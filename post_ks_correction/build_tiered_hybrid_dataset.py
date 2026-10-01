"""
A TIERED hybrid ground-truth dataset: verified-good neurons placed into
deliberately chosen easy and hard situations, with the difficulty recorded.

WHY TIERS. The previous dataset put every injected neuron in a quiet probe
region at a quiet moment. That is the easy case, and an easy case cannot
discriminate between algorithms -- a clean neuron on a quiet channel is found
by every version, so the comparison has no power where it matters. Stratifying
the difficulty on purpose, and recording which tier each unit is in, means a
change can be judged where it is supposed to help instead of on an average
that hides everything.

The design rule, which the previous version violated: putting spikes into
busy or colliding situations is GOOD, as long as it is INTENTIONAL and
LABELLED. The earlier dataset had 12.3% of its spikes colliding with resident
neurons by accident, which is the same thing done invisibly, and therefore
uninterpretable.

SOURCE UNITS ARE VERIFIED FIRST, ALWAYS. Every injected unit must be
`KSLabel == 'good'` with `ContamPct <= MAX_CONTAM_PCT` in Kilosort's own
output, because if the source is a mixture of neurons then the ground truth
is a lie and a "split" may actually be correct. Difficulty is then imposed by
WHERE and WHEN the verified spikes are placed -- never by using a dubious
source.

THE FOUR TIERS

  tier 1  easy          quiet probe region (lowest measured local background
                        density), injection times guarded from every resident
                        spike. The floor: anything that fails here is broken.

  tier 2  noisy_channel busiest probe regions available, injection times still
                        guarded. Isolates the effect of dense neighbouring
                        activity from the effect of temporal overlap.

  tier 3  collision     quiet probe region, but injection times deliberately
                        placed a few samples from a resident spike, so the
                        waveforms genuinely overlap. The offset is recorded
                        per spike, so performance can be plotted against how
                        close the collision was.

  tier 4  pair          two DIFFERENT verified-good neurons placed with
                        heavily overlapping footprints (destination peaks a
                        few channels apart), in a quiet region at guarded
                        times. Correct behaviour is TWO clusters; merging them
                        is an error, and this is the only tier that tests the
                        merge/split decision directly, which is the decision
                        this project's footprint work (log 5q, 5t) is about.

Everything needed to score per tier is written into the truth file: the tier
of every unit, the pair id where applicable, and the collision offset of
every tier-3 spike.

HONEST LIMITS (unchanged from the untiered version)
  * Injected spikes add linearly; this tests the sorter, not biophysics.
  * The injected train is Poisson-like with a refractory floor, so nothing
    burst-dependent is tested.
  * "Quiet" means no SORTED spike nearby; unsorted activity is invisible.

Usage: python build_tiered_hybrid_dataset.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ground_truth_harness import (shift_multichannel, inject_multichannel,
                                   find_quiet_positions)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT_DIR = r"D:\Gil\spike_sorting_agent\hybrid_tiered"
N_CHAN_BIN, FS, NT0MIN, NT, ITEMSIZE = 384, 30000.0, 20, 61, 2

BG_START = 60_000_000
DURATION_S = 120.0
CHUNK = 1_000_000

MIN_SOURCE_SPIKES = 800
MAX_CONTAM_PCT = 10.0
REQUIRE_KSLABEL = "good"

# MUTUAL-INDEPENDENCE CONSTRAINT. "KSLabel == good" only says a cluster looks
# like a single unit on its own. It does NOT rule out that two selected
# clusters are two halves of ONE neuron that Kilosort oversplit. That
# possibility is fatal for the pair tier specifically: if a "pair" is really
# one oversplit neuron, then the CORRECT behaviour is to merge them, and the
# merge-error metric would be scoring exactly backwards. It also distorts the
# other tiers, because half of an oversplit neuron is a biased subsample of
# its spikes rather than a neuron's full output.
#
# So selected sources must be mutually FAR APART on the probe and mutually
# DISSIMILAR by Kilosort's own template-similarity matrix. Oversplit halves of
# one neuron necessarily sit at the same place with near-identical waveforms,
# so both constraints exclude them. The spatial one is the stronger guarantee:
# two clusters 150um apart cannot be one neuron.
MIN_SOURCE_SEPARATION_UM = 150.0
MAX_KS_SIMILARITY = 0.20

N_EASY, N_NOISY, N_COLLISION, N_PAIRS = 3, 3, 3, 2   # N_PAIRS pairs = 2x units
SPIKES_PER_UNIT = 300
QUIET_GUARD = 75             # samples of clearance for non-collision tiers
COLLISION_RANGE = (4, 25)    # samples from the resident spike, tier 3
PAIR_CHANNEL_GAP = 3         # destination peak separation inside a pair
DEST_MIN_SEPARATION = 22     # between units of different groups
MIN_SEP = 150
FOOTPRINT_RADIUS_UM = 60.0
N_SNIPPET_POOL = 400
SEED = 7

os.makedirs(OUT_DIR, exist_ok=True)
OUT_DAT = os.path.join(OUT_DIR, "hybrid_tiered.bin")
OUT_TRUTH = os.path.join(OUT_DIR, "hybrid_tiered_truth.npz")
OUT_PROBE = os.path.join(OUT_DIR, "probe.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "manifest.csv")

N_SAMPLES = int(DURATION_S * FS)
BG_END = BG_START + N_SAMPLES
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
assert BG_END <= TOTAL

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
rng = np.random.default_rng(SEED)

with open(OUT_PROBE, "w") as f:
    json.dump(dict(chanMap=list(range(N_CHAN_BIN)),
                   xc=[float(v) for v in channel_positions[:, 0]],
                   yc=[float(v) for v in channel_positions[:, 1]],
                   kcoords=[0] * N_CHAN_BIN, n_chan=N_CHAN_BIN), f)

# --------------------------------------------- verified-good source pool
qual = (pd.read_csv(KS + r"\cluster_KSLabel.tsv", sep="\t")
        .merge(pd.read_csv(KS + r"\cluster_ContamPct.tsv", sep="\t"), on="cluster_id"))
u_all, c_all = np.unique(spike_clusters, return_counts=True)
qual["n"] = qual.cluster_id.map(dict(zip(u_all.tolist(), c_all.tolist()))).fillna(0)
pool = qual[(qual.KSLabel == REQUIRE_KSLABEL) & (qual.ContamPct <= MAX_CONTAM_PCT)
            & (qual.n >= MIN_SOURCE_SPIKES)
            & (qual.cluster_id < templates.shape[0])].copy()
pool["pk"] = peak_ch_all[pool.cluster_id.to_numpy()]
pool = pool.sort_values("pk").reset_index(drop=True)
n_need = N_EASY + N_NOISY + N_COLLISION + 2 * N_PAIRS
print(f"verified-good pool: {len(pool)} units ('{REQUIRE_KSLABEL}', "
      f"<= {MAX_CONTAM_PCT}% contamination, >= {MIN_SOURCE_SPIKES} spikes)")

# Greedy selection enforcing mutual independence, walking the probe in order
# so the accepted set still spans it.
ks_sim = np.load(KS + r"\similar_templates.npy")
cand_ids = [int(x) for x in pool.cluster_id.to_numpy()]
picked, n_close, n_sim = [], 0, 0
for cid in cand_ids:
    ok = True
    for q in picked:
        dist = float(np.sqrt(((channel_positions[peak_ch_all[cid]]
                               - channel_positions[peak_ch_all[q]]) ** 2).sum()))
        if dist < MIN_SOURCE_SEPARATION_UM:
            n_close += 1
            ok = False
            break
        if float(ks_sim[cid, q]) > MAX_KS_SIMILARITY:
            n_sim += 1
            ok = False
            break
    if ok:
        picked.append(cid)
print(f"mutual-independence filter: >= {MIN_SOURCE_SEPARATION_UM:.0f} um apart, "
      f"KS template similarity <= {MAX_KS_SIMILARITY}")
print(f"  accepted {len(picked)} of {len(cand_ids)} "
      f"(rejected {n_close} too close, {n_sim} too similar)")
assert len(picked) >= n_need, (
    f"only {len(picked)} mutually-independent verified-good units, need "
    f"{n_need}; relax MIN_SOURCE_SEPARATION_UM or reduce the tier counts")
step = max(len(picked) // n_need, 1)
picked = picked[::step][:n_need]

# Prove the constraint holds on the FINAL set and report the worst case, so
# this is a checked guarantee rather than a hopeful comment.
worst_d, worst_s = np.inf, 0.0
for i in range(len(picked)):
    for j in range(i + 1, len(picked)):
        a, b = picked[i], picked[j]
        worst_d = min(worst_d, float(np.sqrt(
            ((channel_positions[peak_ch_all[a]]
              - channel_positions[peak_ch_all[b]]) ** 2).sum())))
        worst_s = max(worst_s, float(ks_sim[a, b]))
print(f"  final {len(picked)}: closest pair {worst_d:.0f} um, "
      f"highest similarity {worst_s:.3f}")
assert worst_d >= MIN_SOURCE_SEPARATION_UM and worst_s <= MAX_KS_SIMILARITY
print(f"  -> no two sources can be oversplit halves of one neuron\n")

# ------------------------------------------- probe density, for tier placement
inwin = (spike_times >= BG_START) & (spike_times < BG_END)
bc_win = spike_clusters[inwin]
busy = np.zeros(N_CHAN_BIN)
for cid, n in zip(*np.unique(bc_win, return_counts=True)):
    if cid < len(peak_ch_all):
        busy[peak_ch_all[cid]] += n
density = np.array([busy[max(0, i - 6):i + 7].sum() for i in range(N_CHAN_BIN)])
valid = np.array([25 < i < N_CHAN_BIN - 25 for i in range(N_CHAN_BIN)])
quietest = [int(i) for i in np.argsort(np.where(valid, density, np.inf))]
busiest = [int(i) for i in np.argsort(np.where(valid, -density, np.inf))]
print(f"local background density: quietest {density[quietest[0]]:.0f}, "
      f"median {np.median(density):.0f}, busiest {density[busiest[0]]:.0f}\n")

# ------------------------------------------------------- assign tiers + slots
assignments = []          # dicts: unit, tier, pair_id, slot
used = []


def take_slot(candidates, src_pk, footprint_len, want=None):
    for c in candidates:
        if want is not None and c != want:
            continue
        if used and min(abs(c - x) for x in used) < DEST_MIN_SEPARATION:
            continue
        if abs(c - src_pk) < footprint_len + 8:
            continue
        half = footprint_len // 2
        if c - half < 0 or c - half + footprint_len >= N_CHAN_BIN:
            continue
        return c
    return None


def footprint_of(uid):
    pc = int(peak_ch_all[uid])
    d = np.sqrt(((channel_positions - channel_positions[pc]) ** 2).sum(axis=1))
    return pc, np.sort(np.where(d <= FOOTPRINT_RADIUS_UM)[0])


idx = 0
plan = ([("easy", quietest)] * N_EASY + [("noisy_channel", busiest)] * N_NOISY
        + [("collision", quietest)] * N_COLLISION)
for tier, cand in plan:
    uid = picked[idx]; idx += 1
    pc, src_ch = footprint_of(uid)
    slot = take_slot(cand, pc, len(src_ch))
    if slot is None:
        print(f"  !! unit {uid}: no slot for tier {tier}, skipped")
        continue
    used.append(slot)
    assignments.append(dict(unit=uid, tier=tier, pair_id=-1, slot=slot,
                            src_ch=src_ch, peak_ch=pc))

for p in range(N_PAIRS):
    a, b = picked[idx], picked[idx + 1]; idx += 2
    pc_a, src_a = footprint_of(a)
    pc_b, src_b = footprint_of(b)
    slot_a = take_slot(quietest, pc_a, len(src_a))
    if slot_a is None:
        print(f"  !! pair {p}: no slot, skipped")
        continue
    used.append(slot_a)
    slot_b = slot_a + PAIR_CHANNEL_GAP
    half_b = len(src_b) // 2
    if (abs(slot_b - pc_b) < len(src_b) + 8
            or slot_b - half_b < 0 or slot_b - half_b + len(src_b) >= N_CHAN_BIN):
        print(f"  !! pair {p}: partner slot invalid, skipped")
        continue
    used.append(slot_b)
    assignments.append(dict(unit=a, tier="pair", pair_id=p, slot=slot_a,
                            src_ch=src_a, peak_ch=pc_a))
    assignments.append(dict(unit=b, tier="pair", pair_id=p, slot=slot_b,
                            src_ch=src_b, peak_ch=pc_b))

print("TIER ASSIGNMENT")
for A in assignments:
    print(f"  unit {A['unit']:>4}  tier {A['tier']:<14} "
          f"pair {A['pair_id']:>2}  ch{A['peak_ch']:>3} -> ch{A['slot']:>3}  "
          f"local density {density[A['slot']]:>7.0f}")

# ------------------------------------------------------------- read snippets
print("\nreading real snippets:")
for A in assignments:
    uid, src_ch = A["unit"], A["src_ch"]
    st_u = np.sort(spike_times[spike_clusters == uid])
    st_u = st_u[(st_u > NT + 80) & (st_u < TOTAL - NT - 80)]
    pick = st_u[rng.permutation(len(st_u))[:N_SNIPPET_POOL]]
    snips = []
    for sp in pick:
        lo, n = int(sp) - NT0MIN - 60, NT + 120
        if lo < 0 or lo + n > TOTAL:
            continue
        with open(DAT_PATH, "rb") as f:
            f.seek(lo * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(n * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(n, N_CHAN_BIN)
        snips.append(np.stack(
            [filtfilt(b_hp, a_hp, blk[:, c].astype(np.float64))[60:60 + NT]
             for c in src_ch], axis=1))
    A["snippets"] = np.stack(snips)
    half = len(src_ch) // 2
    A["dst_ch"] = np.arange(A["slot"] - half, A["slot"] - half + len(src_ch))
    A["residents"] = {int(c) for c in np.unique(bc_win)
                      if c < len(peak_ch_all) and abs(int(peak_ch_all[c]) - A["slot"]) <= 8}
    print(f"  unit {uid}: {len(snips)} snippets, {len(A['residents'])} residents "
          f"at destination")

# ----------------------------------------------------- choose injection times
print("\nchoosing injection times per tier:")
rows = []
for A in assignments:
    uid, tier = A["unit"], A["tier"]
    if tier == "collision":
        # DELIBERATE overlap: sit a few samples off a resident spike
        res_t = np.sort(spike_times[np.isin(spike_clusters, list(A["residents"]))
                                     & inwin]) - BG_START
        res_t = res_t[(res_t > NT + MIN_SEP) & (res_t < N_SAMPLES - NT - MIN_SEP)]
        rng.shuffle(res_t)
        chosen, offs, last = [], [], -10 ** 9
        for rt in res_t:
            off = int(rng.integers(*COLLISION_RANGE)) * (1 if rng.random() < .5 else -1)
            t = int(rt) + off
            if chosen and abs(t - chosen[-1]) < MIN_SEP:
                continue
            chosen.append(t); offs.append(off)
            if len(chosen) >= SPIKES_PER_UNIT:
                break
        pos = np.array(sorted(chosen), dtype=np.int64)
        off_arr = np.array([offs[chosen.index(int(p))] for p in pos])
        print(f"  unit {uid:>4} [{tier}]: {len(pos)} spikes placed "
              f"{COLLISION_RANGE[0]}-{COLLISION_RANGE[1]} samples from a resident spike")
    else:
        pos = find_quiet_positions(spike_times, spike_clusters, A["residents"],
                                   BG_START + NT + MIN_SEP, BG_END - NT - MIN_SEP,
                                   n_positions=SPIKES_PER_UNIT,
                                   guard_samples=QUIET_GUARD,
                                   min_separation=MIN_SEP, rng=rng) - BG_START
        off_arr = np.full(len(pos), np.nan)
        print(f"  unit {uid:>4} [{tier}]: {len(pos)} quiet spikes")
    for k, (t, o_) in enumerate(zip(pos, off_arr)):
        rows.append(dict(t=int(t), unit=uid, tier=tier, pair_id=A["pair_id"],
                         shift=float(rng.uniform(-0.5, 0.5)),
                         snip=int(rng.integers(0, len(A["snippets"]))),
                         collision_offset=float(o_)))

df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
# cross-unit separation, but NEVER drop a tier-3 spike for being near its
# intended resident -- that closeness is the point of that tier
keep, last = [], -10 ** 9
for i, r in df.iterrows():
    if r.t - last < MIN_SEP:
        continue
    keep.append(i); last = r.t
df = df.loc[keep].reset_index(drop=True)
print(f"\ntotal after cross-unit separation: {len(df)} spikes")
print(df.groupby("tier").size().to_string())

# ------------------------------------------------------------ write the file
A_by_unit = {A["unit"]: A for A in assignments}
print("\nstreaming to disk:")
t0 = time.time(); n_inj = 0
with open(OUT_DAT, "wb") as fout:
    pos = 0
    while pos < N_SAMPLES:
        clen = min(CHUNK, N_SAMPLES - pos)
        with open(DAT_PATH, "rb") as f:
            f.seek((BG_START + pos) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(clen * N_CHAN_BIN * ITEMSIZE)
        chunk = np.frombuffer(raw, dtype=np.int16).reshape(clen, N_CHAN_BIN).astype(np.float64)
        sel = df[(df.t >= pos) & (df.t < pos + clen)]
        for _, r in sel.iterrows():
            A = A_by_unit[int(r.unit)]
            # NOTE: r["shift"], never r.shift -- `shift` is a DataFrame method,
            # so attribute access silently returns the method instead of the
            # column and float() then fails (or worse, would not).
            w = shift_multichannel(A["snippets"][int(r.snip)], float(r["shift"]))
            if inject_multichannel(chunk, w, int(r.t) - pos, A["dst_ch"], nt0min=NT0MIN):
                n_inj += 1
        np.clip(chunk, -2 ** 15 + 1, 2 ** 15 - 1).astype(np.int16).tofile(fout)
        pos += clen
print(f"  wrote {N_SAMPLES:,} samples in {time.time()-t0:.0f}s, {n_inj} injected")

fits = np.array([((t % CHUNK) + NT - NT0MIN) <= CHUNK and t >= NT0MIN for t in df.t])
d = df.loc[fits]
tiers = sorted(d.tier.unique())
tier_code = {t: i for i, t in enumerate(tiers)}
np.savez(OUT_TRUTH,
         times=d.t.to_numpy(), labels=d.unit.to_numpy(),
         shifts=d["shift"].to_numpy(), snippet_index=d.snip.to_numpy(),
         tier=np.array([tier_code[t] for t in d.tier]),
         tier_names=np.array(tiers), pair_id=d.pair_id.to_numpy(),
         collision_offset=d.collision_offset.to_numpy(),
         source_units=np.array([A["unit"] for A in assignments]),
         dst_peak_channels=np.array([A["slot"] for A in assignments]),
         unit_tier=np.array([tier_code[A["tier"]] for A in assignments]),
         unit_pair_id=np.array([A["pair_id"] for A in assignments]),
         bg_start=BG_START, n_samples=N_SAMPLES, fs=FS,
         n_chan_bin=N_CHAN_BIN, nt0min=NT0MIN, quiet_guard=QUIET_GUARD)

man = pd.DataFrame([dict(unit=A["unit"], tier=A["tier"], pair_id=A["pair_id"],
                         source_peak_ch=A["peak_ch"], dest_peak_ch=A["slot"],
                         n_footprint_ch=len(A["src_ch"]),
                         local_density=int(density[A["slot"]]),
                         n_residents=len(A["residents"]),
                         contam_pct=float(qual.loc[qual.cluster_id == A["unit"],
                                                   "ContamPct"].iloc[0]),
                         n_spikes=int((d.unit == A["unit"]).sum()))
                    for A in assignments])
man.to_csv(OUT_MANIFEST, index=False)
print(f"\nMANIFEST (also saved to {OUT_MANIFEST}):")
print(man.to_string(index=False))
print(f"\ntruth: {len(d)} spikes, {len(assignments)} units, tiers {tiers}")
print(f"saved {OUT_DAT} ({os.path.getsize(OUT_DAT)/1e9:.2f} GB)")
