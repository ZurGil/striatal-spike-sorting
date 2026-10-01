"""
Build ONE replicate of the redesigned tiered hybrid dataset.

WHAT CHANGED FROM build_tiered_hybrid_dataset.py, and why (all from Gil's
review of the first benchmark):

1. SITES ARE CHOSEN BY AMPLITUDE, NOT BY UNIT COUNT. The old builder ranked
   channels by how many spikes peaked nearby and called the bottom "quiet".
   Section 5ag showed what that bought: the "quiet" easy sites still hosted
   5-11 neurons, and 80.6% of what counted against precision was a resident
   neuron's real spikes. Requiring NO units nearby is impossible here -- the
   probe carries 308 clusters over 384 channels and its most isolated channel
   is 52 um from the nearest unit. Gil's correction is the workable one: what
   matters is whether the neighbours are LARGE, since a small neighbour is just
   background. Sites are now ranked by `max_neighbor_uv`, the largest resident
   unit's real amplitude as seen on that channel, which spans 23-467 uV across
   this probe -- a 20x contrast to build tiers from.

2. COLLISION AND PAIR MOVE TO QUIET SITES. Previously their difficulty mixed
   two causes: temporal overlap AND whatever the channel's neighbourhood was
   doing. The challenge in those tiers is separating units, so channel noise
   must not be confounded with it. Only the `hard` tier gets loud sites.

3. THE SOURCE POOL IS 49 UNITS, NOT 13. select_source_units.py replaced the
   >= 150 um separation proxy with a cross-correlogram refractory test (two
   clusters that are really one cell cannot fire within a millisecond of each
   other), which let distance relax to 40 um. 50 passed; Gil reviewed every
   waveform and rejected one (403, a diffuse smeared footprint), leaving 49.

4. REPLICATES WITH TIER ROTATION. One 120 s recording holds only about 8 quiet
   sites and 9 loud ones at 22-channel spacing, so a single file cannot carry
   many more than 13 units without defeating the tier design. Statistics
   therefore come from several replicates, each on a DIFFERENT window of the
   session so the backgrounds differ too. Each replicate rotates which units go
   to which tier, so tier differences are measured WITHIN unit rather than
   across units -- the old easy tier's three units scored 0.30, 0.45 and 0.86,
   a spread that is far more likely unit identity than tier difficulty.

5. SITE PROPERTIES ARE RECORDED PER UNIT in the manifest (noise, largest
   neighbour, density), so the difficulty of every placement is documented as a
   number rather than asserted by its tier name.

Usage: python build_tiered_v2.py <replicate_index 0..3>
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ground_truth_harness import (shift_multichannel, inject_multichannel,
                                  find_quiet_positions)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
            r"\20260916_110311.probe1.dat")
OUTPUTS = r"D:\Gil\spike_sorting_agent\outputs"
ROOT = r"D:\Gil\spike_sorting_agent"

N_CHAN_BIN, ITEMSIZE, FS = 384, 2, 30000.0
NT, NT0MIN = 61, 20
GAIN_TO_UV = 0.018311105685598315
DURATION_S = 120.0
N_SAMPLES = int(DURATION_S * FS)
CHUNK = 300_000
SEED = 20261001

# four replicates, four different windows of the 210-minute session
BG_STARTS = [60_000_000, 120_000_000, 180_000_000, 240_000_000]

N_EASY, N_HARD, N_COLLISION, N_PAIRS = 3, 3, 3, 2
SPIKES_PER_UNIT = 300
QUIET_GUARD = 75
COLLISION_RANGE = (4, 25)
PAIR_CHANNEL_GAP = 3
# Site capacity is the binding constraint on this design, and it was set too
# tight at first: p35 cuts with 22-channel spacing yield only 8 quiet sites,
# and the quiet tiers need exactly 8 (3 easy + 3 collision + 2 pair anchors).
# Any unit whose own source channel sits near a quiet site loses that site, so
# replicate 2 ran out and the placement assertion fired. Relaxing to p45 and
# 18-channel spacing yields 13 quiet sites -- a real margin -- while keeping the
# quiet ceiling at 73 uV, still far below the loud floor of 95.5 uV. Footprints
# are about 10 channels wide, so 18 channels still leaves a clear gap.
DEST_MIN_SEPARATION = 16
MIN_SEP = 40
FOOTPRINT_RADIUS_UM = 60.0
N_SNIPPET_POOL = 120
QUIET_PCTL, LOUD_PCTL = 45, 65

rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BG_START = BG_STARTS[rep]
BG_END = BG_START + N_SAMPLES
OUT_DIR = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_DAT = os.path.join(OUT_DIR, "hybrid.bin")
OUT_TRUTH = os.path.join(OUT_DIR, "hybrid_truth.npz")
OUT_PROBE = os.path.join(OUT_DIR, "probe.json")

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
rng = np.random.default_rng(SEED + rep)
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
assert BG_END <= TOTAL, f"replicate {rep} window exceeds the file"
print(f"REPLICATE {rep}: samples {BG_START:,}-{BG_END:,} "
      f"({BG_START/FS/60:.1f}-{BG_END/FS/60:.1f} min)")

templates = np.load(os.path.join(KS, "templates.npy"))
spike_times = np.load(os.path.join(KS, "spike_times.npy")).ravel()
spike_clusters = np.load(os.path.join(KS, "spike_clusters.npy")).ravel()
cp = np.load(os.path.join(KS, "channel_positions.npy"))
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
ks_sim = np.load(os.path.join(KS, "similar_templates.npy"))
qual = (pd.read_csv(os.path.join(KS, "cluster_KSLabel.tsv"), sep="\t")
        .merge(pd.read_csv(os.path.join(KS, "cluster_ContamPct.tsv"), sep="\t"),
               on="cluster_id"))

with open(OUT_PROBE, "w") as f:
    json.dump(dict(chanMap=list(range(N_CHAN_BIN)),
                   xc=[float(v) for v in cp[:, 0]],
                   yc=[float(v) for v in cp[:, 1]],
                   kcoords=[0] * N_CHAN_BIN, n_chan=N_CHAN_BIN), f)

# ------------------------------------------------ the reviewed source pool
sel = pd.read_csv(os.path.join(OUTPUTS, "selected_source_units.csv"))
pool = sel[sel.clean].unit.tolist()
print(f"source pool: {len(pool)} reviewed units")

# ------------------------------------------- sites, ranked by amplitude
site = pd.read_csv(os.path.join(OUTPUTS, "probe_amplitude_survey.csv"))
e = site[site.edge_ok]
nz_q = np.percentile(e.noise_uv, QUIET_PCTL)
mn_q = np.percentile(e.max_neighbor_uv, QUIET_PCTL)
nz_l = np.percentile(e.noise_uv, LOUD_PCTL)
mn_l = np.percentile(e.max_neighbor_uv, LOUD_PCTL)
quiet_all = (e[(e.noise_uv <= nz_q) & (e.max_neighbor_uv <= mn_q)]
             .sort_values("max_neighbor_uv").ks_channel.tolist())
loud_all = (e[(e.noise_uv >= nz_l) & (e.max_neighbor_uv >= mn_l)]
            .sort_values("max_neighbor_uv", ascending=False).ks_channel.tolist())


def pack(channels, spacing=DEST_MIN_SEPARATION):
    """Greedy maximal set of mutually separated sites, best-first."""
    out = []
    for c in channels:
        if all(abs(int(c) - s) >= spacing for s in out):
            out.append(int(c))
    return out


# PACK FIRST, THEN SHUFFLE -- the two steps solve two different problems.
#
# Packing in best-first order is what makes the set large: a random walk over
# the qualifying channels packs badly and replicate 3 ran out of slots entirely.
#
# Shuffling the PACKED set is what removes a confound. Three tiers (pair, easy,
# collision) draw from the same quiet pool in sequence, so walking it
# quietest-first hands the first tier the quietest sites and the last tier the
# loudest leftovers -- in the previous build that was easy at 31.5 uV median
# against collision at 45.2 uV, a site difference confounded with the tier label
# being measured. Dealing a shuffled packed set gives all three the same
# distribution.
#
# The loud pool is NOT shuffled: only the hard tier draws from it, so there is
# no competition to debias, and taking the loudest sites is the entire point of
# that tier. Shuffling it dropped hard sites from 193-466 uV to 100-224 uV.
# Quiet sites are packed with the PAIR GAP added: a pair occupies slot_a and
# slot_a + PAIR_CHANNEL_GAP, so a neighbouring site packed only
# DEST_MIN_SEPARATION from slot_a sits closer than that to slot_b and
# take_slot would reject it, silently costing a placement.
quiet_sites = pack(quiet_all, DEST_MIN_SEPARATION + PAIR_CHANNEL_GAP)
loud_sites = pack(loud_all)
quiet_rank = [int(c) for c in rng.permutation(quiet_sites)]
loud_rank = loud_sites
quiet_set = set(quiet_all)
print(f"packed sites: {len(quiet_sites)} quiet, {len(loud_sites)} loud "
      f"(need {N_EASY + N_COLLISION + N_PAIRS} quiet, {N_HARD} loud)")
print(f"quiet candidates {len(quiet_rank)} (noise <= {nz_q:.1f} uV, "
      f"largest neighbour <= {mn_q:.0f} uV)")
print(f"loud  candidates {len(loud_rank)} (noise >= {nz_l:.1f} uV, "
      f"largest neighbour >= {mn_l:.0f} uV)")
sprop = site.set_index("ks_channel")

# ------------------------------------------- residents inside THIS window
inwin = (spike_times >= BG_START) & (spike_times < BG_END)
bc_win = spike_clusters[inwin]
busy = np.zeros(N_CHAN_BIN)
for cid, n in zip(*np.unique(bc_win, return_counts=True)):
    if cid < len(peak_ch_all):
        busy[peak_ch_all[cid]] += n
density = np.array([busy[max(0, i - 6):i + 7].sum() for i in range(N_CHAN_BIN)])

# --------------------------------------------------- assign tiers + slots
used = []


def take_slot(candidates, src_pk, footprint_len):
    for c in candidates:
        if used and min(abs(c - x) for x in used) < DEST_MIN_SEPARATION:
            continue
        if abs(c - src_pk) < footprint_len + 8:      # never land on the source
            continue
        half = footprint_len // 2
        if c - half < 0 or c - half + footprint_len >= N_CHAN_BIN:
            continue
        return c
    return None


def footprint_of(uid):
    pc = int(peak_ch_all[uid])
    d = np.sqrt(((cp - cp[pc]) ** 2).sum(axis=1))
    return pc, np.sort(np.where(d <= FOOTPRINT_RADIUS_UM)[0])


def _slot_ok(c, src_pk, flen):
    if used and min(abs(c - x) for x in used) < DEST_MIN_SEPARATION:
        return False
    if abs(c - src_pk) < flen + 8:                  # never land on the source
        return False
    half = flen // 2
    return not (c - half < 0 or c - half + flen >= N_CHAN_BIN)


def take_pair_slots(pc_a, len_a, pc_b, len_b):
    """Find an anchor whose PARTNER slot is also quiet and valid.

    The partner sits a fixed PAIR_CHANNEL_GAP away, so it cannot be chosen --
    only checked. Searching over anchors (rather than giving up on the unit) is
    what makes this terminate: the binding constraint is the SITE, not which
    units were handed to the pair. The earlier version consumed a new unit pair
    on every failure and burned the whole pool, leaving the other tiers nothing.
    Both gap directions are tried, since the probe is not symmetric.
    """
    for c in quiet_rank:
        if not _slot_ok(c, pc_a, len_a):
            continue
        for d in (c + PAIR_CHANNEL_GAP, c - PAIR_CHANNEL_GAP):
            if d not in quiet_set:
                continue
            if abs(d - pc_b) < len_b + 8:
                continue
            hb = len_b // 2
            if d - hb < 0 or d - hb + len_b >= N_CHAN_BIN:
                continue
            return c, d
    return None, None


# ROTATION: replicate r starts at a different point in the pool, so a unit that
# was `easy` in one replicate becomes `hard` or `collision` in another and tier
# effects are measured within unit.
n_need = N_EASY + N_HARD + N_COLLISION + 2 * N_PAIRS
offset = (rep * n_need) % len(pool)
order = pool[offset:] + pool[:offset]
print(f"rotation: replicate {rep} starts at pool index {offset} "
      f"(unit {order[0]})")

assignments, idx = [], 0

# PLACEMENT ORDER IS BY HOW CONSTRAINED EACH TIER IS, NOT BY TIER NAME.
# The quiet and loud pools compete for the same probe, and a slot taken by one
# tier blocks DEST_MIN_SEPARATION channels on either side for every other tier.
# The first version placed easy -> hard -> collision -> pair and the pairs then
# failed completely: the hard sites (71 and 335) sat within 22 channels of every
# remaining quiet candidate (352, 348, 350, 356, 66), so there was nowhere left
# for a pair to go. Pairs are the most constrained (they need TWO adjacent
# slots), hard is the least (67 loud candidates against 73 quiet ones but with
# far more spread), so they are placed in that order.
PAIRS_FIRST = True

# THE PAIR TIER'S TWO DONORS MUST COME FROM REMOTE PARTS OF THE PROBE.
#
# This is the tier that asks "did Kilosort keep two distinct neurons apart?",
# so the premise -- that they ARE two distinct neurons -- has to be beyond
# doubt. The first version took `order[idx]` and `order[idx+1]`, and since the
# pool is sorted by peak channel that paired each unit with its NEIGHBOUR:
# donors 51 um apart at template similarity 0.197. Two clusters that close
# could plausibly be halves of one oversplit neuron, in which case merging
# them would arguably be the CORRECT answer and the tier would be scoring
# backwards.
#
# The pool-wide filter is 40 um, which is fine for every other tier because
# those units go to separate sites and are never tested against each other.
# The pair tier needs its own, much stronger requirement, so the partner is
# SEARCHED FOR rather than taken adjacent. The CCG refractory test in
# select_source_units.py already passed these pairs; this is belt and braces,
# because distance is a structural guarantee while the CCG test is statistical
# and can lack power on low-rate units.
PAIR_MIN_SOURCE_SEPARATION_UM = 150.0
PAIR_MAX_SIMILARITY = 0.05


def source_distance(a, b):
    return float(np.sqrt(((cp[peak_ch_all[a]] - cp[peak_ch_all[b]]) ** 2).sum()))


taken = set()
for p in range(N_PAIRS):
    placed = False
    for i, a in enumerate(order):
        if placed:
            break
        if a in taken:
            continue
        pc_a, src_a = footprint_of(a)
        for b in order:
            if b == a or b in taken:
                continue
            if source_distance(a, b) < PAIR_MIN_SOURCE_SEPARATION_UM:
                continue
            if float(ks_sim[a, b]) > PAIR_MAX_SIMILARITY:
                continue
            pc_b, src_b = footprint_of(b)
            slot_a, slot_b = take_pair_slots(pc_a, len(src_a), pc_b, len(src_b))
            if slot_a is None:
                continue
            used += [slot_a, slot_b]
            taken.update((a, b))
            assignments.append(dict(unit=a, tier="pair", pair_id=p, slot=slot_a,
                                    src_ch=src_a, peak_ch=pc_a))
            assignments.append(dict(unit=b, tier="pair", pair_id=p, slot=slot_b,
                                    src_ch=src_b, peak_ch=pc_b))
            print(f"  pair {p}: units {a} (ch{pc_a}) + {b} (ch{pc_b}), "
                  f"{source_distance(a, b):.0f} um apart, "
                  f"similarity {ks_sim[a, b]:.3f}")
            placed = True
            break
    if not placed:
        print(f"  !! pair {p}: no remote-enough donor pair could be placed")

# the remaining tiers draw from whatever the pairs did not consume
order = [u for u in order if u not in taken]
idx = 0

plan = ([("easy", quiet_rank)] * N_EASY
        + [("collision", quiet_rank)] * N_COLLISION
        + [("hard", loud_rank)] * N_HARD)
for tier, cand in plan:
    while idx < len(order):
        uid = order[idx]; idx += 1
        pc, src_ch = footprint_of(uid)
        slot = take_slot(cand, pc, len(src_ch))
        if slot is not None:
            used.append(slot)
            assignments.append(dict(unit=uid, tier=tier, pair_id=-1, slot=slot,
                                    src_ch=src_ch, peak_ch=pc))
            break
        print(f"  !! unit {uid}: no {tier} slot, trying next")

# A tier that silently fails to place looks exactly like a smaller benchmark,
# which is how the pair tier vanished from the first v2 build without an error.
got = pd.Series([A["tier"] for A in assignments]).value_counts().to_dict()
want = {"easy": N_EASY, "hard": N_HARD, "collision": N_COLLISION,
        "pair": 2 * N_PAIRS}
assert got == want, f"placement incomplete: got {got}, wanted {want}"

order_key = {"easy": 0, "hard": 1, "collision": 2, "pair": 3}
assignments.sort(key=lambda A: (order_key[A["tier"]], A["pair_id"], A["slot"]))

print("\nTIER ASSIGNMENT  (site properties are MEASURED, not assumed)")
print(f"{'unit':>5} {'tier':<10} {'pair':>4} {'src':>4} {'dest':>5} "
      f"{'Trodes':>7} {'noise':>7} {'maxNbr':>7} {'density':>8}")
for A in assignments:
    s = sprop.loc[A["slot"]]
    print(f"{A['unit']:>5} {A['tier']:<10} {A['pair_id']:>4} {A['peak_ch']:>4} "
          f"{A['slot']:>5} {int(s.ntrode):>7} {s.noise_uv:>7.1f} "
          f"{s.max_neighbor_uv:>7.1f} {int(s.density):>8}")

qsites = [A["slot"] for A in assignments if A["tier"] != "hard"]
hsites = [A["slot"] for A in assignments if A["tier"] == "hard"]
print(f"\n  quiet-tier sites: largest neighbour "
      f"{sprop.loc[qsites].max_neighbor_uv.min():.0f}-"
      f"{sprop.loc[qsites].max_neighbor_uv.max():.0f} uV")
print(f"  hard-tier  sites: largest neighbour "
      f"{sprop.loc[hsites].max_neighbor_uv.min():.0f}-"
      f"{sprop.loc[hsites].max_neighbor_uv.max():.0f} uV")

# ------------------------------------------------------------- snippets
print("\nreading real snippets:")
for A in assignments:
    uid, src_ch = A["unit"], A["src_ch"]
    st_u = np.sort(spike_times[spike_clusters == uid])
    st_u = st_u[(st_u > NT + 80) & (st_u < TOTAL - NT - 80)]
    pick = st_u[rng.permutation(len(st_u))[:N_SNIPPET_POOL]]
    snips = []
    with open(DAT_PATH, "rb") as f:
        for sp in pick:
            lo, n = int(sp) - NT0MIN - 60, NT + 120
            if lo < 0 or lo + n > TOTAL:
                continue
            f.seek(lo * N_CHAN_BIN * ITEMSIZE)
            blk = np.frombuffer(f.read(n * N_CHAN_BIN * ITEMSIZE),
                                dtype=np.int16).reshape(n, N_CHAN_BIN)
            snips.append(np.stack(
                [filtfilt(b_hp, a_hp, blk[:, c].astype(np.float64))[60:60 + NT]
                 for c in src_ch], axis=1))
    A["snippets"] = np.stack(snips)
    half = len(src_ch) // 2
    A["dst_ch"] = np.arange(A["slot"] - half, A["slot"] - half + len(src_ch))
    A["residents"] = {int(c) for c in np.unique(bc_win)
                      if c < len(peak_ch_all)
                      and abs(int(peak_ch_all[c]) - A["slot"]) <= 8}
    print(f"  unit {uid:>4}: {len(snips)} snippets, "
          f"{len(A['residents'])} residents at destination")

# ------------------------------------------------- injection times per tier
print("\nchoosing injection times:")
rows = []
for A in assignments:
    uid, tier = A["unit"], A["tier"]
    if tier == "collision":
        res_t = np.sort(spike_times[np.isin(spike_clusters, list(A["residents"]))
                                    & inwin]) - BG_START
        res_t = res_t[(res_t > NT + MIN_SEP) & (res_t < N_SAMPLES - NT - MIN_SEP)]
        rng.shuffle(res_t)
        chosen, offs = [], []
        for rt in res_t:
            off = int(rng.integers(*COLLISION_RANGE)) * (1 if rng.random() < .5 else -1)
            t = int(rt) + off
            if chosen and abs(t - chosen[-1]) < MIN_SEP:
                continue
            chosen.append(t); offs.append(off)
            if len(chosen) >= SPIKES_PER_UNIT:
                break
        order_ = np.argsort(chosen)
        pos = np.array(chosen, dtype=np.int64)[order_]
        off_arr = np.array(offs)[order_]
        print(f"  unit {uid:>4} [{tier}]: {len(pos)} spikes "
              f"{COLLISION_RANGE[0]}-{COLLISION_RANGE[1]} samples from a resident")
    else:
        pos = find_quiet_positions(spike_times, spike_clusters, A["residents"],
                                   BG_START + NT + MIN_SEP, BG_END - NT - MIN_SEP,
                                   n_positions=SPIKES_PER_UNIT,
                                   guard_samples=QUIET_GUARD,
                                   min_separation=MIN_SEP, rng=rng) - BG_START
        off_arr = np.full(len(pos), np.nan)
        print(f"  unit {uid:>4} [{tier}]: {len(pos)} guarded spikes")
    for t, o_ in zip(pos, off_arr):
        rows.append(dict(t=int(t), unit=uid, tier=tier, pair_id=A["pair_id"],
                         shift=float(rng.uniform(-0.5, 0.5)),
                         snip=int(rng.integers(0, len(A["snippets"]))),
                         collision_offset=float(o_)))

df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
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
t0, n_inj = time.time(), 0
with open(OUT_DAT, "wb") as fout:
    pos = 0
    while pos < N_SAMPLES:
        clen = min(CHUNK, N_SAMPLES - pos)
        with open(DAT_PATH, "rb") as f:
            f.seek((BG_START + pos) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(clen * N_CHAN_BIN * ITEMSIZE)
        chunk = np.frombuffer(raw, dtype=np.int16).reshape(
            clen, N_CHAN_BIN).astype(np.float64)
        for _, r in df[(df.t >= pos) & (df.t < pos + clen)].iterrows():
            A = A_by_unit[int(r.unit)]
            # r["shift"], never r.shift -- `shift` is a DataFrame method and
            # attribute access silently returns it instead of the column.
            w = shift_multichannel(A["snippets"][int(r.snip)], float(r["shift"]))
            if inject_multichannel(chunk, w, int(r.t) - pos, A["dst_ch"],
                                   nt0min=NT0MIN):
                n_inj += 1
        np.clip(chunk, -2 ** 15 + 1, 2 ** 15 - 1).astype(np.int16).tofile(fout)
        pos += clen
print(f"  wrote {N_SAMPLES:,} samples in {time.time()-t0:.0f}s, {n_inj} injected")

fits = np.array([((t % CHUNK) + NT - NT0MIN) <= CHUNK and t >= NT0MIN
                 for t in df.t])
d = df.loc[fits]
tiers = sorted(d.tier.unique())
tc = {t: i for i, t in enumerate(tiers)}
np.savez(OUT_TRUTH,
         times=d.t.to_numpy(), labels=d.unit.to_numpy(),
         shifts=d["shift"].to_numpy(), snippet_index=d.snip.to_numpy(),
         tier=np.array([tc[t] for t in d.tier]), tier_names=np.array(tiers),
         pair_id=d.pair_id.to_numpy(),
         collision_offset=d.collision_offset.to_numpy(),
         source_units=np.array([A["unit"] for A in assignments]),
         dst_peak_channels=np.array([A["slot"] for A in assignments]),
         unit_tier=np.array([tc[A["tier"]] for A in assignments]),
         unit_pair_id=np.array([A["pair_id"] for A in assignments]),
         bg_start=BG_START, n_samples=N_SAMPLES, fs=FS, replicate=rep,
         n_chan_bin=N_CHAN_BIN, nt0min=NT0MIN, quiet_guard=QUIET_GUARD)

man = pd.DataFrame([dict(
    replicate=rep, unit=A["unit"], tier=A["tier"], pair_id=A["pair_id"],
    source_peak_ch=A["peak_ch"], dest_peak_ch=A["slot"],
    dest_ntrode=int(sprop.loc[A["slot"]].ntrode),
    site_noise_uv=float(sprop.loc[A["slot"]].noise_uv),
    site_max_neighbor_uv=float(sprop.loc[A["slot"]].max_neighbor_uv),
    site_sum_neighbor_uv=float(sprop.loc[A["slot"]].sum_neighbor_uv),
    site_density=int(density[A["slot"]]),
    n_footprint_ch=len(A["src_ch"]), n_residents=len(A["residents"]),
    contam_pct=float(qual.loc[qual.cluster_id == A["unit"], "ContamPct"].iloc[0]),
    n_spikes=int((d.unit == A["unit"]).sum())) for A in assignments])
man.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
print(f"\n{man.to_string(index=False)}")
print(f"\nwrote {OUT_DIR}")
