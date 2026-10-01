"""
Build a SMALL hybrid-ground-truth dataset that Kilosort4 can be run on
end to end, so changes to Kilosort can actually be graded.

WHY A NEW DATASET. Kilosort ships `simulation.hybrid_simulation`, which does
exactly this job, but it is the author's own research script with hard-coded
Linux paths rather than a usable API. The approach is right though, and this
follows it using the already-tested pieces in ground_truth_harness.py.

------------------------------------------------------------------------------
TWO FLAWS IN THE PREVIOUS VERSION OF THIS FILE, BOTH FOUND BY THE USER, BOTH
FIXED HERE. They are written out because either one silently invalidates every
number the dataset produces.

FLAW 1 -- SOURCE UNITS WERE NOT CHECKED FOR BEING REAL NEURONS. The previous
version selected source units by spike count and probe position only. Of the
17 it picked, Kilosort itself labelled **8 as `mua`**, two of them at 82%
estimated contamination. Injecting "one neuron's spikes" from a cluster that
is really several neurons mixed together means the ground truth is a lie: when
Kilosort then splits it, splitting may be CORRECT, and the fragmentation
measure is meaningless. Fixed: sources must be `KSLabel == 'good'` with
`ContamPct <= MAX_CONTAM_PCT` in Kilosort's own output.

FLAW 2 -- SPIKES WERE NOT INJECTED INTO QUIET PLACES. The previous version
chose injection times at random, keeping only a separation between injected
spikes. But every destination already has 6-29 resident neurons firing on it,
and measurement showed **12.3% of injected spikes landed within 1 ms of a real
background spike** on their own destination channels (33% for one unit). Those
spikes are collisions with unknown partners, so neither a hit nor a miss can
be interpreted. Worse, `find_quiet_positions` already existed in the harness
and was tested -- it simply was not used here. Fixed: injection times come
from `find_quiet_positions` against the resident units of each destination,
and destinations are chosen in the QUIETEST parts of the probe.

------------------------------------------------------------------------------
THE DESIGN
  background : a real stretch of this session's recording, all 384 channels,
               untouched. Keeps real spikes and real correlated noise, so the
               sorter faces realistic clutter.
  injected   : real multi-channel snippets from VERIFIED GOOD units of this
               same session, relocated to quiet destination channels where
               the source neuron never fired, placed only at times when the
               destination's own residents are silent.
  truth      : every injected spike's time, source unit, destination channels
               and known sub-sample offset.

WHAT IT IS SCORED ON. Only the injected units. The background's own neurons
have no ground truth and are not graded.

HONEST LIMITS
  * Injected spikes add linearly. Real potentials superpose, so this is fair,
    but it tests the sorter, not the biophysics.
  * The injected train is Poisson-like with a refractory floor, not a
    realistic bursting pattern, so nothing burst-dependent is tested.
  * Injected spikes are kept MIN_SEP apart, so collisions BETWEEN injected
    units are not tested.
  * "Quiet" means no SORTED spike nearby. Unsorted or sub-threshold activity
    is invisible to this check and remains in the background.

Usage: python build_small_hybrid_dataset.py
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
OUT_DIR = r"D:\Gil\spike_sorting_agent\hybrid_small"
N_CHAN_BIN, FS, NT0MIN, NT, ITEMSIZE = 384, 30000.0, 20, 61, 2

BG_START = 60_000_000
DURATION_S = 120.0
CHUNK = 1_000_000

N_SOURCE_UNITS = 12
MIN_SOURCE_SPIKES = 800
MAX_CONTAM_PCT = 10.0        # FLAW 1 fix: Kilosort's own contamination estimate
REQUIRE_KSLABEL = "good"     # FLAW 1 fix: must be a single-unit by KS's own call

DEST_MIN_SEPARATION = 22     # channels between injected units
QUIET_GUARD = 75             # FLAW 2 fix: samples of clearance from any
                             # resident spike at the destination
SPIKES_PER_UNIT = 300
MIN_SEP = 150                # samples between injected spikes (any unit)
FOOTPRINT_RADIUS_UM = 60.0
N_SNIPPET_POOL = 400
SEED = 2024

os.makedirs(OUT_DIR, exist_ok=True)
OUT_DAT = os.path.join(OUT_DIR, "hybrid_small.bin")
OUT_TRUTH = os.path.join(OUT_DIR, "hybrid_small_truth.npz")
OUT_PROBE = os.path.join(OUT_DIR, "probe.json")

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

print(f"background: samples {BG_START:,}-{BG_END:,} ({DURATION_S:.0f}s, "
      f"{N_SAMPLES:,} samples, {N_CHAN_BIN} channels)\n")

probe = dict(chanMap=list(range(N_CHAN_BIN)),
             xc=[float(v) for v in channel_positions[:, 0]],
             yc=[float(v) for v in channel_positions[:, 1]],
             kcoords=[0] * N_CHAN_BIN, n_chan=N_CHAN_BIN)
with open(OUT_PROBE, "w") as f:
    json.dump(probe, f)

# ---------------------------------------------------------------- FLAW 1 fix
qual = (pd.read_csv(KS + r"\cluster_KSLabel.tsv", sep="\t")
        .merge(pd.read_csv(KS + r"\cluster_ContamPct.tsv", sep="\t"), on="cluster_id"))
u_all, c_all = np.unique(spike_clusters, return_counts=True)
n_map = dict(zip(u_all.tolist(), c_all.tolist()))
qual["n"] = qual.cluster_id.map(n_map).fillna(0)
pool = qual[(qual.KSLabel == REQUIRE_KSLABEL)
            & (qual.ContamPct <= MAX_CONTAM_PCT)
            & (qual.n >= MIN_SOURCE_SPIKES)
            & (qual.cluster_id < templates.shape[0])].copy()
pool["pk"] = peak_ch_all[pool.cluster_id.to_numpy()]
pool = pool.sort_values("pk")
step = max(len(pool) // N_SOURCE_UNITS, 1)
SOURCE_UNITS = [int(x) for x in pool.cluster_id.to_numpy()[::step][:N_SOURCE_UNITS]]
print(f"source-unit pool: {len(pool)} units that Kilosort calls "
      f"'{REQUIRE_KSLABEL}' with <= {MAX_CONTAM_PCT}% contamination "
      f"and >= {MIN_SOURCE_SPIKES} spikes")
print(f"selected {len(SOURCE_UNITS)} of them, spread across the probe:")
sel = pool[pool.cluster_id.isin(SOURCE_UNITS)]
for _, r in sel.iterrows():
    print(f"    unit {int(r.cluster_id):>4}  contam {r.ContamPct:>5.1f}%  "
          f"{int(r.n):>7,} spikes  peak ch{int(r.pk)}")

# ---------------------------------------------------------------- FLAW 2 fix
# how busy is each part of the probe inside our window?
inwin = (spike_times >= BG_START) & (spike_times < BG_END)
bc_win = spike_clusters[inwin]
busy = np.zeros(N_CHAN_BIN)
for cid, n in zip(*np.unique(bc_win, return_counts=True)):
    if cid < len(peak_ch_all):
        busy[peak_ch_all[cid]] += n
local_density = np.array([busy[max(0, i - 6):i + 7].sum() for i in range(N_CHAN_BIN)])
quiet_order = [int(i) for i in np.argsort(local_density) if 25 < i < N_CHAN_BIN - 25]
print(f"\nlocal background density (spikes within +/-6 ch over {DURATION_S:.0f}s): "
      f"quietest {local_density[quiet_order[0]]:.0f}, "
      f"median {np.median(local_density):.0f}")

print(f"\ncollecting snippets and assigning quiet destinations:")
sources, used_dest = {}, []
for uid in SOURCE_UNITS:
    pc = int(peak_ch_all[uid])
    d = np.sqrt(((channel_positions - channel_positions[pc]) ** 2).sum(axis=1))
    src_ch = np.sort(np.where(d <= FOOTPRINT_RADIUS_UM)[0])
    half = len(src_ch) // 2

    slot = None
    for cand in quiet_order:
        if used_dest and min(abs(cand - x) for x in used_dest) < DEST_MIN_SEPARATION:
            continue
        if abs(cand - pc) < len(src_ch) + 8:
            continue                      # too near the source: ambiguous
        if cand - half < 0 or cand - half + len(src_ch) >= N_CHAN_BIN:
            continue
        slot = cand
        break
    if slot is None:
        print(f"  unit {uid}: no quiet destination available, skipping")
        continue
    used_dest.append(slot)
    dst_ch = np.arange(slot - half, slot - half + len(src_ch))

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
    if not snips:
        print(f"  unit {uid}: no usable snippets, skipping")
        continue
    snips = np.stack(snips)

    # residents at this destination: units whose own peak channel is inside it
    residents = {int(c) for c in np.unique(bc_win)
                 if c < len(peak_ch_all) and abs(int(peak_ch_all[c]) - slot) <= 8}
    sources[uid] = dict(snippets=snips, src_ch=src_ch, dst_ch=dst_ch,
                        peak_ch=pc, dst_peak=slot, residents=residents)
    print(f"  unit {uid}: {len(snips)} snippets, ch{src_ch[0]}-{src_ch[-1]} "
          f"-> ch{dst_ch[0]}-{dst_ch[-1]} (density "
          f"{local_density[slot]:.0f}, {len(residents)} residents)")

assert sources, "no source units usable"

# --------------------------------------------- quiet injection times per unit
print(f"\nfinding quiet injection times (>= {QUIET_GUARD} samples from any "
      f"resident spike):")
all_t, all_lab, all_sh, all_idx = [], [], [], []
for uid, S in sources.items():
    pos = find_quiet_positions(
        spike_times, spike_clusters, S["residents"],
        BG_START + NT + MIN_SEP, BG_END - NT - MIN_SEP,
        n_positions=SPIKES_PER_UNIT, guard_samples=QUIET_GUARD,
        min_separation=MIN_SEP, rng=rng)
    pos = pos - BG_START                              # to window coordinates
    all_t.append(pos)
    all_lab.append(np.full(len(pos), uid, dtype=np.int64))
    all_sh.append(rng.uniform(-0.5, 0.5, size=len(pos)))
    all_idx.append(rng.integers(0, len(S["snippets"]), size=len(pos)))
    print(f"  unit {uid}: {len(pos)} quiet slots found")

times = np.concatenate(all_t); labels = np.concatenate(all_lab)
shifts = np.concatenate(all_sh); snipidx = np.concatenate(all_idx)
o = np.argsort(times)
times, labels, shifts, snipidx = times[o], labels[o], shifts[o], snipidx[o]
keep = np.ones(len(times), bool); last = -10 ** 9
for i, t in enumerate(times):
    if t - last < MIN_SEP:
        keep[i] = False
    else:
        last = t
times, labels, shifts, snipidx = times[keep], labels[keep], shifts[keep], snipidx[keep]
print(f"  total after cross-unit separation: {len(times)} spikes")

# ------------------------------------------------------- write the hybrid file
print(f"\nstreaming background + injections to disk:")
t0 = time.time()
n_inj = 0
with open(OUT_DAT, "wb") as fout:
    pos = 0
    while pos < N_SAMPLES:
        clen = min(CHUNK, N_SAMPLES - pos)
        with open(DAT_PATH, "rb") as f:
            f.seek((BG_START + pos) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(clen * N_CHAN_BIN * ITEMSIZE)
        chunk = np.frombuffer(raw, dtype=np.int16).reshape(clen, N_CHAN_BIN).astype(np.float64)
        for i in np.where((times >= pos) & (times < pos + clen))[0]:
            S = sources[int(labels[i])]
            w = shift_multichannel(S["snippets"][snipidx[i]], float(shifts[i]))
            if inject_multichannel(chunk, w, int(times[i]) - pos, S["dst_ch"],
                                   nt0min=NT0MIN):
                n_inj += 1
        np.clip(chunk, -2 ** 15 + 1, 2 ** 15 - 1).astype(np.int16).tofile(fout)
        pos += clen
print(f"  wrote {N_SAMPLES:,} samples in {time.time()-t0:.0f}s, {n_inj} injected")

fits = np.array([((t % CHUNK) + NT - NT0MIN) <= CHUNK and t >= NT0MIN for t in times])
np.savez(OUT_TRUTH, times=times[fits], labels=labels[fits], shifts=shifts[fits],
         snippet_index=snipidx[fits],
         source_units=np.array(list(sources.keys())),
         dst_peak_channels=np.array([sources[u]["dst_peak"] for u in sources]),
         bg_start=BG_START, n_samples=N_SAMPLES, fs=FS, n_chan_bin=N_CHAN_BIN,
         nt0min=NT0MIN, quiet_guard=QUIET_GUARD,
         contam_pct=np.array([float(qual.loc[qual.cluster_id == u, "ContamPct"].iloc[0])
                              for u in sources]))
print(f"\ntruth: {int(fits.sum())} spikes across {len(sources)} verified-good units")
print(f"saved {OUT_DAT} ({os.path.getsize(OUT_DAT)/1e9:.2f} GB)")
