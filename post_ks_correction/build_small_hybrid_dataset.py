"""
Build a SMALL hybrid-ground-truth dataset that Kilosort4 can be run on
end to end, so changes to Kilosort can actually be graded.

WHY A NEW DATASET. Kilosort ships `simulation.hybrid_simulation`, which does
exactly this job, but it is the author's own research script with hard-coded
Linux paths (`/home/carsen/...`, `/media/carsen/ssd1/...`) rather than a
usable API. The approach is the right one though, and this follows it, using
the already-tested pieces in ground_truth_harness.py.

THE DESIGN
  background : a real stretch of this session's recording, all 384 channels,
               untouched. It keeps every real spike, so the sorter faces
               realistic clutter and realistic correlated noise.
  injected   : real multi-channel snippets from real units of this same
               session, copied to a DIFFERENT set of channels (shifted by
               CHANNEL_OFFSET) and placed at times we choose. Using real
               snippets keeps the genuine spike-to-spike amplitude and shape
               variability -- the thing whose absence made an earlier
               synthetic test pass and then fail on real data (log 5w).
  truth      : every injected spike's time, which injected unit it came from,
               its destination channels, and its known sub-sample offset.

WHY MOVE THEM TO DIFFERENT CHANNELS. If an injected unit sat on its original
channels it would overlay the real unit it was copied from, and a detection
could not be attributed to one or the other. Moving it by CHANNEL_OFFSET
puts it somewhere the source neuron never fired, so any spike found there at
an injected time is unambiguously ours.

WHAT IT IS SCORED ON. Only the injected units. The background's own units
have no ground truth and are deliberately not graded -- they are there to
make the problem realistic, not to be counted.

HONEST LIMITS
  * Injected spikes are added linearly. Real potentials superpose, so this is
    fair, but it tests the sorter, not the biophysics.
  * The injected firing pattern here is a random Poisson-like train with a
    refractory floor, not a realistic bursting pattern, so anything that
    depends on burst structure (amplitude recovery, for one) is not tested.
  * Injected spikes are placed at least MIN_SEP apart, so this dataset does
    not test collision handling between injected units. Given that tight
    collisions turned out to be rare in this recording anyway (12 in 1,458
    spikes for unit 408, log 5w), that is a deliberate simplification.

Usage: python build_small_hybrid_dataset.py
"""
import os
import json
import time
import numpy as np
from scipy.signal import butter, filtfilt

from ground_truth_harness import (extract_real_snippets, shift_multichannel,
                                   inject_multichannel)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT_DIR = r"D:\Gil\spike_sorting_agent\hybrid_small"
N_CHAN_BIN, FS, NT0MIN, NT, ITEMSIZE = 384, 30000.0, 20, 61, 2

BG_START = 60_000_000          # where in the real recording the background comes from
DURATION_S = 120.0
CHUNK = 1_000_000
SOURCE_UNITS = [440, 408, 439, 302, 31]   # span good and poor quality on purpose
CHANNEL_OFFSET = 40           # move injected units this many channels away
RATE_HZ = 6.0                 # injected firing rate per unit
MIN_SEP = 120                 # samples between injected spikes (any unit)
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
assert BG_END <= TOTAL, f"background window runs past the file ({BG_END} > {TOTAL})"

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
rng = np.random.default_rng(SEED)

print(f"background: samples {BG_START:,}-{BG_END:,} "
      f"({DURATION_S:.0f}s, {N_SAMPLES:,} samples, {N_CHAN_BIN} channels)")
print(f"output: {OUT_DAT}")
print(f"size will be {N_SAMPLES * N_CHAN_BIN * ITEMSIZE / 1e9:.2f} GB\n")

# ---------------------------------------------------------------- probe file
probe = dict(chanMap=list(range(N_CHAN_BIN)),
             xc=[float(v) for v in channel_positions[:, 0]],
             yc=[float(v) for v in channel_positions[:, 1]],
             kcoords=[0] * N_CHAN_BIN, n_chan=N_CHAN_BIN)
with open(OUT_PROBE, "w") as f:
    json.dump(probe, f)
print(f"wrote probe for {N_CHAN_BIN} channels -> {OUT_PROBE}")

# ---------------------------------------------- collect real snippets per unit
print("\ncollecting real multi-channel snippets from source units:")
sources = {}
for uid in SOURCE_UNITS:
    pc = int(peak_ch_all[uid])
    d = np.sqrt(((channel_positions - channel_positions[pc]) ** 2).sum(axis=1))
    src_ch = np.sort(np.where(d <= FOOTPRINT_RADIUS_UM)[0])
    dst_ch = src_ch + CHANNEL_OFFSET
    if dst_ch.max() >= N_CHAN_BIN:
        dst_ch = src_ch - CHANNEL_OFFSET
    if dst_ch.min() < 0:
        print(f"  unit {uid}: cannot relocate footprint, skipping")
        continue

    st = np.sort(spike_times[spike_clusters == uid])
    st = st[(st > NT + 10) & (st < TOTAL - NT - 10)]
    if len(st) < 50:
        print(f"  unit {uid}: only {len(st)} spikes, skipping")
        continue
    pick = st[rng.permutation(len(st))[:N_SNIPPET_POOL]]

    # read each snippet's own little block (spikes are spread over 210 min)
    snips = []
    for sp in pick:
        lo = int(sp) - NT0MIN - 60
        n = NT + 120
        if lo < 0 or lo + n > TOTAL:
            continue
        with open(DAT_PATH, "rb") as f:
            f.seek(lo * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(n * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(n, N_CHAN_BIN)
        w = np.stack([filtfilt(b_hp, a_hp, blk[:, c].astype(np.float64))[60:60 + NT]
                      for c in src_ch], axis=1)
        snips.append(w)
        if len(snips) >= N_SNIPPET_POOL:
            break
    if not snips:
        print(f"  unit {uid}: no usable snippets, skipping")
        continue
    snips = np.stack(snips)                      # (n, NT, n_ch)
    amp = np.abs(snips.mean(axis=0)).max()
    sources[uid] = dict(snippets=snips, src_ch=src_ch, dst_ch=dst_ch,
                        peak_ch=pc, dst_peak=int(pc + (dst_ch[0] - src_ch[0])))
    print(f"  unit {uid}: {len(snips)} snippets, {len(src_ch)} channels "
          f"ch{src_ch[0]}-{src_ch[-1]} -> ch{dst_ch[0]}-{dst_ch[-1]}, "
          f"mean |peak| {amp:.1f} ADU")

assert sources, "no source units usable"

# ------------------------------------------------- build the injection schedule
print("\nbuilding injection schedule:")
all_times, all_labels, all_shifts, all_snipidx = [], [], [], []
for uid, S in sources.items():
    n_spk = int(RATE_HZ * DURATION_S)
    # random times with a refractory floor, kept clear of the trace edges
    cand = np.sort(rng.integers(NT + MIN_SEP, N_SAMPLES - NT - MIN_SEP, size=n_spk * 3))
    keep = [cand[0]]
    for t in cand[1:]:
        if t - keep[-1] >= MIN_SEP:
            keep.append(int(t))
        if len(keep) >= n_spk:
            break
    t_u = np.asarray(keep, dtype=np.int64)
    all_times.append(t_u)
    all_labels.append(np.full(len(t_u), uid, dtype=np.int64))
    all_shifts.append(rng.uniform(-0.5, 0.5, size=len(t_u)))
    all_snipidx.append(rng.integers(0, len(S["snippets"]), size=len(t_u)))
    print(f"  unit {uid}: {len(t_u)} spikes scheduled")

times = np.concatenate(all_times)
labels = np.concatenate(all_labels)
shifts = np.concatenate(all_shifts)
snipidx = np.concatenate(all_snipidx)
o = np.argsort(times)
times, labels, shifts, snipidx = times[o], labels[o], shifts[o], snipidx[o]

# enforce separation ACROSS units too, so no injected pair collides
keep = np.ones(len(times), dtype=bool)
last = -10 ** 9
for i, t in enumerate(times):
    if t - last < MIN_SEP:
        keep[i] = False
    else:
        last = t
times, labels, shifts, snipidx = times[keep], labels[keep], shifts[keep], snipidx[keep]
print(f"  total after enforcing {MIN_SEP}-sample separation: {len(times)} spikes")

# ------------------------------------------------------- write the hybrid file
print("\nstreaming background + injections to disk:")
t0 = time.time()
written = 0
n_injected = 0
with open(OUT_DAT, "wb") as fout:
    pos = 0
    while pos < N_SAMPLES:
        clen = min(CHUNK, N_SAMPLES - pos)
        # read with margin so a spike straddling the boundary is handled in
        # the chunk that owns its start, with room for its full length
        with open(DAT_PATH, "rb") as f:
            f.seek((BG_START + pos) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(clen * N_CHAN_BIN * ITEMSIZE)
        chunk = np.frombuffer(raw, dtype=np.int16).reshape(clen, N_CHAN_BIN).astype(np.float64)

        sel = np.where((times >= pos) & (times < pos + clen))[0]
        for i in sel:
            S = sources[int(labels[i])]
            w = shift_multichannel(S["snippets"][snipidx[i]], float(shifts[i]))
            if inject_multichannel(chunk, w, int(times[i]) - pos, S["dst_ch"],
                                   nt0min=NT0MIN):
                n_injected += 1
        np.clip(chunk, -2 ** 15 + 1, 2 ** 15 - 1).astype(np.int16).tofile(fout)
        written += clen
        pos += clen
        if (pos // CHUNK) % 10 == 0:
            print(f"  {pos:,}/{N_SAMPLES:,} samples "
                  f"({100*pos/N_SAMPLES:.0f}%), {n_injected} injected so far")

print(f"  wrote {written:,} samples in {time.time()-t0:.0f}s, "
      f"{n_injected} spikes injected "
      f"({len(times)-n_injected} dropped at chunk edges)")

# spikes that fell too near a chunk edge were skipped -- the truth table must
# record only what actually went in, or recall is understated forever
fits = np.array([((t % CHUNK) + NT - NT0MIN) <= CHUNK and t >= NT0MIN
                 for t in times])
np.savez(OUT_TRUTH, times=times[fits], labels=labels[fits],
         shifts=shifts[fits], snippet_index=snipidx[fits],
         source_units=np.array(list(sources.keys())),
         dst_peak_channels=np.array([sources[u]["dst_peak"] for u in sources]),
         bg_start=BG_START, n_samples=N_SAMPLES, fs=FS,
         n_chan_bin=N_CHAN_BIN, nt0min=NT0MIN, channel_offset=CHANNEL_OFFSET)

print(f"\ntruth table: {int(fits.sum())} spikes across {len(sources)} injected units")
for u in sources:
    print(f"  injected unit {u}: {int((labels[fits]==u).sum())} spikes, "
          f"destination peak channel {sources[u]['dst_peak']}")
print(f"\nsaved:\n  {OUT_DAT}\n  {OUT_TRUTH}\n  {OUT_PROBE}")
print(f"file size: {os.path.getsize(OUT_DAT)/1e9:.2f} GB")
