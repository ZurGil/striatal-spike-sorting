"""How big ARE the injected spikes, and was 0.556 detection recall real?

Two questions the first LOCDEN run raised, both of which decide whether a
threshold detector can work at all on this benchmark.

1. BASE RATE. "55.6% of injected spikes had a detected event within 10
   samples" was computed against events ANYWHERE on the probe. With 90,664
   events over 3.6 M samples, a random time has an event within +/-10 samples
   with probability ~ 90664 * 21 / 3.6e6 = 0.53. So the number may be entirely
   chance. Measured here by scrambling the truth times into matched random
   times, which is the project's rule: chance is measured, never assumed.

2. AMPLITUDE. The diagnostic found the median injected spike reaches only
   z = -3.2 on its own destination channel. If that is right, a single-channel
   threshold detector cannot see these units at any useful purity, and the
   reason Kilosort reaches 0.88 recall on the easy tier is that whitened
   multichannel template matching integrates over ~10 channels and ~60
   samples. Measured here as the spike-triggered average on the destination
   channel (the deterministic signal) against the single-trial noise.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import (ROOT, FS, N_CHAN_BIN, GAIN_TO_UV, HP, LP, CHUNK,
                           PAD, data_dir, cache_dir, probe)

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NT, PRE = 61, 20

cm, X, Y = probe(REP)
T = np.load(os.path.join(data_dir(REP), "hybrid_truth.npz"), allow_pickle=True)
tt, lab = T["times"].ravel(), T["labels"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]
su = [int(x) for x in T["source_units"]]
dst = {u: int(c) for u, c in zip(su, T["dst_peak_channels"])}
unit_tier = {u: tier_names[k] for u, k in zip(su, T["unit_tier"].ravel())}

ev = np.load(os.path.join(cache_dir(REP), "events.npz"))
noise, det_t = ev["noise"], np.sort(ev["t"])
n_samples = int(T["n_samples"])

# ---------------------------------------------------------------- base rate
rng = np.random.default_rng(0)


def frac_near(times, tol=10):
    j = np.searchsorted(det_t, times)
    out = np.zeros(len(times), bool)
    for i, (x, k) in enumerate(zip(times, j)):
        c = [det_t[m] for m in (k - 1, k) if 0 <= m < len(det_t)]
        out[i] = bool(c) and min(abs(y - x) for y in c) <= tol
    return out.mean()


real = frac_near(tt)
sham = [frac_near(rng.integers(1000, n_samples - 1000, len(tt)))
        for _ in range(10)]
print("=" * 78)
print("BASE RATE for 'an event within 10 samples somewhere on the probe'")
print("=" * 78)
print("  at injected times      %.3f" % real)
print("  at matched RANDOM times %.3f  (10 draws, sd %.4f)"
      % (np.mean(sham), np.std(sham)))
print("  excess over chance      %+.3f" % (real - np.mean(sham)))

# ----------------------------------------------------------- amplitude / SNR
acc = {u: np.zeros(NT) for u in su}
cnt = {u: 0 for u in su}
trough = {u: [] for u in su}
sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")
raw = np.memmap(os.path.join(data_dir(REP), "hybrid.bin"), dtype=np.int16,
                mode="r").reshape(-1, N_CHAN_BIN)

for start in range(0, n_samples, CHUNK):
    stop = min(start + CHUNK, n_samples)
    sel = np.where((tt >= start + PRE) & (tt < stop - NT))[0]
    if len(sel) == 0:
        continue
    lo, hi = max(0, start - PAD), min(n_samples, stop + PAD)
    x = raw[lo:hi, :].astype(np.float32) * GAIN_TO_UV
    x -= np.median(x, axis=1, keepdims=True)
    x = sosfiltfilt(sos, x, axis=0).astype(np.float32)
    for i in sel:
        u = int(lab[i])
        a = int(tt[i] - lo) - PRE
        if a < 0 or a + NT > x.shape[0]:
            continue
        w = x[a:a + NT, dst[u]]
        acc[u] += w
        cnt[u] += 1
        trough[u].append(float(w.min()))
    del x
    print("  %6.1fs" % (stop / FS), flush=True)

print("\n" + "=" * 78)
print("INJECTED UNIT AMPLITUDE on its own destination channel (rep%d)" % REP)
print("=" * 78)
print("%5s %-10s %5s %9s %9s %8s %9s"
      % ("unit", "tier", "n", "STA uV", "noise uV", "STA/sd", "trial z"))
rows = []
for u in su:
    if cnt[u] < 10:
        continue
    sta = acc[u] / cnt[u]
    sta -= sta.mean()
    amp = float(-sta.min())
    nz = float(noise[dst[u]])
    tz = float(np.median(trough[u])) / nz
    rows.append((u, unit_tier[u], cnt[u], amp, nz, amp / nz, tz))
    print("%5d %-10s %5d %9.1f %9.2f %8.2f %9.2f"
          % (u, unit_tier[u], cnt[u], amp, nz, amp / nz, tz))
r = np.array([x[5] for x in rows])
print("\n  STA trough / noise sd: median %.2f, range %.2f-%.2f"
      % (np.median(r), r.min(), r.max()))
print("  A single-channel detector needs the SINGLE-TRIAL trough to clear the")
print("  threshold; the STA/sd column is the best case, with noise averaged")
print("  over n trials, and even that is the number above.")
