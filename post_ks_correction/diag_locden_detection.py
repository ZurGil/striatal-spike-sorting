"""Why does LOCDEN's detector miss 44% of the injected spikes?

Detection recall is the ceiling on everything downstream, so this is measured
before any clustering work. Two candidate causes, and they call for opposite
fixes, so they must be told apart rather than guessed at:

  THRESHOLD  the trough never reaches DETECT_Z robust SDs -> lower the
             threshold (costs purity)
  DEDUP      the trough is deep enough but is not the minimum of its
             +/-DEDUP_DT x +/-DEDUP_DC spatiotemporal window, because a
             larger nearby event suppressed it -> shrink the dedup footprint
             (costs duplicate events)

One pass over the recording, recording for every injected spike the deepest
z-score reached near its injection time on its destination channel
neighbourhood, and whether that sample was a local spatiotemporal minimum.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import minimum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import (ROOT, FS, N_CHAN_BIN, GAIN_TO_UV, HP, LP, CHUNK,
                           PAD, DEDUP_DT, DEDUP_DC, data_dir, cache_dir, probe)

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOL = 6          # samples around the injection time to look in
RADIUS_UM = 40.0  # channels around the destination peak channel to look on

cm, X, Y = probe(REP)
order = cm[np.lexsort((X[cm], Y[cm]))]
pos_in_order = np.full(N_CHAN_BIN, -1)
pos_in_order[order] = np.arange(len(order))

T = np.load(os.path.join(data_dir(REP), "hybrid_truth.npz"), allow_pickle=True)
tt, lab = T["times"].ravel(), T["labels"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]
su = [int(x) for x in T["source_units"]]
dst = {u: int(c) for u, c in zip(su, T["dst_peak_channels"])}

ev = np.load(os.path.join(cache_dir(REP), "events.npz"))
noise = ev["noise"]
det_t = np.sort(ev["t"])

sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")
raw = np.memmap(os.path.join(data_dir(REP), "hybrid.bin"), dtype=np.int16,
                mode="r").reshape(-1, N_CHAN_BIN)
n_samples = raw.shape[0]

best_z = np.full(len(tt), np.nan)      # deepest z near the injection
is_locmin = np.zeros(len(tt), bool)    # was that sample a local minimum?

for start in range(0, n_samples, CHUNK):
    stop = min(start + CHUNK, n_samples)
    sel = np.where((tt >= start) & (tt < stop))[0]
    if len(sel) == 0:
        continue
    lo, hi = max(0, start - PAD), min(n_samples, stop + PAD)
    x = raw[lo:hi, :].astype(np.float32) * GAIN_TO_UV
    x -= np.median(x, axis=1, keepdims=True)
    x = sosfiltfilt(sos, x, axis=0).astype(np.float32)
    z = x[:, order] / noise[order]
    mf = minimum_filter(z, size=(2 * DEDUP_DT + 1, 2 * DEDUP_DC + 1),
                        mode="nearest")
    for i in sel:
        c0 = dst[int(lab[i])]
        near = cm[np.sqrt((X[cm] - X[c0]) ** 2 + (Y[cm] - Y[c0]) ** 2) <= RADIUS_UM]
        cols = pos_in_order[near]
        a = int(tt[i] - lo) - TOL
        b = a + 2 * TOL + 1
        if a < 0 or b > z.shape[0]:
            continue
        w = z[a:b][:, cols]
        k = np.unravel_index(np.argmin(w), w.shape)
        best_z[i] = w[k]
        is_locmin[i] = bool(w[k] == mf[a:b][:, cols][k])
    del x, z, mf
    print("  %6.1fs done" % (stop / FS), flush=True)

ok = np.isfinite(best_z)
hit = np.zeros(len(tt), bool)
for i, x in enumerate(tt):
    j = np.searchsorted(det_t, x)
    c = [det_t[k] for k in (j - 1, j) if 0 <= k < len(det_t)]
    hit[i] = bool(c) and min(abs(y - x) for y in c) <= 10

np.savez(os.path.join(cache_dir(REP), "diag_detection.npz"),
         best_z=best_z, is_locmin=is_locmin, hit=hit, tier=tier_code)

print("\n" + "=" * 78)
print("WHY INJECTED SPIKES ARE MISSED (rep%d, %d truth spikes)" % (REP, ok.sum()))
print("=" * 78)
print("%-11s %6s %7s %8s %8s %9s %9s"
      % ("tier", "n", "detect", "medianZ", "|Z|>=5", "locmin", "lost2dedup"))
for k, name in enumerate(tier_names):
    s = ok & (tier_code == k)
    if not s.sum():
        continue
    deep = s & (best_z <= -5.0)
    print("%-11s %6d %7.3f %8.2f %8.3f %9.3f %9.3f"
          % (name, s.sum(), hit[s].mean(), np.median(best_z[s]),
             deep.sum() / s.sum(), is_locmin[s].mean(),
             (deep & ~is_locmin).sum() / max(deep.sum(), 1)))
s = ok
for thr in (3.0, 3.5, 4.0, 4.5, 5.0, 6.0):
    print("  threshold Z=%.1f would reach %.3f of injected spikes"
          % (thr, (best_z[s] <= -thr).mean()))
print("\nof the spikes deep enough at Z=5 but NOT detected, %d of %d were not a "
      "local minimum (dedup suppressed)"
      % (int(((best_z <= -5) & ~hit & ~is_locmin).sum()),
         int(((best_z <= -5) & ~hit).sum())))
np.savez(os.path.join(cache_dir(REP), "diag_detection.npz"),
         best_z=best_z, is_locmin=is_locmin, hit=hit, tier=tier_code)
