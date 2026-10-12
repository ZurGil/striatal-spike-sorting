"""LOCDEN stage 3: density clustering in a localization feature space.

This is the half of LOCDEN that is not Kilosort's clustering. Kilosort seeds
nclust=200 k-means centres (hard-coded, absent from parameters.py -- 5aa) and
then refines on a kNN graph. 5aw measured that handing that seeding extra
feature dimensions CANCELS part of the graph gain, which is what an
over-seeded k-means does: it cuts along the direction of greatest within-unit
spread, slicing a unimodal unit in two.

HDBSCAN instead:
  * has no k, so nothing forces 200 groups onto the data
  * finds clusters of differing density, which matters because a loud unit and
    a near-threshold unit have very different local densities
  * labels unassignable points -1 instead of forcing them into the nearest
    centre, so contamination is dropped rather than misfiled

The feature space is the other half of the idea. Instead of 10 energy ratios
compared by a compressed cosine (5av: 0.876 to the target vs 0.976 to the best
competitor), each event carries the fitted point-source depth y in MICROMETRES
-- a metric coordinate where distance means distance.

DUPLICATES. Detection deliberately used a +/-3 sample, +/-1 channel
local-maximum footprint, because anything larger let a loud spike suppress a
quiet neighbour and pinned detection recall at 0.476. That admits several
events per spike. They are removed HERE, per cluster, by a refractory rule:
two detections of one spike land in the same cluster, whereas two different
neurons' spikes do not, so this is the stage where the distinction is
available.

Writes spike_times.npy and spike_clusters.npy into
hybrid_v2_rep{r}/ks_{name}/, which run_v2_comparison.score() reads -- so this
is scored by the identical rule as every Kilosort configuration.

Usage:
  python cluster_locden.py <rep> <name> [key=value ...]
"""
import os
import sys
import time

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import FS, N_CHAN_BIN, data_dir, cache_dir

# ---- feature scaling. Everything is divided into "how many micrometres is
# ---- one unit of this feature worth", so the Euclidean metric HDBSCAN uses
# ---- is interpretable rather than arbitrary.
SCALE_Y = 12.0        # um per feature unit for fitted depth
SCALE_Z = 40.0        # um per feature unit for distance from the probe
W_LOGAMP = 1.0        # weight on log10 peak-to-peak amplitude
W_WAVE = 0.6          # weight on each standardized waveform PC
N_WAVE_PC = 3
MIN_CLUSTER = 40      # >= this many spikes in 120 s (0.33 Hz)
MIN_SAMPLES = 10
MAX_FIT = 150000      # HDBSCAN is fitted on at most this many events
REFRACTORY = 20       # samples (0.67 ms) for the per-cluster duplicate rule
MIN_R2 = -1.0         # drop events whose point-source fit is worse than this
USE_Z = 1
USE_WAVE = 1
ASSIGN_MAX_D = 3.0    # feature-space distance beyond which a held-out event is
                      # left unassigned rather than forced into a cluster


def main(rep, name, **kw):
    for k, v in kw.items():
        globals()[k.upper()] = type(globals()[k.upper()])(float(v))
        print("  [override] %s = %s" % (k.upper(), globals()[k.upper()]))

    ev = np.load(os.path.join(cache_dir(rep), "events2.npz"))
    lc = np.load(os.path.join(cache_dir(rep), "loc_events2.npz"))
    t, ch, wave, qual = ev["t"], ev["ch"].astype(np.int64), ev["wave"], ev["qual"]
    y, z, r2, amax = lc["y"], lc["z"], lc["r2"], lc["amp_max"]
    n = len(t)

    good = np.isfinite(y) & np.isfinite(z) & (r2 >= MIN_R2)
    print("  %d of %d events kept by the fit-quality gate (R2 >= %.1f)"
          % (good.sum(), n, MIN_R2))

    # ------------------------------------------------------------- features
    cols = [y / SCALE_Y]
    if USE_Z:
        cols.append(np.clip(z, 1, 250) / SCALE_Z)
    cols.append(W_LOGAMP * np.log10(np.maximum(amax, 1.0)))
    if USE_WAVE:
        w = wave - wave.mean(axis=1, keepdims=True)
        nr = np.linalg.norm(w, axis=1)
        w = w / np.maximum(nr, 1e-9)[:, None]
        pcs = PCA(n_components=N_WAVE_PC, random_state=0).fit_transform(w)
        pcs = pcs / pcs.std(axis=0, keepdims=True)
        for j in range(N_WAVE_PC):
            cols.append(W_WAVE * pcs[:, j])
    F = np.ascontiguousarray(np.stack(cols, axis=1).astype(np.float64))
    print("  feature space: %d dims %s" % (F.shape[1], F.shape))

    idx = np.where(good)[0]
    rng = np.random.default_rng(0)
    fit_idx = (idx if len(idx) <= MAX_FIT
               else rng.choice(idx, MAX_FIT, replace=False))
    fit_idx = np.sort(fit_idx)

    t0 = time.time()
    hdb = HDBSCAN(min_cluster_size=int(MIN_CLUSTER),
                  min_samples=int(MIN_SAMPLES), store_centers="centroid")
    lab_fit = hdb.fit_predict(F[fit_idx])
    k = lab_fit.max() + 1
    print("  HDBSCAN: %d clusters, %.1f%% noise, on %d events in %.0fs"
          % (k, 100.0 * (lab_fit == -1).mean(), len(fit_idx), time.time() - t0))
    if k == 0:
        raise SystemExit("no clusters found")

    # ------------------------------- assign every event to the nearest centre
    cen = hdb.centroids_
    labels = np.full(n, -1, dtype=np.int64)
    labels[fit_idx] = lab_fit
    rest = np.setdiff1d(idx, fit_idx)
    if len(rest):
        for a in range(0, len(rest), 20000):
            b = rest[a:a + 20000]
            d = ((F[b][:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
            j = np.argmin(d, axis=1)
            dd = np.sqrt(d[np.arange(len(b)), j])
            labels[b] = np.where(dd <= ASSIGN_MAX_D, j, -1)
        print("  assigned %d held-out events (%.1f%% left unassigned)"
              % (len(rest), 100.0 * (labels[rest] == -1).mean()))

    # --------------------------------------- per-cluster refractory dedup
    keep = np.zeros(n, bool)
    n_dup = 0
    for c in range(k):
        m = np.where(labels == c)[0]
        if len(m) == 0:
            continue
        o = m[np.argsort(t[m])]
        last = -10 ** 9
        best = -1
        for i in o:
            if t[i] - last < REFRACTORY:
                # same spike detected twice: keep the stronger statistic
                if best >= 0 and qual[i] > qual[best]:
                    keep[best] = False
                    keep[i] = True
                    best = i
                    last = t[i]
                n_dup += 1
                continue
            keep[i] = True
            best = i
            last = t[i]
    sel = keep & (labels >= 0)
    print("  refractory rule removed %d duplicate detections (%.1f%%)"
          % (n_dup, 100.0 * n_dup / max(len(t), 1)))

    st = t[sel].astype(np.int64)
    cl = labels[sel].astype(np.int32)
    o = np.argsort(st)
    out = os.path.join(data_dir(rep), "ks_" + name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "spike_times.npy"), st[o])
    np.save(os.path.join(out, "spike_clusters.npy"), cl[o])
    sizes = np.bincount(cl, minlength=k)
    print("  WROTE %d spikes in %d clusters -> %s"
          % (len(st), len(np.unique(cl)), out))
    print("  cluster sizes: median %d, min %d, max %d"
          % (int(np.median(sizes[sizes > 0])), int(sizes[sizes > 0].min()),
             int(sizes.max())))
    print("  rate %.2f spikes/s/channel (Kilosort vanilla: 4.87)"
          % (len(st) / 120.0 / N_CHAN_BIN))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]
    kw = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    main(int(a[0]), a[1], **kw)
