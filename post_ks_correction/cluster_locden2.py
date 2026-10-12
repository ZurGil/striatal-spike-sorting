"""LOCDEN stage 3, iteration 2: BAND-WISE density clustering with a merge pass.

WHY THE FIRST VERSION FAILED, measured not guessed. Iteration 1 of this stage
ran a single global HDBSCAN over the whole probe and scored recall 0.519 /
precision 0.134 against stock Kilosort's 0.882 / 0.710, while the detector
feeding it had a detection recall of 0.978. So the loss was entirely here.
Two causes, both visible in the output:

  UNDER-CLUSTERING  134 clusters where Kilosort finds 216. A single HDBSCAN
                    over a 3.8 mm probe finds broad density modes, not
                    neurons; everything within a mode merges, which is
                    precisely the 0.134 precision.
  TOO FEW FEATURES  3 PCs of the PEAK CHANNEL waveform, where Kilosort uses
                    ~10 channels x 6 features. The across-channel information
                    5aw proved carries identity was present in the cache (the
                    20-channel amplitude vector) and simply unused.

Both are fixed here:

  1. CLUSTER IN DEPTH BANDS. Each band is 8 channels (~80 um) of peak channel,
     clustered with a margin of 8 channels on each side for context. Inside a
     band there are a handful of units rather than two hundred, so the local
     density structure is about neurons.
  2. USE THE FOOTPRINT. The 20-channel peak-to-peak vector, L2-normalised to a
     pure shape, enters as PCs alongside the fitted depth. This is the same
     quantity whose addition to Kilosort's kNN graph produced the only real
     win in the project (5aw, +4.8 recall at p=0.001).
  3. MERGE ACROSS BANDS. A unit whose peak channel straddles a band boundary
     would otherwise be split in two, which costs recall and inflates the
     fragment count. Clusters in adjacent bands are merged when their mean
     footprints agree on the union of their channels AND their mean waveforms
     agree AND their depths are close -- all three, because footprint alone is
     a weak discriminator (5av: 0.876 to the target vs 0.976 to the best
     competitor).

Usage:
  python cluster_locden2.py <rep> <name> [key=value ...]
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

BAND = 8              # channels of peak channel per band (~80 um)
MARGIN = 8            # context channels on each side, clustered but not kept
SCALE_Y = 12.0        # um per feature unit, fitted depth
SCALE_Z = 40.0        # um per feature unit, distance from probe
W_LOGAMP = 1.0
W_FP = 1.5            # weight on each footprint PC
W_WAVE = 0.8          # weight on each peak-channel waveform PC
N_FP_PC = 4
N_WAVE_PC = 3
MIN_CLUSTER = 30
MIN_SAMPLES = 8
REFRACTORY = 20       # samples, per-cluster duplicate removal
MERGE_FP = 0.97       # footprint cosine required to merge across bands
MERGE_WAVE = 0.95     # waveform cosine required
MERGE_DY = 15.0       # um of fitted depth
MIN_SPIKES = 30       # drop clusters smaller than this after merging


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def main(rep, name, **kw):
    for k, v in kw.items():
        globals()[k.upper()] = type(globals()[k.upper()])(float(v))
        print("  [override] %s = %s" % (k.upper(), globals()[k.upper()]))

    ev = np.load(os.path.join(cache_dir(rep), "events2.npz"))
    lc = np.load(os.path.join(cache_dir(rep), "loc_events2.npz"))
    t, ch, wave, qual = ev["t"], ev["ch"].astype(np.int64), ev["wave"], ev["qual"]
    amp, nb, order = ev["amp"], ev["nb"], ev["order"]
    y, z, amax = lc["y"], lc["z"], lc["amp_max"]
    n = len(t)

    pio = np.zeros(N_CHAN_BIN, dtype=np.int64)
    pio[order] = np.arange(len(order))
    depth_idx = pio[ch]                      # 0 = deepest channel

    # ---------------------------------------------------------- features
    fp = _unit(amp.astype(np.float64))       # the footprint: a pure shape
    w = wave.astype(np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    w = _unit(w)
    fp_pc = PCA(n_components=N_FP_PC, random_state=0).fit_transform(fp)
    fp_pc /= fp_pc.std(axis=0, keepdims=True)
    wv_pc = PCA(n_components=N_WAVE_PC, random_state=0).fit_transform(w)
    wv_pc /= wv_pc.std(axis=0, keepdims=True)

    F = np.column_stack(
        [y / SCALE_Y, np.clip(z, 1, 250) / SCALE_Z,
         W_LOGAMP * np.log10(np.maximum(amax, 1.0))]
        + [W_FP * fp_pc[:, j] for j in range(N_FP_PC)]
        + [W_WAVE * wv_pc[:, j] for j in range(N_WAVE_PC)]).astype(np.float64)
    print("  %d events, %d feature dims" % (n, F.shape[1]))

    # ------------------------------------------------------ band clustering
    labels = np.full(n, -1, dtype=np.int64)
    next_id = 0
    t0 = time.time()
    for c0 in range(0, N_CHAN_BIN, BAND):
        core = (depth_idx >= c0) & (depth_idx < c0 + BAND)
        ctx = ((depth_idx >= c0 - MARGIN) & (depth_idx < c0 + BAND + MARGIN))
        if core.sum() < MIN_CLUSTER:
            continue
        ii = np.where(ctx)[0]
        lab = HDBSCAN(min_cluster_size=int(MIN_CLUSTER),
                      min_samples=int(MIN_SAMPLES)).fit_predict(F[ii])
        keep = core[ii] & (lab >= 0)
        if keep.sum() == 0:
            continue
        u, inv = np.unique(lab[keep], return_inverse=True)
        labels[ii[keep]] = next_id + inv
        next_id += len(u)
    print("  band clustering: %d clusters over %d bands in %.0fs"
          % (next_id, N_CHAN_BIN // BAND, time.time() - t0))
    if next_id == 0:
        raise SystemExit("no clusters found")

    # --------------------------------------------- templates for the merge
    K = next_id
    fp_full = np.zeros((K, N_CHAN_BIN))      # footprint in ABSOLUTE channels
    wv_mean = np.zeros((K, wave.shape[1]))
    y_mean = np.zeros(K)
    size = np.zeros(K, dtype=np.int64)
    for c in range(K):
        m = np.where(labels == c)[0]
        size[c] = len(m)
        if not len(m):
            continue
        acc = np.zeros(N_CHAN_BIN)
        cnt = np.zeros(N_CHAN_BIN)
        np.add.at(acc, nb[ch[m]].ravel(), amp[m].ravel())
        np.add.at(cnt, nb[ch[m]].ravel(), 1.0)
        fp_full[c] = acc / np.maximum(cnt, 1)
        wv_mean[c] = w[m].mean(axis=0)
        y_mean[c] = np.median(y[m])

    # merge adjacent-in-depth clusters that agree on all three criteria
    parent = np.arange(K)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cen_depth = np.array([np.median(depth_idx[labels == c]) if size[c] else 1e9
                          for c in range(K)])
    n_merge = 0
    for i in range(K):
        for j in range(i + 1, K):
            if not size[i] or not size[j]:
                continue
            if abs(cen_depth[i] - cen_depth[j]) > BAND + MARGIN:
                continue
            sh = (fp_full[i] > 0) & (fp_full[j] > 0)
            if sh.sum() < 6:
                continue
            cf = float(np.dot(_unit(fp_full[i][sh]), _unit(fp_full[j][sh])))
            cw = float(np.dot(_unit(wv_mean[i]), _unit(wv_mean[j])))
            if (cf >= MERGE_FP and cw >= MERGE_WAVE
                    and abs(y_mean[i] - y_mean[j]) <= MERGE_DY):
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                    n_merge += 1
    root = np.array([find(c) for c in range(K)])
    remap = {r: i for i, r in enumerate(np.unique(root[size > 0]))}
    newlab = np.full(n, -1, dtype=np.int64)
    ok = labels >= 0
    newlab[ok] = [remap[root[c]] for c in labels[ok]]
    labels = newlab
    K2 = len(remap)
    print("  merge pass: %d merges, %d -> %d clusters" % (n_merge, K, K2))

    # -------------------------------------- per-cluster refractory dedup
    keep = np.zeros(n, bool)
    n_dup = 0
    for c in range(K2):
        m = np.where(labels == c)[0]
        if not len(m):
            continue
        o = m[np.argsort(t[m])]
        last, best = -10 ** 9, -1
        for i in o:
            if t[i] - last < REFRACTORY:
                n_dup += 1
                if best >= 0 and qual[i] > qual[best]:
                    keep[best] = False
                    keep[i] = True
                    best, last = i, t[i]
                continue
            keep[i] = True
            best, last = i, t[i]
    sel = keep & (labels >= 0)
    print("  refractory rule removed %d duplicates (%.1f%% of all events)"
          % (n_dup, 100.0 * n_dup / max(n, 1)))

    # drop clusters too small to be a unit in 120 s
    cl = labels[sel]
    cnt = np.bincount(cl, minlength=K2)
    big = cnt >= MIN_SPIKES
    sel2 = sel.copy()
    sel2[sel] = big[cl]
    cl = labels[sel2]
    u, cl = np.unique(cl, return_inverse=True)
    st = t[sel2].astype(np.int64)
    o = np.argsort(st)

    out = os.path.join(data_dir(rep), "ks_" + name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "spike_times.npy"), st[o])
    np.save(os.path.join(out, "spike_clusters.npy"), cl[o].astype(np.int32))
    sz = np.bincount(cl)
    print("  WROTE %d spikes in %d clusters -> ks_%s" % (len(st), len(u), name))
    print("  cluster sizes: median %d, min %d, max %d"
          % (int(np.median(sz)), int(sz.min()), int(sz.max())))
    print("  rate %.2f spikes/s/channel (Kilosort vanilla: 4.87, 216 clusters)"
          % (len(st) / 120.0 / N_CHAN_BIN))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]
    kw = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    main(int(a[0]), a[1], **kw)
