"""Cluster the multi-axis representation with ISO-SPLIT instead of HDBSCAN.

ISO-SPLIT is MountainSort's clustering algorithm and it is available directly
in the si_ms5 environment as `isosplit6`. It is the right tool for this
project's specific diagnosed failure, for a reason that is measurable rather
than aesthetic:

  5aa found Kilosort's nclust=200 is hard-coded and absent from
  parameters.py, and 5aw found that handing that seeding extra feature
  dimensions CANCELS part of the graph gain -- the signature of an
  over-seeded k-means slicing unimodal units along their direction of
  greatest spread.

  ISO-SPLIT takes no k and no min_cluster_size. It splits a pair of candidate
  clusters only when the projection onto the line between them FAILS a
  non-parametric unimodality test (isotonic regression / dip). So it splits
  where the data says there are two modes and refuses otherwise.

Verified before use rather than assumed: on 200,000 points of pure Gaussian
noise in 10 dimensions it returns ONE cluster, not 200 -- the exact opposite
failure mode from the one 5aa identified -- and it recovers 5 unequal
well-separated blobs at ARI 1.000 in 0.06 s. It is also ~35x faster than the
HDBSCAN run it replaces (4.6 s for 200k x 10, against 170 s for 150k x 6).

Axes are chosen from features_locden.py's named set. The wavelet sub-sample
shift is NOT used and the shape axes are not de-jittered, because
diag_wavelet_alignment_truth.py measured that alignment at chance
(correlation -0.061 with the known injected shifts, RMSE 1.402 against 0.292
for answering zero). Shape axes are still measured relative to each event's
OWN integer trough, so they are unaffected by the detector's -2 sample
systematic offset.

Must be run in the si_ms5 environment (D:/conda_envs/si_ms5/python.exe),
which is where isosplit6 lives.

Usage:
  D:/conda_envs/si_ms5/python.exe cluster_isosplit.py <rep> <name> [key=val ...]
"""
import os
import sys

import numpy as np

ROOT = r"D:\Gil\spike_sorting_agent"
N_CHAN_BIN = 384

# axis -> weight after robust standardisation. Depth carries more weight
# because it is the one axis in micrometres on a probe that is effectively
# one-dimensional; everything else is shape or frequency.
WEIGHTS = {
    "fit_y_um": 2.5,
    "fit_z_um": 0.7,
    "ptp_uv": 1.2,
    "trough_to_peak_samp": 1.0,
    "fwhm_samp": 0.8,
    "peak_trough_asym": 0.8,
    "repol_slope": 0.5,
    "f0_hz": 0.8,
    "spectral_centroid_hz": 0.6,
    "fp_spread_um": 0.6,
    "fp_asym": 0.8,
    "n_chan_half": 0.5,
    "wave_kurtosis": 0.4,
}
LOG_AXES = ("ptp_uv", "energy", "fit_A")
N_FP_PC = 4
W_FP = 1.2            # weight on each footprint PC
BAND = 12             # channels per clustering band; 0 = one global run
MARGIN = 6
REFRACTORY = 20
MIN_SPIKES = 30


def robust_z(v):
    med = np.median(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    return (v - med) / (iqr / 1.349 if iqr > 0 else (v.std() or 1.0))


def main(rep, name, **kw):
    for k, v in kw.items():
        K = k.upper()
        if K in globals():
            globals()[K] = type(globals()[K])(float(v))
            print("  [override] %s = %s" % (K, globals()[K]))
        elif k in WEIGHTS:
            WEIGHTS[k] = float(v)
            print("  [override] weight %s = %s" % (k, WEIGHTS[k]))

    from isosplit6 import isosplit6

    d = os.path.join(ROOT, "hybrid_v2_rep%d" % rep, "ks_locden_cache")
    ev = np.load(os.path.join(d, "events2.npz"))
    ft = np.load(os.path.join(d, "feat.npz"))
    names = [str(x) for x in ft["names"]]
    F0 = ft["F"].astype(np.float64)
    done = ft["done"]
    t, ch, amp, order = (ev["t"], ev["ch"].astype(np.int64), ev["amp"],
                         ev["order"])
    qual = ev["qual"]
    n = len(t)

    pio = np.zeros(N_CHAN_BIN, dtype=np.int64)
    pio[order] = np.arange(len(order))
    depth_idx = pio[ch]

    cols, used = [], []
    for ax, wt in WEIGHTS.items():
        if ax not in names:
            print("  (no axis %s)" % ax)
            continue
        v = F0[:, names.index(ax)].copy()
        if ax in LOG_AXES:
            v = np.log10(np.maximum(v, 1e-3))
        cols.append(wt * robust_z(v))
        used.append("%s x%.1f" % (ax, wt))

    # the footprint, as PCs of the L2-normalised 20-channel amplitude vector.
    # This is the quantity whose addition to Kilosort's kNN graph gave the
    # project its only real win (5aw).
    fp = amp.astype(np.float64)
    fp /= np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    fp = fp - fp.mean(0)
    _, _, Vt = np.linalg.svd(fp[np.random.default_rng(0).choice(
        n, min(n, 60000), replace=False)], full_matrices=False)
    pcs = fp @ Vt[:N_FP_PC].T
    for j in range(N_FP_PC):
        cols.append(W_FP * robust_z(pcs[:, j]))
        used.append("fp_pc%d x%.1f" % (j, W_FP))

    F = np.nan_to_num(np.column_stack(cols)).astype(np.float32)
    # ISO-SPLIT's C++ core calls abort() on exactly-duplicate rows ("Unable to
    # invert matrix"), and several axes here are integer-valued
    # (zero_crossings, n_chan_half, trough_to_peak_samp), so exact duplicates
    # among 400k events are common. A jitter far below the axis scale -- these
    # are robust-z standardised, so ~1 -- breaks the ties without moving any
    # point relative to the structure.
    F += np.random.default_rng(1).normal(0.0, 1e-3, F.shape).astype(np.float32)
    print("  %d events x %d axes" % F.shape)
    print("  axes: " + ", ".join(used))
    n_dim = F.shape[1]

    labels = np.full(n, -1, dtype=np.int64)
    nxt = 0
    if BAND <= 0:
        ii = np.where(done)[0]
        lab = isosplit6(np.ascontiguousarray(F[ii]))
        labels[ii] = lab - 1
        nxt = int(lab.max())
        print("  ISO-SPLIT global: %d clusters" % nxt)
    else:
        import time
        t0 = time.time()
        for c0 in range(0, N_CHAN_BIN, int(BAND)):
            core = done & (depth_idx >= c0) & (depth_idx < c0 + BAND)
            ctx = done & (depth_idx >= c0 - MARGIN) & (depth_idx < c0 + BAND + MARGIN)
            # a band needs comfortably more points than feature dimensions or
            # the within-cluster covariance ISO-SPLIT inverts is singular
            if core.sum() < MIN_SPIKES or ctx.sum() < 10 * n_dim:
                continue
            ii = np.where(ctx)[0]
            lab = isosplit6(np.ascontiguousarray(F[ii]))
            keep = core[ii]
            u, inv = np.unique(lab[keep], return_inverse=True)
            labels[ii[keep]] = nxt + inv
            nxt += len(u)
        print("  ISO-SPLIT band-wise (%d chan bands): %d clusters in %.0fs"
              % (BAND, nxt, time.time() - t0))

    # per-cluster refractory dedup: the detector's small local-maximum
    # footprint admits ~3.35 detections per real spike (5ax), and two
    # detections of one spike land in the same cluster while two neurons' do
    # not, so this is the stage where they can be told apart
    keep = np.zeros(n, bool)
    ndup = 0
    for c in range(nxt):
        m = np.where(labels == c)[0]
        if not len(m):
            continue
        o = m[np.argsort(t[m])]
        last, best = -10 ** 9, -1
        for i in o:
            if t[i] - last < REFRACTORY:
                ndup += 1
                if best >= 0 and qual[i] > qual[best]:
                    keep[best] = False
                    keep[i] = True
                    best, last = i, t[i]
                continue
            keep[i] = True
            best, last = i, t[i]
    sel = keep & (labels >= 0)
    print("  refractory rule removed %d duplicates (%.1f%%)"
          % (ndup, 100.0 * ndup / n))

    cl = labels[sel]
    cnt = np.bincount(cl, minlength=nxt)
    sel2 = sel.copy()
    sel2[sel] = cnt[cl] >= MIN_SPIKES
    u, cl = np.unique(labels[sel2], return_inverse=True)
    st = t[sel2].astype(np.int64)
    o = np.argsort(st)

    out = os.path.join(ROOT, "hybrid_v2_rep%d" % rep, "ks_" + name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "spike_times.npy"), st[o])
    np.save(os.path.join(out, "spike_clusters.npy"), cl[o].astype(np.int32))
    sz = np.bincount(cl)
    print("  WROTE %d spikes in %d clusters -> ks_%s" % (len(st), len(u), name))
    print("  sizes: median %d, min %d, max %d | %.2f spikes/s/chan "
          "(Kilosort vanilla 4.87, 216 clusters)"
          % (int(np.median(sz)), sz.min(), sz.max(),
             len(st) / 120.0 / N_CHAN_BIN))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]
    kw = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    main(int(a[0]), a[1], **kw)
