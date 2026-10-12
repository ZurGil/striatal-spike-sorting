"""LOCDEN -- a spike sorter that is not Kilosort, tested on our own benchmark.

WHY BUILD A SECOND SORTER AT ALL. Every result in sections 5a-5aw was a
modification of Kilosort4, and every one of them was judged against Kilosort's
own detection threshold, its own PC basis and its own nclust=200 seeding --
all three tuned on unmodified data. The one real win (5aw) was a footprint in
the kNN graph: +4.8 recall at p=0.001, net flat once precision is counted.
That win says the across-channel energy split carries identity information
Kilosort's features do not use. LOCDEN is the attempt to build a sorter around
that fact from the beginning rather than patch it in at one stage.

THE THREE DIFFERENCES FROM KILOSORT, each aimed at a measured failure:

  1. LOCALIZATION INSTEAD OF A FOOTPRINT VECTOR. 5av measured the energy
     footprint as a weak discriminator against the real impostor population
     (cosine 0.876 to the target vs 0.976 to the best competitor -- a
     candidate matches a NEIGHBOUR better than its own unit). Cosine between
     non-negative energy vectors is badly compressed. So instead of comparing
     10 energy ratios, fit the physical model that generates them: a point
     current source at (x, y, z) with amplitude A, seen as
     a_c = A / sqrt((x-x_c)^2 + (y-y_c)^2 + z^2). That turns 10 noisy ratios
     into 4 parameters, one of which (depth y) is a micrometre coordinate --
     a metric space where distance means distance, not a compressed cosine.

  2. DENSITY CLUSTERING INSTEAD OF SEEDED K-MEANS. 5aa found nclust=200 is
     hard-coded in Kilosort and absent from parameters.py, and 5aw showed
     handing the seeding extra dimensions CANCELS part of the graph gain --
     an over-seeded k-means cuts along the direction of greatest within-unit
     spread, slicing unimodal units. HDBSCAN has no k, finds variable-density
     clusters, and has an explicit noise label, so a spike it cannot assign is
     dropped instead of forced into the nearest centre.

  3. OUR OWN DETECTION. Kilosort's threshold is a confound in every previous
     comparison. Here detection is an explicit spatiotemporal local minimum on
     z-scored filtered voltage, with the threshold a named parameter.

HONEST GEOMETRY CAVEAT, known before the first run: this probe has 2 staggered
columns spanning 48 um in x and 192 rows at 20 um pitch. It is effectively
one-dimensional. So x_hat carries almost no information and the localizer's
useful outputs are depth y, distance-from-probe z and amplitude A. Any claim
that "localization works" here is a claim about depth, not about 3-D position.

Stage 1 (--detect) is the expensive pass over 2.8 GB and is cached; stage 2
(--cluster) runs on the cache in seconds so clustering can be iterated.

Output goes to hybrid_v2_rep{r}/ks_{name}/ as spike_times.npy and
spike_clusters.npy, which is exactly what run_v2_comparison.score() reads --
so a sorter that is not Kilosort is scored by the identical matching rule,
tolerance (10 samples) and fragment definition as every Kilosort config, on
the same 4 replicates and the same easy/collision/pair/hard placements.

Usage:
  python sorter_locden.py --detect <rep>
  python sorter_locden.py --localize <rep>
"""
import os
import sys
import json
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import minimum_filter

ROOT = r"D:\Gil\spike_sorting_agent"
FS = 30000.0
N_CHAN_BIN = 384
GAIN_TO_UV = 0.018311105685598315

# --- detection parameters (all named, unlike Kilosort's internal threshold) --
HP, LP = 300.0, 6000.0
DETECT_Z = 5.0        # trough must be this many robust SDs below zero
DEDUP_DT = 10         # +/- samples: one event per 0.33 ms ...
DEDUP_DC = 5          # ... and per ~50 um of depth (5 channels in depth order)
AMP_PRE, AMP_POST = 8, 16     # window for peak-to-peak amplitude
WAVE_PRE, WAVE_POST = 10, 22  # peak-channel waveform stored for PCA
N_LOCAL = 20          # channels kept per event for the localization fit
CHUNK = 300000        # 10 s
PAD = 3000


def data_dir(rep):
    return os.path.join(ROOT, "hybrid_v2_rep%d" % rep)


def cache_dir(rep):
    d = os.path.join(data_dir(rep), "ks_locden_cache")
    os.makedirs(d, exist_ok=True)
    return d


def probe(rep):
    with open(os.path.join(data_dir(rep), "probe.json")) as f:
        p = json.load(f)
    cm = np.asarray(p["chanMap"], dtype=np.int64)
    xc = np.asarray(p["xc"], dtype=np.float64)
    yc = np.asarray(p["yc"], dtype=np.float64)
    # full-length arrays indexed by RAW channel number, so an event's channel
    # index means the same thing here as it does in the binary file
    X = np.full(N_CHAN_BIN, np.nan)
    Y = np.full(N_CHAN_BIN, np.nan)
    X[cm] = xc
    Y[cm] = yc
    return cm, X, Y


def neighbours(cm, X, Y, k=N_LOCAL):
    """For each raw channel, the k nearest raw channels (itself first)."""
    d = np.sqrt((X[cm][:, None] - X[cm][None, :]) ** 2
                + (Y[cm][:, None] - Y[cm][None, :]) ** 2)
    nb = np.zeros((N_CHAN_BIN, k), dtype=np.int64)
    for i, c in enumerate(cm):
        nb[c] = cm[np.argsort(d[i])[:k]]
    return nb


# ------------------------------------------------------------------ stage 1
def detect(rep):
    cm, X, Y = probe(rep)
    order = cm[np.lexsort((X[cm], Y[cm]))]        # raw channels, deep -> shallow
    nb = neighbours(cm, X, Y)
    sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")

    raw = np.memmap(os.path.join(data_dir(rep), "hybrid.bin"), dtype=np.int16,
                    mode="r").reshape(-1, N_CHAN_BIN)
    n_samples = raw.shape[0]

    def load(a, b):
        x = raw[a:b, :].astype(np.float32) * GAIN_TO_UV
        x -= np.median(x, axis=1, keepdims=True)   # CAR, matching do_CAR=True
        return sosfiltfilt(sos, x, axis=0).astype(np.float32)

    # noise scale from one 2 s block, robust (MAD), per channel
    blk = load(60000, 120000)
    noise = (np.median(np.abs(blk - np.median(blk, axis=0)), axis=0)
             / 0.6744897501960817).astype(np.float32)
    bad = ~np.isfinite(noise) | (noise <= 0)
    noise[bad] = np.inf
    print("  noise median %.2f uV over %d live channels"
          % (np.median(noise[~bad]), (~bad).sum()))
    del blk

    T, C, A, W = [], [], [], []
    t0 = time.time()
    for start in range(0, n_samples, CHUNK):
        stop = min(start + CHUNK, n_samples)
        lo, hi = max(0, start - PAD), min(n_samples, stop + PAD)
        x = load(lo, hi)
        z = x[:, order] / noise[order]

        mf = minimum_filter(z, size=(2 * DEDUP_DT + 1, 2 * DEDUP_DC + 1),
                            mode="nearest")
        ti, ci = np.where((z == mf) & (z < -DETECT_Z))
        del mf, z

        keep = ((ti + lo >= start) & (ti + lo < stop)
                & (ti >= max(AMP_PRE, WAVE_PRE))
                & (ti + max(AMP_POST, WAVE_POST) < hi - lo))
        ti, ci = ti[keep], ci[keep]
        if len(ti) == 0:
            continue
        pk = order[ci]                             # raw peak channel

        # peak-to-peak amplitude on the N_LOCAL nearest channels
        toff = np.arange(-AMP_PRE, AMP_POST)
        blk = x[ti[:, None, None] + toff[None, :, None], nb[pk][:, None, :]]
        amp = (blk.max(axis=1) - blk.min(axis=1)).astype(np.float32)

        woff = np.arange(-WAVE_PRE, WAVE_POST)
        wav = x[ti[:, None] + woff[None, :], pk[:, None]].astype(np.float32)

        T.append(ti + lo)
        C.append(pk)
        A.append(amp)
        W.append(wav)
        print("  %6.1f-%6.1fs %7d events (%5.0fs)"
              % (start / FS, stop / FS, len(ti), time.time() - t0), flush=True)
        del x, blk

    t = np.concatenate(T).astype(np.int64)
    o = np.argsort(t)
    np.savez(os.path.join(cache_dir(rep), "events.npz"),
             t=t[o], ch=np.concatenate(C).astype(np.int16)[o],
             amp=np.concatenate(A)[o], wave=np.concatenate(W)[o],
             nb=nb, order=order, noise=noise, probe_x=X, probe_y=Y,
             detect_z=DETECT_Z, dedup_dt=DEDUP_DT, dedup_dc=DEDUP_DC)
    print("  %d events cached (%.0f/s over the probe, %.2f per channel per s)"
          % (len(t), len(t) / (n_samples / FS),
             len(t) / (n_samples / FS) / N_CHAN_BIN))


# ------------------------------------------------------------------ stage 2
def localize(rep, cache="events2.npz", n_iter=30, z_init=20.0, z_min=1.0,
             z_max=250.0, lam=1e-3):
    """Batched Levenberg-Marquardt fit of a_c = A / sqrt(d_c^2 + z^2).

    Fitted jointly for every event at once: a (n_ev, 4, 4) solve per
    iteration rather than n_ev calls into scipy, which is what makes a
    half-million-event fit take seconds instead of hours.
    """
    ev = np.load(os.path.join(cache_dir(rep), cache))
    amp, ch, nb = ev["amp"], ev["ch"].astype(np.int64), ev["nb"]
    X, Y = ev["probe_x"], ev["probe_y"]
    n = len(ch)

    cx = X[nb[ch]]                      # (n, N_LOCAL) channel coordinates
    cy = Y[nb[ch]]
    a = amp.astype(np.float64)
    w = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(a)
    a = np.where(w, a, 0.0)

    # init: amplitude-weighted centroid, z a fixed guess, A from the peak
    sw = a.sum(axis=1) + 1e-12
    px = (a * np.nan_to_num(cx)).sum(axis=1) / sw
    py = (a * np.nan_to_num(cy)).sum(axis=1) / sw
    pz = np.full(n, z_init)
    pA = a.max(axis=1) * z_init

    cxf, cyf = np.nan_to_num(cx), np.nan_to_num(cy)
    t0 = time.time()
    for _ in range(n_iter):
        dx = px[:, None] - cxf
        dy = py[:, None] - cyf
        d2 = dx ** 2 + dy ** 2 + (pz ** 2)[:, None]
        d = np.sqrt(d2)
        pred = pA[:, None] / d
        r = (pred - a) * w                               # residual
        # Jacobian of pred wrt (x, y, z, A)
        g = -(pA[:, None] / (d2 * d))
        J = np.stack([g * dx, g * dy, g * pz[:, None], (1.0 / d)], axis=2) * w[..., None]
        JtJ = np.einsum("nkp,nkq->npq", J, J)
        Jtr = np.einsum("nkp,nk->np", J, r)
        JtJ[:, np.arange(4), np.arange(4)] *= (1.0 + lam)
        JtJ[:, np.arange(4), np.arange(4)] += 1e-9
        try:
            step = np.linalg.solve(JtJ, Jtr)
        except np.linalg.LinAlgError:
            break
        step = np.nan_to_num(step)
        px -= np.clip(step[:, 0], -40, 40)
        py -= np.clip(step[:, 1], -40, 40)
        pz -= np.clip(step[:, 2], -40, 40)
        pA -= step[:, 3]
        pz = np.clip(np.abs(pz), z_min, z_max)
        pA = np.abs(pA)

    dx = px[:, None] - cxf
    dy = py[:, None] - cyf
    d = np.sqrt(dx ** 2 + dy ** 2 + (pz ** 2)[:, None])
    pred = pA[:, None] / d
    ss = (((pred - a) * w) ** 2).sum(axis=1)
    tot = (((a - (a * w).sum(axis=1, keepdims=True)
             / np.maximum(w.sum(axis=1, keepdims=True), 1)) * w) ** 2).sum(axis=1)
    r2 = 1.0 - ss / np.maximum(tot, 1e-12)

    np.savez(os.path.join(cache_dir(rep), "loc_" + cache),
             x=px, y=py, z=pz, A=pA, r2=r2,
             amp_max=amp.max(axis=1).astype(np.float32))
    print("  localized %d events in %.0fs" % (n, time.time() - t0))
    print("  fit R^2: median %.3f, p10 %.3f, p90 %.3f"
          % (np.median(r2), np.percentile(r2, 10), np.percentile(r2, 90)))
    print("  depth y: %.0f-%.0f um | z: median %.1f um | A: median %.0f"
          % (np.percentile(py, 1), np.percentile(py, 99), np.median(pz),
             np.median(pA)))


# ------------------------------------------------------------ stage 1, v2
# Parameters below are not guesses: they are the operating point chosen from
# the threshold sweep in diag_detect2_sweep.py, which measured detection
# recall against a MEASURED chance rate at each setting. Iteration 1's plain
# per-channel threshold reached 0.214 (chance 0.009); this reaches 0.949 at a
# Kilosort-comparable event rate and 0.978 with headroom.
N_PC = 5              # temporal basis size (PC1 alone is 51.8% of variance)
BASIS_TOP = 8000      # build the basis from the loudest bootstrapped waveforms
POOL_C = 5            # +/- channels pooled in depth order (~50 um)
LM_DT, LM_DC = 3, 1   # local-maximum footprint. SMALL ON PURPOSE: at
                      # (10, 5) a quiet spike beside a loud one is never a
                      # local maximum and recall saturates at 0.476 no matter
                      # how low the threshold goes. Duplicates are removed
                      # later, per cluster, by a refractory rule -- which is
                      # the right place, because two duplicates of one spike
                      # land in the same cluster while two neurons' spikes do
                      # not.
WHITEN_EPS = 1e-3
TARGET_RATE = 8.0     # events per second per channel at the chosen threshold
REFINE = 5            # refine the event time to the trough within +/- this


def _basis(rep):
    ev = np.load(os.path.join(cache_dir(rep), "events.npz"))
    W = ev["wave"].astype(np.float64)
    W = W - W.mean(axis=1, keepdims=True)
    W = W[np.argsort(-(W.max(axis=1) - W.min(axis=1)))[:BASIS_TOP]]
    nrm = np.linalg.norm(W, axis=1)
    W = W[nrm > 0] / nrm[nrm > 0][:, None]
    _, S_, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
    B = Vt[:N_PC]
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B, (S_[:N_PC] ** 2 / (S_ ** 2).sum())


def detect2(rep):
    """Whitened multichannel matched-filter detection.

    Three things iteration 1 lacked, each measured to matter:
      spatial whitening   decorrelates shared noise across 384 channels
      temporal matching   projects onto a learned spike basis, not one sample
      spatial pooling     integrates over the channels a spike appears on
    """
    from scipy.signal import oaconvolve
    from scipy.ndimage import maximum_filter, uniform_filter1d

    cm, X, Y = probe(rep)
    order = cm[np.lexsort((X[cm], Y[cm]))]
    nb = neighbours(cm, X, Y)
    B, vexp = _basis(rep)
    print("  basis: %d PCs, variance %s" % (N_PC, np.round(vexp, 3).tolist()))
    sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")

    raw = np.memmap(os.path.join(data_dir(rep), "hybrid.bin"), dtype=np.int16,
                    mode="r").reshape(-1, N_CHAN_BIN)
    n_samples = raw.shape[0]

    def load(a, b):
        x = raw[a:b, :].astype(np.float32) * GAIN_TO_UV
        x -= np.median(x, axis=1, keepdims=True)
        return sosfiltfilt(sos, x, axis=0).astype(np.float32)[:, order]

    # whitening matrix from one 10 s block, reused for every chunk so the
    # detection statistic means the same thing throughout the recording
    blk = load(600000, 900000)
    sub = blk[::7].astype(np.float64)
    C = (sub.T @ sub) / len(sub)
    d, V = np.linalg.eigh(C)
    d = np.maximum(d, WHITEN_EPS * d.max())
    Wm = ((V * (1.0 / np.sqrt(d))) @ V.T).astype(np.float32)
    del blk, sub, C

    def statistic(x):
        xw = x @ Wm
        E = np.zeros(xw.shape, dtype=np.float32)
        for k in range(N_PC):
            P = oaconvolve(xw, B[k][::-1].astype(np.float32)[:, None],
                           axes=0, mode="same").astype(np.float32)
            E += P * P
            del P
        del xw
        return uniform_filter1d(E, size=2 * POOL_C + 1, axis=1,
                                mode="nearest") * (2 * POOL_C + 1)

    # threshold calibrated once, on the first chunk, to TARGET_RATE
    x = load(0, CHUNK + PAD)
    S = statistic(x)
    lmv = S[maximum_filter(S, size=(2 * LM_DT + 1, 2 * LM_DC + 1),
                           mode="nearest") == S]
    want = TARGET_RATE * (CHUNK / FS) * N_CHAN_BIN
    thr = float(np.quantile(lmv, max(0.0, 1.0 - want / len(lmv))))
    print("  threshold %.1f for %.1f events/s/channel" % (thr, TARGET_RATE))
    del x, S, lmv

    T, C_, A, W_, Q = [], [], [], [], []
    t0 = time.time()
    for start in range(0, n_samples, CHUNK):
        stop = min(start + CHUNK, n_samples)
        lo, hi = max(0, start - PAD), min(n_samples, stop + PAD)
        x = load(lo, hi)
        S = statistic(x)
        mx = maximum_filter(S, size=(2 * LM_DT + 1, 2 * LM_DC + 1),
                            mode="nearest")
        ti, ci = np.where((S == mx) & (S >= thr))
        del mx
        keep = ((ti + lo >= start) & (ti + lo < stop)
                & (ti >= max(AMP_PRE, WAVE_PRE) + REFINE)
                & (ti + max(AMP_POST, WAVE_POST) + REFINE < hi - lo))
        ti, ci = ti[keep], ci[keep]
        if len(ti) == 0:
            del x, S
            continue
        qual = S[ti, ci].astype(np.float32)
        del S

        # the event time must be the TROUGH: the truth times are trough
        # samples and Kilosort agrees with them to +/-1 sample, so anything
        # else would fail the 10-sample matcher for a bookkeeping reason
        roff = np.arange(-REFINE, REFINE + 1)
        loc = x[ti[:, None] + roff[None, :], ci[:, None]]
        ti = ti + roff[np.argmin(loc, axis=1)]

        # peak channel = largest peak-to-peak nearby, in RAW channel numbers
        pool = np.clip(ci[:, None] + np.arange(-POOL_C, POOL_C + 1)[None, :],
                       0, N_CHAN_BIN - 1)
        toff = np.arange(-AMP_PRE, AMP_POST)
        blk = x[ti[:, None, None] + toff[None, :, None], pool[:, None, :]]
        pk = order[pool[np.arange(len(ti)),
                        np.argmax(blk.max(axis=1) - blk.min(axis=1), axis=1)]]
        del blk

        # amplitudes for the localization fit, on filtered uV (the point-source
        # model is about physical voltage, so these are NOT whitened)
        cols = pos_in_order_of(order)[nb[pk]]
        blk = x[ti[:, None, None] + toff[None, :, None], cols[:, None, :]]
        amp = (blk.max(axis=1) - blk.min(axis=1)).astype(np.float32)
        del blk

        woff = np.arange(-WAVE_PRE, WAVE_POST)
        wav = x[ti[:, None] + woff[None, :],
                pos_in_order_of(order)[pk][:, None]].astype(np.float32)

        T.append(ti + lo)
        C_.append(pk)
        A.append(amp)
        W_.append(wav)
        Q.append(qual)
        print("  %6.1f-%6.1fs %7d events (%5.0fs)"
              % (start / FS, stop / FS, len(ti), time.time() - t0), flush=True)
        del x

    t = np.concatenate(T).astype(np.int64)
    o = np.argsort(t)
    np.savez(os.path.join(cache_dir(rep), "events2.npz"),
             t=t[o], ch=np.concatenate(C_).astype(np.int16)[o],
             amp=np.concatenate(A)[o], wave=np.concatenate(W_)[o],
             qual=np.concatenate(Q)[o], nb=nb, order=order,
             probe_x=X, probe_y=Y, basis=B, thr=thr)
    print("  %d events (%.2f/s/channel)"
          % (len(t), len(t) / (n_samples / FS) / N_CHAN_BIN))


_PIO = {}


def pos_in_order_of(order):
    key = id(order)
    if key not in _PIO:
        p = np.full(N_CHAN_BIN, 0)
        p[order] = np.arange(len(order))
        _PIO[key] = p
    return _PIO[key]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--detect":
        detect(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "--detect2":
        detect2(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "--localize":
        localize(int(sys.argv[2]),
                 **({"cache": sys.argv[3]} if len(sys.argv) > 3 else {}))
    else:
        print(__doc__)
