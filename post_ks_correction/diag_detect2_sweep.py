"""Iteration 2: can a WHITENED MULTICHANNEL detector see these units at all?

Iteration 1 (plain per-channel threshold on filtered voltage) reached a true
detection recall of 0.214 against a measured chance rate of 0.009 -- the
headline 0.556 was almost entirely base rate (chance 0.415), because it
allowed a match to any event anywhere on the probe. The cause was then
measured directly: the injected units have a spike-triggered-average trough of
median 3.06 noise SDs on their own best channel, range 0.64 to 5.29. Unit 371's
deterministic signal is 0.64 SD -- BELOW the single-sample noise. No
single-channel threshold can recover that.

What Kilosort does that iteration 1 did not:
  SPATIAL WHITENING   decorrelates the shared noise across channels, which on
                      a 384-channel probe is a large fraction of the variance
  TEMPORAL MATCHING   projects onto spike-shaped templates over ~60 samples
                      instead of looking at one sample's depth
  SPATIAL POOLING     integrates the evidence over the ~10 channels a spike
                      actually appears on

Together those are an SNR gain of roughly sqrt(n_channels * n_effective_samples),
which is what turns a 3 SD spike into a detectable one. This script adds all
three and sweeps the threshold, so the output is a recall-versus-event-count
curve rather than a single number from a guessed threshold.

The temporal basis is SELF-BOOTSTRAPPED, not borrowed from Kilosort: the top
principal components of the 90,664 waveforms iteration 1 detected. Large
spikes teach the shape; the shape is then used to find small ones.

Every recall is reported with its MEASURED chance rate, on the same channel
restriction, because that is the only version of the number that meant
anything in iteration 1.

Runs on a 10 s window so one design iteration costs ~30 s.
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, oaconvolve
from scipy.ndimage import maximum_filter, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import (ROOT, FS, N_CHAN_BIN, GAIN_TO_UV, HP, LP,
                           data_dir, cache_dir, probe)

REP = 0
T0, DUR = 600000, 300000          # analyse 20-30 s
PAD = 3000
N_PC = 3                          # temporal basis size
POOL_C = 5                        # +/- channels in depth order (~50 um)
DT, DC = 10, 5                    # local-maximum footprint
WHITEN_EPS = 1e-3
RAD_UM = 40.0                     # channel restriction for the recall test
BASIS_TOP = 0                     # if >0, build the basis from the N largest
                                  # bootstrapped waveforms only

# every parameter overridable as key=value, so one iteration is one command
for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        if _k.upper() in globals():
            globals()[_k.upper()] = type(globals()[_k.upper()])(float(_v)) \
                if not isinstance(globals()[_k.upper()], float) else float(_v)
            print("  [override] %s = %s" % (_k.upper(), globals()[_k.upper()]))

cm, X, Y = probe(REP)
order = cm[np.lexsort((X[cm], Y[cm]))]
pos_in_order = np.full(N_CHAN_BIN, -1)
pos_in_order[order] = np.arange(len(order))

T = np.load(os.path.join(data_dir(REP), "hybrid_truth.npz"), allow_pickle=True)
tt, lab = T["times"].ravel(), T["labels"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]
su = [int(x) for x in T["source_units"]]
dstc = {u: int(c) for u, c in zip(su, T["dst_peak_channels"])}

# ------------------------------------------------- temporal basis (bootstrap)
ev = np.load(os.path.join(cache_dir(REP), "events.npz"))
W = ev["wave"].astype(np.float64)
W = W - W.mean(axis=1, keepdims=True)
if BASIS_TOP:
    # a basis learned from noise-dominated snippets is itself noise; take the
    # loudest events, whose shape is the real spike shape
    W = W[np.argsort(-(W.max(axis=1) - W.min(axis=1)))[:int(BASIS_TOP)]]
nrm = np.linalg.norm(W, axis=1)
W = W[nrm > 0] / nrm[nrm > 0][:, None]
U, S_, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
basis = Vt[:N_PC]
basis = basis / np.linalg.norm(basis, axis=1, keepdims=True)
print("temporal basis: %d PCs from %d bootstrapped waveforms, "
      "variance explained %s"
      % (N_PC, len(W),
         np.round((S_[:N_PC] ** 2 / (S_ ** 2).sum()), 3).tolist()))

# ----------------------------------------------------------------- load data
sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")
raw = np.memmap(os.path.join(data_dir(REP), "hybrid.bin"), dtype=np.int16,
                mode="r").reshape(-1, N_CHAN_BIN)
lo, hi = T0 - PAD, T0 + DUR + PAD
x = raw[lo:hi, :].astype(np.float32) * GAIN_TO_UV
x -= np.median(x, axis=1, keepdims=True)
x = sosfiltfilt(sos, x, axis=0).astype(np.float32)
x = x[:, order]                                     # depth order from here on

# ------------------------------------------------------- spatial whitening
sub = x[::7].astype(np.float64)
C = (sub.T @ sub) / len(sub)
d, V = np.linalg.eigh(C)
d = np.maximum(d, WHITEN_EPS * d.max())
Wm = (V * (1.0 / np.sqrt(d))) @ V.T
xw = (x.astype(np.float64) @ Wm).astype(np.float32)
print("whitened: per-channel sd before %.2f uV, after %.3f (target 1.0)"
      % (np.median(x.std(0)), np.median(xw.std(0))))

# --------------------------------------- matched filter energy + pooling
E = np.zeros(xw.shape, dtype=np.float32)
for k in range(N_PC):
    b = basis[k][::-1].astype(np.float32)
    P = oaconvolve(xw, b[:, None], axes=0, mode="same").astype(np.float32)
    E += P * P
    del P
# pooling over channels: SUM of squares over +/-POOL_C in depth order
Spool = uniform_filter1d(E, size=2 * POOL_C + 1, axis=1,
                         mode="nearest") * (2 * POOL_C + 1)
del E

# under noise, each projection is ~N(0,1), so Spool ~ chi2(N_PC*(2*POOL_C+1))
dof = N_PC * (2 * POOL_C + 1)
print("pooled statistic: dof %d, median %.1f (chi2 median %.1f)"
      % (dof, np.median(Spool), dof * 0.9551))

mf = maximum_filter(Spool, size=(2 * DT + 1, 2 * DC + 1), mode="nearest")
cand = (Spool == mf)
ti_all, ci_all = np.where(cand)
val_all = Spool[ti_all, ci_all]
abs_t = ti_all + lo
inwin = (abs_t >= T0) & (abs_t < T0 + DUR)
ti_all, ci_all, val_all, abs_t = (ti_all[inwin], ci_all[inwin],
                                  val_all[inwin], abs_t[inwin])
print("%d local maxima in the window" % len(abs_t))

# ------------------------------------------------------------- recall sweep
sel = (tt >= T0) & (tt < T0 + DUR)
tw, lw, tiw = tt[sel], lab[sel], tier_code[sel]
rng = np.random.default_rng(0)
Xo, Yo = X[order], Y[order]


def recall(det_t, det_c, times, labels, tol=10):
    if len(det_t) == 0:
        return 0.0
    o = np.argsort(det_t)
    dt, dc = det_t[o], det_c[o]
    hit = 0
    for a, u in zip(times, labels):
        i = np.searchsorted(dt, a - tol)
        j = np.searchsorted(dt, a + tol, "right")
        if j > i:
            c0 = dstc[int(u)]
            dd = np.sqrt((Xo[dc[i:j]] - X[c0]) ** 2 + (Yo[dc[i:j]] - Y[c0]) ** 2)
            if (dd <= RAD_UM).any():
                hit += 1
    return hit / len(times)


print("\n" + "=" * 86)
print("WHITENED MULTICHANNEL DETECTION, 10 s window, %d injected spikes"
      % sel.sum())
print("=" * 86)
print("%9s %9s %10s %8s %8s %8s %8s %8s"
      % ("thresh", "events", "ev/s/chan", "recall", "chance", "excess",
         "easy", "pair"))
for q in (99.9, 99.7, 99.5, 99.0, 98.0, 96.0, 93.0, 90.0):
    thr = np.percentile(val_all, 100 - (100 - q) * 0 + 0) if False else \
        np.percentile(val_all, q)
    m = val_all >= thr
    dt, dc = abs_t[m], ci_all[m]
    r = recall(dt, dc, tw, lw)
    sham = np.mean([recall(dt, dc, rng.integers(T0 + 100, T0 + DUR - 100, len(tw)),
                           lw) for _ in range(3)])
    rt = {}
    for k, nm in enumerate(tier_names):
        s2 = tiw == k
        rt[nm] = recall(dt, dc, tw[s2], lw[s2]) if s2.sum() else np.nan
    print("%9.1f %9d %10.2f %8.3f %8.3f %+8.3f %8.3f %8.3f"
          % (thr, m.sum(), m.sum() / (DUR / FS) / N_CHAN_BIN, r, sham,
             r - sham, rt.get("easy", np.nan), rt.get("pair", np.nan)))
print("\n(Kilosort vanilla, same benchmark: 224,347 spikes over 120 s = "
      "4.87 events/s/channel after its own clustering and merging)")
