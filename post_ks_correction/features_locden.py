"""Multi-axis per-spike representation: wavelet-aligned, physically named axes.

Gil's proposal, and it targets exactly what 5ax measured as the failure. The
LOCDEN detector reaches 0.978 detection recall, but end to end the sorter
scored 0.519 recall / 0.134 precision because its feature space was thin:
3 PCs of the PEAK CHANNEL waveform, fitted depth, and amplitude. Kilosort
uses ~10 channels x 6 features. The clustering was being asked to separate
neurons using less information than Kilosort has, which it cannot do.

So instead of more PCs -- which are a basis nobody can interpret and which
spend their variance on whatever happens to vary most, including jitter (5ad
measured Kilosort wasting a PC on sub-sample timing) -- this builds axes that
are each a NAMED physical property of the spike:

  SHAPE (peak channel, after sub-sample alignment)
    trough depth, the peak before and after it, peak-to-trough amplitude,
    trough-to-peak DURATION (the classic striatal fast-spiking-interneuron vs
    projection-neuron discriminator), trough half-width, repolarization and
    recovery slope, peak/trough asymmetry, energy, zero crossings,
    skew/kurtosis
  FREQUENCY
    the per-event best-match frequency f0 = argmax f*|W(f)| over a Morlet
    bank, and the spectral centroid. f0 is this project's own quantity, and
    the f*|W(f)| criterion rather than |W(f)| is a measured choice: 5-section
    work found plain argmax|W| gave a median spurious shift of 0.72 samples
    and made template fit WORSE than no alignment (R2 0.780 -> 0.737), while
    the timing criterion drops that to 0.33 samples and 2.8% of spikes moved
    by more than a sample.
  SPACE
    fitted point-source depth/distance/amplitude and fit quality, footprint
    spread in micrometres, dorsoventral footprint asymmetry, channel count
    above half-maximum, and the propagation slope (per-channel trough time
    regressed on depth) -- which is a genuinely independent axis, since it
    measures how the spike travels rather than how it looks.

WHY ALIGNMENT FIRST. Every shape feature above is corrupted by sub-sample
timing jitter: a spike sampled 0.4 samples late has a shallower measured
trough and a different measured width, purely from sampling. Aligning all
events to a common wavelet phase removes that variance before it reaches the
features, which is the whole reason this project built the wavelet machinery.

Two stages so feature engineering iterates in seconds rather than minutes:
  --extract  one 5-minute pass over the recording, caching raw snippets
  --features fast; wavelet alignment and all axes, from the cache

Usage:
  python features_locden.py --extract <rep>
  python features_locden.py --features <rep>
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import (ROOT, FS, N_CHAN_BIN, GAIN_TO_UV, HP, LP, CHUNK,
                           PAD, data_dir, cache_dir, probe)

NT = 61               # snippet length, 2.03 ms -- long enough for recovery
PRE = 20              # samples before the trough
N_LOCAL = 20          # channels kept per event
F_LO, F_HI, N_F = 400.0, 7000.0, 40
ALIGN = False         # see the long note in features(); measured at chance
N_CYCLES = 3.0


# --------------------------------------------------------------- stage A
def extract(rep):
    ev = np.load(os.path.join(cache_dir(rep), "events2.npz"))
    t, ch, nb, order = (ev["t"], ev["ch"].astype(np.int64), ev["nb"],
                        ev["order"])
    cm, X, Y = probe(rep)
    pio = np.zeros(N_CHAN_BIN, dtype=np.int64)
    pio[order] = np.arange(len(order))

    sos = butter(3, [HP, LP], btype="bandpass", fs=FS, output="sos")
    raw = np.memmap(os.path.join(data_dir(rep), "hybrid.bin"), dtype=np.int16,
                    mode="r").reshape(-1, N_CHAN_BIN)
    n_samples = raw.shape[0]

    n = len(t)
    wav = np.zeros((n, NT), dtype=np.float32)        # peak channel
    mc_ptp = np.zeros((n, N_LOCAL), dtype=np.float32)
    mc_tr = np.zeros((n, N_LOCAL), dtype=np.int8)    # trough offset per chan
    mc_min = np.zeros((n, N_LOCAL), dtype=np.float32)
    done = np.zeros(n, bool)

    t0 = time.time()
    for start in range(0, n_samples, CHUNK):
        stop = min(start + CHUNK, n_samples)
        sel = np.where((t >= start) & (t < stop))[0]
        if not len(sel):
            continue
        lo, hi = max(0, start - PAD), min(n_samples, stop + PAD)
        x = raw[lo:hi, :].astype(np.float32) * GAIN_TO_UV
        x -= np.median(x, axis=1, keepdims=True)
        x = sosfiltfilt(sos, x, axis=0).astype(np.float32)[:, order]

        ti = (t[sel] - lo).astype(np.int64)
        ok = (ti >= PRE) & (ti + (NT - PRE) < x.shape[0])
        sel, ti = sel[ok], ti[ok]
        if not len(sel):
            del x
            continue
        off = np.arange(-PRE, NT - PRE)
        cols = pio[nb[ch[sel]]]                       # (k, N_LOCAL)
        blk = x[ti[:, None, None] + off[None, :, None], cols[:, None, :]]
        wav[sel] = blk[:, :, 0]                       # nb[ch] starts with itself
        mc_ptp[sel] = blk.max(axis=1) - blk.min(axis=1)
        mc_min[sel] = blk.min(axis=1)
        mc_tr[sel] = (np.argmin(blk, axis=1) - PRE).astype(np.int8)
        done[sel] = True
        del x, blk
        print("  %6.1f-%6.1fs  %6d events (%4.0fs)"
              % (start / FS, stop / FS, len(sel), time.time() - t0), flush=True)

    np.savez(os.path.join(cache_dir(rep), "snip.npz"),
             wav=wav, mc_ptp=mc_ptp, mc_tr=mc_tr, mc_min=mc_min, done=done,
             nt=NT, pre=PRE)
    print("  cached snippets for %d of %d events" % (done.sum(), n))


# --------------------------------------------------------------- stage B
def _morlet_bank(fs, nt, center=PRE, n_cycles=N_CYCLES):
    """The project's own Morlet, one column per frequency, centred on the
    nominal trough index so that `w @ K` is the wavelet transform at t0 for
    every event at once.

    make_morlet is reused rather than reimplemented so the wavelet definition
    here is identical to the one the 5-section timing work validated.
    """
    from wavelet_features import make_morlet
    f = np.linspace(F_LO, F_HI, N_F)
    K = np.zeros((nt, N_F), dtype=np.complex128)
    for j, fj in enumerate(f):
        _, psi = make_morlet(fj, fs, n_cycles)
        L = len(psi)
        a = center - L // 2
        lo_k, hi_k = max(0, a), min(nt, a + L)
        lo_p, hi_p = lo_k - a, lo_k - a + (hi_k - lo_k)
        if hi_k <= lo_k:
            continue
        col = np.zeros(nt, dtype=np.complex128)
        col[lo_k:hi_k] = np.conj(psi[lo_p:hi_p])
        nrm = np.sqrt(np.sum(np.abs(col) ** 2))
        K[:, j] = col / max(nrm, 1e-12)
    return f, K


def _shift(w, d):
    """Shift each row of w by its own fractional amount d, cubic in frequency
    domain via linear interpolation on a fine grid (fast and adequate at
    |d| < 1 sample)."""
    n = w.shape[1]
    g = np.arange(n)
    out = np.empty_like(w)
    for i in range(0, len(w), 20000):
        b = slice(i, min(i + 20000, len(w)))
        src = g[None, :] + d[b][:, None]
        lo = np.clip(np.floor(src).astype(np.int64), 0, n - 2)
        fr = np.clip(src - lo, 0.0, 1.0)
        r = np.arange(b.stop - b.start)[:, None]
        out[b] = (w[b][r, lo] * (1 - fr) + w[b][r, lo + 1] * fr)
    return out


def features(rep):
    ev = np.load(os.path.join(cache_dir(rep), "events2.npz"))
    sn = np.load(os.path.join(cache_dir(rep), "snip.npz"))
    lc = np.load(os.path.join(cache_dir(rep), "loc_events2.npz"))
    cm, X, Y = probe(rep)
    ch, nb = ev["ch"].astype(np.int64), ev["nb"]
    w = sn["wav"].astype(np.float64)
    mc_ptp, mc_tr = sn["mc_ptp"].astype(np.float64), sn["mc_tr"].astype(np.float64)
    done = sn["done"]
    n, nt = w.shape

    w = w - w.mean(axis=1, keepdims=True)

    # ---- per-event best-match frequency and sub-sample alignment
    f, K = _morlet_bank(FS, nt)
    Wc = w @ K                                  # (n, N_F) complex
    mag = np.abs(Wc)
    score = mag * f[None, :]                    # the TIMING criterion
    j = np.argmax(score, axis=1)
    f0 = f[j]
    r = np.arange(n)
    phase = np.angle(Wc[r, j])

    # REFERENCE PHASE, per frequency, calibrated from the data. The phase of
    # the wavelet transform at a spike's trough is NOT zero -- it depends on
    # the wavelet convention and the spike's own shape -- so a shift computed
    # against a reference of 0 is meaningless. The first version of this did
    # exactly that and every shift saturated at the +/-2 sample clamp.
    # Calibrated here on the loudest decile, whose integer trough alignment is
    # the most reliable, exactly as calibrate_reference_phase does for a
    # single template.
    big = np.argsort(-(w.max(axis=1) - w.min(axis=1)))[:max(2000, n // 10)]
    ref = np.zeros(N_F)
    for jj in range(N_F):
        sub = big[j[big] == jj]
        ref[jj] = (np.angle(np.mean(np.exp(1j * np.angle(Wc[sub, jj]))))
                   if len(sub) >= 30 else np.nan)
    gl = np.angle(np.mean(np.exp(1j * phase[big])))
    ref = np.where(np.isfinite(ref), ref, gl)

    dphi = np.angle(np.exp(1j * (phase - ref[j])))   # wrapped to (-pi, pi]
    omega = 2.0 * np.pi * f0 / FS
    raw_d = -dphi / np.maximum(omega, 1e-9)
    CLAMP = 1.5
    sat = np.abs(raw_d) > CLAMP
    d = np.clip(raw_d, -CLAMP, CLAMP)

    # ---- WAVELET ALIGNMENT IS OFF BY DEFAULT, AND THE REASON IS MEASURED.
    # diag_wavelet_alignment_truth.py tested this estimate against the
    # per-spike shifts the benchmark builder actually injected (stored as
    # `shifts` in the truth file, drawn from U(-0.5, 0.5)). On 1,240 matched
    # injected spikes:
    #     correlation with truth   -0.061   (i.e. none)
    #     RMSE                      1.402 samples
    #     RMSE of answering zero    0.292 samples
    # So it is 4.8x WORSE than no alignment at all, on the loudest spikes
    # too (corr -0.065). It must not be used.
    #
    # This does NOT contradict the project's earlier wavelet results, which
    # were measured with a per-unit template and a calibrated reference
    # phase, through coarse_then_fine_shift. That function's own docstring
    # states the limit being violated here: phase-only correction is
    # trustworthy only within half an oscillation cycle of the true spike
    # time, reached via a COARSE template match first. At this stage of a
    # from-scratch sorter there are no units yet, so there is no template to
    # calibrate against and no coarse stage to run -- phase-only alignment is
    # outside the method's validated operating range.
    #
    # The fix is an ordering, not a parameter: cluster first on the axes that
    # are insensitive to sub-sample timing (amplitude, footprint, fitted
    # position, |W| frequency), then align inside each cluster against that
    # cluster's own template, then recompute the shape axes and re-cluster.
    # That is an alternating scheme with the same shape as the deconvolution
    # loop, and it is the honest next step.
    if ALIGN:
        wa = _shift(w, d)
    else:
        d = np.zeros(n)
        wa = w
    print("  alignment: %s" % ("wavelet phase shift APPLIED" if ALIGN else
                               "OFF (measured at chance; see note in source)"))
    print("  wavelet: f0 median %.0f Hz (p10 %.0f, p90 %.0f)"
          % (np.median(f0), np.percentile(f0, 10), np.percentile(f0, 90)))
    print("  shift: median |d| %.3f samples, p90 %.3f, %.1f%% saturated at "
          "+/-%.1f" % (np.median(np.abs(d)), np.percentile(np.abs(d), 90),
                       100.0 * sat.mean(), CLAMP))
    if sat.mean() > 0.25:
        print("  WARNING: over a quarter of shifts saturate -- the reference")
        print("           phase calibration is not working, and the shape")
        print("           features below are NOT reliably de-jittered.")

    # ---- shape axes, on the ALIGNED waveform
    it = np.argmin(wa, axis=1)
    trough = wa[r, it]
    post = wa.copy()
    post[np.arange(nt)[None, :] <= it[:, None]] = -np.inf
    ip = np.argmax(post, axis=1)
    peak_after = wa[r, ip]
    pre = wa.copy()
    pre[np.arange(nt)[None, :] >= it[:, None]] = -np.inf
    peak_before = wa[r, np.argmax(pre, axis=1)]
    ptp = peak_after - trough
    t2p = (ip - it).astype(np.float64)

    half = trough * 0.5
    below = wa <= half[:, None]
    fwhm = below.sum(axis=1).astype(np.float64)

    repol = np.where(t2p > 0, (peak_after - trough) / np.maximum(t2p, 1), 0.0)
    tail = min(nt - 1, 0)
    rec = (wa[:, -1] - peak_after) / max(nt - 1 - 1, 1)
    asym = peak_after / np.maximum(np.abs(trough), 1e-9)
    energy = (wa ** 2).sum(axis=1)
    area = -wa.clip(max=0).sum(axis=1)
    zc = (np.diff(np.sign(wa), axis=1) != 0).sum(axis=1).astype(np.float64)
    sk = skew(wa, axis=1)
    ku = kurtosis(wa, axis=1)
    cen = (mag * f[None, :]).sum(axis=1) / np.maximum(mag.sum(axis=1), 1e-9)

    # ---- space axes
    y, z, A, r2 = lc["y"], lc["z"], lc["A"], lc["r2"]
    cy = Y[nb[ch]]                              # (n, N_LOCAL) channel depths
    wts = mc_ptp / np.maximum(mc_ptp.sum(axis=1, keepdims=True), 1e-9)
    ybar = (wts * cy).sum(axis=1)
    spread = np.sqrt(np.maximum((wts * (cy - ybar[:, None]) ** 2).sum(axis=1), 0))
    up = (wts * (cy > cy[:, :1])).sum(axis=1)
    dn = (wts * (cy < cy[:, :1])).sum(axis=1)
    fp_asym = up - dn
    n_half = (mc_ptp >= 0.5 * mc_ptp.max(axis=1, keepdims=True)).sum(axis=1)

    # propagation: regress per-channel trough time on depth, weighted by ptp
    dy = cy - cy[:, :1]
    mw = mc_ptp >= 0.3 * mc_ptp.max(axis=1, keepdims=True)
    sw = np.maximum(mw.sum(axis=1), 1)
    mx = (dy * mw).sum(axis=1) / sw
    my = (mc_tr * mw).sum(axis=1) / sw
    cov = ((dy - mx[:, None]) * (mc_tr - my[:, None]) * mw).sum(axis=1)
    var = (((dy - mx[:, None]) ** 2) * mw).sum(axis=1)
    prop = cov / np.maximum(var, 1e-9)

    names = ["trough_uv", "peak_after_uv", "peak_before_uv", "ptp_uv",
             "trough_to_peak_samp", "fwhm_samp", "repol_slope", "recov_slope",
             "peak_trough_asym", "energy", "trough_area", "zero_crossings",
             "wave_skew", "wave_kurtosis", "f0_hz", "spectral_centroid_hz",
             "fit_y_um", "fit_z_um", "fit_A", "fit_r2",
             "fp_spread_um", "fp_asym", "n_chan_half", "prop_slope",
             "subsample_shift"]
    F = np.column_stack([trough, peak_after, peak_before, ptp, t2p, fwhm,
                         repol, rec, asym, energy, area, zc, sk, ku, f0, cen,
                         y, z, A, r2, spread, fp_asym, n_half, prop, d])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez(os.path.join(cache_dir(rep), "feat.npz"),
             F=F.astype(np.float32), names=np.array(names), done=done,
             wav_aligned=wa.astype(np.float32))
    print("  %d axes x %d events -> feat.npz" % (F.shape[1], F.shape[0]))
    print()
    print("%-22s %10s %10s %10s %10s"
          % ("axis", "median", "p10", "p90", "sd"))
    for i, nm in enumerate(names):
        c = F[done, i]
        print("%-22s %10.3f %10.3f %10.3f %10.3f"
              % (nm, np.median(c), np.percentile(c, 10),
                 np.percentile(c, 90), c.std()))


if __name__ == "__main__":
    if sys.argv[1] == "--extract":
        extract(int(sys.argv[2]))
    elif sys.argv[1] == "--features":
        features(int(sys.argv[2]))
    else:
        print(__doc__)
