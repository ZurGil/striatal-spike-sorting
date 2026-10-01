"""
Multi-frequency sub-sample alignment, and the ambiguity flag that comes free
with it.

THE IDEA IN ONE LINE: if an event really is this unit's template arriving at
time delta, then its measured phase is a STRAIGHT LINE in frequency. Fit that
line -- the slope gives the delay, and the leftover scatter says whether it
was really one spike.

WHY THE LINE IS STRAIGHT. A pure time shift multiplies the spectrum by
exp(-i*2*pi*f*delta/fs), so after removing the template's own shape phase
(calibrate_reference_phase, per frequency):

    dphi(f) = -2*pi*f*delta/fs                                   (*)

Phase is linear in frequency with slope proportional to the delay. That is
the ONLY thing a time shift can do. A collision -- two spikes at different
times -- is a SUM of two shifted templates, and the phase of a sum is not a
linear ramp; it bends. So the residual of a straight-line fit is a direct
test of "was this one spike or more than one".

TWO THINGS FROM ONE REGRESSION:

  1. A BETTER DELAY. One probe frequency gives one estimate. K probes give K,
     each with its own precision. From the single-frequency analysis (section
     5o of the research log):

         sigma_k  proportional to  fs / ( f_k * |W(f_k)| )

     so inverse-variance weighting uses w_k proportional to
     (f_k*|W(f_k)|)^2. Note what this means: the existing single-frequency
     rule, argmax f0*|W(f0)|, is just the K=1 case of keeping only the
     heaviest weight. This uses all of them instead of discarding the rest.
     The gain is less than sqrt(K) because the probes share one noise
     realization and their wavelet supports overlap -- measure it, do not
     assume it.

  2. AN AMBIGUITY FLAG. The weighted scatter of the per-probe estimates
     around the fitted line, in SAMPLES, is a per-spike statistic answering
     "how badly does this event fail to be a single shifted copy of the
     template?" Nothing in Kilosort, and nothing else in this project, can
     currently say "I do not know when this spike happened." This can. It is
     also what makes it safe to align spikes BEFORE clustering: correct the
     confident ones and flag the rest, rather than forcing a wrong
     correction onto a collision.

PHASE WRAPPING. Equation (*) is only invertible while |2*pi*f*delta/fs| < pi,
i.e. |delta| < fs/(2*f). At fs=30000 and f=6000 that is 2.5 samples, so the
existing coarse-then-fine structure (get within about a sample on magnitude,
then refine on phase) already protects this. `max_unambiguous_shift` reports
the limit for a given bank so a caller can check rather than assume.

This module deliberately does NOT change any existing behaviour. It is a new,
separate estimator; wavelet_features.coarse_then_fine_shift is untouched.
"""
import numpy as np

from wavelet_features import (make_morlet, wavelet_transform_at,
                               calibrate_reference_phase)


def build_probe_bank(template, fs, n_cycles=3.0, nt0min=None, n_probes=5,
                      f_lo=300.0, f_hi=8000.0, n_grid=60, keep_fraction=0.35):
    """Choose K probe frequencies for one unit's template, and precompute
    everything that does not depend on the data.

    The band is chosen from the template itself: score every candidate
    frequency by f*|W(f)| (the timing criterion established in 5o), then keep
    the contiguous band around the peak where the score stays above
    `keep_fraction` of its maximum, and place `n_probes` log-spaced
    frequencies across it. Probes outside that band would contribute almost
    no information while still adding noise to the fit.

    Returns a dict with:
      freqs            -- (K,) chosen frequencies in Hz
      psis             -- list of K complex probes
      reference_phases -- (K,) the template's own zero-shift phase per probe
      weights          -- (K,) inverse-variance weights, (f*|W|)^2, normalized
                          to sum to 1
      mags             -- (K,) |W(f)| of the template at each probe
      design           -- (K,) the regressor x_k = -2*pi*f_k/fs, so that
                          dphi_k = x_k * delta
    """
    template = np.asarray(template, dtype=float)
    n = len(template)
    if nt0min is None:
        nt0min = n // 2

    def mag_at(f):
        psi = make_morlet(f, fs, n_cycles=n_cycles)[1]
        half = len(psi) // 2
        pad = half + 5
        trace = np.zeros(n + 2 * pad)
        trace[pad:pad + n] = template
        return abs(wavelet_transform_at(trace, psi, pad + nt0min))

    f_grid = np.logspace(np.log10(f_lo), np.log10(f_hi), n_grid)
    mags = np.array([mag_at(f) for f in f_grid])
    scores = mags * f_grid
    i_pk = int(np.nanargmax(scores))
    thr = keep_fraction * scores[i_pk]

    lo = i_pk
    while lo > 0 and scores[lo - 1] >= thr:
        lo -= 1
    hi = i_pk
    while hi < len(f_grid) - 1 and scores[hi + 1] >= thr:
        hi += 1

    if hi == lo:
        freqs = np.array([f_grid[i_pk]])
    else:
        freqs = np.logspace(np.log10(f_grid[lo]), np.log10(f_grid[hi]),
                            max(int(n_probes), 1))

    psis, refs, m = [], [], []
    for f in freqs:
        psi = make_morlet(f, fs, n_cycles=n_cycles)[1]
        psis.append(psi)
        refs.append(calibrate_reference_phase(template, psi, fs,
                                              align_index=nt0min))
        m.append(mag_at(f))
    m = np.asarray(m)
    w = (freqs * m) ** 2
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    return dict(freqs=np.asarray(freqs, dtype=float), psis=psis,
                reference_phases=np.asarray(refs, dtype=float),
                weights=w, mags=m, fs=float(fs),
                design=-2.0 * np.pi * np.asarray(freqs, dtype=float) / float(fs),
                band_hz=(float(f_grid[lo]), float(f_grid[hi])),
                n_cycles=float(n_cycles))


def max_unambiguous_shift(bank):
    """Largest |delta| (in samples) the highest-frequency probe in this bank
    can measure before its phase wraps: fs / (2*f_max)."""
    return float(bank["fs"] / (2.0 * np.max(bank["freqs"])))


def phase_slope_delay(trace, center_index, bank):
    """Estimate the sub-sample delay of an event at `center_index`, and how
    much the probes disagree about it.

    Fits dphi_k = x_k * delta by weighted least squares THROUGH THE ORIGIN.
    There is no intercept, and that is physical rather than a convenience:
    the reference phases already remove the template's own shape phase, so a
    correctly-aligned event has dphi = 0 at every frequency. An intercept
    would silently absorb exactly the kind of inconsistency this is meant to
    detect.

    Returns dict:
      delta            -- the weighted-least-squares delay, in samples
      scatter_samples  -- weighted RMS disagreement of the per-probe
                          estimates around `delta`, in samples. THE
                          AMBIGUITY STATISTIC.
      per_probe_delta  -- (K,) each probe's own independent estimate
      dphi             -- (K,) measured phase differences, rewrapped
      mags             -- (K,) |W| measured on the DATA (not the template)
      weights, freqs   -- passed through from the bank
    """
    psis = bank["psis"]
    refs = bank["reference_phases"]
    w = bank["weights"]
    x = bank["design"]

    dphi = np.empty(len(psis))
    mags = np.empty(len(psis))
    for k, psi in enumerate(psis):
        W = wavelet_transform_at(trace, psi, center_index)
        mags[k] = abs(W)
        # complex division, not angle subtraction -- rewraps correctly near +-pi
        dphi[k] = np.angle(np.exp(1j * (np.angle(W) - refs[k])))

    denom = float(np.sum(w * x * x))
    delta = float(np.sum(w * x * dphi) / denom) if denom > 0 else np.nan

    per = dphi / x                      # each probe's own estimate, in samples
    resid = per - delta
    scatter = float(np.sqrt(np.sum(w * resid ** 2)))

    return dict(delta=delta, scatter_samples=scatter, per_probe_delta=per,
                dphi=dphi, mags=mags, weights=w, freqs=bank["freqs"])


def single_frequency_delay(trace, center_index, bank, which="best"):
    """The K=1 comparison: use only one probe, the heaviest-weighted one by
    default. This is what the existing pipeline does, exposed here so the
    multi-probe estimate can be compared against it on identical data."""
    k = int(np.argmax(bank["weights"])) if which == "best" else int(which)
    W = wavelet_transform_at(trace, bank["psis"][k], center_index)
    dphi = np.angle(np.exp(1j * (np.angle(W) - bank["reference_phases"][k])))
    return float(dphi / bank["design"][k])


def calibrate_ambiguity_threshold(scatters, percentile=95.0):
    """Turn a sample of scatter values measured on KNOWN-GOOD spikes of a
    unit into a flag threshold. Calibrating on the unit's own spikes keeps
    this from being a hand-tuned constant, the same discipline used for every
    other threshold in this project (see RESEARCH_LOG 5s on why a fixed
    percentile is not always the right choice -- the same caveat applies
    here, and the right percentile should be checked against a real impostor
    or collision population rather than assumed)."""
    s = np.asarray([v for v in scatters if np.isfinite(v)], dtype=float)
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, percentile))
