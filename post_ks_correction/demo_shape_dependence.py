"""
Does the choice of wavelet (fixed f0=2000Hz probe) matter for different
real spike shapes -- specifically (1) inverted polarity (MSN-like
negative-first vs TAN-like positive-first, per the design doc's explicit
callout) and (2) different width/frequency content (a broader, slower
waveform)?

Usage: python demo_shape_dependence.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, wavelet_transform_at, calibrate_reference_phase,
                               sub_sample_shift_from_phase)

OUT = r"D:\Gil\spike_sorting_agent\outputs"
fs = 30000.0
n = 61
nt0min = 20
t = np.arange(n) - nt0min
pad = 40

# MSN-like: negative deflection first, narrow (same shape used throughout
# this conversation so far)
msn = (1.0 * np.exp(-0.5 * (t / 2.2) ** 2) - 0.55 * np.exp(-0.5 * ((t - 7) / 5.0) ** 2))
msn -= msn[:5].mean()
msn *= 150.0

# TAN-like: SIGN-FLIPPED (positive-first, matching the design doc's
# explicit note that TANs have "positive-initial-deflection waveforms")
tan = -msn.copy()

# a third, separately-varied case: same POLARITY as MSN but genuinely
# BROADER in time (slower dominant frequency content) -- tests the
# frequency-mismatch question independent of the polarity question
broad = (1.0 * np.exp(-0.5 * (t / 5.5) ** 2) - 0.55 * np.exp(-0.5 * ((t - 16) / 11.0) ** 2))
broad -= broad[:5].mean()
broad *= 150.0

f0_hz = 2000.0
_, psi = make_morlet(f0_hz, fs, n_cycles=3.0)


def recovery_curve(template, label, color, ax):
    ref_phase = calibrate_reference_phase(template, psi, fs, align_index=nt0min)
    interp_fn = interp1d(np.arange(n), template, kind="cubic", bounds_error=False, fill_value=0.0)
    true_center = pad + nt0min
    true_deltas = np.linspace(-3, 3, 13)
    recovered = []
    for true_delta in true_deltas:
        trace = np.zeros(n + 2 * pad)
        idx = np.arange(n) - true_delta
        trace[true_center - nt0min: true_center - nt0min + n] = interp_fn(idx)
        w = wavelet_transform_at(trace, psi, true_center)
        recovered.append(sub_sample_shift_from_phase(np.angle(w), f0_hz, fs, reference_phase=ref_phase))
    recovered = np.array(recovered)
    mean_err = np.mean(np.abs(recovered - true_deltas))
    ax.plot(true_deltas, recovered, color=color, marker=".", label=f"{label} (mean err={mean_err:.2f} samp)")
    return mean_err, ref_phase


# ============================================================
# FIGURE 1: polarity -- does calibrated recovery still work for
# an upside-down (TAN-like) spike, using the SAME probe?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(t / fs * 1000, msn, color="#1f77b4", lw=1.8, label="MSN-like (negative first)")
ax.plot(t / fs * 1000, tan, color="#d62728", lw=1.8, label="TAN-like (positive first) -- literally upside-down")
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel("time (ms)")
ax.set_ylabel("uV")
ax.legend(fontsize=8.5)
ax.set_title("Two spike shapes: exact mirror images of each other\n"
             "(this is the real MSN-vs-TAN polarity difference from the design doc)", fontsize=9.5)

ax = axes[1]
ax.plot(np.linspace(-3, 3, 13), np.linspace(-3, 3, 13), color="grey", lw=1, ls="--", label="perfect recovery")
err_msn, ref_msn = recovery_curve(msn, "MSN-like", "#1f77b4", ax)
err_tan, ref_tan = recovery_curve(tan, "TAN-like (upside-down)", "#d62728", ax)
ax.set_xlabel("true sub-sample shift (samples)")
ax.set_ylabel("recovered shift (samples)")
ax.legend(fontsize=8)
ax.set_title(f"SAME probe (2000Hz) used for both, but each\n"
             f"gets its OWN calibration (ref angle: MSN={np.degrees(ref_msn):.0f} deg, "
             f"TAN={np.degrees(ref_tan):.0f} deg)\n"
             f"both recover about equally well -- polarity doesn't break it,\n"
             f"AS LONG AS each unit is calibrated on its own real shape", fontsize=9.5)

plt.suptitle("Does polarity (upside-down spikes) matter for the wavelet method?", fontsize=13)
plt.tight_layout()
out1 = os.path.join(OUT, "wavelet_07_polarity.png")
plt.savefig(out1, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out1)
print(f"MSN mean error: {err_msn:.3f} samples, TAN (upside-down) mean error: {err_tan:.3f} samples")


# ============================================================
# FIGURE 2: shape WIDTH / frequency content -- does a fixed
# f0=2000Hz probe work equally well for a narrow vs. a broad
# spike, or does the probe need to be tuned per shape?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(t / fs * 1000, msn / msn.max(), color="#1f77b4", lw=1.8, label="MSN-like (narrow, fast)")
ax.plot(t / fs * 1000, broad / broad.max(), color="#2ca02c", lw=1.8, label="broader (e.g. TAN-scale width), slower")
ax.axhline(0, color="grey", lw=0.4)
ax.set_xlabel("time (ms)")
ax.set_ylabel("normalized amplitude (peak=1)")
ax.legend(fontsize=8.5)
ax.set_title("Two DIFFERENT-WIDTH shapes, same polarity\n"
             "(isolating width/frequency-content, separate from the polarity question above)", fontsize=9.5)

ax = axes[1]
f0_scan = np.linspace(300, 4000, 60)
# the lowest frequencies tested need a much wider probe window (a 300Hz
# probe at 3 cycles is 301 samples long) than the pad=40 used everywhere
# else in this file -- BUG CAUGHT while building this: with pad=40, every
# f0 below ~1125Hz silently returned NaN (window ran past the padded
# trace's edge), and argmax over an array containing NaNs is unreliable,
# which is exactly why the first version of this plot came out empty with
# a bogus "300Hz is best" answer for both shapes. Fix: use a pad sized for
# the LOWEST frequency actually scanned, and use nanargmax so any
# remaining edge NaNs are excluded rather than silently corrupting the
# result.
pad_scan = int(np.ceil(3.0 * fs / (2 * f0_scan.min()))) + 20
for template, label, color in [(msn, "MSN-like (narrow)", "#1f77b4"), (broad, "broader shape", "#2ca02c")]:
    mags = []
    for f0_test in f0_scan:
        _, psi_test = make_morlet(f0_test, fs, n_cycles=3.0)
        trace = np.zeros(n + 2 * pad_scan)
        true_center = pad_scan + nt0min
        trace[true_center - nt0min: true_center - nt0min + n] = template
        w = wavelet_transform_at(trace, psi_test, true_center)
        mags.append(abs(w))
    mags = np.array(mags)
    assert not np.any(np.isnan(mags)), "still getting NaNs in the scan -- pad_scan too small"
    best_f0 = f0_scan[np.nanargmax(mags)]
    ax.plot(f0_scan, mags / mags.max() * 100, color=color, lw=1.8,
            label=f"{label}: best probe freq = {best_f0:.0f} Hz")
    ax.axvline(best_f0, color=color, lw=0.8, ls=":")
ax.axvline(2000, color="grey", lw=1.2, label="the fixed 2000Hz probe used everywhere so far")
ax.set_xlabel("probe frequency f0 tested (Hz)")
ax.set_ylabel("|W| achieved (% of that shape's own best)")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Scanning probe frequency against each shape:\n"
              "each shape has its OWN best-matching frequency", fontsize=9.5)

plt.suptitle("Does the SAME probe frequency work equally well for a narrow vs. a broad spike?", fontsize=13)
plt.tight_layout()
out2 = os.path.join(OUT, "wavelet_08_frequency_mismatch.png")
plt.savefig(out2, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out2)
