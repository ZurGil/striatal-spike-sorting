"""
Visual, step-by-step walkthrough of wavelet_features.py, entirely on
synthetic data (no disk I/O -- safe during a live recording). Produces
SEVERAL separate figures (not one crowded grid), each explaining one
concrete computation.

Usage: python demo_wavelet_plots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, wavelet_transform_at, scan_wavelet_transform,
                               calibrate_reference_phase, sub_sample_shift_from_phase)

OUT = r"D:\Gil\spike_sorting_agent\outputs"
fs = 30000.0
n = 61
nt0min = 20
t_samples_spike = np.arange(n) - nt0min
template = (1.0 * np.exp(-0.5 * (t_samples_spike / 2.2) ** 2)
            - 0.55 * np.exp(-0.5 * ((t_samples_spike - 7) / 5.0) ** 2))
template -= template[:5].mean()
amp_uv = 150.0
template = template * amp_uv

f0_hz = 2000.0
t_psi, psi = make_morlet(f0_hz, fs, n_cycles=3.0)
t_psi_ms = t_psi / fs * 1000
ref_phase = calibrate_reference_phase(template, psi, fs, align_index=nt0min)
pad = 40


# ============================================================
# FIGURE 1: the ACTUAL arithmetic behind one point of the blue
# curve in figure 2 -- how |W(t0)| gets computed, step by step.
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

trace = np.zeros(n + 2 * pad)
true_center = pad + nt0min
trace[true_center - nt0min: true_center - nt0min + n] = template
half = len(psi) // 2
t0 = true_center  # evaluate W exactly on-target for this walkthrough
lo, hi = t0 - half, t0 - half + len(psi)
x_segment = trace[lo:hi]
t_local_ms = (np.arange(lo, hi) - t0) / fs * 1000

ax = axes[0]
ax.plot(t_local_ms, x_segment, color="#222", lw=2.2, label="trace x(tau), the spike itself")
ax2 = ax.twinx()
ax2.plot(t_local_ms, psi.real, color="#1f77b4", lw=1.2, label="probe real part: cos(2*pi*f0*tau)*window")
ax2.plot(t_local_ms, -psi.imag, color="#d62728", lw=1.2, label="probe conj. imag part: -sin(2*pi*f0*tau)*window")
ax.set_title("STEP 1 of 4\nLine up the probe on the trace\nat the candidate time t0", fontsize=10)
ax.set_xlabel("time relative to t0 (ms)")
ax.set_ylabel("x(tau), the spike (uV)")
ax2.set_ylabel("probe (a.u.)")
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=6.3, loc="lower left")

prod_real = x_segment * psi.real
prod_imag = x_segment * (-psi.imag)
ax = axes[1]
ax.plot(t_local_ms, prod_real, color="#1f77b4", lw=1.4, label="x(tau) * probe_real(tau)")
ax.plot(t_local_ms, prod_imag, color="#d62728", lw=1.4, label="x(tau) * probe_conj_imag(tau)")
ax.axhline(0, color="grey", lw=0.5)
ax.set_title("STEP 2 of 4\nMultiply, point by point\n(this is what conj(psi) means)", fontsize=10)
ax.set_xlabel("time relative to t0 (ms)")
ax.set_ylabel("product at each tau")
ax.legend(fontsize=7)

cum_real = np.cumsum(prod_real)
cum_imag = np.cumsum(prod_imag)
ax = axes[2]
ax.plot(t_local_ms, cum_real, color="#1f77b4", lw=1.6, label="running total, real part")
ax.plot(t_local_ms, cum_imag, color="#d62728", lw=1.6, label="running total, imag part")
ax.axhline(cum_real[-1], color="#1f77b4", lw=0.6, ls=":")
ax.axhline(cum_imag[-1], color="#d62728", lw=0.6, ls=":")
ax.set_title("STEP 3 of 4\nAdd up the products left to right\n(final value = the running total's end)", fontsize=10)
ax.set_xlabel("time relative to t0 (ms)")
ax.set_ylabel("cumulative sum so far")
ax.legend(fontsize=7)

W = complex(cum_real[-1], cum_imag[-1])
ax = axes[3]
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.plot([0, W.real], [0, W.imag], color="#333", lw=1.8, marker="o", markevery=[1], ms=8)
circle = plt.Circle((0, 0), abs(W), fill=False, color="#888", lw=0.7, ls="--")
ax.add_patch(circle)
ax.set_xlim(-abs(W) * 1.3, abs(W) * 1.3)
ax.set_ylim(-abs(W) * 1.3, abs(W) * 1.3)
ax.set_aspect("equal")
ax.annotate(f"W(t0) = {W.real:.0f} + {W.imag:.0f}i", (W.real, W.imag),
            textcoords="offset points", xytext=(10, 10), fontsize=8.5)
ax.annotate(f"|W| = {abs(W):.0f}\n(length of the arrow)", (0, 0),
            textcoords="offset points", xytext=(-95, 40), fontsize=8.5, color="#1f4e8c")
ax.annotate(f"angle = {np.degrees(np.angle(W)):.1f}\n(direction of the arrow)", (0, 0),
            textcoords="offset points", xytext=(-95, -55), fontsize=8.5, color="#a83232")
ax.set_title("STEP 4 of 4\nThe two running totals ARE the final\nanswer: plot them as one point", fontsize=10)
ax.set_xlabel("real part of W")
ax.set_ylabel("imaginary part of W")

plt.suptitle("How |W(t0)| and the phase are computed, worked through on one real example", fontsize=13)
plt.tight_layout()
out1 = os.path.join(OUT, "wavelet_01_how_W_is_computed.png")
plt.savefig(out1, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out1)


# ============================================================
# FIGURE 2: shift-tolerance -- what the blue and red lines ARE,
# stated explicitly, plus the same worked computation repeated
# at increasing misalignment so you can see the arrow shrink.
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

shifts = np.arange(0, 8)
Ws = []
mf_scores = []
mf_template_norm = template / np.linalg.norm(template)
for shift in shifts:
    trace2 = np.zeros(n + 2 * pad)
    lo2 = true_center - nt0min + shift
    trace2[lo2:lo2 + n] = template
    w = wavelet_transform_at(trace2, psi, true_center)
    Ws.append(w)
    mf_window = trace2[true_center - nt0min: true_center - nt0min + n]
    mf_scores.append(np.dot(mf_window, mf_template_norm))
Ws = np.array(Ws)
mf_scores = np.array(mf_scores)

ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(shifts)))
for i, (w, c) in enumerate(zip(Ws, colors)):
    ax.plot([0, w.real], [0, w.imag], color=c, lw=1.6, marker="o", markevery=[1], ms=6,
            label=f"shift={shifts[i]}")
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.set_aspect("equal")
ax.set_xlabel("real part of W")
ax.set_ylabel("imaginary part of W")
ax.set_title("Left: each arrow is W(t0) for one misalignment\n"
              "(same STEP 4 as figure 1, repeated for shift=0..7)\n"
              "arrows barely shrink -- that's the shift-tolerance", fontsize=10)
ax.legend(fontsize=6.5, ncol=2, loc="lower left")

ax = axes[1]
ax.plot(shifts, np.abs(Ws) / np.abs(Ws[0]) * 100, marker="o", color="#1f77b4",
        label="wavelet |W(t0)|  (arrow LENGTH from the left panel)")
ax.plot(shifts, mf_scores / mf_scores[0] * 100, marker="s", color="#d62728",
        label="matched-filter score  = dot(trace_window, normalized_template)\n"
              "a single real number, no arrow/phase involved")
ax.axhline(100, color="grey", lw=0.5, ls=":")
ax.set_xlabel("misalignment between t0 and the true spike time (samples)")
ax.set_ylabel("% of the score at perfect alignment (shift=0)")
ax.set_title("Right: same data as % of the shift=0 value", fontsize=10)
ax.legend(fontsize=7.5)

plt.suptitle("Why the wavelet magnitude tolerates misalignment better than a matched filter", fontsize=13)
plt.tight_layout()
out2 = os.path.join(OUT, "wavelet_02_shift_tolerance.png")
plt.savefig(out2, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out2)


# ============================================================
# FIGURE 3: the calibration fix, explained as a rotation.
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

interp_fn = interp1d(np.arange(n), template, kind="cubic", bounds_error=False, fill_value=0.0)

ax = axes[0]
trace0 = np.zeros(n + 2 * pad)
trace0[true_center - nt0min: true_center - nt0min + n] = template
w0 = wavelet_transform_at(trace0, psi, true_center)
ax.plot([0, w0.real], [0, w0.imag], color="#333", lw=1.8, marker="o", markevery=[1], ms=8)
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.set_aspect("equal")
lim = abs(w0) * 1.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.annotate(f"angle = {np.degrees(ref_phase):.1f} deg\n(NOT 0, even though\nthis IS the true spike\ntime -- this is the bug)",
            (w0.real, w0.imag), textcoords="offset points", xytext=(10, -10), fontsize=8.5)
ax.set_title("(1) Measure W at the KNOWN-correct\ntime, on the rested template alone.\nIts angle is the template's own\n'shape phase', not a timing error.", fontsize=9.5)

ax = axes[1]
true_deltas = np.linspace(-3, 3, 13)
recovered_uncal, recovered_cal = [], []
for true_delta in true_deltas:
    idxv = np.arange(n) - true_delta
    trace3 = np.zeros(n + 2 * pad)
    trace3[true_center - nt0min: true_center - nt0min + n] = interp_fn(idxv)
    w = wavelet_transform_at(trace3, psi, true_center)
    recovered_uncal.append(sub_sample_shift_from_phase(np.angle(w), f0_hz, fs))
    recovered_cal.append(sub_sample_shift_from_phase(np.angle(w), f0_hz, fs, reference_phase=ref_phase))
recovered_uncal = np.array(recovered_uncal)
recovered_cal = np.array(recovered_cal)
ax.plot(true_deltas, true_deltas, color="grey", lw=1, ls="--", label="perfect (y=x)")
ax.plot(true_deltas, recovered_uncal, color="#d62728", marker=".", label="(2a) angle used directly -- biased")
ax.plot(true_deltas, recovered_cal, color="#2ca02c", marker=".",
        label=f"(2b) angle MINUS {np.degrees(ref_phase):.1f} deg -- fixed")
mean_err_uncal = np.mean(np.abs(recovered_uncal - true_deltas))
mean_err_cal = np.mean(np.abs(recovered_cal - true_deltas))
ax.text(0.03, 0.97, f"mean |error|:\nuncalibrated = {mean_err_uncal:.2f} samples\ncalibrated = {mean_err_cal:.2f} samples",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
ax.set_xlabel("true sub-sample shift (samples)")
ax.set_ylabel("recovered shift (samples)")
ax.set_title("(2) Same test spike, shifted by known\namounts: subtracting that reference\nangle removes almost all the bias", fontsize=9.5)
ax.legend(fontsize=6.8, loc="lower right")

ax = axes[2]
example_delta = 1.5
idxv = np.arange(n) - example_delta
trace4 = np.zeros(n + 2 * pad)
trace4[true_center - nt0min: true_center - nt0min + n] = interp_fn(idxv)
w_ex = wavelet_transform_at(trace4, psi, true_center)
w_corrected = w_ex * np.exp(-1j * ref_phase)
lim2 = max(abs(w_ex), abs(w_corrected)) * 1.3
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.plot([0, w_ex.real], [0, w_ex.imag], color="#d62728", lw=1.6, marker="o", markevery=[1], ms=7,
        label=f"raw W, angle={np.degrees(np.angle(w_ex)):.1f} deg")
ax.plot([0, w_corrected.real], [0, w_corrected.imag], color="#2ca02c", lw=1.6, marker="o", markevery=[1], ms=7,
        label=f"rotated by -{np.degrees(ref_phase):.1f} deg\nangle={np.degrees(np.angle(w_corrected)):.1f} deg")
ax.set_xlim(-lim2, lim2)
ax.set_ylim(-lim2, lim2)
ax.set_aspect("equal")
ax.legend(fontsize=7.3, loc="lower left")
ax.set_title("(3) What 'calibrating' does geometrically:\nrotate the arrow by the reference angle\nbefore reading off shift from its angle", fontsize=9.5)

plt.suptitle("The calibration fix: subtracting the template's own 'shape phase' before using angle as a timing cue", fontsize=13)
plt.tight_layout()
out3 = os.path.join(OUT, "wavelet_03_calibration_fix.png")
plt.savefig(out3, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out3)


# ============================================================
# FIGURE 4: the wraparound failure -- phase itself, then the
# shift estimate it produces, with the REAL measured error
# printed for the safe zone (not just "looks close").
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

half_cycle = (fs / f0_hz) / 2.0
full_cycle = fs / f0_hz
true_deltas_wide = np.linspace(-2 * full_cycle, 2 * full_cycle, 200)
phases_wide, recovered_wide = [], []
for true_delta in true_deltas_wide:
    idxv = np.arange(n) - true_delta
    trace5 = np.zeros(n + 2 * pad)
    trace5[true_center - nt0min: true_center - nt0min + n] = interp_fn(idxv)
    w = wavelet_transform_at(trace5, psi, true_center)
    ang = np.angle(np.exp(1j * (np.angle(w) - ref_phase)))
    phases_wide.append(np.degrees(ang))
    recovered_wide.append(sub_sample_shift_from_phase(np.angle(w), f0_hz, fs, reference_phase=ref_phase))
phases_wide = np.array(phases_wide)
recovered_wide = np.array(recovered_wide)

ax = axes[0]
ax.plot(true_deltas_wide, phases_wide, color="#9467bd", lw=1.4)
ax.axvspan(-half_cycle, half_cycle, color="#2ca02c", alpha=0.12)
for k in [-2, -1, 1, 2]:
    ax.axvline(k * full_cycle, color="grey", lw=0.5, ls=":")
ax.set_ylabel("measured phase (degrees, -180 to +180)")
ax.set_title("TOP: the raw measured angle itself. It can only ever read between -180 and\n"
              "+180 degrees, so it MUST snap back to -180 every time the true shift moves\n"
              "past +180 degrees' worth of delay (dotted lines = one full oscillation cycle,\n"
              f"= {full_cycle:.1f} samples, apart)", fontsize=10)

ax = axes[1]
ax.plot(true_deltas_wide, true_deltas_wide, color="grey", lw=1, ls="--", label="if recovery were perfect")
ax.plot(true_deltas_wide, recovered_wide, color="#9467bd", lw=1.4, label="what the formula actually reports")
ax.axvspan(-half_cycle, half_cycle, color="#2ca02c", alpha=0.12, label="trustworthy region")
safe_mask = np.abs(true_deltas_wide) <= half_cycle
safe_err = np.abs(recovered_wide[safe_mask] - true_deltas_wide[safe_mask])
ax.text(0.02, 0.97,
        f"inside the green band:\n  error ranges {safe_err.min():.2f} to {safe_err.max():.2f} samples\n"
        f"  (worst at the edges, ~0 at center)\noutside the green band:\n  wrong by several samples, no warning given",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
ax.set_xlabel("true shift (samples)")
ax.set_ylabel("shift the formula reports (samples)")
ax.set_title("BOTTOM: same data converted from degrees to samples. Same snap-back,\n"
              "now visible as the sawtooth jumps.", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

plt.suptitle("Direct answer: does timing recovery work? Yes, imperfectly, ONLY inside the green band",
             fontsize=13)
plt.tight_layout()
out4 = os.path.join(OUT, "wavelet_04_wraparound.png")
plt.savefig(out4, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out4)


# ============================================================
# FIGURE 5: narrow vs wide probe -- show the probes themselves
# stacked directly above the localization curves they produce.
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

_, psi_narrow = make_morlet(f0_hz, fs, n_cycles=1.5)
_, psi_wide = make_morlet(f0_hz, fs, n_cycles=8.0)
t_narrow_ms = (np.arange(len(psi_narrow)) - len(psi_narrow) // 2) / fs * 1000
t_wide_ms = (np.arange(len(psi_wide)) - len(psi_wide) // 2) / fs * 1000

ax = axes[0, 0]
ax.plot(t_narrow_ms, psi_narrow.real, color="#1f77b4", lw=1.2)
ax.plot(t_narrow_ms, np.abs(psi_narrow), color="#1f77b4", lw=1, ls="--")
ax.set_title(f"NARROW probe: only 1.5 oscillation cycles\nfit inside the window ({len(psi_narrow)} samples wide)", fontsize=9.5)
ax.set_xlabel("time (ms)")
ax.set_xlim(-2, 2)

ax = axes[0, 1]
ax.plot(t_wide_ms, psi_wide.real, color="#d62728", lw=1.2)
ax.plot(t_wide_ms, np.abs(psi_wide), color="#d62728", lw=1, ls="--")
ax.set_title(f"WIDE probe: 8 oscillation cycles fit\ninside the window ({len(psi_wide)} samples wide)", fontsize=9.5)
ax.set_xlabel("time (ms)")
ax.set_xlim(-2, 2)

window_samples = 40
pad5 = max(len(psi_narrow), len(psi_wide)) // 2 + window_samples + 5  # big enough that
# BOTH probes have valid (non-nan) coverage across the full +/-window_samples scan range --
# the earlier version used the same pad=40 as the rest of this file, which was narrower
# than the wide probe's own 121-sample window, silently clipping its scan to [0,20]
# instead of the intended [-40,40]. Caught by the bottom-right panel's x-axis only
# reaching 20 instead of 40 -- not a property of the method, a test-setup bug.
true_center5 = pad5 + nt0min
trace6 = np.zeros(n + 2 * pad5)
trace6[true_center5 - nt0min: true_center5 - nt0min + n] = template
xs = np.arange(-window_samples, window_samples + 1)
widths = {}
ax_bottom_left = axes[1, 0]
ax_bottom_right = axes[1, 1]
for label, p, color, ax_here in [("narrow probe", psi_narrow, "#1f77b4", ax_bottom_left),
                                  ("wide probe", psi_wide, "#d62728", ax_bottom_right)]:
    scan = scan_wavelet_transform(trace6, p)
    mag = np.abs(scan)
    seg = mag[true_center5 - window_samples: true_center5 + window_samples + 1]
    seg = seg / np.nanmax(seg) * 100
    ax_here.plot(xs, seg, color=color, lw=1.8)
    ax_here.axhline(50, color="grey", lw=0.6, ls=":")
    above = seg >= 50
    width = int(above.sum())
    widths[label] = width
    ax_here.fill_between(xs, 0, seg, where=above, color=color, alpha=0.15)
    ax_here.set_title(f"{label}: sliding it across the spike and\n"
                       f"reading |W| at every position -- width\n"
                       f"above half-max = {width} samples", fontsize=9.5)
    ax_here.set_xlabel("candidate t0, as sample offset from the true spike center")
    ax_here.set_ylabel("|W(t0)| (% of this probe's own peak)")

plt.suptitle(f"Why window width trades off timing precision: narrow (top-left) makes a sharper,\n"
             f"more localized response (bottom-left, {widths['narrow probe']} samples) than wide (bottom-right, {widths['wide probe']} samples)",
             fontsize=12)
plt.tight_layout()
out5 = os.path.join(OUT, "wavelet_05_narrow_vs_wide.png")
plt.savefig(out5, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out5)
