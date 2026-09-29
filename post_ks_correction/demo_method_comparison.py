"""
Direct comparison of THREE ways to find a spike's sub-sample timing,
across a WIDE range of true shifts (-20 to +20 samples), answering:
  - "can we increase the 0.25ms correctable range?"
  - "is there a better method?"
  - "what if we just align by peak?"

Method 1: PHASE-ONLY (everything shown so far) -- fixed 2000Hz probe,
  phase-based correction directly. Known to fail outside +/-7.5 samples.

Method 2: PEAK INTERPOLATION -- find the discrete sample nearest the
  peak, then fit a parabola through it and its two neighbors to estimate
  the true sub-sample peak location (a standard, simple, widely-used
  technique -- this is the "align by peak" alternative asked about).

Method 3: COARSE-THEN-FINE -- first do a coarse INTEGER-sample search
  (try every whole-sample shift in a wide range, keep the best-matching
  one via a simple matched-filter score), THEN apply the phase correction
  only to the small LEFTOVER fractional part. This is the two-stage
  approach the wavelet module's docstring describes but never
  demonstrated end-to-end until now.

Usage: python demo_method_comparison.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, sub_sample_shift_from_phase

OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
N = 61
NT0MIN = 20

t = np.arange(N) - NT0MIN
template = (1.0 * np.exp(-0.5 * (t / 2.2) ** 2) - 0.55 * np.exp(-0.5 * ((t - 7) / 5.0) ** 2))
template -= template[:5].mean()
template *= 150.0
interp_fn = interp1d(np.arange(N), template, kind="cubic", bounds_error=False, fill_value=0.0)

f0_hz = 2000.0
_, psi = make_morlet(f0_hz, FS, n_cycles=3.0)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
half_cycle = (FS / f0_hz) / 2.0
pad = 60
rng = np.random.default_rng(3)
NOISE_SIGMA = 0.05 * 150.0


def make_noisy_trace(true_delta):
    idx = np.arange(N) - true_delta
    clean = interp_fn(idx)
    noisy = clean + rng.normal(0, NOISE_SIGMA, size=N)
    trace = np.zeros(N + 2 * pad)
    true_center = pad + NT0MIN
    trace[true_center - NT0MIN: true_center - NT0MIN + N] = noisy
    return trace, true_center


def method_phase_only(trace, true_center):
    w = wavelet_transform_at(trace, psi, true_center)
    return sub_sample_shift_from_phase(np.angle(w), f0_hz, FS, reference_phase=ref_phase)


def method_peak_interp(trace, true_center):
    # search a modest integer window for the discrete sample extremum
    # (this template's global peak is at t=0 -- the max), then refine with
    # a 3-point parabolic fit around it
    search = trace[true_center - 25: true_center + 25]
    k = np.argmax(search)  # index within the search window
    if k == 0 or k == len(search) - 1:
        return float(k - 25)  # edge case, no room to interpolate
    y_m1, y_0, y_p1 = search[k - 1], search[k], search[k + 1]
    denom = (y_m1 - 2 * y_0 + y_p1)
    delta = 0.5 * (y_m1 - y_p1) / denom if abs(denom) > 1e-9 else 0.0
    discrete_shift = (k - 25)
    return discrete_shift + delta


def method_coarse_then_fine(trace, true_center):
    mf_norm = template / np.linalg.norm(template)
    best_score, best_shift = -np.inf, 0
    for candidate_shift in range(-25, 26):
        window = trace[true_center - NT0MIN + candidate_shift: true_center - NT0MIN + candidate_shift + N]
        if len(window) != N:
            continue
        score = np.dot(window, mf_norm)
        if score > best_score:
            best_score, best_shift = score, candidate_shift
    # fine correction: re-evaluate phase AT the coarse-corrected center --
    # the residual should now be small, safely inside the phase method's
    # trustworthy half-cycle window
    fine_center = true_center + best_shift
    if fine_center - psi.shape[0] // 2 < 0 or fine_center + psi.shape[0] // 2 >= len(trace):
        return float(best_shift)
    w = wavelet_transform_at(trace, psi, fine_center)
    fine_correction = sub_sample_shift_from_phase(np.angle(w), f0_hz, FS, reference_phase=ref_phase)
    return best_shift + fine_correction


true_deltas = np.linspace(-20, 20, 41)
results = {"phase-only": [], "peak-interpolation": [], "coarse-then-fine": []}
n_trials = 8
for true_delta in true_deltas:
    errs = {k: [] for k in results}
    for _ in range(n_trials):
        trace, true_center = make_noisy_trace(true_delta)
        errs["phase-only"].append(abs(method_phase_only(trace, true_center) - true_delta))
        errs["peak-interpolation"].append(abs(method_peak_interp(trace, true_center) - true_delta))
        errs["coarse-then-fine"].append(abs(method_coarse_then_fine(trace, true_center) - true_delta))
    for k in results:
        results[k].append(np.mean(errs[k]))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
colors = {"phase-only": "#9467bd", "peak-interpolation": "#2ca02c", "coarse-then-fine": "#1f77b4"}
for k, color in colors.items():
    ax.plot(true_deltas, np.array(results[k]) / FS * 1000, color=color, lw=1.8, label=k)
ax.axvspan(-half_cycle, half_cycle, color="grey", alpha=0.08)
ax.set_xlabel("true shift (samples)")
ax.set_ylabel("mean absolute error (ms)")
ax.set_yscale("log")
ax.set_title("Error across a WIDE true-shift range (-20 to +20 samples)\n"
              "grey band = phase-only's old safe zone (+/-7.5 samples)", fontsize=10)
ax.legend(fontsize=8.5)

ax = axes[1]
in_band = np.abs(true_deltas) <= half_cycle
out_band = ~in_band
labels = ["inside old safe zone\n(-7.5 to +7.5 samples)", "OUTSIDE old safe zone\n(the rest of the -20..20 range)"]
x = np.arange(2)
width_bar = 0.25
for i, (k, color) in enumerate(colors.items()):
    means = [np.mean(np.array(results[k])[in_band]), np.mean(np.array(results[k])[out_band])]
    means_ms = [m / FS * 1000 for m in means]
    ax.bar(x + i * width_bar, means_ms, width_bar, color=color, label=k)
ax.set_xticks(x + width_bar)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("mean absolute error (ms)")
ax.set_yscale("log")
ax.legend(fontsize=8.5)
ax.set_title("Same data, summarized: which method survives\ngoing outside the old narrow safe zone?", fontsize=10)

plt.suptitle("Is there a better method than phase-only? Yes -- coarse-then-fine keeps phase's precision\nwithout the narrow range limit; peak-interpolation is simpler but less precise",
             fontsize=12.5)
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_14_method_comparison.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)

for k in results:
    means = np.array(results[k])
    print(f"{k:22s}: inside safe zone mean err = {np.mean(means[in_band])/FS*1000:.4f} ms, "
          f"outside = {np.mean(means[out_band])/FS*1000:.4f} ms")
