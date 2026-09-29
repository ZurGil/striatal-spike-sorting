"""
Real-data version of demo_shape_dependence.py's frequency-mismatch check.
Uses 5 REAL units from session 20260901_085606 (already fully recorded,
untouched on D: -- NOT today's live recording), spanning a wide range of
real peak-to-trough widths, pulled from Kilosort's own templates.npy
(small file, ~38MB, loaded once -- no probe1.dat / raw-trace reads at all,
so this is very light I/O, nothing like the earlier 600GB backup).

Answers: (1) do real units actually show the frequency-mismatch problem
found on synthetic shapes? (2) is there a simple, usable relationship
between a unit's own measured width and its best-matching probe
frequency, so a per-unit probe could be chosen from one cheap
measurement instead of a full frequency sweep every time?

Usage: python demo_real_unit_shapes.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
NT0MIN = 20  # this dataset's real Kilosort alignment convention

UNITS = [224, 398, 176, 345, 86]  # spanning real width 2, 5, 9, 13, 17 samples

templates = np.load(VR + r"\templates.npy")  # (n_units, 61, 384) -- small, one-time load
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")

t_axis = (np.arange(61) - NT0MIN) / FS * 1000

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(UNITS)))
shapes, widths_samples, best_f0s = [], [], []

ax = axes[0]
for uid, color in zip(UNITS, colors):
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    wf = templ_all[:, peak_ch]
    width = abs(np.argmin(wf) - np.argmax(wf))
    n_spikes = int(info.loc[uid, "n_spikes"])
    shapes.append(wf)
    widths_samples.append(width)
    ax.plot(t_axis, wf / np.abs(wf).max(), color=color, lw=1.8,
            label=f"unit {uid} (ch{peak_ch}, {n_spikes} spikes, width={width} samp)")
ax.axhline(0, color="grey", lw=0.4)
ax.set_xlabel("time relative to Kilosort's detected spike sample (ms)")
ax.set_ylabel("normalized amplitude (own peak = 1)")
ax.legend(fontsize=7)
ax.set_title("5 REAL units, session 20260901_085606,\n"
              "chosen to span the real width distribution\n"
              "(actual real widths ranged 2-17 samples across 93 good units)", fontsize=9.5)

ax = axes[1]
f0_scan = np.linspace(300, 4000, 60)
pad_scan = int(np.ceil(3.0 * FS / (2 * f0_scan.min()))) + 20
for uid, wf, color in zip(UNITS, shapes, colors):
    trace = np.zeros(61 + 2 * pad_scan)
    true_center = pad_scan + NT0MIN
    trace[true_center - NT0MIN: true_center - NT0MIN + 61] = wf
    mags = []
    for f0_test in f0_scan:
        _, psi_test = make_morlet(f0_test, FS, n_cycles=3.0)
        w = wavelet_transform_at(trace, psi_test, true_center)
        mags.append(abs(w))
    mags = np.array(mags)
    assert not np.any(np.isnan(mags)), f"unit {uid}: NaNs in scan, pad_scan too small"
    best_f0 = f0_scan[np.nanargmax(mags)]
    best_f0s.append(best_f0)
    ax.plot(f0_scan, mags / mags.max() * 100, color=color, lw=1.8,
            label=f"unit {uid}: best f0 = {best_f0:.0f} Hz")
    ax.axvline(best_f0, color=color, lw=0.7, ls=":")
ax.axvline(2000, color="grey", lw=1.3, label="fixed 2000Hz probe (used in all earlier demos)")
ax.set_xlabel("probe frequency f0 tested (Hz)")
ax.set_ylabel("|W| achieved (% of that unit's own best)")
ax.legend(fontsize=6.8, loc="upper right")
ax.set_title("Real frequency-match scan per unit --\n"
              "each unit's own best frequency, found the same way\n"
              "as the synthetic MSN/broad-shape test earlier", fontsize=9.5)

ax = axes[2]
ax.scatter(widths_samples, best_f0s, c=colors, s=90, zorder=5)
for uid, wi, f0i in zip(UNITS, widths_samples, best_f0s):
    ax.annotate(f"unit {uid}", (wi, f0i), textcoords="offset points", xytext=(6, 6), fontsize=8)
# simple candidate rule: best_f0 ~ k / width  (an oscillation that completes
# in time proportional to the peak-to-trough width -- the simplest possible
# physical guess, tested here rather than assumed)
widths_arr = np.array(widths_samples, dtype=float)
best_f0_arr = np.array(best_f0s)
k = np.median(best_f0_arr * widths_arr)
fit_widths = np.linspace(min(widths_arr) * 0.8, max(widths_arr) * 1.2, 50)
ax.plot(fit_widths, k / fit_widths, color="grey", lw=1.2, ls="--",
        label=f"candidate rule: f0 ~ {k:.0f} / width_samples")
ax.set_xlabel("unit's own peak-to-trough width (samples)")
ax.set_ylabel("that unit's best-matching probe frequency (Hz)")
ax.legend(fontsize=8)
ax.set_title("Does width alone predict the best frequency?\n"
              "(if these 5 real points roughly follow the dashed\n"
              "curve, a per-unit rule is realistic -- one cheap width\n"
              "measurement instead of a full sweep every time)", fontsize=9.5)

plt.suptitle("Real units, real templates: frequency mismatch confirmed on real data, and a candidate per-unit rule",
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_09_real_units.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)

print("\nunit  width(samples)  best_f0(Hz)  width*best_f0")
for uid, wi, f0i in zip(UNITS, widths_samples, best_f0s):
    print(f"{uid:4d}  {wi:14d}  {f0i:11.0f}  {wi*f0i:13.0f}")
