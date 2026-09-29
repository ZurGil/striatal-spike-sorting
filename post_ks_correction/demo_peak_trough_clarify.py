"""
Clarifies two things from the previous turn's real-unit analysis:

(1) HOW peak-to-trough width was actually computed: find the single
    sample where the waveform is most positive (the global maximum) and
    the single sample where it is most negative (the global minimum),
    and measure the distance between those two samples. This is a crude,
    TWO-POINT measurement -- it does not look at overall shape, and does
    NOT know or care whether there is a second, smaller trough elsewhere.

(2) The pipeline was SCAN, not FIT: for each real unit, a broad range of
    candidate frequencies was tried against that unit's own real template
    and the best-scoring one was kept (brute-force search). The
    width-vs-frequency "rule" was only checked AFTER the fact, against
    those 5 scan results -- frequency was never computed FROM width by
    formula; width was only tested afterward as a possible EXPLANATION
    for the scan results, and didn't explain them well.

Usage: python demo_peak_trough_clarify.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelmin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
NT0MIN = 20
UNITS = [224, 398, 176, 345, 86]
BEST_F0 = {224: 2808, 398: 1303, 176: 1053, 345: 802, 86: 739}

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
t_axis = (np.arange(61) - NT0MIN) / FS * 1000

fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
widths_report = []

for col, uid in enumerate(UNITS):
    ax = axes[col]
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    wf = templ_all[:, peak_ch]

    peak_idx = int(np.argmax(wf))       # THE single most-positive sample
    global_trough_idx = int(np.argmin(wf))  # THE single most-negative sample
    width = abs(global_trough_idx - peak_idx)
    widths_report.append((uid, peak_idx, global_trough_idx, width))

    # find ALL local minima (not just the global one), to check for a
    # second trough elsewhere in the waveform
    local_min_idx = argrelmin(wf, order=2)[0]
    # keep only "real" local minima: below zero and at least 10% of the
    # global trough's depth, to ignore tiny noise wiggles near baseline
    depth_thresh = 0.10 * abs(wf[global_trough_idx])
    local_min_idx = [i for i in local_min_idx if wf[i] < -depth_thresh]

    ax.plot(t_axis, wf, color="#333", lw=1.8)
    ax.axhline(0, color="grey", lw=0.4)
    ax.plot(t_axis[peak_idx], wf[peak_idx], marker="^", color="#d62728", ms=13, zorder=5,
            label="global max\n(the 'peak')")
    ax.plot(t_axis[global_trough_idx], wf[global_trough_idx], marker="v", color="#1f77b4", ms=13, zorder=5,
            label="global min\n(the 'trough')")
    secondary = [i for i in local_min_idx if i != global_trough_idx]
    if secondary:
        for i in secondary:
            ax.plot(t_axis[i], wf[i], marker="o", mfc="none", mec="#2ca02c", ms=13, mew=2, zorder=5)
        ax.plot([], [], marker="o", mfc="none", mec="#2ca02c", ms=10, mew=2, ls="none",
                label=f"OTHER local min(s)\n(ignored by this method!)")

    ax.annotate("", xy=(t_axis[global_trough_idx], wf[peak_idx] * 0.15),
                xytext=(t_axis[peak_idx], wf[peak_idx] * 0.15),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax.text((t_axis[peak_idx] + t_axis[global_trough_idx]) / 2, wf[peak_idx] * 0.30,
            f"width = {width} samples", ha="center", fontsize=8, color="#555")

    n_troughs_found = 1 + len(secondary)
    flag = "  <-- MULTI-TROUGH" if n_troughs_found > 1 else ""
    ax.set_title(f"unit {uid}\nwidth={width} samp, best f0={BEST_F0[uid]}Hz{flag}", fontsize=9.5)
    ax.set_xlabel("time (ms)")
    if col == 0:
        ax.set_ylabel("template amplitude (a.u.)")
    ax.legend(fontsize=6.3, loc="upper right")

plt.suptitle("How 'peak-to-trough width' was actually measured on each real unit -- and where it breaks",
             fontsize=13)
plt.tight_layout()
out1 = os.path.join(OUT, "wavelet_10_peak_trough_method.png")
plt.savefig(out1, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out1)

print("\nunit  peak_sample  trough_sample  width")
for uid, p, tr, w in widths_report:
    print(f"{uid:4d}  {p:11d}  {tr:13d}  {w:5d}")


# ============================================================
# FIGURE 2: the actual pipeline, drawn explicitly, to correct
# "is frequency fitted as a function of width" -- no, it's a
# scan first, width-based prediction checked only afterward.
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")

steps = [
    (0.02, 0.85, "STEP 1\nTake one real unit's own\nKilosort template\n(its actual measured shape)"),
    (0.27, 0.85, "STEP 2\nTRY every candidate frequency\nin a wide range (300-4000 Hz)\nagainst THAT SAME template\n-- a brute-force SCAN,\nnot a calculation"),
    (0.55, 0.85, "STEP 3\nKeep whichever frequency\ngave the SINGLE HIGHEST\nmagnitude response\n= that unit's 'best f0'\n(found empirically, per unit)"),
    (0.80, 0.85, "STEP 4\nOnly AFTER doing this for\n5 different real units:\ncheck whether 'best f0' can\nbe PREDICTED from width alone\n(found: not tightly)"),
]
for x, y, text in steps:
    ax.add_patch(plt.Rectangle((x, y - 0.22), 0.20, 0.30, fill=True, fc="#eef2f7",
                                ec="#555", transform=ax.transAxes))
    ax.text(x + 0.10, y - 0.07, text, ha="center", va="center", fontsize=8.3, transform=ax.transAxes)
for x0 in [0.22, 0.47, 0.75]:
    ax.annotate("", xy=(x0 + 0.05, 0.78), xytext=(x0, 0.78),
                xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))

ax.text(0.5, 0.35,
    "What this means for your question:\n\n"
    "\"Is the probe's frequency fitted to the spike as a function of peak-to-trough width?\"\n"
    "-- NO. Frequency was found by brute-force SCAN per unit (steps 1-3), independently\n"
    "for each of the 5 units, using nothing but that unit's own real template.\n\n"
    "Width was only brought in AFTERWARD (step 4), as a candidate EXPLANATION to test\n"
    "against those 5 already-found results -- to see whether a cheap width measurement\n"
    "could stand in for repeating the expensive scan on every future unit.\n"
    "It did not explain them tightly (width*best_f0 was not constant), so right now\n"
    "there is no trustworthy shortcut -- the scan (steps 1-3) is still the only\n"
    "validated way to get a real unit's own best frequency.",
    ha="center", va="center", fontsize=9.5, transform=ax.transAxes,
    bbox=dict(boxstyle="round", fc="white", ec="#888"))

plt.suptitle("The actual pipeline: scan first, width-based shortcut only tested afterward", fontsize=13)
out2 = os.path.join(OUT, "wavelet_11_pipeline_clarified.png")
plt.savefig(out2, dpi=115, bbox_inches="tight")
plt.close(fig)
print("saved", out2)
