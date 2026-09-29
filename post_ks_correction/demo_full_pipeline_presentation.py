"""
Presentation-ready, single-figure walkthrough of the ENTIRE pillar-1b
process for ONE real unit (342, session 20260901_085606), start to finish:
  1) the unit's own real template, and what "width" means
  2) how frequency (f0) is chosen for THIS unit
  3) what n_cycles / the probe shape actually is, at that frequency
  4) a real raw spike snippet pulled from probe1.dat
  5) STAGE 1 -- coarse whole-sample alignment (matched-filter search)
  6) STAGE 2 -- fine phase correction (the complex-arrow calculation)
  7) the final corrected result, real number in real time units
  8) a plain-language glossary panel

Real data used: templates.npy, cluster_info.tsv, spike_times.npy /
spike_clusters.npy (all small, already-loaded metadata files) plus ONE
small, targeted raw snippet read from probe1.dat around a single real
spike time (light I/O, same pattern used throughout this session).

Usage: python demo_full_pipeline_presentation.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import (make_morlet, wavelet_transform_at, calibrate_reference_phase,
                               sub_sample_shift_from_phase, coarse_then_fine_shift)

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
UID = 342
ITEMSIZE = 2

# ---- load real metadata (small files) ----
templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]
peak_idx = int(np.argmax(template))
trough_idx = int(np.argmin(template))
width = abs(trough_idx - peak_idx)
n_spikes = int(info.loc[UID, "n_spikes"])

# ---- STEP: frequency scan for THIS unit ----
f0_scan = np.linspace(300, 4000, 60)
pad_scan = int(np.ceil(3.0 * FS / (2 * f0_scan.min()))) + 20
scan_trace = np.zeros(61 + 2 * pad_scan)
scan_center = pad_scan + NT0MIN
scan_trace[scan_center - NT0MIN: scan_center - NT0MIN + 61] = template
mags = np.array([abs(wavelet_transform_at(scan_trace, make_morlet(f0, FS, 3.0)[1], scan_center))
                  for f0 in f0_scan])
best_f0 = float(f0_scan[np.nanargmax(mags)])
rule_f0 = 2955 * width ** (-0.506)

# ---- STEP: build the chosen probe, calibrate ----
N_CYCLES = 3.0
_, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
half_cycle_ms = (FS / best_f0) / 2.0 / FS * 1000

# ---- STEP: pull ONE real raw spike snippet ----
st = np.sort(spike_times[spike_clusters == UID])
example_spike_sample = int(st[0])  # Kilosort's own reported time for this real spike

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
SEARCH_RADIUS = 25
buf = len(psi) // 2 + SEARCH_RADIUS + 30
filt_buf = 60
s0 = example_spike_sample - NT0MIN - buf - filt_buf
s1 = example_spike_sample + (61 - NT0MIN) + buf + filt_buf
n_read = s1 - s0
with open(DAT_PATH, "rb") as f:
    f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
raw_trace_full = block[:, peak_ch].astype(np.float64)
filtered_full = filtfilt(b_hp, a_hp, raw_trace_full) * GAIN_TO_UV
# crop off the filter buffer, keep the coarse-search-ready window
trace = filtered_full[filt_buf: -filt_buf]
nominal_center = buf + NT0MIN  # index of Kilosort's own reported spike TIME within
# `trace` -- BUG CAUGHT running this the first time: this was written as just
# `buf`, missing the + NT0MIN. s0 was built so the reported spike sample maps
# to index (NT0MIN + buf + filt_buf) in raw_trace_full, i.e. (NT0MIN + buf)
# after the filt_buf crop -- forgetting the NT0MIN term shifted the assumed
# "time zero" by exactly 20 samples, which is exactly why the first run's
# coarse-search result was a suspicious, telltale 20-sample correction on a
# spike Kilosort itself already reported -- not a real finding about the data.

# ---- STEP 1: coarse search (unpack manually here to show every score) ----
template_norm = template / (np.linalg.norm(template) + 1e-12)
candidate_shifts = np.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
coarse_scores = []
for cshift in candidate_shifts:
    lo = nominal_center - NT0MIN + cshift
    window = trace[lo:lo + 61]
    coarse_scores.append(np.dot(window, template_norm))
coarse_scores = np.array(coarse_scores)
best_coarse_shift = int(candidate_shifts[np.argmax(coarse_scores)])

# ---- STEP 2: fine phase correction at the coarse-corrected center ----
fine_center = nominal_center + best_coarse_shift
w_fine = wavelet_transform_at(trace, psi, fine_center)
fine_shift = sub_sample_shift_from_phase(np.angle(w_fine), best_f0, FS, reference_phase=ref_phase)
total_shift = best_coarse_shift + fine_shift

# also get the full result via the actual production function, as a
# cross-check that the manual unpacking above matches it exactly
result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
                                 reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
assert abs(result["total_shift"] - total_shift) < 1e-9

print(f"unit {UID}: width={width} samples, best_f0={best_f0:.0f}Hz (rule predicted {rule_f0:.0f}Hz)")
print(f"example real spike, KS reported sample {example_spike_sample}")
print(f"coarse shift = {best_coarse_shift} samples, fine shift = {fine_shift:.3f} samples, "
      f"total = {total_shift:.3f} samples ({total_shift/FS*1000:.4f} ms)")

# ============================================================
# THE FIGURE
# ============================================================
plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.45, wspace=0.32)

t61_ms = (np.arange(61) - NT0MIN) / FS * 1000

# --- Panel 1: the real template + width definition ---
ax = fig.add_subplot(gs[0, 0])
ax.plot(t61_ms, template, color="#333", lw=2.2)
ax.axhline(0, color="grey", lw=0.5)
ax.plot(t61_ms[peak_idx], template[peak_idx], marker="^", color="#d62728", ms=14, zorder=5)
ax.plot(t61_ms[trough_idx], template[trough_idx], marker="v", color="#1f77b4", ms=14, zorder=5)
ax.annotate("", xy=(t61_ms[trough_idx], template[peak_idx] * 0.2),
            xytext=(t61_ms[peak_idx], template[peak_idx] * 0.2),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color="#555"))
ax.text((t61_ms[peak_idx] + t61_ms[trough_idx]) / 2, template[peak_idx] * 0.35,
        f"WIDTH = {width} samples\n({width/FS*1000:.3f} ms)", ha="center", fontsize=10.5, color="#555")
ax.set_title(f"STEP 1: unit {UID}'s own real Kilosort\ntemplate (ch{peak_ch}, {n_spikes} real spikes)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("time (ms)")
ax.set_ylabel("template amplitude (a.u.)")

# --- Panel 2: frequency selection ---
ax = fig.add_subplot(gs[0, 1])
ax.plot(f0_scan, mags / mags.max() * 100, color="#1f77b4", lw=2)
ax.axvline(best_f0, color="#d62728", lw=2, label=f"chosen f0 = {best_f0:.0f} Hz\n(scan maximum)")
ax.axvline(rule_f0, color="#2ca02c", lw=1.5, ls="--",
           label=f"width-based guess = {rule_f0:.0f} Hz\n(from 93-unit population rule)")
ax.set_xlabel("candidate probe frequency (Hz)")
ax.set_ylabel("|W| (% of this unit's own best)")
ax.set_title("STEP 2: choose f0 FOR THIS UNIT\n(scan many, keep the strongest match)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5)

# --- Panel 3: the chosen probe (n_cycles definition) ---
ax = fig.add_subplot(gs[0, 2])
t_psi_ms = (np.arange(len(psi)) - len(psi) // 2) / FS * 1000
ax.plot(t_psi_ms, psi.real, color="#1f77b4", lw=1.6, label="probe (real part)")
ax.plot(t_psi_ms, np.abs(psi), color="#888", lw=1, ls="--", label="fading envelope")
ax.set_title(f"STEP 3: build the probe at f0={best_f0:.0f}Hz\n"
             f"n_cycles={N_CYCLES:.0f} -> {len(psi)} samples long\n"
             f"(fixed default -- not yet tuned per unit)", fontsize=12, fontweight="bold")
ax.set_xlabel("time (ms)")
ax.legend(fontsize=8.5)

# --- Panel 4: real raw snippet around the real spike ---
ax = fig.add_subplot(gs[0, 3])
show_window = 40
xs_ms = (np.arange(-show_window, show_window + 1)) / FS * 1000
ax.plot(xs_ms, trace[nominal_center - show_window: nominal_center + show_window + 1],
        color="#333", lw=1.4)
ax.axvline(0, color="#888", lw=1, ls=":")
ax.plot(0, trace[nominal_center], marker="o", mfc="none", mec="#888", ms=11, mew=1.8,
        label="Kilosort's reported time\n(the STARTING guess)")
ax.set_title(f"STEP 4: a REAL raw spike from this unit\n"
             f"(sample {example_spike_sample}, >300Hz filtered)", fontsize=12, fontweight="bold")
ax.set_xlabel("time relative to Kilosort's guess (ms)")
ax.set_ylabel("uV")
ax.legend(fontsize=8)

# --- Panel 5: STAGE 1, coarse search ---
ax = fig.add_subplot(gs[1, 0])
ax.plot(candidate_shifts, coarse_scores, color="#1f77b4", lw=1.6)
ax.axvline(best_coarse_shift, color="#d62728", lw=2,
           label=f"winner: shift = {best_coarse_shift} samples")
ax.plot(best_coarse_shift, coarse_scores.max(), marker="*", color="#d62728", ms=18, zorder=5)
ax.set_xlabel("candidate WHOLE-SAMPLE shift tried")
ax.set_ylabel("match score\n(dot product with template)")
ax.set_title("STAGE 1: COARSE search\n(try every whole-sample shift,\nkeep the best match)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

# --- Panel 6: STAGE 2, fine phase (complex arrow) ---
ax = fig.add_subplot(gs[1, 1])
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
lim = abs(w_fine) * 1.3
ax.plot([0, w_fine.real], [0, w_fine.imag], color="#333", lw=2, marker="o", markevery=[1], ms=9)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.annotate(f"measured angle = {np.degrees(np.angle(w_fine)):.1f} deg\n"
            f"minus reference = {np.degrees(ref_phase):.1f} deg\n"
            f"-> fine correction = {fine_shift:.3f} samples",
            (w_fine.real, w_fine.imag), textcoords="offset points", xytext=(10, 10), fontsize=9)
ax.set_title("STAGE 2: FINE phase correction\n(at the coarse-corrected center,\n"
             "read the arrow's angle)", fontsize=12, fontweight="bold")
ax.set_xlabel("real part of W")
ax.set_ylabel("imag part of W")

# --- Panel 7: final result ---
ax = fig.add_subplot(gs[1, 2])
ax.plot(xs_ms, trace[nominal_center - show_window: nominal_center + show_window + 1],
        color="#333", lw=1.4)
ax.axvline(0, color="#888", lw=1, ls=":", label="Kilosort's original guess")
final_ms = total_shift / FS * 1000
ax.axvline(final_ms, color="#d62728", lw=1.8, label=f"FINAL corrected time\n({final_ms:+.4f} ms)")
ax.set_xlabel("time relative to Kilosort's guess (ms)")
ax.set_ylabel("uV")
ax.set_title(f"STEP 7: final answer\ncoarse ({best_coarse_shift} samp) + fine ({fine_shift:.2f} samp)\n"
             f"= {total_shift:.2f} samples = {final_ms:.4f} ms", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)

# --- Panel 8: glossary ---
ax = fig.add_subplot(gs[1, 3])
ax.axis("off")
ax.text(0, 1.0,
    "GLOSSARY\n\n"
    "PROBE (= \"wavelet\"): the short artificial\n"
    "test-oscillation compared against the\n"
    "real trace. NOT a copy of the spike.\n\n"
    "f0 (FREQUENCY): how fast the probe\n"
    "itself wiggles, in Hz. Chosen PER UNIT\n"
    "by scanning (panel 2) -- NOT fixed.\n\n"
    "n_cycles: how many full wiggles fit in\n"
    "the probe before it fades out. Currently\n"
    "fixed at 3 for every unit (open item).\n\n"
    "WIDTH: distance (in samples) between a\n"
    "unit's own peak and trough sample in its\n"
    "real template (panel 1). Used only as a\n"
    "rough starting GUESS for f0, not the\n"
    "final answer (green dashed vs red line,\n"
    "panel 2 -- they don't always agree).\n\n"
    "COARSE stage: whole-sample search, no\n"
    "wraparound risk, sets the big correction.\n\n"
    "FINE stage: phase-based, sub-sample\n"
    "precision, only trustworthy for the small\n"
    "leftover AFTER the coarse stage.",
    fontsize=10, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle(f"End-to-end pillar-1b timing correction, real example: unit {UID}, session 20260901_085606",
             fontsize=16, fontweight="bold", y=1.01)
out_path = os.path.join(OUT, "wavelet_15_full_pipeline_presentation.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print("saved", out_path)
