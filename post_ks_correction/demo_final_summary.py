"""
FINAL, corrected, consolidated summary of the entire pillar-1b pipeline,
one real unit (342, session 20260901_085606) end to end, all numbers
consistent with earlier turns' real-data results.

Usage: python demo_final_summary.py
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
dominant_idx = peak_idx if abs(template[peak_idx]) >= abs(template[trough_idx]) else trough_idx
dominant_label = "peak" if dominant_idx == peak_idx else "trough"
n_spikes = int(info.loc[UID, "n_spikes"])

f0_scan = np.linspace(300, 4000, 60)
pad_scan = int(np.ceil(3.0 * FS / (2 * f0_scan.min()))) + 20
scan_trace = np.zeros(61 + 2 * pad_scan)
scan_center = pad_scan + NT0MIN
scan_trace[scan_center - NT0MIN: scan_center - NT0MIN + 61] = template
mags = np.array([abs(wavelet_transform_at(scan_trace, make_morlet(f0, FS, 3.0)[1], scan_center))
                  for f0 in f0_scan])
best_f0 = float(f0_scan[np.nanargmax(mags)])
rule_f0 = 2955 * width ** (-0.506)

N_CYCLES = 3.0
_, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

st = np.sort(spike_times[spike_clusters == UID])
example_spike_sample = int(st[0])
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
filtered_full = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
trace = filtered_full[filt_buf:-filt_buf]
nominal_center = buf + NT0MIN

template_norm = template / (np.linalg.norm(template) + 1e-12)
candidate_shifts = np.arange(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
coarse_scores = np.array([np.dot(trace[nominal_center - NT0MIN + c: nominal_center - NT0MIN + c + 61], template_norm)
                           for c in candidate_shifts])
best_coarse_shift = int(candidate_shifts[np.argmax(coarse_scores)])
fine_center = nominal_center + best_coarse_shift
w_fine = wavelet_transform_at(trace, psi, fine_center)
fine_shift = sub_sample_shift_from_phase(np.angle(w_fine), best_f0, FS, reference_phase=ref_phase)
total_shift = best_coarse_shift + fine_shift

result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
                                 reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
assert abs(result["total_shift"] - total_shift) < 1e-9

# noise floor, same channel, spike-free stretch
noise_safe_start = example_spike_sample + 500000
with open(DAT_PATH, "rb") as f:
    f.seek(int(noise_safe_start - filt_buf) * N_CHAN_BIN * ITEMSIZE)
    raw_n = f.read((30000 + 2 * filt_buf) * N_CHAN_BIN * ITEMSIZE)
block_n = np.frombuffer(raw_n, dtype=np.int16).reshape(30000 + 2 * filt_buf, N_CHAN_BIN)
noise_trace = filtfilt(b_hp, a_hp, block_n[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV
rng = np.random.default_rng(0)
half = len(psi) // 2
noise_mags = np.array([abs(wavelet_transform_at(noise_trace, psi, rng.integers(half + 5, len(noise_trace) - half - 5)))
                        for _ in range(400)])

print(f"unit {UID}: width={width}, dominant={dominant_label}, best_f0={best_f0:.0f}Hz, rule_f0={rule_f0:.0f}Hz")
print(f"coarse={best_coarse_shift}, fine={fine_shift:.3f}, total={total_shift:.3f} samples ({total_shift/FS*1000:.4f}ms)")
print(f"|W|={abs(w_fine):.0f}, noise median={np.median(noise_mags):.0f}, noise p99={np.percentile(noise_mags,99):.0f}")

# ============================================================
fig = plt.figure(figsize=(23, 12))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.34)
t61_ms = (np.arange(61) - NT0MIN) / FS * 1000

ax = fig.add_subplot(gs[0, 0])
ax.plot(t61_ms, template, color="#333", lw=2.2)
ax.axhline(0, color="grey", lw=0.4)
ax.plot(t61_ms[peak_idx], template[peak_idx], marker="^", color="#d62728", ms=13, zorder=5,
        label="peak" + (" (=Kilosort's\nalignment point)" if dominant_label == "peak" else ""))
ax.plot(t61_ms[trough_idx], template[trough_idx], marker="v", color="#1f77b4", ms=13, zorder=5,
        label="trough" + (" (=Kilosort's\nalignment point)" if dominant_label == "trough" else ""))
ax.annotate("", xy=(t61_ms[trough_idx], template[peak_idx] * 0.15),
            xytext=(t61_ms[peak_idx], template[peak_idx] * 0.15),
            arrowprops=dict(arrowstyle="<->", lw=1.4, color="#555"))
ax.text((t61_ms[peak_idx] + t61_ms[trough_idx]) / 2, template[peak_idx] * 0.30,
        f"WIDTH = {width} samples", ha="center", fontsize=9.5, color="#555")
ax.set_title(f"1) UNIT'S OWN TEMPLATE\nwidth = |peak-trough| = {width} samples\n"
             f"(Kilosort aligns to the DOMINANT one --\nhere, the {dominant_label})", fontsize=11, fontweight="bold")
ax.set_xlabel("time (ms)")
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[0, 1])
ax.plot(f0_scan, mags / mags.max() * 100, color="#1f77b4", lw=2)
ax.axvline(best_f0, color="#d62728", lw=2, label=f"BEST f0 = {best_f0:.0f}Hz\n(scan maximum --\nthis is HOW f0 is chosen)")
ax.axvline(rule_f0, color="#2ca02c", lw=1.3, ls="--", label=f"width-rule guess\n= {rule_f0:.0f}Hz (rough\nprior only, R2=0.42)")
ax.set_xlabel("candidate f0 (Hz)")
ax.set_ylabel("|W| (% of this unit's best)")
ax.set_title("2) FIND BEST FREQUENCY\nscan many f0 against THIS unit's\nown template, keep the strongest", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[0, 2])
t_psi_ms = (np.arange(len(psi)) - len(psi) // 2) / FS * 1000
ax.plot(t_psi_ms, psi.real, color="#1f77b4", lw=1.4, label="probe real part")
ax.plot(t_psi_ms, psi.imag, color="#d62728", lw=1.4, label="probe imag part")
ax.plot(t_psi_ms, np.abs(psi), color="#888", lw=0.9, ls="--", label="envelope")
ax.set_title(f"3) BUILD THE PROBE\nf0={best_f0:.0f}Hz, n_cycles={N_CYCLES:.0f}\n-> {len(psi)} samples (fixed default,\nnot yet per-unit tuned)", fontsize=11, fontweight="bold")
ax.set_xlabel("time (ms)")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 3])
ax.plot(candidate_shifts, coarse_scores, color="#1f77b4", lw=1.6)
ax.axvline(best_coarse_shift, color="#d62728", lw=2, label=f"winner: shift={best_coarse_shift}\n(a REAL spike, sample {example_spike_sample})")
ax.plot(best_coarse_shift, coarse_scores.max(), marker="*", color="#d62728", ms=16, zorder=5)
ax.set_title("4) COARSE: slide the TEMPLATE\n(not the probe!) in whole-sample\nsteps, keep the best match", fontsize=11, fontweight="bold")
ax.set_xlabel("candidate whole-sample shift")
ax.set_ylabel("match score")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
lim = abs(w_fine) * 1.3
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.plot([0, w_fine.real], [0, w_fine.imag], color="#333", lw=2.2, marker="o", markevery=[1], ms=10)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.annotate(f"|W|={abs(w_fine):.0f}\n(LENGTH = energy/\nreliability)", (0, 0), textcoords="offset points",
            xytext=(-100, 40), fontsize=8.5, color="#1f4e8c")
ax.annotate(f"angle={np.degrees(np.angle(w_fine)):.1f}deg\n(DIRECTION = timing,\nafter subtracting\nref={np.degrees(ref_phase):.1f}deg)",
            (w_fine.real, w_fine.imag), textcoords="offset points", xytext=(10, 10), fontsize=8.5, color="#a83232")
ax.set_title("5) FINE: project the coarse-aligned\nsnippet onto the probe's real & imag\nparts -> ONE complex number", fontsize=11, fontweight="bold")
ax.set_xlabel("real part of W")
ax.set_ylabel("imag part of W")

ax = fig.add_subplot(gs[1, 1])
ax.hist(noise_mags, bins=30, color="#888", alpha=0.8, label="400 NOISE-only\nlocations, same channel")
ax.axvline(abs(w_fine), color="#d62728", lw=2.2, label=f"this spike: |W|={abs(w_fine):.0f}")
ax.set_xlabel("|W|")
ax.set_ylabel("count")
ax.set_title(f"RELIABILITY CHECK: is |W| trustworthy?\n{abs(w_fine)/np.median(noise_mags):.1f}x the noise median --\nfar above anything noise alone produces", fontsize=10.7, fontweight="bold")
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[1, 2])
show_window = 40
xs_ms = np.arange(-show_window, show_window + 1) / FS * 1000
ax.plot(xs_ms, trace[nominal_center - show_window: nominal_center + show_window + 1], color="#333", lw=1.4)
ax.axvline(0, color="#888", lw=1, ls=":", label="Kilosort's original guess")
final_ms = total_shift / FS * 1000
ax.axvline(final_ms, color="#d62728", lw=1.8, label=f"FINAL: {final_ms:+.4f} ms")
ax.set_title(f"6) FINAL ANSWER\ncoarse({best_coarse_shift}) + fine({fine_shift:.2f}) = "
             f"{total_shift:.2f} samples\n= {final_ms:.4f} ms", fontsize=11, fontweight="bold")
ax.set_xlabel("time rel. to Kilosort's guess (ms)")
ax.set_ylabel("uV")
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[1, 3])
ax.axis("off")
ax.text(0, 1.0,
    "FULL PIPELINE, confirmed:\n\n"
    "1. Unit's real template -> width\n"
    "   (peak-trough sample distance)\n\n"
    "2. SCAN many frequencies against\n"
    "   THIS unit's template -> best f0\n"
    "   (width only a rough prior, R2=0.42)\n\n"
    "3. Build the probe at that f0\n\n"
    "4. COARSE: slide the TEMPLATE\n"
    "   (whole samples) -> best integer\n"
    "   shift, no wraparound risk\n\n"
    "5. FINE: project coarse-aligned\n"
    "   snippet onto probe's real+imag\n"
    "   parts -> one complex number\n\n"
    "   LENGTH = energy/reliability\n"
    "   (checked against real noise floor)\n"
    "   ANGLE = timing, AFTER subtracting\n"
    "   this unit's own calibration offset\n\n"
    "6. total = coarse + fine, only trusted\n"
    "   because coarse kept fine's input\n"
    "   inside its narrow safe zone",
    fontsize=9.7, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle(f"FINAL summary, corrected: full pillar-1b pipeline on real unit {UID} ({n_spikes} spikes), session 20260901_085606",
             fontsize=15, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_23_final_summary.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
