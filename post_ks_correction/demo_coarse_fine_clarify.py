"""
Clarifies panel 5/6 confusion: coarse stage uses the TEMPLATE (61 samples,
real-valued), never the probe. Fine stage uses the PROBE (a different,
longer, complex-valued array) ONCE, at a single position -- not a scan.
Shows exactly what multiplies what, at matched lengths, no mismatch.

Usage: python demo_coarse_fine_clarify.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase

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
template = templ_all[:, peak_ch]  # length 61 -- THE COARSE STAGE'S OBJECT

best_f0 = 739.0
_, psi = make_morlet(best_f0, FS, n_cycles=3.0)  # length 123 -- THE FINE STAGE'S OBJECT, a DIFFERENT array
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

st = np.sort(spike_times[spike_clusters == UID])
example_spike_sample = int(st[0])

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
buf, filt_buf = 100, 60
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

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.5, wspace=0.3)

# ============================================================
# TOP ROW: COARSE STAGE -- template (61 samples) vs trace window
# (61 samples), REAL numbers only, repeated at each candidate shift
# ============================================================
template_norm = template / (np.linalg.norm(template) + 1e-12)

for col, cshift in enumerate([-8, 0]):
    ax = fig.add_subplot(gs[0, col])
    lo = nominal_center - NT0MIN + cshift
    window = trace[lo:lo + 61]
    t61 = np.arange(61)
    ax.plot(t61, window, color="#333", lw=1.8, label=f"trace window\n(61 samples, shift={cshift})")
    ax2 = ax.twinx()
    ax2.plot(t61, template, color="#d62728", lw=1.5, ls="--", label="template (61 samples,\nunit's own real shape)")
    score = np.dot(window, template_norm)
    tag = "  <- WORSE MATCH" if cshift != 0 else "  <- BEST MATCH (winner)"
    ax.set_title(f"COARSE, candidate shift={cshift}: SAME LENGTH (61 vs 61),\n"
                 f"multiply element-by-element, sum -> ONE real number\n"
                 f"score = {score:.0f}{tag}", fontsize=10.5)
    ax.set_xlabel("sample index (0-60)")
    ax.set_ylabel("trace (uV)", color="#333")
    ax2.set_ylabel("template (a.u.)", color="#d62728")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper right")

ax = fig.add_subplot(gs[0, 2])
candidate_shifts = np.arange(-25, 26)
scores = []
for cshift in candidate_shifts:
    lo = nominal_center - NT0MIN + cshift
    window = trace[lo:lo + 61]
    scores.append(np.dot(window, template_norm))
scores = np.array(scores)
ax.plot(candidate_shifts, scores, color="#1f77b4", lw=1.8, marker=".")
ax.axvline(0, color="#d62728", lw=1.5)
ax.axvline(-8, color="#888", lw=1, ls=":")
ax.annotate("shift=0\n(right panel above)", (0, scores.max()), textcoords="offset points",
            xytext=(5, 5), fontsize=8, color="#d62728")
ax.annotate("shift=-8\n(left panel above)", (-8, scores[candidate_shifts == -8][0]),
            textcoords="offset points", xytext=(-60, 10), fontsize=8, color="#888")
ax.set_title("THIS is what panel 5 plots: ONE score number\n"
              "per candidate shift, repeating the left two panels\n"
              "51 times (once per integer from -25 to +25)", fontsize=10.5)
ax.set_xlabel("candidate whole-sample shift")
ax.set_ylabel("match score")

# ============================================================
# BOTTOM ROW: FINE STAGE -- probe (123 samples) vs a 123-sample
# trace window, COMPLEX numbers, done ONCE (no scanning)
# ============================================================
ax = fig.add_subplot(gs[1, 0])
half = len(psi) // 2
fine_center = nominal_center  # coarse winner was shift=0
lo, hi = fine_center - half, fine_center - half + len(psi)
trace_seg = trace[lo:hi]
t123 = np.arange(len(psi))
ax.plot(t123, trace_seg, color="#333", lw=1.6, label=f"trace window\n({len(psi)} samples)")
ax2 = ax.twinx()
ax2.plot(t123, psi.real, color="#1f77b4", lw=1.3, label="probe, real part\n(123 samples --\nLONGER than the\n61-sample template)")
ax.set_title(f"FINE stage: probe ({len(psi)} samples) needs its OWN\n"
             f"{len(psi)}-sample trace window -- DIFFERENT length\n"
             f"than the coarse stage's 61-sample template.\n"
             f"No mismatch: each stage uses a window matched\n"
             f"to ITS OWN array's length.", fontsize=10)
ax.set_xlabel("sample index")
ax.set_ylabel("trace (uV)")
ax2.set_ylabel("probe (a.u.)", color="#1f77b4")
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lb1 + lb2, fontsize=7.3, loc="upper right")

ax = fig.add_subplot(gs[1, 1])
prod_real = trace_seg * psi.real
prod_imag = trace_seg * (-psi.imag)
ax.plot(t123, prod_real, color="#1f77b4", lw=1.3, label="trace * probe_real")
ax.plot(t123, prod_imag, color="#d62728", lw=1.3, label="trace * probe_conj_imag")
ax.axhline(0, color="grey", lw=0.4)
ax.set_title("Multiply element-by-element (COMPLEX now --\n"
              "this is the ONLY place real+imaginary parts\n"
              "appear anywhere in this whole process)", fontsize=10.5)
ax.set_xlabel("sample index")
ax.set_ylabel("product")
ax.legend(fontsize=7.5)

ax = fig.add_subplot(gs[1, 2])
w_fine = wavelet_transform_at(trace, psi, fine_center)
lim = abs(w_fine) * 1.3
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.plot([0, w_fine.real], [0, w_fine.imag], color="#333", lw=2, marker="o", markevery=[1], ms=9)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title("Sum those products (ONE complex number) ->\n"
              "this happens ONCE, at the coarse winner's\n"
              "position -- NOT scanned/repeated like coarse was",
              fontsize=10.5)
ax.set_xlabel("real part of W")
ax.set_ylabel("imag part of W")

plt.suptitle("What actually multiplies what: COARSE (template, real, scanned 51 times) vs FINE (probe, complex, done ONCE)",
             fontsize=15, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_16_coarse_fine_clarified.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
