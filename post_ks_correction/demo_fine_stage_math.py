"""
One consolidated figure explaining the FINE stage's math completely:
explicit formula, BOTH real and imaginary parts of the probe (only the
real part was shown before), the per-sample complex products, the
running sum (and WHY summing works), and what the final angle and
magnitude each mean physically.

Usage: python demo_fine_stage_math.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, sub_sample_shift_from_phase

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
UID = 342
ITEMSIZE = 2
F0_HZ = 739.0
N_CYCLES = 3.0

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]

t_psi, psi = make_morlet(F0_HZ, FS, n_cycles=N_CYCLES)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
t_psi_ms = t_psi / FS * 1000

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
nominal_center = buf + NT0MIN  # coarse stage found shift=0 for this spike, see previous figure

half = len(psi) // 2
lo, hi = nominal_center - half, nominal_center - half + len(psi)
x_segment = trace[lo:hi]
t_local_ms = (np.arange(lo, hi) - nominal_center) / FS * 1000

# the actual calculation, step by step (matches wavelet_transform_at exactly)
prod_real = x_segment * psi.real
prod_imag = x_segment * (-psi.imag)
cum_real = np.cumsum(prod_real)
cum_imag = np.cumsum(prod_imag)
W = complex(cum_real[-1], cum_imag[-1])
angle_deg = np.degrees(np.angle(W))
ref_deg = np.degrees(ref_phase)
fine_shift_samples = sub_sample_shift_from_phase(np.angle(W), F0_HZ, FS, reference_phase=ref_phase)
fine_shift_ms = fine_shift_samples / FS * 1000

sigma_s = (N_CYCLES / F0_HZ) / 2 / 2.5

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1, 1], hspace=0.75, wspace=0.32)

# ---- top: the explicit formula, as text ----
ax_formula = fig.add_subplot(gs[0, :])
ax_formula.axis("off")
ax_formula.text(0.5, 0.5,
    r"THE PROBE, explicitly:      $\psi(t) = e^{i \cdot 2\pi f_0 t} \cdot w(t)$"
    r"     $= \cos(2\pi f_0 t)\cdot w(t) \; + \; i\cdot\sin(2\pi f_0 t)\cdot w(t)$"
    "\n\n"
    rf"THIS unit's numbers:   $f_0$ = {F0_HZ:.0f} Hz     $w(t) = e^{{-t^2/(2\sigma^2)}}$"
    rf"     $\sigma$ = {sigma_s*1000:.3f} ms   ({N_CYCLES:.0f} cycles fit in the window)"
    "\n\n"
    r"THE CALCULATION:      $W(t_0) = \sum_\tau x(\tau)\cdot\overline{\psi(\tau-t_0)}$"
    r"      (the bar means complex conjugate: flips the sign of the imaginary/sine part only)",
    ha="center", va="center", fontsize=13.5, transform=ax_formula.transAxes,
    bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

# ---- row 2: the probe, BOTH parts (real part alone was shown before -- incomplete) ----
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_psi_ms, psi.real, color="#1f77b4", lw=1.6, label=r"Re[$\psi$] = cos(2$\pi f_0 t$)$\cdot w(t)$")
ax.plot(t_psi_ms, psi.imag, color="#d62728", lw=1.6, label=r"Im[$\psi$] = sin(2$\pi f_0 t$)$\cdot w(t)$")
ax.plot(t_psi_ms, np.abs(psi), color="#888", lw=0.9, ls="--", label="envelope w(t)")
ax.axhline(0, color="grey", lw=0.4)
ax.set_title("THE PROBE -- both parts\n(cosine AND sine, same envelope,\nquarter-cycle offset from each other)", fontsize=11)
ax.set_xlabel("time (ms)")
ax.legend(fontsize=7.5)

# ---- the real trace segment used ----
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_local_ms, x_segment, color="#333", lw=1.6)
ax.axhline(0, color="grey", lw=0.4)
ax.set_title(f"THE TRACE, x(tau)\n(real spike, unit {UID}, {len(psi)} samples,\ncentered on the coarse-stage winner)", fontsize=11)
ax.set_xlabel("time (ms)")
ax.set_ylabel("uV")

# ---- per-sample products ----
ax = fig.add_subplot(gs[1, 2])
ax.plot(t_local_ms, prod_real, color="#1f77b4", lw=1.3,
        label=r"x(tau) $\cdot$ Re[$\psi$] = real part of product")
ax.plot(t_local_ms, prod_imag, color="#d62728", lw=1.3,
        label=r"x(tau) $\cdot$ (-Im[$\psi$]) = imag part of product")
ax.axhline(0, color="grey", lw=0.4)
ax.set_title("Multiply x(tau) by EACH probe sample\nat the SAME tau -- one real product,\none imaginary product, per sample", fontsize=11)
ax.set_xlabel("time (ms)")
ax.legend(fontsize=7.5)

# ---- running sum: WHY summing works ----
ax = fig.add_subplot(gs[2, 0])
ax.plot(t_local_ms, cum_real, color="#1f77b4", lw=1.6, label="running total, real")
ax.plot(t_local_ms, cum_imag, color="#d62728", lw=1.6, label="running total, imag")
ax.axhline(0, color="grey", lw=0.3)
ax.set_title("WHY summing works: near t=0 (the real\nspike, which DOES contain ~739Hz content)\nthe products keep pushing the same\ndirection -- they add up steadily,\nnot randomly. Away from t=0 (noise) they\nwiggle and roughly cancel.", fontsize=10.3)
ax.set_xlabel("time (ms)")
ax.legend(fontsize=7.5)

# ---- final complex number ----
ax = fig.add_subplot(gs[2, 1])
ax.axhline(0, color="grey", lw=0.4)
ax.axvline(0, color="grey", lw=0.4)
ax.plot([0, W.real], [0, W.imag], color="#333", lw=2.2, marker="o", markevery=[1], ms=10)
lim = abs(W) * 1.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.annotate(f"angle = {angle_deg:.1f} deg", (W.real, W.imag), textcoords="offset points",
            xytext=(10, 15), fontsize=10, color="#a83232")
ax.annotate(f"|W| = {abs(W):.0f}", (0, 0), textcoords="offset points",
            xytext=(-90, 30), fontsize=10, color="#1f4e8c")
ax.set_title("The final answer: the two running totals'\nLAST values, plotted as one point", fontsize=11)
ax.set_xlabel("real part of W")
ax.set_ylabel("imag part of W")

# ---- interpretation, in words and numbers ----
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
ax.text(0, 1.0,
    "WHAT THE TWO NUMBERS MEAN\n\n"
    f"ANGLE ({angle_deg:.1f} deg) -- tells you HOW FAR\n"
    f"to move your timing guess. Formula:\n"
    r"  $\delta = -\Delta\phi / \omega$" "\n"
    f"this unit's calibration reference is\n"
    f"{ref_deg:.1f} deg (its own shape's zero-shift\n"
    f"angle), so:\n"
    f"  {angle_deg:.1f} - ({ref_deg:.1f}) = {angle_deg-ref_deg:.1f} deg of REAL timing error\n"
    f"  -> {fine_shift_samples:.3f} samples = {fine_shift_ms:.4f} ms\n"
    f"  THIS is the fine correction applied.\n\n"
    f"MAGNITUDE (|W|={abs(W):.0f}) -- does NOT tell you\n"
    f"where to move. It tells you HOW STRONG/\n"
    f"CONFIDENT this match is -- how much real\n"
    f"{F0_HZ:.0f}Hz signal is actually present here.\n"
    f"A tiny |W| means the angle is measuring\n"
    f"mostly noise, not a real timing cue --\n"
    f"a weak match should NOT be trusted the\n"
    f"same as a strong one.",
    fontsize=10, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle(f"The FINE stage, complete: formula, both probe parts, the calculation, and what angle vs. magnitude each mean\n(real example: unit {UID}, spike sample {example_spike_sample})",
             fontsize=14.5, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_17_fine_stage_math_complete.png")
plt.savefig(out_path, dpi=118, bbox_inches="tight")
print("saved", out_path)
print(f"angle={angle_deg:.2f}deg, ref={ref_deg:.2f}deg, fine_shift={fine_shift_samples:.3f} samples = {fine_shift_ms:.4f} ms")
