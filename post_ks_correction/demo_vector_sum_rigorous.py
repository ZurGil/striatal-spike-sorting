"""
Fully rigorous treatment of "why does summing give the right angle/magnitude":
each sample contributes ONE small complex number (a "phasor" in physics/EE
language -- a little rotating arrow). Summing them is literal VECTOR
ADDITION, tip-to-tail, the same math as combining light or radio waves.
NOT a CDF/PDF (that accumulates a scalar probability weight; this
accumulates 2D vectors, which can reinforce OR cancel depending on
direction, not just pile up).

Shows, on the SAME real unit 342, at REAL sample resolution:
  - individual per-sample phasors, drawn tip-to-tail, near the real spike
  - the full random-walk PATH the running sum traces, real-spike region
  - the same, for a noise-only region (no real signal to detect)
  - explicit clarification of what tau (summed over) vs t0 (fixed) mean

Usage: python demo_vector_sum_rigorous.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at

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

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))

_, psi = make_morlet(F0_HZ, FS, n_cycles=3.0)
n_psi = len(psi)
half = n_psi // 2

st = np.sort(spike_times[spike_clusters == UID])
example_spike_sample = int(st[0])

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
buf, filt_buf = 100, 60


def get_filtered(center_sample):
    s0 = center_sample - buf - filt_buf
    s1 = center_sample + buf + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    filtered = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filtered[filt_buf:-filt_buf]  # length 2*buf, center sample at index buf


spike_trace = get_filtered(example_spike_sample)
noise_trace = get_filtered(example_spike_sample + 500000)  # ~16s later, no relation to this spike

t0_spike = buf  # index of the (fixed) evaluation point for the spike case
t0_noise = buf  # same, arbitrary fixed point for the noise case

lo_s, hi_s = t0_spike - half, t0_spike - half + n_psi
x_spike = spike_trace[lo_s:hi_s]
lo_n, hi_n = t0_noise - half, t0_noise - half + n_psi
x_noise = noise_trace[lo_n:hi_n]

# per-sample complex terms: z_tau = x(tau) * conj(psi(tau - t0))
z_spike = x_spike * np.conj(psi)
z_noise = x_noise * np.conj(psi)

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1, 1], hspace=0.65, wspace=0.3)

# ---- top: precise statement of what's fixed vs summed, correcting the CDF framing ----
ax_text = fig.add_subplot(gs[0, :])
ax_text.axis("off")
ax_text.text(0.5, 1.0,
    r"WHAT IS FIXED, WHAT IS SUMMED: $t_0$ is chosen ONCE (the candidate spike time) and held FIXED for one entire calculation."
    r"  $\tau$ (\"tau\") is a DUMMY index that sweeps across every sample INSIDE the probe's window (here, $\tau = t_0-61 \ldots t_0+61$, 123 values)."
    "\n"
    r"For EACH $\tau$, one small complex number is produced: $z_\tau = x(\tau)\cdot\overline{\psi(\tau-t_0)}$ -- physicists/engineers call this a PHASOR: a little rotating arrow with its own length and direction."
    "\n\n"
    r"THIS IS NOT A CDF/PDF. A cumulative distribution accumulates a scalar (non-negative) probability weight -- it can only ever climb or stay flat."
    r"  Summing PHASORS is different: each $z_\tau$ is a 2D VECTOR, so adding them is literal VECTOR ADDITION, tip-to-tail -- exactly the same math used"
    "\n"
    r"to combine light waves or radio signals. Vectors pointing the SAME direction ADD UP (constructive interference, the total grows);"
    r" vectors pointing RANDOM/OPPOSING directions PARTIALLY CANCEL (destructive interference, the total stays small)."
    "\n\n"
    r"$W(t_0)=\sum_\tau z_\tau$ is just: walk tip-to-tail through all 123 little arrows, and see where you end up. THAT end point's distance from the origin is $|W|$,"
    r" and its direction is $\mathrm{angle}(W)$ -- shown concretely below, on REAL data, both for a real spike (arrows agree) and for pure noise (arrows disagree).",
    ha="center", va="top", fontsize=11.6, transform=ax_text.transAxes,
    bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

# ---- row 2: a FEW individual phasors near the peak, tip-to-tail (SPIKE) ----
ax = fig.add_subplot(gs[1, 0])
center_idx = half  # tau=t0 in local coordinates
show_slice = slice(center_idx - 5, center_idx + 6)  # 11 consecutive terms near the peak
terms = z_spike[show_slice]
running = np.cumsum(terms)
start_points = np.concatenate([[0], running[:-1]])
colors = plt.cm.autumn(np.linspace(0, 0.9, len(terms)))
for k, (s, e, c) in enumerate(zip(start_points, running, colors)):
    ax.annotate("", xy=(e.real, e.imag), xytext=(s.real, s.imag),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.plot(0, 0, marker="o", color="black", ms=6)
ax.plot(running[-1].real, running[-1].imag, marker="X", color="#333", ms=12,
        label=f"sum of these {len(terms)} terms")
lim = max(abs(running[-1]), 1) * 1.4
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title(f"REAL SPIKE: 11 individual phasors z_tau,\ntau near the peak, drawn TIP-TO-TAIL\n(color = time order, red early -> yellow late)\nTHEY MOSTLY AGREE -- march the SAME way",
             fontsize=10.5)
ax.legend(fontsize=8)
ax.set_xlabel("real")
ax.set_ylabel("imag")

# ---- same for NOISE ----
ax = fig.add_subplot(gs[1, 1])
terms_n = z_noise[show_slice]
running_n = np.cumsum(terms_n)
start_points_n = np.concatenate([[0], running_n[:-1]])
for k, (s, e, c) in enumerate(zip(start_points_n, running_n, colors)):
    ax.annotate("", xy=(e.real, e.imag), xytext=(s.real, s.imag),
                arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.plot(0, 0, marker="o", color="black", ms=6)
ax.plot(running_n[-1].real, running_n[-1].imag, marker="X", color="#333", ms=12,
        label=f"sum of these {len(terms_n)} terms")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title("NOISE (no real spike here): SAME 11-term\ntip-to-tail construction, same probe.\nTHEY DISAGREE -- zig-zag, barely leave\nthe origin (partial cancellation)",
             fontsize=10.5)
ax.legend(fontsize=8)
ax.set_xlabel("real")
ax.set_ylabel("imag")

# ---- direct side-by-side magnitude comparison ----
ax = fig.add_subplot(gs[1, 2])
labels = ["11 terms\n(near-peak\nregion)", "all 123 terms\n(full window)"]
spike_vals = [abs(running[-1]), abs(np.sum(z_spike))]
noise_vals = [abs(running_n[-1]), abs(np.sum(z_noise))]
x = np.arange(2)
w = 0.32
ax.bar(x - w / 2, spike_vals, w, color="#d62728", label="real spike")
ax.bar(x + w / 2, noise_vals, w, color="#888", label="noise")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("|sum| (final distance from origin)")
ax.set_title("Same construction, two sample counts:\nspike stays large, noise stays small,\nboth with real data", fontsize=10.5)
ax.legend(fontsize=8.5)

# ---- row 3: the FULL random-walk path, all 123 terms, spike vs noise ----
ax = fig.add_subplot(gs[2, 0])
running_full = np.cumsum(z_spike)
pts = np.concatenate([[0], running_full])
colors_full = plt.cm.viridis(np.linspace(0, 1, len(pts)))
for i in range(len(pts) - 1):
    ax.plot(pts[i:i+2].real, pts[i:i+2].imag, color=colors_full[i], lw=1.6)
ax.plot(0, 0, marker="o", color="black", ms=7, zorder=5)
ax.plot(pts[-1].real, pts[-1].imag, marker="X", color="#d62728", ms=13, zorder=5,
        label=f"final W, |W|={abs(pts[-1]):.0f}")
ax.set_aspect("equal")
ax.set_title("FULL walk, all 123 terms, REAL SPIKE:\npurple(early)->yellow(late). Marches\nSTEADILY away from origin -- coherent",
             fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlabel("real")
ax.set_ylabel("imag")

ax = fig.add_subplot(gs[2, 1])
running_full_n = np.cumsum(z_noise)
pts_n = np.concatenate([[0], running_full_n])
for i in range(len(pts_n) - 1):
    ax.plot(pts_n[i:i+2].real, pts_n[i:i+2].imag, color=colors_full[i], lw=1.6)
ax.plot(0, 0, marker="o", color="black", ms=7, zorder=5)
ax.plot(pts_n[-1].real, pts_n[-1].imag, marker="X", color="#888", ms=13, zorder=5,
        label=f"final W, |W|={abs(pts_n[-1]):.0f}")
same_lim = max(np.abs(pts).max(), np.abs(pts_n).max()) * 1.15
for a in [ax, fig.axes[-2]]:
    a.set_xlim(-same_lim, same_lim)
    a.set_ylim(-same_lim, same_lim)
ax.set_aspect("equal")
ax.set_title("FULL walk, all 123 terms, NOISE:\nwanders back and forth, never commits\nto one direction -- stays near origin",
             fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlabel("real")
ax.set_ylabel("imag")

ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
ax.text(0, 1.0,
    "WHY THIS IS THE RIGHT PICTURE\n(not a CDF):\n\n"
    "A CDF accumulates a single number\n"
    "(probability mass) that can only grow\n"
    "or stay flat -- there's no direction to\n"
    "cancel against.\n\n"
    "Here, each z_tau has a DIRECTION as\n"
    "well as a size. Two terms pointing\n"
    "opposite ways don't just 'add less' --\n"
    "they can cancel almost completely,\n"
    "the same way two water waves meeting\n"
    "trough-to-crest flatten each other out.\n\n"
    "A real spike's samples, near a\n"
    "frequency the probe matches, keep\n"
    "producing z_tau that point roughly\n"
    "the SAME way -- so the walk marches\n"
    "off (bottom-left). Pure noise samples\n"
    "point randomly -- so the walk wanders\n"
    "and stays near the origin (bottom-\n"
    "middle). The FINAL position's distance\n"
    "= |W| (confidence), direction = angle\n"
    "(timing).",
    fontsize=10, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Vector (phasor) addition, not a CDF: real spike (coherent walk) vs. real noise (random walk), unit 342",
             fontsize=15, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_19_vector_sum_rigorous.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
print(f"spike: 11-term |sum|={abs(running[-1]):.0f}, full |sum|={abs(np.sum(z_spike)):.0f}")
print(f"noise: 11-term |sum|={abs(running_n[-1]):.0f}, full |sum|={abs(np.sum(z_noise)):.0f}")
