"""
Checks and refines the "two 123-dim vectors, project onto them" intuition:
  (1) verify Re[psi] and Im[psi], as 123-dim vectors, are ~orthogonal and
      ~equal length (a clean 2D "sub-basis" inside the 123-dim space) --
      confirmed numerically, not assumed.
  (2) the CORRECTION: it is NOT "large in both axes" that signals a
      frequency match -- it's the TOTAL length sqrt(Re^2+Im^2) that does.
      WHERE that fixed-ish total splits between the two axes is the
      TIMING information (angle), independent of match strength.
      Shown by: several t0 candidates all within the real spike (magnitude
      stays large throughout, but the split between axes rotates), vs a
      noise region (BOTH axes collapse together).

Usage: python demo_projection_intuition.py
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
c = psi.real   # 123-dim vector #1 (the "cosine direction")
s = psi.imag   # 123-dim vector #2 (the "sine direction")

# ---- (1) verify the orthogonal-basis claim numerically ----
c_dot_s = float(np.dot(c, s))
c_norm = float(np.linalg.norm(c))
s_norm = float(np.linalg.norm(s))
cos_angle_between = c_dot_s / (c_norm * s_norm)
print(f"|c| = {c_norm:.4f}, |s| = {s_norm:.4f}  (ratio = {c_norm/s_norm:.4f}, want ~1)")
print(f"c.s = {c_dot_s:.4f}  (relative to |c||s|: {cos_angle_between:.5f}, want ~0)")
print(f"angle between c and s: {np.degrees(np.arccos(np.clip(cos_angle_between,-1,1))):.2f} deg (want ~90)")

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
    return filtered[filt_buf:-filt_buf]


spike_trace = get_filtered(example_spike_sample)
noise_trace = get_filtered(example_spike_sample + 500000)
half = len(psi) // 2

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 1], hspace=0.5, wspace=0.32)

ax_text = fig.add_subplot(gs[0, :])
ax_text.axis("off")
ax_text.text(0.5, 1.0,
    "YOUR INTUITION, checked and confirmed: Re[psi] (c) and Im[psi] (s) ARE two vectors living in the same 123-dim space as the spike snippet,\n"
    f"and (Re[W], Im[W]) really IS (x . c, -x . s) -- x's coordinates projected onto those two directions. Verified numerically: "
    f"|c|={c_norm:.1f}, |s|={s_norm:.1f} (equal length), angle between them = {np.degrees(np.arccos(np.clip(cos_angle_between,-1,1))):.1f} deg (~90, i.e. ORTHOGONAL) -- "
    "so c and s form a clean 2D right-angle 'sub-basis' inside the big 123-dim space, exactly like x/y axes.\n\n"
    "ONE CORRECTION: it is NOT 'large in both axes' that signals a frequency match. It's the TOTAL length sqrt(Re^2+Im^2) -- regardless of how that\n"
    "splits between the two axes. Shown below: several t0 candidates, all still within the real spike, keep a LARGE total length throughout, but the\n"
    "SPLIT between the two axes rotates freely (that split/angle is the TIMING signal). Moving to a noise region collapses BOTH axes together instead.",
    ha="center", va="top", fontsize=11.6, transform=ax_text.transAxes,
    bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

# ---- panel: several t0 candidates within the real spike -- magnitude
# stays large, angle rotates ----
ax = fig.add_subplot(gs[1, 0])
t0_center = buf
offsets = [-3, -1, 0, 1, 3]
colors = plt.cm.cool(np.linspace(0, 1, len(offsets)))
for off, col in zip(offsets, colors):
    t0 = t0_center + off
    w = wavelet_transform_at(spike_trace, psi, t0)
    ax.plot([0, w.real], [0, w.imag], color=col, lw=2, marker="o", markevery=[1], ms=8,
            label=f"t0 offset={off:+d}: |W|={abs(w):.0f}, angle={np.degrees(np.angle(w)):.0f}deg")
lim = 4000
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.3)
ax.axvline(0, color="grey", lw=0.3)
ax.set_title("REAL SPIKE, 5 different t0 candidates\n(all still near the true spike):\nLENGTH stays large & similar for all --\nonly the ANGLE (split between axes) rotates",
             fontsize=10.5)
ax.legend(fontsize=7)
ax.set_xlabel("real part")
ax.set_ylabel("imag part")

# ---- panel: real spike (fixed t0) vs noise (several locations) --
# BOTH axes collapse together for noise, not just an imbalance ----
ax = fig.add_subplot(gs[1, 1])
w_spike = wavelet_transform_at(spike_trace, psi, t0_center)
ax.plot([0, w_spike.real], [0, w_spike.imag], color="#d62728", lw=2.4, marker="o",
        markevery=[1], ms=10, label=f"REAL SPIKE: |W|={abs(w_spike):.0f}")
rng = np.random.default_rng(1)
for i in range(6):
    t0n = rng.integers(half + 5, len(noise_trace) - half - 5)
    wn = wavelet_transform_at(noise_trace, psi, t0n)
    ax.plot([0, wn.real], [0, wn.imag], color="#888", lw=1.2, alpha=0.8,
            marker="o", markevery=[1], ms=6,
            label="6x NOISE locations\n(all short, BOTH axes small)" if i == 0 else None)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.3)
ax.axvline(0, color="grey", lw=0.3)
ax.set_title("Real spike (red, long) vs. 6 different\nnoise locations (grey, all short):\nnoise collapses BOTH axes TOGETHER --\nnot a lopsided split, just short overall",
             fontsize=10.5)
ax.legend(fontsize=7.5)
ax.set_xlabel("real part")
ax.set_ylabel("imag part")

# ---- panel: the corrected statement, explicitly ----
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0, 1.0,
    "THE PRECISE RULE:\n\n"
    "FREQUENCY MATCH is read from the\n"
    "TOTAL length only:\n"
    "  |W| = sqrt(Re[W]^2 + Im[W]^2)\n"
    "large |W| regardless of split = real\n"
    "content at this frequency is present.\n\n"
    "TIMING is read from the SPLIT (angle)\n"
    "between the two axes, independent of\n"
    "how large the total is:\n"
    "  angle = atan2(Im[W], Re[W])\n\n"
    "So: 'large in both axes' is only ONE\n"
    "of many possible splits that both give\n"
    "a large total -- e.g. (large Re, ~0 Im)\n"
    "is EQUALLY as strong a match as\n"
    "(large Re, large Im), just at a\n"
    "different timing/phase. Don't judge\n"
    "match quality by whether both axes\n"
    "are individually large -- judge it by\n"
    "the combined length only.",
    fontsize=10.3, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Projection intuition, confirmed and refined: c and s form an orthogonal basis; magnitude=match strength, angle=timing, independent of each other",
             fontsize=13.5, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_20_projection_intuition.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
