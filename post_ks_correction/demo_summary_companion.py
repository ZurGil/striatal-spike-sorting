"""
Companion to the final summary figure: restores the "why summing works"
panels (individual phasors + full random-walk path) that got compressed
out of the main pipeline figure, using the SAME real example spike
(unit 342, sample 15169) for full narrative consistency, plus an HONEST
check of what "noise" actually means in the reliability comparison.

Usage: python demo_summary_companion.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, calibrate_reference_phase

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
template = templ_all[:, peak_ch]

_, psi = make_morlet(F0_HZ, FS, n_cycles=3.0)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

st = np.sort(spike_times[spike_clusters == UID])
example_spike_sample = int(st[0])  # SAME spike as the final summary figure

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

# --- HONEST noise definition: check what's actually there ---
# "noise" locations were picked far in time from THIS spike, but were NEVER
# checked against the full spike_times/spike_clusters arrays -- they could
# easily contain real spikes from unit 342 itself (elsewhere in its own
# spike train) or from any of the other 12 units sharing this channel
# neighborhood (339-351, established many turns ago -- this is the same
# densely-packed region as the 342/347 collision investigation). Check now:
noise_center_sample = example_spike_sample + 500000
window_check = 30000  # the 1-second stretch actually used
any_spikes_from_342 = np.sum((st > noise_center_sample - window_check // 2) &
                              (st < noise_center_sample + window_check // 2))
all_spike_times_sorted = np.sort(spike_times)
any_spikes_any_unit = np.sum((all_spike_times_sorted > noise_center_sample - window_check // 2) &
                              (all_spike_times_sorted < noise_center_sample + window_check // 2))
print(f"'noise' window actually contains: {any_spikes_from_342} spikes from unit 342 itself, "
      f"{any_spikes_any_unit} spikes from ANY unit (out of {len(all_spike_times_sorted)} total in session)")

noise_trace = get_filtered(noise_center_sample)
half = len(psi) // 2
z_spike = spike_trace[buf - half: buf - half + len(psi)] * np.conj(psi)
z_noise = noise_trace[buf - half: buf - half + len(psi)] * np.conj(psi)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
running_full = np.cumsum(z_spike)
pts = np.concatenate([[0], running_full])
colors_full = plt.cm.viridis(np.linspace(0, 1, len(pts)))
for i in range(len(pts) - 1):
    ax.plot(pts[i:i+2].real, pts[i:i+2].imag, color=colors_full[i], lw=1.6)
ax.plot(0, 0, marker="o", color="black", ms=7, zorder=5)
ax.plot(pts[-1].real, pts[-1].imag, marker="X", color="#d62728", ms=13, zorder=5,
        label=f"final W, |W|={abs(pts[-1]):.0f}")
ax.set_aspect("equal")
ax.set_title(f"SAME spike as the final summary\n(unit {UID}, sample {example_spike_sample}):\n"
             f"the full 123-term walk that PRODUCES\nstep 5's arrow -- purple(early)->yellow(late)", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlabel("real")
ax.set_ylabel("imag")

ax = axes[1]
running_full_n = np.cumsum(z_noise)
pts_n = np.concatenate([[0], running_full_n])
for i in range(len(pts_n) - 1):
    ax.plot(pts_n[i:i+2].real, pts_n[i:i+2].imag, color=colors_full[i], lw=1.6)
ax.plot(0, 0, marker="o", color="black", ms=7, zorder=5)
ax.plot(pts_n[-1].real, pts_n[-1].imag, marker="X", color="#888", ms=13, zorder=5,
        label=f"final W, |W|={abs(pts_n[-1]):.0f}")
same_lim = max(np.abs(pts).max(), np.abs(pts_n).max()) * 1.15
for a in [axes[0], ax]:
    a.set_xlim(-same_lim, same_lim)
    a.set_ylim(-same_lim, same_lim)
ax.set_aspect("equal")
ax.set_title(f"COMPARISON walk: same construction,\na 'quiet' location ~16.7s later.\n"
             f"NOTE what this location actually contains\n(see honest caveat, right panel)", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlabel("real")
ax.set_ylabel("imag")

ax = axes[2]
ax.axis("off")
ax.text(0, 1.0,
    "WHAT \"NOISE\" ACTUALLY MEANS HERE\n"
    "(checked now, not assumed):\n\n"
    f"The comparison window was picked by\n"
    f"time offset alone (500,000 samples =\n"
    f"~16.7s after the example spike) --\n"
    f"it was NEVER checked against the real\n"
    f"spike_times arrays before use.\n\n"
    f"Checking now, that exact 1-second\n"
    f"window actually contains:\n"
    f"  {any_spikes_from_342} spikes from unit 342 itself\n"
    f"  {any_spikes_any_unit} spikes from ANY real unit\n"
    f"  (out of {len(all_spike_times_sorted)} total spikes this session)\n\n"
    "So this is really 'background activity'\n"
    "-- true electrical noise PLUS whatever\n"
    "real spiking happens to be going on --\n"
    "not a guaranteed, verified spike-free\n"
    "control. A rigorous version would\n"
    "explicitly exclude all known spike\n"
    "times (every unit, not just 342)\n"
    "before sampling. Not yet done.",
    fontsize=9.7, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Companion to the final summary: the summing/projection build-up (step 5, detailed) + honest noise-definition check",
             fontsize=13.5, fontweight="bold")
out_path = os.path.join(OUT, "wavelet_24_summary_companion.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
