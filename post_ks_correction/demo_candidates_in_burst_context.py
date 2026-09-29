"""
Shows each of the 3 promising undetected candidates in the context of
their FULL burst -- the whole high-pass-filtered raw trace spanning the
burst's real detected spikes (red) plus the candidate (green), not just
an isolated +/-2ms snippet around the candidate alone.

Usage: python demo_candidates_in_burst_context.py
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
UID = 342
ISI_THRESH_MS = 10.0
MIN_BURST_LEN = 4

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))

st = np.sort(spike_times[spike_clusters == UID])
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def find_bursts(times_samples, isi_thresh_ms=ISI_THRESH_MS, min_len=MIN_BURST_LEN):
    isi_ms = np.diff(times_samples) / FS * 1000
    breaks = np.where(isi_ms > isi_thresh_ms)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(times_samples)]))
    bursts = []
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            bursts.append(times_samples[s:e])
    return bursts


all_bursts = find_bursts(st)

candidates = [
    (203158239, 7.576, 0.632),
    (312346361, 6.485, 0.650),
    (93780997, 4.660, 0.625),
]

# find which burst each candidate belongs next to: the burst whose spikes
# bracket it, or whose last spike precedes it most closely
def find_containing_burst(sample):
    best_burst, best_dist = None, np.inf
    for burst in all_bursts:
        if burst[0] - 500 <= sample <= burst[-1] + 500:
            d = min(abs(sample - burst[0]), abs(sample - burst[-1]))
            if sample < burst[0] or sample > burst[-1]:
                d = min(abs(sample - burst[0]), abs(sample - burst[-1]))
            else:
                d = 0
            if d < best_dist:
                best_dist, best_burst = d, burst
    return best_burst


def get_filtered(s0, s1, filt_buf=60):
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0 - filt_buf) * N_CHAN_BIN * 2)
        raw = f.read((s1 - s0 + 2 * filt_buf) * N_CHAN_BIN * 2)
    block = np.frombuffer(raw, dtype=np.int16).reshape(s1 - s0 + 2 * filt_buf, N_CHAN_BIN)
    filt = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filt[filt_buf:-filt_buf]


fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for ax, (cand_sample, cand_a, cand_r2) in zip(axes, candidates):
    burst = find_containing_burst(cand_sample)
    pad_before, pad_after = 80, 80
    s0 = min(burst[0], cand_sample) - pad_before
    s1 = max(burst[-1], cand_sample) + pad_after
    trace = get_filtered(s0, s1)
    t_ms = (np.arange(s0, s1)) / FS * 1000
    t0_ms = burst[0] / FS * 1000

    ax.plot((np.arange(s0, s1) - burst[0]) / FS * 1000, trace, color="#333", lw=1.1)
    for k, s in enumerate(burst):
        idx = int(s) - s0
        ax.plot((s - burst[0]) / FS * 1000, trace[idx], marker="v", color="#d62728", ms=11, zorder=5)
        ax.annotate(f"#{k+1}\nDETECTED", ((s - burst[0]) / FS * 1000, trace[idx]),
                    textcoords="offset points", xytext=(0, 10), fontsize=7, color="#d62728", ha="center")
    idx_c = cand_sample - s0
    ax.plot((cand_sample - burst[0]) / FS * 1000, trace[idx_c], marker="^", color="#2ca02c", ms=13, zorder=6)
    ax.annotate(f"CANDIDATE\na={cand_a:.1f}, R2={cand_r2:.2f}", ((cand_sample - burst[0]) / FS * 1000, trace[idx_c]),
                textcoords="offset points", xytext=(0, -22), fontsize=8, color="#2ca02c", ha="center", fontweight="bold")
    ax.axhline(0, color="grey", lw=0.4)
    isi_ms = np.diff(burst) / FS * 1000
    dist_ms = (cand_sample - burst[-1]) / FS * 1000
    ax.set_title(f"burst with {len(burst)} detected spikes (ISIs: " + ", ".join(f"{x:.1f}" for x in isi_ms) +
                 f" ms), candidate {dist_ms:.2f} ms after the last detected spike", fontsize=10.5)
    ax.set_xlabel("time relative to burst start (ms)")
    ax.set_ylabel("uV (>300Hz)")

plt.suptitle("The 3 promising undetected candidates, shown WITH their full burst (real raw, >300Hz high-passed)",
             fontsize=13.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_30_candidates_in_burst.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
