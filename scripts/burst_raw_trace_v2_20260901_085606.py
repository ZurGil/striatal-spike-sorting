"""
v2: more example bursts, RANDOMLY sampled (no cherry-picking, no bias
toward decay or length), and properly high-pass filtered (probe1.dat is
unfiltered broadband per params.py hp_filtered=False -- Phy applies its
own highpass before display, so we replicate that instead of showing raw
broadband). Units 30, 22, 63, 135 only.

Usage: python burst_raw_trace_v2_20260901_085606.py
"""
import os
import numpy as np
import pandas as pd
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
ISI_THRESH_MS = 10.0
MIN_BURST_LEN = 3
PAD_MS = 2.5
PAD_AFTER_MS = 10.0  # extra time shown after the last spike in the burst
FILT_BUFFER_MS = 60.0  # extra context each side for filter settling, then cropped
N_EXAMPLE_BURSTS = 8
ITEMSIZE = 2
RNG_SEED = 999

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

UNITS = [30, 22, 63, 135]

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def find_bursts(times_samples, isi_thresh_ms=ISI_THRESH_MS, min_len=MIN_BURST_LEN):
    times_samples = np.sort(times_samples)
    isi_ms = np.diff(times_samples) / FS * 1000
    breaks = np.where(isi_ms > isi_thresh_ms)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(times_samples)]))
    bursts = []
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            bursts.append(times_samples[s:e])
    return bursts


def get_filtered_segment(chan, s0, s1, f, buf):
    n = (s1 + buf) - (s0 - buf)
    f.seek(int(s0 - buf) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read(n * N_CHAN_BIN * ITEMSIZE)
    if len(raw) != n * N_CHAN_BIN * ITEMSIZE:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(n, N_CHAN_BIN)
    trace_raw = block[:, chan].astype(np.float64)
    trace_filt = filtfilt(b_hp, a_hp, trace_raw)
    return trace_filt[buf:-buf] * GAIN_TO_UV


PAD = int(PAD_MS * FS / 1000)
PAD_AFTER = int(PAD_AFTER_MS * FS / 1000)
BUF = int(FILT_BUFFER_MS * FS / 1000)
rng = np.random.default_rng(RNG_SEED)

fig, axes = plt.subplots(len(UNITS), N_EXAMPLE_BURSTS, figsize=(4.0 * N_EXAMPLE_BURSTS, 3.8 * len(UNITS)))

with open(DAT_PATH, "rb") as f:
    for row, uid in enumerate(UNITS):
        chan = int(info.loc[uid, "ch"])
        st = spike_times_samples[spike_clusters == uid]
        bursts = find_bursts(st)
        print(f"unit {uid} (ch{chan}): {len(bursts)} total bursts")

        n_pick = min(N_EXAMPLE_BURSTS, len(bursts))
        chosen_idx = rng.choice(len(bursts), size=n_pick, replace=False)
        chosen = [bursts[i] for i in chosen_idx]

        for col in range(N_EXAMPLE_BURSTS):
            ax = axes[row, col]
            if col >= n_pick:
                ax.axis("off")
                continue
            bidx = chosen_idx[col]
            burst = chosen[col]
            s0, s1 = int(burst[0]) - PAD, int(burst[-1]) + PAD_AFTER
            trace = get_filtered_segment(chan, s0, s1, f, BUF)
            t_ms = (np.arange(s0, s1) - burst[0]) / FS * 1000
            ax.plot(t_ms, trace, color="#2b2b2b", lw=0.9)

            amps = []
            for k, s in enumerate(burst):
                idx = int(s) - s0
                ax.plot(t_ms[idx], trace[idx], marker="v", color="#d62728", ms=7, zorder=5)
                ax.annotate(f"#{k+1}", (t_ms[idx], trace[idx]), textcoords="offset points",
                            xytext=(0, 8), fontsize=7, color="#d62728", ha="center")
                lo, hi = max(0, idx - 20), min(len(trace), idx + 20)
                amps.append(trace[lo:hi].max() - trace[lo:hi].min())

            isis = np.diff(burst) / FS * 1000
            amp_str = " -> ".join(f"{a:.0f}" for a in amps)
            ax.set_title(f"unit {uid}, burst idx {bidx} (n={len(burst)})\n"
                         f"ISIs: " + ", ".join(f"{x:.1f}" for x in isis) + " ms\n"
                         f"p2p (uV): {amp_str}", fontsize=7.5)
            ax.set_xlabel("ms", fontsize=8)
            if col == 0:
                ax.set_ylabel("uV (>300Hz)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="grey", lw=0.4)

plt.suptitle("20260901_085606 -- RANDOMLY sampled bursts, >300Hz high-pass filtered raw trace, detected spikes marked (red), extended post-burst window",
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT, "burst_raw_trace_v3_20260901_085606.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("saved", out_path)
