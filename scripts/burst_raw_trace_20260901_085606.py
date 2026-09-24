"""
Show the actual CONTINUOUS raw trace (peak channel) spanning each example
burst -- not overlaid individual snippets -- with detected-spike times
marked, so amplitude decay and any undetected spike-like events can be
judged by eye directly in the raw data. Units 30, 22, 63, 135 only
(unit 38 excluded -- doesn't look like real spikes).

Usage: python burst_raw_trace_20260901_085606.py
"""
import os
import numpy as np
import pandas as pd
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
N_EXAMPLE_BURSTS = 3
ITEMSIZE = 2

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

UNITS = [30, 22, 63, 135]


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


def get_continuous(chan, s0, s1, f):
    n = s1 - s0
    f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read(n * N_CHAN_BIN * ITEMSIZE)
    if len(raw) != n * N_CHAN_BIN * ITEMSIZE:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(n, N_CHAN_BIN)
    return block[:, chan].astype(np.float64) * GAIN_TO_UV


PAD = int(PAD_MS * FS / 1000)

fig, axes = plt.subplots(len(UNITS), N_EXAMPLE_BURSTS, figsize=(6.5 * N_EXAMPLE_BURSTS, 4.2 * len(UNITS)))

with open(DAT_PATH, "rb") as f:
    for row, uid in enumerate(UNITS):
        chan = int(info.loc[uid, "ch"])
        st = spike_times_samples[spike_clusters == uid]
        bursts = find_bursts(st)
        bursts_sorted = sorted(bursts, key=len, reverse=True)
        print(f"unit {uid} (ch{chan}): {len(bursts)} bursts, longest={len(bursts_sorted[0])}")

        example_idxs = np.linspace(0, min(len(bursts_sorted), 40) - 1, N_EXAMPLE_BURSTS).astype(int)
        for col, bi in enumerate(example_idxs):
            ax = axes[row, col]
            burst = bursts_sorted[bi]
            s0, s1 = int(burst[0]) - PAD, int(burst[-1]) + PAD
            trace = get_continuous(chan, s0, s1, f)
            t_ms = (np.arange(s0, s1) - burst[0]) / FS * 1000
            ax.plot(t_ms, trace, color="#2b2b2b", lw=0.9)

            amps = []
            for k, s in enumerate(burst):
                idx = int(s) - s0
                ax.plot(t_ms[idx], trace[idx], marker="v", color="#d62728", ms=8, zorder=5)
                ax.annotate(f"#{k+1}", (t_ms[idx], trace[idx]), textcoords="offset points",
                            xytext=(0, 10), fontsize=7, color="#d62728", ha="center")
                lo, hi = max(0, idx - 20), min(len(trace), idx + 20)
                amps.append(trace[lo:hi].max() - trace[lo:hi].min())

            isis = np.diff(burst) / FS * 1000
            amp_str = " -> ".join(f"{a:.0f}" for a in amps)
            ax.set_title(f"unit {uid}, burst {bi+1} (n={len(burst)})\n"
                         f"ISIs: " + ", ".join(f"{x:.1f}" for x in isis) + " ms\n"
                         f"peak-trough (uV): {amp_str}", fontsize=8)
            ax.set_xlabel("ms", fontsize=8)
            if col == 0:
                ax.set_ylabel("uV", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="grey", lw=0.4)

plt.suptitle("20260901_085606 -- continuous raw trace per burst (peak channel), detected spikes marked (red)",
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT, "burst_raw_trace_20260901_085606.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
