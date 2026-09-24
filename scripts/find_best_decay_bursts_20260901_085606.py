"""
Scan ALL bursts (not random) for units 30, 22, 63, 135 and rank by how
cleanly/monotonically amplitude decreases within the burst, to find the
best real examples of decay to look at (explicitly requested cherry-pick,
for illustration -- not the general-purpose random sample).

Usage: python find_best_decay_bursts_20260901_085606.py
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
MIN_BURST_LEN = 4
PAD_MS = 2.5
PAD_AFTER_MS = 10.0
FILT_BUFFER_MS = 60.0
ITEMSIZE = 2
N_BEST_PER_UNIT = 2

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

best_examples = []  # (score, uid, burst, amps, s0, chan)

with open(DAT_PATH, "rb") as f:
    for uid in UNITS:
        chan = int(info.loc[uid, "ch"])
        st = spike_times_samples[spike_clusters == uid]
        bursts = find_bursts(st)
        scored = []
        for burst in bursts:
            s0, s1 = int(burst[0]) - PAD, int(burst[-1]) + PAD_AFTER
            trace = get_filtered_segment(chan, s0, s1, f, BUF)
            if trace is None:
                continue
            amps = []
            for s in burst:
                idx = int(s) - s0
                lo, hi = max(0, idx - 20), min(len(trace), idx + 20)
                amps.append(trace[lo:hi].max() - trace[lo:hi].min())
            amps = np.array(amps)
            diffs = np.diff(amps)
            n_decreasing = int((diffs < 0).sum())
            monotonic_frac = n_decreasing / len(diffs)
            total_drop = amps[0] - amps[-1]
            rel_drop = total_drop / amps[0] if amps[0] > 0 else -np.inf
            # score: fully monotonic decreasing gets priority, then by relative drop size
            score = monotonic_frac * 10 + rel_drop
            scored.append((score, monotonic_frac, rel_drop, burst, amps, s0, s1))
        scored.sort(key=lambda x: x[0], reverse=True)
        for item in scored[:N_BEST_PER_UNIT]:
            score, mono, rel, burst, amps, s0, s1 = item
            best_examples.append((uid, chan, burst, amps, s0, s1, mono, rel))
            print(f"unit {uid}: monotonic_frac={mono:.2f} rel_drop={rel:.2f} "
                  f"amps={[f'{a:.0f}' for a in amps]}")

n = len(best_examples)
fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
if n == 1:
    axes = [axes]

with open(DAT_PATH, "rb") as f:
    for col, (uid, chan, burst, amps, s0, s1, mono, rel) in enumerate(best_examples):
        trace = get_filtered_segment(chan, s0, s1, f, BUF)
        t_ms = (np.arange(s0, s1) - burst[0]) / FS * 1000
        ax = axes[col]
        ax.plot(t_ms, trace, color="#2b2b2b", lw=1.0)
        for k, s in enumerate(burst):
            idx = int(s) - s0
            ax.plot(t_ms[idx], trace[idx], marker="v", color="#d62728", ms=8, zorder=5)
            ax.annotate(f"#{k+1}\n{amps[k]:.0f}uV", (t_ms[idx], trace[idx]), textcoords="offset points",
                        xytext=(0, 10), fontsize=7.5, color="#d62728", ha="center")
        isis = np.diff(burst) / FS * 1000
        ax.set_title(f"unit {uid} (ch{chan}), n={len(burst)} spikes\n"
                     f"ISIs: " + ", ".join(f"{x:.1f}" for x in isis) + " ms\n"
                     f"{n_decreasing if False else int((np.diff(amps)<0).sum())}/{len(amps)-1} decreasing steps, "
                     f"{rel*100:.0f}% drop 1st->last", fontsize=9)
        ax.set_xlabel("ms")
        if col == 0:
            ax.set_ylabel("uV (>300Hz)")
        ax.axhline(0, color="grey", lw=0.4)

plt.suptitle("20260901_085606 -- best examples of within-burst amplitude DECAY (cherry-picked from full burst set), >300Hz filtered", fontsize=12)
plt.tight_layout()
out_path = os.path.join(OUT, "best_decay_bursts_20260901_085606.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
