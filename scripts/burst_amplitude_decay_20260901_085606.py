"""
For units 30, 22, 63, 135 (expected: amplitude DECAY within a burst) and
unit 38 (expected: OPPOSITE trend) from session 20260901_085606:
find bursts (short-ISI spike runs), show a few example bursts' raw
waveforms (peak channel, color-coded by position in burst) plus a
population-level mean-amplitude-vs-within-burst-position curve.

Usage: python burst_amplitude_decay_20260901_085606.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT, NT0MIN = 61, 20
ISI_THRESH_MS = 10.0
MIN_BURST_LEN = 3
N_EXAMPLE_BURSTS = 3
MAX_BURSTS_FOR_STATS = 2000
MAX_WITHIN_BURST_POS = 6

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

UNITS = [30, 22, 63, 135, 38]
EXPECT = {30: "decay", 22: "decay", 63: "decay", 135: "decay", 38: "OPPOSITE"}


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


def waveform(sample, chan, f):
    itemsize = 2
    lo = int(sample) - NT0MIN
    f.seek(lo * N_CHAN_BIN * itemsize)
    raw = f.read(NT * N_CHAN_BIN * itemsize)
    if len(raw) != NT * N_CHAN_BIN * itemsize:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(NT, N_CHAN_BIN)
    return block[:, chan].astype(np.float64) * GAIN_TO_UV


t_ms = (np.arange(NT) - NT0MIN) / FS * 1000

fig, axes = plt.subplots(len(UNITS), N_EXAMPLE_BURSTS + 1,
                          figsize=(4.2 * (N_EXAMPLE_BURSTS + 1), 3.6 * len(UNITS)))

with open(DAT_PATH, "rb") as f:
    for row, uid in enumerate(UNITS):
        chan = int(info.loc[uid, "ch"])
        st = spike_times_samples[spike_clusters == uid]
        bursts = find_bursts(st)
        bursts_sorted = sorted(bursts, key=len, reverse=True)
        print(f"unit {uid} (ch{chan}, {len(st)} spikes): {len(bursts)} bursts found "
              f"(ISI<{ISI_THRESH_MS}ms, len>={MIN_BURST_LEN}), "
              f"longest={len(bursts_sorted[0]) if bursts else 0}")

        # -- example bursts --
        # pick examples spanning a range of burst lengths (not only the longest)
        example_idxs = np.linspace(0, min(len(bursts_sorted), 40) - 1, N_EXAMPLE_BURSTS).astype(int)
        for col, bi in enumerate(example_idxs):
            ax = axes[row, col]
            burst = bursts_sorted[bi]
            n = len(burst)
            colors = cm.autumn(np.linspace(0, 0.85, n))
            amps = []
            for k, s in enumerate(burst):
                wf = waveform(s, chan, f)
                if wf is None:
                    continue
                ax.plot(t_ms, wf, color=colors[k], lw=1.5, label=f"#{k+1}")
                amps.append(wf.max() - wf.min())
            isis = np.diff(burst) / FS * 1000
            ax.set_title(f"unit {uid} burst (n={n} spikes)\nISIs: " +
                         ", ".join(f"{x:.1f}" for x in isis) + " ms", fontsize=8)
            ax.set_xlabel("ms", fontsize=8)
            if col == 0:
                ax.set_ylabel("uV", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.legend(fontsize=6, ncol=2, loc="upper right")

        # -- population amplitude-vs-position curve --
        ax = axes[row, N_EXAMPLE_BURSTS]
        use_bursts = bursts[:MAX_BURSTS_FOR_STATS]
        pos_amps = {p: [] for p in range(MAX_WITHIN_BURST_POS)}
        for burst in use_bursts:
            for k, s in enumerate(burst[:MAX_WITHIN_BURST_POS]):
                wf = waveform(s, chan, f)
                if wf is None:
                    continue
                pos_amps[k].append(wf.max() - wf.min())
        positions = [p for p in range(MAX_WITHIN_BURST_POS) if len(pos_amps[p]) >= 5]
        means = [np.mean(pos_amps[p]) for p in positions]
        sems = [np.std(pos_amps[p]) / np.sqrt(len(pos_amps[p])) for p in positions]
        ax.errorbar([p + 1 for p in positions], means, yerr=sems, marker="o",
                    color="#333", capsize=3)
        ax.set_title(f"unit {uid} ({EXPECT[uid]}) -- amplitude vs position\n"
                     f"({len(use_bursts)} bursts pooled)", fontsize=8)
        ax.set_xlabel("spike # within burst", fontsize=8)
        ax.set_ylabel("peak-to-trough (uV)", fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle("20260901_085606 -- burst waveform amplitude: units 30/22/63/135 (expect decay) vs unit 38 (expect opposite)",
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT, "burst_amplitude_decay_20260901_085606.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("saved", out_path)
