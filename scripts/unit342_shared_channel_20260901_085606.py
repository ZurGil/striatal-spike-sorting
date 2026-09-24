"""
Unit 342 shares its peak channel (ch41) with unit 347 (mua, 277k spikes!)
and has several other units within 60um. Re-plot the same 10 random
unit-342 bursts, but now mark EVERY cluster's spikes (not just 342's)
that fall inside each window, to see whether the "extra" unmarked bumps
are actually already-detected spikes from a different unit sharing the
channel, or genuinely undetected by anything.

Usage: python unit342_shared_channel_20260901_085606.py
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
PAD_AFTER_MS = 10.0
FILT_BUFFER_MS = 60.0
ITEMSIZE = 2
N_EXAMPLE_BURSTS = 10
RNG_SEED = 42
UID = 342

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_all = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters_all = np.load(VR + r"\spike_clusters.npy").ravel()

chan = int(info.loc[UID, "ch"])
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

# all units within 60um of ch41 (same probe neighborhood)
pos = np.load(VR + r"\channel_positions.npy")
d = np.sqrt(((pos - pos[chan]) ** 2).sum(1))
near_chans = set(np.where(d <= 60)[0].tolist())
neighbor_units = info[info.ch.isin(near_chans)].index.tolist()
print("neighborhood units (ch within 60um of 41):", neighbor_units)

st_uid = spike_times_all[spike_clusters_all == UID]
bursts_order = np.argsort(st_uid)
st_uid = st_uid[bursts_order]


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

bursts = find_bursts(st_uid)
n_pick = min(N_EXAMPLE_BURSTS, len(bursts))
chosen_idx = rng.choice(len(bursts), size=n_pick, replace=False)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
color_map = {u: colors[i % 10] for i, u in enumerate(neighbor_units)}

ncols = 5
nrows = int(np.ceil(n_pick / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
axes = np.atleast_2d(axes)

n_other_found_total = 0
with open(DAT_PATH, "rb") as f:
    for i, bidx in enumerate(chosen_idx):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        burst = bursts[bidx]
        s0, s1 = int(burst[0]) - PAD, int(burst[-1]) + PAD_AFTER
        trace = get_filtered_segment(chan, s0, s1, f, BUF)
        t_ms = (np.arange(s0, s1) - burst[0]) / FS * 1000
        ax.plot(t_ms, trace, color="#2b2b2b", lw=0.9, zorder=1)

        for s in burst:
            idx = int(s) - s0
            ax.plot(t_ms[idx], trace[idx], marker="v", color="#d62728", ms=9, zorder=5)

        # find spikes from OTHER neighborhood units inside this window
        in_win = (spike_times_all >= s0) & (spike_times_all < s1) & (spike_clusters_all != UID) \
                 & np.isin(spike_clusters_all, neighbor_units)
        other_times = spike_times_all[in_win]
        other_clusters = spike_clusters_all[in_win]
        seen_labels = set()
        for s, cl in zip(other_times, other_clusters):
            idx = int(s) - s0
            if 0 <= idx < len(trace):
                lbl = f"u{cl}" if cl not in seen_labels else None
                ax.plot(t_ms[idx], trace[idx], marker="o", mfc="none",
                         mec=color_map[cl], ms=10, mew=1.8, zorder=6, label=lbl)
                seen_labels.add(cl)
                n_other_found_total += 1

        isis = np.diff(burst) / FS * 1000
        ax.set_title(f"unit {UID}, burst idx {bidx} (n={len(burst)})\n"
                     f"ISIs: " + ", ".join(f"{x:.1f}" for x in isis) + " ms", fontsize=8)
        ax.set_xlabel("ms", fontsize=8)
        if c == 0:
            ax.set_ylabel("uV (>300Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="grey", lw=0.4)
        if seen_labels:
            ax.legend(fontsize=6, loc="upper right")

for i in range(n_pick, nrows * ncols):
    r, c = i // ncols, i % ncols
    axes[r, c].axis("off")

print(f"\ntotal other-cluster spikes found inside these {n_pick} burst windows: {n_other_found_total}")

plt.suptitle(f"20260901_085606 -- unit {UID} bursts: red=unit342 spikes, open circles=other units sharing this channel neighborhood",
             fontsize=12)
plt.tight_layout()
out_path = os.path.join(OUT, "unit342_shared_channel_20260901_085606.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("saved", out_path)
