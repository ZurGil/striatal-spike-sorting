"""
Scan the WHOLE population (excluding units already shown: 30, 22, 63,
135, 38) using Kilosort's own amplitudes.npy (fast, no raw-file reads)
to find a genuinely different unit with strong, clean, population-level
within-burst amplitude decay. Then pull real raw (>300Hz filtered)
example traces for the winner from probe1.dat to visualize.

Usage: python find_best_decay_unit_20260901_085606.py
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
MAX_POS = 4
MIN_SPIKES = 2000
MIN_BURSTS = 30
EXCLUDE_UNITS = {30, 22, 63, 135, 38}

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
amplitudes = np.load(VR + r"\amplitudes.npy").ravel()

order = np.argsort(spike_times_samples)
spike_times_samples = spike_times_samples[order]
spike_clusters = spike_clusters[order]
amplitudes = amplitudes[order]


def find_bursts_with_amp(times_samples, amps, isi_thresh_ms=ISI_THRESH_MS, min_len=MIN_BURST_LEN):
    isi_ms = np.diff(times_samples) / FS * 1000
    breaks = np.where(isi_ms > isi_thresh_ms)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(times_samples)]))
    bursts_t, bursts_a = [], []
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            bursts_t.append(times_samples[s:e])
            bursts_a.append(amps[s:e])
    return bursts_t, bursts_a


candidates = info[(info.n_spikes >= MIN_SPIKES) & (~info.index.isin(EXCLUDE_UNITS))].index.tolist()
print(f"{len(candidates)} candidate units with >={MIN_SPIKES} spikes")

results = []
for uid in candidates:
    mask = spike_clusters == uid
    st = spike_times_samples[mask]
    am = amplitudes[mask]
    order2 = np.argsort(st)
    st, am = st[order2], am[order2]
    bursts_t, bursts_a = find_bursts_with_amp(st, am)
    if len(bursts_t) < MIN_BURSTS:
        continue
    pos_amps = {p: [] for p in range(MAX_POS)}
    mono_count, total = 0, 0
    for ba in bursts_a:
        for p in range(min(len(ba), MAX_POS)):
            pos_amps[p].append(ba[p])
        d = np.diff(ba[:MAX_POS])
        mono_count += int((d < 0).sum())
        total += len(d)
    meds = [np.median(pos_amps[p]) for p in range(MAX_POS) if len(pos_amps[p]) >= 10]
    if len(meds) < 3 or meds[0] <= 0:
        continue
    rel_drop = (meds[0] - meds[-1]) / meds[0]
    mono_frac = mono_count / total if total > 0 else 0
    score = rel_drop * mono_frac
    results.append((score, uid, rel_drop, mono_frac, len(bursts_t), meds,
                     info.loc[uid, "group"], info.loc[uid, "n_spikes"]))

results.sort(key=lambda x: x[0], reverse=True)
print("\nTop 10 candidates by (rel_drop * monotonic_frac):")
for r in results[:10]:
    score, uid, rel_drop, mono_frac, nb, meds, grp, nsp = r
    print(f"unit {uid} ({grp}, {nsp} spikes): score={score:.3f} rel_drop={rel_drop:.2f} "
          f"mono_frac={mono_frac:.2f} n_bursts={nb} median_amps(a.u.)={[f'{m:.0f}' for m in meds]}")

# --- winner: pull real raw filtered examples from probe1.dat ---
winner_uid = results[0][1]
chan = int(info.loc[winner_uid, "ch"])
print(f"\nWinner: unit {winner_uid}, channel {chan}")

mask = spike_clusters == winner_uid
st = np.sort(spike_times_samples[mask])
bursts_t, _ = find_bursts_with_amp(st, np.zeros_like(st), min_len=4)

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
PAD = int(2.5 * FS / 1000)
PAD_AFTER = int(10.0 * FS / 1000)
BUF = int(60.0 * FS / 1000)
ITEMSIZE = 2


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


scored_bursts = []
with open(DAT_PATH, "rb") as f:
    for burst in bursts_t:
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
        if amps[0] > 400:  # exclude artifact-scale outlier first spikes
            continue
        diffs = np.diff(amps)
        mono_frac = (diffs < 0).sum() / len(diffs)
        rel_drop = (amps[0] - amps[-1]) / amps[0] if amps[0] > 0 else -1
        score = mono_frac * 10 + rel_drop
        scored_bursts.append((score, burst, amps, s0, s1))

scored_bursts.sort(key=lambda x: x[0], reverse=True)
n_show = min(4, len(scored_bursts))
fig, axes = plt.subplots(1, n_show, figsize=(5.5 * n_show, 4.5))
if n_show == 1:
    axes = [axes]
with open(DAT_PATH, "rb") as f:
    for col in range(n_show):
        score, burst, amps, s0, s1 = scored_bursts[col]
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
        ax.set_title(f"unit {winner_uid} (ch{chan}, {info.loc[winner_uid,'group']}), n={len(burst)} spikes\n"
                     f"ISIs: " + ", ".join(f"{x:.1f}" for x in isis) + " ms", fontsize=9)
        ax.set_xlabel("ms")
        if col == 0:
            ax.set_ylabel("uV (>300Hz)")
        ax.axhline(0, color="grey", lw=0.4)

plt.suptitle(f"20260901_085606 -- unit {winner_uid}: best population-wide decay candidate (new unit, not 30/22/63/135)",
             fontsize=12)
plt.tight_layout()
out_path = os.path.join(OUT, "best_decay_new_unit_20260901_085606.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
