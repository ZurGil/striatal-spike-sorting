"""
Investigate unit 347 (ch41, mua, 277383 spikes -- shares peak channel with
unit 342): ACG, mean waveform, amplitude distribution, and specifically:
do 347's LARGEST-amplitude spikes tend to occur right next to 342 spikes
(suggesting some of 347 is actually mis-split/duplicate detections of 342,
or genuine collisions), vs. 347's more typical smaller spikes?

Usage: python investigate_unit347_20260901_085606.py
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
NT, NT0MIN = 61, 20
ITEMSIZE = 2
U347, U342 = 347, 342

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_all = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters_all = np.load(VR + r"\spike_clusters.npy").ravel()
amplitudes_all = np.load(VR + r"\amplitudes.npy").ravel()

chan347 = int(info.loc[U347, "ch"])
chan342 = int(info.loc[U342, "ch"])
print(f"unit 347: ch{chan347}, {int(info.loc[U347,'n_spikes'])} spikes")
print(f"unit 342: ch{chan342}, {int(info.loc[U342,'n_spikes'])} spikes")

mask347 = spike_clusters_all == U347
st347 = spike_times_all[mask347]
amp347 = amplitudes_all[mask347]
order = np.argsort(st347)
st347, amp347 = st347[order], amp347[order]

st342 = np.sort(spike_times_all[spike_clusters_all == U342])

# --- ACG for 347 ---
st_s = st347.astype(np.float64) / FS
WIN_S, BIN_S = 0.05, 0.001
n_bins = int(2 * WIN_S / BIN_S)
acg = np.zeros(n_bins)
j_lo = j_hi = 0
N = len(st_s)
# subsample for speed if huge
if N > 200000:
    sub = np.sort(np.random.default_rng(0).choice(st_s, 200000, replace=False))
else:
    sub = st_s
Ns = len(sub)
for i in range(Ns):
    while j_lo < Ns and sub[j_lo] < sub[i] - WIN_S:
        j_lo += 1
    while j_hi < Ns and sub[j_hi] <= sub[i] + WIN_S:
        j_hi += 1
    diffs = sub[j_lo:j_hi] - sub[i]
    diffs = diffs[diffs != 0]
    idx = ((diffs + WIN_S) / BIN_S).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    for k in idx:
        acg[k] += 1
acg_centers_ms = (np.arange(n_bins) + 0.5) * BIN_S * 1000 - WIN_S * 1000
refractory_frac = acg[(np.abs(acg_centers_ms) < 2.0)].sum() / max(acg.sum(), 1)
print(f"unit 347 ACG (n={Ns} subsampled): fraction within +/-2ms = {refractory_frac:.4f}")

# --- amplitude distribution + distance to nearest 342 spike ---
nearest_342_dist_ms = np.empty(len(st347))
idx_342 = np.searchsorted(st342, st347)
for i, (s, j) in enumerate(zip(st347, idx_342)):
    cands = []
    if j < len(st342):
        cands.append(abs(st342[j] - s))
    if j > 0:
        cands.append(abs(st342[j - 1] - s))
    nearest_342_dist_ms[i] = (min(cands) / FS * 1000) if cands else np.inf

# correlation: amplitude vs proximity to a 342 spike
close_mask = nearest_342_dist_ms < 3.0  # within 3ms of a 342 spike
print(f"\n347 spikes within 3ms of a 342 spike: {close_mask.sum()} / {len(st347)} ({100*close_mask.mean():.1f}%)")
print(f"mean amplitude (close to 342): {amp347[close_mask].mean():.1f}")
print(f"mean amplitude (far from 342): {amp347[~close_mask].mean():.1f}")

# top amplitude 347 spikes
top_n = 12
top_idx = np.argsort(amp347)[::-1][:top_n]
print(f"\ntop {top_n} amplitude 347 spikes: distance to nearest 342 spike (ms):")
for i in top_idx:
    print(f"  amp={amp347[i]:.1f}, dist_to_342={nearest_342_dist_ms[i]:.2f}ms")

# --- plot ---
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered_segment(chan, s0, s1, f, buf=1800):
    n = (s1 + buf) - (s0 - buf)
    f.seek(int(s0 - buf) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read(n * N_CHAN_BIN * ITEMSIZE)
    if len(raw) != n * N_CHAN_BIN * ITEMSIZE:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(n, N_CHAN_BIN)
    trace_raw = block[:, chan].astype(np.float64)
    trace_filt = filtfilt(b_hp, a_hp, trace_raw)
    return trace_filt[buf:-buf] * GAIN_TO_UV


PAD = int(3.0 * FS / 1000)

fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(4, 6)

# row0: ACG, amplitude histogram, amplitude vs distance scatter
ax = fig.add_subplot(gs[0, 0:2])
ax.bar(acg_centers_ms, acg, width=BIN_S * 1000, color="#444")
ax.axvspan(-2, 2, color="red", alpha=0.15)
ax.set_title(f"unit 347 ACG (subsample n={Ns})\n{refractory_frac*100:.2f}% within +/-2ms")
ax.set_xlabel("lag (ms)")

ax = fig.add_subplot(gs[0, 2:4])
ax.hist(amp347, bins=100, color="#555")
ax.set_title("unit 347 amplitude distribution (a.u., Kilosort amplitudes.npy)")
ax.set_xlabel("amplitude"); ax.set_ylabel("count")
ax.axvline(np.median(amp347[top_idx[-1:]]), color="r", ls="--", lw=1)

ax = fig.add_subplot(gs[0, 4:6])
plot_mask = nearest_342_dist_ms < 50
ax.scatter(nearest_342_dist_ms[plot_mask], amp347[plot_mask], s=2, alpha=0.15, color="#1f4e8c")
ax.set_xlabel("distance to nearest unit-342 spike (ms)")
ax.set_ylabel("347 amplitude (a.u.)")
ax.set_title("347 amplitude vs proximity to a 342 spike")

# rows 1-3: top-amplitude 347 spike examples with both units marked
with open(DAT_PATH, "rb") as f:
    for k, i in enumerate(top_idx):
        r, c = 1 + k // 6, k % 6
        ax = fig.add_subplot(gs[r, c])
        s347 = int(st347[i])
        s0, s1 = s347 - PAD, s347 + PAD
        trace = get_filtered_segment(chan347, s0, s1, f)
        t_ms = (np.arange(s0, s1) - s347) / FS * 1000
        ax.plot(t_ms, trace, color="#2b2b2b", lw=1.0)
        ax.plot(0, trace[PAD], marker="v", color="#d62728", ms=9, zorder=5, label="u347")
        # mark any 342 spikes in window
        in_win_342 = st342[(st342 >= s0) & (st342 < s1)]
        for s2 in in_win_342:
            idx2 = int(s2) - s0
            ax.plot(t_ms[idx2], trace[idx2], marker="o", mfc="none", mec="#2ca02c", ms=11, mew=2, label="u342")
        ax.set_title(f"347 amp={amp347[i]:.0f}, dist_to_342={nearest_342_dist_ms[i]:.2f}ms", fontsize=8)
        ax.set_xlabel("ms", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6)

plt.suptitle("20260901_085606 -- unit 347 (mua, ch41, shares channel w/ unit 342): ACG, amplitude, and largest-amplitude spike examples",
             fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT, "investigate_unit347_20260901_085606.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("saved", out_path)
