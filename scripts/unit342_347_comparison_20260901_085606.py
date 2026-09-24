"""
Four-way comparison on the shared channel (ch41):
  A) unit 342 spikes that occur CLOSE (<3ms) to a unit 347 spike
  B) unit 347 spikes that occur CLOSE (<3ms) to a unit 342 spike
  C) unit 342 spikes that are ISOLATED (far from any 347 spike)
  D) unit 347 spikes that are ISOLATED (far from any 342 spike)
Raw >300Hz filtered trace, both units' spike times marked wherever they
land in each window.

Usage: python unit342_347_comparison_20260901_085606.py
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
ITEMSIZE = 2
U347, U342 = 347, 342
WIN_MS = 5.0
CLOSE_THRESH_MS = 3.0
FAR_THRESH_MS = 30.0
N_EXAMPLES = 6
RNG_SEED = 7

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_all = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters_all = np.load(VR + r"\spike_clusters.npy").ravel()

chan = int(info.loc[U342, "ch"])
st347 = np.sort(spike_times_all[spike_clusters_all == U347])
st342 = np.sort(spike_times_all[spike_clusters_all == U342])

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def nearest_dist_ms(times_a, times_b):
    idx = np.searchsorted(times_b, times_a)
    out = np.empty(len(times_a))
    for i, (s, j) in enumerate(zip(times_a, idx)):
        cands = []
        if j < len(times_b):
            cands.append(abs(times_b[j] - s))
        if j > 0:
            cands.append(abs(times_b[j - 1] - s))
        out[i] = (min(cands) / FS * 1000) if cands else np.inf
    return out


dist_342_to_347 = nearest_dist_ms(st342, st347)
dist_347_to_342 = nearest_dist_ms(st347, st342)

close_342 = st342[dist_342_to_347 < CLOSE_THRESH_MS]
far_342 = st342[dist_342_to_347 > FAR_THRESH_MS]
close_347 = st347[dist_347_to_342 < CLOSE_THRESH_MS]
far_347 = st347[dist_347_to_342 > FAR_THRESH_MS]

print(f"342 close-to-347: {len(close_342)} / {len(st342)}")
print(f"342 far-from-347: {len(far_342)} / {len(st342)}")
print(f"347 close-to-342: {len(close_347)} / {len(st347)}")
print(f"347 far-from-342: {len(far_347)} / {len(st347)}")

rng = np.random.default_rng(RNG_SEED)


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


PAD = int(WIN_MS * FS / 1000)

categories = [
    ("A) unit 342 spikes CLOSE to a 347 spike", close_342, "#d62728"),
    ("B) unit 347 spikes CLOSE to a 342 spike", close_347, "#1f4e8c"),
    ("C) unit 342 spikes ISOLATED (far from 347)", far_342, "#d62728"),
    ("D) unit 347 spikes ISOLATED (far from 342)", far_347, "#1f4e8c"),
]

fig, axes = plt.subplots(len(categories), N_EXAMPLES, figsize=(3.6 * N_EXAMPLES, 3.6 * len(categories)))

with open(DAT_PATH, "rb") as f:
    for row, (label, pool, color) in enumerate(categories):
        n_pick = min(N_EXAMPLES, len(pool))
        picks = rng.choice(pool, size=n_pick, replace=False) if len(pool) > 0 else []
        for col in range(N_EXAMPLES):
            ax = axes[row, col]
            if col >= n_pick:
                ax.axis("off")
                continue
            center = int(picks[col])
            s0, s1 = center - PAD, center + PAD
            trace = get_filtered_segment(chan, s0, s1, f)
            t_ms = (np.arange(s0, s1) - center) / FS * 1000
            ax.plot(t_ms, trace, color="#2b2b2b", lw=1.0)

            in_342 = st342[(st342 >= s0) & (st342 < s1)]
            in_347 = st347[(st347 >= s0) & (st347 < s1)]
            for s in in_342:
                idx = int(s) - s0
                ax.plot(t_ms[idx], trace[idx], marker="v", color="#d62728", ms=10, zorder=6,
                        label="u342" if col == 0 else None)
            for s in in_347:
                idx = int(s) - s0
                ax.plot(t_ms[idx], trace[idx], marker="o", mfc="none", mec="#1f4e8c", ms=11, mew=1.8,
                        zorder=5, label="u347" if col == 0 else None)

            ax.axhline(0, color="grey", lw=0.4)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"example {col+1}", fontsize=8)
            if col == 0:
                ax.set_ylabel(label.split(")")[0] + ")", fontsize=8)
                ax.legend(fontsize=6, loc="upper right")

plt.suptitle("20260901_085606 -- 342 vs 347 on shared ch41: red triangle=342, open blue circle=347\n" +
             "\n".join(l for l, _, _ in categories), fontsize=10)
plt.tight_layout()
out_path = os.path.join(OUT, "unit342_347_comparison_20260901_085606.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("saved", out_path)
