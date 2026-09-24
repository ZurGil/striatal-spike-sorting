"""
Investigate unit 42 (20260901_085606, final Phy label = noise, ch357,
98939 spikes): individual raw spike traces (peak channel) vs mean
waveform vs ACG vs reward PSTH, to understand why raw traces look
noisy but mean waveform/ACG/response look real.

Usage: python investigate_unit42_20260901_085606.py
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
NEW = VR + r"\synced"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT, NT0MIN = 61, 20
UID = 42

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
chan = int(info.loc[UID, "ch"])
n_total = int(info.loc[UID, "n_spikes"])

spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
st_samples = spike_times_samples[spike_clusters == UID]
print(f"unit {UID}: ch{chan}, {len(st_samples)} spikes (cluster_info says {n_total})")

# --- 1. individual raw spike snippets + mean waveform ---
def get_snippets(samples, chan, n, seed=0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(samples, size=min(n, len(samples)), replace=False)
    wfs = []
    itemsize = 2
    with open(DAT_PATH, "rb") as f:
        for s in sample:
            lo = int(s) - NT0MIN
            f.seek(lo * N_CHAN_BIN * itemsize)
            raw = f.read(NT * N_CHAN_BIN * itemsize)
            if len(raw) != NT * N_CHAN_BIN * itemsize:
                continue
            block = np.frombuffer(raw, dtype=np.int16).reshape(NT, N_CHAN_BIN)
            wfs.append(block[:, chan].astype(np.float64))
    return np.array(wfs) * GAIN_TO_UV

show_wfs = get_snippets(st_samples, chan, 60)
mean_wfs = get_snippets(st_samples, chan, 500)
mean_wf = mean_wfs.mean(0)
sem_wf = mean_wfs.std(0) / np.sqrt(len(mean_wfs))
t_ms = (np.arange(NT) - NT0MIN) / FS * 1000

# --- 2. ACG ---
sample_rate_hz = FS
st_s = np.sort(st_samples.astype(np.float64) / sample_rate_hz)
WIN_S, BIN_S = 0.05, 0.001
n_bins = int(2 * WIN_S / BIN_S)
acg = np.zeros(n_bins)
j_lo = j_hi = 0
N = len(st_s)
for i in range(N):
    while j_lo < N and st_s[j_lo] < st_s[i] - WIN_S:
        j_lo += 1
    while j_hi < N and st_s[j_hi] <= st_s[i] + WIN_S:
        j_hi += 1
    diffs = st_s[j_lo:j_hi] - st_s[i]
    diffs = diffs[diffs != 0]
    idx = ((diffs + WIN_S) / BIN_S).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    for k in idx:
        acg[k] += 1
acg_centers_ms = (np.arange(n_bins) + 0.5) * BIN_S * 1000 - WIN_S * 1000
refractory_frac = acg[(np.abs(acg_centers_ms) < 2.0)].sum() / max(acg.sum(), 1)
print(f"ACG: fraction of pairs within +/-2ms = {refractory_frac:.4f}")

# --- 3. reward PSTH (already-computed noise-unit spikes table) ---
trials = pd.read_parquet(NEW + r"\trials.parquet")
se = pd.read_parquet(NEW + r"\state_events.parquet")
spikes = pd.read_parquet(OUT + r"\noise_unit_spikes_20260901_085606.parquet",
                          columns=["unit_id", "trial_id", "spike_time_in_trial"])
water = (se[se.state_name.isin(["water_L", "water_R"])].sort_values(["trial_id", "start_time_in_trial"])
         .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
water_start = water["start_time_in_trial"]
reward_valid = trials[trials.Rewarded == 1].trial_id
reward_valid = reward_valid[reward_valid.isin(water_start.index)].to_numpy()
reward_order = water_start.reindex(reward_valid).sort_values().index.to_numpy()

usp = spikes[spikes.unit_id == UID]
bt = usp.groupby("trial_id")["spike_time_in_trial"]
xlim = (-1.0, 2.0)
BIN_S2 = 0.05
bins = np.arange(xlim[0], xlim[1] + 0.001, BIN_S2)
centers = (bins[:-1] + bins[1:]) / 2
mat = np.zeros((len(reward_order), len(centers)))
raster_x, raster_y = [], []
for row_i, tid in enumerate(reward_order):
    t0 = water_start.loc[tid]
    if tid in bt.groups:
        st = bt.get_group(tid).to_numpy() - t0
        stp = st[(st >= xlim[0]) & (st <= xlim[1])]
        raster_x.append(stp)
        raster_y.append(np.full_like(stp, row_i))
        h, _ = np.histogram(st, bins=bins)
        mat[row_i] = h / BIN_S2
mean_rate = gaussian_filter1d(mat.mean(0), sigma=2, mode="nearest")

# --- plot ---
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

ax = axes[0, 0]
for wf in show_wfs:
    ax.plot(t_ms, wf, color="#a83232", alpha=0.2, lw=0.6)
ax.plot(t_ms, mean_wf, color="black", lw=1.8)
ax.set_title(f"unit {UID}: {len(show_wfs)} individual raw spikes (ch{chan})")
ax.set_xlabel("ms"); ax.set_ylabel("uV")

ax = axes[0, 1]
ax.plot(t_ms, mean_wf, color="#1f4e8c", lw=2)
ax.fill_between(t_ms, mean_wf - sem_wf, mean_wf + sem_wf, color="#1f4e8c", alpha=0.25)
ax.axhline(0, color="grey", lw=0.5)
ax.set_title(f"unit {UID}: mean waveform (n={len(mean_wfs)}) +/- SEM")
ax.set_xlabel("ms"); ax.set_ylabel("uV")

ax = axes[0, 2]
ax.bar(acg_centers_ms, acg, width=BIN_S * 1000, color="#444")
ax.axvspan(-2, 2, color="red", alpha=0.15, label="+/-2ms refractory zone")
ax.set_title(f"unit {UID}: ACG ({N} spikes total)\n{refractory_frac*100:.2f}% of pairs within +/-2ms")
ax.set_xlabel("lag (ms)"); ax.set_ylabel("count")
ax.legend(fontsize=8)

ax = axes[1, 0]
for x, y in zip(raster_x, raster_y):
    ax.scatter(x, y, s=1.5, c="#a83232", marker="|", linewidths=0.4)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlim(*xlim)
ax.set_title(f"unit {UID}: reward raster ({len(reward_order)} trials)")
ax.set_xlabel("t from reward (s)"); ax.set_ylabel("trial")

ax = axes[1, 1]
ax.plot(centers, mean_rate, color="#333", lw=1.8)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_title(f"unit {UID}: reward PSTH (p_fdr=5.2e-34)")
ax.set_xlabel("t from reward (s)"); ax.set_ylabel("Hz")

axes[1, 2].axis("off")
axes[1, 2].text(0, 0.9,
    f"unit {UID} summary\n\n"
    f"channel: {chan}\ntotal spikes: {N}\n"
    f"final Phy label: noise\nKSLabel: mua\n\n"
    f"poke: base 5.18Hz -> resp 5.67Hz, p_fdr=0.36 (NS)\n"
    f"reward: base 5.29Hz -> resp 16.07Hz, p_fdr=5.2e-34 (sig)\n\n"
    f"ACG within +/-2ms: {refractory_frac*100:.2f}%",
    fontsize=10, va="top", family="monospace")

plt.suptitle("20260901_085606 -- unit 42 (final label: NOISE) -- individual spikes vs mean waveform vs ACG vs response")
plt.tight_layout()
out_path = os.path.join(OUT, "investigate_unit42_20260901_085606.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
