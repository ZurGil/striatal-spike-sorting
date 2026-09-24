"""
For one 'good' unit (224) and one 'noise' unit (195) -- both strongly
poke- AND reward-significant -- show mean raw waveform (peak channel,
real uV, extracted directly from probe1.dat around real spike times, no
Kilosort template/whitening involved) side by side with their poke and
reward PSTHs.

Usage: python build_waveform_response_comparison_20260901_085606.py
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
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT, NT0MIN = 61, 20
N_WF_SAMPLE = 300  # spikes to average for the mean waveform

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

GOOD_UID, NOISE_UID = 224, 195


def mean_waveform(uid, chan):
    st = spike_times_samples[spike_clusters == uid]
    rng = np.random.default_rng(0)
    sample = rng.choice(st, size=min(N_WF_SAMPLE, len(st)), replace=False)
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
    wfs = np.array(wfs) * GAIN_TO_UV
    return wfs.mean(0), wfs.std(0) / np.sqrt(len(wfs)), len(wfs)


good_chan = int(info.loc[GOOD_UID, "ch"])
noise_chan = int(info.loc[NOISE_UID, "ch"])
good_wf, good_sem, good_n = mean_waveform(GOOD_UID, good_chan)
noise_wf, noise_sem, noise_n = mean_waveform(NOISE_UID, noise_chan)
print(f"good unit {GOOD_UID}: peak channel {good_chan}, {good_n} spikes sampled")
print(f"noise unit {NOISE_UID}: peak channel {noise_chan}, {noise_n} spikes sampled")

# --- PSTH data ---
NEW = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review\synced"
trials = pd.read_parquet(NEW + r"\trials.parquet")
se = pd.read_parquet(NEW + r"\state_events.parquet")
good_spikes = pd.read_parquet(NEW + r"\spikes.parquet", columns=["unit_id", "trial_id", "spike_time_in_trial"])
noise_spikes = pd.read_parquet(OUT + r"\noise_unit_spikes_20260901_085606.parquet",
                                columns=["unit_id", "trial_id", "spike_time_in_trial"])

wait_sin = (se[se.state_name == "wait_Sin"].sort_values(["trial_id", "start_time_in_trial"])
            .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
cue_end, poke_time = wait_sin["start_time_in_trial"], wait_sin["end_time_in_trial"]
water = (se[se.state_name.isin(["water_L", "water_R"])].sort_values(["trial_id", "start_time_in_trial"])
         .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
water_start = water["start_time_in_trial"]

poke_valid = trials[trials.TrialCompleted == 1].trial_id
poke_valid = poke_valid[poke_valid.isin(cue_end.index)].to_numpy()
reward_valid = trials[trials.Rewarded == 1].trial_id
reward_valid = reward_valid[reward_valid.isin(water_start.index)].to_numpy()

decision_s = (poke_time - cue_end).reindex(poke_valid).sort_values()
poke_order = decision_s.index.to_numpy()
reward_order = water_start.reindex(reward_valid).sort_values().index.to_numpy()


def psth(spikes_df, uid, align_times, order, xlim):
    BIN_S = 0.05
    bins = np.arange(xlim[0], xlim[1] + 0.001, BIN_S)
    centers = (bins[:-1] + bins[1:]) / 2
    usp = spikes_df[spikes_df.unit_id == uid]
    bt = usp.groupby("trial_id")["spike_time_in_trial"]
    counts = np.zeros(len(centers))
    for tid in order:
        if tid in bt.groups:
            st = bt.get_group(tid).to_numpy() - align_times.loc[tid]
            h, _ = np.histogram(st, bins=bins)
            counts += h
    rate = counts / BIN_S / len(order)
    return centers, gaussian_filter1d(rate, sigma=2, mode="nearest")


t_ms = (np.arange(NT) - NT0MIN) / FS * 1000

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, (label, wf, sem, uid, spikes_df, color) in enumerate([
    ("GOOD unit 224", good_wf, good_sem, GOOD_UID, good_spikes, "#1f4e8c"),
    ("NOISE unit 195", noise_wf, noise_sem, NOISE_UID, noise_spikes, "#a83232"),
]):
    ax_wf = axes[row, 0]
    ax_wf.plot(t_ms, wf, color=color, lw=1.8)
    ax_wf.fill_between(t_ms, wf - sem, wf + sem, color=color, alpha=0.25)
    ax_wf.set_title(f"{label} -- mean waveform (peak ch)")
    ax_wf.set_xlabel("ms")
    ax_wf.set_ylabel("uV")
    ax_wf.axhline(0, color="grey", lw=0.5)

    ax_poke = axes[row, 1]
    c, r = psth(spikes_df, uid, cue_end, poke_order, (-1.0, 1.5))
    ax_poke.plot(c, r, color="#333", lw=1.6)
    ax_poke.axvline(0, color="k", lw=0.8, ls="--")
    ax_poke.set_title(f"{label} -- poke PSTH")
    ax_poke.set_xlabel("t from stim end (s)")
    ax_poke.set_ylabel("Hz")

    ax_rew = axes[row, 2]
    c, r = psth(spikes_df, uid, water_start, reward_order, (-1.0, 2.0))
    ax_rew.plot(c, r, color="#333", lw=1.6)
    ax_rew.axvline(0, color="k", lw=0.8, ls="--")
    ax_rew.set_title(f"{label} -- reward PSTH")
    ax_rew.set_xlabel("t from reward (s)")
    ax_rew.set_ylabel("Hz")

plt.suptitle("20260901_085606 -- waveform vs response, GOOD unit 224 vs NOISE unit 195 (both strongly poke- and reward-significant)")
plt.tight_layout()
out_path = os.path.join(OUT, "waveform_vs_response_good_vs_noise_20260901_085606.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print("saved", out_path)
