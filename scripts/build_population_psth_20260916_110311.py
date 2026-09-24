import os
import pandas as pd, numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced_NEWFIX"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
trials = pd.read_parquet(NEW + r"\trials.parquet")
se = pd.read_parquet(NEW + r"\state_events.parquet")
spikes = pd.read_parquet(NEW + r"\spikes.parquet", columns=["unit_id", "trial_id", "spike_time_in_trial"])

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

by_unit = spikes.groupby("unit_id")


def make_fig(uids, align_times, order, xlim, align_label, fname, suptitle):
    BIN_S = 0.05
    bins = np.arange(xlim[0], xlim[1] + 0.001, BIN_S)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(2, len(uids), figsize=(4 * len(uids), 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    for col, uid in enumerate(uids):
        ax_r, ax_p = axes[0, col], axes[1, col]
        if uid not in by_unit.groups:
            continue
        usp = by_unit.get_group(uid)
        bt = usp.groupby("trial_id")["spike_time_in_trial"]
        mat = np.zeros((len(order), len(centers)))
        for row_i, tid in enumerate(order):
            t0 = align_times.loc[tid]
            if tid in bt.groups:
                st = bt.get_group(tid).to_numpy() - t0
                stp = st[(st >= xlim[0]) & (st <= xlim[1])]
                ax_r.scatter(stp, np.full_like(stp, row_i), s=2, c="#1f4e8c", marker="|", linewidths=0.5)
                h, _ = np.histogram(st, bins=bins)
                mat[row_i] = h / BIN_S
        ax_r.axvline(0, color="k", lw=0.9, ls="--")
        ax_r.set_xlim(*xlim)
        ax_r.set_title(f"unit {uid}", fontsize=10)
        if col == 0:
            ax_r.set_ylabel("trial")
        mean = mat.mean(0)
        mean_s = gaussian_filter1d(mean, sigma=2, mode="nearest")
        ax_p.plot(centers, mean_s, color="#333", lw=1.6)
        ax_p.axvline(0, color="k", lw=0.9, ls="--")
        ax_p.set_xlabel(f"time from {align_label} (s)")
        if col == 0:
            ax_p.set_ylabel("firing rate (Hz)")
    plt.suptitle(suptitle)
    plt.tight_layout()
    out_path = os.path.join(OUT, fname)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print("saved", out_path)


make_fig([250, 254, 461, 289, 54, 343], cue_end, poke_order, (-1.0, 1.5), "stimulus end",
          "psth_poke_TIMEFIXED_20260916_110311.png",
          "20260916_110311 (TIME-CORRECTED, no Phy yet) -- top poke-responsive good units, aligned to stimulus end")
make_fig([35, 21, 250, 328, 461, 331], water_start, reward_order, (-1.0, 2.0), "reward delivery",
          "psth_reward_TIMEFIXED_20260916_110311.png",
          "20260916_110311 (TIME-CORRECTED, no Phy yet) -- top reward-responsive good units, aligned to water delivery")
