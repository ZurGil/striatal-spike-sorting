import os
import pandas as pd, numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:\Gil\spike_sorting_agent\outputs"
NEW = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review\synced"
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


def make_fig(uids, align_times, order, xlim, align_label, fname, suptitle, ncols=6):
    BIN_S = 0.05
    bins = np.arange(xlim[0], xlim[1] + 0.001, BIN_S)
    centers = (bins[:-1] + bins[1:]) / 2
    nrows = int(np.ceil(len(uids) / ncols))
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), sharex=True)
    for i, uid in enumerate(uids):
        r, c = (i // ncols) * 2, i % ncols
        ax_r, ax_p = axes[r, c], axes[r + 1, c]
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
                ax_r.scatter(stp, np.full_like(stp, row_i), s=1.5, c="#1f4e8c", marker="|", linewidths=0.4)
                h, _ = np.histogram(st, bins=bins)
                mat[row_i] = h / BIN_S
        ax_r.axvline(0, color="k", lw=0.8, ls="--")
        ax_r.set_xlim(*xlim)
        ax_r.set_title(f"unit {uid}", fontsize=9)
        ax_r.set_yticks([])
        mean = mat.mean(0)
        mean_s = gaussian_filter1d(mean, sigma=2, mode="nearest")
        ax_p.plot(centers, mean_s, color="#333", lw=1.4)
        ax_p.axvline(0, color="k", lw=0.8, ls="--")
        if r + 1 == axes.shape[0] - 1 or i >= len(uids) - ncols:
            ax_p.set_xlabel(f"t from {align_label} (s)", fontsize=8)
        ax_p.tick_params(labelsize=7)
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUT, fname)
    plt.savefig(out_path, dpi=105, bbox_inches="tight")
    print("saved", out_path)


poke_uids = [224, 382, 141, 159, 100, 344, 41, 372, 360, 104, 135, 106]
reward_uids = [40, 36, 53, 19, 35, 199, 65, 1, 372, 301, 182, 381]

make_fig(poke_uids, cue_end, poke_order, (-1.0, 1.5), "stim end",
          "psth_gallery_poke_GOOD_20260901_085606.png",
          "20260901_085606, TIME-CORRECTED -- 12 significant GOOD units spanning strong to marginal effect size, aligned to stimulus end")
make_fig(reward_uids, water_start, reward_order, (-1.0, 2.0), "reward",
          "psth_gallery_reward_GOOD_20260901_085606.png",
          "20260901_085606, TIME-CORRECTED -- 12 significant GOOD units spanning strong to marginal effect size, aligned to reward delivery")
