import pandas as pd, numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SYNCED = r"F:\Gil\Shamir\20260911_100049.rec\20260911_100049.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
spikes = pd.read_parquet(os.path.join(SYNCED, "spikes.parquet"))
se = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))

wait_sin = (se[se.state_name == "wait_Sin"].sort_values(["trial_id", "start_time_in_trial"])
            .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
cue_end = wait_sin["start_time_in_trial"]
poke_time = wait_sin["end_time_in_trial"]
decision_s = poke_time - cue_end

valid = trials[trials.TrialCompleted == 1].trial_id
valid = valid[valid.isin(cue_end.index)]
valid = valid[decision_s.reindex(valid).to_numpy() < 3.0]
choice_left = trials.set_index("trial_id")["ChoiceLeft"].reindex(valid)

order = decision_s.reindex(valid).sort_values().index.to_numpy()
sorted_decision = decision_s.reindex(order).to_numpy()
sorted_side = choice_left.reindex(order).to_numpy()

res = pd.read_csv(os.path.join(OUT, "left_right_cue_response_20260911_100049.csv"))
XLIM = (-1.5, 3.0)
BIN_S = 0.05
BINS = np.arange(XLIM[0], XLIM[1] + 0.001, BIN_S)
centers = (BINS[:-1] + BINS[1:]) / 2
SIGMA_BINS = 2  # ~100ms Gaussian smoothing sigma


def make_fig(units_to_plot, fname, suptitle):
    fig, axes = plt.subplots(2, len(units_to_plot), figsize=(5 * len(units_to_plot), 9),
                              sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for col, uid in enumerate(units_to_plot):
        usp = spikes[spikes.unit_id == uid]
        by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
        ax_r = axes[0, col]
        ax_p = axes[1, col]
        mat = np.zeros((len(order), len(centers)))
        for row_i, tid in enumerate(order):
            t0 = cue_end.loc[tid]
            color = "#3d6a9f" if sorted_side[row_i] == 1.0 else "#c07a2f"
            if tid in by_trial.groups:
                st = by_trial.get_group(tid).to_numpy() - t0
                st_plot = st[(st >= XLIM[0]) & (st <= XLIM[1])]
                ax_r.scatter(st_plot, np.full_like(st_plot, row_i), s=2, c=color, marker="|", linewidths=0.5)
                h, _ = np.histogram(st, bins=BINS)
                mat[row_i] = h / BIN_S
        ax_r.plot(sorted_decision, np.arange(len(order)), color="k", lw=1.0)
        ax_r.axvline(0, color="k", lw=0.8, ls="--")
        ax_r.set_xlim(*XLIM)
        row = res[res.unit_id == uid].iloc[0]
        ax_r.set_title(f"unit {uid} ({row.quality_label})  left p={row.left_p:.4f} (fdr={row.left_p_fdr:.3f})  "
                        f"right p={row.right_p:.4f} (fdr={row.right_p_fdr:.3f})", fontsize=9)
        if col == 0:
            ax_r.set_ylabel("trial (sorted by decision time)")
        mean = mat.mean(0)
        mean_smooth = gaussian_filter1d(mean, sigma=SIGMA_BINS, mode="nearest")
        ax_p.plot(centers, mean, color="#ccc", lw=0.8, label="raw (50ms bins)")
        ax_p.plot(centers, mean_smooth, color="#333", lw=1.6, label="Gaussian-smoothed (sigma~100ms)")
        ax_p.axvline(0, color="k", lw=0.8, ls="--")
        ax_p.set_xlabel("time from cue end (s)")
        if col == 0:
            ax_p.set_ylabel("firing rate (Hz)")
            ax_p.legend(fontsize=7, loc="upper left")
    axes[0, 0].plot([], [], color="k", lw=1.0, label="choice poke (sorted)")
    axes[0, 0].scatter([], [], c="#3d6a9f", marker="|", label="left-trial spike")
    axes[0, 0].scatter([], [], c="#c07a2f", marker="|", label="right-trial spike")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=110, bbox_inches="tight")
    print("saved", fname)


make_fig([42, 95, 169, 135, 118, 435], "cue_period_raster_sorted_20260911_100049.png",
          "Session 20260911_100049 -- raster (sorted by decision time) + smoothed PSTH, aligned to cue end")
make_fig([256, 249, 252], "units_256_249_raster_psth_20260911_100049.png",
          "Session 20260911_100049 -- units 256, 249 & 252, raster (sorted by decision time) + smoothed PSTH, aligned to cue end")
