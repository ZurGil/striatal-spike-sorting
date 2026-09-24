"""
Pure visual raster (no statistics) for each of the 3 independently-detected
tracks on raw channel 129 (Trodes 1337) -- track0/track2/track6, synthetic
unit_ids 90000/90002/90006 in independent_track_spikes_20260916_110311_ch129.parquet.

Two alignments, shown separately (2 rows): stimulus end (wait_Sin start) and
choice poke-in (wait_Sin end). Two columns: left-choice trials, right-choice
trials. ALL completed trials (correct + error) included -- TrialCompleted==1
already means "the rat poked a side port at all, regardless of reward
outcome" (see sync_pipeline process_session.py), so no additional
correct/error filtering needed. No decision-time cutoff applied this time
(user wants to see everything, not a filtered subset). Window -1.5 to +3s
around each alignment point. Sorted by decision time (poke_time - cue_end)
within each panel so the diagonal poke-time line is visible.

Usage: python track_raster_two_alignments_20260916_110311.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNCED = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
TRACK_PARQUET = rf"{OUT}\independent_track_spikes_20260916_110311_ch129.parquet"

TRACKS = {90000: "track 0 (higher-amplitude persistent, 236.7uV)",
          90002: "track 2 (lower-amplitude persistent, 161.9uV)",
          90006: "track 6 (transient, active ~min20-100)"}
XLIM = (-1.5, 3.0)

trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))
spikes = pd.read_parquet(TRACK_PARQUET)

wait_sin = (state_events[state_events.state_name == "wait_Sin"]
            .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
            .set_index("trial_id"))
cue_end = wait_sin["start_time_in_trial"]   # stimulus end
poke_in = wait_sin["end_time_in_trial"]     # choice poke-in (left or right, whichever happened)
decision_s = poke_in - cue_end

valid = trials[trials.TrialCompleted == 1].trial_id
valid = valid[valid.isin(wait_sin.index)]
choice_left = trials.set_index("trial_id")["ChoiceLeft"].reindex(valid)
left_trials = valid[choice_left.to_numpy() == 1.0].to_numpy()
right_trials = valid[choice_left.to_numpy() == 0.0].to_numpy()
print(f"{len(valid)} completed trials total (correct+error): {len(left_trials)} left, {len(right_trials)} right")

left_order = decision_s.reindex(left_trials).sort_values().index.to_numpy()
right_order = decision_s.reindex(right_trials).sort_values().index.to_numpy()


def plot_panel(ax, uid, trial_order, align_col, align_label):
    usp = spikes[spikes.unit_id == uid]
    by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
    for row_i, tid in enumerate(trial_order):
        t0 = align_col.loc[tid]
        if tid in by_trial.groups:
            st = by_trial.get_group(tid).to_numpy() - t0
            st_plot = st[(st >= XLIM[0]) & (st <= XLIM[1])]
            ax.scatter(st_plot, np.full_like(st_plot, row_i), s=3, c="#1f4e8c", marker="|", linewidths=0.6)
        # mark the OTHER alignment point's own offset too, for reference
        poke_offset = (poke_in.loc[tid] - cue_end.loc[tid]) if align_label == "stimulus end" else -(poke_in.loc[tid] - cue_end.loc[tid])
        ax.scatter([poke_offset], [row_i], s=4, c="red", marker="o", linewidths=0)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_xlim(*XLIM)
    ax.set_ylim(-1, len(trial_order))


for uid, label in TRACKS.items():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    plot_panel(axes[0, 0], uid, left_order, cue_end, "stimulus end")
    plot_panel(axes[0, 1], uid, right_order, cue_end, "stimulus end")
    plot_panel(axes[1, 0], uid, left_order, poke_in, "poke-in")
    plot_panel(axes[1, 1], uid, right_order, poke_in, "poke-in")

    axes[0, 0].set_title(f"LEFT choice (n={len(left_trials)})\naligned to stimulus end", fontsize=10)
    axes[0, 1].set_title(f"RIGHT choice (n={len(right_trials)})\naligned to stimulus end", fontsize=10)
    axes[1, 0].set_title("aligned to poke-in", fontsize=10)
    axes[1, 1].set_title("aligned to poke-in", fontsize=10)
    axes[0, 0].set_ylabel("trial (sorted by decision time)")
    axes[1, 0].set_ylabel("trial (sorted by decision time)")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[0, 0].scatter([], [], c="red", marker="o", label="poke-in time (top row) / stimulus-end time (bottom row)")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    plt.suptitle(f"Session 20260916_110311, ch129 (Trodes 1337) -- {label}\n"
                 f"ALL completed trials (correct+error), no filtering, raw raster only")
    plt.tight_layout()
    track_num = {90000: 0, 90002: 2, 90006: 6}[uid]
    fname = rf"{OUT}\raster_2align_track{track_num}_20260916_110311.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    print(f"saved {fname}")
