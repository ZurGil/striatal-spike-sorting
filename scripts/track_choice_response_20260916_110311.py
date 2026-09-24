"""
Aligns the 3 independently-detected tracks on raw channel 129 (Trodes 1337) --
track0/track2/track6 from independent_clustering_windowed_overlap_session.py,
now trial-aligned via run_sync_and_track_alignment_session.py into
independent_track_spikes_20260916_110311_ch129.parquet (synthetic unit_ids
90000/90002/90006) -- to stimulus end (wait_Sin state start), split by
left/right choice, same convention as left_right_cue_response_session.py /
build_cue_raster_figs.py for 20260911_100049:
  baseline: stay_Cin state
  response: wait_Sin start (cue end) -> wait_Sin end (choice poke) + 0.2s
  paired Wilcoxon signed-rank, baseline vs response rate, per side

track6 is only active for part of the session (~1203-5998s ephys, per the
sync/alignment run's own printout) -- trials outside that window will
naturally show no track6 spikes; this is expected, not a bug.
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNCED = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
TRACK_PARQUET = rf"{OUT}\independent_track_spikes_20260916_110311_ch129.parquet"

TRACK_LABEL = {90000: "track 0 (2.07Hz persistent)", 90002: "track 2 (1.76Hz persistent)", 90006: "track 6 (transient, ~min20-100)"}
POST_POKE_EXTRA_S = 0.2
MAX_DECISION_S = 3.0
XLIM = (-1.5, 3.0)
BIN_S = 0.05
BINS = np.arange(XLIM[0], XLIM[1] + 0.001, BIN_S)
CENTERS = (BINS[:-1] + BINS[1:]) / 2
SIGMA_BINS = 2

trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))
spikes = pd.read_parquet(TRACK_PARQUET)

baseline = (state_events[state_events.state_name == "stay_Cin"]
            .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
            .set_index("trial_id"))
base_start, base_end = baseline["start_time_in_trial"], baseline["end_time_in_trial"]

wait_sin = (state_events[state_events.state_name == "wait_Sin"]
            .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
            .set_index("trial_id"))
cue_end = wait_sin["start_time_in_trial"]
poke_time = wait_sin["end_time_in_trial"]
resp_end = poke_time + POST_POKE_EXTRA_S
decision_s = poke_time - cue_end

valid = trials[trials.TrialCompleted == 1].trial_id
valid = valid[valid.isin(base_start.index) & valid.isin(cue_end.index)]
valid = valid[decision_s.reindex(valid).to_numpy() < MAX_DECISION_S]
choice_left = trials.set_index("trial_id")["ChoiceLeft"].reindex(valid)
left_trials = valid[choice_left.to_numpy() == 1.0].to_numpy()
right_trials = valid[choice_left.to_numpy() == 0.0].to_numpy()
print(f"{len(valid)} valid completed trials (<{MAX_DECISION_S}s decision): {len(left_trials)} left, {len(right_trials)} right")

order = decision_s.reindex(valid).sort_values().index.to_numpy()
sorted_decision = decision_s.reindex(order).to_numpy()
sorted_side = choice_left.reindex(order).to_numpy()


def window_rate(usp_by_trial, trial_ids, t0s, t1s):
    rate = np.zeros(len(trial_ids))
    for i, tid in enumerate(trial_ids):
        dur = t1s.loc[tid] - t0s.loc[tid]
        if tid in usp_by_trial.groups:
            st = usp_by_trial.get_group(tid).to_numpy()
            n = int(((st >= t0s.loc[tid]) & (st < t1s.loc[tid])).sum())
        else:
            n = 0
        rate[i] = n / dur
    return rate


def test_side(usp_by_trial, trial_ids):
    if len(trial_ids) == 0:
        return None
    base_r = window_rate(usp_by_trial, trial_ids, base_start, base_end)
    resp_r = window_rate(usp_by_trial, trial_ids, cue_end, resp_end)
    try:
        _, p = wilcoxon(resp_r, base_r, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p = np.nan
    return dict(base_hz=base_r.mean(), resp_hz=resp_r.mean(), diff_hz=(resp_r - base_r).mean(),
                n=len(trial_ids), p=p)


fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True,
                          gridspec_kw={"height_ratios": [2.2, 1]})
print()
for col, uid in enumerate(sorted(TRACK_LABEL)):
    usp = spikes[spikes.unit_id == uid]
    by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
    L = test_side(by_trial, left_trials)
    R = test_side(by_trial, right_trials)
    print(f"{TRACK_LABEL[uid]} (unit_id {uid}):")
    print(f"  LEFT : base={L['base_hz']:.2f}Hz resp={L['resp_hz']:.2f}Hz diff={L['diff_hz']:+.2f}Hz p={L['p']:.4g} (n={L['n']} trials)")
    print(f"  RIGHT: base={R['base_hz']:.2f}Hz resp={R['resp_hz']:.2f}Hz diff={R['diff_hz']:+.2f}Hz p={R['p']:.4g} (n={R['n']} trials)")

    ax_r, ax_p = axes[0, col], axes[1, col]
    mat = np.zeros((len(order), len(CENTERS)))
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
    ax_r.set_title(f"{TRACK_LABEL[uid]}\nleft p={L['p']:.3g} ({L['diff_hz']:+.2f}Hz)  right p={R['p']:.3g} ({R['diff_hz']:+.2f}Hz)", fontsize=9)
    if col == 0:
        ax_r.set_ylabel("trial (sorted by decision time)")

    mean_l = mat[sorted_side == 1.0].mean(0) if (sorted_side == 1.0).any() else np.zeros_like(CENTERS)
    mean_r = mat[sorted_side == 0.0].mean(0) if (sorted_side == 0.0).any() else np.zeros_like(CENTERS)
    ax_p.plot(CENTERS, gaussian_filter1d(mean_l, sigma=SIGMA_BINS, mode="nearest"), color="#3d6a9f", lw=1.6, label="left")
    ax_p.plot(CENTERS, gaussian_filter1d(mean_r, sigma=SIGMA_BINS, mode="nearest"), color="#c07a2f", lw=1.6, label="right")
    ax_p.axvline(0, color="k", lw=0.8, ls="--")
    ax_p.set_xlabel("time from stimulus end (s)")
    if col == 0:
        ax_p.set_ylabel("firing rate (Hz)")
        ax_p.legend(fontsize=8, loc="upper left")

axes[0, 0].plot([], [], color="k", lw=1.0, label="choice poke (sorted)")
axes[0, 0].scatter([], [], c="#3d6a9f", marker="|", label="left-trial spike")
axes[0, 0].scatter([], [], c="#c07a2f", marker="|", label="right-trial spike")
axes[0, 0].legend(fontsize=7, loc="upper right")
plt.suptitle("Session 20260916_110311, raw channel 129 (Trodes 1337) -- independently-detected tracks,\n"
             "raster (sorted by decision time) + left/right smoothed PSTH, aligned to stimulus end")
plt.tight_layout()
fig_path = rf"{OUT}\track_choice_response_20260916_110311_ch129.png"
plt.savefig(fig_path, dpi=110, bbox_inches="tight")
print(f"\nsaved {fig_path}")
