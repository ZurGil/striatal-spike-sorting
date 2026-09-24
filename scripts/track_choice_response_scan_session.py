"""
Left/right choice-response scan for a set of independently-detected tracks
on a given channel. Same Wilcoxon baseline-vs-response test as before.

============================================================================
FIX vs. the original version:
============================================================================
Original trial selection:
    valid = trials[trials.TrialCompleted == 1].trial_id
-- no sync_valid filter. TrialCompleted and sync_valid are independent
flags: a trial can be TrialCompleted=1 (the rat made a real choice-port
poke) AND sync_valid=False (that trial's window overlaps a confirmed sync
problem) at the same time. When that happens, build_spikes_long_table()
silently assigns that trial ZERO spikes -- but it was still being counted
here as a real data point with a manufactured zero count, adding noise (or
worse, a systematic downward bias if such trials cluster on one side) to
the statistical test.

Fixed below. Also prints how many trials this filter actually removes, so
you can see directly whether this was contributing anything on this
session (run_sync_and_track_alignment_session.py's own printed
"N/n_trials flagged sync_valid=False" line already tells you the session-
wide number; this prints the same restricted to just the trials being used
here).

NOTE: per the separate discussion on this same investigation, the window
positioning here (baseline = full stay_Cin period, response =
[cue_end, poke_time + 0.2s]) is unchanged from the original -- if you're
looking for a SHORT, phasic, poke-locked burst, that window dilutes it (it
spans the whole variable-length decision period, not a tight window around
the poke). If the response is instead large and sustained across the whole
post-cue period (as you described), the original window positioning is
less of a concern and the sync_valid fix below is the more relevant one.
If you want the tightened, poke-centered window version too, say so and
I'll add it here as well -- kept out for now so this stays a minimal,
single-purpose diff you can review easily.

Usage: python track_choice_response_scan_session.py <session_id> <rec_root> <chan> <unit_id1> [unit_id2] ...
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
CHAN = int(sys.argv[3])
UNIT_IDS = [int(x) for x in sys.argv[4:]]

SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
TRACK_PARQUET = rf"{OUT}\independent_track_spikes_{SESSION_ID}_ch{CHAN}.parquet"
POST_POKE_EXTRA_S = 0.2

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
cue_end, poke_time = wait_sin["start_time_in_trial"], wait_sin["end_time_in_trial"]
resp_end = poke_time + POST_POKE_EXTRA_S

# FIX: added "& trials.sync_valid" -- see module docstring.
_n_before = int((trials.TrialCompleted == 1).sum())
valid = trials[(trials.TrialCompleted == 1) & trials.sync_valid].trial_id
_n_after = len(valid[valid.isin(base_start.index) & valid.isin(cue_end.index)])
print(f"sync_valid filter: {_n_before} trials had TrialCompleted==1; "
      f"{_n_before - _n_after} of those also removed here for sync_valid==False "
      f"or missing state data, leaving {_n_after}.")

valid = valid[valid.isin(base_start.index) & valid.isin(cue_end.index)]
choice_left = trials.set_index("trial_id")["ChoiceLeft"].reindex(valid)
left_trials = valid[choice_left.to_numpy() == 1.0].to_numpy()
right_trials = valid[choice_left.to_numpy() == 0.0].to_numpy()
print(f"{len(valid)} completed, sync-valid trials: {len(left_trials)} left, {len(right_trials)} right\n")


def window_rate(usp_by_trial, trial_ids, t0s, t1s):
    rate = np.zeros(len(trial_ids))
    for i, tid in enumerate(trial_ids):
        dur = t1s.loc[tid] - t0s.loc[tid]
        n = 0
        if tid in usp_by_trial.groups:
            st = usp_by_trial.get_group(tid).to_numpy()
            n = int(((st >= t0s.loc[tid]) & (st < t1s.loc[tid])).sum())
        rate[i] = n / dur
    return rate


def test_side(usp_by_trial, trial_ids):
    base_r = window_rate(usp_by_trial, trial_ids, base_start, base_end)
    resp_r = window_rate(usp_by_trial, trial_ids, cue_end, resp_end)
    try:
        _, p = wilcoxon(resp_r, base_r, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p = np.nan
    return base_r.mean(), resp_r.mean(), (resp_r - base_r).mean(), p


for uid in UNIT_IDS:
    usp = spikes[spikes.unit_id == uid]
    by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
    bl, rl, dl, pl = test_side(by_trial, left_trials)
    br, rr, dr, pr = test_side(by_trial, right_trials)
    flag_l = " <-- p<0.05" if pl < 0.05 else ""
    flag_r = " <-- p<0.05" if pr < 0.05 else ""
    print(f"unit {uid}: LEFT  base={bl:.2f}Hz resp={rl:.2f}Hz diff={dl:+.2f}Hz p={pl:.4g}{flag_l}")
    print(f"          RIGHT base={br:.2f}Hz resp={rr:.2f}Hz diff={dr:+.2f}Hz p={pr:.4g}{flag_r}")
