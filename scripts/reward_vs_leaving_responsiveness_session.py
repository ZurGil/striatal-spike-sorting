"""
Per-neuron reward-vs-choice-poke (port-leaving) responsiveness test, for one
session's synced data. FRESH RECONSTRUCTION -- the original script from the
20260901_085606 analysis (which found 6 leading units: 28, 135, 401, 182, 5,
232) was never saved anywhere reusable and could not be recovered; this is a
new implementation following the same documented methodology (see
WORKFLOW.md / conversation history), not a byte-for-byte match to the
original code. If the original script ever turns up, prefer it.

Design:
  - Two conditions per GOOD unit:
      "reward"  -- rewarded trials, spike count in a window after the
                   water_L/water_R state (actual reward delivery).
      "leaving" -- non-rewarded but completed trials (the rat poked a side
                   port, got no reward), spike count in the same-length
                   window after the LAST recorded poke-Out event for the
                   chosen side port within that trial (i.e. actual physical
                   port-leaving, not a Bpod state transition -- grace-period
                   re-entries show as extra In/Out pairs, so we take the
                   final Out as "left for good").
  - WINDOW_S = 0.5s post-event spike count (short, outcome-locked response
    window -- phasic reward/outcome responses are typically well within
    this).
  - Trial-consistency filters (established methodology, both conditions
    independently, per unit):
      MIN_SPIKES = 20   (total spikes across all trials of that condition)
      MIN_RESP_TRIALS = 15  (trials with >=1 spike in the window)
      MAX_TOP2_SHARE = 0.40 (top-2 trials' spike share of the condition's
                              total spikes must not exceed this -- guards
                              against a handful of outlier trials driving
                              a "significant" result)
  - Mann-Whitney U test (two-sided) on per-trial spike counts, reward vs
    leaving, per unit that passes both conditions' filters.
  - Benjamini-Hochberg FDR correction across all tested units.

Usage: python reward_vs_leaving_responsiveness_session.py <session_id> <rec_root>
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

WINDOW_S = 0.5
MIN_SPIKES = 20
MIN_RESP_TRIALS = 15
MAX_TOP2_SHARE = 0.40
ALPHA = 0.05


def trial_consistency_ok(spike_counts: np.ndarray) -> bool:
    total = spike_counts.sum()
    if total < MIN_SPIKES:
        return False
    n_resp = int((spike_counts > 0).sum())
    if n_resp < MIN_RESP_TRIALS:
        return False
    top2 = np.sort(spike_counts)[-2:].sum()
    if total > 0 and (top2 / total) > MAX_TOP2_SHARE:
        return False
    return True


def main():
    trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
    units = pd.read_parquet(os.path.join(SYNCED, "units.parquet"))
    spikes = pd.read_parquet(os.path.join(SYNCED, "spikes.parquet"))
    state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))
    poke_events = pd.read_parquet(os.path.join(SYNCED, "poke_events.parquet"))

    good_unit_ids = units.loc[units.quality_label == "good", "unit_id"].tolist()
    print(f"{len(good_unit_ids)} good units")

    # --- reward event per rewarded trial: water_L / water_R state start ---
    water = state_events[state_events.state_name.isin(["water_L", "water_R"])]
    water = water.sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
    reward_time = water.set_index("trial_id")["start_time_in_trial"]

    # --- port-leaving event per non-rewarded, completed trial ---
    # BUG FIX (2026-09-16, user caught it): the unbounded "last Out event in
    # the whole trial" picked up spurious pokes from the ITI / next trial,
    # since a trial's spike/event window runs a fixed buffer_s (10s) past its
    # last state and commonly overlaps the next trial entirely (see
    # HOW_TO_RUN.md sec 7). Also confirmed via TASK_TIMELINE.md + a manual
    # trace of individual trials that "non-rewarded completed" trials are NOT
    # all "chose the wrong side, state name unrewarded_Lin/Rin" -- some are a
    # CORRECT-side choice (state rewarded_Lin/Rin) where the rat simply left
    # too early and forfeited the reward before it was delivered. Both cases
    # converge on the same 'ITI' state afterward, so bounding the search to
    # "before this trial's own ITI state begins" is the correct, general fix
    # (not "before skipped_feedback", which happens to cover every trial in
    # this session's data but isn't guaranteed for a session that also hits
    # timeOut_IncorrectChoice, per TASK_TIMELINE.md step 12).
    choice_port = trials.set_index("trial_id")["ChoiceLeft"].map({1.0: "left", 0.0: "right"})
    non_rewarded_completed = trials[(trials.Rewarded == 0) & (trials.TrialCompleted == 1)].trial_id.tolist()

    iti_start = (state_events[state_events.state_name == "ITI"]
                 .sort_values(["trial_id", "start_time_in_trial"])
                 .drop_duplicates("trial_id", keep="first")
                 .set_index("trial_id")["start_time_in_trial"])

    leave_time = {}
    poke_by_trial = poke_events[poke_events.event_type == "Out"].groupby("trial_id")
    for tid in non_rewarded_completed:
        port = choice_port.get(tid)
        if port not in ("left", "right"):
            continue
        if tid not in poke_by_trial.groups or tid not in iti_start.index:
            continue
        rows = poke_by_trial.get_group(tid)
        rows = rows[(rows.port_role == port) & (rows.time_in_trial < iti_start.loc[tid])]
        if len(rows) == 0:
            continue
        leave_time[tid] = rows.time_in_trial.max()  # last Out BEFORE this trial's own ITI = final departure
    leave_time = pd.Series(leave_time)

    reward_trials = reward_time.index.to_numpy()
    leaving_trials = leave_time.index.to_numpy()
    print(f"{len(reward_trials)} rewarded trials with a water event, "
          f"{len(leaving_trials)} non-rewarded completed trials with a port-leaving time")

    spikes_by_unit = {uid: g for uid, g in spikes[spikes.unit_id.isin(good_unit_ids)].groupby("unit_id")}

    def window_counts(unit_spikes: pd.DataFrame, trial_ids: np.ndarray, event_time: pd.Series) -> np.ndarray:
        counts = np.zeros(len(trial_ids), dtype=int)
        if unit_spikes is None or len(unit_spikes) == 0:
            return counts
        by_trial = unit_spikes.groupby("trial_id")["spike_time_in_trial"]
        for i, tid in enumerate(trial_ids):
            if tid not in by_trial.groups:
                continue
            t0 = event_time.loc[tid]
            st = by_trial.get_group(tid).to_numpy()
            counts[i] = int(((st >= t0) & (st < t0 + WINDOW_S)).sum())
        return counts

    results = []
    for uid in good_unit_ids:
        usp = spikes_by_unit.get(uid)
        reward_counts = window_counts(usp, reward_trials, reward_time)
        leaving_counts = window_counts(usp, leaving_trials, leave_time)

        reward_ok = trial_consistency_ok(reward_counts)
        leaving_ok = trial_consistency_ok(leaving_counts)
        if not (reward_ok and leaving_ok):
            results.append(dict(unit_id=uid, tested=False, reward_ok=reward_ok, leaving_ok=leaving_ok,
                                 n_reward_trials=len(reward_counts), n_leaving_trials=len(leaving_counts),
                                 reward_total_spikes=int(reward_counts.sum()), leaving_total_spikes=int(leaving_counts.sum()),
                                 p_value=np.nan, reward_mean=np.nan, leaving_mean=np.nan))
            continue

        stat, p = mannwhitneyu(reward_counts, leaving_counts, alternative="two-sided")
        results.append(dict(
            unit_id=uid, tested=True, reward_ok=True, leaving_ok=True,
            n_reward_trials=len(reward_counts), n_leaving_trials=len(leaving_counts),
            reward_total_spikes=int(reward_counts.sum()), leaving_total_spikes=int(leaving_counts.sum()),
            p_value=p, reward_mean=float(reward_counts.mean()), leaving_mean=float(leaving_counts.mean()),
        ))

    df = pd.DataFrame(results)
    tested = df[df.tested].copy()
    if len(tested) > 0:
        rej, p_fdr, _, _ = multipletests(tested.p_value.to_numpy(), alpha=ALPHA, method="fdr_bh")
        tested["p_fdr"] = p_fdr
        tested["significant"] = rej
        df = df.merge(tested[["unit_id", "p_fdr", "significant"]], on="unit_id", how="left")
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False

    out_path = rf"{OUT}\reward_vs_leaving_responsiveness_{SESSION_ID}.csv"
    df.sort_values("p_fdr").to_csv(out_path, index=False)

    n_tested = int(df.tested.sum())
    n_sig = int(df.significant.fillna(False).sum())
    print(f"\n{n_tested}/{len(good_unit_ids)} good units passed trial-consistency filters and were tested")
    print(f"{n_sig} FDR-significant (alpha={ALPHA})")
    print(f"\nwrote {out_path}")
    if n_sig > 0:
        sig = df[df.significant.fillna(False)].sort_values("p_fdr")
        print("\nleading units (FDR-significant, sorted by p_fdr):")
        print(sig[["unit_id", "p_fdr", "reward_mean", "leaving_mean", "n_reward_trials", "n_leaving_trials"]].to_string(index=False))


if __name__ == "__main__":
    main()
