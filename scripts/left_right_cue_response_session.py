"""
Per-neuron response-to-choice test, split by left/right, for one session's
synced data -- REPLACES the earlier left_right_choice_period_responsiveness_session.py
approach after user clarification: that script only tested whether firing
during the decision window DIFFERED between left- and right-choice trials,
which would miss a neuron that responds equally strongly on both sides (no
left-vs-right difference, but still a real "responds to choice" signal).

Design (per user spec, clarified 2026-09-16):
  - Only trials with TrialCompleted==1 (a real choice was made; this already
    excludes FixBroke/EarlyWithdrawal trials, which by definition never reach
    Step 9's choice poke) AND (choice poke time - cue-end time) < MAX_DECISION_S
    (default 3.0s -- explicit per user request; a no-op for this session,
    since the observed max is 1.11s, but kept as a real, documented filter
    for any future session where it might matter).
  - BASELINE window: the `stay_Cin` state (pre-cue center-fixation hold,
    immediately before the stimulus plays -- median 0.296s, this session).
  - RESPONSE window: `wait_Sin` state start (right after the cue ends) to
    `wait_Sin` end (the actual choice poke) + POST_POKE_EXTRA_S (default
    0.2s, "or right after the poke").
  - Both windows vary in duration per trial -> per-trial FIRING RATE
    (spikes/duration), not raw count, is what's compared.
  - TWO separate paired Wilcoxon signed-rank tests per neuron:
      "left"  -- baseline vs response rate, LEFT-choice trials only.
      "right" -- baseline vs response rate, RIGHT-choice trials only.
    Each family gets its own Benjamini-Hochberg FDR correction across all
    tested units (two independent multiple-comparisons problems, not one).
  - A neuron's final call: 'left' (left family significant only), 'right'
    (right family significant only), 'both' (both families significant),
    or '' (neither) -- this directly answers "responds to left choice, right
    choice, or both."
  - Runs over ALL non-noise units (good+mua), consistent with the other
    "all neurons" analyses this session.

Trial-consistency filters per neuron per family (on raw spike counts, both
windows): >=10 total spikes (baseline+response combined), >=8 trials with a
nonzero baseline/response difference, top-2-trials |diff| share <=40%.

Usage: python left_right_cue_response_session.py <session_id> <rec_root> [post_poke_extra_s] [max_decision_s]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
POST_POKE_EXTRA_S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
MAX_DECISION_S = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

MIN_SPIKES = 10
MIN_NONZERO_DIFF_TRIALS = 8
MAX_TOP2_ABS_DIFF_SHARE = 0.40
ALPHA = 0.05


def trial_consistency_ok(diff: np.ndarray, total_spikes: int) -> bool:
    if total_spikes < MIN_SPIKES:
        return False
    if int((diff != 0).sum()) < MIN_NONZERO_DIFF_TRIALS:
        return False
    abs_diff = np.abs(diff)
    if abs_diff.sum() > 0 and (np.sort(abs_diff)[-2:].sum() / abs_diff.sum()) > MAX_TOP2_ABS_DIFF_SHARE:
        return False
    return True


def main():
    units = pd.read_parquet(os.path.join(SYNCED, "units.parquet"))
    trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
    spikes = pd.read_parquet(os.path.join(SYNCED, "spikes.parquet"))
    state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))

    baseline = (state_events[state_events.state_name == "stay_Cin"]
                .sort_values(["trial_id", "start_time_in_trial"])
                .drop_duplicates("trial_id", keep="first")
                .set_index("trial_id"))
    base_start, base_end = baseline["start_time_in_trial"], baseline["end_time_in_trial"]

    wait_sin = (state_events[state_events.state_name == "wait_Sin"]
                .sort_values(["trial_id", "start_time_in_trial"])
                .drop_duplicates("trial_id", keep="first")
                .set_index("trial_id"))
    resp_start = wait_sin["start_time_in_trial"]
    poke_time = wait_sin["end_time_in_trial"]
    resp_end = poke_time + POST_POKE_EXTRA_S

    decision_s = poke_time - resp_start
    valid_trials = trials[trials.TrialCompleted == 1].trial_id
    valid_trials = valid_trials[valid_trials.isin(base_start.index) & valid_trials.isin(resp_start.index)]
    valid_trials = valid_trials[decision_s.reindex(valid_trials).to_numpy() < MAX_DECISION_S]

    choice_left = trials.set_index("trial_id")["ChoiceLeft"]
    left_trials = valid_trials[choice_left.reindex(valid_trials).to_numpy() == 1.0].to_numpy()
    right_trials = valid_trials[choice_left.reindex(valid_trials).to_numpy() == 0.0].to_numpy()
    print(f"{len(valid_trials)} valid completed trials (<{MAX_DECISION_S}s decision time): "
          f"{len(left_trials)} left, {len(right_trials)} right")
    print(f"baseline (stay_Cin) median duration {(base_end-base_start).median():.3f}s, "
          f"response window median duration {(resp_end-resp_start).median():.3f}s")

    all_unit_ids = units.unit_id.tolist()
    quality = units.set_index("unit_id")["quality_label"]
    print(f"{len(all_unit_ids)} total units ({(quality=='good').sum()} good, {(quality=='mua').sum()} mua)")

    spikes_by_unit = {uid: g for uid, g in spikes[spikes.unit_id.isin(all_unit_ids)].groupby("unit_id")}

    def window_count(usp, trial_ids, t0s, t1s):
        counts = np.zeros(len(trial_ids), dtype=int)
        if usp is None or len(usp) == 0:
            return counts
        by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
        for i, tid in enumerate(trial_ids):
            if tid not in by_trial.groups:
                continue
            st = by_trial.get_group(tid).to_numpy()
            counts[i] = int(((st >= t0s.loc[tid]) & (st < t1s.loc[tid])).sum())
        return counts

    def run_family(usp, trial_ids):
        if len(trial_ids) == 0:
            return None
        base_c = window_count(usp, trial_ids, base_start, base_end)
        resp_c = window_count(usp, trial_ids, resp_start, resp_end)
        base_rate = base_c / (base_end.loc[trial_ids].to_numpy() - base_start.loc[trial_ids].to_numpy())
        resp_rate = resp_c / (resp_end.loc[trial_ids].to_numpy() - resp_start.loc[trial_ids].to_numpy())
        diff = resp_c.astype(float) - base_c.astype(float)  # consistency check on raw counts
        ok = trial_consistency_ok(diff, int(base_c.sum() + resp_c.sum()))
        result = dict(ok=ok, base_mean_hz=float(base_rate.mean()), resp_mean_hz=float(resp_rate.mean()),
                      mean_diff_hz=float((resp_rate - base_rate).mean()), n_trials=len(trial_ids), p_value=np.nan)
        if ok:
            _, p = wilcoxon(resp_rate, base_rate, zero_method="wilcox", alternative="two-sided")
            result["p_value"] = p
        return result

    rows = []
    for uid in all_unit_ids:
        usp = spikes_by_unit.get(uid)
        L = run_family(usp, left_trials)
        R = run_family(usp, right_trials)
        rows.append(dict(
            unit_id=uid, quality_label=quality.loc[uid],
            left_tested=L["ok"], left_p=L["p_value"], left_base_hz=L["base_mean_hz"], left_resp_hz=L["resp_mean_hz"], left_diff_hz=L["mean_diff_hz"],
            right_tested=R["ok"], right_p=R["p_value"], right_base_hz=R["base_mean_hz"], right_resp_hz=R["resp_mean_hz"], right_diff_hz=R["mean_diff_hz"],
        ))
    df = pd.DataFrame(rows)

    for side in ("left", "right"):
        tested = df[df[f"{side}_tested"]].copy()
        df[f"{side}_p_fdr"] = np.nan
        df[f"{side}_significant"] = False
        if len(tested) > 0:
            rej, p_fdr, _, _ = multipletests(tested[f"{side}_p"].to_numpy(), alpha=ALPHA, method="fdr_bh")
            df.loc[tested.index, f"{side}_p_fdr"] = p_fdr
            df.loc[tested.index, f"{side}_significant"] = rej

    df["response"] = np.select(
        [df.left_significant & df.right_significant, df.left_significant, df.right_significant],
        ["both", "left", "right"], default="")

    out_path = rf"{OUT}\left_right_cue_response_{SESSION_ID}.csv"
    df.sort_values(["response", "left_p_fdr", "right_p_fdr"]).to_csv(out_path, index=False)

    n_left = int((df.response == "left").sum())
    n_right = int((df.response == "right").sum())
    n_both = int((df.response == "both").sum())
    n_left_tested = int(df.left_tested.sum())
    n_right_tested = int(df.right_tested.sum())
    print(f"\nleft family: {n_left_tested}/{len(all_unit_ids)} tested")
    print(f"right family: {n_right_tested}/{len(all_unit_ids)} tested")
    print(f"\nFDR-significant (alpha={ALPHA}): {n_left} left-only, {n_right} right-only, {n_both} both")
    print(f"\nwrote {out_path}")
    if n_left + n_right + n_both > 0:
        sig = df[df.response != ""].sort_values("response")
        print("\nsignificant units:")
        print(sig[["unit_id", "quality_label", "response", "left_p_fdr", "right_p_fdr",
                    "left_diff_hz", "right_diff_hz"]].to_string(index=False))


if __name__ == "__main__":
    main()
