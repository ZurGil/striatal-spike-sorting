"""
Per-neuron left- vs right-choice responsiveness during the post-cue,
pre-choice-poke movement/decision period, for one session's synced data.

Window definition (per user spec, mapped onto this task's actual Bpod state
names -- see TASK_TIMELINE.md): starts right when the cue ends (`wait_Sin`
state start -- entered immediately after `stimulus_delivery` finishes, i.e.
"right after cue delivery end") and runs until POST_POKE_EXTRA_S past the
moment the rat actually pokes a side port (`wait_Sin`'s own end = the instant
`start_Lin`/`start_Rin` begins, i.e. "until the actual choice poke ... or
slightly later"). This window's duration is NOT fixed -- it's exactly how
long that trial's movement/decision took (173ms-1.1s this session, median
266ms) -- so per-trial FIRING RATE (spikes / window duration) is used for the
statistical test, not raw spike count, to avoid confounding "responds more"
with "had a longer window."

Only TrialCompleted==1 trials have a wait_Sin/start_Lin/start_Rin at all
(565/1023 here). Runs over ALL non-noise units (good+mua), same as the
pre/post-reward analysis, per the same "check all neurons" precedent.

Test: Mann-Whitney U (two-sided) on per-trial firing rate, left-choice trials
vs right-choice trials. Trial-consistency filters (on raw spike COUNTS, not
rates, since a near-zero-duration window could otherwise produce an inflated
rate from a single spike): per side, >=15 total spikes, >=10 responsive
trials, top-2-trials spike share <=40%. Benjamini-Hochberg FDR across all
tested units.

Usage: python left_right_choice_period_responsiveness_session.py <session_id> <rec_root> [post_poke_extra_s]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
POST_POKE_EXTRA_S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

MIN_SPIKES = 15
MIN_RESP_TRIALS = 10
MAX_TOP2_SHARE = 0.40
ALPHA = 0.05


def trial_consistency_ok(counts: np.ndarray) -> bool:
    total = counts.sum()
    if total < MIN_SPIKES:
        return False
    if int((counts > 0).sum()) < MIN_RESP_TRIALS:
        return False
    top2 = np.sort(counts)[-2:].sum()
    if total > 0 and (top2 / total) > MAX_TOP2_SHARE:
        return False
    return True


def main():
    units = pd.read_parquet(os.path.join(SYNCED, "units.parquet"))
    trials = pd.read_parquet(os.path.join(SYNCED, "trials.parquet"))
    spikes = pd.read_parquet(os.path.join(SYNCED, "spikes.parquet"))
    state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))

    wait_sin = (state_events[state_events.state_name == "wait_Sin"]
                .sort_values(["trial_id", "start_time_in_trial"])
                .drop_duplicates("trial_id", keep="first")
                .set_index("trial_id"))
    win_start = wait_sin["start_time_in_trial"]          # right after cue ends
    poke_time = wait_sin["end_time_in_trial"]             # the actual choice poke
    win_end = poke_time + POST_POKE_EXTRA_S               # ... or slightly later

    choice_left = trials.set_index("trial_id")["ChoiceLeft"]
    left_trials = choice_left[(choice_left == 1.0) & choice_left.index.isin(win_start.index)].index.to_numpy()
    right_trials = choice_left[(choice_left == 0.0) & choice_left.index.isin(win_start.index)].index.to_numpy()
    print(f"{len(win_start)} completed trials with a cue-end-to-choice-poke window "
          f"({len(left_trials)} left, {len(right_trials)} right)")
    print(f"window duration: median {(win_end-win_start).median():.3f}s "
          f"(cue-end to poke+{POST_POKE_EXTRA_S:.2f}s)")

    all_unit_ids = units.unit_id.tolist()
    quality = units.set_index("unit_id")["quality_label"]
    print(f"{len(all_unit_ids)} total units ({(quality=='good').sum()} good, {(quality=='mua').sum()} mua)")

    spikes_by_unit = {uid: g for uid, g in spikes[spikes.unit_id.isin(all_unit_ids)].groupby("unit_id")}

    def counts_and_rates(usp, trial_ids):
        counts = np.zeros(len(trial_ids), dtype=int)
        durs = (win_end.loc[trial_ids] - win_start.loc[trial_ids]).to_numpy()
        if usp is not None and len(usp) > 0:
            by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
            for i, tid in enumerate(trial_ids):
                if tid not in by_trial.groups:
                    continue
                st = by_trial.get_group(tid).to_numpy()
                t0, t1 = win_start.loc[tid], win_end.loc[tid]
                counts[i] = int(((st >= t0) & (st < t1)).sum())
        rates = counts / durs
        return counts, rates

    results = []
    for uid in all_unit_ids:
        usp = spikes_by_unit.get(uid)
        left_counts, left_rates = counts_and_rates(usp, left_trials)
        right_counts, right_rates = counts_and_rates(usp, right_trials)

        left_ok = trial_consistency_ok(left_counts)
        right_ok = trial_consistency_ok(right_counts)
        row = dict(unit_id=uid, quality_label=quality.loc[uid], tested=(left_ok and right_ok),
                   n_left_trials=len(left_trials), n_right_trials=len(right_trials),
                   left_total_spikes=int(left_counts.sum()), right_total_spikes=int(right_counts.sum()),
                   left_mean_hz=float(left_rates.mean()), right_mean_hz=float(right_rates.mean()),
                   p_value=np.nan)
        if left_ok and right_ok:
            stat, p = mannwhitneyu(left_rates, right_rates, alternative="two-sided")
            row["p_value"] = p
        results.append(row)

    df = pd.DataFrame(results)
    tested = df[df.tested].copy()
    if len(tested) > 0:
        rej, p_fdr, _, _ = multipletests(tested.p_value.to_numpy(), alpha=ALPHA, method="fdr_bh")
        tested["p_fdr"] = p_fdr
        tested["significant"] = rej
        tested["prefers"] = np.where(tested.significant & (tested.left_mean_hz > tested.right_mean_hz), "left",
                               np.where(tested.significant & (tested.right_mean_hz > tested.left_mean_hz), "right", ""))
        df = df.merge(tested[["unit_id", "p_fdr", "significant", "prefers"]], on="unit_id", how="left")
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False
        df["prefers"] = ""

    out_path = rf"{OUT}\left_right_choice_period_responsiveness_{SESSION_ID}.csv"
    df.sort_values("p_fdr").to_csv(out_path, index=False)

    n_tested = int(df.tested.sum())
    n_sig = int(df.significant.fillna(False).sum())
    n_left = int((df.prefers == "left").sum())
    n_right = int((df.prefers == "right").sum())
    print(f"\n{n_tested}/{len(all_unit_ids)} units passed trial-consistency filters and were tested")
    print(f"{n_sig} FDR-significant (alpha={ALPHA}): {n_left} prefer left, {n_right} prefer right")
    print(f"\nwrote {out_path}")
    if n_sig > 0:
        sig = df[df.significant.fillna(False)].sort_values("p_fdr")
        print("\nsignificant units:")
        print(sig[["unit_id", "quality_label", "prefers", "p_fdr", "left_mean_hz", "right_mean_hz"]].to_string(index=False))


if __name__ == "__main__":
    main()
