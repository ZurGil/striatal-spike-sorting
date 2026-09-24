"""
Per-neuron pre- vs post-reward-delivery responsiveness test, within rewarded
trials only, for one session's synced data.

Design (per user spec):
  - Rewarded trials only (water_L/water_R delivery = the reward event, same
    event source as reward_vs_leaving_responsiveness_session.py).
  - PRE window:  [-600ms, -100ms) relative to reward delivery.
  - POST window: [+100ms, +600ms) relative to reward delivery.
  - Paired test per neuron: Wilcoxon signed-rank on (post_count - pre_count)
    across rewarded trials (paired because it's the same trial, same neuron,
    before vs after -- more powerful than an unpaired test here since it
    cancels out trial-to-trial excitability).
  - Runs over ALL units in the session (both 'good' and 'mua', quality_label
    kept in the output) -- 'noise' clusters are already excluded upstream by
    the sync pipeline itself, so "all neurons" here means every surviving
    Kilosort cluster, not just the 'good' subset used elsewhere.
  - Direction: sign of the mean (post - pre) difference -- 'increase' or
    'decrease' -- reported only for FDR-significant units.

Trial-consistency filters per neuron (same spirit as
reward_vs_leaving_responsiveness_session.py, adapted for a paired design):
  MIN_TOTAL_SPIKES = 10      (pre+post combined, across all rewarded trials)
  MIN_NONZERO_DIFF_TRIALS = 15  (Wilcoxon needs enough non-tied pairs to be
                                  meaningful; ties -- pre==post -- are
                                  dropped by scipy's default zero_method)
  MAX_TOP2_ABS_DIFF_SHARE = 0.40 (top-2 trials' |post-pre| must not exceed
                                   this share of the total |post-pre| sum --
                                   guards against a couple of outlier trials
                                   driving the result)

Usage: python reward_pre_post_responsiveness_session.py <session_id> <rec_root>
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
SYNCED = rf"{REC_ROOT}\{SESSION_ID}.kilosort\synced"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

PRE_LO, PRE_HI = -0.600, -0.100
POST_LO, POST_HI = 0.100, 0.600
MIN_TOTAL_SPIKES = 10
MIN_NONZERO_DIFF_TRIALS = 15
MAX_TOP2_ABS_DIFF_SHARE = 0.40
ALPHA = 0.05


def main():
    units = pd.read_parquet(os.path.join(SYNCED, "units.parquet"))
    spikes = pd.read_parquet(os.path.join(SYNCED, "spikes.parquet"))
    state_events = pd.read_parquet(os.path.join(SYNCED, "state_events.parquet"))

    water = state_events[state_events.state_name.isin(["water_L", "water_R"])]
    water = water.sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
    reward_time = water.set_index("trial_id")["start_time_in_trial"]
    reward_trials = reward_time.index.to_numpy()
    print(f"{len(reward_trials)} rewarded trials")

    all_unit_ids = units.unit_id.tolist()
    quality = units.set_index("unit_id")["quality_label"]
    print(f"{len(all_unit_ids)} total units ({(quality=='good').sum()} good, {(quality=='mua').sum()} mua)")

    spikes_by_unit = {uid: g for uid, g in spikes[spikes.unit_id.isin(all_unit_ids)].groupby("unit_id")}

    results = []
    for uid in all_unit_ids:
        usp = spikes_by_unit.get(uid)
        pre_counts = np.zeros(len(reward_trials), dtype=int)
        post_counts = np.zeros(len(reward_trials), dtype=int)
        if usp is not None and len(usp) > 0:
            by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
            for i, tid in enumerate(reward_trials):
                if tid not in by_trial.groups:
                    continue
                st = by_trial.get_group(tid).to_numpy() - reward_time.loc[tid]
                pre_counts[i] = int(((st >= PRE_LO) & (st < PRE_HI)).sum())
                post_counts[i] = int(((st >= POST_LO) & (st < POST_HI)).sum())

        diff = post_counts - pre_counts
        total_spikes = int(pre_counts.sum() + post_counts.sum())
        nonzero_trials = int((diff != 0).sum())
        abs_diff = np.abs(diff)
        top2_share = (np.sort(abs_diff)[-2:].sum() / abs_diff.sum()) if abs_diff.sum() > 0 else 1.0

        ok = (total_spikes >= MIN_TOTAL_SPIKES and nonzero_trials >= MIN_NONZERO_DIFF_TRIALS
              and top2_share <= MAX_TOP2_ABS_DIFF_SHARE)

        row = dict(unit_id=uid, quality_label=quality.loc[uid], tested=ok,
                   n_rewarded_trials=len(reward_trials), total_spikes=total_spikes,
                   nonzero_diff_trials=nonzero_trials, top2_abs_diff_share=round(float(top2_share), 3),
                   pre_mean=float(pre_counts.mean()), post_mean=float(post_counts.mean()),
                   mean_diff=float(diff.mean()), p_value=np.nan)
        if ok:
            stat, p = wilcoxon(post_counts, pre_counts, zero_method="wilcox", alternative="two-sided")
            row["p_value"] = p
        results.append(row)

    df = pd.DataFrame(results)
    tested = df[df.tested].copy()
    if len(tested) > 0:
        rej, p_fdr, _, _ = multipletests(tested.p_value.to_numpy(), alpha=ALPHA, method="fdr_bh")
        tested["p_fdr"] = p_fdr
        tested["significant"] = rej
        tested["direction"] = np.where(tested.significant & (tested.mean_diff > 0), "increase",
                                 np.where(tested.significant & (tested.mean_diff < 0), "decrease", ""))
        df = df.merge(tested[["unit_id", "p_fdr", "significant", "direction"]], on="unit_id", how="left")
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False
        df["direction"] = ""

    out_path = rf"{OUT}\reward_pre_post_responsiveness_{SESSION_ID}.csv"
    df.sort_values("p_fdr").to_csv(out_path, index=False)

    n_tested = int(df.tested.sum())
    n_sig = int(df.significant.fillna(False).sum())
    n_inc = int((df.direction == "increase").sum())
    n_dec = int((df.direction == "decrease").sum())
    print(f"\n{n_tested}/{len(all_unit_ids)} units passed trial-consistency filters and were tested")
    print(f"{n_sig} FDR-significant (alpha={ALPHA}): {n_inc} increase, {n_dec} decrease")
    print(f"\nwrote {out_path}")
    if n_sig > 0:
        sig = df[df.significant.fillna(False)].sort_values("p_fdr")
        print("\nsignificant units:")
        print(sig[["unit_id", "quality_label", "direction", "p_fdr", "pre_mean", "post_mean", "n_rewarded_trials"]].to_string(index=False))


if __name__ == "__main__":
    main()
