"""
Population-wide scan: how many units (of 266 total, 122 good + 144 mua,
all Phy-curated) show a significant response to (a) the choice poke and
(b) reward delivery -- run against BOTH the old (buggy) and new (fixed)
synced data for session 20260901_085606, for a direct before/after
comparison of how many real responses the timing bug was hiding.

Event definitions:
  POKE:   baseline = stay_Cin (pre-stimulus fixation hold)
          response = wait_Sin start (stimulus end) -> poke-in + 0.2s
          trials   = TrialCompleted==1 (correct + error), pooled L+R
  REWARD: baseline = stay_Cin (same baseline, for a consistent reference)
          response = water_L or water_R start (whichever occurred) -> +1.0s
          trials   = Rewarded==1 trials only

Per-unit test: paired Wilcoxon signed-rank, baseline vs response firing
rate. FDR (Benjamini-Hochberg) applied separately per event x sync-version
across all 266 units.

Usage: python population_response_scan_20260901_085606.py
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

OUT = r"D:\Gil\spike_sorting_agent\outputs"
BASE = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
SYNC_DIRS = {"OLD": BASE + r"\synced", "NEW": BASE + r"\synced_NEWFIX_TEST"}
ALPHA = 0.05
POST_POKE_EXTRA_S = 0.2
REWARD_WINDOW_S = 1.0


def window_rate(by_trial, trial_ids, t0s, t1s):
    rate = np.zeros(len(trial_ids))
    for i, tid in enumerate(trial_ids):
        dur = t1s.loc[tid] - t0s.loc[tid]
        n = 0
        if tid in by_trial.groups:
            st = by_trial.get_group(tid).to_numpy()
            n = int(((st >= t0s.loc[tid]) & (st < t1s.loc[tid])).sum())
        rate[i] = n / dur if dur > 0 else np.nan
    return rate


def test_unit(by_trial, trial_ids, base_start, base_end, resp_start, resp_end):
    br = window_rate(by_trial, trial_ids, base_start, base_end)
    rr = window_rate(by_trial, trial_ids, resp_start, resp_end)
    ok = ~(np.isnan(br) | np.isnan(rr))
    br, rr = br[ok], rr[ok]
    if len(br) < 10 or (br == rr).all():
        return dict(n=len(br), base_hz=np.nan, resp_hz=np.nan, diff_hz=np.nan, p=np.nan)
    try:
        _, p = wilcoxon(rr, br, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p = np.nan
    return dict(n=len(br), base_hz=br.mean(), resp_hz=rr.mean(), diff_hz=(rr - br).mean(), p=p)


results = {}
for version, d in SYNC_DIRS.items():
    trials = pd.read_parquet(d + r"\trials.parquet")
    se = pd.read_parquet(d + r"\state_events.parquet")
    spikes = pd.read_parquet(d + r"\spikes.parquet")
    units = pd.read_parquet(d + r"\units.parquet")

    baseline = (se[se.state_name == "stay_Cin"].sort_values(["trial_id", "start_time_in_trial"])
                .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
    base_start, base_end = baseline["start_time_in_trial"], baseline["end_time_in_trial"]

    wait_sin = (se[se.state_name == "wait_Sin"].sort_values(["trial_id", "start_time_in_trial"])
                .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
    cue_end, poke_time = wait_sin["start_time_in_trial"], wait_sin["end_time_in_trial"]
    poke_resp_end = poke_time + POST_POKE_EXTRA_S

    water = (se[se.state_name.isin(["water_L", "water_R"])].sort_values(["trial_id", "start_time_in_trial"])
             .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
    water_start = water["start_time_in_trial"]
    water_resp_end = water_start + REWARD_WINDOW_S

    poke_valid = trials[trials.TrialCompleted == 1].trial_id
    poke_valid = poke_valid[poke_valid.isin(base_start.index) & poke_valid.isin(cue_end.index)].to_numpy()

    reward_valid = trials[trials.Rewarded == 1].trial_id
    reward_valid = reward_valid[reward_valid.isin(base_start.index) & reward_valid.isin(water_start.index)].to_numpy()

    print(f"[{version}] poke trials: {len(poke_valid)}, reward trials: {len(reward_valid)}")

    print(f"[{version}] grouping {len(spikes)} spikes by unit (one pass)...")
    by_unit = spikes.groupby("unit_id")

    rows_poke, rows_reward = [], []
    for uid in units.unit_id:
        if uid not in by_unit.groups:
            rows_poke.append(dict(n=0, base_hz=np.nan, resp_hz=np.nan, diff_hz=np.nan, p=np.nan, unit_id=uid))
            rows_reward.append(dict(n=0, base_hz=np.nan, resp_hz=np.nan, diff_hz=np.nan, p=np.nan, unit_id=uid))
            continue
        usp = by_unit.get_group(uid)
        by_trial = usp.groupby("trial_id")["spike_time_in_trial"]
        r_poke = test_unit(by_trial, poke_valid, base_start, base_end, cue_end, poke_resp_end)
        r_poke["unit_id"] = uid
        rows_poke.append(r_poke)
        r_rew = test_unit(by_trial, reward_valid, base_start, base_end, water_start, water_resp_end)
        r_rew["unit_id"] = uid
        rows_reward.append(r_rew)

    df_poke = pd.DataFrame(rows_poke).merge(units[["unit_id", "quality_label"]], on="unit_id")
    df_reward = pd.DataFrame(rows_reward).merge(units[["unit_id", "quality_label"]], on="unit_id")

    for name, df in [("poke", df_poke), ("reward", df_reward)]:
        tested = df[df.p.notna()].copy()
        df["p_fdr"] = np.nan
        df["significant"] = False
        if len(tested) > 0:
            rej, p_fdr, _, _ = multipletests(tested.p.to_numpy(), alpha=ALPHA, method="fdr_bh")
            df.loc[tested.index, "p_fdr"] = p_fdr
            df.loc[tested.index, "significant"] = rej
        n_sig = int(df.significant.sum())
        n_tested = len(tested)
        print(f"[{version}] {name}: {n_sig}/{n_tested} tested units FDR-significant (of {len(df)} total)")

    df_poke.to_csv(rf"{OUT}\population_poke_response_{version}_20260901_085606.csv", index=False)
    df_reward.to_csv(rf"{OUT}\population_reward_response_{version}_20260901_085606.csv", index=False)
    results[version] = dict(poke=df_poke, reward=df_reward, spikes=spikes, trials=trials, se=se,
                             cue_end=cue_end, poke_time=poke_time, water_start=water_start,
                             poke_valid=poke_valid, reward_valid=reward_valid)

print("\nPOPULATION_SCAN_DONE")
import pickle
with open(rf"{OUT}\population_scan_cache_20260901_085606.pkl", "wb") as f:
    pickle.dump(results, f)
