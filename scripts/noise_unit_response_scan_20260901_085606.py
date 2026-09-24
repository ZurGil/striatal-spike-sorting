"""
Same poke/reward population scan as population_response_scan_20260901_085606.py,
but for the 123 units Phy-labeled 'noise' (excluded from build_units_table /
the normal synced spikes.parquet entirely). Pulls their raw spike times
directly from Kilosort's own spike_times.npy/spike_clusters.npy in the
verdict_review working copy, converts to true absolute time with the SAME
fixed timing lookup as everything else this session (units_spikes.load_true_sample_timestamps),
then runs them through the identical sync_result.trodes_to_bpod() +
build_spikes_long_table() pipeline as real units, tagged as a synthetic
unit group so they don't touch the real units.parquet/spikes.parquet.

Usage: python noise_unit_response_scan_20260901_085606.py
"""
import sys
sys.path.insert(0, r"D:\Gil")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from sync_pipeline import bpod_loader, dio_decoder, session_discovery, sync, units_spikes

REC_FOLDER = r"D:\Gil\Shamir\20260901_085606.rec"
BPOD_FILE = r"Z:\Gil\Shamir_1\bpod\Shamir01_Dual2AFC_nat_Sep01_2026_Session1.mat"
RAT_ROOT = r"Z:\Gil\Shamir_1"
VERDICT_REVIEW_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
SESSION_ID = "20260901_085606"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
BUFFER_S = 10.0
ALPHA = 0.05
POST_POKE_EXTRA_S = 0.2
REWARD_WINDOW_S = 1.0

paths = session_discovery.session_paths_from_rec_folder(Path(REC_FOLDER), Path(BPOD_FILE))

# --- 1. Bpod + sync (identical to every other script this session) ---
pkl_cache = Path(RAT_ROOT) / "bpod" / (Path(BPOD_FILE).stem + ".pkl")
session_data = bpod_loader.load_bpod_session(Path(BPOD_FILE), pkl_cache_path=pkl_cache)
bpod_stream = bpod_loader.build_bpod_event_stream(session_data)
trials = bpod_loader.build_trials_table(session_data)
state_events = bpod_loader.build_state_events_table(session_data)
trials["TrialCompleted"] = bpod_loader.compute_trial_completed(trials, state_events)

ttl_codes, ttl_ts = dio_decoder.extract_ttls(paths.dio_folder)
sync_result = sync.synchronize(bpod_stream.label, bpod_stream.time, ttl_codes, ttl_ts, config=sync.SyncConfig())
print(f"Sync: status={sync_result.sync_status} match_fraction={sync_result.match_fraction:.3f}")

next_starts = np.append(trials["trial_start_bpod"].to_numpy()[1:], np.nan)
trials["next_trial_start_bpod"] = next_starts
trials["trial_end_bpod"] = [
    units_spikes.default_trial_window(s, (ns if not np.isnan(ns) else None), le, BUFFER_S)[1]
    for s, ns, le in zip(trials["trial_start_bpod"], next_starts, trials["trial_last_state_end_bpod"])
]
trials["sync_valid"] = [
    not sync.trial_overlaps_long_gap(row.trial_start_bpod, row.trial_end_bpod, sync_result.gaps)
    for row in trials.itertuples()
]

# --- 2. Noise units' raw spikes, TIME-CORRECTED via the same fixed lookup ---
info = pd.read_csv(f"{VERDICT_REVIEW_DIR}\\cluster_info.tsv", sep="\t")
noise_ids = set(info[info.group == "noise"].cluster_id.tolist())
print(f"{len(noise_ids)} noise-labeled units")

sample_rate_hz = units_spikes.read_kilosort_sample_rate(VERDICT_REVIEW_DIR)
spike_times_samples = np.load(f"{VERDICT_REVIEW_DIR}\\spike_times.npy").ravel()
spike_clusters = np.load(f"{VERDICT_REVIEW_DIR}\\spike_clusters.npy").ravel()
noise_mask = np.isin(spike_clusters, list(noise_ids))
spike_times_samples = spike_times_samples[noise_mask]
spike_clusters = spike_clusters[noise_mask]
print(f"{len(spike_times_samples)} raw spikes across noise units")

true_ts = units_spikes.load_true_sample_timestamps(VERDICT_REVIEW_DIR, SESSION_ID)
spike_times_ephys = true_ts[spike_times_samples].astype(np.float64) / sample_rate_hz

fake_units = pd.DataFrame({"unit_id": sorted(noise_ids)})
spikes = units_spikes.build_spikes_long_table(
    trials, fake_units, spike_times_ephys, spike_clusters, sync_result, buffer_s=BUFFER_S,
)
print(f"{len(spikes)} noise-unit spikes assigned across trials")

# --- 3. Same poke/reward test as the real-unit scan ---
baseline = (state_events[state_events.state_name == "stay_Cin"].sort_values(["trial_id", "start_time_in_trial"])
            .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
base_start, base_end = baseline["start_time_in_trial"], baseline["end_time_in_trial"]
wait_sin = (state_events[state_events.state_name == "wait_Sin"].sort_values(["trial_id", "start_time_in_trial"])
            .drop_duplicates("trial_id", keep="first").set_index("trial_id"))
cue_end, poke_time = wait_sin["start_time_in_trial"], wait_sin["end_time_in_trial"]
poke_resp_end = poke_time + POST_POKE_EXTRA_S
water = (state_events[state_events.state_name.isin(["water_L", "water_R"])]
         .sort_values(["trial_id", "start_time_in_trial"]).drop_duplicates("trial_id", keep="first")
         .set_index("trial_id"))
water_start = water["start_time_in_trial"]
water_resp_end = water_start + REWARD_WINDOW_S

poke_valid = trials[trials.TrialCompleted == 1].trial_id
poke_valid = poke_valid[poke_valid.isin(base_start.index) & poke_valid.isin(cue_end.index)].to_numpy()
reward_valid = trials[trials.Rewarded == 1].trial_id
reward_valid = reward_valid[reward_valid.isin(base_start.index) & reward_valid.isin(water_start.index)].to_numpy()
print(f"poke trials: {len(poke_valid)}, reward trials: {len(reward_valid)}")


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


by_unit = spikes.groupby("unit_id")
rows_poke, rows_reward = [], []
for uid in sorted(noise_ids):
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

df_poke = pd.DataFrame(rows_poke)
df_reward = pd.DataFrame(rows_reward)

for name, df in [("poke", df_poke), ("reward", df_reward)]:
    tested = df[df.p.notna()].copy()
    df["p_fdr"] = np.nan
    df["significant"] = False
    if len(tested) > 0:
        rej, p_fdr, _, _ = multipletests(tested.p.to_numpy(), alpha=ALPHA, method="fdr_bh")
        df.loc[tested.index, "p_fdr"] = p_fdr
        df.loc[tested.index, "significant"] = rej
    n_sig = int(df.significant.sum())
    print(f"NOISE units, {name}: {n_sig}/{len(tested)} tested FDR-significant (of {len(df)} total)")

df_poke.to_csv(rf"{OUT}\population_poke_response_NOISE_20260901_085606.csv", index=False)
df_reward.to_csv(rf"{OUT}\population_reward_response_NOISE_20260901_085606.csv", index=False)
spikes.to_parquet(rf"{OUT}\noise_unit_spikes_20260901_085606.parquet")
print("\nNOISE_SCAN_DONE")
