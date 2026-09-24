"""
Runs the sync pipeline for a session against its STANDARD (pre-Phy) kilosort4
folder -- same pattern as 20260911_100049's preliminary sync (WORKFLOW.md
2026-09-16 entry), using build_cluster_info_from_classification_session.py's
fabricated cluster_info.tsv since no Phy pass has happened yet.

Mirrors sync_pipeline.process_session.process_session()'s internals directly
(rather than calling it) so `sync_result` stays in scope afterward -- needed
to align spikes that are NOT part of Kilosort's own units table at all: the
independently-detected tracks from independent_clustering_windowed_overlap_session.py
(track0/track2/track6 on raw channel 129 / Trodes 1337). Each independent
track's spike times (currently raw sample indices from probe1.dat) are
converted to ephys-clock seconds (/sample_rate) -- exactly what
units_spikes.load_spike_arrays() does for real Kilosort units -- then run
through the SAME units_spikes.build_spikes_long_table() as a set of synthetic
extra "units" (IDs 90000/90002/90006 for track0/2/6), giving trial-aligned
spike_time_in_trial values directly comparable to real units and to
state_events' start_time_in_trial.

Usage: python run_sync_and_track_alignment_session.py <session_id> <rec_root> <rat_root> <bpod_file>
"""
import sys, os
sys.path.insert(0, r"D:\Gil")
import numpy as np
import pandas as pd
from pathlib import Path

from sync_pipeline import bpod_loader, dio_decoder, session_discovery, sync, units_spikes
from sync_pipeline.io_utils import save_json, save_table
from sync_pipeline.rat_metadata import find_rat_metadata, load_rat_metadata

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]          # e.g. F:\Gil\Shamir\20260916_110311.rec
RAT_ROOT = sys.argv[3]          # e.g. Z:\Gil\Shamir_1
BPOD_FILE = sys.argv[4]

OUT = r"D:\Gil\spike_sorting_agent\outputs"
BUFFER_S = 10.0
POKE_IN_STATE = "stay_Cin"

TRACK_NPZ = rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch129.npz"
TRACK_IDS = {0: 90000, 2: 90002, 6: 90006}  # local track id -> synthetic global unit_id


def main():
    rec_folder = Path(REC_ROOT)
    rat_root = Path(RAT_ROOT)
    bpod_file = Path(BPOD_FILE)

    paths = session_discovery.session_paths_from_rec_folder(rec_folder, bpod_file)
    rat_meta_path = find_rat_metadata(rat_root)
    rat_meta = load_rat_metadata(rat_meta_path)

    output_dir = paths.kilosort4_folder.parent / "synced"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Processing session {rec_folder.name} -> {output_dir} ===")

    # --- 1. Bpod ---
    pkl_cache = rat_root / "bpod" / (bpod_file.stem + ".pkl")
    session_data = bpod_loader.load_bpod_session(bpod_file, pkl_cache_path=pkl_cache)
    n_trials = int(session_data["nTrials"])
    print(f"Bpod: nTrials={n_trials}")

    bpod_stream = bpod_loader.build_bpod_event_stream(session_data)
    trials = bpod_loader.build_trials_table(session_data)
    gui_varying, gui_constant = bpod_loader.extract_gui_settings(session_data, set(trials.columns))
    if len(gui_varying.columns) > 1:
        trials = trials.merge(gui_varying, on="trial_id", how="left")
    state_events = bpod_loader.build_state_events_table(session_data)
    poke_events = bpod_loader.build_poke_events_table(session_data)
    port_roles = bpod_loader.infer_port_roles(state_events, poke_events)
    role_by_port = {v["port"]: role for role, v in port_roles.items()}
    poke_events["port_role"] = poke_events["port"].map(role_by_port).fillna("unknown")
    trials["TrialCompleted"] = bpod_loader.compute_trial_completed(trials, state_events)
    n_completed = int(trials["TrialCompleted"].sum())
    print(f"Bpod: {n_completed}/{n_trials} TrialCompleted, port_roles={ {r: v['port'] for r, v in port_roles.items()} }")

    # --- 2. Trodes DIO ---
    ttl_codes, ttl_ts = dio_decoder.extract_ttls(paths.dio_folder)
    print(f"Trodes: decoded {len(ttl_codes)} TTL events")

    # --- 3. Synchronize ---
    sync_result = sync.synchronize(bpod_stream.label, bpod_stream.time, ttl_codes, ttl_ts, config=sync.SyncConfig())
    print(f"Sync: status={sync_result.sync_status} n_matched={sync_result.n_matched}/"
          f"{min(sync_result.n_bpod_events, sync_result.n_trodes_events)} "
          f"match_fraction={sync_result.match_fraction:.3f} "
          f"global_fit_median_resid={sync_result.median_residual_s*1000:.3f}ms "
          f"interval_diff_corr={sync_result.interval_diff_correlation:.6f}")
    if sync_result.interp_holdout_median_ms is not None:
        print(f"Sync: local-interp held-out error median={sync_result.interp_holdout_median_ms:.3f}ms "
              f"max={sync_result.interp_holdout_max_ms:.3f}ms")

    sync.plot_sync_qc(bpod_stream.label, bpod_stream.time, ttl_codes, ttl_ts, sync_result, output_dir / "sync_qc.png")

    # --- 4. Trial windows + sync_valid ---
    next_starts = np.append(trials["trial_start_bpod"].to_numpy()[1:], np.nan)
    trials["next_trial_start_bpod"] = next_starts
    trials["trial_end_bpod"] = [
        units_spikes.default_trial_window(s, (ns if not np.isnan(ns) else None), le, BUFFER_S)[1]
        for s, ns, le in zip(trials["trial_start_bpod"], next_starts, trials["trial_last_state_end_bpod"])
    ]
    trials["window_overlaps_next_trial"] = (
        trials["next_trial_start_bpod"].notna() & (trials["trial_end_bpod"] > trials["next_trial_start_bpod"])
    )
    next_poke = bpod_loader.compute_next_poke_in(trials, state_events, POKE_IN_STATE)
    trials = trials.merge(next_poke, on="trial_id", how="left")
    trials["sync_valid"] = [
        not sync.trial_overlaps_long_gap(row.trial_start_bpod, row.trial_end_bpod, sync_result.gaps)
        for row in trials.itertuples()
    ]
    trials["trial_start_trodes"] = sync_result.bpod_to_trodes(trials["trial_start_bpod"].to_numpy())
    trials["trial_end_trodes"] = sync_result.bpod_to_trodes(trials["trial_end_bpod"].to_numpy())
    n_invalid = int((~trials["sync_valid"]).sum())
    print(f"Trial windows: {n_invalid}/{n_trials} flagged sync_valid=False, "
          f"{int(trials['window_overlaps_next_trial'].sum())}/{n_trials} windows overlap next trial")

    state_events_global_bpod_t = (
        trials.set_index("trial_id")["trial_start_bpod"].reindex(state_events["trial_id"]).to_numpy()
        + state_events["start_time_in_trial"].to_numpy()
    )
    state_events["local_calib_gap_s"] = sync_result.local_calibration_gap_s(state_events_global_bpod_t)
    poke_events_global_bpod_t = (
        trials.set_index("trial_id")["trial_start_bpod"].reindex(poke_events["trial_id"]).to_numpy()
        + poke_events["time_in_trial"].to_numpy()
    )
    poke_events["local_calib_gap_s"] = sync_result.local_calibration_gap_s(poke_events_global_bpod_t)

    # --- 5. Units + spikes (real Kilosort units, pre-Phy / fabricated cluster_info) ---
    sample_rate_hz = units_spikes.read_kilosort_sample_rate(paths.kilosort4_folder)
    units = units_spikes.build_units_table(
        paths.kilosort4_folder, paths.channel_map_json,
        implant_depth_mm=rat_meta["implant_depth_mm"], implant_angle_deg=rat_meta["implant_angle_deg"],
        ml_mm=rat_meta.get("implant_ML_mm", units_spikes.DEFAULT_ML_MM),
        ap_mm=rat_meta.get("implant_AP_mm", units_spikes.DEFAULT_AP_MM),
    )
    print(f"Units: {len(units)} clusters (excluding noise)")
    spike_times_ephys, spike_clusters = units_spikes.load_spike_arrays(paths.kilosort4_folder, sample_rate_hz)
    spikes = units_spikes.build_spikes_long_table(trials, units, spike_times_ephys, spike_clusters, sync_result, buffer_s=BUFFER_S)
    print(f"Spikes: {len(spikes)} spikes across {int(trials['sync_valid'].sum())} valid trials")

    # --- 6. Write the standard synced/ bundle ---
    session_meta = {
        "rat_metadata": rat_meta, "session_name": rec_folder.name.replace(".rec", ""),
        "n_trials": n_trials, "sample_rate_hz": sample_rate_hz,
        "sync": sync_result.to_report_dict(), "port_roles": port_roles, "gui_settings": gui_constant,
        "PRELIMINARY_PRE_PHY": True,
    }
    save_json(session_meta, output_dir / "session_meta.json")
    save_table(trials, output_dir / "trials.parquet")
    save_table(units, output_dir / "units.parquet")
    save_table(spikes, output_dir / "spikes.parquet")
    save_table(state_events, output_dir / "state_events.parquet")
    save_table(poke_events, output_dir / "poke_events.parquet")
    print(f"Wrote standard synced/ bundle to {output_dir}")

    # --- 7. Align the independently-detected tracks (NOT part of units.parquet at all) ---
    d = np.load(TRACK_NPZ)
    spike_times_samples = d["spike_times"]   # shared across all tracks, in raw samples
    track_labels = d["track_labels"]
    fake_rows, fake_times, fake_clusters = [], [], []
    for local_tid, global_uid in TRACK_IDS.items():
        m = track_labels == local_tid
        t_ephys = spike_times_samples[m].astype(np.float64) / sample_rate_hz
        fake_times.append(t_ephys)
        fake_clusters.append(np.full(m.sum(), global_uid))
        fake_rows.append(dict(unit_id=global_uid, quality_label="independent_track", n_spikes_total=int(m.sum())))
        print(f"track {local_tid} (ch129/Trodes1337) -> synthetic unit_id {global_uid}: {m.sum()} spikes, "
              f"{t_ephys.min():.1f}-{t_ephys.max():.1f}s ephys range")
    fake_units = pd.DataFrame(fake_rows)
    fake_spike_times = np.concatenate(fake_times)
    fake_spike_clusters = np.concatenate(fake_clusters)

    track_spikes = units_spikes.build_spikes_long_table(
        trials, fake_units, fake_spike_times, fake_spike_clusters, sync_result, buffer_s=BUFFER_S,
    )
    track_spikes_path = rf"{OUT}\independent_track_spikes_{SESSION_ID}_ch129.parquet"
    track_spikes.to_parquet(track_spikes_path)
    print(f"Aligned {len(track_spikes)} independent-track spikes across trials -> {track_spikes_path}")
    for local_tid, global_uid in TRACK_IDS.items():
        n = (track_spikes.unit_id == global_uid).sum()
        n_trials_hit = track_spikes[track_spikes.unit_id == global_uid].trial_id.nunique()
        print(f"  track {local_tid} (unit_id {global_uid}): {n} spikes assigned across {n_trials_hit} trials")

    print("\nSYNC_AND_ALIGN_DONE")


if __name__ == "__main__":
    main()
