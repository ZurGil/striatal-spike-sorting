"""
Lean version of run_sync_and_track_alignment_session.py's steps 1-4+7 ONLY --
recomputes sync_result (Bpod<->Trodes clock mapping, from the full Bpod
session + full Trodes TTL stream, same as always) but SKIPS steps 5/6
(loading Kilosort's 448 units x tens of millions of spikes and rewriting the
whole synced/ bundle) since that part is channel-independent and was already
written once by run_sync_and_track_alignment_session.py -- this script only
adds trial-alignment for a NEW channel's independently-detected tracks
(from independent_clustering_windowed_overlap_session.py) as synthetic
extra "units", same method as before (same sync_result, same
units_spikes.build_spikes_long_table()).

Usage: python align_tracks_to_behavior_session.py <session_id> <rec_root> <rat_root> <bpod_file> <chan> <local_track_id> [more_track_ids...]
"""
import sys
sys.path.insert(0, r"D:\Gil")
import numpy as np
import pandas as pd
from pathlib import Path

from sync_pipeline import bpod_loader, dio_decoder, session_discovery, sync, units_spikes

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
RAT_ROOT = sys.argv[3]
BPOD_FILE = sys.argv[4]
CHAN = int(sys.argv[5])
LOCAL_TRACK_IDS = [int(x) for x in sys.argv[6:]]

OUT = r"D:\Gil\spike_sorting_agent\outputs"
BUFFER_S = 10.0
POKE_IN_STATE = "stay_Cin"
TRACK_NPZ = rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch{CHAN}.npz"


def main():
    rec_folder = Path(REC_ROOT)
    rat_root = Path(RAT_ROOT)
    bpod_file = Path(BPOD_FILE)
    paths = session_discovery.session_paths_from_rec_folder(rec_folder, bpod_file)

    pkl_cache = rat_root / "bpod" / (bpod_file.stem + ".pkl")
    session_data = bpod_loader.load_bpod_session(bpod_file, pkl_cache_path=pkl_cache)
    bpod_stream = bpod_loader.build_bpod_event_stream(session_data)
    trials = bpod_loader.build_trials_table(session_data)
    state_events = bpod_loader.build_state_events_table(session_data)
    trials["TrialCompleted"] = bpod_loader.compute_trial_completed(trials, state_events)

    ttl_codes, ttl_ts = dio_decoder.extract_ttls(paths.dio_folder)
    sync_result = sync.synchronize(bpod_stream.label, bpod_stream.time, ttl_codes, ttl_ts, config=sync.SyncConfig())
    print(f"Sync: status={sync_result.sync_status} match_fraction={sync_result.match_fraction:.3f} "
          f"(recomputed fresh, identical inputs to the full sync already run for this session)")

    next_starts = np.append(trials["trial_start_bpod"].to_numpy()[1:], np.nan)
    trials["next_trial_start_bpod"] = next_starts
    trials["trial_end_bpod"] = [
        units_spikes.default_trial_window(s, (ns if not np.isnan(ns) else None), le, BUFFER_S)
        for s, ns, le in zip(trials["trial_start_bpod"], next_starts, trials["trial_last_state_end_bpod"])
    ]
    trials["trial_end_bpod"] = [w[1] for w in trials["trial_end_bpod"]]
    trials["sync_valid"] = [
        not sync.trial_overlaps_long_gap(row.trial_start_bpod, row.trial_end_bpod, sync_result.gaps)
        for row in trials.itertuples()
    ]

    sample_rate_hz = units_spikes.read_kilosort_sample_rate(paths.kilosort4_folder)

    d = np.load(TRACK_NPZ)
    spike_times_samples = d["spike_times"]
    track_labels = d["track_labels"]
    fake_rows, fake_times, fake_clusters = [], [], []
    global_id_of = {}
    for local_tid in LOCAL_TRACK_IDS:
        global_uid = 900000 + CHAN * 100 + local_tid
        global_id_of[local_tid] = global_uid
        m = track_labels == local_tid
        t_ephys = spike_times_samples[m].astype(np.float64) / sample_rate_hz
        fake_times.append(t_ephys)
        fake_clusters.append(np.full(m.sum(), global_uid))
        fake_rows.append(dict(unit_id=global_uid, quality_label="independent_track"))
        print(f"ch{CHAN} track {local_tid} -> synthetic unit_id {global_uid}: {m.sum()} spikes, "
              f"{t_ephys.min():.1f}-{t_ephys.max():.1f}s ephys range")
    fake_units = pd.DataFrame(fake_rows)
    fake_spike_times = np.concatenate(fake_times)
    fake_spike_clusters = np.concatenate(fake_clusters)

    track_spikes = units_spikes.build_spikes_long_table(
        trials, fake_units, fake_spike_times, fake_spike_clusters, sync_result, buffer_s=BUFFER_S,
    )
    out_path = rf"{OUT}\independent_track_spikes_{SESSION_ID}_ch{CHAN}.parquet"
    track_spikes.to_parquet(out_path)
    print(f"Aligned {len(track_spikes)} spikes -> {out_path}")
    for local_tid, global_uid in global_id_of.items():
        n = (track_spikes.unit_id == global_uid).sum()
        nt = track_spikes[track_spikes.unit_id == global_uid].trial_id.nunique()
        print(f"  track {local_tid} (unit_id {global_uid}): {n} spikes across {nt} trials")

    print("\nALIGN_DONE")


if __name__ == "__main__":
    main()
