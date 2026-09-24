r"""
TIMING BUG FIX applied here vs align_tracks_to_behavior_session.py:
=====================================================================
Discovered by reading the OLD (proven-working) MATLAB pipeline at
C:\Users\Adam\Documents\MATLAB\neuropixels_preprocessing (MakeTTNeuropixel.m,
line 66): spike sample indices must be converted to true Trodes time by
looking them up in the session's own <session>.timestamps.dat file (one
absolute hardware-clock timestamp per sample of probe1.dat), NOT by naive
`sample_index / sample_rate`, "because this accounts for potential lost
packages/data in Trodes."

Checked directly for this session: <session>.timestamps.dat already exists
(D:\...\20260916_110311.kilosort\20260916_110311.timestamps.dat, never
previously read by this project). Verified with certainty:
  - ZERO dropped samples anywhere in probe1.dat (every one of the
    378,086,289 consecutive per-sample timestamp diffs equals exactly 1).
  - probe1.dat's row 0 is NOT at absolute time 0 -- it's at 87.5134s on the
    same absolute Trodes clock that trials.trial_start_trodes etc. are on
    (confirmed: first trial's own trial_start_trodes = 101.5s, not ~0s).

Every spike time in this whole investigation (mine AND Kilosort's real
units, via units_spikes.load_spike_arrays()) has been computed as
`sample_index / sample_rate` with NO offset -- i.e. on a clock that starts
at 0, while the sync's "trodes time" (what sync_result.trodes_to_bpod()
was fit against) starts at 87.5134s. Every spike has therefore been fed
into the sync ~87.5s too early -- easily enough to scramble trial
assignment given trials are only ~10-20s apart.

FIX: since there are zero dropped samples (confirmed above), the correct
per-sample timestamp is just a CONSTANT offset added to the naive
computation -- no need for the full per-sample lookup array. This script
adds that one constant (read fresh from the file's own first value, not
hardcoded) before spike times are handed to build_spikes_long_table().

This does NOT edit anything under D:\Gil\sync_pipeline -- only calls its
functions, same as every other script in this investigation.

Usage: python align_tracks_to_behavior_session_TIMEFIXED.py <session_id> <rec_root> <rat_root> <bpod_file> <chan> <local_track_id> [more_track_ids...]
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
TRACK_NPZ = rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch{CHAN}.npz"


def get_ephys_absolute_offset_s(kilosort4_folder, sample_rate_hz):
    """Read <session>.timestamps.dat's own first value -- the absolute
    Trodes-clock time (seconds) of probe1.dat's sample 0. Also verifies
    zero dropped samples so a constant-offset correction is valid (if this
    assertion ever fails for a different session, the full per-sample
    lookup would be needed instead of a constant add)."""
    ts_path = Path(kilosort4_folder).parent / f"{SESSION_ID}.timestamps.dat"
    parsed = dio_decoder.read_trodes_extracted_data_file(str(ts_path))
    ts = parsed.data["time"]
    diffs = np.diff(ts.astype(np.int64))
    n_gaps = int((diffs != 1).sum())
    if n_gaps != 0:
        raise RuntimeError(
            f"{ts_path} has {n_gaps} non-unit sample gap(s) -- constant-offset "
            "correction is NOT valid here, a full per-sample lookup is required instead."
        )
    offset_s = float(ts[0]) / sample_rate_hz
    print(f"TIME FIX: {ts_path.name} confirms 0 dropped samples; "
          f"absolute offset = {offset_s:.4f}s (raw counter {int(ts[0])})")
    return offset_s


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
    print(f"Sync: status={sync_result.sync_status} match_fraction={sync_result.match_fraction:.3f}")

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
    offset_s = get_ephys_absolute_offset_s(paths.kilosort4_folder, sample_rate_hz)

    d = np.load(TRACK_NPZ)
    spike_times_samples = d["spike_times"]
    track_labels = d["track_labels"]
    fake_rows, fake_times, fake_clusters = [], [], []
    global_id_of = {}
    for local_tid in LOCAL_TRACK_IDS:
        global_uid = 900000 + CHAN * 100 + local_tid
        global_id_of[local_tid] = global_uid
        m = track_labels == local_tid
        # THE FIX: + offset_s, instead of the naive sample_index/sample_rate alone
        t_ephys = spike_times_samples[m].astype(np.float64) / sample_rate_hz + offset_s
        fake_times.append(t_ephys)
        fake_clusters.append(np.full(m.sum(), global_uid))
        fake_rows.append(dict(unit_id=global_uid, quality_label="independent_track"))
        print(f"ch{CHAN} track {local_tid} -> synthetic unit_id {global_uid}: {m.sum()} spikes, "
              f"{t_ephys.min():.1f}-{t_ephys.max():.1f}s ephys range (TIME-CORRECTED)")
    fake_units = pd.DataFrame(fake_rows)
    fake_spike_times = np.concatenate(fake_times)
    fake_spike_clusters = np.concatenate(fake_clusters)

    track_spikes = units_spikes.build_spikes_long_table(
        trials, fake_units, fake_spike_times, fake_spike_clusters, sync_result, buffer_s=BUFFER_S,
    )
    out_path = rf"{OUT}\independent_track_spikes_{SESSION_ID}_ch{CHAN}_TIMEFIXED.parquet"
    track_spikes.to_parquet(out_path)
    print(f"Aligned {len(track_spikes)} spikes -> {out_path}")
    for local_tid, global_uid in global_id_of.items():
        n = (track_spikes.unit_id == global_uid).sum()
        nt = track_spikes[track_spikes.unit_id == global_uid].trial_id.nunique()
        print(f"  track {local_tid} (unit_id {global_uid}): {n} spikes across {nt} trials")

    print("\nALIGN_TIMEFIXED_DONE")


if __name__ == "__main__":
    main()
