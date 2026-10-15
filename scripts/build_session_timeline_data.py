"""
Builds session_timeline_<session_id>.json -- the per-unit firing-rate-across-the-session
timeline + trial-start markers that the published review tool's "Firing rate across the
session" panel reads. Feature added 2026-10-15 (see WORKFLOW.md 1f-pre).

This is a SEPARATE file from pipeline_review_data_<session_id>.json on purpose: the
trial markers and the bin grid are session-level, not per-unit, and the rate arrays are
packed binary (see below) rather than JSON numbers. build_review_html.py embeds it if
present and degrades to "not computed for this session" if absent.


WHY THIS CAN RUN BEFORE SYNC AT ALL (the point worth not re-deriving)
--------------------------------------------------------------------
The review tool runs BEFORE sync_pipeline, so neither trials.parquet nor the Bpod file
is available yet. Trial boundaries are still available, exactly, from the raw Trodes DIO
export alone:

  * Each Din{1..6}.dat bit-plane decodes (dio_decoder.extract_ttls, imported read-only
    from sync_pipeline -- nothing here writes to that package) into a 0-63 TTL code
    stream whose timestamps are already in absolute Trodes-clock seconds.
  * The TTL code IS the Bpod state number (that is what sync.py aligns the two streams
    on), so TTL code 1 = the first state of each trial = the trial start.
  * Verified 2026-10-15 on 20260916_110311 against that session's own fully-synced
    trials.parquet: 954 code-1 events for 953 Bpod trials, and the ordered difference
    against trial_start_trodes is 0.0 ms median AND 0.0 ms max -- not approximately,
    exactly, to the sample. (Codes 2 and 3 are also rock-steady at +49.9 ms and
    +549.9 ms, i.e. deterministic downstream state transitions.)
  * Code 0 is NOT the trial start, despite appearing once per trial: it sits a variable
    ~0.4-4 s BEFORE it (median -2147 ms, max |13991| ms). It is the inter-trial reset,
    and the first "code 0" event is an artifact of the zero-order-hold reconstruction
    starting with all lines low. Do not use it for ticks.

Two honest caveats, both reported in the output JSON rather than hidden:
  * There are usually a FEW MORE code-1 markers than Bpod trials (954 vs 953 here) --
    the last trial typically starts on the rig but never lands in the saved Bpod file.
    So these are "trial start markers", not trial_ids. Mapping marker -> trial_id is
    the sync pipeline's job, later, and it is exact when it happens.
  * --trial_code 1 assumes the protocol's state machine always begins at state 1. True
    for this task; dio_code_census in the output lets a future session's ticks be
    re-checked the same way (count per code vs expected trial count) instead of assumed.

Spike times go through <session>.timestamps.dat, like sync_pipeline does, so the rate
curve and the trial ticks are on ONE clock. Skipping that lookup would shift every
spike by probe1.dat's export offset -- 87.5134 s on this very session -- and silently
slide the whole rate curve relative to the ticks. (Same bug README.md's top section is
about.) We memmap that file and index it rather than read its 1.5 GB into memory.

Output JSON:
  {
    session_id, fs_hz, bin_s, clock:"trodes_absolute_seconds",
    ephys_t0_s, ephys_t1_s, n_bins,                   # bin i = [t0+i*bin_s, t0+(i+1)*bin_s)
    trial_markers: {source, ttl_code, n, t_s:[...]},  # absolute Trodes seconds
    dio_code_census: {code: n_events},
    counts_dtype:"uint16", counts_encoding:"base64",
    counts: {"<unit_id>": "<base64 of n_bins little-endian uint16 spike counts>"},
    n_spikes_total, n_units
  }
Rate in Hz = count / bin_s, done in the browser.

Usage: python build_session_timeline_data.py <session_id> <rec_root> [--bin_s 20]
                                             [--trial_code 1] [--max_units N]
  <rec_root> e.g. F:\Gil\Shamir\20260916_110311.rec
  --max_units: proof-of-concept / smoke-test shortcut, bins only the first N cluster ids.
Any env with numpy (phy2_ky is fine); no GPU, no SpikeInterface.
"""
import argparse, base64, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\Gil")  # so `sync_pipeline` imports; read-only use

OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs"
CHUNK = 4_000_000  # spikes per pass; keeps peak memory flat regardless of session length


def find_payload_offset(path):
    """Byte offset of the first binary record in a Trodes extracted-data file.

    Same rule dio_decoder.read_trodes_extracted_data_file() uses -- one byte past
    '<End settings>' to skip the trailing newline -- reimplemented here only so we can
    np.memmap the 1.5 GB timestamps file instead of reading all of it. Validated by the
    caller against probe1.dat's own row count, which is what would catch a wrong offset.
    """
    with open(path, "rb") as f:
        head = f.read(20000)
    end = head.find(b"<End settings>")
    if end == -1:
        raise ValueError(path + ": no '<End settings>' in the first 20 kB of header.")
    return end + len(b"<End settings>") + 1


def load_timestamp_map(ks_parent, session_id, n_probe_rows=None):
    ts_path = os.path.join(ks_parent, session_id + ".timestamps.dat")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(
            ts_path + " not found. Without it, spike sample indices cannot be put on the "
            "same absolute Trodes clock as the DIO trial markers -- see this script's "
            "docstring and sync_pipeline/README.md."
        )
    off = find_payload_offset(ts_path)
    payload = os.path.getsize(ts_path) - off
    if payload % 4:
        raise ValueError("%s: payload %d B is not a multiple of 4 (uint32 records) -- "
                         "header offset is wrong." % (ts_path, payload))
    mm = np.memmap(ts_path, dtype=np.uint32, mode="r", offset=off, shape=(payload // 4,))
    if n_probe_rows is not None and n_probe_rows != len(mm):
        raise ValueError("%s has %d rows but probe1.dat has %d -- these files are not from "
                         "the same recording." % (ts_path, len(mm), n_probe_rows))
    return mm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root", help=r"e.g. F:\Gil\Shamir\20260916_110311.rec")
    ap.add_argument("--bin_s", type=float, default=20.0)
    ap.add_argument("--trial_code", type=int, default=1,
                    help="TTL code treated as the trial-start marker (default 1 = Bpod state 1).")
    ap.add_argument("--max_units", type=int, default=None)
    ap.add_argument("--fs_hz", type=float, default=30000.0)
    args = ap.parse_args()

    sid, fs, bin_s = args.session_id, args.fs_hz, args.bin_s
    ks_parent = os.path.join(args.rec_root, sid + ".kilosort")
    ks_dir = os.path.join(ks_parent, "kilosort4", "kilosort_original")
    dio_dir = os.path.join(args.rec_root, sid + ".DIO")

    # --- 1. the clock ------------------------------------------------------
    probe_path = os.path.join(ks_parent, sid + ".probe1.dat")
    n_probe_rows = os.path.getsize(probe_path) // (384 * 2) if os.path.exists(probe_path) else None
    ts_map = load_timestamp_map(ks_parent, sid, n_probe_rows)
    t0 = float(ts_map[0]) / fs
    t1 = float(ts_map[-1]) / fs
    n_bins = int(np.ceil((t1 - t0) / bin_s))
    print("clock: probe1.dat row 0 = absolute sample %d = %.4fs; last row = %.4fs -> "
          "%d bins of %gs" % (int(ts_map[0]), t0, t1, n_bins, bin_s))

    # --- 2. trial markers, straight from DIO (no Bpod, no sync) ------------
    from sync_pipeline import dio_decoder  # read-only import
    codes, ev_t = dio_decoder.extract_ttls(dio_dir)
    census = Counter(int(c) for c in codes)
    marker_t = ev_t[codes == args.trial_code]
    print("DIO: %d events over %d distinct codes; code %d -> %d trial-start markers"
          % (len(codes), len(census), args.trial_code, len(marker_t)))
    if len(marker_t) >= 2:
        print("     markers span %.3f..%.3fs, median interval %.3fs"
              % (marker_t[0], marker_t[-1], float(np.median(np.diff(marker_t)))))
    else:
        print("WARNING: only %d marker(s) for code %d. Check dio_code_census in the output "
              "-- this protocol may not start at state 1." % (len(marker_t), args.trial_code))

    # --- 3. per-unit spike counts per bin ----------------------------------
    spike_samples = np.load(os.path.join(ks_dir, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(ks_dir, "spike_clusters.npy")).ravel()
    if len(spike_samples) != len(spike_clusters):
        raise ValueError("spike_times.npy and spike_clusters.npy differ in length.")
    uids = np.unique(spike_clusters)
    if args.max_units:
        uids = uids[: args.max_units]
        keep = np.isin(spike_clusters, uids)
        spike_samples, spike_clusters = spike_samples[keep], spike_clusters[keep]
        print("--max_units %d: restricted to cluster ids %d..%d"
              % (args.max_units, uids.min(), uids.max()))
    row_of = np.full(int(uids.max()) + 1, -1, dtype=np.int64)
    row_of[uids] = np.arange(len(uids))
    print("spikes: %d across %d clusters" % (len(spike_samples), len(uids)))

    # Spike amplitude per bin, alongside the rate. This is NOT a detector and nothing
    # here decides anything: it is a second trace to read by eye, because rate alone
    # cannot tell "the cell went quiet" from "the cell drifted off the electrode".
    # Fading amplitude with falling rate = drift; steady amplitude with falling rate =
    # the cell really stopped firing. kilosort_original/amplitudes.npy is the per-spike
    # template scaling factor -- arbitrary units, so only its change WITHIN a unit means
    # anything, which is exactly what is wanted here (and the same field
    # build_burst_isi_metrics_session.py already uses).
    amp_path = os.path.join(ks_dir, "amplitudes.npy")
    amps = np.load(amp_path).ravel().astype(np.float64) if os.path.exists(amp_path) else None
    if amps is None:
        print("note: %s absent -- building the rate trace only." % amp_path)
    elif args.max_units:
        amps = amps[keep]

    flat = np.zeros(len(uids) * n_bins, dtype=np.int64)
    amp_sum = np.zeros(len(uids) * n_bins, dtype=np.float64) if amps is not None else None
    for i in range(0, len(spike_samples), CHUNK):
        s = spike_samples[i:i + CHUNK].astype(np.int64)
        abs_t = ts_map[s].astype(np.float64) / fs          # the offset correction
        b = np.floor((abs_t - t0) / bin_s).astype(np.int64)
        np.clip(b, 0, n_bins - 1, out=b)
        r = row_of[spike_clusters[i:i + CHUNK]]
        idx = r * n_bins + b
        flat += np.bincount(idx, minlength=flat.size)
        if amps is not None:
            amp_sum += np.bincount(idx, weights=amps[i:i + CHUNK], minlength=flat.size)
        print("  binned %d/%d" % (min(i + CHUNK, len(spike_samples)), len(spike_samples)))
    counts = flat.reshape(len(uids), n_bins)

    hi = int(counts.max())
    if hi > 65535:
        raise ValueError("max bin count %d exceeds uint16; use a smaller --bin_s." % hi)
    packed = {str(int(u)): base64.b64encode(counts[k].astype("<u2").tobytes()).decode("ascii")
              for k, u in enumerate(uids)}

    # Amplitude is stored per unit as uint8 on that unit's OWN scale (0 = no spikes in
    # the bin, 1-250 = mean amplitude x amp_scale[uid]) -- a byte per bin instead of
    # four, which matters because the whole page ships inside the HTML.
    packed_amp, amp_scale = {}, {}
    if amps is not None:
        mean_amp = np.divide(amp_sum.reshape(len(uids), n_bins), np.maximum(counts, 1),
                             where=counts > 0, out=np.zeros((len(uids), n_bins)))
        for k, u in enumerate(uids):
            row, has = mean_amp[k], counts[k] > 0
            top = float(np.percentile(row[has], 99)) if has.any() else 0.0
            sc = top / 250.0 if top > 0 else 1.0
            q = np.where(has, np.clip(np.round(row / sc), 1, 255), 0).astype(np.uint8)
            packed_amp[str(int(u))] = base64.b64encode(q.tobytes()).decode("ascii")
            amp_scale[str(int(u))] = round(sc, 6)

    out = {
        "session_id": sid,
        "built_by": "build_session_timeline_data.py",
        "fs_hz": fs,
        "bin_s": bin_s,
        "clock": "trodes_absolute_seconds",
        "clock_note": ("spike sample indices converted via <session>.timestamps.dat, the same "
                       "absolute Trodes clock dio_decoder.extract_ttls() reports -- rate curve "
                       "and trial markers are directly comparable"),
        "ephys_t0_s": t0,
        "ephys_t1_s": t1,
        "n_bins": n_bins,
        "trial_markers": {
            "source": "dio_ttl_code",
            "ttl_code": args.trial_code,
            "note": ("TTL code = Bpod state number; code 1 = each trial's first state. Verified "
                     "exact (0.0 ms median and max) against synced trials.parquet's "
                     "trial_start_trodes on 20260916_110311. There are usually a few more "
                     "markers than saved Bpod trials -- marker -> trial_id mapping is the sync "
                     "pipeline's job, not this file's."),
            "n": int(len(marker_t)),
            "t_s": [round(float(t), 4) for t in marker_t],
        },
        "dio_code_census": {str(c): int(n) for c, n in sorted(census.items())},
        "counts_dtype": "uint16",
        "counts_encoding": "base64",
        "counts_note": "rate_hz[i] = count[i] / bin_s",
        "amp_dtype": "uint8",
        "amp_encoding": "base64",
        "amp_note": ("mean kilosort template-scaling amplitude per bin, on each unit's OWN "
                     "scale: 0 = no spikes in that bin, otherwise amp[i] * amp_scale[uid]. "
                     "Arbitrary units -- only its change within one unit is meaningful."),
        "n_spikes_total": int(len(spike_samples)),
        "n_units": int(len(uids)),
        "counts": packed,
        "amp": packed_amp,
        "amp_scale": amp_scale,
    }
    out_path = os.path.join(OUT_DIR, "session_timeline_%s.json" % sid)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print("\nwrote %s (%d bytes)" % (out_path, os.path.getsize(out_path)))
    print("peak bin count %d (%.1f Hz); median unit mean rate %.2f Hz"
          % (hi, hi / bin_s, float(np.median(counts.sum(1) / (t1 - t0)))))


if __name__ == "__main__":
    main()
