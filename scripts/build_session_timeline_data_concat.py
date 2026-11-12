"""
Builds session_timeline_<session_id>.json for a CONCATENATED session -- one made by
joining two or more Trodes recordings of the same day into a single raw file and sorting
it once (see RESEARCH_LOG.md 5bd.42 and WORKFLOW.md STAGE 0).

Same output contract as build_session_timeline_data.py, which the review tool reads, so
build_review_html.py needs no change. What is different, and why:


1. A JOINT SESSION HAS NO SINGLE TRODES CLOCK
---------------------------------------------
The plain builder puts spikes and DIO markers on one absolute Trodes clock via
<session>.timestamps.dat. A joint session has one such file PER PART and none for the
whole, because the sample counter restarts when Trodes restarts -- part 2's absolute
seconds overlap part 1's, so "absolute Trodes seconds" is not even a well-defined axis
here. There is no joint timestamps.dat and there must not be one.

So this script reads each part's own timestamps.dat and DIO, converts both to seconds
ELAPSED WITHIN THAT PART, and lays the parts end to end on a single display axis:

    clock = "concat_elapsed_seconds"   (0 = the first sample of part 1)

The real gap between the recordings is NOT drawn as empty bins. 38 minutes of blank
would eat a third of the panel and read as "the cell went silent" rather than "nothing
was recorded". It is reported in the JSON instead (`concat.real_time_gap_seconds`) and
the seam lands on an exact bin boundary so the tool can draw a line there.

THE CLOCK NAME IS LOAD-BEARING, NOT DECORATION. The review tool stamps ephys_t0_s /
ephys_t1_s onto every per-unit time-range selection it exports, and
write_unit_trial_selection.py writes those ranges out as `clock:
trodes_absolute_seconds` for sync_pipeline to apply with inclusion_rule
trial_start_inside_range. Hand it ranges measured on THIS axis under that label and
every trial-inclusion decision for the session is quietly wrong. That is why
`clock_mapping` below is part of the output, and why write_unit_trial_selection.py now
reads the clock from this file instead of assuming it.


2. THE SEAM IS PUT ON A BIN BOUNDARY, AND THE REMAINDER IS DROPPED, NOT SMEARED
-------------------------------------------------------------------------------
Each non-final part contributes floor(span / bin_s) WHOLE bins. Part 1 of
20260915_084812_103710_concat spans 4145.0 s, which is 207 bins of 20 s plus 5 s left
over. Those 5 s are left out of the timeline (about 11 k of 16.7 M spikes, 0.07 %) and
the count is reported as `parts[k].timeline_tail_dropped_s` / `_spikes_dropped`.

The alternatives are worse and both mislead exactly where someone would look hardest:
a short 208th bin draws a 4x dip right at the seam, and folding those spikes into bin
207 inflates it by a quarter. The final part keeps its short tail bin, which is what
every ordinary session already does.


3. EVERYTHING ELSE IS THE PLAIN BUILDER'S LOGIC, IMPORTED NOT COPIED
--------------------------------------------------------------------
find_payload_offset, find_recording_gaps and load_timestamp_map come from
build_session_timeline_data.py directly. Trial markers are still DIO TTL code 1 = Bpod
state 1 -- read that script's docstring for why that is exact and why code 0 is not the
trial start. Dropped-packet gaps are found per part and reported on this axis.

Usage:
  python build_session_timeline_data_concat.py <concat_session_id> <concat_rec_root>
         [--concat_json PATH] [--parts_root Z:\\Gil\\Shamir_1] [--bin_s 20]
         [--trial_code 1] [--max_units N]

  <concat_rec_root>  e.g. F:\\Gil\\Shamir\\20260915_084812_103710_concat.rec
                     (needs kilosort4\\kilosort_original\\ -- so the agent batches must
                     have been consolidated first, same as the plain builder)
  --concat_json      the provenance file written when the parts were joined. Found
                     automatically next to the sort, or in outputs\\, if not given. It
                     is the ONLY source for the seam and the part list -- nothing here
                     is inferred from the folder name.
  --parts_root       where each part's own <part>.rec folder lives, for its
                     timestamps.dat and DIO. Defaults to the rat's archive on Z.

Any env with numpy; no GPU. ~10 s for a 2 h two-part session.
"""
import argparse
import base64
import glob
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\Gil")  # so `sync_pipeline` imports; read-only use

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs"
CHUNK = 4_000_000


def _plain_builder():
    """Import build_session_timeline_data.py as a module without running it."""
    path = os.path.join(HERE, "build_session_timeline_data.py")
    spec = importlib.util.spec_from_file_location("_plain_timeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_concat_json(sid, rec_root):
    candidates = []
    ks_parent = os.path.join(rec_root, sid + ".kilosort")
    candidates += sorted(glob.glob(os.path.join(ks_parent, "concat_*.json")))
    candidates += sorted(glob.glob(os.path.join(OUT_DIR, "concat_*.json")))
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get("concat_session_id") == sid:
            return c, d
    raise SystemExit(
        "no concatenation provenance file for %s. Looked for concat_*.json in\n  %s\n  %s\n"
        "One is required: the seam and the part list are read from it, never guessed."
        % (sid, ks_parent, OUT_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root")
    ap.add_argument("--concat_json", default=None)
    ap.add_argument("--parts_root", default=r"Z:\Gil\Shamir_1")
    ap.add_argument("--bin_s", type=float, default=20.0)
    ap.add_argument("--trial_code", type=int, default=1)
    ap.add_argument("--max_units", type=int, default=None)
    ap.add_argument("--fs_hz", type=float, default=30000.0)
    args = ap.parse_args()

    plain = _plain_builder()
    sid, fs, bin_s = args.session_id, args.fs_hz, args.bin_s

    if args.concat_json:
        with open(args.concat_json, "r", encoding="utf-8") as fh:
            prov = json.load(fh)
        cj = args.concat_json
        if prov.get("concat_session_id") != sid:
            raise SystemExit("%s describes %r, not %r" % (cj, prov.get("concat_session_id"), sid))
    else:
        cj, prov = find_concat_json(sid, args.rec_root)
    print("provenance: %s" % cj)

    parts = sorted(prov["parts"], key=lambda p: p["order"])
    seam = int(prov["seam"]["sample_index"])
    total_samples = int(prov["total"]["samples"])
    print("%d parts, seam at sample %d, %d samples total, real gap %s"
          % (len(parts), seam, total_samples, prov["seam"].get("real_time_gap_text", "?")))

    # ---- the raw joint file must be the one that was sorted ------------------
    ks_parent = os.path.join(args.rec_root, sid + ".kilosort")
    probe_path = os.path.join(ks_parent, sid + ".probe1.dat")
    if os.path.exists(probe_path):
        rows = os.path.getsize(probe_path) // 768
        if rows != total_samples:
            raise SystemExit("%s holds %d samples but the provenance says %d -- the staged "
                             "joint file is not the one this provenance describes."
                             % (probe_path, rows, total_samples))
        print("joint raw file checks out: %d samples" % rows)

    # ---- per part: its own clock, its own DIO -------------------------------
    from sync_pipeline import dio_decoder  # read-only import

    part_info = []
    bin_offset = 0
    marker_t, census = [], Counter()
    for i, p in enumerate(parts):
        psid = p["session"]
        prec = os.path.join(args.parts_root, psid + ".rec")
        pks = os.path.join(prec, psid + ".kilosort")
        used = int(p["used_samples"])
        src_samples = int(p["source_samples"])

        ts = plain.load_timestamp_map(pks, psid, src_samples)
        t0 = float(ts[0]) / fs
        span = (float(ts[used - 1]) - float(ts[0])) / fs
        nominal = float(p["used_seconds"])
        print("\npart %d  %s" % (p["order"], psid))
        print("   trodes t0 %.4f s, %d of %d samples used" % (t0, used, src_samples))
        print("   span by timestamps %.4f s vs nominal %.4f s (difference %.4f s)"
              % (span, nominal, span - nominal))

        gaps = plain.find_recording_gaps(ts[:used], fs)
        if gaps:
            print("   DROPPED-PACKET GAPS: %d, %.3f s never recorded"
                  % (len(gaps), sum(g["missing_s"] for g in gaps)))
        else:
            print("   dropped-packet gaps: none")

        last = (i == len(parts) - 1)
        nb = int(np.ceil(span / bin_s)) if last else int(np.floor(span / bin_s))
        tail = span - nb * bin_s
        print("   %d bins of %gs%s" % (nb, bin_s,
              "" if last else "  (%.3f s tail left out of the timeline)" % tail))

        codes, ev_t = dio_decoder.extract_ttls(os.path.join(prec, psid + ".DIO"))
        census.update(int(c) for c in codes)
        mt = np.asarray(ev_t)[np.asarray(codes) == args.trial_code] - t0
        in_window = mt[(mt >= 0) & (mt < nb * bin_s)]
        dropped_markers = int(len(mt) - len(in_window))
        print("   DIO: %d events, %d code-%d markers, %d inside the binned window%s"
              % (len(codes), len(mt), args.trial_code, len(in_window),
                 "" if not dropped_markers else "  (%d outside, in the trimmed tail or "
                 "before t0)" % dropped_markers))
        marker_t.extend((in_window + bin_offset * bin_s).tolist())

        part_info.append({
            "order": p["order"], "session": psid,
            "sample_offset_in_concat": int(p["offset_in_concat_samples"]),
            "used_samples": used,
            "trodes_t0_s": round(t0, 6),
            "span_by_timestamps_s": round(span, 6),
            "nominal_used_seconds": nominal,
            "bin_first": bin_offset, "bin_last": bin_offset + nb - 1, "n_bins": nb,
            "timeline_t0_s": round(bin_offset * bin_s, 6),
            "timeline_t1_s": round((bin_offset + nb) * bin_s, 6),
            "timeline_tail_dropped_s": 0.0 if last else round(tail, 6),
            "recording_gaps": gaps,
            "n_trial_markers": int(len(in_window)),
            "trial_markers_outside_window": dropped_markers,
            "bpod": p.get("bpod"),
            "ts_map": ts, "_used": used, "_t0": t0, "_nb": nb, "_offset": bin_offset,
        })
        bin_offset += nb

    n_bins = bin_offset
    print("\ntimeline: %d bins of %g s = %.1f s on the concat_elapsed_seconds axis"
          % (n_bins, bin_s, n_bins * bin_s))

    # ---- spikes ------------------------------------------------------------
    ks_dir = os.path.join(ks_parent, "kilosort4", "kilosort_original")
    if not os.path.isdir(ks_dir):
        raise SystemExit("%s does not exist -- run the agent batches and "
                         "consolidate_batches_session.py first (WORKFLOW.md 1f)." % ks_dir)
    spike_samples = np.load(os.path.join(ks_dir, "spike_times.npy")).ravel().astype(np.int64)
    spike_clusters = np.load(os.path.join(ks_dir, "spike_clusters.npy")).ravel()
    if len(spike_samples) != len(spike_clusters):
        raise ValueError("spike_times.npy and spike_clusters.npy differ in length.")
    if spike_samples.max() >= total_samples:
        raise SystemExit("a spike sits at sample %d, past the joint file's %d -- the sort and "
                         "the provenance disagree." % (spike_samples.max(), total_samples))

    uids = np.unique(spike_clusters)
    if args.max_units:
        uids = uids[: args.max_units]
        keep = np.isin(spike_clusters, uids)
        spike_samples, spike_clusters = spike_samples[keep], spike_clusters[keep]
    row_of = np.full(int(uids.max()) + 1, -1, dtype=np.int64)
    row_of[uids] = np.arange(len(uids))
    print("spikes: %d across %d clusters" % (len(spike_samples), len(uids)))

    flat = np.zeros(len(uids) * n_bins, dtype=np.int64)
    dropped_total = 0
    for pi in part_info:
        lo = pi["sample_offset_in_concat"]
        hi = lo + pi["_used"]
        sel = (spike_samples >= lo) & (spike_samples < hi)
        n_in_part = int(sel.sum())
        ss = spike_samples[sel] - lo
        cc = spike_clusters[sel]
        ts, t0, nb, off = pi["ts_map"], pi["_t0"], pi["_nb"], pi["_offset"]
        dropped_here = 0
        for i in range(0, len(ss), CHUNK):
            s = ss[i:i + CHUNK]
            elapsed = ts[s].astype(np.float64) / fs - t0      # the offset correction
            b = np.floor(elapsed / bin_s).astype(np.int64)
            ok = (b >= 0) & (b < nb)
            dropped_here += int((~ok).sum())
            r = row_of[cc[i:i + CHUNK]][ok]
            flat += np.bincount(r * n_bins + (b[ok] + off), minlength=flat.size)
        pi["n_spikes"] = n_in_part
        pi["spikes_dropped_with_timeline_tail"] = dropped_here
        dropped_total += dropped_here
        print("  part %d: %d spikes, %d dropped with the %.3f s display tail (%.3f%%)"
              % (pi["order"], n_in_part, dropped_here, pi["timeline_tail_dropped_s"],
                 100.0 * dropped_here / max(n_in_part, 1)))
    counts = flat.reshape(len(uids), n_bins)

    hi_count = int(counts.max())
    if hi_count > 65535:
        raise ValueError("max bin count %d exceeds uint16; use a smaller --bin_s." % hi_count)
    packed = {str(int(u)): base64.b64encode(counts[k].astype("<u2").tobytes()).decode("ascii")
              for k, u in enumerate(uids)}

    marker_t.sort()
    gaps_on_axis = []
    for pi in part_info:
        for g in pi["recording_gaps"]:
            gg = dict(g)
            gg["t0"] = round(g["t0"] - pi["trodes_t0_s"] + pi["timeline_t0_s"], 4)
            gg["t1"] = round(g["t1"] - pi["trodes_t0_s"] + pi["timeline_t0_s"], 4)
            gg["part"] = pi["session"]
            gaps_on_axis.append(gg)

    for pi in part_info:
        for k in ("ts_map", "_used", "_t0", "_nb", "_offset"):
            pi.pop(k, None)

    out = {
        "session_id": sid,
        "built_by": "build_session_timeline_data_concat.py",
        "fs_hz": fs,
        "bin_s": bin_s,
        "clock": "concat_elapsed_seconds",
        "clock_note": (
            "NOT absolute Trodes seconds. This session is two recordings joined into one raw "
            "file; each has its own Trodes sample counter, and they overlap, so there is no "
            "single absolute clock. 0 here is the first sample of part 1, and each part's "
            "spikes and DIO markers were put on this axis through that part's OWN "
            "timestamps.dat. The 37m56s real-time gap between the recordings is NOT drawn as "
            "empty bins -- the seam is an exact bin boundary and the gap is reported in "
            "concat.real_time_gap_seconds. Use clock_mapping to convert a time on this axis "
            "back to a part and that part's absolute Trodes seconds, which is what "
            "sync_pipeline needs."),
        "ephys_t0_s": 0.0,
        "ephys_t1_s": n_bins * bin_s,
        "n_bins": n_bins,
        "concat": {
            "is_concatenated": True,
            "provenance_file": cj,
            "seam_sample_index": seam,
            "seam_bins": [p["bin_first"] for p in part_info[1:]],
            "seam_timeline_s": [p["timeline_t0_s"] for p in part_info[1:]],
            "real_time_gap_seconds": prov["seam"].get("real_time_gap_seconds"),
            "real_time_gap_text": prov["seam"].get("real_time_gap_text"),
            "warning": ("Each seam is a break in real time, not in the array. Do not smooth, "
                        "average or fit across one."),
        },
        "clock_mapping": {
            "rule": ("for a time t on this axis, find the part whose timeline_t0_s <= t < "
                     "timeline_t1_s; that part's absolute Trodes second is "
                     "t - timeline_t0_s + trodes_t0_s, and its own Bpod file is the one "
                     "listed. A range that spans a seam must be split."),
            "parts": part_info,
        },
        "trial_markers": {
            "source": "dio_ttl_code_per_part_then_shifted",
            "ttl_code": args.trial_code,
            "note": ("TTL code = Bpod state number; code 1 = each trial's first state. Read "
                     "from EACH part's own .DIO folder, converted to seconds elapsed in that "
                     "part, then shifted onto this axis. Markers falling in a part's trimmed "
                     "display tail are left out and counted in clock_mapping.parts. Marker -> "
                     "trial_id mapping is the sync pipeline's job, and for this session it is "
                     "two jobs, one per Bpod file."),
            "n": len(marker_t),
            "t_s": [round(float(t), 4) for t in marker_t],
        },
        "dio_code_census": {str(k): int(v) for k, v in sorted(census.items())},
        "recording_gaps": gaps_on_axis,
        "timeline_spikes_dropped_with_tails": dropped_total,
        "counts_dtype": "uint16",
        "counts_encoding": "base64",
        "counts": packed,
        "n_spikes_total": int(counts.sum()),
        "n_units": len(uids),
    }

    dest = os.path.join(OUT_DIR, "session_timeline_%s.json" % sid)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh)
    print("\nwritten: %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1e6))
    print("markers: %d   bins: %d   units: %d   spikes binned: %d (%d dropped with tails)"
          % (len(marker_t), n_bins, len(uids), int(counts.sum()), dropped_total))


if __name__ == "__main__":
    main()
