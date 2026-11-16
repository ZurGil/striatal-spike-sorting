# -*- coding: utf-8 -*-
"""Make a joined session's single sort syncable, by giving each recording its own copy.

    python scripts/split_concat_sort_for_sync.py 20260915_084812_103710_concat

WHY
---
2026-09-15 was recorded in two halves because Trodes was stopped and restarted, and
the two halves were joined and sorted ONCE, so a neuron that stayed under the probe
has one unit id across the whole day. That is the right way to sort it and the wrong
shape for sync_pipeline, which takes one recording, one DIO stream and one Bpod file.
The joined folder has no DIO and no timestamps.dat of its own -- it is not a Trodes
recording -- and 15 September has two Bpod files, one per half.

So the sort is split, not the behaviour joined. For each half this writes

    <half>.rec/<half>.kilosort/kilosort4_from_concat/

holding the JOINT unit ids with that half's own spike times, converted back to the
half's own sample space. The halves then sync independently, each against its own
DIO and its own Bpod file, and afterwards they share one unit space: unit 137 means
the same neuron in both trial tables. Nothing here modifies the joint sort.

WHAT THE SPLIT IS, EXACTLY
--------------------------
One number: the seam, read from the provenance file rather than recomputed.

    spike at concat sample s  ->  s < seam   : half 1, sample s
                                  s >= seam  : half 2, sample s - seam

Per-spike arrays are masked and reindexed with that. Per-unit and per-channel files
(cluster_info.tsv, templates, the channel map, the whitening matrices) describe the
clustering rather than the spikes, so they are copied unchanged -- which is what
keeps the two halves in one unit space.

A unit can be silent in one half. It stays in cluster_info.tsv there with no spikes,
and the pipeline's own units table drops it, which is the honest outcome: the unit
exists for the day, and contributed nothing to that half's trials.

THE STABILITY WINDOWS NEED THE OTHER CLOCK
------------------------------------------
Trial-range selections were drawn on the review tool's timeline, whose axis is
`concat_elapsed_seconds`. sync_pipeline wants each recording's own absolute Trodes
seconds, which start at that recording's first timestamp -- 29.63 s for half 1,
17.60 s for half 2 -- so the same number means two different moments in the two
halves, and a range that crosses the seam is two ranges. The conversion comes from
`clock_mapping` in outputs/session_timeline_<concat>.json, which states the rule and
each half's offset; it is applied here, written into each half's own selections file
with its provenance, and NOT guessed anywhere.

Note the five seconds between the timeline and the sample seam: the timeline is
binned at 20 s and drops half 1's last partial bin (its part 2 starts at 4140.0 s,
the sample seam is at 4145.0 s). Ranges are converted on the timeline axis they were
drawn on; spikes are split on the sample seam. Each is applied in its own domain.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

# Arrays with one row per spike: masked and, for spike_times, shifted.
PER_SPIKE = ("spike_times.npy", "spike_clusters.npy", "amplitudes.npy",
             "spike_templates.npy", "spike_positions.npy",
             "spike_detection_templates.npy")

# Files that describe the clustering, not the spikes. Copied so both halves keep
# the joint unit space. pc_features.npy is deliberately left out: 4 GB per half,
# and nothing downstream of here reads it.
COPY = ("cluster_info.tsv", "cluster_KSLabel.tsv", "cluster_group.tsv",
        "cluster_Amplitude.tsv", "cluster_ContamPct.tsv",
        "channel_map.npy", "channel_positions.npy", "channel_shanks.npy",
        "templates.npy", "templates_ind.npy", "similar_templates.npy",
        "whitening_mat.npy", "whitening_mat_dat.npy", "whitening_mat_inv.npy",
        "params.py", "run_provenance.json",
        # The joint run's own log and ops. They describe how these units were produced,
        # which is a property of the sort, not of either half, so each half carries them
        # and nothing has to guess the thresholds from somewhere else.
        "kilosort4.log", "ops.npy")


def load_provenance(concat_ks):
    cands = [f for f in os.listdir(concat_ks) if f.startswith("concat_") and f.endswith(".json")]
    if len(cands) != 1:
        raise SystemExit("expected exactly one concat_*.json in %s, found %s"
                         % (concat_ks, cands))
    with open(os.path.join(concat_ks, cands[0])) as fh:
        return json.load(fh)


def convert_ranges(ranges, mapping_parts, part_order):
    """Concat-timeline seconds -> one half's absolute Trodes seconds.

    Returns [] when the unit's windows do not reach into this half at all, which
    means "this unit contributed nothing here", not "keep everything".
    """
    part = next(p for p in mapping_parts if p["order"] == part_order)
    lo, hi = float(part["timeline_t0_s"]), float(part["timeline_t1_s"])
    shift = float(part["trodes_t0_s"]) - lo
    out = []
    for r in ranges:
        t0, t1 = max(float(r["t0"]), lo), min(float(r["t1"]), hi)
        if t1 - t0 <= 0:
            continue
        out.append({"t0": round(t0 + shift, 6), "t1": round(t1 + shift, 6)})
    return out


def split_part(prov, part, concat_ks, out_subdir, timeline, selections, force):
    sid = part["session"]
    rec = os.path.dirname(os.path.dirname(part["source"]))      # <sid>.rec
    ks_root = os.path.join(rec, sid + ".kilosort")
    dest = os.path.join(ks_root, out_subdir)
    if not os.path.isdir(ks_root):
        raise SystemExit("no %s -- is the half's own export still on Z?" % ks_root)
    if os.path.isdir(dest) and not force:
        raise SystemExit("%s already exists. Pass --force to rebuild it." % dest)
    os.makedirs(dest, exist_ok=True)

    seam = int(prov["seam"]["sample_index"])
    offset = int(part["offset_in_concat_samples"])
    used = int(part["used_samples"])

    st = np.load(os.path.join(concat_ks, "spike_times.npy")).ravel()
    mask = (st < seam) if part["order"] == 1 else (st >= seam)
    n = int(mask.sum())
    print("  %s: %d of %d spikes (%.1f%%)" % (sid, n, st.size, 100.0 * n / st.size))

    for name in PER_SPIKE:
        src = os.path.join(concat_ks, name)
        if not os.path.exists(src):
            continue
        a = np.load(src)
        if a.shape[0] != st.size:
            raise SystemExit("%s has %d rows against %d spikes -- refusing to guess "
                             "how to split it" % (name, a.shape[0], st.size))
        b = a[mask]
        if name == "spike_times.npy":
            b = b - offset
            if b.size:
                if b.min() < 0 or b.max() >= used:
                    raise SystemExit(
                        "%s: shifted spike samples fall outside the half (%d..%d "
                        "against 0..%d)" % (sid, b.min(), b.max(), used - 1))
        np.save(os.path.join(dest, name), b)

    for name in COPY:
        src = os.path.join(concat_ks, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))

    # Where this clustering came from. Every merge verdict Gil made is stamped with
    # the sha1 of the JOINT spike_clusters.npy, and this half's is necessarily a
    # different file -- it holds a subset of the same rows. Without this record, a
    # checker comparing the two can only conclude the verdicts were made against
    # some other sorting, which is the one thing it must not conclude here.
    with open(os.path.join(dest, "derived_from.json"), "w") as fh:
        json.dump({
            "_readme": [
                "This sort was not produced by Kilosort on this recording. It is the",
                "joined session's sort, restricted to this recording: the same unit ids,",
                "the same templates, this half's spikes, with the sample indices shifted",
                "back into this recording's own space. Written by",
                "scripts/split_concat_sort_for_sync.py.",
            ],
            "source_session": prov["concat_session_id"],
            "source_sort": concat_ks,
            "source_spike_clusters_sha1": sha1_of(
                os.path.join(concat_ks, "spike_clusters.npy")),
            "spike_clusters_sha1": sha1_of(os.path.join(dest, "spike_clusters.npy")),
            "seam_sample_index": seam,
            "part_order": part["order"],
            "sample_offset_in_concat": offset,
            "n_spikes_here": n,
            "n_spikes_joint": int(st.size),
        }, fh, indent=1)

    # The windows, on this half's clock, with their provenance attached.
    # The offsets live in the timeline's clock_mapping, not in the sort's
    # provenance: one says where the halves sit on the plotted axis, the other
    # where they sit in the raw file, and only the first can convert a window.
    mparts = timeline["clock_mapping"]["parts"]
    mpart = next(p for p in mparts if p["order"] == part["order"])
    sel_out, n_units_with_windows = {}, 0
    for uid, doc in selections.items():
        ranges = convert_ranges(doc.get("ranges", []), mparts, part["order"])
        if ranges:
            # The half's own full span travels with every entry. write_unit_merges.py
            # needs it: merging a restricted parent with an unrestricted one must come
            # out as "every trial", and the only honest way to write that as a range is
            # this recording's own bounds -- not the joined session's.
            sel_out[uid] = {
                "ranges": ranges,
                "ephys_t0_s": round(float(mpart["trodes_t0_s"]), 6),
                "ephys_t1_s": round(float(mpart["trodes_t0_s"])
                                    + float(mpart["span_by_timestamps_s"]), 6),
            }
            n_units_with_windows += 1
    sel_path = os.path.join(OUT, "trial_selections_%s_from_concat.json" % sid)
    with open(sel_path, "w") as fh:
        json.dump({
            "_readme": [
                "Per-unit stability windows for %s, converted from the joined" % sid,
                "session's review tool. The tool's axis is concat_elapsed_seconds;",
                "these are this recording's own ABSOLUTE Trodes seconds, converted",
                "with clock_mapping from session_timeline_%s.json" % prov["concat_session_id"],
                "(t -> t - timeline_t0_s + trodes_t0_s, clipped to this half, a range",
                "crossing the seam split between the halves).",
            ],
            "source_session": prov["concat_session_id"],
            "part_order": part["order"],
            "timeline_t0_s": mpart["timeline_t0_s"],
            "timeline_t1_s": mpart["timeline_t1_s"],
            "trodes_t0_s": mpart["trodes_t0_s"],
            "clock": "trodes_absolute_seconds",
            "selections": sel_out,
        }, fh, indent=1, sort_keys=True)

    print("      %d unit(s) keep a window here (of %d in the joined session)"
          % (n_units_with_windows, len(selections)))
    print("      sort    %s" % dest)
    print("      windows %s" % sel_path)
    print("      bpod    %s" % part["bpod"])
    return {"session_id": sid, "rec": rec, "ks_folder": dest,
            "selections": sel_path, "bpod": part["bpod"], "n_spikes": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("concat_session_id")
    ap.add_argument("--concat_rec", default=None,
                    help="the joined session's folder (default Z:/Gil/Shamir_1/<sid>.rec)")
    ap.add_argument("--out_subdir", default="kilosort4_from_concat",
                    help="folder name created under each half's <sid>.kilosort/")
    ap.add_argument("--selections", default=None,
                    help="the joined session's trial_selections.json dump "
                         "(default outputs/review_verdicts/<sid>/trial_selections.json)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild a half's folder that already exists")
    args = ap.parse_args()

    sid = args.concat_session_id
    rec = args.concat_rec or ("Z:/Gil/Shamir_1/%s.rec" % sid)
    concat_ks = os.path.join(rec, sid + ".kilosort", "kilosort4")
    prov = load_provenance(os.path.join(rec, sid + ".kilosort"))

    with open(os.path.join(OUT, "session_timeline_%s.json" % sid)) as fh:
        timeline = json.load(fh)
    if timeline.get("clock") != "concat_elapsed_seconds":
        raise SystemExit("timeline clock is %r, not concat_elapsed_seconds -- this "
                         "script exists only for the joined case"
                         % timeline.get("clock"))

    sel_path = args.selections or os.path.join(
        OUT, "review_verdicts", sid, "trial_selections.json")
    with open(sel_path) as fh:
        sel_raw = json.load(fh)
    selections = sel_raw.get("selections", sel_raw)

    print("joined session %s" % sid)
    print("  seam at sample %d (%.1f s), %d part(s)"
          % (prov["seam"]["sample_index"], prov["seam"]["second"], len(prov["parts"])))
    print("  %s" % prov["seam"]["real_time_gap_text"])

    parts = []
    for part in sorted(prov["parts"], key=lambda p: p["order"]):
        parts.append(split_part(prov, part, concat_ks, args.out_subdir,
                                timeline, selections, args.force))

    total = sum(p["n_spikes"] for p in parts)
    st_n = np.load(os.path.join(concat_ks, "spike_times.npy")).size
    if total != st_n:
        raise SystemExit("the halves hold %d spikes but the joint sort has %d"
                         % (total, st_n))
    print("\nevery one of the %d joint spikes landed in exactly one half." % st_n)

    plan = os.path.join(OUT, "concat_sync_plan_%s.json" % sid)
    with open(plan, "w") as fh:
        json.dump({"concat_session_id": sid, "parts": parts}, fh, indent=2)
    print("wrote %s" % plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
