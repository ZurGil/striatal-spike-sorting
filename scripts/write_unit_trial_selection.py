"""
Writes the canonical `unit_trial_selection.json` that sync_pipeline picks up
automatically, from a plain export of the review tool's per-unit stability selection.

This is the ONLY bridge between the review tool and the dataset, and it is one-way and
explicit. The review tool never writes to the dataset; sync_pipeline never reads the
review tool. In between sits one file, in one conventional place, in a documented
format you could equally well write by hand:

    <rec_root>/<session>.kilosort/kilosort4/unit_trial_selection.json

Because the location is fixed, there is nothing to remember per session and nothing to
pass on the command line: run the pipeline and it is picked up, or it is not there and
every unit keeps every trial. Either way `session_meta.json` states which happened
(`unit_trial_selection.status` = applied / not_found / disabled_by_flag), so whether a
manual stability pass reached the dataset is a fact you read, never something you have
to remember doing.

The written file is stamped with a fingerprint of that clustering's
`spike_clusters.npy`. Unit ids belong to one specific sorting -- Phy merges renumber
them -- so a selection made against a different clustering would mark the wrong trials
on the wrong neurons. sync_pipeline refuses such a file with a hard error rather than
applying it.

INPUT: a JSON file shaped like the review tool's `trial_selections` collection, i.e.
either

    {"selections": {"163": {"ranges": [{"t0": 4500, "t1": 12690.39}]}, ...}}

or the flat form

    {"163": [{"t0": 4500, "t1": 12690.39}], ...}

Units with no ranges are dropped: "kept everything" and "never reviewed" mean the same
thing and must not produce different files.

Usage:
  python write_unit_trial_selection.py <session_id> <rec_root> <input.json> [--ks_subdir kilosort4]
  e.g. python write_unit_trial_selection.py 20260916_110311 F:\\Gil\\Shamir\\20260916_110311.rec \\
            D:\\Gil\\spike_sorting_agent\\outputs\\trial_selections_20260916_110311.json
"""
import argparse, datetime, hashlib, json, os, sys

import numpy as np

FORMAT = "unit_trial_selection/1"


def fingerprint(ks_dir):
    p = os.path.join(ks_dir, "spike_clusters.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "%s not found -- point --ks_subdir at the folder whose spike_clusters.npy "
            "defines the unit ids the selection refers to." % p)
    with open(p, "rb") as f:
        raw = f.read()
    cl = np.load(p).ravel()
    return {"spike_clusters_sha1": hashlib.sha1(raw).hexdigest(),
            "n_spikes": int(len(cl)), "n_clusters": int(len(np.unique(cl)))}


def normalise(doc):
    """Accept either the nested review-tool export or the flat {uid: ranges} form."""
    raw = doc.get("selections", doc)
    out = {}
    for uid, entry in raw.items():
        if str(uid).startswith("_"):
            continue
        ranges = entry.get("ranges") if isinstance(entry, dict) else entry
        ranges = [{"t0": float(r["t0"]), "t1": float(r["t1"])} for r in (ranges or [])]
        if ranges:                      # empty == keep everything == not listed
            out[str(int(uid))] = {"ranges": ranges}
    return out


def session_window(doc):
    """The session's full ephys span, if the review tool recorded it.

    Needed by `write_unit_merges.py`: merging a restricted parent with an
    UNRESTRICTED one has to yield "every trial", and the only honest way to write
    that as a range is the session's own bounds. The review tool stamps
    ephys_t0_s/ephys_t1_s on every selection document, so it is free to carry here --
    and carrying it means the merge writer never has to guess or re-derive it.
    """
    raw = doc.get("selections", doc)
    lo, hi = [], []
    for uid, entry in raw.items():
        if str(uid).startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("ephys_t0_s") is not None and entry.get("ephys_t1_s") is not None:
            lo.append(float(entry["ephys_t0_s"]))
            hi.append(float(entry["ephys_t1_s"]))
    if not lo:
        return None
    if len(set(lo)) > 1 or len(set(hi)) > 1:
        raise SystemExit(
            "the selection documents disagree about the session's ephys window "
            "(t0 %s, t1 %s). They must all describe one session." % (sorted(set(lo)),
                                                                     sorted(set(hi))))
    return {"t0": lo[0], "t1": hi[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root")
    ap.add_argument("input_json")
    ap.add_argument("--ks_subdir", default="kilosort4",
                    help="folder under <session>.kilosort/ holding spike_clusters.npy "
                         "(default kilosort4; use e.g. verdict_review after Phy)")
    ap.add_argument("--source", default="review tool trial_selections export")
    args = ap.parse_args()

    ks_dir = os.path.join(args.rec_root, args.session_id + ".kilosort", args.ks_subdir)
    with open(args.input_json) as f:
        raw_doc = json.load(f)
    selections = normalise(raw_doc)
    window = session_window(raw_doc)

    fp = fingerprint(ks_dir)
    doc = {
        "format": FORMAT,
        "session_id": args.session_id,
        "clock": "trodes_absolute_seconds",
        "inclusion_rule": "trial_start_inside_range",
        "clustering": fp,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "scripts/write_unit_trial_selection.py",
        "source": args.source,
        "note": ("Per-unit stability selection: the time windows during which each listed "
                 "unit was actually present. Absolute Trodes seconds. A unit not listed "
                 "here keeps every trial. Read automatically by sync_pipeline from this "
                 "filename; see sync_pipeline/HOW_TO_RUN.md section 6b."),
        # The session's full ephys span, as the review tool measured it. Carried so
        # write_unit_merges.py can express "this merged unit keeps every trial" as a
        # real range when one of its parents was never restricted.
        "session_ephys_window": window,
        "selections": selections,
    }
    out_path = os.path.join(ks_dir, "unit_trial_selection.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    print("wrote %s" % out_path)
    print("  clustering fingerprint: sha1=%s n_spikes=%d n_clusters=%d"
          % (fp["spike_clusters_sha1"][:16], fp["n_spikes"], fp["n_clusters"]))
    print("  %d unit(s) restricted:" % len(selections))
    for uid, entry in sorted(selections.items(), key=lambda kv: int(kv[0])):
        spans = ", ".join("%.1f-%.1fs" % (r["t0"], r["t1"]) for r in entry["ranges"])
        print("     unit %-5s %s" % (uid, spans))
    print("\nsync_pipeline will now pick this up automatically on the next run of this "
          "session. session_meta.json will report unit_trial_selection.status='applied'.")


if __name__ == "__main__":
    main()
