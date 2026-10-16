# -*- coding: utf-8 -*-
"""Run sync_pipeline for a session WITH every decision from the review tool applied.

    python scripts/sync_session.py <session_id> --rec <rec_folder> --dumps <dir>

THE POINT OF THIS SCRIPT IS THAT IT REFUSES.

Gil's question, which is the right one: how do we guarantee that every pipeline run
carries every verdict, rather than relying on someone remembering? The answer cannot
be "remember". It has to be that the ordinary way of running the pipeline is a thing
that *fails* when the decisions are missing or stale.

So this is the entry point. It will not run the pipeline unless it has a fresh dump
of every decision collection, and it stamps what it applied into
`outputs/verdict_sync_<session>.json` so a later run can prove what went in.

WHY A DUMP RATHER THAN A LIVE READ
----------------------------------
The review tool is a web page on claude.ai and its decisions live in that artifact's
shared database, NOT in the HTML file. A page in a browser sandbox cannot write to
`F:\\`, and this script has no credentials to read claude.ai. Claude has both: it
reads the database with its artifact tooling and writes the dumps here. That hop is
irreducible without a local server; everything after it is mechanical, which is what
this script makes true.

Ask Claude for: "dump the review tool verdicts for <session>", which writes

    <dumps>/verdicts.json          unit GOOD / MUA / NOISE          -> cluster_group.tsv
    <dumps>/trial_selections.json  per-unit stability windows       -> unit_trial_selection.json
    <dumps>/merge_verdicts.json    per-pair MERGE/SEPARATE/UNSURE   -> unit_merges.json

then run this. --allow_missing names a collection you have deliberately decided not
to apply; it is recorded in the stamp as an explicit omission rather than silence.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "outputs")

# collection -> (dump filename, what it becomes on disk)
COLLECTIONS = {
    "verdicts": ("verdicts.json", "cluster_info.tsv group column (unit GOOD/MUA/NOISE)"),
    "trial_selections": ("trial_selections.json", "unit_trial_selection.json (stability windows)"),
    "merge_verdicts": ("merge_verdicts.json", "unit_merges.json (merges)"),
}


def sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def rows_of(path):
    """Dumps arrive as a list of documents, a list of {id,data} rows, or a dict."""
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, dict):
        doc = list(doc.values())
    return [r.get("data", r) if isinstance(r, dict) else r for r in doc]


def run(args, label):
    print("\n--- %s" % label)
    print("    " + " ".join(str(a) for a in args))
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("FAILED: %s (exit %d). Nothing further was run." % (label, r.returncode))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--rec", required=True, help=r"the .rec folder, e.g. F:\Gil\Shamir\<sid>.rec")
    ap.add_argument("--dumps", required=True, help="directory holding the database dumps")
    ap.add_argument("--ks_subdir", default="kilosort4")
    ap.add_argument("--allow_missing", action="append", default=[], metavar="COLLECTION",
                    help="proceed without this collection, recorded as a deliberate "
                         "omission. Repeatable.")
    ap.add_argument("--no_pipeline", action="store_true",
                    help="apply the decisions but stop before running sync_pipeline")
    args = ap.parse_args()

    sid, py = args.session_id, sys.executable
    ks_dir = os.path.join(args.rec, sid + ".kilosort", args.ks_subdir)
    if not os.path.isdir(ks_dir):
        raise SystemExit("no Kilosort folder at %s" % ks_dir)

    # --- refuse first, act second ------------------------------------------------
    present, missing = {}, []
    for coll, (fname, becomes) in COLLECTIONS.items():
        path = os.path.join(args.dumps, fname)
        if os.path.exists(path):
            present[coll] = path
        elif coll in args.allow_missing:
            print("SKIPPING %-17s (--allow_missing) -> %s" % (coll, becomes))
        else:
            missing.append((coll, fname, becomes))

    if missing:
        print("\nREFUSING TO RUN -- these decisions are not here, so a run now would "
              "silently drop them:\n")
        for coll, fname, becomes in missing:
            print("  %-17s expected %s" % (coll, os.path.join(args.dumps, fname)))
            print("  %-17s would become %s" % ("", becomes))
        print("\nAsk Claude: \"dump the review tool verdicts for %s\".\n"
              "Or pass --allow_missing %s to record the omission deliberately."
              % (sid, " --allow_missing ".join(c for c, _, _ in missing)))
        return 2

    stamp = {
        "session_id": sid,
        "synced_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "kilosort_dir": ks_dir,
        "applied": {},
        "deliberately_skipped": sorted(set(args.allow_missing)),
    }
    for coll, path in present.items():
        rows = rows_of(path)
        stamp["applied"][coll] = {"dump": path, "sha1": sha1(path), "n_documents": len(rows)}
        print("  %-17s %4d document(s)  sha1 %s" % (coll, len(rows), sha1(path)[:12]))

    # --- apply, in dependency order ----------------------------------------------
    # stability windows first: write_unit_merges freezes the UNION of the parents'
    # windows at merge time, so it has to see today's selections before it runs.
    if "trial_selections" in present:
        run([py, os.path.join(SCRIPTS, "write_unit_trial_selection.py"),
             sid, args.rec, present["trial_selections"], "--ks_subdir", args.ks_subdir],
            "stability windows -> unit_trial_selection.json")

    if "merge_verdicts" in present:
        run([py, os.path.join(SCRIPTS, "write_unit_merges.py"),
             sid, args.rec, present["merge_verdicts"], "--from_verdicts",
             "--ks_subdir", args.ks_subdir],
            "merges -> unit_merges.json")

    if "verdicts" in present:
        run([py, os.path.join(SCRIPTS, "apply_unit_verdicts.py"),
             sid, args.rec, present["verdicts"], "--ks_subdir", args.ks_subdir],
            "unit verdicts -> cluster_info.tsv group column")

    stamp_path = os.path.join(OUT, "verdict_sync_%s.json" % sid)
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    print("\nstamped %s" % stamp_path)

    if args.no_pipeline:
        print("--no_pipeline: decisions applied, pipeline NOT run.")
        return 0

    run([py, "-m", "sync_pipeline.process_session", "--rec_folder", args.rec],
        "sync_pipeline")
    print("\nDone. synced/ now reflects every decision in the stamp above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
