# -*- coding: utf-8 -*-
"""Turn a per-document database dump into the three files sync_session.py wants.

    python scripts/write_verdict_dumps.py <session_id> --src <dir> [--also <dir>]

WHY THIS EXISTS
---------------
The review tool's decisions live in its artifact's database on claude.ai. Claude
reads them with its artifact tooling, which writes one JSON file per document:

    <src>/verdicts/<unit>.json
    <src>/merge_verdicts/<pair>.json
    <src>/trial_selections/<unit>.json

sync_session.py wants one file per collection, keyed by document id -- the shape
apply_unit_verdicts.py, write_unit_merges.py and write_unit_trial_selection.py all
read. Doing that by hand is how a collection goes missing, so it is a script.

It also writes manual_verdicts.csv (readable without a JSON parser) and a MANIFEST
with a sha256 per file, because a dump whose provenance is not recorded is a dump
nobody can later trust. --also copies the same set to a second directory, which is
how the durable copy lands next to the sort on Z.

The counts it prints are the whole point of running it: they are what you check
against the review tool before letting a pipeline touch anything.
"""
from __future__ import print_function

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil

COLLECTIONS = ("verdicts", "merge_verdicts", "trial_selections")


def load_collection(src, coll):
    """Read <src>/<coll>/*.json into {document id: document}."""
    out = {}
    pat = os.path.join(src, coll, "*.json")
    for path in sorted(glob.glob(pat)):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fh:
            out[doc_id] = json.load(fh)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tally(docs, field):
    counts = {}
    for d in docs.values():
        counts[d.get(field)] = counts.get(d.get(field), 0) + 1
    return counts


def write(session_id, src, dest, note, source):
    os.makedirs(dest, exist_ok=True)
    data = {c: load_collection(src, c) for c in COLLECTIONS}

    written = []
    for coll in COLLECTIONS:
        path = os.path.join(dest, coll + ".json")
        with open(path, "w") as fh:
            json.dump(data[coll], fh, indent=1, sort_keys=True)
        written.append(path)

    # The same verdicts in a form you can read with your eyes or a spreadsheet.
    csv_path = os.path.join(dest, "manual_verdicts.csv")
    with open(csv_path, "w", newline="") as fh:
        fh.write("unit,verdict,ts\n")
        for uid in sorted(data["verdicts"], key=lambda s: int(s)):
            d = data["verdicts"][uid]
            fh.write("%s,%s,%s\n" % (uid, d.get("verdict", ""), d.get("ts", "")))
    written.append(csv_path)

    vt = tally(data["verdicts"], "verdict")
    mt = tally(data["merge_verdicts"], "verdict")
    lines = [
        "VERDICT DUMP -- %s" % session_id,
        "written %s from the review tool's live database"
        % datetime.datetime.now().isoformat(timespec="seconds"),
        "source  %s" % (source or "(not recorded -- pass --source next time)"),
        "        collections %s" % ", ".join(COLLECTIONS),
        "",
        "unit verdicts      %4d   %s" % (len(data["verdicts"]), vt),
        "merge verdicts     %4d   %s" % (len(data["merge_verdicts"]), mt),
        "trial selections   %4d" % len(data["trial_selections"]),
        "",
    ]
    if note:
        lines += [note, ""]
    lines += ["sha256                                                            file"]
    for path in written:
        lines.append("%s  %s  %d bytes"
                     % (sha256(path), os.path.basename(path), os.path.getsize(path)))
    lines.append("")
    man = os.path.join(dest, "MANIFEST.txt")
    with open(man, "w") as fh:
        fh.write("\n".join(lines))
    written.append(man)

    print("%s -> %s" % (session_id, dest))
    print("    verdicts         %4d  %s" % (len(data["verdicts"]), vt))
    print("    merge_verdicts   %4d  %s" % (len(data["merge_verdicts"]), mt))
    print("    trial_selections %4d" % len(data["trial_selections"]))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--src", required=True,
                    help="directory holding <collection>/<doc id>.json, as Claude's "
                         "artifact database read writes it")
    ap.add_argument("--dest", default=None,
                    help="where the dumps go (default outputs/review_verdicts/<sid>)")
    ap.add_argument("--also", action="append", default=[],
                    help="copy the same files here too, e.g. the review/verdicts "
                         "folder next to the sort on Z. Repeatable.")
    ap.add_argument("--note", default="",
                    help="a line for the MANIFEST, e.g. whether the review is finished")
    ap.add_argument("--source", default="",
                    help="the review artifact's URL, recorded in the MANIFEST. These "
                         "URLs are in no repo file, so a dump that does not carry its "
                         "own is a dump nobody can trace back.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = args.dest or os.path.join(root, "outputs", "review_verdicts", args.session_id)
    written = write(args.session_id, args.src, dest, args.note, args.source)

    for extra in args.also:
        os.makedirs(extra, exist_ok=True)
        for path in written:
            shutil.copy2(path, os.path.join(extra, os.path.basename(path)))
        print("    also copied to %s" % extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
