# -*- coding: utf-8 -*-
"""Overlay the review tool's per-unit verdicts onto `cluster_info.tsv`.

    python scripts/apply_unit_verdicts.py <session_id> <rec_folder> <verdicts.json>

`sync_pipeline.units_spikes.build_units_table()` reads `cluster_info.tsv` from the
Kilosort folder and takes its `group` column as the unit's quality label (and as the
noise filter). So that column is where a manual GOOD/MUA/NOISE verdict has to land.

This is the unit-verdict counterpart of `write_unit_merges.py`, and deliberately the
same shape: a plain documented file in a conventional place, which the pipeline reads
and the review tool never touches.

It replaces `export_phy_curation.py` for this purpose. That script was written for
session 20260901_085606, writes into that session's `agent_processed/` folder, and
reads verdicts from a Claude scratchpad directory that no longer exists -- so it
cannot carry verdicts for any other session, which is easy to miss because it still
runs.

WHAT IT DOES, EXACTLY
---------------------
- Takes a dump of the artifact database's `verdicts` collection: documents keyed by
  unit id, each with a `verdict` of good / mua / noise.
- Rewrites ONLY the `group` cell of the rows it has a verdict for. Every other column
  (`ch`, `depth`, `KSLabel`) and every row it has no verdict for is passed through
  byte-identical.
- Keeps one backup, `cluster_info.tsv.preverdicts`, written only the first time, so
  the automatic classification this started from is always recoverable.
- Prints every change it makes, and a count of verdicts that AGREED with what was
  already there (those are not silent -- an agreeing verdict is still a reviewed
  unit, and the difference matters when you ask how much of the session is reviewed).
"""
import argparse
import json
import os
import shutil
import sys

VALID = ("good", "mua", "noise")
BACKUP_SUFFIX = ".preverdicts"


def load_verdicts(path):
    """Accept a list of documents, a list of {id,data} rows, or a dict keyed by id."""
    with open(path) as f:
        doc = json.load(f)
    out = {}
    if isinstance(doc, dict):
        items = list(doc.items()) if "verdict" not in doc else []
        for k, v in items:
            out[str(k)] = (v.get("data", v) if isinstance(v, dict) else v)
    else:
        for row in doc:
            if not isinstance(row, dict):
                continue
            uid = row.get("id", row.get("unit_id"))
            data = row.get("data", row)
            if uid is None:
                uid = data.get("unit_id")
            if uid is not None:
                out[str(uid)] = data
    clean = {}
    for uid, data in out.items():
        v = str((data or {}).get("verdict", "")).strip().lower()
        if v in VALID:
            clean[str(int(uid))] = v
        elif v:
            raise SystemExit("unit %s: verdict %r is not one of %s" % (uid, v, list(VALID)))
    return clean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root", help=r"the .rec folder, e.g. F:\Gil\Shamir\<sid>.rec")
    ap.add_argument("verdicts_json")
    ap.add_argument("--ks_subdir", default="kilosort4")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ks = os.path.join(args.rec_root, args.session_id + ".kilosort", args.ks_subdir)
    tsv = os.path.join(ks, "cluster_info.tsv")
    if not os.path.exists(tsv):
        raise SystemExit("%s not found -- sync_pipeline requires it; build it with "
                         "scripts/build_cluster_info_from_classification_session.py" % tsv)

    verdicts = load_verdicts(args.verdicts_json)
    print("%d manual verdict(s) in %s" % (len(verdicts), args.verdicts_json))

    with open(tsv) as f:
        lines = f.read().splitlines()
    header = lines[0].split("\t")
    if "cluster_id" not in header or "group" not in header:
        raise SystemExit("%s: expected cluster_id and group columns, found %s" % (tsv, header))
    ci, gi = header.index("cluster_id"), header.index("group")

    changed, agreed, unknown = [], 0, []
    out = [lines[0]]
    seen = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        uid = cells[ci].strip()
        seen.add(uid)
        if uid in verdicts:
            was, now = cells[gi], verdicts[uid]
            if was != now:
                changed.append((uid, was, now))
                cells[gi] = now
            else:
                agreed += 1
        out.append("\t".join(cells))

    for uid in verdicts:
        if uid not in seen:
            unknown.append(uid)
    if unknown:
        raise SystemExit("verdicts name %d unit(s) absent from %s: %s -- these verdicts "
                         "were made against a different clustering."
                         % (len(unknown), tsv, sorted(unknown, key=int)[:10]))

    print("  %d changed, %d already agreed, %d unit(s) left on their automatic label"
          % (len(changed), agreed, len(seen) - len(changed) - agreed))
    for uid, was, now in changed[:40]:
        print("     unit %-5s %-5s -> %s" % (uid, was, now))
    if len(changed) > 40:
        print("     ... and %d more" % (len(changed) - 40))

    if args.dry_run:
        print("--dry_run: nothing written.")
        return 0
    if not changed:
        print("nothing to write -- cluster_info.tsv already matches every verdict.")
        return 0

    backup = tsv + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(tsv, backup)
        print("  kept the pre-verdict table at %s" % backup)
    with open(tsv, "w", newline="") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s" % tsv)
    print("Re-run sync_pipeline for this session to carry the verdicts into "
          "units.parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
