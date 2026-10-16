"""
Writes the canonical `unit_merges.json` that sync_pipeline picks up automatically.

This is the merge counterpart of `write_unit_trial_selection.py`, and it works the same
way: the review tool never writes to your dataset, and sync_pipeline never reads the
review tool. In between sits one plain file, in one conventional place, in a documented
format you could equally well write by hand:

    <rec_root>/<session>.kilosort/kilosort4/unit_merges.json

WHAT A MERGE IS HERE (AND WHAT IT IS NOT)
-----------------------------------------
It is NOT what Phy does. Phy rewrites `spike_clusters.npy`: the parents stop existing,
ids are renumbered, and anything you had keyed to an old id silently points at the
wrong neuron. This writes a manifest instead. `spike_clusters.npy` is never modified,
so unit 38 is still unit 38 in the Kilosort folder forever, the clustering fingerprint
still matches, and your stability selections survive the merge. sync_pipeline applies
the manifest in memory while building `synced/`.

The cost, so it is never a surprise: open the Kilosort folder in Phy or bombcell after
this and you will see the PARENTS, unmerged. The merged view lives in `synced/`.

THE FROZEN UNION
----------------
A merged unit is present wherever ANY parent was present -- parent a kept 1-5,40-60
plus parent b kept 10-20,50-70 gives 1-5,10-20,50-70. Only stretches absent in every
parent drop out.

This script computes that union ONCE, from `unit_trial_selection.json` as it stands
right now, and writes it into the manifest as `kept_ranges`. It is then frozen: if you
later re-mark a parent's windows, a merged unit you already signed off does NOT change
underneath you. Each parent's own ranges are written alongside, under `parent_ranges`,
so the union is always auditable and you can deliberately recompute it (re-run this
script) if that is what you want.

INPUT: a small JSON file naming the groups to merge. Either

    {"merges": [{"parents": [38, 163], "note": "drift pair, CCG dip at 0"}]}

or the bare form

    [[38, 163], [204, 211, 390]]

Merged ids are allocated automatically from 10000 upward (MERGED_ID_BASE), skipping
anything already taken, so a merged unit is recognisable on sight and can never be
confused with a Kilosort cluster id. Pass "merged_id" explicitly in an entry to pin it.

Usage:
  python write_unit_merges.py <session_id> <rec_root> <merges.json> [--ks_subdir kilosort4]
  e.g. python write_unit_merges.py 20260916_110311 F:\\Gil\\Shamir\\20260916_110311.rec \\
            D:\\Gil\\spike_sorting_agent\\outputs\\merges_20260916_110311.json

Then re-run the pipeline for the session. Nothing extra to pass -- the file is found by
convention, and session_meta.json reports unit_merges.status = applied.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

FORMAT = "unit_merges/1"
MERGED_ID_BASE = 10000
SELECTION_FILENAME = "unit_trial_selection.json"


def fingerprint(ks_dir):
    """Identify the clustering these parent ids belong to (sha1 + counts)."""
    path = os.path.join(ks_dir, "spike_clusters.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found -- point --ks_subdir at the folder whose spike_clusters.npy "
            "defines the unit ids the merges refer to." % path)
    with open(path, "rb") as f:
        raw = f.read()
    clusters = np.load(path).ravel()
    return ({"spike_clusters_sha1": hashlib.sha1(raw).hexdigest(),
             "n_spikes": int(len(clusters)), "n_clusters": int(len(np.unique(clusters)))},
            set(int(c) for c in np.unique(clusters)))


def union_of_ranges(ranges):
    """The merge rule: present wherever ANY parent was present.

    Kept deliberately identical to sync_pipeline.unit_merges.union_of_ranges -- the
    pipeline re-derives it in its error messages, and the two must not drift apart.
    """
    if not ranges:
        return []
    ordered = sorted(({"t0": float(r["t0"]), "t1": float(r["t1"])} for r in ranges),
                     key=lambda r: r["t0"])
    out = [dict(ordered[0])]
    for r in ordered[1:]:
        if r["t0"] <= out[-1]["t1"]:
            out[-1]["t1"] = max(out[-1]["t1"], r["t1"])
        else:
            out.append(dict(r))
    return out


def load_selection_ranges(ks_dir):
    """Per-unit stability windows as they stand right now, or {} if none were made."""
    path = os.path.join(ks_dir, SELECTION_FILENAME)
    if not os.path.exists(path):
        return {}, None
    with open(path) as f:
        doc = json.load(f)
    out = {}
    for uid, entry in (doc.get("selections") or {}).items():
        ranges = [{"t0": float(r["t0"]), "t1": float(r["t1"])}
                  for r in (entry.get("ranges") or [])]
        if ranges:
            out[int(uid)] = ranges
    return out, path


def normalise(doc):
    """Accept either {"merges": [{parents, note?, merged_id?}]} or [[a, b], ...]."""
    raw = doc.get("merges", doc) if isinstance(doc, dict) else doc
    out = []
    for entry in raw:
        if isinstance(entry, dict):
            parents = sorted(int(p) for p in entry["parents"])
            out.append({"parents": parents,
                        "merged_id": entry.get("merged_id"),
                        "note": entry.get("note"),
                        "decided": entry.get("decided")})
        else:
            out.append({"parents": sorted(int(p) for p in entry),
                        "merged_id": None, "note": None, "decided": None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root")
    ap.add_argument("input_json")
    ap.add_argument("--ks_subdir", default="kilosort4",
                    help="folder under <session>.kilosort/ holding spike_clusters.npy "
                         "(default kilosort4)")
    ap.add_argument("--source", default="merge review pass")
    args = ap.parse_args()

    ks_dir = os.path.join(args.rec_root, args.session_id + ".kilosort", args.ks_subdir)
    with open(args.input_json) as f:
        entries = normalise(json.load(f))
    if not entries:
        print("No merges in %s -- nothing to write." % args.input_json, file=sys.stderr)
        return 1

    fp, existing_ids = fingerprint(ks_dir)
    selections, selection_path = load_selection_ranges(ks_dir)

    # --- allocate merged ids ------------------------------------------------------
    taken = set(existing_ids) | {e["merged_id"] for e in entries if e["merged_id"]}
    next_id = MERGED_ID_BASE
    for e in entries:
        if e["merged_id"] is None:
            while next_id in taken:
                next_id += 1
            e["merged_id"] = next_id
            taken.add(next_id)

    # --- validate before writing anything ----------------------------------------
    seen_parent = {}
    for e in entries:
        if len(e["parents"]) < 2:
            raise SystemExit("merge %s: needs at least 2 parents." % e["parents"])
        for p in e["parents"]:
            if p not in existing_ids:
                raise SystemExit(
                    "parent %d is not a cluster in %s. Check --ks_subdir and that these "
                    "ids come from this sorting." % (p, ks_dir))
            if p in seen_parent:
                raise SystemExit(
                    "unit %d appears in two merges (%s and %s). A unit can only be "
                    "merged into one place -- combine them into one entry with three "
                    "parents." % (p, seen_parent[p], e["parents"]))
            seen_parent[p] = e["parents"]

    # --- freeze the union --------------------------------------------------------
    merges = []
    for e in entries:
        parent_ranges = {str(p): selections[p] for p in e["parents"] if p in selections}
        kept = union_of_ranges([r for rs in parent_ranges.values() for r in rs])
        merges.append({
            "merged_id": e["merged_id"],
            "parents": e["parents"],
            # FROZEN at merge time. Re-marking a parent later will not change this.
            "kept_ranges": kept,
            "parent_ranges": parent_ranges,
            "decided": e["decided"] or datetime.date.today().isoformat(),
            "note": e["note"] or "",
        })

    doc = {
        "format": FORMAT,
        "session_id": args.session_id,
        "clock": "trodes_absolute_seconds",
        "clustering": fp,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "written_by": "scripts/write_unit_merges.py",
        "source": args.source,
        "stability_windows": "frozen_union_at_merge_time",
        "selection_file_used": selection_path,
        "note": ("Unit merges applied IN MEMORY by sync_pipeline while building "
                 "synced/. spike_clusters.npy is NEVER rewritten, so parent ids stay "
                 "valid and the clustering fingerprint keeps matching -- which is what "
                 "lets unit_trial_selection.json survive a merge. kept_ranges is the "
                 "union of the parents' stability windows, frozen when the merge was "
                 "made; parent_ranges records what it was built from. Read "
                 "automatically from this filename; see sync_pipeline/HOW_TO_RUN.md "
                 "section 6c."),
        "merges": merges,
    }
    out_path = os.path.join(ks_dir, "unit_merges.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    print("wrote %s" % out_path)
    print("  clustering fingerprint: sha1=%s n_spikes=%d n_clusters=%d"
          % (fp["spike_clusters_sha1"][:16], fp["n_spikes"], fp["n_clusters"]))
    print("  stability selection read from: %s" % (selection_path or "(none present)"))
    print("  %d merge(s):" % len(merges))
    for m in merges:
        kept = (", ".join("%.1f-%.1fs" % (r["t0"], r["t1"]) for r in m["kept_ranges"])
                if m["kept_ranges"] else "every trial (no parent had a selection)")
        print("     unit %-6d = %-20s keeps %s"
              % (m["merged_id"], " + ".join(str(p) for p in m["parents"]), kept))
    print("\nspike_clusters.npy was NOT modified -- Phy and bombcell will still show "
          "the parents.\nRe-run sync_pipeline for this session to apply it; "
          "session_meta.json will report\nunit_merges.status='applied'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
