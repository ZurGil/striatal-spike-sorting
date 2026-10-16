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

SCORING THE MERGED UNIT -- YOU DO IT, AND YOUR ANSWER WINS
----------------------------------------------------------
A merged unit cannot honestly be scored before it exists: you have to see its combined
waveform, ACG and rate over the session first. So scoring is a SECOND pass, and it
overrides everything the pipeline would have guessed.

    merge it         ->  re-run the pipeline  ->  LOOK at the merged unit  ->  score it

Set it up front if you already know:

    {"merges": [{"parents": [38, 163], "quality_label": "good", "note": "..."}]}

or, the normal case, score it afterwards without rebuilding the manifest -- this edits
the existing file in place and leaves the frozen kept_ranges, ids, dates and notes
exactly as they were:

    python write_unit_merges.py <session_id> <rec_root> --score 10000=good

If you do not score it, the pipeline inherits the parents' shared label, or falls back
to `mua` when they disagree. That fallback is deliberately pessimistic and WILL mark
good merges as mua -- it is a placeholder, not a verdict, and
`unit_merges.quality_label_source` tells you which you are looking at
(`manual` / `inherited` / `conservative_mua`).

Only "good" and "mua" are accepted. "noise" is not a score for a merged unit -- deciding
it is noise is deciding the merge should not exist, so delete the entry instead.

INPUT: a small JSON file naming the groups to merge. Either

    {"merges": [{"parents": [38, 163], "note": "drift pair, CCG dip at 0",
                 "quality_label": "good"}]}

or the bare form

    [[38, 163], [204, 211, 390]]

Merged ids are allocated automatically from 10000 upward (MERGED_ID_BASE), skipping
anything already taken, so a merged unit is recognisable on sight and can never be
confused with a Kilosort cluster id. Pass "merged_id" explicitly in an entry to pin it.

Usage:
  python write_unit_merges.py <session_id> <rec_root> <merges.json> [--ks_subdir kilosort4]
  python write_unit_merges.py <session_id> <rec_root> --score 10000=good [--score 10001=mua]
  e.g. python write_unit_merges.py 20260916_110311 F:\\Gil\\Shamir\\20260916_110311.rec \\
            D:\\Gil\\spike_sorting_agent\\outputs\\merges_20260916_110311.json

Then re-run the pipeline for the session. Nothing extra to pass -- the file is found by
convention, and session_meta.json reports unit_merges.status = applied. That applies to
a re-score too: the label reaches your dataset on the next run, not when you set it.
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
MERGES_FILENAME = "unit_merges.json"
# Same vocabulary as units.quality_label, so a merged unit answers
# `quality_label == "good"` exactly like any other unit. See
# sync_pipeline/unit_merges.py MERGE_LABELS -- the two must agree.
MERGE_LABELS = ("good", "mua")


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


def check_label(label, where):
    """One place for the label vocabulary, so every entry point rejects the same typos."""
    if label is None:
        return None
    clean = str(label).strip().lower()
    if clean not in MERGE_LABELS:
        extra = ("Labelling a merged unit 'noise' is not a score -- it is a decision "
                 "that the merge should not exist. Delete the entry instead."
                 if clean == "noise" else
                 "Leave it unset to inherit the parents' label.")
        raise SystemExit("%s: quality_label=%r is not one of %s. %s"
                         % (where, label, list(MERGE_LABELS), extra))
    return clean


def from_verdicts(doc):
    """Turn a raw `merge_verdicts` dump from the review tool into merge groups.

    Saves a hop. The review tool writes one document per PAIR, because a pair is what
    you look at and decide; the pipeline needs GROUPS, because a unit cannot parent two
    different merges. Accept 38+163 and 163+204 and that is one merge of three, not
    two merges -- so accepted pairs are chained transitively here (union-find), exactly
    as the tool's own export button does.

    Accepts whatever shape the dump arrives in: a bare list of documents, a list of
    {"id","data"} rows, or a dict keyed by pair id. Anything whose verdict is not
    "merge" is counted and ignored -- a SEPARATE is a real decision, it just has no
    representation in a file that only lists merges.
    """
    if isinstance(doc, dict):
        rows = list(doc.values()) if "merges" not in doc else doc["merges"]
    else:
        rows = list(doc)
    rows = [r.get("data", r) if isinstance(r, dict) else r for r in rows]

    accepted, other = [], 0
    for r in rows:
        if not isinstance(r, dict) or "unit_a" not in r:
            continue
        if str(r.get("verdict", "")).lower() == "merge":
            accepted.append(r)
        else:
            other += 1

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in accepted:
        for u in (int(r["unit_a"]), int(r["unit_b"])):
            parent.setdefault(u, u)
        ra, rb = find(int(r["unit_a"])), find(int(r["unit_b"]))
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for u in parent:
        groups.setdefault(find(u), []).append(u)

    merges = []
    for root in sorted(groups, key=lambda r: min(groups[r])):
        members = sorted(groups[root])
        inside = [r for r in accepted
                  if int(r["unit_a"]) in members and int(r["unit_b"]) in members]
        labels = [r.get("quality_label") for r in inside if r.get("quality_label")]
        notes = [r.get("note") for r in inside if r.get("note")]
        pairs = [r.get("pair_id") or ("%s_%s" % (r["unit_a"], r["unit_b"])) for r in inside]
        m = {"parents": members,
             "note": "; ".join(notes) if notes else
                     "merge review in the unit review tool; pairs " + ", ".join(pairs)}
        if labels:
            # several pairs in one group may each carry a label; the last one wins,
            # and they are all in the note so a disagreement stays visible
            m["quality_label"] = labels[-1]
        merges.append(m)

    print("from the review tool's verdicts: %d pair(s) accepted -> %d merge group(s); "
          "%d pair(s) decided otherwise (not merges, nothing to write)."
          % (len(accepted), len(merges), other))
    for m in merges:
        print("   parents %s%s"
              % (" + ".join(str(p) for p in m["parents"]),
                 "  label %s" % m["quality_label"] if m.get("quality_label") else
                 "  (label inherited -- score it after you have seen it)"))
    return {"merges": merges}


def normalise(doc):
    """Accept {"merges": [{parents, note?, merged_id?, quality_label?}]} or [[a, b], ...]."""
    raw = doc.get("merges", doc) if isinstance(doc, dict) else doc
    out = []
    for entry in raw:
        if isinstance(entry, dict):
            parents = sorted(int(p) for p in entry["parents"])
            out.append({"parents": parents,
                        "merged_id": entry.get("merged_id"),
                        "note": entry.get("note"),
                        "decided": entry.get("decided"),
                        "quality_label": check_label(entry.get("quality_label"),
                                                     "merge %s" % parents)})
        else:
            out.append({"parents": sorted(int(p) for p in entry),
                        "merged_id": None, "note": None, "decided": None,
                        "quality_label": None})
    return out


def score_existing(ks_dir, scores):
    """Set `quality_label` on merges that already exist, editing the manifest in place.

    This is the second pass -- the one you run after looking at the merged unit, which
    is the only moment a score can honestly be given. Everything else in the file is
    left byte-for-byte as it was: the frozen kept_ranges above all, but also the ids,
    the dates and the notes. Re-running the whole script instead would recompute the
    union against today's stability windows, which is exactly what freezing exists to
    prevent.
    """
    path = os.path.join(ks_dir, MERGES_FILENAME)
    if not os.path.exists(path):
        raise SystemExit(
            "%s does not exist -- there are no merges to score. Write the manifest "
            "first (pass a merges.json)." % path)
    with open(path) as f:
        doc = json.load(f)
    by_id = {int(m["merged_id"]): m for m in doc.get("merges") or []}
    for mid, label in scores.items():
        if mid not in by_id:
            raise SystemExit(
                "%s has no merged unit %d. It holds: %s. (Merged ids start at %d; a "
                "parent id is not a merged id.)"
                % (path, mid, sorted(by_id) or "nothing", MERGED_ID_BASE))
        was = by_id[mid].get("quality_label")
        by_id[mid]["quality_label"] = label
        print("unit %d: quality_label %s -> %s  (parents %s)"
              % (mid, was if was else "(unset, inherited from parents)", label,
                 " + ".join(str(p) for p in by_id[mid]["parents"])))
    doc["scored"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print("\nupdated %s (kept_ranges, ids, dates and notes untouched)." % path)
    print("Re-run sync_pipeline for this session to put the new label in "
          "units.parquet.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root")
    ap.add_argument("input_json", nargs="?",
                    help="the groups to merge; omit it when using --score")
    ap.add_argument("--ks_subdir", default="kilosort4",
                    help="folder under <session>.kilosort/ holding spike_clusters.npy "
                         "(default kilosort4)")
    ap.add_argument("--score", action="append", default=[], metavar="MERGED_ID=LABEL",
                    help="score a merge that already exists, e.g. --score 10000=good. "
                         "Edits the manifest in place and leaves the frozen kept_ranges "
                         "alone. Repeatable. This is the normal way to label a merged "
                         "unit, because you have to see it first.")
    ap.add_argument("--from_verdicts", action="store_true",
                    help="input_json is a raw merge_verdicts dump from the review "
                         "tool's database (one document per PAIR) rather than merge "
                         "groups. Accepted pairs are chained transitively, SEPARATE "
                         "and UNSURE are ignored. Saves building an intermediate file.")
    ap.add_argument("--source", default="merge review pass")
    args = ap.parse_args()

    ks_dir = os.path.join(args.rec_root, args.session_id + ".kilosort", args.ks_subdir)

    # --- second pass: score merges that already exist ----------------------------
    if args.score:
        if args.input_json:
            raise SystemExit(
                "--score edits the existing manifest; do not also pass a merges.json "
                "(rebuilding it would recompute the frozen kept_ranges).")
        scores = {}
        for item in args.score:
            if "=" not in item:
                raise SystemExit("--score takes MERGED_ID=LABEL, e.g. 10000=good "
                                 "(got %r)." % item)
            mid, label = item.split("=", 1)
            try:
                mid = int(mid)
            except ValueError:
                raise SystemExit("--score %r: %r is not a merged unit id." % (item, mid))
            scores[mid] = check_label(label, "--score %s" % item)
        return score_existing(ks_dir, scores)

    if not args.input_json:
        raise SystemExit("pass a merges.json to create merges, or --score "
                         "MERGED_ID=LABEL to score ones that already exist.")
    with open(args.input_json) as f:
        doc = json.load(f)
    entries = normalise(from_verdicts(doc) if args.from_verdicts else doc)
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
            # null = nobody has scored the merged unit yet, so the pipeline will
            # inherit the parents' label (or fall back to mua). Set it here if you
            # already know, or afterwards with --score once you have looked at it.
            "quality_label": e["quality_label"],
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
        print("     unit %-6d = %-20s keeps %s  label %s"
              % (m["merged_id"], " + ".join(str(p) for p in m["parents"]), kept,
                 m["quality_label"] or "(unset -- inherited from the parents until you "
                 "--score it)"))
    unscored = [m["merged_id"] for m in merges if not m["quality_label"]]
    if unscored:
        print("\nNot yet scored: %s. Re-run the pipeline, look at the merged unit(s), "
              "then:\n  python write_unit_merges.py %s %s --score %d=good"
              % (", ".join(str(u) for u in unscored), args.session_id, args.rec_root,
                 unscored[0]))
    print("\nspike_clusters.npy was NOT modified -- Phy and bombcell will still show "
          "the parents.\nRe-run sync_pipeline for this session to apply it; "
          "session_meta.json will report\nunit_merges.status='applied'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
