# -*- coding: utf-8 -*-
"""Prove that every manual verdict actually reached `synced/`. Exits non-zero if not.

    python scripts/verify_verdicts_in_dataset.py <session_id> --rec <rec_folder> --dumps <dir>

WHY THIS EXISTS
---------------
Applying verdicts and checking they arrived are different acts, and the second one is
the one that was missing. 132 of Gil's 240 unit verdicts sat in the review tool for
weeks without reaching the dataset, because the script that was supposed to carry them
was hardcoded to a different session and still exited 0. Nothing anywhere compared the
decisions against the result.

So this reads the decisions from one side, reads the finished dataset from the other,
and compares them. It knows nothing about how the verdicts were applied and shares no
code with the scripts that apply them -- that independence is the point. If a writer
silently no-ops, or the pipeline is re-run from a stale manifest, or a verdict is made
against a clustering that has since changed, a check here fails.

It is wired into `sync_session.py` so it runs on every sync. Run it standalone any time
to re-prove the current `synced/` against the current verdicts.

WHAT EACH CHECK WOULD CATCH
---------------------------
  T1 fingerprints      a verdict made against a different sorting (ids mean other cells)
  T2 good/mua labels    a verdict that did not change `quality_label`
  T3 noise exclusions   a unit called noise that still has rows in the dataset
  T4 untouched units    a unit nobody judged whose label moved anyway
  T5 merges             a merge missing, mis-grouped, or with parents left behind
  T6 separates          two units merged although they were judged separate
  T7 trial windows      a stability window that did not reach unit_trial_selection
  T8 accounting         any verdict not accounted for by one of the above
  T9 referential        spikes for a unit that is not in the units table

EXIT CODES: 0 every check passed. 1 at least one FAILED. 2 could not run (missing input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

MERGED_ID_BASE = 10000
# The review tool counts DIO trial-start markers; the pipeline counts Bpod-matched
# trials. They can differ by a marker or two at the session edges, which is not an
# error. The RULE (is trial_start inside a kept window) is what gets compared exactly;
# the tool's own count is reported alongside and allowed to differ by this much.
TRIAL_COUNT_TOLERANCE = 3


class Report:
    """Collects check results so one run reports everything, not just the first fault."""

    def __init__(self):
        self.rows = []

    def add(self, tid, name, ok, detail, offenders=None):
        self.rows.append((tid, name, bool(ok), detail, offenders or []))

    def failed(self):
        return [r for r in self.rows if not r[2]]

    def render(self):
        print("")
        print("=" * 78)
        for tid, name, ok, detail, offenders in self.rows:
            print("%-4s %-9s %-34s %s" % (tid, "PASS" if ok else "FAILED", name, detail))
            for line in offenders[:12]:
                print("         %s" % line)
            if len(offenders) > 12:
                print("         ... and %d more" % (len(offenders) - 12))
        print("=" * 78)
        bad = self.failed()
        if bad:
            print("RESULT: %d of %d checks FAILED -- synced/ does NOT match the verdicts."
                  % (len(bad), len(self.rows)))
        else:
            print("RESULT: all %d checks passed -- every verdict is in synced/."
                  % len(self.rows))
        return 1 if bad else 0


# --------------------------------------------------------------------------- loaders
# Each accepts the shapes a dump can arrive in: a dict keyed by id, a bare list of
# documents, or a list of {"id","data"} rows. The review tool's own export has changed
# shape once already, and a loader that only understands today's shape is a check that
# silently stops checking.

def _rows(doc):
    if isinstance(doc, dict):
        inner = doc.get("selections", doc.get("merges", doc))
        if isinstance(inner, dict):
            return [(k, (val.get("data", val) if isinstance(val, dict) else val))
                    for k, val in inner.items()]
        doc = inner
    out = []
    for r in doc:
        if isinstance(r, dict):
            data = r.get("data", r)
            out.append((str(r.get("id", data.get("unit_id", data.get("pair_id", "")))), data))
    return out


def load_verdicts(path):
    out = {}
    for key, data in _rows(json.load(open(path))):
        v = str((data or {}).get("verdict", "")).strip().lower()
        if v in ("good", "mua", "noise"):
            out[int(key)] = v
        elif v:
            raise SystemExit("verdicts: unit %s has verdict %r" % (key, v))
    return out


def load_merge_verdicts(path):
    out = {}
    for key, data in _rows(json.load(open(path))):
        if not isinstance(data, dict) or "unit_a" not in data:
            continue
        out[str(key)] = data
    return out


def load_trial_selections(path):
    out = {}
    for key, data in _rows(json.load(open(path))):
        ranges = data.get("ranges") if isinstance(data, dict) else data
        ranges = [(float(r["t0"]), float(r["t1"])) for r in (ranges or [])]
        if ranges:                      # empty == keep everything == not listed
            out[int(key)] = {"ranges": ranges,
                             "n_markers_inside": (data or {}).get("n_markers_inside")}
    return out


def merge_groups(mverdicts):
    """Union-find over the ACCEPTED pairs -- recomputed here, independently of
    write_unit_merges.py, so a bug in its chaining shows up as a disagreement."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted = [d for d in mverdicts.values()
                if str(d.get("verdict", "")).lower() == "merge"]
    for d in accepted:
        a, b = int(d["unit_a"]), int(d["unit_b"])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for u in list(parent):
        groups.setdefault(find(u), []).append(u)
    return sorted([sorted(v) for v in groups.values()]), accepted


def sha1_of(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--ks_subdir", default="kilosort4")
    ap.add_argument("--synced_dir", default=None,
                    help="default <rec>/<session>.kilosort/synced")
    args = ap.parse_args()

    sid = args.session_id
    ks = os.path.join(args.rec, sid + ".kilosort", args.ks_subdir)
    synced = args.synced_dir or os.path.join(args.rec, sid + ".kilosort", "synced")

    need = {
        "verdicts": os.path.join(args.dumps, "verdicts.json"),
        "trial_selections": os.path.join(args.dumps, "trial_selections.json"),
        "merge_verdicts": os.path.join(args.dumps, "merge_verdicts.json"),
        "units.parquet": os.path.join(synced, "units.parquet"),
        "spikes.parquet": os.path.join(synced, "spikes.parquet"),
        "trials.parquet": os.path.join(synced, "trials.parquet"),
    }
    for label, p in need.items():
        if not os.path.exists(p):
            print("cannot verify: %s is missing (%s)" % (label, p), file=sys.stderr)
            return 2

    verdicts = load_verdicts(need["verdicts"])
    selections = load_trial_selections(need["trial_selections"])
    mverdicts = load_merge_verdicts(need["merge_verdicts"])
    groups, accepted = merge_groups(mverdicts)
    parents_of = {u: g for g in groups for u in g}

    units = pd.read_parquet(need["units.parquet"])
    trials = pd.read_parquet(need["trials.parquet"])
    spike_units = pd.read_parquet(need["spikes.parquet"], columns=["unit_id"])["unit_id"]
    present = dict(zip(units["unit_id"].astype(int), units["quality_label"].astype(str)))
    spike_unit_ids = set(int(u) for u in pd.unique(spike_units))

    print("session        %s" % sid)
    print("synced/        %s" % synced)
    print("dumps          %s" % args.dumps)
    print("decisions      %d unit verdicts | %d merge pairs (%d accepted -> %d groups) "
          "| %d stability windows"
          % (len(verdicts), len(mverdicts), len(accepted), len(groups), len(selections)))
    print("dataset        units.parquet %d rows | trials.parquet %d rows | "
          "spikes.parquet %d distinct unit ids"
          % (len(units), len(trials), len(spike_unit_ids)))

    R = Report()

    # --- T1 fingerprints -----------------------------------------------------------
    # Unit ids belong to one sorting. If spike_clusters.npy has changed since a verdict
    # was made, that verdict now names a different cell and must not be applied.
    live = sha1_of(os.path.join(ks, "spike_clusters.npy"))
    bad = []
    for fname in ("unit_merges.json", "unit_trial_selection.json"):
        p = os.path.join(ks, fname)
        if os.path.exists(p):
            got = (json.load(open(p)).get("clustering") or {}).get("spike_clusters_sha1")
            if got != live:
                bad.append("%s stamped %s, clustering is %s" % (fname, got, live))
    for pid, d in sorted(mverdicts.items()):
        got = d.get("spike_clusters_sha1")
        if got and got != live:
            bad.append("merge verdict %s was made against clustering %s" % (pid, got))
    R.add("T1", "fingerprints", not bad,
          "manifests and merge verdicts all stamped %s" % live[:12], bad)

    # --- T2 good / mua verdicts ----------------------------------------------------
    bad = []
    checked = 0
    for u, v in sorted(verdicts.items()):
        if v == "noise" or u in parents_of:
            continue                      # T3 and T5 own those
        checked += 1
        if u not in present:
            bad.append("unit %d verdict %s but absent from units.parquet" % (u, v))
        elif present[u] != v:
            bad.append("unit %d verdict %s but units.parquet says %s" % (u, v, present[u]))
    R.add("T2", "good/mua labels", not bad,
          "%d verdicts carried into quality_label" % checked, bad)

    # --- T3 noise exclusions -------------------------------------------------------
    noise = sorted(u for u, v in verdicts.items() if v == "noise")
    bad = []
    for u in noise:
        if u in present:
            bad.append("unit %d judged noise but present in units.parquet as %s"
                       % (u, present[u]))
        if u in spike_unit_ids:
            bad.append("unit %d judged noise but has rows in spikes.parquet" % u)
    R.add("T3", "noise exclusions", not bad,
          "%d noise verdicts, all absent from units and spikes" % len(noise), bad)

    # --- T4 units nobody judged ----------------------------------------------------
    # Their automatic label must be untouched. The pre-verdict table is the baseline;
    # without it this check cannot run, and says so rather than passing vacuously.
    pre = os.path.join(ks, "cluster_info.tsv.preverdicts")
    if os.path.exists(pre):
        rows = open(pre).read().splitlines()
        head = rows[0].split("\t")
        ci, gi = head.index("cluster_id"), head.index("group")
        auto = {}
        for line in rows[1:]:
            if line.strip():
                c = line.split("\t")
                auto[int(c[ci])] = c[gi].strip()
        bad, checked = [], 0
        for u, lab in sorted(auto.items()):
            if u in verdicts or u in parents_of:
                continue
            checked += 1
            if lab == "noise":
                if u in present:
                    bad.append("unit %d was automatically noise but is in units.parquet" % u)
            elif u not in present:
                bad.append("unit %d (auto %s) vanished from units.parquet" % (u, lab))
            elif present[u] != lab:
                bad.append("unit %d nobody judged: auto %s -> dataset %s"
                           % (u, lab, present[u]))
        R.add("T4", "unjudged units untouched", not bad,
              "%d unjudged units keep their automatic label" % checked, bad)
    else:
        R.add("T4", "unjudged units untouched", False,
              "cannot check: %s is missing, so there is no record of the automatic "
              "labels to compare against" % pre)

    # --- T5 merges -----------------------------------------------------------------
    bad = []
    mpath = os.path.join(ks, "unit_merges.json")
    manifest_groups = []
    if groups and not os.path.exists(mpath):
        bad.append("%d merge group(s) accepted but %s does not exist" % (len(groups), mpath))
    elif os.path.exists(mpath):
        doc = json.load(open(mpath))
        manifest_groups = sorted([sorted(int(p) for p in m["parents"])
                                  for m in doc.get("merges", [])])
        if manifest_groups != groups:
            bad.append("manifest groups %s != groups from the verdicts %s"
                       % (manifest_groups, groups))
        label_of = {tuple(sorted(int(p) for p in m["parents"])): m for m in doc.get("merges", [])}
        for g in groups:
            m = label_of.get(tuple(g))
            if m is None:
                continue
            mid = int(m["merged_id"])
            if mid not in present:
                bad.append("merged unit %d (%s) missing from units.parquet"
                           % (mid, "+".join(map(str, g))))
            for p in g:
                if p in present:
                    bad.append("parent %d still in units.parquet after merging into %d"
                               % (p, mid))
                if p in spike_unit_ids:
                    bad.append("parent %d still has rows in spikes.parquet "
                               "(spikes were not re-attributed to %d)" % (p, mid))
            # a label Gil set on the pair must be the label the merged unit carries
            want = None
            for d in accepted:
                if int(d["unit_a"]) in g and int(d["unit_b"]) in g and d.get("quality_label"):
                    want = str(d["quality_label"]).lower()
            if want and mid in present and present[mid] != want:
                bad.append("merged unit %d: you scored it %s, dataset says %s"
                           % (mid, want, present[mid]))
    R.add("T5", "merges", not bad,
          "%d merge group(s) applied, parents absent, labels as scored" % len(groups), bad)

    # --- T6 separate verdicts ------------------------------------------------------
    bad = []
    seps = [d for d in mverdicts.values() if str(d.get("verdict", "")).lower() != "merge"]
    for d in seps:
        a, b = int(d["unit_a"]), int(d["unit_b"])
        ga, gb = parents_of.get(a), parents_of.get(b)
        if ga is not None and gb is not None and ga == gb:
            bad.append("units %d and %d judged %s but ended up in the same merge %s"
                       % (a, b, d.get("verdict"), ga))
    R.add("T6", "separates respected", not bad,
          "%d pair(s) judged not-a-merge, none merged" % len(seps), bad)

    # --- T7 stability windows ------------------------------------------------------
    bad = []
    sel_path = os.path.join(synced, "unit_trial_selection.parquet")
    if selections and not os.path.exists(sel_path):
        bad.append("%d stability window(s) set but %s does not exist -- this pipeline "
                   "run did not apply them" % (len(selections), sel_path))
    elif os.path.exists(sel_path):
        sel = pd.read_parquet(sel_path)
        start = trials.set_index("trial_id")["trial_start_trodes"]
        for u, info in sorted(selections.items()):
            target = u
            if u in parents_of:
                # the parent is gone; its windows live on in the merged unit's frozen
                # union, which T5 already checked exists. Covered there, not here.
                continue
            rows = sel[sel["unit_id"] == target]
            if rows.empty:
                bad.append("unit %d has a stability window but no rows in "
                           "unit_trial_selection.parquet" % u)
                continue
            want = pd.Series(False, index=start.index)
            for t0, t1 in info["ranges"]:
                want |= (start >= t0) & (start <= t1)
            got = rows.set_index("trial_id")["selected"].reindex(start.index)
            if got.isna().any():
                bad.append("unit %d: %d trial(s) have no selected flag"
                           % (u, int(got.isna().sum())))
                continue
            dis = int((got.astype(bool) != want).sum())
            if dis:
                bad.append("unit %d: %d trial(s) disagree with the windows you marked"
                           % (u, dis))
            tool_n = info.get("n_markers_inside")
            if tool_n is not None:
                delta = abs(int(want.sum()) - int(tool_n))
                if delta > TRIAL_COUNT_TOLERANCE:
                    bad.append("unit %d: the tool counted %s trials inside the windows, "
                               "the dataset selects %d (differ by %d, tolerance %d)"
                               % (u, tool_n, int(want.sum()), delta, TRIAL_COUNT_TOLERANCE))
    R.add("T7", "stability windows", not bad,
          "%d restricted unit(s) match the windows trial by trial" % len(selections), bad)

    # --- T8 accounting -------------------------------------------------------------
    # Every decision must be explained by one of the checks above. This is the check
    # that would have caught the original fault: 240 verdicts, 108 reflected, and
    # nothing anywhere noticing the other 132.
    accounted, unaccounted = 0, []
    for u, v in sorted(verdicts.items()):
        if v == "noise":
            accounted += 1 if u not in present else 0
            if u in present:
                unaccounted.append("noise verdict on unit %d not applied" % u)
        elif u in parents_of:
            accounted += 1          # represented by its merge, checked in T5
        elif present.get(u) == v:
            accounted += 1
        else:
            unaccounted.append("unit %d verdict %s is not in the dataset" % (u, v))
    R.add("T8", "every verdict accounted for", not unaccounted,
          "%d/%d unit verdicts reached the dataset" % (accounted, len(verdicts)),
          unaccounted)

    # --- T9 referential integrity --------------------------------------------------
    orphan = sorted(spike_unit_ids - set(present))
    R.add("T9", "spikes reference real units", not orphan,
          "%d distinct unit ids in spikes.parquet, all present in units.parquet"
          % len(spike_unit_ids),
          ["spikes.parquet has unit %d, units.parquet does not" % u for u in orphan])

    return R.render()


if __name__ == "__main__":
    sys.exit(main())
