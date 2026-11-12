# -*- coding: utf-8 -*-
"""Collect the per-batch archives from the parallel worker roots into the real session
root, so consolidate_batches_session.py can see all of them at once.

    python scripts/gather_worker_batches.py <session_id> <rec_root> [--par F:\\Gil\\Shamir\\par]
                                            [--workers 4] [--expect_batches 47] [--dry_run]

WHY THIS EXISTS
---------------
The agent pipeline has to run one batch per fresh process (WORKFLOW.md 1e), and
run_pipeline() writes agent_processed\\agent_report.json and cluster_<label>.tsv at FIXED
names, so batches cannot share a session root without racing. Each worker therefore gets
its own root under <par>\\w<N>\\, and each batch archives itself into THAT root's
agent_processed\\batch_reports\\.

consolidate_batches_session.py, correctly, globs ONE root's batch_reports. So the
archives have to be brought together first. That gathering was done by hand the first
two times and never saved, which is exactly the habit WORKFLOW.md 1f-pre warns about --
so here it is, with the checks the hand version did not have.

WHAT IT CHECKS, AND WHY EACH CHECK EARNS ITS PLACE
--------------------------------------------------
  * Every expected batch number appears EXACTLY ONCE across all the worker roots. A
    missing batch is ten units silently absent from the consolidated report, which looks
    like nothing at all -- the report is simply short, and nobody counts. A duplicated
    batch number (two workers handed overlapping ranges) would have consolidation drop
    one copy as a "duplicate unit_id" and carry on.
  * The units covered by the gathered reports, unioned, are exactly the session's unit
    ids. This is the same question asked from the other end, and it is the one that
    catches an off-by-one in the batch ranges rather than a missing file.
  * Every worker's kilosort_original\\spike_clusters.npy is byte-identical to the one in
    the session root. run_pipeline builds kilosort_original per root from that root's own
    kilosort4\\; if one worker was set up from a different sort, its scores describe
    different units under the same ids, and nothing downstream would ever notice.

It copies kilosort_original\\ and the agent_processed\\ scaffolding from a worker into the
session root when they are not already there, since the session root itself usually never
had a batch run in it.

Safe to re-run: files are overwritten, and the exactly-once check is on batch numbers
found in the WORKER roots, so an existing gathered copy in the session root does not
count as a duplicate.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.session import discover_session


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("rec_root", help=r"the real session root, e.g. F:\Gil\Shamir\<sid>.rec")
    ap.add_argument("--par", default=r"F:\Gil\Shamir\par")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expect_batches", type=int, default=None,
                    help="how many batches there should be. Default: ceil(n_units / 10) "
                         "from the session's own spike_templates.npy.")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    sid = args.session_id
    paths = discover_session(args.rec_root)
    dest_ap = paths.agent_processed_dir
    dest_br = os.path.join(dest_ap, "batch_reports")

    st = np.load(os.path.join(paths.ks4_output_dir, "spike_templates.npy")).ravel()
    all_units = sorted(int(u) for u in np.unique(st))
    n_expected = args.expect_batches
    if n_expected is None:
        n_expected = (len(all_units) + args.batch_size - 1) // args.batch_size
    print("session %s: %d units -> %d batches of %d expected"
          % (sid, len(all_units), n_expected, args.batch_size))

    ref_clusters = os.path.join(paths.ks4_output_dir, "spike_clusters.npy")
    ref_sha = sha1(ref_clusters)

    # ---- find every batch archive, and whose root it is in --------------------
    found = {}            # batch number -> (worker, directory)
    dupes = []
    for w in range(1, args.workers + 1):
        wroot = os.path.join(args.par, "w%d" % w, sid + ".rec")
        wbr = os.path.join(wroot, sid + ".kilosort", "kilosort4",
                           "agent_processed", "batch_reports")
        if not os.path.isdir(wbr):
            print("  w%d: no batch_reports at %s" % (w, wbr))
            continue
        reports = sorted(glob.glob(os.path.join(wbr, "batch_*_report.json")))
        nums = []
        for r in reports:
            n = int(os.path.basename(r).split("_")[1])
            nums.append(n)
            if n in found:
                dupes.append((n, found[n][0], w))
            else:
                found[n] = (w, wbr)
        print("  w%d: %d batches %s" % (w, len(nums),
              ("%d..%d" % (min(nums), max(nums))) if nums else "-"))

        wks = os.path.join(wroot, sid + ".kilosort", "kilosort4", "spike_clusters.npy")
        if os.path.exists(wks):
            s = sha1(wks)
            if s != ref_sha:
                raise SystemExit(
                    "w%d was set up from a DIFFERENT sort than %s\n"
                    "  session root spike_clusters.npy sha1 %s\n"
                    "  w%d              spike_clusters.npy sha1 %s\n"
                    "Its scores describe different units under the same ids. Re-run "
                    "setup_pipeline_workers.ps1 for this session and redo those batches."
                    % (w, args.rec_root, ref_sha[:16], w, s[:16]))

    missing = [n for n in range(n_expected) if n not in found]
    print("\nbatches found: %d of %d" % (len(found), n_expected))
    if dupes:
        for n, a, b in dupes:
            print("  DUPLICATE batch %d in both w%d and w%d" % (n, a, b))
    if missing:
        print("  MISSING batches: %s" % missing)
    if dupes or missing:
        raise SystemExit(
            "refusing to gather an incomplete or overlapping set. A missing batch is ten "
            "units silently absent from the consolidated report -- the report is simply "
            "short, and nobody counts. Re-run the missing batches, or pass "
            "--expect_batches if the real count differs.")

    # ---- the same question from the other end: do the units add up? ----------
    covered = []
    for n, (w, wbr) in sorted(found.items()):
        with open(os.path.join(wbr, "batch_%03d_report.json" % n)) as fh:
            rep = json.load(fh)
        covered.extend(int(r["unit_id"]) for r in rep["per_unit_records"])
    cov_set = sorted(set(covered))
    if len(covered) != len(cov_set):
        from collections import Counter
        rep = [u for u, c in Counter(covered).items() if c > 1]
        raise SystemExit("units scored more than once across batches: %s" % sorted(rep)[:20])
    if cov_set != all_units:
        only_rep = sorted(set(cov_set) - set(all_units))
        only_ses = sorted(set(all_units) - set(cov_set))
        raise SystemExit(
            "the gathered batches do not cover the session's units exactly.\n"
            "  scored but not in the sort: %s\n"
            "  in the sort but never scored: %s"
            % (only_rep[:20] or "none", only_ses[:20] or "none"))
    print("units covered: %d, exactly the session's unit ids" % len(cov_set))

    if args.dry_run:
        print("\n--dry_run: nothing copied")
        return

    # ---- scaffolding, if this root never had a batch run in it ---------------
    src_w, _ = found[min(found)]
    src_root = os.path.join(args.par, "w%d" % src_w, sid + ".rec",
                            sid + ".kilosort", "kilosort4")
    if not os.path.isdir(paths.kilosort_original_dir):
        print("\ncopying kilosort_original from w%d (the session root never ran a batch)" % src_w)
        shutil.copytree(os.path.join(src_root, "kilosort_original"),
                        paths.kilosort_original_dir)
    else:
        print("\nkilosort_original already in the session root")
    os.makedirs(dest_br, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src_root, "agent_processed", "*"))):
        if os.path.isfile(f) and not os.path.exists(os.path.join(dest_ap, os.path.basename(f))):
            shutil.copy2(f, dest_ap)

    # ---- the archives themselves --------------------------------------------
    n_files = 0
    for n, (w, wbr) in sorted(found.items()):
        for f in sorted(glob.glob(os.path.join(wbr, "batch_%03d_*" % n))):
            shutil.copy2(f, dest_br)
            n_files += 1
    print("copied %d archive files for %d batches into\n  %s" % (n_files, len(found), dest_br))
    n_reports = len(glob.glob(os.path.join(dest_br, "batch_*_report.json")))
    print("the session root now holds %d batch reports" % n_reports)
    print("\nnext: python scripts/consolidate_batches_session.py %s %s" % (sid, args.rec_root))


if __name__ == "__main__":
    main()
