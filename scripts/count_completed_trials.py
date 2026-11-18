# -*- coding: utf-8 -*-
"""How many COMPLETED trials does each recorded session have?

    python scripts/count_completed_trials.py
    python scripts/count_completed_trials.py --min 500

Gil's question after the confidence analysis: which sessions are big enough to run
it on? The answer is not `nTrials`. A session can log 1115 trials and give you 593
usable ones, because a trial where the rat broke fixation or withdrew early never
reached a choice and carries no decision to analyse. What every trial-count analysis
actually spends is

    completed = Custom.ChoiceCorrect is not NaN   (the rat poked a side port)

which is 45-55 % of nTrials on these rats. This walks every .rec folder of every rat,
finds its Bpod file the same way the sync pipeline does, and reports both numbers
plus whether the session is sorted and synced -- so "which sessions can I use" is one
table rather than six places to look.

Writes outputs/completed_trial_counts.csv.
"""
from __future__ import print_function

import argparse
import csv
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))          # sync_pipeline is a sibling checkout

RATS = [("Shamir", r"Z:\Gil\Shamir_1"), ("Popper", r"Z:\Gil\Popper"),
        ("Begin", r"Z:\Gil\Begin_1")]


def completed_of(path, cache_dir):
    from sync_pipeline import bpod_loader
    cache = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + ".pkl")
    s = bpod_loader.load_bpod_session(path, pkl_cache_path=cache)
    n = int(s["nTrials"])
    cc = s.get("Custom", {}).get("ChoiceCorrect")
    if cc is None:
        return n, 0
    v = np.asarray(cc, dtype=object).ravel()
    done = 0
    for i in range(min(n, v.size)):
        try:
            if np.isfinite(float(v[i])):
                done += 1
        except (TypeError, ValueError):
            pass
    return n, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=0,
                    help="only print sessions with at least this many completed trials")
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs", "bpod_cache"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    from sync_pipeline import session_discovery

    rows = []
    for rat, root in RATS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            rec = os.path.join(root, name)
            if not (name.endswith(".rec") and os.path.isdir(rec)):
                continue
            sid = name[:-4]
            ks = os.path.join(rec, sid + ".kilosort")
            sorted_ = os.path.isdir(os.path.join(ks, "kilosort4")) or any(
                d.startswith("kilosort4") for d in (os.listdir(ks) if os.path.isdir(ks) else []))
            synced = any(d.startswith("synced") for d in
                         (os.listdir(ks) if os.path.isdir(ks) else []))
            try:
                bpod = session_discovery.find_bpod_file_for_rec_folder(rec, root)
            except Exception as exc:
                rows.append(dict(rat=rat, session=sid, n_trials="", completed="",
                                 sorted="yes" if sorted_ else "no",
                                 synced="yes" if synced else "no",
                                 bpod_file="", note=str(exc)[:90]))
                continue
            try:
                n, done = completed_of(str(bpod), args.cache)
            except Exception as exc:
                rows.append(dict(rat=rat, session=sid, n_trials="", completed="",
                                 sorted="yes" if sorted_ else "no",
                                 synced="yes" if synced else "no",
                                 bpod_file=os.path.basename(str(bpod)), note=str(exc)[:90]))
                continue
            rows.append(dict(rat=rat, session=sid, n_trials=n, completed=done,
                             sorted="yes" if sorted_ else "no",
                             synced="yes" if synced else "no",
                             bpod_file=os.path.basename(str(bpod)), note=""))

    out = os.path.join(ROOT, "outputs", "completed_trial_counts.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rat", "session", "n_trials", "completed",
                                          "sorted", "synced", "bpod_file", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("%-8s %-30s %8s %10s %7s %7s" % ("rat", "session", "trials", "completed",
                                           "sorted", "synced"))
    big = 0
    for r in rows:
        c = r["completed"]
        if c == "" or (args.min and c < args.min):
            continue
        if c >= 500:
            big += 1
        print("%-8s %-30s %8s %10s %7s %7s" % (r["rat"], r["session"], r["n_trials"],
                                               r["completed"], r["sorted"], r["synced"]))
    have = [r for r in rows if r["completed"] != ""]
    print("\n%d recordings with a Bpod file; %d have >= 500 completed trials, "
          "%d have >= 400." % (len(have),
                               sum(1 for r in have if r["completed"] >= 500),
                               sum(1 for r in have if r["completed"] >= 400)))
    print("wrote", out)


if __name__ == "__main__":
    main()
