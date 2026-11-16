# -*- coding: utf-8 -*-
"""Did this session actually deliver catch trials? Read it from the Bpod file.

    python scripts/audit_catch_trials.py --rat Shamir --rat Popper

A catch trial is a correct choice with the reward deliberately withheld, so the
animal keeps waiting and its willingness to wait can be measured. Two things can
go wrong and they are different: the protocol may not have been configured to run
any (`PercentCatch` = 0), or it may have been configured and the trials may not
have behaved as catch trials. A flag in the file is not evidence that anything
happened, so this checks the behaviour as well as the setting:

    configured      TrialSettings.GUI.PercentCatch, and CatchError
    flagged         how many trials carry Custom.CatchTrial = 1
    delivered       of the CORRECT trials, what fraction of catch ones were
                    rewarded (should be 0) against non-catch ones (should be high),
                    and how long the animal waited in each case

Writes outputs/catch_trial_audit.csv and prints the table. Nothing is modified.
"""
from __future__ import print_function

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))          # sync_pipeline is a sibling checkout

RATS = {"Shamir": r"Z:\Gil\Shamir_1", "Popper": r"Z:\Gil\Popper",
        "Begin": r"Z:\Gil\Begin_1"}


def arr(custom, name, n):
    v = custom.get(name)
    if v is None:
        return np.full(n, np.nan)
    v = np.asarray(v, dtype=object).ravel()
    out = np.full(n, np.nan)
    for i in range(min(n, v.size)):
        try:
            out[i] = float(v[i])
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def audit(path, cache_dir):
    from sync_pipeline import bpod_loader
    cache = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + ".pkl")
    s = bpod_loader.load_bpod_session(path, pkl_cache_path=cache)
    n = int(s["nTrials"])
    custom = s.get("Custom", {})
    # TrialSettings is one entry per trial, each a dict with a GUI sub-dict. Settings
    # can be changed mid-session, so trial 1's value is the one to report and a change
    # is worth naming rather than averaging away.
    gui0, changed = {}, []
    ts = s.get("TrialSettings")
    if ts is not None and len(ts):
        first = ts[0]
        gui0 = first.get("GUI", first) if isinstance(first, dict) else {}
        last = ts[-1]
        gui_last = last.get("GUI", last) if isinstance(last, dict) else {}
        for key in ("PercentCatch", "CatchError"):
            if str(gui0.get(key)) != str(gui_last.get(key)):
                changed.append("%s %s->%s" % (key, gui0.get(key), gui_last.get(key)))

    def setting(name):
        v = gui0.get(name)
        try:
            return float(np.asarray(v).ravel()[0])
        except Exception:
            return None

    ct = arr(custom, "CatchTrial", n)
    cc = arr(custom, "ChoiceCorrect", n)
    rw = arr(custom, "Rewarded", n)
    ft = arr(custom, "FeedbackTime", n)

    corr = cc == 1
    m_catch, m_plain = corr & (ct == 1), corr & (ct == 0)
    # The OTHER half of the paradigm, and the one that is easy to lose. A time-
    # investment analysis needs how long the animal waits on ERROR trials as well as
    # on correct catch trials -- the two curves crossing is the whole measurement.
    # Whether error trials give you that is set by CatchError, not by PercentCatch:
    #   CatchError = 1  an error trial is a catch trial. Nothing is signalled, the
    #                   animal waits until it gives up, and the wait is its own.
    #   CatchError = 0  the error is signalled at the end of the feedback delay. The
    #                   "wait" you measure is then the delay WE drew, not a decision
    #                   the animal made, and it is useless for confidence.
    err = cc == 0
    row = {
        "bpod_file": os.path.basename(path),
        "n_trials": n,
        "percent_catch_setting": setting("PercentCatch"),
        "catch_error_setting": setting("CatchError"),
        "setting_changed_mid_session": "; ".join(changed),
        "n_flagged_catch": int(np.nansum(ct == 1)),
        "pct_flagged": round(100.0 * np.nansum(ct == 1) / n, 1) if n else None,
        "n_correct_catch": int(m_catch.sum()),
        "pct_rewarded_catch": (round(100.0 * np.nanmean(rw[m_catch]), 1)
                               if m_catch.sum() else None),
        "pct_rewarded_noncatch": (round(100.0 * np.nanmean(rw[m_plain]), 1)
                                  if m_plain.sum() else None),
        "median_wait_catch_s": (round(float(np.nanmedian(ft[m_catch])), 2)
                                if m_catch.sum() else None),
        "median_wait_noncatch_s": (round(float(np.nanmedian(ft[m_plain])), 2)
                                   if m_plain.sum() else None),
        "n_error": int(err.sum()),
        "median_wait_error_s": (round(float(np.nanmedian(ft[err])), 2)
                                if err.sum() else None),
    }
    # An error trial counts as giving a real wait when the animal waited about as long
    # as it does on a correct catch trial -- half of it is the threshold, which
    # separates 5-6 s from 1.1 s by a wide margin on every session here.
    mw_err, mw_catch = row["median_wait_error_s"], row["median_wait_catch_s"]
    if row["catch_error_setting"] == 0 or (mw_err is not None and mw_catch
                                           and mw_err < 0.5 * mw_catch):
        row["error_catch"] = "NO -- errors are signalled, no time investment on errors"
    elif mw_err is None:
        row["error_catch"] = "unknown"
    else:
        row["error_catch"] = "yes -- errors are catch trials too"
    # The verdict, stated once, from the behaviour rather than the flag.
    if not row["n_flagged_catch"]:
        row["verdict"] = "NO CATCH TRIALS"
    elif row["pct_rewarded_catch"] is None:
        row["verdict"] = "flagged, but no correct catch trial to judge"
    elif row["pct_rewarded_catch"] > 5:
        row["verdict"] = "FLAGGED BUT REWARDED -- not behaving as catch"
    elif (row["median_wait_catch_s"] or 0) <= (row["median_wait_noncatch_s"] or 0):
        row["verdict"] = "unrewarded but no longer wait -- check"
    else:
        row["verdict"] = "catch trials delivered"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat", action="append", default=[], choices=sorted(RATS))
    ap.add_argument("--match", default=None,
                    help="only Bpod files whose name contains this (e.g. Sep1)")
    ap.add_argument("--cache_dir", default=None,
                    help="where the .pkl caches go. Defaults to a temp dir rather than "
                         "next to the Bpod files, so an audit leaves Z unchanged.")
    args = ap.parse_args()
    rats = args.rat or sorted(RATS)
    cache = args.cache_dir or os.path.join(os.environ.get("TEMP", "."), "bpod_pkl_cache")
    os.makedirs(cache, exist_ok=True)

    rows = []
    for rat in rats:
        bpod = os.path.join(RATS[rat], "bpod")
        if not os.path.isdir(bpod):
            print("%s: no bpod folder" % rat)
            continue
        for f in sorted(os.listdir(bpod)):
            if not f.endswith(".mat") or "Session" not in f:
                continue
            if args.match and args.match not in f:
                continue
            try:
                r = audit(os.path.join(bpod, f), cache)
            except Exception as e:                      # one bad file must not stop the audit
                r = {"bpod_file": f, "verdict": "COULD NOT READ: %s" % e}
            r["rat"] = rat
            rows.append(r)
            print("%-7s %-46s n=%-5s CatchError=%-4s flagged=%-4s  wait: catch %-5s "
                  "error %-5s plain %-5s | %s"
                  % (rat, f, r.get("n_trials"), r.get("catch_error_setting"),
                     r.get("n_flagged_catch"), r.get("median_wait_catch_s"),
                     r.get("median_wait_error_s"), r.get("median_wait_noncatch_s"),
                     r.get("error_catch")))

    out = os.path.join(ROOT, "outputs", "catch_trial_audit.csv")
    keys = ["rat", "bpod_file", "n_trials", "percent_catch_setting", "catch_error_setting",
            "setting_changed_mid_session",
            "n_flagged_catch", "pct_flagged", "n_correct_catch", "pct_rewarded_catch",
            "pct_rewarded_noncatch", "median_wait_catch_s", "median_wait_noncatch_s",
            "n_error", "median_wait_error_s", "error_catch", "verdict"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s (%d session(s))" % (out, len(rows)))
    bad = [r for r in rows if r.get("verdict", "").startswith(("NO CATCH", "FLAGGED BUT"))]
    print("%d session(s) with no catch trials delivered at all" % len(bad))
    noerr = [r for r in rows if str(r.get("error_catch", "")).startswith("NO")]
    print("%d session(s) with NO ERROR CATCH TRIALS -- no time investment on errors, "
          "so no confidence analysis: %s"
          % (len(noerr), ", ".join(r["bpod_file"] for r in noerr) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
