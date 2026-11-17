# -*- coding: utf-8 -*-
"""One table of every sorted session, its verdicts, and whether they are safely saved.

    python scripts/session_status_report.py
    python scripts/session_status_report.py --live outputs/artifact_live_counts.json

WHAT THIS ANSWERS
-----------------
"Which sessions are sorted, which ones have my verdicts on them, and are the verdicts
saved?" -- without anybody having to remember. Everything is read from the files on Z
and from the review artifacts, never from a hand-kept list.

THE ONE THING IT CANNOT READ BY ITSELF
--------------------------------------
Verdicts live in each review page's database on claude.ai. Python here cannot reach
that; only the Artifact tooling can. So Claude dumps the live counts into a small JSON
(`--live`) and this folds them in, which makes the interesting comparison possible:

    verdicts in the live page   vs   verdicts in the file on Z

A difference means the page has decisions that exist in exactly one place -- a browser
database -- and a session in that state is one cleared cache away from losing work. The
table says DRIFTED for it rather than quietly showing the larger number.

Without --live the table still lists what is on Z; it just cannot say whether the page
has moved on since.

OUTPUT
    post_ks_correction/SESSION_STATUS.md     committed, so git carries the history
    outputs/session_status.json              the same thing for the dashboard
"""
import argparse
import collections
import csv
import json
import os
import re

RATS = [("Popper", r"Z:\Gil\Popper"),
        ("Shamir", r"Z:\Gil\Shamir_1"),
        ("Begin", r"Z:\Gil\Begin_1")]

# Review pages, per session. Not derivable from anything on disk -- a published
# artifact's URL exists only where someone wrote it down, so here is where it is
# written down.
ARTIFACTS = {
    "20261112_082028": "https://claude.ai/artifact/M54x4TFQzUXJZJWwMcycSG",
    "20260916_110311": "https://claude.ai/artifact/FS2ySxRCF6ecGRduMmA2Ap",
    "20260915_084812_103710_concat": "https://claude.ai/artifact/9wCjhAQncHPDuSYijKwgMb",
    "20260914_122553": "https://claude.ai/artifact/JgJrGazBZmQ6KtjsRGhsn8",
}
EXTRA_TOOLS = {
    "20260916_110311": [("Merge Adjudicator",
                         "https://claude.ai/artifact/DLDrEjL5Sn74TvBuMnmdcW")],
    "20261112_082028": [("Correlogram Atlas",
                         "https://claude.ai/artifact/69cKEsVzCAXDxrAz1KVtD5")],
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def n_units(k4):
    cg = os.path.join(k4, "cluster_group.tsv")
    if not os.path.exists(cg):
        return None
    try:
        with open(cg, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return None


def thresholds(k4):
    p = os.path.join(k4, "ops.npy")
    if not os.path.exists(p):
        return None
    try:
        import numpy as np
        o = np.load(p, allow_pickle=True).item()
        return "%s / %s" % (o.get("Th_universal"), o.get("Th_learned"))
    except Exception:
        return None


def provenance_lines(root, sid):
    """Report-ready notes for a session that is not what its name suggests.

    Gil, 2026-11-13: "if specific session is actually a combination of two sessions,
    you should have also comment for it ... what exactly happened, if something was
    trimmed or not. Those comments will be very important when we come to sync it with
    bpod."

    Everything here is read out of the session's own concat_*.json, so the note cannot
    disagree with what was actually sorted.
    """
    ks = os.path.join(root, sid + ".rec", sid + ".kilosort")
    if not os.path.isdir(ks):
        return []
    man = None
    for f in sorted(os.listdir(ks)):
        if f.startswith("concat_") and f.endswith(".json"):
            man = jload(os.path.join(ks, f))
            break
    if not man:
        return []
    out = []
    seam = man.get("seam", {})
    tot = man.get("total", {})
    out.append("JOINED SESSION -- %s was recorded as %d separate files and sorted ONCE. "
               "This is the sort of record for that day; neither part is sorted on its "
               "own." % (man.get("date", "?"), len(man.get("parts", []))))
    if seam.get("real_time_gap_text"):
        out.append("Gap between the parts: %s. The seam at sample %s is a break in TIME, "
                   "not in the array -- nothing that bins over time, computes a rate or "
                   "looks for drift may interpolate across it."
                   % (seam["real_time_gap_text"], "{:,}".format(seam.get("sample_index", 0))))
    for p in man.get("parts", []):
        a = int(p.get("offset_in_concat_samples", 0))
        b = a + int(p.get("used_samples", 0))
        line = ("part %s  %s  ->  concat samples %s to %s  (%s s of recording)"
                % (p.get("order"), p.get("session"), "{:,}".format(a), "{:,}".format(b),
                   p.get("used_seconds")))
        out.append(line)
        tr = p.get("trimmed")
        if tr:
            out.append("    TRIMMED: the last %s s of this part were dropped before "
                       "sorting, cut at t = %s s. Any Bpod trial that started after "
                       "that has NO ephys behind it. Reason: %s"
                       % (tr.get("dropped_seconds"), tr.get("cut_at_second"),
                          " ".join(tr.get("why", []))))
        if p.get("bpod"):
            have = os.path.exists(p["bpod"])
            out.append("    Bpod: %s%s" % (os.path.basename(p["bpod"]),
                                           "" if have else "   ** NOT FOUND **"))
    out.append("FOR THE BPOD SYNC: split spike_times at sample %s and sync each part "
               "against its own Bpod file. A spike at concat sample s belongs to part 1 "
               "at sample s when s < the seam, otherwise to part 2 at s minus the seam. "
               "Total %s samples, %s h."
               % ("{:,}".format(seam.get("sample_index", 0)),
                  "{:,}".format(tot.get("samples", 0)), tot.get("hours")))
    sup_txt = man.get("supersedes") or []
    if sup_txt:
        # The manifest stores this as an array of source lines, not as separate facts.
        out.append("SUPERSEDES: %s" % " ".join(s.strip() for s in sup_txt))
    return out


def concat_map(root):
    """(session -> joint session that supersedes it, joint session -> its parts).

    A standalone sort of a part is not a session awaiting review; it is a superseded
    result that must never have its unit ids mixed with the joint sort's. Without this
    the register listed 20260915_103710 as an ordinary unreviewed session and
    20260915_084812 as a sorting candidate, and the second of those got a GPU job
    queued for it on 2026-11-13 before Gil caught it.
    """
    sup, parts = {}, {}
    if not os.path.isdir(root):
        return sup, parts
    for name in os.listdir(root):
        if not name.endswith(".rec"):
            continue
        ks = os.path.join(root, name, name[:-4] + ".kilosort")
        if not os.path.isdir(ks):
            continue
        for f in os.listdir(ks):
            if f.startswith("concat_") and f.endswith(".json"):
                d = jload(os.path.join(ks, f)) or {}
                cid = d.get("concat_session_id") or name[:-4]
                ps = [p.get("session") for p in d.get("parts", []) if p.get("session")]
                parts[cid] = ps
                for p in ps:
                    sup[p] = cid
    return sup, parts


MONTHS = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
          "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}


def catch_audit_by_date():
    """outputs/catch_trial_audit.csv, keyed by (rat, YYYYMMDD).

    A session's Bpod file is named by date, so this is how the behavioural audit
    reaches the register: whether that day's protocol made ERROR trials catch
    trials, which is what decides whether a time-investment analysis is possible
    at all. It is kept out of the per-session scan because it is a property of the
    behaviour file, not of the sort.
    """
    path = os.path.join(ROOT, "outputs", "catch_trial_audit.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            m = re.search(r"_([A-Z][a-z]{2})(\d{1,2})_(\d{4})_Session(\d+)",
                          row.get("bpod_file", ""))
            if not m:
                continue
            key = (row["rat"], "%s%s%02d" % (m.group(3), MONTHS[m.group(1)], int(m.group(2))))
            out.setdefault(key, []).append(row)
    return out


def scan():
    out = []
    catch = catch_audit_by_date()
    for rat, root in RATS:
        if not os.path.isdir(root):
            continue
        sup, parts = concat_map(root)
        for name in sorted(os.listdir(root)):
            if not name.endswith(".rec"):
                continue
            sdir = os.path.join(root, name)
            if not os.path.isdir(sdir):
                continue
            sid = name[:-4]
            ks = os.path.join(sdir, sid + ".kilosort")
            # One recording can hold more than one sorting, and then it needs more than
            # one row: the second half of 15 September has both its own superseded
            # standalone sort and the joint sort restricted to it, and those are
            # different unit spaces with different states. Collapsing them into one row
            # made the table report the half as unreviewed while its dataset carried
            # all 322 verdicts.
            for sub in ("kilosort4", "kilosort4_from_concat"):
              k4 = os.path.join(ks, sub)
              if not os.path.exists(os.path.join(k4, "spike_clusters.npy")):
                continue
              vd = os.path.join(ks, "review", "verdicts")
              r = {
                "rat": rat, "session": sid, "date": sid[:8],
                "row_id": sid if sub == "kilosort4" else "%s [from concat]" % sid,
                "n_units": n_units(k4), "thresholds": thresholds(k4),
                "artifact": ARTIFACTS.get(sid),
                "tools": EXTRA_TOOLS.get(sid, []),
                "verdicts_dir": vd if os.path.isdir(vd) else None,
                "cluster_info": os.path.exists(os.path.join(k4, "cluster_info.tsv")),
                # Only the recording's own standalone sort can be superseded by the
                # joint one. The joint sort restricted to this half IS the joint sort.
                "superseded_by": sup.get(sid) if sub == "kilosort4" else None,
                "joins": parts.get(sid),
                "provenance": provenance_lines(root, sid),
                # Bpod sync is a different thing from applying verdicts. sync_session.py
                # writes a `synced/` directory; the verdicts go into cluster_info.tsv.
                # A session can have one without the other, and conflating them is how
                # "is it synced?" gets answered wrongly.
                # A synced/ directory belongs to ONE of the recording's sorts, so it is
                # attributed rather than just listed: synced_from_concat is the joint
                # sort's output and must not be credited to the standalone sort beside it.
                "bpod_synced": sorted(
                    d for d in (os.listdir(ks) if os.path.isdir(ks) else [])
                    if (d == "synced" or d.startswith("synced_"))
                    and ((d == "synced_from_concat") == (sub == "kilosort4_from_concat"))),
                "bpod_files": len(os.listdir(os.path.join(root, "bpod")))
                              if os.path.isdir(os.path.join(root, "bpod")) else 0,
              }
              v = jload(os.path.join(vd, "verdicts.json")) or {}
              m = jload(os.path.join(vd, "merge_verdicts.json")) or {}
              t = jload(os.path.join(vd, "trial_selections.json")) or {}
              r["z_verdicts"] = len(v)
              r["z_tally"] = dict(collections.Counter(x.get("verdict") for x in v.values()))
              r["z_merges"] = len(m)
              r["z_merge_tally"] = dict(collections.Counter(x.get("verdict") for x in m.values()))
              r["z_trials"] = len(t)
              # "verdicts applied" = a stamp exists saying which decisions went into this
              # sort. sync_session.py writes it to outputs/verdict_sync_<sid>.json; the
              # older per-session flow left a copy next to the verdicts on Z. Checking
              # only the Z copy reported sessions as unapplied right after applying them.
              stamp = jload(os.path.join(ROOT, "outputs", "verdict_sync_%s.json" % sid))
              stamped_here = bool(stamp) and os.path.basename(
                  str(stamp.get("kilosort_dir", "")).rstrip("\\/")) == sub
              r["synced"] = (os.path.exists(os.path.join(vd, "verdict_sync.json"))
                             or stamped_here)
              r["ks_folder"] = os.path.basename(k4)
              r["sync_check"] = sync_check(ks, r["bpod_synced"])
              r["catch"] = catch.get((rat, sid[:8]), [])
              snaps = []
              if os.path.isdir(vd):
                  snaps = sorted(d for d in os.listdir(vd)
                                 if d.startswith("snapshot_") and os.path.isdir(os.path.join(vd, d)))
              r["snapshots"] = snaps
              out.append(r)

    # A joined session is synced through its halves, and each half's decisions are the
    # joined session's. Neither fact is visible in either folder on its own, so it is
    # read here from the derived_from.json the splitter leaves in each half.
    by_sid = {r["session"]: r for r in out if r.get("ks_folder") == "kilosort4"}
    for r in out:
        if r.get("ks_folder") != "kilosort4_from_concat":
            continue
        d = None
        for rat, root in RATS:
            p = os.path.join(root, r["session"] + ".rec", r["session"] + ".kilosort",
                             "kilosort4_from_concat", "derived_from.json")
            d = jload(p)
            if d:
                break
        if not d:
            continue
        src = d.get("source_session")
        r["decisions_from"] = src
        parent = by_sid.get(src)
        if parent:
            # The decisions are the joined session's -- but NOT its "applied" flag:
            # each half has its own stamp, and the joined folder has none of its own
            # because there is nothing there to sync.
            for k in ("z_verdicts", "z_tally", "z_merges", "z_merge_tally", "z_trials",
                      "verdicts_dir", "artifact"):
                r[k] = parent[k]
            parent.setdefault("synced_parts", []).append(
                {"session": r["session"], "dirs": r["bpod_synced"],
                 "applied": r["synced"]})
            if r["synced"]:
                parent["synced"] = True
            # The joined session's alignment IS its halves' alignment; it has no clock
            # of its own to check.
            parent.setdefault("sync_check", []).extend(r.get("sync_check") or [])
    return out


def sync_check(ks, synced_dirs):
    """How well the ephys clock and the Bpod clock actually agree, per synced dataset.

    Read from each `synced/session_meta.json`, because "there is a synced directory"
    and "the alignment is good" are different claims and only the second one matters
    to an analysis. Four numbers, and each is there for a reason:

      matched           fraction of Bpod events that found their Trodes TTL. Below 1.0
                        means events went unexplained, which is the loud failure.
      holdout_max_ms    leave-one-out error of the time conversion: each matched pair
                        predicted from the others. This is the honest accuracy figure.
      linear_resid_ms   residual of a single straight line over the whole session. It
                        is large here (hundreds of ms) and that is EXPECTED -- the two
                        clocks drift, which is why the conversion is local rather than
                        linear. It is reported because it is what sets sync_status to
                        `needs_review`, and someone reading that word deserves to see
                        the number behind it and the holdout number beside it.
      trials_valid      trials the pipeline was willing to call synced.
    """
    out = []
    for d in synced_dirs:
        meta = jload(os.path.join(ks, d, "session_meta.json"))
        if not meta:
            continue
        s = meta.get("sync", {})
        trials = meta.get("n_trials")
        tv = meta.get("n_trials_sync_valid")
        if tv is None:                      # older bundles kept it only in the summary
            tv = trials
        out.append({
            "dir": d,
            "status": s.get("sync_status"),
            "matched": s.get("match_fraction"),
            "n_matched": s.get("n_matched"),
            "n_bpod_events": s.get("n_bpod_events"),
            "holdout_median_ms": s.get("interp_holdout_median_ms"),
            "holdout_max_ms": s.get("interp_holdout_max_ms"),
            "linear_median_resid_ms": s.get("median_residual_ms"),
            "linear_max_resid_ms": s.get("max_residual_ms"),
            "method": s.get("conversion_method"),
            "n_trials": trials,
            "n_trials_valid": tv,
            "gaps": len(meta.get("recording_gaps") or []),
            "trials_over_gap": meta.get("n_trials_overlapping_recording_gap"),
            # One word. A dataset passes when every Bpod event matched, every trial is
            # sync_valid, and the conversion's own worst-case holdout error is under
            # 50 ms -- short enough that no trial-aligned analysis cares.
            "verdict": None,
        })
        c = out[-1]
        if c["matched"] is None:
            c["verdict"] = "unknown"
        elif (c["matched"] >= 1.0 and (c["holdout_max_ms"] or 1e9) < 50
              and c["n_trials_valid"] == c["n_trials"]):
            c["verdict"] = "pass"
        elif c["matched"] >= 0.99:
            c["verdict"] = "check"
        else:
            c["verdict"] = "FAIL"
    return out


def sync_cell_md(r):
    """The alignment quality, in the table, next to "is there a synced folder".

    A folder existing says nothing about whether the clocks agree; these numbers do.
    Several synced folders for one recording are collapsed to the worst verdict, since
    that is the one somebody needs to know about.
    """
    checks = r.get("sync_check") or []
    if not checks:
        return "--"
    order = {"FAIL": 0, "check": 1, "unknown": 2, "pass": 3}
    worst = sorted(checks, key=lambda c: order.get(c["verdict"], 9))[0]
    bits = "%s (%.0f%% matched, %.0f ms worst-case, %s/%s trials)" % (
        worst["verdict"], 100.0 * (worst["matched"] or 0),
        worst["holdout_max_ms"] or 0, worst["n_trials_valid"], worst["n_trials"])
    return ("**%s**" % bits) if worst["verdict"] != "pass" else bits


def catch_cell_md(r):
    """Whether this day's protocol makes ERROR trials catch trials.

    PercentCatch gives catch trials on CORRECT choices, and every session here has
    them. CatchError is the separate switch that decides whether an error trial is
    also left unsignalled so the animal reveals how long it is willing to wait. With
    it off, the error is announced at the end of a delay WE drew, so there is no time
    investment to measure on errors and the confidence analysis -- which needs the
    correct and error waiting curves together -- cannot be done at all. It is the
    difference between a session that is merely incomplete and one that cannot answer
    the question, so it belongs in the table and not in a footnote.
    """
    c = r.get("catch") or []
    if not c:
        return "--"
    row = c[0]
    err = str(row.get("error_catch", ""))
    if err.startswith("NO"):
        return ("**NO ERROR CATCH** (CatchError=0; error wait %ss vs %ss on correct "
                "catch)" % (row.get("median_wait_error_s"), row.get("median_wait_catch_s")))
    if err.startswith("yes"):
        return "ok (%s%% catch, error wait %ss)" % (row.get("pct_flagged"),
                                                    row.get("median_wait_error_s"))
    return err or "--"


def state_of(r, live):
    """One word for where this session's decisions stand."""
    if r.get("decisions_from"):
        return "HALF OF A JOIN", (
            "this recording's units and verdicts are %s's, applied to the joint sort "
            "restricted to this half (kilosort4_from_concat) and synced against this "
            "recording's own Bpod file" % r["decisions_from"])
    lv = live.get(r["session"], {}).get("verdicts") if live else None
    if r.get("superseded_by") and not r["z_verdicts"] and not lv:
        return "SUPERSEDED", ("part of %s, which is the sort of record for this day. "
                              "Not for review, and its unit ids do not correspond to "
                              "the joint sort's." % r["superseded_by"])
    if lv is None and r["z_verdicts"] == 0:
        return "NOT REVIEWED", "no verdicts anywhere"
    if lv is None:
        return "ON Z", "%d on Z; live page not checked this run" % r["z_verdicts"]
    if lv == 0 and r["z_verdicts"] == 0:
        return "NOT REVIEWED", "the page is open and empty"
    if lv > r["z_verdicts"]:
        return "DRIFTED", ("the page has %d, Z has %d -- %d decisions exist only in the "
                           "browser database" % (lv, r["z_verdicts"], lv - r["z_verdicts"]))
    if lv < r["z_verdicts"]:
        return "CHECK", ("Z has %d but the page has %d -- Z may be ahead, or the page was "
                         "rebuilt" % (r["z_verdicts"], lv))
    if not r["n_units"] or lv < r["n_units"]:
        # Gil can call a review finished with units still unjudged -- he stops when the
        # ones left are automatic MUA he has no intention of labelling. That is a
        # decision, so it is recorded in artifact_live_counts.json and reported as
        # finished; the count of unjudged units is still shown rather than hidden.
        if r.get("review_finished"):
            return "SAVED, REVIEW CLOSED", (
                "Gil called this review finished: %d of %d units carry a verdict, all "
                "of them on Z; the other %d were left on their automatic label"
                % (lv, r["n_units"], r["n_units"] - lv))
        return "SAVED, PARTIAL", ("%d of %d units decided, all of them on Z"
                                  % (lv, r["n_units"] or 0))
    return "SAVED, COMPLETE", "%d of %d units decided, all on Z" % (lv, r["n_units"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=os.path.join(ROOT, "outputs", "artifact_live_counts.json"),
                    help="JSON of live artifact counts, written by Claude from the "
                         "review-page databases")
    args = ap.parse_args()
    live = jload(args.live) or {}
    if live:
        print("live counts from %s (as of %s)" % (os.path.basename(args.live),
                                                  live.get("_checked", "unknown")))
    else:
        print("no live artifact counts given -- the table will show Z only")
    live_rows = {k: v for k, v in live.items() if not k.startswith("_")}

    rows = scan()
    for r in rows:
        lv = live_rows.get(r["session"], {})
        r["live_verdicts"] = lv.get("verdicts")
        r["live_merges"] = lv.get("merge_verdicts")
        r["live_trials"] = lv.get("trial_selections")
        r["review_finished"] = bool(lv.get("review_finished"))
        # state last: it reads the fields above.
        r["state"], r["state_why"] = state_of(r, live_rows)

    md = []
    md.append("# Sorted sessions, verdicts, and where they are saved")
    md.append("")
    md.append("Generated by `scripts/session_status_report.py`. Every number is read from")
    md.append("Z or from a review page's own database -- nothing here is kept by hand, so it")
    md.append("cannot drift out of date without the script saying so.")
    md.append("")
    md.append("`DRIFTED` is the one that matters: the review page holds decisions that are not")
    md.append("on Z, which means they exist only in a browser database. Fix it by dumping the")
    md.append("three collections to `<session>.kilosort/review/verdicts/` on Z.")
    md.append("")
    hdr = ("| rat | session | units | Th u/l | decided | good | mua | noise | merges | "
           "windows | on Z | verdicts applied | Bpod synced | sync check | catch trials "
           "| joined | state |")
    md.append(hdr)
    md.append("|" + "---|" * 17)
    for r in rows:
        t = r["z_tally"]
        dec = r["live_verdicts"] if r["live_verdicts"] is not None else r["z_verdicts"]
        if r["bpod_synced"]:
            sy = "yes (`%s`)" % "`, `".join(r["bpod_synced"])
        elif r.get("synced_parts"):
            # A joined session has no DIO and no Bpod file of its own, so it is never
            # synced in its own folder. It is synced once per half, each against that
            # half's own behaviour, and saying "no" here would be false.
            sy = "yes, through its halves (`%s`)" % "`, `".join(
                p["session"] for p in r["synced_parts"])
        elif not r["bpod_files"]:
            sy = "**blocked** -- no Bpod files for this rat"
        else:
            sy = "no"
        md.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | **%s** |" % (
            r["rat"], r.get("row_id", r["session"]), r["n_units"] or "?",
            r["thresholds"] or "?",
            dec, t.get("good", 0), t.get("mua", 0), t.get("noise", 0),
            r["z_merges"], r["z_trials"],
            "yes" if r["z_verdicts"] else "no",
            "yes" if r["synced"] else "no", sy, sync_cell_md(r), catch_cell_md(r),
            "yes" if r["provenance"] else ("part of a join" if r["superseded_by"] else "no"),
            r["state"]))
    md.append("")
    md.append("`Bpod synced` is a different question from `verdicts applied`: the first is")
    md.append("a `synced/` directory written by `scripts/sync_session.py`, the second is the")
    md.append("group column in `cluster_info.tsv`. A session can have one without the other.")
    md.append("")
    md.append("## Per session")
    md.append("")
    for r in rows:
        md.append("### %s  (%s)" % (r["session"], r["rat"]))
        md.append("")
        md.append("- %s -- %s" % (r["state"], r["state_why"]))
        md.append("- %s units, Kilosort thresholds %s" % (r["n_units"], r["thresholds"]))
        if r.get("joins"):
            md.append("- **the sort of record for %s-%s-%s**, joining %s"
                      % (r["date"][:4], r["date"][4:6], r["date"][6:],
                         " + ".join("`%s`" % p for p in r["joins"])))
        if r.get("superseded_by"):
            md.append("- **superseded by** `%s` -- do not mix unit ids between them"
                      % r["superseded_by"])
        if r["bpod_synced"]:
            md.append("- Bpod sync: **done** -- `%s`" % "`, `".join(r["bpod_synced"]))
            if len(r["bpod_synced"]) > 1:
                md.append("  - **%d sync directories exist**, so this session was synced more "
                          "than once and which one is current is not recorded in their names. "
                          "Settle that before anything downstream reads one."
                          % len(r["bpod_synced"]))
        elif not r["bpod_files"]:
            md.append("- Bpod sync: **blocked** -- `Z:\\Gil\\%s\\bpod\\` holds no files, so no "
                      "session of this rat can be synced yet"
                      % {"Popper": "Popper", "Shamir": "Shamir_1", "Begin": "Begin_1"}[r["rat"]])
        else:
            md.append("- Bpod sync: not done (%d Bpod files available for this rat)"
                      % r["bpod_files"])
        if r["provenance"]:
            md.append("")
            md.append("  **What this session actually is**")
            md.append("")
            for line in r["provenance"]:
                md.append("  - %s" % line)
        if r["artifact"]:
            md.append("- review page: %s" % r["artifact"])
        else:
            md.append("- review page: **none built yet**")
        for label, url in r["tools"]:
            md.append("- %s: %s" % (label, url))
        if r["z_verdicts"]:
            md.append("- on Z: %d unit verdicts (%s), %d merge verdicts (%s), %d trial selections"
                      % (r["z_verdicts"],
                         ", ".join("%s %d" % kv for kv in sorted(r["z_tally"].items())),
                         r["z_merges"],
                         ", ".join("%s %d" % kv for kv in sorted(r["z_merge_tally"].items()))
                         or "none",
                         r["z_trials"]))
            md.append("- `%s`" % r["verdicts_dir"])
            if r["snapshots"]:
                md.append("- snapshots: %s" % ", ".join(r["snapshots"]))
        else:
            md.append("- nothing saved on Z for this session yet")
        md.append("- verdicts applied into `cluster_info.tsv`: %s"
                  % ("yes" if r["synced"] else "**no**"))
        md.append("")

    mdp = os.path.join(ROOT, "post_ks_correction", "SESSION_STATUS.md")
    with open(mdp, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    jp = os.path.join(ROOT, "outputs", "session_status.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"sessions": rows, "live_source": args.live}, f, indent=2)
    print("wrote %s" % mdp)
    print("wrote %s" % jp)
    print()
    for r in rows:
        print("  %-30s %-16s %s" % (r["session"], r["state"], r["state_why"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
