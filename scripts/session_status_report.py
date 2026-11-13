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


def scan():
    out = []
    for rat, root in RATS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".rec"):
                continue
            sdir = os.path.join(root, name)
            if not os.path.isdir(sdir):
                continue
            sid = name[:-4]
            ks = os.path.join(sdir, sid + ".kilosort")
            k4 = os.path.join(ks, "kilosort4")
            if not os.path.exists(os.path.join(k4, "spike_clusters.npy")):
                continue
            vd = os.path.join(ks, "review", "verdicts")
            r = {
                "rat": rat, "session": sid, "date": sid[:8],
                "n_units": n_units(k4), "thresholds": thresholds(k4),
                "artifact": ARTIFACTS.get(sid),
                "tools": EXTRA_TOOLS.get(sid, []),
                "verdicts_dir": vd if os.path.isdir(vd) else None,
                "cluster_info": os.path.exists(os.path.join(k4, "cluster_info.tsv")),
            }
            v = jload(os.path.join(vd, "verdicts.json")) or {}
            m = jload(os.path.join(vd, "merge_verdicts.json")) or {}
            t = jload(os.path.join(vd, "trial_selections.json")) or {}
            r["z_verdicts"] = len(v)
            r["z_tally"] = dict(collections.Counter(x.get("verdict") for x in v.values()))
            r["z_merges"] = len(m)
            r["z_merge_tally"] = dict(collections.Counter(x.get("verdict") for x in m.values()))
            r["z_trials"] = len(t)
            r["synced"] = os.path.exists(os.path.join(vd, "verdict_sync.json"))
            snaps = []
            if os.path.isdir(vd):
                snaps = sorted(d for d in os.listdir(vd)
                               if d.startswith("snapshot_") and os.path.isdir(os.path.join(vd, d)))
            r["snapshots"] = snaps
            out.append(r)
    return out


def state_of(r, live):
    """One word for where this session's decisions stand."""
    lv = live.get(r["session"], {}).get("verdicts") if live else None
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
        r["state"], r["state_why"] = state_of(r, live_rows)
        lv = live_rows.get(r["session"], {})
        r["live_verdicts"] = lv.get("verdicts")
        r["live_merges"] = lv.get("merge_verdicts")
        r["live_trials"] = lv.get("trial_selections")

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
           "windows | on Z | applied | state |")
    md.append(hdr)
    md.append("|" + "---|" * 13)
    for r in rows:
        t = r["z_tally"]
        dec = r["live_verdicts"] if r["live_verdicts"] is not None else r["z_verdicts"]
        md.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | **%s** |" % (
            r["rat"], r["session"], r["n_units"] or "?", r["thresholds"] or "?",
            dec, t.get("good", 0), t.get("mua", 0), t.get("noise", 0),
            r["z_merges"], r["z_trials"],
            "yes" if r["z_verdicts"] else "no",
            "yes" if r["synced"] else "no", r["state"]))
    md.append("")
    md.append("## Per session")
    md.append("")
    for r in rows:
        md.append("### %s  (%s)" % (r["session"], r["rat"]))
        md.append("")
        md.append("- %s -- %s" % (r["state"], r["state_why"]))
        md.append("- %s units, Kilosort thresholds %s" % (r["n_units"], r["thresholds"]))
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
