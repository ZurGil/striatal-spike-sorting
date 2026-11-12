# -*- coding: utf-8 -*-
"""Report every date that has more than one Trodes recording, with what is needed to
decide whether they are one interrupted session or two deliberate ones.

    python scripts/find_multi_recording_days.py                 # all rats on Z
    python scripts/find_multi_recording_days.py --rat Shamir_1
    python scripts/find_multi_recording_days.py --root Z:/Gil --json out.json

WHY THIS EXISTS
---------------
Gil's rule, 2026-11-11: when a day has more than one recording, find out what happened
before sorting anything, because they may need to be joined. 2026-09-15 is the case that
produced the rule -- Trodes was stopped 69 minutes in when 82 channels went bad, and the
two halves are one session. Sorting them apart would have given the same neuron two
identities and split its spikes across two files, for no reason other than that the
filesystem happened to hold two folders.

A rule nobody can forget is better than a rule everybody is told, so this makes the check
mechanical: run it against the archive and it names the days that need a look.

WHAT IT DOES NOT DO
-------------------
It does not decide. The gap in minutes, the durations and the Bpod files are evidence,
and a short gap is strong evidence of an interruption, but only the person who was in the
room knows whether the second recording was a restart or a planned second block. It also
never claims a duration it cannot measure: an exact duration comes from the exported
`<session>.probe1.dat`, and where that export does not exist the duration is printed as
an estimate from the .rec size, with the ratio's observed spread shown so the estimate
can be judged. The estimate is for triage only -- never for a seam offset.

WHAT TO DO WITH A DAY IT FLAGS
------------------------------
Three measurements settle it, and all three are cheap:
  1. the real-time gap, which this script gives;
  2. whether the probe moved -- correlate the spike-band per-channel RMS fingerprint of
     the end of one recording against the start of the next, and against within-recording
     controls, since a fingerprint that matches as well across the gap as it does within a
     recording means the channels are seeing the same cells;
  3. whether either recording has a damaged tail or head -- the same fingerprint against
     a healthy mid-recording reference, which is what found the 82 bad channels on
     2026-09-15.
Then, if they are one session: concatenate, trim any damaged edge, record the seam sample
index in a provenance file, and sort once. See outputs/concat_20260915_084812_103710.json
for the shape of that record, and RESEARCH_LOG.md 5bd.42.
"""
import argparse
import datetime
import json
import os
import re

SESSION_RE = re.compile(r"^(\d{8})_(\d{6})\.rec$")
NCH = 384
BYTES_PER_SAMPLE = 2 * NCH
FS = 30000.0

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_start(date_str, time_str):
    return datetime.datetime(
        int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]),
        int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6]),
    )


def scan_rat(rat_dir):
    """Every recording in one rat's archive, with an exact duration where one is knowable."""
    out = []
    for name in sorted(os.listdir(rat_dir)):
        m = SESSION_RE.match(name)
        if not m:
            continue
        sid = name[:-4]
        folder = os.path.join(rat_dir, name)
        probe = os.path.join(folder, sid + ".kilosort", sid + ".probe1.dat")
        rec = os.path.join(folder, sid + ".rec")
        rec_bytes = os.path.getsize(rec) if os.path.isfile(rec) else None
        entry = {
            "session": sid,
            "date": m.group(1),
            "start": parse_start(m.group(1), m.group(2)),
            "rec_bytes": rec_bytes,
            "probe_bytes": None,
            "samples": None,
            "seconds": None,
            "exact": False,
        }
        if os.path.isfile(probe):
            b = os.path.getsize(probe)
            entry["probe_bytes"] = b
            entry["samples"] = b // BYTES_PER_SAMPLE
            entry["seconds"] = entry["samples"] / FS
            entry["exact"] = b % BYTES_PER_SAMPLE == 0
        out.append(entry)
    return out


def rec_to_probe_ratio(recordings):
    """Measured from the sessions that have both files, so an estimate can be judged."""
    rs = [r["rec_bytes"] / r["probe_bytes"]
          for r in recordings if r["rec_bytes"] and r["probe_bytes"]]
    if not rs:
        return None, None, None, 0
    rs.sort()
    mid = rs[len(rs) // 2]
    return mid, rs[0], rs[-1], len(rs)


def bpod_for_date(rat_dir, date_str):
    bpod = os.path.join(rat_dir, "bpod")
    if not os.path.isdir(bpod):
        return []
    tag = "%s%02d_%s" % (MONTHS[int(date_str[4:6]) - 1], int(date_str[6:8]), date_str[0:4])
    hits = [f for f in sorted(os.listdir(bpod)) if tag in f and f.lower().endswith(".mat")]
    return hits


def hhmm(dt):
    return dt.strftime("%H:%M:%S")


def fmt_dur(sec, exact):
    if sec is None:
        return "duration unknown (not exported)"
    return "%.1f min%s" % (sec / 60.0, "" if exact else "  [ESTIMATED from .rec size]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"Z:\Gil", help="archive root holding one folder per rat")
    ap.add_argument("--rat", default=None, help="only this rat")
    ap.add_argument("--json", default=None, help="also write the findings here")
    args = ap.parse_args()

    rats = [args.rat] if args.rat else sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))

    findings = []
    for rat in rats:
        rat_dir = os.path.join(args.root, rat)
        recs = scan_rat(rat_dir)
        if not recs:
            continue
        ratio, rlo, rhi, nratio = rec_to_probe_ratio(recs)

        by_date = {}
        for r in recs:
            by_date.setdefault(r["date"], []).append(r)
        multi = {d: v for d, v in by_date.items() if len(v) > 1}

        print("=" * 78)
        print("%s -- %d recordings over %d dates" % (rat, len(recs), len(by_date)))
        if ratio:
            print("   .rec/.probe1.dat size ratio from %d exported sessions: "
                  "median %.4f, range %.4f-%.4f" % (nratio, ratio, rlo, rhi))
        if not multi:
            print("   no date has more than one recording")
            continue
        print("   %d date(s) with more than one recording:" % len(multi))

        for date in sorted(multi):
            parts = sorted(multi[date], key=lambda r: r["start"])
            print()
            print("  %s -- %d recordings" % (date, len(parts)))
            prev_end = None
            for i, p in enumerate(parts, 1):
                sec, exact = p["seconds"], p["exact"]
                if sec is None and ratio and p["rec_bytes"]:
                    sec = (p["rec_bytes"] / ratio) / BYTES_PER_SAMPLE / FS
                    exact = False
                end = p["start"] + datetime.timedelta(seconds=sec) if sec else None
                print("    %d) %-20s %s -> %-8s  %s"
                      % (i, p["session"], hhmm(p["start"]),
                         hhmm(end) if end else "?", fmt_dur(sec, exact)))
                if prev_end is not None and end is not None:
                    gap = (p["start"] - prev_end).total_seconds()
                    note = ("a few minutes -- almost certainly a crash and restart"
                            if gap < 600 else
                            "under an hour -- an interruption that took some fixing"
                            if gap < 3600 else
                            "over an hour -- could be a planned second block; ask")
                    print("       gap before it: %d min %02d s   (%s)"
                          % (int(gap // 60), int(gap % 60), note))
                    if any(not q["exact"] for q in (p,)) or sec is None:
                        print("       NOTE: that gap uses an ESTIMATED duration; export the "
                              "recording before relying on it")
                prev_end = end
            bp = bpod_for_date(rat_dir, date)
            print("    bpod files for the date: %s" % (", ".join(bp) if bp else "none found"))
            if len(bp) != len(parts):
                print("       NOTE: %d bpod file(s) for %d recording(s) -- they do not map "
                      "one to one, so do not infer the split from bpod alone"
                      % (len(bp), len(parts)))
            print("    -> EXAMINE BEFORE SORTING: probe-fingerprint continuity across the gap, "
                  "and a damaged tail or head on either side.")
            findings.append({
                "rat": rat, "date": date,
                "sessions": [p["session"] for p in parts],
                "bpod": bp,
            })

    print()
    print("=" * 78)
    print("%d multi-recording date(s) in total" % len(findings))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(findings, fh, indent=2)
        print("written: %s" % args.json)


if __name__ == "__main__":
    main()
