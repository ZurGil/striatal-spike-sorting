# -*- coding: utf-8 -*-
"""Decide whether a session can be uploaded for a cluster sort, from the files themselves.

    python scripts/ris_preflight_batch.py --rat Shamir --all
    python scripts/ris_preflight_batch.py --rat Shamir 20260914_122553 20260917_100043

WHY THIS EXISTS
---------------
Two things have already cost a whole cluster job each, and both were knowable before
the upload started:

  * `timestamps.dat` was not staged for 20260915_084812_103710_concat. Every one of the
    six post-KS steps begins by describing the session, `discover_session()` refuses to
    describe one without that file, so 111 workers died in 45 s -- and LSF still
    reported the job as DONE (RESEARCH_LOG 5bd.46).
  * a session's channel map is not what the pad-location file says; it is what the
    `.rec` header says. Sorting a whole session on the wrong map produces a complete,
    plausible, wrongly-positioned result (RESEARCH_LOG 5bd.38, 5bd.46).

So this refuses a session rather than letting a 4-hour GPU job find out. It reads only
sizes and headers -- a couple of MiB per session -- so it is safe to run while Trodes
is recording or exporting.

WHAT IT CHECKS
--------------
  1. the five things Kilosort needs are present: probe1.dat, timestamps.dat,
     channelmap_probe1.dat, the six dio_Controller_Din{1..6}.dat, and the .rec itself
  2. probe1.dat is a whole number of 384-channel int16 samples
  3. timestamps.dat holds exactly that many 4-byte stamps past its own measured header
  4. the .rec's length implies the same packet count (within one packet)
  5. the pad-location block of channelmap_probe1.dat is byte-identical to the
     reference session's -- the sort we are extending
  6. the map era in channel_map_history.json (which is derived from the .rec headers,
     not from notes) is the expected one
  7. rat_metadata.json exists, since four of its fields silently set unit coordinates

Nothing is written. Nothing is uploaded. It prints a table and a per-session verdict.
"""
import argparse
import hashlib
import json
import os

NCH = 384
PACKET = 856
REC_MARKER = b"</Configuration>"
TS_MARKER = b"<End settings>"
RAT_DIRS = {"Shamir": "Shamir_1", "Popper": "Popper", "Begin": "Begin_1"}


def header_len(path, marker, probe=4 * 1024 * 1024):
    """Measured, never assumed -- these headers change width between months."""
    try:
        with open(path, "rb") as f:
            head = f.read(probe)
    except OSError:
        return None
    i = head.find(marker)
    return i + len(marker) + 1 if i >= 0 else None


def pad_block_md5(path):
    """md5 of channelmap_probe1.dat AFTER its text header: the geometry alone.

    The header carries the session name, its creation time and the headstage fields,
    so hashing the whole file always differs between sessions and says nothing.
    """
    h = header_len(path, TS_MARKER)
    if h is None:
        return None
    with open(path, "rb") as f:
        f.seek(h)
        return hashlib.md5(f.read()).hexdigest()


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def check(sid, root, era_map, ref_pad, expect_era):
    sdir = os.path.join(root, sid + ".rec")
    ks = os.path.join(sdir, sid + ".kilosort")
    dio = os.path.join(sdir, sid + ".DIO")
    f = {
        "probe": os.path.join(ks, sid + ".probe1.dat"),
        "ts": os.path.join(ks, sid + ".timestamps.dat"),
        "cmap": os.path.join(ks, sid + ".channelmap_probe1.dat"),
        "rec": os.path.join(sdir, sid + ".rec"),
    }
    r = {"sid": sid, "problems": [], "notes": []}

    for k, path in f.items():
        if size(path) <= 0:
            r["problems"].append("%s missing" % os.path.basename(path))
    missing_dio = [i for i in range(1, 7)
                   if size(os.path.join(dio, "%s.dio_Controller_Din%d.dat" % (sid, i))) <= 0]
    if missing_dio:
        r["problems"].append("DIO Din%s missing" % ",".join(str(i) for i in missing_dio))
    if r["problems"]:
        return r

    pn, tn, rn = size(f["probe"]), size(f["ts"]), size(f["rec"])
    r["probe_gb"] = pn / 1024.0 ** 3
    r["rec_gb"] = rn / 1024.0 ** 3
    if pn % (NCH * 2):
        r["problems"].append("probe1.dat is not a whole number of %d-channel samples "
                             "(%d left over)" % (NCH, pn % (NCH * 2)))
    r["packets_probe"] = pn // (NCH * 2)
    r["hours"] = r["packets_probe"] / 30000.0 / 3600.0

    th = header_len(f["ts"], TS_MARKER)
    if th is None:
        r["problems"].append("no %r at the head of timestamps.dat" % TS_MARKER)
    else:
        if (tn - th) % 4:
            r["problems"].append("timestamps.dat is not whole 4-byte stamps past its "
                                 "%d-byte header" % th)
        r["packets_ts"] = (tn - th) // 4
        if r["packets_ts"] != r["packets_probe"]:
            r["problems"].append("probe1.dat holds %d packets, timestamps.dat holds %d"
                                 % (r["packets_probe"], r["packets_ts"]))

    rh = header_len(f["rec"], REC_MARKER)
    if rh is None:
        r["problems"].append("no %r in the first 4 MiB of the .rec" % REC_MARKER)
    else:
        r["packets_rec"] = (rn - rh) // PACKET
        if abs(r["packets_rec"] - r["packets_probe"]) > 1:
            short = (r["packets_rec"] - r["packets_probe"]) * NCH * 2 / 1024.0 ** 3
            r["problems"].append("the .rec implies %d packets but the export holds %d "
                                 "-- short by %.1f GiB"
                                 % (r["packets_rec"], r["packets_probe"], short))

    pad = pad_block_md5(f["cmap"])
    r["pad_md5"] = pad
    if pad is None:
        r["problems"].append("could not read the pad-location block of the channel map")
    elif ref_pad and pad != ref_pad:
        r["problems"].append("channel map pad block %s does not match the reference %s"
                             % (pad[:12], ref_pad[:12]))

    era = era_map.get(sid)
    r["era"] = era
    if era is None:
        r["problems"].append("not in channel_map_history.json -- regenerate it with "
                             "scripts/channel_map_history.py before trusting the map")
    elif expect_era and era != expect_era:
        r["problems"].append("channel_map_history says map %s, expected %s" % (era, expect_era))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--rat", default="Shamir", choices=sorted(RAT_DIRS))
    ap.add_argument("--all", action="store_true",
                    help="every session of this rat that has no kilosort4 output yet")
    ap.add_argument("--reference", default=None,
                    help="an already-sorted session whose channel map the batch must "
                         "match (default: the newest sorted one)")
    ap.add_argument("--expect_era", default=None, help="e.g. A")
    args = ap.parse_args()

    root = os.path.join(r"Z:\Gil", RAT_DIRS[args.rat])
    hist = os.path.join(root, "channel_map_history.json")
    era_map = {}
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            era_map = json.load(f).get("by_session", {})
    else:
        print("WARNING: no channel_map_history.json for %s -- map era cannot be checked"
              % args.rat)

    all_sids = sorted(d[:-4] for d in os.listdir(root)
                      if d.endswith(".rec") and os.path.isdir(os.path.join(root, d)))
    sorted_sids = [s for s in all_sids
                   if os.path.exists(os.path.join(root, s + ".rec", s + ".kilosort",
                                                  "kilosort4", "spike_clusters.npy"))]
    if not os.path.exists(os.path.join(root, "rat_metadata.json")):
        print("WARNING: %s has no rat_metadata.json -- four of its fields silently set "
              "unit coordinates" % root)

    ref = args.reference or (sorted_sids[-1] if sorted_sids else None)
    ref_pad = None
    if ref:
        ref_pad = pad_block_md5(os.path.join(root, ref + ".rec", ref + ".kilosort",
                                             ref + ".channelmap_probe1.dat"))
        print("reference sort: %s   channel map pad block %s"
              % (ref, ref_pad[:16] if ref_pad else "UNREADABLE"))
        print("already sorted: %s" % ", ".join(sorted_sids))
        print()

    todo = args.sessions or ([s for s in all_sids if s not in sorted_sids]
                             if args.all else [])
    if not todo:
        raise SystemExit("name some sessions, or pass --all")

    rows = [check(s, root, era_map, ref_pad, args.expect_era) for s in todo]
    print("%-32s %7s %7s %6s %4s %-14s %s"
          % ("session", "rec_GiB", "dat_GiB", "hours", "map", "pad_md5", "verdict"))
    ok, bad = [], []
    for r in rows:
        v = "READY" if not r["problems"] else "NO"
        (ok if not r["problems"] else bad).append(r)
        print("%-32s %7s %7s %6s %4s %-14s %s"
              % (r["sid"],
                 "%.0f" % r["rec_gb"] if "rec_gb" in r else "-",
                 "%.0f" % r["probe_gb"] if "probe_gb" in r else "-",
                 "%.2f" % r["hours"] if "hours" in r else "-",
                 r.get("era") or "-",
                 (r.get("pad_md5") or "-")[:12], v))
        for p in r["problems"]:
            print("%-32s   ! %s" % ("", p))

    print()
    print("READY: %d of %d" % (len(ok), len(rows)))
    if ok:
        tot = sum(r["probe_gb"] for r in ok)
        hrs = sum(r["hours"] for r in ok)
        print("  %.0f GiB of probe1.dat to upload, %.1f h of recording" % (tot, hrs))
        print("  at the 173 MB/s measured on 2026-11-12 that is %.1f h of transfer"
              % (tot * 1024 / 173.0 / 3600))
        print("  job 1 ran at 0.043 h per GiB on the last three sessions, so about "
              "%.0f h of GPU time if they run one at a time" % (0.043 * tot))
    if bad:
        print("  NOT ready: %s" % ", ".join(r["sid"] for r in bad))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
