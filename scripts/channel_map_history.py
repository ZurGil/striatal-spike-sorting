# -*- coding: utf-8 -*-
"""Which channel map did each session use? Answered from the recordings, not from memory.

    python scripts/channel_map_history.py [--write]

Every Trodes `.rec` carries its whole configuration in its text header, including the
384 `<SpikeChannel>` entries that ARE the channel map: each one's `hwChan` (which
hardware channel feeds that ntrode) and its `coord_ml` / `coord_ap` / `coord_dv`
position in micrometres. So the map a session used is recoverable from the session
itself, for every session ever recorded, whether or not anyone wrote it down.

The script fingerprints that map per session, groups the sessions by fingerprint, and
labels each distinct map A, B, C... in order of first appearance. `--write` also saves
the result next to the rat's metadata, so the answer survives without re-reading 28
recordings over the network.

Deliberately NOT part of the fingerprint: thresholds, filter settings, colours, display
state. Those change constantly and have nothing to do with the channel map. Changing a
threshold must not look like changing the map.

Reads only. Nothing is modified unless --write is given, and then only two new files.
"""
import hashlib
import json
import os
import re
import sys

B = chr(92)
Z_RAT = "Z:" + B + "Gil" + B + "Shamir_1"
F_RAT = "F:" + B + "Gil" + B + "Shamir"
OUT_JSON = os.path.join(Z_RAT, "channel_map_history.json")
OUT_MD = os.path.join(Z_RAT, "channel_map_history.md")
MARKER = b"</Configuration>"
PROBE = 2 * 1024 * 1024

WRITE = "--write" in sys.argv


def log(msg):
    print(msg)
    sys.stdout.flush()


def header_text(rec):
    """The .rec's XML header, as text. Measured, never assumed -- it is variable-length
    and has been 255,039 / 255,102 / 255,106 bytes on this rig in three weeks."""
    with open(rec, "rb") as f:
        head = f.read(PROBE)
    i = head.find(MARKER)
    if i < 0:
        return None
    return head[:i + len(MARKER)].decode("utf8", "replace")


def channel_map(hdr):
    """The map itself: per ntrode, which hardware channel and where it sits.

    Returned in document order, which is the order Trodes writes the ntrodes in and the
    order the exported .dat channels follow.
    """
    ids = re.findall(r'<SpikeNTrode\b[^>]*\bid="([^"]*)"', hdr)
    chans = []
    for tag in re.findall(r"<SpikeChannel\b[^>]*/>", hdr):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        chans.append((attrs.get("hwChan"), attrs.get("coord_ml"),
                      attrs.get("coord_ap"), attrs.get("coord_dv")))
    return ids, chans


def fingerprint(ids, chans):
    blob = json.dumps({"ids": ids, "chans": chans}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def find_recs():
    """Every session .rec reachable, archive first then F, de-duplicated by session id."""
    found = {}
    for root in (Z_RAT, F_RAT):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".rec") or not os.path.isdir(os.path.join(root, name)):
                continue
            sid = name[:-4]
            rec = os.path.join(root, name, sid + ".rec")
            if os.path.isfile(rec) and sid not in found:
                found[sid] = rec
    return [(sid, found[sid]) for sid in sorted(found)]


recs = find_recs()
if not recs:
    raise SystemExit("no .rec files found under %s or %s" % (Z_RAT, F_RAT))

log("reading the channel map out of %d recordings" % len(recs))
log("")

rows = []
maps = {}            # fingerprint -> {"label", "ids", "chans", "sessions"}
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for sid, rec in recs:
    hdr = header_text(rec)
    if hdr is None:
        log("  %s  NO HEADER FOUND -- skipped (%s)" % (sid, rec))
        continue
    ids, chans = channel_map(hdr)
    if len(chans) != 384 or len(ids) != 384:
        log("  %s  unexpected counts: %d ntrodes, %d channels"
            % (sid, len(ids), len(chans)))
    fp = fingerprint(ids, chans)
    if fp not in maps:
        maps[fp] = {"label": labels[len(maps)], "ids": ids, "chans": chans,
                    "sessions": []}
    maps[fp]["sessions"].append(sid)
    rows.append((sid, maps[fp]["label"], fp[:12], len(chans),
                 "Z" if rec.startswith("Z:") else "F"))

log("  %-18s %-5s %-14s %-9s %s" % ("session", "map", "fingerprint", "channels",
                                    "read from"))
prev = None
for sid, label, fp12, n, where in rows:
    mark = "   <-- CHANGED HERE" if prev is not None and label != prev else ""
    log("  %-18s %-5s %-14s %-9d %s%s" % (sid, label, fp12, n, where, mark))
    prev = label

log("")
log("%d distinct channel map(s) across %d sessions" % (len(maps), len(rows)))
log("")

ordered = sorted(maps.values(), key=lambda m: m["label"])
for m in ordered:
    ss = m["sessions"]
    dv = sorted(set(int(c[3]) for c in m["chans"]))
    m["depth_span_um"] = [dv[0], dv[-1]]
    m["depth_steps"] = len(dv)
    log("map %s: %d session(s), %s%s  --  coord_dv %d..%d um in %d steps"
        % (m["label"], len(ss), ss[0],
           " .. " + ss[-1] if len(ss) > 1 else "",
           dv[0], dv[-1], len(dv)))

# --- what actually differs between consecutive maps ---------------------------------
diff_lines = []


def dlog(msg):
    """Print it and keep it, so the written record carries the same findings."""
    log(msg)
    diff_lines.append(msg)


for a, b in zip(ordered, ordered[1:]):
    dlog("")
    dlog("-- map %s vs map %s: what changed" % (a["label"], b["label"]))
    # Indexed by hardware channel: did a given channel's pad position move?
    pa = {c[0]: (c[1], c[2], c[3]) for c in a["chans"]}
    pb = {c[0]: (c[1], c[2], c[3]) for c in b["chans"]}
    if set(pa) == set(pb):
        dv_off = set(int(pa[h][2]) - int(pb[h][2]) for h in pa)
        ml_same = all(pa[h][0] == pb[h][0] for h in pa)
        ap_same = all(pa[h][1] == pb[h][1] for h in pa)
        dlog("   per hardware channel: coord_ml %s, coord_ap %s, coord_dv offsets %s um"
             % ("identical" if ml_same else "DIFFERS",
                "identical" if ap_same else "DIFFERS", sorted(dv_off)))
    # Indexed by row, which is the order the exported .dat channels follow.
    dva = [int(c[3]) for c in a["chans"]]
    dvb = [int(c[3]) for c in b["chans"]]
    if len(dva) == len(dvb):
        row_off = sorted(set(x - y for x, y in zip(dva, dvb)))
        dlog("   per exported row: coord_dv offsets %s um%s"
             % (row_off, "  <- a uniform depth re-reference" if len(row_off) == 1
                else ""))
    hwa = [c[0] for c in a["chans"]]
    hwb = [c[0] for c in b["chans"]]
    if sorted(hwa) == sorted(hwb) and hwa != hwb:
        rot = next((k for k in range(len(hwa)) if hwa[k:] + hwa[:k] == hwb), None)
        dlog("   row order: the same channels, %s"
             % ("rotated by %d positions" % rot if rot is not None else "reordered"))
    if a["ids"] != b["ids"]:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y]
        dlog("   ntrode ids differ at %d of %d positions (A %s..%s, B %s..%s)"
             % (len(diff), len(a["ids"]), min(a["ids"]), max(a["ids"]),
                min(b["ids"]), max(b["ids"])))
    else:
        dlog("   ntrode ids: identical")

    fields = ("hwChan", "coord_ml", "coord_ap", "coord_dv")
    for k, fname in enumerate(fields):
        da = [c[k] for c in a["chans"]]
        db = [c[k] for c in b["chans"]]
        if da == db:
            dlog("   %-9s identical row for row" % fname)
            continue
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(da, db)) if x != y]
        note = ""
        if sorted(da) == sorted(db):
            note = "  (same set of values, reordered)"
        dlog("   %-9s differs at %d of %d rows%s" % (fname, len(diff), len(da), note))
        for i, x, y in diff[:4]:
            dlog("     row %d: %s -> %s" % (i, x, y))
        if len(diff) > 4:
            dlog("     ... and %d more" % (len(diff) - 4))

# --- optionally write it down -------------------------------------------------------
if WRITE:
    payload = {
        "generated_from": "the <SpikeChannel> entries in each session's .rec header",
        "fingerprint_covers": ["SpikeNTrode id", "hwChan", "coord_ml", "coord_ap",
                               "coord_dv"],
        "fingerprint_excludes": ["thresholds", "filters", "colours", "display state"],
        "maps": [{"label": m["label"], "sessions": m["sessions"],
                  "n_channels": len(m["chans"]),
                  "coord_dv_um": m.get("depth_span_um"),
                  "coord_dv_steps": m.get("depth_steps")} for m in ordered],
        "by_session": {sid: label for sid, label, _f, _n, _w in rows},
        "differences": [d for d in diff_lines if d.strip()],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    lines = ["# Channel maps used, by session", "",
             "Derived from the `<SpikeChannel>` entries in each session's own `.rec`",
             "header by `scripts/channel_map_history.py` -- not from anyone's notes, so",
             "it can be regenerated and cannot drift out of date.", "",
             "The fingerprint covers the ntrode ids, `hwChan` and the `coord_ml` /",
             "`coord_ap` / `coord_dv` positions. It deliberately ignores thresholds,",
             "filters and display settings, so changing a threshold does not look like",
             "changing the map.", ""]
    for m in ordered:
        lines.append("## Map %s -- %d session(s)" % (m["label"], len(m["sessions"])))
        lines.append("")
        lines.append("`coord_dv` spans %d..%d um in %d steps."
                     % (m["depth_span_um"][0], m["depth_span_um"][1],
                        m["depth_steps"]))
        lines.append("")
        for s in m["sessions"]:
            lines.append("- %s" % s)
        lines.append("")
    if diff_lines:
        lines.append("## What differs between the maps")
        lines.append("")
        lines.append("```")
        lines.extend(d for d in diff_lines if d.strip())
        lines.append("```")
        lines.append("")
        lines.append("A uniform `coord_dv` offset is a change of DEPTH REFERENCE, not")
        lines.append("of hardware: the same pads, labelled from a different origin. It")
        lines.append("shifts every unit's `depth_mm` by that amount, because the")
        lines.append("pipeline computes `depth_mm = implant_depth_mm - phy_depth/1000`")
        lines.append("and `phy_depth` comes from these coordinates. Sessions on")
        lines.append("different maps are therefore NOT depth-comparable without")
        lines.append("correcting for the offset.")
        lines.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    log("")
    log("written:")
    log("  %s" % OUT_JSON)
    log("  %s" % OUT_MD)
