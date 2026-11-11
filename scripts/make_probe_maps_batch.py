# -*- coding: utf-8 -*-
"""Generate the Kilosort probe map for every archived session, into a local staging dir.

    python scripts/make_probe_maps_batch.py [--rat Shamir] [--out D:/Gil/probe_maps]

WHY THIS EXISTS
---------------
`make_probe_map.py` works on the current working directory and writes its output
there. Pointed at the archive that would mean writing into Z unattended, 30 times,
with no chance to look at the result first. So this stages to a local directory; the
files are copied into the session folders only once they have been looked at.

WHY IT DOES NOT GUESS THE MAP FROM THE DATE
-------------------------------------------
Sessions up to 2026-10-23 use channel map A and those from 2026-10-26 use map B, and
the two differ by a silent 820 um shift in depth -- get it wrong and the sort looks
entirely plausible at the wrong depths. But the date rule is not needed: every session
carries its OWN `channelmap_probe1.dat`, written by Trodes at acquisition time, and the
map is derived from that file. The date is only ever used to CHECK the answer, never to
produce it. Any session whose geometry disagrees with its date is reported loudly rather
than quietly accepted.
"""
import argparse
import json
import os
import re
import sys

sys.path.append(r"C:\Users\Adam\Documents\Trodes_2-4-2_Windows64\Resources\TrodesToPython")

# The Trodes reader calls np.set_printoptions(threshold='nan') at import time. That
# was legal before numpy 1.13 and raises TypeError on anything current, so the module
# cannot be imported at all without this. Only the display setting is suppressed --
# nothing that touches how the data is parsed -- and it is restored immediately.
# It is also a SCRIPT, not a library: at module level it does
# `fields = readTrodesExtractedDataFile(argv[1])`, so importing it with any command
# line at all makes it try to open our first argument as a Trodes file. argv is
# blanked for the duration of the import and restored afterwards. (This is why the
# original make_probe_map.py had to be run with no arguments, from inside the
# session directory.)
import numpy as _np
_real_set_printoptions = _np.set_printoptions
_real_argv = sys.argv
_np.set_printoptions = lambda *a, **k: None
sys.argv = [sys.argv[0]]
try:
    import readTrodesExtractedDataFile3 as trodes
except SystemExit:
    pass  # the module exits when given no file; the functions are still defined
finally:
    _np.set_printoptions = _real_set_printoptions
    sys.argv = _real_argv

RAT_DIRS = {"Shamir": "Shamir_1", "Popper": "Popper", "Begin": "Begin_1"}
# Map B starts here. Sessions on or after this date are expected to sit 820 um deeper.
MAP_B_FROM = "20261026"


def build_map(channelmap_path):
    cmap = trodes.readTrodesExtractedDataFile(channelmap_path)
    raw = cmap["data"]
    xc = raw["ml"].flatten().astype(float)
    yc = raw["dv"].flatten().astype(float)
    n = len(xc)
    return {
        "chanMap": list(range(n)),
        "xc": xc.tolist(),
        "yc": yc.tolist(),
        "kcoords": [0] * n,
        "n_chan": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", default="Shamir", choices=sorted(RAT_DIRS))
    ap.add_argument("--out", default=r"D:\Gil\probe_maps")
    ap.add_argument("--archive-root", default=r"Z:\Gil")
    args = ap.parse_args()

    root = os.path.join(args.archive_root, RAT_DIRS[args.rat])
    outdir = os.path.join(args.out, args.rat)
    os.makedirs(outdir, exist_ok=True)

    sessions = sorted(d for d in os.listdir(root) if d.endswith(".rec"))
    print("%d sessions under %s" % (len(sessions), root))
    print("%-18s %-7s %-9s %-9s %-7s %s" % ("session", "chans", "yc_min", "yc_max", "map", "note"))

    rows, problems = [], []
    for d in sessions:
        sid = d[:-4]
        ksdir = os.path.join(root, d, sid + ".kilosort")
        src = os.path.join(ksdir, sid + ".channelmap_probe1.dat")
        if not os.path.isfile(src):
            print("%-18s %s" % (sid, "NO CHANNELMAP SOURCE -- skipped"))
            problems.append((sid, "no channelmap_probe1.dat"))
            continue
        try:
            m = build_map(src)
        except Exception as e:
            print("%-18s FAILED: %s" % (sid, e))
            problems.append((sid, "read failed: %s" % e))
            continue

        yc0, yc1 = min(m["yc"]), max(m["yc"])
        # Which map does the geometry say? Check it against what the date says.
        expected_b = sid[:8] >= MAP_B_FROM
        dest = os.path.join(outdir, sid + "_custom_neuropixels_map.json")
        with open(dest, "w") as f:
            json.dump(m, f)
        rows.append((sid, m["n_chan"], yc0, yc1, expected_b))
        print("%-18s %-7d %-9.1f %-9.1f %-7s" % (sid, m["n_chan"], yc0, yc1, "B?" if expected_b else "A?"))

    # The check: within each date group the geometry must be self-consistent, and the
    # two groups must differ. If map A and map B produce the same yc range, either the
    # 820 um story is wrong or we are reading the wrong file -- say so, do not assume.
    print()
    a = sorted({(r[2], r[3]) for r in rows if not r[4]})
    b = sorted({(r[2], r[3]) for r in rows if r[4]})
    print("distinct geometries before %s: %s" % (MAP_B_FROM, a))
    print("distinct geometries from   %s: %s" % (MAP_B_FROM, b))
    if a and b and set(a) & set(b):
        print("WARNING: a date group on each side of the boundary shares a geometry.")
    if len(a) > 1 or len(b) > 1:
        print("WARNING: a date group is not internally consistent -- inspect before using.")

    print()
    print("%d maps written to %s" % (len(rows), outdir))
    if problems:
        print("%d session(s) need attention:" % len(problems))
        for sid, why in problems:
            print("   %s  %s" % (sid, why))


if __name__ == "__main__":
    main()
