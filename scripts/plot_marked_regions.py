# -*- coding: utf-8 -*-
"""Draw a unit's raw-data scatter with the hand-drawn regions saved against it.

    python scripts/plot_marked_regions.py 20260925_103539 100 <dir-of-db-json>

Reads the same `cluster_points_<sid>.json` the review page draws from, so this is
the same cloud Gil is looking at, and overlays every polygon in the artifact's
`reassignments` collection for that unit. Nothing is written to Z.
"""
from __future__ import print_function

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = ['#d9a441', '#2f6f9f', '#3f8f5f', '#b0552f', '#7f4f9f', '#3f9f9f',
           '#9f4f7f', '#6f7f2f', '#4f6fd9', '#9f2f4f', '#2f9f6f', '#8f7f3f']


def main():
    sid, uid, dbdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cp = json.load(open(os.path.join(ROOT, "outputs", "cluster_points_%s.json" % sid)))
    u = cp[str(uid)]

    regions = []
    for f in sorted(glob.glob(os.path.join(dbdir, "*.json"))):
        d = json.load(open(f))
        if int(d.get("source_unit", -1)) == uid:
            regions.append(d)
    print("%d region(s) for unit %d" % (len(regions), uid))

    xf, yf = regions[0]["x_field"], regions[0]["y_field"]

    # the page pools every ticked window; all ten were ticked on all three regions
    cats = {"own": [], "other": [], "noise": []}
    unit_of = []
    for wkey in sorted(u["windows"], key=lambda z: int(z)):
        w = u["windows"][wkey]
        for cat in ("own", "other", "noise"):
            g = w.get(cat)
            if not g or xf not in g or yf not in g:
                continue
            for i in range(len(g[xf])):
                cats[cat].append((g[xf][i], g[yf][i]))
                if cat == "other":
                    unit_of.append(w.get("other_unit", [None] * len(g[xf]))[i])

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    n = np.array(cats["noise"])
    if len(n):
        ax.scatter(n[:, 0], n[:, 1], s=4, c="#9aacbd", alpha=0.3, linewidths=0,
                   label="unassigned")
    o = np.array(cats["other"])
    colmap, order = {}, []
    for q in unit_of:
        if q is not None and q not in colmap:
            colmap[q] = PALETTE[len(colmap) % len(PALETTE)]
            order.append(q)
    if len(o):
        cols = [colmap.get(q, "#888888") for q in unit_of]
        ax.scatter(o[:, 0], o[:, 1], s=5, c=cols, alpha=0.75, linewidths=0)
    w_ = np.array(cats["own"])
    ax.scatter(w_[:, 0], w_[:, 1], s=7, c="#c2453f", alpha=0.8, linewidths=0,
               label="unit %d (this unit)" % uid)

    import datetime
    styles = ["-", "--", ":"]
    for i, r in enumerate(regions):
        p = np.array(r["polygon"] + [r["polygon"][0]])
        t = datetime.datetime.utcfromtimestamp(r["ts"] / 1000.0).strftime("%H:%M:%S")
        ax.plot(p[:, 0], p[:, 1], styles[i % 3], color="#111111", lw=1.8,
                label="%s  %d dots -> unit %d (~%d spikes)"
                      % (t, r["n_selected_dots"], r["target_unit"], r["n_estimated"]))

    for q in order:
        ax.scatter([], [], s=24, c=colmap[q], label="unit %d" % q)
    ax.set_xlabel(xf)
    ax.set_ylabel(yf)
    ax.set_title("%s  unit %d  --  every region saved against it, all 10 windows"
                 % (sid, uid), loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = os.path.join(ROOT, "outputs", "unit_%d_marked_regions_%s.png" % (uid, sid))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)
    print("own %d, other %d, unassigned %d; other units present: %s"
          % (len(cats["own"]), len(cats["other"]), len(cats["noise"]), order))


if __name__ == "__main__":
    main()
