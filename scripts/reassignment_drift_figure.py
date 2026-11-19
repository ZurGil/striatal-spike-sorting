# -*- coding: utf-8 -*-
"""Does a region drawn on 20 minutes still mean the same thing 2 hours later?

    python scripts/reassignment_drift_figure.py 20260925_103539

Gil chose (2026-11-19) that a hand-drawn region applies to the WHOLE session rather
than only inside the ten windows it was drawn on, with a drift check. This draws the
check. It reads what apply_reassignments.py measured -- it never re-reads raw data --
and answers three questions in one page:

A  WHAT EACH REGION ACTUALLY DOES, once it is applied everywhere: how many spikes it
   moves, and whether those spikes are being taken from the unit Gil thinks they are.

B  WHERE THE GREY DOTS COME FROM. The review page draws a detection as "unassigned"
   when it matched no sorted spike OR when it matched a unit that does not peak on
   this channel. Those are not the same thing, and a region sourced from the grey pile
   treats them as if they were.

C  THE DRIFT ITSELF. For every region, the fraction of its source events that fall
   inside the boundary, per 2 minutes of recording, with the ten drawn windows marked.
   A flat line means the boundary means the same thing all session. A sloping one
   means the cloud moved underneath it, and the region should not be applied outside
   the windows without looking again.

Writes outputs/reassignment_drift_<session>.png. Nothing on Z is touched.
"""
from __future__ import print_function

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

INK = "#1d2124"
MUT = "#6b7280"
RULE = "#d7dade"
OK = "#2f7a53"
WARN = "#b4541f"
BAD = "#9d2a26"
COOL = "#2b6f9c"


def main():
    sid = sys.argv[1]
    doc = json.load(open(os.path.join(OUT, "reassignment_manifest_%s.json" % sid)))
    arr = np.load(os.path.join(OUT, "reassignment_manifest_%s.npz" % sid))
    recs = doc["records"]
    bin_s = doc["bin_seconds"]
    drawn = set(doc["drawn_bins"])

    n = len(recs)
    fig = plt.figure(figsize=(16.5, 4.6 + 1.45 * n))
    # hspace is a fraction of the AVERAGE row height, and these two rows differ by a
    # factor of eight, so anything but a hair here opens a gap as tall as the top row.
    gs = GridSpec(2, 1, height_ratios=[3.05, 1.45 * n], hspace=0.03,
                  left=0.055, right=0.985, top=0.955, bottom=0.035)

    # ------------------------------------------------------------------ A + B, on top
    top = fig.add_subplot(gs[0])
    top.axis("off")
    labels, relab, offch, ins = [], [], [], []
    for r in recs:
        src = ", ".join("unassigned" if s is None else str(s) for s in r["sources"])
        tgt = "unassigned" if r["target_unit"] is None else str(r["target_unit"])
        labels.append("%s → %s" % (src, tgt))
        relab.append(r["relabel_spikes"])
        offch.append(sum(r["taken_from_offchannel_units"].values()))
        ins.append(r["insert_events"])
    relab, offch, ins = np.array(relab), np.array(offch), np.array(ins)
    y = np.arange(n)[::-1]

    ax = top.inset_axes([0.085, 0.0, 0.375, 1.0])
    ax.barh(y, relab - offch, height=0.62, color=COOL,
            label="moved from the unit named on the panel")
    ax.barh(y, offch, left=relab - offch, height=0.62, color=BAD,
            label="taken from a unit that peaks on another channel")
    ax.barh(y, ins, left=relab, height=0.62, color=OK,
            label="genuinely unsorted (a new spike)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel("spikes moved across the whole session", fontsize=9)
    ax.set_title("A   what each region does once it applies everywhere",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # B: where the grey dots come from, pooled
    ax2 = top.inset_axes([0.555, 0.0, 0.165, 1.0])
    pooled = {}
    for r in recs:
        for u, c in r["taken_from_offchannel_units"].items():
            pooled[int(u)] = pooled.get(int(u), 0) + int(c)
    tot_off = sum(pooled.values())
    tot_ins = int(ins.sum())
    top_u = sorted(pooled.items(), key=lambda kv: -kv[1])[:12]
    if top_u:
        yy = np.arange(len(top_u))[::-1]
        ax2.barh(yy, [c for _, c in top_u], height=0.62, color=BAD)
        ax2.set_yticks(yy)
        ax2.set_yticklabels(["unit %d" % u for u, _ in top_u], fontsize=7.6)
    ax2.set_xlabel("spikes it would lose", fontsize=9)
    ax2.set_title("B   who actually pays for the grey dots", fontsize=10.5,
                  loc="left", color=INK)
    ax2.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    txt = "\n".join([
        "THE GREY PILE IS NOT WHAT IT LOOKS LIKE",
        "",
        "The review page draws a detection grey when it matched",
        "no sorted spike OR when it matched a unit that does not",
        "peak on this channel. Nine of these regions take their",
        "events from that pile.",
        "",
        "Across the session those grey events are",
        "   %7d  spikes belonging to other units" % tot_off,
        "   %7d  genuinely unsorted" % tot_ins,
        "",
        "So 'sweeping up the unassigned' is overwhelmingly",
        "TAKING SPIKES AWAY FROM NEIGHBOURING UNITS, not",
        "recovering spikes the sorter missed. %d units would"
        % len(pooled),
        "lose spikes they currently own.",
        "",
        "That is a different decision from the one the button",
        "looks like it is offering, which is why it is drawn",
        "here before anything is applied.",
    ])
    top.text(0.745, 1.0, txt, fontsize=8.0, va="top", ha="left", color=INK,
             family="monospace", linespacing=1.4)

    # ------------------------------------------------------------------ C, the drift
    host = fig.add_subplot(gs[1])
    host.axis("off")
    host.set_title("C   the drift check: of each region's source events, the fraction "
                   "inside the boundary, per 2 minutes",
                   fontsize=10.5, loc="left", color=INK)
    h = 1.0 / n
    for i, r in enumerate(recs):
        frac = arr["frac_" + r["doc_id"]]
        nsrc = arr["nsrc_" + r["doc_id"]]
        a = host.inset_axes([0.085, 1.0 - (i + 1) * h, 0.9, h * 0.82])
        t = (np.arange(len(frac)) + 0.5) * bin_s / 60.0
        thin = nsrc < 20
        f = np.where(thin, np.nan, frac)
        for b in sorted(drawn):
            if b < len(frac):
                a.axvspan((b * bin_s) / 60.0, ((b + 1) * bin_s) / 60.0,
                          color="#e8eef3", zorder=0)
        a.plot(t, f, "-", color=COOL, lw=1.3)
        dw = np.array(sorted(b for b in drawn if b < len(frac)))
        if len(dw):
            a.plot(t[dw], f[dw], "o", color=INK, ms=3.2, zorder=3)
        fd = r["frac_inside_drawn_windows"]
        fr = r["frac_inside_rest_of_session"]
        if fd is not None and fr is not None and fd > 0:
            ratio = fr / fd
            col = OK if 0.75 <= ratio <= 1.33 else (WARN if 0.5 <= ratio <= 2.0 else BAD)
            verdict = ("holds" if col == OK else
                       ("shifts" if col == WARN else "does not hold"))
            a.text(0.995, 0.86, "%s  (drawn %.3f, rest %.3f)" % (verdict, fd, fr),
                   transform=a.transAxes, ha="right", va="top", fontsize=7.4, color=col)
        src = ", ".join("unassigned" if s is None else str(s) for s in r["sources"])
        tgt = "unassigned" if r["target_unit"] is None else str(r["target_unit"])
        a.set_ylabel("%s→%s" % (src, tgt), fontsize=7.0, rotation=0,
                     ha="right", va="center", labelpad=6, color=INK)
        a.set_ylim(-0.03, max(0.06, np.nanmax(f) * 1.25 if np.isfinite(f).any() else 1))
        a.set_xlim(0, len(frac) * bin_s / 60.0)
        a.tick_params(labelsize=7, length=2)
        if i < n - 1:
            a.set_xticklabels([])
        else:
            a.set_xlabel("minutes into the session   (shaded = the ten windows the "
                         "region was drawn on)", fontsize=9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle("%s  —  applying the hand-drawn regions to the whole session, "
                 "and what that would cost" % sid, fontsize=13, y=0.988)
    p = os.path.join(OUT, "reassignment_drift_%s.png" % sid)
    fig.savefig(p, dpi=135)
    print("wrote", p)
    print("relabels %d, of which %d off-channel; inserts %d"
          % (int(relab.sum()), int(offch.sum()), tot_ins))


if __name__ == "__main__":
    main()
