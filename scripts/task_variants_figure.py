# -*- coding: utf-8 -*-
"""Two proposed changes to the task, drawn against the task as it runs today.

    python scripts/task_variants_figure.py

A  the trial as it is now, with the real numbers from 16/17/18 September
B  Gil's proposal: the cue runs as long as the rat stays in the centre port
C  post-decision evidence: a short extra burst AFTER the choice is committed
D  what C buys -- the two hypotheses it separates, and what each predicts

The timings in A are measured, not drawn from memory: fixation 0.2-0.4 s
(StimDelay), stimulus 0.3501 s on every completed trial, movement 0.25 s median,
waiting ~6.5 s median, 41-43 % of starts lost to early withdrawal.

Writes outputs/task_design/task_variants.png. Nothing on Z is touched.
"""
from __future__ import print_function

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "task_design")

INK = "#23232a"
GREY = "#d7d9de"
FIX = "#9aa3ae"
STIM = "#3c6e47"
MOVE = "#b8bcc4"
WAIT = "#1f6f8b"
NEW = "#c1553b"
MUT = "#6d7179"


def bar(ax, x0, w, y, h, color, label, sub=None, hatch=None, text_col="white"):
    p = FancyBboxPatch((x0, y - h / 2.0), w, h,
                       boxstyle="round,pad=0,rounding_size=0.012",
                       linewidth=0, facecolor=color, hatch=hatch)
    ax.add_patch(p)
    if label:
        ax.text(x0 + w / 2.0, y, label, ha="center", va="center",
                fontsize=8.5, color=text_col, fontweight="bold")
    if sub:
        ax.text(x0 + w / 2.0, y - h / 2.0 - 0.055, sub, ha="center", va="top",
                fontsize=7.6, color=MUT)


def frame(ax, title, note=None):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    if note:
        ax.text(0.0, 0.03, note, fontsize=8.2, color=MUT, va="bottom")


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    fig = plt.figure(figsize=(14.0, 9.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.15, 1.85],
                          hspace=0.75, left=0.055, right=0.975,
                          top=0.915, bottom=0.035)

    y, h = 0.50, 0.30

    # --------------------------------------------------------------- A  as it runs
    ax = fig.add_subplot(gs[0])
    bar(ax, 0.00, 0.14, y, h, FIX, "fixation", "0.2-0.4 s")
    bar(ax, 0.145, 0.16, y, h, STIM, "CLICKS", "0.3501 s, fixed")
    bar(ax, 0.31, 0.09, y, h, MOVE, "move", "0.25 s", text_col=INK)
    bar(ax, 0.405, 0.50, y, h, WAIT, "WAIT AT THE SIDE PORT",
        "6.5 s median -- the confidence report")
    ax.annotate("", xy=(0.145, y + h / 2 + 0.09), xytext=(0.30, y + h / 2 + 0.09),
                arrowprops=dict(arrowstyle="<->", color=NEW, lw=1.3))
    ax.text(0.315, y + h / 2 + 0.09,
            "41-43 % of starts never get past here (early withdrawal)",
            ha="left", va="center", fontsize=8.2, color=NEW)
    frame(ax, "A   the task as it runs today  (16, 17, 18 September)",
          "Every completed trial hears exactly 0.3501 s. The rat has no say in how "
          "much evidence it gets, and 85-89 % correct says 0.35 s is already enough.")

    # --------------------------------------------------------------- B  free cue
    ax = fig.add_subplot(gs[1])
    yb = 0.70
    bar(ax, 0.00, 0.14, yb, h, FIX, "fixation", "0.2-0.4 s")
    bar(ax, 0.145, 0.07, yb, h, STIM, "min", "0.05-0.10 s")
    bar(ax, 0.215, 0.20, yb, h, STIM, "CLICKS CONTINUE WHILE HE STAYS",
        "the rat ends it -> sampling time is the read-out", hatch="//")
    bar(ax, 0.42, 0.09, yb, h, MOVE, "move", None, text_col=INK)
    bar(ax, 0.515, 0.39, yb, h, WAIT, "WAIT AT THE SIDE PORT", None)
    frame(ax, "B   Gil's proposal: the cue lasts as long as he pokes",
          "Gains a second confidence report (how long he listens) and a per-trial "
          "commitment time, which is what a drift-diffusion bound is fitted to.\n"
          "Risk, from A: he already leaves 42 % of the time before 0.35 s. Drop the "
          "minimum BELOW current behaviour and make listening longer pay, or it floors.")

    # --------------------------------------------------------------- C  post-dec
    ax = fig.add_subplot(gs[2])
    y2 = 0.52
    bar(ax, 0.00, 0.14, y2, h, FIX, "fixation", None)
    bar(ax, 0.145, 0.16, y2, h, STIM, "CLICKS", "the decision is made from these")
    bar(ax, 0.31, 0.09, y2, h, MOVE, "move", None, text_col=INK)
    bar(ax, 0.405, 0.085, y2, h, NEW, "BURST", "100-200 ms, 30 % of trials")
    bar(ax, 0.495, 0.41, y2, h, WAIT, "WAIT AT THE SIDE PORT",
        "does he wait longer or give up sooner?")
    ax.annotate("", xy=(0.405, y2 + h / 2 + 0.04), xytext=(0.405, y2 + h / 2 + 0.22),
                arrowprops=dict(arrowstyle="-|>", color=NEW, lw=1.4))
    ax.text(0.415, y2 + h / 2 + 0.17,
            "the choice is already committed and cannot be undone --\n"
            "the ONLY thing this burst can change is what he believes about it",
            fontsize=8.4, color=NEW, va="bottom")
    frame(ax, "C   post-decision evidence: the same burst, pointing two ways",
          "Three kinds of burst, physically matched -- same duration, same total click "
          "count, same loudness -- differing only in which side has more:\n"
          "   CONGRUENT  (supports the side he chose)      NEUTRAL  50/50       "
          "INCONGRUENT  (supports the other side)")

    # --------------------------------------------------------------- D  what you get
    ax = fig.add_subplot(gs[3])
    ax.axis("off")
    inner = ax.inset_axes([0.035, 0.12, 0.215, 0.78])
    x = np.array([0, 1, 2])
    inner.plot(x, [6.5, 6.5, 6.5], "-o", color=MUT, lw=2.2, ms=7,
               label="stimulus echo")
    inner.plot(x, [5.2, 6.5, 7.8], "-o", color=NEW, lw=2.2, ms=7,
               label="true confidence")
    inner.set_xticks(x)
    inner.set_xticklabels(["incong.", "neutral", "cong."], fontsize=8.5)
    inner.set_ylim(4.4, 8.6)
    inner.set_ylabel("waiting time (s)", fontsize=9)
    inner.set_title("predicted behaviour", fontsize=9.5, loc="left")
    inner.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        inner.spines[s].set_visible(False)

    inner2 = ax.inset_axes([0.305, 0.12, 0.215, 0.78])
    inner2.plot(x, [12.0, 12.0, 12.0], "-o", color=MUT, lw=2.2, ms=7)
    inner2.plot(x, [9.4, 12.0, 14.6], "-o", color=NEW, lw=2.2, ms=7)
    inner2.set_xticks(x)
    inner2.set_xticklabels(["incong.", "neutral", "cong."], fontsize=8.5)
    inner2.set_ylim(8.0, 16.0)
    inner2.set_ylabel("firing rate (Hz)", fontsize=9)
    inner2.set_title("predicted neurons, same trials", fontsize=9.5, loc="left")
    for s in ("top", "right"):
        inner2.spines[s].set_visible(False)

    txt = "\n".join([
        "WHAT C BUYS YOU",
        "",
        "A cell firing more on easy trials at the choice port has two",
        "explanations, and today's data cannot separate them:",
        "   (1) it is still echoing the stimulus it just heard",
        "   (2) it encodes how likely the choice just made was right",
        "",
        "The burst changes the belief WITHOUT changing the stimulus the",
        "decision came from, without changing the decision, and without",
        "changing the physical sound: congruent and incongruent bursts",
        "are identical in energy, duration and click count. Only their",
        "meaning relative to his own choice differs.",
        "",
        "So an echo MUST be flat across the three conditions. A real",
        "confidence signal must move, monotonically, and must move the",
        "waiting time on the SAME trials. Nothing else in the task",
        "delivers that contrast.",
    ])
    ax.text(0.565, 0.98, txt, fontsize=7.7, va="top", ha="left", color=INK,
            family="monospace", linespacing=1.32)

    fig.suptitle("Two changes to the task: let him choose how much to hear, "
                 "and tell him something after he has already answered",
                 fontsize=13.5, y=0.962)
    p = os.path.join(OUT, "task_variants.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
