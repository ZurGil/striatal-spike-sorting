# -*- coding: utf-8 -*-
"""Where the logistic comes from: one straight line, seen on three scales.

    python scripts/explain_logistic.py

Gil asked why log-odds are linear in DV and why that is the same equation as
1/(1+exp(-slope*(DV-bias))). The algebra is in the research log; this is the
picture that goes with it. The SAME five trials are marked in all three panels,
so it is visible that nothing changes between them except the vertical scale:

  A  log-odds       log(p/(1-p))   a straight line, which is the model
  B  odds           p/(1-p)        exponentiating bends it, floor at 0, no ceiling
  C  probability    p              dividing by (1+odds) adds the ceiling at 1

Real parameters from 20260917_100043: slope 7.5, bias 0.039.
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_LINE = "#2f6f9f"
C_MARK = "#c2502d"

SLOPE = 7.5
BIAS = 0.039
MARKS = np.array([-0.6, -0.2, BIAS, 0.2, 0.6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_coding"))
    args = ap.parse_args()

    dv = np.linspace(-1, 1, 500)
    x = SLOPE * (dv - BIAS)             # the log-odds
    odds = np.exp(x)
    p = 1.0 / (1.0 + np.exp(-x))

    xm = SLOPE * (MARKS - BIAS)
    om = np.exp(xm)
    pm = 1.0 / (1.0 + np.exp(-xm))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # ---- A: the model is a straight line on the log-odds scale
    ax = axes[0]
    ax.plot(dv, x, "-", color=C_LINE, lw=2.4)
    ax.plot(MARKS, xm, "o", color=C_MARK, ms=7, zorder=5)
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(-9, 9)
    ax.set_xlabel("DV   (positive = left is the rewarded side)")
    ax.set_ylabel(r"log-odds   $\log\frac{p}{1-p}$")
    ax.set_title("A   The model: a straight line")
    ax.text(0.03, 0.96, "runs $-\\infty$ to $+\\infty$, so a line is safe here\n"
                        "symmetric: swap left/right and the sign flips\n"
                        "slope = %.1f,  crosses 0 at DV = bias = %.3f" % (SLOPE, BIAS),
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)

    # ---- B: exponentiate
    ax = axes[1]
    ax.plot(dv, odds, "-", color=C_LINE, lw=2.4)
    ax.plot(MARKS, om, "o", color=C_MARK, ms=7, zorder=5)
    ax.axhline(1, color=MUTED, lw=0.9, ls=":")
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("DV")
    ax.set_ylabel(r"odds   $\frac{p}{1-p}$   (log axis)")
    ax.set_title(r"B   Step 1: $e^{(\,)}$ both sides")
    ax.text(0.03, 0.96, "odds = 1 means even money\n"
                        "floor at 0, but still no ceiling\n"
                        "and lopsided: 3 vs 1/3, not $\\pm$3",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)

    # ---- C: and divide
    ax = axes[2]
    ax.plot(dv, p, "-", color=C_LINE, lw=2.4)
    ax.plot(MARKS, pm, "o", color=C_MARK, ms=7, zorder=5)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls="-")
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("DV")
    ax.set_ylabel(r"probability   $p = \frac{1}{1+e^{-x}}$")
    ax.set_title(r"C   Step 2: $p = \frac{odds}{1+odds}$")
    ax.text(0.03, 0.96, "now it has a ceiling at 1\n"
                        "the S-shape was never assumed --\n"
                        "it is the straight line, squashed",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)

    # the same five trials, written out underneath
    rows = ["DV = %+.2f:   log-odds %+5.2f   odds %8.3f   p %.3f" % (d, a, b, c)
            for d, a, b, c in zip(MARKS, xm, om, pm)]
    fig.text(0.5, 0.015, "the five marked trials:      " + "        ".join(rows[:3]),
             ha="center", fontsize=8, color=MUTED, family="monospace")

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("One straight line, three scales   --   the same five trials marked "
                 "in each panel", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "logistic_three_scales.png")
    fig.savefig(path, dpi=155)
    print("wrote", path)


if __name__ == "__main__":
    main()
