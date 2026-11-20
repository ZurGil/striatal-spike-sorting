# -*- coding: utf-8 -*-
"""What the confidence regressor actually is, drawn in three steps.

    python scripts/explain_confidence_definition.py --session 20260917_100043

Gil asked how conf_norm is computed and whether it is just the stimulus. The
answer is three moves, and each one gets a panel:

  A  the psychometric fit        P(choose left) against DV, the rat's own choices
                                 behind it, with bias, slope and the lapse band
                                 marked, because those three numbers are the model
  B  the mirror                  the same curve read as confidence: p_left if the
                                 rat went left, 1 - p_left if it went right. Same
                                 stimulus, opposite confidence, depending only on
                                 the answer given -- which is the whole difference
                                 between confidence and |DV|
  C  what that buys              conf_norm against |DV|, correct and error apart.
                                 On correct trials the two agree (r ~ +0.8); on
                                 errors they are opposite (r ~ -0.85). The count of
                                 high-evidence errors is printed on the panel,
                                 because that handful of trials is what the whole
                                 distinction rests on.

Writes outputs/confidence_coding/confidence_definition_<session>.png.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from confidence_coding_quick_check import (SESSIONS, build_trials, fit_psychometric,
                                           normative_confidence)

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_LEFT = "#2f6f9f"
C_RIGHT = "#c2502d"
C_FIT = "#1c1c1c"
C_BAND = "#d9d2e9"


def panel_a(ax, t, p):
    bias, slope, lapse = p
    # the rat's actual choices, binned, as the thing the curve is fitted to
    edges = np.linspace(-1, 1, 13)
    idx = np.clip(np.digitize(t.DV.values, edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        x = 0.5 * (edges[b] + edges[b + 1])
        y = t.ChoiceLeft.values[m].mean()
        se = np.sqrt(max(y * (1 - y), 1e-6) / m.sum())
        ax.errorbar(x, y, yerr=se, fmt="o", color=MUTED, ms=5, capsize=2, zorder=3)

    g = np.linspace(-1, 1, 400)
    core = 1.0 / (1.0 + np.exp(-slope * (g - bias)))
    full = lapse + (1.0 - 2.0 * lapse) * core
    ax.axhspan(0, lapse, color=C_BAND, zorder=0)
    ax.axhspan(1 - lapse, 1, color=C_BAND, zorder=0)
    ax.plot(g, full, "-", color=C_FIT, lw=2.4, zorder=4,
            label=r"$\lambda + (1-2\lambda)\,\sigma$   (fitted to the choices)")
    ax.plot(g, core, "--", color=C_LEFT, lw=1.6, zorder=4,
            label=r"$\sigma$ alone   (used for confidence)")

    ax.axvline(bias, color=MUTED, lw=0.9, ls=":")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.annotate("bias = %.3f\n(P = 0.5 here)" % bias, xy=(bias, 0.5),
                xytext=(bias + 0.18, 0.30), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.annotate(r"lapse band, height $\lambda$ = %.3f" % lapse, xy=(-0.95, lapse),
                xytext=(-0.95, 0.17), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    # the slope, as a tangent at the midpoint
    s = slope * 0.25 * 0.5
    ax.plot([bias - 0.18, bias + 0.18], [0.5 - s * 0.36, 0.5 + s * 0.36],
            "-", color=C_RIGHT, lw=1.6, zorder=5)
    ax.text(bias + 0.2, 0.62, "slope = %.1f\nlog-odds per unit DV" % slope,
            fontsize=8, color=C_RIGHT)

    ax.set_xlabel("DV   (positive = left is the rewarded side)")
    ax.set_ylabel("P(the rat chooses left)")
    ax.set_title("A   Step 1: fit the rat's own psychometric curve")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def panel_b(ax, p):
    bias, slope, _ = p
    g = np.linspace(-1, 1, 400)
    p_left = 1.0 / (1.0 + np.exp(-slope * (g - bias)))
    ax.plot(g, p_left, "-", color=C_LEFT, lw=2.4, label="rat chose LEFT    conf = $p_{left}$")
    ax.plot(g, 1 - p_left, "-", color=C_RIGHT, lw=2.4,
            label="rat chose RIGHT  conf = $1-p_{left}$")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.axvline(0, color=MUTED, lw=0.9, ls=":")

    for dv, lab in ((0.8, "easy trial, left is correct"),):
        pl = 1.0 / (1.0 + np.exp(-slope * (dv - bias)))
        ax.plot([dv, dv], [1 - pl, pl], color=MUTED, lw=0.9, ls="-", zorder=1)
        ax.plot(dv, pl, "o", color=C_LEFT, ms=7, zorder=5)
        ax.plot(dv, 1 - pl, "o", color=C_RIGHT, ms=7, zorder=5)
        ax.annotate("went with the evidence\nconf = %.3f" % pl, xy=(dv, pl),
                    xytext=(dv - 0.92, 0.88), fontsize=8, color=C_LEFT,
                    arrowprops=dict(arrowstyle="->", color=C_LEFT, lw=0.9))
        ax.annotate("same stimulus, went against it\nconf = %.3f" % (1 - pl),
                    xy=(dv, 1 - pl), xytext=(dv - 0.95, 0.09), fontsize=8,
                    color=C_RIGHT,
                    arrowprops=dict(arrowstyle="->", color=C_RIGHT, lw=0.9))

    ax.set_xlabel("DV   (positive = left is the rewarded side)")
    ax.set_ylabel("conf_norm")
    ax.set_title("B   Step 2: mirror it by the choice the rat actually made")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="center left")


def panel_c(ax, t):
    a = t.DV.abs().values
    c = t.correct.values > 0.5
    ax.plot(a[c] + np.random.default_rng(0).normal(0, .008, c.sum()),
            t.conf_norm.values[c], ".", color=C_LEFT, ms=4, alpha=0.5)
    ax.plot(a[~c] + np.random.default_rng(1).normal(0, .008, (~c).sum()),
            t.conf_norm.values[~c], ".", color=C_RIGHT, ms=5, alpha=0.8)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    rc = np.corrcoef(a[c], t.conf_norm.values[c])[0, 1]
    re = np.corrcoef(a[~c], t.conf_norm.values[~c])[0, 1]
    # labelled on the clouds themselves; a legend box has nowhere to sit here
    ax.text(0.34, 0.95, "correct   n = %d,  r = %+.2f" % (c.sum(), rc),
            transform=ax.transAxes, color=C_LEFT, fontsize=9, va="top")
    ax.text(0.04, 0.07, "error   n = %d,  r = %+.2f" % ((~c).sum(), re),
            transform=ax.transAxes, color=C_RIGHT, fontsize=9, va="bottom")
    n_hi = int(((~c) & (a >= 0.4)).sum())
    ax.text(0.52, 0.47, "the two regressors part company\nonly out here: %d error "
                        "trials\nat |DV| >= 0.4" % n_hi,
            transform=ax.transAxes, ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("evidence strength  |DV|")
    ax.set_ylabel("conf_norm")
    ax.set_title("C   Why it is not just the stimulus")
    ax.set_ylim(-0.02, 1.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20260917_100043")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_coding"))
    args = ap.parse_args()

    t = build_trials(SESSIONS[args.session], args.session)
    p = fit_psychometric(t.DV.values, t.ChoiceLeft.values)
    t = t.copy()
    t["conf_norm"] = normative_confidence(t.DV.values, t.ChoiceLeft.values, p)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    panel_a(axes[0], t, p)
    panel_b(axes[1], p)
    panel_c(axes[2], t)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("How conf_norm is built   --   %s,  %d completed trials,  "
                 "bias %.3f, slope %.1f, lapse %.3f"
                 % (args.session, len(t), p[0], p[1], p[2]),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "confidence_definition_%s.png" % args.session)
    fig.savefig(path, dpi=155)
    print("wrote", path)


if __name__ == "__main__":
    main()
