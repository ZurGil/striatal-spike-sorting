# -*- coding: utf-8 -*-
"""Why the psychometric curve is smooth: a step function plus noise.

    python scripts/explain_step_plus_noise.py

Gil, on the previous explanation: "I'm not sure I fully understood the bayes
part and how this true." The thing that was run together and should not have
been is that Bayes and the sigmoid are two separate claims:

  Bayes says the DECISION VARIABLE is linear in DV. An observer that simply
  takes the more probable side produces a STEP, not an S.
  The S comes from NOISE -- sampling noise in the stimulus plus internal noise.
  Push a step through noise and you get the noise's own CDF, read sideways.

Three panels:

  A  the noiseless ideal observer      a step at DV = bias
  B  the noise, drawn as a density     sitting on the DV axis at three widths
  C  step convolved with that noise    the psychometrics those widths produce,
                                       with the real fitted slope marked

Real parameters from 20260917_100043: slope 7.5 => logistic noise scale
s = 1/7.5 = 0.133 in DV units, bias 0.039.

Writes outputs/confidence_coding/step_plus_noise.png.
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
C_STEP = "#1c1c1c"
C_FIT = "#2f6f9f"
SHADES = ["#c9d9e8", "#7fa6c4", "#2f6f9f"]

BIAS = 0.039
SLOPE_FIT = 7.5
SCALES = [0.40, 0.25, 1.0 / SLOPE_FIT]      # logistic noise scale s, in DV units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_coding"))
    args = ap.parse_args()

    dv = np.linspace(-1, 1, 800)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # ---- A  the noiseless observer
    ax = axes[0]
    step = (dv > BIAS).astype(float)
    ax.plot(dv, step, "-", color=C_STEP, lw=2.6, drawstyle="steps-post")
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("DV")
    ax.set_ylabel("P(the rat chooses left)")
    ax.set_title("A   What Bayes alone predicts")
    ax.text(0.03, 0.97,
            "take the side with the higher\nposterior -- and you get a STEP.\n"
            "No S anywhere in the algebra.",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.annotate("flips at DV = bias = %.3f" % BIAS, xy=(BIAS, 0.5),
                xytext=(BIAS + 0.22, 0.42), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    # ---- B  the noise itself
    ax = axes[1]
    for s, col in zip(SCALES, SHADES):
        z = (dv - BIAS) / s
        pdf = np.exp(-z) / (s * (1 + np.exp(-z)) ** 2)      # logistic density
        ax.plot(dv, pdf, "-", color=col, lw=2.2,
                label="s = %.3f   (slope %.1f)" % (s, 1.0 / s))
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("DV")
    ax.set_ylabel("density of the noise  $\\varepsilon$")
    ax.set_title("B   The noise on the rat's estimate")
    ax.text(0.03, 0.97,
            "sampling noise in the sounds,\nplus the rat's own. Width s is\n"
            "the only thing that varies here.",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    # ---- C  the step pushed through the noise
    ax = axes[2]
    for s, col in zip(SCALES, SHADES):
        p = 1.0 / (1.0 + np.exp(-(dv - BIAS) / s))
        ax.plot(dv, p, "-", color=col, lw=2.4,
                label="s = %.3f  ->  slope = %.1f" % (s, 1.0 / s))
    ax.plot(dv, (dv > BIAS).astype(float), "-", color=C_STEP, lw=1.2, alpha=0.45,
            drawstyle="steps-post", label="no noise (panel A)")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax.axvline(BIAS, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("DV")
    ax.set_ylabel("P(the rat chooses left)")
    ax.set_title("C   Step + noise = the psychometric curve")
    ax.text(0.03, 0.97,
            "the curve IS the noise's CDF.\nslope = 1/s, so a steeper rat is\n"
            "simply a less noisy one.\n20260917 fitted slope 7.5 -> s = 0.133",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Bayes gives the step; noise gives the S   --   they are two "
                 "separate claims", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "step_plus_noise.png")
    fig.savefig(path, dpi=155)
    print("wrote", path)


if __name__ == "__main__":
    main()
