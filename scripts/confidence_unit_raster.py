# -*- coding: utf-8 -*-
"""One unit, every trial, drawn -- so the confidence result can be seen, not inferred.

    python scripts/confidence_unit_raster.py
    python scripts/confidence_unit_raster.py --session 20260918_100248 --unit 12

Gil, after the summary figure: "it would be cool if I could actually see the neuron
activity and also see the results". So this is the same claim as panel C of
confidence_figure.png, with the raw spikes in front of it:

  A  raster, aligned to the END of the cue. Correct trials on top, errors below,
     each block sorted by evidence strength with the weakest evidence at the
     bottom. Every dot is a spike. The open marker on each row is the moment the
     rat poked the choice port. The shaded band is the 0-200 ms window the
     analysis counted in -- no window was drawn around anything after the fact.
  B  the same trials as firing rate, correct only, split by evidence strength.
  C  the same for errors.
  D  what the analysis actually fitted: mean rate in the shaded band against
     evidence, correct and error apart.

Writes outputs/confidence_coding/raster_<session>_unit<id>.png.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from confidence_coding_quick_check import SESSIONS, build_trials, epoch_windows

OUT = os.path.join(ROOT, "outputs", "confidence_coding")

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_CORRECT = "#2f6f9f"
C_ERROR = "#c2502d"
# Sequential ramps: one hue, light to dark, because evidence strength is a magnitude.
RAMP_CORRECT = ["#bcd4e6", "#89b2d1", "#4f84b0", "#20496e"]
RAMP_ERROR = ["#f0b9a4", "#b8441f"]
WINDOW = (0.0, 0.20)
PRE, POST = 0.50, 0.80
BIN = 0.020
SIGMA = 0.030

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def spike_times_by_trial(S, uid, t):
    """Spike times relative to cue end, one array per trial, for one unit."""
    sp = pd.read_parquet(S + "/spikes.parquet",
                         columns=["unit_id", "trial_id", "spike_time_in_trial"],
                         filters=[("unit_id", "==", uid)])
    pos = pd.Series(np.arange(len(t)), index=t.trial_id.values)
    idx = sp.trial_id.map(pos)
    ok = idx.notna()
    idx = idx[ok].values.astype(int)
    rel = sp.spike_time_in_trial.values[ok.values] - t.cue_end.values[idx]
    order = np.argsort(idx, kind="stable")
    idx, rel = idx[order], rel[order]
    bounds = np.searchsorted(idx, np.arange(len(t) + 1))
    return [rel[bounds[i]:bounds[i + 1]] for i in range(len(t))]


def smoothed_psth(trains, rows):
    """Gaussian-smoothed mean rate over a set of trials, in Hz."""
    edges = np.arange(-PRE, POST + BIN, BIN)
    centres = edges[:-1] + BIN / 2.0
    if not len(rows):
        return centres, np.zeros_like(centres), np.zeros_like(centres)
    counts = np.zeros((len(rows), len(centres)))
    for k, i in enumerate(rows):
        counts[k], _ = np.histogram(trains[i], bins=edges)
    rate = counts / BIN
    half = int(round(3 * SIGMA / BIN))
    g = np.exp(-0.5 * (np.arange(-half, half + 1) * BIN / SIGMA) ** 2)
    g /= g.sum()
    sm = np.array([np.convolve(r, g, mode="same") for r in rate])
    return centres, sm.mean(axis=0), sm.std(axis=0) / max(np.sqrt(len(rows)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20260917_100043")
    ap.add_argument("--unit", type=int, default=287)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    S = SESSIONS[args.session]
    t = build_trials(S, args.session)
    trains = spike_times_by_trial(S, args.unit, t)

    win = epoch_windows(t)
    a, b = win["cue"]
    dur = np.maximum(b - a, 1e-6)
    rate_win = np.array([((v >= 0) & (v < dur[i])).sum() / dur[i]
                         for i, v in enumerate(trains)])
    mt = t.choice_in.values - t.cue_end.values
    absdv = t.absDV.values
    correct = t.correct.values > 0.5

    # Correct trials in four evidence bins, errors in two -- there are only a handful
    # of errors at strong evidence, and four bins would put three trials in one.
    cedges = np.array([0, .2, .4, .6, 1.0001])
    eedges = np.array([0, .3, 1.0001])
    cbin = np.clip(np.digitize(absdv, cedges) - 1, 0, 3)
    ebin = np.clip(np.digitize(absdv, eedges) - 1, 0, 1)

    corr_rows = np.argsort(absdv + np.where(correct, 0, 1e6))[:correct.sum()]
    err_rows = np.argsort(absdv + np.where(~correct, 0, 1e6))[:(~correct).sum()]

    fig = plt.figure(figsize=(14.6, 8.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1.05],
                          hspace=0.42, wspace=0.26)
    ax_r = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_s = fig.add_subplot(gs[2, 1])

    # ---------------------------------------------------------------- A: the raster
    GAP = 18
    y = 0
    bands = []
    # Errors drawn first so the correct block ends up on TOP, which is the way round
    # everyone reads these. Within each block the weakest evidence is at the bottom.
    for rows, bins, ramp, label, col in ((err_rows, ebin, RAMP_ERROR, "ERROR", C_ERROR),
                                         (corr_rows, cbin, RAMP_CORRECT, "CORRECT", C_CORRECT)):
        y0 = y
        xs, ys, cs = [], [], []
        for i in rows:
            v = trains[i]
            v = v[(v >= -PRE) & (v <= POST)]
            xs.append(v)
            ys.append(np.full(len(v), y))
            cs.append(ramp[bins[i]])
            y += 1
        for v, yy, c in zip(xs, ys, cs):
            ax_r.plot(v, yy, "|", color=c, ms=3.0, mew=0.75, zorder=3)
        ax_r.plot(mt[rows], np.arange(y0, y), "o", mfc="none", mec=MUTED, ms=2.4,
                  mew=0.7, zorder=4)
        bands.append((y0, y, label, col, rows, bins, ramp))
        y += GAP

    ax_r.add_patch(Rectangle((WINDOW[0], -5), WINDOW[1] - WINDOW[0], y + 5,
                             color="#f0c040", alpha=0.16, zorder=1, lw=0))
    ax_r.axvline(0, color=INK, lw=1.1, zorder=2)
    for y0, y1, label, col, rows, bins, ramp in bands:
        # Spikes already carry the evidence ramp, so the only extra mark the block
        # needs is a hairline where one evidence bin ends and the next begins.
        prev = None
        for k, i in enumerate(rows):
            if bins[i] != prev:
                if prev is not None:
                    ax_r.plot([-PRE, POST], [y0 + k, y0 + k], color="#ffffff",
                              lw=0.9, zorder=5)
                prev = bins[i]
        ax_r.text(POST - 0.012, y1 - 3, "%s  (%d)" % (label, y1 - y0), color=col,
                  fontsize=10, fontweight="bold", va="top", ha="right", zorder=7,
                  bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.6))
        # The "evidence strengthens upward" arrow goes OUTSIDE the data area; drawn
        # inside, it lands in the densest part of the raster and hides spikes.
        ax_r.annotate("", xy=(-0.022, y1 - 4), xytext=(-0.022, y0 + 4),
                      xycoords=("axes fraction", "data"),
                      textcoords=("axes fraction", "data"), annotation_clip=False,
                      arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax_r.set_xlim(-PRE, POST)
    ax_r.set_ylim(-4, y - GAP + 4)
    ax_r.set_yticks([])
    ax_r.set_xlabel("time from the end of the cue (s)")
    ax_r.set_ylabel("one row = one trial, sorted by evidence strength\n"
                    "(weakest at the bottom of each block)")
    ax_r.set_title("A   Unit %d, every completed trial" % args.unit)
    ax_r.text(WINDOW[1] + 0.012, y - GAP - 2, "the 200 ms the analysis counts",
              fontsize=8.5, color="#8a6d1f", va="top", zorder=7,
              bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.6))
    ax_r.text(POST - 0.012, -2.5, "open circle = choice port in", fontsize=8.5,
              color=MUTED, ha="right", va="top")

    # --------------------------------------------------------------- B, C: the PSTHs
    for ax, rows_all, bins, ramp, edges, title, col in (
            (ax_c, corr_rows, cbin, RAMP_CORRECT, cedges, "B   Correct trials", C_CORRECT),
            (ax_e, err_rows, ebin, RAMP_ERROR, eedges, "C   Error trials", C_ERROR)):
        for bi in range(len(edges) - 1):
            rows = [i for i in rows_all if bins[i] == bi]
            if len(rows) < 3:
                continue
            x, m, s = smoothed_psth(trains, rows)
            ax.fill_between(x, m - s, m + s, color=ramp[bi], alpha=0.18, lw=0)
            ax.plot(x, m, color=ramp[bi], lw=2.0,
                    label="|DV| %.1f-%.1f  (n=%d)" % (edges[bi], min(edges[bi + 1], 1.0),
                                                      len(rows)))
        ax.axvspan(WINDOW[0], WINDOW[1], color="#f0c040", alpha=0.16, lw=0)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_xlim(-PRE, POST)
        ax.set_ylabel("rate (Hz)")
        ax.set_title(title, color=col)
        ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.4)

        # The cue itself drives a 30 Hz transient, which is five times the effect
        # being asked about and flattens it to nothing at this scale. So the counted
        # window gets its own axes, on its own y scale.
        iz = ax.inset_axes([0.60, 0.46, 0.38, 0.50])
        for bi in range(len(edges) - 1):
            rows = [i for i in rows_all if bins[i] == bi]
            if len(rows) < 3:
                continue
            x, m, s = smoothed_psth(trains, rows)
            k = (x >= -0.03) & (x <= 0.28)
            iz.fill_between(x[k], (m - s)[k], (m + s)[k], color=ramp[bi], alpha=0.18, lw=0)
            iz.plot(x[k], m[k], color=ramp[bi], lw=1.8)
        iz.axvspan(WINDOW[0], WINDOW[1], color="#f0c040", alpha=0.16, lw=0)
        iz.set_xlim(-0.03, 0.28)
        iz.tick_params(labelsize=7.5)
        iz.set_title("zoom on the counted window", fontsize=8, fontweight="normal",
                     color=MUTED, pad=3)
        for sp in ("top", "right"):
            iz.spines[sp].set_visible(False)
    ax_e.set_xlabel("time from the end of the cue (s)")

    # ------------------------------------------------------------------ D: the result
    for lab, col, sel, edges in (("correct", C_CORRECT, correct, cedges),
                                 ("error", C_ERROR, ~correct, eedges)):
        cs, ms, ss, ns = [], [], [], []
        idx = np.digitize(absdv, edges) - 1
        for bi in range(len(edges) - 1):
            v = rate_win[sel & (idx == bi)]
            if len(v) < 3:
                continue
            cs.append(0.5 * (edges[bi] + min(edges[bi + 1], 1.0)))
            ms.append(v.mean()); ss.append(v.std() / np.sqrt(len(v))); ns.append(len(v))
        ax_s.errorbar(cs, ms, yerr=ss, fmt="o-", color=col, lw=2.2, ms=6,
                      capsize=2, label=lab)
        for x, m, n in zip(cs, ms, ns):
            ax_s.annotate("n=%d" % n, (x, m), textcoords="offset points",
                          xytext=(0, -15 if col == C_ERROR else 10), ha="center",
                          fontsize=7.5, color=col)
    ax_s.set_xlabel("evidence strength  |DV|")
    ax_s.set_ylabel("rate in the shaded\n200 ms window (Hz)")
    ax_s.set_title("D   What the analysis fitted")
    ax_s.set_xlim(-0.02, 1.0)
    ax_s.legend(frameon=False, fontsize=9, loc="upper left")

    stats = ("%s unit %d  |  %d completed trials (%d correct, %d error)  |  "
             "mean rate %.1f Hz" % (args.session, args.unit, len(t), int(correct.sum()),
                                    int((~correct).sum()), rate_win.mean()))
    fig.suptitle("Seeing the confidence effect in one neuron\n" + stats,
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.012, 0.012, 1, 0.935))
    p = args.out or os.path.join(OUT, "raster_%s_unit%d.png" % (args.session, args.unit))
    fig.savefig(p, dpi=170)
    print("wrote", p)


if __name__ == "__main__":
    main()
