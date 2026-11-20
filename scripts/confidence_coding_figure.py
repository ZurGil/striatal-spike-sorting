# -*- coding: utf-8 -*-
"""The figure for the confidence analysis. Reads what the check already wrote.

    python scripts/confidence_coding_figure.py
    python scripts/confidence_coding_figure.py --dir outputs/confidence_coding_four \
        --sessions 20260916_110311 20260917_100043 20260918_100248 20261112_082028

Six panels, in the order the argument runs:

  A  the task works              psychometric curve, every session
  B  what confidence IS here     how long the rat waits with no reward, against
                                 evidence, split by whether it was right
  C  one neuron                  the strongest single unit, firing rate against
                                 evidence, correct and error apart
  D  how many neurons do it      share of units with a significant slope, per
                                 epoch, with the pre-stimulus baseline as control
  E  the population read-out     cross-validated R^2 for choice side, |DV| and
                                 |DV| within correct trials only
  F  the answer                  correct-vs-error decoded BEFORE feedback, as AUC,
                                 raw and with each cell's own baseline subtracted

Writes outputs/confidence_coding/confidence_figure.png.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from confidence_coding_quick_check import (SESSIONS, NO_ERROR_CATCH, EPOCHS,
                                           DEFAULT_SESSIONS,
                                           bh_fdr, build_trials, epoch_windows)

OUT = os.path.join(ROOT, "outputs", "confidence_coding")
# Match the check's default exactly. This used to be every key in SESSIONS, which
# silently pulled in the never-reviewed 20260911 and the other rat, so the figure
# described a different population from the numbers it was drawn beside.
USE = list(DEFAULT_SESSIONS)
SHAMIR = [s for s in USE if s not in NO_ERROR_CATCH]


def completed(t, sid):
    """The trials an analysis may spend, filtered the way the check filters them."""
    ok = t.ChoiceCorrect.notna() & t.sync_valid
    if "overlaps_recording_gap" in t.columns:
        ok &= ~t.overlaps_recording_gap
    else:
        # Syncs written before the gap column existed. Say so rather than quietly
        # counting trials the check itself would have dropped.
        print("warning: %s has no overlaps_recording_gap column; "
              "gap filter not applied" % sid)
    return t[ok].copy()

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_CORRECT = "#2f6f9f"      # right
C_ERROR = "#c2502d"        # wrong
C_SIDE = "#6b6b6b"
C_DV = "#2f6f9f"
C_DVC = "#4fa08b"
EPOCH_LABEL = {"base": "baseline\n(before cue)", "stim": "during\ncue",
               "cue": "cue end\n+0-200 ms", "chin": "choice port\n+0-450 ms",
               "wait": "waiting\n+0.5-1.5 s"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def binned(x, y, edges):
    """Mean, sem and n of y in bins of x."""
    idx = np.digitize(x, edges) - 1
    m, s, n, c = [], [], [], []
    for b in range(len(edges) - 1):
        v = y[idx == b]
        if len(v) == 0:
            m.append(np.nan); s.append(np.nan); n.append(0)
        else:
            m.append(v.mean()); s.append(v.std() / max(np.sqrt(len(v)), 1)); n.append(len(v))
        c.append(0.5 * (edges[b] + edges[b + 1]))
    return np.array(c), np.array(m), np.array(s), np.array(n)


# --------------------------------------------------------------- data for A and B

def behaviour():
    psy, inv = [], []
    for sid in USE:
        t = pd.read_parquet(SESSIONS[sid] + "/trials.parquet")
        c = completed(t, sid)
        c["sid"] = sid
        psy.append(c[["sid", "DV", "ChoiceLeft", "ChoiceCorrect"]])
        if sid in NO_ERROR_CATCH:       # the error wait there is our delay, not the rat's
            continue
        c["invest"] = (((c.CatchTrial > 0.5) & (c.ChoiceCorrect > 0.5))
                       | (c.ChoiceCorrect < 0.5))
        k = c[c.invest & c.FeedbackTime.notna()].copy()
        k["lw"] = np.log(k.FeedbackTime.clip(lower=0.05))
        inv.append(k)
    inv = pd.concat(inv)
    inv["lwz"] = inv.groupby("sid").lw.transform(lambda v: (v - v.mean()) / v.std())
    inv["absDV"] = inv.DV.abs()
    return pd.concat(psy), inv


def example_unit(sid="20260917_100043", uid=287, epoch="cue"):
    """Spike rate of one unit in one epoch, trial by trial."""
    S = SESSIONS[sid]
    t = build_trials(S, sid)
    sp = pd.read_parquet(S + "/spikes.parquet",
                         columns=["unit_id", "trial_id", "spike_time_in_trial"],
                         filters=[("unit_id", "==", uid)])
    win = epoch_windows(t)
    a, b = win[epoch]
    pos = pd.Series(np.arange(len(t)), index=t.trial_id.values)
    ti = sp.trial_id.map(pos)
    ok = ti.notna()
    ti = ti[ok].values.astype(int)
    tm = sp.spike_time_in_trial.values[ok.values]
    m = (tm >= a[ti]) & (tm < b[ti])
    cnt = np.bincount(ti[m], minlength=len(t))
    rate = cnt / np.maximum(b - a, 1e-6)
    return t.assign(rate=rate)


# ------------------------------------------------------------------------- panels

def panel_a(ax, psy):
    edges = np.linspace(-1, 1, 11)
    for sid, g in psy.groupby("sid"):
        c, m, s, n = binned(g.DV.values, g.ChoiceLeft.values, edges)
        ax.plot(c, m, "-", color=MUTED, lw=1.0, alpha=0.75, zorder=2)
    c, m, s, n = binned(psy.DV.values, psy.ChoiceLeft.values, edges)
    ax.errorbar(c, m, yerr=s, fmt="o-", color=INK, lw=2.2, ms=5, zorder=3,
                capsize=2, label="all %d sessions" % psy.sid.nunique())
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.axvline(0.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("DV   (negative = right is correct,  positive = left)")
    ax.set_ylabel("P(chose left)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("A   The rat is doing the task")
    ax.text(0.02, 0.96, "%d trials, grey = one session"
            % len(psy), transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def panel_b(ax, inv):
    edges = np.array([0, 0.2, 0.4, 0.6, 1.0001])
    for lab, col, sel in (("chose right (reward withheld)", C_CORRECT, inv.ChoiceCorrect > 0.5),
                          ("chose wrong", C_ERROR, inv.ChoiceCorrect < 0.5)):
        g = inv[sel]
        c, m, s, n = binned(g.absDV.values, g.lwz.values, edges)
        ax.errorbar(c, m, yerr=s, fmt="o-", color=col, lw=2.2, ms=6, capsize=2, label=lab)
        for x, y, k in zip(c, m, n):
            if np.isfinite(y):
                ax.annotate("%d" % k, (x, y), textcoords="offset points",
                            xytext=(0, -13 if col == C_ERROR else 9),
                            ha="center", fontsize=7, color=col)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("evidence strength  |DV|")
    ax.set_ylabel("how long it waits\n(z of log seconds)")
    ax.margins(y=0.16)
    ax.set_title("B   What the rat itself calls confidence")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.04, "same evidence, different wait\n$\\Rightarrow$ |DV| alone is not confidence",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)


def panel_c(ax, t):
    """Correct trials in four bins; errors in two, because there are only a handful
    of errors at strong evidence (3 trials in one of the fine bins) and plotting a
    mean of three as if it were a measurement would be the dishonest part of this
    figure, not the interesting part."""
    for lab, col, sel, edges in (
            ("correct", C_CORRECT, t.correct.values > 0.5, np.array([0, .2, .4, .6, 1.0001])),
            ("error", C_ERROR, t.correct.values < 0.5, np.array([0, .3, 1.0001]))):
        g = t[sel]
        c, m, s, n = binned(g.absDV.values, g.rate.values, edges)
        ax.errorbar(c, m, yerr=s, fmt="o-", color=col, lw=2.2, ms=6, capsize=2, label=lab)
        for x, y, k in zip(c, m, n):
            if np.isfinite(y):
                ax.annotate("n=%d" % k, (x, y), textcoords="offset points",
                            xytext=(0, -15 if col == C_ERROR else 10),
                            ha="center", fontsize=7, color=col)
    ax.set_xlabel("evidence strength  |DV|")
    ax.set_ylabel("firing rate (Hz)\n0-200 ms after cue end")
    ax.set_title("C   One neuron that does it")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-0.08, 1.0)
    ax.margins(y=0.14)
    ax.text(0.98, 0.05, "20260917 unit 287\none of the units with a significant slope",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)


def panel_d(ax, units):
    eps = ["base", "stim", "cue", "chin"]
    w = 0.38
    xs = np.arange(len(eps))
    allv = []
    for off, key, col, lab in ((-w / 2, "absDV", C_DV, "evidence strength |DV|"),
                               (w / 2, "conf", C_DVC, "P(correct | DV, choice)")):
        vals = []
        for ep in eps:
            # FDR is corrected WITHIN a session, as the check does it, then the
            # sessions are pooled -- not one correction over the pile.
            n_sig = n_tot = 0
            for sid, g in units[units.epoch == ep].groupby("session"):
                n_sig += int(bh_fdr(g[key + "_p"].values).sum())
                n_tot += len(g)
            vals.append(100.0 * n_sig / max(n_tot, 1))
        allv += vals
        b = ax.bar(xs + off, vals, w, color=col, label=lab, edgecolor="white", linewidth=1.2)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.12, "%.1f%%" % v,
                    ha="center", fontsize=7.5, color=col)
    ax.set_xticks(xs)
    ax.set_xticklabels([EPOCH_LABEL[e] for e in eps], fontsize=8)
    ax.set_ylabel("% of units with a\nsignificant slope")
    ax.set_title("D   Single neurons that carry it")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    top = max(allv + [3.0])
    ax.set_ylim(0, top * 1.45)
    ax.annotate("control: should be ~0", xy=(0, 0.4), xytext=(0.35, top * 0.45),
                fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))


def panel_e(ax, dec):
    eps = ["base", "stim", "cue", "chin"]
    targets = [("side", C_SIDE, "which side it will choose"),
               ("absDV", C_DV, "evidence strength |DV|"),
               ("absDV_correct_only", C_DVC, "|DV|, correct trials only")]
    xs = np.arange(len(eps))
    w = 0.26
    for i, (tg, col, lab) in enumerate(targets):
        off = (i - 1) * w
        med, pts = [], []
        for ep in eps:
            v = dec[(dec.epoch == ep) & (dec.target == tg)].r2.values
            med.append(np.median(v) if len(v) else np.nan)
            pts.append(v)
        ax.bar(xs + off, med, w, color=col, label=lab, edgecolor="white", linewidth=1.2)
        for x, v in zip(xs + off, pts):
            if len(v):
                ax.plot(np.full(len(v), x) + np.linspace(-0.05, 0.05, len(v)), v,
                        ".", color="white", ms=3.5, zorder=4, mec=INK, mew=0.4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([EPOCH_LABEL[e] for e in eps], fontsize=8)
    ax.set_ylabel("cross-validated $R^2$\nof the population read-out")
    ax.set_title("E   The population: side is huge, confidence is small but real")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.99, 0.99, "bar = median of %d sessions\ndot = one session"
            % dec.session.nunique(),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)


def panel_f(ax, dec):
    eps = ["base", "stim", "cue", "chin"]
    xs = np.arange(len(eps))
    w = 0.38
    for off, tg, col, lab in ((-w / 2, "correct_vs_error", C_CORRECT, "all spikes"),
                              (w / 2, "correct_vs_error_minus_base", "#9a7fb8",
                               "minus each cell's own baseline")):
        med, pts = [], []
        for ep in eps:
            v = dec[(dec.epoch == ep) & (dec.target == tg)].r2.values
            med.append(np.median(v) if len(v) else np.nan)
            pts.append(v)
        ax.bar(xs + off, med, w, color=col, label=lab, edgecolor="white", linewidth=1.2)
        for x, v in zip(xs + off, pts):
            if len(v):
                ax.plot(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v,
                        ".", color="white", ms=4, zorder=4, mec=INK, mew=0.4)
    ax.axhline(0.5, color=INK, lw=1.0, ls="--")
    ax.text(3.45, 0.505, "chance", fontsize=8, color=INK, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([EPOCH_LABEL[e] for e in eps], fontsize=8)
    ax.set_ylabel("AUC: is this choice\ngoing to be correct?")
    ax.set_ylim(0.35, 0.95)
    ax.set_title("F   The answer: it knows before the feedback does")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main():
    global OUT, USE, SHAMIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT, help="where the check wrote its CSVs")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--name", default="confidence_figure.png")
    ap.add_argument("--unit", type=int, default=287)
    ap.add_argument("--unit_session", default="20260917_100043")
    args = ap.parse_args()
    OUT = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    if args.sessions:
        USE = sorted(args.sessions)
    SHAMIR = [s for s in USE if s not in NO_ERROR_CATCH]

    psy, inv = behaviour()
    units = pd.concat([pd.read_csv(os.path.join(OUT, "%s_units.csv" % s))
                       for s in USE
                       if os.path.exists(os.path.join(OUT, "%s_units.csv" % s))])
    dec = pd.read_csv(os.path.join(OUT, "decoding.csv"))
    ex = example_unit(sid=args.unit_session, uid=args.unit)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    panel_a(axes[0, 0], psy)
    panel_b(axes[0, 1], inv)
    panel_c(axes[0, 2], ex)
    panel_d(axes[1, 0], units)
    panel_e(axes[1, 1], dec)
    panel_f(axes[1, 2], dec)

    n_tr = int(dec[dec.epoch == "chin"].groupby("session").n_trials.max().sum())
    n_un = int(units[units.epoch == "chin"].groupby("session").size().sum())
    fig.suptitle("Does the response after the cue carry decision confidence?   "
                 "%d synced sessions, %d completed trials, %d good units"
                 % (dec.session.nunique(), n_tr, n_un),
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    p = os.path.join(OUT, args.name)
    fig.savefig(p, dpi=170)
    print("wrote", p)


if __name__ == "__main__":
    main()
