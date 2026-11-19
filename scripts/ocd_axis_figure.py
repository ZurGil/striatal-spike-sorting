# -*- coding: utf-8 -*-
"""The dorsal-to-ventral axis of the implant, drawn against the OCD-circuit claim.

    python scripts/ocd_axis_figure.py

Panels
  A  where the recorded units actually are: good-unit depth histogram over the
     probe's span, with the dorsal / ventral split used everywhere else
  B  fraction of good units coding |DV| and P(correct), in 5 depth bins, with the
     shuffle-null expectation drawn as the chance line
  C  error decoding at the cue and at the choice port, dorsal vs ventral, with the
     two populations subsampled to the SAME number of units
  D  the behavioural adjustment after an error -- the measure an OCD model has to
     move, and the one we can read today without touching the rig

Reads outputs/confidence_depth/ (written by scripts/confidence_depth_axis.py) and
the three sessions' trials.parquet. Writes outputs/confidence_depth/ocd_axis.png.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import confidence_coding_quick_check as C  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "confidence_depth")
SESSIONS = ["20260916_110311", "20260917_100043", "20260918_100248"]
SPLIT_MM = 4.70
BINS = np.array([2.70, 3.45, 4.20, 4.95, 5.70, 6.45])

INK = "#2b2b31"
DORS = "#1f6f8b"
VENT = "#c1553b"


def frac_sig_by_bin(U, epoch, key):
    """Per-session FDR, then pooled into depth bins. Returns frac, n per bin."""
    sub = U[U.epoch == epoch].copy()
    sig = np.zeros(len(sub), dtype=bool)
    for sid, g in sub.groupby("session"):
        s = C.bh_fdr(g[key + "_p"].values)
        sig[sub.index.get_indexer(g.index[s])] = True
    sub["sig"] = sig
    b = np.digitize(sub.depth_mm.values, BINS) - 1
    fr, nn = [], []
    for i in range(len(BINS) - 1):
        m = b == i
        nn.append(int(m.sum()))
        fr.append(100.0 * sub.sig.values[m].sum() / max(m.sum(), 1))
    return np.array(fr), np.array(nn)


def post_error(sid):
    S = C.SESSIONS[sid]
    t = C.build_trials(S, sid)
    t = t.sort_values("trial_id").reset_index(drop=True)
    prev_err = np.concatenate([[np.nan], 1.0 - t.correct.values[:-1]])
    out = {}
    # ST is a fixed 0.35 s in this protocol, so it carries no adjustment and is
    # left out -- which is itself the argument for freeing it (see the log).
    for name, v in (("MT", t.MT.values),):
        a = v[prev_err == 0]
        b = v[prev_err == 1]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        out[name] = (np.median(a), np.median(b), len(a), len(b))
    acc = t.correct.values.astype(float)
    out["accuracy"] = (np.nanmean(acc[prev_err == 0]), np.nanmean(acc[prev_err == 1]),
                       int((prev_err == 0).sum()), int((prev_err == 1).sum()))
    # the rat's own time investment on the trial AFTER an error
    w = t.wait.values.copy()
    w[~t.time_investment.values] = np.nan
    a = w[(prev_err == 0) & np.isfinite(w)]
    b = w[(prev_err == 1) & np.isfinite(w)]
    out["wait"] = (np.median(a), np.median(b), len(a), len(b))
    return out


def main():
    U = pd.read_csv(os.path.join(OUT, "units_with_depth.csv"))
    D = pd.read_csv(os.path.join(OUT, "decode_by_depth.csv"))

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1.35, 1.1],
                          hspace=0.42, wspace=0.34,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)

    # ---------------------------------------------------------------- A  geometry
    ax = fig.add_subplot(gs[:, 0])
    dd = U.drop_duplicates(["session", "unit_id"]).depth_mm.values
    ax.hist(dd, bins=np.arange(2.7, 6.65, 0.15), orientation="horizontal",
            color="#b9bec7", edgecolor="white", linewidth=0.6)
    ax.axhline(SPLIT_MM, color=INK, lw=1.2, ls="--")
    ax.set_ylim(6.7, 2.5)
    ax.set_ylabel("depth below brain surface (mm)")
    ax.set_xlabel("good units (3 sessions)")
    # The y axis is inverted, so the TOP of this panel is the shallow, dorsal end.
    ax.text(0.97, 0.97, "dorsal striatum\n(sensorimotor / habit limb)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=DORS)
    ax.text(0.97, 0.03, "ventral striatum, NAc border\n(OFC / ACC limb)"
                        "\nDiI track still to confirm",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=VENT)
    ax.set_title("A  one shank, the whole axis\nML -2.45, AP +1.2, %d units"
                 % U.drop_duplicates(["session", "unit_id"]).shape[0],
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------------------------------------------------------- B  profiles
    centres = 0.5 * (BINS[:-1] + BINS[1:])
    for row, (ep, label) in enumerate((("cue", "0-200 ms after the cue ends"),
                                       ("chin", "0-450 ms after the choice poke"))):
        ax = fig.add_subplot(gs[row, 1])
        for key, col, nm in (("absDV", "#3c6e47", "|DV|  (stimulus strength)"),
                             ("conf", "#7b4397", "P(correct)  (confidence)")):
            fr, nn = frac_sig_by_bin(U, ep, key)
            ax.plot(fr, centres, "-o", color=col, lw=1.8, ms=5, label=nm)
        ax.axhline(SPLIT_MM, color=INK, lw=1.0, ls="--", alpha=0.5)
        ax.axvline(5.0, color="#999999", lw=1.0, ls=":")
        ax.text(5.2, 2.75, "chance (shuffle)", fontsize=8, color="#777777")
        ax.set_ylim(6.7, 2.5)
        ax.set_xlim(0, 26)
        ax.set_xlabel("% of good units significant (FDR 0.05)")
        ax.set_ylabel("depth (mm)")
        ax.set_title("%s  %s\n%s" % ("B" if row == 0 else "C", ep, label),
                     fontsize=10, loc="left")
        if row == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------------------------------------------------------- C  decoding
    ax = fig.add_subplot(gs[0, 2])
    x = {"cue": 0, "chin": 1}
    for sid, mk in zip(SESSIONS, ("o", "s", "^")):
        for ep in ("cue", "chin"):
            g = D[(D.session == sid) & (D.epoch == ep)]
            if len(g) != 2:
                continue
            a = float(g[g.half == "dorsal"].auc_sub.iloc[0])
            b = float(g[g.half == "ventral"].auc_sub.iloc[0])
            ax.plot([x[ep] - 0.14, x[ep] + 0.14], [a, b], "-", color="#bbbbbb", lw=1.0,
                    zorder=1)
            ax.plot(x[ep] - 0.14, a, mk, color=DORS, ms=6, zorder=2)
            ax.plot(x[ep] + 0.14, b, mk, color=VENT, ms=6, zorder=2)
    ax.axhline(0.5, color="#999999", lw=1.0, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cue end", "choice port"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("AUC, decoding THIS trial is an error")
    ax.set_title("D  error read-out, dorsal vs ventral\nequal unit count in both",
                 fontsize=10, loc="left")
    ax.text(0.02, 0.97, "blue dorsal / red ventral\ncircle 16, square 17, triangle 18 Sep",
            transform=ax.transAxes, fontsize=8, va="top", color="#555555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------------------------------------------------------- D  behaviour
    ax = fig.add_subplot(gs[1, 2])
    pe = {sid: post_error(sid) for sid in SESSIONS}
    lbl = [("MT", "movement time"), ("wait", "time investment"), ("accuracy", "accuracy")]
    for i, (k, nm) in enumerate(lbl):
        for sid, mk in zip(SESSIONS, ("o", "s", "^")):
            a, b, na, nb = pe[sid][k]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            ax.plot([i - 0.14, i + 0.14], [0.0, 100.0 * (b - a) / a], "-",
                    color="#bbbbbb", lw=1.0, zorder=1)
            ax.plot(i + 0.14, 100.0 * (b - a) / a, mk, color=INK, ms=6, zorder=2)
    ax.axhline(0.0, color="#999999", lw=1.0, ls=":")
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels([n for _, n in lbl], fontsize=9)
    ax.set_ylabel("% change on the trial AFTER an error")
    ax.set_title("E  what an OCD model has to move\nbaseline = after a correct trial",
                 fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("The implant spans the dorsal-to-ventral striatal axis "
                 "-- what it already says, and where the OCD question lives",
                 fontsize=13, y=0.955)
    p = os.path.join(OUT, "ocd_axis.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)
    for sid in SESSIONS:
        print(sid, {k: tuple(round(float(x), 3) for x in v[:2]) for k, v in pe[sid].items()})


if __name__ == "__main__":
    main()
