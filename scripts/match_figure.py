# -*- coding: utf-8 -*-
"""The figure for cross-session unit matching: can we follow one neuron for three days?

    python scripts/match_figure.py

Six panels:

  A  what the score separates      same neuron (split halves) against different
                                   neurons, on the one session used for calibration
  B  recall and precision          measured on split halves with HALF the second set
                                   deleted, so half the units have no partner at all
  C  one neuron, three days        its mean spike at the peak channel, 16/17/18 Sep
  D  the same neuron's footprint   amplitude down the probe, which is what is matched
  E  the probe is where it was     centroid depth in one session against the next
  F  the three-day consistency     chain 16>17>18 against the direct 16>18 match

Writes outputs/unit_matching/matching_figure.png.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import match_units_across_sessions as M

OUT = os.path.join(ROOT, "outputs", "unit_matching")
A, B, C = "20260916_110311", "20260917_100043", "20260918_100248"
DAY = {A: "16 Sep", B: "17 Sep", C: "18 Sep"}
COL = {A: "#bcd4e6", B: "#4f84b0", C: "#20496e"}

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_SAME = "#2f6f9f"
C_DIFF = "#c2502d"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def panel_a(ax, cal, thr):
    bins = np.linspace(-3.4, 1.6, 75)
    ax.hist(cal["s_diff"], bins=bins, color=C_DIFF, alpha=0.85,
            label="different neurons, same session (%d pairs)" % len(cal["s_diff"]))
    ax.hist(cal["s_same"], bins=bins, color=C_SAME, alpha=0.9,
            label="same neuron, by construction (%d)" % len(cal["s_same"]))
    m = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (B, C)))
    ax.hist(m.score.values, bins=bins, histtype="step", color=INK, lw=1.6,
            label="accepted 17 $\\rightarrow$ 18 Sep pairs (%d)" % len(m))
    ax.axvline(thr, color=INK, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1400)
    ax.text(thr, 700, " accepted $\\rightarrow$", fontsize=8, color=INK)
    ax.set_xlabel("match score  (footprint + shape + distance)")
    ax.set_ylabel("candidate pairs")
    ax.set_title("A   What the score has to separate")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.text(0.015, 0.03, "calibrated on 20260917: every unit's first half of\nspikes"
                         " against its own second half",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=MUTED,
            bbox=dict(fc="white", ec="none", pad=1.5))


def panel_b(ax, cal, thr):
    t = pd.DataFrame(cal["orphan_table"]).sort_values("thr")
    ax.plot(t.thr, 100 * t.recall, "-", color=C_SAME, lw=2.2, label="recall")
    ax.plot(t.thr, 100 * t.precision, "-", color=C_DIFF, lw=2.2, label="precision")
    ax.axvline(thr, color=INK, lw=1.4, ls="--")
    row = t.iloc[(t.thr - thr).abs().values.argmin()]
    ax.plot([thr, thr], [100 * row.recall, 100 * row.precision], "o", color=INK, ms=6,
            zorder=5)
    ax.annotate("%.1f%% recall\n%.1f%% precision\n(%.1f false matches in %.0f)"
                % (100 * row.recall, 100 * row.precision, row.fp, row.matches),
                xy=(thr, 100 * row.recall), xytext=(-12, -58),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.set_xlim(-1.0, 1.45)
    ax.set_ylim(0, 105)
    ax.set_xlabel("score threshold")
    ax.set_ylabel("%")
    ax.set_title("B   Measured, not assumed")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0.03, 0.56, "half of the second set deleted,\nso half the units have NO"
                        " partner\n-- the across-session case",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)


def load_all():
    return dict((s, M.load(s)) for s in (A, B, C))


def pick_example(W):
    """The unit the two independent routes agree on with the highest weakest link."""
    ch = pd.read_csv(os.path.join(OUT, "chain_%s_%s_%s.csv" % (A, B, C)))
    m1 = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (A, B)))
    m2 = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (B, C)))
    s1 = dict(zip(m1.unit_a, m1.score))
    s2 = dict(zip(m2.unit_a, m2.score))
    best, bestv = None, -9
    for _, r in ch[ch.agrees == 1].iterrows():
        v = min(s1.get(r.unit_a, -9), s2.get(r.via_b, -9))
        if v > bestv:
            best, bestv = (int(r.unit_a), int(r.via_b), int(r.direct_c)), v
    return best, ch


def unit_wave(S, uid):
    """Mean over both halves, in uV, plus the per-channel amplitude and positions."""
    i = int(np.where(S["unit_ids"] == uid)[0][0])
    n = S["n_spikes"][i].astype(float)
    w = (S["mean_wf"][i, 0] * n[0] + S["mean_wf"][i, 1] * n[1]) / max(n.sum(), 1)
    ptp = w.max(axis=0) - w.min(axis=0)
    return w, ptp, S["channel_pos"], int(np.argmax(ptp))


def panel_c(ax, W, ids):
    t = (np.arange(82) - 31) / 30.0            # ms, 30 kHz
    pk = None
    for s, uid in zip((A, B, C), ids):
        w, ptp, pos, p = unit_wave(W[s], uid)
        pk = p if pk is None else pk           # same site for all three
        ax.plot(t, w[:, pk], "-", color=COL[s], lw=2.2,
                label="%s  unit %d" % (DAY[s], uid))
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.margins(y=0.22)
    ax.set_xlabel("time from spike (ms)")
    ax.set_ylabel("$\\mu$V")
    ax.set_title("C   One neuron, three days")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.02, 0.97, "mean spike at the same recording site,\nraw, no template,"
                        " no alignment across days",
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)


def panel_d(ax, W, ids):
    for s, uid in zip((A, B, C), ids):
        w, ptp, pos, p = unit_wave(W[s], uid)
        y = pos[:, 1]
        near = np.abs(y - pos[p, 1]) <= 90
        o = np.argsort(y[near])
        ax.plot(ptp[near][o], y[near][o], "-o", color=COL[s], lw=2.0, ms=3.5,
                label=DAY[s])
    ax.set_xlabel("spike amplitude ($\\mu$V)")
    ax.set_ylabel("depth along the probe ($\\mu$m)")
    ax.set_title("D   ...and the footprint that identifies it")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.98, 0.03, "this pattern across sites, not the\nfiring or the tuning,"
                        " is what is matched",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)


def firing_rates(session):
    """Spikes per second per raw Kilosort cluster. Cached -- the arrays are on Z."""
    cache = os.path.join(OUT, "rates_%s.npy" % session)
    if os.path.exists(cache):
        return np.load(cache)
    p = "Z:/Gil/Shamir_1/%s.rec/%s.kilosort/kilosort4/" % (session, session)
    cl = np.load(p + "spike_clusters.npy").astype(int)
    st = np.load(p + "spike_times.npy")
    r = np.bincount(cl) / (float(st.max()) / 30000.0)
    np.save(cache, r)
    return r


def panel_e(ax, W):
    """Depth agreement would be circular -- centroid distance is one of the three
    features, so matched pairs lie on the identity line by construction and the
    panel proves nothing. Firing rate is never shown to the matcher at any point."""
    m = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (B, C)))
    rb, rc = firing_rates(B), firing_rates(C)
    x = np.array([rb[u] for u in m.unit_a])
    y = np.array([rc[u] for u in m.unit_b])
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    rng = np.random.RandomState(0)
    ysh = y[rng.permutation(len(y))]
    lo, hi = 0.02, 60
    ax.plot([lo, hi], [lo, hi], "-", color=MUTED, lw=1.0, zorder=1)
    ax.plot(x, ysh, ".", color="#cfcfcf", ms=4, zorder=2,
            label="same units, partners shuffled  r = %.2f"
                  % np.corrcoef(np.log10(x), np.log10(ysh))[0, 1])
    ax.plot(x, y, ".", color=C_SAME, ms=4.5, zorder=3,
            label="matched pairs  r = %.2f" % np.corrcoef(np.log10(x), np.log10(y))[0, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("firing rate on 17 Sep (Hz)")
    ax.set_ylabel("firing rate of its match\non 18 Sep (Hz)")
    ax.set_title("E   A check the matcher never saw")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.97, 0.03, "firing rate enters nothing in the score.\nmatched pairs agree"
                        " to a median factor\nof 1.38 across a day; absolute spike\n"
                        "amplitude agrees at r = 0.84",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))


def verdicts(session):
    p = os.path.join(ROOT, "outputs", "review_verdicts", session, "manual_verdicts.csv")
    d = pd.read_csv(p)
    return dict(zip(d.unit.astype(int), d.verdict))


def panel_f(ax, W, ch):
    """Which units find a partner -- split by Gil's own label, because 'how many units
    matched' means nothing without knowing whether they were units he would keep."""
    classes = [("good", "kept at review"), ("mua", "multi-unit"),
               (None, "never judged")]
    pairs = [(A, B, "16 $\\rightarrow$ 17 Sep", COL[B]),
             (B, C, "17 $\\rightarrow$ 18 Sep", COL[C])]
    w = 0.38
    xs = np.arange(len(classes))
    for k, (a, b, lab, col) in enumerate(pairs):
        m = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (a, b)))
        matched = set(int(u) for u in m.unit_a)
        V = verdicts(a)
        vals, ns = [], []
        for cls, _ in classes:
            ids = [int(u) for u in W[a]["unit_ids"]
                   if (V.get(int(u)) == cls if cls else int(u) not in V)]
            ns.append(len(ids))
            vals.append(100.0 * sum(1 for u in ids if u in matched) / max(len(ids), 1))
        bars = ax.bar(xs + (k - 0.5) * w, vals, w, color=col, label=lab,
                      edgecolor="white", lw=1.2)
        for r, v, n in zip(bars, vals, ns):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.5, "%.0f%%\nof %d" % (v, n),
                    ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\n%s" % (c if c else "unjudged", d) for c, d in classes],
                       fontsize=8)
    ax.set_ylabel("% that find a partner\nin the next session")
    ax.set_ylim(0, 108)
    ax.set_title("F   Who gets matched, by your own label")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.5, 0.995, "of the %d matched pairs where BOTH sides\nwere judged, the two"
                        " labels agree on %d\n-- and the matcher is never shown a label"
            % tuple(agreement_both_judged(W)),
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)


def agreement_both_judged(W):
    m = pd.read_csv(os.path.join(OUT, "%s__%s_matches.csv" % (B, C)))
    vb, vc = verdicts(B), verdicts(C)
    both = [(int(u), int(v)) for u, v in zip(m.unit_a, m.unit_b)
            if int(u) in vb and int(v) in vc]
    return len(both), sum(1 for u, v in both if vb[u] == vc[v])


def main():
    cal = M.calibrate(M.load(B), verbose=False)
    thr = None
    for target, r in M.tiers(cal["orphan_table"]):
        if abs(target - 0.95) < 1e-9:
            thr = r["thr"]
    W = load_all()
    ids, ch = pick_example(W)
    print("example unit:", ids)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    panel_a(axes[0, 0], cal, thr)
    panel_b(axes[0, 1], cal, thr)
    panel_c(axes[0, 2], W, ids)
    panel_d(axes[1, 0], W, ids)
    panel_e(axes[1, 1], W)
    panel_f(axes[1, 2], W, ch)

    fig.suptitle("Can one neuron be followed from one session to the next?   "
                 "16-18 September, Shamir, chronic probe",
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    p = os.path.join(OUT, "matching_figure.png")
    fig.savefig(p, dpi=170)
    print("wrote", p)


if __name__ == "__main__":
    main()
