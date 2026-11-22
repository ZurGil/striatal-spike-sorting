# -*- coding: utf-8 -*-
"""Is pi the same at every leave? -- which turns kappa from a guess into a result.

    python scripts/pi_at_leaving.py

Gil, 2026-11-22: "assuming we know the reward rate (and this is a big if, because
im not sure how to estimate it), and assuing the rat nows the reward timing
distribution ..., we shouold be able to calcualte the confidence for every trial,
because at the leaving time, the hazard (which is function of reard time and
condifence), is equal to the oppprtunity cost."

Right, and the big if is avoidable. His own deck (slide 7) defines

    pi = C S(T) / (1 - C + C S(T))      the posterior that the choice was correct,
                                        given no reward has arrived by T

and the leaving rule is pi(T) h(T) = kappa. The delay hazard is flat, h = lambda,
so the rule reads

    pi at the moment of leaving = kappa / lambda      -- the SAME on every trial.

So kappa does not have to be known in advance. Compute pi at each leave from a C
that owes nothing to the policy (conf_wait, the isotonic calibration curve of
5bd.93) and an S(T) read off the delays the program drew, and then:

  * if pi at leaving is constant, the framework holds and kappa = lambda * pi;
  * if it is not, the spread says by how much the rat departs from the policy,
    in the units of Gil's own slide.

A note on what the inversion would and would not add. Solving pi(T) lambda = kappa
for C gives a confidence that is a deterministic function of T, exactly as the
calibration curve is. The two therefore carry the SAME information and differ only
in scale -- any monotone relabelling of the waiting time. The reason to prefer the
calibration curve is that it is calibrated by construction, whereas the inversion
is calibrated only if the rat follows the policy, which panel A measures.

Panel C asks the separate empirical question behind Gil's "big if": if kappa is
the recent reward rate, the rat should wait LESS right after a good run. Tested
against a leaky average of reward over the preceding trials, with |DV| controlled.

Writes outputs/confidence_coding/pi_at_leaving.png.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from confidence_coding_quick_check import SESSIONS, DEFAULT_SESSIONS
from confidence_from_waiting import (analyse, delay_law, wls_logistic, weighted_bins,
                                     INK, MUTED, C_CORR, C_ERR, C_PRED, SHADES, BOX)

TAUS = (20.0, 50.0, 120.0)      # s, the leaky-average timescales to try


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_coding"))
    args = ap.parse_args()

    rows, lams, rhos = [], [], []
    for sid in args.sessions:
        r = analyse(sid, SESSIONS[sid])
        tb = r["table"]
        # S(T) is the RAT'S survival belief, so it has to be the smooth exponential
        # it is assumed to have learned, not the empirical one. The program never
        # draws past about 8 s, so the counted survival is exactly 0 beyond that and
        # pi collapses to exactly 0 -- an artefact of our sample, not of the rat,
        # which cannot have learned a hard cutoff from a few hundred draws.
        a0s, lams_ = delay_law(tb.delay.values)
        S = np.exp(-lams_ * np.maximum(tb.wait.values - a0s, 0.0))
        C = tb.conf_wait.values
        # The deck's pi = C S/(1-C+C S) because its simulation has no catch trials:
        # there, correct always means reward eventually. Here 15% of correct trials
        # are deliberately unrewarded, so the quantity the LEAVING rule turns on --
        # "is reward still coming?" -- is driven by a = C (1 - p_c), not C. Both are
        # carried: pi_deck is the slide's posterior-correct, pi is the policy's.
        a = C * (1.0 - r["p_c"])
        tb = tb.assign(S=S, lam=r["lam"], p_c=r["p_c"],
                       pi_deck=C * S / np.maximum(1 - C + C * S, 1e-12),
                       pi=a * S / np.maximum(a * S + 1 - a, 1e-12))

        # local reward rate before each trial, from the Bpod clock
        t = pd.read_parquet(SESSIONS[sid] + "/trials.parquet")
        t = t[t.ChoiceCorrect.notna() & t.sync_valid]
        if "overlaps_recording_gap" in t.columns:
            t = t[~t.overlaps_recording_gap]
        tt = t.trial_start_bpod.values.astype(float)
        rw = (t.Rewarded.fillna(0).values > 0.5).astype(float)
        for tau in TAUS:
            loc = np.zeros(len(tt))
            acc = 0.0
            for i in range(1, len(tt)):
                acc = acc * np.exp(-(tt[i] - tt[i - 1]) / tau) + rw[i - 1]
                loc[i] = acc / tau                       # rewards per second
            tb["rate_%d" % int(tau)] = loc
        rows.append(tb)
        lams.append(r["lam"]); rhos.append(r["rho"])

    p = pd.concat(rows, ignore_index=True)
    q = p[(p.usable > 0.5) & np.isfinite(p.pi)].copy()
    lam = float(np.mean(lams)); rho = float(np.mean(rhos))
    q["kappa_imp"] = q.lam * q.pi
    p.to_csv(os.path.join(args.out, "pi_at_leaving.csv"), index=False)

    print("usable leaves: %d" % len(q))
    print("\npi at the moment of leaving -- the policy says this is CONSTANT:")
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)):
        m = ((q.wait >= np.percentile(q.wait, lo)) & (q.wait <= np.percentile(q.wait, hi)))
        print("   wait %5.1f-%5.1f s   pi = %.4f   -> kappa = %.4f /s   n=%d"
              % (np.percentile(q.wait, lo), np.percentile(q.wait, hi),
                 q.pi[m].median(), q.kappa_imp[m].median(), int(m.sum())))
    print("\n   overall median pi = %.4f  -> kappa = %.4f /s" %
          (q.pi.median(), q.kappa_imp.median()))
    print("   measured background reward rate rho/R = %.4f /s" % rho)
    print("   spread of the implied kappa, p10-p90: %.4f - %.4f  (x%.0f)"
          % (q.kappa_imp.quantile(.10), q.kappa_imp.quantile(.90),
             q.kappa_imp.quantile(.90) / max(q.kappa_imp.quantile(.10), 1e-9)))

    # Time in session is the confound that would fake this: late on, the rat is
    # sated, so it both earns less AND waits less, which manufactures a positive
    # slope out of nothing. So trial index goes in, and session identity with it.
    q["frac"] = q.groupby("sid").cumcount() / q.groupby("sid").trial_id.transform("size")
    dums = pd.get_dummies(q.sid, drop_first=True).values.astype(float)
    print("\nis kappa the RECENT reward rate? wait ~ local rate, with |DV|,")
    print("time-in-session and session identity controlled:")
    res_tau = {}
    yv = np.log(np.maximum(q.wait.values, 0.05))
    for tau in TAUS:
        col = "rate_%d" % int(tau)
        for lab, extra in (("|DV| only", np.zeros((len(q), 0))),
                           ("+ time + session", np.column_stack([q.frac.values, dums]))):
            X = np.column_stack([np.ones(len(q)),
                                 (q.absDV - q.absDV.mean()) / q.absDV.std(),
                                 (q[col] - q[col].mean()) / q[col].std(), extra])
            b = np.linalg.lstsq(X, yv, rcond=None)[0]
            resid = yv - X.dot(b)
            se = np.sqrt(resid.dot(resid) / (len(yv) - X.shape[1]) *
                         np.diag(np.linalg.pinv(X.T.dot(X))))
            if lab.startswith("+"):
                res_tau[tau] = (b[2], b[2] / se[2])
            print("   tau %5.0f s  %-18s beta = %+.4f   t = %+.1f"
                  % (tau, lab, b[2], b[2] / se[2]))

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.2))

    ax = axes[0]
    ax.plot(q.wait, q.pi, ".", color=SHADES[1], ms=4, alpha=0.45)
    ed = np.percentile(q.wait, np.linspace(0, 100, 9))
    med = [(np.median(q.wait[(q.wait >= ed[i]) & (q.wait <= ed[i + 1])]),
            np.median(q.pi[(q.wait >= ed[i]) & (q.wait <= ed[i + 1])]))
           for i in range(len(ed) - 1)]
    med = np.array(med)
    ax.plot(med[:, 0], med[:, 1], "o-", color=INK, lw=2.2, ms=6, zorder=4,
            label="median per bin")
    ax.axhline(q.pi.median(), color=C_PRED, lw=2.0, ls="--", zorder=3,
               label="what the policy predicts: one constant")
    ax.axhline(rho / lam, color=C_ERR, lw=1.6, ls=":", zorder=3,
               label="$\\rho/(R\\lambda)$ from the measured reward rate")
    ax.set_yscale("log")
    ax.set_xlabel("waiting time at the leave  T  (s)")
    ax.set_ylabel("$\\pi$ at the moment of leaving")
    ax.set_title("A   $\\pi$ at the leave is NOT constant")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0.97, 0.95,
            "median $\\pi$ = %.3f  ->  $\\kappa$ = %.3f /s\nmeasured $\\rho/R$ = %.3f /s\n\n"
            "but $\\pi$ falls %.0fx across the range\ninstead of holding still, which is\n"
            "the shallow slope of 5bd.93 said\nin the units of your slide."
            % (q.pi.median(), q.kappa_imp.median(), rho,
               med[0, 1] / max(med[-1, 1], 1e-9)),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color=INK,
            bbox=BOX)

    ax = axes[1]
    ax.hist(np.log10(np.maximum(q.kappa_imp, 1e-5)), bins=40, color=SHADES[1],
            edgecolor="white", lw=0.5)
    ax.axvline(np.log10(rho), color=C_ERR, lw=2.0,
               label="measured $\\rho/R$ = %.3f /s" % rho)
    ax.axvline(np.log10(q.kappa_imp.median()), color=C_PRED, lw=2.0, ls="--",
               label="median implied $\\kappa$ = %.3f /s" % q.kappa_imp.median())
    ax.set_xlabel("$\\log_{10}$  implied $\\kappa = \\lambda\\,\\pi(T)$   (rewards / s)")
    ax.set_ylabel("trials")
    ax.set_title("B   The threshold the rat would need, per trial")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.text(0.97, 0.60,
            "the two agree in the MIDDLE --\nwhich is why the single-number\n"
            "recovery in 5bd.93 worked -- but\nthe per-trial spread is p10-p90\n"
            "%.4f to %.4f, a factor of %.0f.\nNo plausible reward rate moves\n"
            "that much inside a session."
            % (q.kappa_imp.quantile(.10), q.kappa_imp.quantile(.90),
               q.kappa_imp.quantile(.90) / max(q.kappa_imp.quantile(.10), 1e-9)),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color=INK,
            bbox=BOX)

    ax = axes[2]
    for tau, col in zip(TAUS, SHADES):
        c = "rate_%d" % int(tau)
        ed = np.percentile(q[c], np.linspace(0, 100, 7))
        xs, ys, es = [], [], []
        for i in range(len(ed) - 1):
            m = (q[c] >= ed[i]) & (q[c] <= ed[i + 1])
            if m.sum() < 20:
                continue
            lw_ = np.log(np.maximum(q.wait[m], 0.05))
            xs.append(q[c][m].median()); ys.append(lw_.mean())
            es.append(lw_.std() / np.sqrt(m.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, ms=6, lw=2.0, capsize=2,
                    label="$\\tau$ = %.0f s   t = %+.1f" % (tau, res_tau[tau][1]))
    ax.set_xlabel("local reward rate before the trial  (rewards / s)")
    ax.set_ylabel("mean  log(waiting time)")
    ax.set_title("C   Does a good run make the rat leave sooner?")
    ax.legend(frameon=False, fontsize=8.2, loc="best")
    ax.text(0.03, 0.06,
            "if $\\kappa$ really is the recent reward\nrate, these should slope DOWN.\n"
            "|DV| is in the regression behind\nthe reported t values.",
            transform=ax.transAxes, va="bottom", fontsize=8.3, color=INK, bbox=BOX)

    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("$\\pi$ at the leave, and whether $\\kappa$ has to be guessed   --   "
                 "Shamir_1, %d usable leaves" % len(q), fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "pi_at_leaving.png")
    fig.savefig(path, dpi=150)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
