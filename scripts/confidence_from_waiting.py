# -*- coding: utf-8 -*-
"""Confidence read out of the waiting time, the way Lak/Kepecs define it.

    python scripts/confidence_from_waiting.py
    python scripts/confidence_from_waiting.py --sessions 20260917_100043

Gil, 2026-11-22: "as measurment for confidence you used the psychoemetric. the
question we have to ask ourself is wether this is the correct why to do it. in
lak and kepecs 2014, the confidence is the value of the haazrd function right at
the very first moment the rat start the wating time. is their a way to estimate
this value? myabe from the anout of time the rat wait per trial. of course this
is much more comlicated because we: A assume the rat learned the time
ditribution (which is reasonable). B. we dont know the thredhould value per rat,
but myabe there is still way of doing it?"

Yes, and the two unknowns separate -- A sets a slope we can predict, B sets a
shift we can recover.

THE POLICY
----------
Waiting at the side port, nothing has arrived at time t. Reward comes only if the
choice was correct AND the trial is not a catch trial, at a delay D the program
drew. With c = P(correct | this choice) and p_c = P(catch):

    a     = c (1 - p_c)                      P(reward will ever come)
    pi(t) = a S(t) / [a S(t) + 1 - a]        P(still coming | nothing yet at t)
    rate of expected reward = pi(t) h(t) R

Leave the moment that rate falls below the opportunity cost rho of standing here
rather than starting the next trial. With kappa = rho/R, at the leaving time T:

    a S(T) h(T) / (a S(T) + 1 - a) = kappa
    a f(T)                         = kappa [a S(T) + 1 - a]
    a                              = kappa / (kappa + f(T) - kappa S(T))

Confidence is not the hazard itself -- it is the SCALE on it. The rat leaves when
the hazard, scaled by confidence and eroded by the growing belief that the trial
was an error, drops through kappa.

MEASURED, NOT ASSUMED: the programmed delay is a shifted exponential, floor
0.5 s, hazard FLAT at lambda = 0.69 /s in all eight sessions (panel A). So
S = exp(-lambda (t - a0)), f = lambda S, and the policy collapses to a logistic
in the waiting time:

    logit(a) = lambda (T - T0),     T0 = a0 - log(kappa/(lambda-kappa))/lambda

  A  "the rat learned the distribution" fixes the SLOPE of logit(a) against T at
     lambda = 0.69 /s -- read off the delays the program drew, nothing fitted. So
     it is a prediction, and panel C tests it.
  B  "we don't know the threshold" moves ONLY T0, i.e. slides the line sideways.
     One scalar, recovered by anchoring the line to the measured calibration
     curve at its own midpoint, then inverted back to kappa and compared against
     the session's measured reward rate rho/R -- which is not fitted either.

WHAT IS ACTUALLY ESTIMATED, AND HOW
-----------------------------------
conf_wait is NOT the normative inversion. It is the calibration curve itself:

    conf_wait(trial) = P(correct | the rat waited this long)

estimated by weighted isotonic regression (PAVA) on the waiting time. That needs
no threshold, no hazard, no psychometric and no functional form -- only that
confidence is monotone in waiting time -- and it is calibrated by construction:
trials assigned 0.8 really are correct 80% of the time. The normative model is
then kept for the one thing it is good for, which is being WRONG in a measurable
way (panel C).

WHICH TRIALS SHOW A LEAVE -- the part that is easy to get wrong
--------------------------------------------------------------
On a rewarded trial the rat stops because water came, so its leaving time is
unobserved. Three classes are unrewarded and show a leave:

  correct + CATCH     usable. Catch is drawn independently of the wait, so these
                      are an unbiased p_c-sample of all correct trials, whatever
                      the wait. Weight 1/p_c.
  ERROR               usable, never censored at all. Weight 1.
  correct, not catch, not rewarded   NOT usable, and this was checked rather than
                      assumed. The rat was out of the port past the 0.35 s grace
                      and the protocol fired skipped_feedback: median wait 0.9 s
                      against 6.7 s on catch, a separate mode. Matched on the
                      drawn delay -- restricted to trials where a leave under
                      2.5 s was observable at all -- the short-leave rate is 5.0%
                      on correct trials and 5.3% on errors. Accuracy-independent,
                      so it is a grace abort carrying no confidence. And it is
                      only VISIBLE when the drawn delay happened to be long, so
                      keeping it makes the wait depend on the delay and breaks the
                      censoring model: with them in, the calibration curve INVERTS
                      and the shortest waits come out the most accurate (97%).

Dropping that class makes the weighting a constant 1/p_c on correct trials and
needs no survival model at all.

Writes outputs/confidence_coding/confidence_from_waiting.png and
outputs/confidence_coding/conf_wait_<session>.csv (per-trial conf_wait).
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
from confidence_coding_quick_check import (SESSIONS, DEFAULT_SESSIONS,
                                           fit_psychometric, normative_confidence)

INK = "#1c1c1c"
MUTED = "#8a8a8a"
C_CORR = "#2f6f9f"
C_ERR = "#c2502d"
C_ISO = "#1c1c1c"
C_PRED = "#4a9b7f"
SHADES = ["#c9d9e8", "#7fa6c4", "#2f6f9f"]

# s. Below the shortest plausible time investment. The calibration curve is flat
# in this choice anywhere from 0 to 2.5 s, because the abort mode is almost
# entirely confined to the non-catch correct trials that are dropped anyway.
ABORT_FLOOR = 1.0
CONF_LO, CONF_HI = 0.50, 0.99
# every in-panel note sits on top of data somewhere; give them all a quiet ground
BOX = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2.0)
N_BOOT = 2000


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ the delay draws

def delay_law(fd):
    """Programmed floor and the ML hazard of the shifted exponential."""
    fd = np.asarray(fd, float)
    fd = fd[np.isfinite(fd)]
    a0 = float(np.percentile(fd, 1))
    sh = fd - a0
    return a0, float(1.0 / sh[sh > 0].mean())


# ------------------------------------------------------- the calibration estimator

def isotonic(x, y, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators.

    The non-parametric ML estimate of a monotone P(correct | x). Returns the sorted
    x and the fitted level at each, which is all the interpolation below needs.
    """
    o = np.argsort(x, kind="mergesort")
    xs, ys, ws = np.asarray(x)[o], np.asarray(y, float)[o], np.asarray(w, float)[o]
    # blocks of (weight, weighted mean)
    bw, bv, bn = [], [], []
    for i in range(len(xs)):
        bw.append(ws[i]); bv.append(ys[i]); bn.append(1)
        while len(bv) > 1 and bv[-2] > bv[-1] + 1e-15:
            w2, v2, n2 = bw.pop(), bv.pop(), bn.pop()
            w1, v1, n1 = bw.pop(), bv.pop(), bn.pop()
            tw = w1 + w2
            bw.append(tw); bv.append((w1 * v1 + w2 * v2) / tw); bn.append(n1 + n2)
    fit = np.concatenate([np.full(n, v) for v, n in zip(bv, bn)])
    return xs, fit


def iso_predict(xs, fit, q):
    """Step-interpolate the isotonic fit, clamped at both ends."""
    return np.interp(np.asarray(q, float), xs, fit, left=fit[0], right=fit[-1])


def weighted_bins(T, y, wt, edges):
    """IPW fraction correct per bin, with a Kish effective-n error bar."""
    idx = np.clip(np.digitize(T, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 8:
            continue
        ww, yy = wt[m], y[m]
        p = float((ww * yy).sum() / ww.sum())
        n_eff = float(ww.sum() ** 2 / (ww ** 2).sum())
        rows.append((float(np.median(T[m])), p,
                     np.sqrt(max(p * (1 - p), 1e-6) / max(n_eff, 1.0)), int(m.sum())))
    return np.array(rows)


def local_slope(T, y, wt, p_c, lo=25.0, hi=75.0):
    """d logit(a) / dT across the central half of the waits.

    This is the quantity the policy predicts to equal lambda. A single robust
    number, taken where the data is dense: a global parametric fit is not usable
    here because the normative form must climb to c = 1 while the measured curve
    plateaus near 0.92, so the fit trades its slope away to reach an asymptote
    the rat does not have.
    """
    xs, fit = isotonic(T, y, wt)
    t1, t2 = np.percentile(T, lo), np.percentile(T, hi)
    if t2 - t1 < 0.3:
        return np.nan, np.nan, np.nan
    a1 = iso_predict(xs, fit, t1) * (1 - p_c)
    a2 = iso_predict(xs, fit, t2) * (1 - p_c)
    return float((logit(a2) - logit(a1)) / (t2 - t1)), float(t1), float(t2)


def boot_slope(T, y, wt, p_c, rng, n=N_BOOT):
    out = []
    n_t = len(T)
    for _ in range(n):
        i = rng.integers(0, n_t, n_t)
        s, _, _ = local_slope(T[i], y[i], wt[i], p_c)
        if np.isfinite(s):
            out.append(s)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def wls_logistic(X, y, wt, iters=80):
    """Weighted IRLS logistic regression. Returns beta and its standard errors."""
    X = np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X.dot(beta)
        p = sig(eta)
        W = wt * p * (1 - p)
        z = eta + (y - p) / np.maximum(p * (1 - p), 1e-9)
        A = X.T.dot(X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        try:
            new = np.linalg.solve(A, X.T.dot(W * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < 1e-11:
            beta = new
            break
        beta = new
    eta = X.dot(beta); p = sig(eta); W = wt * p * (1 - p)
    cov = np.linalg.pinv(X.T.dot(X * W[:, None]))
    return beta, np.sqrt(np.maximum(np.diag(cov), 1e-30))


# ----------------------------------------------------------------------- per session

def analyse(sid, path):
    t = pd.read_parquet(path + "/trials.parquet")
    t = t[t.ChoiceCorrect.notna() & t.sync_valid]
    if "overlaps_recording_gap" in t.columns:
        t = t[~t.overlaps_recording_gap]
    t = t.reset_index(drop=True)

    a0, lam = delay_law(t.FeedbackDelay.values)
    p_c = float((t.CatchTrial.fillna(0) > 0.5).mean())
    corr = t.ChoiceCorrect.values > 0.5
    ct = t.CatchTrial.fillna(0).values > 0.5
    rew = t.Rewarded.fillna(0).values > 0.5
    w = t.FeedbackTime.values.astype(float)

    usable = ((ct & corr) | (~corr)) & np.isfinite(w) & (w >= ABORT_FLOOR)
    abort = corr & ~ct & ~rew & np.isfinite(w)
    T, y = w[usable], corr[usable].astype(float)
    wt = np.where(y > 0.5, 1.0 / max(p_c, 1e-6), 1.0)

    # the threshold the measured reward rate implies, with nothing fitted
    span = float(np.nanmax(t.trial_start_bpod.values) - np.nanmin(t.trial_start_bpod.values))
    rho = float(rew.sum()) / span if span > 0 else np.nan

    # does the wait predict correctness BEYOND the stimulus?
    aDV = np.abs(t.DV.values)[usable]
    X = np.column_stack([np.ones_like(T), (aDV - aDV.mean()) / aDV.std(),
                         (T - T.mean()) / T.std()])
    beta, se = wls_logistic(X, y, wt)

    # conf_wait: the calibration curve, non-parametric.
    # Isotonic regression leaves single-trial blocks at both tails, which come out
    # at exactly 0 and 1. Clipped to [0.5, 0.99] because confidence in the option
    # you just chose cannot be below 0.5 -- you would have chosen the other one --
    # and cannot be certainty. The clip touches only the tail blocks.
    xs, fit = isotonic(T, y, wt)
    ps = fit_psychometric(t.DV.values, t.ChoiceLeft.values)
    conf_wait = np.full(len(t), np.nan)
    raw = iso_predict(xs, fit, T)
    conf_wait[usable] = np.clip(raw, CONF_LO, CONF_HI)
    n_clip = int(((raw < CONF_LO) | (raw > CONF_HI)).sum())

    table = pd.DataFrame(dict(
        trial_id=t.trial_id.values, sid=sid, DV=t.DV.values, absDV=np.abs(t.DV.values),
        ChoiceLeft=t.ChoiceLeft.values, correct=corr.astype(float),
        catch=ct.astype(float), rewarded=rew.astype(float), wait=w,
        delay=t.FeedbackDelay.values, usable=usable.astype(int),
        abort=abort.astype(int),
        conf_norm=normative_confidence(t.DV.values, t.ChoiceLeft.values, ps),
        conf_wait=conf_wait))

    sl, _, _ = local_slope(T, y, wt, p_c)
    return dict(sid=sid, trials=len(t), a0=a0, lam=lam, p_c=p_c, rho=rho,
                n_use=int(usable.sum()), n_catch=int((usable & corr).sum()),
                n_err=int((usable & ~corr).sum()), n_abort=int(abort.sum()),
                n_rew=int(rew.sum()), T=T, y=y, wt=wt, slope=sl, n_clip=n_clip,
                b_dv=beta[1], t_dv=beta[1] / se[1], b_T=beta[2], t_T=beta[2] / se[2],
                med_corr=float(np.median(T[y > 0.5])),
                med_err=float(np.median(T[y < 0.5])), table=table)


def recover_kappa(T, y, wt, p_c, lam, a0):
    """Anchor the predicted line to the measured curve at its midpoint -> kappa.

    logit(a) = lam (T - T0), so one point on the calibration curve fixes T0, and
    T0 inverts to kappa. Taken at the median wait, where the estimate is densest.
    """
    xs, fit = isotonic(T, y, wt)
    tm = float(np.median(T))
    am = float(iso_predict(xs, fit, tm)) * (1 - p_c)
    T0 = tm - logit(am) / lam
    kappa = lam / (1.0 + np.exp(lam * (T0 - a0)))
    return float(T0), float(kappa), tm, am


# ----------------------------------------------------------------------- the figure

def figure(res, pooled, args):
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.9))
    lam = float(np.mean([r["lam"] for r in res]))
    a0 = float(np.mean([r["a0"] for r in res]))
    pc = float(np.mean([r["p_c"] for r in res]))
    T = np.concatenate([r["T"] for r in res])
    y = np.concatenate([r["y"] for r in res])
    wt = np.concatenate([r["wt"] for r in res])
    rng = np.random.default_rng(0)

    # ---- A  the distribution that makes the slope a prediction
    ax = axes[0, 0]
    fd = pooled.delay.dropna().values
    ax.hist(fd, bins=np.arange(0, 8.3, 0.25), color=SHADES[1], edgecolor="white", lw=0.6)
    ax.set_xlabel("programmed reward delay  D  (s)")
    ax.set_ylabel("trials")
    ax.set_title("A   Why the slope is predicted, not fitted")
    ax.text(0.97, 0.95, "shifted exponential\nfloor $a_0$ = %.2f s\n"
                        "hazard $\\lambda$ = %.2f /s\nall %d sessions: %.2f-%.2f"
            % (a0, lam, len(res), min(r["lam"] for r in res), max(r["lam"] for r in res)),
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    iax = ax.inset_axes([0.46, 0.28, 0.50, 0.33])
    xs = np.sort(fd); sv = 1.0 - np.arange(len(xs)) / float(len(xs))
    k = (xs >= a0) & (sv > 0.01)
    iax.plot(xs[k], np.log(sv[k]), "-", color=INK, lw=1.8)
    iax.plot(xs[k], -lam * (xs[k] - a0), "--", color=C_ERR, lw=1.2)
    iax.set_xlabel("D (s)", fontsize=7); iax.set_ylabel("log S(D)", fontsize=7)
    iax.tick_params(labelsize=6)
    iax.set_title("straight = FLAT hazard", fontsize=7.5)
    for sp in ("top", "right"):
        iax.spines[sp].set_visible(False)
    ax.text(0.03, 0.62, "a flat hazard is why the rat\nleaves at all: waiting never\n"
                        "gets less promising, so the\nonly thing eroding the reason\n"
                        "to stay is the belief that the\ntrial was an error.",
            transform=ax.transAxes, va="top", fontsize=8.2, color=INK, bbox=BOX)

    # ---- B  which trials show a leave
    ax = axes[0, 1]
    p = pooled
    bins = np.arange(0, 14.3, 0.5)
    cat = ((p.correct > .5) & (p["catch"] > .5) & np.isfinite(p.wait)).values
    err = ((p.correct < .5) & np.isfinite(p.wait)).values
    ab = (p.abort > 0.5).values
    base = np.zeros(len(bins) - 1)
    for lab, m, col in (("correct + CATCH  n=%d" % int(cat.sum()), cat, C_CORR),
                        ("ERROR  n=%d" % int(err.sum()), err, C_ERR)):
        h, _ = np.histogram(p.wait.values[m], bins=bins)
        ax.bar(bins[:-1], h, width=0.5, bottom=base, align="edge", color=col,
               edgecolor="white", lw=0.4, label=lab)
        base += h
    h, _ = np.histogram(p.wait.values[ab], bins=bins)
    ax.bar(bins[:-1], -h, width=0.5, align="edge", color="#c9c9c9", edgecolor="white",
           lw=0.4, label="grace ABORT, dropped  n=%d" % int(ab.sum()))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("time at the side port  (s)")
    ax.set_ylabel("trials        dropped  <-   used  ->")
    ax.set_title("B   Two modes, and only one of them is a decision")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.03, 0.42,
            "the abort mode is accuracy-\nindependent: matched on the drawn\n"
            "delay, leaves under 2.5 s happen on\n5.0%% of correct and 5.3%% of error\n"
            "trials -- so it carries no confidence.\nIt is also only VISIBLE when the\n"
            "drawn delay was long, so keeping it\ninverts panel C.\n"
            "%d rewarded trials show no leave."
            % int((p.rewarded > 0.5).sum()),
            transform=ax.transAxes, va="top", fontsize=8.2, color=INK, bbox=BOX)

    # ---- C  the calibration curve: the answer, plus the test
    ax = axes[0, 2]
    edges = np.percentile(T, np.linspace(0, 100, 10))
    edges[0], edges[-1] = T.min() - 1e-6, T.max() + 1e-6
    cb = weighted_bins(T, y, wt, edges)
    xs_i, fit_i = isotonic(T, y, wt)
    g = np.linspace(T.min(), np.percentile(T, 99), 400)
    ax.plot(g, iso_predict(xs_i, fit_i, g), "-", color=C_ISO, lw=2.6, zorder=4,
            label="isotonic fit = conf_wait")
    ax.errorbar(cb[:, 0], cb[:, 1], yerr=cb[:, 2], fmt="o", color=C_CORR, ms=6.5,
                capsize=2.5, zorder=5, label="measured, catch reweighted by $1/p_c$")
    T0, kappa, tm, am = recover_kappa(T, y, wt, pc, lam, a0)
    ax.plot(g, np.clip(sig(lam * (g - T0)) / (1 - pc), 0, 1), "--", color=C_PRED,
            lw=2.0, zorder=3,
            label="policy with $\\lambda$=%.2f pinned, anchored at the median" % lam)
    sl, t1, t2 = local_slope(T, y, wt, pc)
    lo, hi = boot_slope(T, y, wt, pc, rng)
    # Motivation drifts between sessions -- the median wait runs 5.7 to 7.5 s
    # against an IQR of about 2 s. That is variance in T with no confidence in it,
    # and it attenuates the slope, so re-read it with each session centred on the
    # grand median. Any rise is drift that the pooled number was paying for.
    gm = float(np.median(T))
    Tc = np.concatenate([r["T"] - np.median(r["T"]) + gm for r in res])
    sl_c, _, _ = local_slope(Tc, y, wt, pc)
    lo_c, hi_c = boot_slope(Tc, y, wt, pc, rng)
    ax.axvspan(t1, t2, color="#f0ece4", zorder=0)
    ax.axhline(float((wt * y).sum() / wt.sum()), color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("waiting time the rat chose  T  (s)")
    ax.set_ylabel("P(correct | the rat waited T)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("C   Confidence, measured without a psychometric")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.text(0.03, 0.97,
            "%.2f at %.1f s  ->  %.2f at %.1f s.\nNo model, no threshold, no\n"
            "psychometric: just the fraction\ncorrect, binned by how long the\n"
            "rat chose to wait.\n\nshaded = the central half, where\n"
            "the slope is read:\n  predicted  $\\lambda$ = %.2f /s\n"
            "  measured   %.2f  [%.2f, %.2f]\n  attenuated x%.1f\n"
            "  session-centred %.2f  (x%.1f)"
            % (cb[0, 1], cb[0, 0], cb[-1, 1], cb[-1, 0], lam, sl, lo, hi, lam / sl,
               sl_c, lam / sl_c),
            transform=ax.transAxes, va="top", fontsize=8.2, color=INK, bbox=BOX)

    # ---- D  the threshold, which is all assumption B costs
    ax = axes[1, 0]
    cg = np.linspace(0.02, 0.999, 400)
    for kap, col in zip((0.30, 0.10, 0.03), SHADES):
        tt0 = a0 - np.log(kap / (lam - kap)) / lam
        a = np.clip(cg * (1 - pc), 1e-6, 1 - 1e-6)
        ax.plot(cg, tt0 + np.log(a / (1 - a)) / lam, "-", color=col, lw=2.2,
                label="$\\kappa$ = %.2f /s  ->  $T_0$ = %.2f s" % (kap, tt0))
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.set_ylim(-1, 15)
    ax.set_xlabel("confidence  c = P(correct | this choice)")
    ax.set_ylabel("waiting time the policy prescribes  (s)")
    ax.set_title("D   The unknown threshold slides the map; it cannot bend it")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    rho = float(np.mean([r["rho"] for r in res]))
    ax.text(0.97, 0.05,
            "recovered from panel C:\n   $T_0$ = %.2f s  ->  $\\kappa$ = %.3f /s\n"
            "measured reward rate:\n   $\\rho/R$ = %.3f /s\n"
            "ratio %.1fx, and nothing was fitted\non the second number."
            % (T0, kappa, rho, max(kappa, rho) / max(min(kappa, rho), 1e-9)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4, color=INK, bbox=BOX)

    # ---- E  what conf_wait buys over conf_norm
    ax = axes[1, 1]
    q = pooled[pooled.usable > 0.5]
    c_ = q.correct.values > 0.5
    jit = np.random.default_rng(1).normal(0, 0.004, len(q))
    ax.plot(q.conf_norm.values[c_], (q.conf_wait.values + jit)[c_], ".", color=C_CORR,
            ms=4.5, alpha=0.5)
    ax.plot(q.conf_norm.values[~c_], (q.conf_wait.values + jit)[~c_], ".", color=C_ERR,
            ms=5, alpha=0.75)
    r = np.corrcoef(q.conf_norm.values, q.conf_wait.values)[0, 1]
    # the cell must be within a session too: conf_norm is fitted per session, so
    # the same DV in two sessions gets two values and the spread would not be zero.
    key = (q.sid + "|" + q.DV.round(3).astype(str) + "|"
           + (q.ChoiceLeft > 0.5).astype(int).astype(str))
    gw = q.groupby(key).conf_wait.std().dropna()
    gn = q.groupby(key).conf_norm.std().dropna()
    ax.set_xlabel("conf_norm   (from the psychometric)")
    ax.set_ylabel("conf_wait   (from the waiting time)")
    ax.set_title("E   Nearly independent -- which is the whole point")
    ax.text(0.03, 0.97,
            "r = %+.2f over %d usable trials\n\nwithin ONE stimulus x choice cell:\n"
            "   conf_norm  sd = %.3f\n   conf_wait  sd = %.3f\n\n"
            "conf_norm is a FUNCTION of DV and\nthe choice, so it has no within-cell\n"
            "spread at all, by construction.\nconf_wait's spread is the rat's own\n"
            "trial-to-trial belief, and it is the\nonly handle that can test confidence\n"
            "coding with the stimulus held fixed."
            % (r, len(q), gn.mean(), gw.mean()),
            transform=ax.transAxes, va="top", fontsize=8.3, color=INK, bbox=BOX)

    # ---- F  does the wait beat the stimulus?
    ax = axes[1, 2]
    aDV = q.absDV.values
    terc = np.percentile(aDV, [0, 33.3, 66.7, 100])
    for i, col in enumerate(SHADES):
        m = (aDV >= terc[i]) & (aDV <= terc[i + 1])
        if m.sum() < 40:
            continue
        Tq, yq = q.wait.values[m], q.correct.values[m]
        wq = np.where(yq > 0.5, 1.0 / pc, 1.0)
        ed = np.percentile(Tq, np.linspace(0, 100, 5)); ed[0] -= 1e-6; ed[-1] += 1e-6
        bb = weighted_bins(Tq, yq, wq, ed)
        ax.errorbar(bb[:, 0], bb[:, 1], yerr=bb[:, 2], fmt="o-", color=col, ms=6,
                    lw=2.0, capsize=2,
                    label="|DV| %.2f-%.2f   n=%d" % (terc[i], terc[i + 1], int(m.sum())))
    ax.set_xlabel("waiting time  T  (s)")
    ax.set_ylabel("P(correct | T), within a |DV| tercile")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("F   The wait carries confidence the stimulus does not")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    Xp = np.column_stack([np.ones(len(q)), (aDV - aDV.mean()) / aDV.std(),
                          (q.wait.values - q.wait.values.mean()) / q.wait.values.std()])
    bb2, ss2 = wls_logistic(Xp, q.correct.values,
                            np.where(q.correct.values > .5, 1 / pc, 1.0))
    ax.text(0.03, 0.97,
            "pooled logistic,  correct ~ |DV| + T:\n"
            "   |DV|   beta %+.2f   t %+.1f\n   T      beta %+.2f   t %+.1f\n\n"
            "per-session mean t:  |DV| %+.1f,  T %+.1f\n"
            "T stays significant with the stimulus\nin the model, so the wait is not a\n"
            "re-reading of the stimulus."
            % (bb2[1], bb2[1] / ss2[1], bb2[2], bb2[2] / ss2[2],
               np.mean([r["t_dv"] for r in res]), np.mean([r["t_T"] for r in res])),
            transform=ax.transAxes, va="top", fontsize=8.3, color=INK, bbox=BOX)

    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Confidence from the waiting time, not the psychometric   --   "
                 "Shamir_1, %d sessions, %d usable leaves of %d trials"
                 % (len(res), int(pooled.usable.sum()), len(pooled)),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "confidence_from_waiting.png")
    fig.savefig(path, dpi=150)
    print("\nwrote", path)
    return dict(lam=lam, a0=a0, pc=pc, slope=sl, lo=lo, hi=hi, T0=T0, kappa=kappa,
                slope_c=sl_c, lo_c=lo_c, hi_c=hi_c,
                rho=rho, cb=cb, bT=bb2[2] / ss2[2], bD=bb2[1] / ss2[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_coding"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = []
    print("%-17s %6s %6s %5s %5s %5s %6s %7s %7s %7s %7s"
          % ("session", "trials", "usable", "ctch", "err", "abrt", "lam_D",
             "medC", "medE", "slope", "t(T|DV)"))
    for sid in args.sessions:
        r = analyse(sid, SESSIONS[sid])
        res.append(r)
        r["table"].to_csv(os.path.join(args.out, "conf_wait_%s.csv" % sid), index=False)
        print("%-17s %6d %6d %5d %5d %5d %6.2f %7.2f %7.2f %7.2f %+7.1f"
              % (sid, r["trials"], r["n_use"], r["n_catch"], r["n_err"], r["n_abort"],
                 r["lam"], r["med_corr"], r["med_err"], r["slope"], r["t_T"]))

    pooled = pd.concat([r["table"] for r in res], ignore_index=True)
    pooled.to_csv(os.path.join(args.out, "conf_wait_pooled.csv"), index=False)
    s = figure(res, pooled, args)
    q = pooled[pooled.usable > 0.5]

    print("\nPOOLED  (%d usable leaves)" % len(q))
    print("  A  slope the delay distribution predicts      lambda = %.2f /s" % s["lam"])
    print("     slope the rat's calibration curve shows           %.2f /s "
          "[%.2f, %.2f]  -> attenuated x%.1f"
          % (s["slope"], s["lo"], s["hi"], s["lam"] / s["slope"]))
    print("     same, each session centred on the grand median      %.2f /s "
          "[%.2f, %.2f]  -> attenuated x%.1f"
          % (s["slope_c"], s["lo_c"], s["hi_c"], s["lam"] / s["slope_c"]))
    print("  B  threshold recovered from the curve         kappa  = %.3f /s  (T0 = %.2f s)"
          % (s["kappa"], s["T0"]))
    print("     measured background reward rate            rho/R  = %.3f /s" % s["rho"])
    print("  wait predicts correctness given |DV|:  t(T) = %+.1f   t(|DV|) = %+.1f"
          % (s["bT"], s["bD"]))
    print("  corr(conf_wait, conf_norm) = %+.3f   corr(conf_wait, |DV|) = %+.3f"
          % (np.corrcoef(q.conf_norm, q.conf_wait)[0, 1],
             np.corrcoef(q.absDV, q.conf_wait)[0, 1]))
    print("  conf_wait range %.3f - %.3f   median wait  correct %.2f s  error %.2f s"
          % (q.conf_wait.min(), q.conf_wait.max(),
             np.median(np.concatenate([r["T"][r["y"] > .5] for r in res])),
             np.median(np.concatenate([r["T"][r["y"] < .5] for r in res]))))


if __name__ == "__main__":
    main()
