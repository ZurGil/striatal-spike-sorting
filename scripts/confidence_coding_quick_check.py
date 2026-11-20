# -*- coding: utf-8 -*-
"""Does the post-cue / choice-port response scale with decision confidence?

    python scripts/confidence_coding_quick_check.py
    python scripts/confidence_coding_quick_check.py --sessions 20260917_100043

Gil's question, in his words: "is the response right after the cue (aligned to the
end of the cue), or to the choice port in, scaled with the confidence, or can you
encode the confidence from it -- of course confidence needs to be defined, so for
now maybe just assume there is a 1-to-1 map between DV and confidence, I'm not sure".

HE IS RIGHT TO BE UNSURE, AND THE DATA SETTLES IT
-------------------------------------------------
|DV| is stimulus strength. Confidence is the animal's belief that THIS choice was
correct, so it depends on the choice as well as the stimulus: a strong stimulus
answered wrongly is the LOWEST-confidence event there is, not the highest. The two
definitions agree on correct trials and are opposite on errors, which is exactly
where the test lives. So three confidence regressors are carried side by side:

  absDV     |DV|. Gil's 1-to-1 assumption. Pure stimulus strength.
  conf_norm P(correct | DV, choice) from this session's own psychometric fit.
            The normative quantity: high on easy-correct, LOW on easy-error.
  wait      the animal's own report -- how long it waits at the side port when no
            reward comes (correct catch trials and error trials alike). Shamir only;
            on Popper CatchError = 0, so the error wait is our delay, not the rat's
            decision. See outputs/READ_ME_CatchError_0_NO_ERROR_CATCH_TRIALS.txt.

EPOCHS
------
  base   300 ms before stimulus onset, nose in the centre port
  stim   the stimulus itself (fixed, ~350 ms)
  cue    0-200 ms after the END of the stimulus          <- Gil's first alignment
  chin   0-450 ms after the choice port In               <- Gil's second alignment
         (feedback delay is ~500 ms, so this stays clear of the valve)
  wait   500-1500 ms after the choice port In, UNREWARDED trials only -- the
         time-investment epoch, where a confidence signal should be largest

CONTROLS, because every one of these can manufacture a fake confidence cell
  choice side   enormous at chin; always in the model
  movement time correlates with difficulty; always in the model
  trial index   slow drift in firing rate plus block structure in DV
  shuffle null  trial labels permuted 200x, refit, to get the count of "significant"
                units expected by chance with the same drift and the same n

Writes outputs/confidence_coding/<session>_units.csv (one row per unit per epoch)
and outputs/confidence_coding/summary.csv. Nothing on Z is touched.
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "confidence_coding")

SESSIONS = {
    "20260914_122553": r"Z:/Gil/Shamir_1/20260914_122553.rec/20260914_122553.kilosort/synced",
    "20260915_084812": r"Z:/Gil/Shamir_1/20260915_084812.rec/20260915_084812.kilosort/synced_from_concat",
    "20260915_103710": r"Z:/Gil/Shamir_1/20260915_103710.rec/20260915_103710.kilosort/synced_from_concat",
    "20260916_110311": r"Z:/Gil/Shamir_1/20260916_110311.rec/20260916_110311.kilosort/synced",
    "20260917_100043": r"Z:/Gil/Shamir_1/20260917_100043.rec/20260917_100043.kilosort/synced",
    "20260918_100248": r"Z:/Gil/Shamir_1/20260918_100248.rec/20260918_100248.kilosort/synced",
    "20260922_144404": r"Z:/Gil/Shamir_1/20260922_144404.rec/20260922_144404.kilosort/synced",
    "20260925_103539": r"Z:/Gil/Shamir_1/20260925_103539.rec/20260925_103539.kilosort/synced",
    "20261112_082028": r"Z:/Gil/Popper/20261112_082028.rec/20261112_082028.kilosort/synced",
    # Synced long ago but never reviewed: its "good" labels are Kilosort's, not Gil's.
    # Kept out of the default set and run only when asked, because pooling a
    # hand-curated unit population with an automatic one hides which is which.
    "20260911_100049": r"Z:/Gil/Shamir_1/20260911_100049.rec/20260911_100049.kilosort/synced",
}
UNREVIEWED = {"20260911_100049"}

# Which animal each session belongs to. Pooling per-unit counts across rats answers
# no question -- Gil, 2026-11-18, after one Popper session with 27 of 62 units
# significant for P(correct) at the port carried the pooled headline and halved it
# when removed (5bd.73). So the default is one rat, and mixing takes --allow_mixed_rats.
RAT = {
    "20260914_122553": "Shamir_1",
    "20260915_084812": "Shamir_1",
    "20260915_103710": "Shamir_1",
    "20260916_110311": "Shamir_1",
    "20260917_100043": "Shamir_1",
    "20260918_100248": "Shamir_1",
    "20260922_144404": "Shamir_1",
    "20260925_103539": "Shamir_1",
    "20260911_100049": "Shamir_1",
    "20261112_082028": "Popper",
}
DEFAULT_RAT = "Shamir_1"
DEFAULT_SESSIONS = [s for s in sorted(SESSIONS)
                    if s not in UNREVIEWED and RAT.get(s) == DEFAULT_RAT]
# CatchError = 0 here, so the error "wait" is our feedback delay and means nothing.
NO_ERROR_CATCH = {"20261112_082028"}

N_SHUFFLE = 200
# How long the animal must have stayed at the side port for an epoch's window to be
# entirely inside the wait. Epochs before the choice poke need no guard.
WAIT_GUARD = {"base": 0.0, "stim": 0.0, "cue": 0.0, "chin": 0.46, "wait": 1.55}
FDR_Q = 0.05
MIN_RATE_HZ = 0.5          # a unit below this in an epoch cannot support a slope test
MIN_TRIALS = 60


# ----------------------------------------------------------------- small statistics

def ols(X, y):
    """Coefficients, their t statistics, and R^2. Plain OLS, no package."""
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X.dot(beta)
    dof = n - rank
    if dof <= 0:
        return beta, np.zeros(k), 0.0
    s2 = resid.dot(resid) / dof
    try:
        cov = s2 * np.linalg.pinv(X.T.dot(X))
    except np.linalg.LinAlgError:
        return beta, np.zeros(k), 0.0
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - resid.dot(resid) / ss_tot if ss_tot > 0 else 0.0
    return beta, beta / se, r2


def t_to_p(t, dof):
    """Two-sided p from a t statistic, via the normal approximation above 30 dof.

    Every test here has hundreds of degrees of freedom, so the approximation costs
    nothing and keeps scipy out of the dependency list.
    """
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def bh_fdr(p, q=FDR_Q):
    """Benjamini-Hochberg. Returns the boolean reject vector."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / float(n))
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        last = np.max(np.nonzero(passed)[0])
        keep[order[:last + 1]] = True
    return keep


def fit_psychometric(dv, chose_left):
    """P(choose left) = lapse + (1 - 2*lapse) * logistic(slope * (DV - bias)).

    Positive DV means left is the rewarded side in this protocol (checked: P(left)
    runs 0.00 -> 0.97 across the DV range). One symmetric lapse rather than two,
    because a few hundred trials cannot separate two tails and an unbounded lapse
    pair is exactly what ran away in the first version of this fit.
    """
    dv = np.asarray(dv, float)
    y = np.asarray(chose_left, float)

    def nll(bias, slope, lapse):
        pr = lapse + (1.0 - 2.0 * lapse) / (1.0 + np.exp(-slope * (dv - bias)))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).sum()

    grid_b = np.linspace(-0.5, 0.5, 41)
    grid_s = np.concatenate([np.linspace(0.5, 10, 20), np.linspace(11, 40, 15)])
    grid_l = np.linspace(0.0, 0.25, 26)
    best, bestv = (0.0, 5.0, 0.05), np.inf
    for b in grid_b:
        for sl in grid_s:
            for la in grid_l:
                v = nll(b, sl, la)
                if v < bestv:
                    best, bestv = (b, sl, la), v
    b, sl, la = best
    for step in (0.02, 0.005, 0.001):
        for _ in range(40):
            moved = False
            for i, sc in enumerate((1.0, 20.0, 1.0)):
                for d in (step * sc, -step * sc):
                    c = [b, sl, la]
                    c[i] += d
                    if c[1] <= 0.1 or not (0.0 <= c[2] <= 0.3):
                        continue
                    v = nll(*c)
                    if v < bestv - 1e-9:
                        b, sl, la, bestv, moved = c[0], c[1], c[2], v, True
            if not moved:
                break
    return np.array([b, sl, la])


def normative_confidence(dv, chose_left, params):
    """P(the choice just made is the correct one), from the fitted psychometric.

    The lapse-free part of the fit is used: a lapse is a trial on which the animal
    did not consult the evidence, and folding it in here would flatten confidence
    toward 0.5 everywhere for a reason that has nothing to do with the evidence.
    """
    bias, slope, _ = params
    p_left = 1.0 / (1.0 + np.exp(-slope * (np.asarray(dv, float) - bias)))
    return np.where(np.asarray(chose_left, float) > 0.5, p_left, 1.0 - p_left)


# --------------------------------------------------------------------- trial events

def build_trials(S, sid):
    trials = pd.read_parquet(S + "/trials.parquet")
    se = pd.read_parquet(S + "/state_events.parquet")
    pe = pd.read_parquet(S + "/poke_events.parquet")

    stim_on = (se[se.state_name == "stimulus_delivery_min"]
               .groupby("trial_id").start_time_in_trial.min().rename("stim_on"))
    # The stimulus ends when stimulus_delivery ends; on the handful of trials where
    # that state is absent the trial did not complete and is dropped below anyway.
    cue_end = (se[se.state_name == "stimulus_delivery"]
               .groupby("trial_id").end_time_in_trial.max().rename("cue_end"))
    # The valve. Used only as a hard cap, so a long feedback delay can never let the
    # choice window swallow the reward.
    water = (se[se.state_name.isin(["water_L", "water_R"])]
             .groupby("trial_id").start_time_in_trial.min().rename("water_on"))

    t = trials.set_index("trial_id").join([stim_on, cue_end, water])
    t = t[t.ChoiceCorrect.notna() & t.sync_valid & ~t.overlaps_recording_gap]
    t = t[t.cue_end.notna() & t.stim_on.notna()]

    side = pe[(pe.port_role.isin(["left", "right"])) & (pe.event_type == "In")]
    by_trial = {tid: g.time_in_trial.values for tid, g in side.groupby("trial_id")}
    choice_in = []
    for tid, ce in zip(t.index.values, t.cue_end.values):
        a = by_trial.get(tid)
        v = np.nan
        if a is not None:
            a = a[a >= ce - 1e-6]
            if len(a):
                v = a[0]
        choice_in.append(v)
    t["choice_in"] = choice_in
    t = t[t.choice_in.notna()]

    t["absDV"] = t.DV.abs()
    t["chose_right"] = (1.0 - t.ChoiceLeft).astype(float)   # ChoiceLeft 1 = left
    t["correct"] = t.ChoiceCorrect.astype(float)
    t["rewarded"] = t.Rewarded.fillna(0).astype(float)
    # A trial on which the animal's wait is its own decision: a correct CATCH trial
    # (reward deliberately withheld) or an error (never announced on Shamir). A
    # correct non-catch trial with Rewarded = 0 is one where the rat left before the
    # valve -- median wait 1.2 s against 5.7 s on catch -- so it is not an investment
    # and including it was what inverted the first version of the behaviour table.
    t["time_investment"] = (((t.CatchTrial.fillna(0) > 0.5) & (t.ChoiceCorrect > 0.5))
                            | (t.ChoiceCorrect < 0.5)).astype(bool)
    t["wait"] = t.FeedbackTime.astype(float)
    t["trial_idx"] = np.arange(len(t), dtype=float)

    params = fit_psychometric(t.DV.values, t.ChoiceLeft.values)
    t["conf_norm"] = normative_confidence(t.DV.values, t.ChoiceLeft.values, params)
    t.attrs["psychometric"] = dict(bias=params[0], slope=params[1], lapse=params[2])
    return t.reset_index()


EPOCHS = ("base", "stim", "cue", "chin", "wait")


def epoch_windows(t):
    """Start and stop in trial time for each epoch, as two arrays per epoch."""
    ce = t.cue_end.values
    so = t.stim_on.values
    ci = t.choice_in.values
    cap = np.where(np.isfinite(t.water_on.values), t.water_on.values, np.inf)
    w = {}
    w["base"] = (so - 0.30, so)
    w["stim"] = (so, ce)
    w["cue"] = (ce, np.minimum(ce + 0.20, ci))     # never past the choice poke
    w["chin"] = (ci, np.minimum(ci + 0.45, cap))
    w["wait"] = (ci + 0.50, ci + 1.50)             # unrewarded trials only, masked later
    return w


# ------------------------------------------------------------------- spikes -> rates

def count_spikes(S, t, good_ids):
    """counts[epoch] -> (n_trials, n_units) integer array, plus the window durations."""
    uids = np.asarray(sorted(good_ids), dtype=np.int64)
    tids = t.trial_id.values.astype(np.int64)
    nU, nT = len(uids), len(tids)

    sp = pd.read_parquet(S + "/spikes.parquet",
                         columns=["unit_id", "trial_id", "spike_time_in_trial"],
                         filters=[("unit_id", "in", uids.tolist())])
    su = sp.unit_id.values.astype(np.int64)
    st_ = sp.trial_id.values.astype(np.int64)
    tm = sp.spike_time_in_trial.values.astype(np.float64)
    del sp

    # searchsorted maps ids to row/column indices; -1 for anything not in our sets.
    order_t = np.argsort(tids)
    pos = np.searchsorted(tids[order_t], st_)
    pos = np.clip(pos, 0, nT - 1)
    ti = np.where(tids[order_t][pos] == st_, order_t[pos], -1)
    ui = np.searchsorted(uids, su)
    ui = np.clip(ui, 0, nU - 1)
    ui = np.where(uids[ui] == su, ui, -1)
    keep = (ti >= 0) & (ui >= 0)
    ti, ui, tm = ti[keep], ui[keep], tm[keep]

    win = epoch_windows(t)
    counts, durs = {}, {}
    for ep in EPOCHS:
        a, b = win[ep]
        m = (tm >= a[ti]) & (tm < b[ti])
        flat = np.bincount(ti[m] * nU + ui[m], minlength=nT * nU)
        counts[ep] = flat.reshape(nT, nU)
        durs[ep] = np.maximum(b - a, 0.0)
    return uids, counts, durs


def selection_mask(S, t, uids):
    """(n_trials, n_units) boolean: was this unit present for this trial?"""
    nT, nU = len(t), len(uids)
    mask = np.ones((nT, nU), dtype=bool)
    try:
        sel = pd.read_parquet(S + "/unit_trial_selection.parquet")
    except (FileNotFoundError, OSError):
        return mask
    sel = sel[~sel.selected]
    if not len(sel):
        return mask
    tpos = pd.Series(np.arange(nT), index=t.trial_id.values)
    upos = pd.Series(np.arange(nU), index=uids)
    ti = sel.trial_id.map(tpos)
    ui = sel.unit_id.map(upos)
    ok = ti.notna() & ui.notna()
    mask[ti[ok].values.astype(int), ui[ok].values.astype(int)] = False
    return mask


# ------------------------------------------------------------------------ the models

def design(t, rows, conf, with_interaction=False):
    """Intercept, confidence regressor, and the three nuisance terms."""
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    cols = [np.ones(rows.sum()),
            z(conf[rows]),
            t.chose_right.values[rows] - 0.5,
            z(np.log(np.maximum(t.MT.values[rows], 1e-3))),
            z(t.trial_idx.values[rows])]
    names = ["const", "conf", "side", "logMT", "drift"]
    if with_interaction:
        corr = t.correct.values[rows] - 0.5
        cols += [corr, z(conf[rows]) * corr]
        names += ["correct", "conf_x_correct"]
    return np.column_stack(cols), names


def analyse_session(sid, S, verbose=True):
    t = build_trials(S, sid)
    units = pd.read_parquet(S + "/units.parquet")
    good = units.loc[units.quality_label == "good", "unit_id"].values
    if verbose:
        print("\n=== %s   %d completed trials, %d good units" % (sid, len(t), len(good)))
        print("    psychometric:", {k: round(v, 3) for k, v in t.attrs["psychometric"].items()})

    uids, counts, durs = count_spikes(S, t, good)
    keep = selection_mask(S, t, uids)

    invest = t.time_investment.values
    rows = []
    for ep in EPOCHS:
        dur = durs[ep]
        valid_trial = dur > 0.02
        if ep == "wait":
            valid_trial = valid_trial & invest
            if sid in NO_ERROR_CATCH:
                # The wait on an error here is our feedback delay, not a decision.
                valid_trial = valid_trial & (t.correct.values > 0.5)
        for ui in range(len(uids)):
            sel = valid_trial & keep[:, ui]
            if sel.sum() < MIN_TRIALS:
                continue
            rate = counts[ep][sel, ui] / dur[sel]
            if rate.mean() < MIN_RATE_HZ:
                continue
            rec = {"session": sid, "unit_id": int(uids[ui]), "epoch": ep,
                   "n_trials": int(sel.sum()), "mean_rate": float(rate.mean())}

            for key, conf in (("absDV", t.absDV.values), ("conf", t.conf_norm.values)):
                X, names = design(t, sel, conf)
                beta, tt, r2 = ols(X, rate)
                j = names.index("conf")
                rec[key + "_beta"] = float(beta[j])
                rec[key + "_t"] = float(tt[j])
                rec[key + "_p"] = float(t_to_p(tt[j], sel.sum() - X.shape[1]))
                rec[key + "_r2"] = float(r2)

            # The X pattern: slope against |DV| separately for correct and error.
            X, names = design(t, sel, t.absDV.values, with_interaction=True)
            beta, tt, _ = ols(X, rate)
            j, k = names.index("conf"), names.index("conf_x_correct")
            rec["slope_correct"] = float(beta[j] + 0.5 * beta[k])
            rec["slope_error"] = float(beta[j] - 0.5 * beta[k])
            rec["interaction_t"] = float(tt[k])
            rec["interaction_p"] = float(t_to_p(tt[k], sel.sum() - X.shape[1]))

            # The animal's own report, where it exists.
            w = t.wait.values
            wsel = sel & np.isfinite(w) & invest & (w >= WAIT_GUARD[ep])
            if sid in NO_ERROR_CATCH:
                wsel = wsel & (t.correct.values > 0.5)
            if wsel.sum() >= 40:
                rate_w = counts[ep][wsel, ui] / dur[wsel]
                X, names = design(t, wsel, np.log(np.maximum(w, 0.05)))
                beta, tt, _ = ols(X, rate_w)
                j = names.index("conf")
                rec["wait_beta"] = float(beta[j])
                rec["wait_t"] = float(tt[j])
                rec["wait_p"] = float(t_to_p(tt[j], wsel.sum() - X.shape[1]))
                rec["wait_n"] = int(wsel.sum())
            rows.append(rec)

    out = pd.DataFrame(rows)

    # ---- the shuffle null: same drift, same n, confidence destroyed
    rng = np.random.default_rng(7)
    null = {}
    for ep in EPOCHS:
        dur = durs[ep]
        valid_trial = dur > 0.02
        if ep == "wait":
            valid_trial = valid_trial & invest
            if sid in NO_ERROR_CATCH:
                valid_trial = valid_trial & (t.correct.values > 0.5)
        # Units nobody restricted all share one trial set, so the whole null can be
        # refit as one matrix product per permutation. Restricted units are a small
        # minority and are left out of the null, not out of the analysis.
        full = valid_trial
        cols = [ui for ui in range(len(uids)) if keep[:, ui].all()]
        if full.sum() < MIN_TRIALS or not cols:
            continue
        R = counts[ep][full][:, cols] / dur[full][:, None]
        alive = R.mean(axis=0) >= MIN_RATE_HZ
        R = R[:, alive]
        if R.shape[1] == 0:
            continue
        conf = t.conf_norm.values
        hits = np.zeros(N_SHUFFLE)
        for s in range(N_SHUFFLE):
            perm = rng.permutation(full.sum())
            c = conf[full][perm]
            z = (c - c.mean()) / (c.std() + 1e-12)
            X = np.column_stack([np.ones(full.sum()), z,
                                 t.chose_right.values[full] - 0.5,
                                 t.trial_idx.values[full]])
            XtXi = np.linalg.pinv(X.T.dot(X))
            B = XtXi.dot(X.T).dot(R)
            res = R - X.dot(B)
            s2 = (res ** 2).sum(axis=0) / (full.sum() - X.shape[1])
            se = np.sqrt(np.maximum(XtXi[1, 1] * s2, 1e-30))
            tt = B[1] / se
            p = np.array([t_to_p(v, full.sum() - X.shape[1]) for v in tt])
            hits[s] = bh_fdr(p).sum()
        null[ep] = (float(hits.mean()), float(np.percentile(hits, 95)), int(R.shape[1]))

    # ---- population read-out, on the units nobody restricted
    dec = {}
    cols = [ui for ui in range(len(uids)) if keep[:, ui].all()]
    for ep in EPOCHS:
        dur = durs[ep]
        vt = dur > 0.02
        if ep == "wait":
            vt = vt & invest
            if sid in NO_ERROR_CATCH:
                vt = vt & (t.correct.values > 0.5)
        if vt.sum() < MIN_TRIALS or not cols:
            continue
        R = counts[ep][vt][:, cols] / dur[vt][:, None]
        alive = R.mean(axis=0) >= MIN_RATE_HZ
        R = R[:, alive]
        if R.shape[1] < 5:
            continue
        # The same cells over the 300 ms before the stimulus, for the control below.
        Rb = (counts["base"][vt][:, cols] / np.maximum(durs["base"][vt], 1e-6)[:, None]
              )[:, alive]
        tg = {"absDV": t.absDV.values[vt],
              "conf_norm": t.conf_norm.values[vt],
              "side": t.ChoiceLeft.values[vt].astype(float)}
        corr = t.correct.values[vt] > 0.5
        d = decode_epoch(R, tg)
        # confidence read-out tested on correct trials alone, where |DV| and
        # confidence agree, so a hit there cannot be an error signal in disguise
        if corr.sum() >= MIN_TRIALS:
            dc = decode_epoch(R[corr], {"absDV_correct_only": t.absDV.values[vt][corr]})
            d.update(dc)
        # Confidence read out WITHIN one choice side at a time. The side read-out in
        # panel E is enormous (R^2 ~ 0.85) and the confidence one is small, and the
        # two decoders are fitted independently on the same cells -- so the obvious
        # worry is that the small one is the big one leaking. |DV| and conf_norm are
        # unsigned and so side-independent by construction, but that is an argument,
        # not a measurement. Holding the choice fixed is the measurement: within
        # left-choice trials every trial has the same side, so nothing a side signal
        # does can produce a confidence read-out here.
        for lab, sel in (("left", t.ChoiceLeft.values[vt] > 0.5),
                         ("right", t.ChoiceLeft.values[vt] < 0.5)):
            if sel.sum() >= MIN_TRIALS:
                d.update(decode_epoch(R[sel], {
                    "conf_norm_%s_only" % lab: t.conf_norm.values[vt][sel]}))
        # The animal's own confidence report. Guarded: a window that runs past the
        # moment the rat gave up and left the port would be reading the withdrawal,
        # not the confidence that set it, and short waits are exactly the
        # low-confidence trials -- the confound would manufacture the result.
        w = t.wait.values[vt]
        ok = np.isfinite(w) & invest[vt]
        if sid in NO_ERROR_CATCH:
            ok = ok & corr
        ok = ok & (w >= WAIT_GUARD[ep])
        if ok.sum() >= MIN_TRIALS:
            d.update(decode_epoch(R[ok], {"log_wait": np.log(np.maximum(w[ok], 0.05))}))
            d["n_wait_trials"] = int(ok.sum())
        # Does the population already know this choice is going to be wrong? A
        # confidence signal must say yes; a pure stimulus-strength signal need not.
        ab = decode_binary(R, t.correct.values[vt])
        if ab is not None:
            d["correct_vs_error"] = dict(r2=ab["auc"], null_median=ab["null_median"],
                                         p=ab["p"], shift_median=np.nan, p_shift=np.nan)
        # ...and is any of it evoked, rather than the state the animal was already in?
        # Same cells, same trials, with each cell's own pre-stimulus rate subtracted.
        # A signal that survives this started after the trial did.
        if ep in ("cue", "chin"):
            D = R - Rb
            dd = decode_epoch(D, {"absDV_minus_base": t.absDV.values[vt],
                                  "conf_norm_minus_base": t.conf_norm.values[vt]})
            d.update(dd)
            ab2 = decode_binary(D, t.correct.values[vt])
            if ab2 is not None:
                d["correct_vs_error_minus_base"] = dict(
                    r2=ab2["auc"], null_median=ab2["null_median"], p=ab2["p"],
                    shift_median=np.nan, p_shift=np.nan)
        d["n_units"] = R.shape[1]
        d["n_trials"] = int(vt.sum())
        dec[ep] = d

    return t, out, null, dec



# ------------------------------------------------------------- population decoding

ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def ridge_cv_r2(R, y, n_folds=5, seed=0):
    """Cross-validated R^2 of a ridge read-out of y from the population matrix R.

    The ridge penalty is picked on the training half of each fold, so the number
    reported is out-of-sample end to end and the permutation null below runs the
    identical procedure -- any optimism from picking alpha is in the null too.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    fold = rng.permutation(n) % n_folds
    pred = np.zeros(n)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if tr.sum() < 20 or te.sum() < 3:
            return np.nan
        mu, sd = R[tr].mean(axis=0), R[tr].std(axis=0) + 1e-9
        Xtr, Xte = (R[tr] - mu) / sd, (R[te] - mu) / sd
        ym = y[tr].mean()
        G = Xtr.T.dot(Xtr)
        b = Xtr.T.dot(y[tr] - ym)
        # inner split, to choose alpha without ever looking at the test fold
        inner = rng.permutation(tr.sum()) % 2
        best, bestsc = ALPHAS[0], -np.inf
        for a in ALPHAS:
            i0, i1 = inner == 0, inner == 1
            if i0.sum() < 5 or i1.sum() < 5:
                sc = 0.0
            else:
                w0 = np.linalg.solve(Xtr[i0].T.dot(Xtr[i0]) + a * np.eye(Xtr.shape[1]),
                                     Xtr[i0].T.dot(y[tr][i0] - y[tr][i0].mean()))
                r = y[tr][i1] - (Xtr[i1].dot(w0) + y[tr][i0].mean())
                sc = -r.dot(r)
            if sc > bestsc:
                best, bestsc = a, sc
        w = np.linalg.solve(G + best * np.eye(Xtr.shape[1]), b)
        pred[te] = Xte.dot(w) + ym
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def auc(y, score):
    """Area under the ROC curve, by the rank identity. y is 0/1."""
    y = np.asarray(y) > 0.5
    if y.all() or not y.any():
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ridge_cv_pred(R, y, n_folds=5, seed=0):
    """The cross-validated predictions themselves, for a binary target's AUC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fold = rng.permutation(n) % n_folds
    pred = np.full(n, np.nan)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if tr.sum() < 20 or te.sum() < 3:
            return pred
        mu, sd = R[tr].mean(axis=0), R[tr].std(axis=0) + 1e-9
        Xtr, Xte = (R[tr] - mu) / sd, (R[te] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T.dot(Xtr) + 100.0 * np.eye(Xtr.shape[1]),
                            Xtr.T.dot(y[tr] - ym))
        pred[te] = Xte.dot(w) + ym
    return pred


def decode_binary(R, y, n_perm=100, seed=0):
    """Cross-validated AUC for a 0/1 target, with a permutation null."""
    rng = np.random.default_rng(seed)
    obs = auc(y, ridge_cv_pred(R, y, seed=seed))
    if not np.isfinite(obs):
        return None
    null = []
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        null.append(auc(yp, ridge_cv_pred(R, yp, seed=seed)))
    null = np.array([v for v in null if np.isfinite(v)])
    return dict(auc=float(obs),
                null_median=float(np.median(null)) if len(null) else np.nan,
                p=float((1 + (null >= obs).sum()) / (1.0 + len(null))))


def decode_epoch(R, targets, n_perm=100, seed=0):
    """CV R^2 per target, against two nulls.

    A plain permutation destroys everything, including the slow drift that both
    firing rates and behaviour have across a two-hour session -- so it answers
    "better than nothing" and not "better than drift". The circular shift keeps the
    target's own autocorrelation and only breaks its trial-by-trial correspondence
    with the spikes, which is the null that a drift artefact survives. A read-out is
    only reported as real when it beats BOTH.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, y in targets.items():
        n = len(y)
        obs = ridge_cv_r2(R, y, seed=seed)
        if not np.isfinite(obs):
            continue
        null = np.array([ridge_cv_r2(R, y[rng.permutation(n)], seed=seed)
                         for _ in range(n_perm)])
        null = null[np.isfinite(null)]
        shifts = rng.integers(max(10, n // 10), n - max(10, n // 10),
                              size=min(n_perm, max(1, n - 1)))
        snull = np.array([ridge_cv_r2(R, np.roll(y, int(k)), seed=seed) for k in shifts])
        snull = snull[np.isfinite(snull)]
        out[name] = dict(
            r2=float(obs),
            null_median=float(np.median(null)) if len(null) else np.nan,
            p=float((1 + (null >= obs).sum()) / (1.0 + len(null))),
            shift_median=float(np.median(snull)) if len(snull) else np.nan,
            p_shift=float((1 + (snull >= obs).sum()) / (1.0 + len(snull))) if len(snull) else np.nan)
    return out


# ----------------------------------------------------------------------- the report

def behaviour_table(t, sid):
    """The animal's own confidence report, which is what licenses the whole analysis."""
    unrew = t[t.time_investment & t.wait.notna()].copy()
    if sid in NO_ERROR_CATCH:
        unrew = unrew[unrew.correct > 0.5]
    b = pd.cut(unrew.absDV, [0, .2, .4, .6, .8, 1.0001], right=False)
    return unrew.groupby([b, "correct"], observed=False).wait.agg(["median", "size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    ap.add_argument("--min_completed", type=int, default=0,
                    help="skip a session with fewer completed trials than this. The "
                         "two sessions that showed nothing in the first run were the "
                         "two shortest (172 and 318), so this is the knob for asking "
                         "the question again with only the sessions that can answer "
                         "it. Counted after the sync_valid and recording-gap filters, "
                         "which is the number the analysis actually spends.")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--allow_mixed_rats", action="store_true",
                    help="permit sessions from more than one animal in one pooled "
                         "run. Off by default: the per-unit counts are sums over "
                         "units, so one animal with an unusual session moves the "
                         "headline for all of them.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rats = sorted({RAT.get(s, "?") for s in args.sessions})
    if len(rats) > 1 and not args.allow_mixed_rats:
        sys.exit("refusing to pool %s in one run -- per-unit counts are summed over "
                 "units, so one animal can carry the headline (5bd.73). Run each rat "
                 "separately, or pass --allow_mixed_rats if you really mean it."
                 % " and ".join(rats))

    summaries, behav, decrows = [], [], []
    for sid in args.sessions:
        S = SESSIONS[sid]
        if not os.path.isdir(S):
            print("skip %s: no synced folder" % sid)
            continue
        if args.min_completed:
            n_done = len(build_trials(S, sid))
            if n_done < args.min_completed:
                print("skip %s: %d completed trials, under the %d asked for"
                      % (sid, n_done, args.min_completed))
                continue
        t, out, null, dec = analyse_session(sid, S)
        out.to_csv(os.path.join(args.out, "%s_units.csv" % sid), index=False)

        bt = behaviour_table(t, sid)
        print("    waiting time (s) on unrewarded trials, by |DV| and outcome:")
        print("      " + bt.round(2).to_string().replace("\n", "\n      "))
        behav.append(bt.assign(session=sid))

        print("    %-5s %-5s %-28s %-28s %s"
              % ("epoch", "n", "|DV| (Gil's assumption)", "P(correct|DV,choice)", "shuffle null"))
        for ep in EPOCHS:
            e = out[out.epoch == ep]
            if not len(e):
                continue
            row = {"session": sid, "epoch": ep, "n_units": len(e)}
            for key in ("absDV", "conf"):
                sig = bh_fdr(e[key + "_p"].values)
                pos = sig & (e[key + "_beta"].values > 0)
                row[key + "_n_sig"] = int(sig.sum())
                row[key + "_n_pos"] = int(pos.sum())
                row[key + "_frac"] = float(sig.mean())
            isig = bh_fdr(e.interaction_p.values)
            xpat = isig & (e.slope_correct.values > 0) & (e.slope_error.values < 0)
            xpat_inv = isig & (e.slope_correct.values < 0) & (e.slope_error.values > 0)
            row["n_xpattern"] = int(xpat.sum())
            row["n_xpattern_inverted"] = int(xpat_inv.sum())
            if "wait_p" in e and e.wait_p.notna().any():
                ww = e.dropna(subset=["wait_p"])
                row["n_wait_sig"] = int(bh_fdr(ww.wait_p.values).sum())
                row["n_wait_units"] = int(len(ww))
            if ep in null:
                row["null_mean"], row["null_p95"], _ = null[ep]
            summaries.append(row)
            print("    %-5s %-5d %-28s %-28s %s"
                  % (ep, len(e),
                     "%d sig (%d up)" % (row["absDV_n_sig"], row["absDV_n_pos"]),
                     "%d sig (%d up)" % (row["conf_n_sig"], row["conf_n_pos"]),
                     ("%.1f exp, p95 %.0f" % (row.get("null_mean", float("nan")),
                                              row.get("null_p95", float("nan")))
                      if ep in null else "-")))

        print("    population read-out (5-fold CV R^2, permutation p):")
        for ep in EPOCHS:
            if ep not in dec:
                continue
            d = dec[ep]
            bits = []
            for k, v in d.items():
                if isinstance(v, dict):
                    bits.append("%s R2=%.3f p=%.3f pshift=%.3f"
                                % (k, v["r2"], v["p"], v["p_shift"]))
            print("      %-5s %3d units %4d trials" % (ep, d["n_units"], d["n_trials"]))
            for bit in bits:
                print("            " + bit)
            for k, v in d.items():
                if isinstance(v, dict):
                    decrows.append(dict(session=sid, epoch=ep, target=k,
                                        n_units=d["n_units"], n_trials=d["n_trials"], **v))

    s = pd.DataFrame(summaries)
    pd.DataFrame(decrows).to_csv(os.path.join(args.out, "decoding.csv"), index=False)
    s.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.concat(behav).to_csv(os.path.join(args.out, "behaviour_waiting_time.csv"))
    print("\nwrote %s" % os.path.join(args.out, "summary.csv"))


if __name__ == "__main__":
    main()
