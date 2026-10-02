"""Can the decision variable be decoded from the population, and when?

Session 20260916_110311, the 111 units Gil labelled `good` in the review tool,
478 completed sync-valid trials.

DV is the signed decision variable: sign = the objectively correct side,
magnitude = how easy the trial is. It is continuous here (478 distinct values
spanning -1.00 to +1.00), so it supports regression, not just classification.

THE TRIAL TAXONOMY, which this script keeps separate throughout because the
three groups answer different questions:

  ERROR              71 trials -- the rat chose the wrong side.
  CORRECT + REWARDED 291 trials -- chose right, water arrived.
  CORRECT + OMISSION 116 trials -- chose right, no water (67 of them catch
                     trials, 49 ordinary trials the rat abandoned early).
                     The rat cannot tell these from a rewarded trial until the
                     water fails to come, so how long it waits is a readout of
                     its confidence -- which this script measures as a sanity
                     check on the behaviour before touching the neural data.

THE CONFOUND THAT DECIDES WHETHER ANY OF THIS MEANS ANYTHING. sign(DV) is the
correct side, and the rat is correct on 407 of 478 trials, so sign(DV) and the
rat's CHOICE agree 85% of the time. A decoder that reads sign(DV) well after
the choice may simply be reading the movement. Three things separate them:

  1. |DV| (difficulty) is independent of which side was correct, so decoding it
     cannot be a motor readout.
  2. Choice itself is decoded alongside, so the two time courses can be
     compared directly.
  3. THE DECISIVE TEST: a sign(DV) decoder trained on correct trials only is
     evaluated on ERROR trials, where choice and correct side DISAGREE. Above
     chance there means the population carries the stimulus; at-or-below chance
     means it was carrying the choice all along.

METHODS. Ridge, linear SVR and random forest for continuous DV; logistic
regression, LDA and linear SVM for the binary targets. Each inside a pipeline
that z-scores on training folds only. Scored by 5-fold CV repeated 5 times.

EVERY NUMBER CARRIES ITS OWN CHANCE LEVEL, measured by permutation (the target
is shuffled across trials and the whole CV is redone), never assumed. This
project has walked into a base-rate trap four times; cross-validated R2 on 111
features and 478 trials is exactly the kind of statistic that looks impressive
at chance.

Usage: python decode_dv.py
"""
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNC = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
        r"\synced_20261002")
VERDICTS = r"D:\Gil\spike_sorting_agent\outputs\manual_verdicts_20260916_110311.csv"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FIGDIR = os.path.join(OUT, "reward_cue_figs")
os.makedirs(FIGDIR, exist_ok=True)

WIN = 0.200            # sliding window width, s
STEP = 0.050
N_PERM_COURSE = 200    # permutations per sliding window
N_PERM_EPOCH = 1000    # permutations at the named epochs
N_REPEATS = 5
N_FOLDS = 5
SEED = 0
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ load
trials = pd.read_parquet(os.path.join(SYNC, "trials.parquet"))
se = pd.read_parquet(os.path.join(SYNC, "state_events.parquet"))
units = pd.read_parquet(os.path.join(SYNC, "units.parquet"))
ver = pd.read_csv(VERDICTS)
good_units = sorted(set(ver.loc[ver.verdict == "good", "unit"].astype(int))
                    & set(units.unit_id.astype(int)))

clean = trials[trials.sync_valid & trials.TrialCompleted].copy()


def first_state(names, field):
    s = se[se.state_name.isin(names)].sort_values(["trial_id", "occurrence"])
    return s.groupby("trial_id").first()[field]


ev = clean.set_index("trial_id")[
    ["ChoiceLeft", "ChoiceCorrect", "Rewarded", "CatchTrial", "DV", "MT"]].copy()
ev["stim_on"] = first_state(["stimulus_delivery_min"], "start_time_in_trial")
ev["cue_end"] = first_state(["stimulus_delivery"], "end_time_in_trial")
ev["choice_poke"] = first_state(["start_Lin", "start_Rin"], "start_time_in_trial")
ev["reward_time"] = first_state(["water_L", "water_R"], "start_time_in_trial")
ev["giveup"] = first_state(["skipped_feedback"], "start_time_in_trial")

ev["group"] = np.where(ev.ChoiceCorrect == 0, "error",
                       np.where(ev.Rewarded == 1, "correct_rewarded",
                                "correct_omission"))
ev = ev[ev.stim_on.notna() & ev.cue_end.notna() & ev.choice_poke.notna()].copy()

print("=" * 90)
print("TRIAL TAXONOMY (completed, sync-valid)")
print("=" * 90)
print(ev.group.value_counts().to_string())
print(f"  of the omission trials, {int(ev[ev.group=='correct_omission'].CatchTrial.sum())} "
      f"are catch trials and "
      f"{int((ev[ev.group=='correct_omission'].CatchTrial==0).sum())} are ordinary "
      f"trials the rat abandoned early")

# ---------------- confidence: how long does the rat wait when no water comes?
w = ev[ev.giveup.notna()].copy()
w["wait_s"] = w.giveup - w.choice_poke
print("\n" + "=" * 90)
print("CONFIDENCE READOUT: waiting time before giving up (no water arrived)")
print("=" * 90)
for grp in ["correct_omission", "error"]:
    g = w[w.group == grp]
    if not len(g):
        continue
    r, p = spearmanr(g.DV.abs(), g.wait_s)
    print(f"  {grp:<18} n={len(g):<4} median wait {g.wait_s.median():.2f}s   "
          f"wait vs |DV|: rho={r:+.3f}, p={p:.4f}")
gc, ge = w[w.group == "correct_omission"], w[w.group == "error"]
if len(gc) and len(ge):
    from scipy.stats import mannwhitneyu
    pu = mannwhitneyu(gc.wait_s, ge.wait_s, alternative="two-sided").pvalue
    print(f"  correct-omission waits {gc.wait_s.median():.2f}s vs error "
          f"{ge.wait_s.median():.2f}s  (Mann-Whitney p={pu:.4f})")
    print("  [a longer wait on correct trials, and a wait that grows with |DV|,")
    print("   is the signature of a confidence report]")

# ------------------------------------------------------------- spikes
print("\nloading spikes...")
want = set(good_units)
keep = []
pf = pq.ParquetFile(os.path.join(SYNC, "spikes.parquet"))
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=["unit_id", "trial_id",
                                      "spike_time_in_trial"]).to_pandas()
    keep.append(t[t.unit_id.isin(want)])
sp = pd.concat(keep, ignore_index=True)
del keep
sp = sp.sort_values(["unit_id", "trial_id", "spike_time_in_trial"])
idx = {}
for u, g in sp.groupby("unit_id", sort=False):
    idx[int(u)] = {int(t): gg.to_numpy()
                   for t, gg in g.groupby("trial_id", sort=False)["spike_time_in_trial"]}
del sp
units_used = [u for u in good_units if u in idx]
print(f"  {len(units_used)} units, {len(ev)} trials")

tids = ev.index.to_numpy()


def design(align_col, lo, hi, rows=None):
    """X[trial, unit] = spike count in [event+lo, event+hi)."""
    r = ev if rows is None else ev.loc[rows]
    t0 = r[align_col].to_numpy()
    ids = r.index.to_numpy()
    X = np.zeros((len(ids), len(units_used)))
    for j, u in enumerate(units_used):
        iu = idx[u]
        for i, (tid, a) in enumerate(zip(ids, t0)):
            s = iu.get(int(tid))
            if s is None:
                continue
            X[i, j] = np.searchsorted(s, a + hi, "left") - \
                np.searchsorted(s, a + lo, "left")
    return X


def cv_score(X, y, kind, model, n_repeats=N_REPEATS):
    scores = []
    for rep in range(n_repeats):
        if kind == "reg":
            cv = KFold(N_FOLDS, shuffle=True, random_state=rep)
            sc = cross_val_score(model, X, y, cv=cv, scoring="r2")
        else:
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)
            sc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        scores.append(np.mean(sc))
    return float(np.mean(scores))


def perm_test(X, y, kind, model, n_perm, seed=0):
    obs = cv_score(X, y, kind, model, n_repeats=2)
    r = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = cv_score(X, r.permutation(y), kind, model, n_repeats=1)
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    return obs, float(null.mean()), float(np.percentile(null, 95)), p


RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 12)))
LOGIT = make_pipeline(StandardScaler(),
                      LogisticRegression(C=0.05, max_iter=3000))

# --------------------------------------------------- time course
print("\n" + "=" * 90)
print("TIME COURSE: cross-validated decoding in 200 ms windows, 50 ms steps")
print("every window carries its own permutation chance level (n=200)")
print("=" * 90)

y_dv = ev.DV.to_numpy()
y_absdv = np.abs(y_dv)
y_side = (ev.DV > 0).astype(int).to_numpy()        # correct side
y_choice = (ev.ChoiceLeft == 1).astype(int).to_numpy()

ALIGN = [("stim_on", -0.6, 1.2, "stimulus onset"),
         ("choice_poke", -0.6, 1.2, "choice poke")]
course = []
for col, t_lo, t_hi, label in ALIGN:
    starts = np.arange(t_lo, t_hi - WIN + 1e-9, STEP)
    print(f"\n--- aligned to {label} ---")
    print(f"{'win (s)':>12}{'DV r2':>10}{'chance':>9}{'p':>8}"
          f"{'|DV| r2':>10}{'p':>8}{'side AUC':>10}{'p':>8}{'choice AUC':>12}")
    for s0 in starts:
        X = design(col, s0, s0 + WIN)
        r_dv, c_dv, _, p_dv = perm_test(X, y_dv, "reg", RIDGE, N_PERM_COURSE, 1)
        r_ab, c_ab, _, p_ab = perm_test(X, y_absdv, "reg", RIDGE, N_PERM_COURSE, 2)
        a_sd, c_sd, _, p_sd = perm_test(X, y_side, "clf", LOGIT, N_PERM_COURSE, 3)
        a_ch = cv_score(X, y_choice, "clf", LOGIT, n_repeats=2)
        course.append(dict(align=col, t_start=s0, t_mid=s0 + WIN / 2,
                           dv_r2=r_dv, dv_chance=c_dv, dv_p=p_dv,
                           absdv_r2=r_ab, absdv_chance=c_ab, absdv_p=p_ab,
                           side_auc=a_sd, side_chance=c_sd, side_p=p_sd,
                           choice_auc=a_ch))
        print(f"{s0:+.2f}..{s0+WIN:+.2f}{r_dv:>10.3f}{c_dv:>9.3f}{p_dv:>8.3f}"
              f"{r_ab:>10.3f}{p_ab:>8.3f}{a_sd:>10.3f}{p_sd:>8.3f}{a_ch:>12.3f}")

cdf = pd.DataFrame(course)
cdf.to_csv(os.path.join(OUT, "dv_decoding_timecourse.csv"), index=False)

# --------------------------------------------------- named epochs, all methods
EPOCHS = [
    ("pre-stimulus", "stim_on", -0.30, 0.00),
    ("stimulus", "stim_on", 0.00, 0.35),
    ("cue end -> choice", "cue_end", 0.00, 0.25),
    ("at choice poke", "choice_poke", -0.10, 0.20),
    ("after choice (wait)", "choice_poke", 0.20, 0.70),
    ("late wait", "choice_poke", 0.70, 1.20),
]
REG = [("ridge", RIDGE),
       ("linear SVR", make_pipeline(StandardScaler(), LinearSVR(C=0.01, max_iter=20000))),
       ("random forest", make_pipeline(StandardScaler(),
                                       RandomForestRegressor(n_estimators=200,
                                                             min_samples_leaf=5,
                                                             random_state=0, n_jobs=-1)))]
CLF = [("logistic", LOGIT),
       ("LDA", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr",
                                                                         shrinkage="auto"))),
       ("linear SVM", make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=20000)))]

print("\n" + "=" * 90)
print("NAMED EPOCHS, SEVERAL METHODS (cross-validated; chance by permutation)")
print("=" * 90)
rows = []
for name, col, lo, hi in EPOCHS:
    X = design(col, lo, hi)
    print(f"\n### {name}   [{lo:+.2f},{hi:+.2f}] s from {col}")
    for mname, m in REG:
        obs, ch, p95, p = perm_test(X, y_dv, "reg", m, N_PERM_EPOCH // 4, 11)
        rows.append(dict(epoch=name, target="DV", method=mname, score=obs,
                         chance=ch, chance_p95=p95, p=p))
        print(f"  DV    {mname:<15} r2 = {obs:+.3f}   chance {ch:+.3f} "
              f"(95th {p95:+.3f})   p={p:.4f}")
    for mname, m in CLF:
        if mname == "linear SVM":
            continue   # no predict_proba -> roc_auc scorer unavailable
        obs, ch, p95, p = perm_test(X, y_side, "clf", m, N_PERM_EPOCH // 4, 12)
        rows.append(dict(epoch=name, target="sign(DV)", method=mname, score=obs,
                         chance=ch, chance_p95=p95, p=p))
        print(f"  side  {mname:<15} AUC= {obs:.3f}    chance {ch:.3f} "
              f"(95th {p95:.3f})    p={p:.4f}")
    obs, ch, p95, p = perm_test(X, y_absdv, "reg", RIDGE, N_PERM_EPOCH // 4, 13)
    rows.append(dict(epoch=name, target="|DV|", method="ridge", score=obs,
                     chance=ch, chance_p95=p95, p=p))
    print(f"  |DV|  {'ridge':<15} r2 = {obs:+.3f}   chance {ch:+.3f}   p={p:.4f}")
    obs = cv_score(X, y_choice, "clf", LOGIT)
    rows.append(dict(epoch=name, target="choice", method="logistic", score=obs,
                     chance=np.nan, chance_p95=np.nan, p=np.nan))
    print(f"  choice{'logistic':<15} AUC= {obs:.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "dv_decoding_epochs.csv"), index=False)

# ------------------------------- THE DECISIVE TEST: generalise to error trials
print("\n" + "=" * 90)
print("STIMULUS OR MOVEMENT? train sign(DV) on CORRECT trials, test on ERRORS")
print("on error trials the rat's choice and the correct side DISAGREE, so a")
print("decoder reading the stimulus scores above 0.5 and one reading the")
print("movement scores below it")
print("=" * 90)
is_err = (ev.group == "error").to_numpy()
print(f"  {int((~is_err).sum())} correct trials to train on, "
      f"{int(is_err.sum())} error trials to test on")
print(f"{'epoch':<22}{'AUC on errors':>15}{'chance (perm 95%)':>20}{'p':>8}"
      f"{'AUC on held-out correct':>26}")
gen = []
for name, col, lo, hi in EPOCHS:
    X = design(col, lo, hi)
    Xc, yc = X[~is_err], y_side[~is_err]
    Xe, ye = X[is_err], y_side[is_err]
    if len(np.unique(ye)) < 2:
        continue
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    m.fit(Xc, yc)
    auc_err = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    auc_cor = cv_score(Xc, yc, "clf", LOGIT, n_repeats=3)
    r = np.random.default_rng(7)
    null = np.empty(400)
    for k in range(400):
        mm = make_pipeline(StandardScaler(),
                           LogisticRegression(C=0.05, max_iter=3000))
        mm.fit(Xc, r.permutation(yc))
        null[k] = roc_auc_score(ye, mm.predict_proba(Xe)[:, 1])
    p = (1 + int((null >= auc_err).sum())) / 401
    gen.append(dict(epoch=name, auc_error=auc_err, chance=float(null.mean()),
                    chance_p95=float(np.percentile(null, 95)), p=p,
                    auc_correct_cv=auc_cor))
    print(f"{name:<22}{auc_err:>15.3f}{np.percentile(null,95):>20.3f}"
          f"{p:>8.4f}{auc_cor:>26.3f}")
pd.DataFrame(gen).to_csv(os.path.join(OUT, "dv_decoding_error_generalisation.csv"),
                         index=False)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (col, _, _, label) in zip(axes, ALIGN):
    c = cdf[cdf.align == col]
    ax.axhline(0, color="#999", lw=0.8)
    ax.plot(c.t_mid, c.dv_r2, color="#8c3b2e", lw=2, label="DV (ridge $R^2$)")
    ax.plot(c.t_mid, c.dv_chance, color="#8c3b2e", lw=1, ls=":", label="DV chance")
    ax.plot(c.t_mid, c.absdv_r2, color="#2e6f8c", lw=2, label="|DV| difficulty ($R^2$)")
    ax.plot(c.t_mid, (c.side_auc - 0.5) * 2, color="#3f6b4a", lw=2,
            label="correct side (2·(AUC−0.5))")
    ax.plot(c.t_mid, (c.choice_auc - 0.5) * 2, color="#9a9288", lw=1.6, ls="--",
            label="rat's choice (2·(AUC−0.5))")
    ax.axvline(0, color="#333", lw=1.2)
    if col == "stim_on":
        ax.axvspan(0, 0.35, color="#c8a", alpha=0.12, lw=0)
        ax.text(0.175, ax.get_ylim()[1] * 0.95, "stimulus", ha="center",
                fontsize=8, color="#666")
    ax.set_xlabel(f"time from {label} (s)", fontsize=9)
    ax.set_title(f"aligned to {label}", fontsize=10)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("cross-validated decoding performance", fontsize=9)
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Decoding the decision variable from 111 good units, session 20260916_110311\n"
             "478 completed trials · 200 ms windows · dotted = permutation chance",
             fontsize=11.5, fontweight="bold")
fig.tight_layout()
p = os.path.join(FIGDIR, "dv_decoding_timecourse.png")
fig.savefig(p, dpi=125, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {p}")
print("saved dv_decoding_timecourse.csv, dv_decoding_epochs.csv, "
      "dv_decoding_error_generalisation.csv")
