"""Can the population predict WHEN the rat will give up waiting?

Gil's questions, both addressed here:

(A) THE DV/CHOICE CONFOUND, settled cleanly. 5ao showed that a DV decoder is
    really a choice decoder, using an error-trial generalisation test. The
    cleanest possible version of that test is run here instead: decode DV
    WITHIN a single choice. Among left-choice trials only, the rat's choice is
    constant by construction, so ANY decodable DV variance must be stimulus or
    difficulty and cannot be motor. Same for right-choice trials. If DV
    decoding collapses to zero within choice, the earlier conclusion is exact,
    not merely suggestive.

(B) LEAVING TIME. On trials where no water ever comes, how long the rat holds
    on is its confidence report. Can that be read from the population, and does
    it sharpen as the moment approaches?

    ERROR and OMISSION trials are kept SEPARATE throughout, per Gil: they are
    different internal states (the rat was wrong vs the rat was right and is
    being probed), even though both end the same way. Rewarded trials are
    excluded entirely -- there the ending is imposed by the valve, not chosen
    by the rat, so there is no leaving decision to predict. The 49 correct
    trials the rat abandoned after a median 0.83 s are also excluded: they are
    not confidence probes, they are trials it barely engaged with.

THE CONFOUND THAT WOULD OTHERWISE FAKE THIS ENTIRE RESULT. If a window at, say,
3 s after the choice poke is applied to every trial, then on a trial where the
rat left at 2 s the window is measuring activity AFTER it has already gone --
different port, different posture, different everything. A decoder would then
"predict" waiting time by detecting whether the rat is still there, which is
circular. So every window here includes ONLY trials still waiting when the
window closes, and the surviving n is printed for every row. This is also why
Gil's instinct to use long-wait trials is the right one: it is the only regime
where the question is well posed.

Two measurements:
  1. PREDICT REMAINING WAIT from a window at increasing lag after the choice
     poke. Ridge, cross-validated R2, against a permutation null. If the signal
     accumulates, R2 should grow with lag.
  2. IS THE LEAVE IMMINENT? A classifier over time points sampled throughout
     the wait: from a 300 ms window, will the rat leave within the next 500 ms?
     Cross-validated BY TRIAL (GroupKFold) so no trial contributes to both
     train and test, with a null built by shuffling which trial got which leave
     time.

Usage: python decode_leaving_time.py
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNC = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
        r"\synced_20261002")
VERD = r"D:\Gil\spike_sorting_agent\outputs\manual_verdicts_20260916_110311.csv"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FIG = os.path.join(OUT, "reward_cue_figs")
N_PERM = 300
MARGIN = 0.05          # a trial must still be waiting this long past window end

trials = pd.read_parquet(os.path.join(SYNC, "trials.parquet"))
se = pd.read_parquet(os.path.join(SYNC, "state_events.parquet"))
units = pd.read_parquet(os.path.join(SYNC, "units.parquet"))
ver = pd.read_csv(VERD)
good = sorted(set(ver.loc[ver.verdict == "good", "unit"].astype(int))
              & set(units.unit_id.astype(int)))
clean = trials[trials.sync_valid & trials.TrialCompleted]


def fs(names, f):
    s = se[se.state_name.isin(names)].sort_values(["trial_id", "occurrence"])
    return s.groupby("trial_id").first()[f]


ev = clean.set_index("trial_id")[
    ["ChoiceLeft", "ChoiceCorrect", "Rewarded", "CatchTrial", "DV"]].copy()
ev["stim_on"] = fs(["stimulus_delivery_min"], "start_time_in_trial")
ev["cue_end"] = fs(["stimulus_delivery"], "end_time_in_trial")
ev["poke"] = fs(["start_Lin", "start_Rin"], "start_time_in_trial")
ev["giveup"] = fs(["skipped_feedback"], "start_time_in_trial")
ev = ev[ev.stim_on.notna() & ev.cue_end.notna() & ev.poke.notna()]
ev["wait"] = ev.giveup - ev.poke
ev["grp"] = np.where(ev.ChoiceCorrect == 0, "error",
                     np.where(ev.Rewarded == 1, "rewarded",
                              np.where(ev.CatchTrial == 1, "omission_catch",
                                       "abandoned")))

print("loading spikes...")
keep = []
pf = pq.ParquetFile(os.path.join(SYNC, "spikes.parquet"))
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=["unit_id", "trial_id",
                                      "spike_time_in_trial"]).to_pandas()
    keep.append(t[t.unit_id.isin(set(good))])
sp = pd.concat(keep, ignore_index=True).sort_values(
    ["unit_id", "trial_id", "spike_time_in_trial"])
idx = {int(u): {int(t): gg.to_numpy()
                for t, gg in g.groupby("trial_id", sort=False)["spike_time_in_trial"]}
       for u, g in sp.groupby("unit_id", sort=False)}
del sp, keep
U = [u for u in good if u in idx]
print(f"  {len(U)} units, {len(ev)} trials")

RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 15)))


def counts(ids, t0, lo, hi):
    X = np.zeros((len(ids), len(U)))
    for j, u in enumerate(U):
        iu = idx[u]
        for i, (tid, a) in enumerate(zip(ids, t0)):
            s = iu.get(int(tid))
            if s is None:
                continue
            X[i, j] = np.searchsorted(s, a + hi, "left") - \
                np.searchsorted(s, a + lo, "left")
    return X


def cv_r2(X, y, reps=5):
    return float(np.mean([np.mean(cross_val_score(
        RIDGE, X, y, cv=KFold(5, shuffle=True, random_state=r), scoring="r2"))
        for r in range(reps)]))


def perm_r2(X, y, n=N_PERM, seed=0):
    obs = cv_r2(X, y)
    r = np.random.default_rng(seed)
    null = np.array([cv_r2(X, r.permutation(y), reps=1) for _ in range(n)])
    return obs, float(null.mean()), float(np.percentile(null, 95)), \
        (1 + int((null >= obs).sum())) / (1 + n)


# ====================================================================== (A)
print("\n" + "=" * 96)
print("(A) DV DECODING *WITHIN* A SINGLE CHOICE -- the clean version of the")
print("    confound test. Choice is constant inside each group, so anything")
print("    decodable here is stimulus/difficulty and cannot be motor.")
print("=" * 96)
EPOCHS = [("stimulus", "stim_on", 0.00, 0.35),
          ("cue end -> choice", "cue_end", 0.00, 0.25),
          ("after choice (wait)", "poke", 0.20, 0.70)]
rowsA = []
for name, col, lo, hi in EPOCHS:
    print(f"\n### {name}")
    for side, mask in [("ALL trials", np.ones(len(ev), bool)),
                       ("LEFT choice only", (ev.ChoiceLeft == 1).to_numpy()),
                       ("RIGHT choice only", (ev.ChoiceLeft == 0).to_numpy())]:
        sub = ev[mask]
        X = counts(sub.index.to_numpy(), sub[col].to_numpy(), lo, hi)
        for tgt, y in [("DV", sub.DV.to_numpy()),
                       ("|DV|", np.abs(sub.DV.to_numpy()))]:
            o, c, p95, p = perm_r2(X, y, 200, seed=hash((name, side, tgt)) % 1000)
            rowsA.append(dict(epoch=name, subset=side, target=tgt, n=len(sub),
                              r2=o, chance=c, chance_p95=p95, p=p))
            print(f"  {side:<19}{tgt:<5} n={len(sub):<4} r2={o:+.3f}  "
                  f"chance {c:+.3f} (95th {p95:+.3f})  p={p:.4f}")
pd.DataFrame(rowsA).to_csv(os.path.join(OUT, "dv_within_choice.csv"), index=False)

# ====================================================================== (B1)
print("\n" + "=" * 96)
print("(B1) PREDICTING HOW MUCH LONGER THE RAT WILL STAY")
print("     window at increasing lag after the choice poke; only trials still")
print("     waiting when the window closes are included")
print("=" * 96)
LAGS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
        (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]
rowsB = []
for grp in ["omission_catch", "error"]:
    g0 = ev[ev.grp == grp]
    print(f"\n### {grp}  (n={len(g0)} total)")
    print(f"{'window (s)':>14}{'n':>6}{'SD remaining':>14}{'R2':>9}"
          f"{'chance95':>11}{'p':>8}")
    for lo, hi in LAGS:
        sub = g0[g0.wait > hi + MARGIN]
        if len(sub) < 25:
            print(f"{lo:>6.1f}-{hi:<7.1f}{len(sub):>6}   too few trials, skipped")
            continue
        X = counts(sub.index.to_numpy(), sub.poke.to_numpy(), lo, hi)
        y = (sub.wait - hi).to_numpy()          # time still to go
        o, c, p95, p = perm_r2(X, y, N_PERM, seed=int(lo * 10) + (grp == "error") * 50)
        rowsB.append(dict(group=grp, win_lo=lo, win_hi=hi, n=len(sub),
                          sd_remaining=float(y.std()), r2=o, chance=c,
                          chance_p95=p95, p=p))
        print(f"{lo:>6.1f}-{hi:<7.1f}{len(sub):>6}{y.std():>14.2f}{o:>+9.3f}"
              f"{p95:>+11.3f}{p:>8.4f}")
pd.DataFrame(rowsB).to_csv(os.path.join(OUT, "leaving_time_decoding.csv"),
                           index=False)

# ====================================================================== (B2)
print("\n" + "=" * 96)
print("(B2) IS THE LEAVE IMMINENT?  from a 300 ms window, will the rat go")
print("     within the next 500 ms?  time points sampled every 250 ms through")
print("     the wait; cross-validated BY TRIAL; null shuffles leave times")
print("=" * 96)
WINW, STEPW, HORIZON = 0.300, 0.250, 0.500
rowsC = []
for grp in ["omission_catch", "error"]:
    g0 = ev[(ev.grp == grp) & (ev.wait > 1.5)]
    rows_t, labels, groups = [], [], []
    for tid, r in g0.iterrows():
        t = WINW
        while t <= r.wait - 0.05:
            rows_t.append((tid, r.poke + t - WINW, r.poke + t))
            labels.append(1 if (r.wait - t) <= HORIZON else 0)
            groups.append(tid)
            t += STEPW
    if len(rows_t) < 100 or len(set(labels)) < 2:
        print(f"  {grp}: not enough sampled points")
        continue
    ids = np.array([a for a, _, _ in rows_t])
    X = np.zeros((len(rows_t), len(U)))
    for j, u in enumerate(U):
        iu = idx[u]
        for i, (tid, a, b) in enumerate(rows_t):
            s = iu.get(int(tid))
            if s is None:
                continue
            X[i, j] = np.searchsorted(s, b, "left") - np.searchsorted(s, a, "left")
    y = np.array(labels)
    groups = np.array(groups)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))

    def auc_grouped(yy):
        return float(np.mean(cross_val_score(clf, X, yy, cv=GroupKFold(5),
                                             groups=groups, scoring="roc_auc")))

    obs = auc_grouped(y)
    # null: reassign leave times among trials of this group, relabel
    waits = g0.wait.to_dict()
    tids_list = list(waits)
    r = np.random.default_rng(3)
    null = []
    for _ in range(200):
        perm = dict(zip(tids_list, r.permutation([waits[t] for t in tids_list])))
        yy = []
        ok = True
        for (tid, a, b) in rows_t:
            el = b - ev.loc[tid, "poke"]
            yy.append(1 if (perm[tid] - el) <= HORIZON else 0)
        yy = np.array(yy)
        if len(set(yy)) < 2:
            continue
        null.append(auc_grouped(yy))
    null = np.array(null)
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    rowsC.append(dict(group=grp, n_timepoints=len(y), n_trials=len(g0),
                      frac_positive=float(y.mean()), auc=obs,
                      chance=float(null.mean()),
                      chance_p95=float(np.percentile(null, 95)), p=p))
    print(f"  {grp:<16} {len(g0)} trials, {len(y)} time points "
          f"({100*y.mean():.0f}% within {HORIZON:.1f}s of leaving)")
    print(f"  {'':<16} AUC = {obs:.3f}   chance {null.mean():.3f} "
          f"(95th {np.percentile(null,95):.3f})   p = {p:.4f}")
pd.DataFrame(rowsC).to_csv(os.path.join(OUT, "leaving_imminent.csv"), index=False)

# ------------------------------------------------------------------ figure
if rowsB:
    b = pd.DataFrame(rowsB)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cols = {"omission_catch": "#2e6f8c", "error": "#8c3b2e"}
    nice = {"omission_catch": "correct, no water (catch)", "error": "error"}
    for grp, c in cols.items():
        g = b[b.group == grp]
        if not len(g):
            continue
        mid = (g.win_lo + g.win_hi) / 2
        ax.plot(mid, g.r2, "o-", color=c, lw=2, ms=5, label=nice[grp])
        ax.plot(mid, g.chance_p95, color=c, lw=1, ls=":", alpha=.7)
        for _, r in g.iterrows():
            ax.annotate(f"n={int(r.n)}", ((r.win_lo + r.win_hi) / 2, r.r2),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color=c)
    ax.axhline(0, color="#999", lw=0.9)
    ax.set_xlabel("centre of 500 ms window, from the choice poke (s)", fontsize=9.5)
    ax.set_ylabel("cross-validated $R^2$ for remaining wait", fontsize=9.5)
    ax.set_title("Can the population predict how much longer the rat will hold on?\n"
                 "only trials still waiting when the window closes; "
                 "dotted = 95th percentile of permutation null",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "leaving_time_decoding.png")
    fig.savefig(p, dpi=125, bbox_inches="tight")
    print(f"\nsaved {p}")

print("\nsaved dv_within_choice.csv, leaving_time_decoding.csv, leaving_imminent.csv")
