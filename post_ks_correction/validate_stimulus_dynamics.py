"""Validation of the stimulus-period decoding, three ways.

Gil's concern: the result is strong for a single session. Three independent
checks, two of which he asked for directly.

1. CROSS-SESSION REPLICATION is handled by running decode_stimulus_dynamics.py
   on 20260901_085606 and 20260911_100049 (both reprocessed from scratch,
   because their old `synced` folders predate the ephys-offset fix -- the Sep 1
   session was out by 25.79 s and the Sep 11 one by 28.8 s, which would have
   silently wrecked any alignment). Those sessions have no hand review, so they
   use KSLabel 'good' -- a weaker curation, which makes it a harder test.

2. SPLIT-HALF WITHIN A SESSION (here). Fit the whole analysis independently on
   the first and second half of the trials. A result that is a fluke of one
   subset of trials will not survive; one driven by slow drift across the
   session will differ between halves.

3. THE PRE-STIMULUS LEAK, and what causes it (here). The two validation
   sessions decode the rat's CHOICE above chance BEFORE the stimulus starts
   (AUC 0.60-0.63 and 0.54-0.60, p=0.010), where 20260916 is clean
   (0.48-0.52, p>=0.21). DV is at chance pre-stimulus in all three, so no
   stimulus information leaks -- but choice-predictive activity before the
   stimulus would contaminate any onset estimate for choice.

   The obvious biological candidate is CHOICE HISTORY: rats repeat and
   alternate, so if pre-stimulus activity carries the PREVIOUS trial's choice
   it will partly predict the current one for free. That is tested directly
   here by decoding the previous trial's choice from the same pre-stimulus
   window, and by asking whether current-choice decodability survives once
   trials are split by what the rat did last.

WHAT THIS CANNOT DO, and it is the analysis Gil actually wanted: compare trials
where evidence for the correct side arrives EARLY in the stimulus against those
where it arrives LATE. That needs the per-tone sequence, and `Custom.AudSound`
holds only a rolling buffer of unplayed look-ahead trials. No amount of
analysis recovers it; the protocol has to log tone times or the RNG seed.

Usage: python validate_stimulus_dynamics.py
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

OUT = r"D:\Gil\spike_sorting_agent\outputs"
SCR = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
       r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad")
SESSIONS = [
    ("20260916_110311",
     r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced_20261002",
     os.path.join(OUT, "manual_verdicts_20260916_110311.csv")),
    ("20260901_085606", os.path.join(SCR, "synced_20260901_085606"), "kslabel"),
    ("20260911_100049", os.path.join(SCR, "synced_20260911_100049"), "kslabel"),
]
WIN = 0.100
STIM_DUR = 0.35
PRE = (-0.150, -0.050)       # pre-stimulus window
EARLY = (0.050, 0.150)       # early stimulus
LATE = (0.250, 0.350)        # late stimulus
N_PERM = 200

LOGIT = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 12)))


def cvs(X, y, kind, model, reps=3):
    out = []
    for r in range(reps):
        cv = (KFold(5, shuffle=True, random_state=r) if kind == "reg"
              else StratifiedKFold(5, shuffle=True, random_state=r))
        try:
            out.append(np.mean(cross_val_score(model, X, y, cv=cv,
                                               scoring="r2" if kind == "reg" else "roc_auc")))
        except Exception:
            return np.nan
    return float(np.mean(out))


def perm(X, y, kind, model, n=N_PERM, seed=0):
    obs = cvs(X, y, kind, model)
    if not np.isfinite(obs):
        return obs, np.nan, np.nan
    r = np.random.default_rng(seed)
    null = np.array([cvs(X, r.permutation(y), kind, model, reps=1) for _ in range(n)])
    return obs, float(np.percentile(null, 95)), (1 + int((null >= obs).sum())) / (1 + n)


rows_half, rows_hist = [], []
for tag, sync, unit_src in SESSIONS:
    trials = pd.read_parquet(os.path.join(sync, "trials.parquet"))
    se = pd.read_parquet(os.path.join(sync, "state_events.parquet"))
    units = pd.read_parquet(os.path.join(sync, "units.parquet"))
    if unit_src == "kslabel":
        good = sorted(units.loc[units.quality_label == "good", "unit_id"].astype(int))
    else:
        v = pd.read_csv(unit_src)
        good = sorted(set(v.loc[v.verdict == "good", "unit"].astype(int))
                      & set(units.unit_id.astype(int)))

    s = se[se.state_name == "stimulus_delivery_min"].sort_values(
        ["trial_id", "occurrence"]).groupby("trial_id").first()
    clean = trials[trials.sync_valid & trials.TrialCompleted]
    ev = clean.set_index("trial_id")[["ChoiceLeft", "DV"]].copy()
    ev["stim_on"] = s.start_time_in_trial
    ev = ev[ev.stim_on.notna()].sort_index()

    # previous COMPLETED trial's choice, in trial order
    ev["prev_choice"] = ev.ChoiceLeft.shift(1)

    keep = []
    pf = pq.ParquetFile(os.path.join(sync, "spikes.parquet"))
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

    def design(sub, lo, hi):
        t0 = sub.stim_on.to_numpy(); ids = sub.index.to_numpy()
        X = np.zeros((len(ids), len(U)))
        for j, u in enumerate(U):
            iu = idx[u]
            for i, (tid, a) in enumerate(zip(ids, t0)):
                arr = iu.get(int(tid))
                if arr is None:
                    continue
                X[i, j] = np.searchsorted(arr, a + hi, "left") - \
                    np.searchsorted(arr, a + lo, "left")
        return X

    print(f"\n{'='*94}\n{tag}  —  {len(ev)} trials, {len(U)} units\n{'='*94}")

    # ---------------- 2. SPLIT-HALF ----------------
    n = len(ev)
    halves = {"first half": ev.iloc[:n // 2], "second half": ev.iloc[n // 2:]}
    print("\nSPLIT-HALF (independent fits; a fluke or a drift artefact should differ)")
    print(f"{'window':<16}{'half':<14}{'n':>5}{'choice AUC':>12}{'p':>8}"
          f"{'DV r2':>9}{'p':>8}")
    for wname, (lo, hi) in [("pre-stimulus", PRE), ("early stim", EARLY),
                            ("late stim", LATE)]:
        for hname, sub in halves.items():
            X = design(sub, lo, hi)
            yc = (sub.ChoiceLeft == 1).astype(int).to_numpy()
            a, _, pa = perm(X, yc, "clf", LOGIT, seed=1)
            r2, _, pr = perm(X, sub.DV.to_numpy(), "reg", RIDGE, seed=2)
            rows_half.append(dict(session=tag, window=wname, half=hname,
                                  n=len(sub), choice_auc=a, choice_p=pa,
                                  dv_r2=r2, dv_p=pr))
            print(f"{wname:<16}{hname:<14}{len(sub):>5}{a:>12.3f}{pa:>8.3f}"
                  f"{r2:>+9.3f}{pr:>8.3f}", flush=True)

    # ---------------- 3. THE PRE-STIMULUS LEAK ----------------
    print("\nPRE-STIMULUS LEAK: is it choice HISTORY?")
    sub = ev[ev.prev_choice.notna()]
    Xp = design(sub, *PRE)
    yc = (sub.ChoiceLeft == 1).astype(int).to_numpy()
    yp = (sub.prev_choice == 1).astype(int).to_numpy()
    a_cur, _, p_cur = perm(Xp, yc, "clf", LOGIT, seed=3)
    a_prev, _, p_prev = perm(Xp, yp, "clf", LOGIT, seed=4)
    rep = float((yc == yp).mean())
    # does current-choice decoding survive INSIDE each previous-choice group?
    within = []
    for pv in (0, 1):
        m = yp == pv
        if m.sum() >= 60 and len(np.unique(yc[m])) == 2:
            within.append(cvs(Xp[m], yc[m], "clf", LOGIT))
    w = float(np.nanmean(within)) if within else np.nan
    rows_hist.append(dict(session=tag, n=len(sub), auc_current=a_cur, p_current=p_cur,
                          auc_previous=a_prev, p_previous=p_prev,
                          repeat_rate=rep, auc_current_within_prev=w))
    print(f"  pre-stimulus window {PRE[0]:+.3f}..{PRE[1]:+.3f}s, n={len(sub)}")
    print(f"    decode CURRENT choice           AUC {a_cur:.3f}  p={p_cur:.3f}")
    print(f"    decode PREVIOUS trial's choice  AUC {a_prev:.3f}  p={p_prev:.3f}")
    print(f"    rat repeated its last choice on {100*rep:.1f}% of trials")
    print(f"    CURRENT choice, within previous-choice groups  AUC {w:.3f}"
          f"   <- if this falls to ~0.5 the leak was history")

pd.DataFrame(rows_half).to_csv(os.path.join(OUT, "validation_split_half.csv"), index=False)
pd.DataFrame(rows_hist).to_csv(os.path.join(OUT, "validation_prestim_history.csv"), index=False)
print("\nsaved validation_split_half.csv, validation_prestim_history.csv")
