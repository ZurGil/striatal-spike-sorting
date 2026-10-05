"""How do choice and DV signals evolve DURING the 350 ms stimulus?

Gil asked two things. The second one -- does the neural signal track the
individual evidence events as they arrive -- cannot be answered on this
session, and it is worth being exact about why.

WHAT THE STIMULUS ACTUALLY IS. This is NOT the click-train variant
(`ClickTask = 0`, `AuditoryTrial = 1`, `TaskType = 3`). It is the tone-cloud
variant: 30 ms tones drawn from 18 frequencies spanning 200 Hz - 20 kHz
(`Aud_ToneDuration = 0.03`, `Aud_nFreq = 18`), overlapping by 2/3
(`Aud_ToneOverlap = 0.6667`), so a new tone starts every ~10 ms and the 350 ms
stimulus carries roughly 35 of them. Each tone is above or below
`CategoryBoundary` and is therefore evidence for one side, exactly analogous to
a left or right click. So the question is the right one to ask of this task.

WHY IT CANNOT BE ANSWERED HERE. The realized sequence was not saved. The Bpod
file's `Custom.AudSound` has 957 entries and only FOUR are non-empty -- indices
953-956, which are the pre-generated look-ahead trials that were never played
(the session ran 953). It is a rolling buffer for upcoming trials, not a record
of what was delivered. Nothing else in the file carries per-tone identity or
timing: `AudFracHigh` is a single 2-element array, and `DV` is one number per
trial. So for every analysed trial we know the NET evidence and not the stream
that produced it.

To do the per-tone analysis on future sessions, the protocol needs to log, per
trial, either the tone frequencies and their onset times or the random seed
that generated them. That is a protocol change, not an analysis one.

WHAT CAN BE MEASURED, AND THE PROXY THAT GETS CLOSEST. Two things:

  1. The fine-grained time course of choice and DV decoding through the
     stimulus, in 100 ms windows stepped by 25 ms, instead of the 200 ms
     windows used before -- which were wider than half the stimulus and
     smeared exactly the period of interest.

  2. AN ACCUMULATION PROXY. If the population were integrating evidence as it
     arrives, the choice signal should rise FASTER when the evidence is
     stronger. Trials are split into |DV| tertiles and the choice time course
     is measured within each. A steeper, earlier rise on strong-evidence
     trials is the signature of accumulation; identical rise times across
     tertiles would argue the signal reflects a decision already made rather
     than one being built. This is not a substitute for the per-tone
     regression, but it is the same question asked with the data that exists.

     Caveat kept with the result: easy trials also produce more consistent
     behaviour, so better decodability there is not by itself proof of
     accumulation.

Usage: python decode_stimulus_dynamics.py
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--sync", default=(r"F:\Gil\Shamir\20260916_110311.rec"
                                    r"\20260916_110311.kilosort\synced_20261002"))
_ap.add_argument("--units", default=(r"D:\Gil\spike_sorting_agent\outputs"
                                     r"\manual_verdicts_20260916_110311.csv"),
                 help="path to a review-tool verdicts csv, or the word 'kslabel'")
_ap.add_argument("--tag", default="20260916_110311")
_A = _ap.parse_args()
SYNC, UNIT_SRC, TAG = _A.sync, _A.units, _A.tag
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FIG = os.path.join(OUT, "reward_cue_figs")

WIN, STEP = 0.100, 0.025
T_LO, T_HI = -0.150, 0.600
N_PERM = 100
STIM_DUR = 0.35

trials = pd.read_parquet(os.path.join(SYNC, "trials.parquet"))
se = pd.read_parquet(os.path.join(SYNC, "state_events.parquet"))
units = pd.read_parquet(os.path.join(SYNC, "units.parquet"))
if UNIT_SRC.lower() == "kslabel":
    good = sorted(units.loc[units.quality_label == "good", "unit_id"].astype(int))
    CURATION = "KSLabel 'good' (automatic — this session has no hand review)"
else:
    ver = pd.read_csv(UNIT_SRC)
    good = sorted(set(ver.loc[ver.verdict == "good", "unit"].astype(int))
                  & set(units.unit_id.astype(int)))
    CURATION = "hand-reviewed 'good' in the review tool"
print(f"session {TAG}: units = {CURATION}")
clean = trials[trials.sync_valid & trials.TrialCompleted]


def fs(names, f):
    s = se[se.state_name.isin(names)].sort_values(["trial_id", "occurrence"])
    return s.groupby("trial_id").first()[f]


ev = clean.set_index("trial_id")[["ChoiceLeft", "ChoiceCorrect", "DV", "MT"]].copy()
ev["stim_on"] = fs(["stimulus_delivery_min"], "start_time_in_trial")
ev = ev[ev.stim_on.notna()]
print(f"{len(ev)} completed trials; stimulus {STIM_DUR*1000:.0f} ms, "
      f"~{int(STIM_DUR/0.01)} tones at ~10 ms spacing")
print(f"median movement time {ev.MT.median():.3f}s -> the choice poke lands "
      f"~{STIM_DUR + ev.MT.median():.2f}s after stimulus onset")

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
print(f"{len(U)} good units")

LOGIT = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 12)))


def design(sub, lo, hi):
    t0 = sub.stim_on.to_numpy()
    ids = sub.index.to_numpy()
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


def cvs(X, y, kind, model, reps=3):
    out = []
    for r in range(reps):
        cv = (KFold(5, shuffle=True, random_state=r) if kind == "reg"
              else StratifiedKFold(5, shuffle=True, random_state=r))
        out.append(np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring="r2" if kind == "reg" else "roc_auc")))
    return float(np.mean(out))


def perm(X, y, kind, model, n=N_PERM, seed=0):
    obs = cvs(X, y, kind, model)
    r = np.random.default_rng(seed)
    null = np.array([cvs(X, r.permutation(y), kind, model, reps=1) for _ in range(n)])
    return obs, float(np.percentile(null, 95)), (1 + int((null >= obs).sum())) / (1 + n)


starts = np.arange(T_LO, T_HI - WIN + 1e-9, STEP)
y_choice = (ev.ChoiceLeft == 1).astype(int).to_numpy()
y_dv = ev.DV.to_numpy()
absdv = np.abs(y_dv)
tert = pd.Series(pd.qcut(absdv, 3, labels=["weak", "medium", "strong"]),
                 index=ev.index)

print("\n" + "=" * 96)
print("FINE TIME COURSE THROUGH THE STIMULUS (100 ms windows, 25 ms steps)")
print("=" * 96)
print(f"{'window':>16}{'choice AUC':>12}{'p':>8}{'DV r2':>9}{'p':>8}"
      f"{'  |  choice AUC by evidence strength':>40}")
print(f"{'':>16}{'':>12}{'':>8}{'':>9}{'':>8}{'weak':>12}{'medium':>9}{'strong':>9}")
rows = []
for s0 in starts:
    X = design(ev, s0, s0 + WIN)
    a_ch, p95_ch, p_ch = perm(X, y_choice, "clf", LOGIT, seed=1)
    r_dv, p95_dv, p_dv = perm(X, y_dv, "reg", RIDGE, seed=2)
    by = {}
    for lab in ["weak", "medium", "strong"]:
        m = (tert == lab).to_numpy()
        if m.sum() >= 40 and len(np.unique(y_choice[m])) == 2:
            by[lab] = cvs(X[m], y_choice[m], "clf", LOGIT, reps=3)
        else:
            by[lab] = np.nan
    rows.append(dict(t_start=s0, t_mid=s0 + WIN / 2,
                     choice_auc=a_ch, choice_p=p_ch, choice_chance95=p95_ch,
                     dv_r2=r_dv, dv_p=p_dv, dv_chance95=p95_dv,
                     auc_weak=by["weak"], auc_medium=by["medium"],
                     auc_strong=by["strong"]))
    print(f"{s0:+.3f}..{s0+WIN:+.3f}{a_ch:>12.3f}{p_ch:>8.3f}{r_dv:>+9.3f}{p_dv:>8.3f}"
          f"{by['weak']:>12.3f}{by['medium']:>9.3f}{by['strong']:>9.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"stimulus_dynamics_{TAG}.csv"), index=False)


def onset(col, thresh, chance_col=None):
    """First window centre where the curve crosses `thresh` and stays above it."""
    v = df[col].to_numpy()
    t = df.t_mid.to_numpy()
    for i in range(len(v)):
        if np.all(v[i:] >= thresh):
            return t[i]
    return np.nan


print("\n" + "=" * 96)
print("WHEN DOES EACH SIGNAL APPEAR? (first window centre that stays above threshold)")
print("=" * 96)
print(f"  choice AUC >= 0.60 : {onset('choice_auc', 0.60):+.3f} s from stimulus onset")
print(f"  choice AUC >= 0.70 : {onset('choice_auc', 0.70):+.3f} s")
print(f"  choice AUC >= 0.80 : {onset('choice_auc', 0.80):+.3f} s")
print(f"  DV R2     >= 0.10  : {onset('dv_r2', 0.10):+.3f} s")
print(f"  stimulus ends at   : {STIM_DUR:+.3f} s")
print(f"  median choice poke : {STIM_DUR + ev.MT.median():+.3f} s")

print("\n" + "=" * 96)
print("ACCUMULATION PROXY: does choice decoding rise FASTER on strong evidence?")
print("=" * 96)
for lab in ["weak", "medium", "strong"]:
    col = f"auc_{lab}"
    n = int((tert == lab).sum())
    md = np.abs(y_dv[(tert == lab).to_numpy()]).mean()
    print(f"  {lab:<8} n={n:<4} mean |DV|={md:.2f}   "
          f"AUC>=0.60 at {onset(col,0.60):+.3f}s   "
          f"AUC>=0.70 at {onset(col,0.70):+.3f}s   "
          f"peak-in-stimulus {df.loc[df.t_mid<=STIM_DUR, col].max():.3f}")
print("\n  [a steeper, earlier rise on strong evidence is the signature of")
print("   accumulation; identical rise times would argue the signal reflects a")
print("   decision already made. Caveat: easy trials also give more consistent")
print("   behaviour, so this is suggestive, not proof.]")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axes[0]
ax.axvspan(0, STIM_DUR, color="#b07", alpha=0.10, lw=0)
ax.axhline(0.5, color="#aaa", lw=0.8)
ax.plot(df.t_mid, df.choice_auc, color="#9a9288", lw=2.4, label="rat's choice (AUC)")
ax.plot(df.t_mid, df.choice_chance95, color="#9a9288", lw=1, ls=":")
ax.plot(df.t_mid, df.dv_r2 + 0.5, color="#8c3b2e", lw=2.4, label="DV ($R^2$, offset +0.5)")
ax.axvline(0, color="#222", lw=1.2)
ax.axvline(STIM_DUR, color="#222", lw=1.2, ls="--")
ax.axvline(STIM_DUR + ev.MT.median(), color="#3f6b4a", lw=1.2, ls=":")
ax.text(STIM_DUR / 2, 0.97, "stimulus", ha="center", fontsize=8.5, color="#666")
ax.text(STIM_DUR + ev.MT.median() + .01, 0.52, "median\nchoice poke", fontsize=7,
        color="#3f6b4a")
ax.set_xlabel("time from stimulus onset (s)", fontsize=9.5)
ax.set_ylabel("decoding performance", fontsize=9.5)
ax.set_title("Choice and DV through the stimulus", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.tick_params(labelsize=8)

ax = axes[1]
ax.axvspan(0, STIM_DUR, color="#b07", alpha=0.10, lw=0)
ax.axhline(0.5, color="#aaa", lw=0.8)
for lab, c in [("weak", "#c9b8a8"), ("medium", "#8c6a4f"), ("strong", "#4a2c18")]:
    ax.plot(df.t_mid, df[f"auc_{lab}"], color=c, lw=2.2, label=f"{lab} evidence")
ax.axvline(0, color="#222", lw=1.2)
ax.axvline(STIM_DUR, color="#222", lw=1.2, ls="--")
ax.set_xlabel("time from stimulus onset (s)", fontsize=9.5)
ax.set_ylabel("choice decoding (AUC)", fontsize=9.5)
ax.set_title("Does the choice signal build faster on strong evidence?", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.tick_params(labelsize=8)

fig.suptitle(f"Stimulus-period dynamics, session {TAG} — {len(ev)} completed trials, "
             f"{len(U)} good units\n100 ms windows stepped by 25 ms · dotted grey = "
             "permutation chance",
             fontsize=11, fontweight="bold")
fig.tight_layout()
p = os.path.join(FIG, f"stimulus_dynamics_{TAG}.png")
fig.savefig(p, dpi=125, bbox_inches="tight")
print(f"\nsaved {p}")
print(f"saved stimulus_dynamics_{TAG}.csv")
