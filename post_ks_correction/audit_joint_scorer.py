"""AUDIT E1: does the plan's "train one regularized joint scorer" beat our
three independent gates, on the EXACT candidate table the 2.3% run produced?

Zero new heavy compute: outputs/v2_post_hoc_candidates.csv already holds every
refined candidate with mf / r2 / fp and whether it was a real missed spike.

Reported with base rates throughout (project rule: every "is X near Y" number
carries its chance rate).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

CAND = r"D:\Gil\spike_sorting_agent\outputs\v2_post_hoc_candidates.csv"
d = pd.read_csv(CAND)
print(f"loaded {len(d):,} candidates | configs {sorted(d.config.unique())}")

def auc(score, y):
    """Rank AUC, nan-safe."""
    m = np.isfinite(score) & np.isfinite(y)
    s, yy = score[m], y[m].astype(bool)
    if yy.sum() == 0 or (~yy).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = yy.sum(), (~yy).sum()
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

for cfg in ["vanilla", "footprint_cluster_strong", "coarse_then_align"]:
    c = d[d.config == cfg].copy()
    if not len(c):
        continue
    y = c.is_true.values.astype(bool)
    print("\n" + "=" * 72)
    print(f"CONFIG {cfg}:  {len(c):,} candidates, {y.sum():,} real "
          f"({100*y.mean():.2f}% base rate)")
    print("=" * 72)

    print("\n-- single-score AUC (chance = 0.500) --")
    for col in ["mf", "r2", "fp"]:
        a = auc(c[col].values, y)
        print(f"   {col:>3}: AUC {a:.3f}")
    print("\n-- per tier --")
    print(f"   {'tier':<10} {'n':>7} {'%real':>7} {'mf':>6} {'r2':>6} {'fp':>6}")
    for t, g in c.groupby("tier"):
        yg = g.is_true.values.astype(bool)
        print(f"   {t:<10} {len(g):>7,} {100*yg.mean():>6.2f}% "
              f"{auc(g.mf.values, yg):>6.3f} {auc(g.r2.values, yg):>6.3f} "
              f"{auc(g.fp.values, yg):>6.3f}")

    # ---- the plan's proposal: ONE regularized joint scorer, not 3 gates.
    # Grouped CV by unit so a unit never trains and tests together.
    c2 = c.dropna(subset=["mf", "r2", "fp"]).copy()
    X = c2[["mf", "r2", "fp"]].values
    yj = c2.is_true.values.astype(int)
    grp = c2.unit.values
    oof = np.full(len(c2), np.nan)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(grp))))
    for tr, te in gkf.split(X, yj, grp):
        if yj[tr].sum() < 5:
            continue
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000,
                                               class_weight="balanced"))
        mdl.fit(X[tr], yj[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    print(f"\n-- joint ridge-logistic scorer, unit-grouped CV --")
    print(f"   out-of-fold AUC {auc(oof, yj.astype(bool)):.3f}")

    # ---- precision at MATCHED acceptance count (the only fair comparison)
    ok_gate = (c2.r2 >= c2.bar_r2) & (c2.fp >= c2.bar_fp) & np.isfinite(c2.fp)
    n_acc = int(ok_gate.sum())
    prec_gate = yj[ok_gate.values].mean() if n_acc else np.nan
    print(f"\n-- recovery precision at matched acceptance (n = {n_acc:,}) --")
    print(f"   current two gates (r2 AND fp percentile) : "
          f"{100*prec_gate:.2f}%   ({yj[ok_gate.values].sum()} real)")
    for nm, sc in [("mf alone", c2.mf.values), ("r2 alone", c2.r2.values),
                   ("fp alone", c2.fp.values), ("JOINT scorer", oof)]:
        s = np.where(np.isfinite(sc), sc, -np.inf)
        top = np.argsort(-s)[:n_acc]
        print(f"   {nm:<40} : {100*yj[top].mean():.2f}%   ({yj[top].sum()} real)")
