"""Decompose the footprint win into its two halves.

`clustering_qr.cluster()` uses its feature matrix in exactly two places:
`neigh_mat()` (the kNN graph the assignment loop runs on) and
`kmeans_plusplus()` (the seeding of nclust=200 centres). The original
footprint patch appends to Xd and so moves both at once. These configs move
one at a time:

    footprint_cluster / _strong   graph AND seeding   (the measured win)
    footprint_graph   / _strong   graph only
    footprint_seed                seeding only

Three questions, each a paired test on the identical placements:
  1. does graph-only reproduce the combined win?
  2. does seeding-only do anything on its own?
  3. is the combination additive, or do the halves cancel?

Reported per tier with recall AND precision together, and never without the
noise floor from analyse_v2_results.py in mind: the hard tier is not
measurable and no hard-tier number here should be believed.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = r"D:\Gil\spike_sorting_agent\outputs"
df = pd.read_csv(os.path.join(OUT, "v2_scores.csv"))
KEY = ["replicate", "unit", "tier"]
TIERS = ["easy", "collision", "pair", "hard"]

PAIRS = [
    ("footprint_graph", "vanilla", "graph only vs stock"),
    ("footprint_graph_strong", "vanilla", "graph only (w3) vs stock"),
    ("footprint_seed", "vanilla", "seeding only vs stock"),
    ("footprint_cluster", "vanilla", "graph+seed vs stock (known win)"),
    ("footprint_cluster_strong", "vanilla", "graph+seed (w3) vs stock"),
    ("footprint_graph", "footprint_cluster", "DECOMP: removing the seeding half"),
    ("footprint_graph_strong", "footprint_cluster_strong",
     "DECOMP: removing the seeding half (w3)"),
    ("footprint_seed", "footprint_cluster", "DECOMP: removing the graph half"),
]


def paired(cfg, ref, metric, scope):
    a = df[df.config == cfg].set_index(KEY)
    b = df[df.config == ref].set_index(KEY)
    idx = a.index.intersection(b.index)
    if scope != "ALL":
        idx = idx[idx.get_level_values("tier") == scope]
    d = (a.loc[idx, metric] - b.loc[idx, metric]).dropna()
    if len(d) < 3:
        return None
    if d.abs().sum() == 0:
        return dict(n=len(d), diff=0.0, p=1.0, win=0, loss=0)
    p = float(wilcoxon(d, zero_method="zsplit").pvalue)
    return dict(n=len(d), diff=float(d.mean()), p=p,
                win=int((d > 0).sum()), loss=int((d < 0).sum()))


have = set(df.config.unique())
rows = []
for cfg, ref, label in PAIRS:
    if cfg not in have or ref not in have:
        print(f"  (skipping {label}: missing {cfg if cfg not in have else ref})")
        continue
    for metric in ("recall", "precision"):
        for scope in ["ALL"] + TIERS:
            r = paired(cfg, ref, metric, scope)
            if r is None:
                continue
            rows.append(dict(comparison=label, config=cfg, ref=ref,
                             metric=metric, scope=scope, **r))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "graph_vs_seed_paired.csv"), index=False)

for metric in ("recall", "precision"):
    print("\n" + "=" * 94)
    print(f"{metric.upper()}: paired difference, mean (wins/losses), p")
    print("=" * 94)
    sub = res[res.metric == metric]
    print(f"  {'comparison':<44} {'ALL':>16} {'pair':>16} {'collision':>16}")
    for label in sub.comparison.unique():
        g = sub[sub.comparison == label].set_index("scope")
        cells = []
        for scope in ("ALL", "pair", "collision"):
            if scope in g.index:
                r = g.loc[scope]
                star = "*" if r.p < 0.05 else " "
                cells.append(f"{r['diff']:+.4f} p{r['p']:.3f}{star}")
            else:
                cells.append("--")
        print(f"  {label:<44} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

print("\n" + "=" * 94)
print("OVERALL MEANS (every scored placement, 4 replicates)")
print("=" * 94)
keep = [c for c in ["vanilla", "footprint_cluster", "footprint_cluster_strong",
                    "footprint_graph", "footprint_graph_strong",
                    "footprint_seed"] if c in have]
agg = (df[df.config.isin(keep)].groupby("config")
       .agg(recall=("recall", "mean"), precision=("precision", "mean"),
            fragments=("n_fragments", "mean"), n=("recall", "size")))
print(agg.reindex(keep).round(4).to_string())

print("\nPAIR TIER ONLY (the tier the footprint idea was built for)")
pt = df[df.config.isin(keep) & (df.tier == "pair")]
print(pt.groupby("config").agg(recall=("recall", "mean"),
                               precision=("precision", "mean"),
                               n=("recall", "size")).reindex(keep)
      .round(4).to_string())

print("\nADDITIVITY CHECK on pair-tier recall "
      "(is graph+seed the sum of its halves?)")
base = df[(df.config == "vanilla") & (df.tier == "pair")].recall.mean()
for w, comb in (("1.0", "footprint_cluster"), ("3.0", "footprint_cluster_strong")):
    gcfg = "footprint_graph" if w == "1.0" else "footprint_graph_strong"
    if comb not in have or gcfg not in have:
        continue
    g = df[(df.config == gcfg) & (df.tier == "pair")].recall.mean() - base
    c = df[(df.config == comb) & (df.tier == "pair")].recall.mean() - base
    s = (df[(df.config == "footprint_seed") & (df.tier == "pair")].recall.mean()
         - base) if "footprint_seed" in have else np.nan
    print(f"  weight {w}: graph {g:+.4f} | seed {s:+.4f} | "
          f"sum {g + (s if np.isfinite(s) else 0):+.4f} | "
          f"combined MEASURED {c:+.4f}")
