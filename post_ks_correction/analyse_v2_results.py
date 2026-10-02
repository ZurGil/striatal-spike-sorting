"""
Turn the v2 sweep into a defensible comparison: measure the noise floor, then
test each configuration against stock Kilosort PAIRWISE on the same placements.

TWO THINGS THE RAW TABLES DO NOT TELL YOU, BOTH OF WHICH CHANGE THE READING:

1. THE NOISE FLOOR IS NOT ZERO ANY MORE. In round one `vanilla_repeat`
   reproduced `vanilla` exactly, which is what licensed reading every
   difference as signal. On the v2 benchmark it does NOT: the two identical
   runs differ on the `hard` tier. So a difference smaller than that gap is
   indistinguishable from re-running the same code twice, and any claim built
   on one has to be withdrawn. This script measures the gap per tier and prints
   it beside every comparison instead of leaving it implicit.

2. COMPARING MEANS WASTES THE DESIGN. Every configuration sees exactly the same
   52 placements, so the comparison is PAIRED: the right question is "on how
   many placements did this config beat vanilla, and by how much", not "is its
   mean higher". A paired test removes the between-unit variance that otherwise
   swamps everything -- and that variance is enormous here (per-replicate
   overall recall ranges 0.60-0.93), which is exactly why round one's n=3 tier
   means were unreadable.

Reported per configuration: paired mean difference against vanilla, the
Wilcoxon signed-rank p-value (non-parametric, since per-unit recall is bounded
and skewed), win/loss counts, and whether the effect clears the measured noise
floor for that tier.

Usage: python analyse_v2_results.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = r"D:\Gil\spike_sorting_agent\outputs"
df = pd.read_csv(os.path.join(OUT, "v2_scores.csv"))

TIERS = ["easy", "hard", "collision", "pair"]
CONFIGS = [c for c in df.config.unique() if c not in ("vanilla", "vanilla_repeat")]
KEY = ["replicate", "unit", "tier"]

print("=" * 86)
print("1. THE NOISE FLOOR: vanilla vs vanilla_repeat (identical code, two runs)")
print("=" * 86)
a = df[df.config == "vanilla"].set_index(KEY)
b = df[df.config == "vanilla_repeat"].set_index(KEY)
common = a.index.intersection(b.index)
a, b = a.loc[common], b.loc[common]
rows = []
for t in TIERS:
    m = a.index.get_level_values("tier") == t
    for metric in ("recall", "precision"):
        d = (a.loc[m, metric] - b.loc[m, metric]).dropna()
        rows.append(dict(tier=t, metric=metric, n=len(d),
                         mean_diff=round(float(d.mean()), 4),
                         max_abs_diff=round(float(d.abs().max()), 4),
                         n_placements_differing=int((d.abs() > 1e-9).sum())))
nf = pd.DataFrame(rows)
print(nf.to_string(index=False))

floor = {}
for t in TIERS:
    r = nf[(nf.tier == t) & (nf.metric == "recall")].iloc[0]
    floor[t] = float(r.max_abs_diff)
print("\nPer-tier recall noise floor (largest difference between two identical runs):")
for t in TIERS:
    print(f"  {t:<10} {floor[t]:.4f}"
          + ("   <- NOT deterministic" if floor[t] > 1e-9 else "   (bit-identical)"))
nondet = [t for t in TIERS if floor[t] > 1e-9]
print(f"\nKilosort is non-deterministic on: {nondet if nondet else 'nothing'}")
print("Any effect on those tiers smaller than the floor above is NOT evidence.")

print("\n" + "=" * 86)
print("2. PAIRED COMPARISON AGAINST STOCK KILOSORT (same 52 placements)")
print("=" * 86)
van = df[df.config == "vanilla"].set_index(KEY)
out = []
for cfg in CONFIGS:
    c = df[df.config == cfg].set_index(KEY)
    idx = van.index.intersection(c.index)
    for metric in ("recall", "precision"):
        for scope in ["ALL"] + TIERS:
            if scope == "ALL":
                sel = idx
            else:
                sel = idx[idx.get_level_values("tier") == scope]
            v = van.loc[sel, metric]
            x = c.loc[sel, metric]
            d = (x - v).dropna()
            if len(d) < 3:
                continue
            try:
                p = float(wilcoxon(d, zero_method="zsplit").pvalue) if d.abs().sum() > 0 else 1.0
            except Exception:
                p = np.nan
            fl = floor.get(scope, 0.0) if metric == "recall" else np.nan
            out.append(dict(
                config=cfg, metric=metric, scope=scope, n=len(d),
                vanilla=round(float(v.mean()), 4), patched=round(float(x.mean()), 4),
                diff=round(float(d.mean()), 4),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                p=round(p, 4) if np.isfinite(p) else np.nan,
                clears_noise=("n/a" if not np.isfinite(fl)
                              else ("yes" if abs(float(d.mean())) > fl else "NO"))))
res = pd.DataFrame(out)
res.to_csv(os.path.join(OUT, "v2_paired_vs_vanilla.csv"), index=False)

for metric in ("recall", "precision"):
    print(f"\n----- {metric.upper()}, overall (all 52 placements) -----")
    r = res[(res.metric == metric) & (res.scope == "ALL")].sort_values("diff", ascending=False)
    print(r[["config", "vanilla", "patched", "diff", "wins", "losses", "p",
             "clears_noise"]].to_string(index=False))

print("\n" + "=" * 86)
print("3. THE PAIR TIER (the tier footprint weighting was built for)")
print("=" * 86)
for metric in ("recall", "precision"):
    r = res[(res.metric == metric) & (res.scope == "pair")].sort_values("diff", ascending=False)
    print(f"\n--- {metric} on the pair tier ---")
    print(r[["config", "vanilla", "patched", "diff", "wins", "losses", "p",
             "clears_noise"]].to_string(index=False))

print("\n" + "=" * 86)
print("4. WHAT MAKES A PLACEMENT HARD? (vanilla only, Spearman vs recall)")
print("=" * 86)
v = df[df.config == "vanilla"]
for col in ("contam_pct", "site_max_neighbor_uv", "site_noise_uv", "site_density",
            "n_true"):
    if col in v.columns:
        print(f"  {col:<24} {v.recall.corr(v[col], method='spearman'):+.3f}")
print("\n  (the hard tier was built on site amplitude; if contamination dominates,")
print("   the donor matters more than where it was placed)")

print("\n" + "=" * 86)
print("5. MERGE ERRORS (pair tier: correct answer is TWO clusters)")
print("=" * 86)
mg = os.path.join(OUT, "v2_merge_errors.csv")
if os.path.exists(mg):
    m = pd.read_csv(mg)
    g = m.groupby("config").merged.agg(["sum", "size"])
    g.columns = ["merged", "pairs"]
    print(g.to_string())
    bad = m[m.merged]
    if len(bad):
        print("\nthe merged pairs:")
        print(bad[["config", "replicate", "pair_id", "units"]].to_string(index=False))
print("\nsaved v2_paired_vs_vanilla.csv")
