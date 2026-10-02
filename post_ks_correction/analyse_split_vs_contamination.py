"""
Is Kilosort's problem ONLY that it splits neurons, or does splitting also drag
in contamination?

THE QUESTION, in Gil's words: if an easy unit was split into three clusters,
what are the recall and precision of those three COMBINED? If combining the
fragments recovers nearly all the spikes AND stays clean, the problem is purely
splitting and the fix is a merge tool. If combining recovers the spikes but the
precision collapses, the fragments are contaminated and merging them would just
pool the contamination -- a completely different problem needing a different
fix.

TWO MEASUREMENTS PER PLACEMENT

  BEST-CLUSTER (what every table so far reports)
    recall    = our spikes in the single best-matching cluster / our spikes
    precision = our spikes in that cluster / all spikes in that cluster

  COMBINED (new)
    Take every cluster holding a meaningful share of our spikes -- the
    fragments -- and pool them.
    recall    = our spikes found in ANY fragment / our spikes
    precision = our spikes found in any fragment / all spikes in all fragments

  The difference between the two is exactly what splitting costs, and the
  precision of the combined set is exactly what a merge tool would inherit.

THE THRESHOLD MATTERS, SO IT IS SWEPT. "Every cluster containing at least one
of our spikes" is the base-rate trap this project has fallen into three times:
with 200+ clusters and a +-10 sample window, chance alone puts a few of our
spikes in many clusters, so a loose threshold inflates the fragment count and
destroys precision for free. A fragment therefore has to hold at least
`min(MIN_ABS, frac * n_true)` of our spikes, and results are reported at
several thresholds so the reader can see how much the conclusion depends on it.

A chance-level reference is computed too: the same counting against RANDOM time
points instead of our injected times, which says how many "fragments" this
procedure invents when there is nothing to find.

Usage: python analyse_split_vs_contamination.py
"""
import os
import numpy as np
import pandas as pd

ROOT = r"D:\Gil\spike_sorting_agent"
OUT = os.path.join(ROOT, "outputs")
N_REPLICATES = 4
TOLERANCE = 10
CONFIGS = ["vanilla", "subsample_align", "coarse_then_align",
           "amplitude_normalize", "align_and_amp_norm", "coarse_align_amp_norm",
           "footprint_cluster", "footprint_cluster_strong"]
# a cluster counts as a fragment if it holds >= this many of our spikes
THRESHOLDS = [(5, 0.02), (15, 0.02), (30, 0.05)]
MAIN = (15, 0.02)          # the one used for the headline tables


def hits_per_cluster(st_sorted, cl_sorted, tt, tol=TOLERANCE):
    """For each cluster, how many of our true spikes it accounts for.

    One-to-one within a cluster: a detection is consumed once, so a cluster
    cannot claim the same spike twice. Returns (counts, matched_any) where
    matched_any marks true spikes picked up by ANY cluster -- that is the union
    used for combined recall, and it counts each true spike at most once.
    """
    counts = {}
    matched_any = np.zeros(len(tt), bool)
    used = {}
    for i, t in enumerate(np.sort(tt)):
        lo = np.searchsorted(st_sorted, t - tol, "left")
        hi = np.searchsorted(st_sorted, t + tol, "right")
        if hi <= lo:
            continue
        best_j, best_d = -1, None
        for j in range(lo, hi):
            if used.get(j):
                continue
            d = abs(int(st_sorted[j]) - int(t))
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j < 0:
            continue
        used[best_j] = True
        c = int(cl_sorted[best_j])
        counts[c] = counts.get(c, 0) + 1
        matched_any[i] = True
    return counts, matched_any


rows = []
for cfg in CONFIGS:
    for rep in range(N_REPLICATES):
        d = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
        res = os.path.join(d, f"ks_{cfg}")
        if not os.path.exists(os.path.join(res, "spike_times.npy")):
            continue
        st = np.load(os.path.join(res, "spike_times.npy")).ravel()
        cl = np.load(os.path.join(res, "spike_clusters.npy")).ravel()
        o = np.argsort(st)
        st, cl = st[o], cl[o]
        size = pd.Series(cl).value_counts().to_dict()

        T = np.load(os.path.join(d, "hybrid_truth.npz"), allow_pickle=True)
        times, labels = T["times"].ravel(), T["labels"].ravel()
        tier_names = [str(x) for x in T["tier_names"]]
        tier = np.array([tier_names[i] for i in T["tier"].ravel()])

        rng = np.random.default_rng(0)
        n_samples = int(T["n_samples"])

        for u in np.unique(labels):
            sel = labels == u
            tt = np.sort(times[sel])
            ut = tier[sel][0]
            counts, matched_any = hits_per_cluster(st, cl, tt)
            if not counts:
                continue
            best_c = max(counts, key=counts.get)
            best_hits = counts[best_c]

            # chance reference: same procedure on random times
            fake = np.sort(rng.integers(60, n_samples - 60, size=len(tt)))
            fcounts, _ = hits_per_cluster(st, cl, fake)

            rec = dict(config=cfg, replicate=rep, unit=int(u), tier=ut,
                       n_true=len(tt),
                       best_recall=best_hits / len(tt),
                       best_precision=best_hits / size.get(best_c, 1))
            for (mn, fr) in THRESHOLDS:
                thr = max(mn, int(fr * len(tt)))
                frags = [c for c, n in counts.items() if n >= thr]
                if best_c not in frags:
                    frags.append(best_c)
                tot = sum(size.get(c, 0) for c in frags)
                ours = sum(counts[c] for c in frags)
                fake_frags = sum(1 for c, n in fcounts.items() if n >= thr)
                key = f"t{mn}"
                rec[f"{key}_nfrag"] = len(frags)
                rec[f"{key}_recall"] = ours / len(tt)
                rec[f"{key}_precision"] = ours / tot if tot else np.nan
                rec[f"{key}_chance_nfrag"] = fake_frags
            rec["union_recall_all"] = float(matched_any.mean())
            rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "v2_split_vs_contamination.csv"), index=False)
k = f"t{MAIN[0]}"

print("=" * 94)
print("IS THE PROBLEM ONLY SPLITTING?  best single cluster vs ALL fragments combined")
print(f"(a fragment = a cluster holding >= max({MAIN[0]}, {MAIN[1]:.0%} of the unit's spikes))")
print("=" * 94)

for cfg in CONFIGS:
    c = df[df.config == cfg]
    if not len(c):
        continue
    print(f"\n### {cfg}")
    g = c.groupby("tier").agg(
        n=("unit", "size"),
        frags=(f"{k}_nfrag", "mean"),
        best_rec=("best_recall", "mean"), best_prec=("best_precision", "mean"),
        comb_rec=(f"{k}_recall", "mean"), comb_prec=(f"{k}_precision", "mean"))
    g["rec_gain"] = g.comb_rec - g.best_rec
    g["prec_cost"] = g.comb_prec - g.best_prec
    print(g.round(3).to_string())

print("\n" + "=" * 94)
print("THE HEADLINE: stock Kilosort, pooled over all tiers")
print("=" * 94)
v = df[df.config == "vanilla"]
print(f"  fragments per unit        : {v[f'{k}_nfrag'].mean():.2f}")
print(f"  best-cluster  recall      : {v.best_recall.mean():.3f}")
print(f"  COMBINED      recall      : {v[f'{k}_recall'].mean():.3f}"
      f"   (+{v[f'{k}_recall'].mean()-v.best_recall.mean():.3f})")
print(f"  best-cluster  precision   : {v.best_precision.mean():.3f}")
print(f"  COMBINED      precision   : {v[f'{k}_precision'].mean():.3f}"
      f"   ({v[f'{k}_precision'].mean()-v.best_precision.mean():+.3f})")
print(f"  union recall, ANY cluster : {v.union_recall_all.mean():.3f}"
      f"   <- ceiling: every spike Kilosort detected and put somewhere")
print(f"  chance fragments          : {v[f'{k}_chance_nfrag'].mean():.2f}"
      f"   <- what this counting invents from nothing")

print("\n" + "=" * 94)
print("THRESHOLD SENSITIVITY (stock Kilosort) -- does the conclusion depend on it?")
print("=" * 94)
for (mn, fr) in THRESHOLDS:
    kk = f"t{mn}"
    print(f"  fragment >= max({mn:>2}, {fr:.0%}): "
          f"frags {v[f'{kk}_nfrag'].mean():.2f}  "
          f"combined recall {v[f'{kk}_recall'].mean():.3f}  "
          f"combined precision {v[f'{kk}_precision'].mean():.3f}  "
          f"(chance frags {v[f'{kk}_chance_nfrag'].mean():.2f})")

print("\n" + "=" * 94)
print("WHAT A PERFECT MERGE TOOL WOULD BUY (stock Kilosort, per tier)")
print("=" * 94)
g = v.groupby("tier").agg(
    frags=(f"{k}_nfrag", "mean"),
    best_rec=("best_recall", "mean"), comb_rec=(f"{k}_recall", "mean"),
    best_prec=("best_precision", "mean"), comb_prec=(f"{k}_precision", "mean"))
g["recall_recovered"] = (g.comb_rec - g.best_rec).round(3)
g["precision_paid"] = (g.comb_prec - g.best_prec).round(3)
print(g.round(3).to_string())
print("\nsaved v2_split_vs_contamination.csv")
