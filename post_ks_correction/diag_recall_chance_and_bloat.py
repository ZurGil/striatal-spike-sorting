"""Recall does not penalise MERGING: a cluster holding three neurons contains
nearly all of each one's spikes. locden_iso emits 76 clusters where Kilosort
emits 238, so its high recall must be read against two controls.

(1) CHANCE RATE -- shuffle cluster labels across spikes, preserving each
    cluster's size and every spike time, and recompute best-cluster recall.
(2) BLOAT -- how many times bigger the matched cluster is than the part of it
    that actually belongs to the unit.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Gil\spike_sorting_agent\post_ks_correction")
from run_v2_comparison import _match, TOLERANCE
ROOT = r"D:\Gil\spike_sorting_agent"
rng = np.random.default_rng(0)
T = np.load(os.path.join(ROOT, "hybrid_v2_rep0", "hybrid_truth.npz"),
            allow_pickle=True)
t_true, lab = T["times"].ravel(), T["labels"].ravel()

print("%-20s %8s %8s %9s %8s" % ("sorter", "recall", "chance", "bloat", "ncl"))
for nm in ["footprint_cluster", "vanilla", "lupin", "locden_iso",
           "tridesclous2", "mountainsort5"]:
    d = os.path.join(ROOT, "hybrid_v2_rep0", "ks_" + nm)
    st = np.load(os.path.join(d, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
    pu = pd.read_csv(os.path.join(ROOT, "outputs",
                                  "per_unit_fair_%s_rep0.csv" % nm))
    sizes = pd.Series(cl).value_counts()
    bloat = []
    for _, r in pu.iterrows():
        if r.cluster < 0 or not np.isfinite(r.precision) or r.precision <= 0:
            continue
        bloat.append(1.0 / r.precision)
    # chance: permute labels, keep sizes and times, re-score at each unit's
    # own already-chosen offset so the only thing destroyed is cluster identity
    ch = []
    for rep in range(3):
        perm = cl[rng.permutation(len(cl))]
        for _, r in pu.iterrows():
            tt = np.sort(t_true[lab == int(r.unit)]).astype(np.int64)
            lo = np.searchsorted(st, tt - TOLERANCE - abs(int(r.off)), "left")
            hi = np.searchsorted(st, tt + TOLERANCE + abs(int(r.off)), "right")
            cand = set()
            for a_, b_ in zip(lo, hi):
                cand.update(perm[a_:b_].tolist())
            bh = 0
            for c in cand:
                td = np.sort(st[perm == c])
                h, _ = _match(td + int(r.off), tt)
                bh = max(bh, h)
            ch.append(bh / len(tt))
    print("%-20s %8.3f %8.3f %9.2f %8d"
          % (nm, pu.recall.mean(), float(np.mean(ch)),
             float(np.median(bloat)), len(sizes)), flush=True)
