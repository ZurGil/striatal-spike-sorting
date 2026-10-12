"""Does tridesclous2's rep0 precision survive the other three replicates?

The one live lead out of 5ay/5az was tridesclous2 holding 0.862-0.965
precision in EVERY tier on rep0, against stock Kilosort's 0.486-0.765. 5ba
established that rep0 is the easiest of the four windows, so that lead was
never established. This scores it on all four, with the per-unit alignment of
score_per_unit_fair.py (needed: its median offset is +12, +2, +7, +10 by
replicate) and stock Kilosort scored at shift 0 as the reference.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Gil\spike_sorting_agent\post_ks_correction")
from run_v2_comparison import _match, TOLERANCE
ROOT = r"D:\Gil\spike_sorting_agent"
SHIFTS = np.arange(-24, 9)
W = TOLERANCE + int(np.abs(SHIFTS).max())
TIERS = ["easy", "collision", "pair", "hard"]


def pairs(td, tt):
    lo = np.searchsorted(td, tt - W, "left")
    hi = np.searchsorted(td, tt + W, "right")
    n = hi - lo
    keep = n > 0
    if not keep.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    lo, n, idx = lo[keep], n[keep], np.flatnonzero(keep)
    starts = np.repeat(lo, n)
    within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ti = np.repeat(idx, n)
    return ti, td[starts + within] - tt[ti]


def score_rep(nm, rep, per_unit):
    d = os.path.join(ROOT, "hybrid_v2_rep%d" % rep)
    T = np.load(os.path.join(d, "hybrid_truth.npz"), allow_pickle=True)
    t_true, lab = T["times"].ravel(), T["labels"].ravel()
    tier_code = T["tier"].ravel()
    tier_names = [str(x) for x in T["tier_names"]]
    p = os.path.join(d, "ks_" + nm)
    st = np.load(os.path.join(p, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(p, "spike_clusters.npy")).ravel()
    o = np.argsort(st); st, cl = st[o], cl[o]
    rows = []
    for u in np.unique(lab):
        sel = lab == u
        tt = np.sort(t_true[sel]).astype(np.int64)
        tier = tier_names[int(tier_code[sel][0])]
        lo = np.searchsorted(st, tt - W, "left")
        hi = np.searchsorted(st, tt + W, "right")
        cand = set()
        for a_, b_ in zip(lo, hi):
            cand.update(cl[a_:b_].tolist())
        best = (0, np.nan, 0, -1)
        for c in sorted(cand):
            td = np.sort(st[cl == c])
            if not len(td):
                continue
            if not per_unit:
                h, _ = _match(td, tt)
                if h > best[0]:
                    best = (h, h / len(td), 0, int(c))
                continue
            ti, off = pairs(td, tt)
            if not len(ti):
                continue
            bs, bn = 0, -1
            for s in SHIFTS:
                m = np.abs(off + s) <= TOLERANCE
                if not m.any():
                    continue
                k = len(np.unique(ti[m]))
                if k > bn or (k == bn and abs(s) < abs(bs)):
                    bn, bs = k, int(s)
            h, _ = _match(td + bs, tt)
            if h > best[0]:
                best = (h, h / len(td), bs, int(c))
        h, pr, s, c = best
        rows.append(dict(rep=rep, unit=int(u), tier=tier, off=s,
                         recall=h / len(tt), precision=pr,
                         bloat=(1.0 / pr) if pr and np.isfinite(pr) else np.nan))
    return pd.DataFrame(rows)


JOBS = [("tridesclous2", True), ("vanilla", False),
        ("footprint_cluster", False)]
allr = {}
for nm, pu in JOBS:
    fr = []
    for rep in range(4):
        if not os.path.isfile(os.path.join(ROOT, "hybrid_v2_rep%d" % rep,
                                           "ks_" + nm, "spike_times.npy")):
            print("  (%s rep%d missing)" % (nm, rep)); continue
        a = score_rep(nm, rep, pu)
        fr.append(a)
        print("  %-18s rep%d  recall %.3f  precision %.3f  (offsets %+d..%+d)"
              % (nm, rep, a.recall.mean(), float(np.nanmean(a.precision)),
                 a.off.min(), a.off.max()), flush=True)
    d = pd.concat(fr)
    allr[nm] = d
    d.to_csv(os.path.join(ROOT, "outputs",
                          "allreps_per_unit_%s.csv" % nm), index=False)

print("\n" + "=" * 92)
print("POOLED OVER 4 REPLICATES, 52 placements (recall / precision)")
print("=" * 92)
for nm in allr:
    d = allr[nm]
    line = "  %-18s overall %.3f/%.3f  " % (
        nm, d.recall.mean(), float(np.nanmean(d.precision)))
    line += " ".join("%s %.3f/%.3f" % (t[:4], d[d.tier == t].recall.mean(),
                     float(np.nanmean(d[d.tier == t].precision)))
                     for t in TIERS)
    print(line)
print("\nPER-REPLICATE SPREAD (recall, then precision)")
for nm in allr:
    d = allr[nm]
    g = d.groupby("rep")
    print("  %-18s recall %s | precision %s" % (
        nm, ", ".join("%.3f" % v for v in g.recall.mean()),
        ", ".join("%.3f" % v for v in g.precision.mean())))
print("\nMATCHED-CLUSTER BLOAT (median)")
for nm in allr:
    print("  %-18s %.2fx" % (nm, float(np.nanmedian(allr[nm].bloat))))
