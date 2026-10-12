"""Score with a PER-UNIT time offset, because the benchmark's convention is
arbitrary per unit, not per sorter.

MEASURED REASON. The truth time is sample 20 of each donor's 61-sample
extraction window. Where that sits relative to the physical trough is a
property of each donor waveform -- measured at +8..+17 across the 13 rep0
units. A sorter that reports real troughs therefore inherits a DIFFERENT
offset per unit: measured per-unit medians are locden_iso +5..+15,
tridesclous2 +7..+17, mountainsort5 +4..+17, lupin +9..+20. One global integer
shift cannot bring a 13-sample spread inside a 10-sample tolerance, so the
global-shift scores in 5ay UNDERSTATE those sorters.

THE CONTROL THAT MAKES THIS READABLE. Kilosort shares the benchmark's
convention, so per-unit freedom should buy it nothing. Whatever it gains is
the selection bias of taking a max over 33 shifts, and that much is discounted
from every other sorter's gain.

METHOD. The shift SEARCH is a vectorised offset histogram (cheap); the
reported score at the chosen shift is the project's own exact greedy
one-to-one _match, so the numbers stay comparable to every other table.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Gil\spike_sorting_agent\post_ks_correction")
from run_v2_comparison import _match, TOLERANCE
ROOT = r"D:\Gil\spike_sorting_agent"
SHIFTS = np.arange(-24, 9)
W = TOLERANCE + int(np.abs(SHIFTS).max())
TIERS = ["easy", "collision", "pair", "hard"]

T = np.load(os.path.join(ROOT, "hybrid_v2_rep0", "hybrid_truth.npz"),
            allow_pickle=True)
t_true, lab = T["times"].ravel(), T["labels"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]


def pairs(td, tt):
    """All (true index, offset) pairs with |offset| <= W, vectorised."""
    lo = np.searchsorted(td, tt - W, "left")
    hi = np.searchsorted(td, tt + W, "right")
    n = hi - lo
    keep = n > 0
    if not keep.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    lo, n, idx = lo[keep], n[keep], np.flatnonzero(keep)
    starts = np.repeat(lo, n)
    within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    di = starts + within
    ti = np.repeat(idx, n)
    return ti, td[di] - tt[ti]


JOBS = ["footprint_cluster", "vanilla", "lupin", "locden_iso",
        "tridesclous2", "mountainsort5"]
out = []
for nm in JOBS:
    d = os.path.join(ROOT, "hybrid_v2_rep0", "ks_" + nm)
    st = np.load(os.path.join(d, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
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
            if len(td) == 0:
                continue
            ti, off = pairs(td, tt)
            if len(ti) == 0:
                continue
            # cheap search: how many DISTINCT true spikes are reachable at s
            # offsets are (detected - true), so the CORRECTIVE shift is -off.
            # Ties are broken toward s = 0, otherwise a flat optimum (which is
            # what a correctly-aligned sorter has) reports an arbitrary shift.
            best_s, best_n = 0, -1
            for s in SHIFTS:
                m = np.abs(off + s) <= TOLERANCE
                if not m.any():
                    continue
                k = len(np.unique(ti[m]))
                if k > best_n or (k == best_n and abs(s) < abs(best_s)):
                    best_n, best_s = k, int(s)
            h, _ = _match(td + best_s, tt)       # exact, project's own rule
            if h > best[0]:
                best = (h, h / len(td), best_s, int(c))
        h, p, s, c = best
        rows.append(dict(unit=int(u), tier=tier, n_true=len(tt), off=s,
                         cluster=c, recall=h / len(tt), precision=p))
    a = pd.DataFrame(rows)
    rec = dict(sorter=nm, recall=a.recall.mean(),
               precision=float(np.nanmean(a.precision)),
               off_min=int(a.off.min()), off_max=int(a.off.max()),
               off_median=float(a.off.median()))
    for t in TIERS:
        x = a[a.tier == t]
        rec["rec_" + t] = x.recall.mean()
        rec["prec_" + t] = float(np.nanmean(x.precision))
    out.append(rec)
    print("  %-20s recall %.3f  precision %.3f   per-unit offsets %+d..%+d "
          "(median %+.0f)" % (nm, rec["recall"], rec["precision"],
          rec["off_min"], rec["off_max"], rec["off_median"]), flush=True)
    a.to_csv(os.path.join(ROOT, "outputs",
                          "per_unit_fair_%s_rep0.csv" % nm), index=False)

df = pd.DataFrame(out)
df.to_csv(os.path.join(ROOT, "outputs", "per_unit_fair_rep0.csv"), index=False)
print("\nPER TIER at per-unit alignment (recall/precision)")
for _, r in df.sort_values("recall", ascending=False).iterrows():
    print("  %-20s " % r.sorter + "   ".join(
        "%s %.3f/%.3f" % (t[:4], r["rec_"+t], r["prec_"+t]) for t in TIERS))
