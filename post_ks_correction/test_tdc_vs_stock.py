"""Paired Wilcoxon on the same 52 placements, the project's standard test, plus
the alignment control run POOLED rather than only on rep0.

The control matters because tridesclous2 is scored with 33 shifts of per-unit
freedom and stock Kilosort at a fixed 0. If that freedom were worth anything to
a sorter already on the benchmark's convention, the comparison would be rigged.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, (r"C:\Users\Adam\AppData\Local\Temp\claude"
                    r"\D--Gil-spike-sorting-agent"
                    r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad"))
ROOT = r"D:\Gil\spike_sorting_agent"
import tdc_all_reps as M      # reuse score_rep / pairs

# ---- control: does per-unit freedom help a sorter that needs none?
print("ALIGNMENT CONTROL, pooled over 4 replicates")
for nm in ["vanilla", "footprint_cluster"]:
    fr0, fr1 = [], []
    for rep in range(4):
        fr0.append(M.score_rep(nm, rep, False))
        fr1.append(M.score_rep(nm, rep, True))
    a, b = pd.concat(fr0), pd.concat(fr1)
    print("  %-18s fixed 0: %.4f/%.4f   per-unit: %.4f/%.4f   gain "
          "%+.4f/%+.4f" % (nm, a.recall.mean(), np.nanmean(a.precision),
          b.recall.mean(), np.nanmean(b.precision),
          b.recall.mean() - a.recall.mean(),
          np.nanmean(b.precision) - np.nanmean(a.precision)), flush=True)

# ---- paired test, tridesclous2 vs stock, same placements
t = pd.read_csv(os.path.join(ROOT, "outputs",
                             "allreps_per_unit_tridesclous2.csv"))
v = pd.read_csv(os.path.join(ROOT, "outputs", "allreps_per_unit_vanilla.csv"))
k = ["rep", "unit"]
m = t.merge(v, on=k, suffixes=("_t", "_v"))
print("\nPAIRED WILCOXON, tridesclous2 vs stock Kilosort4, n=%d placements"
      % len(m))
for metric in ["recall", "precision"]:
    x, y = m[metric + "_t"].values, m[metric + "_v"].values
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    st, p = wilcoxon(x[ok], y[ok])
    print("  %-10s delta %+.3f  (median %+.3f)  p = %.2e  n=%d  "
          "better in %d of %d placements"
          % (metric, d.mean(), np.median(d), p, ok.sum(), (d > 0).sum(),
             ok.sum()))
# per replicate, so a single window cannot carry it
print("\n  precision by replicate (tridesclous2 vs stock):")
for rep in range(4):
    s = m[m.rep == rep]
    print("    rep%d  %.3f vs %.3f  (better in %d of %d)"
          % (rep, np.nanmean(s.precision_t), np.nanmean(s.precision_v),
             int((s.precision_t > s.precision_v).sum()), len(s)))
print("\n  pair tier only, the audit's worst cell:")
s = m[m.tier_t == "pair"]
st, p = wilcoxon(s.precision_t, s.precision_v)
print("    precision %.3f vs %.3f, delta %+.3f, p = %.2e, n=%d"
      % (np.nanmean(s.precision_t), np.nanmean(s.precision_v),
         np.nanmean(s.precision_t - s.precision_v), p, len(s)))
