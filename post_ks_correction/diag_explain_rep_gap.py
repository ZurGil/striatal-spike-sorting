"""How much of the rep0-vs-rest gap does donor contamination explain?

Fit stock Kilosort's per-placement score on donor contamination alone, pooled
over all 52 placements, then compare the replicate means of the RESIDUALS. If
contamination is the whole story the residual means flatten out; whatever
remains is unexplained and is most likely just which 13 neurons were drawn.
"""
import os, numpy as np, pandas as pd
ROOT = r"D:\Gil\spike_sorting_agent"
a = pd.read_csv(os.path.join(ROOT, "outputs", "donor_amplitude_by_rep.csv"))
v = pd.read_csv(os.path.join(ROOT, "outputs", "allreps_per_unit_vanilla.csv"))
m = a.merge(v[["rep", "unit", "recall", "precision"]], on=["rep", "unit"])

for metric in ["recall", "precision"]:
    y = m[metric].values
    X = np.column_stack([np.ones(len(m)), m.contam.values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    print("\n%s  ~  contam:  slope %+.4f per %% contamination, R2 = %.3f"
          % (metric.upper(), beta[1], r2))
    raw = m.groupby("rep")[metric].mean()
    m["_r"] = resid
    adj = m.groupby("rep")["_r"].mean() + y.mean()
    print("  rep            " + "  ".join("rep%d" % r for r in range(4))
          + "   spread")
    print("  raw            " + "  ".join("%.3f" % v_ for v_ in raw)
          + "   %.3f" % (raw.max() - raw.min()))
    print("  contam-adjusted" + "  ".join("%.3f" % v_ for v_ in adj)
          + "   %.3f" % (adj.max() - adj.min()))
    print("  gap rep0 vs mean of rep1-3: raw %+.3f -> adjusted %+.3f "
          "(%.0f%% explained)"
          % (raw[0] - raw[1:].mean(), adj[0] - adj[1:].mean(),
             100 * (1 - abs(adj[0] - adj[1:].mean())
                    / abs(raw[0] - raw[1:].mean()))))

# the 18 donors used in more than one replicate: same neuron, different
# window and tier -- the within-neuron check the benchmark was built for
print("\nSAME DONOR IN MORE THAN ONE REPLICATE (within-neuron comparison)")
cnt = m.groupby("unit").rep.nunique()
rep_units = cnt[cnt > 1].index
mm = m[m.unit.isin(rep_units)]
print("  %d donors appear in %d placements across replicates"
      % (len(rep_units), len(mm)))
w = mm.pivot_table(index="unit", columns="rep", values="recall")
for pair in [(0, 2), (0, 3), (1, 3)]:
    s = w[[pair[0], pair[1]]].dropna()
    if len(s):
        print("  rep%d vs rep%d on the SAME %d neurons: %.3f vs %.3f "
              "(delta %+.3f)" % (pair[0], pair[1], len(s),
              s[pair[0]].mean(), s[pair[1]].mean(),
              s[pair[0]].mean() - s[pair[1]].mean()))
