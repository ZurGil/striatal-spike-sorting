import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Gil\spike_sorting_agent\post_ks_correction")
from run_v2_comparison import score
TIERS = ["easy", "collision", "pair", "hard"]
for nm in ["vanilla", "footprint_cluster"]:
    print("=== %s" % nm)
    frames = []
    for r in range(4):
        try:
            a, _ = score(nm, r)
        except Exception as e:
            print("  rep%d missing (%s)" % (r, type(e).__name__)); continue
        a["rep"] = r
        frames.append(a)
        print("  rep%d  n=%2d  recall %.3f  precision %.3f   "
              % (r, len(a), a.recall.mean(), float(np.nanmean(a.precision)))
              + " ".join("%s %.2f/%.2f" % (t[:4], a[a.tier==t].recall.mean(),
                float(np.nanmean(a[a.tier==t].precision))) for t in TIERS),
              flush=True)
    d = pd.concat(frames)
    print("  POOLED over %d reps, %d placements: recall %.3f  precision %.3f"
          % (d.rep.nunique(), len(d), d.recall.mean(),
             float(np.nanmean(d.precision))))
    for t in TIERS:
        x = d[d.tier == t]
        per = x.groupby("rep").recall.mean()
        print("     %-10s n=%2d  recall %.3f  precision %.3f   "
              "per-rep recall %s" % (t, len(x), x.recall.mean(),
              float(np.nanmean(x.precision)),
              ", ".join("%.2f" % v for v in per)))
