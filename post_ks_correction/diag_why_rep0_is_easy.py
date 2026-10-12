"""Why is rep0 so much easier than rep1? Measure the thing most likely to
explain it: how big the injected spikes actually are.

The manifests already rule out the site: destination channels are REUSED
across replicates, so site noise (17.6-18.0 uV), largest neighbour
(108-113 uV), resident count (~20) and spike counts (3755-3787) are matched by
construction. The windows are evenly spaced (2000/4000/6000/8000 s) and rep1,
the worst, is second, so it is not drift either. What rotates is WHICH donor
neuron goes where, and rep0 and rep1 share only 1 donor of 13.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = r"D:\Gil\spike_sorting_agent"
NCH, NT, PRE = 384, 61, 20
NSNIP = 150

rows = []
for r in range(4):
    d = os.path.join(ROOT, "hybrid_v2_rep%d" % r)
    man = pd.read_csv(os.path.join(d, "manifest.csv")).set_index("unit")
    T = np.load(os.path.join(d, "hybrid_truth.npz"), allow_pickle=True)
    t, lab = T["times"].ravel(), T["labels"].ravel()
    n = T["n_samples"]
    x = np.memmap(os.path.join(d, "hybrid.bin"), dtype=np.int16, mode="r",
                  shape=(int(n), NCH))
    for u in np.unique(lab):
        tt = t[lab == u]
        ch = int(man.loc[int(u)].dest_peak_ch)
        tt = tt[(tt > PRE + 5) & (tt < n - NT)]
        if len(tt) > NSNIP:
            tt = tt[np.linspace(0, len(tt) - 1, NSNIP).astype(int)]
        snip = np.stack([np.asarray(x[a - PRE:a - PRE + NT, ch],
                                    dtype=np.float32) for a in tt])
        snip -= np.median(snip[:, :10], axis=1, keepdims=True)
        sta = snip.mean(0)
        noise = float(man.loc[int(u)].site_noise_uv)
        rows.append(dict(rep=r, unit=int(u), tier=str(man.loc[int(u)].tier),
                         contam=float(man.loc[int(u)].contam_pct),
                         sta_trough_uv=float(-sta.min() * 0.195),
                         sta_ptp_uv=float((sta.max() - sta.min()) * 0.195),
                         snr=float(-sta.min() * 0.195 / noise),
                         site_noise=noise))
    print("  rep%d measured" % r, flush=True)

a = pd.DataFrame(rows)
a.to_csv(os.path.join(ROOT, "outputs", "donor_amplitude_by_rep.csv"),
         index=False)
pd.set_option("display.width", 220)
print("\nDONOR SPIKE SIZE BY REPLICATE (spike-triggered average on the "
      "destination channel)")
g = a.groupby("rep").agg(trough_uv_mean=("sta_trough_uv", "mean"),
                         trough_uv_med=("sta_trough_uv", "median"),
                         snr_mean=("snr", "mean"), snr_med=("snr", "median"),
                         contam_mean=("contam", "mean"))
print(g.round(2).to_string())

# join to stock Kilosort per-placement recall
v = pd.read_csv(os.path.join(ROOT, "outputs", "allreps_per_unit_vanilla.csv"))
m = a.merge(v[["rep", "unit", "recall", "precision"]], on=["rep", "unit"])
print("\nWHAT PREDICTS STOCK KILOSORT'S RECALL, over all %d placements"
      % len(m))
for col in ["snr", "sta_trough_uv", "contam", "site_noise"]:
    rho, p = spearmanr(m[col], m.recall)
    rho2, p2 = spearmanr(m[col], m.precision)
    print("  %-14s vs recall rho %+.3f p %.4f | vs precision rho %+.3f "
          "p %.4f" % (col, rho, p, rho2, p2))

print("\nSNR BY REPLICATE AND TIER (STA trough / site noise)")
for r in range(4):
    s = m[m.rep == r]
    print("  rep%d  overall %.2f   " % (r, s.snr.mean()) + "  ".join(
        "%s %.2f" % (t[:4], s[s.tier == t].snr.mean())
        for t in ["easy", "collision", "pair", "hard"]))
