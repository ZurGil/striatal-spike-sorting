"""Is a sorter's offset from truth ONE constant (a convention) or a WIDE spread
(poor timing)? The shift sweep cannot tell these apart; this can.

For each injected unit, take its matched cluster, pair each true spike with its
nearest detected spike within a generous +-25 samples, and report the
distribution of (detected - true)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Gil\spike_sorting_agent\post_ks_correction")
import score_all_fair as SAF
ROOT = r"D:\Gil\spike_sorting_agent"
W = 25

T = np.load(os.path.join(ROOT, "hybrid_v2_rep0", "hybrid_truth.npz"),
            allow_pickle=True)
t_true, lab = T["times"].ravel(), T["labels"].ravel()

rows = []
for nm, sh in [("footprint_cluster", 0), ("vanilla", 0), ("locden_iso", -14),
               ("tridesclous2", -14), ("lupin", -16), ("mountainsort5", -14)]:
    a, _ = SAF.score_shifted(0, nm, sh)       # which cluster matched each unit
    d = os.path.join(ROOT, "hybrid_v2_rep0", "ks_" + nm)
    st = np.load(os.path.join(d, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
    per_unit_med, allofs = [], []
    for _, r in a.iterrows():
        if r.matched_cluster < 0:
            continue
        td = np.sort(st[cl == r.matched_cluster])
        tt = t_true[lab == r.unit]
        if len(td) == 0:
            continue
        j = np.searchsorted(td, tt)
        j = np.clip(j, 1, len(td) - 1) if len(td) > 1 else np.zeros(len(tt), int)
        cand = np.stack([td[np.clip(j - 1, 0, len(td)-1)],
                         td[np.clip(j, 0, len(td)-1)]])
        off = cand - tt[None, :]
        pick = np.argmin(np.abs(off), axis=0)
        o = off[pick, np.arange(len(tt))]
        o = o[np.abs(o) <= W]
        if len(o) >= 20:
            per_unit_med.append(np.median(o))
            allofs.append(o)
    allofs = np.concatenate(allofs)
    pum = np.array(per_unit_med)
    # spread WITHIN a unit, after removing that unit's own median = pure jitter
    jit = np.concatenate([o - np.median(o) for o in
                          [x for x in np.split(allofs, np.cumsum(
                              [0]))[:0]]]) if False else None
    rows.append(dict(sorter=nm, sweep_shift=sh, n_units=len(pum),
                     median_offset=np.median(allofs),
                     unit_median_min=pum.min(), unit_median_max=pum.max(),
                     unit_median_iqr=np.subtract(*np.percentile(pum, [75, 25])),
                     spread_p5=np.percentile(allofs, 5),
                     spread_p95=np.percentile(allofs, 95),
                     mad=np.median(np.abs(allofs - np.median(allofs))),
                     frac_within_2=float(np.mean(
                         np.abs(allofs - np.median(allofs)) <= 2))))
    print("  %-20s n=%2d  median %+5.1f  per-unit medians %+.0f..%+.0f  "
          "p5..p95 %+.0f..%+.0f  MAD %.1f  within+-2 of own median %.0f%%"
          % (nm, len(pum), np.median(allofs), pum.min(), pum.max(),
             np.percentile(allofs, 5), np.percentile(allofs, 95),
             np.median(np.abs(allofs - np.median(allofs))),
             100*np.mean(np.abs(allofs - np.median(allofs)) <= 2)), flush=True)

pd.DataFrame(rows).to_csv(
    os.path.join(ROOT, "outputs", "timing_jitter_rep0.csv"), index=False)
