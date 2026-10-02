"""Of the spikes the post-hoc tool recovered, which KIND were they?

Gil's question, which the 5am summary collapsed into one number: when the
correction pipeline ran on a Kilosort cluster, did it pull back spikes that
Kilosort had NEVER detected, or spikes Kilosort DID detect and filed under
somebody else's cluster? Those are different problems with different fixes and
the headline "259 recovered" hid the distinction.

Every injected spike absent from the unit's best-matching cluster falls into
exactly one of two classes:

  MISFILED   -- some Kilosort cluster has a spike within TOLERANCE of it, just
                not this one. The spike was detected; the attribution is wrong.
                Section 5ad measured this as 84% of all failures, and the v2
                union recall (94-96%) says the same thing.
  UNDETECTED -- no cluster anywhere has a spike there. Kilosort's detector
                genuinely never fired.

The recovery tool's value is completely different in the two cases. Recovering
an UNDETECTED spike is new information no amount of re-clustering could get.
Recovering a MISFILED one is re-attribution -- the spike was already in the
output, and a tool that moves it is competing with better clustering, not
adding to it.

This script re-derives both classes per placement and matches the accepted
candidates against each separately, so the 259 splits into its two parts and
the ceiling for each is visible.

Usage: python analyse_recovery_breakdown.py
"""
import os
import numpy as np
import pandas as pd

ROOT = r"D:\Gil\spike_sorting_agent"
OUT = os.path.join(ROOT, "outputs")
N_REPLICATES = 4
TOLERANCE = 10
CONFIGS = ["vanilla", "footprint_cluster_strong", "coarse_then_align"]


def _match(det_t, true_t, tolerance=TOLERANCE):
    det_t = np.asarray(det_t)
    if len(det_t) == 0 or len(true_t) == 0:
        return 0, np.full(len(true_t), -1)
    order = np.argsort(det_t)
    ds = det_t[order]
    used = np.zeros(len(ds), bool)
    hit, which = 0, []
    for t in np.sort(true_t):
        lo = np.searchsorted(ds, t - tolerance, "left")
        hi = np.searchsorted(ds, t + tolerance, "right")
        cand = [i for i in range(lo, hi) if not used[i]]
        if not cand:
            which.append(-1)
            continue
        j = min(cand, key=lambda i: abs(ds[i] - t))
        used[j] = True
        hit += 1
        which.append(int(order[j]))
    return hit, np.asarray(which)


cands = pd.read_csv(os.path.join(OUT, "v2_post_hoc_candidates.csv"))
rows = []

for rep in range(N_REPLICATES):
    d = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
    T = np.load(os.path.join(d, "hybrid_truth.npz"), allow_pickle=True)
    t_true, lab = T["times"].ravel(), T["labels"].ravel()
    tier_names = [str(x) for x in T["tier_names"]]
    tier_code = T["tier"].ravel()

    for cfg in CONFIGS:
        res = os.path.join(d, f"ks_{cfg}")
        if not os.path.exists(os.path.join(res, "spike_times.npy")):
            continue
        st = np.load(os.path.join(res, "spike_times.npy")).ravel()
        cl = np.load(os.path.join(res, "spike_clusters.npy")).ravel()
        o = np.argsort(st)
        st, cl = st[o], cl[o]

        for u in np.unique(lab):
            sel = lab == u
            tt = np.sort(t_true[sel])
            tier = tier_names[int(tier_code[sel][0])]

            cand_cl = set()
            for t in tt:
                cand_cl.update(cl[np.abs(st - t) <= TOLERANCE].tolist())
            best = None
            for c in sorted(cand_cl):
                h, _ = _match(st[cl == c], tt)
                if best is None or h > best[1]:
                    best = (int(c), h)
            if best is None or best[1] == 0:
                continue
            best_c = best[0]

            _, which = _match(st[cl == best_c], tt)
            missed = tt[which < 0]
            if not len(missed):
                continue

            # split the missed spikes: detected by SOMEONE, or by no one
            other = st[cl != best_c]
            _, w_other = _match(other, missed)
            misfiled = missed[w_other >= 0]
            undetected = missed[w_other < 0]

            # what the tool accepted for this placement
            cc = cands[(cands.config == cfg) & (cands.replicate == rep)
                       & (cands.unit == u)]
            acc = cc[(cc.r2 >= cc.bar_r2) & (cc.fp >= cc.bar_fp)]
            A = np.sort(acc.t.to_numpy())

            rec_mis, _ = _match(A, misfiled)
            rec_und, _ = _match(A, undetected)

            rows.append(dict(
                config=cfg, replicate=rep, unit=int(u), tier=tier,
                cluster=best_c, n_true=len(tt), n_missed=len(missed),
                n_misfiled=len(misfiled), n_undetected=len(undetected),
                n_accepted=len(A),
                rec_misfiled=int(rec_mis), rec_undetected=int(rec_und)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "v2_recovery_breakdown.csv"), index=False)

print("=" * 92)
print("WHAT KIND OF SPIKE DID THE POST-HOC TOOL RECOVER?")
print("MISFILED  = Kilosort detected it, filed it under another cluster")
print("UNDETECTED = no Kilosort cluster has a spike there at all")
print("=" * 92)

for cfg in CONFIGS:
    c = df[df.config == cfg]
    if not len(c):
        continue
    print(f"\n### {cfg}   ({len(c)} placements that had any missed spike)")
    g = c.groupby("tier").agg(
        n=("unit", "size"),
        missed=("n_missed", "sum"),
        misfiled=("n_misfiled", "sum"),
        undetected=("n_undetected", "sum"),
        rec_mis=("rec_misfiled", "sum"),
        rec_und=("rec_undetected", "sum"))
    g["pct_misfiled"] = (100 * g.misfiled / g.missed).round(1)
    g["rec_mis_pct"] = (100 * g.rec_mis / g.misfiled.clip(lower=1)).round(1)
    g["rec_und_pct"] = (100 * g.rec_und / g.undetected.clip(lower=1)).round(1)
    print(g.to_string())
    tm, tu = int(c.n_misfiled.sum()), int(c.n_undetected.sum())
    rm, ru = int(c.rec_misfiled.sum()), int(c.rec_undetected.sum())
    print(f"  POOLED: {tm + tu} missed = {tm} misfiled ({100*tm/(tm+tu):.1f}%) "
          f"+ {tu} undetected ({100*tu/(tm+tu):.1f}%)")
    print(f"          recovered {rm} of {tm} misfiled ({100*rm/max(1,tm):.1f}%) "
          f"and {ru} of {tu} undetected ({100*ru/max(1,tu):.1f}%)")
    print(f"          so of {rm+ru} recovered spikes, "
          f"{100*rm/max(1,rm+ru):.0f}% were re-attribution and "
          f"{100*ru/max(1,rm+ru):.0f}% were genuinely new detections")
print("\nsaved v2_recovery_breakdown.csv")
