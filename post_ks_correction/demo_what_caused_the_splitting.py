"""
WHAT ACTUALLY CAUSED THE SPLITTING?

We know Kilosort cut each injected unit into 6-10 clusters, and we know
sub-sample alignment barely changed that (7.2 -> 6.8 fragments). So timing is
not the cause. This finds out what is, and it can be answered properly
because for these spikes we know the ground truth: every spike carrying the
same injected label came from the SAME neuron, so every split is by
definition an error, and we can ask what distinguishes the fragments.

WHAT KILOSORT'S SPLIT DECISION ACTUALLY TESTS (read from the installed
source, kilosort/swarmsplitter.py, function `split`). For each candidate
pair of branches it decides merge-or-keep-split in this order:

  1. tstat[kk,0] < 0.2                      -> keep split
  2. refractoriness(spikes1, spikes2)       -> CCG-based decision
  3. bimod_score of the 1-D projection:
         criterion = 2 * (score < 0.6) - 1  -> merge only if UNIMODAL
  4. tstat[kk,-1] > 0.15                    -> merge

Step 3 is the geometric heart of it: two groups stay separate if the spikes,
projected onto the direction that best separates them, look BIMODAL. Note
what that implies -- and it answers directly why 16x tighter timing changed
nothing. Shrinking the spread along the TIMING direction cannot merge two
groups whose bimodality lies along a DIFFERENT direction. If a unit's spikes
are bimodal in amplitude, the splitter will split them however perfectly
their timing is corrected.

ALSO FOUND, and it answers the "is there a fixed number of clusters"
question: `clustering_qr.cluster()` is declared with `nclust = 200` and both
call sites use that default. It seeds 200 clusters per spatial region with
kmeans++ (`kmeans_plusplus(Xg, niter=nclust)`) and then iterates. `nclust`
appears ZERO times in parameters.py, so it is not a user-visible setting.
Our runs produced 210 and 205 total clusters.

THE TEST HERE: for each injected unit, group its spikes by which output
cluster they landed in, then ask what separates the fragments --
  * AMPLITUDE (Kilosort's own per-spike amplitude)
  * TIME (drift: do fragments occupy different parts of the recording?)
  * TRUE SUB-SAMPLE SHIFT (the thing we corrected)
Reported as the between-fragment spread divided by the within-fragment
spread for each variable -- an F-like ratio. Whichever variable separates the
fragments is what the splitter latched onto.

Usage: python demo_what_caused_the_splitting.py
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = r"D:\Gil\spike_sorting_agent\hybrid_small"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
TRUTH = os.path.join(DATA_DIR, "hybrid_small_truth.npz")
TOLERANCE = 10
MIN_FRAGMENT = 15
CONFIGS = ["vanilla", "subsample_align"]

T = np.load(TRUTH)
t_true, lab_true, shift_true = T["times"].ravel(), T["labels"].ravel(), T["shifts"].ravel()
FS = float(T["fs"])

rows, frag_rows = [], []
for cfg in CONFIGS:
    d = os.path.join(DATA_DIR, f"ks_{cfg}")
    st = np.load(os.path.join(d, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
    amp = np.load(os.path.join(d, "amplitudes.npy")).ravel()

    print(f"\n{'='*78}\n{cfg}: {len(st):,} spikes, {len(np.unique(cl))} clusters")

    for u in np.unique(lab_true):
        tt = t_true[lab_true == u]
        ss = shift_true[lab_true == u]

        # match each injected spike to the nearest output spike, one-to-one
        used = np.zeros(len(st), dtype=bool)
        m_cl, m_amp, m_t, m_shift = [], [], [], []
        order = np.argsort(st)
        st_s, idx_s = st[order], order
        for t, s in zip(tt, ss):
            lo = np.searchsorted(st_s, t - TOLERANCE, side="left")
            hi = np.searchsorted(st_s, t + TOLERANCE, side="right")
            if hi <= lo:
                continue
            cand = idx_s[lo:hi]
            cand = cand[~used[cand]]
            if len(cand) == 0:
                continue
            j = cand[np.argmin(np.abs(st[cand] - t))]
            used[j] = True
            m_cl.append(int(cl[j])); m_amp.append(float(amp[j]))
            m_t.append(float(st[j])); m_shift.append(float(s))
        if not m_cl:
            continue
        df = pd.DataFrame(dict(cluster=m_cl, amplitude=m_amp, time=m_t,
                               true_shift=m_shift))

        sizes = df["cluster"].value_counts()
        frags = sizes[sizes >= MIN_FRAGMENT].index.tolist()
        if len(frags) < 2:
            print(f"  unit {u}: only {len(frags)} real fragment, nothing to explain")
            continue
        sub = df[df["cluster"].isin(frags)]

        def f_ratio(col):
            """between-fragment spread / within-fragment spread"""
            g = [sub.loc[sub["cluster"] == c, col].to_numpy() for c in frags]
            means = np.array([x.mean() for x in g])
            n = np.array([len(x) for x in g])
            grand = sub[col].mean()
            between = np.sum(n * (means - grand) ** 2) / max(len(g) - 1, 1)
            within = np.sum([((x - x.mean()) ** 2).sum() for x in g]) / max(len(sub) - len(g), 1)
            return float(between / within) if within > 0 else np.nan

        r_amp, r_time, r_shift = f_ratio("amplitude"), f_ratio("time"), f_ratio("true_shift")
        print(f"  unit {u}: {len(frags)} fragments covering {len(sub)} matched spikes")
        print(f"      separation by AMPLITUDE   F = {r_amp:>9.1f}")
        print(f"      separation by TIME        F = {r_time:>9.1f}")
        print(f"      separation by TRUE SHIFT  F = {r_shift:>9.1f}   <- what we corrected")
        winner = max([("amplitude", r_amp), ("time", r_time), ("true_shift", r_shift)],
                     key=lambda x: (x[1] if np.isfinite(x[1]) else -1))
        print(f"      -> fragments are separated mainly by {winner[0].upper()}")

        rows.append(dict(config=cfg, unit=int(u), n_fragments=len(frags),
                         n_matched=len(sub), F_amplitude=round(r_amp, 2),
                         F_time=round(r_time, 2), F_true_shift=round(r_shift, 3),
                         dominant=winner[0]))
        for c in frags:
            g = sub[sub["cluster"] == c]
            frag_rows.append(dict(config=cfg, unit=int(u), cluster=int(c), n=len(g),
                                  mean_amplitude=round(float(g["amplitude"].mean()), 2),
                                  mean_time_s=round(float(g["time"].mean() / FS), 1),
                                  mean_true_shift=round(float(g["true_shift"].mean()), 3)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "what_caused_splitting.csv"), index=False)
pd.DataFrame(frag_rows).to_csv(os.path.join(OUT, "what_caused_splitting_fragments.csv"),
                                index=False)
print(f"\n{'='*78}\nSUMMARY")
print(res.to_string(index=False))
print(f"\nmedian F by cause (vanilla):")
v = res[res["config"] == "vanilla"]
print(f"  amplitude  {v['F_amplitude'].median():>9.1f}")
print(f"  time       {v['F_time'].median():>9.1f}")
print(f"  true shift {v['F_true_shift'].median():>9.3f}")
print("\nsaved what_caused_splitting.csv, what_caused_splitting_fragments.csv")
