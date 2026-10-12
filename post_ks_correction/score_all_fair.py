"""Score every sorter at its OWN best time-alignment, because the benchmark's
time convention is not the physical trough and not every sorter shares it.

THE CORRECTION THIS SCRIPT EXISTS FOR. The benchmark's truth times are the
nt0min=20 index of the 61-sample extraction window, which measurement puts a
median of +13 samples BEFORE the physical trough of the injected waveform
(range +8 to +17 over the 13 rep0 units; one unit, 331, is inverted at -1).
Kilosort uses the same nt0min convention, which is why truth and Kilosort
agree at a median offset of 0 -- but that agreement never implied either sits
at the trough, and an earlier note in this project read it that way.

Consequence: a sorter that reports the physical trough, as MountainSort5 and
tridesclous2 both do, is shifted ~13 samples from the benchmark's convention,
which is outside the scorer's 10-sample matching tolerance. Scored naively:

    tridesclous2   recall 0.276  precision 0.492
    mountainsort5  recall 0.089  precision 0.221

Scored at shift -14:

    tridesclous2   recall 0.562  precision 0.900
    mountainsort5  recall 0.323  precision 0.691

Those naive numbers are bookkeeping, not sorting, and publishing them would
misrepresent both sorters.

IS SWEEPING THE SHIFT CHEATING? It uses the ground truth, so it has to be
justified rather than waved through. It is a one-integer-per-sorter
convention calibration, not parameter tuning: it cannot invent spikes, cannot
change which clusters exist, and cannot trade recall against precision -- a
wrong shift lowers BOTH, as the sweeps show. The optimum is also a broad
plateau (for tridesclous2, shifts -16 through -10 all land within 0.003
recall), so it is identifying a convention rather than fitting noise. Both
columns are reported so the size of the correction is always visible, and a
sorter whose best shift is 0 is shown to need none.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_v2_comparison import score

ROOT = r"D:\Gil\spike_sorting_agent"
SHIFTS = list(range(-20, 7, 2))
TIERS = ["easy", "collision", "pair", "hard"]


def available(rep):
    d = os.path.join(ROOT, "hybrid_v2_rep%d" % rep)
    out = []
    for nm in sorted(os.listdir(d)):
        if not nm.startswith("ks_"):
            continue
        p = os.path.join(d, nm)
        if (os.path.isfile(os.path.join(p, "spike_times.npy"))
                and os.path.isfile(os.path.join(p, "spike_clusters.npy"))):
            out.append(nm[3:])
    return out


def score_shifted(rep, name, shift):
    d = os.path.join(ROOT, "hybrid_v2_rep%d" % rep)
    st = np.load(os.path.join(d, "ks_" + name, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(d, "ks_" + name, "spike_clusters.npy")).ravel()
    if shift == 0:
        return score(name, rep)
    tmp = os.path.join(d, "ks_ZZtmp")
    os.makedirs(tmp, exist_ok=True)
    t2 = st.astype(np.int64) + shift
    k = (t2 > 0) & (t2 < 3600000)
    o = np.argsort(t2[k])
    np.save(os.path.join(tmp, "spike_times.npy"), t2[k][o])
    np.save(os.path.join(tmp, "spike_clusters.npy"),
            cl[k][o].astype(np.int32))
    return score("ZZtmp", rep)


def main(rep, names):
    rows = []
    for nm in names:
        d = os.path.join(ROOT, "hybrid_v2_rep%d" % rep, "ks_" + nm)
        nsp = len(np.load(os.path.join(d, "spike_times.npy")))
        ncl = len(np.unique(np.load(os.path.join(d, "spike_clusters.npy"))))
        best = None
        raw = None
        for sh in SHIFTS:
            try:
                a, _ = score_shifted(rep, nm, sh)
            except Exception as e:
                print("  %-26s shift %+3d FAILED %s" % (nm, sh, e))
                continue
            f1 = 2 * a.recall.mean() * np.nanmean(a.precision) / max(
                a.recall.mean() + np.nanmean(a.precision), 1e-9)
            rec = dict(config=nm, shift=sh, spikes=nsp, clusters=ncl,
                       recall=a.recall.mean(),
                       precision=float(np.nanmean(a.precision)), f1=f1,
                       frag=a.n_fragments.mean())
            for t in TIERS:
                x = a[a.tier == t]
                rec["rec_" + t] = x.recall.mean()
                rec["prec_" + t] = float(np.nanmean(x.precision))
            if sh == 0:
                raw = rec
            if best is None or f1 > best["f1"]:
                best = rec
        if best is None:
            continue
        best["raw_recall"] = raw["recall"] if raw else np.nan
        best["raw_precision"] = raw["precision"] if raw else np.nan
        rows.append(best)
        print("  %-26s best shift %+3d  recall %.3f  precision %.3f "
              "(unshifted %.3f / %.3f)"
              % (nm, best["shift"], best["recall"], best["precision"],
                 best["raw_recall"], best["raw_precision"]), flush=True)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(os.path.join(ROOT, "outputs", "score_all_fair_rep%d.csv" % rep),
              index=False)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("ALL SORTERS AT THEIR OWN BEST TIME ALIGNMENT -- rep%d, "
          "13 placements, identical matching rule" % rep)
    print("=" * 110)
    print(df[["config", "shift", "spikes", "clusters", "recall", "precision",
              "f1", "frag", "raw_recall", "raw_precision"]]
          .round(3).to_string(index=False))
    print("\nPER TIER (recall / precision)")
    for _, r in df.iterrows():
        print("  %-26s " % r.config + "  ".join(
            "%s %.3f/%.3f" % (t[:4], r["rec_" + t], r["prec_" + t])
            for t in TIERS))


if __name__ == "__main__":
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    names = sys.argv[2:] or [n for n in available(rep)
                             if not n.startswith(("ZZtmp", "locden_cache",
                                                  "tmpshift"))]
    print("scoring %d configurations on rep%d" % (len(names), rep))
    main(rep, names)
