"""
Run Kilosort4 on the TIERED hybrid dataset and score it per difficulty tier.

  python run_tiered_comparison.py vanilla
  python run_tiered_comparison.py subsample_align
  python run_tiered_comparison.py --compare [configs...]

Any config whose name starts with "vanilla" runs unpatched, which is what
makes a vanilla-vs-vanilla repeat possible -- the control that establishes
Kilosort's own run-to-run variability. On the previous dataset that was 0.0006
in overall recall (about 2 spikes), so real effects sit far above it, but it
has to be re-measured per dataset rather than assumed.

WHY PER-TIER SCORING. An average over all injected units hides exactly what
we need to see. A clean neuron on a quiet channel is found by every version of
the algorithm, so including it only dilutes. Reporting `easy`,
`noisy_channel`, `collision` and `pair` separately shows WHERE a change helps
and where it hurts -- and those can have opposite signs, which an average
turns into "no effect".

THE PAIR TIER IS SCORED DIFFERENTLY. Two different verified-good neurons sit
with heavily overlapping footprints, 3 channels apart. Correct behaviour is
TWO distinct clusters. So besides recall and precision, it reports whether
both members of a pair got matched to the SAME output cluster -- a merge
error, the failure this project's footprint work is aimed at. A merge error
can look like excellent recall, so it must be reported separately or it
reads as success.

THE COLLISION TIER also reports recall split by how close the collision was,
because a spike 4 samples from a neighbour is a different problem from one 25
samples away, and lumping them hides the gradient.

MUST BE RUN IN THE kilosort4 ENVIRONMENT for the run step:
  C:/Users/Adam/anaconda3/envs/kilosort4/python.exe run_tiered_comparison.py vanilla
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA_DIR = r"D:\Gil\spike_sorting_agent\hybrid_tiered"
DAT = os.path.join(DATA_DIR, "hybrid_tiered.bin")
TRUTH = os.path.join(DATA_DIR, "hybrid_tiered_truth.npz")
PROBE = os.path.join(DATA_DIR, "probe.json")
RESULTS = r"D:\Gil\spike_sorting_agent\outputs"
TOLERANCE = 10
MIN_FRAGMENT_SPIKES = 15
MIN_FRAGMENT_FRAC = 0.02


def _match(det_t, true_t, tolerance=TOLERANCE):
    """Greedy one-to-one match; returns (n_hit, matched_det_indices)."""
    det_t = np.asarray(det_t)
    order = np.argsort(det_t)
    ds = det_t[order]
    used = np.zeros(len(ds), bool)
    hit = 0
    which = []
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


def score(config):
    res_dir = os.path.join(DATA_DIR, f"ks_{config}")
    st = np.load(os.path.join(res_dir, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(res_dir, "spike_clusters.npy")).ravel()
    T = np.load(TRUTH, allow_pickle=True)
    t_true = T["times"].ravel()
    lab = T["labels"].ravel()
    tier_code = T["tier"].ravel()
    tier_names = [str(x) for x in T["tier_names"]]
    coll_off = T["collision_offset"].ravel()
    unit_pair = dict(zip([int(u) for u in T["source_units"]],
                         [int(p) for p in T["unit_pair_id"]]))

    rows = []
    for u in np.unique(lab):
        sel = lab == u
        tt = t_true[sel]
        tier = tier_names[int(tier_code[sel][0])]
        offs = coll_off[sel]

        cand = set()
        for t in tt:
            cand.update(cl[np.abs(st - t) <= TOLERANCE].tolist())
        min_share = max(MIN_FRAGMENT_SPIKES, int(MIN_FRAGMENT_FRAC * len(tt)))
        best, fragments = None, 0
        for c in sorted(cand):
            td = st[cl == c]
            h, _ = _match(td, tt)
            if h >= min_share:
                fragments += 1
            if best is None or h > best[1]:
                best = (int(c), h, len(td))
        if best is None:
            rows.append(dict(config=config, unit=int(u), tier=tier,
                             pair_id=unit_pair.get(int(u), -1), n_true=len(tt),
                             matched_cluster=-1, n_matched=0, recall=0.0,
                             precision=np.nan, n_fragments=0,
                             recall_close=np.nan, recall_far=np.nan))
            continue
        c, h, n_in = best

        # collision gradient: close (<=12 samples) vs far (>12)
        rc = rf = np.nan
        if tier == "collision" and np.isfinite(offs).any():
            td = st[cl == c]
            a = np.abs(offs)
            for name, mask in (("close", a <= 12), ("far", a > 12)):
                if mask.sum() >= 20:
                    hh, _ = _match(td, tt[mask])
                    if name == "close":
                        rc = hh / mask.sum()
                    else:
                        rf = hh / mask.sum()

        rows.append(dict(config=config, unit=int(u), tier=tier,
                         pair_id=unit_pair.get(int(u), -1), n_true=len(tt),
                         matched_cluster=c, n_matched=h,
                         recall=round(h / len(tt), 4),
                         precision=round(h / n_in, 4) if n_in else np.nan,
                         n_fragments=fragments,
                         recall_close=round(rc, 4) if np.isfinite(rc) else np.nan,
                         recall_far=round(rf, 4) if np.isfinite(rf) else np.nan))

    df = pd.DataFrame(rows)
    df["n_total_clusters"] = len(np.unique(cl))

    # pair merge errors: did both members land on the same output cluster?
    merge_err = []
    for p in sorted({v for v in unit_pair.values() if v >= 0}):
        members = df[df.pair_id == p]
        if len(members) == 2:
            same = members.matched_cluster.iloc[0] == members.matched_cluster.iloc[1]
            merge_err.append(dict(config=config, pair_id=p,
                                  units=list(members.unit),
                                  clusters=list(members.matched_cluster),
                                  merged=bool(same)))
    return df, pd.DataFrame(merge_err)


def run(config):
    from kilosort import run_kilosort
    if not config.startswith("vanilla"):
        import ks_patches
        ks_patches.enable(config)
        print(f"[patch] {config}")
    else:
        print("[patch] none -- stock Kilosort4")
    res_dir = os.path.join(DATA_DIR, f"ks_{config}")
    os.makedirs(res_dir, exist_ok=True)
    with open(PROBE) as f:
        p = json.load(f)
    probe = dict(chanMap=np.array(p["chanMap"]),
                 xc=np.array(p["xc"], dtype="float32"),
                 yc=np.array(p["yc"], dtype="float32"),
                 kcoords=np.array(p["kcoords"], dtype="float32"),
                 n_chan=int(p["n_chan"]))
    t0 = time.time()
    run_kilosort(settings=dict(n_chan_bin=384, fs=30000, nblocks=0), probe=probe,
                 filename=DAT, results_dir=res_dir, do_CAR=True,
                 save_preprocessed_copy=False)
    print(f"[run] {config} finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    if args[0] == "--compare":
        cfgs = args[1:] or ["vanilla"]
        frames, merges = [], []
        for c in cfgs:
            try:
                d, m = score(c)
                frames.append(d)
                merges.append(m)
            except FileNotFoundError:
                print(f"  (no results for '{c}')")
        df = pd.concat(frames)
        mg = pd.concat(merges) if merges else pd.DataFrame()
        df.to_csv(os.path.join(RESULTS, "tiered_comparison_scores.csv"), index=False)
        if len(mg):
            mg.to_csv(os.path.join(RESULTS, "tiered_pair_merges.csv"), index=False)

        print("\n================ RECALL BY TIER ================")
        piv = df.pivot_table(index="tier", columns="config", values="recall",
                             aggfunc="mean")
        print(piv.round(4).to_string())
        print("\n================ PRECISION BY TIER ================")
        print(df.pivot_table(index="tier", columns="config", values="precision",
                             aggfunc="mean").round(4).to_string())
        print("\n================ FRAGMENTS BY TIER ================")
        print(df.pivot_table(index="tier", columns="config", values="n_fragments",
                             aggfunc="mean").round(2).to_string())
        coll = df[df.tier == "collision"]
        if len(coll):
            print("\n===== COLLISION TIER, by how close the collision was =====")
            print(coll.pivot_table(index="config",
                                   values=["recall_close", "recall_far"],
                                   aggfunc="mean").round(4).to_string())
        if len(mg):
            print("\n===== PAIR TIER: merge errors (two neurons -> one cluster) =====")
            print(mg.to_string(index=False))
            print("\nmerge-error rate by config:")
            print(mg.groupby("config")["merged"].mean().round(3).to_string())
        print("\n================ OVERALL ================")
        agg = df.groupby("config").agg(
            recall=("recall", "mean"), precision=("precision", "mean"),
            fragments=("n_fragments", "mean"),
            clusters=("n_total_clusters", "first"))
        print(agg.round(4).to_string())
        print("\nsaved tiered_comparison_scores.csv")
    else:
        run(args[0])
        d, m = score(args[0])
        print(f"\nSCORE for '{args[0]}' by tier:")
        print(d.groupby("tier")[["recall", "precision", "n_fragments"]]
              .mean().round(4).to_string())
        if len(m):
            print(f"\npair merge errors: {int(m.merged.sum())} of {len(m)}")
