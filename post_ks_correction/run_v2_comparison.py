"""
Run and score every configuration across all replicates of the v2 benchmark.

WHY REPLICATES. The first benchmark had 13 units, 3 per tier, and its tier
means were therefore n=3 statistics: the easy tier's three units scored 0.30,
0.45 and 0.86 (section 5ag), a spread far more likely to be unit identity than
tier difficulty. One 120 s recording cannot hold many more units without
placing them close enough to defeat the tier design -- the probe only offers
about 8 quiet sites and 9 loud ones at 22-channel spacing. So the extra
statistics come from four replicates on four different windows of the session,
with units ROTATED through tiers between replicates. That makes tier effects
measurable within unit rather than only across units.

WHAT IS REPORTED
  * recall AND precision per tier, pooled over replicates, with the spread
    across replicates -- a mean with no spread hides exactly the instability
    that made the first benchmark hard to read.
  * per-replicate tables, so a result that only appears in one window is
    visible as such.
  * pair merge errors separately, because a merge looks like excellent recall.
  * every per-unit row saved with its MEASURED site properties (noise, largest
    neighbour amplitude, density), so performance can be regressed on what the
    site actually was instead of only on its tier label.

Usage:
  python run_v2_comparison.py --run <config> <replicate>
  python run_v2_comparison.py --compare <config> [<config> ...]
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = r"D:\Gil\spike_sorting_agent"
RESULTS = os.path.join(ROOT, "outputs")
N_REPLICATES = 4
TOLERANCE = 10
MIN_FRAGMENT_SPIKES = 15
MIN_FRAGMENT_FRAC = 0.02


def paths(rep):
    d = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
    return (d, os.path.join(d, "hybrid.bin"), os.path.join(d, "hybrid_truth.npz"),
            os.path.join(d, "probe.json"), os.path.join(d, "manifest.csv"))


def _match(det_t, true_t, tolerance=TOLERANCE):
    """Greedy one-to-one match; returns (n_hit, matched_det_indices)."""
    det_t = np.asarray(det_t)
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


def score(config, rep):
    data_dir, _, truth, _, man_path = paths(rep)
    res_dir = os.path.join(data_dir, f"ks_{config}")
    st = np.load(os.path.join(res_dir, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(res_dir, "spike_clusters.npy")).ravel()
    T = np.load(truth, allow_pickle=True)
    t_true, lab = T["times"].ravel(), T["labels"].ravel()
    tier_code = T["tier"].ravel()
    tier_names = [str(x) for x in T["tier_names"]]
    coll_off = T["collision_offset"].ravel()
    unit_pair = dict(zip([int(u) for u in T["source_units"]],
                         [int(p) for p in T["unit_pair_id"]]))
    man = pd.read_csv(man_path).set_index("unit")

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

        m = man.loc[int(u)]
        base = dict(config=config, replicate=rep, unit=int(u), tier=tier,
                    pair_id=unit_pair.get(int(u), -1), n_true=len(tt),
                    site_noise_uv=float(m.site_noise_uv),
                    site_max_neighbor_uv=float(m.site_max_neighbor_uv),
                    site_density=int(m.site_density),
                    dest_ntrode=int(m.dest_ntrode),
                    contam_pct=float(m.contam_pct))
        if best is None:
            rows.append(dict(base, matched_cluster=-1, n_matched=0, recall=0.0,
                             precision=np.nan, n_fragments=0,
                             recall_close=np.nan, recall_far=np.nan))
            continue
        c, h, n_in = best

        rc = rf = np.nan
        if tier == "collision" and np.isfinite(offs).any():
            td = st[cl == c]
            for close, name in ((np.abs(offs) <= 12, "close"),
                                (np.abs(offs) > 12, "far")):
                mask = close & np.isfinite(offs)
                if mask.sum():
                    hh, _ = _match(td, tt[mask])
                    if name == "close":
                        rc = hh / mask.sum()
                    else:
                        rf = hh / mask.sum()

        rows.append(dict(base, matched_cluster=c, n_matched=h,
                         recall=round(h / len(tt), 4),
                         precision=round(h / n_in, 4) if n_in else np.nan,
                         n_fragments=fragments,
                         recall_close=round(rc, 4) if np.isfinite(rc) else np.nan,
                         recall_far=round(rf, 4) if np.isfinite(rf) else np.nan))

    df = pd.DataFrame(rows)
    df["n_total_clusters"] = len(np.unique(cl))

    merge_err = []
    for p in sorted({v for v in unit_pair.values() if v >= 0}):
        mem = df[df.pair_id == p]
        if len(mem) == 2:
            merge_err.append(dict(
                config=config, replicate=rep, pair_id=p,
                units=list(mem.unit), clusters=list(mem.matched_cluster),
                merged=bool(mem.matched_cluster.iloc[0]
                            == mem.matched_cluster.iloc[1])))
    return df, pd.DataFrame(merge_err)


def run(config, rep):
    from kilosort import run_kilosort
    data_dir, dat, _, probe_json, _ = paths(rep)
    if not config.startswith("vanilla"):
        import ks_patches
        ks_patches.enable(config)
        print(f"[patch] {config}")
    else:
        print("[patch] none -- stock Kilosort4")
    res_dir = os.path.join(data_dir, f"ks_{config}")
    os.makedirs(res_dir, exist_ok=True)
    with open(probe_json) as f:
        p = json.load(f)
    probe = dict(chanMap=np.array(p["chanMap"]),
                 xc=np.array(p["xc"], dtype="float32"),
                 yc=np.array(p["yc"], dtype="float32"),
                 kcoords=np.array(p["kcoords"], dtype="float32"),
                 n_chan=int(p["n_chan"]))
    t0 = time.time()
    run_kilosort(settings=dict(n_chan_bin=384, fs=30000, nblocks=0), probe=probe,
                 filename=dat, results_dir=res_dir, do_CAR=True,
                 save_preprocessed_copy=False)
    print(f"[run] {config} rep{rep} finished in {time.time()-t0:.0f}s")


def compare(cfgs):
    frames, merges = [], []
    for c in cfgs:
        for r in range(N_REPLICATES):
            try:
                d, m = score(c, r)
                frames.append(d)
                if len(m):
                    merges.append(m)
            except FileNotFoundError:
                print(f"  (missing: {c} rep{r})")
    if not frames:
        raise SystemExit("no results found")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "v2_scores.csv"), index=False)

    n_rep = df.groupby("config").replicate.nunique()
    print(f"\nreplicates scored per config:\n{n_rep.to_string()}")
    print(f"total unit-placements scored: {len(df)}")

    for metric in ("recall", "precision"):
        print(f"\n{'='*86}\n{metric.upper()} BY TIER "
              f"(mean over all units and replicates)")
        piv = df.pivot_table(index="tier", columns="config", values=metric,
                             aggfunc="mean")
        print(piv.round(4).to_string())
        print(f"\n{metric} SPREAD ACROSS REPLICATES (std of per-replicate means)")
        per = df.groupby(["config", "replicate", "tier"])[metric].mean()
        print(per.groupby(level=["config", "tier"]).std().unstack(0)
              .round(4).to_string())

    print(f"\n{'='*86}\nOVERALL")
    agg = df.groupby("config").agg(
        recall=("recall", "mean"), precision=("precision", "mean"),
        fragments=("n_fragments", "mean"), n=("recall", "size"))
    print(agg.round(4).sort_values("recall", ascending=False).to_string())

    print(f"\nPER-REPLICATE OVERALL RECALL (is a result stable?)")
    print(df.pivot_table(index="replicate", columns="config", values="recall",
                         aggfunc="mean").round(4).to_string())

    if merges:
        mg = pd.concat(merges, ignore_index=True)
        mg.to_csv(os.path.join(RESULTS, "v2_merge_errors.csv"), index=False)
        print(f"\n{'='*86}\nPAIR MERGE ERRORS (two neurons -> one cluster)")
        print(mg.groupby("config").merged.agg(["sum", "size"]).to_string())

    # the point of recording site properties: separate WHY a site is hard
    print(f"\n{'='*86}\nWHAT ACTUALLY PREDICTS RECALL? (vanilla, Spearman)")
    v = df[df.config == "vanilla"]
    if len(v) > 3:
        for col in ("site_max_neighbor_uv", "site_density", "site_noise_uv",
                    "contam_pct"):
            print(f"  {col:<24} {v.recall.corr(v[col], method='spearman'):+.3f}")
    print("\nsaved v2_scores.csv")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        raise SystemExit(1)
    if a[0] == "--run":
        run(a[1], int(a[2]))
    elif a[0] == "--compare":
        compare(a[1:])
    else:
        print(__doc__)
        raise SystemExit(1)
