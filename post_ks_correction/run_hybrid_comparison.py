"""
Run Kilosort4 on the small hybrid dataset and score it against the injected
ground truth. This is the grading harness for any change to Kilosort.

  python run_hybrid_comparison.py vanilla
  python run_hybrid_comparison.py subsample_align
  python run_hybrid_comparison.py --compare

"vanilla" runs stock Kilosort4 with nothing patched. Any other config name
applies the corresponding patch from ks_patches.py before running, and the
patch is flag-gated so that running with it disabled must reproduce vanilla
exactly.

SCORING. Only the injected units are graded -- the background's own neurons
have no ground truth and are there to make the problem realistic, not to be
counted. For each injected unit we find the output cluster that matches it
best (most one-to-one matched spikes within TOLERANCE samples) and report:

  recall    = matched / injected        -- did it find the spikes?
  precision = matched / cluster size    -- is that cluster pure?
  n_clusters_touching = how many output clusters contain any matched spike,
                        which is the oversplit indicator -- the specific
                        failure sub-sample jitter is predicted to cause

Reporting all three matters. Splitting one injected neuron across three
clusters shows up as poor recall with good precision and a high touching
count; merging it with a background neuron shows up as good recall with poor
precision. A single accuracy number would hide which happened.

MUST BE RUN IN THE kilosort4 ENVIRONMENT:
  C:/Users/Adam/anaconda3/envs/kilosort4/python.exe run_hybrid_comparison.py vanilla
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA_DIR = r"D:\Gil\spike_sorting_agent\hybrid_small"
DAT = os.path.join(DATA_DIR, "hybrid_small.bin")
TRUTH = os.path.join(DATA_DIR, "hybrid_small_truth.npz")
PROBE = os.path.join(DATA_DIR, "probe.json")
RESULTS = r"D:\Gil\spike_sorting_agent\outputs"
TOLERANCE = 10


def score(config):
    """Score an already-completed Kilosort run for `config`."""
    from ground_truth_harness import match_detections
    res_dir = os.path.join(DATA_DIR, f"ks_{config}")
    st = np.load(os.path.join(res_dir, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(res_dir, "spike_clusters.npy")).ravel()
    T = np.load(TRUTH)
    t_true, lab = T["times"].ravel(), T["labels"].ravel()

    rows = []
    for u in np.unique(lab):
        tt = np.sort(t_true[lab == u])
        # candidate clusters: any cluster with a spike near a true spike
        cand = set()
        for t in tt:
            cand.update(cl[np.abs(st - t) <= TOLERANCE].tolist())
        best, touching = None, 0
        for c in cand:
            td = np.sort(st[cl == c])
            mr = match_detections(td, tt, tolerance=TOLERANCE)
            if mr["n_hit"] > 0:
                touching += 1
            if best is None or mr["n_hit"] > best[1]["n_hit"]:
                best = (int(c), mr, len(td))
        if best is None:
            rows.append(dict(config=config, injected_unit=int(u), n_true=len(tt),
                             matched_cluster=-1, n_in_cluster=0, n_matched=0,
                             recall=0.0, precision=np.nan, n_clusters_touching=0))
            continue
        c, mr, n_in = best
        rows.append(dict(config=config, injected_unit=int(u), n_true=len(tt),
                         matched_cluster=c, n_in_cluster=n_in,
                         n_matched=mr["n_hit"],
                         recall=round(mr["n_hit"] / len(tt), 4),
                         precision=round(mr["n_hit"] / n_in, 4) if n_in else np.nan,
                         n_clusters_touching=touching))
    df = pd.DataFrame(rows)
    df["n_total_clusters"] = len(np.unique(cl))
    df["n_total_spikes"] = len(st)
    return df


def run(config):
    from kilosort import run_kilosort
    from kilosort.io import load_probe

    if config != "vanilla":
        import ks_patches
        ks_patches.enable(config)
        print(f"[patch] enabled: {config}")
    else:
        print("[patch] none -- stock Kilosort4")

    res_dir = os.path.join(DATA_DIR, f"ks_{config}")
    os.makedirs(res_dir, exist_ok=True)
    with open(PROBE) as f:
        p = json.load(f)
    probe = dict(chanMap=np.array(p["chanMap"]), xc=np.array(p["xc"], dtype="float32"),
                 yc=np.array(p["yc"], dtype="float32"),
                 kcoords=np.array(p["kcoords"], dtype="float32"),
                 n_chan=int(p["n_chan"]))

    settings = dict(n_chan_bin=384, fs=30000, nblocks=0)
    t0 = time.time()
    out = run_kilosort(settings=settings, probe=probe, filename=DAT,
                       results_dir=res_dir, do_CAR=True, save_preprocessed_copy=False)
    print(f"[run] {config} finished in {time.time()-t0:.0f}s")
    return res_dir


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)

    if args[0] == "--compare":
        frames = []
        for cfg in args[1:] or ["vanilla", "subsample_align"]:
            try:
                frames.append(score(cfg))
            except FileNotFoundError:
                print(f"  (no results yet for '{cfg}')")
        if not frames:
            raise SystemExit("nothing to compare")
        df = pd.concat(frames)
        df.to_csv(os.path.join(RESULTS, "hybrid_comparison_scores.csv"), index=False)
        print("\nPER INJECTED UNIT")
        print(df[["config", "injected_unit", "n_true", "n_matched", "recall",
                  "precision", "n_clusters_touching"]].to_string(index=False))
        print("\nPOOLED BY CONFIG")
        agg = df.groupby("config").agg(
            mean_recall=("recall", "mean"), mean_precision=("precision", "mean"),
            total_matched=("n_matched", "sum"), total_true=("n_true", "sum"),
            mean_touching=("n_clusters_touching", "mean"),
            n_clusters=("n_total_clusters", "first"),
            n_spikes=("n_total_spikes", "first"))
        agg["overall_recall"] = (agg["total_matched"] / agg["total_true"]).round(4)
        print(agg.to_string())
        print(f"\nsaved hybrid_comparison_scores.csv")
    else:
        cfg = args[0]
        run(cfg)
        df = score(cfg)
        print(f"\nSCORE for '{cfg}':")
        print(df[["injected_unit", "n_true", "n_matched", "recall", "precision",
                  "n_clusters_touching"]].to_string(index=False))
        print(f"\noverall recall = {df['n_matched'].sum()/df['n_true'].sum():.4f}"
              f" | mean precision = {df['precision'].mean():.4f}"
              f" | total clusters found = {df['n_total_clusters'].iloc[0]}")
