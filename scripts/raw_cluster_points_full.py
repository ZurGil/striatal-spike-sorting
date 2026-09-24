"""
Best-chunk cluster point-cloud (PC1, ptp, own/other/noise assignment) for all 401
units, for the interactive HTML review tool. Subsampled per category to keep the
published page's size reasonable. Runs independently of raw_separation_metric_full.py
(that one computes 10-chunk separation scores; this one just needs one representative
chunk's actual points per unit for visualization).
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig
from striatal_agent.session import discover_session
from striatal_agent.raw_io import RawReader, highpass_filter

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
OUT_PATH = os.path.join(OUT, "raw_cluster_points_full_401.json")
PROGRESS_PATH = os.path.join(OUT, "raw_cluster_points_full_401_progress.txt")

FS = 30000
N_CHUNKS = 5
CHUNK_SEC = 120
NT, NT0MIN = 60, 20
DEAD_SAMPLES = int(0.001 * FS)
MATCH_WINDOW = int(0.0005 * FS)
MAX_PTS_PER_CATEGORY = 250  # subsample cap for page size


def detect(reader, cfg, chan, start, dur_samples):
    block, s, e = reader.read_window(start - 200, start + dur_samples + 200, chans=[chan])
    filt = highpass_filter(block, FS, cfg.highpass_hz, axis=0)[:, 0]
    trace = filt[200:200 + dur_samples]
    mad = np.median(np.abs(trace - np.median(trace))) * 1.4826
    thresh = 4.5 * mad
    below = trace < -thresh
    crossings = np.where(below & ~np.roll(below, 1))[0]
    keep, last = [], -DEAD_SAMPLES
    for c in crossings:
        if c - last >= DEAD_SAMPLES:
            keep.append(c); last = c
    crossings = np.array(keep)
    crossings = crossings[(crossings > NT0MIN + 2) & (crossings < len(trace) - (NT - NT0MIN) - 2)]
    if len(crossings) < 20:
        return None
    abs_times = crossings + start
    snips = np.array([trace[c - NT0MIN:c - NT0MIN + NT] for c in crossings])
    ptp = snips.max(axis=1) - snips.min(axis=1)
    teo_max = np.array([(s[1:-1]**2 - s[:-2]*s[2:]).max() for s in snips])
    # same additional axes validated and added to raw_separation_metric_full.py -- see
    # conversation: psi evaluated at the detector's own trough sample (complementary to
    # teo_max's whole-window search), and the lag-2 generalization of teo_max.
    trough_idx = np.clip(np.argmin(snips, axis=1), 1, NT - 2)
    psi_trough = np.array([s[i]**2 - s[i-1]*s[i+1] for s, i in zip(snips, trough_idx)])
    teo2 = np.array([(s[2:-2]**2 - s[:-4]*s[4:]).max() for s in snips])
    X = snips - snips.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return dict(abs_times=abs_times, pc1=X @ Vt[0], pc2=X @ Vt[1], ptp=ptp, teo_max=teo_max,
                psi_trough=psi_trough, teo2=teo2)


def subsample(idx_bool, rng, cap=MAX_PTS_PER_CATEGORY):
    idx = np.where(idx_bool)[0]
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    return idx


def main():
    t0 = time.time()
    cfg = AgentConfig()
    paths = discover_session(SESSION_ROOT)
    ko_dir = paths.kilosort_original_dir
    spike_times_all = np.load(os.path.join(ko_dir, "spike_times.npy")).ravel()
    spike_clusters_all = np.load(os.path.join(ko_dir, "spike_clusters.npy")).ravel()
    reader = RawReader(paths.probe_dat, cfg.n_chan)

    with open(os.path.join(OUT, "corrected_classification_with_waveforms.json")) as f:
        cls_rows = json.load(f)
    by_id = {u["unit_id"]: u for u in cls_rows}
    cls = pd.DataFrame([{k: v for k, v in u.items() if k != "ks_template"} for u in cls_rows])
    chan_to_units = cls.groupby("peak_channel")["unit_id"].apply(list).to_dict()

    all_units = sorted(by_id.keys())
    chunk_starts = np.linspace(0, reader.n_samples - CHUNK_SEC * FS, N_CHUNKS).astype(int)
    rng = np.random.default_rng(0)

    results = {}
    for i, uid in enumerate(all_units):
        chan = by_id[uid]["peak_channel"]
        other_units = [o for o in chan_to_units.get(chan, []) if o != uid]
        own_st = spike_times_all[spike_clusters_all == uid]
        other_sts = {o: spike_times_all[spike_clusters_all == o] for o in other_units}
        best, best_n = None, -1
        for cs in chunk_starts:
            d = detect(reader, cfg, chan, int(cs), CHUNK_SEC * FS)
            if d is None:
                continue
            own_local = own_st[(own_st >= cs) & (own_st < cs + CHUNK_SEC * FS)]
            assign = np.full(len(d["abs_times"]), "noise", dtype=object)
            for k, t in enumerate(d["abs_times"]):
                if np.any(np.abs(own_local - t) <= MATCH_WINDOW):
                    assign[k] = "own"
            for o, ost in other_sts.items():
                o_local = ost[(ost >= cs) & (ost < cs + CHUNK_SEC * FS)]
                if len(o_local) == 0:
                    continue
                for k, t in enumerate(d["abs_times"]):
                    if assign[k] == "noise" and np.any(np.abs(o_local - t) <= MATCH_WINDOW):
                        assign[k] = "other"
            n_own = int((assign == "own").sum())
            if n_own > best_n:
                best_n = n_own
                best = dict(pc1=d["pc1"], pc2=d["pc2"], ptp=d["ptp"], teo_max=d["teo_max"],
                            psi_trough=d["psi_trough"], teo2=d["teo2"], assign=assign)

        if best is None:
            results[uid] = None
        else:
            a = best["assign"]
            out = {}
            for cat in ["own", "other", "noise"]:
                idx = subsample(a == cat, rng)
                out[cat] = {field: [round(float(x), 2) for x in best[field][idx]]
                            for field in ("pc1", "pc2", "ptp", "teo_max", "psi_trough", "teo2")}
            results[uid] = out

        if (i + 1) % 20 == 0 or (i + 1) == len(all_units):
            elapsed = time.time() - t0
            with open(PROGRESS_PATH, "w") as f:
                f.write(f"{i+1}/{len(all_units)} units done, {elapsed:.0f}s elapsed, "
                        f"~{elapsed/(i+1)*(len(all_units)-(i+1)):.0f}s remaining\n")
            with open(OUT_PATH, "w") as f:
                json.dump(results, f)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    with open(PROGRESS_PATH, "w") as f:
        f.write(f"DONE: {len(all_units)}/{len(all_units)} units, {time.time()-t0:.0f}s total\n")
    print(f"DONE: {len(all_units)} units in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
