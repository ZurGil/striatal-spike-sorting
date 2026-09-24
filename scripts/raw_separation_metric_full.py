"""
Stage A extension: raw-data separation metric (vs noise floor, vs co-located units)
for ALL 401 units. Generalizes the ad-hoc per-unit investigation from earlier this
session into a reusable, full-session pipeline stage.

Detection is KS-independent (blind threshold-crossing on the highpass-filtered raw
trace); labeling which detected event belongs to which group uses KS's own spike
times (this unit's own vs. a co-located unit's vs. unassigned/noise). See conversation
for full design rationale and validation on the 60-unit pilot.
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
OUT_PATH = os.path.join(OUT, "raw_separation_full_401.npy")
PROGRESS_PATH = os.path.join(OUT, "raw_separation_full_401_progress.txt")

FS = 30000
N_CHUNKS = 10
CHUNK_SEC = 120
NT, NT0MIN = 60, 20
DEAD_SAMPLES = int(0.001 * FS)
MATCH_WINDOW = int(0.0005 * FS)


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
    # Teager-Kaiser energy operator: psi[n] = x[n]^2 - x[n-1]*x[n+1], sensitive to fast
    # transients (amplitude+frequency) not just amplitude -- see conversation: dramatically
    # improves own-vs-noise separation for low-SNR units (70, 58, 59), roughly neutral
    # elsewhere, occasionally hurts an already-easy unit (40) whose noise floor itself has
    # real high-frequency content. Included as a 4th axis; Mahalanobis's covariance term
    # should down-weight it automatically where it doesn't help.
    teo_max = np.array([(s[1:-1]**2 - s[:-2]*s[2:]).max() for s in snips])
    # psi evaluated at the trough sample specifically (our detector's own alignment point),
    # rather than searched over the whole snippet -- validated in conversation as a genuinely
    # complementary axis (not a replacement for teo_max): whole-window search adaptively
    # finds whichever phase (trough or rebound) is sharpest per-event, while this pins down
    # sharpness specifically at the detected point. Confirmed additive (never harmful) across
    # 13 test units spanning SNR 6-129, with gains up to +15% on units where the two capture
    # different information.
    trough_idx = np.clip(np.argmin(snips, axis=1), 1, NT - 2)
    psi_trough = np.array([s[i]**2 - s[i-1]*s[i+1] for s, i in zip(snips, trough_idx)])
    # lag-2 generalization: psi_2[n] = x[n]^2 - x[n-2]*x[n+2]. For a pure tone this equals
    # A^2*sin^2(2*omega) instead of A^2*sin^2(omega) -- tuned to a lower frequency band, so it
    # helps units whose dominant transient is a bit slower (e.g. unit 70: lag-2 alone beat
    # lag-1 alone) without replacing lag-1 (which still wins standalone on faster-transient
    # units). Validated as additive, not harmful, across the same 13-unit test set.
    teo2 = np.array([(s[2:-2]**2 - s[:-4]*s[4:]).max() for s in snips])
    X = snips - snips.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    feat = np.column_stack([X @ Vt[0], X @ Vt[1], ptp, teo_max, psi_trough, teo2])
    return dict(abs_times=abs_times, feat=feat)


def maha_sep(a, b):
    if len(a) < 5 or len(b) < 5:
        return np.nan
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov = (np.cov(a.T) + np.cov(b.T)) / 2 + np.eye(a.shape[1]) * 1e-6
    diff = mu_a - mu_b
    return float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))


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
    cls = pd.DataFrame([{k: v for k, v in u.items() if k != "ks_template"} for u in cls_rows])
    by_id = {u["unit_id"]: u for u in cls_rows}
    chan_to_units = cls.groupby("peak_channel")["unit_id"].apply(list).to_dict()
    lut = pd.read_csv(os.path.join(OUT, "unit_trodes_channel_lookup.csv")).set_index("unit_id")

    all_units = sorted(by_id.keys())
    chunk_starts = np.linspace(0, reader.n_samples - CHUNK_SEC * FS, N_CHUNKS).astype(int)

    results = {}
    for i, uid in enumerate(all_units):
        chan = by_id[uid]["peak_channel"]
        other_units_on_chan = [o for o in chan_to_units.get(chan, []) if o != uid]
        own_st = spike_times_all[spike_clusters_all == uid]
        other_sts = {o: spike_times_all[spike_clusters_all == o] for o in other_units_on_chan}
        sep_vs_noise, sep_vs_others = [], []
        for cs in chunk_starts:
            d = detect(reader, cfg, chan, int(cs), CHUNK_SEC * FS)
            if d is None:
                sep_vs_noise.append(np.nan); sep_vs_others.append(np.nan); continue
            feat_z = (d["feat"] - d["feat"].mean(0)) / (d["feat"].std(0) + 1e-9)
            assign = np.full(len(d["abs_times"]), "", dtype=object)
            own_local = own_st[(own_st >= cs) & (own_st < cs + CHUNK_SEC * FS)]
            for t_idx, t in enumerate(d["abs_times"]):
                if np.any(np.abs(own_local - t) <= MATCH_WINDOW):
                    assign[t_idx] = "own"
            for o, ost in other_sts.items():
                o_local = ost[(ost >= cs) & (ost < cs + CHUNK_SEC * FS)]
                if len(o_local) == 0:
                    continue
                for t_idx, t in enumerate(d["abs_times"]):
                    if assign[t_idx] == "" and np.any(np.abs(o_local - t) <= MATCH_WINDOW):
                        assign[t_idx] = "other"
            unassigned = assign == ""
            is_own = assign == "own"
            is_other = ~is_own & ~unassigned
            sep_vs_noise.append(maha_sep(feat_z[is_own], feat_z[unassigned]))
            sep_vs_others.append(maha_sep(feat_z[is_own], feat_z[is_other]) if is_other.sum() >= 5 else np.nan)

        trodes = int(lut.loc[uid].trodes_ntrode_id) if uid in lut.index else None
        results[uid] = dict(
            trodes=trodes, other_units=other_units_on_chan,
            sep_vs_noise=sep_vs_noise, sep_vs_others=sep_vs_others,
        )

        if (i + 1) % 20 == 0 or (i + 1) == len(all_units):
            elapsed = time.time() - t0
            with open(PROGRESS_PATH, "w") as f:
                f.write(f"{i+1}/{len(all_units)} units done, {elapsed:.0f}s elapsed, "
                        f"~{elapsed/(i+1)*(len(all_units)-(i+1)):.0f}s remaining\n")
            np.save(OUT_PATH, results, allow_pickle=True)  # incremental save

    np.save(OUT_PATH, results, allow_pickle=True)
    with open(PROGRESS_PATH, "w") as f:
        f.write(f"DONE: {len(all_units)}/{len(all_units)} units, {time.time()-t0:.0f}s total\n")
    print(f"DONE: {len(all_units)} units in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
