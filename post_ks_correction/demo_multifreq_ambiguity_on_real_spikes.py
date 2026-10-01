"""
Does the multi-frequency ambiguity flag work on REAL spikes, and how common
are ambiguous events?

THE VALIDATION TRICK: we do not need injected ground truth to test the
collision flag, because Kilosort's own spike list already tells us when two
nearby neurons fired almost simultaneously. Split a unit's own spikes into:

  ISOLATED  -- no other unit within 60 um has a spike within +/-40 samples
  COLLIDING -- another unit within 60 um has a spike within +/-12 samples

If the scatter statistic is doing its job, the COLLIDING group must score
higher. That is a real-data test with a real (if imperfect) label, and it
costs nothing to run.

WHAT ELSE IS MEASURED:
  * how often the flag fires on ISOLATED spikes -- the false-alarm rate, and
    the answer to "how common are ambiguous events?"
  * whether the multi-probe delay differs from the single-probe delay the
    current pipeline uses, and by how much
  * per-probe disagreement, so the shape of the failure is visible rather
    than just its summary

IMPORTANT LIMIT ON THE LABEL: "colliding" here means another SORTED unit
fired nearby. Collisions with unsorted/sub-threshold activity are invisible
to this label and will sit in the ISOLATED group, which inflates the apparent
false-alarm rate. So a high flag rate on ISOLATED spikes is an upper bound on
false alarms, not a measurement of them.

Usage: python demo_multifreq_ambiguity_on_real_spikes.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from multifreq_alignment import (build_probe_bank, phase_slope_delay,
                                  single_frequency_delay, max_unambiguous_shift,
                                  calibrate_ambiguity_threshold)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, RADIUS_UM = 25, 3.0, 60.0
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_PROBES = 5
ISO_GUARD, COLLIDE_WIN = 40, 12
N_PER_GROUP = 400
UNITS = [440, 408, 439]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL)

order = np.argsort(spike_times)
st_sorted, sc_sorted = spike_times[order], spike_clusters[order]


def load_channel(ch):
    out = np.empty(WINDOW_END - WINDOW_START, dtype=np.float32)
    pos = WINDOW_START
    while pos < WINDOW_END:
        clen = min(CHUNK, WINDOW_END - pos)
        rlo, rhi = max(0, pos - OVERLAP), min(TOTAL, pos + clen + OVERLAP)
        with open(DAT_PATH, "rb") as f:
            f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
        filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
        lt = pos - rlo
        out[pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt:lt + clen]
        pos += clen
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def neighbour_distance_ok(uid, other):
    if other >= n_templ:
        return False
    d = np.sqrt(((channel_positions[peak_ch_all[other]]
                  - channel_positions[peak_ch_all[uid]]) ** 2).sum())
    return d <= RADIUS_UM


summary, detail = [], []
for uid in UNITS:
    PC = int(peak_ch_all[uid])
    TM = templates[uid][:, PC].astype(float)

    bank = build_probe_bank(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                            n_probes=N_PROBES)
    f_single, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES,
                                            nt0min=NT0MIN, criterion="timing")
    lim = max_unambiguous_shift(bank)
    print(f"\n{'='*76}")
    print(f"unit {uid}: peak ch{PC}")
    print(f"  probe bank: {len(bank['freqs'])} frequencies "
          f"{np.round(bank['freqs']).astype(int).tolist()} Hz")
    print(f"  band kept: {bank['band_hz'][0]:.0f}-{bank['band_hz'][1]:.0f} Hz "
          f"| weights {np.round(bank['weights'], 3).tolist()}")
    print(f"  single-probe frequency the current pipeline would pick: {f_single:.0f} Hz")
    print(f"  phase-wrap limit of this bank: +/-{lim:.2f} samples")

    t0 = time.time()
    trace = load_channel(PC)
    print(f"  loaded channel in {time.time()-t0:.0f}s")

    st_u = np.sort(spike_times[spike_clusters == uid])
    st_in = st_u[(st_u >= WINDOW_START + 1000) & (st_u < WINDOW_END - 1000)]

    # label each own-spike as isolated or colliding, using the sorted output
    lo_i = np.searchsorted(st_sorted, st_in - ISO_GUARD, side="left")
    hi_i = np.searchsorted(st_sorted, st_in + ISO_GUARD, side="right")
    lo_c = np.searchsorted(st_sorted, st_in - COLLIDE_WIN, side="left")
    hi_c = np.searchsorted(st_sorted, st_in + COLLIDE_WIN, side="right")
    iso, coll = [], []
    for s, a, b, ca, cb in zip(st_in, lo_i, hi_i, lo_c, hi_c):
        near_far = {int(c) for c in sc_sorted[a:b].tolist()} - {uid}
        near_cls = {int(c) for c in sc_sorted[ca:cb].tolist()} - {uid}
        if any(neighbour_distance_ok(uid, c) for c in near_cls):
            coll.append(int(s))
        elif not any(neighbour_distance_ok(uid, c) for c in near_far):
            iso.append(int(s))
    rng = np.random.default_rng(5)
    iso = np.array(iso)[rng.permutation(len(iso))[:N_PER_GROUP]] if iso else np.array([], int)
    coll = np.array(coll)[rng.permutation(len(coll))[:N_PER_GROUP]] if coll else np.array([], int)
    print(f"  own spikes in window: {len(st_in):,} -> "
          f"{len(iso)} isolated / {len(coll)} colliding sampled")

    # alignment reference for the coarse stage (unchanged pipeline behaviour)
    _, psi_single = make_morlet(f_single, FS, n_cycles=N_CYCLES)
    ref_single = calibrate_reference_phase(TM, psi_single, FS, align_index=NT0MIN)

    def measure(sample, group):
        i = int(sample) - WINDOW_START
        pad = 200
        if i - pad < 0 or i + pad >= len(trace):
            return None
        loc = trace[i - pad:i + pad].astype(np.float64)
        # coarse stage first (magnitude-based), exactly as the pipeline does,
        # so the phase stage is inside its unambiguous range
        r = coarse_then_fine_shift(loc, pad, TM, psi_single, f_single, FS,
                                   reference_phase=ref_single,
                                   search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
        c = pad + int(round(r["coarse_shift"]))
        if c - 100 < 0 or c + 100 >= len(loc):
            return None
        m = phase_slope_delay(loc, c, bank)
        s1 = single_frequency_delay(loc, c, bank)
        return dict(unit=uid, group=group, sample=int(sample),
                    delta_multi=m["delta"], delta_single=s1,
                    scatter=m["scatter_samples"],
                    spread=float(np.ptp(m["per_probe_delta"])))

    rows = [x for x in ([measure(s, "isolated") for s in iso]
                        + [measure(s, "colliding") for s in coll]) if x]
    df = pd.DataFrame(rows)
    detail.append(df)
    g_iso = df[df["group"] == "isolated"]
    g_col = df[df["group"] == "colliding"]

    thr = calibrate_ambiguity_threshold(g_iso["scatter"], percentile=95.0)
    a = auc(g_col["scatter"], g_iso["scatter"])
    flag_iso = float((g_iso["scatter"] > thr).mean())
    flag_col = float((g_col["scatter"] > thr).mean()) if len(g_col) else float("nan")
    disagree = float(np.mean(np.abs(df["delta_multi"] - df["delta_single"])))

    print(f"  scatter (samples): isolated median {g_iso['scatter'].median():.3f}, "
          f"colliding median {g_col['scatter'].median():.3f}")
    print(f"  AUC separating colliding from isolated by scatter = {a:.3f}")
    print(f"  flag threshold (isolated p95) = {thr:.3f}  ->  fires on "
          f"{flag_iso:.1%} of isolated, {flag_col:.1%} of colliding")
    print(f"  multi vs single probe delay: mean |difference| = {disagree:.3f} samples")

    summary.append(dict(unit=uid, peak_ch=PC, n_probes=len(bank["freqs"]),
                        band_lo=round(bank["band_hz"][0], 1),
                        band_hi=round(bank["band_hz"][1], 1),
                        f_single=round(f_single, 1), wrap_limit=round(lim, 3),
                        n_isolated=len(g_iso), n_colliding=len(g_col),
                        scatter_iso_med=round(float(g_iso["scatter"].median()), 4),
                        scatter_col_med=round(float(g_col["scatter"].median()), 4),
                        auc_collision=round(a, 4), flag_threshold=round(thr, 4),
                        flag_rate_isolated=round(flag_iso, 4),
                        flag_rate_colliding=round(flag_col, 4),
                        mean_abs_multi_minus_single=round(disagree, 4)))
    del trace

sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "multifreq_ambiguity_summary.csv"), index=False)
pd.concat(detail).to_csv(os.path.join(OUT, "multifreq_ambiguity_per_spike.csv"),
                          index=False)
print(f"\n{'='*76}")
print(sdf[["unit", "scatter_iso_med", "scatter_col_med", "auc_collision",
           "flag_rate_isolated", "flag_rate_colliding",
           "mean_abs_multi_minus_single"]].to_string(index=False))
print("\nsaved multifreq_ambiguity_summary.csv, multifreq_ambiguity_per_spike.csv")
