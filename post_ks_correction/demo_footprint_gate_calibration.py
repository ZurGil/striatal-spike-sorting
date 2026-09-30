"""
Calibrating the footprint gate -- and correcting the reading of stage 1.

WHAT STAGE 1 LOOKED LIKE: the footprint gate wiped out unit 440's
contamination (470 of 470 attributable-to-439 detections removed) but barely
touched unit 408's (2 of 162 attributable-to-412 removed). The obvious
conclusion was "footprint cannot separate 408 from 412."

WHY THAT CONCLUSION WAS WRONG: stage 2 measured the separation directly and
got AUC 0.827 for 408 vs 412 -- clearly separable, just not as cleanly as
440 vs 439 (AUC 0.999). The gate failed not because the feature is blind but
because the THRESHOLD was in the wrong place. The gate was set at the 5th
percentile of the target's own spikes, chosen to be permissive so that any
loss of detections could not be blamed on a tight threshold. For unit 440
that bar landed at 0.887 while impostor 439 averaged 0.705 -- far below, so
everything was rejected. For unit 408 the bar landed at 0.814 while impostor
412 averaged 0.850 -- ABOVE the bar, so the impostor sailed through.

A single fixed percentile of the target's own distribution cannot work,
because where the impostor sits relative to that bar is a property of the
PAIR, not of the target. This script sweeps the bar and reports the actual
trade-off, so the operating point is chosen from evidence.

Reported at each bar: what fraction of the unit's own real spikes survive
(recall), and how many detections attributable to the dominant impostor
survive (contamination).

Usage: python demo_footprint_gate_calibration.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from spatial_footprint import (multichannel_template, spatial_energy_vector,
                                footprint_similarity)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, RADIUS_UM, JITTER_WINDOW = 25, 3.0, 60.0, 15
N_SPIKES = 300
CASES = [(408, 412), (440, 439)]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
st_sorted = np.sort(spike_times)
order = np.argsort(spike_times)
sc_sorted = spike_clusters[order]
stt_sorted = spike_times[order]


def read_block(center, pad=250, fb=60):
    s0 = int(center) - pad - fb
    nr = (int(center) + pad + fb) - s0
    if s0 < 0 or s0 + nr > TOTAL:
        return None, fb
    with open(DAT_PATH, "rb") as f:
        f.seek(s0 * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nr * N_CHAN_BIN * ITEMSIZE)
    return np.frombuffer(raw, dtype=np.int16).reshape(nr, N_CHAN_BIN), fb


def filt_channel(block, ch, fb):
    return filtfilt(b_hp, a_hp, block[:, ch].astype(np.float64))[fb:-fb] * GAIN_TO_UV


rows = []
for uid, imp in CASES:
    mct = multichannel_template(templates, uid, channel_positions,
                                radius_um=RADIUS_UM, min_amplitude_fraction=0.0)
    CH = list(mct["channels"])
    PC = mct["peak_channel"]
    TM = templates[uid][:, PC]
    FP_EXP = mct["per_channel_amplitude"] / np.linalg.norm(mct["per_channel_amplitude"])
    f0, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                      criterion="timing")
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    refph = calibrate_reference_phase(TM, psi, FS, align_index=NT0MIN)

    def fp_of(sample):
        block, fb = read_block(sample)
        if block is None:
            return None
        pad = 250
        trace = filt_channel(block, PC, fb)
        r = coarse_then_fine_shift(trace, pad, TM, psi, f0, FS, reference_phase=refph,
                                   search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
        tc = pad + r["total_shift"]
        lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
        if lo < 0 or hi > len(trace):
            return None
        grid = tc - NT0MIN + np.arange(N)
        snips = [interp1d(np.arange(lo, hi), filt_channel(block, ch, fb)[lo:hi],
                          kind="cubic", bounds_error=False, fill_value=0.0)(grid)
                 for ch in CH]
        return footprint_similarity(spatial_energy_vector(np.stack(snips, axis=1)), FP_EXP)

    rng = np.random.default_rng(1)
    own_st = np.sort(spike_times[spike_clusters == uid])
    pick = own_st[rng.choice(len(own_st), size=min(N_SPIKES, len(own_st)), replace=False)]
    own_fp = np.array([v for v in (fp_of(int(s)) for s in pick) if v is not None])

    # candidate events from the stage-1 scan that passed the R2 gate,
    # split by whether unit `imp` holds a spike at that time
    cdf = pd.read_csv(os.path.join(OUT, f"fpgate_unit{uid}_candidates.csv"))
    cdf = cdf[cdf["pass_r2"]].copy()
    samp = cdf["sample"].to_numpy()
    lo = np.searchsorted(stt_sorted, samp - JITTER_WINDOW, side="left")
    hi = np.searchsorted(stt_sorted, samp + JITTER_WINDOW, side="right")
    cdf["is_impostor"] = [imp in set(sc_sorted[a:b].tolist()) for a, b in zip(lo, hi)]
    imp_fp = cdf.loc[cdf["is_impostor"], "fp"].to_numpy()
    oth_fp = cdf.loc[~cdf["is_impostor"], "fp"].to_numpy()

    print(f"\n{'='*78}")
    print(f"unit {uid} (peak ch{PC}) vs impostor {imp}")
    print(f"  own real spikes scored: {len(own_fp)}  (footprint mean {own_fp.mean():.3f})")
    print(f"  R2-passing detections: {len(cdf)} total, "
          f"{len(imp_fp)} attributable to {imp}, {len(oth_fp)} not")
    print(f"  {'bar':>7}{'own kept':>11}{'impostor kept':>15}{'other kept':>12}")
    best = None
    for bar in np.arange(0.70, 0.991, 0.01):
        keep_own = float((own_fp >= bar).mean())
        keep_imp = int((imp_fp >= bar).sum()) if len(imp_fp) else 0
        keep_oth = int((oth_fp >= bar).sum()) if len(oth_fp) else 0
        rows.append(dict(unit=uid, impostor=imp, bar=round(float(bar), 3),
                         own_kept_frac=round(keep_own, 4),
                         impostor_kept=keep_imp, other_kept=keep_oth,
                         impostor_total=len(imp_fp)))
        if bar * 100 % 5 < 1e-6 or abs(bar - 0.99) < 1e-9:
            print(f"  {bar:>7.2f}{keep_own:>11.1%}{keep_imp:>15,}{keep_oth:>12,}")
        # operating point: highest recall while keeping <=10% of the impostor
        if len(imp_fp) and keep_imp <= 0.10 * len(imp_fp) and best is None:
            best = (float(bar), keep_own, keep_imp)
    if best:
        print(f"  -> lowest bar that removes >=90% of impostor {imp}: {best[0]:.2f}, "
              f"keeping {best[1]:.1%} of own spikes ({best[2]} impostor events left)")
    else:
        print(f"  -> no bar in [0.70, 0.99] removes 90% of impostor {imp}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "footprint_gate_calibration.csv"), index=False)
print(f"\nsaved footprint_gate_calibration.csv")
