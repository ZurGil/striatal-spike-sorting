"""
Was the collision LABEL too loose, or is the ambiguity statistic genuinely weak?

The band sweep settled one question: narrowing the probe band cuts the
baseline scatter 2.2x (0.738 -> 0.334 samples) but leaves the AUC flat at
~0.62. Both the isolated and colliding groups shrink together, so the
baseline is not probe-band noise -- it is something that affects clean and
colliding spikes alike.

But there is still a confound in the LABEL. "Colliding" was defined as
another nearby unit having a spike within +/-12 samples. With a 61-sample
template, two spikes 12 samples apart overlap only in their tails -- the
peaks do not interfere at all. If the statistic really detects interference,
it should care enormously about HOW CLOSE the second spike is, and lumping
everything inside +/-12 samples together would hide that.

So this measures scatter as a function of the actual lag to the nearest
neighbouring spike, in bins. The prediction if the method works:

    scatter should rise sharply as |lag| -> 0 and flatten out by ~15 samples

If instead scatter is flat across lag, the statistic is not responding to
interference at all and the ~0.62 AUC is the method's real ceiling on this
data -- a negative result worth recording rather than tuning around.

Uses the narrow band that minimized baseline scatter (keep_fraction=0.85).

Usage: python demo_multifreq_scatter_vs_collision_lag.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from multifreq_alignment import build_probe_bank, phase_slope_delay

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, RADIUS_UM = 25, 3.0, 60.0
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
KEEP_FRACTION, N_PROBES = 0.85, 5
UNITS = [408, 439]

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

BINS = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40), (40, 10 ** 9)]
rows = []

for UID in UNITS:
    PC = int(peak_ch_all[UID])
    TM = templates[UID][:, PC].astype(float)
    bank = build_probe_bank(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                            n_probes=N_PROBES, keep_fraction=KEEP_FRACTION)
    f_single, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES,
                                            nt0min=NT0MIN, criterion="timing")
    _, psi_single = make_morlet(f_single, FS, n_cycles=N_CYCLES)
    ref_single = calibrate_reference_phase(TM, psi_single, FS, align_index=NT0MIN)

    t0 = time.time()
    trace = np.empty(WINDOW_END - WINDOW_START, dtype=np.float32)
    pos = WINDOW_START
    while pos < WINDOW_END:
        clen = min(CHUNK, WINDOW_END - pos)
        rlo, rhi = max(0, pos - OVERLAP), min(TOTAL, pos + clen + OVERLAP)
        with open(DAT_PATH, "rb") as f:
            f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
        filt = filtfilt(b_hp, a_hp, blk[:, PC].astype(np.float64)) * GAIN_TO_UV
        trace[pos - WINDOW_START: pos - WINDOW_START + clen] = filt[pos - rlo: pos - rlo + clen]
        pos += clen

    print(f"\n{'='*70}\nunit {UID} ch{PC}: loaded in {time.time()-t0:.0f}s")
    print(f"  band {bank['band_hz'][0]:.0f}-{bank['band_hz'][1]:.0f} Hz, "
          f"{len(bank['freqs'])} probes "
          f"{np.round(bank['freqs']).astype(int).tolist()}")

    st_u = np.sort(spike_times[spike_clusters == UID])
    st_in = st_u[(st_u >= WINDOW_START + 1000) & (st_u < WINDOW_END - 1000)]

    # exact lag to the nearest spike of any unit within RADIUS_UM
    mp = channel_positions[peak_ch_all[UID]]
    near_units = set()
    for oc in np.unique(sc_sorted):
        if oc == UID or oc >= n_templ:
            continue
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        if d <= RADIUS_UM:
            near_units.add(int(oc))
    mask_near = np.isin(sc_sorted, list(near_units))
    t_near = st_sorted[mask_near]
    print(f"  {len(near_units)} units within {RADIUS_UM:.0f} um, "
          f"{len(t_near):,} of their spikes in the recording")

    idx = np.searchsorted(t_near, st_in)
    lag = np.minimum(
        np.abs(st_in - t_near[np.clip(idx, 0, len(t_near) - 1)]),
        np.abs(st_in - t_near[np.clip(idx - 1, 0, len(t_near) - 1)]))

    rng = np.random.default_rng(3)
    print(f"  {'lag (samples)':>16}{'n':>7}{'scatter median':>16}{'scatter p75':>13}")
    for lo, hi in BINS:
        sel = st_in[(lag >= lo) & (lag < hi)]
        if len(sel) > 350:
            sel = sel[rng.permutation(len(sel))[:350]]
        vals = []
        for s in sel:
            i = int(s) - WINDOW_START
            pad = 200
            if i - pad < 0 or i + pad >= len(trace):
                continue
            loc = trace[i - pad:i + pad].astype(np.float64)
            r = coarse_then_fine_shift(loc, pad, TM, psi_single, f_single, FS,
                                       reference_phase=ref_single,
                                       search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
            c = pad + int(round(r["coarse_shift"]))
            if 100 <= c < len(loc) - 100:
                vals.append(phase_slope_delay(loc, c, bank)["scatter_samples"])
        if not vals:
            continue
        lbl = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {lbl:>16}{len(vals):>7}{np.median(vals):>16.3f}"
              f"{np.percentile(vals,75):>13.3f}")
        rows.append(dict(unit=UID, lag_lo=lo, lag_hi=(hi if hi < 10**9 else -1),
                         n=len(vals), scatter_median=round(float(np.median(vals)), 4),
                         scatter_p75=round(float(np.percentile(vals, 75)), 4),
                         scatter_mean=round(float(np.mean(vals)), 4)))
    del trace

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "multifreq_scatter_vs_lag.csv"), index=False)
print(f"\n{'='*70}")
for UID in UNITS:
    d = df[df["unit"] == UID]
    if len(d) < 2:
        continue
    close = d.iloc[0]["scatter_median"]
    far = d.iloc[-1]["scatter_median"]
    print(f"unit {UID}: closest bin {close:.3f} vs farthest bin {far:.3f} "
          f"-> ratio {close/far:.2f}x")
print("\nIf those ratios are near 1.0, the statistic is not responding to")
print("interference and ~0.62 AUC is the real ceiling, not a tuning problem.")
print("\nsaved multifreq_scatter_vs_lag.csv")
