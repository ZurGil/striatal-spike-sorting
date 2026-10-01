"""
Is the weak ambiguity result a design choice or something fundamental?

The first real-data run gave a disappointing answer: the scatter statistic
separated known collisions from isolated spikes at only AUC 0.55-0.66, and
isolated spikes already scattered by ~0.70 samples -- enormous, given that on
synthetic clean spikes the scatter is near zero.

Two suspects:
  (a) THE BAND IS TOO WIDE. keep_fraction=0.35 kept 523-8000 Hz for unit 440
      (and 8000 is the scan ceiling, i.e. the range rather than the data set
      it -- the same edge problem found in section 5r). The extreme probes
      carry little signal (weights 0.047 and 0.06) but still inject noise
      into the fit, and at 8000 Hz with n_cycles=3 the wavelet spans only
      ~11 samples, so it is largely measuring high-frequency noise.
  (b) IT IS PHYSIOLOGY. Real spikes are not pure shifted copies of their
      template -- amplitude varies, shape varies -- so the phase is not
      exactly linear in frequency no matter how well the probes are chosen,
      and the baseline scatter is a real property of the data.

If (a), narrowing the band should lower the isolated-spike baseline and raise
the AUC. If (b), nothing will help much, and that is a real finding about the
method's ceiling that should be recorded rather than tuned around.

Run on unit 408, which has the most usable collision labels (116 events vs
12 for unit 440 -- that small count is itself a reason the first run's AUC is
noisy).

Usage: python demo_multifreq_band_sweep.py
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
ISO_GUARD, COLLIDE_WIN, N_PER_GROUP = 40, 12, 400
UID = 408

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
PC = int(peak_ch_all[UID])
TM = templates[UID][:, PC].astype(float)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


pos = WINDOW_START
trace = np.empty(WINDOW_END - WINDOW_START, dtype=np.float32)
t0 = time.time()
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
print(f"unit {UID} ch{PC}: channel loaded in {time.time()-t0:.0f}s")


def near_ok(other):
    if other >= n_templ:
        return False
    d = np.sqrt(((channel_positions[peak_ch_all[other]]
                  - channel_positions[peak_ch_all[UID]]) ** 2).sum())
    return d <= RADIUS_UM


st_u = np.sort(spike_times[spike_clusters == UID])
st_in = st_u[(st_u >= WINDOW_START + 1000) & (st_u < WINDOW_END - 1000)]
lo_i = np.searchsorted(st_sorted, st_in - ISO_GUARD, side="left")
hi_i = np.searchsorted(st_sorted, st_in + ISO_GUARD, side="right")
lo_c = np.searchsorted(st_sorted, st_in - COLLIDE_WIN, side="left")
hi_c = np.searchsorted(st_sorted, st_in + COLLIDE_WIN, side="right")
iso, coll = [], []
for s, a, b, ca, cb in zip(st_in, lo_i, hi_i, lo_c, hi_c):
    if any(near_ok(c) for c in {int(c) for c in sc_sorted[ca:cb].tolist()} - {UID}):
        coll.append(int(s))
    elif not any(near_ok(c) for c in {int(c) for c in sc_sorted[a:b].tolist()} - {UID}):
        iso.append(int(s))
rng = np.random.default_rng(5)
iso = np.array(iso)[rng.permutation(len(iso))[:N_PER_GROUP]]
coll = np.array(coll)[rng.permutation(len(coll))[:N_PER_GROUP]]
print(f"  {len(iso)} isolated, {len(coll)} colliding\n")

f_single, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES,
                                        nt0min=NT0MIN, criterion="timing")
_, psi_single = make_morlet(f_single, FS, n_cycles=N_CYCLES)
ref_single = calibrate_reference_phase(TM, psi_single, FS, align_index=NT0MIN)

# precompute the coarse centre once per spike -- identical across all banks,
# so the sweep compares banks and nothing else
centres = {}
for s in np.concatenate([iso, coll]):
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
        centres[int(s)] = (loc, c)
print(f"  {len(centres)} spikes with a usable coarse centre")

rows = []
print(f"\n{'keep':>6}{'probes':>8}{'band (Hz)':>16}{'iso med':>10}"
      f"{'coll med':>10}{'AUC':>8}{'flag@iso95 on coll':>21}")
for keep in (0.35, 0.50, 0.70, 0.85):
    for npr in (3, 5, 7):
        bank = build_probe_bank(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                n_probes=npr, keep_fraction=keep)
        if len(bank["freqs"]) < 2:
            continue
        sc_iso, sc_col = [], []
        for s in iso:
            if int(s) in centres:
                loc, c = centres[int(s)]
                sc_iso.append(phase_slope_delay(loc, c, bank)["scatter_samples"])
        for s in coll:
            if int(s) in centres:
                loc, c = centres[int(s)]
                sc_col.append(phase_slope_delay(loc, c, bank)["scatter_samples"])
        a = auc(sc_col, sc_iso)
        thr = float(np.percentile(sc_iso, 95))
        fr = float(np.mean(np.asarray(sc_col) > thr))
        band = f"{bank['band_hz'][0]:.0f}-{bank['band_hz'][1]:.0f}"
        print(f"{keep:>6.2f}{len(bank['freqs']):>8}{band:>16}"
              f"{np.median(sc_iso):>10.3f}{np.median(sc_col):>10.3f}"
              f"{a:>8.3f}{fr:>20.1%}")
        rows.append(dict(keep_fraction=keep, n_probes=len(bank["freqs"]),
                         band_lo=round(bank["band_hz"][0], 1),
                         band_hi=round(bank["band_hz"][1], 1),
                         freqs=";".join(str(int(round(f))) for f in bank["freqs"]),
                         iso_median=round(float(np.median(sc_iso)), 4),
                         coll_median=round(float(np.median(sc_col)), 4),
                         auc=round(a, 4), flag_rate_colliding=round(fr, 4)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "multifreq_band_sweep.csv"), index=False)
best = df.loc[df["auc"].idxmax()]
print(f"\nbest AUC {best['auc']:.3f} at keep_fraction={best['keep_fraction']}, "
      f"{int(best['n_probes'])} probes, band {best['band_lo']:.0f}-{best['band_hi']:.0f} Hz")
print(f"baseline isolated scatter there: {best['iso_median']:.3f} samples")
print("\nsaved multifreq_band_sweep.csv")
