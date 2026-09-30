"""
Corrected ablation for the 20-minute scan.

WHY THIS REPLACES THE FIRST ATTEMPT: the first version set each
configuration's detection bar at the 99th percentile of a NOISE-ONLY null.
That bar answers "is this distinguishable from silence", not "is this a
spike of THIS unit" -- and it sat far below what the units' own real
spikes score (unit 440: noise bar 0.426 vs its own real spikes' 25th
percentile of 0.720). So it admitted other neurons' spikes and ordinary
noise excursions wholesale, producing 5-12x more "detections" than the
units even fire. Those counts were meaningless.

CORRECTED BAR: for each configuration independently, score the unit's OWN
known-real spikes under that configuration and take their 25th percentile.
Each configuration is then asked the same question -- "how many candidate
events reach the quality this configuration assigns to known-real spikes
of this unit" -- which is a fair, matched comparison across configurations.

Restricted to units 440 and 408, the only two of the four that are usable:
units 302 and 31 have own-spike scores at or below the noise floor
(own p25 0.370 / 0.380 vs noise bar 0.530 / 0.554), so no threshold can
separate their real spikes from noise and any count for them is
meaningless.

Usage: python demo_20min_ablation_corrected.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, find_peaks
from scipy.interpolate import interp1d

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, MARGIN, MIN_PEAK_DISTANCE = 25, 3.0, 20, 20
RADIUS_UM, JITTER_WINDOW = 60.0, 15

UNITS = [440, 408]
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_REF = 120
CONFIGS = ["full", "no_wavelet", "no_whiten", "neither"]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL_SAMPLES = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL_SAMPLES)

unit_cfg = {}
for uid in UNITS:
    ta = templates[uid]
    pc = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
    tm = ta[:, pc]
    f0s = np.linspace(300, 4000, 50)
    ps = int(np.ceil(N_CYCLES * FS / (2 * f0s.min()))) + 20
    tr = np.zeros(N + 2 * ps); ctr = ps + NT0MIN
    tr[ctr - NT0MIN: ctr - NT0MIN + N] = tm
    mg = [abs(wavelet_transform_at(tr, make_morlet(f, FS, N_CYCLES)[1], ctr)) for f in f0s]
    f0 = float(f0s[int(np.nanargmax(mg))])
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    s0b, _, qb = build_basis(tm, dt=1.0)
    unit_cfg[uid] = dict(peak_ch=pc, template=tm, template_norm=tm / (np.linalg.norm(tm) + 1e-12),
                          f0=f0, psi=psi, ref_phase=calibrate_reference_phase(tm, psi, FS, align_index=NT0MIN),
                          s0=s0b, q=qb, st=np.sort(spike_times[spike_clusters == uid]))

channels = sorted({unit_cfg[u]["peak_ch"] for u in UNITS})
t0 = time.time()
traces = {ch: np.empty(WINDOW_END - WINDOW_START, dtype=np.float64) for ch in channels}
pos = WINDOW_START
while pos < WINDOW_END:
    clen = min(CHUNK, WINDOW_END - pos)
    rlo, rhi = max(0, pos - OVERLAP), min(TOTAL_SAMPLES, pos + clen + OVERLAP)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
    lt = pos - rlo
    for ch in channels:
        filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
        traces[ch][pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt: lt + clen]
    pos += clen
print(f"read+filtered {len(channels)} channels in {time.time()-t0:.0f}s")


def snip_raw(trace, s):
    i = s - WINDOW_START - NT0MIN
    if i < 0 or i + N > len(trace):
        return None
    return trace[i:i + N].copy()


def snip_aligned(trace, s, cfg):
    i = s - WINDOW_START
    pad = 200
    if i - pad < 0 or i + pad >= len(trace):
        return None
    loc = trace[i - pad:i + pad]
    r = coarse_then_fine_shift(loc, pad, cfg["template"], cfg["psi"], cfg["f0"], FS,
                                reference_phase=cfg["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(loc):
        return None
    ip = interp1d(np.arange(lo, hi), loc[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(tc - NT0MIN + np.arange(N))


def all_configs(trace, s, cfg, W):
    sa, sr = snip_aligned(trace, s, cfg), snip_raw(trace, s)
    if sa is None or sr is None:
        return None
    return dict(
        full=fit_nuisance_prealigned(sa, cfg["s0"], cfg["q"], whitening_matrix=W)["r_squared"],
        no_whiten=fit_nuisance_prealigned(sa, cfg["s0"], cfg["q"], whitening_matrix=None)["r_squared"],
        no_wavelet=fit_nuisance_prealigned(sr, cfg["s0"], cfg["q"], whitening_matrix=W)["r_squared"],
        neither=fit_nuisance_prealigned(sr, cfg["s0"], cfg["q"], whitening_matrix=None)["r_squared"])


def neighbor(uid, s):
    mp = channel_positions[peak_ch_all[uid]]
    m = np.abs(spike_times - s) <= JITTER_WINDOW
    best = None
    for oc in sorted(set(spike_clusters[m].tolist()) - {uid}):
        if oc >= n_templ:
            continue
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (oc, d)
    return best


rows, detail = [], []
for uid in UNITS:
    cfg = unit_cfg[uid]
    trace = traces[cfg["peak_ch"]]
    st = cfg["st"]
    st_in = st[(st >= WINDOW_START + 1000) & (st < WINDOW_END - 1000)]
    W, _, _ = build_whitening_from_noise(trace[:60000], N, order=4)

    # per-configuration bar from this unit's OWN real spikes
    rng = np.random.default_rng(42)
    ref = st_in[rng.choice(len(st_in), size=min(N_REF, len(st_in)), replace=False)]
    ref_scores = {c: [] for c in CONFIGS}
    for s in ref:
        sc_ = all_configs(trace, int(s), cfg, W)
        if sc_:
            for c in CONFIGS:
                ref_scores[c].append(sc_[c])
    bars = {c: float(np.percentile(ref_scores[c], 25)) for c in CONFIGS}

    own_coarse = [np.dot(x, cfg["template_norm"]) for x in
                  (snip_raw(trace, int(s)) for s in st_in[:200]) if x is not None]
    floor = np.percentile(own_coarse, 10)
    score = correlate(trace, cfg["template_norm"], mode="valid", method="fft")
    peaks, _ = find_peaks(score, height=floor, distance=MIN_PEAK_DISTANCE)
    pabs = peaks + WINDOW_START + NT0MIN
    di = np.searchsorted(st, pabs)
    d1 = np.abs(pabs - st[np.clip(di, 0, len(st) - 1)])
    d2 = np.abs(pabs - st[np.clip(di - 1, 0, len(st) - 1)])
    cands = pabs[np.minimum(d1, d2) > MARGIN]

    cand_scores = []
    for c_ in cands:
        sc_ = all_configs(trace, int(c_), cfg, W)
        if sc_:
            cand_scores.append(dict(sample=int(c_), **sc_))
    cdf = pd.DataFrame(cand_scores)
    counts = {c: int((cdf[c] >= bars[c]).sum()) for c in CONFIGS}

    passing = cdf[cdf["full"] >= bars["full"]]
    n_el = n_new = 0
    for _, r in passing.iterrows():
        nb = neighbor(uid, int(r["sample"]))
        if nb:
            n_el += 1
        else:
            n_new += 1
        detail.append(dict(uid=uid, sample=int(r["sample"]), r2=r["full"],
                            neighbor_uid=nb[0] if nb else None))

    print(f"\n=== unit {uid} (ch{cfg['peak_ch']}) ===")
    print(f"  own spikes in window        : {len(st_in):,}")
    print(f"  candidate locations scanned : {len(cdf):,}")
    print(f"  per-config bars (own-spike p25): " + ", ".join(f"{c}={bars[c]:.3f}" for c in CONFIGS))
    print(f"  DETECTIONS (matched to each config's own real-spike quality):")
    for c in CONFIGS:
        print(f"      {c:<11}: {counts[c]:,}")
    print(f"  of the {counts['full']:,} full-pipeline detections:")
    print(f"      already detected under another unit  : {n_el:,}")
    print(f"      NEW (first detected in this analysis): {n_new:,}")
    if len(passing):
        print(f"      top neighbours: {dict(passing.assign(nb=[neighbor(uid,int(s))[0] if neighbor(uid,int(s)) else None for s in passing['sample']]).nb.value_counts().head(4))}")

    rows.append(dict(uid=uid, own_spikes=len(st_in), scanned=len(cdf),
                      **{f"det_{c}": counts[c] for c in CONFIGS},
                      already_elsewhere=n_el, new=n_new))

df = pd.DataFrame(rows)
print("\n\n========== CORRECTED SUMMARY (20 min of recording) ==========")
print(df.to_string(index=False))
print("\nCOMPONENT CONTRIBUTION (each config judged against its own real-spike quality)")
for c in CONFIGS:
    print(f"  {c:<11}: {df['det_'+c].sum():,}")
df.to_csv(os.path.join(OUT, "scan20min_ablation_corrected.csv"), index=False)
pd.DataFrame(detail).to_csv(os.path.join(OUT, "scan20min_detections_corrected.csv"), index=False)
print("\nsaved")
