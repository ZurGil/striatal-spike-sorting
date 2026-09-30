"""
20-minute scan across several good units (session 20260916_110311), with
clear counts and a component ablation.

Answers three questions with real numbers, per unit:
  1. How many credible undetected spike-like events are found?
  2. How many of those are already detected under a DIFFERENT unit?
  3. How many are detected for the first time here (new to this analysis)?

Plus: what do the wavelet alignment and the precision matrix (whitening)
each actually contribute? Measured as an ABLATION AT MATCHED FALSE-POSITIVE
RATE -- each of four configurations gets its own noise-only null
distribution from the same data, its detection bar set at that null's 99th
percentile, so every configuration is allowed the same 1% false-positive
rate and the detection counts are directly comparable:

    full        : wavelet alignment + whitened fit
    no_wavelet  : raw position (no alignment)   + whitened fit
    no_whiten   : wavelet alignment             + plain (unweighted) fit
    neither     : raw position                  + plain fit

Window: 20 real minutes of recording (36,000,000 samples), read ONCE and
shared across all units (their channels are extracted from the same
chunks), so this costs one pass over ~27GB rather than one per unit.

Usage: python demo_20min_scan_with_ablation.py
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
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
MARGIN = 20
MIN_PEAK_DISTANCE = 20
RADIUS_UM = 60.0
JITTER_WINDOW = 15

UNITS = [440, 408, 302, 31]
WINDOW_START = 50_000_000
WINDOW_SAMPLES = 36_000_000        # 20 minutes
CHUNK = 2_000_000
OVERLAP = 400
N_NULL = 400                       # noise-only locations per unit for the null

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

file_size = os.path.getsize(DAT_PATH)
TOTAL_SAMPLES = file_size // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL_SAMPLES)
print(f"window: samples {WINDOW_START:,}-{WINDOW_END:,} "
      f"({(WINDOW_END-WINDOW_START)/FS/60:.1f} min of recording)")

# ---- per-unit setup ----
unit_cfg = {}
for uid in UNITS:
    ta = templates[uid]
    pc = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
    tm = ta[:, pc]
    f0_scan = np.linspace(300, 4000, 50)
    ps = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
    tr = np.zeros(N + 2 * ps); ctr = ps + NT0MIN
    tr[ctr - NT0MIN: ctr - NT0MIN + N] = tm
    mg = [abs(wavelet_transform_at(tr, make_morlet(f, FS, N_CYCLES)[1], ctr)) for f in f0_scan]
    f0 = float(f0_scan[int(np.nanargmax(mg))])
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    rp = calibrate_reference_phase(tm, psi, FS, align_index=NT0MIN)
    s0b, _, qb = build_basis(tm, dt=1.0)
    st = np.sort(spike_times[spike_clusters == uid])
    unit_cfg[uid] = dict(peak_ch=pc, template=tm, template_norm=tm / (np.linalg.norm(tm) + 1e-12),
                          f0=f0, psi=psi, ref_phase=rp, s0=s0b, q=qb, st=st)
    print(f"unit {uid}: ch{pc}, f0={f0:.0f}Hz, {len(st):,} detected spikes")

channels_needed = sorted({unit_cfg[u]["peak_ch"] for u in UNITS})
ch_index = {ch: i for i, ch in enumerate(channels_needed)}

# ---- single chunked pass: build filtered traces for all needed channels ----
t0 = time.time()
traces = {ch: np.empty(WINDOW_END - WINDOW_START, dtype=np.float64) for ch in channels_needed}
pos = WINDOW_START
while pos < WINDOW_END:
    clen = min(CHUNK, WINDOW_END - pos)
    rlo = max(0, pos - OVERLAP)
    rhi = min(TOTAL_SAMPLES, pos + clen + OVERLAP)
    nread = rhi - rlo
    with open(DAT_PATH, "rb") as f:
        f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nread * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(nread, N_CHAN_BIN)
    lo_trim = pos - rlo
    for ch in channels_needed:
        filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
        traces[ch][pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lo_trim: lo_trim + clen]
    pos += clen
print(f"read+filtered {len(channels_needed)} channels over the window in {time.time()-t0:.0f}s")


def snippet_raw(trace, abs_sample):
    """Snippet at the raw candidate position -- NO alignment correction."""
    i = abs_sample - WINDOW_START
    lo = i - NT0MIN
    if lo < 0 or lo + N > len(trace):
        return None
    return trace[lo: lo + N].copy()


def snippet_aligned(trace, abs_sample, cfg):
    """Snippet after wavelet coarse+fine alignment."""
    i = abs_sample - WINDOW_START
    pad = 200
    if i - pad < 0 or i + pad >= len(trace):
        return None
    local = trace[i - pad: i + pad]
    r = coarse_then_fine_shift(local, pad, cfg["template"], cfg["psi"], cfg["f0"], FS,
                                reference_phase=cfg["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo = int(np.floor(tc - NT0MIN)) - 2
    hi = int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(local):
        return None
    ip = interp1d(np.arange(lo, hi), local[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(tc - NT0MIN + np.arange(N))


def score_all_configs(trace, abs_sample, cfg, W):
    """R2 under all four configurations for one location."""
    out = {}
    sn_a = snippet_aligned(trace, abs_sample, cfg)
    sn_r = snippet_raw(trace, abs_sample)
    if sn_a is None or sn_r is None:
        return None
    out["full"] = fit_nuisance_prealigned(sn_a, cfg["s0"], cfg["q"], whitening_matrix=W)["r_squared"]
    out["no_whiten"] = fit_nuisance_prealigned(sn_a, cfg["s0"], cfg["q"], whitening_matrix=None)["r_squared"]
    out["no_wavelet"] = fit_nuisance_prealigned(sn_r, cfg["s0"], cfg["q"], whitening_matrix=W)["r_squared"]
    out["neither"] = fit_nuisance_prealigned(sn_r, cfg["s0"], cfg["q"], whitening_matrix=None)["r_squared"]
    return out


def spatial_neighbor(uid, sample):
    my_pos = channel_positions[peak_ch_all[uid]]
    mask = np.abs(spike_times - sample) <= JITTER_WINDOW
    other = sorted(set(spike_clusters[mask].tolist()) - {uid})
    other = [oc for oc in other if oc < n_templ]
    best = None
    for oc in other:
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - my_pos) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (oc, d)
    return best


CONFIGS = ["full", "no_wavelet", "no_whiten", "neither"]
summary_rows = []
all_detail = []

for uid in UNITS:
    cfg = unit_cfg[uid]
    trace = traces[cfg["peak_ch"]]
    st = cfg["st"]
    st_in = st[(st >= WINDOW_START) & (st < WINDOW_END)]

    # whitening operator from real noise on this channel, inside this window
    noise_probe = trace[:60000]
    W, _, _ = build_whitening_from_noise(noise_probe, N, order=4)

    # ---- coarse peak scan over the whole window ----
    own_coarse = []
    for s in st_in[:200]:
        sn = snippet_raw(trace, int(s))
        if sn is not None:
            own_coarse.append(np.dot(sn, cfg["template_norm"]))
    floor = np.percentile(own_coarse, 10)

    score = correlate(trace, cfg["template_norm"], mode="valid", method="fft")
    peaks, _ = find_peaks(score, height=floor, distance=MIN_PEAK_DISTANCE)
    peak_abs = peaks + WINDOW_START + NT0MIN

    if len(st) > 0:
        di = np.searchsorted(st, peak_abs)
        d1 = np.abs(peak_abs - st[np.clip(di, 0, len(st) - 1)])
        d2 = np.abs(peak_abs - st[np.clip(di - 1, 0, len(st) - 1)])
        keep = np.minimum(d1, d2) > MARGIN
        candidates = peak_abs[keep]
    else:
        candidates = peak_abs

    # ---- noise-only null, per configuration ----
    rng = np.random.default_rng(0)
    # "Noise" must mean: no spike from a unit that actually puts signal on THIS
    # channel. Excluding on ALL ~470 clusters is both physically wrong (a unit
    # 3mm away contributes nothing here) and impossible -- the population is so
    # dense that essentially every sample in the session is within 200 samples
    # of some cluster's spike, which is what made the first attempt find zero
    # valid noise locations.
    my_pos_ch = channel_positions[peak_ch_all[uid]]
    near_units = [u for u in range(n_templ)
                  if np.sqrt(((channel_positions[peak_ch_all[u]] - my_pos_ch) ** 2).sum()) <= 100.0]
    near_mask = np.isin(spike_clusters, near_units)
    all_spk_in = spike_times[near_mask & (spike_times >= WINDOW_START) & (spike_times < WINDOW_END)]
    all_spk_in = np.sort(all_spk_in)
    print(f"  noise baseline excludes spikes from {len(near_units)} units within 100um "
          f"({len(all_spk_in):,} spikes in window)")
    null_scores = {c: [] for c in CONFIGS}
    tries = 0
    while len(null_scores["full"]) < N_NULL and tries < N_NULL * 40:
        tries += 1
        s = int(rng.integers(WINDOW_START + 1000, WINDOW_END - 1000))
        j = np.searchsorted(all_spk_in, s)
        near = min(abs(s - all_spk_in[min(j, len(all_spk_in) - 1)]),
                   abs(s - all_spk_in[max(j - 1, 0)])) if len(all_spk_in) else 1e9
        if near < 200:
            continue
        sc_ = score_all_configs(trace, s, cfg, W)
        if sc_ is None:
            continue
        for c in CONFIGS:
            null_scores[c].append(sc_[c])
    bars = {c: float(np.percentile(null_scores[c], 99)) for c in CONFIGS}

    # ---- score every candidate under every configuration ----
    rows = []
    for c_samp in candidates:
        sc_ = score_all_configs(trace, int(c_samp), cfg, W)
        if sc_ is None:
            continue
        rows.append(dict(sample=int(c_samp), **sc_))
    cdf = pd.DataFrame(rows)

    counts = {}
    for c in CONFIGS:
        counts[c] = int((cdf[c] >= bars[c]).sum()) if len(cdf) else 0

    # ---- classify the FULL-pipeline detections ----
    passing = cdf[cdf["full"] >= bars["full"]] if len(cdf) else cdf
    n_elsewhere, n_new = 0, 0
    for _, r in passing.iterrows():
        nb = spatial_neighbor(uid, int(r["sample"]))
        if nb is not None:
            n_elsewhere += 1
        else:
            n_new += 1
        all_detail.append(dict(uid=uid, sample=int(r["sample"]), r2_full=r["full"],
                                neighbor_uid=nb[0] if nb else None,
                                neighbor_dist=round(nb[1], 1) if nb else None))

    print(f"\n=== unit {uid} (ch{cfg['peak_ch']}, {len(st_in):,} own spikes in window) ===")
    print(f"  raw candidate locations scanned : {len(cdf):,}")
    print(f"  noise-null 99th pct bars        : " +
          ", ".join(f"{c}={bars[c]:.3f}" for c in CONFIGS))
    print(f"  DETECTIONS at matched 1% FP rate:")
    for c in CONFIGS:
        print(f"      {c:<11}: {counts[c]:,}")
    print(f"  of the {counts['full']:,} full-pipeline detections:")
    print(f"      already detected under another unit : {n_elsewhere:,}")
    print(f"      NEW (first detected in this analysis): {n_new:,}")

    summary_rows.append(dict(uid=uid, peak_ch=cfg["peak_ch"], own_spikes_in_window=len(st_in),
                              candidates_scanned=len(cdf),
                              det_full=counts["full"], det_no_wavelet=counts["no_wavelet"],
                              det_no_whiten=counts["no_whiten"], det_neither=counts["neither"],
                              already_elsewhere=n_elsewhere, new_detections=n_new))

sdf = pd.DataFrame(summary_rows)
print("\n\n================ SUMMARY (20 min of recording) ================")
print(sdf.to_string(index=False))
print("\nTOTALS")
print(f"  full-pipeline detections            : {sdf.det_full.sum():,}")
print(f"    already detected by another unit  : {sdf.already_elsewhere.sum():,}")
print(f"    NEW, first detected here          : {sdf.new_detections.sum():,}")
print("\nCOMPONENT CONTRIBUTION (detections at matched 1% false-positive rate)")
print(f"  full (wavelet + precision matrix)   : {sdf.det_full.sum():,}")
print(f"  without wavelet alignment           : {sdf.det_no_wavelet.sum():,}")
print(f"  without precision matrix            : {sdf.det_no_whiten.sum():,}")
print(f"  without either                      : {sdf.det_neither.sum():,}")

sdf.to_csv(os.path.join(OUT, "scan20min_summary.csv"), index=False)
pd.DataFrame(all_detail).to_csv(os.path.join(OUT, "scan20min_detections.csv"), index=False)
print("\nsaved scan20min_summary.csv / scan20min_detections.csv")
