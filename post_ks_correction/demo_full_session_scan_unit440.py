"""
A REAL full-session scan, not a narrow spike-triggered search -- run only
on unit 440 (session 20260916_110311), the cleanest validated unit
available (R2=0.73 baseline, zero candidates found near its own spikes in
the earlier narrow test), per the instruction to spend this much more
expensive computation only on a unit already trusted.

This session's raw file is 290GB / 378 million samples / 210 minutes --
every sample must be touched at least once regardless of which channel we
care about, since channels are interleaved. Two passes:

PASS 1 (cheap, vectorized, covers the ENTIRE session): read the whole
recording in large sequential chunks (with overlap so a real event isn't
missed or double-counted at a chunk boundary), high-pass filter, compute a
plain matched-filter score against unit 440's own template at EVERY
sample via FFT-based correlation (not a python loop -- this is what makes
scanning 378 million samples tractable at all), find local-maximum peaks
above a real, data-derived floor (this unit's own real spikes' 10th-
percentile coarse score, not an arbitrary number), excluding anything
already within MARGIN samples of one of this unit's own detected spikes.

PASS 2 (expensive, only on the survivors from pass 1): the full wavelet
coarse+fine alignment + whitened amplitude/stretch fit, exactly as used
everywhere else in this project, plus the spatially-relevant-neighbor
provenance check.

Usage: python demo_full_session_scan_unit440.py
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
UID = 440

CHUNK_SAMPLES = 2_000_000
OVERLAP = 400  # >> filter transient + fine-stage's own window needs
MIN_PEAK_DISTANCE = 20  # samples, avoid double-counting one event

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

file_size = os.path.getsize(DAT_PATH)
TOTAL_SAMPLES = file_size // (N_CHAN_BIN * ITEMSIZE)
print(f"session: {TOTAL_SAMPLES:,} samples ({TOTAL_SAMPLES/FS/60:.1f} min), file {file_size/1e9:.1f}GB")

templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]
template_norm = template / (np.linalg.norm(template) + 1e-12)
s0_basis, _, q_basis = build_basis(template, dt=1.0)

f0_scan = np.linspace(300, 4000, 50)
pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
scan_trace = np.zeros(N + 2 * pad_scan)
scan_center = pad_scan + NT0MIN
scan_trace[scan_center - NT0MIN: scan_center - NT0MIN + N] = template
mags = [abs(wavelet_transform_at(scan_trace, make_morlet(f0, FS, N_CYCLES)[1], scan_center)) for f0 in f0_scan]
best_f0 = float(f0_scan[int(np.nanargmax(mags))])
_, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
print(f"unit {UID}: ch{peak_ch}, best_f0={best_f0:.0f}Hz")

st = np.sort(spike_times[spike_clusters == UID])
detected_sorted = st.copy()


def get_filtered(peak_ch, center_sample, pad, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


# ---- establish a REAL, data-derived floor from this unit's own real spikes' coarse score ----
rng = np.random.default_rng(3)
ref_idx = rng.choice(len(st), size=min(80, len(st)), replace=False)
own_coarse_scores = []
for s in st[ref_idx]:
    trace = get_filtered(peak_ch, int(s), pad=100)
    window = trace[100 - NT0MIN: 100 - NT0MIN + N]
    own_coarse_scores.append(np.dot(window, template_norm))
own_coarse_scores = np.array(own_coarse_scores)
height_floor = np.percentile(own_coarse_scores, 10)
print(f"own real-spike coarse score: median={np.median(own_coarse_scores):.1f}, "
      f"10th pctile (peak-finding floor)={height_floor:.1f}")

noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
noise_start = max(noise_start, 10000)
with open(DAT_PATH, "rb") as f:
    f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

# ============================================================
# PASS 1: full-session, chunked, vectorized peak scan
# ============================================================
t0 = time.time()
all_peak_samples = []
n_chunks = int(np.ceil(TOTAL_SAMPLES / CHUNK_SAMPLES))
chunk_start = 0
chunk_idx = 0
while chunk_start < TOTAL_SAMPLES:
    chunk_len = min(CHUNK_SAMPLES, TOTAL_SAMPLES - chunk_start)
    read_lo = max(0, chunk_start - OVERLAP)
    read_hi = min(TOTAL_SAMPLES, chunk_start + chunk_len + OVERLAP)
    n_read = read_hi - read_lo
    with open(DAT_PATH, "rb") as f:
        f.seek(int(read_lo) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    ch_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV

    score = correlate(ch_trace, template_norm, mode="valid", method="fft")
    # score[i] corresponds to window ch_trace[i : i+N] -- align "detected sample"
    # convention to i + NT0MIN (matches how real spikes are indexed elsewhere)
    score_abs_offset = read_lo + NT0MIN

    peaks, props = find_peaks(score, height=height_floor, distance=MIN_PEAK_DISTANCE)
    peak_samples_global = peaks + score_abs_offset

    # keep only peaks whose "core" region is inside this chunk's non-overlap zone
    # (avoids double-counting the same peak from two neighboring chunks)
    keep = (peak_samples_global >= chunk_start) & (peak_samples_global < chunk_start + chunk_len)
    peak_samples_global = peak_samples_global[keep]

    all_peak_samples.append(peak_samples_global)
    chunk_idx += 1
    if chunk_idx % 20 == 0:
        elapsed = time.time() - t0
        frac = chunk_start / TOTAL_SAMPLES
        print(f"  chunk {chunk_idx}/{n_chunks} ({frac*100:.1f}%), "
              f"{elapsed:.0f}s elapsed, {len(np.concatenate(all_peak_samples)):,} raw peaks so far")

    chunk_start += chunk_len

all_peaks = np.concatenate(all_peak_samples)
all_peaks.sort()
print(f"\nPASS 1 done in {time.time()-t0:.0f}s: {len(all_peaks):,} raw peaks across the full session")

# exclude anything within MARGIN samples of this unit's OWN detected spikes
det_idx = np.searchsorted(detected_sorted, all_peaks)
dist_to_detected = np.minimum(
    np.abs(all_peaks - detected_sorted[np.clip(det_idx, 0, len(detected_sorted) - 1)]),
    np.abs(all_peaks - detected_sorted[np.clip(det_idx - 1, 0, len(detected_sorted) - 1)]))
candidates = all_peaks[dist_to_detected > MARGIN]
print(f"after excluding proximity to unit {UID}'s own {len(st)} detected spikes: "
      f"{len(candidates):,} genuinely independent candidate locations")

np.save(os.path.join(OUT, f"fullscan_unit{UID}_raw_candidates.npy"), candidates)

# ============================================================
# PASS 2: refine every survivor with the full pipeline
# ============================================================
def find_spatial_neighbor_match(uid, sample, radius_um=60.0, jitter_window=15):
    my_pos = channel_positions[peak_ch_all[uid]]
    mask = np.abs(spike_times - sample) <= jitter_window
    other = sorted(set(spike_clusters[mask].tolist()) - {uid})
    other = [oc for oc in other if oc < n_templ]
    best = None
    for oc in other:
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - my_pos) ** 2).sum())
        if d <= radius_um and (best is None or d < best[1]):
            best = (oc, d)
    return best


refined = []
for c in candidates:
    trace = get_filtered(peak_ch, int(c), pad=200)
    nominal_center = 200
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        continue
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, s0_basis, q_basis, whitening_matrix=W)
    refined.append(dict(sample=int(c), r_squared=fit["r_squared"], a=fit["a"]))

refined_df = pd.DataFrame(refined)
refined_df.to_csv(os.path.join(OUT, f"fullscan_unit{UID}_refined.csv"), index=False)
print(f"\nPASS 2 done: {len(refined_df)} candidates refined through full wavelet+whitened fit")

# use the SAME reference-quality-bar convention as the narrow-scan tests
ref_r2 = []
for s in st[ref_idx]:
    trace = get_filtered(peak_ch, int(s), pad=200)
    nominal_center = 200
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        continue
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, s0_basis, q_basis, whitening_matrix=W)
    ref_r2.append(fit["r_squared"])
ref_r2 = np.array(ref_r2)
bar_p25 = np.percentile(ref_r2, 25)
print(f"unit {UID}'s own real-spike R2: median={np.median(ref_r2):.3f}, p25={bar_p25:.3f}")

passing = refined_df[refined_df.r_squared >= bar_p25] if len(refined_df) else refined_df
print(f"\ncandidates reaching real-spike quality (R2 >= {bar_p25:.3f}): {len(passing)} / {len(refined_df)}")

n_elsewhere, n_unexplained = 0, 0
rows_out = []
for _, row in passing.iterrows():
    nb = find_spatial_neighbor_match(UID, int(row["sample"]))
    if nb is not None:
        n_elsewhere += 1
    else:
        n_unexplained += 1
    rows_out.append(dict(sample=int(row["sample"]), r_squared=row["r_squared"],
                          neighbor_uid=nb[0] if nb else None, neighbor_dist=nb[1] if nb else None))

print(f"  already belong to a spatially-relevant different unit: {n_elsewhere}")
print(f"  genuinely unexplained: {n_unexplained}")
for r in rows_out:
    tag = f"-> unit {r['neighbor_uid']} ({r['neighbor_dist']:.0f}um)" if r["neighbor_uid"] is not None else "-> UNEXPLAINED"
    print(f"    sample {r['sample']}: R2={r['r_squared']:.2f} {tag}")

pd.DataFrame(rows_out).to_csv(os.path.join(OUT, f"fullscan_unit{UID}_final_candidates.csv"), index=False)
print("\ndone")
