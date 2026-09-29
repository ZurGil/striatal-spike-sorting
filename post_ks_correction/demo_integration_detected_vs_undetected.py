"""
First real integration test: pillar 1b (wavelet timing correction) +
pillar 2 (nuisance amplitude/timing/stretch fit), run together on REAL
bursts from unit 342. For every DETECTED (Kilosort-reported) spike in a
burst: coarse+fine align, then fit (a,tau,beta) on the realigned snippet.
Separately, scan the GAPS between detected spikes (and just after the
last one) for candidate events Kilosort did NOT report, run the identical
pipeline on them, and compare fit quality between the two groups.

Diagnostic only -- nothing here writes to any Kilosort output or spike
train. Real, targeted raw-data reads only (light I/O, same pattern used
throughout this session).

Usage: python demo_integration_detected_vs_undetected.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
UID = 342
ITEMSIZE = 2
F0_HZ = 739.0
ISI_THRESH_MS = 10.0
MIN_BURST_LEN = 4

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]
template_norm = template / (np.linalg.norm(template) + 1e-12)

_, psi = make_morlet(F0_HZ, FS, n_cycles=3.0)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)

st = np.sort(spike_times[spike_clusters == UID])
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def find_bursts(times_samples, isi_thresh_ms=ISI_THRESH_MS, min_len=MIN_BURST_LEN):
    isi_ms = np.diff(times_samples) / FS * 1000
    breaks = np.where(isi_ms > isi_thresh_ms)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(times_samples)]))
    bursts = []
    for s, e in zip(starts, ends):
        if e - s >= min_len:
            bursts.append(times_samples[s:e])
    return bursts


bursts = find_bursts(st)
bursts_sorted = sorted(bursts, key=len, reverse=True)
print(f"unit {UID}: {len(bursts)} bursts found, using top 5 longest")
chosen_bursts = bursts_sorted[:5]

PAD = 100
SEARCH_RADIUS = 25


def get_filtered(center_sample, pad=PAD, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    filtered = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filtered[filt_buf:-filt_buf]  # length 2*pad, center at index pad


def process_candidate(center_sample):
    trace = get_filtered(center_sample)
    nominal_center = PAD
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, F0_HZ, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    # BUG FOUND AND FIXED: extracting the snippet at round(total_shift) throws
    # away exactly the sub-sample correction pillar 1b computed, so
    # nuisance_model's own tau ends up re-absorbing that discarded fractional
    # part (plus real burst effects), inflating tau far above what alignment
    # alone should leave behind. Fix: INTERPOLATE the raw trace at the exact
    # fractional-shifted sample positions instead of rounding to an integer.
    from scipy.interpolate import interp1d
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + 61)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    local_interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int],
                             kind="cubic", bounds_error=False, fill_value=0.0)
    sample_positions = true_center - NT0MIN + np.arange(61)
    snippet = local_interp(sample_positions)
    fit = fit_nuisance(snippet, s0_basis, s0p_basis, q_basis)
    return dict(coarse=result["coarse_shift"], fine=result["fine_shift"],
                total_shift=result["total_shift"], coarse_score=result["coarse_score"],
                a=fit["a"], tau=fit["tau"], beta=fit["beta"], r_squared=fit["r_squared"])


# ---- process every DETECTED spike in the chosen bursts ----
detected_results = []
for burst in chosen_bursts:
    for s in burst:
        r = process_candidate(int(s))
        if r is not None:
            r["burst_len"] = len(burst)
            r["sample"] = int(s)
            detected_results.append(r)

# ---- scan GAPS for UNDETECTED candidates ----
# candidate positions: every integer sample strictly between consecutive
# detected spikes (skipping a small margin around each real spike so we
# don't just re-find the same detected spike), plus a window after the
# last spike in each burst
undetected_results = []
MARGIN = 8  # samples to exclude around each real detected spike
GAP_SCAN_STEP = 2  # coarse pre-scan step (refined by coarse_then_fine afterward anyway)
for burst in chosen_bursts:
    gap_candidates = []
    for i in range(len(burst) - 1):
        lo, hi = int(burst[i]) + MARGIN, int(burst[i + 1]) - MARGIN
        gap_candidates += list(range(lo, hi, GAP_SCAN_STEP))
    # also scan a window after the last spike
    lo, hi = int(burst[-1]) + MARGIN, int(burst[-1]) + 150
    gap_candidates += list(range(lo, hi, GAP_SCAN_STEP))

    for c in gap_candidates:
        trace = get_filtered(c)
        nominal_center = PAD
        window = trace[nominal_center - NT0MIN: nominal_center - NT0MIN + 61]
        score = np.dot(window, template_norm)
        undetected_results.append(dict(sample=c, coarse_score=score, burst_len=len(burst)))

undetected_df = pd.DataFrame(undetected_results)
detected_df = pd.DataFrame(detected_results)
print(f"detected spikes processed: {len(detected_df)}")
print(f"gap candidate LOCATIONS scanned: {len(undetected_df)}")

# rank gap candidates by coarse_score, keep local maxima above a real
# threshold: the noise floor established earlier this session (median
# ~308 in wavelet-magnitude units on this same channel) -- here using
# matched-filter score instead, so recompute a comparable reference: the
# median detected-spike score, and flag candidates reaching a meaningful
# fraction of it
median_detected_score = detected_df["coarse_score"].median()
print(f"median DETECTED spike matched-filter score: {median_detected_score:.0f}")

# keep only local maxima (score higher than both neighbors in the scan) to
# avoid double-counting a single bump many times at GAP_SCAN_STEP spacing
undetected_df = undetected_df.sort_values("sample").reset_index(drop=True)
is_local_max = np.zeros(len(undetected_df), dtype=bool)
scores = undetected_df["coarse_score"].to_numpy()
for i in range(1, len(scores) - 1):
    if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
        is_local_max[i] = True
peak_candidates = undetected_df[is_local_max].copy()
peak_candidates = peak_candidates.sort_values("coarse_score", ascending=False)
print(f"local-maximum gap candidates: {len(peak_candidates)}")

TOP_N_UNDETECTED = 12
top_undetected = peak_candidates.head(TOP_N_UNDETECTED)
undetected_processed = []
for _, row in top_undetected.iterrows():
    r = process_candidate(int(row["sample"]))
    if r is not None:
        r["sample"] = int(row["sample"])
        r["burst_len"] = row["burst_len"]
        undetected_processed.append(r)
undetected_processed_df = pd.DataFrame(undetected_processed)

print("\n=== DETECTED spikes: fit quality ===")
print(detected_df[["a", "tau", "beta", "r_squared", "coarse_score"]].describe())
print("\n=== TOP undetected gap candidates: fit quality ===")
if len(undetected_processed_df) > 0:
    print(undetected_processed_df[["a", "tau", "beta", "r_squared", "coarse_score"]].describe())
else:
    print("none processed")

detected_df.to_csv(os.path.join(OUT, "integration_detected_unit342.csv"), index=False)
undetected_processed_df.to_csv(os.path.join(OUT, "integration_undetected_unit342.csv"), index=False)
print("\nsaved CSVs")
