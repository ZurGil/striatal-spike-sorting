"""
Scaled-up version of the detected-vs-undetected integration test: ALL real
bursts for unit 342 (not just the top 5 longest), using the CORRECTED
2-parameter prealigned fit throughout. Margin around detected spikes
widened at the source (20 samples, not 8) based on the earlier lesson,
PLUS a post-hoc distance safety check, so margin-artifacts don't
contaminate the results again.

Diagnostic only -- no spike-train modification.

Usage: python demo_full_scale_search.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned

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
SEARCH_RADIUS = 25
PAD = 100
MARGIN = 20          # widened from 8 -- the earlier margin-artifact lesson
GAP_SCAN_STEP = 2
SAFETY_DIST = 30      # post-hoc double-check, same as before
TOP_N_UNDETECTED = 40  # widened from 12, matching the larger scan

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


all_bursts = find_bursts(st)
print(f"unit {UID}: using ALL {len(all_bursts)} bursts (was 5 before), "
      f"{sum(len(b) for b in all_bursts)} total spikes in bursts")


def get_filtered(center_sample, pad=PAD, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    filtered = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filtered[filt_buf:-filt_buf]


def process_candidate(center_sample):
    trace = get_filtered(center_sample)
    nominal_center = PAD
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, F0_HZ, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + 61)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    local_interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int],
                             kind="cubic", bounds_error=False, fill_value=0.0)
    sample_positions = true_center - NT0MIN + np.arange(61)
    snippet = local_interp(sample_positions)
    fit = fit_nuisance_prealigned(snippet, s0_basis, q_basis)
    return dict(coarse=result["coarse_shift"], fine=result["fine_shift"],
                total_shift=result["total_shift"], coarse_score=result["coarse_score"],
                a=fit["a"], beta=fit["beta"], r_squared=fit["r_squared"])


# ---- every DETECTED spike, ALL bursts ----
detected_results = []
for burst in all_bursts:
    for s in burst:
        r = process_candidate(int(s))
        if r is not None:
            r["burst_len"] = len(burst)
            r["sample"] = int(s)
            detected_results.append(r)
detected_df = pd.DataFrame(detected_results)
det_samples_all = detected_df["sample"].to_numpy()
print(f"detected spikes processed: {len(detected_df)}")

# ---- scan ALL gaps, ALL bursts ----
undetected_results = []
for burst in all_bursts:
    gap_candidates = []
    for i in range(len(burst) - 1):
        lo, hi = int(burst[i]) + MARGIN, int(burst[i + 1]) - MARGIN
        gap_candidates += list(range(lo, hi, GAP_SCAN_STEP))
    lo, hi = int(burst[-1]) + MARGIN, int(burst[-1]) + 150
    gap_candidates += list(range(lo, hi, GAP_SCAN_STEP))

    for c in gap_candidates:
        trace = get_filtered(c)
        nominal_center = PAD
        window = trace[nominal_center - NT0MIN: nominal_center - NT0MIN + 61]
        score = np.dot(window, template_norm)
        undetected_results.append(dict(sample=c, coarse_score=score, burst_len=len(burst)))

undetected_df = pd.DataFrame(undetected_results)
print(f"gap candidate LOCATIONS scanned: {len(undetected_df)}")

undetected_df = undetected_df.sort_values("sample").reset_index(drop=True)
scores = undetected_df["coarse_score"].to_numpy()
is_local_max = np.zeros(len(undetected_df), dtype=bool)
for i in range(1, len(scores) - 1):
    if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
        is_local_max[i] = True
peak_candidates = undetected_df[is_local_max].copy()

# safety check: distance to nearest REAL detected spike (of this unit,
# not just within the burst) -- catches margin-artifacts even if MARGIN
# itself wasn't wide enough somewhere
peak_candidates["dist_to_nearest_detected"] = [
    np.min(np.abs(det_samples_all - s)) for s in peak_candidates["sample"]]
n_before = len(peak_candidates)
peak_candidates = peak_candidates[peak_candidates["dist_to_nearest_detected"] > SAFETY_DIST].copy()
print(f"local-maximum candidates: {n_before}, after safety-distance filter: {len(peak_candidates)}")

peak_candidates = peak_candidates.sort_values("coarse_score", ascending=False)
top_undetected = peak_candidates.head(TOP_N_UNDETECTED)

undetected_processed = []
for _, row in top_undetected.iterrows():
    r = process_candidate(int(row["sample"]))
    if r is not None:
        r["sample"] = int(row["sample"])
        r["burst_len"] = row["burst_len"]
        r["dist_to_nearest_detected"] = row["dist_to_nearest_detected"]
        undetected_processed.append(r)
undetected_processed_df = pd.DataFrame(undetected_processed)

print(f"\nprocessed {len(undetected_processed_df)} genuinely independent undetected candidates")
print("\n=== DETECTED spikes: fit quality ===")
print(detected_df[["a", "beta", "r_squared"]].describe())
print("\n=== INDEPENDENT undetected candidates: fit quality ===")
print(undetected_processed_df[["a", "beta", "r_squared"]].describe())

# how many undetected candidates look genuinely spike-like: R2 within the
# range real detected spikes occupy (using detected spikes' own 10th
# percentile as the bar, not an arbitrary number)
r2_bar = detected_df["r_squared"].quantile(0.10)
promising = undetected_processed_df[undetected_processed_df["r_squared"] >= r2_bar]
print(f"\ndetected spikes' own 10th-percentile R2 = {r2_bar:.3f}")
print(f"undetected candidates reaching that bar: {len(promising)}/{len(undetected_processed_df)}")
if len(promising) > 0:
    print(promising[["sample", "a", "beta", "r_squared", "dist_to_nearest_detected"]].to_string())

detected_df.to_csv(os.path.join(OUT, "fullscale_detected_unit342.csv"), index=False)
undetected_processed_df.to_csv(os.path.join(OUT, "fullscale_undetected_unit342.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
ax = axes[0]
ax.scatter(detected_df["a"], detected_df["r_squared"], color="#1f77b4", s=45, label=f"DETECTED (n={len(detected_df)})")
ax.scatter(undetected_processed_df["a"], undetected_processed_df["r_squared"], color="#2ca02c", s=45, marker="^",
           label=f"independent undetected (n={len(undetected_processed_df)})")
ax.axhline(r2_bar, color="grey", lw=1, ls=":", label=f"detected 10th-pctile R2={r2_bar:.2f}")
ax.set_xlabel("fitted amplitude a")
ax.set_ylabel("R-squared")
ax.set_title(f"FULL SCALE: all {len(all_bursts)} bursts,\n{len(detected_df)} detected + {len(undetected_processed_df)} independent candidates", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(detected_df["beta"], bins=15, alpha=0.6, color="#1f77b4", label="DETECTED")
ax.hist(undetected_processed_df["beta"], bins=15, alpha=0.6, color="#2ca02c", label="undetected candidates")
ax.set_xlabel("beta (stretch)")
ax.set_ylabel("count")
ax.set_title("beta distributions at full scale", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[2]
ax.axis("off")
ax.text(0, 1.0,
    f"FULL-SCALE RESULT\n\n"
    f"{len(all_bursts)} bursts (not 5), {len(detected_df)} detected spikes fit\n"
    f"{n_before} local-max gap candidates found\n"
    f"{n_before - len(peak_candidates)} were margin/safety-distance artifacts (excluded)\n"
    f"{len(peak_candidates)} genuinely independent candidates existed\n"
    f"top {len(undetected_processed_df)} processed through full pipeline\n\n"
    f"detected R2: mean={detected_df.r_squared.mean():.2f}, "
    f"10th pctile={r2_bar:.2f}\n"
    f"undetected R2: mean={undetected_processed_df.r_squared.mean():.2f}, "
    f"max={undetected_processed_df.r_squared.max():.2f}\n\n"
    f"candidates reaching detected-spike quality bar:\n"
    f"  {len(promising)} / {len(undetected_processed_df)}",
    fontsize=10, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Full-scale search: ALL unit-342 bursts, corrected 2-parameter fit", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_28_fullscale_search.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
