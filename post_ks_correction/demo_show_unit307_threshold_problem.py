"""
Makes the unit-307 threshold problem visible, not just numeric. Shows,
side by side, on real raw traces:
  TOP ROW    -- 6 real spikes Kilosort itself already reported for unit 307
  BOTTOM ROW -- 6 of the "candidate missed spike" locations that passed
                unit 307's own (too-lenient) quality bar in the sanity check

Point: if the bottom row looks visibly worse / less spike-like than the
top row, but still "passed", that proves the bar itself was the problem,
not that 14 real spikes were actually missed.

Usage: python demo_show_unit307_threshold_problem.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
UID = 307
MARGIN = 20
GAP_SCAN_STEP = 2
MISS_SCAN_AFTER = 150

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]
template_norm = template / (np.linalg.norm(template) + 1e-12)
s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)

f0_scan = np.linspace(300, 4000, 50)
pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
scan_trace = np.zeros(N + 2 * pad_scan)
scan_center = pad_scan + NT0MIN
scan_trace[scan_center - NT0MIN: scan_center - NT0MIN + N] = template
mags = [abs(wavelet_transform_at(scan_trace, make_morlet(f0, FS, N_CYCLES)[1], scan_center)) for f0 in f0_scan]
best_f0 = float(f0_scan[int(np.nanargmax(mags))])
_, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

st = np.sort(spike_times[spike_clusters == UID])
noise_start = int(st[0]) + 500000
with open(DAT_PATH, "rb") as f:
    f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
W, phi, sigma2 = build_whitening_from_noise(noise_trace, N, order=4)


def get_filtered(center_sample, pad=200, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


def process(center_sample, pad=200):
    trace = get_filtered(center_sample, pad=pad)
    nominal_center = pad
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, s0_basis, q_basis, whitening_matrix=W)
    return snippet, fit["r_squared"], fit["a"]


rng = np.random.default_rng(42)
sample_idx = rng.choice(len(st), size=40, replace=False)
sample_idx.sort()
sampled_spikes = st[sample_idx]

detected_examples = []
detected_r2_all = []
for s in sampled_spikes:
    r = process(int(s))
    if r is not None:
        detected_examples.append((int(s), r[0], r[1], r[2]))
        detected_r2_all.append(r[1])
detected_r2_all = np.array(detected_r2_all)
r2_low_bar = max(0.3, np.quantile(detected_r2_all, 0.10))
print(f"unit {UID}: mean detected R2={detected_r2_all.mean():.3f}, "
      f"10th-percentile bar used for 'missed spike' flag = {r2_low_bar:.3f}")

# real, ALREADY-DETECTED spikes -- 6 spanning the quality range (not cherry-picked
# to look bad: 3 near this unit's own median, 3 near this unit's own worst)
order = np.argsort([d[2] for d in detected_examples])
top_examples = [detected_examples[i] for i in order[-3:]]     # best-fitting real spikes
mid_examples = [detected_examples[i] for i in order[len(order)//2-1:len(order)//2+2]]  # median real spikes

# candidate "missed spike" locations -- same scan logic as the sanity check
detected_set = set(st.tolist())
candidates = []
for s in sampled_spikes:
    lo, hi = int(s) + MARGIN, int(s) + MISS_SCAN_AFTER
    trace = get_filtered(int(s), pad=max(200, hi - int(s) + 50))
    center_in_trace = max(200, hi - int(s) + 50)
    local_scores = []
    for c in range(lo, hi, GAP_SCAN_STEP):
        offset = c - int(s)
        lo_w = center_in_trace + offset - NT0MIN
        window = trace[lo_w: lo_w + N]
        if len(window) != N:
            continue
        local_scores.append((c, np.dot(window, template_norm)))
    if len(local_scores) < 3:
        continue
    samples_arr = np.array([c for c, _ in local_scores])
    scores_arr = np.array([sc for _, sc in local_scores])
    for i in range(1, len(scores_arr) - 1):
        if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
            cand_sample = int(samples_arr[i])
            if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                candidates.append(cand_sample)

candidate_examples = []
for c in candidates:
    r = process(c)
    if r is not None and r[1] >= r2_low_bar:
        candidate_examples.append((c, r[0], r[1], r[2]))
candidate_examples.sort(key=lambda d: -d[2])
shown_candidates = candidate_examples[:6]
print(f"showing {len(shown_candidates)} of {len(candidate_examples)} candidates that passed the bar")

# ============================================================
fig, axes = plt.subplots(2, 6, figsize=(22, 7), sharey=True)
t_ms = (np.arange(N) - NT0MIN) / FS * 1000
template_scaled_for_plot = template  # a=1 reference

real_examples_to_show = (mid_examples + top_examples)[:6]
for col, (sample, snip, r2, a) in enumerate(real_examples_to_show):
    ax = axes[0, col]
    ax.plot(t_ms, snip, color="#333", lw=1.6, label="real raw snippet")
    ax.plot(t_ms, template * a, color="#d62728", lw=1.3, ls="--", label=f"template * a={a:.1f}")
    ax.set_title(f"REAL detected spike\nsample {sample}\nR2={r2:.2f}", fontsize=9.5)
    ax.axhline(0, color="grey", lw=0.4)
    if col == 0:
        ax.set_ylabel("uV")
        ax.legend(fontsize=6.5, loc="upper right")

for col in range(6):
    ax = axes[1, col]
    if col < len(shown_candidates):
        sample, snip, r2, a = shown_candidates[col]
        ax.plot(t_ms, snip, color="#2ca02c", lw=1.6, label="candidate raw snippet")
        ax.plot(t_ms, template * a, color="#d62728", lw=1.3, ls="--", label=f"template * a={a:.1f}")
        ax.set_title(f"\"MISSED SPIKE\" CANDIDATE\nsample {sample}\nR2={r2:.2f} (passed bar={r2_low_bar:.2f})",
                     fontsize=9.5, color="#2ca02c")
    else:
        ax.axis("off")
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_xlabel("time (ms)")
    if col == 0:
        ax.set_ylabel("uV")
        ax.legend(fontsize=6.5, loc="upper right")

plt.suptitle(f"Unit {UID}: real detected spikes (top) vs. candidates that passed the SAME unit's own "
             f"low quality bar (bottom)\nmean R2 of this unit's real detected spikes = {detected_r2_all.mean():.2f} "
             f"-- the bar itself ({r2_low_bar:.2f}) is set low BECAUSE this unit's own real spikes are messy",
             fontsize=12.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "multiunit_02_unit307_threshold_problem.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
