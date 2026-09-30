"""
Sanity check requested directly: rerun the missed-spike-search pipeline on
3 units never touched by any earlier demo in this project (285, 295, 48 --
picked by sep_vs_noise, high quality, no prior history), and report two
plain numbers per unit: how many candidate ("potential missed spike")
events were found, and what fraction of those already belong to a
spatially-relevant OTHER unit (i.e. are not really "missed" at all, just
mis-labeled) -- exactly the pattern found across the earlier 6-unit batch,
tested here on a genuinely fresh sample.

Usage: python demo_sanity_check_3fresh_units.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
N_REF_SPIKES = 60
N_SCAN_SPIKES = 60
SCAN_HALF_WINDOW = 150
MARGIN = 20
GAP_SCAN_STEP = 2
TOP_N_CANDIDATES = 25
JITTER_WINDOW = 15
RADIUS_UM = 60.0

FRESH_UNITS = [285, 295, 232]  # 48 dropped: 0 spikes in final spike_clusters.npy (stale review-JSON id)

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
channel_positions = np.load(VR + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(peak_ch, center_sample, pad, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


def best_f0_for(template):
    f0_scan = np.linspace(300, 4000, 50)
    pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
    trace = np.zeros(N + 2 * pad_scan)
    center = pad_scan + NT0MIN
    trace[center - NT0MIN: center - NT0MIN + N] = template
    mags = [abs(wavelet_transform_at(trace, make_morlet(f0, FS, N_CYCLES)[1], center)) for f0 in f0_scan]
    return float(f0_scan[int(np.nanargmax(mags))])


def align_and_fit(peak_ch, center_sample, template, psi, f0, ref_phase, s0_basis, q_basis, W, pad=200):
    trace = get_filtered(peak_ch, center_sample, pad=pad)
    nominal_center = pad
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, f0, FS,
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
    return dict(r_squared=fit["r_squared"], a=fit["a"])


def find_spatial_neighbor_match(uid, sample):
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


summary = []
for uid in FRESH_UNITS:
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    s0_basis, _, q_basis = build_basis(template, dt=1.0)

    f0 = best_f0_for(template)
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

    st = np.sort(spike_times[spike_clusters == uid])
    detected_set = set(st.tolist())

    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

    rng = np.random.default_rng(7)
    ref_idx = rng.choice(len(st), size=min(N_REF_SPIKES, len(st)), replace=False)
    ref_r2 = np.array([r["r_squared"] for r in
                        (align_and_fit(peak_ch, int(s), template, psi, f0, ref_phase, s0_basis, q_basis, W)
                         for s in st[ref_idx]) if r is not None])
    bar_p25 = np.percentile(ref_r2, 25)

    rng2 = np.random.default_rng(11)
    scan_idx = rng2.choice(len(st), size=min(N_SCAN_SPIKES, len(st)), replace=False)
    scan_spikes = st[scan_idx]

    coarse_candidates = []
    for s in scan_spikes:
        lo, hi = int(s) - SCAN_HALF_WINDOW, int(s) + SCAN_HALF_WINDOW
        trace = get_filtered(peak_ch, int(s), pad=SCAN_HALF_WINDOW + 60)
        center_in_trace = SCAN_HALF_WINDOW + 60
        local = []
        for c in range(lo, hi, GAP_SCAN_STEP):
            offset = c - int(s)
            lo_w = center_in_trace + offset - NT0MIN
            window = trace[lo_w: lo_w + N]
            if len(window) != N:
                continue
            local.append((c, np.dot(window, template_norm)))
        if len(local) < 3:
            continue
        samples_arr = np.array([c for c, _ in local])
        scores_arr = np.array([sc for _, sc in local])
        for i in range(1, len(scores_arr) - 1):
            if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
                cand_sample = int(samples_arr[i])
                if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                    coarse_candidates.append((cand_sample, scores_arr[i]))
    coarse_candidates = sorted(set(coarse_candidates), key=lambda x: -x[1])[:TOP_N_CANDIDATES]

    detected_potential = []
    for cand_sample, _ in coarse_candidates:
        r = align_and_fit(peak_ch, cand_sample, template, psi, f0, ref_phase, s0_basis, q_basis, W)
        if r is not None and r["r_squared"] >= bar_p25:
            neighbor = find_spatial_neighbor_match(uid, cand_sample)
            detected_potential.append(dict(sample=cand_sample, r_squared=r["r_squared"],
                                            neighbor_uid=neighbor[0] if neighbor else None))

    n_detected = len(detected_potential)
    n_assigned_elsewhere = sum(1 for d in detected_potential if d["neighbor_uid"] is not None)
    pct = 100 * n_assigned_elsewhere / n_detected if n_detected > 0 else float("nan")

    print(f"unit {uid} (ch{peak_ch}, {len(st)} spikes): ref R2 p25={bar_p25:.3f} | "
          f"{n_detected} 'potential missed spike' candidates detected, "
          f"{n_assigned_elsewhere}/{n_detected} ({pct:.0f}%) already belong to a spatially-relevant other unit")
    if n_detected > 0:
        for d in detected_potential:
            tag = f"-> unit {d['neighbor_uid']}" if d["neighbor_uid"] is not None else "-> no match, unexplained"
            print(f"    sample {d['sample']}: R2={d['r_squared']:.2f} {tag}")

    summary.append(dict(uid=uid, n_detected=n_detected, n_assigned_elsewhere=n_assigned_elsewhere, pct=pct))

print("\n=== SUMMARY ===")
df = pd.DataFrame(summary)
print(df.to_string(index=False))
total_detected = df.n_detected.sum()
total_elsewhere = df.n_assigned_elsewhere.sum()
print(f"\nTOTAL across 3 fresh units: {total_detected} candidates, "
      f"{total_elsewhere}/{total_detected} ({100*total_elsewhere/total_detected:.0f}%) already assigned elsewhere")
