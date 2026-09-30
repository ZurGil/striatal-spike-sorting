"""
Full missed-spike search, ported to session 20260916_110311, on units
already validated as clean/good in that session (440, 408, 302 --
mean R2 0.52-0.73 in the earlier sanity check). Same method built and
used throughout this project on 20260901_085606:

  1. wavelet alignment: pick this unit's own matching frequency, coarse
     (whole-sample) search, then phase-based fine sub-sample correction
  2. whitened (precision-matrix) scoring: fit amplitude+stretch against
     this unit's template using the noise-aware fit, not a plain
     unweighted one
  3. scan a window both before and after real detected spikes for
     un-detected local-maximum candidate events
  4. score each candidate the same way real spikes are scored, and
     compare against THIS unit's own real-spike score distribution
  5. for every candidate that scores as well as real spikes: check
     whether it already coincides (within 15 samples) with a spike
     ALREADY detected under a different, spatially-relevant (within 60um)
     unit, or whether it is genuinely unexplained by anyone

Reports, per unit: how many candidates were found, how many of those
already belong to a different unit, and how many are truly unexplained.

Usage: python demo_session_20260916_missed_spike_search.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315  # carried over from 20260901_085606, same rig -- not independently confirmed
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

GOOD_UNITS = [440, 408, 302]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
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
for uid in GOOD_UNITS:
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
                                            neighbor_uid=neighbor[0] if neighbor else None,
                                            neighbor_dist=neighbor[1] if neighbor else None))

    n_detected = len(detected_potential)
    n_elsewhere = sum(1 for d in detected_potential if d["neighbor_uid"] is not None)
    n_unexplained = n_detected - n_elsewhere

    print(f"unit {uid} (ch{peak_ch}, {len(st)} spikes, f0={f0:.0f}Hz): ref R2 p25={bar_p25:.3f}")
    print(f"  {n_detected} potential-missed-spike candidates detected")
    print(f"  {n_elsewhere}/{n_detected} ({100*n_elsewhere/n_detected if n_detected else float('nan'):.0f}%) "
          f"already belong to a different, spatially-relevant unit")
    print(f"  {n_unexplained}/{n_detected} ({100*n_unexplained/n_detected if n_detected else float('nan'):.0f}%) "
          f"genuinely unexplained (no match to any unit)")
    for d in detected_potential:
        tag = f"-> ALSO belongs to unit {d['neighbor_uid']} ({d['neighbor_dist']:.0f}um away)" if d["neighbor_uid"] is not None else "-> UNEXPLAINED"
        print(f"    sample {d['sample']}: R2={d['r_squared']:.2f} {tag}")

    summary.append(dict(uid=uid, n_ref=len(ref_r2), ref_p25=bar_p25, n_detected=n_detected,
                         n_elsewhere=n_elsewhere, n_unexplained=n_unexplained))

df = pd.DataFrame(summary)
print("\n=== SUMMARY, session 20260916_110311 ===")
print(df.to_string(index=False))
total_detected = df.n_detected.sum()
total_elsewhere = df.n_elsewhere.sum()
total_unexplained = df.n_unexplained.sum()
print(f"\nTOTAL across {len(GOOD_UNITS)} units: {total_detected} candidates found, "
      f"{total_elsewhere} ({100*total_elsewhere/total_detected if total_detected else float('nan'):.0f}%) already belong elsewhere, "
      f"{total_unexplained} ({100*total_unexplained/total_detected if total_detected else float('nan'):.0f}%) unexplained")
df.to_csv(os.path.join(OUT, "session_20260916_missed_spike_summary.csv"), index=False)
