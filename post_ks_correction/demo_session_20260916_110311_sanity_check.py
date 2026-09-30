"""
First port of the whitening + wavelet-alignment pipeline to a SECOND
session (20260916_110311), per the user's direction that this session's
sorted units are the ones to use going forward, not 20260901_085606.

Sanity check only: does the pipeline built and validated entirely on
20260901_085606 work at all on a completely different session's data? Same
recording rig/hardware assumed (same GAIN_TO_UV, NT0MIN=20, nt=61, 384
channels, 30kHz -- confirmed matching via this session's own ops.npy/
params.py), but GAIN_TO_UV itself could not be independently confirmed
from this session's own metadata (no gain field in ops.npy) -- carried
over from the other session's known value as an assumption, flagged here
rather than silently assumed.

Picks 4 "good"-labelled units spanning a range of spike counts, runs the
per-unit whitening + wavelet coarse/fine alignment + amplitude/stretch fit
on real detected spikes, and reports real R2/cosine-similarity numbers --
exactly the same diagnostic used throughout this project on the other
session.

Usage: python demo_session_20260916_110311_sanity_check.py
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
GAIN_TO_UV = 0.018311105685598315  # ASSUMED carried over from 20260901_085606 -- same rig, not independently confirmed here
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
N_TEST_SPIKES = 40

UNITS = [440, 302, 408, 317, 154]  # random, moderate spike-count sample (2000-25000) -- not top-by-count this time

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(peak_ch, center_sample, pad=100, filt_buf=60):
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


results = []
for uid in UNITS:
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    s0_basis, _, q_basis = build_basis(template, dt=1.0)

    f0 = best_f0_for(template)
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

    st = np.sort(spike_times[spike_clusters == uid])

    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, phi, sigma2 = build_whitening_from_noise(noise_trace, N, order=4)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(st), size=min(N_TEST_SPIKES, len(st)), replace=False)
    test_spikes = st[idx]

    r2s, cos_sims = [], []
    for s in test_spikes:
        trace = get_filtered(peak_ch, int(s), pad=100)
        nominal_center = 100
        result = coarse_then_fine_shift(trace, nominal_center, template, psi, f0, FS,
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
        r2s.append(fit["r_squared"])
        snippet_norm = snippet / (np.linalg.norm(snippet) + 1e-12)
        cos_sims.append(float(np.dot(snippet_norm, template_norm)))

    r2s, cos_sims = np.array(r2s), np.array(cos_sims)
    print(f"unit {uid} (ch{peak_ch}, {len(st)} spikes, f0={f0:.0f}Hz): "
          f"mean R2={r2s.mean():.3f}, mean cos_sim={cos_sims.mean():.3f}, n_tested={len(r2s)}")
    results.append(dict(uid=uid, peak_ch=peak_ch, n_spikes=len(st), f0=f0,
                         mean_r2=r2s.mean(), mean_cos_sim=cos_sims.mean()))

df = pd.DataFrame(results)
print("\n=== SUMMARY, session 20260916_110311 ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "session_20260916_110311_sanity_check.csv"), index=False)
