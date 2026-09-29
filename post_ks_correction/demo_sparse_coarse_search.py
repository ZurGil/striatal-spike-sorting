"""
Tests the proposed optimization: instead of an EXHAUSTIVE coarse search
(every integer shift, step=1, currently 51 evaluations per spike), use a
SPARSE search with step size set from the unit's own half-cycle safe-zone
width (fs/(2*f0)) -- the coarse stage only needs to land within that zone,
not find the exact best integer, since the fine stage handles the rest.

Predicts: low-f0 units (wide safe zone) get big step sizes -> few
evaluations; high-f0 units (narrow safe zone) need small step sizes ->
still expensive. Verified here on two real units at very different
frequencies, checking BOTH speed and whether final accuracy survives.

Usage: python demo_sparse_coarse_search.py
"""
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, sub_sample_shift_from_phase

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
ITEMSIZE = 2
SEARCH_RADIUS = 25

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(peak_ch, center_sample, buf, filt_buf):
    s0 = center_sample - buf - filt_buf
    s1 = center_sample + buf + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    filtered = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filtered[filt_buf:-filt_buf]


def exhaustive_coarse(trace, nominal_center, template_norm, radius=SEARCH_RADIUS):
    n_eval = 0
    best_score, best_shift = -np.inf, 0
    for c in range(-radius, radius + 1):
        lo = nominal_center - NT0MIN + c
        window = trace[lo:lo + 61]
        score = np.dot(window, template_norm)
        n_eval += 1
        if score > best_score:
            best_score, best_shift = score, c
    return best_shift, n_eval


def sparse_coarse(trace, nominal_center, template_norm, step, radius=SEARCH_RADIUS):
    n_eval = 0
    best_score, best_shift = -np.inf, 0
    candidates = list(range(-radius, radius + 1, step))
    if candidates[-1] != radius:
        candidates.append(radius)  # make sure the edge is covered
    for c in candidates:
        lo = nominal_center - NT0MIN + c
        window = trace[lo:lo + 61]
        score = np.dot(window, template_norm)
        n_eval += 1
        if score > best_score:
            best_score, best_shift = score, c
    return best_shift, n_eval


def test_unit(uid, f0_hz, n_spikes_to_test=15, safety_fraction=0.5):
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    _, psi = make_morlet(f0_hz, FS, n_cycles=3.0)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

    half_cycle_samples = FS / (2 * f0_hz)
    step = max(1, int(half_cycle_samples * safety_fraction))
    print(f"\nunit {uid}: f0={f0_hz:.0f}Hz, half_cycle={half_cycle_samples:.1f} samples, "
          f"chosen sparse step={step} samples")

    st = np.sort(spike_times[spike_clusters == uid])
    test_spikes = st[:n_spikes_to_test]

    # buf must cover BOTH the probe's own half-width AND the coarse search's
    # radius plus the template's 61-sample extension beyond the nominal
    # center -- using just len(psi)//2 (as first written) under-sized this
    # for a short, high-f0 probe, silently truncating the coarse window at
    # the search edges (caught via a shape-mismatch crash, not silently)
    buf = max(len(psi) // 2, 61) + SEARCH_RADIUS + 30
    filt_buf = 60

    n_match, n_total = 0, 0
    t_exhaustive, t_sparse = 0.0, 0.0
    evals_exhaustive, evals_sparse = 0, 0
    diffs = []
    for spike_sample in test_spikes:
        trace = get_filtered(peak_ch, int(spike_sample), buf, filt_buf)
        nominal_center = buf + NT0MIN

        t0 = time.perf_counter()
        cshift_ex, ne = exhaustive_coarse(trace, nominal_center, template_norm)
        t_exhaustive += time.perf_counter() - t0
        evals_exhaustive += ne

        t0 = time.perf_counter()
        cshift_sp, ns = sparse_coarse(trace, nominal_center, template_norm, step)
        t_sparse += time.perf_counter() - t0
        evals_sparse += ns

        # finish BOTH with the identical fine stage, compare final totals
        def finish(cshift):
            fine_center = nominal_center + cshift
            w = wavelet_transform_at(trace, psi, fine_center)
            fine = sub_sample_shift_from_phase(np.angle(w), f0_hz, FS, reference_phase=ref_phase)
            return cshift + fine

        total_ex = finish(cshift_ex)
        total_sp = finish(cshift_sp)
        diff = abs(total_ex - total_sp)
        diffs.append(diff)
        n_total += 1
        if diff < 0.05:  # samples -- essentially identical final answer
            n_match += 1
        else:
            print(f"    spike {spike_sample}: exhaustive_coarse={cshift_ex}, sparse_coarse={cshift_sp}, "
                  f"exhaustive_total={total_ex:.2f}, sparse_total={total_sp:.2f}, diff={diff:.2f} samples")

    diffs = np.array(diffs)
    print(f"  final total_shift matched (within 0.05 samples) on {n_match}/{n_total} real spikes")
    print(f"  mismatch sizes (samples): mean={diffs.mean():.3f}, median={np.median(diffs):.3f}, "
          f"max={diffs.max():.3f}")
    print(f"  evaluations per spike: exhaustive={evals_exhaustive/n_total:.0f}, "
          f"sparse={evals_sparse/n_total:.0f}  ({evals_exhaustive/evals_sparse:.1f}x fewer)")
    print(f"  wall time (this test batch): exhaustive={t_exhaustive*1000:.1f}ms, sparse={t_sparse*1000:.1f}ms")


test_unit(342, 739.0)    # low f0 -- wide safe zone -- predict BIG step, big speedup
test_unit(224, 2808.0)   # high f0 -- narrow safe zone -- predict small step, small speedup
