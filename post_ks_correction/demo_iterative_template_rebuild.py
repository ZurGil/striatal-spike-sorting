"""
Rebuilds a unit's template from its OWN real spikes, using wavelet
alignment ONLY (no whitening in this loop -- see module docstring in the
user-requested build order): start from Kilosort's own template, pick the
matching frequency, coarse (sparse) search + wavelet fine correction for
every sampled real spike, average the aligned snippets into a new
template, then optionally repeat using the NEW template as the reference
and check whether it converges (stops moving).

Logic: if a unit is one real, coherent neuron, Kilosort's own template is
already an average of spikes that were only aligned to the nearest whole
sample -- sub-sample jitter blurs it. Realigning precisely first should
produce a SHARPER template. If it does NOT get sharper no matter how
carefully we align, that itself is informative: the unit's poor fit
quality is a real property of the unit, not an artifact of a blurry
reference template.

Whitening is applied only AFTERWARD, to score real spikes against the
FINAL rebuilt template -- not used inside the realignment/averaging loop
itself (see conversation: this keeps the two ideas separately testable).

Usage: python demo_iterative_template_rebuild.py
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
N_SPIKES_REBUILD = 150     # more spikes than the earlier 40-spike sanity sample -- averaging benefits from more
MAX_ITERS = 5
UNITS = [342, 307]  # 342 = control (already works well), 307 = the real test

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
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


def align_snippet(peak_ch, center_sample, template, psi, f0, ref_phase):
    trace = get_filtered(peak_ch, center_sample, pad=100)
    nominal_center = 100
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, f0, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    return interp(true_center - NT0MIN + np.arange(N))


results_summary = []
fig, axes = plt.subplots(2, len(UNITS), figsize=(7 * len(UNITS), 10))

for col, uid in enumerate(UNITS):
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    ks4_template = templ_all[:, peak_ch].copy()

    st = np.sort(spike_times[spike_clusters == uid])
    rng = np.random.default_rng(3)
    sample_idx = rng.choice(len(st), size=min(N_SPIKES_REBUILD, len(st)), replace=False)
    sampled_spikes = st[sample_idx]

    current_template = ks4_template.copy()
    history = [dict(iter=0, template=current_template.copy(),
                     ptp=current_template.max() - current_template.min())]

    for it in range(1, MAX_ITERS + 1):
        f0 = best_f0_for(current_template)
        _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
        ref_phase = calibrate_reference_phase(current_template, psi, FS, align_index=NT0MIN)

        aligned = []
        for s in sampled_spikes:
            snip = align_snippet(peak_ch, int(s), current_template, psi, f0, ref_phase)
            if snip is not None:
                aligned.append(snip)
        new_template = np.mean(aligned, axis=0)

        corr_to_prev = np.corrcoef(new_template, current_template)[0, 1]
        ptp_new = new_template.max() - new_template.min()
        print(f"unit {uid} iter {it}: f0={f0:.0f}Hz, n_aligned={len(aligned)}, "
              f"corr-to-previous={corr_to_prev:.5f}, peak-to-trough={ptp_new:.2f}uV")

        history.append(dict(iter=it, template=new_template.copy(), ptp=ptp_new, corr_to_prev=corr_to_prev))
        current_template = new_template
        if corr_to_prev > 0.99999:
            print(f"  converged (correlation to previous iteration > 0.99999), stopping early")
            break

    rebuilt_template = current_template
    ks4_ptp = ks4_template.max() - ks4_template.min()
    rebuilt_ptp = rebuilt_template.max() - rebuilt_template.min()
    shape_corr = np.corrcoef(ks4_template, rebuilt_template)[0, 1]

    # ---- score real spikes against BOTH templates, with whitening, to see if it actually matters ----
    noise_start = int(st[0]) + 500000
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

    def score_against(template, n_test=40):
        f0 = best_f0_for(template)
        _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
        ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
        s0_basis, _, q_basis = build_basis(template, dt=1.0)
        rng2 = np.random.default_rng(42)  # SAME seed/spikes as the earlier sanity check, for direct comparison
        test_idx = rng2.choice(len(st), size=min(n_test, len(st)), replace=False)
        r2s = []
        for s in st[test_idx]:
            snip = align_snippet(peak_ch, int(s), template, psi, f0, ref_phase)
            if snip is None:
                continue
            fit = fit_nuisance_prealigned(snip, s0_basis, q_basis, whitening_matrix=W)
            r2s.append(fit["r_squared"])
        return np.array(r2s)

    r2_ks4 = score_against(ks4_template)
    r2_rebuilt = score_against(rebuilt_template)

    print(f"\nunit {uid}: KS4 template peak-to-trough={ks4_ptp:.2f}uV, "
          f"REBUILT peak-to-trough={rebuilt_ptp:.2f}uV ({rebuilt_ptp/ks4_ptp:.2f}x)")
    print(f"unit {uid}: shape correlation KS4-vs-rebuilt = {shape_corr:.4f}")
    print(f"unit {uid}: mean R2 vs KS4 template = {r2_ks4.mean():.3f}, "
          f"mean R2 vs REBUILT template = {r2_rebuilt.mean():.3f}\n")

    results_summary.append(dict(uid=uid, n_iters=len(history) - 1, ks4_ptp=ks4_ptp, rebuilt_ptp=rebuilt_ptp,
                                 ptp_ratio=rebuilt_ptp / ks4_ptp, shape_corr=shape_corr,
                                 mean_r2_ks4_template=r2_ks4.mean(), mean_r2_rebuilt_template=r2_rebuilt.mean()))

    t_ms = (np.arange(N) - NT0MIN) / FS * 1000
    ax = axes[0, col]
    ax.plot(t_ms, ks4_template, color="#888", lw=2, label=f"Kilosort's own template\n(peak-to-trough={ks4_ptp:.1f}uV)")
    ax.plot(t_ms, rebuilt_template, color="#1f77b4", lw=2,
            label=f"REBUILT (wavelet-realigned avg)\n(peak-to-trough={rebuilt_ptp:.1f}uV)")
    ax.set_title(f"unit {uid}: template before/after realignment\nshape correlation={shape_corr:.3f}", fontsize=11)
    ax.set_xlabel("time (ms)")
    ax.legend(fontsize=8)

    ax = axes[1, col]
    ax.hist(r2_ks4, bins=15, alpha=0.55, color="#888", label=f"vs KS4 template\n(mean={r2_ks4.mean():.2f})")
    ax.hist(r2_rebuilt, bins=15, alpha=0.55, color="#1f77b4", label=f"vs REBUILT template\n(mean={r2_rebuilt.mean():.2f})")
    ax.set_xlabel("R2 (whitened fit), same 40 real spikes")
    ax.set_ylabel("count")
    ax.set_title(f"unit {uid}: does scoring against the rebuilt\ntemplate actually improve fit quality?", fontsize=10.5)
    ax.legend(fontsize=8)

summary_df = pd.DataFrame(results_summary)
print("=== SUMMARY ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT, "iterative_template_rebuild_summary.csv"), index=False)

plt.suptitle("Iterative wavelet-realigned template rebuild: control (342) vs. weak unit (307)", fontsize=13.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "template_rebuild_01.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
