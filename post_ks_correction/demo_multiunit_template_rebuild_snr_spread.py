"""
Same iterative wavelet-realigned template rebuild as
demo_iterative_template_rebuild.py, generalized to more units, chosen
deliberately to span a wide range of bombcell's independent SNR metric
(outputs/bombcell_results/templates._bc_qMetrics.csv, confirmed to match
this session via exact n_spikes agreement) -- NOT picked to confirm a
story, picked to cover the range: SNR from ~26 (weakest) to ~182
(strongest), plus units 342 and 307 kept from the earlier single-pair test
as anchors (bombcell SNR 79 and 77 respectively -- nearly identical to
each other despite behaving completely differently in this project's own
diagnostics, which is itself worth checking against a wider set).

For each unit: converges or doesn't, how much the shape changes
(correlation to Kilosort's own template), and whether scoring real spikes
against the rebuilt template helps or hurts, vs. bombcell's independent
SNR number -- does OUR metric track theirs, or catch something different?

Usage: python demo_multiunit_template_rebuild_snr_spread.py
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
BC_PATH = r"D:\Gil\spike_sorting_agent\outputs\bombcell_results\templates._bc_qMetrics.csv"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
N_SPIKES_REBUILD = 100
MAX_ITERS = 4
UNITS = [336, 6, 397, 171, 313, 135, 342, 307]  # spans bombcell SNR ~26 to ~182; 342/307 are known anchors

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
bc = pd.read_csv(BC_PATH).set_index("phy_clusterID")


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


results = []
for uid in UNITS:
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    ks4_template = templ_all[:, peak_ch].copy()
    snr = float(bc.loc[uid, "signalToNoiseRatio"]) if uid in bc.index else np.nan

    st = np.sort(spike_times[spike_clusters == uid])
    rng = np.random.default_rng(3)
    sample_idx = rng.choice(len(st), size=min(N_SPIKES_REBUILD, len(st)), replace=False)
    sampled_spikes = st[sample_idx]

    current_template = ks4_template.copy()
    n_iters_run = 0
    converged = False
    for it in range(1, MAX_ITERS + 1):
        f0 = best_f0_for(current_template)
        _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
        ref_phase = calibrate_reference_phase(current_template, psi, FS, align_index=NT0MIN)
        aligned = [a for a in (align_snippet(peak_ch, int(s), current_template, psi, f0, ref_phase)
                                for s in sampled_spikes) if a is not None]
        new_template = np.mean(aligned, axis=0)
        corr_to_prev = np.corrcoef(new_template, current_template)[0, 1]
        current_template = new_template
        n_iters_run = it
        if corr_to_prev > 0.99999:
            converged = True
            break

    rebuilt_template = current_template
    shape_corr = np.corrcoef(ks4_template, rebuilt_template)[0, 1]

    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

    def score_against(template, n_test=30):
        f0t = best_f0_for(template)
        _, psit = make_morlet(f0t, FS, n_cycles=N_CYCLES)
        ref_phase_t = calibrate_reference_phase(template, psit, FS, align_index=NT0MIN)
        s0_basis, _, q_basis = build_basis(template, dt=1.0)
        rng2 = np.random.default_rng(42)
        test_idx = rng2.choice(len(st), size=min(n_test, len(st)), replace=False)
        r2s = []
        for s in st[test_idx]:
            snip = align_snippet(peak_ch, int(s), template, psit, f0t, ref_phase_t)
            if snip is None:
                continue
            fit = fit_nuisance_prealigned(snip, s0_basis, q_basis, whitening_matrix=W)
            r2s.append(fit["r_squared"])
        return np.array(r2s)

    r2_ks4 = score_against(ks4_template)
    r2_rebuilt = score_against(rebuilt_template)

    print(f"unit {uid} (SNR={snr:.1f}, n_spikes={len(st)}): converged={converged} after {n_iters_run} iters, "
          f"shape_corr={shape_corr:.3f}, R2 ks4={r2_ks4.mean():.3f} -> rebuilt={r2_rebuilt.mean():.3f}")

    results.append(dict(uid=uid, snr=snr, n_spikes=len(st), n_iters=n_iters_run, converged=converged,
                         shape_corr=shape_corr, r2_ks4=r2_ks4.mean(), r2_rebuilt=r2_rebuilt.mean(),
                         r2_delta=r2_rebuilt.mean() - r2_ks4.mean()))

df = pd.DataFrame(results).sort_values("snr")
print("\n=== SUMMARY, sorted by bombcell SNR ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "multiunit_template_rebuild_snr_spread.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
colors = ["#2ca02c" if c else "#d62728" for c in df["converged"]]
ax.scatter(df["snr"], df["shape_corr"], c=colors, s=90)
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.snr, r.shape_corr), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("bombcell SNR (independent metric)")
ax.set_ylabel("shape correlation: rebuilt vs. Kilosort's own template")
ax.set_title("Does realigning change the template's shape?\ngreen=converged within 4 iters, red=did not", fontsize=10.5)

ax = axes[1]
ax.scatter(df["snr"], df["r2_delta"], c=colors, s=90)
ax.axhline(0, color="grey", lw=1, ls=":")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.snr, r.r2_delta), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("bombcell SNR (independent metric)")
ax.set_ylabel("R2 change (rebuilt template minus KS4 template)")
ax.set_title("Does the rebuilt template fit BETTER or WORSE\nthan Kilosort's own?", fontsize=10.5)

ax = axes[2]
ax.scatter(df["snr"], df["r2_ks4"], s=90, color="#1f77b4")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.snr, r.r2_ks4), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("bombcell SNR (independent metric)")
ax.set_ylabel("mean R2 vs Kilosort's own template")
ax.set_title("Does bombcell's SNR predict OUR fit-quality\nmetric at all?", fontsize=10.5)

plt.suptitle("Template rebuild across units spanning bombcell SNR ~26-180", fontsize=13.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "template_rebuild_02_snr_spread.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
