"""
Real, deployable two-pass frequency correction, compared against both the
naive fixed-f0 method and the (unrealistic) oracle from the previous test.

PASS 1: coarse_then_fine_shift with the unit's fixed rest-state f0, then
        fit_nuisance_prealigned on the aligned snippet -> a rough,
        first-pass beta estimate (itself somewhat biased/noisy -- that's
        fine, it only needs to be informative, not perfect).
PASS 2: f0_adjusted = f0_rest / (1 + beta_estimate)  -- exact consequence
        of time-stretching a signal by (1+beta) compressing its frequency
        content by the same factor, verified against real scan data in
        this project's conversation history (predicted vs observed
        best_f0 differed by only 6-37 Hz, within the scan's own ~92Hz
        resolution). Build a fresh probe at f0_adjusted, recalibrate its
        reference phase against the RESTED template (still the only
        thing known ahead of time), and re-run coarse_then_fine_shift on
        the SAME raw trace for a refined timing correction.

Usage: python demo_two_pass_vs_oracle.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import synthesize_deformed_spike, build_basis, fit_nuisance_prealigned

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
NT0MIN = 20
UID = 342
F0_REST = 739.0
SEARCH_RADIUS = 25
N_CYCLES = 3.0
BETA_CLIP = (-0.3, 1.0)   # guard against pathological first-pass beta estimates
F0_CLIP = (200.0, 4000.0)  # guard against pathological adjusted frequencies

templates = np.load(VR + r"\templates.npy")
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]
s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)

_, psi_rest = make_morlet(F0_REST, FS, n_cycles=N_CYCLES)
ref_phase_rest = calibrate_reference_phase(template, psi_rest, FS, align_index=NT0MIN)

# cache of (adjusted f0 -> (psi, ref_phase)) so repeated nearby beta
# estimates don't rebuild the probe from scratch every single trial
_probe_cache = {}


def get_probe_for_f0(f0_hz):
    key = round(f0_hz)
    if key not in _probe_cache:
        _, psi = make_morlet(f0_hz, FS, n_cycles=N_CYCLES)
        ref = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
        _probe_cache[key] = (psi, ref)
    return _probe_cache[key]


def two_pass_shift(trace, center_index):
    # PASS 1: fixed rest-state f0
    r1 = coarse_then_fine_shift(trace, center_index, template, psi_rest, F0_REST, FS,
                                 reference_phase=ref_phase_rest, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center1 = center_index + r1["total_shift"]
    from scipy.interpolate import interp1d
    lo_int = int(np.floor(true_center1 - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center1 - NT0MIN + 61)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return r1["total_shift"], np.nan
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center1 - NT0MIN + np.arange(61))
    fit1 = fit_nuisance_prealigned(snippet, s0_basis, q_basis)
    beta_est = np.clip(fit1["beta"], *BETA_CLIP) if not np.isnan(fit1["beta"]) else 0.0

    # PASS 2: adjust f0 using the EXACT time-stretch relationship, re-align
    f0_adjusted = np.clip(F0_REST / (1 + beta_est), *F0_CLIP)
    psi_adj, ref_adj = get_probe_for_f0(f0_adjusted)
    r2 = coarse_then_fine_shift(trace, center_index, template, psi_adj, f0_adjusted, FS,
                                 reference_phase=ref_adj, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    return r2["total_shift"], beta_est


rng = np.random.default_rng(11)
noise_sigma = 0.05 * np.abs(template).max()
pad = 200
beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
true_taus = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
N_TRIALS = 6

results = []
for beta in beta_values:
    for true_tau in true_taus:
        errs_fixed, errs_twopass = [], []
        beta_ests = []
        for _ in range(N_TRIALS):
            clean = synthesize_deformed_spike(template, a=1.0, tau=true_tau, beta=beta, noise_sigma=0.0)
            noisy = clean + rng.normal(0, noise_sigma, size=len(clean))
            trace = np.zeros(len(noisy) + 2 * pad)
            true_center = pad + NT0MIN
            trace[true_center - NT0MIN: true_center - NT0MIN + len(noisy)] = noisy

            r_fixed = coarse_then_fine_shift(trace, true_center, template, psi_rest, F0_REST, FS,
                                              reference_phase=ref_phase_rest, search_radius=SEARCH_RADIUS,
                                              nt0min=NT0MIN)
            errs_fixed.append(abs(r_fixed["total_shift"] - true_tau))

            shift_2p, beta_est = two_pass_shift(trace, true_center)
            errs_twopass.append(abs(shift_2p - true_tau))
            beta_ests.append(beta_est)

        results.append(dict(beta=beta, true_tau=true_tau,
                             mean_err_fixed=np.mean(errs_fixed),
                             mean_err_twopass=np.mean(errs_twopass),
                             mean_beta_est_pass1=np.mean(beta_ests)))

df = pd.DataFrame(results)
summary = df.groupby("beta")[["mean_err_fixed", "mean_err_twopass", "mean_beta_est_pass1"]].mean().reset_index()
print(summary.to_string())

# oracle numbers from the previous experiment, for the 3-way comparison
oracle_ms = {0.00: 0.0059, 0.05: 0.0089, 0.10: 0.0110, 0.15: 0.0192, 0.20: 0.0248,
             0.25: 0.0215, 0.30: 0.0266, 0.40: 0.0423, 0.50: 0.1999}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
ax.plot(summary["beta"], summary["mean_err_fixed"] / FS * 1000, marker="o", color="#d62728",
        label="PASS 1 only (fixed f0, no correction)")
ax.plot(summary["beta"], summary["mean_err_twopass"] / FS * 1000, marker="^", color="#2ca02c",
        label="TWO-PASS (real, deployable)")
ax.plot(list(oracle_ms.keys()), list(oracle_ms.values()), marker="s", color="#1f77b4", ls="--",
        label="oracle (knows true beta,\nidealized upper bound)")
ax.set_xlabel("beta (stretch) of the TEST spike")
ax.set_ylabel("mean timing error (ms)")
ax.set_title("Does the REAL two-pass method recover\nmost of the oracle's benefit?", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(summary["beta"], summary["mean_beta_est_pass1"], marker="o", color="#9467bd")
ax.plot(summary["beta"], summary["beta"], color="grey", ls=":", label="perfect (y=x)")
ax.set_xlabel("TRUE beta")
ax.set_ylabel("pass-1 beta ESTIMATE (biased, from fixed-f0 alignment)")
ax.set_title("How biased is the pass-1 beta estimate\nthat pass 2 relies on?", fontsize=10.5)
ax.legend(fontsize=8)

plt.suptitle("Real two-pass frequency correction vs. fixed-f0 and the oracle upper bound", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_32_two_pass_vs_oracle.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
