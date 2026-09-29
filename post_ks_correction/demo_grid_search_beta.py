"""
Implements the user's proposal: instead of bootstrapping (align with ONE
fixed frequency, estimate beta from that possibly-biased alignment, then
patch the frequency), build a GRID of candidate beta values, each with
its own correctly-matched frequency (f0_rest/(1+beta), the exact relation
confirmed previously), align at EACH candidate's own frequency, then
"de-stretch" (resample the aligned snippet's time axis by 1/(1+beta)) and
check how well it matches the RESTED template directly. Whichever
candidate is most SELF-CONSISTENT (best de-stretched match) wins -- no
single biased estimate feeding a single correction; many parallel
hypotheses, each tested on its own terms.

Usage: python demo_grid_search_beta.py
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import synthesize_deformed_spike

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
NT0MIN = 20
UID = 342
F0_REST = 739.0
SEARCH_RADIUS = 25
N_CYCLES = 3.0
BETA_GRID = np.arange(-0.10, 0.65, 0.05)
F0_CLIP = (200.0, 4000.0)

templates = np.load(VR + r"\templates.npy")
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]

_, psi_rest = make_morlet(F0_REST, FS, n_cycles=N_CYCLES)
ref_phase_rest = calibrate_reference_phase(template, psi_rest, FS, align_index=NT0MIN)

# precompute the whole family ONCE (per unit, not per spike -- exactly
# the "sub-unit template library" the user described)
_family = {}
for beta_cand in BETA_GRID:
    f0_cand = float(np.clip(F0_REST / (1 + beta_cand), *F0_CLIP))
    _, psi_cand = make_morlet(f0_cand, FS, n_cycles=N_CYCLES)
    ref_cand = calibrate_reference_phase(template, psi_cand, FS, align_index=NT0MIN)
    _family[round(beta_cand, 2)] = dict(f0=f0_cand, psi=psi_cand, ref=ref_cand)


def grid_search_shift_and_beta(trace, center_index):
    best_r2, best_beta, best_tau = -np.inf, 0.0, 0.0
    for beta_cand, entry in _family.items():
        r = coarse_then_fine_shift(trace, center_index, template, entry["psi"], entry["f0"], FS,
                                    reference_phase=entry["ref"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
        true_center = center_index + r["total_shift"]
        # de-stretch: sample the ALIGNED snippet at t*(1+beta_cand) to undo
        # the hypothesized stretch, then compare directly to the rested template
        lo_int = int(np.floor(true_center - NT0MIN * (1 + abs(beta_cand)))) - 3
        hi_int = int(np.ceil(true_center - NT0MIN + 61 * (1 + abs(beta_cand)))) + 3
        if lo_int < 0 or hi_int > len(trace):
            continue
        interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                           bounds_error=False, fill_value=0.0)
        t_local = np.arange(61) - NT0MIN
        sample_positions = true_center + t_local * (1 + beta_cand)
        destretched = interp(sample_positions)

        s0_norm = template / (np.linalg.norm(template) + 1e-12)
        a_fit = np.dot(destretched, s0_norm)
        fit = a_fit * s0_norm
        resid = destretched - fit
        ss_res = np.dot(resid, resid)
        ss_tot = np.dot(destretched - destretched.mean(), destretched - destretched.mean())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

        if r2 > best_r2:
            best_r2, best_beta, best_tau = r2, beta_cand, r["total_shift"]
    return best_tau, best_beta, best_r2


rng = np.random.default_rng(11)
noise_sigma = 0.05 * np.abs(template).max()
pad = 200
beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
true_taus = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
N_TRIALS = 6

results = []
for beta in beta_values:
    for true_tau in true_taus:
        errs_tau, errs_beta = [], []
        for _ in range(N_TRIALS):
            clean = synthesize_deformed_spike(template, a=1.0, tau=true_tau, beta=beta, noise_sigma=0.0)
            noisy = clean + rng.normal(0, noise_sigma, size=len(clean))
            trace = np.zeros(len(noisy) + 2 * pad)
            true_center = pad + NT0MIN
            trace[true_center - NT0MIN: true_center - NT0MIN + len(noisy)] = noisy

            tau_est, beta_est, r2 = grid_search_shift_and_beta(trace, true_center)
            errs_tau.append(abs(tau_est - true_tau))
            errs_beta.append(abs(beta_est - beta))
        results.append(dict(beta=beta, true_tau=true_tau,
                             mean_err_tau=np.mean(errs_tau), mean_err_beta=np.mean(errs_beta)))

df = pd.DataFrame(results)
summary = df.groupby("beta")[["mean_err_tau", "mean_err_beta"]].mean().reset_index()
print(summary.to_string())

# numbers from the earlier experiments, for direct comparison
fixed_ms = {0.0: 0.0059, 0.1: 0.0136, 0.2: 0.0304, 0.3: 0.0496, 0.4: 0.0699, 0.5: 0.0915}
twopass_ms = {0.0: 0.0061, 0.1: 0.0111, 0.2: 0.0237, 0.3: 0.0380, 0.4: 0.0540, 0.5: 0.0727}
oracle_ms = {0.0: 0.0059, 0.1: 0.0110, 0.2: 0.0248, 0.3: 0.0266, 0.4: 0.0423, 0.5: 0.1999}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
ax.plot(list(fixed_ms.keys()), list(fixed_ms.values()), marker="o", color="#d62728", label="fixed f0 only")
ax.plot(list(twopass_ms.keys()), list(twopass_ms.values()), marker="^", color="#2ca02c", label="two-pass (bootstrapped)")
ax.plot(list(oracle_ms.keys()), list(oracle_ms.values()), marker="s", color="#1f77b4", ls="--", label="oracle (cheats: knows true beta)")
ax.plot(summary["beta"], summary["mean_err_tau"] / FS * 1000, marker="D", color="#9467bd", lw=2.2,
        label="GRID SEARCH (this idea, no cheating)")
ax.set_xlabel("beta (stretch) of the TEST spike")
ax.set_ylabel("mean TIMING error (ms)")
ax.set_title("Does grid-search beat bootstrapping,\nwithout needing the oracle's cheat?", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(summary["beta"], summary["mean_err_beta"], marker="D", color="#9467bd")
ax.set_xlabel("true beta")
ax.set_ylabel("mean BETA estimation error")
ax.set_title("How accurate is beta itself\nwith grid search?", fontsize=10.5)

plt.suptitle("Grid-search (parallel hypotheses) vs. bootstrapped two-pass vs. oracle", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_33_grid_search_beta.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
