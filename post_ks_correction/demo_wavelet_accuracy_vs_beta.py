"""
THE key unresolved question from this project's conversation history:
does pillar 1b's timing correction stay accurate for heavily-stretched
(high-beta) spikes, given it uses ONE FIXED, rest-state frequency per
unit (never re-tuned per spike)?

Uses SYNTHETIC ground truth (known true tau, known true beta) since no
ground truth exists for real spike timing -- built on unit 342's own real
template and its own real best-fit f0=739Hz, exactly as actually used in
practice: reference_phase calibrated from the RESTED (beta=0) template,
same fixed f0 used regardless of how stretched the test spike is.

Also computes a best-case comparison: what if f0 WERE re-tuned per beta
level (an idealized upper bound), to quantify how much of any accuracy
loss is specifically attributable to using one fixed frequency.

Usage: python demo_wavelet_accuracy_vs_beta.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import synthesize_deformed_spike

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
NT0MIN = 20
UID = 342
F0_REST = 739.0  # this unit's own real best-fit frequency, established via the scan
SEARCH_RADIUS = 25
N_CYCLES = 3.0

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]

_, psi_rest = make_morlet(F0_REST, FS, n_cycles=N_CYCLES)
ref_phase_rest = calibrate_reference_phase(template, psi_rest, FS, align_index=NT0MIN)

rng = np.random.default_rng(11)
noise_sigma = 0.05 * np.abs(template).max()  # same realistic 5% noise level used throughout
pad = 200  # generous, covers the widest probe tested

beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
true_taus = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]  # spans well within the 739Hz safe zone (+/-20.3 samples)
N_TRIALS = 6  # noise realizations per (beta, tau) combo

# for the best-case comparison: what f0 would this template's own scan find
# AT each beta level (i.e. if we re-tuned per spike based on its own
# instantaneous stretch)? Re-run the same frequency scan used throughout
# this project, but on the STRETCHED template instead of the rested one.
def best_f0_for_beta(beta):
    stretched = synthesize_deformed_spike(template, a=1.0, tau=0.0, beta=beta, noise_sigma=0.0, rng=rng)
    f0_scan = np.linspace(300, 4000, 40)
    pad_scan = int(np.ceil(3.0 * FS / (2 * f0_scan.min()))) + 20
    trace = np.zeros(61 + 2 * pad_scan)
    center = pad_scan + NT0MIN
    trace[center - NT0MIN: center - NT0MIN + 61] = stretched
    mags = [abs(wavelet_transform_at(trace, make_morlet(f0, FS, N_CYCLES)[1], center)) for f0 in f0_scan]
    return f0_scan[int(np.argmax(mags))]


results = []
for beta in beta_values:
    best_f0 = best_f0_for_beta(beta)
    _, psi_best = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
    ref_phase_best = calibrate_reference_phase(template, psi_best, FS, align_index=NT0MIN)

    for true_tau in true_taus:
        errors_fixed, errors_bestf0 = [], []
        for _ in range(N_TRIALS):
            clean = synthesize_deformed_spike(template, a=1.0, tau=true_tau, beta=beta, noise_sigma=0.0)
            noisy = clean + rng.normal(0, noise_sigma, size=len(clean))
            trace = np.zeros(len(noisy) + 2 * pad)
            true_center = pad + NT0MIN
            trace[true_center - NT0MIN: true_center - NT0MIN + len(noisy)] = noisy

            r_fixed = coarse_then_fine_shift(trace, true_center, template, psi_rest, F0_REST, FS,
                                              reference_phase=ref_phase_rest, search_radius=SEARCH_RADIUS,
                                              nt0min=NT0MIN)
            errors_fixed.append(abs(r_fixed["total_shift"] - true_tau))

            r_best = coarse_then_fine_shift(trace, true_center, template, psi_best, best_f0, FS,
                                             reference_phase=ref_phase_best, search_radius=SEARCH_RADIUS,
                                             nt0min=NT0MIN)
            errors_bestf0.append(abs(r_best["total_shift"] - true_tau))

        results.append(dict(beta=beta, true_tau=true_tau, best_f0=best_f0,
                             mean_err_fixed=np.mean(errors_fixed), mean_err_bestf0=np.mean(errors_bestf0)))

df = pd.DataFrame(results)
summary = df.groupby("beta")[["mean_err_fixed", "mean_err_bestf0"]].mean().reset_index()
print(summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.plot(summary["beta"], summary["mean_err_fixed"] / FS * 1000, marker="o", color="#d62728",
        label="FIXED f0=739Hz (rest-state,\nas actually used in practice)")
ax.plot(summary["beta"], summary["mean_err_bestf0"] / FS * 1000, marker="s", color="#1f77b4",
        label="RE-TUNED f0 per beta level\n(idealized best case)")
ax.set_xlabel("beta (stretch) of the TEST spike")
ax.set_ylabel("mean timing error (ms), averaged\nover true_tau in [-10,15] samples")
ax.set_title("THE key test: does fixed-f0 correction accuracy\ndegrade as spikes stretch further from rest?", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[1]
ax.axis("off")
err0 = summary[summary.beta == 0.0]["mean_err_fixed"].values[0]
err_max = summary[summary.beta == summary.beta.max()]["mean_err_fixed"].values[0]
degradation = err_max / err0 if err0 > 0 else np.nan
real_beta_range_note = "real burst spikes on unit 342 measured beta up to ~0.5-0.7"
ax.text(0, 1.0,
    "RESULT\n\n"
    f"error at beta=0 (rested): {err0/FS*1000:.4f} ms\n"
    f"error at beta={summary.beta.max():.1f} (heavily stretched): {err_max/FS*1000:.4f} ms\n"
    f"degradation factor: {degradation:.1f}x\n\n"
    f"({real_beta_range_note})\n\n"
    "best_f0 found per beta level:\n" +
    "\n".join(f"  beta={row.beta:.2f}: best_f0={row.best_f0:.0f}Hz"
              for row in df.drop_duplicates('beta').itertuples()),
    fontsize=9.7, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Does pillar 1b's fixed per-unit frequency stay accurate as spikes stretch? (synthetic ground truth)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_31_accuracy_vs_beta.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
