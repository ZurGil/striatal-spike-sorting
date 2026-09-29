"""
Control test motivated directly by the user's question: does the
tau-vs-beta correlation found on BURST spikes (r=0.77) reflect real burst
physiology (wavelet correction less complete for more-stretched, more-
decayed spikes), or is it a property of the fitting method itself that
would show up on ANY spike regardless of context?

Runs the IDENTICAL pipeline (coarse+fine alignment, then nuisance fit) on
a matched-size sample of ISOLATED unit 342 spikes -- long ISI on both
sides, i.e. NOT part of any burst by the same ISI<10ms criterion used to
find bursts.

Usage: python demo_isolated_control.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
UID = 342
ITEMSIZE = 2
F0_HZ = 739.0
ISOLATION_MS = 200.0  # both neighboring ISIs must exceed this -- "fully rested"
SEARCH_RADIUS = 25
PAD = 100

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templ_all = templates[UID]
peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
template = templ_all[:, peak_ch]

_, psi = make_morlet(F0_HZ, FS, n_cycles=3.0)
ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)

st = np.sort(spike_times[spike_clusters == UID])
isi_samples = np.diff(st)
isolation_samples = ISOLATION_MS / 1000 * FS

# a spike at index i is "isolated" if BOTH its preceding and following ISI
# exceed the threshold (skip the very first/last spike, no neighbor on one side)
isolated_mask = np.zeros(len(st), dtype=bool)
isolated_mask[1:-1] = (isi_samples[:-1] > isolation_samples) & (isi_samples[1:] > isolation_samples)
isolated_spikes = st[isolated_mask]
print(f"unit {UID}: {len(st)} total spikes, {len(isolated_spikes)} isolated "
      f"(ISI>{ISOLATION_MS}ms on both sides)")

rng = np.random.default_rng(7)
n_test = 22  # match the burst-spike sample size from the previous test
chosen = rng.choice(isolated_spikes, size=min(n_test, len(isolated_spikes)), replace=False)

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(center_sample, pad=PAD, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    filtered = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64)) * GAIN_TO_UV
    return filtered[filt_buf:-filt_buf]


def process_candidate(center_sample):
    trace = get_filtered(center_sample)
    nominal_center = PAD
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, F0_HZ, FS,
                                     reference_phase=ref_phase, search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + 61)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    local_interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int],
                             kind="cubic", bounds_error=False, fill_value=0.0)
    sample_positions = true_center - NT0MIN + np.arange(61)
    snippet = local_interp(sample_positions)
    fit = fit_nuisance(snippet, s0_basis, s0p_basis, q_basis)
    return dict(sample=int(center_sample), coarse=result["coarse_shift"], fine=result["fine_shift"],
                total_shift=result["total_shift"], coarse_score=result["coarse_score"],
                a=fit["a"], tau=fit["tau"], beta=fit["beta"], r_squared=fit["r_squared"])


isolated_results = [r for r in (process_candidate(int(s)) for s in chosen) if r is not None]
isolated_df = pd.DataFrame(isolated_results)
isolated_df.to_csv(os.path.join(OUT, "integration_isolated_control_unit342.csv"), index=False)

print(f"\nprocessed {len(isolated_df)} isolated spikes")
print(isolated_df[["a", "tau", "beta", "r_squared", "coarse_score"]].describe())
print(f"\ncorr(tau, beta) ISOLATED: {isolated_df['tau'].corr(isolated_df['beta']):.3f}")
print(f"corr(tau, a) ISOLATED: {isolated_df['tau'].corr(isolated_df['a']):.3f}")

# direct comparison against the burst-spike results from the previous test
burst_df = pd.read_csv(os.path.join(OUT, "integration_detected_unit342.csv"))
print(f"\n=== DIRECT COMPARISON ===")
print(f"{'metric':<20}{'BURST spikes':>15}{'ISOLATED spikes':>18}")
for col in ["a", "tau", "beta", "r_squared"]:
    print(f"{col:<20}{burst_df[col].mean():>15.3f}{isolated_df[col].mean():>18.3f}")
print(f"{'tau-beta corr':<20}{burst_df['tau'].corr(burst_df['beta']):>15.3f}"
      f"{isolated_df['tau'].corr(isolated_df['beta']):>18.3f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

ax = axes[0]
ax.scatter(burst_df["beta"], burst_df["tau"], color="#1f77b4", s=50, label=f"BURST (n={len(burst_df)}), r={burst_df['tau'].corr(burst_df['beta']):.2f}")
ax.scatter(isolated_df["beta"], isolated_df["tau"], color="#d62728", s=50, marker="^",
           label=f"ISOLATED (n={len(isolated_df)}), r={isolated_df['tau'].corr(isolated_df['beta']):.2f}")
ax.set_xlabel("beta (stretch)")
ax.set_ylabel("tau (residual timing, samples)")
ax.set_title("THE key test: does tau-vs-beta correlation\nshow up WITHOUT real burst decay?", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[1]
ax.hist(burst_df["a"], bins=10, alpha=0.6, color="#1f77b4", label="BURST")
ax.hist(isolated_df["a"], bins=10, alpha=0.6, color="#d62728", label="ISOLATED")
ax.set_xlabel("fitted amplitude a")
ax.set_ylabel("count")
ax.set_title("Amplitude: isolated should be\nhigher/less variable (rested state)", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[2]
ax.axis("off")
r_burst = burst_df["tau"].corr(burst_df["beta"])
r_isolated = isolated_df["tau"].corr(isolated_df["beta"])
verdict = ("BURST-SPECIFIC (real physiology)" if abs(r_isolated) < 0.3 and abs(r_burst) > 0.5
           else "METHOD ARTIFACT (not burst-specific)" if abs(r_isolated) > 0.5
           else "AMBIGUOUS -- needs more data")
ax.text(0, 1.0,
    "CONTROL TEST RESULT\n\n"
    f"BURST tau-beta correlation:    r={r_burst:.2f}\n"
    f"ISOLATED tau-beta correlation: r={r_isolated:.2f}\n\n"
    f"amplitude a: burst mean={burst_df.a.mean():.1f}, "
    f"isolated mean={isolated_df.a.mean():.1f}\n"
    f"beta: burst mean={burst_df.beta.mean():.3f}, "
    f"isolated mean={isolated_df.beta.mean():.3f}\n\n"
    f"VERDICT: {verdict}",
    fontsize=10.5, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Control test: burst vs. isolated unit-342 spikes, same pipeline", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_26_isolated_control.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
