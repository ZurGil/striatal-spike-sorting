"""
Bigger, better-selected version of the template-rebuild check, addressing
three points raised directly:

1. Unit selection: uses THIS PROJECT'S OWN "sep_vs_noise" metric (the
   "Sep. vs noise (p80)" score shown in outputs/unit_review_pipeline.html,
   from outputs/pipeline_review_data.json) -- NOT bombcell's SNR, which
   was used by mistake last time. Units chosen evenly spaced across its
   real range, not hand-picked.

2. Metric choice: alongside the existing R2 (from the amplitude+stretch
   fit, nuisance_model.fit_nuisance_prealigned -- the SAME metric used
   throughout this project since pillar 2, not something invented new for
   this check), also computes a much simpler, more standard metric: plain
   NORMALIZED SIMILARITY (cosine similarity) between each aligned spike
   snippet and the template -- a single dot product ratio, no stretch
   parameter, no regression. cos_sim = dot(x, s0) / (||x|| * ||s0||).

3. Independent cross-check: Kilosort's OWN per-spike amplitude
   (amplitudes.npy) for the exact same sampled spikes -- this is the
   score Kilosort itself already used when it decided these spikes belong
   to this unit (its own matching-pursuit result). If a unit is good,
   Kilosort's own scoring for it should already be stable/consistent --
   checked here via coefficient of variation (std/mean) of Kilosort's own
   amplitude across the sampled spikes, completely independent of
   anything built in this project.

Usage: python demo_multiunit_metric_comparison.py
"""
import os
import json
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
REVIEW_JSON = r"D:\Gil\spike_sorting_agent\outputs\pipeline_review_data.json"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
N = 61
ITEMSIZE = 2
SEARCH_RADIUS = 25
N_CYCLES = 3.0
N_TEST_SPIKES = 40
N_UNITS = 16

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
ks_amplitudes = np.load(VR + r"\amplitudes.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

with open(REVIEW_JSON) as f:
    review = json.load(f)
review_df = pd.DataFrame(review)[["unit_id", "label", "sep_vs_noise", "n_spikes"]]
review_df = review_df.dropna(subset=["sep_vs_noise"])
review_df = review_df[(review_df.n_spikes.between(2000, 90000)) & (~review_df.label.str.contains("NOISE"))]

# GUARD: pipeline_review_data.json is from an earlier curation stage than
# this session's FINAL verdict_review output -- some unit_ids it lists
# (e.g. unit 12) no longer exist at all in spike_clusters.npy (merged away
# or removed during final curation). Only keep candidates that actually
# have real spikes in the FINAL data, checked directly, not assumed.
final_unit_ids = set(np.unique(spike_clusters).tolist())
n_before = len(review_df)
review_df = review_df[review_df.unit_id.isin(final_unit_ids)]
n_dropped = n_before - len(review_df)
if n_dropped:
    print(f"dropped {n_dropped} candidate unit(s) present in the review JSON but absent from "
          f"this session's final spike_clusters.npy (stale/merged unit_ids)")

review_df = review_df.sort_values("sep_vs_noise").reset_index(drop=True)

# evenly spaced across the REAL range of sep_vs_noise, not hand-picked
pick_idx = np.linspace(0, len(review_df) - 1, N_UNITS).astype(int)
UNITS = review_df.iloc[pick_idx]["unit_id"].astype(int).tolist()
print(f"selected {len(UNITS)} units, evenly spaced across sep_vs_noise "
      f"[{review_df.sep_vs_noise.min():.2f}, {review_df.sep_vs_noise.max():.2f}]:")
print(review_df.iloc[pick_idx][["unit_id", "label", "sep_vs_noise", "n_spikes"]].to_string(index=False))


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

    unit_mask = spike_clusters == uid
    st = np.sort(spike_times[unit_mask])
    sep_vs_noise = float(review_df.loc[review_df.unit_id == uid, "sep_vs_noise"].iloc[0])

    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(st), size=min(N_TEST_SPIKES, len(st)), replace=False)
    test_spikes = st[idx]

    # Kilosort's OWN amplitude for these exact spikes -- fully independent,
    # no alignment/whitening/wavelet involved at all
    spike_time_to_amp = dict(zip(spike_times[unit_mask], ks_amplitudes[unit_mask]))
    ks_amp_sample = np.array([spike_time_to_amp[s] for s in test_spikes])
    ks_amp_cv = ks_amp_sample.std() / ks_amp_sample.mean()

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
    print(f"unit {uid}: sep_vs_noise={sep_vs_noise:.2f}, n_tested={len(r2s)}, "
          f"mean R2={r2s.mean():.3f}, mean cos_sim={cos_sims.mean():.3f}, KS amp CV={ks_amp_cv:.3f}")

    results.append(dict(uid=uid, sep_vs_noise=sep_vs_noise, n_spikes=len(st),
                         mean_r2=r2s.mean(), mean_cos_sim=cos_sims.mean(),
                         ks_amp_cv=ks_amp_cv))

df = pd.DataFrame(results).sort_values("sep_vs_noise")
print("\n=== SUMMARY, sorted by sep_vs_noise ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "multiunit_metric_comparison.csv"), index=False)

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
ax = axes[0]
ax.scatter(df["sep_vs_noise"], df["mean_r2"], s=80, color="#1f77b4")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.sep_vs_noise, r.mean_r2), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax.set_xlabel("sep_vs_noise (this project's own metric)")
ax.set_ylabel("mean R2 (amplitude+stretch fit)")
ax.set_title("R2 vs sep_vs_noise", fontsize=10.5)

ax = axes[1]
ax.scatter(df["sep_vs_noise"], df["mean_cos_sim"], s=80, color="#2ca02c")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.sep_vs_noise, r.mean_cos_sim), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax.set_xlabel("sep_vs_noise")
ax.set_ylabel("mean cosine similarity (plain, no stretch)")
ax.set_title("Simple similarity vs sep_vs_noise", fontsize=10.5)

ax = axes[2]
ax.scatter(df["sep_vs_noise"], df["ks_amp_cv"], s=80, color="#d62728")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.sep_vs_noise, r.ks_amp_cv), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax.set_xlabel("sep_vs_noise")
ax.set_ylabel("Kilosort's OWN amplitude, coeff. of variation")
ax.set_title("Kilosort's own scoring vs sep_vs_noise", fontsize=10.5)

ax = axes[3]
ax.scatter(df["ks_amp_cv"], df["mean_r2"], s=80, color="#9467bd")
for _, r in df.iterrows():
    ax.annotate(f"{int(r.uid)}", (r.ks_amp_cv, r.mean_r2), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax.set_xlabel("Kilosort's own amplitude CV")
ax.set_ylabel("mean R2 (this project's fit)")
ax.set_title("Does OUR metric agree with\nKILOSORT'S own scoring?", fontsize=10.5)

plt.suptitle(f"{len(UNITS)} units spanning sep_vs_noise: three independent metrics compared", fontsize=13.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "multiunit_04_metric_comparison.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
