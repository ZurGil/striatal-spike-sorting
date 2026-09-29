"""
Generalization sanity check, per the user's explicit request: does the
whole pillar-1a+1b(+whitening) pipeline -- built and iterated on unit 342
specifically -- actually work on a few ORDINARY units, not just the one
unit it was tuned against? And does the same raw-data scan approach used
to look for unit 342's undetected burst-tail candidates also surface (a)
candidate MISSED spikes and (b) currently-DETECTED spikes that fit this
unit's own template poorly enough to flag as possibly not belonging, on
units in general -- not just the one hand-picked bursty example?

This is explicitly a SANITY CHECK, not a validated production pipeline:
small spike samples per unit, no ground truth, results are meant to give a
first real-data sense of whether this generalizes, not a final answer.
Nothing here writes to any Kilosort/Phy file -- diagnostic only, same as
everything else in this module so far.

Units tested: 342 (the bursty unit everything else was built against, kept
as a reference point) plus three "ordinary" good units picked by spike
count alone (240 - sparse, 307 - dense, 325 - medium), not cherry-picked
for any known property.

Usage: python demo_multiunit_sanity_check.py
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
AR_ORDER = 4
N_SPIKES_SAMPLE = 40   # per unit -- a sanity-check sample, not exhaustive
MARGIN = 20
GAP_SCAN_STEP = 2
MISS_SCAN_AFTER = 150  # samples after each sampled spike to scan for un-detected candidates

UNITS = [342, 240, 307, 325]

templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
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


def best_f0_for_template(template):
    f0_scan = np.linspace(300, 4000, 50)
    pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
    trace = np.zeros(N + 2 * pad_scan)
    center = pad_scan + NT0MIN
    trace[center - NT0MIN: center - NT0MIN + N] = template
    mags = [abs(wavelet_transform_at(trace, make_morlet(f0, FS, N_CYCLES)[1], center)) for f0 in f0_scan]
    return float(f0_scan[int(np.nanargmax(mags))])


results_per_unit = []
per_unit_detected = {}
per_unit_missed = {}

for uid in UNITS:
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    n_spikes_total = int(info.loc[uid, "n_spikes"])

    best_f0 = best_f0_for_template(template)
    _, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
    s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)

    st = np.sort(spike_times[spike_clusters == uid])

    # real, per-unit noise segment -- far from this unit's own spikes,
    # used ONLY to fit the whitening model, never touched again below
    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] - 100000 else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, phi, sigma2 = build_whitening_from_noise(noise_trace, N, order=AR_ORDER)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(st), size=min(N_SPIKES_SAMPLE, len(st)), replace=False)
    sample_idx.sort()
    sampled_spikes = st[sample_idx]

    def process(center_sample):
        trace = get_filtered(peak_ch, int(center_sample))
        nominal_center = 100
        result = coarse_then_fine_shift(trace, nominal_center, template, psi, best_f0, FS,
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
        return dict(a=fit["a"], beta=fit["beta"], r_squared=fit["r_squared"], total_shift=result["total_shift"])

    detected_rows = []
    for s in sampled_spikes:
        r = process(int(s))
        if r is not None:
            r["sample"] = int(s)
            detected_rows.append(r)
    detected_df = pd.DataFrame(detected_rows)
    per_unit_detected[uid] = detected_df

    # -------- flag currently-detected spikes that fit this unit's own
    # template poorly (possibly-doesn't-belong candidates), using this
    # unit's OWN r_squared distribution as the bar, not an absolute number
    r2_low_bar = max(0.3, detected_df["r_squared"].quantile(0.10))
    flagged = detected_df[detected_df["r_squared"] < r2_low_bar]

    # -------- scan shortly after each sampled DETECTED spike for
    # candidate events this unit's own cluster did NOT report
    detected_set = set(st.tolist())
    candidates = []
    for s in sampled_spikes:
        lo, hi = int(s) + MARGIN, int(s) + MISS_SCAN_AFTER
        trace = get_filtered(peak_ch, int(s), pad=max(200, hi - int(s) + 50))
        center_in_trace = max(200, hi - int(s) + 50)
        local_scores = []
        for c in range(lo, hi, GAP_SCAN_STEP):
            offset = c - int(s)
            lo_w = center_in_trace + offset - NT0MIN
            window = trace[lo_w: lo_w + N]
            if len(window) != N:
                continue
            score = np.dot(window, template_norm)
            local_scores.append((c, score))
        if len(local_scores) < 3:
            continue
        samples_arr = np.array([c for c, _ in local_scores])
        scores_arr = np.array([sc for _, sc in local_scores])
        for i in range(1, len(scores_arr) - 1):
            if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
                cand_sample = int(samples_arr[i])
                if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                    candidates.append(dict(sample=cand_sample, coarse_score=scores_arr[i], after_spike=int(s)))

    cand_df = pd.DataFrame(candidates)
    n_promising_missed = 0
    if len(cand_df) > 0:
        cand_df = cand_df.sort_values("coarse_score", ascending=False).head(15)
        promising_rows = []
        for _, row in cand_df.iterrows():
            r = process(int(row["sample"]))
            if r is not None:
                r["sample"] = int(row["sample"])
                promising_rows.append(r)
        cand_processed = pd.DataFrame(promising_rows)
        if len(cand_processed) > 0:
            n_promising_missed = int((cand_processed["r_squared"] >= detected_df["r_squared"].quantile(0.10)).sum())
        per_unit_missed[uid] = cand_processed
    else:
        per_unit_missed[uid] = pd.DataFrame()

    results_per_unit.append(dict(
        uid=uid, peak_ch=peak_ch, n_spikes_total=n_spikes_total, best_f0=best_f0,
        n_sampled=len(detected_df),
        mean_r2=detected_df["r_squared"].mean(), median_r2=detected_df["r_squared"].median(),
        mean_a=detected_df["a"].mean(), mean_beta=detected_df["beta"].mean(),
        r2_low_bar=r2_low_bar, n_flagged_low_quality=len(flagged),
        n_gap_candidates_scanned=len(cand_df) if len(cand_df) else 0,
        n_promising_missed=n_promising_missed,
    ))
    print(f"unit {uid} (ch{peak_ch}, {n_spikes_total} total spikes, best_f0={best_f0:.0f}Hz): "
          f"sampled {len(detected_df)}, mean R2={detected_df['r_squared'].mean():.3f}, "
          f"flagged-low-quality={len(flagged)}/{len(detected_df)}, "
          f"promising-missed-candidates={n_promising_missed}")

summary_df = pd.DataFrame(results_per_unit)
print("\n=== SUMMARY, all units ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT, "multiunit_sanity_summary.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for uid in UNITS:
    df = per_unit_detected[uid]
    ax.hist(df["r_squared"], bins=12, alpha=0.5, label=f"unit {uid} (n={len(df)})")
ax.set_xlabel("fit R-squared (whitened, prealigned fit)")
ax.set_ylabel("count")
ax.set_title("Fit quality distribution, sampled detected\nspikes, across 4 units (1 bursty + 3 ordinary)", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
ax.axis("off")
rows = "\n".join(
    f"  unit {r.uid:4d} (ch{r.peak_ch:3d}, {r.n_spikes_total:6d} spikes): "
    f"mean R2={r.mean_r2:.2f}, flagged-low-quality={r.n_flagged_low_quality}/{r.n_sampled}, "
    f"promising-missed={r.n_promising_missed}"
    for r in summary_df.itertuples())
ax.text(0, 1.0,
    "SANITY CHECK RESULT (small samples, not a validated\npipeline -- see RESEARCH_LOG.md)\n\n" + rows,
    fontsize=9.5, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle("Generalization sanity check: pipeline run on 4 units (not just 342), whitened fit", fontsize=12.5, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "multiunit_01_sanity_check.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
