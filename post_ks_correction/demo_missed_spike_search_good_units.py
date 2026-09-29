"""
Stage: search for spikes Kilosort MISSED, restricted to units already
validated as trustworthy (high sep_vs_noise, high R2/similarity after
wavelet alignment + whitening -- section 5g). Unlike the earlier unit-307
attempt, the reference "what does a real spike score" distribution here is
itself trustworthy, since these units already passed the quality check --
so comparing a candidate against THIS unit's own real-spike score
distribution is a fair bar, not the fragile relative-percentile-of-a-bad-
distribution problem found on unit 307.

Hypothesis being tested (user's, matches the design doc): Kilosort's
matching-pursuit detection is a RIGID template match with no phase/
sub-sample tolerance, so a real spike landing at an awkward sub-sample
timing can score too low and get missed even for a clean, well-isolated
neuron. If true, some genuinely real spikes should be sitting near this
unit's own detected spikes, un-detected only because of jitter -- and our
wavelet-aligned, whitened scoring (which explicitly finds the best
sub-sample alignment) should be able to find and score them as well as
real spikes.

Method: for each unit, scan a window around (before AND after, not just
after) each of many real detected spikes for local-maximum candidate
events not already detected, using a cheap coarse matched-filter score
first, then run only the most promising candidates through the full
wavelet align + whitened fit -- exactly the coarse-then-expensive pattern
used everywhere else in this project.

Nothing here writes to any Kilosort/Phy file -- diagnostic only.

Usage: python demo_missed_spike_search_good_units.py
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
N_REF_SPIKES = 60       # real spikes used to build the trustworthy reference distribution
N_SCAN_SPIKES = 60      # real spikes whose neighborhood gets scanned for candidates
SCAN_HALF_WINDOW = 150  # samples scanned BEFORE and AFTER each scanned spike
MARGIN = 20             # exclusion zone around any of this unit's own real spikes
GAP_SCAN_STEP = 2
TOP_N_CANDIDATES = 25   # per unit, refine only the top-scoring coarse candidates

GOOD_UNITS = [303, 313, 245, 332, 306, 342]  # validated in section 5g: high sep_vs_noise, high R2/cos_sim

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def get_filtered(peak_ch, center_sample, pad, filt_buf=60):
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


def align_and_fit(peak_ch, center_sample, template, psi, f0, ref_phase, s0_basis, q_basis, W, pad=200):
    trace = get_filtered(peak_ch, center_sample, pad=pad)
    nominal_center = pad
    result = coarse_then_fine_shift(trace, nominal_center, template, psi, f0, FS,
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
    return dict(snippet=snippet, r_squared=fit["r_squared"], a=fit["a"], total_shift=result["total_shift"])


all_results = []
example_plots = []

for uid in GOOD_UNITS:
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    s0_basis, _, q_basis = build_basis(template, dt=1.0)

    f0 = best_f0_for(template)
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)

    st = np.sort(spike_times[spike_clusters == uid])
    detected_set = set(st.tolist())

    noise_start = int(st[0]) + 500000 if st[0] + 500000 < st[-1] else int(st[0]) - 500000
    noise_start = max(noise_start, 10000)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
    noise_trace = filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[60:-60] * GAIN_TO_UV
    W, _, _ = build_whitening_from_noise(noise_trace, N, order=4)

    # ---- reference distribution: real detected spikes, same pipeline ----
    rng = np.random.default_rng(7)
    ref_idx = rng.choice(len(st), size=min(N_REF_SPIKES, len(st)), replace=False)
    ref_results = []
    for s in st[ref_idx]:
        r = align_and_fit(peak_ch, int(s), template, psi, f0, ref_phase, s0_basis, q_basis, W)
        if r is not None:
            ref_results.append(r["r_squared"])
    ref_r2 = np.array(ref_results)
    bar_p25 = np.percentile(ref_r2, 25)
    bar_p10 = np.percentile(ref_r2, 10)

    # ---- scan around (before AND after) many real spikes for local maxima ----
    rng2 = np.random.default_rng(11)
    scan_idx = rng2.choice(len(st), size=min(N_SCAN_SPIKES, len(st)), replace=False)
    scan_spikes = st[scan_idx]

    coarse_candidates = []
    for s in scan_spikes:
        lo, hi = int(s) - SCAN_HALF_WINDOW, int(s) + SCAN_HALF_WINDOW
        trace = get_filtered(peak_ch, int(s), pad=SCAN_HALF_WINDOW + 60)
        center_in_trace = SCAN_HALF_WINDOW + 60
        local = []
        for c in range(lo, hi, GAP_SCAN_STEP):
            offset = c - int(s)
            lo_w = center_in_trace + offset - NT0MIN
            window = trace[lo_w: lo_w + N]
            if len(window) != N:
                continue
            local.append((c, np.dot(window, template_norm)))
        if len(local) < 3:
            continue
        samples_arr = np.array([c for c, _ in local])
        scores_arr = np.array([sc for _, sc in local])
        for i in range(1, len(scores_arr) - 1):
            if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
                cand_sample = int(samples_arr[i])
                if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                    coarse_candidates.append((cand_sample, scores_arr[i]))

    coarse_candidates = sorted(set(coarse_candidates), key=lambda x: -x[1])[:TOP_N_CANDIDATES]

    cand_rows = []
    for cand_sample, coarse_score in coarse_candidates:
        r = align_and_fit(peak_ch, cand_sample, template, psi, f0, ref_phase, s0_basis, q_basis, W)
        if r is not None:
            cand_rows.append(dict(sample=cand_sample, coarse_score=coarse_score, r_squared=r["r_squared"],
                                   a=r["a"], snippet=r["snippet"]))
    cand_df = pd.DataFrame(cand_rows)
    n_above_p25 = int((cand_df["r_squared"] >= bar_p25).sum()) if len(cand_df) else 0
    n_above_p10 = int((cand_df["r_squared"] >= bar_p10).sum()) if len(cand_df) else 0

    print(f"unit {uid} (ch{peak_ch}, {len(st)} spikes, f0={f0:.0f}Hz): "
          f"ref R2 median={np.median(ref_r2):.3f}, p25={bar_p25:.3f}, p10={bar_p10:.3f} | "
          f"{len(coarse_candidates)} candidates refined, "
          f"{n_above_p25}/{len(cand_df)} above ref p25, {n_above_p10}/{len(cand_df)} above ref p10")

    all_results.append(dict(uid=uid, peak_ch=peak_ch, n_spikes=len(st), f0=f0,
                             ref_r2_median=np.median(ref_r2), ref_r2_p25=bar_p25, ref_r2_p10=bar_p10,
                             n_candidates_refined=len(cand_df), n_above_p25=n_above_p25, n_above_p10=n_above_p10))

    if len(cand_df) > 0:
        top_cands = cand_df.sort_values("r_squared", ascending=False).head(3)
        for _, row in top_cands.iterrows():
            example_plots.append(dict(uid=uid, sample=int(row["sample"]), r_squared=row["r_squared"],
                                       a=row["a"], snippet=row["snippet"], template=template,
                                       ref_r2_p25=bar_p25))

summary_df = pd.DataFrame(all_results)
print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT, "missed_spike_search_good_units_summary.csv"), index=False)

# ---- figure: best candidate per unit vs. a real spike, overlaid ----
n_show = min(6, len(example_plots))
if n_show > 0:
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4.5), squeeze=False)
    t_ms = (np.arange(N) - NT0MIN) / FS * 1000
    shown_by_unit = set()
    col = 0
    for ex in sorted(example_plots, key=lambda e: -e["r_squared"]):
        if ex["uid"] in shown_by_unit or col >= n_show:
            continue
        shown_by_unit.add(ex["uid"])
        ax = axes[0, col]
        ax.plot(t_ms, ex["snippet"], color="#2ca02c", lw=1.6, label="candidate raw snippet")
        ax.plot(t_ms, ex["template"] * ex["a"], color="#d62728", lw=1.3, ls="--", label=f"template*a={ex['a']:.1f}")
        ax.set_title(f"unit {ex['uid']}: best candidate\nsample {ex['sample']}\n"
                     f"R2={ex['r_squared']:.2f} (unit's own ref p25={ex['ref_r2_p25']:.2f})", fontsize=9.5)
        ax.axhline(0, color="grey", lw=0.4)
        ax.set_xlabel("time (ms)")
        if col == 0:
            ax.set_ylabel("uV")
            ax.legend(fontsize=7)
        col += 1
    plt.suptitle("Best missed-spike candidate per unit (validated 'good' units only)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT, "missed_spike_search_01_good_units.png")
    plt.savefig(out_path, dpi=115, bbox_inches="tight")
    print("\nsaved", out_path)
