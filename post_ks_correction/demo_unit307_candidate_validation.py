"""
Follow-up on the unit-307 'missed spike' candidates -- three concrete
questions the user raised, none of which the earlier sanity check answered:

  1. Is a score of 0.3-0.7 R2 even ABOVE what pure noise achieves by chance
     on this weak unit's own channel, run through the identical pipeline?
     (a real null distribution, not an assumed cutoff)

  2. Does any candidate actually already belong to a DIFFERENT unit
     Kilosort already detected nearby (same or adjacent channel) -- i.e.
     is this "Kilosort's threshold caught it, just jittered / assigned
     elsewhere", not a genuinely missed event? And separately: does a
     spatially neighboring unit's own template explain the candidate
     BETTER than unit 307's template does?

  3. Is WHITENING what's making these candidates pass -- would the plain
     (unwhitened) fit have flagged the same ones, or fewer?

Nothing here writes to any Kilosort/Phy file -- diagnostic only.

Usage: python demo_unit307_candidate_validation.py
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
UID = 307
NEARBY_UIDS = [302, 311, 299]  # 302,311 share UID 307's exact peak channel; 299 is 25.6um away, huge MUA cluster
MARGIN = 20
GAP_SCAN_STEP = 2
MISS_SCAN_AFTER = 150
JITTER_MATCH_WINDOW = 15  # samples -- "already detected elsewhere, just jittered" search radius

templates = np.load(VR + r"\templates.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def unit_pipeline(uid):
    templ_all = templates[uid]
    peak_ch = int(np.argmax(templ_all.max(axis=0) - templ_all.min(axis=0)))
    template = templ_all[:, peak_ch]
    f0_scan = np.linspace(300, 4000, 50)
    pad_scan = int(np.ceil(N_CYCLES * FS / (2 * f0_scan.min()))) + 20
    trace = np.zeros(N + 2 * pad_scan)
    center = pad_scan + NT0MIN
    trace[center - NT0MIN: center - NT0MIN + N] = template
    mags = [abs(wavelet_transform_at(trace, make_morlet(f0, FS, N_CYCLES)[1], center)) for f0 in f0_scan]
    best_f0 = float(f0_scan[int(np.nanargmax(mags))])
    _, psi = make_morlet(best_f0, FS, n_cycles=N_CYCLES)
    ref_phase = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
    s0_basis, s0p_basis, q_basis = build_basis(template, dt=1.0)
    template_norm = template / (np.linalg.norm(template) + 1e-12)
    return dict(peak_ch=peak_ch, template=template, template_norm=template_norm,
                best_f0=best_f0, psi=psi, ref_phase=ref_phase, s0=s0_basis, q=q_basis)


main = unit_pipeline(UID)
neighbors = {u: unit_pipeline(u) for u in NEARBY_UIDS}

st = np.sort(spike_times[spike_clusters == UID])
noise_start = int(st[0]) + 500000
with open(DAT_PATH, "rb") as f:
    f.seek(int(noise_start - 60) * N_CHAN_BIN * ITEMSIZE)
    raw = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
block = np.frombuffer(raw, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
noise_trace_fit = filtfilt(b_hp, a_hp, block[:, main["peak_ch"]].astype(np.float64))[60:-60] * GAIN_TO_UV
W, phi, sigma2 = build_whitening_from_noise(noise_trace_fit, N, order=4)


def get_filtered(peak_ch, center_sample, pad=200, filt_buf=60):
    s0 = center_sample - pad - filt_buf
    s1 = center_sample + pad + filt_buf
    n_read = s1 - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(n_read * N_CHAN_BIN * ITEMSIZE)
    block = np.frombuffer(raw, dtype=np.int16).reshape(n_read, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, block[:, peak_ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


def align_and_fit(center_sample, unit, whitening=None, pad=200):
    trace = get_filtered(unit["peak_ch"], center_sample, pad=pad)
    nominal_center = pad
    result = coarse_then_fine_shift(trace, nominal_center, unit["template"], unit["psi"], unit["best_f0"], FS,
                                     reference_phase=unit["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(trace):
        return None
    interp = interp1d(np.arange(lo_int, hi_int), trace[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, unit["s0"], unit["q"], whitening_matrix=whitening)
    return dict(snippet=snippet, r_squared=fit["r_squared"], a=fit["a"], total_shift=result["total_shift"])


# ============================================================
# regenerate the same candidate list as before (identical logic/seed)
# ============================================================
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(st), size=40, replace=False)
sample_idx.sort()
sampled_spikes = st[sample_idx]

detected_r2 = []
for s in sampled_spikes:
    r = align_and_fit(int(s), main, whitening=W)
    if r is not None:
        detected_r2.append(r["r_squared"])
detected_r2 = np.array(detected_r2)
r2_low_bar = max(0.3, np.quantile(detected_r2, 0.10))

detected_set = set(st.tolist())
candidates = []
for s in sampled_spikes:
    lo, hi = int(s) + MARGIN, int(s) + MISS_SCAN_AFTER
    trace = get_filtered(main["peak_ch"], int(s), pad=max(200, hi - int(s) + 50))
    center_in_trace = max(200, hi - int(s) + 50)
    local_scores = []
    for c in range(lo, hi, GAP_SCAN_STEP):
        offset = c - int(s)
        lo_w = center_in_trace + offset - NT0MIN
        window = trace[lo_w: lo_w + N]
        if len(window) != N:
            continue
        local_scores.append((c, np.dot(window, main["template_norm"])))
    if len(local_scores) < 3:
        continue
    samples_arr = np.array([c for c, _ in local_scores])
    scores_arr = np.array([sc for _, sc in local_scores])
    for i in range(1, len(scores_arr) - 1):
        if scores_arr[i] >= scores_arr[i - 1] and scores_arr[i] >= scores_arr[i + 1]:
            cand_sample = int(samples_arr[i])
            if min(abs(cand_sample - ss) for ss in detected_set) > MARGIN:
                candidates.append(cand_sample)
candidates = sorted(set(candidates))
print(f"unit {UID}: r2_low_bar={r2_low_bar:.3f}, {len(candidates)} raw candidate locations")

# ============================================================
# Q1: NOISE-ONLY null distribution, identical pipeline, unit 307's channel
# ============================================================
noise_start2 = int(st[0]) + 2000000
with open(DAT_PATH, "rb") as f:
    f.seek(int(noise_start2 - 60) * N_CHAN_BIN * ITEMSIZE)
    raw2 = f.read((60000 + 120) * N_CHAN_BIN * ITEMSIZE)
block2 = np.frombuffer(raw2, dtype=np.int16).reshape(60000 + 120, N_CHAN_BIN)
held_out_noise = filtfilt(b_hp, a_hp, block2[:, main["peak_ch"]].astype(np.float64))[60:-60] * GAIN_TO_UV

rng2 = np.random.default_rng(99)
n_null = 150
null_centers = rng2.integers(400, len(held_out_noise) - 400, size=n_null)
null_r2 = []
for c in null_centers:
    # treat this held-out noise buffer itself as the "trace" -- reuse align_and_fit's
    # inner logic directly on a slice, since it expects to read from disk normally;
    # simplest here: run coarse_then_fine + fit directly on the in-memory noise buffer
    nominal_center = int(c)
    result = coarse_then_fine_shift(held_out_noise, nominal_center, main["template"], main["psi"], main["best_f0"], FS,
                                     reference_phase=main["ref_phase"], search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal_center + result["total_shift"]
    lo_int = int(np.floor(true_center - NT0MIN)) - 2
    hi_int = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_int < 0 or hi_int > len(held_out_noise):
        continue
    interp = interp1d(np.arange(lo_int, hi_int), held_out_noise[lo_int:hi_int], kind="cubic",
                       bounds_error=False, fill_value=0.0)
    snippet = interp(true_center - NT0MIN + np.arange(N))
    fit = fit_nuisance_prealigned(snippet, main["s0"], main["q"], whitening_matrix=W)
    null_r2.append(fit["r_squared"])
null_r2 = np.array(null_r2)
null_p90 = np.percentile(null_r2, 90)
null_p99 = np.percentile(null_r2, 99)
print(f"\nQ1 -- NOISE-ONLY null (n={len(null_r2)}, same channel, same pipeline):")
print(f"  null R2: mean={null_r2.mean():.3f}, median={np.median(null_r2):.3f}, "
      f"90th pctile={null_p90:.3f}, 99th pctile={null_p99:.3f}, max={null_r2.max():.3f}")
print(f"  detected-spike R2 low bar was {r2_low_bar:.3f} -- {'BELOW' if r2_low_bar < null_p90 else 'above'} "
      f"the null's own 90th percentile")

# ============================================================
# Q2: for each candidate -- already-detected-elsewhere? better fit to a neighbor?
# ============================================================
rows = []
for c in candidates:
    r_main = align_and_fit(c, main, whitening=W)
    r_main_plain = align_and_fit(c, main, whitening=None)
    if r_main is None:
        continue
    if r_main["r_squared"] < r2_low_bar:
        continue  # only look closely at the ones that actually passed

    # already an existing spike of a DIFFERENT cluster nearby?
    other_mask = spike_clusters != UID
    other_times = spike_times[other_mask]
    other_clusters = spike_clusters[other_mask]
    diffs = np.abs(other_times - c)
    jmin = np.argmin(diffs)
    jitter_dist = int(diffs[jmin])
    jitter_cluster = int(other_clusters[jmin]) if jitter_dist <= JITTER_MATCH_WINDOW else None

    neighbor_r2 = {}
    for u, unit in neighbors.items():
        r_n = align_and_fit(c, unit, whitening=None)
        neighbor_r2[u] = r_n["r_squared"] if r_n is not None else np.nan

    best_neighbor = max(neighbor_r2, key=lambda k: neighbor_r2[k]) if neighbor_r2 else None
    rows.append(dict(
        sample=c, r2_whitened=r_main["r_squared"], r2_plain=r_main_plain["r_squared"] if r_main_plain else np.nan,
        already_detected_cluster=jitter_cluster, jitter_dist=jitter_dist if jitter_cluster is not None else np.nan,
        **{f"r2_vs_{u}": neighbor_r2[u] for u in NEARBY_UIDS},
        best_neighbor=best_neighbor, best_neighbor_r2=neighbor_r2.get(best_neighbor, np.nan),
        null_percentile=float((null_r2 < r_main["r_squared"]).mean() * 100),
    ))

df = pd.DataFrame(rows)
print(f"\nQ2/Q3 -- {len(df)} candidates that passed the bar, checked in detail:")
pd.set_option("display.width", 200)
print(df.to_string(index=False))

n_already_elsewhere = df["already_detected_cluster"].notna().sum()
n_neighbor_wins = (df["best_neighbor_r2"] > df["r2_whitened"]).sum()
n_would_fail_unwhitened = (df["r2_plain"] < r2_low_bar).sum()
n_above_null_p90 = (df["r2_whitened"] > null_p90).sum()
n_above_null_p99 = (df["r2_whitened"] > null_p99).sum()

print(f"\n=== ANSWERS ===")
print(f"Q1: candidates scoring above noise's own 90th percentile ({null_p90:.3f}): {n_above_null_p90}/{len(df)}")
print(f"    candidates scoring above noise's own 99th percentile ({null_p99:.3f}): {n_above_null_p99}/{len(df)}")
print(f"Q2a: candidates that are ALREADY a detected spike of a different cluster "
      f"(within {JITTER_MATCH_WINDOW} samples): {n_already_elsewhere}/{len(df)}")
print(f"Q2b: candidates a spatially-nearby unit's template explains BETTER than unit 307's: "
      f"{n_neighbor_wins}/{len(df)}")
print(f"Q3: candidates that would NOT have passed the bar WITHOUT whitening: "
      f"{n_would_fail_unwhitened}/{len(df)}")

df.to_csv(os.path.join(OUT, "unit307_candidate_validation.csv"), index=False)

# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.hist(null_r2, bins=20, color="#888", alpha=0.8, label=f"NOISE-only (n={len(null_r2)})")
ax.hist(df["r2_whitened"], bins=20, color="#2ca02c", alpha=0.6, label=f"candidates (n={len(df)})")
ax.axvline(r2_low_bar, color="#d62728", lw=1.5, ls="--", label=f"bar used = {r2_low_bar:.2f}")
ax.set_xlabel("R2 (whitened fit)")
ax.set_ylabel("count")
ax.set_title("Q1: are candidate scores above what\nPURE NOISE achieves by chance?", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
ax.axis("off")
ax.text(0, 1.0,
    f"unit {UID} candidate follow-up\n\n"
    f"noise null: median={np.median(null_r2):.2f}, p90={null_p90:.2f}, p99={null_p99:.2f}\n"
    f"bar used for flagging: {r2_low_bar:.2f}\n\n"
    f"of {len(df)} candidates that passed the bar:\n"
    f"  above noise p90: {n_above_null_p90}\n"
    f"  above noise p99: {n_above_null_p99}\n"
    f"  already a spike of ANOTHER cluster (jittered): {n_already_elsewhere}\n"
    f"  fit a NEARBY unit's template better: {n_neighbor_wins}\n"
    f"  would fail WITHOUT whitening: {n_would_fail_unwhitened}",
    fontsize=10, va="top", family="monospace", transform=ax.transAxes)

plt.suptitle(f"Unit {UID}: are the 'missed spike' candidates real, noise, or someone else's?", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "multiunit_03_unit307_candidate_validation.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
