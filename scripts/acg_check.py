"""
Direct-ACG check to test whether KS's ContamPct (Hill et al. 2011-style, assumes a
homogeneous/uniform spike rate across the recording) is over-flagging genuinely clean,
bursty units as contaminated -- per the user's finding that bursty MSN-like units violate
that homogeneity assumption by construction, which can inflate ContamPct without any real
mixed-identity problem.

Computes the actual autocorrelogram (all pairwise spike-time differences within a lag
window, not just KS's formula) for each candidate unit, and compares the observed density
right at zero lag (the refractory zone) against the *empirical* density a bit further out
(the "shoulder", 5-25ms) -- no assumption about uniform rate across the whole recording,
just the unit's own local ACG shape.

  violation_ratio = (observed count in |lag|<=1.5ms) / (expected count if it were as dense
                     as the 5-25ms shoulder)

  ~0   -> sharp, clean dip at zero -> real single-unit refractoriness, ContamPct is
          probably an artifact of the homogeneity assumption
  ~1   -> no suppression at zero relative to the shoulder -> genuine mixed-identity
          contamination, consistent with high ContamPct
"""
import os
import numpy as np
import pandas as pd

KS_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4"
OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0

REFRACTORY_MS = 1.5
SHOULDER_LO_MS = 5.0
SHOULDER_HI_MS = 25.0
BIN_MS = 0.5
MAX_LAG_MS = 30.0

UNITS = [187, 188, 153, 156, 395, 266, 393, 221]  # the 8 units from the bursty/lower-SNR batch
ELEVATED_THRESHOLD = 0.25  # ratio above this => call it "elevated near zero" (group a)


def compute_acg_hist(times_samples, fs=FS, max_lag_ms=MAX_LAG_MS, bin_ms=BIN_MS):
    """Vectorized full-pair ACG histogram (forward pairs only, mirrored)."""
    times = np.sort(times_samples).astype(np.int64)
    n = len(times)
    max_lag = max_lag_ms / 1000.0 * fs

    right_idx = np.searchsorted(times, times + max_lag, side="right")
    counts = np.clip(right_idx - (np.arange(n) + 1), 0, None)
    total_pairs = int(counts.sum())

    if total_pairs == 0:
        bins = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
        return np.zeros(len(bins) - 1), bins, 0

    row_idx = np.repeat(np.arange(n), counts)
    cum_counts = np.cumsum(counts)
    starts_group = cum_counts - counts
    flat_idx = np.arange(total_pairs)
    within_group_offset = flat_idx - np.repeat(starts_group, counts)
    j_idx = row_idx + 1 + within_group_offset

    diffs_ms = (times[j_idx] - times[row_idx]) / fs * 1000.0
    all_diffs_ms = np.concatenate([diffs_ms, -diffs_ms])

    bins = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    hist, edges = np.histogram(all_diffs_ms, bins=bins)
    return hist, edges, total_pairs


def violation_ratio_from_hist(hist, edges):
    centers = (edges[:-1] + edges[1:]) / 2.0
    refr_mask = np.abs(centers) <= REFRACTORY_MS
    shoulder_mask = (np.abs(centers) >= SHOULDER_LO_MS) & (np.abs(centers) <= SHOULDER_HI_MS)

    refr_count = hist[refr_mask].sum()
    shoulder_mean_per_bin = hist[shoulder_mask].mean() if shoulder_mask.sum() else 0.0
    n_refr_bins = refr_mask.sum()
    expected_if_flat = shoulder_mean_per_bin * n_refr_bins
    ratio = refr_count / expected_if_flat if expected_if_flat > 0 else (np.inf if refr_count > 0 else 0.0)
    return float(ratio), int(refr_count), float(expected_if_flat)


def main():
    spike_templates = np.load(os.path.join(KS_DIR, "spike_templates.npy")).ravel()
    spike_times = np.load(os.path.join(KS_DIR, "spike_times.npy")).ravel()

    rows = []
    for u in UNITS:
        st = spike_times[spike_templates == u]
        hist, edges, total_pairs = compute_acg_hist(st)
        ratio, refr_count, expected = violation_ratio_from_hist(hist, edges)
        group = "a_elevated_real_contam" if ratio >= ELEVATED_THRESHOLD else "b_clean_likely_ContamPct_artifact"
        rows.append(dict(
            unit_id=u, n_spikes=len(st), total_acg_pairs=total_pairs,
            refractory_pair_count=refr_count, expected_if_flat_shoulder=round(expected, 1),
            violation_ratio=round(ratio, 4), group=group,
        ))
        print(f"unit {u}: n_spikes={len(st)}, refr_count={refr_count}, "
              f"expected_if_flat={expected:.1f}, violation_ratio={ratio:.4f} -> {group}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "acg_check_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    print("\n=== GROUP SPLIT ===")
    print(df[["unit_id", "violation_ratio", "group"]].sort_values("violation_ratio").to_string(index=False))

    # cross-reference with previously computed recovery numbers (no raw I/O needed, reuse the
    # matched-filter results already computed in the bursty validation-gate run)
    prev = pd.read_csv(os.path.join(OUT_DIR, "validation_gate_results_bursty.csv"))
    merged = df.merge(prev, on="unit_id")
    group_b = merged[merged["group"] == "b_clean_likely_ContamPct_artifact"]

    print("\n=== GROUP B (clean-by-ACG, ContamPct likely an artifact) RECOVERY NUMBERS ===")
    if len(group_b):
        print(group_b[["unit_id", "violation_ratio", "pct_spikes_recovery_would_add",
                        "n_candidates_clean_after_refractory", "baseline_refractory_violation_rate_pct"]].to_string(index=False))
        print(f"\nGroup B mean %% spikes recovery would add: {group_b['pct_spikes_recovery_would_add'].mean():.3f}%%")
        print(f"Group B median %% spikes recovery would add: {group_b['pct_spikes_recovery_would_add'].median():.3f}%%")
    else:
        print("(no units classified as group B)")

    group_a = merged[merged["group"] == "a_elevated_real_contam"]
    print("\n=== GROUP A (elevated ACG, likely real contamination) RECOVERY NUMBERS (for comparison) ===")
    if len(group_a):
        print(group_a[["unit_id", "violation_ratio", "pct_spikes_recovery_would_add",
                        "n_candidates_clean_after_refractory", "baseline_refractory_violation_rate_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
