"""
Shared autocorrelogram utilities. Used by: structural_score (refractory-contamination
term), recovery (ACG-inflection stopping rule for the threshold sweep, spec Section 2.7's
"required control"), and audit (before/after violation reporting, spec Section 9.3).

violation_ratio is deliberately NOT KS's ContamPct formula: this session's validation
work found ContamPct over-flags genuinely clean bursty units because it assumes a
homogeneous spike rate across the whole recording. violation_ratio instead compares the
observed near-zero-lag density against the unit's own empirical shoulder density (5-25ms)
-- no rate-homogeneity assumption. Calibrated in-session against known-good units
(ratio 0.06-0.22) and known-bad units (ContamPct>100%, ratio 0.59-0.94).
"""
import numpy as np

from .config import AgentConfig


def compute_acg_hist(times_samples, fs, max_lag_ms=30.0, bin_ms=0.5):
    times = np.sort(np.asarray(times_samples)).astype(np.int64)
    n = len(times)
    max_lag = max_lag_ms / 1000.0 * fs
    bins = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)

    if n < 2:
        return np.zeros(len(bins) - 1), bins, 0

    right_idx = np.searchsorted(times, times + max_lag, side="right")
    counts = np.clip(right_idx - (np.arange(n) + 1), 0, None)
    total_pairs = int(counts.sum())
    if total_pairs == 0:
        return np.zeros(len(bins) - 1), bins, 0

    row_idx = np.repeat(np.arange(n), counts)
    cum_counts = np.cumsum(counts)
    starts_group = cum_counts - counts
    flat_idx = np.arange(total_pairs)
    within_group_offset = flat_idx - np.repeat(starts_group, counts)
    j_idx = row_idx + 1 + within_group_offset

    diffs_ms = (times[j_idx] - times[row_idx]) / fs * 1000.0
    all_diffs_ms = np.concatenate([diffs_ms, -diffs_ms])
    hist, edges = np.histogram(all_diffs_ms, bins=bins)
    return hist, edges, total_pairs


def violation_ratio(times_samples, fs, cfg: AgentConfig = None):
    cfg = cfg or AgentConfig()
    hist, edges, total_pairs = compute_acg_hist(
        times_samples, fs, max_lag_ms=max(cfg.acg_shoulder_hi_ms, cfg.acg_refractory_ms) + 5)
    centers = (edges[:-1] + edges[1:]) / 2.0
    refr_mask = np.abs(centers) <= cfg.acg_refractory_ms
    shoulder_mask = (np.abs(centers) >= cfg.acg_shoulder_lo_ms) & (np.abs(centers) <= cfg.acg_shoulder_hi_ms)

    refr_count = hist[refr_mask].sum()
    shoulder_mean = hist[shoulder_mask].mean() if shoulder_mask.sum() else 0.0
    expected_if_flat = shoulder_mean * refr_mask.sum()
    if expected_if_flat > 0:
        ratio = refr_count / expected_if_flat
    else:
        ratio = np.inf if refr_count > 0 else 0.0
    return float(ratio), int(refr_count), float(expected_if_flat)


def refractory_violation_rate(spike_times_samples, fs, refractory_ms=1.5):
    """Nearest-neighbour ISI violation fraction -- sufficient for refractory-period
    violations specifically (any violating pair is necessarily adjacent in the sorted
    spike train), cheaper than the full ACG when that's all that's needed."""
    st = np.sort(spike_times_samples)
    if len(st) < 2:
        return 0.0
    isi_ms = np.diff(st) / fs * 1000.0
    return float(np.mean(isi_ms < refractory_ms))


def cross_unit_acg_similarity(acg_hists: dict, peak_channels: dict, geometry=None):
    """Session-wide check (not a single-unit one): flags units whose autocorrelogram
    shape is near-identical to some OTHER unit's, which a real, independently-generated
    spike train essentially never produces by chance -- each neuron's ACG reflects its
    own refractory/bursting dynamics. Caught in this session by eye: units 187/156/151
    had pairwise ACG shape correlations of 0.97-0.99 despite sitting hundreds of microns
    apart on the probe, while every other pair in the same batch was 0.06-0.74. That
    pattern (near-identical timing statistics + physically separate locations) reads as
    a shared external artifact each channel independently picks up, not three real
    neurons that happen to fire alike.

    acg_hists: {unit_id: counts array}, same binning for every unit (acg_tools.compute_acg_hist).
    Returns {unit_id: dict(best_match_unit, correlation, distance_um)}.
    """
    ids = list(acg_hists.keys())
    z = {u: (np.asarray(acg_hists[u], dtype=float) - np.mean(acg_hists[u])) /
            (np.std(acg_hists[u]) + 1e-9) for u in ids}
    results = {}
    for u in ids:
        best_r, best_u = -2.0, None
        for v in ids:
            if v == u:
                continue
            r = float(np.corrcoef(z[u], z[v])[0, 1])
            if r > best_r:
                best_r, best_u = r, v
        dist = None
        if geometry is not None and best_u is not None and u in peak_channels and best_u in peak_channels:
            dist = geometry.distance(peak_channels[u], peak_channels[best_u])
        results[u] = dict(best_match_unit=best_u, correlation=best_r, distance_um=dist)
    return results


def find_inflection_threshold(thresholds, violation_rates):
    """Spec Section 2.7's 'required control': as detection threshold is swept down,
    track refractory-violation rate; the inflection point (where violations start
    climbing) is the principled stopping point rather than an arbitrary fixed value.

    thresholds must be sorted descending (KS convention: higher Th = stricter).
    Returns (threshold, found) where found=True means a clear inflection (violation
    rate climbing) was located, and the threshold is the point just before it.

    Calibration-set finding (this session): with large candidate pools, the violation
    ratio can *decrease* rather than climb as more low-score candidates are admitted --
    thousands of near-noise local maxima dilute the ACG numerator and denominator
    together rather than concentrating excess density at zero lag, so "no climbing
    signal seen" is NOT evidence of safety. When no clear inflection is found, this
    must fail conservative (strictest tested cutoff) rather than permissive (most
    permissive tested cutoff) -- silently defaulting to the most permissive threshold
    let ~40-90% of a unit's spike count get added as "recovered" for a bug, not a
    finding. found=False should route the caller to tier2 (human review of the sweep
    curve), never to tier1 auto-apply.
    """
    thresholds = np.asarray(thresholds)
    violation_rates = np.asarray(violation_rates)
    if len(thresholds) < 3:
        return (thresholds[0] if len(thresholds) else None), False

    d = np.diff(violation_rates)
    median_step = np.median(np.abs(d[: max(1, len(d) // 2)])) + 1e-9
    for i in range(1, len(d)):
        if d[i] > 4 * median_step and d[i] > 0.005:
            return float(thresholds[i]), True
    return float(thresholds[0]), False
