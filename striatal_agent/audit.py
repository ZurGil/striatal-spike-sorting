"""
Section 9.3 -- audit/summary module. Required output, not optional: per-unit audit
record plus a session-level summary ending in one headline sentence, so a null or weak
result is visible as a finding rather than buried in tables.
"""
import numpy as np


def per_unit_audit(unit_id, profile, structural_detail, sweep_result, collision_summary,
                    tier, acg_violation_before, acg_violation_after, unit_class):
    n_candidates = len(sweep_result.get("sweep", []))
    n_recovered = sweep_result.get("n_recovered", 0)
    n_considered = int(len(structural_detail.get("footprint_channels", [])) and
                        sweep_result.get("recovered_scores", np.array([])).shape[0]) or n_recovered

    return dict(
        unit_id=int(unit_id),
        unit_class=unit_class,
        ks_original_spike_count=profile["n_spikes"],
        mean_rate_hz=profile["mean_rate_hz"],
        burst_fraction=profile["burst_fraction"],
        pause_time_fraction=profile["pause_time_fraction"],
        structural_score=structural_detail["structural_score"],
        acg_violation_ratio=structural_detail["acg_violation_ratio"],
        n_recovery_candidates_scored=n_candidates,
        n_spikes_added=n_recovered,
        pct_spikes_added=sweep_result.get("pct_spikes_added", 0.0),
        recovered_score_distribution_summary=(
            dict(mean=float(np.mean(sweep_result["recovered_scores"])),
                 min=float(np.min(sweep_result["recovered_scores"])),
                 max=float(np.max(sweep_result["recovered_scores"])))
            if sweep_result.get("n_recovered", 0) > 0 else None
        ),
        acg_violation_before=acg_violation_before,
        acg_violation_after=acg_violation_after,
        acg_degraded=(acg_violation_after - acg_violation_before) > 0,
        footprint_stability_score=structural_detail["footprint_stability_detail"].get("score"),
        collision_tag_breakdown=collision_summary.get("counts", {}),
        decision_tier=tier,
    )


def session_summary(per_unit_records, calibration_set_records=None):
    n_units = len(per_unit_records)
    if n_units == 0:
        return dict(headline="No units audited.", n_units=0)

    pct_added = np.array([r["pct_spikes_added"] for r in per_unit_records])
    n_zero_additions = int(np.sum(pct_added == 0))
    n_degraded = int(np.sum([r["acg_degraded"] for r in per_unit_records]))

    tier_counts = {}
    for r in per_unit_records:
        tier_counts[r["decision_tier"]] = tier_counts.get(r["decision_tier"], 0) + 1

    class_counts = {}
    for r in per_unit_records:
        class_counts[r["unit_class"]] = class_counts.get(r["unit_class"], 0) + 1

    frac_zero = n_zero_additions / n_units
    frac_degraded = n_degraded / n_units
    median_pct = float(np.median(pct_added))
    mean_pct = float(np.mean(pct_added))

    headline = (
        f"Recovery added a median {median_pct:.2f}% (mean {mean_pct:.2f}%) of spikes across "
        f"{n_units} units audited; {100*(1-frac_zero):.0f}% of units gained at least one spike, "
        f"{100*frac_degraded:.1f}% showed increased ACG violations post-addition (red flag list "
        f"below); decision tiers: {tier_counts}."
    )

    summary = dict(
        n_units=n_units,
        pct_spikes_added_distribution=dict(
            median=median_pct, mean=mean_pct,
            p25=float(np.percentile(pct_added, 25)), p75=float(np.percentile(pct_added, 75)),
            min=float(pct_added.min()), max=float(pct_added.max()),
        ),
        fraction_units_zero_additions=frac_zero,
        fraction_units_acg_degraded=frac_degraded,
        red_flag_units=[r["unit_id"] for r in per_unit_records if r["acg_degraded"]],
        tier_distribution=tier_counts,
        unit_class_distribution=class_counts,
        headline=headline,
    )

    if calibration_set_records:
        summary["calibration_set_performance"] = calibration_set_records

    return summary
