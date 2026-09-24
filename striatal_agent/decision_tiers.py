"""
Section 6 -- decision authority tiers. Turns scores from recovery.py / merge_split.py
into an explicit tier, never silently defaulting a decision into any category:

  Tier 1 - auto-applied: high, unambiguous structural evidence. Written directly to
           agent_processed output.
  Tier 2 - scored and logged, not auto-applied: ambiguous merges/splits, near-threshold
           recovery candidates, unexplained-residual cases. Ranked for human review.
  Tier 3 - flagged as insufficient evidence: low spike count, failed profiling-stability
           checks. Explicitly marked, never silently folded into tier 1 or 2.

Thresholds are the Section 7 calibration set defaults (config.py) -- NOT yet tuned
against a visually-verified reference set. Treat tier boundaries as provisional until
that calibration pass (spec Section 7) is done.
"""
from .config import AgentConfig


def unit_tier3_check(profile, structural_detail, cfg: AgentConfig = None):
    cfg = cfg or AgentConfig()
    if profile["n_spikes"] < cfg.tier3_min_spike_count:
        return True, f"n_spikes ({profile['n_spikes']}) below tier3_min_spike_count ({cfg.tier3_min_spike_count})"
    fp_detail = structural_detail.get("footprint_stability_detail", {})
    if fp_detail.get("n_segments_with_data", 0) < 2:
        return True, "insufficient raw-data coverage for footprint-stability check"
    return False, None


def recovery_tier(structural_score, acg_violation_before, acg_violation_after, cfg: AgentConfig = None,
                   inflection_found=True):
    """inflection_found=False (acg_tools.find_inflection_threshold could not locate a
    clear stopping point) always caps at tier2 -- calibration-set finding: no detected
    inflection is not evidence of safety, so it must never reach tier1 auto-apply."""
    cfg = cfg or AgentConfig()
    degradation = acg_violation_after - acg_violation_before
    if not inflection_found:
        return "tier2_scored_logged", degradation
    if structural_score >= cfg.tier1_structural_score_min and degradation <= cfg.tier1_acg_degradation_max:
        return "tier1_auto_applied", degradation
    return "tier2_scored_logged", degradation  # never below tier2 for a decision that reached this stage; tier3 is a unit-level gate applied earlier


def merge_tier(merge_score_value, cfg: AgentConfig = None):
    cfg = cfg or AgentConfig()
    if merge_score_value >= cfg.tier1_structural_score_min:
        return "tier1_auto_applied"
    if merge_score_value >= cfg.tier2_structural_score_min:
        return "tier2_scored_logged"
    return "not_surfaced"  # below the noise floor for even a logged candidate


def collision_tier(collision_result, cfg: AgentConfig = None):
    """Recovery/QC candidates classified as unexplained-residual (2.5c) are never
    auto-applied -- they're evidence of a real gap (spec Section 5), always tier 2."""
    label = collision_result.get("label", "")
    if label == "clean":
        return "eligible_for_tier1"
    if label == "unexplained_residual":
        return "tier2_scored_logged"
    return "tier2_scored_logged"  # known collision -- always surfaced, never silently dropped or kept
