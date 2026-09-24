"""
Section 7 calibration parameters, collected in one place. Defaults below are the
spec's priors / the validation-gate findings from this session -- NOT yet calibrated
against a visually-verified reference set (spec Section 7's calibration approach).
Every field that still needs that calibration pass is marked below.
"""
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    fs: float = 30000.0
    n_chan: int = 384

    # --- preprocessing (2.1) ---
    highpass_hz: float = 300.0
    # Persisting a full filtered copy of a 250GB/3hr/384ch recording as float32 would
    # roughly double raw disk usage and may not fit (see session disk-space check before
    # committing). Default: filter on-the-fly in per-unit/per-window reads, matching what
    # the Section 1 validation gate already did. Set True only after checking free space.
    persist_filtered_traces: bool = False

    # --- structural scoring (2.3) --- NEEDS CALIBRATION (Section 7)
    footprint_channel_radius_um: float = 100.0
    structural_score_weights: dict = field(default_factory=lambda: dict(
        footprint=0.20, waveform_shape=0.15, ccg=0.20, refractory=0.15,
        footprint_concentration=0.15, cross_unit_acg=0.15,
    ))
    # cross_unit_acg is NOT used as a hard-flag trigger (see structural_score.py) --
    # calibrated against a null sample of unrelated unit pairs and found to flag 58-72%
    # of units by chance alone (multiple-comparisons artifact of searching for each
    # unit's single best match among ~80-400 candidates). It first caught 187/156/151
    # (r=0.97-0.99, mutual matches within a small 8-unit pool), but the same check on a
    # random 120-unit sample flagged 58% of units, most with no other supporting
    # evidence -- a false-positive rate, not a real finding. Left here only as reported
    # context (cross_unit_score, a mild capped soft penalty) for human review.
    cross_unit_acg_flag_threshold: float = 0.9  # unused by the hard-flag path; kept for cross_unit_score's soft penalty curve
    # skip_cross_unit_acg: when True, don't compute cross_unit_acg_info at all (every
    # unit gets cross_unit_score=1.0, i.e. neutral). Given the note above -- this signal
    # was already found to be a multiple-comparisons false-positive artifact, not a real
    # finding, and never drives the hard likely_shared_artifact flag -- this is a safe,
    # well-justified thing to disable, and also removes the only per-unit dependency on
    # which OTHER units happen to be processed in the same run (needed for batched runs
    # to be split without affecting results; see conversation, 2026-09-15).
    skip_cross_unit_acg: bool = False

    # likely_shared_artifact hard-flag thresholds -- calibrated against all 401 units in
    # this session (see conversation): each independently sits at the ~5-10% rarity mark,
    # unlike cross_unit_acg above. footprint_concentration_flag_threshold=0.2 -> ~5.7% of
    # units; width_outlier_threshold_ms=0.55 -> ~10.0% of units (0.60ms -> ~5.5%).
    footprint_concentration_flag_threshold: float = 0.2
    width_outlier_threshold_ms: float = 0.55
    # footprint_flatness_flag_threshold: the mirror-image failure mode of
    # footprint_concentration (decays too slowly/broadly instead of too fast). Chosen at
    # 0.75 specifically to be the SAFE end of the calibration range -- 0.7 already gave
    # 89% noise precision but included one manually-confirmed GOOD unit (0.741) as a
    # false positive; 0.75 excludes it entirely (0/12 GOOD, 1 MUA, 11 NOISE = 92%
    # precision in the 326-unit ground truth) while still hitting a comparable ~4.0% of
    # all 401 units to concentration_flag_threshold's ~5.7%.
    footprint_flatness_flag_threshold: float = 0.75

    # --- firing pattern profiling (Section 3) ---
    burst_seed_isi_ms: float = 15.0          # prior for MSNs, spec Sec 7 -- NEEDS CALIBRATION per-unit against log-ISI dip
    burst_min_spikes: int = 3
    burst_min_surprise: float = 2.0          # S = -log10(P); NEEDS CALIBRATION
    pause_isi_multiplier: float = 4.0        # prior 3-5x local median ISI, spec Sec 7
    pause_local_window_s: float = 30.0
    min_spikes_for_profiling: int = 200      # NEEDS CALIBRATION

    # --- footprint stability (2.6) ---
    footprint_stability_n_segments: int = 10  # session split into N segments to track shape drift

    # --- collision / contamination (2.5) ---
    collision_neighbor_radius_um: float = 100.0
    collision_score_margin: float = 0.15     # min relative-score margin to call "clean" vs "collision"

    # --- recovery (2.7) ---
    footprint_n_channels: int = 10
    th_sweep_start: float = 7.0              # KS's own Th_learned operating point (from this session's ops.npy)
    th_sweep_stop: float = 2.0
    th_sweep_step: float = 0.5
    acg_refractory_ms: float = 1.5
    acg_shoulder_lo_ms: float = 5.0
    acg_shoulder_hi_ms: float = 25.0
    recovery_burst_only: bool = False        # sweep whole session vs. burst windows only; validation gate used bursts only
    max_bursts_per_unit_io_cap: int = 150    # I/O cap, matches validation-gate script
    recovery_n_random_windows: int = 40      # additional non-burst windows sampled session-wide (full 3hr scan per unit is too costly at scale)
    recovery_random_window_s: float = 2.0
    recovery_sweep_percentiles: tuple = (50, 40, 30, 20, 10, 5, 2, 1)  # accepted-score percentile grid, strict->permissive
    recovery_min_candidate_score_frac: float = 0.15  # candidates below this fraction of mean(accepted_scores) are dropped before the sweep -- calibration finding: without this floor, tens of thousands of near-noise local maxima dilute the ACG-inflection control until it stops detecting anything

    # --- burst/pause submodule (Section 4) ---
    isi_bin_edges_ms: tuple = (0, 5, 10, 15, 25, 40, 65, 100, 1e9)  # preceding-ISI bins for position-indexed shape

    # --- merge/split scoring (2.8) --- NEEDS CALIBRATION
    merge_score_weights: dict = field(default_factory=lambda: dict(
        structural=0.4, ccg=0.35, amplitude=0.25,
    ))

    # --- decision tiers (Section 6) --- NEEDS CALIBRATION against 5-10 unit reference set
    tier1_structural_score_min: float = 0.85
    tier1_acg_degradation_max: float = 0.02   # max allowed increase in refractory-violation rate post-recovery
    tier2_structural_score_min: float = 0.5
    tier3_min_spike_count: int = 200          # below this -> "insufficient evidence", never silently defaulted

    # --- ContamPct: per this session's finding, KS's ContamPct assumes a homogeneous
    # spike rate and over-flags genuinely clean bursty units. Never used as a trust/
    # selection filter anywhere in this pipeline -- ACG shape (acg_tools) is used instead.
    use_ks_contam_pct: bool = False
