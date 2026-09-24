"""
Orchestrator -- single entry point (see run_agent.py at repo root). Runs the Section 2
core curation loop per unit, then session-wide merge scoring, then writes the
non-destructive Section 9 output.

Deliberately conservative defaults for this first scaffolded run:
  - apply_tier1_merges=False: merge candidates are scored and logged (Section 2.8), but
    the pipeline does NOT auto-rewrite spike_clusters.npy via merges yet, even for
    tier-1-scored pairs -- relabeling cluster identity is a stronger action than adding a
    clearly-separated child cluster, and this is the first run against real data. Flip on
    once tier-1 merge calls have been checked against a few examples.
  - Tier-1 RECOVERY (adding a child cluster of new spikes) *is* applied, per spec Section
    9.2's explicit design: it's reversible-by-construction in phy (a separate cluster you
    can merge back with one click), which is exactly why the spec treats it differently
    from merges.
"""
import os
import json
import numpy as np

from .config import AgentConfig
from .session import discover_session, SessionPaths
from .raw_io import RawReader, ChannelGeometry, footprint_channels
from .firing_pattern import profile_unit
from .structural_score import structural_score_for_unit
from .recovery import compute_candidate_scores, sweep_and_select
from .burst_module import classify_unit_class, apply_pause_conservatism
from .collision import classify_unit_spikes, find_spatial_neighbors
from .merge_split import candidate_pairs_from_similarity, merge_score
from .decision_tiers import unit_tier3_check, recovery_tier, merge_tier
from .acg_tools import violation_ratio, compute_acg_hist, cross_unit_acg_similarity
from .audit import per_unit_audit, session_summary
from . import output_writer as ow


def run_structural_screen(session_root, units=None, cfg: AgentConfig = None, seed=0, verbose=True):
    """Cheap, already-validated-only screening pass: structural score (footprint
    stability + waveform shape + CCG/ACG cleanliness + refractory + footprint
    concentration + cross-unit ACG similarity) across many/all units, WITHOUT the
    recovery sweep, collision check, or merge scoring -- those still have open issues
    (see conversation: the ACG-inflection control doesn't reliably fire, merges aren't
    auto-applied yet). Answers: how many likely-artifact units (like 187/156/151) exist
    across the full session, using exactly the checks that caught those three.

    Writes score columns into agent_processed/ (phy-visible) but does not touch
    spike_clusters.npy/spike_times.npy -- this is read-only analysis, not a curation pass.
    """
    cfg = cfg or AgentConfig()
    rng = np.random.default_rng(seed)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    import time
    t_start = time.time()

    paths = discover_session(session_root)
    log(f"Session: {paths.session_id}")
    ks_orig = ow.build_kilosort_original(paths)
    ow.init_agent_processed(paths)

    spike_times_all = ks_orig["spike_times"]
    spike_clusters_all = ks_orig["spike_clusters"]
    templates = ks_orig["templates"]

    geometry = ChannelGeometry.from_json(paths.channel_map_json)
    reader = RawReader(paths.probe_dat, cfg.n_chan)
    session_duration_samples = reader.n_samples
    session_duration_s = session_duration_samples / cfg.fs

    ops = np.load(os.path.join(paths.ks4_output_dir, "ops.npy"), allow_pickle=True).item()
    nt0min = int(ops.get("nt0min", 20))

    all_unit_ids = np.unique(spike_clusters_all).tolist()
    target_units = [u for u in units if u in all_unit_ids] if units is not None else all_unit_ids
    log(f"Screening {len(target_units)} / {len(all_unit_ids)} units")

    peak_channels = {u: int(np.argmax(templates[u].max(axis=0) - templates[u].min(axis=0)))
                      for u in all_unit_ids}

    log("Pass 1/2: profiling + ACG histograms...")
    profiles, acg_hists, spike_times_cache = {}, {}, {}
    for unit_id in target_units:
        st = spike_times_all[spike_clusters_all == unit_id]
        spike_times_cache[unit_id] = st
        profiles[unit_id] = profile_unit(st, cfg.fs, session_duration_s, cfg)
        hist, edges, total_pairs = compute_acg_hist(st, cfg.fs, max_lag_ms=30.0, bin_ms=0.5)
        acg_hists[unit_id] = hist
    cross_unit_info = cross_unit_acg_similarity(acg_hists, peak_channels, geometry)

    log("Pass 2/2: structural scoring (raw-data reads for footprint/waveform checks)...")
    results = []
    score_columns = dict(agent_structural_score={}, footprint_stability={}, waveform_shape={},
                          ccg={}, refractory={}, footprint_concentration={}, footprint_flatness={},
                          cross_unit_acg_correlation={}, likely_shared_artifact={}, tier3={})
    t_pass2 = time.time()
    for i, unit_id in enumerate(target_units):
        st = spike_times_cache[unit_id]
        profile = profiles[unit_id]

        neighbor_ids = find_spatial_neighbors(unit_id, templates, geometry,
                                               cfg.collision_neighbor_radius_um, peak_channels)
        neighbor_spike_times = {nb: spike_times_all[spike_clusters_all == nb] for nb in neighbor_ids[:5]}

        structural = structural_score_for_unit(
            unit_id, st, templates, geometry, reader, templates.shape[1], nt0min, cfg.fs,
            session_duration_samples, neighbor_spike_times, cfg, rng,
            cross_unit_acg_info=cross_unit_info.get(unit_id))
        is_tier3, tier3_reason = unit_tier3_check(profile, structural, cfg)

        results.append(dict(unit_id=unit_id, n_spikes=profile["n_spikes"],
                             mean_rate_hz=profile["mean_rate_hz"],
                             structural_score=structural["structural_score"],
                             components=structural["components"],
                             likely_shared_artifact=structural["likely_shared_artifact"],
                             cross_unit_acg_info=structural["cross_unit_acg_info"],
                             footprint_concentration_ratio=structural["footprint_concentration_ratio"],
                             footprint_flatness_ratio=structural["footprint_flatness_ratio"],
                             is_tier3=is_tier3, tier3_reason=tier3_reason))

        score_columns["agent_structural_score"][unit_id] = structural["structural_score"]
        score_columns["footprint_stability"][unit_id] = structural["footprint_stability_detail"].get("score", np.nan)
        score_columns["waveform_shape"][unit_id] = structural["waveform_shape_detail"].get("score", np.nan)
        score_columns["ccg"][unit_id] = structural["components"]["ccg"]
        score_columns["refractory"][unit_id] = structural["components"]["refractory"]
        score_columns["footprint_concentration"][unit_id] = structural["footprint_concentration_ratio"]
        score_columns["footprint_flatness"][unit_id] = structural["footprint_flatness_ratio"]
        score_columns["cross_unit_acg_correlation"][unit_id] = (
            structural["cross_unit_acg_info"]["correlation"] if structural["cross_unit_acg_info"] else np.nan)
        score_columns["likely_shared_artifact"][unit_id] = int(structural["likely_shared_artifact"])
        score_columns["tier3"][unit_id] = int(is_tier3)

        if (i + 1) % 25 == 0 or (i + 1) == len(target_units):
            elapsed = time.time() - t_pass2
            rate = (i + 1) / elapsed
            eta = (len(target_units) - (i + 1)) / rate if rate > 0 else float("nan")
            flagged = sum(r["likely_shared_artifact"] for r in results)
            log(f"  [{i+1}/{len(target_units)}] {elapsed:.0f}s elapsed, "
                f"{rate:.2f} units/s, ETA {eta:.0f}s -- {flagged} flagged so far")
            # checkpoint: an earlier run was killed mid-pass with zero results saved --
            # write progress every 25 units so an interruption doesn't lose everything
            report_path = os.path.join(paths.agent_processed_dir, "structural_screen.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    for label, mapping in score_columns.items():
        ow.write_cluster_tsv(paths.agent_processed_dir, label, mapping)

    n_flagged = sum(r["likely_shared_artifact"] for r in results)
    n_tier3 = sum(r["is_tier3"] for r in results)
    scores = np.array([r["structural_score"] for r in results])
    log(f"\n=== SCREEN SUMMARY ===")
    log(f"{len(results)} units screened in {time.time()-t_start:.0f}s")
    log(f"likely_shared_artifact: {n_flagged} ({100*n_flagged/len(results):.1f}%)")
    log(f"tier3 (insufficient evidence): {n_tier3} ({100*n_tier3/len(results):.1f}%)")
    log(f"structural_score: median={np.median(scores):.2f}, p25={np.percentile(scores,25):.2f}, "
        f"p75={np.percentile(scores,75):.2f}")
    log(f"flagged unit ids: {sorted(r['unit_id'] for r in results if r['likely_shared_artifact'])}")

    report_path = os.path.join(paths.agent_processed_dir, "structural_screen.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"Full results written to {report_path}")
    return results


def run_pipeline(session_root, units=None, n_units=None, apply_tier1_merges=False,
                  cfg: AgentConfig = None, seed=0, verbose=True):
    cfg = cfg or AgentConfig()
    rng = np.random.default_rng(seed)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    paths = discover_session(session_root)
    log(f"Session: {paths.session_id}")
    log("Building kilosort_original/ (reconstructed pre-phy baseline)...")
    ks_orig = ow.build_kilosort_original(paths)
    log("Initializing agent_processed/...")
    ow.init_agent_processed(paths)

    spike_times_all = ks_orig["spike_times"]
    spike_clusters_all = ks_orig["spike_clusters"]  # == original spike_templates
    templates = ks_orig["templates"]

    geometry = ChannelGeometry.from_json(paths.channel_map_json)
    reader = RawReader(paths.probe_dat, cfg.n_chan)
    session_duration_samples = reader.n_samples

    ops = np.load(os.path.join(paths.ks4_output_dir, "ops.npy"), allow_pickle=True).item()
    nt0min = int(ops.get("nt0min", 20))
    similar_templates = np.load(os.path.join(paths.kilosort_original_dir, "similar_templates.npy"))

    # KS's own ContamPct, shown for side-by-side comparison only (Section 9.1) -- never
    # used internally as a filter/trust signal (this session's finding: it over-flags
    # genuinely clean bursty units by assuming a homogeneous rate). Sourced from the
    # mid-curation snapshot in kilosort4\ since no pristine pre-merge version exists;
    # covers 352/401 original templates, NaN for the rest (see conversation).
    ks_contam_path = os.path.join(paths.ks4_output_dir, "cluster_ContamPct.tsv")
    ks_contam = {}
    if os.path.exists(ks_contam_path):
        import pandas as pd
        ks_contam = pd.read_csv(ks_contam_path, sep="\t").set_index("cluster_id")["ContamPct"].to_dict()

    all_unit_ids = np.unique(spike_clusters_all).tolist()
    if units is not None:
        target_units = [u for u in units if u in all_unit_ids]
    elif n_units is not None:
        target_units = all_unit_ids[:n_units]
    else:
        target_units = all_unit_ids
    log(f"Processing {len(target_units)} / {len(all_unit_ids)} units")

    peak_channels = {u: int(np.argmax(templates[u].max(axis=0) - templates[u].min(axis=0)))
                      for u in all_unit_ids}

    # pass 0: ACG histograms for every target unit, needed before cross-unit similarity
    # can run (cheap -- spike times only, no raw I/O). Comparison pool is target_units
    # only, not the full session -- fine for a subset run; for a full run this scans
    # all processed units against each other, which is O(n^2) in unit count and may need
    # tightening (e.g. only compare units with similar firing rate) if that's slow at
    # full (401-unit) scale.
    if cfg.skip_cross_unit_acg:
        log("skip_cross_unit_acg=True: skipping cross-unit ACG similarity check "
            "(already-documented false-positive-prone signal, never drives the hard "
            "likely_shared_artifact flag -- see config.py note).")
        cross_unit_info = {}
    else:
        log("Computing ACG histograms for cross-unit similarity check...")
        acg_hists = {}
        for unit_id in target_units:
            st = spike_times_all[spike_clusters_all == unit_id]
            hist, edges, total_pairs = compute_acg_hist(st, cfg.fs, max_lag_ms=30.0, bin_ms=0.5)
            acg_hists[unit_id] = hist
        cross_unit_info = cross_unit_acg_similarity(acg_hists, peak_channels, geometry)

    per_unit_records = []
    tier1_recoveries = {}
    score_columns = dict(ks_score={}, agent_structural_score={}, pct_spikes_added={},
                          n_recovered={}, acg_violation_pre={}, acg_violation_post={},
                          footprint_stability={}, footprint_concentration={}, footprint_flatness={},
                          cross_unit_acg_correlation={}, likely_shared_artifact={}, tier={})
    profile_cache, structural_cache, spike_times_cache = {}, {}, {}

    for i, unit_id in enumerate(target_units):
        log(f"[{i+1}/{len(target_units)}] unit {unit_id}")
        st = spike_times_all[spike_clusters_all == unit_id]
        spike_times_cache[unit_id] = st
        session_duration_s = session_duration_samples / cfg.fs

        profile = profile_unit(st, cfg.fs, session_duration_s, cfg)
        profile_cache[unit_id] = profile

        chans, peak_chan = footprint_channels(templates, unit_id, cfg.footprint_n_channels,
                                               geometry, cfg.footprint_channel_radius_um)
        neighbor_ids = find_spatial_neighbors(unit_id, templates, geometry,
                                               cfg.collision_neighbor_radius_um, peak_channels)
        neighbor_spike_times = {nb: spike_times_all[spike_clusters_all == nb] for nb in neighbor_ids[:5]}

        structural = structural_score_for_unit(
            unit_id, st, templates, geometry, reader, templates.shape[1], nt0min, cfg.fs,
            session_duration_samples, neighbor_spike_times, cfg, rng,
            cross_unit_acg_info=cross_unit_info.get(unit_id))
        structural_cache[unit_id] = structural
        if structural["likely_shared_artifact"]:
            if structural["cross_unit_acg_info"] is not None:
                log(f"  [!] likely shared artifact (cross-unit ACG r="
                    f"{structural['cross_unit_acg_info']['correlation']:.2f} with unit "
                    f"{structural['cross_unit_acg_info']['best_match_unit']}, "
                    f"{structural['cross_unit_acg_info']['distance_um']:.0f}um away)")
            else:
                log("  [!] likely shared artifact (footprint/width-based; "
                    "cross-unit ACG check skipped)")

        is_tier3, tier3_reason = unit_tier3_check(profile, structural, cfg)
        if is_tier3:
            log(f"  tier3 (insufficient evidence): {tier3_reason}")
            per_unit_records.append(per_unit_audit(
                unit_id, profile, structural, dict(sweep=[], n_recovered=0, pct_spikes_added=0.0),
                dict(counts={}), "tier3_insufficient_evidence",
                structural["acg_violation_ratio"], structural["acg_violation_ratio"],
                "unclassified_low_confidence"))
            continue

        unit_class = classify_unit_class(profile, cfg)

        scan_result = compute_candidate_scores(unit_id, st, templates, geometry, reader,
                                                cfg.fs, session_duration_samples, cfg, rng, nt0min)
        sweep_result = sweep_and_select(unit_id, st, scan_result, cfg.fs, cfg)

        recovered_times = sweep_result["recovered_times"]
        if unit_class == "tan_like" and len(recovered_times) > 0:
            recovered_times, dropped = apply_pause_conservatism(
                recovered_times, profile["pauses"], margin_ms=5.0, fs=cfg.fs)
            sweep_result["recovered_times"] = recovered_times
            sweep_result["n_recovered"] = len(recovered_times)
            sweep_result["pct_spikes_added"] = 100.0 * len(recovered_times) / max(len(st), 1)

        acg_before = structural["acg_violation_ratio"]
        union_times = np.sort(np.concatenate([st, recovered_times])) if len(recovered_times) else st
        acg_after, _, _ = violation_ratio(union_times, cfg.fs, cfg)

        tier, degradation = recovery_tier(structural["structural_score"], acg_before, acg_after, cfg,
                                           inflection_found=sweep_result.get("inflection_found", False))

        collision_summary = classify_unit_spikes(unit_id, st, chans, templates, geometry, reader,
                                                   peak_channels, templates.shape[1], nt0min, cfg, rng=rng)

        per_unit_records.append(per_unit_audit(
            unit_id, profile, structural, sweep_result, collision_summary, tier,
            acg_before, acg_after, unit_class))

        score_columns["ks_score"][unit_id] = ks_contam.get(unit_id, float("nan"))
        score_columns["agent_structural_score"][unit_id] = structural["structural_score"]
        score_columns["pct_spikes_added"][unit_id] = sweep_result["pct_spikes_added"]
        score_columns["n_recovered"][unit_id] = sweep_result["n_recovered"]
        score_columns["acg_violation_pre"][unit_id] = acg_before
        score_columns["acg_violation_post"][unit_id] = acg_after
        score_columns["footprint_stability"][unit_id] = structural["footprint_stability_detail"].get("score", np.nan)
        score_columns["footprint_concentration"][unit_id] = structural["footprint_concentration_ratio"]
        score_columns["footprint_flatness"][unit_id] = structural["footprint_flatness_ratio"]
        score_columns["cross_unit_acg_correlation"][unit_id] = (
            structural["cross_unit_acg_info"]["correlation"] if structural["cross_unit_acg_info"] else np.nan)
        score_columns["likely_shared_artifact"][unit_id] = int(structural["likely_shared_artifact"])
        score_columns["tier"][unit_id] = tier

        if tier == "tier1_auto_applied" and len(recovered_times) > 0:
            tier1_recoveries[unit_id] = recovered_times
            log(f"  tier1: +{len(recovered_times)} spikes ({sweep_result['pct_spikes_added']:.2f}%)")
        else:
            log(f"  {tier}: structural_score={structural['structural_score']:.2f}, "
                f"candidates={sweep_result['n_recovered']}")

    # --- merge/split scoring (session-wide, Section 2.8) ---
    log("\nScoring merge candidates...")
    merge_records = []
    pairs = candidate_pairs_from_similarity(similar_templates, peak_channels, geometry,
                                             cfg.collision_neighbor_radius_um)
    for pair in pairs:
        ua, ub = pair["unit_a"], pair["unit_b"]
        if ua not in structural_cache or ub not in structural_cache:
            continue
        chans_a, _ = footprint_channels(templates, ua, cfg.footprint_n_channels, geometry, cfg.footprint_channel_radius_um)
        ms = merge_score(ua, ub, spike_times_cache[ua], spike_times_cache[ub],
                          structural_cache[ua]["structural_score"], structural_cache[ub]["structural_score"],
                          templates, chans_a, cfg.fs, cfg)
        ms["tier"] = merge_tier(ms["merge_score"], cfg)
        ms["ks_similarity"] = pair["ks_similarity"]
        merge_records.append(ms)
    log(f"  {len(merge_records)} candidate pairs scored "
        f"({sum(1 for m in merge_records if m['tier']=='tier1_auto_applied')} tier1, "
        f"{sum(1 for m in merge_records if m['tier']=='tier2_scored_logged')} tier2)")
    if not apply_tier1_merges:
        log("  apply_tier1_merges=False: merges scored/logged only, not applied to spike_clusters.npy")

    # --- write output (Section 9) ---
    log("\nWriting agent_processed/ output...")
    child_id_map = ow.append_recovered_spikes(paths.agent_processed_dir, tier1_recoveries)
    for label, mapping in score_columns.items():
        ow.write_cluster_tsv(paths.agent_processed_dir, label, mapping)

    summary = session_summary(per_unit_records)
    log("\n" + summary["headline"])

    report = dict(session_id=paths.session_id, per_unit_records=per_unit_records,
                  merge_records=merge_records, summary=summary,
                  child_cluster_id_map=child_id_map, config=cfg.__dict__)
    report_path = os.path.join(paths.agent_processed_dir, "agent_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"Full report written to {report_path}")

    return report
