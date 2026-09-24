"""
Section 7 calibration set: run the full per-unit pipeline stages on a handful of units
spanning the ContamPct/amplitude range this session already gathered evidence about
(not just the cleanest units), and capture rich enough diagnostics (waveforms, ACG
histograms, recovery sweep curves) to support a visual ground-truth-by-eye check --
per spec, tune Tier 1 thresholds so automated calls match the visual read before
trusting auto-apply at scale.

Selected units (8, spanning the full tested range):
  285, 295  - clean batch, ContamPct 0.0%          (Section 1 gate)
  187, 156  - mua batch, ContamPct 26.5-29.2%, ACG-clean (violation_ratio 0.20-0.22)
  139, 151  - 30-45% ContamPct batch, ACG-clean (violation_ratio 0.23-0.27)
  69        - 30-45% batch OUTLIER, ACG ratio 0.63 -- flagged as likely real contamination,
              a negative control the pipeline should NOT rate highly
  229       - known-bad calibration unit from the original ACG-metric check, ContamPct
              102.4%, ratio 0.94 -- extreme negative control
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from striatal_agent.config import AgentConfig
from striatal_agent.session import discover_session
from striatal_agent.raw_io import RawReader, ChannelGeometry, footprint_channels, read_filtered_snippet
from striatal_agent.firing_pattern import profile_unit
from striatal_agent.structural_score import structural_score_for_unit
from striatal_agent.recovery import compute_candidate_scores, sweep_and_select
from striatal_agent.collision import classify_unit_spikes, find_spatial_neighbors
from striatal_agent.decision_tiers import unit_tier3_check, recovery_tier
from striatal_agent.acg_tools import compute_acg_hist, violation_ratio, cross_unit_acg_similarity
from striatal_agent import output_writer as ow

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
CALIBRATION_UNITS = [285, 295, 187, 156, 139, 151, 69, 229]
OUT_JSON = r"D:\Gil\spike_sorting_agent\outputs\calibration_set.json"


def sample_waveform(reader, spike_times, chans, nt, nt0min, fs, cutoff, n_sample=200, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(spike_times) == 0:
        return None
    idx = rng.choice(len(spike_times), size=min(n_sample, len(spike_times)), replace=False)
    snippets = []
    for t in spike_times[idx]:
        start = int(t) - nt0min
        block = read_filtered_snippet(reader, chans, start, nt, fs, cutoff)
        if block is not None:
            snippets.append(block)
    if not snippets:
        return None
    arr = np.array(snippets)  # (n, nt, k)
    return dict(mean=arr.mean(axis=0), std=arr.std(axis=0))


def sample_waveform_unfiltered_peak(reader, spike_times, peak_chan, nt, nt0min, n_sample=200, rng=None):
    """Old (buggy) behavior for comparison: raw, unfiltered, single channel."""
    rng = rng or np.random.default_rng(0)
    if len(spike_times) == 0:
        return None
    idx = rng.choice(len(spike_times), size=min(n_sample, len(spike_times)), replace=False)
    snippets = []
    for t in spike_times[idx]:
        start, end = int(t) - nt0min, int(t) - nt0min + nt
        block, a, b = reader.read_window(start, end, [peak_chan])
        if block.shape[0] == nt:
            snippets.append(block[:, 0])
    if not snippets:
        return None
    arr = np.array(snippets)
    return dict(mean=arr.mean(axis=0).tolist(), std=arr.std(axis=0).tolist())


def main():
    cfg = AgentConfig()
    rng = np.random.default_rng(0)
    paths = discover_session(SESSION_ROOT)

    print("Building/reusing kilosort_original...")
    ks_orig = ow.build_kilosort_original(paths)
    spike_times_all = ks_orig["spike_times"]
    spike_clusters_all = ks_orig["spike_clusters"]
    templates = ks_orig["templates"]

    geometry = ChannelGeometry.from_json(paths.channel_map_json)
    reader = RawReader(paths.probe_dat, cfg.n_chan)
    session_duration_samples = reader.n_samples
    session_duration_s = session_duration_samples / cfg.fs

    ops = np.load(os.path.join(paths.ks4_output_dir, "ops.npy"), allow_pickle=True).item()
    nt0min = int(ops.get("nt0min", 20))

    import pandas as pd
    contam = pd.read_csv(os.path.join(paths.ks4_output_dir, "cluster_ContamPct.tsv"), sep="\t").set_index("cluster_id")["ContamPct"].to_dict()
    amp_tsv = pd.read_csv(os.path.join(paths.ks4_output_dir, "cluster_Amplitude.tsv"), sep="\t").set_index("cluster_id")["Amplitude"].to_dict()
    kslabel = pd.read_csv(os.path.join(paths.ks4_output_dir, "cluster_KSLabel.tsv"), sep="\t").set_index("cluster_id")["KSLabel"].to_dict()

    all_unit_ids = np.unique(spike_clusters_all).tolist()
    peak_channels = {u: int(np.argmax(templates[u].max(axis=0) - templates[u].min(axis=0)))
                      for u in all_unit_ids}

    # pass 1: ACG histograms for every calibration unit, needed before cross-unit
    # comparison can run (cheap -- spike times only, no raw I/O)
    print("\nPass 1: computing ACG histograms for cross-unit comparison...")
    acg_hists = {}
    for unit_id in CALIBRATION_UNITS:
        st = spike_times_all[spike_clusters_all == unit_id]
        hist, edges, total_pairs = compute_acg_hist(st, cfg.fs, max_lag_ms=30.0, bin_ms=0.5)
        acg_hists[unit_id] = hist
    cross_unit_info = cross_unit_acg_similarity(acg_hists, peak_channels, geometry)
    for uid, info in cross_unit_info.items():
        print(f"  unit {uid}: best match = unit {info['best_match_unit']}, "
              f"r={info['correlation']:.3f}, distance={info['distance_um']:.0f}um")

    results = []
    for unit_id in CALIBRATION_UNITS:
        print(f"\n--- unit {unit_id} ---")
        st = spike_times_all[spike_clusters_all == unit_id]

        profile = profile_unit(st, cfg.fs, session_duration_s, cfg)
        chans, peak_chan = footprint_channels(templates, unit_id, cfg.footprint_n_channels, geometry, cfg.footprint_channel_radius_um)
        neighbor_ids = find_spatial_neighbors(unit_id, templates, geometry, cfg.collision_neighbor_radius_um, peak_channels)
        neighbor_spike_times = {nb: spike_times_all[spike_clusters_all == nb] for nb in neighbor_ids[:5]}

        structural = structural_score_for_unit(unit_id, st, templates, geometry, reader, templates.shape[1],
                                                nt0min, cfg.fs, session_duration_samples, neighbor_spike_times, cfg, rng,
                                                cross_unit_acg_info=cross_unit_info.get(unit_id))

        acg_hist, acg_edges, acg_total_pairs = compute_acg_hist(st, cfg.fs, max_lag_ms=30.0, bin_ms=0.5)
        v_ratio, refr_count, expected = violation_ratio(st, cfg.fs, cfg)

        is_tier3, tier3_reason = unit_tier3_check(profile, structural, cfg)

        scan_result = compute_candidate_scores(unit_id, st, templates, geometry, reader, cfg.fs,
                                                session_duration_samples, cfg, rng, nt0min)
        sweep_result = sweep_and_select(unit_id, st, scan_result, cfg.fs, cfg)

        recovered = sweep_result["recovered_times"]
        union_times = np.sort(np.concatenate([st, recovered])) if len(recovered) else st
        acg_after, _, _ = violation_ratio(union_times, cfg.fs, cfg)
        tier, degradation = recovery_tier(structural["structural_score"], v_ratio, acg_after, cfg,
                                           inflection_found=sweep_result.get("inflection_found", False))

        collision_summary = classify_unit_spikes(unit_id, st, chans, templates, geometry, reader,
                                                   peak_channels, templates.shape[1], nt0min, cfg, rng=rng)

        top4 = chans[np.argsort(-(templates[unit_id][:, chans].max(axis=0) - templates[unit_id][:, chans].min(axis=0)))[:4]]
        wave = sample_waveform(reader, st, top4, templates.shape[1], nt0min, cfg.fs, cfg.highpass_hz, rng=rng)
        wave_unfiltered = sample_waveform_unfiltered_peak(reader, st, peak_chan, templates.shape[1], nt0min, rng=rng)

        # KS's own template (averaged over far more spikes, spatially whitened) -- the
        # most reliable shape estimate available, used as ground truth for the
        # peak-trough width/amplitude check (see conversation: raw-snippet averaging on
        # a single channel is too low-SNR to trust for this on its own)
        ks_templ = templates[unit_id][:, top4]
        peak_ch_templ = templates[unit_id][:, peak_chan]
        peak_idx, trough_idx = int(np.argmax(peak_ch_templ)), int(np.argmin(peak_ch_templ))
        ks_peak_trough_width_ms = abs(trough_idx - peak_idx) / cfg.fs * 1000
        ks_template_amplitude = float(peak_ch_templ.max() - peak_ch_templ.min())
        ks_trough_before_peak = trough_idx < peak_idx

        print(f"  n_spikes={profile['n_spikes']}, structural_score={structural['structural_score']:.3f}, "
              f"acg_ratio={v_ratio:.3f}, tier3={is_tier3}, tier={tier if not is_tier3 else 'tier3'}, "
              f"pct_added={sweep_result['pct_spikes_added']:.2f}%")

        results.append(dict(
            unit_id=int(unit_id),
            ks_contam_pct=contam.get(unit_id), ks_amplitude=amp_tsv.get(unit_id), ks_label=kslabel.get(unit_id),
            n_spikes=profile["n_spikes"], mean_rate_hz=profile["mean_rate_hz"],
            cv=profile["cv"], lv=profile["lv"], burst_fraction=profile["burst_fraction"],
            pause_time_fraction=profile["pause_time_fraction"], n_bursts=profile["n_bursts"],
            structural_score=structural["structural_score"], components=structural["components"],
            footprint_concentration_ratio=structural["footprint_concentration_ratio"],
            cross_unit_acg_info=structural["cross_unit_acg_info"],
            likely_shared_artifact=structural["likely_shared_artifact"],
            peak_channel=peak_chan, footprint_channels=chans.tolist(),
            acg_bin_centers_ms=((acg_edges[:-1] + acg_edges[1:]) / 2).tolist(), acg_counts=acg_hist.tolist(),
            acg_violation_ratio=v_ratio,
            is_tier3=is_tier3, tier3_reason=tier3_reason,
            tier=("tier3_insufficient_evidence" if is_tier3 else tier),
            acg_violation_before=v_ratio, acg_violation_after=acg_after,
            n_recovered=sweep_result["n_recovered"], pct_spikes_added=sweep_result["pct_spikes_added"],
            sweep_curve=sweep_result["sweep"], chosen_cutoff=sweep_result["chosen_cutoff"],
            inflection_found=sweep_result.get("inflection_found", False),
            n_candidates_before_floor=scan_result.get("n_candidates_before_floor"),
            n_candidates_after_floor=scan_result.get("n_candidates_after_floor"),
            collision_counts=collision_summary["counts"], collision_n_sampled=collision_summary["n_sampled"],
            waveform_mean=wave["mean"].tolist() if wave else None,
            waveform_std=wave["std"].tolist() if wave else None,
            waveform_channels=top4.tolist(),
            waveform_unfiltered_peak=wave_unfiltered,
            ks_template=ks_templ.tolist(),
            ks_peak_trough_width_ms=ks_peak_trough_width_ms,
            ks_template_amplitude=ks_template_amplitude,
            ks_trough_before_peak=ks_trough_before_peak,
        ))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved calibration diagnostics to {OUT_JSON}")


if __name__ == "__main__":
    main()
