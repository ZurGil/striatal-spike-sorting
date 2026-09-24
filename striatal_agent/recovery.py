"""
Section 2.7 -- raw-data recovery search.

Primary method (implemented): reuses KS's own learned template (templates.npy) as a
matched filter against the filtered raw trace -- same principle as the spec's "rerun
deconvolution with Th_learned swept downward", but as a *simplified single-template
correlation* rather than a literal call into Kilosort4's internal multi-unit peeling
deconvolution. This is the same method validated in the Section 1 gate (8+8+8 units,
consistent 0.6-7% recovery yield, well-calibrated against known-bad-ACG units).

Whether KS4's own deconvolution step can be re-run standalone with an adjusted
Th_learned (kickoff prompt's open question 3) is still unresolved -- that would be the
natural upgrade path for this module if confirmed feasible; interface here
(sweep_and_select's inputs/outputs) is written so that swap would not require changing
callers.

Required control (implemented): ACG-inflection stopping rule, not an arbitrary fixed
threshold -- rerun automatically any time the candidate cutoff changes, since it's cheap
once candidate scores are computed (compute once, sweep the cutoff for free).

Secondary method (Section 2.7, ISI-indexed pass for burst-capable units): stubbed here,
implemented in burst_module.py once a unit's profile (firing_pattern.py) marks it
burst-capable -- kept separate per spec's framing as a targeted supplement, not a
parallel general system.
"""
import numpy as np
from scipy import signal

from .config import AgentConfig
from .raw_io import RawReader, footprint_channels, highpass_filter
from .firing_pattern import detect_bursts
from .acg_tools import compute_acg_hist, violation_ratio, find_inflection_threshold


def _sample_scan_windows(spike_times, session_n_samples, fs, cfg: AgentConfig, rng):
    pad = int(cfg.acg_refractory_ms * 5 / 1000.0 * fs)  # generous padding for filter edge effects
    bursts = detect_bursts(spike_times, fs, cfg)
    if len(bursts) > cfg.max_bursts_per_unit_io_cap:
        idx = rng.choice(len(bursts), size=cfg.max_bursts_per_unit_io_cap, replace=False)
        bursts = [bursts[i] for i in sorted(idx)]
    windows = [(b["start_t"] - pad, b["end_t"] + pad) for b in bursts]

    if not cfg.recovery_burst_only and cfg.recovery_n_random_windows > 0:
        win_len = int(cfg.recovery_random_window_s * fs)
        starts = rng.integers(0, max(1, session_n_samples - win_len), size=cfg.recovery_n_random_windows)
        windows += [(int(s), int(s) + win_len) for s in starts]

    return windows, bursts


def compute_candidate_scores(unit_id, spike_times, templates, geometry, reader: RawReader,
                              fs, session_n_samples, cfg: AgentConfig = None, rng=None, nt0min=20):
    cfg = cfg or AgentConfig()
    rng = rng or np.random.default_rng(0)
    spike_times = np.sort(spike_times)

    chans, peak_chan = footprint_channels(templates, unit_id, cfg.footprint_n_channels,
                                           geometry, cfg.footprint_channel_radius_um)
    templ = templates[unit_id][:, chans]
    templ_norm = templ / (np.linalg.norm(templ) + 1e-9)
    nt = templ.shape[0]

    windows, bursts = _sample_scan_windows(spike_times, session_n_samples, fs, cfg, rng)
    excl = int(0.5 / 1000.0 * fs)
    refr = int(cfg.acg_refractory_ms / 1000.0 * fs)

    accepted_scores, candidate_scores, candidate_times = [], [], []
    n_windows_used = 0
    for w_start, w_end in windows:
        raw, actual_start, actual_end = reader.read_window(w_start, w_end, chans)
        if raw.shape[0] < nt + 10:
            continue
        filt = highpass_filter(raw, fs, cfg.highpass_hz)

        score = np.zeros(filt.shape[0] - nt + 1, dtype=np.float32)
        for c in range(filt.shape[1]):
            score += np.correlate(filt[:, c], templ_norm[:, c], mode="valid")

        ks_in_window = spike_times[(spike_times >= w_start) & (spike_times < w_end)]
        ks_local_onsets = (ks_in_window - actual_start - nt0min).astype(int)

        matched_mask = np.zeros(len(score), dtype=bool)
        for onset in ks_local_onsets:
            lo, hi = max(0, onset - excl), min(len(score), onset + excl + 1)
            if hi <= lo:
                continue
            local_max_idx = lo + np.argmax(score[lo:hi])
            accepted_scores.append(float(score[local_max_idx]))
            matched_mask[max(0, local_max_idx - excl):local_max_idx + excl + 1] = True

        peaks, _ = signal.find_peaks(score, distance=refr)
        for p in peaks:
            if matched_mask[p]:
                continue
            candidate_scores.append(float(score[p]))
            candidate_times.append(int(actual_start + p + nt0min))
        n_windows_used += 1

    accepted_scores = np.array(accepted_scores)
    candidate_scores = np.array(candidate_scores)
    candidate_times = np.array(candidate_times, dtype=np.int64)

    # noise floor: drop candidates that are obviously not spike-like before they ever
    # reach the sweep (calibration finding -- see acg_tools.find_inflection_threshold)
    n_candidates_before_floor = len(candidate_scores)
    if len(accepted_scores) >= 5 and len(candidate_scores) > 0:
        floor = cfg.recovery_min_candidate_score_frac * np.mean(accepted_scores)
        keep = candidate_scores >= floor
        candidate_scores, candidate_times = candidate_scores[keep], candidate_times[keep]

    return dict(
        unit_id=int(unit_id), chans=chans, peak_channel=peak_chan, nt=nt, nt0min=nt0min,
        accepted_scores=accepted_scores, candidate_scores=candidate_scores,
        candidate_times=candidate_times,
        n_windows_scanned=n_windows_used, n_bursts_found=len(bursts),
        n_candidates_before_floor=n_candidates_before_floor, n_candidates_after_floor=len(candidate_scores),
    )


def sweep_and_select(unit_id, spike_times, scan_result, fs, cfg: AgentConfig = None):
    """ACG-inflection sweep over the accepted-score percentile grid (spec Section 2.7's
    'required control'). Returns the sweep curve plus the final recovered spike set."""
    cfg = cfg or AgentConfig()
    accepted = scan_result["accepted_scores"]
    cand_t = scan_result["candidate_times"]
    cand_s = scan_result["candidate_scores"]
    base_times = np.sort(np.asarray(spike_times))

    if len(accepted) < 10 or len(cand_t) == 0:
        return dict(unit_id=int(unit_id), sweep=[], chosen_cutoff=None, inflection_found=False,
                     recovered_times=np.array([], dtype=np.int64), recovered_scores=np.array([]),
                     n_recovered=0, pct_spikes_added=0.0)

    cutoffs = [np.percentile(accepted, p) for p in cfg.recovery_sweep_percentiles]
    order = np.argsort(cand_t)
    cand_t_sorted, cand_s_sorted = cand_t[order], cand_s[order]

    sweep = []
    for cutoff in cutoffs:
        included = cand_t_sorted[cand_s_sorted >= cutoff]
        union = np.sort(np.concatenate([base_times, included]))
        ratio, refr_count, expected = violation_ratio(union, fs, cfg)
        sweep.append(dict(cutoff=float(cutoff), n_included=int(len(included)), acg_violation_ratio=ratio))

    thresholds = np.array([s["cutoff"] for s in sweep])          # descending (strict->permissive)
    viol = np.array([s["acg_violation_ratio"] for s in sweep])
    chosen_cutoff, inflection_found = find_inflection_threshold(thresholds, viol)

    final_mask = cand_s_sorted >= chosen_cutoff
    recovered_times = cand_t_sorted[final_mask]
    recovered_scores = cand_s_sorted[final_mask]

    # final refractory safety check against the base (KS-confirmed) train
    if len(recovered_times) > 0:
        refr = int(cfg.acg_refractory_ms / 1000.0 * fs)
        nearest = base_times[np.clip(np.searchsorted(base_times, recovered_times), 0, len(base_times) - 1)]
        clean_mask = np.abs(nearest - recovered_times) >= refr
        recovered_times = recovered_times[clean_mask]
        recovered_scores = recovered_scores[clean_mask]

    return dict(unit_id=int(unit_id), sweep=sweep, chosen_cutoff=float(chosen_cutoff),
                inflection_found=inflection_found,
                recovered_times=recovered_times, recovered_scores=recovered_scores,
                n_recovered=int(len(recovered_times)),
                pct_spikes_added=float(100.0 * len(recovered_times) / max(len(base_times), 1)))
