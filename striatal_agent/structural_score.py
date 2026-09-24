"""
Section 2.3 -- per-unit structural scoring. Soft score (not pass/fail): combines
spatial footprint consistency, waveform shape similarity, CCG evidence, and refractory
contamination into one weighted score per unit. Real drift/amplitude fluctuation can look
like a channel swap without being a real merge problem, hence a combined soft score
rather than a single hard filter (this is v2's explicit revision of v1's binary Stage 3).
"""
import numpy as np

from .config import AgentConfig
from .raw_io import RawReader, footprint_channels, ChannelGeometry, read_filtered_snippet
from .footprint_stability import footprint_stability_score, _mean_snippet_ptp_per_channel
from .acg_tools import violation_ratio, refractory_violation_rate


def waveform_shape_consistency(reader: RawReader, spike_times, chans, nt, nt0min, fs,
                                cutoff, n_sample=150, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(spike_times) < 5:
        return dict(score=np.nan, n_sampled=0)
    idx = rng.choice(len(spike_times), size=min(n_sample, len(spike_times)), replace=False)
    snippets = []
    for t in spike_times[idx]:
        start = int(t) - nt0min
        block = read_filtered_snippet(reader, chans, start, nt, fs, cutoff)
        if block is not None:
            snippets.append(block.ravel())
    if len(snippets) < 5:
        return dict(score=np.nan, n_sampled=len(snippets))
    snippets = np.array(snippets)
    mean_wave = snippets.mean(axis=0)
    mean_wave_norm = mean_wave / (np.linalg.norm(mean_wave) + 1e-9)
    sims = snippets @ mean_wave_norm / (np.linalg.norm(snippets, axis=1) + 1e-9)
    return dict(score=float(np.clip(np.mean(sims), 0, 1)), n_sampled=len(snippets))


def peak_trough_width_ms(templates, unit_id, peak_chan, fs):
    """Peak-to-trough width on the unit's own peak channel. Calibrated against the full
    401-unit population (see conversation): width >=0.55-0.60ms is genuinely rare
    (~5-10% of units), unlike the cross-unit ACG best-match check below, which turned
    out to flag ~58-72% of units in a null (unrelated-pairs) sample -- a multiple-
    comparisons artifact of searching for each unit's best match among hundreds of
    candidates, not a real anomaly signal. Width, by contrast, held up under the same
    scrutiny and is what actually separated 187/156/151 (0.60-0.63ms, clearly atypical)
    from a random comparison batch that visually looked like real neurons (0.33-0.47ms,
    solidly within the ordinary range)."""
    tr = templates[unit_id][:, peak_chan]
    peak_idx, trough_idx = int(np.argmax(tr)), int(np.argmin(tr))
    return abs(trough_idx - peak_idx) / fs * 1000.0


def footprint_concentration_ratio(templates, unit_id, chans):
    """Spatial-cliff check: ratio of the 2nd-strongest footprint channel's amplitude to
    the peak channel's. A real dipole source decays smoothly over a few tens of microns
    -- it doesn't vanish between adjacent contacts. Caught in this session by eye + this
    ratio: units 187 and 156 had ratios of 0.15 and 0.07 (essentially isolated to one
    channel) vs. 0.70-1.00 for every genuine-looking unit in the same comparison batch.
    Near 1 = smooth spread (good); near 0 = single-channel-isolated (suspicious)."""
    templ = templates[unit_id][:, chans]
    ptp = templ.max(axis=0) - templ.min(axis=0)
    sorted_ptp = np.sort(ptp)[::-1]
    if len(sorted_ptp) < 2 or sorted_ptp[0] <= 0:
        return 0.0
    return float(sorted_ptp[1] / sorted_ptp[0])


def footprint_flatness_ratio(templates, unit_id, chans):
    """Opposite failure mode from concentration_ratio above: instead of decaying too
    fast (isolated to one channel), a footprint can decay too slowly -- amplitude still
    nearly full-strength many channels away, consistent with broad common-mode/reference
    noise rather than a localized dipole source. Ratio of the average of the 4th- and
    5th-strongest footprint channels' amplitude to the peak channel's (needs the 2nd/3rd
    strongest to still be informative, so this deliberately looks one ring further out
    than concentration_ratio). Calibrated against the 326-unit manually-verdicted ground
    truth (see conversation): >=0.75 hits 4.0% of all 401 units, 11/12 (92%) of which are
    manually-labeled NOISE and none GOOD -- comparable rarity and precision to
    concentration_ratio<0.2's calibration. Near 0 = normal decay; near 1 = suspiciously
    flat/broad."""
    templ = templates[unit_id][:, chans]
    ptp = templ.max(axis=0) - templ.min(axis=0)
    sorted_ptp = np.sort(ptp)[::-1]
    if len(sorted_ptp) < 5 or sorted_ptp[0] <= 0:
        return 0.0
    return float((sorted_ptp[3] + sorted_ptp[4]) / 2.0 / sorted_ptp[0])


def zero_lag_synchrony_excess(times_a, times_b, fs, window_ms=1.0):
    """Cheap CCG cross-check: fraction of unit B's spikes within +-window_ms of a unit-A
    spike, vs. the chance level under B's own mean rate. A large excess flags possible
    over-splitting (two templates catching the same true spikes) -- feeds a *penalty*,
    not a merge decision (that's Section 2.8's job)."""
    if len(times_a) == 0 or len(times_b) == 0:
        return 0.0
    ta = np.sort(times_a)
    tb = np.sort(times_b)
    w = window_ms / 1000.0 * fs
    lo = np.searchsorted(tb, ta - w)
    hi = np.searchsorted(tb, ta + w)
    n_coincident = int((hi - lo).sum())
    duration_s = (max(ta[-1], tb[-1]) - min(ta[0], tb[0])) / fs
    rate_b = len(tb) / duration_s if duration_s > 0 else 0.0
    expected = len(ta) * rate_b * (2 * window_ms / 1000.0)
    return float(n_coincident / (expected + 1e-9))


def structural_score_for_unit(unit_id, spike_times, templates, geometry: ChannelGeometry,
                               reader: RawReader, nt, nt0min, fs, session_duration_samples,
                               neighbor_spike_times=None, cfg: AgentConfig = None, rng=None,
                               cross_unit_acg_info=None):
    """neighbor_spike_times: optional dict {neighbor_unit_id: spike_times_samples} for
    units sharing footprint channels, used for the CCG cross-check term.
    cross_unit_acg_info: optional dict(best_match_unit, correlation, distance_um) from
    acg_tools.cross_unit_acg_similarity -- a session-wide batch computation, so it's
    passed in rather than computed here. None (default) means the check is skipped
    (neutral score), not silently treated as "passed"."""
    cfg = cfg or AgentConfig()
    rng = rng or np.random.default_rng(0)

    chans, peak_chan = footprint_channels(templates, unit_id, cfg.footprint_n_channels,
                                           geometry, cfg.footprint_channel_radius_um)

    footprint = footprint_stability_score(reader, spike_times, chans, nt, nt0min,
                                           session_duration_samples, cfg, rng)
    waveform = waveform_shape_consistency(reader, spike_times, chans, nt, nt0min, fs,
                                           cfg.highpass_hz, rng=rng)
    v_ratio, refr_count, expected = violation_ratio(spike_times, fs, cfg)
    refr_rate = refractory_violation_rate(spike_times, fs, cfg.acg_refractory_ms)
    concentration_ratio = footprint_concentration_ratio(templates, unit_id, chans)
    flatness_ratio = footprint_flatness_ratio(templates, unit_id, chans)

    ccg_penalty = 0.0
    if neighbor_spike_times:
        excesses = [zero_lag_synchrony_excess(spike_times, nb_times, fs)
                    for nb_times in neighbor_spike_times.values()]
        ccg_penalty = float(np.clip(np.max(excesses) - 1.0, 0, None)) if excesses else 0.0
    ccg_score = float(np.clip(1.0 - v_ratio, 0, 1) * np.clip(1.0 - ccg_penalty, 0, 1))

    refractory_score = float(np.clip(1.0 - refr_rate / 0.05, 0, 1))  # 5%+ violations -> score 0
    width_ms = peak_trough_width_ms(templates, unit_id, peak_chan, fs)

    # cross_unit_acg_info is reported for human context (it's what first caught
    # 187/156/151) but NOT used as an automatic trigger: calibrated against a null
    # sample of unrelated unit pairs and found to flag 58-72% of units by chance alone
    # (searching for each unit's single best match among hundreds of candidates is a
    # multiple-comparisons problem -- see conversation). cross_unit_score below is a
    # mild, capped soft penalty only, never a hard flag on its own.
    cross_unit_score = 1.0
    if cross_unit_acg_info is not None and cross_unit_acg_info.get("correlation") is not None:
        corr = cross_unit_acg_info["correlation"]
        cross_unit_score = float(np.clip(1.0 - max(0.0, corr - 0.5) / 0.4, 0, 1))

    # likely_shared_artifact: hard flag based on the two signals that survived
    # null-distribution calibration against all 401 units -- footprint_concentration
    # <0.2 and peak-trough width >=0.55ms are each independently rare (~5-10% of
    # units), unlike cross-unit ACG correlation. Catches 187/156 via footprint
    # concentration (0.07-0.15) and 151 via width alone (0.60ms, normal footprint) --
    # matching what visual inspection + this calibration converged on.
    likely_shared_artifact = (concentration_ratio < cfg.footprint_concentration_flag_threshold or
                               flatness_ratio >= cfg.footprint_flatness_flag_threshold or
                               width_ms >= cfg.width_outlier_threshold_ms)

    components = dict(
        footprint=footprint["score"] if not np.isnan(footprint.get("score", np.nan)) else 0.5,
        waveform_shape=waveform["score"] if not np.isnan(waveform.get("score", np.nan)) else 0.5,
        ccg=ccg_score,
        refractory=refractory_score,
        footprint_concentration=concentration_ratio,
        cross_unit_acg=cross_unit_score,
    )
    w = cfg.structural_score_weights
    total = sum(w[k] * components[k] for k in w) / sum(w.values())
    if likely_shared_artifact:
        total = min(total, 0.15)  # explicit ceiling, not just a weighted-average dilution

    return dict(
        unit_id=int(unit_id), peak_channel=peak_chan, footprint_channels=chans.tolist(),
        structural_score=float(total), components=components,
        footprint_stability_detail=footprint, waveform_shape_detail=waveform,
        acg_violation_ratio=v_ratio, refractory_violation_rate=refr_rate,
        ccg_synchrony_penalty=ccg_penalty, footprint_concentration_ratio=concentration_ratio,
        footprint_flatness_ratio=flatness_ratio,
        peak_trough_width_ms=width_ms,
        cross_unit_acg_info=cross_unit_acg_info, likely_shared_artifact=likely_shared_artifact,
    )
