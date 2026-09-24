"""
Section 4 -- burst/pause sub-module. Context-aware, triggered by firing_pattern.py's
continuous profile -- not the pipeline's core purpose (spec Section 0), so this module
only ever *supplements* recovery.py's general-purpose sweep for units whose profile
calls for it, and only ever narrows TAN-like units' accepted set, never widens it
carelessly near a pause.

Preceding-ISI (not rank-in-burst) is the primary predictor, per spec's revision from v1:
physiology is a time process, not a counting process.
"""
import numpy as np

from .config import AgentConfig
from .raw_io import RawReader, highpass_filter
from .acg_tools import violation_ratio


def classify_unit_class(profile, cfg: AgentConfig = None):
    """Soft dispatch label for *which handling branch* to use -- not a hard filter
    anywhere else in the pipeline (structural_score, recovery's primary sweep, and the
    audit all use the continuous profile directly)."""
    if profile["n_spikes"] < (cfg or AgentConfig()).min_spikes_for_profiling:
        return "unclassified_low_confidence"
    if profile["pause_time_fraction"] > 0.05 and profile["mean_rate_hz"] < 5:
        return "tan_like"
    if profile["burst_fraction"] > 0.15:
        return "bursty_msn_like"
    return "tonic_fsi_like"


def preceding_isi_ms(spike_time, reference_times_sorted, fs):
    idx = np.searchsorted(reference_times_sorted, spike_time)
    if idx == 0:
        return np.inf
    return float((spike_time - reference_times_sorted[idx - 1]) / fs * 1000.0)


def build_isi_indexed_template(reader: RawReader, clean_spike_times, chans, nt, nt0min, fs,
                                cfg: AgentConfig = None, rng=None):
    """Position/ISI-indexed expected waveform: average shape per preceding-ISI bin, built
    only from clean, already-validated (KS-confirmed) spikes -- never from candidates
    (anti-circularity, spec Section 4/0)."""
    cfg = cfg or AgentConfig()
    rng = rng or np.random.default_rng(0)
    st = np.sort(clean_spike_times)
    edges = cfg.isi_bin_edges_ms

    bin_templates = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_spike_idx = []
        for i in range(1, len(st)):
            pisi = (st[i] - st[i - 1]) / fs * 1000.0
            if lo <= pisi < hi:
                bin_spike_idx.append(i)
        if len(bin_spike_idx) < 5:
            bin_templates[(lo, hi)] = None
            continue
        sample_idx = rng.choice(bin_spike_idx, size=min(100, len(bin_spike_idx)), replace=False)
        snippets = []
        for i in sample_idx:
            t = st[i]
            start, end = int(t) - nt0min, int(t) - nt0min + nt
            block, a, b = reader.read_window(start, end, chans)
            if block.shape[0] == nt:
                snippets.append(block)
        if snippets:
            bin_templates[(lo, hi)] = np.mean(snippets, axis=0)
        else:
            bin_templates[(lo, hi)] = None

    return dict(edges=edges, bin_templates=bin_templates)


def _template_for_isi(isi_indexed, pisi_ms):
    edges = isi_indexed["edges"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= pisi_ms < hi:
            t = isi_indexed["bin_templates"].get((lo, hi))
            if t is not None:
                return t
    # fall back to the nearest non-empty bin
    for (lo, hi), t in isi_indexed["bin_templates"].items():
        if t is not None:
            return t
    return None


def isi_indexed_supplement(unit_id, base_recovered_union, rejected_candidate_times,
                            rejected_candidate_scan, isi_indexed_template, fs,
                            shape_match_min=0.5, cfg: AgentConfig = None):
    """Secondary method (2.7): re-score candidates the primary KS-native sweep rejected,
    against the ISI-indexed expected shape *at their own preceding-ISI position* instead
    of the fixed average template. Targeted supplement for burst-capable units only --
    catches genuinely decremented late-burst spikes the fixed-template sweep is blind to.
    """
    cfg = cfg or AgentConfig()
    reference = np.sort(base_recovered_union)
    recovered = []
    for t in rejected_candidate_times:
        pisi = preceding_isi_ms(t, reference, fs)
        expected_wave = _template_for_isi(isi_indexed_template, pisi)
        if expected_wave is None:
            continue
        # candidate's own snippet is looked up from the same raw scan cache the primary
        # sweep already read (rejected_candidate_scan carries {time: snippet}) to avoid
        # a second raw-data pass
        snippet = rejected_candidate_scan.get(int(t))
        if snippet is None:
            continue
        num = np.sum(snippet * expected_wave)
        denom = np.linalg.norm(snippet) * np.linalg.norm(expected_wave) + 1e-9
        shape_match = float(num / denom)
        if shape_match >= shape_match_min:
            recovered.append(dict(time=int(t), preceding_isi_ms=pisi, shape_match=shape_match))
    return recovered


def apply_pause_conservatism(recovered_times, pauses, margin_ms, fs):
    """TAN-like handling: recovered spikes landing inside a detected pause are more
    likely noise than a real miss -- drop them rather than trust the raw sweep."""
    if not pauses or len(recovered_times) == 0:
        return recovered_times, np.array([], dtype=bool)
    margin = margin_ms / 1000.0 * fs
    drop_mask = np.zeros(len(recovered_times), dtype=bool)
    for p in pauses:
        in_pause = (recovered_times >= p["start_t"] - margin) & (recovered_times <= p["end_t"] + margin)
        drop_mask |= in_pause
    return recovered_times[~drop_mask], drop_mask
