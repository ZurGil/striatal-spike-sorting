"""
Section 2.8 -- merge/split scoring. Combines structural score (2.3) + CCG evidence +
amplitude/footprint relationship into one merge-confidence score per candidate pair.
High-confidence, unambiguous merges may be auto-applied (Section 6 tier 1); ambiguous
cases are scored and logged, never silently resolved (spec's repeated anti-silent-
resolution principle).

Candidate pairs come from KS's own similar_templates.npy (already-computed template
similarity ranking) intersected with spatial adjacency -- cheap, and avoids an O(n^2)
scan over all unit pairs.
"""
import numpy as np

from .config import AgentConfig
from .structural_score import zero_lag_synchrony_excess


def candidate_pairs_from_similarity(similar_templates, peak_channels, geometry, radius_um,
                                     top_k=5, min_similarity=0.5):
    """similar_templates: KS4's similar_templates.npy, (n_templates, n_templates) similarity matrix."""
    pairs = []
    n = similar_templates.shape[0]
    seen = set()
    for u in range(n):
        if u not in peak_channels:
            continue
        row = similar_templates[u].copy()
        row[u] = -np.inf
        top = np.argsort(row)[::-1][:top_k]
        for v in top:
            v = int(v)
            if v not in peak_channels or row[v] < min_similarity:
                continue
            if geometry.distance(peak_channels[u], peak_channels[v]) > radius_um:
                continue
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(dict(unit_a=key[0], unit_b=key[1], ks_similarity=float(similar_templates[key[0], key[1]])))
    return pairs


def footprint_similarity(templates, unit_a, unit_b, chans):
    ta = templates[unit_a][:, chans]
    tb = templates[unit_b][:, chans]
    ptp_a = ta.max(axis=0) - ta.min(axis=0)
    ptp_b = tb.max(axis=0) - tb.min(axis=0)
    na, nb = np.linalg.norm(ptp_a), np.linalg.norm(ptp_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(ptp_a, ptp_b) / (na * nb))


def merge_score(unit_a, unit_b, spike_times_a, spike_times_b, structural_score_a,
                 structural_score_b, templates, chans, fs, cfg: AgentConfig = None):
    cfg = cfg or AgentConfig()
    struct_component = float(np.mean([structural_score_a, structural_score_b]))
    synchrony = zero_lag_synchrony_excess(spike_times_a, spike_times_b, fs)
    ccg_component = float(np.clip(synchrony / 3.0, 0, 1))  # excess>=3x chance -> full score; heuristic, needs calibration
    amplitude_component = footprint_similarity(templates, unit_a, unit_b, chans)

    w = cfg.merge_score_weights
    total = (w["structural"] * struct_component + w["ccg"] * ccg_component +
             w["amplitude"] * amplitude_component) / sum(w.values())

    return dict(unit_a=int(unit_a), unit_b=int(unit_b), merge_score=float(total),
                structural_component=struct_component, ccg_component=ccg_component,
                zero_lag_synchrony_excess=synchrony, amplitude_component=amplitude_component)
