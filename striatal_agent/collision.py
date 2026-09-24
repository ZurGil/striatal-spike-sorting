"""
Section 2.5 -- three-way collision/contamination check. For a spike assigned to unit U,
decide whether the raw snippet is:
  (a) clean            -- fully explained by U's own template
  (b) known collision   -- residual after subtracting U's template is better explained by
                            a spatially-neighboring, already-identified unit's template
  (c) unexplained residual -- real residual signal, not attributable to any known unit
                            (spec's acknowledged gap: likely an undetected low-amplitude
                            neuron, common in dense striatal recordings -- must not be
                            misattributed to a known unit, and must not be called clean)

This can only ever identify collisions with units KS already found (spec Section 5's
limitation, carried over unchanged) -- (c) exists specifically so that gap isn't silently
folded into "clean".
"""
import numpy as np

from .config import AgentConfig
from .raw_io import RawReader, ChannelGeometry, read_filtered_snippet


def find_spatial_neighbors(unit_id, templates, geometry: ChannelGeometry, radius_um, peak_channels):
    """peak_channels: dict {unit_id: peak_channel}, precomputed once for all units."""
    peak = peak_channels[unit_id]
    near_chans = set(geometry.neighbors_within(peak, radius_um).tolist())
    return [u for u, pc in peak_channels.items() if u != unit_id and pc in near_chans]


def _best_fit_scale(snippet, template):
    num = np.sum(snippet * template)
    denom = np.sum(template * template) + 1e-9
    a = num / denom
    residual = snippet - a * template
    return a, residual


def classify_spike(snippet, assigned_template, neighbor_templates, cfg: AgentConfig = None):
    """snippet, assigned_template, and each entry in neighbor_templates must all be on the
    SAME channel set (typically the assigned unit's footprint channels)."""
    cfg = cfg or AgentConfig()
    a, residual = _best_fit_scale(snippet, assigned_template)
    snippet_energy = float(np.sum(snippet ** 2)) + 1e-9
    residual_energy_ratio = float(np.sum(residual ** 2) / snippet_energy)

    clean_threshold = 0.15
    if residual_energy_ratio < clean_threshold:
        return dict(label="clean", residual_energy_ratio=residual_energy_ratio,
                    assigned_scale=float(a), best_neighbor=None, neighbor_explained_frac=0.0)

    best_neighbor, best_frac = None, 0.0
    for nb_id, nb_template in neighbor_templates.items():
        _, nb_residual = _best_fit_scale(residual, nb_template)
        explained_frac = 1.0 - float(np.sum(nb_residual ** 2) / (np.sum(residual ** 2) + 1e-9))
        if explained_frac > best_frac:
            best_frac, best_neighbor = explained_frac, nb_id

    if best_neighbor is not None and best_frac >= cfg.collision_score_margin + 0.25:
        label = f"collision_with_unit_{best_neighbor}"
    else:
        label = "unexplained_residual"

    return dict(label=label, residual_energy_ratio=residual_energy_ratio,
                assigned_scale=float(a), best_neighbor=best_neighbor,
                neighbor_explained_frac=float(best_frac))


def classify_unit_spikes(unit_id, spike_times, chans, templates, geometry, reader: RawReader,
                          peak_channels, nt, nt0min, cfg: AgentConfig = None, n_sample=300, rng=None):
    """Batch three-way classification for a sample of a unit's spikes -- used both as a
    general QC pass and to sanity-check recovery candidates before they're trusted."""
    cfg = cfg or AgentConfig()
    rng = rng or np.random.default_rng(0)
    spike_times = np.asarray(spike_times)
    if len(spike_times) == 0:
        return dict(unit_id=int(unit_id), n_sampled=0, counts={})

    idx = rng.choice(len(spike_times), size=min(n_sample, len(spike_times)), replace=False)
    neighbor_ids = find_spatial_neighbors(unit_id, templates, geometry, cfg.collision_neighbor_radius_um, peak_channels)
    assigned_template = templates[unit_id][:, chans]
    neighbor_templates = {nb: templates[nb][:, chans] for nb in neighbor_ids}

    counts = {}
    detail = []
    for t in spike_times[idx]:
        start = int(t) - nt0min
        snippet = read_filtered_snippet(reader, chans, start, nt, cfg.fs, cfg.highpass_hz)
        if snippet is None:
            continue
        result = classify_spike(snippet, assigned_template, neighbor_templates, cfg)
        counts[result["label"].split("_with_unit_")[0]] = counts.get(result["label"].split("_with_unit_")[0], 0) + 1
        detail.append(dict(time=int(t), **result))

    return dict(unit_id=int(unit_id), n_sampled=len(detail), counts=counts,
                neighbor_ids=neighbor_ids, detail=detail)
