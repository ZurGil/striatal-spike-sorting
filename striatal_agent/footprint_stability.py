"""
Section 2.6 -- footprint stability module. Standalone, always-on QC metric (not gated
on burst-recovery use). Checks that the *normalized shape* of the per-channel amplitude
pattern stays stable across the session, even if absolute amplitude drifts. Shape
instability (not just scale change) is direct evidence against single-source identity.

Needs real raw snippets (not just KS's per-spike scalar amplitude, which is a uniform
template scaling factor and therefore shape-invariant by construction) -- so this module
samples a modest number of raw waveforms per session segment, matching the I/O-light
approach validated in the Section 1 gate.
"""
import numpy as np

from .config import AgentConfig
from .raw_io import RawReader, read_filtered_snippet


def _mean_snippet_ptp_per_channel(reader: RawReader, spike_times, chans, nt, nt0min, fs, cutoff,
                                   n_sample=100, rng=None):
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
    mean_wave = np.mean(snippets, axis=0)  # (nt, k)
    ptp = mean_wave.max(axis=0) - mean_wave.min(axis=0)
    return ptp


def footprint_stability_score(reader: RawReader, spike_times_samples, chans, nt, nt0min,
                               session_duration_samples, cfg: AgentConfig = None, rng=None):
    cfg = cfg or AgentConfig()
    rng = rng or np.random.default_rng(0)
    st = np.sort(spike_times_samples)
    n_seg = cfg.footprint_stability_n_segments
    edges = np.linspace(0, session_duration_samples, n_seg + 1)

    shapes = []
    for i in range(n_seg):
        seg_mask = (st >= edges[i]) & (st < edges[i + 1])
        seg_times = st[seg_mask]
        ptp = _mean_snippet_ptp_per_channel(reader, seg_times, chans, nt, nt0min, cfg.fs, cfg.highpass_hz, rng=rng)
        if ptp is not None and np.linalg.norm(ptp) > 0:
            shapes.append(ptp / np.linalg.norm(ptp))

    if len(shapes) < 2:
        return dict(score=np.nan, n_segments_with_data=len(shapes), pairwise_cosine=[])

    sims = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            sims.append(float(np.dot(shapes[i], shapes[j])))

    return dict(score=float(np.mean(sims)), min_pairwise_cosine=float(np.min(sims)),
                n_segments_with_data=len(shapes), pairwise_cosine=sims)
