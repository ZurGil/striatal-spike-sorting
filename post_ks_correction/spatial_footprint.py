"""
Pillar 1c -- pooled energy / spatial footprint.

WHAT THIS IS, concretely: every module built so far (nuisance_model.py,
wavelet_features.py) looks at ONE channel -- the unit's single peak
channel. But a real spike's voltage shows up on MANY nearby channels at
once, with a characteristic PATTERN of how much signal reaches each one
(strongest on the true peak channel, falling off with distance). That
pattern -- the unit's "spatial footprint" -- is a second, independent
signature a unit carries, separate from its single-channel waveform SHAPE
(nuisance_model.py) or its single-channel TIMING (wavelet_features.py).

WHY IT MATTERS HERE SPECIFICALLY: units 342 and 347 (this project's
running real example) share the EXACT SAME peak channel (41) -- confirmed
many turns ago, and the direct motivation for the whole collision
investigation earlier in this project's history. A single-channel view
cannot use footprint at all to help tell them apart. If their footprints
(how the signal falls off across the OTHER nearby channels) differ, that
is new, independent information neither nuisance_model.py nor
wavelet_features.py can see.

Per the original design doc: pooled energy is ALIGNMENT-INSENSITIVE (like
wavelet magnitude, unlike wavelet phase) -- "how much, roughly where",
not a precise timing tool. And per that same doc: these features are
CORRELATED (same underlying voltage reaching multiple electrodes), not
independent votes -- do not treat a match/mismatch on many channels as
many separate pieces of evidence without accounting for that.
"""
import numpy as np


def unit_footprint(templates, unit_id, channel_positions, radius_um=60.0):
    """The unit's OWN expected spatial footprint, from its real Kilosort
    template (the same object used throughout this project -- NOT a new
    data source).

    templates : the full (n_units, n_timepoints, n_channels) array, e.g.
        np.load(".../templates.npy") -- already used everywhere else.
    unit_id : which unit.
    channel_positions : (n_channels, 2) array of (x, y) electrode
        coordinates in microns, e.g. np.load(".../channel_positions.npy").
    radius_um : which channels count as "this unit's neighborhood" --
        same convention (60um) used for the 342/347 neighborhood
        investigation earlier in this project.

    Returns dict(channels, footprint, peak_channel):
      channels    -- sorted array of channel indices within radius_um of
                      the unit's own peak channel
      footprint   -- per-channel amplitude (peak-to-trough of that
                      channel's own template trace), normalized to unit
                      L2 norm -- a SHAPE descriptor (relative pattern
                      across channels), not an absolute amplitude
      peak_channel -- the single channel with the largest amplitude
                      (same definition used everywhere else in this
                      project: argmax of max-min per channel)
    """
    templ_all = templates[unit_id]  # (n_timepoints, n_channels)
    amp_per_channel = templ_all.max(axis=0) - templ_all.min(axis=0)
    peak_channel = int(np.argmax(amp_per_channel))

    dist = np.sqrt(((channel_positions - channel_positions[peak_channel]) ** 2).sum(axis=1))
    channels = np.where(dist <= radius_um)[0]
    channels = np.sort(channels)

    raw_footprint = amp_per_channel[channels]
    norm = np.linalg.norm(raw_footprint)
    footprint = raw_footprint / norm if norm > 0 else raw_footprint

    return dict(channels=channels, footprint=footprint, peak_channel=peak_channel)


def spatial_energy_vector(multichannel_snippet):
    """Per-channel amplitude (peak-to-trough) for one real, observed
    multi-channel spike snippet -- the OBSERVED counterpart to
    unit_footprint's expected pattern. ALIGNMENT-INSENSITIVE by
    construction (peak-to-trough over a small window doesn't require
    precise sub-sample timing the way wavelet phase does -- consistent
    with the design doc's "pooled energy" framing).

    multichannel_snippet : (n_timepoints, n_channels) array -- same
        channel SET and ORDER as unit_footprint's `channels` output.

    Returns the per-channel amplitude vector, normalized to unit L2 norm
    (a shape, not an absolute-amplitude, comparison -- consistent with
    unit_footprint's own normalization, so the two are directly
    comparable regardless of this particular spike's overall amplitude).
    """
    amp_per_channel = multichannel_snippet.max(axis=0) - multichannel_snippet.min(axis=0)
    norm = np.linalg.norm(amp_per_channel)
    return amp_per_channel / norm if norm > 0 else amp_per_channel


def footprint_similarity(observed_vector, expected_footprint):
    """Cosine similarity between an observed spike's spatial pattern and
    a unit's expected footprint -- 1.0 = identical shape (regardless of
    overall amplitude, since both inputs are already unit-normalized),
    0.0 = unrelated/orthogonal patterns, negative = anti-correlated
    (shouldn't normally happen for real spike amplitudes, which are
    non-negative by construction here).

    Both vectors must already be over the SAME channel set/order (see
    unit_footprint's `channels` output) -- this function does not check
    that itself.
    """
    denom = np.linalg.norm(observed_vector) * np.linalg.norm(expected_footprint)
    if denom == 0:
        return np.nan
    return float(np.dot(observed_vector, expected_footprint) / denom)
