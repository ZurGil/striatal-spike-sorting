"""
Raw-data access: channel geometry (verified in step 1 against Trodes' own
channelmap_probe1.dat -- exact per-channel match, JSON is authoritative) and a
memmap-based reader that only ever touches the byte ranges actually requested
(confirmed fast: ~5ms per scattered 5000-sample x 16-channel window on this session's
SSD -- safe for the targeted, per-unit/per-window reads every stage below uses).
"""
import json
import os
import numpy as np
from scipy import signal

from .config import AgentConfig


class ChannelGeometry:
    def __init__(self, chan_map, xc, yc, kcoords, n_chan):
        self.chan_map = np.asarray(chan_map)
        self.xc = np.asarray(xc, dtype=np.float64)
        self.yc = np.asarray(yc, dtype=np.float64)
        self.kcoords = np.asarray(kcoords)
        self.n_chan = n_chan

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["chanMap"], d["xc"], d["yc"], d["kcoords"], d["n_chan"])

    def neighbors_within(self, channel, radius_um):
        dx = self.xc - self.xc[channel]
        dy = self.yc - self.yc[channel]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        return np.where(dist <= radius_um)[0]

    def distance(self, chan_a, chan_b):
        return float(np.hypot(self.xc[chan_a] - self.xc[chan_b], self.yc[chan_a] - self.yc[chan_b]))


class RawReader:
    """Headerless int16 binary, (n_samples, n_chan) row-major -- matches this session's
    probe1.dat (verified in step 1: file size / (n_chan*2) is an exact integer, and the
    sample count matches timestamps.dat exactly)."""

    def __init__(self, probe_dat_path, n_chan, dtype=np.int16):
        self.path = probe_dat_path
        self.n_chan = n_chan
        self.dtype = dtype
        itemsize = np.dtype(dtype).itemsize
        n_bytes = os.path.getsize(probe_dat_path)
        if n_bytes % (n_chan * itemsize) != 0:
            raise ValueError(f"{probe_dat_path}: size not divisible by n_chan*itemsize; "
                              f"channel count or dtype is probably wrong")
        self.n_samples = n_bytes // (n_chan * itemsize)
        self._arr = np.memmap(probe_dat_path, dtype=dtype, mode="r", shape=(self.n_samples, n_chan))

    def read_window(self, start, end, chans=None):
        start = max(0, int(start))
        end = min(self.n_samples, int(end))
        if chans is None:
            block = self._arr[start:end, :]
        else:
            block = self._arr[start:end, :][:, chans]
        return np.array(block, dtype=np.float32), start, end


def highpass_filter(x, fs, cutoff, order=3, axis=0):
    sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def read_filtered_snippet(reader: RawReader, chans, snippet_start, nt, fs, cutoff, pad=150):
    """A bare nt-sample (~2ms) snippet is too short to highpass-filter on its own --
    filtfilt needs padding or the edges are unstable. Reads a wider padded window,
    filters that, then crops back to the nt samples actually wanted. Every raw
    snippet read anywhere in this package (waveform display, structural score,
    footprint stability, collision check) must go through this, not RawReader
    directly -- an earlier version skipped filtering here, letting LFP content
    dominate averaged waveforms for busy units (found via visual inspection of the
    calibration report, see conversation)."""
    wide_start, wide_end = snippet_start - pad, snippet_start + nt + pad
    block, actual_start, actual_end = reader.read_window(wide_start, wide_end, chans)
    if block.shape[0] < nt:
        return None
    filt = highpass_filter(block, fs, cutoff)
    offset = snippet_start - actual_start
    if offset < 0 or offset + nt > filt.shape[0]:
        return None
    return filt[offset:offset + nt]


def footprint_channels(templates, unit_id, n_keep, geometry: ChannelGeometry = None, radius_um=None):
    """Top-N channels by template peak-to-peak amplitude. If geometry+radius given,
    intersect with the spatial-radius set around the peak channel (keeps footprint
    contiguous rather than picking up spatially unrelated high-amplitude channels)."""
    templ = templates[unit_id]  # (nt, n_chan)
    ptp = templ.max(axis=0) - templ.min(axis=0)
    order = np.argsort(ptp)[::-1]
    peak_chan = int(order[0])
    if geometry is not None and radius_um is not None:
        near = set(geometry.neighbors_within(peak_chan, radius_um).tolist())
        order = [c for c in order if c in near] or order[:n_keep]
    chosen = np.sort(np.array(order[:n_keep]))
    return chosen, peak_chan
