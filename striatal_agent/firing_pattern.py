"""
Section 3 (spec's "Section 4 from v1") -- firing-pattern profiling math. Produces a
continuous profile vector per unit; Section 2.4 says this determines whether/how the
burst/pause submodule (Section 5 / burst_module.py) activates, with no forced binary
label -- so this module returns continuous quantities, not a class.
"""
import numpy as np
from scipy.stats import poisson

from .config import AgentConfig


def firing_rate_stats(spike_times_samples, fs, session_duration_s, window_s=45.0):
    st = np.sort(spike_times_samples)
    mean_rate = len(st) / session_duration_s if session_duration_s > 0 else 0.0

    if len(st) < 2:
        return dict(mean_rate_hz=mean_rate, rate_cv_across_windows=np.nan)

    edges = np.arange(0, session_duration_s + window_s, window_s) * fs
    counts, _ = np.histogram(st, bins=edges)
    window_rates = counts / window_s
    rate_cv = float(np.std(window_rates) / (np.mean(window_rates) + 1e-9))
    return dict(mean_rate_hz=float(mean_rate), rate_cv_across_windows=rate_cv)


def isi_regularity(spike_times_samples, fs):
    """CV (Poisson reference = 1) and LV (Shinomoto et al. 2003, robust to slow rate drift)."""
    st = np.sort(spike_times_samples)
    if len(st) < 3:
        return dict(cv=np.nan, lv=np.nan)
    isi = np.diff(st) / fs  # seconds

    cv = float(np.std(isi) / (np.mean(isi) + 1e-12))

    a = isi[:-1]
    b = isi[1:]
    lv = float(3.0 / (len(isi) - 1) * np.sum(((a - b) / (a + b + 1e-12)) ** 2))
    return dict(cv=cv, lv=lv)


def poisson_surprise(k, lam, T):
    if T <= 0:
        return 0.0
    p = poisson.sf(k - 1, lam * T)
    p = max(p, 1e-300)
    return -np.log10(p)


def detect_bursts(spike_times_samples, fs, cfg: AgentConfig = None):
    """Seed on short-ISI runs, grow while Poisson Surprise increases, trim edges.
    Returns list of dicts: start_t, end_t (samples), n_spikes, S.

    Edge trimming: after growing to the max-S point, drop leading/trailing spikes whose
    removal would not decrease S (per spec Section 3) -- catches seed spikes that
    happened to start the run but weren't really part of the elevated-rate epoch.
    """
    cfg = cfg or AgentConfig()
    st = np.sort(spike_times_samples)
    if len(st) < cfg.burst_min_spikes + 1:
        return []
    duration_s = (st[-1] - st[0]) / fs
    if duration_s <= 0:
        return []
    lam = len(st) / duration_s
    isi_ms = np.diff(st) / fs * 1000.0

    bursts = []
    i = 0
    n = len(st)
    used_until = -1
    while i < n - 1:
        if isi_ms[i] < cfg.burst_seed_isi_ms and i > used_until:
            start_idx = i
            # vectorized: compute Poisson Surprise for the whole candidate growth
            # window (up to 200 spikes) in one scipy call instead of one call per
            # step -- same result, ~10-50x faster (dominant cost on high-rate units:
            # a 356k-spike unit took 20s here before this change, some units in this
            # session have 890k spikes)
            j_hi = min(start_idx + 1 + 200, n)
            j_range = np.arange(start_idx + 1, j_hi)
            T_arr = (st[j_range] - st[start_idx]) / fs
            k_arr = j_range - start_idx + 1
            S_arr = -np.log10(np.clip(poisson.sf(k_arr - 1, lam * T_arr), 1e-300, None))

            best_S, best_j = -np.inf, None
            for local_idx, j in enumerate(j_range):
                S = S_arr[local_idx]
                if S > best_S:
                    best_S, best_j = S, int(j)
                if j > start_idx + 1 and isi_ms[j - 1] > cfg.burst_seed_isi_ms * 3 and S < best_S:
                    break
            end_idx = best_j if best_j is not None else start_idx + 1

            # trim edges: drop leading/trailing spikes that don't reduce S
            while end_idx - start_idx > cfg.burst_min_spikes - 1:
                T_trim = (st[end_idx] - st[start_idx + 1]) / fs
                k_trim = end_idx - (start_idx + 1) + 1
                if poisson_surprise(k_trim, lam, T_trim) >= best_S:
                    start_idx += 1
                else:
                    break

            n_spk = end_idx - start_idx + 1
            if n_spk >= cfg.burst_min_spikes and best_S >= cfg.burst_min_surprise:
                bursts.append(dict(
                    start_t=int(st[start_idx]), end_t=int(st[end_idx]),
                    n_spikes=int(n_spk), S=float(best_S),
                    start_idx=int(start_idx), end_idx=int(end_idx),
                ))
                used_until = end_idx
            i = end_idx + 1
        else:
            i += 1
    return bursts


def detect_pauses(spike_times_samples, fs, cfg: AgentConfig = None):
    """ISI > pause_isi_multiplier x local median ISI (windowed). Returns list of dicts:
    start_t, end_t, duration_s, rebound_rate_hz (rate in the 1s after the pause ends)."""
    cfg = cfg or AgentConfig()
    st = np.sort(spike_times_samples)
    if len(st) < 5:
        return []
    isi = np.diff(st) / fs  # seconds

    win = int(cfg.pause_local_window_s * len(st) / max((st[-1] - st[0]) / fs, 1e-9))
    win = max(5, min(win, len(isi)))
    # centered rolling median, min_periods=1 to match the original's clipped-boundary
    # window -- vectorized (pandas' C rolling-median impl) instead of one np.median
    # call per spike, which was O(n * win) and the dominant cost on high-rate units
    import pandas as pd
    local_median = pd.Series(isi).rolling(window=win, center=True, min_periods=1).median().to_numpy()

    pauses = []
    is_pause = isi > cfg.pause_isi_multiplier * (local_median + 1e-12)
    for idx in np.where(is_pause)[0]:
        start_t, end_t = int(st[idx]), int(st[idx + 1])
        duration_s = (end_t - start_t) / fs
        post_mask = (st > end_t) & (st <= end_t + fs)
        rebound_rate = float(post_mask.sum())  # spikes in 1s after pause end = Hz
        pauses.append(dict(start_t=start_t, end_t=end_t, duration_s=float(duration_s),
                            rebound_rate_hz=rebound_rate))
    return pauses


def profile_unit(spike_times_samples, fs, session_duration_s, cfg: AgentConfig = None):
    cfg = cfg or AgentConfig()
    rate_stats = firing_rate_stats(spike_times_samples, fs, session_duration_s)
    reg_stats = isi_regularity(spike_times_samples, fs)
    bursts = detect_bursts(spike_times_samples, fs, cfg)
    pauses = detect_pauses(spike_times_samples, fs, cfg)

    n_spikes = len(spike_times_samples)
    burst_spike_count = sum(b["n_spikes"] for b in bursts)
    burst_fraction = burst_spike_count / n_spikes if n_spikes else 0.0
    pause_time_fraction = (sum(p["duration_s"] for p in pauses) / session_duration_s
                            if session_duration_s > 0 else 0.0)

    return dict(
        n_spikes=n_spikes,
        **rate_stats, **reg_stats,
        n_bursts=len(bursts), burst_fraction=burst_fraction,
        n_pauses=len(pauses), pause_time_fraction=pause_time_fraction,
        bursts=bursts, pauses=pauses,
        low_confidence=n_spikes < cfg.min_spikes_for_profiling,
    )
