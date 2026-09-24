"""
Computes burst_curve, burst_shape, isi_amp_corr_all, isi_amp_corr_burst for one
session -- fields the review tool displays but whose original computation script was
lost (see WORKFLOW.md 1f-pre). This is a FRESH RECONSTRUCTION, not recovered original
code -- the exact shape-classification thresholds below are a new, reasonable design,
not a match to whatever exact logic produced the old session's values. If you ever find
the original script, prefer it and note the discrepancy.

Design (documented here since there's no prior script to point to):
- burst_curve: spike-times amplitude (from kilosort_original/amplitudes.npy, KS's own
  template-scaled amplitude) as a function of position within a burst, where a burst
  is detected via the pipeline's own striatal_agent.firing_pattern.detect_bursts()
  (same definition already used elsewhere for burst_fraction/n_true_bursts -- reusing
  it rather than a second, different burst definition). For each burst with
  >=cfg.burst_min_spikes spikes, record (position-in-burst, amplitude) for each
  spike. Aggregate mean/sem/n per position across all bursts; keep only positions with
  n>=3 to avoid single-burst noise; cap at 6 positions (later positions get sparse).
  excluded_high_rate: mean_rate_hz >= 5 (same threshold the review tool's own message
  already documents). insufficient_data: fewer than 3 qualifying bursts.
- burst_shape: classified from the resulting curve's mean values -- decay (monotonic
  non-increasing), facilitation (monotonic non-decreasing), dip_then_rise (interior
  minimum below both endpoints), rise_then_dip (interior maximum above both
  endpoints), flat (none of the above, i.e. first/last within ~10% of each other and
  no clear interior extremum).
- isi_amp_corr_all: Pearson correlation between preceding-ISI (ms) and amplitude,
  across ALL spikes (except each unit's first, which has no preceding ISI).
- isi_amp_corr_burst: same correlation, restricted to spike pairs with preceding
  ISI < 20ms (matching the burst_seed_isi_ms-scale threshold used elsewhere).
- isi_amp_scatter: intentionally NOT computed -- confirmed unused by the current
  review tool's rendering code (dead field in the old session's data).

Usage: python build_burst_isi_metrics_session.py <session_id> <rec_root>
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig
from striatal_agent.firing_pattern import detect_bursts

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
KO_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4\kilosort_original"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
OUT_PATH = rf"{OUT}\burst_isi_metrics_{SESSION_ID}.json"

FS = 30000.0
MIN_N_PER_POS = 3
MAX_POS = 6
HIGH_RATE_HZ = 5.0
BURST_ISI_MS = 20.0
FLAT_TOL_FRAC = 0.10  # first/last within 10% of the larger -> "flat"


def classify_shape(means):
    if len(means) < 2:
        return "insufficient_data"
    first, last = means[0], means[-1]
    scale = max(abs(first), abs(last), 1e-9)
    interior = means[1:-1]
    if len(interior) > 0:
        imin, imax = min(interior), max(interior)
        if imin < min(first, last) - 0.05 * scale:
            return "dip_then_rise"
        if imax > max(first, last) + 0.05 * scale:
            return "rise_then_dip"
    if abs(first - last) <= FLAT_TOL_FRAC * scale:
        return "flat"
    return "decay" if last < first else "facilitation"


def main():
    cfg = AgentConfig()
    spike_times = np.load(os.path.join(KO_DIR, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(KO_DIR, "spike_clusters.npy")).ravel()
    amplitudes = np.load(os.path.join(KO_DIR, "amplitudes.npy")).ravel()
    all_units = sorted(np.unique(spike_clusters).tolist())

    results = {}
    for i, uid in enumerate(all_units):
        mask = spike_clusters == uid
        st = spike_times[mask]
        amp = amplitudes[mask]
        order = np.argsort(st)
        st, amp = st[order], amp[order]

        if len(st) < 2:
            results[uid] = dict(burst_curve=[], burst_shape="insufficient_data",
                                 isi_amp_corr_all=None, isi_amp_corr_burst=None)
            continue

        duration_s = (st[-1] - st[0]) / FS
        mean_rate_hz = len(st) / duration_s if duration_s > 0 else 0.0

        isi_ms = np.diff(st) / FS * 1000.0
        amp_after = amp[1:]  # amplitude of the spike that ENDS each ISI
        valid = ~np.isnan(amp_after)
        corr_all = None
        if valid.sum() >= 10 and np.std(isi_ms[valid]) > 0 and np.std(amp_after[valid]) > 0:
            corr_all = float(np.corrcoef(isi_ms[valid], amp_after[valid])[0, 1])
        burst_mask = valid & (isi_ms < BURST_ISI_MS)
        corr_burst = None
        if burst_mask.sum() >= 10 and np.std(isi_ms[burst_mask]) > 0 and np.std(amp_after[burst_mask]) > 0:
            corr_burst = float(np.corrcoef(isi_ms[burst_mask], amp_after[burst_mask])[0, 1])

        if mean_rate_hz >= HIGH_RATE_HZ:
            results[uid] = dict(burst_curve=[], burst_shape="excluded_high_rate",
                                 isi_amp_corr_all=corr_all, isi_amp_corr_burst=corr_burst)
            continue

        bursts = detect_bursts(st, FS, cfg)
        pos_amps = {}
        for b in bursts:
            for pos, idx in enumerate(range(b["start_idx"], b["end_idx"] + 1)):
                if pos >= MAX_POS:
                    break
                if not np.isnan(amp[idx]):
                    pos_amps.setdefault(pos, []).append(float(amp[idx]))

        curve = []
        for pos in sorted(pos_amps.keys()):
            vals = np.array(pos_amps[pos])
            if len(vals) < MIN_N_PER_POS:
                continue
            curve.append(dict(pos=pos + 1, mean=round(float(vals.mean()), 2),
                               sem=round(float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0, 2),
                               n=int(len(vals))))

        if len(bursts) < 3 or len(curve) < 2:
            shape = "insufficient_data"
            curve = []
        else:
            shape = classify_shape([c["mean"] for c in curve])

        results[uid] = dict(burst_curve=curve, burst_shape=shape,
                             isi_amp_corr_all=corr_all, isi_amp_corr_burst=corr_burst)

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(all_units)}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"wrote {len(results)} units to {OUT_PATH}")
    from collections import Counter
    print(Counter(r["burst_shape"] for r in results.values()))


if __name__ == "__main__":
    main()
