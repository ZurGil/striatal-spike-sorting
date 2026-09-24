"""
Section 1 validation gate (striatal_spike_sorting_agent_spec.md).

Quantifies, on a handful of clean/high-SNR/well-isolated units, whether raw-data
spike recovery is likely to add anything before investing in the full recovery
machinery (spec Section 2.7). This is deliberately a lean, exploratory
implementation -- not the calibrated Section 2.7 pipeline.

Method per unit:
  1. Poisson-Surprise burst detection on the unit's own KS-confirmed ISI train.
  2. Within burst windows (+padding), read the raw trace on the unit's footprint
     channels only (targeted memmap reads, not full-file), high-pass filter it.
  3. Raw-threshold detector: count simple MAD-threshold crossings vs KS spike
     count, within burst windows.
  4. Matched-filter detector: cross-correlate the unit's own KS template against
     the filtered window trace, get a per-sample match score; read off the score
     at each KS-confirmed spike, and separately find other local maxima
     ("candidates") not already matched to a KS spike.
  5. Compare candidate-score distribution to accepted-score distribution, flag
     candidates whose score falls within the accepted distribution and whose
     insertion would not introduce a refractory (<1.5ms) violation.
  6. Report, per unit and in aggregate, what fraction of spikes recovery would add.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
KS_DIR = os.path.join(SESSION_ROOT, r"20260901_085606.kilosort\kilosort4")
PROBE_PATH = os.path.join(SESSION_ROOT, r"20260901_085606.kilosort\20260901_085606.probe1.dat")
OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs"

FS = 30000.0
N_CHAN = 384
N_SAMPLES = 325_904_509

N_FOOTPRINT_CHANS = 10          # channels kept per unit for raw analysis
HIGHPASS_HZ = 300.0
BURST_SEED_ISI_MS = 15.0        # prior for MSN-like burst seed cutoff (spec Sec 7)
MIN_BURST_SPIKES = 3
BURST_MIN_SURPRISE = 2.0        # S = -log10(P), S>=2 => p<=0.01
PAD_MS = 8.0                    # padding added to each burst window for filtering context
RAW_THRESH_K = 4.0              # MAD multiplier for the naive raw-threshold detector
REFRACTORY_MS = 1.5             # standard refractory period for violation checks
SPIKE_EXCLUSION_MS = 0.5        # min separation to call something a "different" event from a KS spike
N_UNITS_TO_TEST = 8
MIN_SPIKE_COUNT = 3000
MAX_CONTAM_PCT = 5.0
MAX_BURSTS_PER_UNIT = 150       # cap raw I/O per unit

rng = np.random.default_rng(0)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. Load KS outputs (reconstructed pre-phy original: spike_templates.npy)
# ---------------------------------------------------------------------------
def load_ks():
    spike_templates = np.load(os.path.join(KS_DIR, "spike_templates.npy")).ravel()
    spike_times = np.load(os.path.join(KS_DIR, "spike_times.npy")).ravel()
    templates = np.load(os.path.join(KS_DIR, "templates.npy"))  # (n_templates, nt, n_chan)
    channel_positions = np.load(os.path.join(KS_DIR, "channel_positions.npy"))
    kslabel = pd.read_csv(os.path.join(KS_DIR, "cluster_KSLabel.tsv"), sep="\t")
    contam = pd.read_csv(os.path.join(KS_DIR, "cluster_ContamPct.tsv"), sep="\t")
    amp_tsv = pd.read_csv(os.path.join(KS_DIR, "cluster_Amplitude.tsv"), sep="\t")
    return spike_templates, spike_times, templates, channel_positions, kslabel, contam, amp_tsv


def merged_away_template_ids():
    """Template IDs the human already merged away in phy -- excluded from the
    validation-gate candidate pool so we only test units nobody has flagged."""
    spike_clusters = np.load(os.path.join(KS_DIR, "spike_clusters.npy")).ravel()
    spike_templates = np.load(os.path.join(KS_DIR, "spike_templates.npy")).ravel()
    diff_mask = spike_clusters != spike_templates
    return set(np.unique(spike_templates[diff_mask]).tolist())


def select_units(spike_templates, spike_times, kslabel, contam, amp_tsv):
    uniq, counts = np.unique(spike_templates, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))

    m = kslabel.merge(contam, on="cluster_id").merge(amp_tsv, on="cluster_id")
    m["n_spikes"] = m["cluster_id"].map(count_map)
    m = m.dropna(subset=["n_spikes"])

    excluded = merged_away_template_ids()
    m = m[~m["cluster_id"].isin(excluded)]

    good = m[(m.KSLabel == "good") & (m.ContamPct < MAX_CONTAM_PCT) & (m.n_spikes > MIN_SPIKE_COUNT)]
    good = good.sort_values("Amplitude", ascending=False)
    chosen = good.head(N_UNITS_TO_TEST)["cluster_id"].astype(int).tolist()
    log(f"Candidate clean units after excluding phy-merged templates: {len(good)}")
    log(f"Selected {len(chosen)} units for validation gate: {chosen}")
    return chosen


def burst_fraction(spike_times_samples, isi_cut_ms=BURST_SEED_ISI_MS, fs=FS):
    st = np.sort(spike_times_samples)
    if len(st) < 10:
        return 0.0
    isi = np.diff(st) / fs * 1000.0
    return float(np.mean(isi < isi_cut_ms))


def select_burstier_lowersnr_units(spike_templates, spike_times, kslabel, contam, amp_tsv,
                                    n_units=N_UNITS_TO_TEST, max_contam=30.0, min_spikes=2000):
    """Lower-SNR (bottom half of amplitude among labeled units), more contaminated
    (but still interpretable, <30%) and high burst_fraction (ISI<15ms) units --
    the population the burst/decrement hypothesis (spec Sec 4/5) predicts should
    show a bigger recovery effect than the cleanest units tested first."""
    uniq, counts = np.unique(spike_templates, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))

    m = kslabel.merge(contam, on="cluster_id").merge(amp_tsv, on="cluster_id")
    m["n_spikes"] = m["cluster_id"].map(count_map)
    m = m.dropna(subset=["n_spikes"])

    excluded = merged_away_template_ids()
    m = m[~m["cluster_id"].isin(excluded)]
    m = m[(m.n_spikes > min_spikes) & (m.ContamPct < max_contam) & (m.KSLabel.isin(["good", "mua"]))]

    amp_median = m["Amplitude"].median()
    m = m[m["Amplitude"] <= amp_median]

    log("Computing burst_fraction for candidate pool (in-memory, no raw I/O)...")
    burst_fracs = {}
    for cid in m["cluster_id"].astype(int):
        st = spike_times[spike_templates == cid]
        burst_fracs[cid] = burst_fraction(st)
    m = m.copy()
    m["burst_fraction"] = m["cluster_id"].astype(int).map(burst_fracs)

    m = m.sort_values("burst_fraction", ascending=False)
    chosen = m.head(n_units)["cluster_id"].astype(int).tolist()
    log(f"Candidate burstier/lower-SNR pool: {len(m)} (amplitude <= median {amp_median:.1f}, "
        f"ContamPct<{max_contam}, KSLabel in good/mua)")
    log(m.head(n_units)[["cluster_id", "KSLabel", "ContamPct", "Amplitude", "n_spikes", "burst_fraction"]].to_string(index=False))
    log(f"Selected {len(chosen)} burstier/lower-SNR units for validation gate: {chosen}")
    return chosen


# ---------------------------------------------------------------------------
# 2. Poisson-Surprise burst detection
# ---------------------------------------------------------------------------
def poisson_surprise(k, lam, T):
    if T <= 0:
        return 0.0
    p = poisson.sf(k - 1, lam * T)  # P(N >= k)
    p = max(p, 1e-300)
    return -np.log10(p)


def detect_bursts(spike_times_samples, fs=FS, seed_isi_ms=BURST_SEED_ISI_MS,
                   min_spikes=MIN_BURST_SPIKES, min_surprise=BURST_MIN_SURPRISE):
    """Seed on short-ISI runs, grow while Poisson Surprise increases, trim edges.
    Returns list of dicts: start_t, end_t (samples), n_spikes, S."""
    st = np.sort(spike_times_samples)
    if len(st) < min_spikes + 1:
        return []
    duration_s = (st[-1] - st[0]) / fs
    if duration_s <= 0:
        return []
    lam = len(st) / duration_s  # spikes/sec, mean rate
    isi = np.diff(st) / fs * 1000.0  # ms
    seed_cut = seed_isi_ms

    bursts = []
    i = 0
    n = len(st)
    used_until = -1
    while i < n - 1:
        if isi[i] < seed_cut and i > used_until:
            # seed burst at spike i (start), grow forward
            start_idx = i
            j = i + 1
            best_S = -np.inf
            best_j = None
            while j < n:
                T = (st[j] - st[start_idx]) / fs
                k = j - start_idx + 1
                S = poisson_surprise(k, lam, T)
                if S > best_S:
                    best_S = S
                    best_j = j
                # stop growing once ISI gets long (burst clearly ended) and S no
                # longer improving for a few steps
                if j > start_idx + 1 and isi[j - 1] > seed_cut * 3 and S < best_S:
                    break
                if j - start_idx > 200:  # safety cap
                    break
                j += 1
            end_idx = best_j if best_j is not None else start_idx + 1
            n_spk = end_idx - start_idx + 1
            if n_spk >= min_spikes and best_S >= min_surprise:
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


# ---------------------------------------------------------------------------
# 3. Raw data access + filtering
# ---------------------------------------------------------------------------
def get_footprint_channels(templates, unit_id, n_keep=N_FOOTPRINT_CHANS):
    templ = templates[unit_id]  # (nt, n_chan)
    ptp = templ.max(axis=0) - templ.min(axis=0)
    order = np.argsort(ptp)[::-1]
    chosen = np.sort(order[:n_keep])
    peak_chan = int(order[0])
    return chosen, peak_chan


def highpass_filter(x, fs=FS, cutoff=HIGHPASS_HZ, order=3):
    sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


class RawReader:
    def __init__(self, path, n_samples=N_SAMPLES, n_chan=N_CHAN):
        self.arr = np.memmap(path, dtype=np.int16, mode="r", shape=(n_samples, n_chan))
        self.n_samples = n_samples

    def read_window(self, start, end, chans):
        start = max(0, start)
        end = min(self.n_samples, end)
        return np.array(self.arr[start:end, :][:, chans], dtype=np.float32), start, end


# ---------------------------------------------------------------------------
# 4. Per-unit analysis
# ---------------------------------------------------------------------------
def analyze_unit(unit_id, spike_times, templates, reader):
    unit_spike_times = np.sort(spike_times)
    bursts = detect_bursts(unit_spike_times)
    n_bursts_total = len(bursts)
    if len(bursts) > MAX_BURSTS_PER_UNIT:
        idx = rng.choice(len(bursts), size=MAX_BURSTS_PER_UNIT, replace=False)
        bursts = [bursts[i] for i in sorted(idx)]

    footprint_chans, peak_chan_global = get_footprint_channels(templates, unit_id)
    peak_local = int(np.where(footprint_chans == peak_chan_global)[0][0])

    templ_full = templates[unit_id][:, footprint_chans]  # (nt, k)
    templ_norm = templ_full / (np.linalg.norm(templ_full) + 1e-9)
    nt = templ_full.shape[0]

    pad = int(PAD_MS / 1000.0 * FS)
    excl = int(SPIKE_EXCLUSION_MS / 1000.0 * FS)
    refr = int(REFRACTORY_MS / 1000.0 * FS)

    raw_thresh_crossings_total = 0
    ks_spikes_in_bursts_total = 0
    accepted_scores = []
    candidate_scores = []
    candidate_times = []
    n_bursts_used = 0

    for b in bursts:
        w_start, w_end = b["start_t"] - pad, b["end_t"] + pad
        raw, actual_start, actual_end = reader.read_window(w_start, w_end, footprint_chans)
        if raw.shape[0] < nt + 10:
            continue
        filt = highpass_filter(raw)
        peak_trace = filt[:, peak_local]

        # --- raw threshold detector ---
        mad = np.median(np.abs(peak_trace - np.median(peak_trace))) * 1.4826 + 1e-9
        thresh = -RAW_THRESH_K * mad
        below = peak_trace < thresh
        # count separate crossing events (falling below thresh after being above)
        crossings = np.where(below & ~np.roll(below, 1))[0]
        # restrict to core burst window (exclude padding)
        core_lo, core_hi = b["start_t"] - actual_start, b["end_t"] - actual_start
        crossings_core = crossings[(crossings >= core_lo) & (crossings <= core_hi)]
        raw_thresh_crossings_total += len(crossings_core)

        ks_in_window = unit_spike_times[(unit_spike_times >= b["start_t"]) & (unit_spike_times <= b["end_t"])]
        ks_spikes_in_bursts_total += len(ks_in_window)

        # --- matched filter detector ---
        score = np.zeros(filt.shape[0] - nt + 1, dtype=np.float32)
        for c in range(filt.shape[1]):
            score += np.correlate(filt[:, c], templ_norm[:, c], mode="valid")
        # score[t] corresponds to template placed starting at sample t (in window coords);
        # KS spike_time convention marks the trough at offset nt0min within the template
        ks_local_onsets = (ks_in_window - actual_start - 20).astype(int)  # nt0min=20

        matched_mask = np.zeros(len(score), dtype=bool)
        for onset in ks_local_onsets:
            lo, hi = max(0, onset - excl), min(len(score), onset + excl + 1)
            if hi <= lo:
                continue
            local_max_idx = lo + np.argmax(score[lo:hi])
            accepted_scores.append(float(score[local_max_idx]))
            matched_mask[max(0, local_max_idx - excl):local_max_idx + excl + 1] = True

        # find local maxima elsewhere in the window as recovery candidates
        peaks, _ = signal.find_peaks(score, distance=refr)
        for p in peaks:
            if matched_mask[p]:
                continue
            candidate_scores.append(float(score[p]))
            candidate_times.append(int(actual_start + p + 20))

        n_bursts_used += 1

    accepted_scores = np.array(accepted_scores)
    candidate_scores = np.array(candidate_scores)
    candidate_times = np.array(candidate_times)

    if len(accepted_scores) >= 5:
        accept_p5 = np.percentile(accepted_scores, 5)
    else:
        accept_p5 = np.inf

    plausible_mask = candidate_scores >= accept_p5
    plausible_times = candidate_times[plausible_mask]

    # refractory check: would adding these introduce <1.5ms violations against
    # the unit's *existing* full spike train (not just this burst)?
    if len(plausible_times) > 0:
        all_times_sorted = np.sort(unit_spike_times)
        nearest_gap = np.abs(all_times_sorted[np.searchsorted(all_times_sorted, plausible_times).clip(0, len(all_times_sorted)-1)] - plausible_times)
        violates = nearest_gap < refr
    else:
        violates = np.array([], dtype=bool)

    n_clean_recoverable = int((~violates).sum())

    baseline_isi = np.diff(unit_spike_times) / FS * 1000.0
    baseline_viol_rate = float(np.mean(baseline_isi < REFRACTORY_MS)) if len(baseline_isi) else np.nan

    result = dict(
        unit_id=int(unit_id),
        n_ks_spikes_total=int(len(unit_spike_times)),
        n_bursts_detected=n_bursts_total,
        n_bursts_analyzed=n_bursts_used,
        ks_spikes_in_bursts=int(ks_spikes_in_bursts_total),
        raw_thresh_crossings_in_bursts=int(raw_thresh_crossings_total),
        raw_thresh_excess_pct=float(100.0 * (raw_thresh_crossings_total - ks_spikes_in_bursts_total) / max(ks_spikes_in_bursts_total, 1)),
        n_accepted_scores=int(len(accepted_scores)),
        accepted_score_mean=float(np.mean(accepted_scores)) if len(accepted_scores) else np.nan,
        accepted_score_p5=float(accept_p5) if np.isfinite(accept_p5) else np.nan,
        n_candidates_raw=int(len(candidate_scores)),
        n_candidates_plausible_score=int(len(plausible_times)),
        n_candidates_clean_after_refractory=n_clean_recoverable,
        pct_spikes_recovery_would_add=float(100.0 * n_clean_recoverable / max(len(unit_spike_times), 1)),
        baseline_refractory_violation_rate_pct=float(100.0 * baseline_viol_rate) if not np.isnan(baseline_viol_rate) else np.nan,
    )
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clean", "bursty", "explicit"], default="clean")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--units", default="", help="comma-separated cluster_ids, for --mode explicit")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log("Loading KS outputs...")
    spike_templates, spike_times, templates, channel_positions, kslabel, contam, amp_tsv = load_ks()

    if args.mode == "clean":
        chosen_units = select_units(spike_templates, spike_times, kslabel, contam, amp_tsv)
    elif args.mode == "bursty":
        chosen_units = select_burstier_lowersnr_units(spike_templates, spike_times, kslabel, contam, amp_tsv)
    else:
        chosen_units = [int(x) for x in args.units.split(",") if x.strip()]
        log(f"Explicit unit list: {chosen_units}")

    reader = RawReader(PROBE_PATH)

    results = []
    for u in chosen_units:
        log(f"\n--- Unit {u} ---")
        st = spike_times[spike_templates == u]
        res = analyze_unit(u, st, templates, reader)
        log(json.dumps(res, indent=2))
        results.append(res)

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, f"validation_gate_results{args.out_suffix}.csv")
    df.to_csv(out_csv, index=False)
    log(f"\nSaved per-unit results to {out_csv}")

    log("\n=== AGGREGATE SUMMARY ===")
    log(df[["unit_id", "n_ks_spikes_total", "n_bursts_detected", "ks_spikes_in_bursts",
            "raw_thresh_excess_pct", "n_candidates_clean_after_refractory",
            "pct_spikes_recovery_would_add", "baseline_refractory_violation_rate_pct"]].to_string(index=False))

    log(f"\nMean %% spikes recovery would add: {df['pct_spikes_recovery_would_add'].mean():.3f}%%")
    log(f"Median %% spikes recovery would add: {df['pct_spikes_recovery_would_add'].median():.3f}%%")
    log(f"Units with zero plausible clean recoveries: {(df['n_candidates_clean_after_refractory']==0).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
