"""
STAGE 1: the detection scan rerun with the spatial footprint actually used as
the discriminator.

WHY THIS RUN EXISTS: every detection number this project has produced so far
(sections 5h-5p of the research log) scored candidate events on ONE channel --
the unit's peak channel. Section 5q then measured that single-channel scoring
is the binding limitation: unit 440 and unit 439 are different neurons at
different probe depths, but on the one channel they share their waveforms
correlate 0.972, so a single-channel score cannot reject 439's spikes when
hunting for 440's. Footprint similarity separated that pair at AUC 0.999
versus 0.749 for the single-channel fit.

But 5q was a separation test on two KNOWN populations. It never ran inside a
detection scan. This script does that, and asks the only question that
matters: when the footprint gate is added, do the neighbour-attributable
detections go away while the unit's own real spikes survive?

THE GATES (both calibrated on the unit's OWN real spikes, so the unit defines
its own standard and nothing is hand-tuned):
  R2 gate        : whitened amplitude+stretch fit on the peak channel,
                   bar = 25th percentile of the unit's own real spikes
                   -- IDENTICAL to the previous runs, deliberately unchanged
                   so the comparison is apples-to-apples
  footprint gate : cosine similarity of the observed per-channel amplitude
                   pattern against the unit's expected footprint,
                   bar = 5th percentile of the unit's own real spikes
                   -- a permissive gate that by construction keeps ~95% of
                   real spikes, so any drop in detections is specificity,
                   not a tighter threshold in disguise

THE IMPOSTOR TEST: the nearest competing unit's own known spikes are pushed
through the target unit's gates. Their pass rate IS the false-positive rate
against the hardest possible impostor. This is the decisive measurement.

Alignment uses the corrected frequency rule (criterion="timing") throughout,
done ONCE on the peak channel, with the resulting shift applied to every
channel -- one spike, one time.

Usage: python demo_footprint_gated_detection_scan.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, find_peaks
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise
from spatial_footprint import (multichannel_template, spatial_energy_vector,
                                footprint_similarity)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, MARGIN, MIN_PEAK_DISTANCE = 25, 3.0, 20, 20
RADIUS_UM, JITTER_WINDOW = 60.0, 15
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_REF = 250
R2_PCT, FP_PCT = 25.0, 5.0

# target unit -> the nearest competing unit whose spikes it must reject.
# 440/439 is the pair measured in 5q; 408's competitor is found automatically.
UNITS = [440, 408]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL)
live = np.array(sorted(set(spike_clusters.tolist())))

print(f"window: samples {WINDOW_START:,}-{WINDOW_END:,} "
      f"({(WINDOW_END-WINDOW_START)/FS/60:.1f} min of {TOTAL/FS/60:.0f} min)")


def nearest_competitor(uid):
    """The live unit within RADIUS_UM whose peak channel is closest to this
    unit's, excluding itself -- the impostor a detector for `uid` must reject."""
    mp = channel_positions[peak_ch_all[uid]]
    best = None
    for oc in live:
        if oc == uid or oc >= n_templ:
            continue
        if (spike_clusters == oc).sum() < 200:
            continue
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (int(oc), float(d))
    return best


def load_channels(chans):
    """Read + highpass-filter the window for a specific channel set."""
    out = {ch: np.empty(WINDOW_END - WINDOW_START, dtype=np.float32) for ch in chans}
    pos = WINDOW_START
    while pos < WINDOW_END:
        clen = min(CHUNK, WINDOW_END - pos)
        rlo, rhi = max(0, pos - OVERLAP), min(TOTAL, pos + clen + OVERLAP)
        with open(DAT_PATH, "rb") as f:
            f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
        lt = pos - rlo
        for ch in chans:
            filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
            out[ch][pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt:lt + clen]
        pos += clen
    return out


def neighbor(uid, s):
    """Nearest unit within RADIUS_UM that already has a spike within
    JITTER_WINDOW of sample s. Spatially restricted on purpose -- an
    unrestricted version is meaningless at this spike density (log caveat)."""
    mp = channel_positions[peak_ch_all[uid]]
    m = np.abs(spike_times - s) <= JITTER_WINDOW
    best = None
    for oc in sorted(set(spike_clusters[m].tolist()) - {uid}):
        if oc >= n_templ:
            continue
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (int(oc), float(d))
    return best


summary, all_rows = [], []
for uid in UNITS:
    t_unit = time.time()
    mct = multichannel_template(templates, uid, channel_positions,
                                radius_um=RADIUS_UM, min_amplitude_fraction=0.0)
    CH = list(mct["channels"])
    PC = mct["peak_channel"]
    TM = templates[uid][:, PC]
    pc_idx = CH.index(PC)
    exp_amp = mct["per_channel_amplitude"]
    FP_EXPECTED = exp_amp / np.linalg.norm(exp_amp)

    f0, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                      criterion="timing")
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
    refph = calibrate_reference_phase(TM, psi, FS, align_index=NT0MIN)
    s0b, _, qb = build_basis(TM, dt=1.0)
    tmn = TM / (np.linalg.norm(TM) + 1e-12)

    comp = nearest_competitor(uid)
    print(f"\n{'='*74}\nunit {uid}: peak ch{PC}, footprint {len(CH)} channels {CH}")
    print(f"  f0 = {f0:.0f} Hz (cycle {FS/f0:.1f} samples)")
    print(f"  nearest competitor: unit {comp[0]} at {comp[1]:.0f} um "
          f"(peak ch{peak_ch_all[comp[0]]})" if comp else "  no competitor in radius")

    t0 = time.time()
    traces = load_channels(CH)
    print(f"  loaded {len(CH)} channels in {time.time()-t0:.0f}s")

    W_ops = []
    for ch in CH:
        Wc, _, _ = build_whitening_from_noise(traces[ch][:60000].astype(np.float64),
                                              N, order=4)
        W_ops.append(Wc)
    W_PC = W_ops[pc_idx]

    def score(s):
        """Align once on the peak channel; return single-channel whitened R2
        and footprint similarity across the whole footprint."""
        i = s - WINDOW_START
        pad = 200
        if i - pad < 0 or i + pad >= len(traces[PC]):
            return None
        loc = traces[PC][i - pad:i + pad].astype(np.float64)
        r = coarse_then_fine_shift(loc, pad, TM, psi, f0, FS, reference_phase=refph,
                                   search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
        tc = pad + r["total_shift"]
        lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
        if lo < 0 or hi > 2 * pad:
            return None
        grid = tc - NT0MIN + np.arange(N)
        snips = []
        for ch in CH:
            tr = traces[ch][i - pad:i + pad].astype(np.float64)
            ip = interp1d(np.arange(lo, hi), tr[lo:hi], kind="cubic",
                          bounds_error=False, fill_value=0.0)
            snips.append(ip(grid))
        multi = np.stack(snips, axis=1)
        r2 = fit_nuisance_prealigned(multi[:, pc_idx], s0b, qb,
                                     whitening_matrix=W_PC)["r_squared"]
        fp = footprint_similarity(spatial_energy_vector(multi), FP_EXPECTED)
        return dict(r2=r2, fp=fp, shift=r["total_shift"])

    # ---- calibrate both gates on the unit's own real spikes ----
    st = np.sort(spike_times[spike_clusters == uid])
    st_in = st[(st >= WINDOW_START + 1000) & (st < WINDOW_END - 1000)]
    rng = np.random.default_rng(42)
    ref = st_in[rng.choice(len(st_in), size=min(N_REF, len(st_in)), replace=False)]
    ref_rows = [x for x in (score(int(s)) for s in ref) if x is not None]
    rdf = pd.DataFrame(ref_rows)
    r2_bar = float(np.percentile(rdf["r2"], R2_PCT))
    fp_bar = float(np.percentile(rdf["fp"], FP_PCT))
    own_pass_r2 = float((rdf["r2"] >= r2_bar).mean())
    own_pass_both = float(((rdf["r2"] >= r2_bar) & (rdf["fp"] >= fp_bar)).mean())
    print(f"  own spikes: {len(st_in):,} in window, {len(rdf)} scored")
    print(f"  R2 bar (p{R2_PCT:.0f}) = {r2_bar:.3f} | footprint bar (p{FP_PCT:.0f}) = {fp_bar:.3f}")
    print(f"  own-spike retention: R2 only {own_pass_r2:.1%} -> R2+footprint {own_pass_both:.1%}")

    # ---- impostor test: the competitor's own spikes through our gates ----
    imp_line = "  (no competitor to test)"
    imp_r2 = imp_both = float("nan")
    if comp:
        ist = np.sort(spike_times[spike_clusters == comp[0]])
        ist_in = ist[(ist >= WINDOW_START + 1000) & (ist < WINDOW_END - 1000)]
        ipick = ist_in[rng.choice(len(ist_in), size=min(N_REF, len(ist_in)), replace=False)]
        irows = [x for x in (score(int(s)) for s in ipick) if x is not None]
        idf = pd.DataFrame(irows)
        imp_r2 = float((idf["r2"] >= r2_bar).mean())
        imp_both = float(((idf["r2"] >= r2_bar) & (idf["fp"] >= fp_bar)).mean())
        imp_line = (f"  IMPOSTOR unit {comp[0]} ({len(idf)} spikes) passes our gate: "
                    f"R2 only {imp_r2:.1%} -> R2+footprint {imp_both:.1%}")
    print(imp_line)

    # ---- the detection scan ----
    own_coarse = [float(np.dot(x, tmn)) for x in
                  (traces[PC][int(s) - WINDOW_START - NT0MIN:
                              int(s) - WINDOW_START - NT0MIN + N] for s in st_in[:300])
                  if len(x) == N]
    floor = np.percentile(own_coarse, 10)
    sc_full = correlate(traces[PC].astype(np.float64), tmn, mode="valid", method="fft")
    peaks, _ = find_peaks(sc_full, height=floor, distance=MIN_PEAK_DISTANCE)
    pabs = peaks + WINDOW_START + NT0MIN
    di = np.searchsorted(st, pabs)
    d1 = np.abs(pabs - st[np.clip(di, 0, len(st) - 1)])
    d2 = np.abs(pabs - st[np.clip(di - 1, 0, len(st) - 1)])
    cands = pabs[np.minimum(d1, d2) > MARGIN]
    print(f"  candidates scanned: {len(cands):,}")

    rows = []
    for c_ in cands:
        r = score(int(c_))
        if r:
            rows.append(dict(sample=int(c_), **r))
    cdf = pd.DataFrame(rows)

    pass_r2 = cdf["r2"] >= r2_bar
    pass_both = pass_r2 & (cdf["fp"] >= fp_bar)
    res = {}
    for label, mask in [("R2 only (old behaviour)", pass_r2),
                        ("R2 + footprint (1c)", pass_both)]:
        sub = cdf[mask]
        nb = [neighbor(uid, int(s)) for s in sub["sample"]]
        att = sum(1 for x in nb if x is not None)
        res[label] = dict(det=len(sub), att=att, new=len(sub) - att,
                          frac=att / len(sub) if len(sub) else float("nan"))

    print(f"  {'gate':<26}{'detections':>12}{'attributable':>14}{'frac':>8}{'NEW':>7}")
    for label in res:
        d = res[label]
        print(f"  {label:<26}{d['det']:>12,}{d['att']:>14,}{d['frac']:>8.3f}{d['new']:>7,}")

    cdf["pass_r2"] = pass_r2
    cdf["pass_both"] = pass_both
    cdf["unit"] = uid
    cdf.to_csv(os.path.join(OUT, f"fpgate_unit{uid}_candidates.csv"), index=False)
    all_rows.append(cdf)
    summary.append(dict(
        unit=uid, peak_ch=PC, n_footprint_ch=len(CH), f0_hz=round(f0, 1),
        competitor=comp[0] if comp else None,
        competitor_dist_um=round(comp[1], 1) if comp else None,
        own_in_window=len(st_in), r2_bar=round(r2_bar, 4), fp_bar=round(fp_bar, 4),
        own_retention_r2=round(own_pass_r2, 4), own_retention_both=round(own_pass_both, 4),
        impostor_pass_r2=round(imp_r2, 4), impostor_pass_both=round(imp_both, 4),
        scanned=len(cdf),
        det_r2=res["R2 only (old behaviour)"]["det"],
        att_r2=res["R2 only (old behaviour)"]["att"],
        new_r2=res["R2 only (old behaviour)"]["new"],
        det_both=res["R2 + footprint (1c)"]["det"],
        att_both=res["R2 + footprint (1c)"]["att"],
        new_both=res["R2 + footprint (1c)"]["new"],
    ))
    del traces
    print(f"  unit done in {time.time()-t_unit:.0f}s")

sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "fpgate_detection_summary.csv"), index=False)
pd.concat(all_rows).to_csv(os.path.join(OUT, "fpgate_all_candidates.csv"), index=False)

print(f"\n{'='*74}\nTOTALS ACROSS {len(UNITS)} UNITS")
print(f"  detections  R2 only: {sdf['det_r2'].sum():,}  ->  R2+footprint: {sdf['det_both'].sum():,}")
print(f"  attributable to a neighbour: {sdf['att_r2'].sum():,}  ->  {sdf['att_both'].sum():,}")
print(f"  genuinely new:               {sdf['new_r2'].sum():,}  ->  {sdf['new_both'].sum():,}")
print("\nsaved fpgate_detection_summary.csv, fpgate_all_candidates.csv")
