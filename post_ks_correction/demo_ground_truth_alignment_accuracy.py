"""
FIRST REAL USE OF THE HARNESS: is the multi-frequency delay estimate actually
better than the single-frequency one?

This question has been open since the estimator was built. On synthetic data
multi-frequency won. On real spikes the two disagree by 0.33-0.44 samples and
there was no way to say which was closer to the truth (RESEARCH_LOG 5w). That
is exactly the gap the harness exists to close.

THE DESIGN (see ground_truth_harness module docstring for why it is valid):

  * background = real recorded voltage from this session, at positions where
    no unit within 60 um has a spike nearby -- real noise with its real
    temporal correlation, not Gaussian white noise.
  * injected spikes = the unit's OWN REAL SNIPPETS, so they carry genuine
    amplitude and shape variability. This is the thing my earlier synthetic
    test got wrong, and the reason it passed while the real-data test failed.
  * ground truth = the DIFFERENCE design. Each real snippet has an unknown
    intrinsic offset e. Inject copies of the SAME snippet shifted by known
    amounts and grade differences; e cancels exactly. So the truth is
    absolute even though real snippets are used.
  * one injected spike per trace, so nothing can interfere with anything.
  * measurement is taken at the known injection index, so both estimators are
    asked exactly the same question -- recover the fractional part. This
    isolates the FINE stage, which is what is being compared; the coarse
    magnitude search is identical for both and would only add common noise.

WHAT IS COMPARED
  single_wide / single_narrow : one probe, the heaviest-weighted frequency of
                                the respective bank (what the pipeline does)
  multi_wide   / multi_narrow : all probes, inverse-variance weighted
  wide   = keep_fraction 0.35 (the original default)
  narrow = keep_fraction 0.85 (the band that cut scatter 2.2x in 5w)

TWO NUMBERS PER CONFIGURATION
  rms   -- root-mean-square error in samples, after cancelling e
  slope -- regression slope of estimate on true shift. 1.0 means the
           estimator tracks real shifts at the correct GAIN. A slope of 0.6
           means it systematically under-corrects by 40%, which an RMS at
           small shifts would partly hide. This matters more than rms for a
           correction that is going to be APPLIED.

Usage: python demo_ground_truth_alignment_accuracy.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from wavelet_features import make_morlet, calibrate_reference_phase
from multifreq_alignment import (build_probe_bank, phase_slope_delay,
                                  single_frequency_delay)
from ground_truth_harness import (find_quiet_positions, extract_real_snippets,
                                   shift_waveform, inject, grade_shift_ladder)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
N_CYCLES, RADIUS_UM = 3.0, 60.0
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_SNIPPETS, PAD = 150, 300
SHIFTS = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
UNITS = [440, 408, 439]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL)


def load_channel(ch):
    out = np.empty(WINDOW_END - WINDOW_START, dtype=np.float32)
    pos = WINDOW_START
    while pos < WINDOW_END:
        clen = min(CHUNK, WINDOW_END - pos)
        rlo, rhi = max(0, pos - OVERLAP), min(TOTAL, pos + clen + OVERLAP)
        with open(DAT_PATH, "rb") as f:
            f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
            raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
        filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
        out[pos - WINDOW_START: pos - WINDOW_START + clen] = filt[pos - rlo: pos - rlo + clen]
        pos += clen
    return out


rows, per_snip = [], []
for uid in UNITS:
    PC = int(peak_ch_all[uid])
    TM = templates[uid][:, PC].astype(float)

    banks = {
        "wide": build_probe_bank(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                 n_probes=5, keep_fraction=0.35),
        "narrow": build_probe_bank(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                   n_probes=5, keep_fraction=0.85),
    }
    print(f"\n{'='*78}\nunit {uid}: peak ch{PC}")
    for nm, bk in banks.items():
        print(f"  {nm:>6} band {bk['band_hz'][0]:.0f}-{bk['band_hz'][1]:.0f} Hz, "
              f"probes {np.round(bk['freqs']).astype(int).tolist()}")

    t0 = time.time()
    trace = load_channel(PC)
    print(f"  loaded channel in {time.time()-t0:.0f}s")

    # units whose spikes would contaminate the background
    mp = channel_positions[PC]
    relevant = {int(u) for u in np.unique(spike_clusters)
                if u < n_templ and
                np.sqrt(((channel_positions[peak_ch_all[u]] - mp) ** 2).sum()) <= RADIUS_UM}

    rng = np.random.default_rng(17)
    quiet = find_quiet_positions(spike_times, spike_clusters, relevant,
                                 WINDOW_START + PAD + 100, WINDOW_END - PAD - 100,
                                 n_positions=N_SNIPPETS, guard_samples=150,
                                 min_separation=1000, rng=rng)
    st_u = np.sort(spike_times[spike_clusters == uid])
    st_in = st_u[(st_u >= WINDOW_START + 100) & (st_u < WINDOW_END - 100)]
    snips, used = extract_real_snippets(trace, WINDOW_START, st_in, nt=N,
                                        nt0min=NT0MIN, max_snippets=N_SNIPPETS,
                                        rng=rng)
    n_use = min(len(quiet), len(snips))
    print(f"  {len(quiet)} quiet injection sites, {len(snips)} real snippets "
          f"-> using {n_use}")
    print(f"  ({len(relevant)} units within {RADIUS_UM:.0f} um treated as contaminants)")

    # single-probe references, one per bank
    singles = {}
    for nm, bk in banks.items():
        k = int(np.argmax(bk["weights"]))
        singles[nm] = (bk["freqs"][k], bk["psis"][k], bk["reference_phases"][k],
                       bk["design"][k])

    results = {f"{m}_{nm}": [] for m in ("single", "multi") for nm in banks}
    for m in range(n_use):
        p = int(quiet[m])
        snip = snips[m]
        seg0 = trace[p - PAD - WINDOW_START: p + PAD - WINDOW_START].astype(np.float64)
        if len(seg0) != 2 * PAD:
            continue
        est = {k: [] for k in results}
        for d in SHIFTS:
            seg = seg0.copy()
            if not inject(seg, shift_waveform(snip, float(d)), PAD, nt0min=NT0MIN):
                est = None
                break
            for nm, bk in banks.items():
                est[f"multi_{nm}"].append(phase_slope_delay(seg, PAD, bk)["delta"])
                est[f"single_{nm}"].append(single_frequency_delay(seg, PAD, bk))
        if est is None:
            continue
        for k in results:
            g = grade_shift_ladder(est[k], SHIFTS)
            results[k].append((g["rms"], g["slope"]))
            per_snip.append(dict(unit=uid, snippet=m, config=k,
                                 rms=g["rms"], slope=g["slope"]))

    print(f"\n  {'config':<16}{'RMS error':>12}{'slope':>9}{'n':>6}")
    for k in ("single_wide", "multi_wide", "single_narrow", "multi_narrow"):
        v = np.array(results[k])
        if len(v) == 0:
            continue
        rms = float(np.median(v[:, 0]))
        slp = float(np.median(v[:, 1]))
        print(f"  {k:<16}{rms:>12.4f}{slp:>9.3f}{len(v):>6}")
        rows.append(dict(unit=uid, config=k, median_rms=round(rms, 4),
                         median_slope=round(slp, 4), n_snippets=len(v)))

    sw = np.array(results["single_narrow"])[:, 0]
    mw = np.array(results["multi_narrow"])[:, 0]
    if len(sw) and len(mw):
        win = float(np.mean(mw < sw))
        print(f"  -> narrow band: multi beats single on {win:.1%} of snippets, "
              f"median RMS {np.median(mw):.4f} vs {np.median(sw):.4f}")
    del trace

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "groundtruth_alignment_accuracy.csv"), index=False)
pd.DataFrame(per_snip).to_csv(
    os.path.join(OUT, "groundtruth_alignment_per_snippet.csv"), index=False)

print(f"\n{'='*78}\nPOOLED ACROSS UNITS (median over snippets and units)")
piv = df.groupby("config")[["median_rms", "median_slope"]].median()
print(piv.to_string())
print("\nsaved groundtruth_alignment_accuracy.csv, "
      "groundtruth_alignment_per_snippet.csv")
