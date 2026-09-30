"""
Rerun of the 20-minute detection scan with the CORRECTED frequency-selection
rule (wavelet_features.select_probe_frequency, criterion="timing"), because
every detection number produced before used the old argmax|W| rule that was
shown to make the fine stage actively harmful.

Runs both frequency rules through the identical pipeline so the difference
is directly visible. Note that the two configurations which do no alignment
at all (no_wavelet, neither) don't depend on f0, so they appear once.

Configurations:
  full_old    : old f0 (argmax |W|)      + wavelet alignment + whitening
  full_new    : new f0 (argmax f0*|W|)   + wavelet alignment + whitening
  nowhit_old  : old f0 + wavelet alignment, plain fit
  nowhit_new  : new f0 + wavelet alignment, plain fit
  no_wavelet  : no alignment, whitened fit          (f0-independent)
  neither     : no alignment, plain fit             (f0-independent)

Each configuration is judged against its OWN real-spike 25th percentile, so
every configuration is asked the same question: how many candidate events
reach the quality that configuration assigns to this unit's known spikes.

Usage: python demo_20min_rerun_corrected_frequency.py
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

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, MARGIN, MIN_PEAK_DISTANCE = 25, 3.0, 20, 20
RADIUS_UM, JITTER_WINDOW = 60.0, 15
UNITS = [440, 408]
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_REF = 150

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL)

cfgs = {}
for uid in UNITS:
    ta = templates[uid]
    pc = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
    tm = ta[:, pc]
    f_old, _, _ = select_probe_frequency(tm, FS, n_cycles=N_CYCLES, nt0min=NT0MIN, criterion="strength")
    f_new, _, _ = select_probe_frequency(tm, FS, n_cycles=N_CYCLES, nt0min=NT0MIN, criterion="timing")
    _, psi_old = make_morlet(f_old, FS, n_cycles=N_CYCLES)
    _, psi_new = make_morlet(f_new, FS, n_cycles=N_CYCLES)
    s0b, _, qb = build_basis(tm, dt=1.0)
    cfgs[uid] = dict(pc=pc, tm=tm, tmn=tm / (np.linalg.norm(tm) + 1e-12),
                      f_old=f_old, f_new=f_new, psi_old=psi_old, psi_new=psi_new,
                      rp_old=calibrate_reference_phase(tm, psi_old, FS, align_index=NT0MIN),
                      rp_new=calibrate_reference_phase(tm, psi_new, FS, align_index=NT0MIN),
                      s0=s0b, q=qb, st=np.sort(spike_times[spike_clusters == uid]))
    print(f"unit {uid} ch{pc}: old f0={f_old:.0f}Hz (cycle {FS/f_old:.1f} samp), "
          f"new f0={f_new:.0f}Hz (cycle {FS/f_new:.1f} samp)")

channels = sorted({cfgs[u]["pc"] for u in UNITS})
t0 = time.time()
traces = {ch: np.empty(WINDOW_END - WINDOW_START, dtype=np.float64) for ch in channels}
pos = WINDOW_START
while pos < WINDOW_END:
    clen = min(CHUNK, WINDOW_END - pos)
    rlo, rhi = max(0, pos - OVERLAP), min(TOTAL, pos + clen + OVERLAP)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
    lt = pos - rlo
    for ch in channels:
        filt = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64)) * GAIN_TO_UV
        traces[ch][pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt:lt + clen]
    pos += clen
print(f"read+filtered in {time.time()-t0:.0f}s\n")

CONFIGS = ["full_old", "full_new", "nowhit_old", "nowhit_new", "no_wavelet", "neither"]


def snip_raw(trace, s):
    i = s - WINDOW_START - NT0MIN
    if i < 0 or i + N > len(trace):
        return None
    return trace[i:i + N].copy()


def snip_aligned(trace, s, C, which):
    i = s - WINDOW_START
    pad = 200
    if i - pad < 0 or i + pad >= len(trace):
        return None
    loc = trace[i - pad:i + pad]
    psi = C["psi_old"] if which == "old" else C["psi_new"]
    f0 = C["f_old"] if which == "old" else C["f_new"]
    rp = C["rp_old"] if which == "old" else C["rp_new"]
    r = coarse_then_fine_shift(loc, pad, C["tm"], psi, f0, FS, reference_phase=rp,
                                search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(loc):
        return None
    ip = interp1d(np.arange(lo, hi), loc[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(tc - NT0MIN + np.arange(N)), r["total_shift"]


def score(trace, s, C, W):
    sr = snip_raw(trace, s)
    ao = snip_aligned(trace, s, C, "old")
    an = snip_aligned(trace, s, C, "new")
    if sr is None or ao is None or an is None:
        return None
    a_o, sh_o = ao
    a_n, sh_n = an
    fit = lambda x, w: fit_nuisance_prealigned(x, C["s0"], C["q"], whitening_matrix=w)["r_squared"]
    return dict(full_old=fit(a_o, W), full_new=fit(a_n, W),
                nowhit_old=fit(a_o, None), nowhit_new=fit(a_n, None),
                no_wavelet=fit(sr, W), neither=fit(sr, None),
                shift_old=sh_o, shift_new=sh_n)


def neighbor(uid, s):
    mp = channel_positions[peak_ch_all[uid]]
    m = np.abs(spike_times - s) <= JITTER_WINDOW
    best = None
    for oc in sorted(set(spike_clusters[m].tolist()) - {uid}):
        if oc >= n_templ:
            continue
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        if d <= RADIUS_UM and (best is None or d < best[1]):
            best = (oc, d)
    return best


summary = []
for uid in UNITS:
    C = cfgs[uid]
    trace = traces[C["pc"]]
    st = C["st"]
    st_in = st[(st >= WINDOW_START + 1000) & (st < WINDOW_END - 1000)]
    W, _, _ = build_whitening_from_noise(trace[:60000], N, order=4)

    rng = np.random.default_rng(42)
    ref = st_in[rng.choice(len(st_in), size=min(N_REF, len(st_in)), replace=False)]
    ref_sc = {c: [] for c in CONFIGS}
    sh_old, sh_new = [], []
    for s in ref:
        r = score(trace, int(s), C, W)
        if r:
            for c in CONFIGS:
                ref_sc[c].append(r[c])
            sh_old.append(abs(r["shift_old"])); sh_new.append(abs(r["shift_new"]))
    bars = {c: float(np.percentile(ref_sc[c], 25)) for c in CONFIGS}

    own_coarse = [float(np.dot(x, C["tmn"])) for x in
                  (snip_raw(trace, int(s)) for s in st_in[:200]) if x is not None]
    floor = np.percentile(own_coarse, 10)
    sc_full = correlate(trace, C["tmn"], mode="valid", method="fft")
    peaks, _ = find_peaks(sc_full, height=floor, distance=MIN_PEAK_DISTANCE)
    pabs = peaks + WINDOW_START + NT0MIN
    di = np.searchsorted(st, pabs)
    d1 = np.abs(pabs - st[np.clip(di, 0, len(st) - 1)])
    d2 = np.abs(pabs - st[np.clip(di - 1, 0, len(st) - 1)])
    cands = pabs[np.minimum(d1, d2) > MARGIN]

    rows = []
    for c_ in cands:
        r = score(trace, int(c_), C, W)
        if r:
            rows.append(dict(sample=int(c_), **r))
    cdf = pd.DataFrame(rows)
    sets = {c: set(cdf.loc[cdf[c] >= bars[c], "sample"]) for c in CONFIGS}

    print(f"=== unit {uid} (ch{C['pc']}) ===")
    print(f"  own spikes in window: {len(st_in):,} | candidates scanned: {len(cdf):,}")
    print(f"  spurious |shift| on own real spikes: old f0 median={np.median(sh_old):.2f} "
          f"(frac>1samp {np.mean(np.array(sh_old)>1):.3f}) | "
          f"new f0 median={np.median(sh_new):.2f} (frac>1samp {np.mean(np.array(sh_new)>1):.3f})")
    print(f"  {'config':<12}{'bar':>8}{'detections':>12}{'attributable':>14}{'frac':>8}{'NEW':>8}")
    row = dict(uid=uid, own=len(st_in), scanned=len(cdf))
    for c in CONFIGS:
        sub = cdf[cdf["sample"].isin(sets[c])]
        att = sum(1 for s in sub["sample"] if neighbor(uid, int(s)) is not None)
        new = len(sub) - att
        frac = att / len(sub) if len(sub) else float("nan")
        print(f"  {c:<12}{bars[c]:>8.3f}{len(sub):>12,}{att:>14,}{frac:>8.3f}{new:>8,}")
        row[f"det_{c}"] = len(sub); row[f"new_{c}"] = new; row[f"frac_attrib_{c}"] = frac
    j = len(sets["full_new"] & sets["neither"]) / max(len(sets["full_new"] | sets["neither"]), 1)
    j_old = len(sets["full_old"] & sets["neither"]) / max(len(sets["full_old"] | sets["neither"]), 1)
    print(f"  overlap with 'neither': full_old={j_old:.3f}  full_new={j:.3f}")
    print()
    summary.append(row)
    cdf.to_csv(os.path.join(OUT, f"rerun_corrfreq_unit{uid}_candidates.csv"), index=False)

sdf = pd.DataFrame(summary)
sdf.to_csv(os.path.join(OUT, "rerun_corrfreq_summary.csv"), index=False)
print("========== TOTALS ==========")
for c in CONFIGS:
    print(f"  {c:<12} detections={sdf['det_'+c].sum():>8,}   new={sdf['new_'+c].sum():>8,}")
print("\nsaved rerun_corrfreq_summary.csv")
