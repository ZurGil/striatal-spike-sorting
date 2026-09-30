"""
Ablation, but asking WHICH events each configuration finds -- not just how
many. Raised directly: two configurations can report similar detection
counts while detecting largely different events, and a weaker
configuration's extra detections may simply be noise.

Unit 440 only -- the one unit whose numbers were stable across independent
runs (unit 408's channel has four crowding neighbours and its counts swing
with the bar, so membership analysis there would be measuring the
instability, not the components).

For each configuration (full / no_wavelet / no_whiten / neither), scored at
its own real-spike 25th-percentile bar, this computes:

  - the detection SET, and pairwise overlap between configurations
  - a PRECISION PROXY per configuration: the fraction of its detections
    that coincide (within 15 samples) with a real spike already detected
    under a nearby unit -- overwhelmingly unit 439 here. A real spike of
    the 439/440 pair is usually already in Kilosort's output as 439, so a
    configuration detecting more noise should show a LOWER attributable
    fraction.
  - what is unique to the full pipeline vs unique to 'neither', and how
    those unique events score on that same precision proxy

Usage: python demo_ablation_overlap_unit440.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, find_peaks
from scipy.interpolate import interp1d

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, MARGIN, MIN_PEAK_DISTANCE = 25, 3.0, 20, 20
RADIUS_UM, JITTER_WINDOW = 60.0, 15
UID = 440
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_REF = 120
CONFIGS = ["full", "no_wavelet", "no_whiten", "neither"]
MATCH_TOL = 20  # samples, for calling two configs' detections "the same event"

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL_SAMPLES = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)
WINDOW_END = min(WINDOW_START + WINDOW_SAMPLES, TOTAL_SAMPLES)

ta = templates[UID]
PC = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
TM = ta[:, PC]
TMN = TM / (np.linalg.norm(TM) + 1e-12)
f0s = np.linspace(300, 4000, 50)
ps = int(np.ceil(N_CYCLES * FS / (2 * f0s.min()))) + 20
tr = np.zeros(N + 2 * ps); ctr = ps + NT0MIN
tr[ctr - NT0MIN: ctr - NT0MIN + N] = TM
mg = [abs(wavelet_transform_at(tr, make_morlet(f, FS, N_CYCLES)[1], ctr)) for f in f0s]
F0 = float(f0s[int(np.nanargmax(mg))])
_, PSI = make_morlet(F0, FS, n_cycles=N_CYCLES)
REFPH = calibrate_reference_phase(TM, PSI, FS, align_index=NT0MIN)
S0B, _, QB = build_basis(TM, dt=1.0)
ST = np.sort(spike_times[spike_clusters == UID])

t0 = time.time()
trace = np.empty(WINDOW_END - WINDOW_START, dtype=np.float64)
pos = WINDOW_START
while pos < WINDOW_END:
    clen = min(CHUNK, WINDOW_END - pos)
    rlo, rhi = max(0, pos - OVERLAP), min(TOTAL_SAMPLES, pos + clen + OVERLAP)
    with open(DAT_PATH, "rb") as f:
        f.seek(int(rlo) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read((rhi - rlo) * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(rhi - rlo, N_CHAN_BIN)
    filt = filtfilt(b_hp, a_hp, blk[:, PC].astype(np.float64)) * GAIN_TO_UV
    lt = pos - rlo
    trace[pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt: lt + clen]
    pos += clen
print(f"read+filtered ch{PC} in {time.time()-t0:.0f}s")

W, _, _ = build_whitening_from_noise(trace[:60000], N, order=4)


def snip_raw(s):
    i = s - WINDOW_START - NT0MIN
    if i < 0 or i + N > len(trace):
        return None
    return trace[i:i + N].copy()


def snip_aligned(s):
    i = s - WINDOW_START
    pad = 200
    if i - pad < 0 or i + pad >= len(trace):
        return None
    loc = trace[i - pad:i + pad]
    r = coarse_then_fine_shift(loc, pad, TM, PSI, F0, FS, reference_phase=REFPH,
                                search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(loc):
        return None
    ip = interp1d(np.arange(lo, hi), loc[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(tc - NT0MIN + np.arange(N))


def all_cfgs(s):
    sa, sr = snip_aligned(s), snip_raw(s)
    if sa is None or sr is None:
        return None
    return dict(
        full=fit_nuisance_prealigned(sa, S0B, QB, whitening_matrix=W)["r_squared"],
        no_whiten=fit_nuisance_prealigned(sa, S0B, QB, whitening_matrix=None)["r_squared"],
        no_wavelet=fit_nuisance_prealigned(sr, S0B, QB, whitening_matrix=W)["r_squared"],
        neither=fit_nuisance_prealigned(sr, S0B, QB, whitening_matrix=None)["r_squared"],
        amp=float(np.max(sa) - np.min(sa)))


# nearby units' spikes -- the attribution reference
my_pos = channel_positions[peak_ch_all[UID]]
near_units = [u for u in range(n_templ)
              if u != UID and np.sqrt(((channel_positions[peak_ch_all[u]] - my_pos) ** 2).sum()) <= RADIUS_UM]
near_spk = np.sort(spike_times[np.isin(spike_clusters, near_units)])
print(f"attribution reference: {len(near_units)} units within {RADIUS_UM:.0f}um, {len(near_spk):,} spikes")


def attributable(s):
    if len(near_spk) == 0:
        return False
    j = np.searchsorted(near_spk, s)
    d = min(abs(s - near_spk[min(j, len(near_spk) - 1)]), abs(s - near_spk[max(j - 1, 0)]))
    return bool(d <= JITTER_WINDOW)


ST_in = ST[(ST >= WINDOW_START + 1000) & (ST < WINDOW_END - 1000)]
rng = np.random.default_rng(42)
ref = ST_in[rng.choice(len(ST_in), size=min(N_REF, len(ST_in)), replace=False)]
ref_scores = {c: [] for c in CONFIGS}
for s in ref:
    sc = all_cfgs(int(s))
    if sc:
        for c in CONFIGS:
            ref_scores[c].append(sc[c])
bars = {c: float(np.percentile(ref_scores[c], 25)) for c in CONFIGS}
print("bars (own-spike p25):", {c: round(bars[c], 3) for c in CONFIGS})

own_coarse = [float(np.dot(x, TMN)) for x in (snip_raw(int(s)) for s in ST_in[:200]) if x is not None]
floor = np.percentile(own_coarse, 10)
score = correlate(trace, TMN, mode="valid", method="fft")
peaks, _ = find_peaks(score, height=floor, distance=MIN_PEAK_DISTANCE)
pabs = peaks + WINDOW_START + NT0MIN
di = np.searchsorted(ST, pabs)
d1 = np.abs(pabs - ST[np.clip(di, 0, len(ST) - 1)])
d2 = np.abs(pabs - ST[np.clip(di - 1, 0, len(ST) - 1)])
cands = pabs[np.minimum(d1, d2) > MARGIN]
print(f"candidate locations: {len(cands):,}")

rows = []
for c_ in cands:
    sc = all_cfgs(int(c_))
    if sc:
        rows.append(dict(sample=int(c_), attributable=attributable(int(c_)), **sc))
cdf = pd.DataFrame(rows)
cdf.to_csv(os.path.join(OUT, f"ablation_overlap_unit{UID}_candidates.csv"), index=False)

sets = {c: set(cdf.loc[cdf[c] >= bars[c], "sample"].tolist()) for c in CONFIGS}

print("\n=== DETECTION COUNTS and PRECISION PROXY ===")
print(f"{'config':<12}{'n_det':>8}{'attributable':>14}{'frac_attrib':>13}{'median_amp_uV':>15}")
for c in CONFIGS:
    sub = cdf[cdf["sample"].isin(sets[c])]
    na = int(sub.attributable.sum())
    print(f"{c:<12}{len(sub):>8,}{na:>14,}{na/len(sub) if len(sub) else float('nan'):>13.3f}"
          f"{sub.amp.median():>15.1f}")

print("\n=== PAIRWISE OVERLAP (events shared / union) ===")
print(f"{'':<12}" + "".join(f"{c:>13}" for c in CONFIGS))
for c1 in CONFIGS:
    line = f"{c1:<12}"
    for c2 in CONFIGS:
        inter = len(sets[c1] & sets[c2])
        union = len(sets[c1] | sets[c2])
        line += f"{inter/union if union else float('nan'):>13.3f}"
    print(line)

core = set.intersection(*[sets[c] for c in CONFIGS])
print(f"\ncore set found by ALL four configurations: {len(core):,}")
sub_core = cdf[cdf["sample"].isin(core)]
print(f"  attributable fraction of core: {sub_core.attributable.mean():.3f}")

only_full = sets["full"] - sets["neither"]
only_neither = sets["neither"] - sets["full"]
print(f"\n=== UNIQUE EVENTS: full vs neither ===")
for name, s in [("only full (wavelet+precision)", only_full), ("only neither (no correction)", only_neither)]:
    sub = cdf[cdf["sample"].isin(s)]
    if len(sub) == 0:
        print(f"  {name}: 0")
        continue
    print(f"  {name}: {len(sub):,} events | attributable {sub.attributable.sum():,} "
          f"({sub.attributable.mean():.3f}) | median amp {sub.amp.median():.1f} uV")

only_full_vs_nw = sets["full"] - sets["no_wavelet"]
only_nw_vs_full = sets["no_wavelet"] - sets["full"]
print(f"\n=== UNIQUE EVENTS: full vs no_wavelet (isolating wavelet alignment) ===")
for name, s in [("only full", only_full_vs_nw), ("only no_wavelet", only_nw_vs_full)]:
    sub = cdf[cdf["sample"].isin(s)]
    if len(sub) == 0:
        print(f"  {name}: 0")
        continue
    print(f"  {name}: {len(sub):,} events | attributable {sub.attributable.sum():,} "
          f"({sub.attributable.mean():.3f}) | median amp {sub.amp.median():.1f} uV")
print("\nsaved candidate-level scores")
