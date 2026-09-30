"""
STAGE 3 on real data: does Anderson acceleration actually shorten the
template-rebuild loop?

THE MAP. The rebuild is a genuine fixed-point iteration:
    T_{k+1} = G(T_k),   G = "wavelet-align every sampled spike to T_k,
                             then average the aligned snippets"
A real, coherent neuron should have a template that is a fixed point of
this map: align spikes to it, average them, and you get it back.

ONE DELIBERATE CHANGE FROM THE ORIGINAL LOOP. The original rebuild
re-selected the probe frequency f0 from the CURRENT template on every
iteration. That makes the map's own definition change from step to step,
which is no longer a fixed-point iteration at all -- and it is what
produced the drift recorded in section 5e (f0 walking 802 -> 739 -> 676 Hz
while the amplitude decayed 8%). Here f0 is selected ONCE, from Kilosort's
template, with the corrected timing criterion, and then held fixed. That
makes G a fixed map on T alone, which is both the honest thing to
accelerate and the thing that actually converges.

THE COMPARISON. depth=0 is the plain Picard iteration and depth=3 is
Anderson-accelerated, both through the identical code path in
fixed_point_acceleration.iterate_to_fixed_point, both from the identical
starting template, both to the identical tolerance. What is counted is
evaluations of G, because G is the expensive part: one wavelet alignment
per sampled spike per iteration.

Usage: python demo_accelerated_template_rebuild.py
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from fixed_point_acceleration import iterate_to_fixed_point

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES = 25, 3.0
WINDOW_START, WINDOW_SAMPLES, CHUNK, OVERLAP = 50_000_000, 36_000_000, 2_000_000, 400
N_FIT = 250
TOL, MAX_ITER = 1e-6, 60
UNITS = [440, 408, 439]

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
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
        lt = pos - rlo
        out[pos - WINDOW_START: pos - WINDOW_START + clen] = filt[lt:lt + clen]
        pos += clen
    return out


rows = []
for uid in UNITS:
    ta = templates[uid]
    PC = int(np.argmax(ta.max(0) - ta.min(0)))
    T0 = ta[:, PC].astype(np.float64)
    f0, _, _ = select_probe_frequency(T0, FS, n_cycles=N_CYCLES, nt0min=NT0MIN,
                                      criterion="timing")
    _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)

    print(f"\n{'='*74}\nunit {uid}: peak ch{PC}, f0 = {f0:.0f} Hz (held fixed)")
    t0 = time.time()
    trace = load_channel(PC)
    print(f"  loaded channel in {time.time()-t0:.0f}s")

    st = np.sort(spike_times[spike_clusters == uid])
    st_in = st[(st >= WINDOW_START + 1000) & (st < WINDOW_END - 1000)]
    rng = np.random.default_rng(7)
    fit_spikes = st_in[rng.choice(len(st_in), size=min(N_FIT, len(st_in)), replace=False)]
    print(f"  {len(st_in):,} spikes in window, using {len(fit_spikes)} for the rebuild")

    pad = 200

    def G(T):
        """One rebuild step: align every sampled spike to T, average."""
        T = np.asarray(T, dtype=float).ravel()
        refph = calibrate_reference_phase(T, psi, FS, align_index=NT0MIN)
        acc, n = np.zeros(N), 0
        for s in fit_spikes:
            i = int(s) - WINDOW_START
            if i - pad < 0 or i + pad >= len(trace):
                continue
            loc = trace[i - pad:i + pad].astype(np.float64)
            r = coarse_then_fine_shift(loc, pad, T, psi, f0, FS, reference_phase=refph,
                                       search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
            tc = pad + r["total_shift"]
            lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
            if lo < 0 or hi > 2 * pad:
                continue
            ip = interp1d(np.arange(lo, hi), loc[lo:hi], kind="cubic",
                          bounds_error=False, fill_value=0.0)
            acc += ip(tc - NT0MIN + np.arange(N))
            n += 1
        if n == 0:
            return T
        new = acc / n
        # G must be scale-matched to its input, or "T stopped moving" would
        # be measuring the arbitrary microvolt scale of the average rather
        # than the template's shape. Rescale to the input's norm.
        nn = np.linalg.norm(new)
        return new * (np.linalg.norm(T) / nn) if nn > 0 else T

    # Run each method ONCE at the tightest tolerance, keeping the full
    # residual history, then read off how many G-evaluations each needed to
    # first reach a range of tolerances. This is both cheaper and fairer
    # than re-running per tolerance, and it exposes the NOISE FLOOR: with a
    # finite spike sample the averaged template cannot be reproduced to
    # arbitrary precision, so tolerances below that floor are unreachable
    # by any method and a speedup quoted there would be meaningless.
    res = {}
    for label, depth in (("plain Picard", 0), ("Anderson depth=3", 3)):
        t1 = time.time()
        T_fin, n_eval, hist = iterate_to_fixed_point(
            G, T0.copy(), depth=depth, tol=TOL, max_iter=MAX_ITER,
            return_history=True)
        res[label] = dict(n_eval=n_eval, secs=time.time() - t1, T=T_fin,
                          hist=np.asarray(hist), floor=float(np.min(hist)))
        print(f"  {label:<18} G-evals={n_eval:>3}  {res[label]['secs']:>5.1f}s  "
              f"best residual reached={res[label]['floor']:.2e}")

    a, b = res["plain Picard"], res["Anderson depth=3"]

    def evals_to(hist, tol):
        hit = np.nonzero(hist < tol)[0]
        return int(hit[0]) + 1 if len(hit) else None

    print(f"  {'tolerance':>11}{'Picard':>9}{'Anderson':>10}{'speedup':>9}")
    tol_rows = {}
    for tol in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-6):
        ea, eb = evals_to(a["hist"], tol), evals_to(b["hist"], tol)
        sp = (ea / eb) if (ea and eb) else None
        tol_rows[tol] = (ea, eb, sp)
        fa = f"{ea}" if ea else "never"
        fb = f"{eb}" if eb else "never"
        fs = f"{sp:.2f}x" if sp else "-"
        print(f"  {tol:>11.0e}{fa:>9}{fb:>10}{fs:>9}")

    agree = float(np.corrcoef(a["T"], b["T"])[0, 1])
    corr_to_ks = float(np.corrcoef(b["T"], T0)[0, 1])
    print(f"  -> noise floor of this loop: Picard {a['floor']:.1e}, "
          f"Anderson {b['floor']:.1e}")
    print(f"  -> final templates correlate {agree:.6f} with each other, "
          f"{corr_to_ks:.4f} with Kilosort's")

    rec = dict(unit=uid, peak_ch=PC, f0_hz=round(f0, 1), n_spikes=len(fit_spikes),
               picard_floor=a["floor"], anderson_floor=b["floor"],
               templates_agree_r=round(agree, 6), rebuilt_vs_ks_r=round(corr_to_ks, 4),
               rebuilt_vs_ks_ptp_ratio=round(float(np.ptp(b["T"]) / np.ptp(T0)), 4))
    for tol, (ea, eb, sp) in tol_rows.items():
        key = f"tol{tol:.0e}".replace("-", "m").replace("+", "")
        rec[f"picard_{key}"] = ea
        rec[f"anderson_{key}"] = eb
        rec[f"speedup_{key}"] = round(sp, 3) if sp else None
    rows.append(rec)
    del trace

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "accelerated_template_rebuild.csv"), index=False)
print(f"\n{'='*74}")
cols = ["unit", "picard_floor", "anderson_floor",
        "picard_tol1em03", "anderson_tol1em03", "speedup_tol1em03",
        "templates_agree_r"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
print("saved accelerated_template_rebuild.csv")
