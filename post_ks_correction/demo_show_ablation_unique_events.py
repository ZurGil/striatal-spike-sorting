"""
Shows the events that the ablation configurations DISAGREE about, for unit
440 -- the point being to see what the corrections actually add and drop.

Two groups (definitions stated explicitly to avoid ambiguity):
  GROUP A -- detected by 'neither' (no wavelet alignment, no whitening)
             but NOT by the full pipeline.   80 events total.
  GROUP B -- detected by the full pipeline (wavelet + whitening)
             but NOT by 'neither'.          100 events total.

Each panel shows, for one event:
  - the RAW snippet at the candidate position (how 'neither' sees it)
  - the WAVELET-ALIGNED snippet (how the full pipeline sees it)
  - unit 440's own template, scaled to best fit the aligned snippet
and is annotated with the R2 each configuration assigned, plus whether a
nearby unit already has a spike there ("corroborated") or not.

Usage: python demo_show_ablation_unique_events.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import make_morlet, wavelet_transform_at, calibrate_reference_phase, coarse_then_fine_shift
from nuisance_model import build_basis

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES = 25, 3.0
UID = 440
BARS = {"full": 0.630, "no_wavelet": 0.714, "no_whiten": 0.687, "neither": 0.725}
N_SHOW_PER_GROUP = 5

templates = np.load(KS + r"\templates.npy")
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

ta = templates[UID]
PC = int(np.argmax(ta.max(axis=0) - ta.min(axis=0)))
TM = ta[:, PC]
f0s = np.linspace(300, 4000, 50)
ps = int(np.ceil(N_CYCLES * FS / (2 * f0s.min()))) + 20
tr = np.zeros(N + 2 * ps); ctr = ps + NT0MIN
tr[ctr - NT0MIN: ctr - NT0MIN + N] = TM
mg = [abs(wavelet_transform_at(tr, make_morlet(f, FS, N_CYCLES)[1], ctr)) for f in f0s]
F0 = float(f0s[int(np.nanargmax(mg))])
_, PSI = make_morlet(F0, FS, n_cycles=N_CYCLES)
REFPH = calibrate_reference_phase(TM, PSI, FS, align_index=NT0MIN)

cdf = pd.read_csv(os.path.join(OUT, f"ablation_overlap_unit{UID}_candidates.csv"))
set_full = set(cdf.loc[cdf["full"] >= BARS["full"], "sample"])
set_neither = set(cdf.loc[cdf["neither"] >= BARS["neither"], "sample"])
group_a = sorted(set_neither - set_full)      # only 'neither'
group_b = sorted(set_full - set_neither)      # only full pipeline
print(f"GROUP A (only 'neither'): {len(group_a)}   GROUP B (only full): {len(group_b)}")


def get_filtered(ch, center, pad=250, fb=60):
    s0 = center - pad - fb
    nr = (center + pad + fb) - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nr * N_CHAN_BIN * ITEMSIZE)
    blk = np.frombuffer(raw, dtype=np.int16).reshape(nr, N_CHAN_BIN)
    return filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64))[fb:-fb] * GAIN_TO_UV


def raw_and_aligned(sample):
    trace = get_filtered(PC, sample, pad=250)
    pad = 250
    raw_snip = trace[pad - NT0MIN: pad - NT0MIN + N].copy()
    r = coarse_then_fine_shift(trace, pad, TM, PSI, F0, FS, reference_phase=REFPH,
                                search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    ip = interp1d(np.arange(lo, hi), trace[lo:hi], kind="cubic", bounds_error=False, fill_value=0.0)
    return raw_snip, ip(tc - NT0MIN + np.arange(N)), r["total_shift"]


def pick(group):
    """A spread: prefer a mix of corroborated and uncorroborated events."""
    sub = cdf[cdf["sample"].isin(group)].sort_values("amp", ascending=False)
    uncorr = sub[~sub.attributable]
    corr = sub[sub.attributable]
    out = list(uncorr["sample"].head(2)) + list(corr["sample"].head(N_SHOW_PER_GROUP - min(2, len(uncorr))))
    return out[:N_SHOW_PER_GROUP]


picks = {"A: only 'neither' (no corrections)": pick(group_a),
         "B: only full (wavelet + whitening)": pick(group_b)}

fig, axes = plt.subplots(2, N_SHOW_PER_GROUP, figsize=(4.1 * N_SHOW_PER_GROUP, 7.6), squeeze=False)
t_ms = (np.arange(N) - NT0MIN) / FS * 1000

for row, (gname, samples) in enumerate(picks.items()):
    for col in range(N_SHOW_PER_GROUP):
        ax = axes[row][col]
        if col >= len(samples):
            ax.axis("off")
            continue
        s = int(samples[col])
        rec = cdf[cdf["sample"] == s].iloc[0]
        raw_s, ali_s, shift = raw_and_aligned(s)
        denom = float(np.dot(TM, TM))
        amp_fit = float(np.dot(ali_s, TM) / denom) if denom > 0 else 0.0

        ax.plot(t_ms, raw_s, color="#999", lw=1.3, label="raw (uncorrected)")
        ax.plot(t_ms, ali_s, color="#111", lw=1.8, label="wavelet-aligned")
        ax.plot(t_ms, TM * amp_fit, color="#1f77b4", lw=1.5, ls="--", label=f"unit {UID} template")
        ax.axhline(0, color="grey", lw=0.4)
        corro = "corroborated by neighbour" if rec.attributable else "NOT corroborated"
        ax.set_title(f"sample {s}\nR2 full={rec['full']:.2f}  neither={rec['neither']:.2f}\n"
                     f"shift={shift:+.2f} samp | {corro}", fontsize=8.5)
        ax.set_xlabel("time (ms)")
        if col == 0:
            ax.set_ylabel(f"{gname}\n\nuV", fontsize=9)
            ax.legend(fontsize=6.5, loc="lower right")

plt.suptitle(f"Unit {UID}: events the corrections DISAGREE about\n"
             f"top row = found only without corrections   |   bottom row = found only with wavelet+whitening",
             fontsize=12.5, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT, f"ablation_unique_events_unit{UID}.png")
plt.savefig(out, dpi=115, bbox_inches="tight")
print("saved", out)

for gname, samples in picks.items():
    print(f"\n{gname}")
    for s in samples:
        rec = cdf[cdf["sample"] == int(s)].iloc[0]
        print(f"  sample {int(s)}: full={rec['full']:.3f} neither={rec['neither']:.3f} "
              f"amp={rec['amp']:.1f}uV corroborated={bool(rec.attributable)}")
