"""
Pillar 1c wired into the scoring path, tested on the exact failure case
that motivated it.

THE FAILURE: on session 20260916_110311, ~99% of everything that scored as
a unit-440 spike on its peak channel (23) was already a detected spike of
unit 439. Units 440 and 439 peak on DIFFERENT channels (23 vs 19/21) and
have genuinely different spatial footprints, so they should not be merged
-- but on the one channel they share, their waveforms correlate at 0.972,
so a single-channel template cannot tell them apart.

THE TEST: score unit 440's own real spikes and unit 439's own real spikes
(the impostors a 440-detector must reject) three ways, and measure how well
each separates the two populations:

  1. single-channel R2      -- what the pipeline does today (peak channel only)
  2. multi-channel R2       -- pillar 1c integrated: the same whitened
                               amplitude+stretch fit, but against the
                               template concatenated across every channel
                               in the footprint, with per-channel whitening
                               assembled block-diagonally
  3. footprint similarity   -- pillar 1c's original metric: cosine
                               similarity of the observed per-channel
                               amplitude pattern against the unit's expected
                               footprint

Separation is reported as AUC (probability a random 440 spike scores above
a random 439 spike). 0.5 = no separation at all, 1.0 = perfect.

Alignment is done once on the peak channel with the CORRECTED frequency
criterion, and the resulting shift is applied to every channel -- one
spike, one time.

Usage: python demo_pillar1c_integration_test.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelet_features import (make_morlet, calibrate_reference_phase,
                               coarse_then_fine_shift, select_probe_frequency)
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise
from spatial_footprint import (multichannel_template, multichannel_snippet_concat,
                                block_diagonal_whitening, unit_footprint,
                                spatial_energy_vector, footprint_similarity)

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DAT_PATH = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\20260916_110311.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, N, ITEMSIZE = 384, 30000.0, 20, 61, 2
SEARCH_RADIUS, N_CYCLES, RADIUS_UM = 25, 3.0, 60.0
TARGET, IMPOSTOR = 440, 439
N_SPIKES = 250

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")

mct = multichannel_template(templates, TARGET, channel_positions, radius_um=RADIUS_UM,
                            min_amplitude_fraction=0.05)
CHANS = mct["channels"]
PC = mct["peak_channel"]
TM = templates[TARGET][:, PC]
print(f"unit {TARGET}: peak ch{PC}, footprint = {mct['n_channels']} channels {list(CHANS)}")
print(f"per-channel template amplitude: {np.round(mct['per_channel_amplitude'], 2)}")

F0, _, _ = select_probe_frequency(TM, FS, n_cycles=N_CYCLES, nt0min=NT0MIN, criterion="timing")
_, PSI = make_morlet(F0, FS, n_cycles=N_CYCLES)
REFPH = calibrate_reference_phase(TM, PSI, FS, align_index=NT0MIN)
print(f"probe frequency (timing criterion): {F0:.0f} Hz")

# bases: build PER CHANNEL then concatenate -- a gradient taken across the
# concatenation seam would be meaningless
s0_single, _, q_single = build_basis(TM, dt=1.0)
s0_parts, q_parts = [], []
for ch in CHANS:
    s0c, _, qc = build_basis(templates[TARGET][:, ch], dt=1.0)
    s0_parts.append(s0c); q_parts.append(qc)
S0_MULTI = np.concatenate(s0_parts)
Q_MULTI = np.concatenate(q_parts)

fp = unit_footprint(templates, TARGET, channel_positions, radius_um=RADIUS_UM)
FP_CHANS = fp["channels"]
FP_EXPECTED = fp["footprint"]


def read_block(center, pad=250, filt_buf=60):
    s0 = center - pad - filt_buf
    nr = (center + pad + filt_buf) - s0
    with open(DAT_PATH, "rb") as f:
        f.seek(int(s0) * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(nr * N_CHAN_BIN * ITEMSIZE)
    return np.frombuffer(raw, dtype=np.int16).reshape(nr, N_CHAN_BIN), filt_buf


def filt_channel(block, ch, filt_buf):
    return filtfilt(b_hp, a_hp, block[:, ch].astype(np.float64))[filt_buf:-filt_buf] * GAIN_TO_UV


# per-channel whitening operators, from real noise on each channel
st_target = np.sort(spike_times[spike_clusters == TARGET])
noise_start = max(int(st_target[0]) + 500000, 10000)
nb, nfb = read_block(noise_start + 30000, pad=30000)
W_ops = []
for ch in CHANS:
    ntr = filt_channel(nb, ch, nfb)
    Wc, _, _ = build_whitening_from_noise(ntr, N, order=4)
    W_ops.append(Wc)
W_MULTI = block_diagonal_whitening(W_ops)
W_SINGLE = W_ops[list(CHANS).index(PC)]
print(f"built {len(W_ops)} per-channel whitening operators; "
      f"block-diagonal operator is {W_MULTI.shape[0]}x{W_MULTI.shape[1]}")


def score_spike(sample):
    """Align once on the peak channel, then score all three ways."""
    block, fb = read_block(int(sample), pad=250)
    pad = 250
    trace_pc = filt_channel(block, PC, fb)
    r = coarse_then_fine_shift(trace_pc, pad, TM, PSI, F0, FS, reference_phase=REFPH,
                                search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    tc = pad + r["total_shift"]
    lo, hi = int(np.floor(tc - NT0MIN)) - 2, int(np.ceil(tc - NT0MIN + N)) + 2
    if lo < 0 or hi > len(trace_pc):
        return None
    pos = tc - NT0MIN + np.arange(N)

    snips = []
    for ch in CHANS:
        tr = filt_channel(block, ch, fb)
        ip = interp1d(np.arange(lo, hi), tr[lo:hi], kind="cubic",
                      bounds_error=False, fill_value=0.0)
        snips.append(ip(pos))
    multi = np.stack(snips, axis=1)                       # (n_samples, n_channels)
    concat = multichannel_snippet_concat(multi)

    r2_single = fit_nuisance_prealigned(multi[:, list(CHANS).index(PC)],
                                         s0_single, q_single,
                                         whitening_matrix=W_SINGLE)["r_squared"]
    r2_multi = fit_nuisance_prealigned(concat, S0_MULTI, Q_MULTI,
                                        whitening_matrix=W_MULTI)["r_squared"]

    fp_snips = []
    for ch in FP_CHANS:
        tr = filt_channel(block, ch, fb)
        ip = interp1d(np.arange(lo, hi), tr[lo:hi], kind="cubic",
                      bounds_error=False, fill_value=0.0)
        fp_snips.append(ip(pos))
    fp_obs = spatial_energy_vector(np.stack(fp_snips, axis=1))
    fp_sim = footprint_similarity(fp_obs, FP_EXPECTED)

    return dict(r2_single=r2_single, r2_multi=r2_multi, fp_sim=fp_sim)


def auc(pos_scores, neg_scores):
    """P(random positive > random negative), rank-based."""
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


rng = np.random.default_rng(0)
results = {}
for label, uid in [("target_440", TARGET), ("impostor_439", IMPOSTOR)]:
    st = np.sort(spike_times[spike_clusters == uid])
    pick = st[rng.choice(len(st), size=min(N_SPIKES, len(st)), replace=False)]
    rows = [s for s in (score_spike(int(x)) for x in pick) if s is not None]
    results[label] = pd.DataFrame(rows)
    print(f"{label}: scored {len(rows)} spikes")

t, i = results["target_440"], results["impostor_439"]
print("\n================ SEPARATION OF UNIT 440 FROM UNIT 439 ================")
print(f"{'scorer':<24}{'440 mean':>10}{'439 mean':>10}{'AUC':>8}")
for col, name in [("r2_single", "single-channel R2"),
                  ("r2_multi", "MULTI-channel R2 (1c)"),
                  ("fp_sim", "footprint similarity")]:
    print(f"{name:<24}{t[col].mean():>10.3f}{i[col].mean():>10.3f}{auc(t[col], i[col]):>8.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (col, name) in zip(axes, [("r2_single", "single-channel R2 (current)"),
                                   ("r2_multi", "multi-channel R2 (pillar 1c)"),
                                   ("fp_sim", "footprint similarity (pillar 1c)")]):
    lo = min(t[col].min(), i[col].min()); hi = max(t[col].max(), i[col].max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(t[col], bins=bins, alpha=0.6, color="#1f77b4", label=f"unit {TARGET} (real)")
    ax.hist(i[col], bins=bins, alpha=0.6, color="#d62728", label=f"unit {IMPOSTOR} (impostor)")
    ax.set_title(f"{name}\nAUC = {auc(t[col], i[col]):.3f}", fontsize=11)
    ax.set_xlabel(col)
    ax.legend(fontsize=8)
axes[0].set_ylabel("count")
plt.suptitle(f"Can each scorer tell unit {TARGET}'s spikes from unit {IMPOSTOR}'s? "
             f"(templates correlate 0.972 on the shared channel)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUT, "pillar1c_integration_separation.png")
plt.savefig(out, dpi=115, bbox_inches="tight")
pd.concat([t.assign(which="440"), i.assign(which="439")]).to_csv(
    os.path.join(OUT, "pillar1c_integration_scores.csv"), index=False)
print(f"\nsaved {out}")
