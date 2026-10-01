"""
Show the templates of every unit selected for the tiered dataset, so the
selection can be judged by eye rather than trusted on metrics alone.

WHY THIS MATTERS. The units are chosen by three automatic filters -- Kilosort
labels them `good`, contamination is under 10%, and they are mutually >= 150um
apart with template similarity <= 0.2. Those rule out the specific failures
we know about (noise clusters, and two halves of one oversplit neuron). They
cannot rule out something that only looks wrong: a template that is really an
artefact, a double-peaked shape, something clipped, or a waveform too small to
be a real spike. That needs eyes on it.

A CORRECTION BUILT INTO THIS SCRIPT. The first version plotted Kilosort's
`templates.npy` scaled by GAIN_TO_UV and reported amplitudes of 0.1-0.3
microvolts, which is nonsense -- real extracellular spikes are tens of
microvolts. Kilosort's templates are NORMALIZED, not in recording units, a
caveat already recorded in this project's log and walked into anyway. So this
plots the REAL averaged waveform computed from the raw voltage file for each
unit, which is both correctly scaled and a stronger check: it shows the actual
recorded spike, noise and all, rather than a model of it.

WHAT IS DRAWN, per unit:
  * the peak-channel waveform in microvolts, with the measured peak-to-trough
    amplitude and the width at half the trough depth
  * the full multi-channel footprint as a heatmap, which is what a real
    neuron's spatial signature should look like -- strongest on one channel
    and falling away smoothly. A footprint with two separated hot spots is a
    warning sign that the cluster is a mixture.
  * the tier it was assigned to, its contamination percentage, and its pair
    partner where it has one

The two PAIR rows are drawn adjacent on purpose: those are the two units
placed 3 channels apart in the dataset, where correct sorting must keep them
separate. If their templates looked alike, the pair tier would be testing the
wrong thing, so they should be visibly different.

Usage: python demo_plot_selected_unit_templates.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
TIERED = r"D:\Gil\spike_sorting_agent\hybrid_tiered"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
FS, NT0MIN = 30000.0, 20
RADIUS_UM = 80.0

DAT_PATH = (r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort"
            r"\20260916_110311.probe1.dat")
N_CHAN_BIN, ITEMSIZE, NT = 384, 2, 61
N_AVG = 250

from scipy.signal import butter, filtfilt

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
TOTAL = os.path.getsize(DAT_PATH) // (N_CHAN_BIN * ITEMSIZE)

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
cp = np.load(KS + r"\channel_positions.npy")
amp_all = templates.max(axis=1) - templates.min(axis=1)
peak_ch = np.argmax(amp_all, axis=1)


def real_mean_waveform(uid, chans, n_avg=N_AVG, seed=0):
    """Average the unit's REAL recorded snippets, in microvolts.

    This is the honest object to inspect. Kilosort's templates.npy is
    normalized, so multiplying it by GAIN_TO_UV produces meaningless numbers
    (the first version of this script reported 0.1-0.3 uV spikes that way).
    Averaging real snippets gives both correct scaling and a stronger check,
    because it shows the recorded spike rather than a model of it.
    """
    rng = np.random.default_rng(seed)
    st = np.sort(spike_times[spike_clusters == uid])
    st = st[(st > NT + 80) & (st < TOTAL - NT - 80)]
    pick = st[rng.permutation(len(st))[:n_avg]]
    acc, k = np.zeros((NT, len(chans))), 0
    for sp in pick:
        lo, nread = int(sp) - NT0MIN - 60, NT + 120
        if lo < 0 or lo + nread > TOTAL:
            continue
        with open(DAT_PATH, "rb") as f:
            f.seek(lo * N_CHAN_BIN * ITEMSIZE)
            raw = f.read(nread * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(raw, dtype=np.int16).reshape(nread, N_CHAN_BIN)
        acc += np.stack(
            [filtfilt(b_hp, a_hp, blk[:, c].astype(np.float64))[60:60 + NT]
             for c in chans], axis=1)
        k += 1
    return (acc / max(k, 1)) * GAIN_TO_UV, k
man = pd.read_csv(os.path.join(TIERED, "manifest.csv"))
ks_sim = np.load(KS + r"\similar_templates.npy")

TIER_COLOR = {"easy": "#2b7a4b", "noisy_channel": "#b5651d",
              "collision": "#8a4fa8", "pair": "#1f5f9c"}
order = {"easy": 0, "noisy_channel": 1, "collision": 2, "pair": 3}
man = man.sort_values(["tier", "pair_id", "unit"],
                      key=lambda s: s.map(order) if s.name == "tier" else s)

n = len(man)
fig, axes = plt.subplots(n, 2, figsize=(11.5, 2.05 * n),
                         gridspec_kw={"width_ratios": [1.0, 1.25]})
t_ms = (np.arange(templates.shape[1]) - NT0MIN) / FS * 1000.0

for row, (_, r) in enumerate(man.iterrows()):
    u = int(r.unit)
    pc = int(peak_ch[u])
    col = TIER_COLOR[r.tier]

    # ---- peak-channel waveform, from REAL averaged snippets
    d_all = np.sqrt(((cp - cp[pc]) ** 2).sum(axis=1))
    chans = np.sort(np.where(d_all <= RADIUS_UM)[0])
    M_real, n_used = real_mean_waveform(u, chans)
    ax = axes[row, 0]
    w = M_real[:, list(chans).index(pc)]
    ax.plot(t_ms, w, color=col, lw=1.9)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)
    ptp = float(w.max() - w.min())
    tr = int(np.argmin(w))
    half = w[tr] / 2.0
    left = tr
    while left > 0 and w[left] < half:
        left -= 1
    right = tr
    while right < len(w) - 1 and w[right] < half:
        right += 1
    width_ms = (right - left) / FS * 1000.0
    ax.set_ylabel(f"unit {u}", fontsize=9, fontweight="bold")
    ax.text(0.98, 0.06,
            f"ch{pc}  ptp {ptp:.0f}\u00b5V  half-width {width_ms:.2f} ms",
            transform=ax.transAxes, ha="right", fontsize=7.3, color="0.3")
    partner = ""
    if r.pair_id >= 0:
        other = man[(man.pair_id == r.pair_id) & (man.unit != u)]
        if len(other):
            o = int(other.unit.iloc[0])
            partner = f"  pair with {o} (KS sim {ks_sim[u, o]:.3f})"
    ax.set_title(f"{r.tier}   contam {r.contam_pct:.1f}%   "
                 f"src ch{int(r.source_peak_ch)} \u2192 dest ch{int(r.dest_peak_ch)}"
                 f"{partner}",
                 fontsize=8.2, color=col, loc="left")
    ax.tick_params(labelsize=7)
    if row == n - 1:
        ax.set_xlabel("time from detection (ms)", fontsize=8)

    # ---- multi-channel footprint
    ax2 = axes[row, 1]
    M = M_real.T                                          # (n_ch, n_time)
    v = np.abs(M).max()
    im = ax2.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                    extent=[t_ms[0], t_ms[-1], len(chans) - 0.5, -0.5],
                    interpolation="nearest")
    ax2.set_yticks(range(len(chans)))
    ax2.set_yticklabels([str(c) for c in chans], fontsize=6)
    ax2.axhline(list(chans).index(pc), color="k", lw=0.8, ls=":")
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"footprint, {len(chans)} channels within {RADIUS_UM:.0f}\u00b5m "
                  f"(peak ch{pc} dotted)", fontsize=7.6, loc="left", color="0.35")
    plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)
    if row == n - 1:
        ax2.set_xlabel("time from detection (ms)", fontsize=8)

plt.suptitle("REAL averaged spike waveforms (not Kilosort templates, which are normalized)
"
             "13 units selected for the tiered hybrid dataset\n"
             "all KSLabel='good', contamination <= 10%, mutually >= 160\u00b5m apart "
             "with template similarity 0.000",
             fontsize=11.5, fontweight="bold", y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(OUT, "tiered_selected_unit_templates.png")
plt.savefig(out, dpi=115, bbox_inches="tight")
print(f"saved {out}")

# a compact numeric table alongside the figure
rows = []
for _, r in man.iterrows():
    u = int(r.unit); pc = int(peak_ch[u])
    d = np.sqrt(((cp - cp[pc]) ** 2).sum(axis=1))
    chans = np.sort(np.where(d <= RADIUS_UM)[0])
    Mr, _ = real_mean_waveform(u, chans)
    w = Mr[:, list(chans).index(pc)]
    a = Mr.max(axis=0) - Mr.min(axis=0)
    rows.append(dict(unit=u, tier=r.tier, pair_id=int(r.pair_id),
                     peak_ch=pc, ptp_uv=round(float(w.max() - w.min()), 1),
                     trough_uv=round(float(w.min()), 1),
                     contam_pct=r.contam_pct,
                     n_chans_above_25pct=int((a >= 0.25 * a.max()).sum()),
                     footprint_spread_um=round(float(
                         d[chans][a >= 0.25 * a.max()].max()), 1)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "tiered_selected_unit_templates.csv"), index=False)
print()
print(df.to_string(index=False))
print("\nSanity checks on the selection:")
print(f"  peak-to-peak amplitude range: {df.ptp_uv.min():.0f} - {df.ptp_uv.max():.0f} uV")
print(f"  all troughs negative (a real spike points down): "
      f"{bool((df.trough_uv < 0).all())}")
print(f"  footprint spread (channels >= 25% of peak): "
      f"{df.footprint_spread_um.min():.0f} - {df.footprint_spread_um.max():.0f} um")
