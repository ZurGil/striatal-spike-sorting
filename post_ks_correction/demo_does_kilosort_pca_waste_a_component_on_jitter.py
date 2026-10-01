"""
THE PREMISE TEST for putting sub-sample alignment inside Kilosort.

THE CLAIM BEING TESTED: Kilosort4 clusters spikes on `tF`, the projection of
each spike's snippet onto a learned 6-component basis `wPCA` (shape (6, 61),
found in ops.npy). Kilosort aligns spikes only to the nearest whole sample
(`align_U` uses torch.roll, an integer shift). So every snippet carries up to
half a sample of residual timing jitter.

A time-shifted waveform is, to first order, the waveform plus a multiple of
its own derivative:
    s(t - d) ~ s(t) - d * s'(t)
So if the snippets fed to PCA carry random jitter, the derivative direction
becomes a real axis of variance in the data, and PCA will spend a component
on it. That component then encodes TIMING rather than SHAPE -- and Kilosort
clusters on it, which could split one neuron into several on the basis of
nothing but sub-sample timing.

If that is happening, aligning before feature extraction has a compounding
payoff and the fork is justified. If no component looks like a derivative,
the mechanism is wrong and we should rethink before touching Kilosort.

TWO TESTS:

  Test 1 (geometry) -- is any wPCA component close to the derivative of the
  leading component? Reported as |cosine| between each PC and d(PC0)/dt.
  This is necessary but weak evidence on its own: the derivative of a
  smooth spike is itself smooth, so some overlap is expected.

  Test 2 (sensitivity, the decisive one) -- take real unit templates, shift
  them by sub-sample amounts in [-0.5, +0.5] samples, project onto wPCA, and
  measure how far the feature vector MOVES. Compare that movement to the
  spread of feature vectors ACROSS DIFFERENT REAL UNITS, which is the signal
  Kilosort is trying to cluster on. The ratio answers the question that
  matters: how much of the apparent difference between units is really just
  timing?

Usage: python demo_does_kilosort_pca_waste_a_component_on_jitter.py
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

ops = np.load(KS + r"\ops.npy", allow_pickle=True).item()
wPCA = np.asarray(ops["wPCA"], dtype=float)          # (n_pc, nt)
templates = np.load(KS + r"\templates.npy")
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
NT = wPCA.shape[1]
NPC = wPCA.shape[0]

print(f"Kilosort's feature basis wPCA: {NPC} components x {NT} samples")
print("(every spike is reduced to these 6 numbers per channel before clustering)\n")

# wPCA rows are orthonormal in Kilosort; confirm, because the whole
# interpretation below assumes projections are independent coordinates.
G = wPCA @ wPCA.T
off = np.abs(G - np.diag(np.diag(G))).max()
print(f"basis check: diagonal {np.round(np.diag(G), 4)}, "
      f"largest off-diagonal {off:.2e} -> "
      f"{'orthonormal' if off < 1e-6 else 'NOT orthogonal'}\n")


def deriv(x):
    """Centered first difference, same convention as build_basis."""
    d = np.gradient(x)
    return d / (np.linalg.norm(d) + 1e-12)


# ---------------------------------------------------------------- test 1
print("=" * 74)
print("TEST 1 -- is any component shaped like a time-derivative?")
d0 = deriv(wPCA[0])
rows1 = []
for k in range(NPC):
    pk = wPCA[k] / (np.linalg.norm(wPCA[k]) + 1e-12)
    c = float(abs(np.dot(pk, d0)))
    rows1.append(dict(component=k, abs_cos_with_dPC0=round(c, 4),
                      angle_deg=round(float(np.degrees(np.arccos(min(c, 1.0)))), 1)))
    print(f"  PC{k}: |cos| with d(PC0)/dt = {c:.3f}   "
          f"({np.degrees(np.arccos(min(c,1.0))):.1f} deg apart)")
best = max(rows1[1:], key=lambda r: r["abs_cos_with_dPC0"])
print(f"  -> closest to the derivative direction: PC{best['component']} "
      f"at |cos| {best['abs_cos_with_dPC0']:.3f}")
# how much of the derivative direction does the 6-D basis capture at all?
cap = float(np.linalg.norm(wPCA @ d0))
print(f"  -> the 6-component basis captures {cap**2:.1%} of the derivative "
      f"direction's energy")
print("     (if this were near 0, jitter could not enter the features at all)")

# ---------------------------------------------------------------- test 2
print("\n" + "=" * 74)
print("TEST 2 -- how far does sub-sample jitter move a spike's features,")
print("          compared with how far apart DIFFERENT UNITS sit?")

live, counts = np.unique(spike_clusters, return_counts=True)
live = live[(counts >= 300) & (live < templates.shape[0])]
print(f"  using {len(live)} units with >=300 spikes")

amps = templates.max(axis=1) - templates.min(axis=1)
peak_ch = np.argmax(amps, axis=1)


def shift_template(t, d):
    """Shift a template by d samples (fractional) via cubic interpolation."""
    x = np.arange(len(t))
    ip = interp1d(x, t, kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(x - d)


# feature vector of each unit's own template on its peak channel
feats, kept = [], []
for u in live:
    t = templates[u][:, peak_ch[u]].astype(float)
    n = np.linalg.norm(t)
    if n <= 0:
        continue
    feats.append(wPCA @ (t / n))        # unit-norm so scale is not the story
    kept.append(int(u))
feats = np.asarray(feats)
between = float(np.mean(np.linalg.norm(feats - feats.mean(axis=0), axis=1)))
print(f"  between-unit feature spread (mean distance from centre) = {between:.4f}")

shifts = np.array([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5])
rows2 = []
for u, f_ref in zip(kept, feats):
    t = templates[u][:, peak_ch[u]].astype(float)
    t = t / np.linalg.norm(t)
    for d in shifts:
        ts = shift_template(t, d)
        ts = ts / (np.linalg.norm(ts) + 1e-12)
        move = float(np.linalg.norm(wPCA @ ts - f_ref))
        rows2.append(dict(unit=u, shift_samples=float(d), feature_move=move,
                          move_frac_of_between=move / between))
df2 = pd.DataFrame(rows2)

print(f"\n  {'shift (samples)':>16}{'feature move':>15}{'as % of between-unit spread':>30}")
for d in shifts:
    s = df2[df2["shift_samples"] == d]
    print(f"  {d:>16.2f}{s['feature_move'].mean():>15.4f}"
          f"{s['move_frac_of_between'].mean()*100:>29.1f}%")

half = df2[np.abs(df2["shift_samples"]) == 0.5]
print(f"\n  A half-sample shift -- the WORST residual error Kilosort's integer")
print(f"  alignment can leave -- moves a unit's features by "
      f"{half['move_frac_of_between'].mean()*100:.1f}% of the")
print(f"  distance that separates genuinely different units, on average.")

# how many real unit PAIRS are closer together than jitter moves one unit?
from itertools import combinations
pair_d = np.array([np.linalg.norm(feats[i] - feats[j])
                   for i, j in combinations(range(len(feats)), 2)])
jit = float(half["feature_move"].mean())
frac_closer = float((pair_d < jit).mean())
print(f"\n  Of all {len(pair_d):,} unit pairs, {frac_closer:.1%} are closer together")
print(f"  in feature space than a half-sample shift displaces a single unit.")
print(f"  Those are pairs that sub-sample jitter alone could confuse.")

# which components carry the jitter?
print(f"\n  Which components absorb the shift (mean |change| per component,")
print(f"  at a half-sample shift):")
per_pc = np.zeros(NPC)
for u, f_ref in zip(kept, feats):
    t = templates[u][:, peak_ch[u]].astype(float)
    t = t / np.linalg.norm(t)
    for d in (-0.5, 0.5):
        ts = shift_template(t, d)
        ts = ts / (np.linalg.norm(ts) + 1e-12)
        per_pc += np.abs(wPCA @ ts - f_ref)
per_pc /= (2 * len(kept))
for k in range(NPC):
    bar = "#" * int(round(60 * per_pc[k] / per_pc.max()))
    print(f"    PC{k}: {per_pc[k]:.4f}  {bar}")
print(f"  -> jitter lands hardest on PC{int(np.argmax(per_pc))}")

df2.to_csv(os.path.join(OUT, "kilosort_pca_jitter_sensitivity.csv"), index=False)
pd.DataFrame(rows1).to_csv(os.path.join(OUT, "kilosort_pca_derivative_geometry.csv"),
                            index=False)
print("\nsaved kilosort_pca_jitter_sensitivity.csv, "
      "kilosort_pca_derivative_geometry.csv")
