"""
The actual merge test for the candidate pairs flagged in 5i (near-identical
template shapes: 306-305, 306-298, 332-333, 332-334, 333-334, 303-299,
303-306, 342-341, 313-318) -- per the original design doc's own
"Stage 2: suggest, don't auto-merge" plan: does the combined spike train
look like ONE real neuron, or two?

THE LOGIC: a single real neuron cannot fire twice within its own absolute
refractory period (~1.5ms used here) -- that's a hard biophysical limit,
not a clustering choice. So if unit A and unit B are actually ONE neuron
that Kilosort split into two clusters, MERGING their spike trains must
still respect that limit: there should be few/no pairs of spikes (one
from A, one from B) closer together than ~1.5ms, because those would
represent one real neuron firing twice almost instantly, which real
neurons cannot do. If merging creates MANY such violations, A and B are
much more likely to be two genuinely different, independent neurons that
happen to have similar templates and can legitimately fire close together
in time (no refractory constraint BETWEEN different real neurons).

Reports, per pair:
  - refractory violation rate of A alone, B alone (sanity: should already
    be low for two clusters KS4 itself considered real units)
  - refractory violation rate of the MERGED (A+B) train -- the key number
  - the cross-correlogram (CCG) shape -- a real split-neuron case should
    show an asymmetric peak (B tends to follow A at short, burst-
    compatible lags), not a flat/symmetric relationship

This is diagnostic only -- suggests candidates for manual review, per the
design doc's own plan. Nothing here merges anything in Kilosort/Phy.

Usage: python demo_merge_candidate_ccg_test.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
FS = 30000.0
REFRACTORY_MS = 1.5
CCG_WINDOW_MS = 25.0
BIN_MS = 0.5

spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
templates = np.load(VR + r"\templates.npy")
info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)

PAIRS = [(306, 305), (306, 298), (332, 333), (332, 334), (333, 334),
         (303, 299), (303, 306), (342, 341), (313, 318)]


def get_times_ms(uid):
    return np.sort(spike_times[spike_clusters == uid]) / FS * 1000.0


def refractory_violation_rate(times_ms, refractory_ms=REFRACTORY_MS):
    if len(times_ms) < 2:
        return 0.0, 0
    isis = np.diff(times_ms)
    n_viol = int(np.sum(isis < refractory_ms))
    return n_viol / len(times_ms), n_viol


def crosscorrelogram(times_a_ms, times_b_ms, window_ms=CCG_WINDOW_MS, bin_ms=BIN_MS):
    """For every spike in A, time differences to all B spikes within
    +/-window_ms. Positive lag = B occurs AFTER A."""
    diffs = []
    j_lo = 0
    times_b_ms = np.asarray(times_b_ms)
    for ta in times_a_ms:
        lo = ta - window_ms
        hi = ta + window_ms
        j_lo = np.searchsorted(times_b_ms, lo, side="left")
        j_hi = np.searchsorted(times_b_ms, hi, side="right")
        if j_hi > j_lo:
            diffs.append(times_b_ms[j_lo:j_hi] - ta)
    if not diffs:
        return np.array([])
    return np.concatenate(diffs)


def autocorrelogram_diffs(times_ms, window_ms=CCG_WINDOW_MS):
    diffs = []
    n = len(times_ms)
    for i, t in enumerate(times_ms):
        lo = np.searchsorted(times_ms, t - window_ms, side="left")
        hi = np.searchsorted(times_ms, t + window_ms, side="right")
        seg = times_ms[lo:hi] - t
        diffs.append(seg[seg != 0])
    return np.concatenate(diffs) if diffs else np.array([])


results = []
fig, axes = plt.subplots(3, len(PAIRS), figsize=(4.3 * len(PAIRS), 11))
bins = np.arange(-CCG_WINDOW_MS, CCG_WINDOW_MS + BIN_MS, BIN_MS)

for col, (a, b) in enumerate(PAIRS):
    ta = get_times_ms(a)
    tb = get_times_ms(b)
    merged = np.sort(np.concatenate([ta, tb]))

    rate_a, n_a = refractory_violation_rate(ta)
    rate_b, n_b = refractory_violation_rate(tb)
    rate_m, n_m = refractory_violation_rate(merged)
    n_cross_viol = n_m - n_a - n_b  # violations created SPECIFICALLY by interleaving A and B

    ccg = crosscorrelogram(ta, tb)
    acg_merged_diffs = autocorrelogram_diffs(merged)

    shape_corr = np.corrcoef(templates[a][:, peak_ch_all[a]], templates[b][:, peak_ch_all[b]])[0, 1]

    results.append(dict(unit_a=a, unit_b=b, n_a=len(ta), n_b=len(tb), shape_corr=shape_corr,
                         viol_rate_a=rate_a, viol_rate_b=rate_b, viol_rate_merged=rate_m,
                         n_cross_violations=n_cross_viol,
                         cross_viol_rate=n_cross_viol / len(merged) if len(merged) else np.nan))

    ax = axes[0, col]
    ax.hist(ccg, bins=bins, color="#1f77b4")
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvspan(-REFRACTORY_MS, REFRACTORY_MS, color="red", alpha=0.1)
    ax.set_title(f"{a} vs {b}: CROSS-correlogram\n(template shape corr={shape_corr:.3f})", fontsize=9.5)
    ax.set_xlabel("lag, B relative to A (ms)")

    ax = axes[1, col]
    ax.hist(acg_merged_diffs, bins=bins, color="#2ca02c")
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvspan(-REFRACTORY_MS, REFRACTORY_MS, color="red", alpha=0.15)
    ax.set_title(f"MERGED ({a}+{b}) autocorrelogram\ncross-violations={n_cross_viol} "
                 f"({n_cross_viol/len(merged)*100:.2f}% of merged spikes)", fontsize=9.5)
    ax.set_xlabel("lag (ms)")

    ax = axes[2, col]
    ax.axis("off")
    ax.text(0, 1.0,
        f"unit {a}: n={len(ta)}, own viol rate={rate_a*100:.2f}%\n"
        f"unit {b}: n={len(tb)}, own viol rate={rate_b*100:.2f}%\n"
        f"MERGED: n={len(merged)}, viol rate={rate_m*100:.2f}%\n"
        f"violations created BY merging: {n_cross_viol}\n"
        f"({n_cross_viol/len(merged)*100:.3f}% of merged spikes)\n\n"
        f"template shape corr: {shape_corr:.3f}",
        fontsize=9.5, va="top", family="monospace", transform=ax.transAxes)

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "merge_candidate_ccg_test.csv"), index=False)

plt.suptitle("Merge test: cross-correlograms and merged-train refractory violations for candidate pairs", fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "merge_candidate_ccg_test.png")
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("\nsaved", out_path)
