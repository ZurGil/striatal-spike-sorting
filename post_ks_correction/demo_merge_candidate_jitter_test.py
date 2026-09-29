"""
Proper significance test for the merge candidates in 5j, addressing a real
gap flagged directly: the earlier "expected chance" number
(n_a*n_b*2w/T) assumes each unit fires at a CONSTANT rate for the whole
3-hour session. Real neurons don't -- up/down states, bursting, task
modulation all make the true rate non-stationary. Two genuinely
independent neurons that are simply co-modulated by the same slow process
(both busier during the same behavioral epochs, say) could show an
"excess" of near-simultaneous spikes that has nothing to do with being the
same cell -- the naive constant-rate calculation can't tell the
difference, and with a small number of spikes, an apparently "clean dip"
could just be a low-count fluke.

FIX: a jitter-based permutation test (standard method, e.g. Fujisawa et
al. 2008's approach to CCG significance). For each pair, jitter every
spike of the smaller-count unit by a random offset within +/-10ms (much
bigger than the 1.5ms refractory window, so any GENUINE fine-timescale
relationship is destroyed, but each unit's own real, slow-timescale
firing-rate structure -- and therefore any shared slow co-modulation --
is preserved almost exactly). Rebuild the null by repeating this 1000
times and recomputing how many cross-unit spike pairs land within the
1.5ms refractory window each time. Compare the REAL cross-refractory
count against this empirical null distribution, not a closed-form
approximation.

Usage: python demo_merge_candidate_jitter_test.py
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
JITTER_MS = 10.0
N_PERM = 1000

spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

PAIRS = [(306, 305), (306, 298), (332, 333), (332, 334), (333, 334),
         (303, 299), (303, 306), (342, 341), (313, 318)]


def times_ms(uid):
    return np.sort(spike_times[spike_clusters == uid]) / FS * 1000.0


def cross_refractory_count(ta_sorted, tb_sorted, w=REFRACTORY_MS):
    """Vectorized count of pairs with |ta_i - tb_j| < w."""
    lo = np.searchsorted(tb_sorted, ta_sorted - w)
    hi = np.searchsorted(tb_sorted, ta_sorted + w)
    return int(np.sum(hi - lo))


results = []
rng = np.random.default_rng(0)

for a, b in PAIRS:
    ta, tb = times_ms(a), times_ms(b)
    # jitter whichever train has FEWER spikes -- cheaper, and the test is
    # symmetric in which side gets jittered (either way destroys the same
    # fine-timescale relationship while preserving each side's own
    # slow-timescale structure)
    if len(ta) <= len(tb):
        fixed, to_jitter, fixed_label = tb, ta, "a"
    else:
        fixed, to_jitter, fixed_label = ta, tb, "b"

    obs = cross_refractory_count(ta, tb)

    null_counts = np.empty(N_PERM, dtype=int)
    T_max = max(ta.max(), tb.max())
    for i in range(N_PERM):
        jitter = rng.uniform(-JITTER_MS, JITTER_MS, size=len(to_jitter))
        jittered = np.sort(np.clip(to_jitter + jitter, 0, T_max))
        if fixed_label == "a":
            null_counts[i] = cross_refractory_count(fixed, jittered)
        else:
            null_counts[i] = cross_refractory_count(jittered, fixed)

    null_mean, null_std = null_counts.mean(), null_counts.std()
    z = (obs - null_mean) / (null_std + 1e-9)
    p_low = (np.sum(null_counts <= obs) + 1) / (N_PERM + 1)   # is observed unusually LOW (depleted)?
    p_high = (np.sum(null_counts >= obs) + 1) / (N_PERM + 1)  # is observed unusually HIGH (excess)?

    verdict = ("DEPLETED vs jittered null -- supports shared identity" if p_low < 0.05
               else "ELEVATED vs jittered null -- argues against merge" if p_high < 0.05
               else "not distinguishable from jittered null -- no real evidence either way")

    print(f"{a} vs {b}: n_a={len(ta)}, n_b={len(tb)}, observed cross-refractory count={obs}, "
          f"jittered-null mean={null_mean:.1f} (std={null_std:.1f}), z={z:.2f}, "
          f"p_depleted={p_low:.3f}, p_elevated={p_high:.3f} -> {verdict}")

    results.append(dict(unit_a=a, unit_b=b, n_a=len(ta), n_b=len(tb), observed=obs,
                         null_mean=null_mean, null_std=null_std, z=z,
                         p_depleted=p_low, p_elevated=p_high, verdict=verdict))

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT, "merge_candidate_jitter_test.csv"), index=False)
print("\n=== SUMMARY ===")
print(df[["unit_a", "unit_b", "observed", "null_mean", "null_std", "z", "p_depleted", "p_elevated"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
labels = [f"{r.unit_a}-{r.unit_b}" for r in df.itertuples()]
colors = ["#2ca02c" if r.p_depleted < 0.05 else "#d62728" if r.p_elevated < 0.05 else "#888"
          for r in df.itertuples()]
ax.bar(labels, df["z"], color=colors)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("z-score (observed vs. jittered null)")
ax.set_title("Jitter-test z-scores: negative = fewer cross-refractory pairs than a\n"
              "rate-matched, comodulation-preserving null predicts (green = significant, p<0.05)", fontsize=11)
plt.xticks(rotation=30)
plt.tight_layout()
out_path = os.path.join(OUT, "merge_candidate_jitter_test.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
