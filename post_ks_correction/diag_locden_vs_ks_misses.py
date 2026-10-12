"""Does LOCDEN's detector find the spikes KILOSORT misses?

This is the measurement that matters most from the LOCDEN work, and it is
independent of whether LOCDEN is a good sorter end to end (it is not).

5av established the ceiling on post-hoc recovery: holding the template fixed,
the best candidate-proposal variant captured only 26.1% of a unit's genuinely
missed spikes in an 800-candidate pool, and the two changes the external
proposal recommended (whitening the detection statistic, scoring the full
footprint) made it no better or much worse. That 26.1% is a hard ceiling --
everything downstream can only lose spikes from there.

LOCDEN's detector is a different instrument: spatially whitened, projected
onto a learned 5-component temporal basis, pooled over 11 channels. It raised
detection recall on injected spikes from 0.214 (per-channel threshold) to
0.863 over the full recording. The question here is whether that gain lands
where it would be useful -- on the spikes Kilosort's own clustering lost.

Reported with a MEASURED chance rate, on the identical channel restriction,
because the first version of this number was 0.556 against a chance rate of
0.415 and meant nothing.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import data_dir, cache_dir
from run_v2_comparison import score, _match

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOL = 10
RAD_UM = 40.0

ev = np.load(os.path.join(cache_dir(REP), "events2.npz"))
et, ech = ev["t"], ev["ch"].astype(np.int64)
X, Y = ev["probe_x"], ev["probe_y"]

T = np.load(os.path.join(data_dir(REP), "hybrid_truth.npz"), allow_pickle=True)
tt, lab = T["times"].ravel(), T["labels"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]
su = [int(x) for x in T["source_units"]]
dstc = {u: int(c) for u, c in zip(su, T["dst_peak_channels"])}
n_samples = int(T["n_samples"])

ks = os.path.join(data_dir(REP), "ks_vanilla")
st = np.load(os.path.join(ks, "spike_times.npy")).ravel().astype(np.int64)
cl = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
sc, _ = score("vanilla", REP)
matched = dict(zip(sc.unit.astype(int), sc.matched_cluster.astype(int)))


def locden_has(times, u):
    """Does a LOCDEN event sit within TOL samples and RAD_UM of the unit?"""
    c0 = dstc[int(u)]
    out = np.zeros(len(times), bool)
    for i, a in enumerate(times):
        p = np.searchsorted(et, a - TOL)
        q = np.searchsorted(et, a + TOL, "right")
        if q > p:
            dd = np.sqrt((X[ech[p:q]] - X[c0]) ** 2 + (Y[ech[p:q]] - Y[c0]) ** 2)
            out[i] = bool((dd <= RAD_UM).any())
    return out


rng = np.random.default_rng(0)
rows = []
for u in su:
    c = matched.get(u, -1)
    if c < 0:
        continue
    truth = np.sort(tt[lab == u])
    own = np.sort(st[cl == c])
    h, which = _match(own, truth, TOL)
    missed = truth[np.asarray(which) == -1]
    hitv = truth[np.asarray(which) >= 0]
    if len(missed) == 0:
        continue
    cap = locden_has(missed, u)
    cap_hit = locden_has(hitv, u) if len(hitv) else np.array([], bool)
    sham = np.mean([locden_has(rng.integers(1000, n_samples - 1000, len(missed)),
                               u).mean() for _ in range(5)])
    rows.append(dict(unit=u, tier=tier_names[int(tier_code[lab == u][0])],
                     n_true=len(truth), ks_recall=h / len(truth),
                     n_missed=len(missed), captured=int(cap.sum()),
                     capture_frac=cap.mean(), chance=sham,
                     capture_on_ks_hits=(cap_hit.mean() if len(cap_hit)
                                         else np.nan)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                       "locden_vs_ks_misses.csv"), index=False)

print("=" * 92)
print("OF THE SPIKES KILOSORT MISSED, HOW MANY DOES LOCDEN'S DETECTOR FIND?")
print("  rep%d, %d units, within %d samples and %.0f um of the destination"
      % (REP, len(df), TOL, RAD_UM))
print("=" * 92)
print("%5s %-10s %7s %8s %8s %9s %8s %9s"
      % ("unit", "tier", "n_true", "KS rec", "missed", "captured", "chance",
         "on KS hits"))
for _, r in df.iterrows():
    print("%5d %-10s %7d %8.3f %8d %9.3f %8.3f %9.3f"
          % (r.unit, r.tier, r.n_true, r.ks_recall, r.n_missed,
             r.capture_frac, r.chance, r.capture_on_ks_hits))
print("-" * 92)
tot_m = df.n_missed.sum()
tot_c = df.captured.sum()
print("POOLED: %d of %d missed spikes captured = %.3f (chance %.3f, excess %+.3f)"
      % (tot_c, tot_m, tot_c / tot_m, df.chance.mean(),
         tot_c / tot_m - df.chance.mean()))
print("  5av ceiling with the shipped single-channel matched filter: 0.261")
print("  best proposal variant measured in 5av:                      0.261")
print("\nper tier:")
g = df.groupby("tier").apply(
    lambda d: pd.Series(dict(units=len(d), missed=d.n_missed.sum(),
                             captured=d.captured.sum(),
                             frac=d.captured.sum() / d.n_missed.sum(),
                             chance=d.chance.mean())),
    include_groups=False)
print(g.round(3).to_string())
