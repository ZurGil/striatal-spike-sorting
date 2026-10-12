"""The honest version: LOCDEN capture of Kilosort's misses at MATCHED pool size.

The raw comparison is not apples to apples. 5av's 26.1% ceiling was measured
with an 800-candidate pool per unit. LOCDEN's detector produces 403,654 events
over the whole probe; the ones within 40 um of a given unit over 120 s number
about ten thousand, so a 13x larger pool is being credited with a higher
capture rate. Pool size has to be held fixed or the number means nothing --
the same discipline that turned a 0.556 detection recall into 0.214 once its
0.415 chance rate was measured.

So: rank the LOCDEN events near each unit by the detection statistic, keep the
top N for N = 800 (5av's size) and a few others, and recompute. Also reported
per tier, because 371 of the 445 missed spikes in rep0 belong to the HARD
tier, which this project has established is not measurable (two identical
Kilosort runs disagree by up to 0.444 recall there).
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
POOLS = (200, 400, 800, 1600, 3200, 0)      # 0 = every event near the unit

ev = np.load(os.path.join(cache_dir(REP), "events2.npz"))
et, ech, equal = ev["t"], ev["ch"].astype(np.int64), ev["qual"]
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

# channel neighbourhood per unit, fixed once
near_ev = {}
for u in su:
    c0 = dstc[u]
    d = np.sqrt((X[ech] - X[c0]) ** 2 + (Y[ech] - Y[c0]) ** 2)
    near_ev[u] = np.where(d <= RAD_UM)[0]

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
    if len(missed) == 0:
        continue
    idx = near_ev[u]
    rank = idx[np.argsort(-equal[idx])]          # best statistic first
    for N in POOLS:
        pool = np.sort(et[rank[:N]] if N else et[rank])
        if len(pool) == 0:
            continue

        def cap(times):
            if len(pool) == 0:
                return 0.0
            j = np.searchsorted(pool, times)
            ok = np.zeros(len(times), bool)
            for i, (a, k) in enumerate(zip(times, j)):
                cc = [pool[m] for m in (k - 1, k) if 0 <= m < len(pool)]
                ok[i] = bool(cc) and min(abs(y - a) for y in cc) <= TOL
            return ok.mean()

        sham = np.mean([cap(rng.integers(1000, n_samples - 1000, len(missed)))
                        for _ in range(5)])
        rows.append(dict(unit=u, tier=tier_names[int(tier_code[lab == u][0])],
                         pool_req=N, pool=len(pool), n_missed=len(missed),
                         capture=cap(missed), chance=sham))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                       "locden_matched_pool.csv"), index=False)

print("=" * 90)
print("CAPTURE OF KILOSORT'S MISSED SPIKES AT MATCHED POOL SIZE (rep%d)" % REP)
print("=" * 90)
print("%10s %9s %9s %9s %9s %10s"
      % ("pool req", "mean pool", "missed", "capture", "chance", "excess"))
for N in POOLS:
    d = df[df.pool_req == N]
    if not len(d):
        continue
    w = d.capture * d.n_missed
    cap = w.sum() / d.n_missed.sum()
    print("%10s %9.0f %9d %9.3f %9.3f %+10.3f"
          % (N or "all", d.pool.mean(), d.n_missed.sum(), cap,
             d.chance.mean(), cap - d.chance.mean()))
print("\n  5av, shipped single-channel matched filter, 800-candidate pool: 0.261")

print("\n" + "=" * 90)
print("PER TIER at the 800-candidate pool size 5av used")
print("=" * 90)
d = df[df.pool_req == 800]
g = d.groupby("tier").apply(
    lambda x: pd.Series(dict(units=len(x), missed=x.n_missed.sum(),
                             capture=(x.capture * x.n_missed).sum()
                             / x.n_missed.sum(), chance=x.chance.mean())),
    include_groups=False)
print(g.round(3).to_string())
print("\nNOTE: the hard tier is NOT measurable in this benchmark -- two")
print("identical Kilosort runs disagree there by up to 0.444 recall -- and it")
print("supplies most of the missed spikes, so the non-hard rows are the ones")
print("that carry weight.")
nh = d[d.tier != "hard"]
if len(nh):
    print("\nNON-HARD TIERS ONLY: %d missed spikes, capture %.3f (chance %.3f)"
          % (nh.n_missed.sum(),
             (nh.capture * nh.n_missed).sum() / nh.n_missed.sum(),
             nh.chance.mean()))
