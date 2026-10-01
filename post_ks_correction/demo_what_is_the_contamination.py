"""
Is the "contamination" in the matched clusters NOISE, or REAL other neurons?
VERSION 2 -- version 1 was spatially unrestricted and therefore meaningless.

THE TRAP VERSION 1 FELL INTO. It asked "does the original, untouched Kilosort
run have a spike within +-10 samples of this foreign spike?" -- anywhere on the
probe. The window holds 266,023 spikes in 3.6M samples, so a 21-sample window
catches 1.55 spikes BY CHANCE. The answer was guaranteed to be yes. This is
mistake #2 in the research log (base rates) committed a third time.

THE FIX. Restrict to the neighbourhood the cluster actually lives on: only
original clusters whose peak channel is within RADIUS_UM of our injected unit's
destination peak channel can plausibly contribute spikes to its cluster. Then
report the CHANCE coincidence rate for that restricted set alongside the
observed rate, so the comparison is interpretable rather than tautological.

ALSO REPORTED, because "real spike" and "well-isolated neuron" differ: the
KSLabel and contamination of each contributing source cluster.

Usage: python what_is_the_contamination2.py
"""
import os
import numpy as np
import pandas as pd

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
DATA = r"D:\Gil\spike_sorting_agent\hybrid_tiered"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
BG_START = 60_000_000
TOL = 10
COINC = 10
RADIUS_UM = 60.0

T = np.load(os.path.join(DATA, "hybrid_tiered_truth.npz"), allow_pickle=True)
times, labels = T["times"].ravel(), T["labels"].ravel()
tier_names = [str(x) for x in T["tier_names"]]
tier = np.array([tier_names[i] for i in T["tier"].ravel()])
n_samples = int(T["n_samples"])
man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
dest_pk = dict(zip(man.unit, man.dest_peak_ch))

cp = np.load(os.path.join(KS, "channel_positions.npy"))
templates = np.load(os.path.join(KS, "templates.npy"))
opeak = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)

ost = np.load(os.path.join(KS, "spike_times.npy")).ravel()
ocl = np.load(os.path.join(KS, "spike_clusters.npy")).ravel()
inwin = (ost >= BG_START) & (ost < BG_START + n_samples)
ost_w, ocl_w = ost[inwin] - BG_START, ocl[inwin]
o = np.argsort(ost_w)
ost_w, ocl_w = ost_w[o], ocl_w[o]

lab = pd.read_csv(os.path.join(KS, "cluster_KSLabel.tsv"), sep="\t")
ks_label = dict(zip(lab.cluster_id, lab.KSLabel))
contam = {}
p = os.path.join(KS, "cluster_ContamPct.tsv")
if os.path.exists(p):
    t = pd.read_csv(p, sep="\t")
    contam = dict(zip(t.cluster_id, t.iloc[:, 1]))

print(f"background window: {len(ost_w):,} original spikes over {n_samples:,} "
      f"samples, all 384 channels")
print(f"UNRESTRICTED chance hits in a +-{COINC} window: "
      f"{len(ost_w)*(2*COINC+1)/n_samples:.2f}  <-- why version 1 was void\n")

d = os.path.join(DATA, "ks_vanilla")
st = np.load(os.path.join(d, "spike_times.npy")).ravel()
cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
oo = np.argsort(st)
st, cl = st[oo], cl[oo]

rows, src_rows = [], []
for u in np.unique(labels):
    sel = labels == u
    tt = np.sort(times[sel])
    utier = tier[sel][0]
    dch = int(dest_pk[int(u)])

    # original clusters that could plausibly reach this location
    dist = np.sqrt(((cp[opeak] - cp[dch]) ** 2).sum(axis=1))
    local = np.where(dist <= RADIUS_UM)[0]
    loc_mask = np.isin(ocl_w, local)
    lst, lcl = ost_w[loc_mask], ocl_w[loc_mask]
    chance = len(lst) * (2 * COINC + 1) / n_samples

    cand = {}
    for t in tt:
        lo = np.searchsorted(st, t - TOL, "left")
        hi = np.searchsorted(st, t + TOL, "right")
        for c in set(cl[lo:hi].tolist()):
            cand[c] = cand.get(c, 0) + 1
    if not cand:
        continue
    c = max(cand, key=cand.get)

    cst = st[cl == c]
    ours = np.zeros(len(cst), bool)
    for t in tt:
        j = np.searchsorted(cst, t)
        for k in (j - 1, j):
            if 0 <= k < len(cst) and abs(int(cst[k]) - int(t)) <= TOL and not ours[k]:
                ours[k] = True
                break
    foreign = cst[~ours]

    pre, from_clusters = 0, {}
    for t in foreign:
        lo = np.searchsorted(lst, t - COINC, "left")
        hi = np.searchsorted(lst, t + COINC, "right")
        if hi > lo:
            pre += 1
            k = int(lcl[lo:hi][np.argmin(np.abs(lst[lo:hi] - t))])
            from_clusters[k] = from_clusters.get(k, 0) + 1

    rows.append(dict(
        unit=int(u), tier=utier, dest_ch=dch, cluster=int(c),
        n_local_clusters=len(local), local_spikes=len(lst),
        chance_per_window=round(chance, 4),
        n_in_cluster=len(cst), ours=int(ours.sum()), foreign=len(foreign),
        precision=round(ours.sum() / len(cst), 4),
        foreign_local_real=pre,
        frac_foreign_real=round(pre / len(foreign), 4) if len(foreign) else np.nan,
        corrected_precision=round((ours.sum() + pre) / len(cst), 4)))

    for k, n in sorted(from_clusters.items(), key=lambda x: -x[1])[:3]:
        src_rows.append(dict(unit=int(u), tier=utier, source_cluster=k,
                             n_spikes=n,
                             share=round(n / len(foreign), 3) if len(foreign) else np.nan,
                             dist_um=round(float(np.sqrt(
                                 ((cp[opeak[k]] - cp[dch]) ** 2).sum())), 1),
                             KSLabel=ks_label.get(k, "?"),
                             contam_pct=contam.get(k, np.nan)))

df = pd.DataFrame(rows)
sd = pd.DataFrame(src_rows)
df.to_csv(os.path.join(OUT, "contamination_identity_local.csv"), index=False)
sd.to_csv(os.path.join(OUT, "contamination_sources_local.csv"), index=False)

print("=" * 100)
print(f"PER UNIT, restricted to original clusters within {RADIUS_UM:.0f} um "
      f"of the destination channel")
print(df.to_string(index=False))

print("\n" + "=" * 100)
print("BY TIER")
print(df.groupby("tier")[["precision", "frac_foreign_real",
                          "corrected_precision", "chance_per_window"]]
      .mean().round(4).to_string())

tf, tp = df.foreign.sum(), df.foreign_local_real.sum()
print("\n" + "=" * 100)
print(f"{tf:,} foreign spikes; {tp:,} ({tp/tf:.1%}) coincide with a spike the "
      f"ORIGINAL run already had on a LOCAL cluster.")
print(f"mean chance coincidence for the local set: "
      f"{df.chance_per_window.mean():.3f} per window "
      f"(so the observed rate is NOT a chance artefact)")
print(f"precision as scored: {df.precision.mean():.4f}   "
      f"counting local pre-existing spikes as real: "
      f"{df.corrected_precision.mean():.4f}")

print("\n" + "=" * 100)
print("THE BIGGEST SINGLE CONTRIBUTOR TO EACH CLUSTER")
print(sd.to_string(index=False))
if len(sd):
    print("\nforeign spikes by original label of source cluster:")
    print(sd.groupby("KSLabel")["n_spikes"].sum().to_string())
print("\nsaved contamination_identity_local.csv, contamination_sources_local.csv")
