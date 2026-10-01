"""
WHY were the missed spikes missed?

Stock Kilosort finds 78% of the injected spikes on the tiered benchmark. The
other 22% have never been examined. "Recall 0.78" is a score, not a diagnosis,
and the fix for each possible cause is completely different -- so this asks
what actually happened to each missed spike.

THE CRITICAL SPLIT, which decides where any fix belongs:

  NOT DETECTED      no spike of ANY cluster appears near that time on those
                    channels. Kilosort's detection stage never saw it. A fix
                    would have to live in detection/thresholding.

  DETECTED, MISFILED  a spike IS there at the right time, but it was assigned
                    to a different cluster than the one matched to our unit.
                    Detection worked; CLUSTERING scattered it. A fix would
                    have to live in the clustering/merge stage.

Those two numbers have never been separated in this project, and almost
everything built so far (alignment, whitening, footprint) targets one or the
other. Spending effort on detection when the failure is clustering -- or the
reverse -- is the expensive mistake this avoids.

ALSO REPORTED, for the missed spikes versus the found ones:
  * amplitude, measured directly from the hybrid recording at the injected
    time on the destination peak channel. If missed spikes are simply the
    small ones, that is a threshold story and nothing clever will fix it.
  * tier, so the causes can differ by difficulty
  * collision offset for tier 3, to see whether closeness drives misses

Usage: python demo_why_were_spikes_missed.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DATA = r"D:\Gil\spike_sorting_agent\hybrid_tiered"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
DAT = os.path.join(DATA, "hybrid_tiered.bin")
TRUTH = os.path.join(DATA, "hybrid_tiered_truth.npz")
MANIFEST = os.path.join(DATA, "manifest.csv")
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN, FS, NT0MIN, NT, ITEMSIZE = 384, 30000.0, 20, 61, 2
TOLERANCE = 10
CONFIGS = ["vanilla", "subsample_align"]

T = np.load(TRUTH, allow_pickle=True)
times, labels = T["times"].ravel(), T["labels"].ravel()
tier_names = [str(x) for x in T["tier_names"]]
tier = np.array([tier_names[i] for i in T["tier"].ravel()])
coll = T["collision_offset"].ravel()
man = pd.read_csv(MANIFEST)
dest_pk = dict(zip(man.unit, man.dest_peak_ch))
b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")
N_SAMPLES = int(T["n_samples"])

# ---- amplitude of every injected spike, measured from the hybrid file itself
print("measuring the amplitude of each injected spike from the recording...")
amp_uv = np.full(len(times), np.nan)
order = np.argsort(times)
with open(DAT, "rb") as f:
    for i in order:
        ch = int(dest_pk[int(labels[i])])
        lo = int(times[i]) - NT0MIN - 60
        n = NT + 120
        if lo < 0 or lo + n > N_SAMPLES:
            continue
        f.seek(lo * N_CHAN_BIN * ITEMSIZE)
        blk = np.frombuffer(f.read(n * N_CHAN_BIN * ITEMSIZE),
                            dtype=np.int16).reshape(n, N_CHAN_BIN)
        w = filtfilt(b_hp, a_hp, blk[:, ch].astype(np.float64))[60:60 + NT] * GAIN_TO_UV
        amp_uv[i] = float(w.max() - w.min())
print(f"  injected amplitudes: median {np.nanmedian(amp_uv):.0f} uV, "
      f"range {np.nanmin(amp_uv):.0f}-{np.nanmax(amp_uv):.0f} uV\n")


def classify(config):
    d = os.path.join(DATA, f"ks_{config}")
    st = np.load(os.path.join(d, "spike_times.npy")).ravel()
    cl = np.load(os.path.join(d, "spike_clusters.npy")).ravel()
    o = np.argsort(st)
    st_s, cl_s = st[o], cl[o]

    # which output cluster is matched to each injected unit (most hits)
    matched = {}
    for u in np.unique(labels):
        tt = times[labels == u]
        cand = {}
        for t in tt:
            lo = np.searchsorted(st_s, t - TOLERANCE, "left")
            hi = np.searchsorted(st_s, t + TOLERANCE, "right")
            for c in set(cl_s[lo:hi].tolist()):
                cand[c] = cand.get(c, 0) + 1
        matched[int(u)] = max(cand, key=cand.get) if cand else -1

    rows = []
    used = np.zeros(len(st_s), bool)
    for i in np.argsort(times):
        u = int(labels[i])
        t = int(times[i])
        lo = np.searchsorted(st_s, t - TOLERANCE, "left")
        hi = np.searchsorted(st_s, t + TOLERANCE, "right")
        near = [j for j in range(lo, hi) if not used[j]]
        if not near:
            status = "not_detected"
            got = -1
        else:
            j = min(near, key=lambda k: abs(st_s[k] - t))
            used[j] = True
            got = int(cl_s[j])
            status = "found" if got == matched[u] else "detected_misfiled"
        rows.append(dict(config=config, unit=u, tier=tier[i], time=t,
                         amp_uv=amp_uv[i], collision_offset=coll[i],
                         status=status, assigned_cluster=got,
                         matched_cluster=matched[u]))
    return pd.DataFrame(rows)


frames = [classify(c) for c in CONFIGS]
df = pd.concat(frames)
df.to_csv(os.path.join(OUT, "why_spikes_missed.csv"), index=False)

for c in CONFIGS:
    d = df[df.config == c]
    print(f"{'='*72}\n{c}: {len(d)} injected spikes")
    vc = d.status.value_counts()
    for k in ("found", "detected_misfiled", "not_detected"):
        n = int(vc.get(k, 0))
        print(f"  {k:<20}{n:>6}  {n/len(d):>7.1%}")
    miss = d[d.status != "found"]
    print(f"\n  OF THE {len(miss)} MISSES: "
          f"{int((miss.status=='detected_misfiled').sum())} were DETECTED but "
          f"filed under another cluster, "
          f"{int((miss.status=='not_detected').sum())} were never detected")
    print(f"\n  by tier:")
    tb = d.pivot_table(index="tier", columns="status", values="time",
                       aggfunc="count").fillna(0).astype(int)
    tb["recall"] = (tb.get("found", 0) / tb.sum(axis=1)).round(3)
    print(tb.to_string())
    print(f"\n  amplitude (uV) by outcome:")
    print(d.groupby("status")["amp_uv"].describe()[["count", "mean", "50%"]]
          .round(1).to_string())

print(f"\n{'='*72}\nDOES AMPLITUDE EXPLAIN THE MISSES? (vanilla)")
v = df[df.config == "vanilla"]
qs = np.nanpercentile(v.amp_uv, [0, 20, 40, 60, 80, 100])
v = v.assign(amp_bin=pd.cut(v.amp_uv, qs, include_lowest=True))
t = v.pivot_table(index="amp_bin", columns="status", values="time",
                  aggfunc="count", observed=False).fillna(0).astype(int)
t["recall"] = (t.get("found", 0) / t.sum(axis=1)).round(3)
print(t.to_string())

print(f"\nDOES COLLISION CLOSENESS EXPLAIN MISSES? (vanilla, collision tier)")
cv = v[(v.tier == "collision") & np.isfinite(v.collision_offset)]
cv = cv.assign(absoff=np.abs(cv.collision_offset))
cb = cv.assign(bin=pd.cut(cv.absoff, [0, 8, 14, 20, 26])).pivot_table(
    index="bin", columns="status", values="time", aggfunc="count",
    observed=False).fillna(0).astype(int)
cb["recall"] = (cb.get("found", 0) / cb.sum(axis=1)).round(3)
print(cb.to_string())
print("\nsaved why_spikes_missed.csv")
