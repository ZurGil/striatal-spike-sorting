"""AUDIT E2: footprint-first ATTRIBUTION -- the plan's section 6 reformulated as
argmax over competing units, instead of a one-sided threshold on the target unit.

This is the one idea in the proposal our data has never tested, and it is also
this project's own open lead. Run on the EXACT candidates the 2.3% run produced,
so the comparison is like-for-like.

Three things measured at once:
  1. ARGMAX vs THRESHOLD: accept a candidate only if the target unit's footprint
     beats every nearby unit's footprint on the SAME channel set (the plan's
     "same union of channels across competing explanations").
  2. The DOMAIN MISMATCH found by code reading: the shipped pipeline built
     fp_expected from UNFILTERED raw voltage while scoring candidates on
     HIGHPASS-FILTERED voltage. Both are computed here.
  3. The chance rate for every acceptance rule (project rule 1).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = r"D:\Gil\spike_sorting_agent"
sys.path.insert(0, os.path.join(ROOT, "post_ks_correction"))
from spatial_footprint import spatial_energy_vector  # noqa: E402

REP, CONFIG = 0, "vanilla"
FS, N_CHAN_BIN, N, NT0MIN = 30000.0, 384, 61, 20
GAIN_TO_UV = 0.018311105685598315
RADIUS_UM = 60.0
NEIGH_UM = 120.0
MIN_SPK = 60
N_TPL = 200

sdir = os.path.join(ROOT, "hybrid_v2_rep%d" % REP)
ks = os.path.join(sdir, "ks_vanilla")
pos = np.load(os.path.join(ks, "channel_positions.npy"))
cmap = np.load(os.path.join(ks, "channel_map.npy"))
st = np.load(os.path.join(ks, "spike_times.npy")).astype(np.int64).ravel()
cl = np.load(os.path.join(ks, "spike_clusters.npy")).astype(np.int64).ravel()
print("KS: %d spikes, %d clusters" % (len(st), len(np.unique(cl))))

full_pos = np.full((N_CHAN_BIN, 2), 1e9)
full_pos[cmap] = pos

raw = np.memmap(os.path.join(sdir, "hybrid.bin"), dtype=np.int16,
                mode="r").reshape(-1, N_CHAN_BIN)
n_samples = raw.shape[0]
b_hp, a_hp = butter(3, 300.0 / (FS / 2), btype="high")
_fcache = {}


def filt(c):
    c = int(c)
    if c not in _fcache:
        _fcache[c] = (filtfilt(b_hp, a_hp, raw[:, c].astype(np.float64))
                      * GAIN_TO_UV).astype(np.float32)
    return _fcache[c]


def mean_wave_on(chans, times, filtered):
    times = times[(times > 500) & (times < n_samples - 500)]
    if len(times) < 20:
        return None
    pick = times[np.random.default_rng(3).permutation(len(times))[:N_TPL]]
    acc = np.zeros((N, len(chans)))
    for s in pick:
        lo = int(s) - NT0MIN
        if filtered:
            acc += np.stack([filt(c)[lo:lo + N] for c in chans], axis=1)
        else:
            acc += raw[lo:lo + N, chans].astype(np.float64) * GAIN_TO_UV
    mw = acc / len(pick)
    return mw - mw.mean(axis=0, keepdims=True)


def footprint_of(chans, times, filtered):
    mw = mean_wave_on(chans, times, filtered)
    if mw is None:
        return None
    amp = mw.max(axis=0) - mw.min(axis=0)
    nn = np.linalg.norm(amp)
    return amp / nn if nn > 0 else None


def peak_channel(times, n=60):
    s = times[(times > 500) & (times < n_samples - 500)]
    if len(s) < 20:
        return None
    s = s[np.random.default_rng(1).permutation(len(s))[:n]]
    acc = np.zeros((N, N_CHAN_BIN))
    for t in s:
        acc += raw[int(t) - NT0MIN:int(t) - NT0MIN + N, :].astype(np.float64)
    mw = acc / len(s) * GAIN_TO_UV
    mw -= mw.mean(axis=0, keepdims=True)
    return int(np.argmax(mw.max(axis=0) - mw.min(axis=0)))


cand = pd.read_csv(os.path.join(ROOT, "outputs", "v2_post_hoc_candidates.csv"))
cand = cand[(cand.config == CONFIG) & (cand.replicate == REP)].copy()
print("candidates: %d (%d real, %.2f%%)"
      % (len(cand), cand.is_true.sum(), 100 * cand.is_true.mean()))

clusters = [int(c) for c in np.unique(cl) if (cl == c).sum() >= MIN_SPK]
print("%d clusters with >= %d spikes" % (len(clusters), MIN_SPK))

print("computing peak channel for every cluster once ...", flush=True)
pk = {}
for c in clusters:
    p = peak_channel(st[cl == c], n=40)
    if p is not None:
        pk[c] = p
print("  done (%d)" % len(pk), flush=True)

rows = []
for unit, g in cand.groupby("unit"):
    tgt_c = int(g.cluster.iloc[0])
    tier = g.tier.iloc[0]
    t_tgt = st[cl == tgt_c]
    if len(t_tgt) < MIN_SPK or tgt_c not in pk:
        continue
    peak_ch = pk[tgt_c]
    d = np.sqrt(((full_pos - full_pos[peak_ch]) ** 2).sum(axis=1))
    fp_chans = np.sort(np.where(d <= RADIUS_UM)[0])
    if len(fp_chans) < 3:
        continue

    comp = [c for c in clusters
            if c != tgt_c and c in pk
            and np.sqrt(((full_pos[pk[c]] - full_pos[peak_ch]) ** 2).sum()) <= NEIGH_UM]

    fps_f, fps_r = {}, {}
    for c in [tgt_c] + comp:
        a = footprint_of(fp_chans, st[cl == c], True)
        b = footprint_of(fp_chans, st[cl == c], False)
        if a is not None:
            fps_f[c] = a
        if b is not None:
            fps_r[c] = b
    if tgt_c not in fps_f:
        continue

    gg = g[(g.t > 500) & (g.t < n_samples - 500)].copy()
    times = gg.t.values.astype(np.int64)
    istrue = gg.is_true.values.astype(bool)
    if len(times) == 0:
        continue

    obs = np.empty((len(times), len(fp_chans)))
    for i, t in enumerate(times):
        lo = int(t) - NT0MIN
        snip = np.stack([filt(c)[lo:lo + N] for c in fp_chans], axis=1)
        obs[i] = spatial_energy_vector(snip)

    gate_ok = ((gg.r2 >= gg.bar_r2) & (gg.fp >= gg.bar_fp)).values
    for dom, fps in (("filtered", fps_f), ("raw", fps_r)):
        if tgt_c not in fps:
            continue
        order = [tgt_c] + [c for c in fps if c != tgt_c]
        S = np.stack([obs @ fps[c] for c in order], axis=1)
        win = np.argmax(S, axis=1) == 0
        both = win & gate_ok
        rows.append(dict(
            unit=unit, tier=tier, cluster=tgt_c, domain=dom,
            n_comp=len(order) - 1, n_cand=len(times), n_true=int(istrue.sum()),
            fp_tgt_mean=float(S[:, 0].mean()),
            fp_best_comp_mean=(float(S[:, 1:].max(axis=1).mean())
                               if S.shape[1] > 1 else np.nan),
            argmax_acc=int(win.sum()), argmax_true=int(istrue[win].sum()),
            gate_acc=int(gate_ok.sum()), gate_true=int(istrue[gate_ok].sum()),
            both_acc=int(both.sum()), both_true=int(istrue[both].sum()),
            n_missed=int(gg.n_missed.iloc[0])))
    r = rows[-1]
    print("  unit %4s %-10s cl%-4d | %2d competitors | cand %4d true %3d "
          "| argmax keeps %4d (%3d real)"
          % (unit, tier, tgt_c, r["n_comp"], r["n_cand"], r["n_true"],
             r["argmax_acc"], r["argmax_true"]), flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "outputs", "audit_argmax_attribution.csv"),
           index=False)

print("\n" + "=" * 74)
print("ARGMAX-OVER-UNITS vs ONE-SIDED THRESHOLD  (rep0, vanilla)")
print("=" * 74)
for dom, dd in out.groupby("domain"):
    print("\n--- expected footprints built on %s voltage ---" % dom.upper())
    nc, nt_ = dd.n_cand.sum(), dd.n_true.sum()
    print("  candidate pool %d, real %d (base rate %.2f%%)"
          % (nc, nt_, 100.0 * nt_ / max(nc, 1)))
    for nm, a, t in (("shipped two gates", dd.gate_acc.sum(), dd.gate_true.sum()),
                     ("ARGMAX over units", dd.argmax_acc.sum(), dd.argmax_true.sum()),
                     ("argmax AND gates ", dd.both_acc.sum(), dd.both_true.sum())):
        print("  %s : accept %6d  real %4d  precision %.2f%%"
              % (nm, a, t, 100.0 * t / max(a, 1)))
    print("  mean fp to target %.4f vs best competitor %.4f"
          % (dd.fp_tgt_mean.mean(), dd.fp_best_comp_mean.mean()))

print("\nper tier (filtered domain):")
f = out[out.domain == "filtered"]
print("  %-10s %6s %5s %8s %9s %9s" % ("tier", "cand", "real", "gate_p",
                                       "argmax_p", "argmax_n"))
for t, gg in f.groupby("tier"):
    print("  %-10s %6d %5d %7.2f%% %8.2f%% %9d"
          % (t, gg.n_cand.sum(), gg.n_true.sum(),
             100.0 * gg.gate_true.sum() / max(gg.gate_acc.sum(), 1),
             100.0 * gg.argmax_true.sum() / max(gg.argmax_acc.sum(), 1),
             gg.argmax_acc.sum()))
