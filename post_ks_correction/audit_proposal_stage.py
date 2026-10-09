"""AUDIT E3: the PROPOSAL stage, which the argmax experiment just identified as
the binding constraint.

Tests three implementation points the proposal makes that our shipped code does
NOT do, holding the template fixed so only the filter changes:

  A  unwhitened single-channel matched filter        <- what shipped
  B  temporally whitened, AR(4) from the shipped 2 s min-spike-count window
     (the plan's 5.A.1: "apply it consistently to data and templates" -- our
      code applied whitening only to a post-hoc R2 gate, never to the filter)
  C  same, but AR(4) from a REPRESENTATIVE noise sample across the recording
     (the plan warns against estimating noise from a window selected for being
      quiet, which underestimates it)
  D  multichannel matched filter over the whole footprint, unwhitened
     (the plan's 5.A.2: score the full informative footprint, not one channel)
  E  multichannel + whitened

Endpoint is what matters at this stage, not precision: of the unit's genuinely
missed spikes, how many does the candidate pool CONTAIN at a matched pool size?
Everything downstream can only lose spikes from there.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter, fftconvolve

ROOT = r"D:\Gil\spike_sorting_agent"
sys.path.insert(0, os.path.join(ROOT, "post_ks_correction"))
from noise_whitening import fit_ar_model  # noqa: E402

REP = 0
FS, N_CHAN_BIN, N, NT0MIN = 30000.0, 384, 61, 20
GAIN_TO_UV = 0.018311105685598315
RADIUS_UM = 60.0
TOL = 10
MARGIN = 20
MIN_PEAK_SEP = 15
TOP_K = 800
N_TPL = 200
NOISE_SECONDS = 2.0

sdir = os.path.join(ROOT, "hybrid_v2_rep%d" % REP)
ks = os.path.join(sdir, "ks_vanilla")
pos = np.load(os.path.join(ks, "channel_positions.npy"))
cmap = np.load(os.path.join(ks, "channel_map.npy"))
st = np.load(os.path.join(ks, "spike_times.npy")).astype(np.int64).ravel()
cl = np.load(os.path.join(ks, "spike_clusters.npy")).astype(np.int64).ravel()
full_pos = np.full((N_CHAN_BIN, 2), 1e9)
full_pos[cmap] = pos

tr = np.load(os.path.join(sdir, "hybrid_truth.npz"), allow_pickle=True)
t_times, t_labels = tr["times"].ravel(), tr["labels"].ravel()
tier_names = [str(s) for s in tr["tier_names"]]
unit_tier = tr["unit_tier"].ravel()
source_units = tr["source_units"].ravel()

raw = np.memmap(os.path.join(sdir, "hybrid.bin"), dtype=np.int16,
                mode="r").reshape(-1, N_CHAN_BIN)
n_samples = raw.shape[0]
b_hp, a_hp = butter(3, 300.0 / (FS / 2), btype="high")
_fc = {}


def filt(c):
    c = int(c)
    if c not in _fc:
        _fc[c] = (filtfilt(b_hp, a_hp, raw[:, c].astype(np.float64))
                  * GAIN_TO_UV).astype(np.float32)
    return _fc[c]


def quiet_window_shipped(ch, all_spikes, seconds=NOISE_SECONDS):
    """Exactly the shipped rule: of 40 candidate windows take the one with the
    fewest detected spikes."""
    n = int(seconds * FS)
    s = np.sort(all_spikes)
    best, bk = None, None
    for start in np.linspace(1000, n_samples - n - 1000, 40).astype(int):
        k = np.searchsorted(s, start + n) - np.searchsorted(s, start)
        if bk is None or k < bk:
            best, bk = start, k
    return filt(ch)[best:best + n].astype(np.float64)


def representative_noise(ch, all_spikes, n_blocks=40, block=0.25):
    """Concatenate many short spike-free blocks spread over the recording,
    rather than one window chosen for being the quietest."""
    nb = int(block * FS)
    s = np.sort(all_spikes)
    segs = []
    for start in np.linspace(1000, n_samples - nb - 1000, n_blocks * 3).astype(int):
        k = np.searchsorted(s, start + nb) - np.searchsorted(s, start)
        if k == 0:
            segs.append(filt(ch)[start:start + nb].astype(np.float64))
        if len(segs) >= n_blocks:
            break
    if not segs:
        return quiet_window_shipped(ch, all_spikes)
    return np.concatenate(segs)


def ar_filter(x, phi):
    """Prediction-error (whitening) filter from AR coefficients."""
    return lfilter(np.concatenate([[1.0], -np.asarray(phi, float)]), [1.0], x)


def peaks_from_score(score, shift, existing, k=TOP_K):
    """Local maxima, thinned, de-duplicated against the cluster's own spikes."""
    thr = np.percentile(score, 99.0)
    above = np.where(score >= thr)[0]
    if not len(above):
        return np.array([], np.int64)
    ismax = ((score[above] >= score[np.maximum(above - 1, 0)]) &
             (score[above] >= score[np.minimum(above + 1, len(score) - 1)]))
    pk = above[ismax]
    pk = pk[np.argsort(-score[pk])]
    taken = np.zeros(len(score), bool)
    out = []
    for p in pk:
        lo, hi = max(0, p - MIN_PEAK_SEP), min(len(score), p + MIN_PEAK_SEP + 1)
        if taken[lo:hi].any():
            continue
        taken[p] = True
        out.append(int(p) + shift)
        if len(out) >= k * 3:
            break
    c = np.array(sorted(out), dtype=np.int64)
    if len(c) and len(existing):
        e = np.sort(existing)
        idx = np.searchsorted(e, c)
        near = np.full(len(c), np.inf)
        for off in (-1, 0):
            j = np.clip(idx + off, 0, len(e) - 1)
            near = np.minimum(near, np.abs(c - e[j]))
        c = c[near > MARGIN]
    c = c[(c > 500) & (c < n_samples - 500)]
    return c[:k]


cand = pd.read_csv(os.path.join(ROOT, "outputs", "v2_post_hoc_candidates.csv"))
cand = cand[(cand.config == "vanilla") & (cand.replicate == REP)]
placements = cand.groupby("unit").agg(cluster=("cluster", "first"),
                                      tier=("tier", "first")).reset_index()
print("%d placements" % len(placements))

rows = []
for _, p in placements.iterrows():
    unit, tgt_c, tier = int(p.unit), int(p.cluster), p.tier
    own = np.sort(st[cl == tgt_c])
    truth = np.sort(t_times[t_labels == unit])
    if len(own) < 60 or len(truth) < 20:
        continue

    # the unit's genuinely MISSED spikes: no spike of its matched cluster near
    missed = []
    for t in truth:
        j = np.searchsorted(own, t)
        d = min([abs(t - own[k]) for k in (j - 1, j)
                 if 0 <= k < len(own)] or [np.inf])
        if d > TOL:
            missed.append(t)
    missed = np.array(missed, np.int64)
    if len(missed) == 0:
        continue

    # template: empirical filtered mean on the peak channel, held FIXED
    sub = truth[(truth > 500) & (truth < n_samples - 500)]
    acc = np.zeros((N, N_CHAN_BIN))
    pick = own[(own > 500) & (own < n_samples - 500)]
    pick = pick[np.random.default_rng(3).permutation(len(pick))[:N_TPL]]
    for s in pick:
        acc += raw[int(s) - NT0MIN:int(s) - NT0MIN + N, :].astype(np.float64)
    mw = acc / len(pick) * GAIN_TO_UV
    mw -= mw.mean(axis=0, keepdims=True)
    amp = mw.max(axis=0) - mw.min(axis=0)
    peak_ch = int(np.argmax(amp))
    d = np.sqrt(((full_pos - full_pos[peak_ch]) ** 2).sum(axis=1))
    fp_chans = np.sort(np.where(d <= RADIUS_UM)[0])

    # rebuild the template on FILTERED data over the footprint
    accf = np.zeros((N, len(fp_chans)))
    for s in pick:
        lo = int(s) - NT0MIN
        accf += np.stack([filt(c)[lo:lo + N] for c in fp_chans], axis=1)
    mwf = accf / len(pick)
    mwf -= mwf.mean(axis=0, keepdims=True)
    ip = int(np.where(fp_chans == peak_ch)[0][0]) if peak_ch in fp_chans else \
        int(np.argmax(mwf.max(axis=0) - mwf.min(axis=0)))
    tpl1 = mwf[:, ip]
    shift = NT0MIN - (N // 2)

    trace = filt(peak_ch).astype(np.float64)
    phi_s, s2s = fit_ar_model(quiet_window_shipped(peak_ch, st), order=4)
    phi_r, s2r = fit_ar_model(representative_noise(peak_ch, st), order=4)

    variants = {}
    tn = tpl1 / (np.linalg.norm(tpl1) + 1e-12)
    variants["A_single_unwhitened"] = fftconvolve(trace, tn[::-1], mode="same")
    for nm, phi in (("B_single_whitened_shipped_noise", phi_s),
                    ("C_single_whitened_repr_noise", phi_r)):
        tw = ar_filter(tpl1, phi)
        tw = tw / (np.linalg.norm(tw) + 1e-12)
        variants[nm] = fftconvolve(ar_filter(trace, phi), tw[::-1], mode="same")

    mc = np.zeros(len(trace))
    mcw = np.zeros(len(trace))
    for j, c in enumerate(fp_chans):
        tj = mwf[:, j]
        if np.linalg.norm(tj) < 1e-9:
            continue
        xj = filt(c).astype(np.float64)
        mc += fftconvolve(xj, tj[::-1], mode="same")
        pj, _ = fit_ar_model(representative_noise(c, st), order=4)
        tjw = ar_filter(tj, pj)
        mcw += fftconvolve(ar_filter(xj, pj), tjw[::-1], mode="same")
    variants["D_multichannel_unwhitened"] = mc
    variants["E_multichannel_whitened"] = mcw

    for nm, sc in variants.items():
        c = peaks_from_score(sc, shift, own, k=TOP_K)
        if len(c) == 0:
            hit = 0
        else:
            idx = np.searchsorted(c, missed)
            hit = 0
            for t, j in zip(missed, idx):
                dd = min([abs(t - c[k]) for k in (j - 1, j)
                          if 0 <= k < len(c)] or [np.inf])
                if dd <= TOL:
                    hit += 1
        # purity: candidates that are a real missed spike of THIS unit
        rows.append(dict(unit=unit, tier=tier, variant=nm,
                         n_missed=len(missed), pool=len(c), captured=hit,
                         capture_frac=hit / len(missed),
                         purity=hit / max(len(c), 1)))
    print("  unit %4d %-10s ch%-3d fp%2d missed %3d | "
          % (unit, tier, peak_ch, len(fp_chans), len(missed))
          + " ".join("%s %3d" % (k.split("_")[0], rows[-5 + i]["captured"])
                     for i, k in enumerate(variants)), flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "outputs", "audit_proposal_stage.csv"), index=False)

print("\n" + "=" * 78)
print("PROPOSAL-STAGE CAPTURE  (rep0, vanilla, %d placements, template held fixed)"
      % out.unit.nunique())
print("=" * 78)
print("%-34s %8s %9s %10s %9s" % ("variant", "pool", "missed", "captured",
                                  "purity"))
for v, g in out.groupby("variant"):
    print("%-34s %8d %9d %9d (%4.1f%%) %8.2f%%"
          % (v, g.pool.sum(), g.n_missed.sum(), g.captured.sum(),
             100.0 * g.captured.sum() / max(g.n_missed.sum(), 1),
             100.0 * g.captured.sum() / max(g.pool.sum(), 1)))
print("\nper tier, captured fraction of missed spikes:")
pv = out.pivot_table(index="tier", columns="variant", values="capture_frac",
                     aggfunc="mean")
print(pv.round(3).to_string())
