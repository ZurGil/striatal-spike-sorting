"""
Run this project's post-Kilosort correction pipeline on the v2 hybrid
benchmark, where for the first time we KNOW the right answer.

WHY THIS RUN IS DIFFERENT FROM EVERY EARLIER ONE. The missed-spike search
was built and run twice on real data (5h, 5p) and both times the headline
number had to be withdrawn, for the same structural reason: on real data a
recovered candidate cannot be checked. The only available test was "does
some other unit already have a spike here", which answers a different
question -- it tells you the candidate is a real spike, not whose. 5h ended
at "these look like collisions, not misses"; 5p watched the genuinely-new
count fall from 115 to 40 and, for the cleanest unit, from 29 to 1, once the
frequency rule was fixed. Neither run could say whether the pipeline had
recovered a spike that truly belonged to the unit.

The v2 benchmark removes that limitation completely. Every injected spike's
time and owner is known, so a recovered candidate is either one of this
unit's spikes that Kilosort missed, or it is not, and nothing is left to
interpretation. That makes this the first honest measurement of whether
post-hoc correction is worth more than patching Kilosort itself -- which is
the actual decision on the table.

THE PIPELINE UNDER TEST, in the order the modules were built, and strictly
truth-blind: it sees Kilosort's output and the raw voltage, never the truth.

  1. TEMPLATE CONVERGENCE (demo_iterative_template_rebuild.py). Kilosort's
     own template is an average of spikes aligned only to the nearest whole
     sample, so sub-sample jitter blurs it. Start from the cluster's own
     empirical mean waveform, pick the unit's probe frequency by the timing
     criterion (f0*|W|, the corrected rule from 5o/5r -- the strength rule
     is what made 5n's result evaporate), wavelet-align every sampled spike
     to the current template, average, and iterate to a fixed point. The
     template is rebuilt from the DATA rather than read from templates.npy
     so that cluster ids beyond the template array (merges/splits) are not a
     special case.

  2. TEMPORAL WHITENING (noise_whitening.py). Fit an AR(4) model to a real
     spike-free stretch of this channel and derive the whitening operator,
     so the fit is `s^T Sigma^-1 x` rather than implicitly assuming white
     noise. Kilosort whitens SPATIALLY; this is the orthogonal piece.

  3. FULL-SESSION MATCHED-FILTER SEARCH. The earlier runs scanned only
     +-150 samples around existing spikes, which cannot find a missed spike
     that happens to fall between two detected ones -- and at these firing
     rates most do. Here the rebuilt template sweeps the entire 120 s trace,
     local maxima are taken above a threshold set by the cluster's OWN real
     spikes, and anything within MARGIN of a spike the cluster already has
     is dropped (we want the misses, not the hits).

  4. TWO GATES, both calibrated on the cluster's own real spikes, never on a
     fixed constant: the whitened fit R2, and footprint similarity. The
     second is there because 5p identified single-channel specificity as the
     binding limitation (~99% of what scored as a unit-440 spike on its peak
     channel was already unit 439's) and 5q measured footprint similarity as
     the fix (AUC 0.999 vs 0.749 for the single-channel score).

WHAT IS MEASURED, AND THE CONTROLS THAT MAKE IT MEAN SOMETHING

  before : recall and precision of the best-matching cluster, scored with
           run_v2_comparison's own matcher so the numbers are comparable to
           the headline tables.
  after  : the same two numbers once the accepted candidates are added to
           that cluster. This is the whole point -- a correction tool is only
           worth having if recall rises without precision collapsing.
  recovery precision : of the candidates it accepted, what fraction really
           are this unit's missed spikes.
  chance : the same accepted candidates matched against RANDOM times at the
           same rate. With 4000+ accepted candidates over 120 s, a +-10
           sample window gives a non-trivial hit rate for free, and this
           project has walked into that trap three times. The chance row is
           printed beside every result, not in a footnote.
  foreign: of the false accepts, how many coincide with a spike Kilosort
           already filed under a DIFFERENT cluster. This is the only check
           the old real-data runs could do, kept here because it separates
           "the tool is finding noise" from "the tool is finding real spikes
           belonging to someone else" -- a specificity failure, not a
           detection failure, and a different fix.

Usage: python test_post_hoc_on_v2.py [config ...]   (default: vanilla)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.interpolate import interp1d

from wavelet_features import (make_morlet, calibrate_reference_phase,
                              coarse_then_fine_shift, select_probe_frequency)
from nuisance_model import build_basis, fit_nuisance_prealigned
from noise_whitening import build_whitening_from_noise
from spatial_footprint import unit_footprint, spatial_energy_vector, footprint_similarity

ROOT = r"D:\Gil\spike_sorting_agent"
OUT = os.path.join(ROOT, "outputs")
N_REPLICATES = int(os.environ.get("N_REPLICATES", 4))
MAX_PLACEMENTS = int(os.environ.get("MAX_PLACEMENTS", 0))  # 0 = all; smoke-test hook
FS = 30000.0
N_CHAN_BIN = 384
N = 61
NT0MIN = 20
GAIN_TO_UV = 0.018311105685598315
TOLERANCE = 10            # same as run_v2_comparison, so before/after are comparable
SEARCH_RADIUS = 25
N_CYCLES = 3.0
RADIUS_UM = 60.0

N_TPL_SPIKES = 120        # spikes averaged per template-rebuild iteration
MAX_ITERS = 4
CONVERGE_CORR = 0.99999
N_REF_SPIKES = 80         # spikes used to calibrate the two gates
MF_PCTL = 5.0             # matched-filter threshold: this percentile of real spikes
R2_PCTL = 25.0            # whitened-fit gate
FP_PCTL = 25.0            # footprint-similarity gate
MIN_PEAK_SEP = 15         # samples between accepted local maxima
MARGIN = 20               # a candidate this close to an existing spike is not new
TOP_N_CANDIDATES = 800    # cap on candidates refined per placement (reported)
NOISE_SECONDS = 2.0
MIN_CLUSTER_SPIKES = 60

b_hp, a_hp = butter(3, 300 / (FS / 2), btype="high")


def _match(det_t, true_t, tolerance=TOLERANCE):
    """Greedy one-to-one match -- copied from run_v2_comparison so the
    'before' numbers here are identical to the headline tables."""
    det_t = np.asarray(det_t)
    if len(det_t) == 0:
        return 0, np.full(len(true_t), -1)
    order = np.argsort(det_t)
    ds = det_t[order]
    used = np.zeros(len(ds), bool)
    hit, which = 0, []
    for t in np.sort(true_t):
        lo = np.searchsorted(ds, t - tolerance, "left")
        hi = np.searchsorted(ds, t + tolerance, "right")
        cand = [i for i in range(lo, hi) if not used[i]]
        if not cand:
            which.append(-1)
            continue
        j = min(cand, key=lambda i: abs(ds[i] - t))
        used[j] = True
        hit += 1
        which.append(int(order[j]))
    return hit, np.asarray(which)


class Session:
    """One replicate's raw data, filtered lazily and cached per channel.

    The whole int16 file (2.8 GB) is held in RAM deliberately: the
    full-session matched filter needs every sample of several channels for
    each of 13 placements, and streaming it per placement would read the
    file 13 times.
    """

    def __init__(self, rep):
        self.dir = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
        t0 = time.time()
        raw = np.fromfile(os.path.join(self.dir, "hybrid.bin"), dtype=np.int16)
        self.raw = raw.reshape(-1, N_CHAN_BIN)
        self.n_samples = self.raw.shape[0]
        self._cache = {}
        print(f"  loaded {self.raw.shape} int16 in {time.time()-t0:.0f}s")

    def chan(self, c):
        """Filtered, gain-corrected full trace for one channel, in uV."""
        c = int(c)
        if c not in self._cache:
            x = filtfilt(b_hp, a_hp, self.raw[:, c].astype(np.float64))
            self._cache[c] = (x * GAIN_TO_UV).astype(np.float32)
        return self._cache[c]

    def snippet(self, chans, center, pad=0):
        """(n_time, n_chan) filtered snippet, channels in the given order."""
        lo = int(center) - NT0MIN - pad
        hi = lo + N + 2 * pad
        if lo < 0 or hi > self.n_samples:
            return None
        return np.stack([self.chan(c)[lo:hi] for c in chans], axis=1)


def align_snippet(trace, center, template, psi, f0, ref_phase, pad=120):
    """Wavelet-align one event on one channel and return the resampled
    61-sample snippet at its corrected sub-sample position."""
    lo = int(center) - pad
    hi = int(center) + pad
    if lo < 0 or hi > len(trace):
        return None, None
    seg = trace[lo:hi].astype(np.float64)
    nominal = pad
    r = coarse_then_fine_shift(seg, nominal, template, psi, f0, FS,
                               reference_phase=ref_phase,
                               search_radius=SEARCH_RADIUS, nt0min=NT0MIN)
    true_center = nominal + r["total_shift"]
    lo_i = int(np.floor(true_center - NT0MIN)) - 2
    hi_i = int(np.ceil(true_center - NT0MIN + N)) + 2
    if lo_i < 0 or hi_i > len(seg):
        return None, None
    f = interp1d(np.arange(lo_i, hi_i), seg[lo_i:hi_i], kind="cubic",
                 bounds_error=False, fill_value=0.0)
    snip = f(true_center - NT0MIN + np.arange(N))
    return snip, float(r["total_shift"])


def quiet_noise_trace(sess, ch, all_spike_times, seconds=NOISE_SECONDS):
    """A stretch of this channel with no detected spike in it, for the AR fit.

    Scans candidate windows and takes the one with the fewest detected
    spikes (ties broken by lowest MAD), rather than assuming any particular
    offset is quiet -- on a 120 s file with ~300 clusters, a blindly chosen
    window is not.
    """
    n = int(seconds * FS)
    st = np.sort(all_spike_times)
    best, best_key = None, None
    for start in np.linspace(1000, sess.n_samples - n - 1000, 40).astype(int):
        k = np.searchsorted(st, start + n) - np.searchsorted(st, start)
        if best_key is None or k < best_key:
            best, best_key = start, k
    seg = sess.chan(ch)[best:best + n].astype(np.float64)
    return seg


def run_placement(sess, st, cl, u, tt, tier, best_c, rebuilt_cache):
    """The correction pipeline for one placement. Truth-blind: `tt` is used
    only AFTER the fact, by the caller, to score what came back."""
    st_c = np.sort(st[cl == best_c])
    if len(st_c) < MIN_CLUSTER_SPIKES:
        return None

    # ---- 1. the cluster's empirical mean waveform -> peak channel, footprint
    if best_c in rebuilt_cache:
        R = rebuilt_cache[best_c]
    else:
        rng = np.random.default_rng(3)
        pick = st_c[rng.permutation(len(st_c))[:N_TPL_SPIKES]]
        pick = pick[(pick > 500) & (pick < sess.n_samples - 500)]
        if len(pick) < 20:
            return None
        acc = np.zeros((N, N_CHAN_BIN))
        nacc = 0
        for s in pick:
            lo = int(s) - NT0MIN
            blk = sess.raw[lo:lo + N, :].astype(np.float64)
            acc += blk
            nacc += 1
        mean_wave = (acc / nacc) * GAIN_TO_UV
        mean_wave -= mean_wave.mean(axis=0, keepdims=True)
        amp = mean_wave.max(axis=0) - mean_wave.min(axis=0)
        peak_ch = int(np.argmax(amp))

        # footprint channels from the probe geometry around that peak channel
        pos = np.load(os.path.join(sess.dir, "probe_positions.npy")) \
            if os.path.exists(os.path.join(sess.dir, "probe_positions.npy")) else None
        if pos is None:
            pos = CHANNEL_POSITIONS
        d = np.sqrt(((pos - pos[peak_ch]) ** 2).sum(axis=1))
        fp_chans = np.sort(np.where(d <= RADIUS_UM)[0])
        fp_expected = amp[fp_chans] / (np.linalg.norm(amp[fp_chans]) + 1e-12)

        trace = sess.chan(peak_ch)
        template = mean_wave[:, peak_ch].copy()

        # ---- 1b. converge the template by iterative wavelet realignment
        hist = []
        for it in range(1, MAX_ITERS + 1):
            f0, _, _ = select_probe_frequency(template, FS, n_cycles=N_CYCLES,
                                              nt0min=NT0MIN, warn_at_edge=False)
            _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
            refp = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
            aligned = []
            for s in pick:
                snip, _ = align_snippet(trace, int(s), template, psi, f0, refp)
                if snip is not None:
                    aligned.append(snip)
            if len(aligned) < 20:
                break
            new_t = np.mean(aligned, axis=0)
            corr = float(np.corrcoef(new_t, template)[0, 1])
            hist.append(dict(it=it, f0=f0, corr=corr, n=len(aligned),
                             ptp=float(new_t.max() - new_t.min())))
            template = new_t
            if corr > CONVERGE_CORR:
                break

        f0, _, _ = select_probe_frequency(template, FS, n_cycles=N_CYCLES,
                                          nt0min=NT0MIN, warn_at_edge=False)
        _, psi = make_morlet(f0, FS, n_cycles=N_CYCLES)
        refp = calibrate_reference_phase(template, psi, FS, align_index=NT0MIN)
        s0b, _, qb = build_basis(template, dt=1.0)

        # ---- 2. temporal whitening from a genuinely quiet stretch
        noise = quiet_noise_trace(sess, peak_ch, st)
        try:
            W, _, _ = build_whitening_from_noise(noise, N, order=4)
        except Exception:
            W = None

        # ---- 4a. calibrate both gates on the cluster's own real spikes
        rng2 = np.random.default_rng(11)
        ref_pick = st_c[rng2.permutation(len(st_c))[:N_REF_SPIKES]]
        ref_pick = ref_pick[(ref_pick > 500) & (ref_pick < sess.n_samples - 500)]
        tnorm = template / (np.linalg.norm(template) + 1e-12)
        r2s, fps, mfs = [], [], []
        for s in ref_pick:
            snip, _ = align_snippet(trace, int(s), template, psi, f0, refp)
            if snip is None:
                continue
            fit = fit_nuisance_prealigned(snip, s0b, qb, whitening_matrix=W)
            r2s.append(fit["r_squared"])
            mfs.append(float(np.dot(snip, tnorm)))
            ms = sess.snippet(fp_chans, int(s))
            if ms is not None:
                fps.append(footprint_similarity(spatial_energy_vector(ms), fp_expected))
        if len(r2s) < 20:
            return None
        R = dict(template=template, tnorm=tnorm, f0=f0, psi=psi, refp=refp,
                 s0b=s0b, qb=qb, W=W, peak_ch=peak_ch, fp_chans=fp_chans,
                 fp_expected=fp_expected, hist=hist,
                 bar_r2=float(np.percentile(r2s, R2_PCTL)),
                 bar_fp=float(np.percentile(fps, FP_PCTL)) if fps else -1.0,
                 bar_mf=float(np.percentile(mfs, MF_PCTL)),
                 ref_r2=float(np.mean(r2s)), ref_fp=float(np.mean(fps)) if fps else np.nan,
                 n_iters=len(hist))
        rebuilt_cache[best_c] = R

    trace = sess.chan(R["peak_ch"])

    # ---- 3. full-session matched filter with the rebuilt template
    mf = fftconvolve(trace.astype(np.float64), R["tnorm"][::-1], mode="same")
    # Convention, verified against known truth times rather than reasoned
    # about: with mode="same", output index i scores the template window
    # starting at i - N//2, and the spike sits at index NT0MIN inside that
    # window, so the event is at i - N//2 + NT0MIN. Getting this sign
    # backwards put every candidate 20 samples late and silently recovered
    # nothing -- the gates still passed, because align_snippet's +-25 coarse
    # search walked back to the event, so the error showed up only as zero
    # true recoveries with plausible-looking fit quality.
    shift = NT0MIN - (N // 2)
    above = np.where(mf >= R["bar_mf"])[0]
    cands = []
    if len(above):
        # keep local maxima, thinned to MIN_PEAK_SEP
        is_max = (mf[above] >= mf[np.maximum(above - 1, 0)]) & \
                 (mf[above] >= mf[np.minimum(above + 1, len(mf) - 1)])
        peaks = above[is_max]
        order = peaks[np.argsort(-mf[peaks])]
        taken = np.zeros(len(mf), bool)
        for p in order:
            lo = max(0, p - MIN_PEAK_SEP)
            hi = min(len(mf), p + MIN_PEAK_SEP + 1)
            if taken[lo:hi].any():
                continue
            taken[p] = True
            cands.append(int(p) + shift)
    cands = np.array(sorted(cands), dtype=np.int64)

    # drop anything the cluster already has
    st_c_sorted = np.sort(st_c)
    if len(cands):
        idx = np.searchsorted(st_c_sorted, cands)
        near = np.full(len(cands), np.inf)
        for off in (-1, 0):
            j = np.clip(idx + off, 0, len(st_c_sorted) - 1)
            near = np.minimum(near, np.abs(cands - st_c_sorted[j]))
        cands = cands[near > MARGIN]
    n_cand_total = len(cands)
    cands = cands[(cands > 500) & (cands < sess.n_samples - 500)]
    if len(cands) > TOP_N_CANDIDATES:
        keep = np.argsort(-mf[np.clip(cands - shift, 0, len(mf) - 1)])[:TOP_N_CANDIDATES]
        cands = np.sort(cands[keep])

    # ---- 4b. refine EVERY candidate and record its scores, gate afterwards.
    # Both gates are recorded rather than applied inline so the thresholds can
    # be swept offline: the percentile chosen for a gate is the single biggest
    # lever on the recall/precision trade a correction tool offers, and fixing
    # it at one value here would hide exactly the trade we are trying to
    # measure.
    recs = []
    for c in cands:
        snip, tot = align_snippet(trace, int(c), R["template"], R["psi"],
                                  R["f0"], R["refp"])
        if snip is None:
            continue
        fit = fit_nuisance_prealigned(snip, R["s0b"], R["qb"],
                                      whitening_matrix=R["W"])
        ms = sess.snippet(R["fp_chans"], int(c))
        fp = footprint_similarity(spatial_energy_vector(ms), R["fp_expected"]) \
            if ms is not None else np.nan
        recs.append(dict(t=int(c) + int(round(tot)),
                         mf=float(mf[int(np.clip(c - shift, 0, len(mf) - 1))]),
                         r2=float(fit["r_squared"]), fp=float(fp)))

    cd = pd.DataFrame(recs)
    if len(cd):
        ok = (cd.r2 >= R["bar_r2"]) & (cd.fp >= R["bar_fp"]) & np.isfinite(cd.fp)
        accepted = np.array(sorted(cd.loc[ok, "t"].tolist()), dtype=np.int64)
        n_fail_r2 = int((cd.r2 < R["bar_r2"]).sum())
        n_fail_fp = int(((cd.r2 >= R["bar_r2"]) & ~((cd.fp >= R["bar_fp"])
                                                    & np.isfinite(cd.fp))).sum())
        acc_r2 = float(cd.loc[ok, "r2"].mean()) if ok.any() else np.nan
        acc_fp = float(cd.loc[ok, "fp"].mean()) if ok.any() else np.nan
    else:
        accepted = np.array([], dtype=np.int64)
        n_fail_r2 = n_fail_fp = 0
        acc_r2 = acc_fp = np.nan

    return dict(R=R, accepted=accepted, cand_table=cd,
                n_cand_total=n_cand_total, n_refined=len(cands),
                n_fail_r2=n_fail_r2, n_fail_fp=n_fail_fp,
                acc_r2=acc_r2, acc_fp=acc_fp)


# --------------------------------------------------------------- main
CONFIGS = sys.argv[1:] or ["vanilla"]
CHANNEL_POSITIONS = None
rows = []
merge_rows = []
cand_tables = []

for rep in range(N_REPLICATES):
    data_dir = os.path.join(ROOT, f"hybrid_v2_rep{rep}")
    T = np.load(os.path.join(data_dir, "hybrid_truth.npz"), allow_pickle=True)
    t_true, lab = T["times"].ravel(), T["labels"].ravel()
    tier_names = [str(x) for x in T["tier_names"]]
    tier_code = T["tier"].ravel()
    unit_pair = dict(zip([int(x) for x in T["source_units"]],
                         [int(p) for p in T["unit_pair_id"]]))

    have = [c for c in CONFIGS
            if os.path.exists(os.path.join(data_dir, f"ks_{c}", "spike_times.npy"))]
    if not have:
        continue
    if CHANNEL_POSITIONS is None:
        CHANNEL_POSITIONS = np.load(os.path.join(data_dir, f"ks_{have[0]}",
                                                 "channel_positions.npy"))
    print(f"\n=== replicate {rep} ===")
    sess = Session(rep)

    for cfg in have:
        res = os.path.join(data_dir, f"ks_{cfg}")
        st = np.load(os.path.join(res, "spike_times.npy")).ravel()
        cl = np.load(os.path.join(res, "spike_clusters.npy")).ravel()
        o = np.argsort(st)
        st, cl = st[o], cl[o]
        rebuilt_cache = {}
        rng_ch = np.random.default_rng(100 + rep)

        done_here = 0
        for u in np.unique(lab):
            if MAX_PLACEMENTS and done_here >= MAX_PLACEMENTS:
                break
            done_here += 1
            sel = lab == u
            tt = np.sort(t_true[sel])
            tier = tier_names[int(tier_code[sel][0])]

            # ---- before: best-matching cluster, same matcher as the headline
            cand_cl = set()
            for t in tt:
                cand_cl.update(cl[np.abs(st - t) <= TOLERANCE].tolist())
            best = None
            for c in sorted(cand_cl):
                h, _ = _match(st[cl == c], tt)
                if best is None or h > best[1]:
                    best = (int(c), h)
            if best is None or best[1] == 0:
                continue
            best_c, hits0 = best
            n_det0 = int((cl == best_c).sum())
            rec0 = hits0 / len(tt)
            prec0 = hits0 / n_det0

            t0 = time.time()
            out = run_placement(sess, st, cl, u, tt, tier, best_c, rebuilt_cache)
            if out is None:
                continue
            A = out["accepted"]

            # ---- which of this unit's spikes the cluster already had
            _, which = _match(st[cl == best_c], tt)
            missed = tt[which < 0]

            rec_hits, _ = _match(A, missed) if len(A) else (0, None)
            n_false = len(A) - rec_hits

            # label every REFINED candidate (gated or not) so the gate
            # thresholds can be swept offline against the real answer
            cd = out["cand_table"]
            if len(cd):
                ct = cd.t.to_numpy()
                _, w_all = _match(ct, missed)
                # _match returns indices into the ORIGINAL det_t array
                is_true = np.zeros(len(ct), bool)
                for j in w_all:
                    if j >= 0:
                        is_true[j] = True
                cd = cd.assign(is_true=is_true, config=cfg, replicate=rep,
                               unit=int(u), tier=tier, cluster=best_c,
                               n_missed=len(missed), n_true_spikes=len(tt),
                               bar_r2=out["R"]["bar_r2"], bar_fp=out["R"]["bar_fp"])
                cand_tables.append(cd)

            # chance control: same accepted set against random times, same count
            chance_hits = 0
            for _ in range(5):
                fake = np.sort(rng_ch.integers(600, sess.n_samples - 600,
                                               size=len(missed)))
                h, _ = _match(A, fake) if len(A) else (0, None)
                chance_hits += h
            chance_hits /= 5.0

            # of the false accepts, how many are somebody else's detected spike
            foreign = 0
            if n_false > 0 and len(A):
                _, w2 = _match(A, missed)
                claimed = set(int(x) for x in w2 if x >= 0)
                order = np.argsort(A)
                fa = [int(A[order[i]]) for i in range(len(A)) if i not in claimed]
                for t in fa:
                    lo = np.searchsorted(st, t - TOLERANCE)
                    hi = np.searchsorted(st, t + TOLERANCE, "right")
                    if any(int(x) != best_c for x in cl[lo:hi]):
                        foreign += 1

            rec1 = (hits0 + rec_hits) / len(tt)
            prec1 = (hits0 + rec_hits) / (n_det0 + len(A)) if (n_det0 + len(A)) else np.nan
            R = out["R"]
            rows.append(dict(
                config=cfg, replicate=rep, unit=int(u), tier=tier,
                cluster=best_c, n_true=len(tt), n_missed=len(missed),
                n_det_before=n_det0,
                recall_before=rec0, precision_before=prec0,
                n_cand_total=out["n_cand_total"], n_refined=out["n_refined"],
                n_accepted=len(A), n_recovered=int(rec_hits),
                n_false=int(n_false), n_false_foreign=int(foreign),
                chance_recovered=chance_hits,
                recovery_precision=(rec_hits / len(A)) if len(A) else np.nan,
                recall_after=rec1, precision_after=prec1,
                d_recall=rec1 - rec0, d_precision=prec1 - prec0,
                n_iters=R["n_iters"], f0=R["f0"], peak_ch=R["peak_ch"],
                bar_r2=R["bar_r2"], bar_fp=R["bar_fp"],
                ref_r2=R["ref_r2"], ref_fp=R["ref_fp"],
                acc_r2=out["acc_r2"], acc_fp=out["acc_fp"],
                n_fail_r2=out["n_fail_r2"], n_fail_fp=out["n_fail_fp"],
                secs=round(time.time() - t0, 1)))
            print(f"  {cfg} r{rep} u{u:<4}{tier:<10} ch{R['peak_ch']:<4}"
                  f"cl{best_c:<5} rec {rec0:.3f}->{rec1:.3f}  "
                  f"prec {prec0:.3f}->{prec1:.3f}  "
                  f"acc {len(A)} (true {rec_hits}, chance {chance_hits:.1f}, "
                  f"foreign {foreign})  {time.time()-t0:.0f}s")

        # ---------------- merge detection, with a real answer for once
        # same-neuron pairs: a fragment of one injected unit vs its best
        # cluster (ground truth says MERGE). different-neuron pairs: the two
        # units of a pair-tier placement (ground truth says DO NOT MERGE).
        def viol_rate(ta, tb=None, refractory_ms=1.5):
            if tb is None:
                t = np.sort(ta)
            else:
                t = np.sort(np.concatenate([ta, tb]))
            if len(t) < 2:
                return np.nan
            d = np.diff(t) / FS * 1000.0
            return float((d < refractory_ms).mean())

        MIN_FRAG = 15
        pair_best = {}
        for u in np.unique(lab):
            sel = lab == u
            tt = np.sort(t_true[sel])
            tier = tier_names[int(tier_code[sel][0])]
            cand_cl = set()
            for t in tt:
                cand_cl.update(cl[np.abs(st - t) <= TOLERANCE].tolist())
            scored = []
            for c in sorted(cand_cl):
                h, _ = _match(st[cl == c], tt)
                scored.append((int(c), h))
            if not scored:
                continue
            scored.sort(key=lambda x: -x[1])
            bc = scored[0][0]
            pair_best[int(u)] = (bc, tier)
            for c, h in scored[1:]:
                if h < max(MIN_FRAG, int(0.02 * len(tt))):
                    continue
                merge_rows.append(dict(
                    config=cfg, replicate=rep, kind="same_neuron", tier=tier,
                    a=bc, b=c, truth="merge",
                    v_a=viol_rate(st[cl == bc]), v_b=viol_rate(st[cl == c]),
                    v_merged=viol_rate(st[cl == bc], st[cl == c]),
                    n_a=int((cl == bc).sum()), n_b=int((cl == c).sum())))
        seen = set()
        for u, (bc, tier) in pair_best.items():
            p = unit_pair.get(u, -1)
            if p < 0 or tier != "pair":
                continue
            partner = [v for v, (_, tv) in pair_best.items()
                       if v != u and unit_pair.get(v, -2) == p]
            for v in partner:
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                cb = pair_best[v][0]
                if cb == bc:
                    continue
                merge_rows.append(dict(
                    config=cfg, replicate=rep, kind="different_neuron", tier=tier,
                    a=bc, b=cb, truth="do_not_merge",
                    v_a=viol_rate(st[cl == bc]), v_b=viol_rate(st[cl == cb]),
                    v_merged=viol_rate(st[cl == bc], st[cl == cb]),
                    n_a=int((cl == bc).sum()), n_b=int((cl == cb).sum())))

    del sess

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "v2_post_hoc_correction.csv"), index=False)
md = pd.DataFrame(merge_rows)
md.to_csv(os.path.join(OUT, "v2_merge_detector_truth.csv"), index=False)
if cand_tables:
    pd.concat(cand_tables, ignore_index=True).to_csv(
        os.path.join(OUT, "v2_post_hoc_candidates.csv"), index=False)

if not len(df):
    print("no placements scored")
    sys.exit(0)

print("\n" + "=" * 100)
print("POST-HOC CORRECTION ON GROUND TRUTH: did the recovered spikes belong to the unit?")
print("=" * 100)
for cfg in df.config.unique():
    c = df[df.config == cfg]
    print(f"\n### {cfg}   ({len(c)} placements)")
    g = c.groupby("tier").agg(
        n=("unit", "size"),
        missed=("n_missed", "mean"),
        accepted=("n_accepted", "mean"),
        recovered=("n_recovered", "mean"),
        chance=("chance_recovered", "mean"),
        rec_prec=("recovery_precision", "mean"),
        rec_before=("recall_before", "mean"), rec_after=("recall_after", "mean"),
        prec_before=("precision_before", "mean"), prec_after=("precision_after", "mean"))
    g["d_recall"] = (g.rec_after - g.rec_before).round(3)
    g["d_prec"] = (g.prec_after - g.prec_before).round(3)
    print(g.round(3).to_string())
    print(f"  pooled: recall {c.recall_before.mean():.3f} -> {c.recall_after.mean():.3f}"
          f"   precision {c.precision_before.mean():.3f} -> {c.precision_after.mean():.3f}")
    print(f"  accepted {c.n_accepted.sum()} candidates; {c.n_recovered.sum()} were real "
          f"missed spikes of the unit ({100*c.n_recovered.sum()/max(1,c.n_accepted.sum()):.1f}%), "
          f"chance would give {c.chance_recovered.sum():.0f} "
          f"({100*c.chance_recovered.sum()/max(1,c.n_accepted.sum()):.1f}%)")
    print(f"  of {c.n_false.sum()} false accepts, {c.n_false_foreign.sum()} "
          f"({100*c.n_false_foreign.sum()/max(1,c.n_false.sum()):.1f}%) coincide with a "
          f"spike Kilosort filed under another cluster")
    print(f"  template rebuild: {c.n_iters.mean():.1f} iterations, f0 "
          f"{c.f0.min():.0f}-{c.f0.max():.0f} Hz")
    print(f"  gates: real-spike R2 {c.ref_r2.mean():.3f} (bar {c.bar_r2.mean():.3f}), "
          f"footprint {c.ref_fp.mean():.3f} (bar {c.bar_fp.mean():.3f}); "
          f"rejected {c.n_fail_r2.sum()} on R2, {c.n_fail_fp.sum()} on footprint")

if cand_tables:
    allc = pd.concat(cand_tables, ignore_index=True)
    print("\n" + "=" * 100)
    print("GATE SWEEP: what the two gates actually buy, per config")
    print("(quantiles are of each cluster's OWN real spikes, so 'r2 q10' = keep a")
    print(" candidate whose whitened fit beats the worst 10% of real spikes)")
    print("=" * 100)
    for cfg in allc.config.unique():
        c = allc[allc.config == cfg]
        tot_missed = df[df.config == cfg].n_missed.sum()
        print(f"\n### {cfg}  ({len(c)} candidates refined, {int(c.is_true.sum())} of them "
              f"are real missed spikes, out of {tot_missed} missed in total)")
        print(f"{'gate':<34}{'kept':>8}{'true':>8}{'recovery prec':>15}"
              f"{'of all misses':>15}")
        for name, mask in [
            ("no gate (matched filter only)", np.ones(len(c), bool)),
            ("footprint only", c.fp >= c.bar_fp),
            ("whitened R2 only", c.r2 >= c.bar_r2),
            ("both (what the tool ships)", (c.r2 >= c.bar_r2) & (c.fp >= c.bar_fp)),
            ("both, strict (R2 median)",
             (c.r2 >= c.groupby('cluster').r2.transform('median')) & (c.fp >= c.bar_fp)),
        ]:
            k = int(mask.sum())
            t = int(c.loc[mask, "is_true"].sum())
            print(f"  {name:<32}{k:>8}{t:>8}"
                  f"{(t/k if k else float('nan')):>15.3f}"
                  f"{(t/tot_missed if tot_missed else float('nan')):>15.3f}")

if len(md):
    print("\n" + "=" * 100)
    print("MERGE DETECTOR vs TRUTH (refractory violation rate of the merged train)")
    print("=" * 100)
    for cfg in md.config.unique():
        m = md[md.config == cfg]
        print(f"\n### {cfg}")
        g = m.groupby("kind").agg(n=("a", "size"), v_a=("v_a", "mean"),
                                  v_b=("v_b", "mean"), v_merged=("v_merged", "mean"))
        print((g * 1).round(4).to_string())
        for thr in (0.005, 0.01, 0.02):
            same = m[m.kind == "same_neuron"]
            diff = m[m.kind == "different_neuron"]
            tp = int((same.v_merged <= thr).sum())
            fp = int((diff.v_merged <= thr).sum())
            print(f"  threshold merged-violation <= {thr:.3f}: "
                  f"says MERGE for {tp}/{len(same)} true fragments (sensitivity "
                  f"{tp/max(1,len(same)):.2f}), and wrongly for {fp}/{len(diff)} "
                  f"genuinely different pairs (false-merge {fp/max(1,len(diff)):.2f})")

print("\nsaved v2_post_hoc_correction.csv, v2_merge_detector_truth.csv")
