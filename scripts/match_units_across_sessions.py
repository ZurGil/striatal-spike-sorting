# -*- coding: utf-8 -*-
"""Is unit 137 of one session the same neuron as unit 212 of the next?

    python scripts/match_units_across_sessions.py --calibrate 20260917_100043
    python scripts/match_units_across_sessions.py --a 20260917_100043 --b 20260918_100248

Gil's question: "are you capable of matching one neuron from one session to another
session?" The answer is only worth anything with an error rate attached, so this script
does not just produce matches -- it produces matches whose recall and precision were
measured on pairs that are the same neuron BY CONSTRUCTION.

THE POSITIVE CONTROL IS THE WHOLE METHOD
----------------------------------------
`extract_unit_waveforms.py` saved every unit's raw mean waveform TWICE: the first half
of its spikes and the second half, split by time. Treat those two halves as two
recordings and every unit has a partner that is certainly itself, and 500-odd other
candidates that are certainly not. Run the exact matching procedure on that pair of
half-sessions and recall and precision fall out by counting, with no threshold taken on
faith from a paper.

Splitting by time (not alternate spikes) is what makes the control honest: within-session
drift lands inside the same-neuron distribution, which is the thing that will actually
degrade a real across-session match.

WHAT IS MATCHED ON -- ELECTRICAL SIGNATURE ONLY
----------------------------------------------
Three features per candidate pair, all from the raw mean waveform in uV:

  r_amp    correlation of the peak-to-peak amplitude across local channels -- the
           footprint, i.e. which sites the cell reaches and how strongly
  r_shape   correlation of the full waveform matrix (time x local channels) -- shape
  d_cent    distance between the amplitude-weighted centroids, in um

Never on tuning, PSTH or firing rate. These sessions are a learning series, so matching
on responses would manufacture the very stability the series is meant to measure.
Functional agreement afterwards is validation; it is never a feature.

The three features are combined by a Fisher linear discriminant fit on the split-half
control of the session being matched, so the weighting is calibrated on this probe and
this rat rather than assumed.

CANDIDATES ARE GEOMETRICALLY LOCAL
----------------------------------
Only pairs whose centroids sit within --radius um of each other are scored. Two units
600 um apart are not a hard case and counting them as correct rejections would inflate
precision into meaninglessness. Every number below is therefore "among units that could
plausibly be confused".

DRIFT
-----
The probe never moved between these sessions, but the tissue does. The median y-offset
between sessions is estimated from the confidently matched pairs and reported; pass
--apply-drift to subtract it and re-match.

Writes outputs/unit_matching/<a>__<b>_matches.csv and calibration_<session>.csv.
Nothing on Z is written.
"""
from __future__ import print_function

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "unit_matching")

WAVEFORMS = {
    "20260914_122553": "Z:/Gil/Shamir_1/20260914_122553.rec/20260914_122553.kilosort/waveforms_cluster",
    "20260915_103710": "Z:/Gil/Shamir_1/20260915_103710.rec/20260915_103710.kilosort/waveforms",
    "20260916_110311": "Z:/Gil/Shamir_1/20260916_110311.rec/20260916_110311.kilosort/waveforms_curated_z",
    "20260917_100043": "Z:/Gil/Shamir_1/20260917_100043.rec/20260917_100043.kilosort/waveforms_cluster",
    "20260918_100248": "Z:/Gil/Shamir_1/20260918_100248.rec/20260918_100248.kilosort/waveforms_cluster",
    "20261112_082028": "Z:/Gil/Popper/20261112_082028.rec/20261112_082028.kilosort/waveforms",
}

RADIUS = 60.0        # um, channels included around a unit's peak site
CAND_RADIUS = 80.0   # um, centroid distance inside which a pair is even considered
MIN_SPIKES = 20      # per half


def load(session):
    d = WAVEFORMS[session]
    z = np.load(os.path.join(d, "unit_waveforms.npz"))
    with open(os.path.join(d, "unit_waveforms.json")) as f:
        meta = json.load(f)
    return dict(unit_ids=z["unit_ids"], mean_wf=z["mean_wf"], n_spikes=z["n_spikes"],
                peak_chan=z["peak_chan"], centroid=z["centroid"],
                channel_pos=z["channel_pos"], meta=meta, session=session)


def halves(S):
    """Flatten a session into a list of half-records, one per (unit, half)."""
    wf = S["mean_wf"]
    pos = S["channel_pos"]
    recs = []
    for i, uid in enumerate(S["unit_ids"]):
        for h in (0, 1):
            if S["n_spikes"][i, h] < MIN_SPIKES:
                continue
            w = wf[i, h]                                  # (82, nchan), uV
            ptp = w.max(axis=0) - w.min(axis=0)
            pk = int(np.argmax(ptp))
            loc = np.where(np.sum((pos - pos[pk]) ** 2, axis=1) <= RADIUS ** 2)[0]
            wsub = w[:, loc]
            amp = ptp[loc]
            cx, cy = S["centroid"][i, h]
            recs.append(dict(uid=int(uid), half=h, pk=pk, loc=loc, amp=amp,
                             wsub=wsub, x=float(cx), y=float(cy),
                             max_ptp=float(ptp[pk]), n=int(S["n_spikes"][i, h])))
    return recs


def _r(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na = np.sqrt(np.dot(a, a))
    nb = np.sqrt(np.dot(b, b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def features(A, B, dy=0.0):
    """Three features for one candidate pair. B's y is shifted by dy first."""
    pos_a, pos_b = A["loc"], B["loc"]
    shared = np.intersect1d(pos_a, pos_b)
    if shared.size < 4:
        return None
    ia = np.searchsorted(pos_a, shared)
    ib = np.searchsorted(pos_b, shared)
    r_amp = _r(A["amp"][ia], B["amp"][ib])
    r_shape = _r(A["wsub"][:, ia].ravel(), B["wsub"][:, ib].ravel())
    d = np.hypot(A["x"] - B["x"], A["y"] - (B["y"] - dy))
    return np.array([r_amp, r_shape, -d / 100.0])


def score_matrix(recs_a, recs_b, w, dy=0.0, cand_radius=CAND_RADIUS):
    """Pairwise score, NaN where the pair is not even a candidate."""
    na, nb = len(recs_a), len(recs_b)
    S = np.full((na, nb), np.nan)
    ya = np.array([r["y"] for r in recs_a])
    xa = np.array([r["x"] for r in recs_a])
    yb = np.array([r["y"] for r in recs_b]) - dy
    xb = np.array([r["x"] for r in recs_b])
    F = {}
    for i in range(na):
        near = np.where(np.hypot(xa[i] - xb, ya[i] - yb) <= cand_radius)[0]
        for j in near:
            f = features(recs_a[i], recs_b[j], dy=dy)
            if f is None:
                continue
            F[(i, j)] = f
            S[i, j] = np.nan if w is None else float(np.dot(w, f))
    return S, F


def fisher(pos, neg):
    """Fisher linear discriminant: the direction that separates two clouds."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    mu = pos.mean(axis=0) - neg.mean(axis=0)
    Sw = np.cov(pos.T) * (len(pos) - 1) + np.cov(neg.T) * (len(neg) - 1)
    Sw = Sw / (len(pos) + len(neg) - 2) + np.eye(pos.shape[1]) * 1e-6
    w = np.linalg.solve(Sw, mu)
    return w / np.sqrt(np.dot(w, w))


def mutual_best(S, thr):
    """Reciprocal best match above threshold. Returns list of (i, j, score)."""
    out = []
    Sm = np.where(np.isnan(S), -np.inf, S)
    best_j = np.argmax(Sm, axis=1)
    best_i = np.argmax(Sm, axis=0)
    for i in range(S.shape[0]):
        j = int(best_j[i])
        if Sm[i, j] >= thr and int(best_i[j]) == i:
            out.append((i, j, float(Sm[i, j])))
    return out


def calibrate(S_sess, verbose=True, orphan_frac=0.5):
    """Fit the score and measure recall/precision on split halves of one session."""
    recs = halves(S_sess)
    h1 = [r for r in recs if r["half"] == 0]
    h2 = [r for r in recs if r["half"] == 1]
    if verbose:
        print("%s: %d units, %d/%d halves usable"
              % (S_sess["session"], len(S_sess["unit_ids"]), len(h1), len(h2)))

    # pass 1: raw features for every candidate pair, to fit the discriminant
    _, F = score_matrix(h1, h2, None)
    same, diff = [], []
    for (i, j), f in F.items():
        if h1[i]["uid"] == h2[j]["uid"]:
            same.append(f)
        else:
            diff.append(f)
    if len(same) < 20 or len(diff) < 20:
        raise SystemExit("not enough candidate pairs to calibrate")
    w = fisher(same, diff)
    if verbose:
        print("  candidate pairs: %d same-neuron, %d different-neuron (within %.0f um)"
              % (len(same), len(diff), CAND_RADIUS))
        print("  score = %+.3f*r_amp %+.3f*r_shape %+.3f*(-d/100um)" % tuple(w))

    S = np.full((len(h1), len(h2)), np.nan)
    for (i, j), f in F.items():
        S[i, j] = float(np.dot(w, f))

    truth = {}
    for j, r in enumerate(h2):
        truth[r["uid"]] = j

    rows = []
    s_same = np.array([np.dot(w, f) for f in same])
    s_diff = np.array([np.dot(w, f) for f in diff])
    if verbose:
        print("  score, same neuron : median %.3f, 5th pct %.3f, min %.3f"
              % (np.median(s_same), np.percentile(s_same, 5), s_same.min()))
        print("  score, different   : median %.3f, 95th pct %.3f, max %.3f"
              % (np.median(s_diff), np.percentile(s_diff, 95), s_diff.max()))
    grid = np.unique(np.round(np.concatenate([
        np.percentile(s_same, np.arange(0, 100, 1)),
        np.percentile(s_diff, np.arange(50, 100, 2)),
        [s_diff.min() - 1e-6]]), 4))
    for thr in grid:
        m = mutual_best(S, thr)
        tp = sum(1 for i, j, _ in m if h1[i]["uid"] == h2[j]["uid"])
        fp = len(m) - tp
        n_true = sum(1 for r in h1 if r["uid"] in truth)
        rows.append(dict(thr=float(thr), matches=len(m), tp=tp, fp=fp,
                         recall=tp / float(n_true) if n_true else 0.0,
                         precision=tp / float(len(m)) if m else 1.0))
    # The split-half control above is too kind in one specific way: there, EVERY unit
    # has a true partner, so reciprocal-best-match alone can hardly go wrong. Across
    # sessions many units simply are not there the next day, and an orphan paired with
    # another orphan is a false match that no mutual-best rule catches. So: delete a
    # fraction of the second half-set at random, which makes that fraction of the first
    # set orphans, and measure what the procedure does with them.
    rng = np.random.RandomState(0)
    orows = []
    for thr in grid:
        tp = fp = n_true = n_m = 0
        for rep in range(5):
            keep = rng.rand(len(h2)) >= orphan_frac
            Sm = np.where(keep[None, :], S, np.nan)
            m = mutual_best(Sm, thr)
            alive = set(h2[j]["uid"] for j in np.where(keep)[0])
            tp += sum(1 for i, j, _ in m if h1[i]["uid"] == h2[j]["uid"])
            n_m += len(m)
            n_true += sum(1 for r in h1 if r["uid"] in alive)
        fp = n_m - tp
        orows.append(dict(thr=float(thr), matches=n_m / 5.0, tp=tp / 5.0, fp=fp / 5.0,
                          recall=tp / float(n_true) if n_true else 0.0,
                          precision=tp / float(n_m) if n_m else 1.0))

    return dict(w=w, table=rows, orphan_table=orows, recs=recs, h1=h1, h2=h2, S=S,
                s_same=s_same,
                s_diff=np.array([np.dot(w, f) for f in diff]))


def tiers(table):
    """Highest-recall threshold reaching each precision target."""
    out = []
    for target in (0.99, 0.95, 0.90):
        ok = [r for r in table if r["precision"] >= target]
        if ok:
            best = max(ok, key=lambda r: r["recall"])
            out.append((target, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--radius", type=float, default=CAND_RADIUS)
    ap.add_argument("--apply-drift", action="store_true")
    ap.add_argument("--precision", type=float, default=0.95,
                    help="precision target that sets the matching threshold")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sessions = [s for s in (args.calibrate, args.a, args.b) if s]
    if not sessions:
        raise SystemExit("give --calibrate SESSION or --a A --b B")

    cal_sess = args.calibrate or args.a
    cal = calibrate(load(cal_sess))
    print("\nsplit-half control on %s -- the same neuron by construction:" % cal_sess)
    print("  %-10s %8s %6s %6s %8s %10s" % ("threshold", "matches", "tp", "fp",
                                            "recall", "precision"))
    for target, r in tiers(cal["table"]):
        print("  %-10.3f %8d %6d %6d %7.1f%% %9.1f%%  <- precision >= %.0f%%"
              % (r["thr"], r["matches"], r["tp"], r["fp"], 100 * r["recall"],
                 100 * r["precision"], 100 * target))
    print("\n  same control, but with half of the second half-set deleted, so half the\n"
          "  units have no true partner at all -- the across-session case:")
    print("  %-10s %8s %6s %6s %8s %10s" % ("threshold", "matches", "tp", "fp",
                                            "recall", "precision"))
    for target, r in tiers(cal["orphan_table"]):
        print("  %-10.3f %8.1f %6.1f %6.1f %7.1f%% %9.1f%%  <- precision >= %.0f%%"
              % (r["thr"], r["matches"], r["tp"], r["fp"], 100 * r["recall"],
                 100 * r["precision"], 100 * target))

    with open(os.path.join(OUT, "calibration_%s.csv" % cal_sess), "w") as f:
        f.write("control,thr,matches,tp,fp,recall,precision\n")
        for r in cal["table"]:
            f.write("all_present,%.4f,%.1f,%.1f,%.1f,%.4f,%.4f\n"
                    % (r["thr"], r["matches"], r["tp"], r["fp"], r["recall"],
                       r["precision"]))
        for r in cal["orphan_table"]:
            f.write("half_deleted,%.4f,%.1f,%.1f,%.1f,%.4f,%.4f\n"
                    % (r["thr"], r["matches"], r["tp"], r["fp"], r["recall"],
                       r["precision"]))

    thr = None
    for target, r in tiers(cal["orphan_table"]):
        if abs(target - args.precision) < 1e-9:
            thr = r["thr"]
    if thr is None:
        thr = float(np.percentile(cal["s_same"], 5))

    if not (args.a and args.b):
        return

    A, B = load(args.a), load(args.b)
    ra, rb = halves(A), halves(B)
    a2 = [r for r in ra if r["half"] == 1]      # last half of A
    b1 = [r for r in rb if r["half"] == 0]      # first half of B -- closest in time
    w = cal["w"]

    dy = 0.0
    for it in range(3 if args.apply_drift else 1):
        S, _ = score_matrix(a2, b1, w, dy=dy, cand_radius=args.radius)
        m = mutual_best(S, thr)
        if not args.apply_drift or not m:
            break
        dys = [a2[i]["y"] - b1[j]["y"] for i, j, _ in m]
        dy = dy + float(np.median(dys))
    print("\n%s -> %s : %d x %d units, y-offset %s"
          % (args.a, args.b, len(a2), len(b1),
             ("%+.1f um (applied)" % dy) if args.apply_drift else "not applied"))
    print("  %d matches at threshold %.3f (split-half precision %.0f%%)"
          % (len(m), thr, 100 * args.precision))

    path = os.path.join(OUT, "%s__%s_matches.csv" % (args.a, args.b))
    with open(path, "w") as f:
        f.write("unit_a,unit_b,score,r_amp,r_shape,d_centroid_um,"
                "max_ptp_a_uV,max_ptp_b_uV,n_spikes_a,n_spikes_b\n")
        for i, j, s in sorted(m, key=lambda t: -t[2]):
            fe = features(a2[i], b1[j], dy=dy)
            f.write("%d,%d,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f,%d,%d\n"
                    % (a2[i]["uid"], b1[j]["uid"], s, fe[0], fe[1], -fe[2] * 100.0,
                       a2[i]["max_ptp"], b1[j]["max_ptp"], a2[i]["n"], b1[j]["n"]))
    print("  wrote", path)
    top = sorted(m, key=lambda t: -t[2])[:10]
    print("  %-10s %-10s %7s %7s %8s" % ("unit A", "unit B", "score", "r_amp", "d (um)"))
    for i, j, s in top:
        fe = features(a2[i], b1[j], dy=dy)
        print("  %-10d %-10d %7.3f %7.3f %8.1f"
              % (a2[i]["uid"], b1[j]["uid"], s, fe[0], -fe[2] * 100.0))


if __name__ == "__main__":
    main()
