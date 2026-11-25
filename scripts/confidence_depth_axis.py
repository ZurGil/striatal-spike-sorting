# -*- coding: utf-8 -*-
"""Where along the probe does the confidence / error signal live?

    python scripts/confidence_depth_axis.py

The implant is one straight Neuropixels shank at ML -2.45, AP +1.2, and the good
units span 2.76 to 6.56 mm below the brain surface -- i.e. the whole dorsal-to-
ventral striatal axis in one penetration. Every OCD model of this circuit makes a
claim about THAT axis and not about the striatum as a lump: the OFC/ACC -> ventro-
medial striatum limb is the one implicated, the sensorimotor dorsolateral limb is
the habit side, and the two are supposed to carry different things.

Nothing new is fitted for the single-unit part: it joins depth_mm onto the per-unit
rows the confidence check already wrote, so the depth profile and the pooled number
in the paper figure are the same fit.

The population part IS new and has to be: a dorsal and a ventral half-population
are decoded separately, subsampled to the same unit count so the comparison is not
a comparison of n.

Writes outputs/confidence_depth/{units_with_depth.csv, decode_by_depth.csv}.
Nothing on Z is touched.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import confidence_coding_quick_check as C  # noqa: E402

# The same sessions the check and the figure use. This was a hand-written list of
# three, frozen in October, which silently answered the depth question on a third
# of the data every time the main analysis grew.
SESSIONS = list(C.DEFAULT_SESSIONS)
SPLIT_MM = 4.70        # dorsal / ventral boundary, below the brain surface
N_SUB = 10             # subsample repeats for the equal-n decode
EPOCHS_DECODE = ["cue", "chin"]


def depth_table(sid):
    S = C.SESSIONS[sid]
    u = pd.read_parquet(S + "/units.parquet")
    u = u[u.quality_label == "good"]
    return dict(zip(u.unit_id.astype(int), u.depth_mm.astype(float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "outputs", "confidence_coding_shamir"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "confidence_depth"))
    args = ap.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # ---------------------------------------------------------------- single units
    frames = []
    for sid in SESSIONS:
        d = pd.read_csv(os.path.join(args.src, "%s_units.csv" % sid))
        dep = depth_table(sid)
        d["depth_mm"] = d.unit_id.map(dep)
        frames.append(d)
    U = pd.concat(frames, ignore_index=True)
    U = U[U.depth_mm.notna()].copy()
    U["half"] = np.where(U.depth_mm >= SPLIT_MM, "ventral", "dorsal")
    U.to_csv(os.path.join(args.out, "units_with_depth.csv"), index=False)

    print("good units with depth: %d   span %.2f-%.2f mm   split at %.2f"
          % (U.unit_id.nunique(), U.depth_mm.min(), U.depth_mm.max(), SPLIT_MM))

    # significance is FDR-corrected within session, as the check itself does it
    rows = []
    for ep in C.EPOCHS:
        for key in ("absDV", "conf"):
            sig = np.zeros(len(U), dtype=bool)
            for (sid, _), g in U[U.epoch == ep].groupby(["session", "epoch"]):
                s = C.bh_fdr(g[key + "_p"].values)
                sig[U.index.get_indexer(g.index[s])] = True
            sub = U[U.epoch == ep]
            m = sig[U.index.get_indexer(sub.index)]
            for half in ("dorsal", "ventral"):
                h = (sub.half == half).values
                rows.append(dict(epoch=ep, target=key, half=half,
                                 n_units=int(h.sum()),
                                 n_sig=int((m & h).sum()),
                                 frac=float((m & h).sum()) / max(int(h.sum()), 1)))
    byhalf = pd.DataFrame(rows)
    byhalf.to_csv(os.path.join(args.out, "frac_sig_by_half.csv"), index=False)
    print("\n  epoch target   dorsal              ventral")
    for ep in C.EPOCHS:
        for key in ("absDV", "conf"):
            a = byhalf[(byhalf.epoch == ep) & (byhalf.target == key) & (byhalf.half == "dorsal")].iloc[0]
            b = byhalf[(byhalf.epoch == ep) & (byhalf.target == key) & (byhalf.half == "ventral")].iloc[0]
            print("  %-5s %-6s %3d/%3d (%4.1f%%)   %3d/%3d (%4.1f%%)"
                  % (ep, key, a.n_sig, a.n_units, 100 * a.frac,
                     b.n_sig, b.n_units, 100 * b.frac))

    # ---------------------------------------------------------------- populations
    dec = []
    for sid in SESSIONS:
        S = C.SESSIONS[sid]
        t = C.build_trials(S, sid)
        units = pd.read_parquet(S + "/units.parquet")
        good = units.loc[units.quality_label == "good", "unit_id"].values
        uids, counts, durs = C.count_spikes(S, t, good)
        keep = C.selection_mask(S, t, uids)
        dep = depth_table(sid)
        udep = np.array([dep.get(int(u), np.nan) for u in uids])

        for ep in EPOCHS_DECODE:
            dur = durs[ep]
            ok_t = dur > 0.02
            R = counts[ep][ok_t] / dur[ok_t][:, None]
            # A population read-out needs one trial set, so only units that were
            # present for the whole session enter it -- dropping the unit, never
            # the trial, which is what keep.all(axis=1) was doing by accident.
            whole = keep.all(axis=0)
            live = whole & (R.mean(axis=0) >= C.MIN_RATE_HZ) & np.isfinite(udep)
            err = 1.0 - t.correct.values[ok_t]
            if ok_t.sum() < 150 or err.sum() < 30:
                continue
            idx_d = np.where(live & (udep < SPLIT_MM))[0]
            idx_v = np.where(live & (udep >= SPLIT_MM))[0]
            n = min(len(idx_d), len(idx_v))
            if n < 15:
                continue
            rng = np.random.default_rng(3)
            for half, idx in (("dorsal", idx_d), ("ventral", idx_v)):
                aucs = []
                for r in range(N_SUB):
                    pick = rng.choice(idx, size=n, replace=False)
                    p = C.ridge_cv_pred(R[:, pick], err, seed=r)
                    a = C.auc(err, p)
                    if np.isfinite(a):
                        aucs.append(a)
                full = C.decode_binary(R[:, idx], err, n_perm=100, seed=0)
                dec.append(dict(session=sid, epoch=ep, half=half, n_avail=len(idx),
                                n_sub=n, auc_sub=float(np.mean(aucs)),
                                auc_sub_sd=float(np.std(aucs)),
                                auc_all=full["auc"] if full else np.nan,
                                null=full["null_median"] if full else np.nan,
                                p=full["p"] if full else np.nan,
                                n_trials=int(ok_t.sum()), n_err=int(err.sum())))
                print("  %s %-5s %-7s n=%3d/%3d  AUC(equal n)=%.3f+-%.3f  AUC(all)=%.3f p=%.3f"
                      % (sid, ep, half, n, len(idx), dec[-1]["auc_sub"],
                         dec[-1]["auc_sub_sd"], dec[-1]["auc_all"], dec[-1]["p"]))
    pd.DataFrame(dec).to_csv(os.path.join(args.out, "decode_by_depth.csv"), index=False)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
