"""Compare two Kilosort4 runs of the SAME recording, unit by unit.

This is NOT a ground-truth comparison. Both sorts are estimates of the same
truth, so "accuracy" is meaningless here and the only honest question is
agreement: did the second run reproduce the first? That makes every number
directional, which is why each matched pair gets BOTH

    recall    = matched / n_spikes in the reference unit
    precision = matched / n_spikes in the test unit

A unit the test run split in two scores high precision on each fragment and low
recall on both. A unit the test run merged with a neighbour scores high recall
and low precision. One number cannot tell those apart, so there is never one
number here.

Results are tiered by the reference sort's own KSLabel, because the expectation
differs by tier: well-isolated units should reproduce almost exactly, while MUA
clusters are partly an arbitrary partition of the noise floor and will not.
Reporting them together hides both facts.

    python compare_two_sorts.py --ref <dir> --test <dir> --out <csv>

HOW THE MATCHING WORKS, AND WHY IT IS NOT THE OBVIOUS THING

Across 384 channels a 3.5 h session has 25 M spikes, and 75% of them have some
other spike within 20 samples. So "for each reference spike, find the nearest
test spike and credit its cluster" is wrong -- the nearest spike in the whole
recording usually belongs to a different cell firing at nearly the same moment.
An earlier version of this script did exactly that and scored a sort against
ITSELF at 0.963 instead of 1.000, which is how the bug was found.

Comparing every cluster pair directly is correct but is 200 x 200 Python loops
over millions of spikes, which does not finish. So it is done in two stages:

  1. a cheap upper bound -- for each reference spike take the WINDOW of all test
     spikes within +/- tol and tally their clusters. Pairs, not nearest, so no
     real partner is ever missed.
  2. an exact count for the few clusters that bound says are plausible. Within
     one cluster the refractory period keeps spikes far apart compared with tol,
     so a binary search for the nearest spike of THAT cluster is unambiguous and
     the count is exact. The script verifies that assumption instead of trusting
     it, and warns per cluster if it does not hold.
"""

import argparse
import os
import sys

import numpy as np


def read_kslabel(d):
    """KSLabel for each cluster, looking one directory up as well.

    kilosort_original/ holds the pristine arrays but not always the TSVs -- the
    labels sit in the parent phy directory. Falling back there beats silently
    reporting every unit as unlabelled, which is what the first version did.
    """
    for cand in (d, os.path.dirname(os.path.normpath(d))):
        p = os.path.join(cand, "cluster_KSLabel.tsv")
        if not os.path.exists(p):
            continue
        out = {}
        with open(p) as fh:
            next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip().isdigit():
                    out[int(parts[0])] = parts[1].strip()
        if out:
            return out, p
    return {}, None


def load(d):
    for f in ("spike_times.npy", "spike_clusters.npy"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            raise SystemExit("missing %s" % p)
    st = np.load(os.path.join(d, "spike_times.npy")).astype(np.int64).ravel()
    sc = np.load(os.path.join(d, "spike_clusters.npy")).astype(np.int64).ravel()
    if st.size != sc.size:
        raise SystemExit("%s: spike_times (%d) and spike_clusters (%d) disagree"
                         % (d, st.size, sc.size))
    order = np.argsort(st, kind="stable")
    st, sc = st[order], sc[order]
    label, lpath = read_kslabel(d)
    return st, sc, label, lpath


def candidate_counts(ref, t_st, t_sc_dense, n_test, tol):
    """Stage 1: an upper bound on the overlap with every test cluster.

    Counts (reference spike, test spike) PAIRS within +/- tol, so a cluster that
    shares anything with this reference unit is guaranteed a non-zero tally. It
    can over-count when two test spikes of one cluster fall inside one window,
    which is exactly why stage 2 exists.
    """
    lo = np.searchsorted(t_st, ref - tol, side="left")
    hi = np.searchsorted(t_st, ref + tol, side="right")
    k = hi - lo
    total = int(k.sum())
    if total == 0:
        return np.zeros(n_test, dtype=np.int64)
    # Expand each [lo, hi) window into flat indices without a Python loop.
    starts = np.repeat(lo, k)
    offsets = np.arange(total) - np.repeat(np.cumsum(k) - k, k)
    return np.bincount(t_sc_dense[starts + offsets],
                       minlength=n_test).astype(np.int64)


def exact_overlap(ref, test, tol):
    """Stage 2: how many reference spikes have a partner in THIS cluster.

    Exact, not approximate. searchsorted brackets each reference spike between
    test[lo] and test[hi], and the nearest spike of the train is necessarily one
    of those two -- so if any spike of this cluster is within tol, the nearer of
    the pair is too. Internal spacing of the cluster does not matter.

    The direction that CAN go wrong is injectivity: two reference spikes close
    together could both claim the same test spike, pushing the count above the
    test train's length and precision above 1. Returns (count, ok) with ok False
    in that case, so the caller reports it rather than printing a precision of
    1.04 as if it meant something.
    """
    if ref.size == 0 or test.size == 0:
        return 0, True
    hi = np.searchsorted(test, ref, side="left")
    lo = np.clip(hi - 1, 0, test.size - 1)
    hi = np.clip(hi, 0, test.size - 1)
    d = np.minimum(np.abs(ref - test[lo]), np.abs(ref - test[hi]))
    n = int((d <= tol).sum())
    return min(n, test.size), n <= test.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the sort to reproduce (phy dir)")
    ap.add_argument("--test", required=True, help="the new sort (phy dir)")
    ap.add_argument("--out", required=True, help="per-unit CSV to write")
    ap.add_argument("--tol", type=int, default=10,
                    help="samples; 10 = 0.33 ms at 30 kHz. Two runs of the same "
                         "algorithm on the same data align to within a sample or "
                         "two, so this is already generous")
    ap.add_argument("--min-spikes", type=int, default=50,
                    help="skip reference units below this; they carry no signal "
                         "about agreement either way")
    ap.add_argument("--candidates", type=int, default=12,
                    help="how many stage-1 candidates to score exactly")
    a = ap.parse_args()

    r_st, r_sc, r_lab, r_lp = load(a.ref)
    t_st, t_sc, t_lab, _ = load(a.test)

    print("reference : %s" % a.ref)
    print("            %d spikes, %d clusters, span %d..%d samples"
          % (r_st.size, np.unique(r_sc).size, r_st[0], r_st[-1]))
    print("            KSLabel from %s" % (r_lp or "NOWHERE -- units unlabelled"))
    print("test      : %s" % a.test)
    print("            %d spikes, %d clusters, span %d..%d samples"
          % (t_st.size, np.unique(t_sc).size, t_st[0], t_st[-1]))
    print("            spike count differs by %+.2f%%"
          % (100.0 * (t_st.size - r_st.size) / r_st.size))
    print("tolerance : +/- %d samples (%.2f ms)" % (a.tol, a.tol / 30.0))
    print()

    # A different recording length means the two runs did not see the same data,
    # and every number below would be quietly wrong rather than loudly wrong.
    span_ratio = float(t_st[-1]) / float(r_st[-1])
    if not 0.98 < span_ratio < 1.02:
        print("STOP: the two sorts span different durations (ratio %.4f) -- these "
              "are not the same recording" % span_ratio)
        return 2

    r_ids = np.unique(r_sc)
    t_ids, t_sc_dense = np.unique(t_sc, return_inverse=True)
    t_sizes = np.bincount(t_sc_dense, minlength=t_ids.size)
    n_test = t_ids.size
    T = {k: t_st[t_sc_dense == k] for k in range(n_test)}

    rows, claimed, shaky = [], set(), []
    for cid in sorted((int(c) for c in r_ids),
                      key=lambda c: -int((r_sc == c).sum())):
        ref = r_st[r_sc == cid]
        if ref.size < a.min_spikes:
            continue
        bound = candidate_counts(ref, t_st, t_sc_dense, n_test, a.tol)
        cand = [int(k) for k in np.argsort(-bound)[:a.candidates] if bound[k] > 0]

        # Exact counts once per candidate, reused by both matchings below.
        counts = {}
        for k in cand:
            n, ok = exact_overlap(ref, T[k], a.tol)
            counts[k] = n
            if not ok and k not in shaky:
                shaky.append(k)

        # Ties on matched count are common for small units in dense data: a
        # 60-spike unit's own cluster overlaps all 60, and so can a 200k-spike
        # cluster that happens to have something within tol of each one. Break
        # the tie on precision -- among equal overlaps the purest cluster is the
        # better match. Without this a self-comparison reports a spurious merge.
        def rank(k):
            return (counts[k], counts[k] / float(t_sizes[k]))

        # OPEN match: best overlap regardless of who else wants it. This answers
        # "were the spikes found at all", and the structural verdicts read off it.
        o_best = max(counts, key=rank) if counts else None
        if o_best is not None and counts[o_best] == 0:
            o_best = None

        # EXCLUSIVE match: each test cluster claimed once, so the leftovers at
        # the end are real. Kept because it is the only way to count units
        # unaccounted for on each side -- but on its own it LIES about merges.
        # When the test sort fuses two reference units, the larger claims the
        # cluster and the smaller is handed an unrelated leftover, scoring ~0.00
        # and looking like a unit the test sort missed. Measured here: reference
        # unit 94 scores 0.006 exclusive and 0.861 open, its spikes sitting
        # inside a 197k-spike cluster it simply does not own.
        free = [k for k, v in counts.items() if k not in claimed and v > 0]
        x_best = max(free, key=rank) if free else None
        if x_best is not None:
            claimed.add(x_best)

        def q(k):
            if k is None:
                return (-1, 0, 0, 0.0, 0.0, "")
            nt, nm = int(t_sizes[k]), counts[k]
            return (int(t_ids[k]), nt, nm, nm / float(ref.size), nm / float(nt),
                    t_lab.get(int(t_ids[k]), ""))

        rows.append((cid, ref.size, r_lab.get(cid, "")) + q(x_best) + q(o_best)[:5])

    with open(a.out, "w") as fh:
        fh.write("ref_unit,ref_n,ref_kslabel,"
                 "test_unit,test_n,n_matched,recall,precision,test_kslabel,"
                 "open_test_unit,open_test_n,open_matched,open_recall,"
                 "open_precision\n")
        for r in rows:
            fh.write("%d,%d,%s,%d,%d,%d,%.4f,%.4f,%s,%d,%d,%d,%.4f,%.4f\n" % r)
    print("wrote %s (%d reference units scored)" % (a.out, len(rows)))
    if shaky:
        print("NOTE: %d test cluster(s) had more reference spikes claim them than "
              "they hold, so two reference spikes paired with one test spike. "
              "Their matched counts are capped at the cluster size: %s"
              % (len(shaky), ", ".join(str(int(t_ids[k])) for k in shaky[:10])))
    print()

    def report(name, sel, ri, pi):
        if not sel:
            print("%-14s none" % name)
            return
        rec = np.array([r[ri] for r in sel])
        pre = np.array([r[pi] for r in sel])
        print("%-14s n=%d" % (name, len(sel)))
        for lbl, v in (("recall", rec), ("precision", pre)):
            print("    %-10s median %.3f   IQR %.3f-%.3f   min %.3f   >=0.90: %d/%d"
                  % (lbl, np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                     v.min(), int((v >= 0.90).sum()), len(v)))

    g = [r for r in rows if r[2] == "good"]
    m = [r for r in rows if r[2] == "mua"]
    rest = [r for r in rows if r[2] not in ("good", "mua")]

    # Weighted by spikes, not by unit, because a unit with 300 spikes and one
    # with 300,000 are not the same evidence. Both are printed: the per-unit
    # medians above say how a typical cluster fared, this says how much of the
    # recording the two sorts actually agree about.
    def mass(sel, ri):
        tot = sum(r[1] for r in sel)
        if not tot:
            return 0.0, 0.0
        agree = sum(r[1] for r in sel if r[ri] >= 0.80)
        return 100.0 * agree / tot, tot

    for title, ri, pi in (("OPEN match -- were the spikes found at all?", 12, 13),
                          ("EXCLUSIVE match -- one test cluster per unit", 6, 7)):
        print("=== %s ===" % title)
        print("(recall = reference spikes found; precision = test unit purity)")
        print()
        report("good", g, ri, pi)
        report("mua", m, ri, pi)
        if rest:
            report("unlabelled", rest, ri, pi)
        pc, tot = mass(rows, ri)
        print("    %-10s %.2f%% of all %d reference spikes sit in units with "
              "recall >= 0.80" % ("by mass", pc, tot))
        print()

    # Split and merge are the two failure modes a per-pair score cannot show on
    # its own, so they are counted separately. Read off the OPEN match, because
    # a merge is precisely the case the exclusive match cannot represent.
    splits = [r for r in rows if r[12] < 0.80 <= r[13]]
    merges = [r for r in rows if r[13] < 0.80 <= r[12]]
    neither = [r for r in rows if r[12] < 0.80 and r[13] < 0.80]
    print("=== structural verdicts, from the OPEN match ===")
    print("  SPLIT by the test sort (high precision, low recall)  : %d" % len(splits))
    print("  MERGED by the test sort (high recall, low precision) : %d" % len(merges))
    print("  no counterpart at all, even unclaimed                : %d  (%.2f%% of spikes)"
          % (len(neither),
             100.0 * sum(r[1] for r in neither) / max(1, sum(r[1] for r in rows))))
    print("  reference units with NO overlapping test cluster     : %d"
          % len([r for r in rows if r[9] == -1]))
    print()
    print("=== leftovers, from the EXCLUSIVE match ===")
    print("  reference units left without a free counterpart      : %d"
          % len([r for r in rows if r[3] == -1]))
    print("  test clusters never claimed by any reference unit    : %d"
          % (n_test - len(claimed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
