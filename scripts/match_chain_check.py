# -*- coding: utf-8 -*-
"""Do three sessions agree with themselves about who is who?

    python scripts/match_chain_check.py 20260916_110311 20260917_100043 20260918_100248

The split-half control measures recall and precision against pairs that are the same
neuron by construction, but only within one session. Across sessions there is no ground
truth, so the next best thing is a consistency check the matcher cannot game: match
A->B and B->C, follow the chain, and compare it against the direct A->C match computed
independently. Every disagreement is at least one error in one of the three matches.
Agreement does not prove identity, but a procedure with a high error rate cannot produce
it, so the number is a real upper bound on how often all three agree by luck.

Expects outputs/unit_matching/<a>__<b>_matches.csv for all three pairs, i.e. run
match_units_across_sessions.py on A->B, B->C and A->C first.
"""
from __future__ import print_function

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "unit_matching")


def pairs(a, b):
    p = os.path.join(OUT, "%s__%s_matches.csv" % (a, b))
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run the matcher on that pair first" % p)
    m = pd.read_csv(p)
    return dict(zip(m.unit_a, m.unit_b)), m


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    a, b, c = sys.argv[1:]
    ab, mab = pairs(a, b)
    bc, mbc = pairs(b, c)
    ac, mac = pairs(a, c)
    print("%s->%s %d matches, %s->%s %d, %s->%s %d"
          % (a, b, len(mab), b, c, len(mbc), a, c, len(mac)))

    chain = dict((u, bc[v]) for u, v in ab.items() if v in bc)
    both = [u for u in chain if u in ac]
    agree = [u for u in both if chain[u] == ac[u]]
    print("chain %s->%s->%s survives for %d units; the direct %s->%s match also found "
          "%d of them" % (a, b, c, len(chain), a, c, len(both)))
    print("the two routes name the same %s unit for %d of %d (%.1f%%)"
          % (c, len(agree), len(both), 100.0 * len(agree) / max(1, len(both))))
    bad = [(u, ab[u], chain[u], ac[u]) for u in both if chain[u] != ac[u]]
    if bad:
        print("\ndisagreements (unit in %s, via %s, chain says, direct says):" % (a, b))
        for row in bad[:25]:
            print("  %6d  %6d  %6d  %6d" % row)
    out = os.path.join(OUT, "chain_%s_%s_%s.csv" % (a, b, c))
    pd.DataFrame([dict(unit_a=u, via_b=ab[u], chain_c=chain[u], direct_c=ac[u],
                       agrees=int(chain[u] == ac[u])) for u in both]).to_csv(out,
                                                                            index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
