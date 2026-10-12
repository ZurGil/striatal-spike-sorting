"""Does the wavelet alignment recover the KNOWN sub-sample shifts?

The multi-axis feature build reported 61.6% of its sub-sample shifts
saturating at the clamp, which means the estimate is wrong rather than merely
noisy, and every shape feature downstream is then de-jittered incorrectly.
Guessing at the cause twice already failed, so this measures it against
ground truth instead.

The benchmark builder drew a sub-sample shift for every injected spike from
U(-0.5, 0.5) and stored it in the truth file as `shifts`. So for the injected
spikes the correct answer is known per spike, which makes this a direct
accuracy test of the alignment rather than an indirect one:

  * is the estimate correlated with the truth at all?
  * what is the residual error, against the 0.289 sample SD that guessing
    zero for every spike would give (the SD of U(-0.5, 0.5))?

A method that cannot beat "always answer zero" is not an alignment method.
That baseline is the thing to beat and it is not trivial: the shifts are
small by construction.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sorter_locden import FS, data_dir, cache_dir, probe

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOL = 3          # tight: we want the SAME spike, not a nearby one
RAD_UM = 40.0

ev = np.load(os.path.join(cache_dir(REP), "events2.npz"))
ft = np.load(os.path.join(cache_dir(REP), "feat.npz"))
names = [str(x) for x in ft["names"]]
F = ft["F"]
et, ech = ev["t"], ev["ch"].astype(np.int64)
cm, X, Y = probe(REP)

T = np.load(os.path.join(data_dir(REP), "hybrid_truth.npz"), allow_pickle=True)
tt, lab, sh = T["times"].ravel(), T["labels"].ravel(), T["shifts"].ravel()
tier_code, tier_names = T["tier"].ravel(), [str(x) for x in T["tier_names"]]
su = [int(x) for x in T["source_units"]]
dstc = {u: int(c) for u, c in zip(su, T["dst_peak_channels"])}

i_shift = names.index("subsample_shift")
i_f0 = names.index("f0_hz")
i_ptp = names.index("ptp_uv")

rows = []
for a, u, s in zip(tt, lab, sh):
    c0 = dstc[int(u)]
    p = np.searchsorted(et, a - TOL)
    q = np.searchsorted(et, a + TOL, "right")
    if q <= p:
        continue
    cand = np.arange(p, q)
    d = np.sqrt((X[ech[cand]] - X[c0]) ** 2 + (Y[ech[cand]] - Y[c0]) ** 2)
    cand = cand[d <= RAD_UM]
    if not len(cand):
        continue
    k = cand[np.argmax(F[cand, i_ptp])]      # the biggest one is the injected
    rows.append((int(u), tier_names[int(tier_code[lab == u][0])], float(s),
                 float(F[k, i_shift]), float(F[k, i_f0]), float(F[k, i_ptp]),
                 int(et[k] - a)))

import pandas as pd
df = pd.DataFrame(rows, columns=["unit", "tier", "true_shift", "est_shift",
                                 "f0_hz", "ptp_uv", "int_offset"])
df.to_csv(os.path.join(r"D:\Gil\spike_sorting_agent\outputs",
                       "wavelet_alignment_truth.csv"), index=False)

print("=" * 80)
print("WAVELET SUB-SAMPLE ALIGNMENT vs KNOWN INJECTED SHIFTS (rep%d)" % REP)
print("=" * 80)
print("matched %d injected spikes within %d samples and %.0f um"
      % (len(df), TOL, RAD_UM))
print()
base = df.true_shift.std()
print("  truth shifts: sd %.4f samples (U(-0.5,0.5) has sd 0.2887)" % base)
print("  BASELINE, answer 0 for every spike:  RMSE %.4f"
      % np.sqrt((df.true_shift ** 2).mean()))
print("  wavelet estimate:                    RMSE %.4f"
      % np.sqrt(((df.est_shift - df.true_shift) ** 2).mean()))
r = np.corrcoef(df.true_shift, df.est_shift)[0, 1]
print("  correlation(estimate, truth): %+.4f" % r)
print("  estimate spread: sd %.3f, |est| median %.3f, %.1f%% at |1.5|"
      % (df.est_shift.std(), df.est_shift.abs().median(),
         100.0 * (df.est_shift.abs() >= 1.4999).mean()))

# the integer part should be ~0 because the detector refines to the trough
print("\n  integer offset of the matched event from the injection time:")
print("    median %+.1f, 5-95%% [%+.0f, %+.0f]"
      % (df.int_offset.median(), df.int_offset.quantile(.05),
         df.int_offset.quantile(.95)))

print("\n  by tier:")
print("%-11s %7s %10s %10s %10s" % ("tier", "n", "corr", "RMSE", "baseline"))
for t, g in df.groupby("tier"):
    rr = (np.corrcoef(g.true_shift, g.est_shift)[0, 1] if len(g) > 5
          else np.nan)
    print("%-11s %7d %10.4f %10.4f %10.4f"
          % (t, len(g), rr, np.sqrt(((g.est_shift - g.true_shift) ** 2).mean()),
             np.sqrt((g.true_shift ** 2).mean())))

print("\n  loudest spikes only (ptp above the median), where the phase is")
print("  least noisy and the method should look its best:")
hi = df[df.ptp_uv >= df.ptp_uv.median()]
print("    n %d, corr %+.4f, RMSE %.4f vs baseline %.4f"
      % (len(hi), np.corrcoef(hi.true_shift, hi.est_shift)[0, 1],
         np.sqrt(((hi.est_shift - hi.true_shift) ** 2).mean()),
         np.sqrt((hi.true_shift ** 2).mean())))
