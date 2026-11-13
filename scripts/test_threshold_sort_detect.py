"""Prove troughs_on / features_of / cut_windows on synthetic data before touching Z."""
import sys

import numpy as np

sys.path.insert(0, r"D:\Gil\spike_sorting_agent\scripts")
import threshold_sort_validate as T

fail = 0


def check(name, got, want):
    global fail
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    ok = g.shape == w.shape and np.allclose(g, w, rtol=0, atol=1e-6)
    print("%-46s %s   got %s want %s" % (name, "ok" if ok else "FAIL", got, want))
    if not ok:
        fail += 1


# --- troughs_on ------------------------------------------------------------------
x = np.zeros(200, dtype=np.float32)
check("empty: nothing below threshold", T.troughs_on(x, -10.0), [])

x = np.zeros(200, dtype=np.float32); x[50] = -20
check("one spike", T.troughs_on(x, -10.0), [50])

x = np.zeros(200, dtype=np.float32)
x[50] = -20
x[50 + T.DEAD_SAMP] = -15          # exactly at the dead time: both kept
check("two spikes exactly DEAD_SAMP apart", T.troughs_on(x, -10.0), [50, 50 + T.DEAD_SAMP])

x = np.zeros(200, dtype=np.float32)
x[50] = -20
x[55] = -30                         # inside dead time and deeper: the deeper one wins
check("deeper event inside the dead time replaces", T.troughs_on(x, -10.0), [55])

x = np.zeros(200, dtype=np.float32)
x[50] = -30
x[55] = -20                         # inside dead time and shallower: dropped
check("shallower event inside the dead time dropped", T.troughs_on(x, -10.0), [50])

x = np.zeros(200, dtype=np.float32)
x[40:45] = [-12, -18, -25, -14, -11]   # one run, trough in the middle
check("trough of a run, not its first sample", T.troughs_on(x, -10.0), [42])

x = np.zeros(200, dtype=np.float32)
x[0:3] = [-25, -14, -11]               # a run that starts at sample 0
check("run starting at sample 0", T.troughs_on(x, -10.0), [0])

x = np.zeros(200, dtype=np.float32)
x[-3:] = [-11, -14, -25]               # a run that never ends
check("run running off the end", T.troughs_on(x, -10.0), [199])

x = np.zeros(400, dtype=np.float32)
want = []
for i, s in enumerate(range(50, 350, 40)):
    x[s] = -20 - i
    want.append(s)
check("eight well-separated spikes", T.troughs_on(x, -10.0), want)

# --- features_of -----------------------------------------------------------------
W = np.zeros((1, 60), dtype=np.float32)
W[0, 20] = -100.0
W[0, 29] = 40.0                       # peak 9 samples (0.3 ms) after the trough
tr, p2, en, wd = T.features_of(W)
check("trough", np.round(tr, 3), [100.0])
check("peak-to-trough", np.round(p2, 3), [140.0])
check("energy = sum of squares", np.round(en, 1), [100.0 ** 2 + 40.0 ** 2])
check("width trough->peak in ms", np.round(wd, 4), [round(9 / 30000.0 * 1000, 4)])

# --- cut_windows -----------------------------------------------------------------
f = np.zeros(1000, dtype=np.float32)
f[500] = -80.0
f[:5] = 7.0                            # a baseline offset that must be removed
W, ok = T.cut_windows(f, 0, np.array([500]))
check("window length", [W.shape[1]], [T.WF_PRE + T.WF_POST])
check("trough sits at WF_PRE", [int(W[0].argmin())], [T.WF_PRE])
f2 = np.full(1000, 25.0, dtype=np.float32)
f2[500] = -80.0
W2, _ = T.cut_windows(f2, 0, np.array([500]))
check("baseline removed (flat 25 uV offset)", [float(np.round(W2[0, 0], 3))], [0.0])
check("amplitude survives baseline removal", [float(np.round(-W2[0].min(), 1))], [105.0])
W3, ok3 = T.cut_windows(f, 0, np.array([2, 500, 995]))
check("events too close to an edge are dropped", ok3, [False, True, False])

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
