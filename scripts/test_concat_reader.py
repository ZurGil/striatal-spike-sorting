"""Prove ConcatReader returns exactly what the two part files hold, seam included."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\Gil\spike_sorting_agent\scripts")
import build_merge_candidates as B

MAN = (r"Z:\Gil\Shamir_1\20260915_084812_103710_concat.rec"
       r"\20260915_084812_103710_concat.kilosort\concat_20260915_084812_103710.json")
NCH = 384
man = json.load(open(MAN, encoding="utf-8"))
r = B.ConcatReader(man, NCH)

p1, p2 = man["parts"]
m1 = np.memmap(p1["source"], dtype=np.int16, mode="r").reshape(-1, NCH)
m2 = np.memmap(p2["source"], dtype=np.int16, mode="r").reshape(-1, NCH)
SEAM = int(man["seam"]["sample_index"])
cols = np.array([0, 7, 100, 191, 383])
fail = 0


def check(name, got, want):
    global fail
    ok = got.shape == want.shape and np.array_equal(got, want)
    print("%-54s %s  shape %s" % (name, "ok" if ok else "FAIL", got.shape))
    if not ok:
        fail += 1


print("manifest: %d parts, total %d samples" % (len(man["parts"]), r.shape[0]))
print("  part 1 %s: file %d samples, used %d (trimmed %s)"
      % (p1["session"], m1.shape[0], p1["used_samples"],
         m1.shape[0] != p1["used_samples"]))
print("  part 2 %s: file %d samples, used %d"
      % (p2["session"], m2.shape[0], p2["used_samples"]))
print("  seam at %d" % SEAM)
print()

check("shape[0] equals the manifest total",
      np.array([r.shape[0]]), np.array([man["total"]["samples"]]))

# deep inside part 1
a = 50_000_000
check("window wholly inside part 1", r[a:a + 60, cols], m1[a:a + 60, cols])

# the first samples
check("window at sample 0", r[0:60, cols], m1[0:60, cols])

# the last 60 samples of part 1's USED range (not of its file)
a = SEAM - 60
check("last used window of part 1", r[a:SEAM, cols], m1[a:SEAM, cols])

# deep inside part 2
a = SEAM + 40_000_000
check("window wholly inside part 2", r[a:a + 60, cols],
      m2[a - SEAM:a - SEAM + 60, cols])

# the very first window of part 2
check("first window of part 2", r[SEAM:SEAM + 60, cols], m2[0:60, cols])

# straddling the seam: 20 from part 1, 40 from part 2
a = SEAM - 20
want = np.concatenate((m1[a:SEAM, cols], m2[0:40, cols]), axis=0)
check("window straddling the seam (20 + 40)", r[a:a + 60, cols], want)

# the last window of the whole concat
a = r.shape[0] - 60
check("last window of the joined session", r[a:a + 60, cols],
      m2[a - SEAM:a - SEAM + 60, cols])

# the trimmed tail of part 1 must be UNREACHABLE: a window just past the seam
# must come from part 2, never from part 1's discarded 112 s
got = r[SEAM:SEAM + 60, cols]
bad = m1[SEAM:SEAM + 60, cols]
same_as_trimmed = np.array_equal(got, bad)
print("%-54s %s" % ("the trimmed tail of part 1 is unreachable",
                    "FAIL -- it leaked in" if same_as_trimmed else "ok"))
if same_as_trimmed:
    fail += 1

# a real mean waveform through the reader, to be sure the call pattern works
rows = np.array([1_000_000, 50_000_000, SEAM - 100, SEAM + 1_000_000], dtype=np.int64)
wf, used = B.mean_waveforms(r, NCH, rows, cols, 10)
print("%-54s %s  used %d of %d, peak %.1f ADC"
      % ("mean_waveforms runs through the reader",
         "ok" if used == len(rows) else "FAIL", used, len(rows), float(np.abs(wf).max())))
if used != len(rows):
    fail += 1

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
