The decisions for 20260915_103710 are the JOINT 2026-09-15 session's decisions.

verdicts.json and merge_verdicts.json are copied unchanged from
review_verdicts/20260915_084812_103710_concat/ -- there is one review and one unit
space for the day, and a unit is good or mua for the day, not per half.

trial_selections.json is NOT a copy. The windows were drawn on the joined session's
concat_elapsed_seconds axis and are converted here to this recording's own absolute
Trodes seconds, clipped to this half, with a window that crosses the seam split
between the halves. The conversion and the offsets it used are stamped in the file.
Written by scripts/split_concat_sort_for_sync.py.
