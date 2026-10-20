"""Worked example: read a synced/ folder and apply every filter correctly.

Run this against a processed session to see exactly what reaches the dataset from each
manual-curation pass and how an analysis consumes it.

    python read_synced_with_selection.py <synced_folder>

THE FILTERS, IN THE ORDER THEY MUST BE APPLIED

  1. trials.sync_valid                -- the Bpod/Trodes clock fit is trustworthy here
  2. ~trials.overlaps_recording_gap   -- neural data for this trial actually exists
  3. unit_trial_selection.parquet     -- this *unit* was present for this trial

The first two are properties of the recording and drop whole trials. The third is
per (unit, trial), so it is applied last, as a join -- never as a trial filter.

Every unit appears in unit_trial_selection.parquet with a true/false for every trial;
a unit nobody restricted has selected=True throughout. There is no absent-means-
everything rule to remember, and an inner join is as safe as a left join.

MERGES ARE NOT A FILTER

unit_merges.parquet needs NO change to an analysis: a merged unit is just a unit, with
an id from 10000 up -- it is in units.parquet and spikes.parquet by default, and a
`quality_label == "good"` filter picks it up like any other unit (provided somebody
scored it -- see quality_label_source below). It already has its row in units.parquet, its spikes in
spikes.parquet, and (if its parents had stability windows) its rows in
unit_trial_selection.parquet carrying the frozen union of them. You read
unit_merges.parquet only when you want the provenance -- which clusters went in, and
how lopsidedly. The section at the end of this script shows that.

Note that spike_clusters.npy in the Kilosort folder still contains the PARENTS: merges
are applied in memory while building synced/, never written back. See
sync_pipeline/HOW_TO_RUN.md section 6c.
"""
import json
import sys

import pandas as pd

DEFAULT = r"F:/Gil/Shamir/20260916_110311.rec/20260916_110311.kilosort/synced_20261015"
S = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
print("reading %s" % S)
print()

meta   = json.load(open(S + "/session_meta.json"))
trials = pd.read_parquet(S + "/trials.parquet")
units  = pd.read_parquet(S + "/units.parquet")
sel    = pd.read_parquet(S + "/unit_trial_selection.parquet")
try:
    merges = pd.read_parquet(S + "/unit_merges.parquet")
except (FileNotFoundError, OSError):
    # A synced/ folder built before merges existed has no such file -- and never gains
    # one retroactively. Re-run the pipeline for the session to get it.
    merges = pd.DataFrame(columns=["merged_unit_id", "parent_unit_id", "n_parents",
                                   "parent_n_spikes"])
    print("NOTE: no unit_merges.parquet here -- this output predates the merge "
          "feature. Re-run the pipeline for this session to add it.")

# ------------------------------------------------- what this dataset was built with
print("=== session_meta.json: the two manual-curation blocks ===")
print("unit_trial_selection:", json.dumps(meta["unit_trial_selection"], indent=1))
print("unit_merges        :", json.dumps(meta.get("unit_merges", {}), indent=1))

# ------------------------------------------------------------------ stability pass
print("\n=== unit_trial_selection.parquet ===")
print("shape", sel.shape, "| dtypes:", dict(sel.dtypes.astype(str)))
if len(sel):
    print(sel.head(3).to_string(index=False))
    print("\nper unit:")
    print(sel.groupby("unit_id").selected.agg(kept="sum", total="size").to_string())
else:
    print("(empty -- no unit was restricted; every unit keeps every trial)")

print("\n=== units.parquet: has_stability_selection ===")
print(units.has_stability_selection.value_counts().to_string())
print(units.loc[units.has_stability_selection,
                ["unit_id", "has_stability_selection"]].to_string(index=False))

# ------------------------------------------------------------------------- merges
print("\n=== unit_merges.parquet (provenance; not a filter) ===")
if len(merges):
    print(merges.to_string(index=False))
    print("\nmerged units in units.parquet:")
    print(units.loc[units.get("is_merged", False),
                    ["unit_id", "quality_label", "curated", "channel", "depth_mm",
                     "n_parents", "parent_unit_ids", "has_stability_selection"]]
          .to_string(index=False))
    print("(quality_label above is only a verdict where quality_label_source is "
          "'manual'.)")
    # Has anyone actually scored these merged units? `manual` means yes; the other
    # two mean the label was inherited from the parents, and `conservative_mua` in
    # particular is a placeholder that undersells a real merge. Score them with
    # scripts/write_unit_merges.py --score <merged_id>=good once you have looked.
    unscored = merges.loc[merges.quality_label_source != "manual", "merged_unit_id"]
    print("\nmerged units not scored by hand: %s"
          % (sorted(unscored.unique()) or "none — every merge has a real verdict"))

    # Share of each merged unit's spikes coming from each parent. A merge that is
    # ~98% one parent is a different animal from a 50/50 one, and worth a second look.
    share = merges.parent_n_spikes / merges.groupby(
        "merged_unit_id").parent_n_spikes.transform("sum")
    lopsided = merges.loc[share > 0.95, ["merged_unit_id", "parent_unit_id"]]
    print("\nmerges where one parent contributes >95%% of the spikes: %d"
          % len(lopsided))
else:
    print("(empty -- this session has no merges; every cluster stays separate)")

# ----------------------------------------------------------------- the actual filter
print("\n=== applying the filters to spikes ===")
spikes = pd.read_parquet(S + "/spikes.parquet",
                         columns=["unit_id", "trial_id", "spike_time_in_trial"])
good   = trials.loc[trials.sync_valid & ~trials.overlaps_recording_gap, "trial_id"]
spikes = spikes[spikes.trial_id.isin(good)]

excluded = sel.loc[~sel.selected, ["unit_id", "trial_id"]]
m = spikes.merge(excluded, on=["unit_id", "trial_id"], how="left", indicator=True)
clean = m[m._merge == "left_only"].drop(columns="_merge")
print("spikes before stability filter: %d" % len(spikes))
print("spikes after  stability filter: %d  (-%d)" % (len(clean), len(spikes) - len(clean)))

print("\n=== what the stability pass buys you, per restricted unit ===")
# NB: a crude per-trial-window rate only. Trial windows carry a 10 s trailing buffer
# and overlap, so these Hz are inflated in absolute terms -- the before/after RATIO is
# the meaningful part, not the value.
win = 10.0
for uid in sorted(sel.unit_id.unique()):
    a = spikes[spikes.unit_id == uid]
    b = clean[clean.unit_id == uid]
    na, nb = a.trial_id.nunique(), b.trial_id.nunique()
    tag = (" (merged)" if "is_merged" in units.columns
           and bool(units.loc[units.unit_id == uid, "is_merged"].any()) else "")
    print("unit %-6d%s all trials: %3d trials, %7d spikes, %5.2f Hz   |   "
          "kept: %3d trials, %7d spikes, %5.2f Hz"
          % (uid, tag, na, len(a), len(a) / (na * win), nb, len(b), len(b) / (nb * win)))
