"""Worked example: read a synced/ folder and apply all three trial filters correctly.

Run this against a session that has been processed with a per-unit stability selection
in place, to see exactly what reaches the dataset and how an analysis consumes it.

The three filters, in the order they must be applied:

  1. trials.sync_valid                -- the Bpod/Trodes clock fit is trustworthy here
  2. ~trials.overlaps_recording_gap   -- neural data for this trial actually exists
  3. unit_trial_selection.parquet     -- this *unit* was present for this trial

The first two are properties of the recording and drop whole trials. The third is
per (unit, trial), so it is applied last, as a join -- never as a trial filter.

A unit that does not appear in unit_trial_selection.parquet keeps every trial, which
is what makes the join safe to add to an analysis written before the table existed.
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

print("=== session_meta.json -> unit_trial_selection ===")
print(json.dumps(meta["unit_trial_selection"], indent=1))

print("\n=== unit_trial_selection.parquet ===")
print("shape", sel.shape, "| dtypes:", dict(sel.dtypes.astype(str)))
print(sel.head(3).to_string(index=False))
print("\nper unit:")
print(sel.groupby("unit_id").selected.agg(kept="sum", total="size").to_string())

print("\n=== units.parquet -> has_stability_selection ===")
print(units.has_stability_selection.value_counts().to_string())
print(units.loc[units.has_stability_selection, ["unit_id", "has_stability_selection"]].to_string(index=False))

print("\n=== the actual filter, on spikes ===")
spikes = pd.read_parquet(S + "/spikes.parquet", columns=["unit_id", "trial_id", "spike_time_in_trial"])
good   = trials.loc[trials.sync_valid & ~trials.overlaps_recording_gap, "trial_id"]
spikes = spikes[spikes.trial_id.isin(good)]

excluded = sel.loc[~sel.selected, ["unit_id", "trial_id"]]
m = spikes.merge(excluded, on=["unit_id", "trial_id"], how="left", indicator=True)
clean = m[m._merge == "left_only"].drop(columns="_merge")
print("spikes before stability filter: %d" % len(spikes))
print("spikes after  stability filter: %d  (-%d)" % (len(clean), len(spikes) - len(clean)))

print("\n=== what it buys you, per restricted unit ===")
# NB: a crude per-trial-window rate only. Trial windows carry a 10 s trailing buffer
# and overlap, so these Hz are inflated in absolute terms -- the before/after RATIO is
# the meaningful part, not the value.
win = 10.0
for uid in sorted(sel.unit_id.unique()):
    a = spikes[spikes.unit_id == uid]
    b = clean[clean.unit_id == uid]
    na, nb = a.trial_id.nunique(), b.trial_id.nunique()
    print("unit %-4d  all trials: %3d trials, %7d spikes, %5.2f Hz   |   kept: %3d trials, %7d spikes, %5.2f Hz"
          % (uid, na, len(a), len(a)/(na*win), nb, len(b), len(b)/(nb*win)))
