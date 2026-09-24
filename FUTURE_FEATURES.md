# Future features / roadmap (not yet started)

Deferred ideas, parked pending evaluation of the current tool's day-to-day
usefulness across more sessions. Not scheduled — revisit when there's an
actual reason to prioritize one of these, not on a timer. When one gets
started, move its notes into `WORKFLOW.md` as it becomes real process, and
remove it from here.

---

## 1. In-tool merge (and later, split) decisions -- avoid needing to open Phy

**Status**: discussed 2026-09-15, not started. Revisit after the current
per-session review tool (labels, ACG, waveform, footprint -- no merge UI) has
been used across a few more sessions and its usefulness is actually assessed.

**The idea**: let the review tool itself suggest merge candidates and record
merge decisions, the way Phy's similarity view does, so Phy doesn't need to be
opened separately just for merging.

**Key constraint (already worked out)**: a web page can't write to
`spike_clusters.npy` directly. So "merge in the tool" can only ever mean
*recording* the decision there -- applying it (rewriting the clustering,
rebuilding `cluster_info.tsv`, resyncing) still has to be a separate script
step, run after review. This is the same shape as Stage 2's Phy-merge handling
already built and proven (2026-09-15, `20260901_085606`) -- that pipeline
would mostly just need to read "pending merge decisions from our db" instead
of "Phy's merge log," see WORKFLOW.md stage 2.

**What's already available to build on**:
- Candidate pairs: Kilosort's own `similar_templates.npy` +
  `striatal_agent.merge_split.candidate_pairs_from_similarity()` (already used
  by the agent pipeline's merge-scoring step)
- Per-unit visuals (waveform, ACG, footprint) already computed and rendered
- A working "apply merges -> rebuild cluster_info.tsv -> resync" pipeline
  (Stage 2, proven on `20260901_085606`)

**What's genuinely new**:
- Cross-correlogram between two candidate units (the actual "are these the
  same neuron" signal -- similar computation to the existing per-unit ACG,
  done pairwise instead of self-only)
- A two-unit side-by-side comparison UI + a "propose merge" action, wired into
  the shared database
- The "apply pending decisions" script (mostly reusing Stage 2 logic)

**Open design question, explicitly flagged by the user -- do not just copy
Phy's UI wholesale**: there are things about how *we* want to review/decide
merges that Phy doesn't do, and Phy's own similarity-ranking (from
`similar_templates.npy`) is only one input, not necessarily the final design.
Before implementing, have an actual design conversation about:
- What Phy is missing that matters for this workflow
- How to combine Kilosort's own template-similarity ranking with whatever
  else we want (e.g. the structural-score/ACG/footprint metrics the agent
  pipeline already computes per unit) rather than just reproducing Phy's
  candidate list as-is
- Splits are explicitly out of scope for a first pass -- a real new
  sub-clustering decision within one unit's spikes, harder than picking
  between two existing candidates. Separate design conversation later.

---

## 2. Reusable multi-session review tool (one URL, session picker)

**Status**: discussed 2026-09-15, not started. Same "revisit after evaluating
current usefulness" gating as item 1 -- explicitly deferred by the user
pending seeing how much the per-session-rebuild approach actually costs in
practice across more sessions.

**The idea**: one permanently-published page that fetches whichever session's
data you select, instead of building and publishing a brand-new HTML file
(and getting a new URL) per session.

**Why it's not urgent**: measured 2026-09-15 -- the actual review-tool-specific
build cost (bombcell + classification + review-data json + html render +
publish, now that the 4-script chain in WORKFLOW.md 1f exists and is
debugged) is only ~10-15 minutes per session. The real time cost (Stage 1's
agent pipeline, ~90 min for a 466-unit session) is identical either way, so
the reusable version mainly buys link/maintenance convenience, not speed.

**What it would need** (see conversation 2026-09-15 for the full tradeoff
discussion): the `assets` capability to store each session's data file, a
session-picker UI with async fetch-on-select, and per-session-namespaced
verdict storage in the shared db (so sessions' verdicts don't collide). Real
but contained UI work; some risk of a new bug in what's currently a simple,
proven, working page.

---

## 3. Checkpointed Kilosort4 runner (survive a mid-run kill)

**Status**: discussed 2026-09-16, not started. Triggered by a real incident:
running `independent_channel_clustering_session.py` concurrently with
Kilosort4 on session `20260916_110311` caused a genuine system-level
out-of-memory kill (see WORKFLOW.md 2026-09-16 INCIDENT entry for the root
cause -- `RawReader`'s row-major single-channel reads touch the whole file at
the OS page-cache level) that wiped out ~2.5 hours of Kilosort4 progress,
since it had to restart the entire spike-detection stage from scratch.

**The immediate trigger is already fixed** (don't run other heavy I/O jobs
concurrently with Kilosort4) -- this item is about the underlying fragility
itself: ANY kill during a multi-hour run (OOM, power blip, accidental
process kill, machine reboot for an update) currently costs the *entire*
run, however far it got.

**What's already true, worth building on**: Kilosort4's top-level
`run_kilosort()` (in `kilosort/run_kilosort.py`, confirmed by reading the
installed package source, `D:\conda_envs\kilosort\Lib\site-packages\kilosort\run_kilosort.py`)
is internally modular -- `_sort()` calls separate functions in sequence:
`compute_preprocessing()` -> `compute_drift_correction()` (the long stage
that keeps getting killed) -> `detect_spikes()` -> `cluster_spikes()` -- each
producing real intermediate results in memory. Only the very last step calls
`io.save_ops(ops, results_dir)` -- nothing is persisted between stages.

**The idea**: a custom runner script that calls those internal stage
functions directly (instead of the `run_kilosort()` wrapper), and manually
saves/pickles each stage's output (at minimum: post-drift-correction state)
to `results_dir` before starting the next stage. On a subsequent run, check
for an existing checkpoint file and skip straight to the next stage instead
of recomputing from scratch.

**Real trade-off, be upfront about it before building**: this means calling
Kilosort4's internal/private functions instead of its public API --
undocumented, not guaranteed stable across a Kilosort4 version update, and a
non-trivial amount of engineering to correctly replicate `_sort()`'s own
setup/state handling (ops initialization, binary file object, GPU device
setup) around each manually-called stage. Worth it only if multi-hour
Kilosort4 runs on this machine turn out to be a recurring pain point, not a
one-off.
