# Spike-sorting agent workflow (living document)

**Read this before running the pipeline, before re-syncing a session after Phy, or
before answering "where's the review tool / final data for session X."** This file
records *how things are actually done*, including workarounds for real problems
hit in practice. Keep it in sync with reality: whenever we make a permanent
decision (a new default, a fixed bug, a new convention), edit this file in the
same sitting. Don't let decisions live only in chat history — a future agent
starting cold should be able to reproduce everything from this file alone.

Companion file: `outputs/session_registry.json` — the per-session index (paths,
artifact URLs, curation status). Update it every time a session moves to a new
stage.

---

## Two-stage process, in order

```
Kilosort4 output
      |
      v
[STAGE 1] agent pipeline (run_agent.py)  --writes-->  agent_processed/ (cluster_*.tsv, agent_report.json)
      |
      v
corrected_classification.py  --writes-->  corrected_classification.csv / _with_waveforms.json
      |
      v
review-tool build script  --writes-->  pipeline_review_data.json  -->  published HTML Artifact
      |
      v
   >>> HUMAN reviews in the review tool, saves verdicts (current_verdicts.json / artifact DB) <<<
      |
      |--- stability windows + merges (NO Phy needed, section 1h) ----------------+
      |      scripts/write_unit_trial_selection.py --> unit_trial_selection.json  |
      |      scripts/write_unit_merges.py          --> unit_merges.json           |
      |      both land in <session>.kilosort/kilosort4/, read by convention       |
      |                                                                           |
      v                                                                           |
   >>> HUMAN opens Phy, labels units, and may MERGE or SPLIT clusters <<<         |
      |      (only needed for splits and label changes now -- see 1h)             |
      v                                                                           |
[STAGE 2] rebuild cluster_info.tsv + final verdict mapping  -->  run sync_pipeline <+
      |
      v
synced/{spikes,trials,units,state_events,poke_events}.parquet   <-- THE analysis dataset
       + unit_trial_selection.parquet + unit_merges.parquet     <-- manual curation
```

**Stage 2 only ever happens after Stage 1 and after human Phy review.** Never run
the sync pipeline for analysis purposes against pre-Phy data if a Phy pass has
since happened — the whole point of Stage 2 is to fold in Phy's merges/splits and
final labels, which change actual spike-to-cluster assignments, not just display
labels.

If this ordering ever changes (e.g., we decide to sync before Phy, or skip the
review-tool step for some sessions), **update this section first**, since
everything below assumes this order.

---

## STAGE 1 — Kilosort4 output -> agent pipeline -> review tool

### 1a. Prerequisites

- Standard Kilosort4 output at `<session_id>.rec\<session_id>.kilosort\kilosort4\`
  (needs `spike_times.npy`, `spike_clusters.npy` (== `spike_templates.npy` pre-curation),
  `templates.npy`, `amplitudes.npy`, `ops.npy`, `params.py`, `similar_templates.npy`, etc. —
  whatever Kilosort4 itself writes).
- Channel map at `<session_id>.kilosort\<session_id>_custom_neuropixels_map.json`
  (sibling of the `kilosort4/` folder, **not** inside it). If this is missing, the
  pipeline can't auto-discover the session — check for it before running.
- Raw `.probe1.dat` file readable (the pipeline memory-maps it via `RawReader`).

### 1b. Run it

```
D:\conda_envs\kilosort\python.exe D:\Gil\spike_sorting_agent\run_agent.py "<drive>:\path\to\<session_id>.rec"
```

Only this one conda env (`D:\conda_envs\kilosort`) has the right scipy + pandas +
statsmodels + pyarrow combination for everything in this project — use it for
every script in this whole workflow, not just this step.

Default flags: no `--apply-tier1-merges` (non-destructive: merges are scored and
logged, never auto-applied to `spike_clusters.npy` by the agent itself — only Phy,
by a human, actually performs merges). `--screen-only` skips the recovery
sweep/collision/merge steps if you only want the structural/footprint/ACG screen.

### 1c. What it computes, per unit

- **Structural score** (0-1, weighted average): footprint shape, waveform shape,
  CCG cross-check, refractory-violation ratio, footprint concentration.
  `cross_unit_acg` is in the weight dict but **disabled by default as of
  2026-09-15** (`AgentConfig.skip_cross_unit_acg = False` — we currently *don't*
  flip it True in `run_agent.py`'s default path, but pass `cfg=AgentConfig(skip_cross_unit_acg=True)`
  explicitly when scripting a run; see "Known issues" below for why).
- **`likely_shared_artifact`** hard flag: footprint concentration `<0.2` OR
  footprint flatness `>=0.75` OR waveform width `>=0.55ms` — each independently
  rare (~5-10% of units), calibrated against the 401-unit ground truth for session
  `20260901_085606`. Caps `structural_score` at 0.15 when tripped. **Does NOT use
  cross-unit ACG** — see below.
- **Firing-pattern profile**: bursts, pauses, cell-class guess (e.g. `tan_like`).
- **Recovery sweep**: scans raw data (bounded to a capped number of burst-windows
  + a fixed number of random windows per unit — NOT unbounded) for below-threshold
  candidate spikes matching the unit's own template, and scores whether adding
  them helps or hurts the unit's ACG.
- **Collision/contamination classification** against spatial neighbors (found via
  the *full* session's `templates`/`peak_channels`, not just the units in the
  current invocation — this is why batching this step is safe, see below).
- **Decision tier**: `tier1_auto_applied` (recovery only, reversible child
  cluster) / `tier2_scored_logged` (scored, not applied) / `tier3_insufficient_evidence`.

Session-wide, after the per-unit loop: **merge-candidate scoring** — pairs from
Kilosort's own `similar_templates.npy`, filtered to spatially close pairs
(`collision_neighbor_radius_um = 100µm` by default), each scored and logged to
`merge_records` (never auto-applied). This is purely advisory — it does not
change any unit's own `structural_score` or label.

### 1d. `cross_unit_acg` — known to be unreliable, disabled

Documented directly in `striatal_agent/config.py` (read the comment there, it's
the primary source): calibrated against a null sample of unrelated unit pairs and
found to flag 58-72% of units **by chance alone** (a multiple-comparisons
artifact of searching each unit's single best match among 80-400 candidates).
**It is not used by the hard `likely_shared_artifact` flag at all** — only as a
mild, capped soft penalty (weight 0.15) in the overall structural score, and as
incidental context in the pipeline's log line.

Decision (2026-09-15): pass `AgentConfig(skip_cross_unit_acg=True)` for any run
where you want either (a) faster runs (skips an O(n^2) all-pairs ACG comparison),
or (b) safe batching (see below — it's the only per-unit computation that
otherwise depends on which *other* units are in the same invocation). If you ever
want it back, it degrades gracefully — `cross_unit_acg_info=None` is already a
handled case, not a crash.

### 1e. KNOWN ISSUE — large sessions crash around unit 20-24 regardless of free RAM

Confirmed empirically on session `20260911_100049` (466 units) on 2026-09-15:
running the full pipeline in one process gets killed consistently around unit
20-24, **independent of total requested unit count and independent of free
system memory** (reproduced with 2GB free and with 58GB free — same crash
point). Code review of the per-unit hot path (`recovery.py`'s
`compute_candidate_scores`) found no obvious bug (no unclosed figures, no
unbounded caches, raw-data reads are capped via `max_bursts_per_unit_io_cap` +
a fixed random-window count) — best working explanation is memory-allocator
fragmentation from many large temporary numpy arrays across units, possibly
compounded by a resource ceiling specific to how background processes are
launched in this environment (not confirmed either way — see the open question
below).

**Workaround that works**: process in small batches (~10 units), each a **fresh
Python process** (so cache/fragmentation resets), rather than one long-running
invocation.

- Per-unit results (`structural_score`, `tier`, label) are **100% identical**
  regardless of batch membership *as long as `skip_cross_unit_acg=True`* — the
  collision/neighbor lookup already uses the full session's spike data, not just
  the current batch, so nothing about a unit's own score depends on which other
  units share its batch.
- The only batch-sensitive thing is the separate `merge_records` list (a pair
  split across batches never gets scored). This is advisory-only and doesn't
  affect any unit's label — acceptable to leave incomplete, or do one lightweight
  consolidation pass afterward using each batch's saved `structural_score` values
  (no need to redo the expensive raw-data scan) if you want it complete.
- **Do NOT try to fix this with a spatial-overlap batching scheme** — tried and
  abandoned same day: every unit in a batch gets *fully* heavy-processed
  regardless of whether it's "core" or "margin," so padding batches with a
  spatial radius to catch merge-candidate pairs just inflates batch size back up
  past the crash threshold (observed: 8-unit "core" batches ballooned to
  50-145-unit "extended" batches once given a 400µm depth margin, given typical
  probe unit density). Plain small sequential batches (arbitrary order, no
  spatial logic) is simpler and correct once you accept the merge-list caveat
  above.

Scripts used for this (scratchpad, not in the repo — recreate from this
description if needed): `plan_batches_v2.py` (simple sequential split by
`BATCH_SIZE`), `run_batch.py` (imports `run_pipeline` directly, runs one batch,
copies `agent_report.json` + `cluster_*.tsv` to a batch-specific output folder
before the next batch overwrites them), `run_all_batches.sh` (loops batches
sequentially, logs start/done per batch to a status file for polling).

**Open question, not yet resolved**: is the crash a property of this specific
environment (a resource ceiling on background-task processes) or something that
would also happen running the same code directly in a user terminal outside this
tool? Untested. If you get a chance to test this cleanly (run a 40+ unit batch
directly in a terminal, not through an agent's background-task mechanism) and it
*doesn't* crash at ~20-24 units, that confirms the ceiling is environment-specific
— update this section with the answer.

### 1f-pre. The review tool needs MORE than corrected_classification.py -- map of all inputs

**Found 2026-09-15, while building the review tool for a second session for the first
time**: `pipeline_review_data.json` (what the published review tool actually reads) has
far more fields than `corrected_classification_with_waveforms.json` alone provides.
Table of every field the review tool actually uses, and where it comes from. **This
table is now COMPLETE and CURRENT as of 2026-09-15 (second pass, same day)** — every
row was resolved (built, confirmed reusable, or confirmed not needed at all). All
session-parameterized scripts below live in `scripts/*_session.py`; the originals they
were adapted from (hardcoded to `20260901_085606`) are left untouched as reference.

| Field(s) in `pipeline_review_data.json` | Source script | Status |
|---|---|---|
| `label`, `mua_reasons`, `noise_reasons`, `violation_ratio`, `footprint_concentration_ratio`, `footprint_flatness_ratio`, `peak_trough_width_ms`, `rawAmplitude`, `signalToNoiseRatio`, `mainPeakToTroughRatio`, `mean_rate_hz`, `n_spikes`, `peak_channel` | `scripts/corrected_classification_session.py` (needs bombcell qMetrics -- see 1f step 1-2) | DONE -- reusable |
| `waveform_top4` | `corrected_classification_session.py`'s own `ks_template` field, renamed in `build_pipeline_review_data.py` | DONE -- free, no separate computation |
| `acg_hist` + its bin/lag/refractory/shoulder params | `build_pipeline_review_data.py` via `striatal_agent.acg_tools.compute_acg_hist()` (a proper package function, already used by the Stage 1 pipeline itself) | DONE -- cheap, spike-times only |
| `trodes_channel` (+ everything downstream that needs a trodes id) | `scripts/build_trodes_channel_lookup_session.py` | DONE -- reusable. **Note**: not every session has a standalone `<session_id>.trodesconf` file (e.g. `20260911_100049` didn't) -- this script falls back to extracting the same `SpikeNTrode`/`SpikeChannel` XML config directly from the raw `.rec` file's own embedded header (same trick as verifying `gain_to_uV`, see step 1) when no standalone file exists. |
| `cluster_points` (2D scatter panel) | `scripts/raw_cluster_points_session.py` | DONE -- reusable. ~500-600s per 400-ish units (scales roughly linearly; real timing logged per-session in the progress file) |
| `sep_vs_noise`, `sep_vs_other`, `sep_vs_noise_vector`, `sep_vs_other_vector`, `n_valid_noise`, `n_valid_other`, `co_located_units` | `scripts/raw_separation_metric_session.py` (needs the trodes lookup above) | DONE -- reusable. ~1000-1300s per 400-ish units, the slowest of the per-session steps (10 chunks vs. 5) |
| `burst_curve`, `burst_shape`, `isi_amp_corr_all`, `isi_amp_corr_burst` | `scripts/build_burst_isi_metrics_session.py` | DONE 2026-09-15, but **this is a FRESH RECONSTRUCTION, not recovered original code** -- the original script that produced these fields for `20260901_085606` was never found. The shape-classification thresholds (decay/facilitation/dip_then_rise/rise_then_dip/flat) are a new, reasonable design documented in that script's own docstring, not a verified match to whatever exact logic produced the old session's values for those 5 shape categories. Uses `striatal_agent.firing_pattern.detect_bursts()` (the pipeline's own existing burst definition, not a second one) + `kilosort_original/amplitudes.npy`. Cheap -- spike-times/amplitudes only, no raw-data reads. |
| `isi_amp_scatter` | **NOT COMPUTED, BY DESIGN -- do not add this back.** Confirmed 2026-09-15 by checking the review tool's own rendering code: this field is present in the *old* session's `pipeline_review_data.json` but nothing in `build_review_html.py`'s JS ever reads or renders it (only the two scalar correlations, `isi_amp_corr_all`/`isi_amp_corr_burst`, are actually displayed). It was dead weight even in the original session's data. Skip it entirely in any future session -- there is no panel that needs it, and no script should be written to produce it. |
| Raw spike-trace examples (the "raw spikes" toggle panel) | `scripts/compute_raw_trace_snippets_session.py`, output pushed into the artifact's own `raw_traces` database collection (**not** baked into the HTML like everything else above -- see below) | DONE -- reusable, cheap (~100-110s per session, small windowed reads only) |
| Firing-rate-across-the-session curve + trial-start ticks + the per-unit kept-time-range selection (the "Firing rate across the session" panel) | `scripts/build_session_timeline_data.py` -> `outputs/session_timeline_<session_id>.json`, embedded by `build_review_html.py`. **Not** part of `pipeline_review_data.json`: the bin grid and the trial markers are session-level, and the rate arrays ship as base64 uint16 rather than JSON numbers. | DONE 2026-10-15. Cheap -- 5.5s for 472 units / 25.2M spikes / a 3.5h session. **Trial ticks come from the raw DIO stream, not from sync**: the TTL code is the Bpod state number, so code 1 = each trial's first state = the trial start, and it needs no Bpod file and no sync fit (which is the point -- the review tool runs before sync exists). Verified exact against this session's own fully-synced `trials.parquet`: 954 code-1 markers vs 953 Bpod trials, ordered difference to `trial_start_trodes` 0.0 ms median **and** 0.0 ms max. Code 0 looks tempting (also once per trial) but sits a variable 0.4-4s earlier -- it is the inter-trial reset, don't use it. |
| `cell_type`, `postSpikeSuppression_ms`, `propLongISI` | `outputs/striatal_celltypes.csv` (from `scripts/run_striatal_celltype.py`) + `scripts/build_full_unit_browser.py`'s merge pattern | **Still not built for a second session** -- `build_full_unit_browser.py` is from an early iteration (predates most of the fields above) and isn't a drop-in script; treat as a reference for the merge pattern only, not yet re-verified as needed by the current tool's rendering code the way `isi_amp_scatter` was checked and ruled out. If a future session needs it, check the JS rendering first (same method that ruled out `isi_amp_scatter`) before spending time computing it. |

**Raw spike traces are architecturally different from every other field above** --
they're fetched live by the browser (`db.collection('raw_traces').doc(String(uid)).get()`),
not embedded in the HTML at build time. After running
`compute_raw_trace_snippets_session.py`, its output JSON must be split into one small
file per unit and pushed via the Artifact tool's `write_db` (`db_op: "batch"`,
`collection: "raw_traces"`, `doc_id` = unit_id as a string, one write per unit,
`file_path` per entry rather than inline `data` to keep each tool call small). **Keep
each batch's total payload under ~900KB** (the actual limit is 1,048,576 bytes).
**The safe batch size depends on N_SPIKES and must be recomputed whenever that
changes**: at N_SPIKES=8 (original), ~46KB average/51KB max per unit -> 15/batch was
safe; at N_SPIKES=16 (current, see 2026-09-16 entry below), ~93KB average/101KB max
per unit -> dropped to 8/batch. Always check actual on-disk file sizes
(`os.path.getsize` over the split directory) before picking a batch size rather than
assuming the old number still holds.

**Lesson (why this file exists):** several of the fields above were originally computed
as one-off scratchpad/inline code during a conversation and never promoted to
`scripts/`, which is exactly why they had to be reconstructed (or, for `isi_amp_scatter`,
confirmed safe to drop) the second time a session needed them. **Going forward: any
script that computes a field the published review tool actually displays must be saved
under `scripts/`, not left in a session-temp scratchpad directory that gets cleaned
up.** If you write one ad-hoc during a conversation and it works, copy it into
`scripts/` in the same sitting and update the table above — don't wait until the next
session needs it to find out it's gone. Conversely, if you ever discover another field
in `pipeline_review_data.json` that the review tool's own JS doesn't actually render
(the way `isi_amp_scatter` turned out to be dead), remove it from this table's "needed"
list and say so explicitly, the same way — a future pass should never waste time
recomputing something nothing displays.

### 1f. Building the review tool from agent output

**Prerequisite, easy to forget (see 2026-09-17 change-log entry — it was
forgotten once already): the actual `striatal_agent.run_pipeline()` agent
run must already be done first** — batched via `run_pipeline_batch_session.py`
(one batch per fresh process, BATCH_SIZE=10, `skip_cross_unit_acg=True`, see
1e below for why batching is required at all) then
`consolidate_batches_session.py`. This is what builds `kilosort_original/`
and `agent_processed/` — bombcell (step 1 below) reads directly from
`kilosort_original/` and fails with a plain `FileNotFoundError` if it
doesn't exist yet. Do not skip straight from "Kilosort4 finished" to step 1.

**Proven, working 4-script chain (2026-09-15, second session)** — run in this
order, all under `scripts/`, all take `<session_id>` (and most `<rec_root>`,
e.g. `F:\Gil\Shamir\20260911_100049.rec`) as CLI args:

1. **`run_bombcell_session.py <session_id> <rec_root> <gain_to_uV>`** — needs
   the **`D:\conda_envs\bombcell`** conda env specifically (not `kilosort` —
   bombcell isn't installed there), and needs
   `PYTHONIOENCODING=utf-8` **exported at the shell level before launching
   python** (bombcell prints a unicode char on import; setting the env var
   inside the script is too late — Windows fixes console encoding at
   interpreter startup, not from anything the running script does). Find
   `gain_to_uV` fresh per session — grep the session's own raw `.rec` file
   header for `spikeScalingToUv=` (embedded Trodes config XML near the top of
   the file) rather than assuming a prior session's value carries over, even
   though it happened to be identical (`0.018311105685598315`) for both
   sessions done so far. Writes `outputs/bombcell_results_<session_id>/`,
   including `templates._bc_qMetrics.csv` (needed by the next step). Takes
   a few minutes (466 units: ~2-3 min with 16 parallel workers).
2. **`corrected_classification_session.py <session_id> <rec_root>`** — session-
   parameterized version of `scripts/corrected_classification.py` (that
   original file is hardcoded to `20260901_085606` — it's the calibration
   reference, leave it alone). Self-contained given bombcell's qMetrics +
   `kilosort_original` + channel map — does NOT depend on the batched agent
   pipeline's own per-unit output at all. Run with the `kilosort` env. Writes
   `outputs/corrected_classification_<session_id>.csv` +
   `_with_waveforms.json`. Calibrated thresholds live in this script and in
   `striatal_agent/config.py` — check both before assuming a threshold is
   still what you remember.
3. **`build_pipeline_review_data.py <session_id> <rec_root>`** — adds
   `acg_hist` (+ its bin/lag/refractory/shoulder params) via the already-
   proper `striatal_agent.acg_tools.compute_acg_hist()`, and renames
   `corrected_classification`'s already-computed `ks_template` field to
   `waveform_top4` (same data, just the name the review tool's JS expects —
   **no extra computation needed for that field**, don't re-derive it).
   Writes `outputs/pipeline_review_data_<session_id>.json`. Run with `kilosort`.
   **Does NOT yet compute** `cluster_points`, `sep_vs_noise`/`sep_vs_other`/
   `co_located_units`, `burst_curve`, `isi_amp_scatter`/`isi_amp_corr_*` — see
   the field-source table in 1f-pre above for what's still missing and why.
   The published tool degrades gracefully without these (shows "not yet
   computed" in the relevant panel) rather than crashing — confirmed safe,
   except `co_located_units` needed a one-line null-guard fix in step 4's
   script (`(u.co_located_units||[]).length`, already applied).
3b. **`build_session_timeline_data.py <session_id> <rec_root> [--bin_s 20]`** —
   builds `outputs/session_timeline_<session_id>.json`: per-unit spike counts
   per time bin across the whole session, plus trial-start markers decoded
   from the raw Trodes DIO stream (TTL code 1). Feeds the "Firing rate across
   the session" panel and its kept-time-range selection. Any numpy env
   (`phy2_ky` is fine); ~5s per session. **Optional** — step 4 builds without
   it and the panel says "not computed for this session". Two things not to
   re-derive (both explained at length in the script's own docstring): spike
   sample indices MUST go through `<session>.timestamps.dat` or the rate curve
   slides relative to the trial ticks by the ephys export offset (87.5134s on
   `20260916_110311`), and the trial ticks come from TTL code 1, not code 0.
4. **`build_review_html.py <session_id>`** — reads
   `pipeline_review_data_<session_id>.json`, renders
   `outputs/unit_review_pipeline_<session_id>.html`, sets the page's `<title>`
   and header to the session id. Run with `kilosort`. Then publish via the
   Artifact tool with **`capabilities: {"db": {}}`** explicitly declared — the
   tool calls `claude.use('db')` to save verdicts, and the publish silently
   produces a tool that *looks* fine but can't save anything if you forget
   this flag (caught this the first publish attempt — republished with the
   flag to fix, same URL). `favicon` required on first publish for a session's
   artifact (pick something, e.g. 🧠); omit on republishes to the same URL.

5. Human reviews units in the tool; verdicts get saved to the artifact's own
   database. Pull the current verdict snapshot into
   `outputs/current_verdicts_<session_id>.json`
   (`{unit_id_str: "good"|"mua"|"noise"}`) when you need it as a plain file for
   scripting (the review tool's live DB is the source of truth; this JSON is a
   point-in-time export).

**The lesson this chain exists to prevent repeating**: building a second
session's review tool from memory alone (no saved scripts) would have meant
re-deriving all of this — which conda env bombcell needs, the encoding gotcha,
which fields are free vs. need real computation vs. are simply unavailable.
Whenever you add a new field to the review tool or fix something in this
chain, **edit the numbered steps above in the same sitting** — don't let the
description drift from what the scripts actually do.

### 1g. Where the review tool lives per session

**Convention (adopt going forward)**: one published Artifact per session, titled
distinctly (e.g. include the session id), tracked in
`outputs/session_registry.json` under that session's `review_tool_url` field —
don't republish over a *different* session's artifact URL, and don't assume
there's only ever one review tool in flight. See the registry file for the
current list; update it immediately after every publish/republish.

---

### 1h. Manual curation that does NOT need Phy: stability windows and merges

Two curation passes now go straight from the review tool into the dataset **without
touching `spike_clusters.npy` and without a Phy session at all**. Both work the same
way, and the rule is worth stating once:

> The review tool **never writes to the dataset**. It writes only to its own artifact
> database. A bridge script exports a plain JSON file into the Kilosort folder, and
> `sync_pipeline` reads it **by convention** on the next run. The pipeline depends on a
> **format**, never on a tool — you can hand-write either file.

| pass | bridge script | file written (into `<session>.kilosort/kilosort4/`) | dataset output |
|---|---|---|---|
| stability windows | `scripts/write_unit_trial_selection.py` | `unit_trial_selection.json` | `unit_trial_selection.parquet`, `units.has_stability_selection` |
| merges | `scripts/write_unit_merges.py` | `unit_merges.json` | `unit_merges.parquet`, `units.is_merged` / `n_parents` / `parent_unit_ids` |

Full reference: `sync_pipeline/HOW_TO_RUN.md` §6b (stability) and §6c (merges).
`scripts/read_synced_with_selection.py <synced_folder>` is a runnable worked example of
consuming both.

**The step people forget is the third one.** Marking windows or deciding merges changes
nothing in the dataset by itself:

```
1. decide it (review tool)
2. export it     (bridge script -> JSON in the kilosort folder)
3. RE-RUN THE PIPELINE for that session   <-- without this, nothing reaches synced/
```

A `synced/` folder built before a curation file existed does **not** contain it and
never gains it retroactively. `session_meta.json` always states what happened, per
pass: `unit_trial_selection.status` and `unit_merges.status`, each one of
`applied` / `not_found` / `disabled_by_flag`.

#### Why merges go through a manifest instead of Phy

Phy's merge **rewrites `spike_clusters.npy`**: the parents stop existing and ids get
renumbered. From that moment, every unit id written down anywhere else — a review-tool
verdict, a stability window, a figure, a note in a log — points at a *different neuron*
than it did before. Not missing: **wrong**, and silent.

The manifest route leaves the clustering immutable and applies merges in memory while
building `synced/`. So:

- parent ids stay valid forever (unit 38 is still unit 38 in the Kilosort folder);
- the clustering fingerprint keeps matching, so **merging does not invalidate a
  stability selection** — the two passes are independent, in either order;
- un-merging is deleting a line, and two competing manifests can be compared by
  processing the session twice (`--unit_merges PATH`);
- merged unit ids start at **10000**, so a synthetic unit is recognisable on sight.

Stability windows combine as the **union** of the parents', **frozen at merge time** —
re-marking a parent later does not change a merged unit already signed off. The
parents' own ranges are kept beside it as provenance.

**You score the merged unit, and that is a separate step on purpose.** A merge's
quality can only be judged after the merge exists, so the cycle is: merge → re-run →
look at the merged unit → `write_unit_merges.py --score 10000=good` → re-run. An
explicit label wins over everything; without one the pipeline inherits the parents'
label, or falls back to `mua` when they disagree, which is a placeholder that
undersells real merges. `unit_merges.quality_label_source` (`manual` / `inherited` /
`conservative_mua`) is how you tell a judged merge from an unjudged one.

**The cost, so it is never a surprise:** the Kilosort folder on disk no longer shows the
final truth. Phy, bombcell and SpikeInterface read `spike_clusters.npy`, so after
merging they will show the **parents, unmerged**. The merged view lives in `synced/`.

### Getting your review into the dataset — one command

```
python scripts/sync_session.py <session_id> --rec <rec_folder> --rat_root <data_root>
```

That applies **every** decision and then proves it arrived. For Shamir:

```
python scripts/sync_session.py 20260916_110311 ^
    --rec     F:\Gil\Shamir\20260916_110311.rec ^
    --rat_root Z:\Gil\Shamir_1
```

`--rat_root` is needed because the behaviour (`bpod/`, `rat_metadata.json`) lives with
the archive on Z: while the ephys works from F:. Without it the pipeline looks for
`bpod/` next to the `.rec` folder and cannot auto-match the session.

#### The one-button path (no Claude in the loop)

The left panel of the review tool has two numbered steps:

1. **↓ 1 · export all for sync** — saves the three files, with a live count of what
   it would write.
2. **2 · then paste this into cmd** — the command, with this session's id already
   filled in, and a **copy** button next to it.

```
cd /d D:\Gil\spike_sorting_agent && "C:\Users\Adam\anaconda3\envs\phy2_ky\python.exe" scripts\sync_session.py <sid> --rec "F:\Gil\Shamir\<sid>.rec" --rat_root "Z:\Gil\Shamir_1" --collect_from_downloads
```

That is the whole hop, and there is **no copying files by hand**:
`--collect_from_downloads` runs `scripts/collect_verdict_dumps.py`, which finds the
three files in your Downloads folder itself and copies them into
`outputs/review_verdicts/<session>/`.

Three things in that command are easy to get wrong by hand, which is why the page
prints it instead of leaving it to be retyped:

- `cd /d` first, because the script path is relative.
- the **full** interpreter path. `python` on PATH is the base conda install, which
  has no pyarrow — the pipeline would run for minutes and then die writing its
  first parquet file.
- `--rat_root`, because the Bpod behaviour lives with the archive on Z: while the
  ephys works from F:.

Those four paths are constants at the top of
`scripts/patch_review_tool_sync_command.py`. If the repo moves, the conda
environment is rebuilt, or another rat is recorded, change them there and rebuild
the tool.

**A button cannot run the Python itself.** The review tool is a web page: its sandbox
has no filesystem and cannot start a process, so no button anywhere can launch
`sync_session.py`. What a page *can* do is hand you files, which is what this one
does. The local command is irreducible without replacing the page with a local server.

`collect_verdict_dumps.py` is deliberately suspicious of what it finds, because a
stale export is worse than a missing one:

- browsers never overwrite, so a second export lands as `verdicts_<sid> (1).json`.
  It considers every spelling (` (1)`, `(1)`, `-1`, `_1`), takes the **newest**, and
  prints which ones it ignored.
- it prints each file's age and refuses anything older than `--max_age_hours`
  (default 24).
- it checks each file really is the collection its name claims before copying.
- if any of the three fails, **nothing** is copied.

#### If you would rather I do it

Ask: *"dump the review tool verdicts for &lt;session&gt;"*. I read the three collections
with the `ArtifactData` tool and write them to
`outputs/review_verdicts/<session>/`, after which the same `sync_session.py` command
runs without `--collect_from_downloads`. Same destination, same files — the button
just removes the wait for me.

**It refuses to run without your decisions.** It expects three dumps of the review
tool's database in `outputs/review_verdicts/<session_id>/`, and names what is missing
and what that would have become:

| dump | becomes | effect |
|---|---|---|
| `verdicts.json` | `cluster_info.tsv` `group` column | per-unit GOOD / MUA / NOISE |
| `trial_selections.json` | `unit_trial_selection.json` | per-unit stability windows |
| `merge_verdicts.json` | `unit_merges.json` | merges, chained transitively |

Those dumps live only in the artifact's database until someone writes them here, so ask
Claude: *"dump the review tool verdicts for &lt;session&gt;"*. They are committed to the
repo, which means your review also has a backup outside claude.ai.

Applied in dependency order — stability windows, then merges (a merge freezes the
**union** of its parents' windows at merge time), then unit verdicts. Then
`sync_pipeline` runs, and then:

```
python scripts/verify_verdicts_in_dataset.py <session_id> --rec <rec> --dumps <dir>
```

runs automatically. Nine checks compare your decisions against the finished `synced/`:

| | catches |
|---|---|
| T1 | a verdict made against a different sorting — the ids would name other cells |
| T2 | a good/mua verdict that did not change `quality_label` |
| T3 | a unit you called noise that still has rows in `units`/`spikes` |
| T4 | a unit nobody judged whose label moved anyway |
| T5 | a merge missing, mis-grouped, parents left behind, or mislabelled |
| T6 | two units merged although you judged them separate |
| T7 | a stability window that did not reach the dataset (compared trial by trial) |
| T8 | **any verdict not accounted for** — the check that catches silent drops |
| T9 | spikes referencing a unit absent from the units table |

It shares no code with the scripts that apply the verdicts, and recomputes the merge
grouping itself, so a bug in a writer shows up as a disagreement rather than being
reproduced on both sides. On failure the run raises and `outputs/verdict_sync_<sid>.json`
records `verified: false`. `--no_verify` exists for debugging and is recorded too.

This is why it exists: **132 of 240 verdicts sat unapplied for weeks** because the
script meant to carry them was hardcoded to another session and still exited 0. Nothing
compared the decisions against the result. Now the ordinary way of running the pipeline
is a thing that fails when they are missing or wrong.

#### Two merge rules that are easy to get wrong

**A parent with no stability window keeps every trial.** So merging a restricted parent
with an unrestricted one gives the WHOLE session, not the restricted parent's window.
Unioning only the parents that happen to have windows once cut merged unit 10004
(262+263+264) down to 27 of 953 trials. `freeze_union()` handles this and records
`unrestricted_parents` in the manifest so the wide window does not look like a bug.

**Merged ids are pinned to their parent set.** `--score 10002=good` has to mean the same
neuron next month, so re-running the merge writer reuses the id a parent set already
has, and a retired merge keeps its id reserved. Allocating by position silently
renumbered: 233+236 was 10001 before four new merges sorted ahead of it.

#### Building the review tool for a new session — one command

```
python scripts/build_review_tool.py <session_id>
```

That is the whole thing. It produces `outputs/unit_review_tool_<sid>.html`, complete:
unit panels, session timeline and stability windows, AND the merge panel with every
addition. It runs, in order and failing loudly if any anchor has drifted:

| step | adds |
|---|---|
| `build_review_html.py` | the base page — waveforms, scatter, ACG, burst curve, timeline, verdicts |
| `patch_review_tool_merge_panel.py` | the possible-merges panel |
| `patch_review_tool_merge_v2.py` | dip measured on the shoulders + Poisson test, the across-channel footprint, firing rates above the merged ACG |
| `patch_review_tool_merge_v3.py` | working verdict buttons, panel at the bottom, jump to the other unit, twin templates |

The merge panel grew as patches spliced onto an already-published page, which is only
reproducible if you remember the order — so `build_review_tool.py` is the order, kept
in one place. Each patch asserts its anchors match **exactly once**, so editing
`build_review_html.py` in a way that moves one stops the build instead of quietly
publishing a half-patched page. Verified: the chain run against a clean
`build_review_html.py` output reproduces the published page byte for byte apart from
the publisher's own HTML skeleton.

Build its three inputs first:

```
python scripts/build_pipeline_review_data.py <sid>     # outputs/pipeline_review_data_<sid>.json
python scripts/build_session_timeline_data.py <sid>    # outputs/session_timeline_<sid>.json
python scripts/build_merge_candidates.py <sid>         # outputs/merge_candidates_<sid>.json
```

and publish the page **with** its data file, which it fetches lazily at the relative
path `data/merge_candidates.json`:

```
file_path    = outputs/unit_review_tool_<sid>.html
files        = {"data/merge_candidates.json": "outputs/merge_candidates_<sid>.json"}
capabilities = {"db": {}, "downloads": true}
```

Pass the **existing artifact URL** to keep its verdicts — publishing without one makes
a new artifact with an empty database.

#### The merge tool: suggestions while you review a unit

Merge candidates are **inside the review tool**, in the unit you are looking at. Open a
unit and, under the session timeline, *Possible merges* lists the other clusters that
might be the **same neuron** — with the evidence for each, and a verdict you can set
without leaving the unit.

> Review tool (merge panel included): https://claude.ai/code/artifact/74dec064-c86a-46a3-8c03-f112ee4ecb05

Two things keep the list honest on a dense probe, where almost every unit has *some*
neighbour within 60 µm:

- the panel headline is **"N worth a look of M neighbours"**, and only the N are shown.
  A candidate is worth a look when it has a readable refractory dip or a drift-pair
  signature. The rest sit behind *show the other M neighbours*, never gone, never in
  the way;
- the unit list marks only those N, as a `N⇄` chip, so scanning the list shows you
  where the real candidates are. `with a merge candidate worth a look` is also a filter
  in the review dropdown.

Both come from `scripts/build_merge_candidates.py <session_id>`, which exports every
pair within 60 µm (2790 of them for 20260916_110311, ~2 min, 9.6 MB) and is published
alongside the page as `data/merge_candidates.json`, fetched lazily so the page still
loads at its old speed. `scripts/patch_review_tool_merge_panel.py` is how the panel is
spliced into a freshly-read copy of the published page.

There is also a standalone queue over the same data and the same database —
https://claude.ai/artifact/DLDrEjL5Sn74TvBuMnmdcW — for working through candidates in bulk rather than
unit by unit. A decision made in either shows up in the other.

Each candidate shows the four things a merge decision actually rests on, in the order
they should be weighed:

| panel | what it answers | why it is there |
|---|---|---|
| **cross-correlogram** | do they ever fire too close together? | the decider. One neuron cannot fire inside its own refractory period, so one neuron = a DIP at zero lag. The dip is measured against the **2–5 ms shoulders beside the refractory window**, never the far flanks — a flank baseline is invalid whenever the correlogram has broad structure, and on 20260916 it called 30 pairs one neuron where only 13 survive shoulders plus a Poisson test. Both baselines are displayed, with any disagreement flagged, and the expected count always sits beside the observed one |
| **firing rates** | is the pair lopsided? | shown above the merged ACG. At a large rate ratio the merged train is mostly the fast unit, so its violation *percentage* is nearly unchanged however wrong the merge is; the panel leads with the **count** of violations the merge creates instead |
| **footprint across channels** | one neuron, or two at one spot? | both mean waveforms on every nearby channel, each at its real position on the probe, one µV scale. One neuron has one footprint |
| **merged ACG** | what does the unit this merge creates look like? | the closest thing to seeing the merged unit before it exists. If merging fills in the refractory gap, it is not one neuron whatever else matches |
| **raw waveforms + footprint** | same shape, same size, same place? | means read from `probe1.dat`, in **microvolts on one shared scale**, on the channels the two units share. A normalised overlay would hide a 40 µV / 400 µV mismatch |
| **session timelines** | is this a drift pair? | one unit fading out as the other fades in. Then the CCG is empty and uninformative *by construction*, and only this panel can decide |

Kilosort's `similar_templates.npy` never *decides* a merge — it compares templates,
so two different neurons with similar-shaped waveforms score high. But it is not
noise either, and the first version of this document oversold that point. Measured on
20260916_110311 against a chance rate of **0.51%** (the fraction of the 2554 readable
pairs showing a one-neuron dip):

| similarity | pairs | one-neuron dip | vs chance |
|---|---|---|---|
| 0.50–0.80 | 418 | 1.4% | 2.8× |
| 0.80–0.90 | 57 | 1.8% | 3.4× |
| 0.90–0.95 | 15 | 6.7% | **13×** |
| ≥0.95 | 17 | 0.0% — every one has a partial dip the CCG cannot settle | — |

Read it as a prior, never a verdict. Above 0.95 it selects almost perfectly for pairs
whose correlogram has too few coincident spikes to decide, which is exactly when the
footprint and the session timelines carry the decision — so the tool surfaces those
as **twin templates** rather than burying them.

**Nothing about a decision reaches the dataset on its own.** The chain is three
hops, and only the first is automatic:

```
click a verdict        ->  artifact db `merge_verdicts`     (instant, nothing else needed)
write_unit_merges.py --from_verdicts  ->  <rec>/<session>.kilosort/kilosort4/unit_merges.json
re-run the pipeline    ->  synced/ now reflects the merge
```

`--from_verdicts` takes the raw database dump (one document per PAIR) and does the
transitive chaining itself, so there is no intermediate file to build:

```
python scripts/write_unit_merges.py <sid> <rec>/<sid>.rec verdicts.json --from_verdicts
```

SEPARATE and UNSURE are counted and ignored — a rejection is a real decision, it just
has no representation in a file that only lists merges.

**Why this cannot be fully automatic, and why the unit verdicts are not either.** The
review tool is a web page in a browser sandbox on claude.ai: it cannot write to
`F:\` any more than any other web page can. So *something on this machine* has to
pull from the artifact database. That is equally true of the GOOD/MUA/NOISE verdicts,
which reach the dataset through `export_phy_curation.py` writing `cluster_group.tsv`
into the Kilosort folder — the same shape of bridge, the same number of hops. Neither
is automatic; the verdict bridge has simply been run more often. Ask Claude to sync
the verdicts and the whole chain runs in one step.

The pipeline reads **only** that last file, and picks it up automatically once it
exists — it never talks to the artifact, and the artifact never writes to the dataset.
One plain documented file in one conventional place sits between them, exactly as for
stability windows.

Decisions live in the artifact database (`merge_verdicts`, one document per pair,
stamped with the clustering sha1) and export as the input JSON for
`write_unit_merges.py`, with accepted pairs **chained transitively** — accept 38+163
and 163+204 and you get one three-parent merge, which is what the pipeline requires.
The merged unit's `quality_label` is set in the same row, from the merged ACG and the
combined waveform, which is the only honest moment to judge it.

And the step that is easy to miss, exactly as for stability windows: deciding in the
tool changes **nothing**. Export → `write_unit_merges.py` → **re-run the pipeline**.

#### What this means for STAGE 2 and §2c

Phy is still the route for **splits** and for **label changes**, and §2c's
merge-verification procedure still applies to any merge that was done *in Phy*. But a
merge decided in the review tool should go through `write_unit_merges.py`, not Phy —
then there is no `phy.log` to chase, no renumbering to verify, and §2c has nothing to
do for it.

If a session has both (Phy merges *and* a manifest), note that the manifest's
fingerprint is stamped against the clustering it was made for. Post-Phy, that
fingerprint will not match, and the pipeline will refuse the manifest with a hard
error — which is the correct behaviour, not a bug. Redo the merge pass against the
post-Phy clustering.

---

## STAGE 2 — after Phy: rebuild sync output from the final clustering

### 2a. Precondition: Phy runs on a clean copy, never on the irreplaceable original

Never launch Phy directly against `kilosort_original/` (the only pristine,
pre-curation backup — treat it read-only, always). Before a human curation pass,
make a **fresh working copy** (e.g. `kilosort4/verdict_review/`) containing:

- Every file from `kilosort_original/` needed by Phy (`spike_times.npy`,
  `spike_clusters.npy`, `templates.npy`, `amplitudes.npy`, `channel_*.npy`,
  `pc_features*.npy`, `similar_templates.npy`, `whitening_mat*.npy`, `ops.npy`,
  `params.py`) — **except** any bombcell-derived `cluster_*.tsv` files
  (`cluster_Lratio.tsv`, `cluster_bc_unitType.tsv`, `cluster_isolationDistance.tsv`,
  etc. — there's a long list of these if bombcell was ever run on this session;
  exclude all of them) and except `phy.log` / `.phy/` (stale cache from wherever
  the source snapshot came from — let Phy build a fresh one).
- Two extra columns for phy's cluster view, generated fresh (not copied):
  - `cluster_verdict.tsv` — the current manual review-tool verdict per cluster
    (from `current_verdicts.json`), for units that have one.
  - `cluster_violation_ratio.tsv` — fraction of ISIs under a 2ms refractory
    window, computed directly from that copy's own `spike_times.npy`/`spike_clusters.npy`.
- **No `cluster_group.tsv`** — start Phy's own group field empty/unsorted, so the
  human's labeling in this pass is unambiguous (not mixed with a stale prior
  `group` column).

Launch: `D:\conda_envs\kilosort\Scripts\phy.exe template-gui params.py` from
inside that copy's folder (not `python -m phy` — that fails, phy has no
`__main__.py`; the earlier `ls` failure that looked like `phy.exe` was missing
was a bash quoting bug, not a real absence — check with `ls Scripts | grep phy`,
not a glob with a stray quote).

### 2b. Verdict precedence when building the final label set

**Phy label (this session's `cluster_group.tsv`) > manual review-tool verdict
(`current_verdicts.json`) > automatic classifier
(`corrected_classification.csv`, `label` field, `+NON-SOMA` suffix stripped).**

Every unit gets a verdict this way — there should be zero "truly unreviewed"
units left once you fall all the way back to the automatic classifier, since that
covers all original units. Newly-created cluster IDs from a Phy merge (see
below) only ever have a Phy label (there's no "previous" verdict for a brand-new
ID) — that's expected and fine, not a gap.

### 2c. Merges (and splits, if any happen in the future) — verify empirically, don't trust the log alone

> **Read §1h first.** Merges no longer have to go through Phy at all. A merge decided
> in the review tool is written to `unit_merges.json` and applied in memory by
> `sync_pipeline`, leaving `spike_clusters.npy` untouched — so none of the
> renumber-chasing below applies to it. This section covers merges actually performed
> **in Phy**, which do rewrite the clustering.

Phy logs each merge as two identical lines (`Merge clusters A, B to C.`) in
`phy.log`. **This is not sufficient by itself** — a merge can be immediately
undone (`Undo cluster assign.`), and the log doesn't make this obvious at a
glance (an "Undo move." line is a *different* action — undoing a group-label
move, not a merge — don't conflate the two when scanning for undos).

**Always cross-check against the actual current `spike_clusters.npy`**, not just
log semantics:

```python
# for every merge output C with inputs [A, B] (chase multi-step chains, e.g.
# a merge result that itself got merged again):
#   - if C is present in current spike_clusters.npy and A,B are absent -> real, kept
#   - if C is absent and A,B are present -> merge was undone, ignore C entirely
#   - if C is absent and A,B are also absent -> C itself got consumed by a further
#     merge; trace forward to find its final surviving descendant
```

This exact check caught 4 apparently-real merges (out of 15 attempted, on session
`20260901_085606`'s Phy pass on 2026-09-15) that had actually been undone
seconds later — trusting the log alone would have produced a final unit count
mismatch (verified: 401 original units - 15 "naive" merges = 386 expected, but
the *real* count after accounting for undos was 389; empirical cross-check
against `spike_clusters.npy` is what resolved the discrepancy, hand-tracing log
semantics alone was getting it wrong).

A unit consumed by a real merge (e.g. original units 275 and 278, merged into new
unit 401) **no longer exists as a separate entity** — it must not appear
anywhere in the final unit list, verdict mapping, or review tool. Only the merge
result (401) appears, carrying the combined spikes of both.

### 2d. Build `cluster_info.tsv` for the post-Phy clustering

The sync pipeline's `units_spikes.build_units_table()` requires a
`cluster_info.tsv` with columns `cluster_id, ch, depth, group, KSLabel` (plus
whatever else is present) — the clean Phy working copy won't have one (we
deliberately didn't copy the original). Build it fresh for the *current* cluster
set (post-merge):

- `ch` / `depth`: for original (pre-merge) unit IDs, read directly from that
  unit's row in `templates.npy` (peak channel = argmax of `template.max(axis=0) -
  template.min(axis=0)`, depth = that channel's `channel_positions.npy` y-coordinate).
  **Merge-result cluster IDs have no row of their own in `templates.npy`**
  (Kilosort never computed a template for a cluster that didn't exist at sort
  time) — use one of the merge's original constituent units as a representative
  proxy (reasonable since merges only happen between spatially-close, similar
  units by construction). Resolve chained merges to their earliest original
  constituent.
- `group`: the final verdict from the precedence rule above (2b) — this makes
  the sync pipeline's own `noise`-exclusion logic (`build_units_table` drops
  `group=='noise'` clusters) correctly reflect the *final*, human-reviewed
  labels, not the stale pre-Phy ones.
- `KSLabel`: not really used once `group` is populated for every row (quality
  label falls back to `KSLabel` only when `group` is blank) — `"good"`/`"mua"`
  passthrough of `group` is a fine placeholder.

### 2e. Run the sync pipeline against the Phy working copy, not the standard kilosort4/ folder

`sync_pipeline`'s CLI (`process_session.py`) auto-discovers the Kilosort4 folder
by naming convention (`<session>.kilosort\kilosort4\`) with no direct override
flag. To point it at `kilosort4\verdict_review\` instead, monkeypatch
`session_discovery.session_paths_from_rec_folder` to return a `SessionPaths`
with `kilosort4_folder` swapped (it's a plain dataclass — `dataclasses.replace`
works cleanly). `channel_map_json` discovery is unaffected since it globs the
*parent* (`kilosort4\`'s sibling), which doesn't change.

```python
from unittest.mock import patch
import dataclasses
from sync_pipeline import process_session, session_discovery

_orig = session_discovery.session_paths_from_rec_folder
def _patched(rec_folder, bpod_mat_file):
    paths = _orig(rec_folder, bpod_mat_file)
    return dataclasses.replace(paths, kilosort4_folder=Path(VERDICT_REVIEW_DIR))

with patch.object(session_discovery, "session_paths_from_rec_folder", side_effect=_patched):
    summary = process_session.process_session(
        rec_folder=REC_FOLDER, bpod_file=BPOD_FILE, rat_root=RAT_ROOT,
        output_dir=VERDICT_REVIEW_DIR + r"\synced",
    )
```

Find `bpod_file` explicitly (e.g. under `<rat_root>\bpod\`, matched by session
date) rather than relying on auto-match if the working copy's `.rec` folder
location doesn't have the usual `bpod/` sibling folder auto-match expects.

### 2f. Output — this is now THE dataset

The new `synced/` folder inside the Phy working copy (e.g.
`kilosort4\verdict_review\synced\`) — `spikes.parquet`, `trials.parquet`,
`units.parquet`, `state_events.parquet`, `poke_events.parquet` — reflects the
complete chain (automatic -> manual review -> Phy merges/labels). **Use this,
not any earlier `synced/` folder or any pre-Phy spikes table, for all further
analysis on this session.** Same column schema as before
(`spikes.parquet`: `unit_id, trial_id, spike_time_in_trial, spike_time_ephys,
local_calib_gap_s`) — drop-in compatible with every existing analysis script.

Update `outputs/session_registry.json` with the new `synced_data_path` and
`final_verdicts_path` immediately after this step.

---

## Change log (append, don't delete)

- **2026-09-15**: Documented Stage 1/Stage 2 split for the first time, after
  doing it manually for `20260901_085606`. Found + worked around the
  large-session pipeline crash (batching). Established the merge-undo
  verification method. `skip_cross_unit_acg=True` adopted as the default
  scripting choice (not yet changed as `run_agent.py`'s own CLI default).
- **2026-09-15 (later)**: Fixed a real (non-memory-related) bug found during
  the `20260911_100049` batched run: `recovery.py`'s `sweep_and_select()` has
  an early-return path (taken when a unit has `<10` KS-matched "accepted"
  spikes or zero raw candidates -- i.e. very low-activity units) that was
  missing the `n_recovered` / `pct_spikes_added` keys present on the normal
  return path, causing a `KeyError` in `pipeline.py` whenever a batch happened
  to include such a unit (crashed batch 36 -- unit 365 -- losing all 10 units
  in that batch, since the exception hit mid-loop before any of that batch's
  output was written). Fixed by adding the missing keys
  (`n_recovered=0, pct_spikes_added=0.0`) to the early-return dict. This was
  latent in the code before this session too -- it just happens to not have
  been hit by any of `20260901_085606`'s 401 units. **If re-processing that
  session from scratch in the future, this fix applies there too** (harmless
  no-op for units that don't hit the edge case, correct instead of crashing
  for any that do).
- **2026-09-15 (evening)**: Closed the review-tool "1f-pre" gap for
  `20260911_100049` -- built and ran the remaining 4 session-parameterized
  scripts (`build_trodes_channel_lookup_session.py`,
  `raw_cluster_points_session.py`, `raw_separation_metric_session.py`,
  `build_burst_isi_metrics_session.py`, `compute_raw_trace_snippets_session.py`),
  merged everything via the new `merge_review_data_session.py`, rebuilt and
  republished the HTML. The "1f-pre" table above is now the authoritative,
  current field-source map -- treat it as finished, not a to-do list, for any
  future session. Specific things worth remembering:
  - `isi_amp_scatter` is **confirmed dead and permanently dropped** from the
    pipeline -- grepped `build_review_html.py`'s JS and found nothing reads
    it, even in the original `20260901_085606` data where the field existed.
    Do not compute it for any future session.
  - Two more fields turned out to be dead the same way, found while reverse
    engineering the merge formulas: `n_burst_pairs` and `n_true_bursts` (only
    `burst_curve` and `burst_shape` are actually rendered). Neither is
    produced by `build_burst_isi_metrics_session.py`, and that's correct --
    don't add them.
  - The exact merge formulas for `sep_vs_noise`/`sep_vs_other` (scalars) were
    reverse-engineered from the one surviving old-session raw intermediate
    file that matches `20260901_085606` by unit count (`raw_separation_full_401.npy`,
    `raw_cluster_points_full_401.json` -- the "_401" scratch files are that
    session's leftovers, not a separate test session) and confirmed exactly
    against `outputs\pipeline_review_data.json` (the old session's own file,
    still present, no suffix): the scalar is `round(np.percentile(valid_chunk_values, 80), 4)`
    (numpy default linear interpolation) over the 10 chunk values, excluding
    NaN chunks; `n_valid_noise`/`n_valid_other` are the count of non-NaN
    chunks; `co_located_units` is exactly `raw_separation`'s own `other_units`
    list, unmodified (verified unit 0: `other_units=[]` -> `co_located_units=[]`;
    unit 5: `other_units=[7,8]` -> `co_located_units=[7,8]`, `sep_vs_noise`/`sep_vs_other`
    p80 values matched to 4 decimal places). This formula was NOT written down
    anywhere before this pass -- it's now captured in
    `merge_review_data_session.py`'s own docstring too, so it survives even if
    this file is skimmed past.
  - `raw_traces` pushed to the artifact DB in 32 batches of <=15 (466 total
    documents, unit ids 0-465, all committed -- confirmed via the tool's own
    running document count reaching 472/5000, +6 for pre-existing `verdicts`
    docs from manual curation testing).
  - Session registry updated to mark `20260911_100049`'s review tool as fully
    complete (no more "KNOWN GAP" note) -- see below.
- **2026-09-16**: Found and fixed a real bug (user caught it, not a
  self-review): `merge_review_data_session.py` merged cluster_points,
  separation metrics, and burst/ISI data into the review JSON, but never
  merged `trodes_channel` -- even though
  `unit_trodes_channel_lookup_<session>.csv` (built earlier from this
  session's own real Trodes hardware config, values ~1086-1466) was already
  sitting there, correctly computed. The merge step for that one field was
  simply never written. Every unit in the first published
  `20260911_100049` review tool therefore had `trodes_channel: null`, and the
  UI's `u.trodes_channel ?? '—'` fallback silently displayed the 0-383 raw
  Kilosort channel index instead (labeled "KS channel" right next to the
  dashed-out Trodes one, but easy to read as *the* channel number). **The
  user caught this by noticing the displayed numbers were far smaller than
  the ~1083-1466 range they expected for this probe** -- root-caused to the
  missing merge step (not a cosmetic renumbering), fixed by adding the CSV
  merge to `merge_review_data_session.py`, re-run, verified all 466 units now
  carry the correct value, HTML rebuilt and republished. **Lesson: building a
  lookup table is not the same as confirming it actually made it into the
  final merged JSON** -- after adding any new per-unit data source, spot-check
  the actual field in the built output (e.g. grep the built HTML for
  `"trodes_channel": null`) rather than only confirming the source file looks
  right in isolation.
- **2026-09-16**: Raw spike-traces panel upgraded per user request, for
  `20260911_100049` (should be the default going forward for any new session
  too -- the old 8-trace, single-red-line version is superseded, not an
  alternate mode):
  - Every trace window now marks ALL spikes detected in it, not just the one
    it was centered on. "Detected in it" = any spike (from this unit, or any
    other unit sharing this unit's peak channel -- i.e. exactly the same set
    already surfaced elsewhere as `co_located_units`) whose recorded spike
    time falls inside the trace's `[t0-pad, t0+pad)` window. Units on OTHER
    channels are deliberately excluded -- a coincidental spike on a
    physically distant channel wouldn't actually appear as a deflection in
    this channel's voltage trace, so marking it would mislead rather than
    help the "does this deflection correspond to a real, sorted spike" check
    this feature exists for. This required a new per-trace `spikes: [{unit_id,
    offset_ms}]` field, added in `compute_raw_trace_snippets_session.py`
    (`spikes_in_window()`), using the same per-unit spike-time cache already
    built for the "own" spike train (no new raw-data reads -- just extra
    `searchsorted` calls against arrays already in memory).
  - JS (`build_review_html.py`): `renderTracePair` now draws one vertical
    line per entry in `t.spikes` -- solid red for this unit (`unit_id===uid`),
    dashed + a distinct color + a small "#<id>" label for every other unit,
    assigned from a fixed palette in first-seen order per render (so the same
    other-unit id keeps the same color across all traces shown for one unit).
    A small legend renders above the trace list mapping each color back to a
    unit number.
  - N_SPIKES doubled from 8 to 16 (in `compute_raw_trace_snippets_session.py`)
    per the user's ask to be able to inspect roughly twice as many examples.
    Doubling only doubles the number of small raw-data window reads (no other
    cost scales with it) -- 466 units went from ~110s to ~181s, still cheap.
    The review tool does NOT render all 16 up front, though: `renderRawSpikesPanel`
    shows the first `RAW_TRACES_INITIAL_N=8` on "Show" and adds a "Show N more
    traces" button that reveals the rest on demand, so the common case (someone
    just spot-checking a unit) still only pays for 8 SVG renders.
  - **Gotcha hit and fixed**: doubling N_SPIKES roughly doubled each unit's
    `raw_traces` doc size (~46KB avg -> ~93KB avg, ~101KB max), which meant the
    previously-safe 15-writes-per-batch `write_db` batching (see the
    "Raw spike traces are architecturally different" note above) would have
    exceeded the 1MB/request limit. Recomputed and switched to 8/batch for
    this push. **Recompute the safe batch size any time N_SPIKES or the trace
    window changes** -- don't assume the old constant still holds.
  - Also hit and fixed a **Windows-path-via-Bash escaping bug**: running
    `compute_raw_trace_snippets_session.py` from the Bash tool with an
    unquoted `F:\\Gil\\Shamir\\...` argument silently stripped ALL backslashes
    (bash consumed them as escape characters twice over), producing a mangled
    path and a `FileNotFoundError` that only showed up in the background
    task's own output log, not the initial tool result (the process had
    already been backgrounded and reported "completed, exit code 0" -- always
    check the actual log/progress file after a backgrounded run, don't trust
    "completed" alone). **Fix: always double-quote a Windows path argument
    passed through the Bash tool** (`"F:\Gil\Shamir\...\rec"`), single
    backslashes, not double -- quoting stops bash from treating the
    backslashes as escapes at all, which is what's needed on Windows paths.
- **2026-09-16**: Ran the sync pipeline for `20260911_100049` for the first
  time (Bpod file `Shamir01_Dual2AFC_nat_Sep11_2026_Session1.mat` only became
  available on this machine today -- it had to be transferred in). Two new
  things worth keeping for the next session:
  - **`sync_pipeline.units_spikes.build_units_table()` hard-requires a real
    `cluster_info.tsv`** in the kilosort4 folder (columns `cluster_id`, `ch`,
    `depth`, `group`, `KSLabel`) and raises `FileNotFoundError` without one --
    this file only exists if Phy was actually launched on that folder at
    some point (Phy writes it). A session with no Phy pass yet (like
    `20260911_100049` right now) has no such file, only the raw
    `cluster_group.tsv`/`cluster_KSLabel.tsv` (both just plain KS auto-labels,
    not curation). **New script `build_cluster_info_from_classification_session.py`**
    fabricates a stand-in `cluster_info.tsv` directly from our own
    `corrected_classification_<session>_with_waveforms.json` (`ch` =
    peak_channel, `depth` = that channel's y-coordinate in µm from
    `channel_positions.npy`, `group` = our GOOD/MUA/NOISE label lowercased
    with the `+NON-SOMA` suffix dropped). **This is explicitly a
    pre-curation stand-in, not curation** -- when a session later goes
    through Phy, re-run the sync pipeline against the real post-Phy
    `cluster_info.tsv` the same way `20260901_085606` was (stage 2), and
    treat any synced/ folder built with the fabricated version as
    preliminary only (see `session_registry.json`'s
    `synced_data_status` for that session).
  - Re-ran the reward-vs-port-leaving responsiveness analysis (the one that
    found 6 leading units for `20260901_085606`) on `20260911_100049`.
    **The original script was never found/saved** (same situation as
    `burst_curve` earlier) -- reconstructed fresh as
    `scripts/reward_vs_leaving_responsiveness_session.py`, same documented
    methodology (Mann-Whitney per good unit, 0.5s post-event window,
    rewarded-trials-aligned-to-water-delivery vs
    non-rewarded-completed-trials-aligned-to-last-poke-Out-on-the-chosen-port,
    trial-consistency filters [>=20 spikes, >=15 responsive trials, top-2-
    trial-share <=40%], BH-FDR across units). **Result: null** -- 165/183
    good units tested, 0 FDR-significant. See `session_registry.json`'s
    `reward_vs_leaving_analysis` block for the full result and the important
    caveat: this session's "good" units are automatic-classifier-only (no
    Phy/manual check yet, per the point above), so a null result here isn't
    directly comparable to the other session's Phy-curated positive result --
    can't yet tell how much of the difference (if any) is real
    session-to-session variability vs. noisier pre-curation unit quality.
- **2026-09-15/16 (multi-day)**: Copied two new raw sessions
  (`20260915_084812.rec`, 193.8GB; `20260915_103710.rec`, 167.1GB) from their
  original extraction location `F:\Gil\Shamir\` to the rat's shared backup
  root `Z:\Gil\Shamir_1\` via `robocopy /E /Z /MT:16 /R:3 /W:5`, at the
  user's request. Both verified 0 failed / 0 mismatch in robocopy's own
  summary block. **Gotcha**: invoking `robocopy` (or any native Windows exe)
  from the Bash tool with flags like `/E`, `/Z` unquoted gets silently
  mangled by Git Bash/MSYS's automatic POSIX-path conversion (it rewrites
  `/E` into something like a drive path), producing an immediate
  `ERROR : Invalid Parameter` with no real copy attempted. **Fix: prefix the
  command with `MSYS_NO_PATHCONV=1`** (or quote every flag) when calling a
  native Windows exe with single-letter `/flag`-style options from Bash on
  this machine. Neither of these two sessions has been kilosorted yet
  (`.kilosort` subfolder exists but is empty/minimal) -- Stage 1 hasn't
  started for either.
  - Also hit: after editing `build_review_html.py`'s JS (spike-marking,
    legend, show-more button), the artifact was republished from the STALE
    locally-built HTML file (from an earlier `build_review_html.py` run,
    before the JS edits) -- a script edit does not retroactively change an
    already-built HTML file on disk. Caught only because a routine "is the
    background job actually done" check happened to re-verify the published
    file's contents. **Whenever `build_review_html.py` itself is edited,
    always rerun it before the next Artifact publish** -- don't assume the
    HTML file on disk reflects the current script just because it was built
    "recently" in the same session. A quick sanity check: grep the built HTML
    for a string unique to the just-added JS before publishing.
- **2026-09-16**: User asked "are you sure you have the correct time point
  from bpod?" about the reward-vs-leaving plots -- a real, well-founded
  challenge, not a false alarm. Investigation found:
  - The **reward-delivery time point (`water_L`/`water_R` state start) was
    already correct** -- confirmed against `TASK_TIMELINE.md`'s documented
    state sequence: that state's start time IS exactly when the valve opens.
  - The **"port-leaving" time point had a real bug**:
    `reward_vs_leaving_responsiveness_session.py` took the unbounded max of
    ALL poke-Out events anywhere in the trial's event window as "when the
    rat left" -- but per `HOW_TO_RUN.md` sec 7, a trial's window runs a fixed
    10s past its last state and commonly overlaps the *next* trial entirely,
    so this could (and did, for real trials) pick up a spurious later poke
    from the ITI or the next trial instead of the actual departure. Confirmed
    concretely on trial 18: true departure at t=4.2522s (right before
    `skipped_feedback` fires at t=4.3341s) vs. the buggy unbounded search
    returning t=5.093s. **Fix**: bound the Out-event search to before that
    trial's own `ITI` state begins (general -- not hardcoded to
    `skipped_feedback`, so it stays correct on a session where
    `timeOut_IncorrectChoice` also fires, unlike this one).
  - Along the way, also discovered "non-rewarded completed" trials are NOT
    all wrong-side choices: some are a CORRECT-side choice
    (`rewarded_Lin`/`Rin` state) where the rat left too early and forfeited
    the reward before delivery. Both converge on the same `ITI` state
    afterward, which is exactly why bounding by `ITI` (rather than by state
    name) is the right general fix.
  - Re-ran the corrected analysis: **conclusion unchanged (still 0
    FDR-significant)**, confirming the null result itself was real, not an
    artifact of the timing bug -- but individual units' rankings/values did
    shift, so the published figure/table were regenerated and republished.
  - **Open question, not yet resolved**: the original `20260901_085606`
    reward-vs-leaving analysis (the one that found 6 leading units) was never
    saved as a script (see 1f-pre and the 2026-09-15 entry above) -- it's
    unknown whether it had the same unbounded-last-Out-event bug. If that
    session's synced data and trial tables are ever revisited, re-derive its
    port-leaving times the same bounded way and confirm the 6-unit result
    still holds before relying on it further.
- **2026-09-16**: New analysis -- does any neuron respond to the choice
  period (left, right, or both), right after the cue ends? First pass
  (`left_right_choice_period_responsiveness_session.py`) only tested whether
  firing DIFFERED between left- and right-choice trials in that window --
  user correctly pointed out this would miss a neuron that responds equally
  strongly on both sides (no left-vs-right difference, but a real "responds
  to choice" signal either way). **Superseded** by
  `left_right_cue_response_session.py`: a genuine pre-cue baseline
  (`stay_Cin` state, median 0.296s) is compared to the post-cue/pre-poke
  response window (`wait_Sin` state, cue-end to poke+0.2s, median 0.466s) via
  a PAIRED Wilcoxon signed-rank test, run SEPARATELY for left-choice trials
  and right-choice trials (two independent FDR corrections, one per side) --
  a unit can then be called 'left', 'right', 'both', or neither, which is
  what was actually asked for. Also added the explicit user-specified filter:
  only trials with decision time (cue-end to poke) <3s (a no-op for this
  session -- observed max is 1.11s -- but a real, applied parameter, not
  assumed away). Restricted to TrialCompleted==1 throughout (excludes
  FixBroke/EarlyWithdrawal by construction, per user's clarification that
  this is exactly what "completed trials" means here).
  - Result: still null after FDR (0/430 left-family, 0/433 right-family), but
    notably closer than the previous three analyses -- units 135 and 118
    (both RIGHT family) tied at p_fdr=0.116, and their PSTHs
    (`outputs/left_right_cue_response_top6_20260911_100049.png`) show a
    visible ramp right after cue-end that the earlier analyses' top
    candidates didn't show as clearly. Still not significant, but flagged in
    the published results page as the closest call so far -- worth
    revisiting with more trials or once this session has real Phy curation.
  - Same pre-Phy caveat applies as the other analyses on this page.
- **2026-09-16**: New session `20260916_110311` -- user recorded it
  specifically as a ground-truth test case (knows a few channels have real
  responses by fact) to diagnose why the pipeline/analyses keep coming back
  null. Process notes worth keeping for next time:
  - **Detecting Trodes extraction completion without watching it live**:
    the raw `.rec` file itself is fully written well before extraction
    finishes (its mtime stops advancing early) -- the actual "still working"
    signal is the `.kilosort\<session>.probe1.dat` file's size still growing,
    plus the separate `trodesexport` process (distinct from the main
    `Trodes` GUI process) still running. Polled `probe1.dat`'s size every
    30s in a background loop, declared done after 4 consecutive unchanged
    reads (~2 min stable); cross-checked that `trodesexport` had actually
    exited (`Get-Process -Name trodesexport`) before trusting it. This let
    the user step away from the computer during the last ~15 min of a
    multi-hour extraction.
  - Closed the main `Trodes` GUI process (`Stop-Process`) immediately after
    confirming extraction was done, per user request, to free RAM before
    launching Kilosort4 -- **left `trodesexport` alone** (it exits on its
    own; killing it mid-extraction would have corrupted the output, hence
    checking it had already exited first, not killing it).
  - Kilosort4 launched with **non-default thresholds Th_universal=9.0,
    Th_learned=8.0** (explicit user request, differs from the 9.2/7.0 used
    for `20260901_085606`/`20260910_095200`) -- deliberately testing whether
    threshold choice is part of why known-responsive channels aren't showing
    up as responsive in the analysis. Script: `scripts/run_kilosort4_20260916.py`
    (same pattern as `run_kilosort4_20260910.py`, just the two threshold
    values and paths changed). Probe map built via `make_probe_map.py`, run
    from inside the session's own `.kilosort` directory (it operates on
    `os.getcwd()`, not an argument -- easy to get wrong by running it from
    the wrong directory).
  - Also synced the STALE `Z:\Gil\Shamir_1\20260911_100049.rec` backup (only
    489GB, predating all of this session's kilosort4/pipeline work which
    brought the F: copy to 596GB) via the same `MSYS_NO_PATHCONV=1 robocopy
    /E` pattern from the 2026-09-15 entry -- a plain `/E` copy only adds/
    updates files, never deletes, so it's safe to run without confirmation;
    freeing the F: copy afterward to make room for this new session is a
    separate, deliberately NOT-yet-taken step -- confirm with the user before
    deleting anything on F:.
  - Full pipeline (Stage 1 through review tool) still to run once Kilosort4
    finishes -- see `session_registry.json`'s `20260916_110311` entry for
    live status.
  - **Independent (non-Kilosort) single-channel clustering, for validating
    Kilosort against known-responsive channels**: new script
    `independent_channel_clustering_session.py` -- threshold detection (4.5x
    MAD) + PCA + BIC-selected GMM on one raw channel's full-session waveform
    snippets, built specifically to cross-check whether a channel the user
    knows has real responses actually shows up as a distinguishable cluster
    independent of Kilosort's own machinery. Given a Trodes ntrode id (e.g.
    "channel 1337"), the raw channel index it corresponds to is NOT the same
    number and must be looked up: parse the `.rec` file's own embedded
    `<SpikeNTrode id=... ><SpikeChannel coord_ap=... coord_dv=... coord_ml=...
    hwChan=...>` header (first ~5MB, same trick as the gain_to_uV extraction),
    build a `(ml,dv) -> trodes_id` map, and join it against the channel map
    JSON's own `xc`/`yc` arrays (same coordinates, different source) to get
    `trodes_id -> raw_channel_index`. **Always verify this join has zero
    duplicate coordinate keys on both sides before trusting a single result**
    (user explicitly asked for this check, and it matters -- two electrodes
    sharing a coordinate would silently produce a wrong or ambiguous match).
    For `20260916_110311`: verified clean 384/384 unique match, Trodes 1337
    -> raw channel index 129 (hwChan=28), confirmed unambiguous both
    directions.
  - **INCIDENT: running the independent clustering script concurrently with
    Kilosort4 caused a real out-of-memory event that killed BOTH background
    jobs (plus an unrelated small grep job) -- lost ~2.5 hours of Kilosort4
    progress**, which had to restart from scratch (Kilosort4 has no
    incremental checkpoint for the spike-detection stage). **Root cause**:
    `RawReader.read_window()` in `striatal_agent/raw_io.py` does
    `self._arr[start:end, :][:, chans]` -- because `probe1.dat` is stored
    row-major (all 384 channels interleaved per sample), extracting even a
    SINGLE channel still requires the OS to page in every byte of every row
    in that range (a row is smaller than a disk page, so there's no way to
    skip the other 383 channels' bytes). The clustering script scanned the
    entire ~270GB file this way while Kilosort4 was simultaneously streaming
    through its own ~270GB file -- combined, enough real memory/page-cache
    pressure to trigger a genuine system-level low-memory kill, not a bug in
    either script individually. **Rule going forward: never run a
    full-session single-channel raw-data scan concurrently with Kilosort4 (or
    any other large streaming job) on this machine** -- either wait for the
    other job to finish, or restrict the scan to a short time window (a few
    minutes, not hours) if it must run concurrently. This is a property of
    `RawReader`'s row-major access pattern, not something a smarter Python
    script alone can fix -- true single-channel-only disk I/O would require
    the data to be stored channel-major (transposed) or re-chunked, which
    `probe1.dat`'s format doesn't support.
  - **Explicit user instruction, easy to forget across a long overnight
    Kilosort4 run: once Kilosort4 finishes for `20260916_110311`, run the
    ENTIRE pipeline through to a published review tool -- do not stop at just
    Kilosort4 finishing.** This is the same proven Stage 1 chain used for
    `20260911_100049` (see section 1f/1f-pre above), applied in full:
    `run_bombcell_session.py` -> `corrected_classification_session.py` ->
    `build_pipeline_review_data.py` -> `build_trodes_channel_lookup_session.py`
    -> `raw_cluster_points_session.py` -> `raw_separation_metric_session.py`
    -> `build_burst_isi_metrics_session.py` ->
    `compute_raw_trace_snippets_session.py` (N_SPIKES=16 + per-window spike
    marking, the current standard, not the superseded N_SPIKES=8 version) ->
    `merge_review_data_session.py` (**must include the `trodes_channel`
    merge** -- that step was accidentally left out and had to be bug-fixed
    for `20260911_100049`, see the 2026-09-16 trodes_channel entry above,
    don't repeat the omission) -> `build_review_html.py` -> publish as a new
    Artifact (`capabilities:{db:{}}`) -> push `raw_traces` to that artifact's
    db (batch size depends on N_SPIKES=16's real per-unit file size, was
    8/batch last time at ~93KB/unit average -- recompute, don't assume).
    gain_to_uV for this session is already confirmed (0.018311105685598315,
    same as `20260911_100049`). The full checklist with all the specific
    gotchas is also written into `session_registry.json`'s
    `20260916_110311.stage1_pipeline_status` field -- keep both in sync if
    either changes. **Do not run any of the raw-data-scanning steps
    (cluster_points, separation_metric, trace_snippets) concurrently with
    anything else heavy** -- see the OOM incident entry above.
- **2026-09-17: the OOM diagnosis above was INCOMPLETE.** After Kilosort4
  finished, `independent_channel_clustering_session.py` was run again --
  ALONE, nothing else running -- and got OOM-killed a second time. This rules
  out "concurrency with Kilosort4" as the actual root cause; the script had a
  standalone problem. **Real cause**: the script used
  `striatal_agent.raw_io.RawReader`, which `np.memmap()`s the ENTIRE
  `probe1.dat` (hundreds of GB) ONCE and keeps that single mapping alive for
  the reader's whole lifetime; every `read_window()` call is just a slice
  into that one persistent mapping. Sequentially scanning a file far larger
  than physical RAM (here ~270GB against 64GB) through one persistent memmap
  lets Windows' page cache (standby list) grow to consume nearly all
  physical memory by the end of the scan -- technically reclaimable by
  Windows itself under real pressure, but apparently enough to trip
  whatever low-memory guard actually kills these background jobs (not a
  true Windows OOM in the classic unreclaimable-memory sense).
  **Fix, applied to `independent_channel_clustering_session.py`**: replaced
  the persistent-memmap `RawReader` with plain buffered file I/O
  (`open()`+`seek()`+`read()`) inside the chunk loop itself -- each chunk's
  bytes are read into an ordinary buffer, the target channel's column
  extracted, and the buffer explicitly `del`eted (+ periodic `gc.collect()`)
  before the next chunk, so no single long-lived object ever accumulates the
  whole file's cache footprint. Chunk size also cut 300s -> 60s. Verified
  correct against the old memmap-based reader's output for a short window
  (same values, same shape) before committing to the full-session run.
  **This fix is specific to the diagnostic clustering script** -- it does
  NOT change `striatal_agent/raw_io.py`'s `RawReader` itself, which is used
  throughout the rest of the pipeline (`raw_cluster_points_session.py`,
  `raw_separation_metric_session.py`, `compute_raw_trace_snippets_session.py`)
  successfully, because those scripts only ever read scattered small windows
  (a handful of 120s chunks per unit) rather than sequentially scanning the
  ENTIRE file end-to-end the way this one-channel full-session scan does --
  the persistent-memmap pattern is fine for sparse/bounded access, it's
  specifically an exhaustive sequential full-file scan that's risky. **If any
  future script needs to sequentially scan a whole multi-hundred-GB
  `probe1.dat` start-to-finish, use this same buffered-read-per-chunk
  pattern, not `RawReader`.**
- **2026-09-17: real gap found in the Stage 1 checklist itself** -- it
  jumped straight from "Kilosort4 finished" to `run_bombcell_session.py`,
  skipping the actual `striatal_agent.run_pipeline()` batch run (the real
  agent pipeline: per-unit structural/violation/footprint scoring, spike
  recovery, and building `kilosort_original/` + `agent_processed/`) that
  bombcell and everything downstream actually depend on. Caught immediately
  by bombcell's own `FileNotFoundError` on
  `kilosort_original/spike_templates.npy` (that folder didn't exist).
  **Worse compounding problem**: the batch-orchestration scripts that made
  this run work for `20260911_100049` (`plan_batches_v2.py`, `run_batch.py`,
  `run_all_batches.sh`, `consolidate_batches.py`) were written during that
  earlier conversation and never saved to `scripts/` -- exactly the mistake
  this file's own "Lesson" notes (1f-pre) warn against, and it happened
  again anyway. Rebuilt and this time actually saved:
  `run_pipeline_batch_session.py` (runs one batch as a fresh process --
  still required, the pipeline reliably crashes around unit ~20-24
  regardless of requested count or free RAM, see 1e) and
  `consolidate_batches_session.py`. Two things worth knowing about how
  `run_pipeline()` behaves across repeated calls, confirmed by reading
  `output_writer.py`: `build_kilosort_original()`/`init_agent_processed()`
  are idempotent (`overwrite=False` checks if already built, safe to call
  once per batch), and `agent_processed/spike_times.npy`/`spike_clusters.npy`
  are genuinely cumulative across calls (`append_recovered_spikes()` reads
  the current file and appends, doesn't overwrite) -- but
  `agent_processed/agent_report.json` and every `cluster_<label>.tsv` ARE
  fully overwritten by each batch, containing only that batch's own units,
  which is why each batch script archives its own output to
  `agent_processed/batch_reports/` immediately, and a separate consolidation
  pass merges everything at the end. **Going forward: the correct Stage 1
  order is run_pipeline batches -> consolidate -> bombcell -> everything
  else in 1f/1f-pre** -- update that section's own text to say this
  explicitly rather than relying on this change-log entry alone.

- **2026-10-15**: Added the review tool's first per-unit *time-selection*
  feature, plus the data step behind it (new `scripts/build_session_timeline_data.py`,
  new step 3b in 1f, new row in 1f-pre). The panel draws that unit's firing
  rate across the whole session with a trial-start rug underneath, and Gil
  marks the stretches worth keeping -- click start / click end, or drag,
  repeatable for several disjoint ranges. Saved to the artifact db
  (collection `trial_selections`, doc id = unit_id) alongside the verdicts.
  **The contract the not-yet-written sync-pipeline consumer must honour**, and
  the reason it is written onto every saved doc rather than kept here: ranges
  are in ABSOLUTE TRODES SECONDS; an empty `ranges` list (or no doc at all)
  means KEEP EVERY TRIAL for that unit; a trial counts as inside a range iff
  its *start* falls inside it (`inclusion_rule`); and the pipeline must ADD a
  per-(unit, trial) selected/unselected flag, never drop a trial row -- every
  trial always stays in the output. Nothing in `D:\Gil\sync_pipeline` was
  touched (deliberately, per Gil: that code stays isolated); the only contact
  with it is a read-only `import dio_decoder` for TTL decoding.
  **The finding worth keeping**: trial boundaries ARE available before sync.
  The DIO TTL code is the Bpod state number, so code 1 is each trial's first
  state; checked against `20260916_110311`'s own synced `trials.parquet`,
  the 954 code-1 markers match `trial_start_trodes` to 0.0 ms median AND
  0.0 ms max over all 953 trials. (Code 0 also fires once per trial and is
  the obvious-looking candidate, but it is the inter-trial reset: a variable
  0.4-4s early, median -2147 ms, max |13991| ms. The first code-0 event is
  not an event at all, just the zero-order-hold reconstruction starting with
  all lines low.) There are normally a few more markers than saved Bpod
  trials (954 vs 953 -- the last trial starts on the rig but never lands in
  the Bpod file), so these are trial-start MARKERS, not trial_ids; mapping
  marker -> trial_id stays the sync pipeline's job, where it is exact.
  Two incidental fixes while in there: the page had no `<meta charset>`, so
  every em-dash and middot in it rendered as mojibake when opened as a local
  file; and the left header was hardcoded to the first session's "401 units"
  (and "0 / 401 reviewed") for every session built since. **Size watch**: the
  built HTML is now 14.3 MiB of the Artifact platform's 16 MiB page limit --
  812 KB of that is the new timeline. Gil has said more panels are coming, so
  the next one that adds real bulk should go into a published supporting file
  or the artifact db (like `raw_traces` already does) rather than be embedded.
