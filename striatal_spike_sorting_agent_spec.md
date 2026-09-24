# Striatal Neuropixels Post-Kilosort Curation Agent — Design Spec (v2)

## 0. Purpose and scope

**Goal:** an AI agent that performs as much of the post-Kilosort, pre-phy curation job
itself as possible — with real decision-making authority (accept / reject / merge / split
/ recover), not just flagging for human review. The agent has access to **both** Kilosort's
output and the **original raw data**, and uses the raw data directly for validation and
recovery, not just KS's summary.

**This is NOT a replacement for Kilosort.** Kilosort remains the primary sorter — detection,
clustering, drift correction. This agent is a validation/curation layer operating on top of
KS output, informed by direct raw-data inspection.

**This is NOT primarily a burst-recovery tool.** Burst/decrement handling is a
**context-aware sub-module**: because the agent is aware it's working on striatal data, it
profiles each unit's firing pattern and applies burst-appropriate logic *when the data
calls for it*. It is one possible outcome of good general curation, not the reason the
pipeline exists. The pipeline must be useful and complete for general curation even if
burst-related sorting errors turn out to be minor.

**Decisions have real effect.** Where evidence is strong and structural (not
hypothesis-dependent), the agent writes directly to phy-compatible outputs
(`cluster_group.tsv`, spike train edits). Where evidence is ambiguous, decisions are scored
and logged for review, not silently made — see Section 6 (decision authority tiers).

**Core anti-circularity principle (unchanged):** all inclusion/exclusion decisions rest on
*structural* evidence (spatial footprint, waveform shape, refractory consistency) — never
on the burst/decrement hypothesis. Decrement is measured only after the fact, as a
downstream, purely observational analysis.

---

## 1. Validation gate — run before investing in recovery machinery

Per ChatGPT's critique (worth taking seriously): before building/trusting the recovery
stage, get quantitative evidence it's needed, on a handful of clean, high-SNR, well-isolated
units:

- Compare threshold-crossing counts vs. KS-detection counts specifically *during* bursts
- Compare template-match scores of candidate "missed" spikes against accepted spikes
- Quantify what fraction of spikes recovery would actually add

**Caveat on prior null results:** an earlier check (KS-output-only, low-SNR dataset) found
no amplitude-vs-ISI relationship. This is weak evidence against the hypothesis, for two
reasons: (1) low SNR could mask a real but modest effect, and (2) if KS is dropping the
smallest late-burst spikes, the surviving KS-confirmed sample is biased *against* showing
decrement by construction. Needs re-testing on cleaner data before drawing conclusions
either way.

This gate determines how much investment Stage 6/7-equivalent work deserves — not whether
the rest of the agent is worth building.

---

## 2. Core curation loop (general-purpose, applies to every unit)

### 2.1 Preprocessing
Bandpass/high-pass (>300 Hz), common median reference, drift correction (SpikeInterface).
Filtered raw traces retained on disk — the agent needs direct access to these throughout,
not just at one stage.

### 2.2 Kilosort4 first pass
Tuned toward over-splitting rather than over-merging — over-split units are recoverable via
merge; incorrect merges are hard to undo.

### 2.3 Per-unit structural scoring (soft score, not pass/fail)
Combines multiple structural signals into one isolation-quality score per unit:
- **Spatial footprint consistency** — max-channel stability, tolerant of adjacent-channel
  flips consistent with electrode geometry; penalized (not hard-failed) for jumps between
  distant channels
- **Waveform shape similarity** across the unit's own spikes
- **CCG evidence** — refractory-period cleanliness in the autocorrelogram; cross-correlogram
  structure with spatially neighboring units
- **Refractory contamination rate**

Real recordings show drift and amplitude fluctuation that can superficially resemble a
channel swap without indicating a merge problem — hence a combined soft score, not a single
hard filter (this revises v1 Stage 3, which was too binary).

### 2.4 Firing-pattern profiling
Computed on KS-confirmed spikes. Produces a continuous profile vector per unit (rate,
CV, LV, burst_fraction, pause_fraction, etc. — full math in Section 4). This is what
determines whether the burst/pause sub-module (Section 5) activates for a given unit, and
with what parameters. No forced binary label.

### 2.5 Collision / contamination check (three-way, not binary)
For any spike, check whether the observed waveform is fully explained by the assigned
unit's known footprint, or whether residual signal is better explained by:
- **(a) Clean** — no unexplained residual
- **(b) Collision with a known unit** — residual matches another already-identified unit's
  footprint
- **(c) Unexplained residual** — signal present but not attributable to any known unit
  (likely an undetected low-amplitude neuron — common in dense striatal recordings; do not
  misattribute this to a specific known unit, and do not treat it as clean)

This three-way output (revised from v1's binary clean/contaminated) is a more honest
reflection of the fact that not all interfering sources are already in KS's output.

### 2.6 Footprint stability module — standalone, always-on
**Promoted from a Stage-7-only check (v1) to a general QC module run on every unit**,
independent of whether burst-recovery is used at all. Normalized amplitude pattern across
channels should stay stable in *shape* even if absolute amplitude varies (drift, decrement,
noise). Shape instability (not just scale change) is direct evidence against single-source
identity — one of the most information-dense, cheapest-to-compute metrics in the whole
pipeline, and useful on its own regardless of firing pattern.

### 2.7 Raw-data recovery search (candidate spikes, all profiles)

**Primary method — KS-native threshold sweep.** Kilosort4 detects spikes via template
deconvolution against learned, unit-specific templates, gated by `Th_learned` (default ~8;
`Th_universal`, default ~9, gates the earlier universal-template detection stage). Rather
than building an external matched filter, rerun deconvolution for a unit's own learned
template with `Th_learned` swept downward from KS's operating point. This reuses KS's own
already-optimal, multi-channel, temporally-aligned template — the recovered/rejected
distinction is then attributable to exactly one variable (match score vs. threshold),
rather than to differences between KS's algorithm and an external one.

**Required control (your correlogram idea):** at each threshold step, recompute the unit's
autocorrelogram and track the refractory-violation rate as a function of threshold. Use the
inflection point — where violations start climbing — as the principled stopping point, not
an arbitrary fixed value. Rerun this control any time the threshold is adjusted, not once.

**Secondary method — ISI-indexed matched pass (bursty units only, fallback/supplement).**
KS's learned template is a single global-average shape, so the sweep above is well-suited
to tonic/FSI-type units but may still miss genuinely real, strongly-decremented late-burst
spikes for units profiled as burst-capable (Section 4) — those deviate further from the
average template even though they're real. For those units only, run a second pass using
the position/ISI-indexed expected shape (Section 4) against whatever the KS-native sweep
still rejected. This is a targeted supplement for a known blind spot, not a parallel
general-purpose system.

Inclusion criteria (structural only, applies to all units regardless of profile/method):
- Footprint match (2.3)
- Shape match against the profile-appropriate expected shape (2.4 — see Section 5 for
  burst-indexed case)
- No refractory-period violation introduced (per the ACG control above)
- Not a better structural match to a neighboring unit

### 2.8 Merge/split scoring
Combine structural score (2.3) + CCG evidence + amplitude relationship into a merge
confidence score. High-confidence, low-ambiguity merges may be auto-applied (see Section 6
for the authority threshold); ambiguous cases are scored and logged, not silently resolved.

### 2.9 Output
Writes to a **separate, non-destructive output directory** (Section 9) for high-confidence
decisions and everything else. Produces a per-unit report (profile, footprint plot,
collision summary, recovery spike count/confidence, merge/split scores) plus the audit/
summary module (Section 9.3) as required outputs, not optional extras.

---

## 3. Section 4 (from v1) — firing-pattern profiling math

*(unchanged from v1; reproduced here for completeness)*

**Firing rate:** mean over session, plus sliding-window (30–60s) stability check.

**Regularity:** CV = std(ISI)/mean(ISI) (Poisson reference = 1); LV (Shinomoto et al. 2003)
as primary metric, more robust to slow rate drift.

**Burst detection — Poisson Surprise:**
$$P(N=k \mid \lambda, T) = \frac{(\lambda T)^k e^{-\lambda T}}{k!} \qquad S = -\log_{10} P(N \geq k \mid \lambda, T)$$
Seed on short-ISI runs → grow while $S$ increases → trim edges → record burst (start, end,
count, $S$). $S$ ranks confidence; not a binary call.

**Pause detection:** ISI > 3–5× local median ISI (windowed). Report duration, rebound rate.

**Why not raw ISI variance:** rate-dependent, can't localize bursts, dominated by outlier
gaps. CV/LV = cheap screen; Poisson Surprise = actual localization.

---

## 4. Burst/pause sub-module (context-aware, triggered — not the pipeline's core purpose)

Activates per-unit based on Section 2.4 profile output. This section captures the
striatum-specific extensions layered on top of the general curation loop:

**Preceding-ISI, not rank-in-burst, as the primary predictor** (revised from v1). Physiology
(channel recovery, membrane state) is a time process, not a counting process — preceding-ISI
or cumulative time-since-burst-onset is more mechanistically grounded than literal spike
rank. Use rank as a secondary/diagnostic variable only, not the primary axis.

**Position/ISI-indexed expected shape:** for units profiled as burst-capable, the "expected
waveform" is a trajectory indexed by preceding-ISI, not a single fixed template — built from
clean (2.5a), already-validated spikes. Candidate spikes are compared against the expected
shape *at their position*, avoiding false rejection of genuine late-burst spikes.

**Class-appropriate handling:**
- Bursty (MSN-like): ISI-indexed shape matching, decrement expected and accounted for
- Tonic/FSI-like: tight fixed-template match; unusual deviation is treated with *more*
  suspicion, not less
- TAN-like, near a pause: conservative — recovered spikes inside a pause are more likely
  noise than a real miss

**Downstream measurement (not an inclusion criterion — anti-circularity, unchanged):**
amplitude/width vs. preceding-ISI, fit linear vs. saturating-exponential (AIC/BIC), only on
clean, validated spikes, only for units already classified as burst-capable. This is a
*result* the pipeline can produce for burst-relevant units, not a mechanism the pipeline
depends on.

---

## 5. Known limitations (carried over, still open)

- Two genuinely distinct units with very similar spatial footprints are not fully
  separable by footprint/amplitude reasoning alone — reduced, not eliminated, by the
  combined structural score.
- Collision detection can only identify interference from *already-known* units;
  unexplained residual (2.5c) from undetected neurons remains a real, acknowledged gap in
  dense striatal recordings.

---

## 6. Decision authority tiers

To make "real effect" concrete and safe:

- **Tier 1 — auto-applied:** decisions where structural score is high and unambiguous
  (e.g. clean recovery spike passing all 2.7 criteria with no ACG degradation; obvious
  merge with strong CCG + footprint + amplitude agreement). Written directly to output.
- **Tier 2 — scored and logged, not auto-applied:** ambiguous merges/splits, recovery
  candidates near threshold, unexplained-residual cases. Appears in the report with a
  confidence score, ranked for efficient human review rather than blind scrolling in phy.
- **Tier 3 — flagged as insufficient evidence:** low spike count units, units failing
  profiling stability checks. Explicitly marked, not silently defaulted into any category.

Thresholds between tiers are part of Section 7's calibration set, tuned against a
visually-verified reference set before running at scale.

---

## 7. Parameters requiring calibration against real data

- Burst seed-ISI cutoff (prior ~10–15ms for MSNs; validate per-unit against log-ISI dip)
- Poisson Surprise detection threshold
- Pause threshold multiplier (prior 3–5× local median ISI)
- Collision window duration / footprint-overlap threshold
- Structural score weights (2.3) and Tier 1/2/3 cutoffs (Section 6)
- Minimum spike count for reliable profiling

**Calibration approach:** ~5–10 visually clear, multi-channel-visible units as a
ground-truth-by-eye reference set; tune thresholds so automated Tier 1 calls match visual
read before trusting auto-apply at scale.

---

## 8. Numerical vs. AI-visual vs. human roles

- **Primary substrate:** raw numerical vectors — exact, reproducible, scales without
  degradation. All scoring/decision logic operates here.
- **AI visual inspection:** secondary/confirmatory — spot-checks on numerically ambiguous
  cases, or fast triage across channels/segments. Not the primary decision mechanism.
- **Human review:** focused on Tier 2/3 items only (Section 6) — the goal is that most
  Tier 1 volume never needs manual phy review at all.

---

## 9. Non-destructive output architecture & phy integration

**Principle: the agent never mutates Kilosort's original output.** Two folders, both
independently openable in phy:

```
/kilosort_original/      ← untouched KS4 output, read-only, kept as permanent baseline
/agent_processed/        ← all agent decisions and analysis, phy-loadable, fully separate
```

This means you can always open the original in phy exactly as KS produced it, or open the
agent-processed version — never a forced, irreversible edit. `/kilosort_original` is the
ground truth to diff against, forever.

### 9.1 What lives in `/agent_processed`
Standard phy-required files (`spike_times.npy`, `spike_clusters.npy`, `templates.npy`,
`cluster_group.tsv`, etc.), reflecting the agent's Tier 1 decisions (Section 6) — plus:

**Extra per-cluster score columns, visible natively in phy.** Phy auto-detects any file
named `cluster_<label>.tsv` (columns: `cluster_id`, `<label>`) and renders it as a
sortable/filterable column in the cluster view — no custom phy plugin needed. Use this
directly for side-by-side comparison:
- `cluster_ks_score.tsv` — KS's own native detection/match score for the unit
- `cluster_agent_structural_score.tsv` — Section 2.3 combined score
- `cluster_pct_spikes_added.tsv`, `cluster_n_recovered.tsv`
- `cluster_acg_violation_pre.tsv`, `cluster_acg_violation_post.tsv` — before/after recovery
- `cluster_footprint_stability.tsv`
- `cluster_collision_summary.tsv` — clean / known-collision / unexplained-residual counts
- `cluster_profile.tsv` — rate, LV, burst_fraction, pause_fraction (Section 2.4)
- `cluster_tier.tsv` — Section 6 decision tier

This means opening `/agent_processed` in phy shows KS's original numbers *and* the agent's
numbers, side by side, for every unit — directly addressing the "I want to see both scores"
requirement, using phy's existing extensibility rather than a custom viewer.

### 9.2 Representing recovered spikes as a visible, reversible split
Rather than silently merging recovered spikes into a unit's existing cluster ID, write them
as a **separate child cluster** at Tier 1 (e.g. unit `15` → `15` for original KS spikes,
`1015` for agent-added spikes). Phy's native similarity/merge view then lets you visually
compare the two clusters' waveforms and, if you agree, merge them yourself with one click —
or leave them split if something looks off. This makes every addition inspectable and
reversible by construction, not just logged in a report you might not read.

### 9.3 Audit / summary module (required output, not optional)

**Per-unit audit record:**
- KS original spike count
- Candidates considered (KS-native sweep +, for bursty units, ISI-indexed pass), split
  into accepted / rejected / borderline
- Spikes added, with confidence score distribution (not just a count)
- ACG violation rate, before vs. after
- Footprint stability score, before vs. after
- Collision-tag breakdown of final spike train
- Decision tier breakdown

**Session-level summary:**
- Distribution across units of % spikes added (histogram) — shows immediately whether
  recovery is doing something broadly, narrowly, or effectively nothing
- Fraction of units with zero additions; fraction where ACG violations *increased*
  post-addition (an explicit red-flag list)
- Tier distribution across the whole session — how much the agent decided autonomously vs.
  deferred
- For bursty units: agreement rate between the KS-native sweep and the ISI-indexed pass —
  high agreement is reassuring; large disagreement is the evidence that the burst-specific
  module is earning its added complexity
- Calibration-set performance (Section 7) reported **separately** from the full-session
  stats — the closest available proxy for real precision, since there's no ground truth
  elsewhere
- **One headline sentence**, not just tables — e.g. "recovery added <2% of spikes across
  85% of units with no ACG degradation; burst module contributed materially in 4 of 12
  burst-classified units" — so a null or weak result is visible as a finding, not buried.
