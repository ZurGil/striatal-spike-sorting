# Research log — post-Kilosort correction module

Running, human-readable record of what was planned, what was built, what was
tested against real data, what worked, what broke, and what's still open.
Update this file every time a real step is taken or a real conclusion is
reached — this is the project's memory, not a summary written after the fact.

---

## 0. Where this started

The starting point was a pure design document (no code): `striatal_spike_sorting_handoff.md`.
It proposed three "pillars" for describing a spike better than Kilosort's rigid
template match — (1a) a matched filter allowing amplitude/timing/stretch as
legitimate variation of one neuron, (1b) complex wavelet phase for precise,
shift-tolerant timing, (1c) spatial energy footprint across channels — plus a
plan for folding this into Kilosort as an external, diagnostic-first module.
The rat is "Shamir," striatal Neuropixels recordings, Kilosort4 + Phy curation.
Everything below happened after that document, inside `post_ks_correction/`.

---

## 1. Pillar 1a — the amplitude/timing/stretch model, and a real problem we found

**Built:** `nuisance_model.py`. The model is `m(t; a, tau, beta) = a * s0((t-tau)/(1+beta))`
— one neuron's spike, allowed to be scaled (a), shifted in time (tau), and
stretched (beta, which happens during bursts). Linearized into a 3-term
regression against the template itself, its time-derivative (carries tau),
and a stretch-direction vector `q` (carries beta) — fast, closed-form, stable
even at low SNR.

**What went wrong, found on real data:** when this 3-parameter fit was run on
real burst spikes from unit 342, the fitted tau and beta came out strongly
correlated (r=0.77). The natural first read is "makes sense, stretched spikes
also look shifted." To test that, we ran the *identical* fit on a control
group: isolated, non-bursting spikes from the *same* unit, which should show
essentially zero real stretch. If tau/beta correlation were real burst
physiology, it should vanish on this control group. **It didn't — it got
worse** (`demo_isolated_control.py`). That ruled out real physiology and
pointed at the regression itself: the derivative vector (drives tau) and the
stretch vector `q` (drives beta) turn out to be nearly parallel for a real
spike shape (~24° apart, not 90°) — the fit genuinely cannot separate "shifted"
from "stretched," and noise gets arbitrarily split between the two. Locked in
permanently as `test_sprime_and_q_are_severely_non_orthogonal` in `tests.py`,
with a large docstring warning in `nuisance_model.py` so this can't be
silently "fixed" by accident later.

**The fix:** don't try to fix the regression (Gram-Schmidt-style tricks
reproduce the exact same ambiguous answer — least squares has one unique
solution regardless of method). Instead, drop tau from this fit entirely.
Get timing from an independent method that doesn't share this problem —
pillar 1b's wavelet phase alignment — *first*, then fit only amplitude and
beta on the now-aligned snippet. This is `fit_nuisance_prealigned` (2-parameter
fit, template + `q` only, no derivative term).

**Confirmed this fix actually works:** re-ran the same burst-vs-isolated
control test with the corrected 2-parameter fit (`demo_isolated_prealigned_control.py`).
Beta now cleanly separates burst spikes (higher, more scattered) from isolated
spikes (centered near 0, tight) — and beta's own estimation noise (std) dropped
compared to the old 3-parameter fit, exactly as expected from removing a
collinear regressor. This is the real, validated pipeline used everywhere else
in the project since.

---

## 2. Pillar 1b — wavelet timing, three real problems found and fixed along the way

**Built:** `wavelet_features.py`. Core idea: probe the trace with a short,
complex (cosine + sine bundled together) oscillating test-waveform at a chosen
frequency. Its magnitude says "how strongly does a spike-like oscillation
sit here" (shift-tolerant, good for coarse localization); its phase says
"exactly where within one oscillation cycle" (precise, but only unambiguous
within half a cycle — beyond that it silently wraps around and gives a
confidently wrong answer).

**Problem 1 — calibration used the wrong reference sample.** A real spike
isn't a pure tone, so even a perfectly-aligned probe reads a nonzero phase
that depends on the spike's own shape, not on any real misalignment. This has
to be measured once per (template, probe) pair and subtracted off
(`calibrate_reference_phase`). First version used the array midpoint as the
alignment reference; the real convention is Kilosort's own `nt0min` (the
actual detected-spike sample). Using the wrong one produced a ~10-sample
spurious offset — caught by comparing recovered-vs-true shift against the
y=x diagonal, not by reading the code. Fixed; regression-tested
(`test_calibration_reduces_phase_bias`).

**Problem 2 — phase-only timing correction has a hard cliff.** Confirmed on
real and synthetic data: inside ±half a cycle, mean error ~0.008ms; step
outside that boundary and it jumps to ~0.51ms — not a gradual degradation,
a cliff. Fixed with a two-stage approach (`coarse_then_fine_shift`): a coarse,
whole-integer-sample search first (matched-filter score, no wraparound risk
because it only compares whole numbers), then phase-based fine correction on
the small leftover fractional part, which is now guaranteed inside the safe
zone. Result: stayed accurate (~0.002-0.004ms) across the *entire* tested
±20-sample range, not just the old narrow sliver
(`test_coarse_then_fine_survives_outside_old_safe_zone`). Also compared
against a simpler standard alternative (3-point parabolic peak interpolation,
`demo_method_comparison.py`) — coarse-then-fine is more precise, peak
interpolation is simpler but worse.

**Problem 3 (efficiency, not a bug) — the coarse search doesn't need to check
every sample.** The coarse stage's only real job is landing inside the fine
stage's safe zone, not finding the literal best integer. Made the coarse step
size scale with each unit's own half-cycle width — confirmed on two real units
at very different frequencies (739Hz -> step 10 samples, 2808Hz -> step 2
samples), with final answers after the fine stage still agreeing to within
~0.15 samples of the exhaustive (step=1) search, for an 8.5x speedup at low
frequency and 2x at high frequency (`demo_sparse_coarse_search.py`).

**Choosing the probe frequency per unit, not one fixed number.** Real units
have real, different waveform widths, and a wavelet tuned to the wrong
frequency underperforms — confirmed on 5 real units spanning widths 2-17
samples (`demo_real_unit_shapes.py`). The frequency is chosen per unit by
scanning many candidate frequencies against that unit's own template and
keeping the strongest match. A simple width-based shortcut rule (`f0 ~ k/width`)
was tested as a cheap predictor — it's only a rough prior (R²=0.42), not a
replacement for the real scan.

**The stretch problem: a fixed frequency degrades on stretched spikes.**
Since the frequency is chosen once from the *rested* template, a heavily
stretched (high-beta) burst spike is a worse match for that same fixed probe.
Confirmed on synthetic ground truth built on unit 342's own real template:
timing error grows from ~0.006ms (beta=0) to ~0.09ms (beta=0.5) with a fixed
frequency (`demo_wavelet_accuracy_vs_beta.py`). Two fixes were built and
compared against an idealized "oracle" that cheats by knowing the true beta
in advance:
- **Two-pass bootstrapping** (`demo_two_pass_vs_oracle.py`): align once at the
  fixed rest frequency, get a first (biased) beta estimate from that alignment,
  then rebuild the probe at `f0_rest / (1+beta_estimate)` — the exact
  frequency-compression relationship for a time-stretched signal — and
  re-align. Cuts the error roughly in half at high beta (~0.07ms at beta=0.5,
  vs ~0.09ms uncorrected).
- **Grid search** (`demo_grid_search_beta.py`): instead of one biased
  bootstrapped guess, test many candidate beta values in parallel, each with
  its own correctly-matched frequency, and keep whichever one "de-stretches"
  most self-consistently against the rested template. Performs about as well
  as two-pass, without the bootstrapping step.
- Notably, the **oracle is not a clean upper bound** — its error actually
  *increases* at beta=0.5 (0.1999ms) rather than staying best, apparently
  overfitting a single noisy spike to its own re-tuned frequency. Honest,
  useful finding: knowing the true beta perfectly doesn't guarantee the best
  practical result.

**Reliability, checked against real noise, not asserted.** Built a real,
per-channel noise-only magnitude distribution (400 random locations on unit
342's own channel, spike-free stretch of real recording) to answer "how do I
know a candidate spike's phase is trustworthy?" A real spike's magnitude
(3068) sat far above the 99th percentile of pure noise — several-fold above
the noise median (`demo_derivation_and_reliability.py`). This gives a
concrete, per-unit, data-derived way to flag low-confidence detections,
rather than an arbitrary cutoff. **Not yet wired into the rest of the
pipeline as an automatic flag** — currently just a diagnostic check.

**The phase formula itself was derived, not just asserted.** Worked out
algebraically on an idealized pure tone (real spikes are broadband, which
obscures the algebra) — confirms `angle(W) = -omega*delta` from first
principles (`demo_derivation_and_reliability.py`, part 1).

---

## 3. Real integration test on unit 342 — bugs found, and an honest open question

**Built:** ran pillars 1a+1b *together* on real bursts from unit 342
(`demo_integration_detected_vs_undetected.py`, then scaled to all bursts in
`demo_full_scale_search.py`): for every Kilosort-detected spike in a burst,
align (coarse+fine) then fit (a, beta). Separately, scan the *gaps* between
detected spikes (and just after the last one) for candidate events Kilosort
never reported, and run the identical pipeline on them.

**Bug found and fixed:** the first version extracted the aligned snippet by
rounding the total shift to the nearest integer sample — silently discarding
exactly the sub-sample correction pillar 1b had just computed. That discarded
fraction then got re-absorbed into `tau`/`beta` downstream, inflating them for
no real reason. Fixed by interpolating the raw trace at the exact fractional
shifted sample positions instead of rounding.

**Full-scale result:** across all of unit 342's bursts, a systematic scan of
gap locations found a modest number of candidates whose fit quality (R²)
reached into the range real detected spikes themselves occupy — quantitatively
promising. These candidates were then plotted in their full real burst
context, next to the real detected spikes on the same raw trace
(`demo_candidates_in_burst_context.py`), specifically so they could be judged
visually against real raw data rather than trusted on the R² number alone.
**This is the current honest state: the quantitative signal is real, but this
has not yet been converted into a validated "yes, these are real missed
spikes" conclusion — the next step (a full ablation against real,
independently-known ground truth, see design doc, still not built) is what
would actually settle this, not eyeballing more examples.**

---

## 4. Pillar 1c — spatial footprint

**Built:** `spatial_footprint.py`. Per-channel peak-to-trough amplitude
(not the integral — that cancels for a symmetric biphasic waveform),
normalized into a shape vector, compared by cosine similarity against a
unit's own template-derived footprint.

**Real test 1 — hardest real case available:** units 342 and 347 share the
exact same peak channel. Template-vs-template footprint similarity was 0.93
(deceptively high, not a clean separation on its own) — but scoring real
individual spikes against 342's footprint separated them cleanly: 342's own
spikes 0.93-0.98, 347's spikes 0.68-0.87, no overlap
(`demo_footprint_342_vs_347.py`). Asymmetric the other direction — 347's own
footprint is a weaker discriminator, consistent with 347 being a noisy,
hash-like cluster (no ACG refractory dip, 277k spikes) whose averaged template
is blurrier.

**Real test 2 — searched systematically for the cleanest illustrative case:**
scanned 98 real "good" units for the pair with the largest gap between
waveform-shape similarity and footprint similarity. Found units 381 & 397:
near-identical waveform shape (corr 0.985 — indistinguishable by shape alone),
80µm apart, template footprint similarity 0.002 (near-total separation)
(`demo_footprint_ideal_pair.py`). Honest caveat: that separation is close to
perfect on the *averaged template*, but real single spikes only reached 0.551
(381's own spikes) vs 0.424 (397's spikes) against 381's footprint — visibly
overlapping, much noisier than the template comparison suggests. Single-spike
per-channel noise is large relative to the true signal on the weaker channels
of a footprint, so this is a useful extra vote, not a standalone classifier on
one spike.

---

## 5. What's tested and protected against regression

`tests.py` — 17 permanent, assertion-based tests, run via
`python tests.py` (also pytest-discoverable), all passing as of the last
check. Covers: the basis-geometry claims above (including the *known,
deliberately-not-fixed* collinearity problem, locked in as a tracked fact,
not a silent gap), the corrected 2-parameter fit's validity and stability,
the linearization's real breakdown point at large beta, the phase-calibration
fix, the coarse-then-fine method's accuracy inside and outside the old safe
zone, the sparse-coarse-step speed/accuracy tradeoff, and the footprint math
and channel-selection logic.

---

## 5b. Temporal noise whitening — built, validated on real data, wired in

**Built:** `noise_whitening.py`. Fits a low-order (p=4) autoregressive model
to a real, spike-free stretch of a unit's own channel via the Yule-Walker
equations, derives the autocovariance that AR model implies over a
61-sample window, and builds a whitening operator (the inverse of that
covariance's Cholesky factor). This is the design doc's step-A "temporal
whitening" piece, and it addresses a genuine gap: Kilosort4's own whitening
is *spatial* (its ZCA transform decorrelates nearby channels at one time
sample) — it does not touch correlation across nearby time samples on one
channel, which real background noise clearly has.

**Validated on real data, two ways** (`demo_noise_whitening_validation.py`,
unit 342's channel, AR model fit on one real noise segment, checked on a
different held-out one):
- Real background noise is genuinely far from white: raw lag-1
  autocorrelation 0.72. After whitening: -0.09 at lag 1, 0.007 at lag 5 —
  substantially flattened.
- Injected the unit's own real template into real held-out noise at a
  modest amplitude (0.35x the template's own scale) and measured signal-vs-
  noise separation (d') for a plain matched filter vs. the whitened version
  of the same comparison: d'=0.032 plain, d'=0.102 whitened — roughly a 3x
  improvement in discriminability from whitening alone, on real noise, at
  matched signal amplitude.

**Wired in:** `fit_nuisance_prealigned` (the recommended fit) now takes an
optional `whitening_matrix` parameter; passing one turns the plain
least-squares fit into the noise-aware (generalized least squares) version.
Default `None` reproduces the exact old behavior — existing callers/tests
unaffected (`test_fit_nuisance_prealigned_whitening_matches_unweighted_when_identity`).
4 new tests added (AR fit correctness against a known synthetic process,
the AR(1) closed-form check, whitening actually decorrelating a known
synthetic process, and the identity-equivalence regression guard) — 21/21
tests passing.

**Not yet done:** whitening is only wired into the nuisance-model fit, not
into the wavelet coarse/fine timing search or the spatial footprint
comparison — both still implicitly assume white noise. Natural next
integration step, not done in this pass.

## 5c. Multi-unit generalization sanity check — real result, and a real gap it surfaced

**Built:** `demo_multiunit_sanity_check.py`, run per the user's explicit
request to check this isn't just tuned to unit 342. Ran the full pipeline
(per-unit frequency selection, wavelet coarse+fine alignment, whitened
amplitude/stretch fit) on unit 342 (the reference unit everything was built
against) plus three ordinary "good" units picked only by spike count (240 -
sparse, 307 - dense, 325 - medium), on a 40-spike sample each. Also added,
directly on the same real data: (a) flagging currently-detected spikes
whose fit quality is anomalously poor relative to that unit's own
distribution ("possibly doesn't belong"), and (b) scanning shortly after
each sampled spike for un-detected candidate events, exactly as done for
unit 342 earlier but now applied generically to any unit, not burst-gated.

**Real result — mixed, and informative:**

| unit | width (samp) | amplitude (uV) | mean R² | flagged low-quality | "promising" missed candidates |
|---|---|---|---|---|---|
| 342 (reference) | 8 | 16.7 | 0.82 | 4/40 | 0/15 scanned |
| 240 | 14 | 10.4 | 0.82 | 4/40 | 0/15 scanned |
| 307 | 7 | 4.7 | **0.41** | 12/40 | **14/15 scanned** |
| 325 | 14 | 5.4 | 0.57 | 4/40 | 7/15 scanned |

The pipeline generalizes cleanly to unit 240. It does **not** generalize
cleanly to units 307 and 325 — both are visibly lower-amplitude units, and
their fit quality is much worse even on Kilosort's own already-detected
spikes. Because the "possibly missed" flag is currently defined *relative
to each unit's own R² distribution* (10th percentile of that unit's own
detected spikes), a unit with generally poor fit quality gets a very
lenient bar (0.30 for unit 307), so nearly every scanned candidate clears
it (14/15) — this is very unlikely to mean "14 real missed spikes" and much
more likely means **the relative-threshold approach itself breaks down on
noisier units** — exactly the gap that pillar 3's not-yet-built reliability
calibration (an absolute, per-unit noise-floor-referenced bar, not a
relative percentile of possibly-already-poor fits) exists to close. This
sanity check is useful precisely because it demonstrates, on real data, why
that missing piece matters, rather than leaving it as a theoretical
concern.

**One more honest observation, not yet explained:** three of the four units
(342, 240, 325 — real widths 8, 14, and 14 samples, real amplitudes 16.7,
10.4, and 5.4 uV) landed on the *exact same* best-fit frequency
(753.061... Hz) from the per-unit frequency scan, despite visibly different
template shapes. This could be a real, broad low-frequency plateau in the
scan (plausible given a 50-point grid across 300-4000Hz and a
moderate-selectivity 3-cycle probe), or could indicate the scan is coarser
or more degenerate than assumed. Not investigated further this pass —
flagged here so it isn't lost.

## 5d. Following up on unit 307's "missed spike" candidates — three real checks, one real mistake caught

Prompted directly by the user asking three concrete questions about the
unit-307 candidates from 5c: (1) is their score even above what pure noise
achieves by chance, (2) could they actually belong to a spatially nearby
unit instead — including "did Kilosort's own threshold already catch this,
just jittered/mis-assigned", and (3) is whitening itself what's making them
pass. All three checked directly on real data
(`demo_unit307_candidate_validation.py`).

**First, a visual check that walked back the original "these are probably
junk" framing** (`demo_show_unit307_threshold_problem.py`): plotted 6 real
detected spikes against 6 of the candidates side by side. They look
similarly spike-shaped — the candidates are not obviously noise by eye.
Guessing they were junk without looking was a mistake; correcting course
here rather than letting that guess stand.

**Q1 — noise-only null**, same channel, identical pipeline, 150 pure-noise
locations: median R²=0.13, 90th percentile=0.30, 99th percentile=0.49. The
bar actually used to flag candidates (0.30) sits almost exactly AT noise's
own 90th percentile — meaning "passing the bar" is only barely better than
chance for most candidates. Of the 58 that passed: only 10/58 clear noise's
99th percentile, which is a real, meaningfully-above-chance signal; most of
the rest are not distinguishable from noise by this check alone.

**Q2 — spatial provenance, and a real mistake caught along the way.** First
attempt checked whether each candidate coincided (within 15 samples) with
an already-detected spike from ANY other cluster on the whole 384-channel
probe — got 56/58. That number is **meaningless and was reported in error
before being checked**: this session has 41 million spikes across 181
minutes of recording, dense enough that a *uniformly random* time point
already lands within 15 samples of *some* cluster's spike 98.2% of the time
purely by population base rate — confirmed directly with a random-time
control. Redone correctly, restricted to only the units that are
spatially close enough to physically matter (302 and 311, which share unit
307's exact peak channel, and 299, a large MUA cluster 25.6um away): only
**6 of 58** candidates coincide with an existing detection, and all six
coincide specifically with unit 299 (never with 302 or 311). The other 52
do not correspond to any nearby unit's existing detection at all — for
those, the "Kilosort already caught it, just jittered elsewhere" story the
user proposed does not apply; they are genuinely undetected locations, not
reassigned ones. Separately: for **20 of 58**, a spatially nearby unit's
own template (302, 311, or 299) fits the candidate's raw snippet better
than unit 307's own template does (checked via each neighbor's own
matched-filter score at the same alignment — an approximation, not a full
independent re-alignment per neighbor, noted as a limitation) — a real
signal that some of these candidates may be a neighboring neuron's spike
rather than a missed unit-307 spike.

**Q3 — is whitening responsible?** No. Recomputed every candidate's fit
without whitening: 0 of 58 would have failed the bar without it — the
unwhitened score was consistently as high or slightly higher than the
whitened one for every single candidate. Whitening is not manufacturing
these; if anything it makes the fit slightly more conservative here.

**Honest bottom line:** of the 58 original candidates, roughly 10 are
statistically convincing on their own, a different ~20 look like they may
actually belong to a neighboring unit rather than 307, only 6 coincide with
an already-known nearby detection, and whitening isn't the cause either
way. This is not a clean resolved story — the groups overlap and weren't
fully cross-tabulated this pass. What it does clearly show: the current
"possibly missed spike" flag (a relative percentile of the unit's own,
possibly-already-poor fit quality) is not trustworthy as implemented on a
weak unit, and needs, at minimum, (a) an absolute noise-floor-referenced
bar instead of a relative percentile, and (b) a same-time cross-check
against spatially relevant neighboring units built in automatically, not
run as a manual follow-up after the fact.

## 5e. Iterative wavelet-realigned template rebuild — a real bug caught, and a real, useful signal found

Built per the user's explicit description of the order: start from
Kilosort's own template, pick the matching frequency, coarse (sparse) +
fine wavelet-align every sampled real spike, average the aligned snippets
into a new template, repeat using the new template as reference and check
for convergence -- whitening intentionally left OUT of this loop (used
only afterward, to score spikes against the final template), so the two
ideas stay separately attributable (`demo_iterative_template_rebuild.py`).
Run on unit 342 (control, already known to work) and unit 307 (the weak
unit from 5c/5d).

**A real bug caught before it became a false claim:** the first run
reported the rebuilt template as 9-19x bigger in amplitude than Kilosort's
own template for both units. That number is wrong and was caught by
plotting the two templates together: Kilosort's `templates.npy` is not
stored in real microvolts -- it's in Kilosort's own internal working
scale, different from the real voltage read directly from the raw `.dat`
file. Every fit in this project lets amplitude float freely, so this
never broke any actual result, but the raw amplitude-ratio number itself
is meaningless and is retracted. Shape correlation (scale-invariant) is
unaffected and remains valid.

**Unit 342 (control):** converged after 2 rounds, final shape 98%
correlated with Kilosort's own template -- realignment sharpened it
slightly without reshaping it, and fit quality improved a little (mean R²
0.82 → 0.84). This is what a genuinely clean, single, well-isolated
neuron should do under this procedure.

**Unit 307:** did not converge within 5 rounds -- the matching frequency
kept drifting round to round (980 → 904 → 829 → 753 → 678 Hz) with no
sign of settling, unlike 342's immediate lock-in. The final rebuilt shape
is only 67% correlated with Kilosort's own template -- a real, visible
difference, not just sharpening. And counterintuitively, scoring the same
40 real spikes against this "improved" template made fit quality WORSE,
not better (mean R² 0.41 → 0.21).

**Honest interpretation:** if the spikes Kilosort grouped into unit 307
genuinely shared one true waveform shape, refining toward the best average
should have helped everyone, the way it did for 342. Instead refining made
the average fit the population worse and never stabilized -- consistent
with the spikes not actually sharing one shape, i.e. unit 307 being a
mixed/hash cluster rather than one coherent neuron. This is a second,
independent line of evidence pointing the same direction as 5c/5d's
findings (poor fit quality even on Kilosort's own confident spikes), not
proof on its own.

**Open, not yet done:** whitening the alignment search itself (currently
only the final scoring fit is whitened, not the coarse/fine alignment or
the averaging step) — a natural next test, kept separate deliberately so
its effect can be measured on its own rather than mixed in with this
result.

## 5f. Template rebuild across 8 units spanning independent SNR — our metric tracks something SNR doesn't

Found this session's independent quality metrics: `outputs/bombcell_results/templates._bc_qMetrics.csv`
(bombcell, a standard spike-sorting QC toolbox), confirmed to match this
session by exact `n_spikes` agreement with `cluster_info.tsv`. Picked 6
more units by bombcell's own `signalToNoiseRatio` column, deliberately
spanning its range (~26 to ~182, the weakest to the strongest "good" units
available), plus kept 342 and 307 as anchors — notably, bombcell rates
342 and 307 as nearly identical (SNR 79 vs 77) despite this project's own
diagnostics treating them completely differently.

Ran the same iterative rebuild (`demo_multiunit_template_rebuild_snr_spread.py`,
100 spikes/unit, up to 4 iterations). Real result, and it does not track
bombcell's SNR at all:

| unit | bombcell SNR | shape corr (rebuilt vs KS4) | R² change from rebuilding |
|---|---|---|---|
| 336 | 26.5 (lowest) | 0.97 | +0.020 |
| 6 | 36.8 | 0.61 | -0.071 |
| 397 | 50.4 | 0.94 | -0.056 |
| 171 | 67.4 | 0.69 | **-0.120 (worst)** |
| 307 | 77.0 | 0.75 | -0.034 |
| 342 | 79.2 | 0.98 | +0.010 |
| 313 | 99.8 | 0.98 | +0.003 |
| 135 | 181.7 (highest) | 0.98 | +0.020 |

The lowest-SNR unit in the set (336) behaves like the cleanest ones
(336, 342, 313, 135 all have shape correlation 0.97-0.98 and flat-to-
positive R² change from rebuilding). The worst-behaved unit (171) has a
solidly middling SNR, not the lowest. There is no visible trend between
bombcell's SNR and either shape stability or the rebuild's effect on fit
quality — plotted directly, panel 3 of `template_rebuild_02_snr_spread.png`
shows no relationship at all. The split instead falls cleanly into two
groups (336/342/313/135 stable; 6/397/171/307 unstable) that cut straight
across the SNR range. One exception, noted rather than smoothed over: unit
397 has a fairly high shape correlation (0.94) but still lands in the
"rebuild hurts" group on R² — doesn't fit the pattern perfectly.

**Honest reading:** whatever this project's fit-quality/convergence
diagnostic is sensitive to is a genuinely different property than
amplitude-based SNR — most likely something closer to "do this unit's
spikes actually share one consistent shape" (cluster coherence) than "how
big is the signal relative to background noise." That's a useful,
non-obvious finding on its own, but 8 units is a small sample — this
is a real pattern worth taking seriously, not yet a settled rule.

## 5g. Using the RIGHT quality metric, a simpler comparison metric, and a cross-check against Kilosort's own scoring — all three requested directly, and the result is much cleaner

Three corrections to 5f, all prompted directly:

1. **Wrong metric source.** 5f used bombcell's SNR by mistake. This
   project already has its own quality metric from earlier work: `sep_vs_noise`
   (shown as "Sep. vs noise (p80)" in `outputs/unit_review_pipeline.html`,
   underlying data in `outputs/pipeline_review_data.json`) — a completely
   different, project-specific measure, not bombcell's.
2. **Is R² even the right metric?** It's not something built new for this
   check — it's `nuisance_model.fit_nuisance_prealigned`'s R², the same
   fit-quality number used throughout this project since pillar 2. Worth
   checking whether a much simpler, more standard metric tells the same
   story: added plain **cosine similarity** between each aligned spike and
   the template (one dot-product ratio, no stretch parameter, no
   regression) alongside R².
3. **Cross-check against Kilosort's own scoring.** These spikes were
   already accepted by Kilosort's own matching-pursuit process, which has
   its own per-spike amplitude score (`amplitudes.npy`). For a genuinely
   good unit, Kilosort's own scoring shouldn't be unstable either. Computed
   the coefficient of variation of Kilosort's own amplitude for the exact
   same sampled spikes, fully independent of anything built in this
   project.

`demo_multiunit_metric_comparison.py`: selected 16 units evenly spaced
across the real range of `sep_vs_noise` (0.72 to 12.19), not hand-picked.
One bug caught mid-run and fixed: the review JSON is from an earlier
curation stage than this session's final output — unit 12 no longer
exists at all in `spike_clusters.npy` (0 spikes, absent from
`cluster_info.tsv` too, presumably merged away before finalization),
which crashed the first attempt. Fixed by filtering candidates to only
unit_ids that actually exist in the final data, not assumed to carry over.

**Real result, and it's a clean one this time:**
- **R² and cosine similarity both split cleanly** on `sep_vs_noise`: the
  11 units below ~2.2 (all labeled MUA/MUA+NON-SOMA in the review data)
  score R²=0.27-0.48, cosine similarity=0.39-0.66. The 5 units above ~2.4
  (all labeled GOOD/GOOD+NON-SOMA) score R²=0.73-0.90, cosine
  similarity=0.84-0.93. The two metrics agree with each other closely —
  the simpler, more standard one tells the same story as the more
  complex one, which is reassuring rather than a reason to prefer one
  over the other.
- **Kilosort's own amplitude CV shows no such split** — scattered
  0.074-0.170 with no visible relationship to `sep_vs_noise` at all (one
  of the GOOD units, 332, has the single *highest* CV of all 16 units).
  Plotted directly against our R² (panel 4 of `multiunit_04_metric_comparison.png`),
  there's no relationship there either.
- This generalizes what unit 342 vs 307 first suggested by hand: Kilosort's
  own amplitude scoring, based on a rigid single-shape template match with
  no phase/shape sensitivity, does not catch the same distinction our
  wavelet-aligned shape/similarity check does. `sep_vs_noise` — a
  metric built earlier in this project — tracks our result far better
  than either bombcell's SNR or Kilosort's own per-spike amplitude does.

**Honest reading:** this is a real, clean, generalized result across 16
properly-selected units (not 2 hand-picked ones), and it converges with
the MUA/GOOD labels already present in the review data — which is a good
sign this metric is measuring something real, not an artifact. Still
worth keeping in mind this uses the SAME session's own labels as a
soft validation, not truly independent ground truth.

## 5h. First real missed-spike search on VALIDATED good units — real signal, but mostly collisions, not clean misses

Direct test of the user's hypothesis: Kilosort's matching-pursuit detector
is a rigid template match with no sub-sample tolerance, so a real spike
from an otherwise clean, well-isolated neuron could still get missed
purely from landing at an awkward timing phase. Restricted this test to
the 6 units already validated as trustworthy in 5g (303, 313, 245, 332,
306, 342 — high `sep_vs_noise`, high R²/similarity), specifically so the
"what does a real spike score" reference distribution used as the bar is
itself trustworthy, unlike the fragile relative bar that broke down on
unit 307 in 5d (`demo_missed_spike_search_good_units.py`).

Method: for each unit, scanned a window both before and after 60 sampled
real spikes for un-detected local-maximum events (coarse matched-filter
score first, cheap), refined the top 25 candidates per unit through the
full wavelet-align + whitened-fit pipeline, and compared each candidate's
R² against that SAME unit's own real-spike R² distribution (25th/10th
percentile).

**Real result:** small but consistent — 7 of 150 candidates (across all 6
units) clear the real-spike 25th percentile, 20 of 150 clear the 10th
percentile, and every single unit has at least one candidate at or above
its own real-spike quality bar. Visually, plotting the best candidate per
unit against the template, 4 of 6 (306, 332, 342, 303) look genuinely
spike-shaped, closely tracking the template — not noise.

**Then checked provenance, exactly as in 5d, and it changed the
interpretation:** of those same 6 best candidates, 5 coincide (within 15
samples / 0.5ms) with an ALREADY-DETECTED spike from a spatially relevant
neighboring unit (two — 306's and 332's candidates — coincide with a
neighbor sharing the exact same peak channel). The single candidate with
no nearby match (unit 245's) was also the visually weakest, least
convincing one of the six.

**Honest interpretation:** this batch of candidates looks much more like
genuine COLLISIONS (two nearby neurons firing close together in time, one
neuron's real spike bleeding onto the other's channel and partially
matching its template) than clean, isolated misses caused by timing
jitter alone. This is a real, different, harder problem than plain
missed-detection — exactly the "collision/joint-refitting" gap the
original design doc flagged as needed but not designed. The jitter
hypothesis isn't disproven (some fraction of the 150 candidates weren't
checked for provenance, and jitter and collisions aren't mutually
exclusive in a dense recording), but the clearest, most convincing
examples found so far are better explained as collisions.

**Not yet done:** provenance-checking all 150 candidates (only the top 6
were checked); distinguishing "collision with a real neighbor spike" from
"genuine isolated jitter miss" systematically rather than case by case;
any attempt at joint refitting for the collision cases.

## 5i. Before/after correction, and template shape — the story shifts again, from "collision" to "probable oversplit"

Two follow-ups requested directly: (1) don't assume the already-assigned
unit is correctly assigned — check whether the two units might actually be
the same neuron split in two; (2) show the match score BEFORE any wavelet
correction (as close as achievable to what a rigid, non-phase-tolerant
matcher like Kilosort's own matching pursuit would see) versus AFTER, to
test the jitter hypothesis directly (`demo_candidate_before_after_and_provenance.py`).

One implementation bug caught and fixed before trusting the numbers: the
first version compared a raw, unnormalized dot-product score ("before") to
an R² fraction ("after") — different scales entirely, not a fair
comparison. Fixed to compute the SAME whitened R² fit on both the
uncorrected and corrected snippet, so before/after are on the same 0-1
scale.

**Result 1 — alignment usually makes the fit WORSE, not better, on this
candidate population.** Across all 150 candidates (25 per unit x 6
units): applying the full coarse+fine wavelet search made R² worse 63% of
the time, better only 37% of the time, mean change -0.05. This is the
opposite of what happens on real confirmed spikes throughout this
project, where alignment reliably helps. Most likely explanation: the
±25-sample search radius is free to lock onto a different nearby feature
within its window (noise, or a genuinely different real event) rather
than refining a true but jittered spike of this unit — which argues
against "jitter alone caused these to be missed."

**Result 2 — every single spatially-close "other unit" match has a
near-identical template shape.** Computed shape correlation directly for
all 8 unit-neighbor pairs found in 5h: 306-vs-305 (same channel) 0.996,
306-vs-298 0.977, 303-vs-299 0.980, 303-vs-306 0.979, 332-vs-333 (same
channel) 0.956, 332-vs-334 (same channel) 0.984, 342-vs-341 0.955,
313-vs-318 0.884. Every one is at or above 0.88 — higher than the 0.93
footprint similarity that first flagged the original 342-vs-347
investigation much earlier in this project as a plausible split.

**Revised, honest interpretation:** this combination — realignment not
reliably helping, plus every "colliding" neighbor having an almost
identical template — points away from both "isolated jitter miss" and
"two neurons colliding," and toward **Kilosort having over-split one real
neuron into multiple clusters** (e.g. 306/305, 332/333/334, 303/299/306,
342/341 each look like duplicates of one neuron, not two distinct ones).
That would explain both findings at once: a "candidate" is often a
genuinely real spike that correctly belongs to the OTHER (near-duplicate)
cluster, which is exactly why realignment doesn't reliably improve its
match to THIS cluster's template, and exactly why it was never assigned
here in the first place.

**Not yet done:** this is still an inference from template similarity and
score behavior, not a direct merge test. The natural next step, per the
original design doc's own "Stage 2: suggest, don't auto-merge" plan, is a
real cross-correlogram check between each candidate pair (does the
combined spike train show a genuine refractory violation, the way two
real distinct neurons should, or does it look like one coherent unit) --
not yet built.

## 5j. The actual merge test — cross-correlograms, and a real correction to 5i

Built the real test 5i called for (`demo_merge_candidate_ccg_test.py`):
a single real neuron cannot fire twice within its own absolute refractory
period (1.5ms used here) -- that is a hard biophysical limit, not a
clustering choice. So if two clusters are really one neuron Kilosort
split apart, MERGING their spike trains should still show a clean
refractory gap at short lag in the combined autocorrelogram. If merging
instead creates an excess of very-short-latency pairs between the two
clusters, that is evidence against them being one neuron.

Computed, for all 9 candidate pairs from 5i: each cluster's own
refractory violation rate, the merged train's violation rate, and --
more rigorously -- the chance-expected number of sub-1.5ms coincidences
two truly INDEPENDENT clusters would produce given their firing rates and
this session's 181-minute duration (standard point-process calculation:
`n_a * n_b * 2w / T`). Every pair showed MORE such coincidences than
chance predicts (1.7x to 20x excess) -- ruling out "these are just two
unrelated neurons that happen to fire near each other sometimes."

**The decisive test is the SHAPE of the merged autocorrelogram**, not just
the raw violation count: computed a "dip ratio" (density of merged-train
spike pairs inside the 1.5ms refractory zone, divided by the baseline
density from 5-25ms lag). 0 = a clean single-neuron refractory gap, 1 = no
dip at all, above 1 = an actual PEAK at short lag -- which a real single
neuron cannot produce.

| pair | dip ratio | reading |
|---|---|---|
| 332-333 | 0.10 | clean dip -- strong merge candidate |
| 342-341 | 0.16 | clean dip -- strong merge candidate |
| 303-306 | 0.27 | good dip -- merge candidate |
| 306-305 | 0.28 | good dip -- merge candidate |
| 332-334 | 0.48 | shallow -- ambiguous |
| 333-334 | 0.48 | shallow -- ambiguous |
| 306-298 | 0.63 | weak -- ambiguous |
| 303-299 | 0.70 | weak -- ambiguous |
| 313-318 | **1.46** | **an actual PEAK, not a dip** |

The four ambiguous pairs all involve one of three huge clusters (298:
188,492 spikes; 299: 176,743; 334: 93,459) that are almost certainly
multi-source hash clusters on their own, not clean single neurons -- the
real relationship there is more likely "a clean unit's spikes sit inside a
much bigger catch-all cluster" than a simple two-cluster merge.

**313-318 is a real correction to 5i's inference.** It had a plausible
(0.88) template correlation and was flagged as a collision candidate, but
the actual merged autocorrelogram shows a PEAK at short lag rather than a
dip -- the opposite of what one real neuron's spike train should produce.
This pair should NOT be merged; template similarity and score behavior
alone were not sufficient to establish that, which is exactly why this
direct test was needed rather than stopping at the earlier inference.

**Bottom line (SUPERSEDED by 5k below):** four pairs (332/333, 342/341,
303/306, 306/305) pass a real, quantitative merge test and are strong
candidates for manual review in Phy. This remains "suggest, don't
auto-merge" per the design doc -- nothing here has changed any Kilosort/
Phy output.

## 5k. A real statistical flaw caught in 5j, and the "merge candidates" don't survive the proper test

Flagged directly: 5j's "chance expectation" (`n_a*n_b*2w/T`) assumes each
unit fires at a CONSTANT rate across the whole 3-hour session. Real
neurons don't -- up/down states, bursting, task modulation all make true
firing rate non-stationary. Two genuinely independent neurons that are
simply co-modulated by the same slow process (busier during the same
behavioral epochs, say) could show an "excess" of near-simultaneous
spikes with nothing to do with being the same cell, and with small spike
counts, an apparently clean dip could just be a low-count fluke -- the
closed-form Poisson approximation can't tell the difference.

**Fix: a jitter-based permutation test** (`demo_merge_candidate_jitter_test.py`,
standard method for CCG significance, e.g. Fujisawa et al. 2008). For each
pair, jittered every spike of the smaller-count unit by a random offset
within +/-10ms (much bigger than the 1.5ms refractory window -- destroys
any genuine fine-timescale relationship while preserving each unit's real,
slow-timescale rate structure, so real shared comodulation is preserved in
the null too) and rebuilt the null 1000 times, comparing the REAL
cross-refractory-window count against this empirical distribution instead
of a formula.

**Result: none of the four "clean merge candidates" from 5j survive.**
306-305 (observed 37 vs null 31.5±4.9), 332-333 (11 vs 11.3±3.2), 303-306
(35 vs 30.9±5.0), and 342-341 (93 vs 96.3±8.9) are all statistically
indistinguishable from the rate-matched chance null (p_depleted > 0.3 for
all four) -- the dip-ratio metric in 5j was not properly accounting for
each unit's actual firing-rate structure, exactly the gap flagged. The
other four pairs (306-298, 333-334, 303-299, 313-318) instead show a
significant EXCESS over the jittered null (p_elevated < 0.01 for all
four, 313-318 extremely so at z=16.6) -- the opposite of a merge
signature, more consistent with the giant hash clusters (298, 299, 334)
partially containing real spikes that also belong to the smaller, cleaner
nearby units (contamination) rather than genuine shared identity.

**Corrected bottom line (ITSELF SUPERSEDED by 5l below): with a properly
rate-matched null, ZERO of the 9 candidate pairs tested are statistically
supported merge candidates.** The 5j conclusion was wrong -- not because
the pairs are proven to be genuinely separate neurons, but because the
evidence for merging any of them does not hold up once tested rigorously.

## 5l. A second, more fundamental correction: the refractory test can only DISPROVE a merge, never prove one

Directly corrected: the right criteria for merging two units are (1)
similar waveform shape, (2) similar spatial footprint across nearby
channels (unless probe drift separated them in time, in which case they
should fire in complementary, non-overlapping time windows instead), and
(3) no excess of refractory-period violations when merged. Critically,
**(3) is a one-directional test** -- a significant EXCESS of
short-latency cross-unit spikes is real disproof (a single neuron cannot
violate its own refractory period, so an excess rules out "same neuron").
But the ABSENCE of a significant excess is not positive evidence for
merging -- it just means the refractory test doesn't rule the merge out.
5k's conclusion ("zero candidates supported") incorrectly treated a
NON-significant jitter-test result as if it disproved the merge, which is
the same category of error as 5j's original mistake (treating a clean dip
as proof), just in the opposite direction.

Re-applied the three criteria properly to all 9 pairs, adding a real
spatial-footprint check (`spatial_footprint.py`'s actual footprint vector,
not just peak-channel distance) and a presence-over-time check (10
deciles across the session) to test for the drift-separated-time-windows
case:

| pair | shape corr | footprint sim | refractory veto (5k)? | verdict |
|---|---|---|---|---|
| 306-305 | 0.996 | 0.83 | no | **good merge candidate** |
| 332-333 | 0.956 | 0.90 | no | **good merge candidate** |
| 303-306 | 0.979 | 0.83 | no | **good merge candidate** |
| 342-341 | 0.955 | 0.60 (weaker) | no | plausible, weaker spatial support |
| 306-298 | 0.977 | 0.79 | **yes** (p=0.001) | excluded |
| 333-334 | 0.942 | 0.96 | **yes** (p=0.002) | excluded |
| 303-299 | 0.980 | 0.95 | **yes** (p=0.001) | excluded |
| 313-318 | 0.884 | 0.84 | **yes** (p=0.001, z=16.6) | excluded -- weakest shape match too |

None of the four surviving pairs showed the anti-correlated,
complementary-time-window presence pattern that genuine drift separation
would produce -- all four instead show mild positive co-variation in
firing rate across the session (most likely a shared session-wide trend,
not evidence either way for merging).

**Corrected, final bottom line for this batch: 306-305, 332-333, and
303-306 are well-supported merge candidates (strong shape and spatial
match, not vetoed by the refractory test); 342-341 is plausible but weaker
on spatial grounds; the other four pairs remain properly excluded by the
refractory veto.** These are suggestions for manual review in Phy, per
the design doc's "suggest, don't auto-merge" plan -- nothing here changes
any Kilosort/Phy output.

## 5m. Checking the location score itself — cosine similarity vs. real correlation

Asked directly how the "location" score actually works, and whether plain
correlation (not just cosine similarity) had been checked between the two
units' per-channel amplitudes. It hadn't -- only cosine similarity
(`footprint_similarity`: each unit's per-channel peak-to-trough amplitude,
L2-normalized, compared by `dot(v1,v2)/(||v1||*||v2||)`) had been used.
Cosine similarity doesn't subtract each vector's own mean first, so for
two all-positive, single-peaked vectors it can look more similar than a
proper (mean-subtracted) correlation would find. Computed Pearson
correlation on the same channel sets directly.

**Real result, and it's stricter than the cosine number suggested:**

| pair | cosine sim | Pearson r | p-value |
|---|---|---|---|
| 306-305 | 0.83 | 0.52 | 0.13 (not significant, only 10 channels) |
| 332-333 | 0.90 | 0.74 | 0.015 |
| 303-306 | 0.83 | 0.47 | 0.17 (not significant) |
| 342-341 | 0.60 | 0.13 | 0.70 (essentially no correlation) |
| 333-334 (excluded) | 0.96 | 0.89 | 0.0005 |
| 303-299 (excluded) | 0.95 | 0.79 | 0.006 |
| 306-298 (excluded) | 0.79 | 0.50 | 0.10 |
| 313-318 (excluded) | 0.84 | 0.70 | 0.012 |

Two honest findings: (1) 342-341's spatial support essentially disappears
under real correlation (r=0.13) -- cosine similarity's 0.60 was
misleadingly generous because both vectors are positive and roughly
co-peaked, not because the channel-by-channel pattern actually tracks.
Downgraded from "plausible" to not well-supported spatially. (2) plain
correlation does not cleanly separate the retained candidates from the
excluded ones -- two of the refractory-vetoed pairs (333-334, 303-299)
have STRONGER, more statistically significant spatial correlation than
any retained candidate. This isn't a contradiction (those stay excluded on
refractory grounds regardless), but it shows spatial similarity alone,
with only 10-12 overlapping channels, is a fairly weak/noisy test here,
not a strong independent discriminator.

**Further-corrected bottom line: 332-333 is the only pair with solid,
statistically meaningful support on all three criteria at once. 306-305
and 303-306 have real shape support and reasonable but not statistically
significant spatial support. 342-341 is dropped from the candidate list --
its spatial case does not hold up under real correlation.**

## 5n. Closing insight for the 5h-5m arc: this was a detection-recall search that found a precision problem instead

Stepping back: the whole arc from 5h onward set out to answer "does
Kilosort miss real spikes on otherwise good units due to timing jitter"
(Stage 3 of the original design doc -- recovery). It didn't find that.
Every candidate with enough signal to look credible turned out, on
investigation, to already exist in the spike list -- just assigned to a
different, almost always near-identical, unit. Not one of the ~150
scanned candidates across 6 validated units turned into a confirmed
genuinely-new detection.

**That is itself the finding, and it changes what the open problem
actually is.** If real, credible spikes are essentially always already
present in the combined spike list under some cluster label, then
detection recall is not where this session's problem lives -- Kilosort's
own threshold/matching is already catching what's really there. The
problem that kept surfacing instead, every time, was cluster PRECISION:
whether the right spikes are grouped under the right label (Stage 2 of the
design doc -- burst-split/oversplit reconciliation), not whether spikes
are missing (Stage 3). Section 5h's "missed spike search" became, in
effect, a burst-split/oversplit finder, and that shift was not the
original plan -- it's what the evidence actually supported.

**Practical implication for what to build next:** prioritize Stage 2
(automatic detection of candidate split/duplicate cluster pairs, and the
three-part test worked out in 5i-5m -- shape, spatial correlation,
refractory-veto) over further Stage 3 (missed-spike recovery) work,
unless a future test on a different unit/session actually turns up a
genuinely novel, non-duplicate detection. Real, usable output from this
whole arc: two to three concrete merge candidates (332-333 solid;
306-305 and 303-306 plausible) ready for manual Phy review.

## 5o. The frequency-selection rule was wrong, and it made the wavelet fine stage actively harmful

Found while answering a direct question -- not just "how many detections
does each configuration produce", but "are they the SAME events, and could
the weaker configuration's extras simply be noise". Measuring set overlap
rather than counts is what exposed this.

**Overlap, unit 440 (session 20260916_110311, 20-min window):** the four
configurations do NOT find the same events. Jaccard overlap of full vs
no_wavelet was only 0.69; full vs no_whiten 0.93 (whitening barely changes
membership, consistent with its ~1% count effect). Only 470 events were
found by all four. So the near-identical detection COUNTS reported in 5n
were hiding substantial membership differences.

**Then the events themselves showed the problem.** Plotting events found
only WITHOUT corrections: large, clean, neighbour-corroborated spikes
(258-266uV) that the wavelet stage had shifted by -1.8 to -2.2 samples,
pushing them off the template and collapsing their fit (R2 0.80->0.50,
0.73->0.44, 0.77->0.47). The raw trace was already correctly aligned; the
correction broke it.

**Quantified on 300 known-real spikes** (Kilosort's own detections for unit
440, already correctly placed, so any large shift is by definition an
error): median |shift| applied was 0.95 samples, 48% moved >1 sample, max
26 samples. Mean R2 went 0.777 (no alignment) -> 0.707 (with alignment);
p25 0.747 -> 0.624. Damage was entirely in the spikes it moved: R2 change
-0.142 for |shift|>1, -0.002 for |shift|<=1.

**Diagnosis, in three steps:**
1. NOT the coarse search locking onto neighbours -- narrowing search_radius
   from 25 to 2 changed nothing (median shift still ~0.97, R2 still 0.704).
2. NOT a broken method -- on ground truth (this unit's own template,
   injected at known shifts, no noise) recovery was accurate to 0.024
   samples.
3. It is noise being converted into spurious timing. With the true shift
   held at ZERO and only noise added: 0% noise -> 0.02 samples spurious,
   5% -> 0.17, 20% -> 0.79, 30% -> 1.18. Real spikes' observed 0.95 median
   corresponds to ~20-25% noise, which is ordinary for extracellular data.

**Root cause, which is arithmetic:** the fine stage converts phase to time
via delta = -delta_phi/omega, so timing_error ~= phase_error * (fs/f0)/360.
The cycle length is a straight multiplier on phase error. These striatal
units selected f0 ~800Hz, where one cycle spans 37 samples -- so a
10-degree phase error, easily produced by noise, becomes a FULL SAMPLE of
spurious shift.

**The rule was optimizing the wrong thing.** f0 was chosen as
argmax |W(f0)| -- maximum match strength, ignoring that the same f0 sets
timing precision. Since phase error scales as ~1/|W(f0)|, the expected
timing error goes as fs/(f0*|W(f0)|), which is minimized by maximizing
**f0*|W(f0)|**, not |W(f0)|. This is a real optimum, not "pick a higher
frequency" -- too high and |W| collapses faster than the shorter cycle
helps.

**Fix implemented and validated** (`wavelet_features.select_probe_frequency`,
two new tests, 23/23 passing):

| unit | rule | f0 | median spurious shift | frac >1 samp | mean R2 | p25 |
|---|---|---|---|---|---|---|
| 440 | no alignment | -- | -- | -- | 0.780 | 0.743 |
| 440 | argmax \|W\| (old) | 802 Hz | 0.72 | 34% | 0.737 | 0.691 |
| 440 | argmax f0·\|W\| (new) | 2620 Hz | 0.33 | 2.8% | 0.787 | 0.743 |
| 408 | no alignment | -- | -- | -- | 0.732 | 0.636 |
| 408 | argmax \|W\| (old) | 802 Hz | 0.86 | 46% | 0.713 | 0.547 |
| 408 | argmax f0·\|W\| (new) | 1742 Hz | 0.43 | 20% | 0.740 | 0.633 |

Alignment goes from actively harmful to mildly helpful. Honest caveat: the
gain over NO alignment is small (0.780 -> 0.787), because Kilosort already
places its own detections well -- little real misalignment is left to fix.
The benefit should be larger on candidate events that are not already
well-placed, which this test does not measure.

**This also explains the anomaly logged in 5f** ("three of four units landed
on the exact same best-fit frequency despite visibly different waveform
shapes"): |W(f0)| alone is broad and flat across low frequencies, so argmax
lands in the same plateau for many units. Weighting by f0 sharpens the peak
-- units 440 and 408 separate to 2620 and 1742 Hz instead of both sitting
at 802.

**What this invalidates:** the "wavelet alignment contributes +14% more
detections" claim in 5n was an artifact. Those extra detections came from a
corrupted, lowered threshold (the damaged spikes dragged the reference
distribution's p25 from 0.747 down to 0.624, making the bar more
permissive), not from better sensitivity. Any detection count in 5n/5m that
used the full pipeline with the old frequency rule is affected.

**What it confirms:** sub-sample phase genuinely matters for detection --
alignment changed ~30% of the detection set membership, and a ~1 sample
misalignment costs 0.14 R2. That was this project's founding premise and it
holds; it was the implementation, not the premise, that was wrong.

---

## 5p. Rerun with the corrected frequency rule -- the "new spike" count mostly evaporates

Every detection number before this used the broken argmax|W| frequency
rule, so the 20-minute scan was rerun with both rules through the identical
pipeline (`demo_20min_rerun_corrected_frequency.py`).

**The fix validates on real spikes:** spurious shift on known-correctly-
placed spikes, unit 440: median 1.01 samples with 53% moved >1 sample (old)
-> median 0.37 with 1.3% (new). Unit 408: 0.72/40% -> 0.33/10.7%. The
configurations also stop disagreeing with each other -- overlap between the
full pipeline and the no-correction baseline went 0.688 -> 0.851 (unit 440)
and 0.565 -> 0.815 (unit 408), confirming that most of what the old rule
"uniquely found" was its own misalignment artifact.

**Corrected detection numbers (20 min):**

| unit | config | detections | attributable to neighbour | genuinely new |
|---|---|---|---|---|
| 440 | full, old f0 | 655 | 95.6% | 29 |
| 440 | full, NEW f0 | 476 | 99.8% | **1** |
| 408 | full, old f0 | 412 | 79.1% | 86 |
| 408 | full, NEW f0 | 314 | 87.6% | **39** |

The genuinely-new count drops from 115 to 40 across both units, and for the
cleanest unit (440) from 29 to **1**. Most apparent new detections were an
artifact of the damaged threshold, exactly as 5o predicted.

**Component contribution, now measured on a sound basis** (totals, both
units): full with new f0 = 790 detections / 40 new; no_wavelet = 730 / 26;
neither = 836 / 45; no_whitening = 835 / 44. Alignment gives a modest real
benefit (unit 408: 314 detections at 87.6% precision vs 267 at 90.6%
without) -- not the +14% claimed in 5n, which was the artifact, but not
nothing either. Whitening contributes essentially nothing to DETECTION
(476 vs 524 for unit 440; 314 vs 311 for unit 408), consistent with every
earlier measurement -- it helps against noise (the ~3x d' result in 5b
stands) but not when the competing explanation is a neighbour's
near-identical spike rather than noise.

**The dominant fact, unchanged and now on solid footing:** ~99% of
everything detectable on unit 440's channel already belongs to unit 439.
Since 5o/5m established those two have genuinely different spatial
footprints (440 peaks on ch23, 439 on ch19/21) and should NOT be merged,
this is a single-channel SPECIFICITY problem -- the one-channel template
cannot tell a neighbour's spike from its own. Multi-channel footprint
scoring (pillar 1c, built but never wired into the detection path) is the
obvious thing that would address it, and has not been tried.

---

## 6. What is NOT built yet (real gaps, not forgotten — tracked deliberately)

- ~~Noise whitening / precision matrix~~ — **built and validated**, see
  section 5b. Only wired into the nuisance-model fit so far, not into
  wavelet timing or spatial footprint.
- **Burst-history amplitude-recovery curve** `E[a|deltaT]` and the associated
  gating (only trust weak/deformed states near a confirmed anchor spike): not
  built.
- **The learned 4th deformation direction** `d_k`: deliberately deferred per
  the original design doc, still deferred.
- **Robust estimation** (Huber loss, soft/marginalized alignment for weak
  events, preventing recovered spikes from feeding back into their own
  model): not started.
- **Reliability calibration as a formal step**: the noise-floor comparison in
  section 2 exists as a diagnostic on one unit/channel, not as a systematic,
  per-unit calibration wired into acceptance decisions.
- **Combining pillars 1a/1b/1c into one joint score**: each has been validated
  separately; there is no single function yet that takes one spike and
  returns one combined, correlation-aware decision.
- **Automatic burst-split-candidate detection** (the CCG/ISI trigger rule from
  the design doc): the 342/347 and 381/397 pairs were both hand-picked from
  prior investigation, not surfaced automatically.
- **The real decisive experiment**: real bursting units' natural deformation
  injected into independent real background, with negative controls
  (anchor-only, anchor-then-different-neuron), compared against Kilosort at
  matched false-positive rate. Only a synthetic-in-synthetic-noise version
  exists so far (the beta/tau grid tests in section 1-2).

---

## 7. Git status

All of the above lives in `D:\Gil\spike_sorting_agent`, branch
`post-ks-correction`. Committed locally; **not yet pushed** to the GitHub
remote (`origin` = `https://github.com/ZurGil/striatal-spike-sorting.git`) —
push needs explicit confirmation before it happens (an earlier push attempt
was intentionally interrupted by the user over a live-recording concern, never
resolved either way since).
