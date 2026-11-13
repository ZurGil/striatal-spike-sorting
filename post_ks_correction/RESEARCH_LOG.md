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

## 5q. Pillar 1c integrated into scoring -- and it solves the specificity blocker

The blocker identified in 5p: on session 20260916_110311, ~99% of
everything scoring as a unit-440 spike on its peak channel was already a
detected spike of unit 439. Those two units peak on DIFFERENT channels
(440 on ch23, 439 on ch19/21) with genuinely different spatial footprints,
so they should NOT be merged -- but their waveforms on the one channel they
share correlate at 0.972, so a single-channel template cannot separate them.

**Built** (`spatial_footprint.py`): `multichannel_template` (the template
concatenated channel-major across the whole footprint),
`multichannel_snippet_concat` (same layout for observed data), and
`block_diagonal_whitening` (each channel whitened with its OWN temporal
noise operator, channels kept independent -- cross-channel structure is
Kilosort's spatial whitening's job and modelling it again here would
double-count). Two new layout/independence tests; 25/25 passing.

**The decisive test** (`demo_pillar1c_integration_test.py`): score 250 real
unit-440 spikes and 250 real unit-439 spikes (the impostors a 440-detector
must reject) three ways, and measure separation as AUC.

| scorer | unit 440 mean | unit 439 mean | AUC |
|---|---|---|---|
| single-channel R2 (what the pipeline does today) | 0.782 | 0.698 | 0.749 |
| multi-channel R2 (1c, concatenated fit) | 0.501 | 0.333 | 0.897 |
| **footprint similarity (1c's original metric)** | **0.928** | **0.705** | **0.999** |

**Footprint similarity separates them essentially perfectly** (AUC 0.999 --
a random 440 spike outscores a random 439 spike 99.9% of the time), where
the current single-channel score is a heavily overlapping mess (0.749).

**Design conclusion worth recording:** the concatenated multi-channel FIT
(0.897) is clearly better than single-channel but clearly worse than the
plain footprint-similarity metric (0.999). Two reasons: the concatenated
fit is dominated by the peak channel (amplitude 9.46 there vs 0.82-4.35
elsewhere) which is exactly where the two units look alike, and the
low-amplitude channels contribute mostly noise that no template can explain
-- which is why even unit 440's own spikes only reach R2=0.501 on the
concatenated fit. Footprint similarity instead compares the SHAPE of the
amplitude profile across channels, normalized, which is precisely where the
two units differ. So the right integration is footprint similarity as the
discriminator, not a bigger least-squares fit -- validating pillar 1c as
originally formulated over the "concatenate and fit" instinct.

**Caveat:** alignment for both populations was done with unit 440's own
template on its peak channel, so unit 439's spikes are aligned by a
template that isn't theirs. Peak-to-trough amplitude over a 61-sample
window is deliberately alignment-insensitive (1c was designed that way), so
this should not drive the result, but it has not been separately controlled.

---

## 5r. Frequency scan range was truncating 30% of units; adaptive search added

Prompted by asking why the selection hit the scan edge, whether that was
odd, whether all units do it, and whether an adaptive search (try one, jump
higher/lower) exists.

**The edge-hitting was real and widespread, not a quirk of two units.**
Measured across 20 good units with the old 300-4000Hz range: under the
timing criterion **6/20 selected the 4000Hz endpoint itself** -- their true
optimum was above the bound, so the scan range, not the data, was choosing.
Under p=1.5 it was 10/20 and under p=2, 15/20. (This also corrects 5o's
claim that p=1 was "the last non-degenerate value" -- that was based on two
units whose optima happened to be interior.)

**Widening to 8000Hz fixes it:** 0/20 at the edge, and 80% of units'
selections are unchanged to within one grid step -- only the truncated
minority moves, to 4033-5589Hz.

**Those higher frequencies genuinely align better, verified on real
spikes** (not merely scoring higher). Unit 39, against a no-alignment
baseline of R2=0.459: 1500Hz -> 0.347 (alignment actively HURTING),
4000Hz -> 0.465, 5589Hz -> 0.473, 7000Hz -> 0.473 (plateau). Unit 15 shows
the same shape. So below the old bound alignment was damaging these units,
and above it the benefit appears and then plateaus -- a real interior
optimum the bound was hiding.

**On the adaptive search:** the algorithm described (bracket, jump
higher/lower, narrow) is golden-section / ternary search, and it is only
valid for unimodal functions. Checked directly: f0*|W(f0)| across
300-8000Hz has a median of **2 local maxima** per unit (max 5), and only
**10/20 units are strictly unimodal**. So a pure bracket-and-narrow search
over the full range would sometimes converge onto the wrong peak.
Implemented instead as a hybrid, which is the safe form: coarse grid to
bracket the GLOBAL peak, then golden-section refinement confined to that
bracket. Validated against a 1200-point reference grid -- agrees to within
a few Hz on 5/6 units (unit 440 differs by 129Hz, a nearby competing peak),
at **14.8x less compute** than the equivalent-resolution grid.

**On FFT:** with n_cycles fixed the window width scales as 1/f0, so this is
a true constant-Q wavelet transform and the whole sweep cannot collapse
into a single FFT. What IS available: compute the template's spectrum once
(O(N log N)) and evaluate each f0 as a Gaussian-weighted inner product in
the frequency domain, O(N) per frequency instead of a time-domain sum. For
a 61-sample template this is irrelevant -- selection is per-unit and already
milliseconds; disk reads dominate everything. It would only matter if
frequency selection ever moved inside a per-spike loop.

`select_probe_frequency` now defaults to f_hi=8000, refines by golden
section, and prints a warning if the selection still lands on a scan
endpoint. 26/26 tests passing.

---


## 5s. Pillar 1c used as a real detection gate at last — and a correction to my own first reading of the result

Everything in 5h–5p scored candidate events on a single channel. 5q proved
that was the binding limitation but only as a separation test on two known
populations; it never ran inside a detection scan. This is that scan.
Script: `demo_footprint_gated_detection_scan.py`, 20 minutes of session
20260916_110311 (samples 50M–86M), units 440 and 408.

Two gates, both calibrated on the unit's *own* real spikes so nothing is
hand-tuned:

- **R² gate** — whitened amplitude+stretch fit on the peak channel, bar at
  the 25th percentile of the unit's own spikes. Deliberately left exactly as
  it was in 5p so the comparison is apples-to-apples.
- **Footprint gate** — cosine similarity of the observed per-channel
  amplitude pattern against the unit's expected footprint, bar at the 5th
  percentile. Chosen to be *permissive* on purpose: it keeps ~95% of real
  spikes by construction, so any drop in detections could not be blamed on a
  tightened threshold.

| unit | detections R² only | attributable | new | detections R²+footprint | attributable | new | own-spike retention |
|---|---|---|---|---|---|---|---|
| 440 | 474 | 473 | 1 | **1** | 1 | 0 | 74.8% → 74.0% |
| 408 | 311 | 274 | 37 | 246 | 209 | 37 | 74.8% → 74.8% |

Unit 440 is the result the whole 1c effort was aiming at: contamination
essentially eliminated at a cost of 0.8 percentage points of recall.

Unit 408 barely moved, and my first reading of that was **wrong**. I wrote
that 408 and 412 were "footprint-indistinguishable." They are not — see 5t,
which measures the separation directly and gets AUC 0.827. The gate failed
on 408 not because the feature is blind but because the *threshold* was in
the wrong place.

### Who actually got thrown out
`demo_footprint_gate_attribution_breakdown.py` breaks the aggregates down by
which neighbour each detection was attributable to. This also corrects a flaw
in the scan itself: it picked its impostor by spatial *distance*, which for
unit 440 selected unit 438 (same peak channel, 0 um) — and 438 turns out to
be rejected by the R² gate alone at a 0.4% pass rate. Distance does not
identify the hardest impostor; the contamination counts do.

Unit 440, before to after the footprint gate:

| neighbour | peak ch | dist um | before | after | removed |
|---|---|---|---|---|---|
| **439** | 19 | 40 | **470** | **0** | **470** |
| 432 | 24 | 52 | 4 | 0 | 4 |
| 450 | 18 | 51 | 2 | 0 | 2 |
| 436 / 426 | 27 | 40 | 2 each | 0 | 2 each |
| 434 | 25 | 26 | 2 | 1 | 1 |

Unit 439 alone accounted for 470 of 474 detections — 99.2% — and the gate
removed every single one. That is exactly the behaviour 5q predicted.

Unit 408 tells a different story: unit **412** accounted for 162 of 311 and
the gate removed only **2**. But it *did* clear out the spatially-mismatched
contaminants — unit 402 went 51 to 7, unit 416 went 11 to 0, unit 413 went
12 to 4. So the gate works as designed. 412 simply is not spatially
mismatched enough to fall below a bar set at 0.814.

### Why one fixed percentile cannot work
`demo_footprint_gate_calibration.py` sweeps the bar and shows the real
trade-off. The problem is structural: where the impostor sits *relative to
the target's own distribution* is a property of the **pair**, not of the
target, so no fixed percentile of the target's own spikes can be right for
every pair.

Unit 440 (own spikes mean 0.930, impostor 439 mean 0.705):

| bar | own kept | 439 events kept (of 470) |
|---|---|---|
| 0.75 | 99.3% | 229 |
| **0.80** | **99.3%** | **9** |
| 0.85 | 98.3% | 0 |
| 0.95 | 21.0% | 0 |

Unit 408 (own spikes mean 0.922, impostor 412 mean 0.850):

| bar | own kept | 412 events kept (of 162) | other events kept (of 149) |
|---|---|---|---|
| 0.80 | 97.0% | 161 | 96 |
| 0.85 | 79.3% | 144 | 59 |
| **0.90** | **73.0%** | **23** | 12 |
| 0.95 | 54.0% | 0 | 1 |

For 440 the gate is nearly free: bar 0.80–0.85 removes almost all
contamination and keeps 98–99% of real spikes. For 408 it costs real recall:
removing 86% of the 412 contamination needs bar 0.90, which keeps 73% of own
spikes. The impostor's 0.850 mean sits inside the target's own distribution,
and no threshold can fix that — the two overlap.

**One caveat that cuts against the good news:** at bar 0.90 unit 408's
non-412 detections drop from 148 to 12, which means the 37 "genuinely new"
events do *not* survive a properly-set gate. The new-spike count shrinks
again, for the third time in this project.

**Design conclusion:** the footprint bar must be set per pair, against the
measured impostor distribution, not at a fixed percentile of the target's own
spikes. The sweep is the calibration procedure.


## 5t. Merge re-examination with the corrected tooling — and 408/412 is not a merge

Script: `demo_merge_reexamination_corrected.py`. The pair came from the data
rather than from a label: stage 1's attribution breakdown identified 412 as
the dominant contaminant of 408, and 440/439 is carried along as a known
negative control.

The earlier plan listed re-running session 20260901_085606's candidates
(332-333, 306-305, 303-306) here. Those are deliberately **not** rerun: that
session's curation labels were flagged as untrustworthy, which is the whole
reason the project changed sessions, so re-deciding merges from its labels
would rest on the same bad foundation.

All four criteria, applied as specified — and the refractory test used as a
**veto only**, never as positive evidence (the 5l correction, honoured
explicitly in the code):

| criterion | 408 vs 412 | 440 vs 439 |
|---|---|---|
| peak channels | ch44 / ch42, 26 um | ch23 / ch19, 40 um |
| **shape** — waveform r on shared ch | 0.991 | 0.972 |
| footprint cosine | 0.767 | 0.589 |
| **footprint Pearson r** | 0.481 (p=0.16) | -0.068 (p=0.85) |
| **footprint AUC on real spikes** | **0.827** | **0.999** |
| refractory: observed vs jitter null | 35 vs 38.6+/-5.6, z=-0.64 | 20 vs 25.8+/-4.6, z=-1.26 |
| p(excess) | 0.771 | 0.912 |
| veto triggered? | no | no |
| active-window overlap | 99.9% | 99.9% |
| **verdict** | **do not merge** | **do not merge** |

Both pairs have near-identical waveforms on the channel they share (0.991 and
0.972) and both are separated by their spatial footprint. Neither refractory
test fires, and per 5l that is explicitly *not* taken as support for merging —
it only means nothing rules a merge out. What rules them out is the footprint.

Two things worth noting:

- **Cosine similarity keeps overstating agreement**, exactly as 5m found.
  408/412 looks like a 0.767 match by cosine but only 0.481 by Pearson r
  (p=0.16, i.e. not distinguishable from no relationship at 10 channels).
  For 440/439 the gap is starker still: 0.589 cosine versus -0.068 Pearson.
  All-positive amplitude vectors are never far apart in angle. Cosine should
  not be quoted on its own.
- **AUC 0.827 is the number that overturns 5s's first reading.** Footprint
  does separate 408 from 412; the gate was just set below the useful range.
- Both pairs coexist in time (99.9% overlap), so probe drift cannot be
  offered as an explanation for the footprint difference.


## 5u. Anderson acceleration on the template-rebuild loop — and the loop's real noise floor

New module `fixed_point_acceleration.py`, five new tests, plus
`demo_accelerated_template_rebuild.py` on real data.

### Which iteration this applies to, and which it does not
The template rebuild is a genuine fixed point: `T_{k+1} = G(T_k)` where G
means "wavelet-align every sampled spike to T_k, then average." Aitken /
Anderson extrapolation is exactly the right tool. The **frequency search is
not** a fixed point — it maximizes `f0*|W(f0)|`, and applying Aitken to a
maximization is a category error. That search stays on grid + bracketed
golden-section (5r). This distinction is written into the module docstring so
it does not get muddled again.

Aitken's delta-squared is scalar; a template is a 61-sample vector, so the
vector generalization — Anderson acceleration, least-squares over the last
`depth` residuals — is what the loop actually needs. `depth=0` reproduces
plain Picard through the identical code path, which is what makes the
comparison honest.

### One deliberate change to the loop
The original rebuild re-selected f0 from the *current* template every
iteration, which means the map itself changed step to step — not a fixed-point
iteration at all, and the direct cause of the drift recorded in 5e (f0 walking
802 to 739 to 676 Hz, amplitude decaying 8%). Here f0 is selected once from
Kilosort's template with the corrected timing criterion and then **held
fixed**. G is also rescaled to its input's norm, or "T stopped moving" would
be measuring the arbitrary microvolt scale of the average instead of shape.

### Result: Anderson wins everywhere, but the loop has a floor
G-evaluations needed to first reach each tolerance (250 spikes, 20-min window):

| unit | tol 1e-2 | tol 3e-3 | tol 1e-3 | tol 1e-4 | best residual reached |
|---|---|---|---|---|---|
| 440 Picard | 2 | 39 | 55 | never | 6.0e-04 |
| 440 Anderson | 2 | 9 | 29 | 35 | 4.4e-05 |
| 408 Picard | 11 | 29 | 46 | never | 3.7e-04 |
| 408 Anderson | 7 | 11 | 12 | 12 | **1.9e-08** |
| 439 Picard | 18 | 37 | 53 | never | 6.2e-04 |
| 439 Anderson | 8 | 9 | 10 | 13 | 9.0e-05 |

Speedups at tol 1e-3: **1.9x (440), 3.8x (408), 5.3x (439)**.

The more interesting finding is the column on the right. **Plain Picard
stalls at ~6e-4 on all three units and never gets below it**, no matter how
many iterations it is given. Anderson reaches 4e-5 to 2e-8. So the original
loop's "converged" test was never actually being satisfied — my first run
here used tol=1e-6, and *both* methods hit the 60-iteration cap on units 440
and 439, which is why their final templates only correlated 0.93 and 0.86
with each other. That disagreement was Picard failing to converge, not
Anderson misbehaving: where both got close (unit 408) the two templates agree
to r=0.999986.

Unit 408 reaching 1.9e-08 means a true fixed point exists there — every
spike's alignment stops changing and the average becomes exactly
reproducible. Units 440 and 439 plateau around 1e-4 even under Anderson,
which means a handful of their spikes have genuinely ambiguous alignment that
keeps flip-flopping between iterations. That is a per-unit diagnostic worth
keeping, not just a numerical nuisance.

**Practical upshot:** use `depth=3` and set the tolerance at 1e-3, which is
reachable for every unit tested. Asking for 1e-6 is asking for precision
below the sampling noise floor of a 250-spike average, and no accelerator can
deliver it.


## 5v. Does sub-sample jitter actually pollute Kilosort's clustering features? Testing the premise before forking anything

This is the premise test for the proposed change of strategy: instead of
using these analyses *after* Kilosort, put them *inside* it as flag-gated
options (vanilla behaviour when the flags are off). The highest-leverage
proposed insertion point is stage 3, feature extraction — so the premise had
to be checked first on real data.

### What Kilosort actually does (verified in source, not from memory)
Installed at `C:\Users\Adam\anaconda3\envs\kilosort4\Lib\site-packages\kilosort`.
Pipeline from `run_kilosort.py`:

1. `compute_preprocessing` — 300 Hz high-pass, then a whitening matrix built
   as `CC = X @ X.T / X.shape[1]` with X of shape (channels, time). The
   covariance is **channel x channel**: Kilosort whitens SPACE and never
   touches TIME. Our AR(4) temporal precision matrix is genuinely
   complementary, not a duplicate.
2. `compute_drift_correction` — datashift.
3. `detect_spikes` — `spikedetect.run` (universal templates, integer spike
   times, PC features `tF`), then a first `clustering_qr.run(mode='spikes')`,
   then `postprocess_templates` -> `align_U`, then
   `template_matching.extract` (the deconvolution that produces most spikes).
4. `cluster_spikes` — `clustering_qr.run(mode='template')` then
   `template_matching.merging_function`.
5. `save_sorting`.

Two openings, both verified:
- `align_U` aligns templates with `torch.roll(..., ops['nt']//2 - j, -2)` —
  an **integer** shift. Kilosort has no sub-sample alignment anywhere, so
  every template is an average of spikes each up to half a sample misplaced.
- `CCG.similarity` scores merge candidates with a normalized template
  cross-correlation **maximized over lag** — effectively a cosine, the exact
  measure 5m/5t showed overstates spatial agreement. And
  `merging_function` skips any cluster with `is_ref[kk]==0`, i.e. it uses
  refractoriness as a *precondition* for merging — structurally the same
  "absence of violation = permission" error corrected in 5l.

### The claim under test, and the correction to it
Claim: since a shifted waveform is `s(t-d) ~ s(t) - d*s'(t)`, random jitter
makes the derivative a real axis of variance, so PCA should spend a whole
component on timing — and Kilosort would then be clustering partly on
sub-sample timing.

Script: `demo_does_kilosort_pca_waste_a_component_on_jitter.py`. `wPCA` from
ops.npy is 6 components x 61 samples, confirmed orthonormal (largest
off-diagonal 2.2e-07).

**The strong form of the claim is WRONG.** No component is the derivative:

| component | \|cos\| with d(PC0)/dt | angle |
|---|---|---|
| PC0 | 0.000 | 90.0° |
| PC1 | 0.255 | 75.2° |
| **PC2** | **0.615** | **52.0°** |
| PC3 | 0.415 | 65.5° |
| PC4 | 0.554 | 56.3° |
| PC5 | 0.112 | 83.6° |

PC2 is the closest at 52° — related, not identical. So "Kilosort wastes a
component on timing" is not true as stated, and I should not have asserted
the mechanism that confidently before testing it.

**The weaker, true form is still consequential.** The 6-component basis
captures **93.5%** of the derivative direction's energy, so jitter enters the
features fully — just *distributed* across PC2/PC3/PC4 rather than isolated
in one. That distinction matters practically: you cannot fix this by dropping
a component. The jitter itself has to be removed.

### How big is the effect? (the decisive measurement)
Shift each unit's peak-channel template by a sub-sample amount, project onto
`wPCA`, and compare how far the feature vector moves against the spread of
feature vectors across 443 real units (the signal Kilosort clusters on).

| shift (samples) | feature displacement | as % of between-unit spread |
|---|---|---|
| 0.10 | 0.0350 | 3.6% |
| 0.25 | 0.0876 | 9.0% |
| **0.50** | **0.1754** | **18.0%** |

A half-sample error — the worst residual integer alignment can leave — moves
a unit's features by **18% of the distance separating genuinely different
units**. Typical residual (~0.25 samples) gives ~9%. Jitter lands hardest on
PC2, consistent with the geometry above.

### Pair statistics, with the spatial restriction applied
How many unit pairs sit closer together in feature space than jitter
displaces a single unit? The all-pairs figure is 6.2% of 97,903 — and that
number is **misleading** in the same way the "98% of random timepoints"
error was (see caveats), because most pairs are far apart on the probe and
could never be confused. Restricted to pairs Kilosort could actually confuse:

| neighbourhood | pairs | closer than jitter | fraction |
|---|---|---|---|
| within 40 um | 1,876 | 221 | **11.8%** |
| within 60 um | 2,727 | 294 | 10.8% |
| within 100 um | 4,864 | 505 | 10.4% |

The spatial restriction makes the case **stronger**, not weaker — nearby
units have more similar waveforms, so more of them fall inside jitter
distance. About **one in nine spatially-adjacent unit pairs** is separated by
less than the feature displacement a half-sample timing error produces.

### Conclusion
The premise holds in its useful form: sub-sample jitter is a real,
quantified contaminant of the features Kilosort clusters on, at ~9-18% of the
between-unit signal, affecting ~11% of adjacent pairs. That justifies
building sub-sample alignment as a flag-gated option at stage 3. It does NOT
establish that fixing it improves sorting — that still needs the
ground-truth harness. Caveats: this uses templates, not individual spikes
(so it measures the systematic effect, not the per-spike noise), and
peak-channel features only, whereas Kilosort uses a channel neighbourhood.


## 5w. Multi-frequency alignment built and tested — the estimator is sound, the ambiguity flag does not work on real data

New module `multifreq_alignment.py`, five new tests (36/36 passing), and three
real-data runs. The ambiguity flag was the thing I argued would make it safe
to align spikes before clustering. It does not hold up, and that changes the
plan.

### The idea
If an event really is this unit's template arriving at delay `delta`, then
after removing the template's own shape phase per frequency,

    dphi(f) = -2*pi*f*delta/fs

Phase is LINEAR in frequency, slope proportional to delay. That is the only
thing a time shift can do. A collision is a SUM of two shifted templates and
the phase of a sum bends. So one weighted line fit through the origin gives
two things: the slope is a delay estimate using all probes (weights
`(f_k*|W(f_k)|)^2`, the inverse-variance rule — note the existing
single-frequency rule `argmax f0*|W(f0)|` is just the K=1 case), and the
residual scatter is a per-spike "was this one spike?" statistic.

### It works perfectly on synthetic data
`test_clean_spike_has_low_scatter_collision_has_high_scatter` passes with a
clean margin: every clean shifted template scores below every collision.
`test_multifrequency_beats_single_frequency_under_noise` confirms the
variance reduction over 300 noise realizations. The mechanism is not in
doubt.

### On real spikes it is weak
`demo_multifreq_ambiguity_on_real_spikes.py`. Collisions labelled from
Kilosort's own spike list (another unit within 60 um firing within +/-12
samples), isolated = nothing within +/-40 samples:

| unit | isolated scatter (median) | colliding | AUC | flag rate iso / coll |
|---|---|---|---|---|
| 440 | 0.707 | 0.745 | **0.550** | 5.2% / 8.3% |
| 408 | 0.738 | 0.943 | **0.611** | 5.0% / 12.9% |
| 439 | 0.694 | 0.886 | **0.661** | 5.0% / 11.1% |

AUC 0.55-0.66, against 0.999 for the footprint on a comparable task. And the
baseline is alarming on its own: isolated spikes already scatter by ~0.70
samples, where synthetic clean spikes scatter near zero.

### Two candidate explanations, and the data picks one
`demo_multifreq_band_sweep.py` on unit 408 (the only unit with a usable
number of collision labels — 116, versus 12 for unit 440):

| keep_fraction | band (Hz) | isolated median | colliding median | AUC |
|---|---|---|---|---|
| 0.35 | 495-6403 | 0.738 | 0.943 | 0.611 |
| 0.50 | 585-4848 | 0.632 | 0.820 | 0.629 |
| 0.70 | 773-3284 | 0.483 | 0.667 | 0.621 |
| 0.85 | 965-2629 | **0.355** | 0.479 | 0.611 |

Narrowing the band cuts the baseline scatter **2.2x** — so the wide band
really was injecting noise (unit 440's reached 8000 Hz, the scan ceiling, the
same edge problem as 5r). But **the AUC does not move at all**. Both groups
shrink together. So the baseline is not probe-band noise; it is something
that affects clean and colliding spikes identically.

### The decisive test: scatter versus actual collision lag
If the statistic detects interference it must care how close the second spike
is. `demo_multifreq_scatter_vs_collision_lag.py`, narrow band, unit 408:

| lag to nearest neighbour spike | n | scatter median |
|---|---|---|
| 0-3 samples | 12 | **0.383** |
| 3-6 | 23 | 0.301 |
| 6-10 | 48 | 0.563 |
| 10-15 | 54 | 0.615 |
| 15-25 | 102 | 0.456 |
| 25-40 | 108 | 0.346 |
| 40+ (no collision) | 350 | **0.375** |

**The tightest collisions score the same as no collision at all** — 0.383 vs
0.375, a ratio of 1.02. The scatter even peaks in the 10-15 bin, where
interference should be weaker than at 0-3. There is no monotonic relationship
with lag. Unit 439's apparent 0.13x ratio comes from a single spike in the
0-3 bin and means nothing.

**Conclusion: the statistic is not responding to interference.** On real
spikes the scatter is dominated by ordinary spike-to-spike shape variability
— amplitude changes, burst-state changes, noise — all of which also make
phase non-linear in frequency. Collision is a minor contributor. The ~0.62
AUC is the method's real ceiling here, not a tuning problem, and I should
not have presented the flag as the safety mechanism before testing it on real
data.

### A second, unplanned finding that matters more
Unit 408 has **12 spikes out of 1,458** with a neighbour inside 3 samples —
about 0.8%. Unit 439 has **one**. Tight collisions are RARE in this data. So
the premise that collision handling is an important problem here is itself
weak, independent of whether the detector works.

### What survives
- **The weighted multi-probe estimator.** Sound in theory, validated on
  synthetic data. On real spikes it disagrees with the single-probe estimate
  by 0.33-0.44 samples on average — substantial, but with no ground truth we
  cannot say which is closer to the truth. Unverified, not disproven.
- **The jitter premise from 5v is untouched** — it rests on feature geometry
  (18% of between-unit spread, 11.8% of adjacent pairs), not on this flag.
- **A band-width lesson that applies to the existing pipeline too:**
  `keep_fraction=0.85` (band ~965-2629 Hz) cuts phase-estimate scatter 2.2x
  versus a wide band. Narrow is better for timing, and the default should
  reflect that.

### Consequence for the plan
The ambiguity flag was going to be the interlock that made pre-clustering
alignment safe — align the confident spikes, flag the rest. There is no
working flag, so that safety argument is gone. The jitter correction may
still be worth doing on the 5v evidence, but it now has to be justified by
the ground-truth harness directly rather than by a per-spike confidence
measure. **The harness moves from "prerequisite I keep recommending" to the
only remaining way to settle this.**


## 5x. The ground-truth harness, and its first verdict: multi-frequency is unnecessary, and the simple estimator is good enough to install in Kilosort

New module `ground_truth_harness.py`, seven new tests (43/43 passing), and
`demo_ground_truth_alignment_accuracy.py`. This is the thing that has been
listed as "the only way to establish real performance" since section 6 was
first written, and it settled an open question on its first run.

### Why it was finally built
Four results in this project have evaporated under better testing: the "new
spikes" twice, the footprint-gate reading I had backwards, and the
multi-frequency ambiguity flag that passed synthetic tests and failed
completely on real spikes (5w). Every number before this was one
configuration against another, with no known truth anywhere.

### The design, and the mistake it is built to avoid
5w's failure was instructive: the synthetic ambiguity test passed with a huge
margin and meant nothing, because every "clean" spike in it was an IDENTICAL
copy of the template, so the only thing that could disturb the phase was the
collision I had planted. Real spikes vary in amplitude and shape from firing
to firing, and that variation disturbs the phase just as much. A harness that
injects identical template copies would flatter us the same way.

So the harness uses the unit's **own real snippets**, which carry that
variability. The apparent problem is that a real snippet's intrinsic
sub-sample offset `e` is unknown, which seems to destroy the ground truth. It
does not, if the question is asked as a difference: inject copies of the SAME
snippet at known shifts `d_j` and grade

    estimate(d_j) - estimate(d_0)  ==  d_j - d_0

and `e` cancels exactly. Absolute accuracy, real variability, no assumption
anywhere that a real spike resembles its template.

Other design points: background is real recorded voltage at positions where
no unit within 60 um has a spike within 150 samples (so real noise with its
real lag-1 correlation of ~0.72, not Gaussian white noise); one injected
spike per trace so nothing can interfere; measurement taken at the known
injection index so both estimators are asked the identical question, which
isolates the fine stage being compared.

Seven tests cover the harness itself, because if the thing defining truth is
wrong then every number it produces is wrong: interpolation accuracy and
invertibility, integer shift matching `np.roll` (catches sign and off-by-one
errors that would silently invert every result), injection landing on the
requested sample, the offset-cancellation claim verified against secret
pre-shifts, the slope diagnostic catching a deliberately under-correcting
estimator, quiet positions genuinely avoiding spikes, and one-to-one
detection matching.

### First verdict: multi-frequency does not earn its place
150 real snippets per unit, shifts [-0.4, -0.2, 0, 0.2, 0.4]:

| config | median RMS (samples) | median slope |
|---|---|---|
| **single probe, narrow band** | **0.0389** | **0.919** |
| single probe, wide band | 0.0389 | 0.930 |
| multi probe, narrow band | 0.0363 | 0.895 |
| multi probe, wide band | 0.0570 | 0.807 |

Multi-probe beat single-probe on 48.7%, 50.7% and 48.7% of snippets for units
440, 408 and 439 — a coin flip. Marginally better median RMS in the narrow
band, slightly worse gain, and clearly worse with a wide band.

The relationship between the two estimators makes this interpretable. The
multi-probe estimate solves `Δφ_k = x_k·δ` with `x_k = -2πf_k/fs` by weighted
least squares, `w_k ∝ (f_k|W(f_k)|)²`. At K=1 that collapses exactly to
`δ̂ = -Δφ/ω`, the existing estimator, and `argmax f|W(f)|` is just
`argmax w_k`. So the single-probe rule is the K=1 case keeping only the
heaviest weight. The theoretical variance gain from using the rest requires
the measurements to be independent; they share one noise realization and the
wavelets overlap in time, so the gain does not materialize.

**Decision: keep the single-probe estimator. `multifreq_alignment.py` stays
in the repository as a tested, measured negative result, not as a pipeline
component.**

### Two findings only the harness could produce
1. **The single-probe fine stage is already excellent: RMS 0.028-0.039
   samples**, about one microsecond. There was nothing left for extra probes
   to win. This also explains why multi-frequency looked good synthetically —
   that test was unrealistically hard in the wrong dimension.
2. **Every variant systematically under-corrects by ~7-8%** (slope ~0.92, not
   1.00). This was completely unknown, is invisible to an RMS figure, and
   only shows up against known truth. Every correction the pipeline applies
   is slightly too small. Worth chasing: likely candidates are the
   first-order phase linearization and the cubic interpolation used both to
   inject and to resample.

### The consequence for the Kilosort integration question
This is the strongest evidence yet, and it points in favour:

- Kilosort's `align_U` aligns to whole samples, leaving up to 0.5 samples of
  residual error (RMS ~0.29 for a uniform residual).
- Our estimator's RMS is ~0.035 samples.
- That is an **8x reduction** in timing error.
- Section 5v measured that 0.5 samples of jitter displaces a unit's
  clustering features by 18% of the between-unit spread; scaling to 0.035
  samples puts that near 1%.

So the tool is accurate enough to be worth installing, and the simple
corrected single-probe version is sufficient — the multi-frequency
elaboration is not needed. What remains before touching Kilosort is to use
the harness for the question it was built for: does replacing integer
alignment actually improve clustering, measured end to end at a matched
false-positive rate.


## 5y. Sub-sample alignment implanted in Kilosort4 and graded end to end — the feature-space effect is real, the sorting benefit is a wash, and oversplitting is NOT caused by jitter

First actual change to Kilosort, measured against injected ground truth.
Files: `ks_patches.py` (the patch), `build_small_hybrid_dataset.py` (the test
data), `run_hybrid_comparison.py` (the grader).

### Reversibility, as required
Nothing modifies the installed kilosort package. `ks_patches.enable(name)`
swaps one function at runtime; `disable()` restores it; the patched function
checks the flag at the top and calls the original when off, so stock
behaviour is reproduced by construction rather than by assertion.

### Where the patch goes, and the correction to my earlier plan
I had said `align_U` in `template_matching.py`, since it is the function
that visibly shifts by whole samples (`torch.roll`). **That was wrong.**
`align_U` aligns templates to each other AFTER they are built, and a
template is already an average of jittered spikes — realigning an average
cannot un-blur it. The blur must be prevented where snippets are measured,
which is in `spikedetect.run`:

    xsub  = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]   # integer positions
    xfeat = xsub @ ops['wPCA'].T                     # <- the replaced line

Implementation uses `<shift(x,-d), w> == <x, shift(w,+d)>`: the PC basis is
precomputed at 41 sub-sample offsets and each spike uses the nearest, so no
snippet is ever resampled. One detail that would have been a silent
disaster: `spikedetect` snippets are centred at `nt//2`, not at `nt0min`.
f0 is chosen once from `wPCA[0]` by `argmax f*|W(f)|` (1507 Hz here), because
at detection time no units exist to select per-unit frequencies from.

### The mechanism works in isolation
On Kilosort's own PC basis, feeding in the same waveform at shifts from
-0.4 to +0.4 samples: feature spread across those shifts drops from 0.1015
to 0.0061, a **16.6x reduction**. Recovered shifts have the right sign and
magnitude (-0.40 -> -0.383, +0.45 -> +0.431; the ~4% shortfall matches the
slope-0.92 bias found in 5x).

### The control that makes the comparison readable
`vanilla_repeat` reproduced the first vanilla run **exactly** — every
per-unit recall, precision and fragment count identical, 210 clusters both
times. Kilosort4 is deterministic on this dataset, so the run-to-run noise
floor is zero and every difference below is real signal rather than variance.
Without this control a 2-point change could not have been interpreted at all.

### The result
2,876 injected spikes, 5 real units relocated 40 channels away, 120 s of real
background:

| unit | recall van -> patched | precision van -> patched | fragments van -> patched |
|---|---|---|---|
| 31  | 0.765 -> 0.765 | 0.502 -> 0.493 | 6 -> 6 |
| 302 | 0.830 -> **0.892** | 0.934 -> **0.853** | 7 -> 8 |
| 408 | 0.679 -> **0.630** | 1.000 -> 1.000 | 10 -> 9 |
| 439 | 0.998 -> 0.998 | 0.993 -> 0.992 | 7 -> 6 |
| 440 | 0.777 -> **0.865** | 0.985 -> 0.988 | 6 -> 5 |

Pooled: overall recall **0.8098 -> 0.8303** (+2.05 points), mean precision
**0.8828 -> 0.8650** (-1.78 points), total clusters 210 -> 205, mean
fragments per unit 7.2 -> 6.8.

**Verdict: a wash.** Two points of recall bought at the cost of nearly two
points of precision, with one clear winner and one clear loser:
- unit 440 genuinely improved — 456 -> 508 spikes found (+52), precision
  slightly up, one fewer fragment.
- unit 408 genuinely got worse — 393 -> 365 (-28) — and it is the unit that
  was most oversplit to begin with, which is the opposite of the prediction.
- unit 302 traded recall for precision (+6.2 / -8.2).
- units 31 and 439 did not move at all.

### The central hypothesis does not survive
The prediction was explicit: sub-sample jitter pollutes the clustering
features, so removing it should reduce oversplitting. **Fragments per unit
went from 7.2 to 6.8.** That is essentially unchanged, and unit 408 — the
worst case at 10 fragments with perfect precision, the textbook jitter
signature — improved by only one fragment while losing recall.

So jitter is NOT the main driver of oversplitting in this recording, even
though both supporting measurements were real and reproducible: the 18%
feature displacement at half a sample (5v) and the 16.6x spread reduction
above. The chain "jitter moves features -> moved features cause splits"
breaks at the second link. Kilosort's clustering is evidently more robust to
that displacement than the geometry suggested, and whatever sets the number
of clusters here is something else — candidates are amplitude variability,
bursting, drift, or the clustering algorithm's own granularity.

### Honest notes
- **The runtime difference is a disk-cache artifact, not a speedup.** First
  vanilla run 1287 s (cold read of the 2.76 GB file), patched 131 s,
  vanilla_repeat 132 s. The patch costs no measurable time, but it does not
  save any either.
- Kilosort's thresholds and its learned PC basis were tuned on unaligned
  data. Changing the feature distribution may perturb behaviour for reasons
  unrelated to jitter, and the mixed per-unit results are consistent with
  that. A fair test of the idea might require re-tuning `Th_universal` and
  `Th_learned` alongside, which has not been done.
- Only 5 injected units on one 120 s dataset. Unit-level differences of
  +52 and -28 spikes are real but this is not a sample size that supports a
  general claim about Kilosort.
- The dataset enforces 120 samples between injected spikes, so it does not
  test collisions between injected units.

### What this means for the plan
The flag-gated patch framework works and the grading harness works — that
infrastructure is the durable result here, and it is what lets any future
change be judged in about two minutes per run on cached data. But
`subsample_align` as it stands does not earn being switched on by default.
Before continuing down this path the next question is whether oversplitting
has a different cause, since that is the problem worth solving and jitter
has now been measured not to be it.


## 5z. What actually caused the splitting: amplitude, not timing

Follow-up to 5y, which found that sub-sample alignment barely changed
fragmentation (7.2 -> 6.8 clusters per injected unit) despite making the
features 16.6x more consistent. The obvious objection is that those two facts
do not sit together: if the features got much tighter, why did the number of
clusters not move? This answers it, and the answer is specific.

### What Kilosort's split decision actually tests
Read from the installed source, `kilosort/swarmsplitter.py`, function
`split`. For each candidate pair of branches it decides merge-or-keep-split
in this order:

    1.  tstat[kk,0] < 0.2                       -> keep split
    2.  refractoriness(spikes1, spikes2)        -> CCG-based decision
    3.  criterion = 2 * (bimod_score < 0.6) - 1 -> merge only if UNIMODAL
    4.  tstat[kk,-1] > 0.15                     -> merge

Step 3 is the geometric heart of it: two groups stay separate if the spikes,
projected onto the direction that best separates them, look BIMODAL. That
immediately explains the paradox. Shrinking the spread along the TIMING
direction cannot merge two groups whose bimodality lies along a DIFFERENT
direction.

### And yes, there is a fixed cluster count baked in
`clustering_qr.cluster()` is declared with `nclust = 200`, and BOTH call
sites (lines 237 and 405) use that default. It seeds 200 clusters per spatial
region with kmeans++ (`kmeans_plusplus(Xg, niter=nclust)`) then iterates.
`nclust` appears ZERO times in `parameters.py`, so it is not a user-visible
setting. Our runs produced 210 and 205 total clusters. The final count is
then reshaped by `hierarchical.maketree` plus `swarmsplitter`, so 200 is a
starting point rather than a hard ceiling -- but it is a hard-coded one.

### The measurement
For each injected unit we know every spike came from the SAME neuron, so
every split is by definition an error. Group the matched spikes by which
output cluster they landed in, then ask what separates the fragments.
Reported as an F-like ratio: between-fragment spread over within-fragment
spread. Script: `demo_what_caused_the_splitting.py`.

| config | unit | fragments | F by AMPLITUDE | F by TIME | F by TRUE SHIFT |
|---|---|---|---|---|---|
| vanilla | 302 | 2 | **34.0** | 0.91 | 0.202 |
| vanilla | 408 | 3 | **112.6** | 0.24 | 0.301 |
| vanilla | 440 | 2 | **309.9** | 6.68 | 1.506 |
| subsample_align | 408 | 3 | **134.6** | 0.00 | 0.816 |
| subsample_align | 440 | 2 | **231.8** | 4.04 | 0.049 |

Medians for vanilla: amplitude **112.6**, time 0.9, true sub-sample shift
**0.301**. Amplitude separates the fragments roughly **370x** more strongly
than the quantity the alignment patch corrects.

**Kilosort is splitting these neurons by spike amplitude.** Not by timing,
not by drift. The alignment patch was correcting a real but almost
irrelevant axis, which is exactly why 16.6x tighter timing features changed
the cluster count by 0.4.

One secondary observation worth keeping: under `subsample_align`, units 31
and 302 dropped to a single real fragment, so the patch did consolidate
those two. Unit 408 stayed at 3. That matches 5y's mixed per-unit picture.

### Why amplitude varies, and the link back to the original design doc
A neuron's spike amplitude drops during a burst and recovers over tens of
milliseconds. That produces exactly the bimodal amplitude distribution step 3
splits on. The handoff document's **burst-history amplitude-recovery curve
and its gating** has been sitting in this log's own "still never built"
section since the beginning, and this is the first time the project has a
MEASURED reason to build it rather than an argument from first principles.

### Note on fragment counts
The counts here (2-3) are lower than 5y's 6-10 because this analysis assigns
each injected spike to at most one output spike globally before counting,
while 5y's fragment count evaluated each cluster independently. The stricter
accounting is the right one for attribution; it does not change 5y's
conclusion, and a 370-fold effect is not sensitive to it.

### Next step this points to
A second flag-gated patch: normalize or regress out each spike's amplitude
before the features are computed, which is precisely what `nuisance_model.py`
pillar 1a was built to do. If amplitude bimodality is what drives the splits,
removing it should collapse the fragmentation -- and unlike the timing
hypothesis, this one has a measured 370x effect size behind it rather than an
18% geometric argument.


## 5aa. The amplitude story does not survive a population check — the axis was right, the cause was wrong

Prompted by the user pointing out that 5z leaned on too few neurons. That was
correct and the objection was decisive: the "370x" figure is a median of
**three** units (302, 408, 440), because the other two injected units never
fragmented and contributed nothing. Three numbers, one session, one animal.
Two cheap checks were run before building anything on top of it, and both
went against the story I told.

### Check 1: is amplitude bimodality a population property?
`demo_amplitude_bimodality_population.py`, using **Kilosort's own
`bimod_score`** copied verbatim from `swarmsplitter.py` so the number is
exactly the one its split decisions act on (its rule: score >= 0.6 keeps a
split, below merges). Across all 443 units with >= 300 spikes in session
20260916_110311:

| percentile | amplitude bimod_score |
|---|---|
| p10 | 0.000 |
| p50 | **0.000** |
| p75 | 0.012 |
| p90 | 0.093 |

**Units reaching the 0.6 split threshold on amplitude alone: 3 of 443 =
0.7%.** Real amplitude distributions in this recording are essentially
unimodal. Median amplitude coefficient of variation is 0.101, p90 is 0.150 —
modest.

### Check 2: is bursting the mechanism?
The handoff document predicts amplitude drops within a burst and recovers, so
amplitude should correlate POSITIVELY with the preceding ISI.

- correlation of amplitude with log(previous ISI): median **-0.083**, the
  WRONG SIGN, and positive in only **19.6%** of units
- amplitude for ISI < 10 ms versus ISI > 100 ms: median **-2.2%**, i.e.
  short-ISI spikes are very slightly LARGER, not smaller; a real drop
  (> 2%) appears in only **14.7%** of units
- median fraction of spikes with ISI < 10 ms: 0.076

**Bursting does not drive amplitude in this population.** Note that the five
units 5z rested on are not special here either: their bimod_scores are
-0.006 to 0.100, indistinguishable from the population median of 0.000.

### Check 3: did my own dataset manufacture the effect?
A real worry: snippets were drawn from the unit's spikes across the whole
210-minute recording, so compressing 210 minutes of drift-driven amplitude
variation into a 120-second dataset could have inflated amplitude spread
beyond anything realistic. Measured directly — amplitude CV across the whole
recording versus the median CV inside a random 120 s window:

| unit | CV whole recording | CV in 120 s | inflation |
|---|---|---|---|
| 440 | 0.0982 | 0.0980 | 1.00x |
| 408 | 0.1009 | 0.0965 | 1.05x |
| 439 | 0.0715 | 0.0650 | 1.10x |
| 302 | 0.0914 | 0.0901 | 1.01x |
| 31  | 0.1333 | 0.1272 | 1.05x |

Median inflation **1.05x**. The dataset did NOT manufacture extra amplitude
spread; the injected amplitude variation is realistic. That artifact
hypothesis is ruled out.

### What was actually wrong with 5z's reasoning
The measurement stands: Kilosort's fragments of one injected neuron do differ
in amplitude far more than in timing. But the inference from it was bad in
two ways.

1. **The F-ratio is partly circular.** If a split is made along the amplitude
   direction for ANY reason, the resulting fragments must differ in
   amplitude. A large F identifies the AXIS of the cut. It cannot establish
   that amplitude structure caused the cut.
2. **The causal story is refuted.** "Bursting makes amplitude bimodal, and
   the bimodality test splits on it" fails at both links: amplitude is not
   bimodal (check 1) and bursting does not drive it (check 2).

So Kilosort is slicing a **unimodal** amplitude distribution. Something is
cutting a continuous distribution into pieces, and bimodality is not the
reason.

### The new live hypothesis
Two mechanisms in the source now look more likely than bimodality, and both
are consistent with cutting a unimodal distribution:

- **Over-seeding.** `clustering_qr.cluster()` seeds **200** clusters per
  spatial region with kmeans++ (`nclust=200`, both call sites use the
  default, absent from `parameters.py` so not user-settable). An over-seeded
  k-means places boundaries along the direction of greatest within-unit
  spread — which for these spikes is amplitude. That produces
  amplitude-separated fragments from a unimodal distribution, exactly what is
  observed.
- **`swarmsplitter.split`'s FIRST criterion**, `tstat[kk,0] < 0.2`, keeps a
  split without ever consulting bimodality. Only if that passes does the
  bimodality test run at all.

### Consequence for the planned patch
`amplitude_normalize` is still the right experiment, but for a different
reason than stated in 5z. Under the new hypothesis, dividing each snippet by
its own magnitude collapses the largest within-unit variance direction, so an
over-seeded k-means has nothing to cut along. The honest risk, written down
before the result: the over-seeding may simply cut along whatever direction
is next-largest, in which case normalization changes nothing. There is also
an implementation risk — `tF` feeds template construction as well as
clustering, so removing scale may degrade template matching for reasons
unrelated to the hypothesis.

Normalization is taken over channels AND time jointly, never per channel: a
spike is one event with one amplitude, and per-channel normalization would
destroy the spatial footprint, the one feature this project has measured to
be decisive (5q).

### Dataset widened in response
`build_small_hybrid_dataset.py` now selects source units automatically rather
than by hand: **17 units** (18 requested, one found no free destination slot)
with >= 800 spikes, sampled across the whole probe, **4,689 injected spikes**,
destinations laid out on a 20-channel stride so no two injected units can
overlap. Destination peak channels span ch30-ch349. That replaces the
five-hand-picked-units base that this entire line of reasoning rested on.


## 5ab. A tiered benchmark: verified-good neurons placed into deliberately easy and hard situations

Built at the user's direction, replacing the all-easy dataset. The point:
a clean neuron on a quiet channel is found by every version of the algorithm,
so an average over such units cannot discriminate. Difficulty is stratified on
purpose and recorded per unit, so a change can be judged where it is meant to
help.

The design rule the previous dataset violated: injecting into busy or
colliding situations is GOOD, provided it is INTENTIONAL and LABELLED. The
earlier version had 12.3% of its spikes colliding by accident, which is the
same thing done invisibly and therefore uninterpretable.

**Sources are verified first, always.** All 13 are KSLabel=good with
ContamPct <= 10%. Difficulty comes from WHERE and WHEN verified spikes are
placed, never from using a dubious source.

**Mutual independence is enforced, not hoped for.** KSLabel=good says a
cluster looks like a single unit on its own; it does not rule out two
clusters being halves of one neuron Kilosort oversplit. That is fatal for the
pair tier, where correct behaviour would then be to MERGE. Selection now
requires every pair of sources to be >= 150um apart with Kilosort template
similarity <= 0.20, which rejects 114 of 131 candidates and leaves 13 whose
closest pair is 160um apart at similarity 0.000.

**The four tiers** (3,449 spikes total):

| tier | placement | local background density |
|---|---|---|
| easy | quietest probe regions, guarded times | 1,744-1,883 |
| noisy_channel | busiest probe regions, guarded times | 21,659-26,185 |
| collision | quiet region, times deliberately 4-25 samples from a resident spike | 1,910-3,601 |
| pair | two different neurons, destination peaks 3 channels apart | 6,058-8,488 |

**The pair tier is scored differently** -- correct behaviour is TWO clusters,
so merge errors are reported separately. A merge error can look like
excellent recall (fusing two neurons finds all spikes of both), so averaging
it in would read as success.

**The selection was also checked by eye** ().
That found a mistake worth recording: plotting Kilosort's templates.npy scaled
by GAIN_TO_UV gave 0.1-0.3 uV spikes, because those templates are
normalized -- a caveat already in this log, walked into anyway. Replaced with
real averaged waveforms from the raw file: 94-345 uV peak-to-peak, every
trough negative, footprints 26-77um, every unit a single smooth deflection
decaying with distance. Pair members are visibly different in shape, which
matters or the pair tier would test the wrong thing.


## 5ac. The tiered benchmark's verdict: every patch makes Kilosort worse, with exactly one real exception

The tiered dataset (5ab) run through all configurations. `vanilla_repeat`
reproduced `vanilla` **exactly** -- same recall, precision, fragment counts
and cluster count -- so the run-to-run noise floor on this dataset is zero and
every difference below is real signal.

### Recall by tier

| tier | vanilla | subsample_align | amplitude_normalize | both |
|---|---|---|---|---|
| easy | **0.9303** | 0.9129 | 0.7786 | 0.7512 |
| collision | **0.8746** | 0.8616 | 0.8390 | 0.8275 |
| noisy_channel | **0.8037** | 0.6584 | 0.7926 | 0.5749 |
| pair | **0.5696** | 0.4374 | 0.4290 | 0.4299 |
| **overall** | **0.7773** | 0.6960 | 0.6882 | 0.6293 |

### Precision by tier

| tier | vanilla | subsample_align | amplitude_normalize | both |
|---|---|---|---|---|
| easy | 0.5383 | **0.5429** | 0.4770 | 0.4628 |
| **collision** | 0.5887 | **0.6576** | 0.4420 | 0.4424 |
| noisy_channel | **0.2696** | 0.2453 | 0.2575 | 0.1307 |
| pair | **0.4701** | 0.2856 | 0.2908 | 0.2834 |
| **overall** | **0.4669** | 0.4215 | 0.3610 | 0.3262 |

**Stock Kilosort wins recall in every single tier, and wins precision in three
of four.** `amplitude_normalize` is clearly harmful, which is consistent with
5aa having already refuted the hypothesis behind it. Combining the two patches
is worse than either alone.

### The one real win, and the tiers are what exposed it
`subsample_align` improves precision in the **collision** tier: **0.6576
versus 0.5887**, nearly 7 points. That is the one tier where sub-sample timing
should matter most -- spikes deliberately placed 4-25 samples from a resident
spike, where getting the alignment right is what separates the two waveforms.
The effect is real (zero noise floor) and it is in exactly the predicted
place.

It was invisible in the previous all-easy dataset, where everything averaged
to "slightly worse". That is precisely the argument for stratifying difficulty
rather than averaging over it, and the tiers earned their cost on this one
result.

It still does not make the patch worth enabling: the same configuration loses
1.3 points of collision recall and 8 points of overall recall.

### What the tiers revealed about difficulty
The intended gradient came out, and not in the order I would have guessed:

| tier | vanilla recall | vanilla precision |
|---|---|---|
| easy | 0.9303 | 0.5383 |
| collision | 0.8746 | 0.5887 |
| noisy_channel | 0.8037 | 0.2696 |
| **pair** | **0.5696** | 0.4701 |

**The pair tier is by far the hardest** -- two different neurons 3 channels
apart cost Kilosort 36 points of recall relative to the easy case. Deliberate
collisions were *easier* than a busy channel, which is worth remembering given
how much effort this project spent on collisions (and consistent with 5w
finding tight collisions rare in this recording).

### Merge errors: none, by any configuration
Both pairs were kept as separate clusters by every configuration, including
vanilla. So Kilosort does not merge nearby distinct neurons here -- it
*fragments* them instead (2.75 fragments per pair unit). The failure mode on
closely-spaced neurons is oversplitting with poor recall, not merging.

That matters for this project's direction. Sections 5q and 5t built the
footprint machinery to tell apart neurons that a single channel confuses, on
the premise that wrongly merging them was the risk. On this benchmark that
risk does not materialize for Kilosort itself.

### Honest limits
- 13 units, one 120 s dataset, one session, one animal.
- The patches were tested against Kilosort's thresholds and learned PC basis,
  both tuned on unaligned, unnormalized data. A fair test of the underlying
  ideas might require retuning `Th_universal`/`Th_learned` alongside, which has
  not been done. The measurement here is "does switching this flag on help as
  Kilosort currently stands", and the answer is no.
- Absolute precision is low across the board (0.27-0.59) because the
  background contains hundreds of real neurons whose spikes legitimately
  enter the matched clusters. Only RELATIVE comparisons between configurations
  are meaningful here.

### Status of the Kilosort integration work
Three patches built, all flag-gated, all measured against ground truth:
**none should be enabled.** The infrastructure stands -- a tiered benchmark
with verified-good, mutually-independent sources, deliberate difficulty, and a
scorer that reports per tier plus merge errors, running in about two minutes
per configuration. The hypotheses it was built to test have all failed, which
is a result rather than a loss, and it is the first time this project could
have known that before shipping something.


## 5ad. Why the missed spikes were missed: 84% are a CLUSTERING failure, not a detection failure

Prompted by the question "were the missed spikes missed for a specific
reason?" -- which had never been asked. Recall 0.78 is a score, not a
diagnosis, and the fix for each possible cause lives in a different part of
the algorithm. Script: `demo_why_were_spikes_missed.py`.

Every injected spike is classified into one of three outcomes:

  **found**              assigned to the cluster matched to its unit
  **detected_misfiled**  a spike IS present at the right time, but filed under
                         a different cluster. Detection worked; clustering
                         scattered it.
  **not_detected**       no spike of any cluster near that time. Detection
                         never saw it.

### The headline, on stock Kilosort

| outcome | count | share |
|---|---|---|
| found | 2,396 | 69.5% |
| **detected, misfiled** | **886** | **25.7%** |
| never detected | 167 | 4.8% |

Of the 1,053 failures, **886 (84%) were detected and then misfiled**, and only
167 (16%) were never detected at all. Kilosort's detection stage sees 95% of
the injected spikes. The loss is almost entirely in deciding which cluster
they belong to.

**This reframes the whole integration effort.** Work aimed at the detection
stage can address at most 4.8% of injected spikes. Work aimed at clustering
addresses 25.7%. Those are not close.

### Does amplitude explain it?
Partly, and it is a clustering effect rather than a threshold effect, since
these spikes are being detected:

| injected amplitude | recall |
|---|---|
| 65-117 uV | 0.532 |
| 117-138 uV | 0.619 |
| 138-161 uV | 0.691 |
| 161-206 uV | 0.755 |
| 206-609 uV | 0.877 |

Median amplitude by outcome: found 157 uV, misfiled 136 uV, never detected
124 uV. Smaller spikes are both harder to detect and much harder to file
correctly -- but even the largest quintile loses 12%.

### Does collision closeness explain it? No.
Recall across the deliberate-collision tier, by how close the resident spike was:

| offset | recall |
|---|---|
| 0-8 samples | 0.699 |
| 8-14 | 0.719 |
| 14-20 | 0.721 |
| 20-26 | 0.664 |

Flat. Collision distance does not predict misses, consistent with 5w finding
tight collisions rare and 5ac finding the collision tier easier than a busy
channel. Collisions have absorbed a lot of this project's effort and keep
coming back as a non-problem on this data.

### By tier (vanilla)

| tier | misfiled | found | not detected | recall |
|---|---|---|---|---|
| easy | 110 | 696 | 9 | 0.854 |
| noisy_channel | 172 | 601 | 35 | 0.744 |
| collision | 210 | 537 | 15 | 0.705 |
| **pair** | **394** | 562 | **108** | **0.528** |

The pair tier fails hardest in both ways at once, and it is the only tier
where "never detected" is substantial (108). Two neurons three channels apart
degrade detection as well as assignment.

### What this says about the remaining pillars
- **Temporal whitening** was never tested inside Kilosort. It is a
  DETECTION-stage improvement, so its ceiling here is the 4.8% never-detected
  bucket. Even a perfect detector gains less than five points. That makes it
  the lowest-value of the remaining ideas, which is worth knowing before
  building it.
- **Footprint-based clustering** targets the 25.7% misfiled bucket directly.
  The specific proposal -- cluster only within matched-footprint events -- is
  aimed exactly at where the failure is. This is now the highest-value
  untested idea in the project, and the first one whose target has been
  measured rather than assumed.
- `subsample_align` makes misfiling WORSE (34.9% versus 25.7%), which explains
  its poor showing in 5ac in mechanistic terms rather than as a bare score.

### Caveat
The per-tier recalls here run slightly below 5ac's because this analysis
assigns each injected spike to at most one output spike globally, in time
order, before classifying. The stricter accounting is the right one for
attribution; the 84/16 split is far too large to be sensitive to it.


## 5ae. Coarse sliding alignment before the wavelet stage: a real improvement to the patch, and a correction to my own diagnosis

Prompted by "what about sliding template correction first before wavelet
correction?" The post-hoc pipeline has always done coarse-then-fine
(`wavelet_features.coarse_then_fine_shift`); the Kilosort port dropped the
coarse stage, and I never checked whether that mattered.

### The diagnosis I gave was measured at the wrong stage
I measured Kilosort's output spike times against the known injected times and
found 55.2% sitting more than half a sample off, 20.1% more than two samples,
5th-95th percentile -4.5 to +5.4 samples. From that I concluded the patch's
+/-0.5 clamp was being violated for most spikes.

**That was the wrong quantity.** `spike_times.npy` is produced by
`template_matching.extract` at the END of the pipeline. The patch operates on
`xy[:,1]` inside `spikedetect.run`, which is a different, earlier index. When
the coarse stage was actually built and run, it reported a **median integer
correction of 0.00 samples** (per-batch medians ranging 0 to 1) -- Kilosort's
detection index is already close to optimally aligned against the average
spike shape, which is unsurprising since detection itself maximizes a template
match.

So the mechanism I proposed for `subsample_align`'s poor showing is not
established. The misfiling increase from 25.7% to 34.9% remains real and
remains unexplained.

### The coarse stage helps anyway, substantially, on the hardest tier
Even with a median correction of zero, a minority of spikes shift by one or
more samples, and that minority matters. `coarse_then_align` versus
`subsample_align`:

| tier | recall | | precision | | fragments | |
|---|---|---|---|---|---|---|
| | sub | coarse | sub | coarse | sub | coarse |
| easy | 0.9129 | 0.9167 | 0.5429 | 0.5402 | 1.67 | 1.33 |
| collision | 0.8616 | 0.8683 | **0.6576** | 0.4749 | 3.67 | 3.00 |
| noisy_channel | 0.6584 | **0.5886** | 0.2453 | 0.1411 | 2.33 | 2.00 |
| **pair** | 0.4374 | **0.5297** | 0.2856 | **0.4331** | 2.75 | 2.25 |

The pair tier -- the hardest, two neurons three channels apart -- gains **+9.2
points of recall and +14.8 points of precision**, and fragmentation drops in
every tier. That is the largest improvement any patch change has produced in
this project. The noisy-channel tier gets worse, and collision precision drops
sharply (0.6576 to 0.4749), so it is a trade rather than a clean win.

### It still does not beat stock Kilosort

| tier | vanilla | best patch | 
|---|---|---|
| easy | **0.9303** | 0.9167 (coarse) |
| collision | **0.8746** | 0.8683 (coarse) |
| noisy_channel | **0.8037** | 0.6584 (subsample) |
| pair | **0.5696** | 0.5413 (coarse+ampnorm) |

Stock Kilosort still wins recall in all four tiers. Five patch configurations
have now been built and measured; none is worth enabling.

### What this points at
The coarse stage closing most of the pair-tier gap (0.4374 to 0.5297 against
vanilla's 0.5696) while cutting fragmentation says the remaining loss on
closely-spaced neurons is about ASSIGNMENT, not timing -- consistent with 5ad
finding 84% of all failures are detected-but-misfiled. Timing work is close to
exhausted; the open lever is the clustering decision itself.


## 5af. Footprint-aware clustering: the first patch to beat stock Kilosort on anything

The user's proposal -- cluster only within events whose spatial footprints
match -- implemented and measured. It is the first idea in this project whose
target was quantified before it was built (5ad: 84% of failures are
detected-but-misfiled, i.e. clustering, not detection), and it is the first
patch of six to beat stock Kilosort on a tier.

### Implementation, and a failure worth recording
The obvious version -- append footprint columns to `Xd` in
`clustering_qr.get_data_cpu` -- crashes:

    RuntimeError: shape '[-1, 6]' is invalid for input of size 70

because `clustering_qr.run` uses the SAME `Xd` twice: once to cluster, and
again to build each cluster's template, where it reshapes each row to
(n_channels, n_pcs). That is an architectural constraint, not a typo -- the
matrix fed to clustering IS the matrix templates are averaged from.

Fixed by letting the footprint reach the clustering step ONLY: computed in
`get_data_cpu` where the layout is known, stashed, and concatenated inside a
patched `cluster()` where the augmented matrix stays local to the graph
build. `Xd` is returned untouched, so template construction sees exactly what
it always saw. That is also a cleaner statement of the idea -- group spikes
using footprint information without distorting what a template is.

The appended block is the L2-NORMALIZED per-channel energy vector, scaled to
`FOOTPRINT_WEIGHT` times the median row norm of the existing features.
Normalizing makes it a pure shape descriptor, so a loud and a quiet spike
from one neuron stay close (encoding raw amplitude is already known to be
harmful, 5ac).

### The result

| tier | vanilla | footprint w=1.0 | **footprint w=3.0** |
|---|---|---|---|
| **recall** | | | |
| easy | 0.9303 | 0.9303 | **0.9303** |
| collision | **0.8746** | 0.8348 | 0.7428 |
| noisy_channel | **0.8037** | 0.7409 | 0.7591 |
| **pair** | 0.5696 | 0.4356 | **0.6644** |
| **precision** | | | |
| easy | 0.5383 | 0.5384 | 0.5368 |
| collision | 0.5887 | **0.6506** | 0.6187 |
| noisy_channel | **0.2696** | 0.2927 | 0.1893 |
| **pair** | 0.4701 | 0.2941 | **0.5188** |

**On the pair tier -- two different neurons three channels apart, the hardest
case in the benchmark -- weight 3.0 beats vanilla on recall by +9.5 points
AND on precision by +4.9 points simultaneously**, with fewer fragments (2.50
vs 2.75). Not a trade: both measures move the right way. The easy tier is
untouched (0.9303 both). No configuration produced a merge error.

The dose-response is systematic and in the predicted direction: weight 1.0
gives pair recall 0.4356, weight 3.0 gives 0.6644. A spurious effect would
not order itself that way.

### The cost, and why it makes mechanistic sense
Weight 3.0 loses 13.2 points of recall on the COLLISION tier (0.7428 vs
0.8746) and 4.5 on the noisy channel. That is exactly what should happen: when
two spikes overlap in time, the measured per-channel energy is contaminated by
the other spike, so a footprint-weighted clustering is being fed a corrupted
descriptor and weighting it heavily amplifies the error. Footprint helps where
neurons are separated in SPACE and hurts where they overlap in TIME.

That is a coherent, falsifiable story rather than a shrug, and it suggests the
obvious refinement: apply footprint weighting only to spikes that are not
collisions. The obstacle is that this project's collision detector failed
(5w, AUC 0.62), so there is currently no reliable way to know which spikes to
exempt.

### Status
Six patch configurations built and measured. Five are worth nothing.
`footprint_cluster_strong` is the first with a real, mechanistically explained
win on the hardest case, at a real cost elsewhere. It is not yet a default --
overall it still trails vanilla because the collision and noisy tiers
dominate the average -- but it is the first time the direction has been
validated rather than argued.


## 5ag. What the low precision numbers actually mean: resident neurons, not noise

Gil asked the obvious question nobody had asked: easy-tier precision is 0.538,
so is half of what Kilosort detects not from the unit? And then the sharper
follow-up -- the easy tier injects into a *quiet* channel, so what other
neuron is even there?

### "Quiet" was a ranking, never an emptiness test
`build_tiered_hybrid_dataset.py:206` is
`quietest = argsort(where(valid, density, inf))`. It picks the lowest-density
channels **available on this probe**. On a striatal Neuropixels shank there is
no empty channel. Measured within 60 um of the three easy destinations:

| easy unit | dest ch | clusters within 60 um | their spikes in 120 s |
|---|---|---|---|
| 413 | 288 | 11 | 1,860 |
| 440 | 223 | 5 | 1,273 |
| 461 | 188 | 8 | 1,591 |

The quietest spot on the probe holds 5-11 neurons firing ~10-15 spikes/s
between them. `QUIET_GUARD = 75` guards the injection *times*, so our spikes
never overlap a resident's -- but residents keep firing at every other moment
and land in the same cluster.

### The measurement
For every "foreign" spike in a matched cluster (any spike the scorer counts
against precision), ask whether the ORIGINAL Kilosort run on the untouched
session has a spike at the same moment (`time + BG_START`) on a cluster whose
peak channel is within 60 um. If yes, that spike existed before we touched
anything -- a real detection, not noise.

**5,868 of 7,283 foreign spikes (80.6%) are pre-existing local real spikes,
against a chance rate of 0.039 per window (~20x chance).**

| tier | precision as scored | foreign that are pre-existing | precision counting those as real |
|---|---|---|---|
| easy | 0.5383 | 42% | **0.8026** |
| collision | 0.5887 | 79% | **0.9137** |
| noisy_channel | 0.2696 | 70% | **0.8071** |
| pair | 0.4701 | 57% | **0.8994** |

Overall 0.4669 -> **0.8591**.

### THE BASE-RATE TRAP, COMMITTED A THIRD TIME
The first version of this script asked "is there any original spike within +-10
samples?" **anywhere on the probe**. The window holds 266,023 spikes in 3.6M
samples, so a 21-sample window catches **1.55 spikes by chance** -- the answer
was guaranteed yes, and it returned a meaningless 96.2%. Restricting to
clusters within 60 um drops chance to 0.039 and the observed rate to 80.6%,
which is now interpretable. This is mistake #2 in section 6 made for the third
time in this project. **Any "is X near Y" statistic must carry its chance rate
in the same table as the observed rate**, or it cannot be read.

### What this does NOT excuse
1. **It is still a genuine isolation failure.** The residents are DIFFERENT
   neurons. Kilosort built one cluster from two or three cells. That is a merge
   problem rather than a noise problem -- real, and exactly what the footprint
   work targets -- but it is not the problem the raw number implies.
2. **~19% stays unexplained.** 1,415 foreign spikes have no local pre-existing
   counterpart. Could be genuine new detections (injection changes the local
   signal, and the hybrid run whitens 120 s on its own) or real noise. Not
   separable with what exists now.
3. **"Real spike" is not "clean neuron."** The largest single contributors are
   frequently `mua` at 54%, 63%, 72%, even 224% contamination (cluster 159).
   Those sources are themselves mixtures.
4. **Precision is averaged UNWEIGHTED over units.** Units 163 and 186 score
   0.0202 and 0.0318 -- each absorbed into a large resident cluster -- and carry
   the same weight as units at 0.97. That is most of why the pair tier reads
   0.4701.

### The variance argument for a bigger run
Within the easy tier alone: unit 461 scores 0.8618, unit 440 0.4490, unit 413
0.3041. At n=3 per tier the tier means are close to meaningless. This is the
strongest quantitative argument yet for the larger-neuron benchmark.

### How to read every precision number in this log
Precision here measures **"what fraction of this cluster is the neuron we
planted, at a location that already hosted other neurons"**. It is a
merge-with-resident measure, not a noise measure, and it is a severe lower
bound on isolation quality. Only RELATIVE comparisons between configurations
were ever meaningful, which section 5ac already said -- this section explains
why, with numbers.

Scripts: `demo_what_is_the_contamination.py`; outputs
`contamination_identity_local.csv`, `contamination_sources_local.csv`.


## 5ah. Redesigning the benchmark: 50 candidate units, and sites chosen by amplitude

Gil asked for a bigger, better-designed rerun: (a) more units, each good and
provably isolated from every other selected unit, with all of them shown
before anything runs; (b) sites chosen deliberately -- easy = low noise, few
events, no large units nearby; hard = the opposite; **collision and pair also
quiet**, because there the challenge is separating units and channel noise must
not be confounded with it.

### The Trodes <-> Kilosort channel mapping (asked for, now established)
The `.rec` header holds 384 `<SpikeNTrode>` entries in `.dat` column order with
ids descending 1466 -> 1083, each carrying pad coordinates that match
`channel_positions.npy` exactly (verified, not assumed):

**`ntrode = 1466 - ks_channel`**  and  **`ks_channel = 1466 - ntrode`**

So Gil's eyeballed quiet stretch, Trodes 1275-1211, is **KS channels 191-255**.

### Gil's visual read was correct, and it identified the right criterion
| measure | Trodes 1275-1211 | rest of probe | percentile |
|---|---|---|---|
| max neighbour amplitude | 47.1 uV | 86.0 uV | **p12** |
| sum neighbour amplitude | 180 uV | 353 uV | p15 |
| event density | 3,722 | 7,963 | p18 |
| p99.9 voltage | 67.1 uV | 72.7 uV | p22 |
| noise (MAD sigma) | 18.6 uV | 19.7 uV | p29 |

Quiet on every measure, and most strongly on neighbour AMPLITUDE (p12) rather
than on raw noise (p29). The criterion he named is the discriminating one.

### "No units nearby" is impossible here; amplitude is the right measure
`survey_probe_sites.py` required no units nearby and found **zero** qualifying
channels of 334 eligible. This probe has 308 clusters with >= 100 spikes over
384 channels -- roughly one unit per channel -- and the most isolated channel on
the whole shank is 52 um from the nearest unit. Striatum has nowhere empty.

Gil's correction: what matters is not whether neighbours exist but whether they
are LARGE, since a small neighbour is just background. `survey_probe_amplitude.py`
measures that directly -- every resident unit's real averaged waveform in
microvolts, projected onto every channel, in one pass over 60 s of sampled
chunks. The contrast available is **20x**: the quietest eligible site has a
largest-neighbour of 22.9 uV (KS 166 / Trodes 1300), the loudest 466.5 uV
(KS 139 / Trodes 1327).

Criteria agreement (Spearman): noise vs max-neighbour +0.54, density vs
max-neighbour +0.40, p99.9 vs max-neighbour +0.79. Related but far from
interchangeable, so they must be applied jointly.

### Source units: 50 candidates, 32 clean
`select_source_units.py` replaces the >= 150 um mutual-separation proxy with a
direct test. Two clusters that are really halves of ONE neuron must show a
**refractory dip in their cross-correlogram** -- a single cell cannot fire twice
within ~1 ms. So distance relaxes to 40 um and the CCG test carries the
guarantee.

Result: **50 accepted of 143** quality candidates (rejected 59 too close, 34 too
similar, 0 by CCG dip). Final-set guarantees: closest pair 40 um, highest
similarity 0.187, **worst CCG ratio 0.60** against a 0.30 dip threshold -- so no
accepted pair is one neuron. 11 pair tests lacked power (< 50 flank counts) and
distance carried those.

Then shape flags, because the automatic filters cannot catch a cluster that is
real but is not a somatic spike. 18 of 50 flagged, **32 clean**:
- **broad-positive (16)**: positive peak >= 2x the trough with half-width
  >= 0.40 ms -- the axonal/fibre-tract signature. These cluster at the probe
  tip (channels 13-116), consistent with that part of the shank sitting outside
  striatum. Not somatic spikes; excluded.
- **single-channel (2)**: units 251 and 195, only the peak channel above 25% of
  max amplitude. A real soma is seen by several sites.
- **few-spikes (3)**: under 1,000 spikes, too few for a trustworthy average.

32 clean units spanning channels 5-367 is 2.5x the previous 13.

### The feasibility constraint that shapes the whole experiment
At p35 cuts (noise <= 18.8 uV, max neighbour <= 60 uV) and 22-channel site
spacing, one recording holds **8 quiet sites and 9 loud ones**. Easy, collision
AND pair all draw from the quiet pool, so a single 120 s recording cannot hold
many more than the previous 13 units. Getting robust statistics therefore means
**several replicate recordings** (different `BG_START` windows, so backgrounds
differ too) rather than cramming units together.

Quiet sites: KS [166, 230, 272, 205, 352, 108, 66, 294] = Trodes
[1300, 1236, 1194, 1261, 1114, 1358, 1400, 1172].
Loud sites: KS [139, 71, 40, 335, 101, 303, 271, 195, 357] = Trodes
[1327, 1395, 1426, 1131, 1365, 1163, 1195, 1271, 1109].

Scripts: `survey_probe_sites.py` (count-based, kept as the negative result),
`survey_probe_amplitude.py`, `select_source_units.py`. Outputs:
`probe_amplitude_survey.csv`, `probe_amplitude_sites.json`,
`selected_source_units.csv`, `selected_source_units.png`, `source_unit_audit.csv`.


## 5ai. The v2 benchmark as built, and four silent placement failures

The dataset Gil asked for, built and verified but not yet run.

### What exists
`build_tiered_v2.py <rep>` for rep 0-3, writing
`D:\Gil\spike_sorting_agent\hybrid_v2_rep{0,1,2,3}\`.

| | old benchmark | v2 |
|---|---|---|
| placements | 13 | **52** |
| distinct units | 13 | **49** |
| ground-truth spikes | 3,449 | **15,013** |
| per tier | 3/3/3/4 | **12 easy / 12 hard / 12 collision / 16 pair** |
| replicates | 1 | **4**, on four different windows of the session |

Site amplitudes achieved (largest resident neighbour, measured in uV):

| tier | median | range |
|---|---|---|
| pair | 31.7 | 23-70 |
| easy | 38.4 | 23-66 |
| collision | 47.4 | 23-70 |
| **hard** | **333.8** | **191-467** |

Quiet and hard do not overlap, and easy/collision now sit on comparable sites
so collision difficulty is temporal only -- which was the point of moving them.

### FOUR PLACEMENT BUGS, ALL OF WHICH PRODUCED A PLAUSIBLE DATASET
This is the lesson worth keeping. None of these raised an error; each produced
a dataset that looked fine and was quietly wrong.

1. **Tier ordering.** Placing easy -> hard -> collision -> pair made the pairs
   vanish entirely: hard sites at channels 71 and 335 sat within 22 channels of
   every remaining quiet candidate. Fixed by placing most-constrained first
   (pair -> easy -> collision -> hard) and by **asserting every tier got its
   full count**, which caught the next three failures immediately.
2. **Ordering bias.** Walking the quiet pool quietest-first gave the
   first-placed tier the quietest sites and later tiers louder leftovers --
   easy 31.5 uV median against collision 45.2 uV, a site difference confounded
   with the tier label being measured. Fixed by packing best-first (which makes
   the set large) then SHUFFLING the packed set (which removes the bias). The
   loud pool is deliberately left sorted: only `hard` draws from it, so there is
   no competition to debias and taking the loudest is the whole point.
3. **Unchecked partner slot.** A pair's partner sits at a fixed channel gap, so
   it must be CHECKED for quietness rather than chosen for it. One partner
   landed on a 119 uV channel -- in the tier whose entire purpose is that the
   only difficulty is separating two neurons.
4. **Destructive search.** The pair loop consumed a new unit pair on every
   failed placement, so one unplaceable pair burned the whole 49-unit pool and
   left the other tiers nothing. The binding constraint is the SITE, not which
   units were handed to the pair; the search now runs over anchor channels and
   tries both gap directions.

**Carry this forward: the v1 benchmark was never checked for silent gaps of
this kind.** Its results in section 6 should be read with that in mind.

### The expert-review correction on unit shape
The first shape filter flagged on waveform polarity (positive peak >= 2x trough,
wide half-width), rejecting 18 of 50 as putative fibre-tract signals. Gil
reviewed the figure and overruled it -- those are fine -- and rejected the one
unit the rule had passed, 403. Comparing 403 to accepted units, the real
difference is spatial: a diffuse smear over 16 channels, where unit 428 has
nearly identical polarity and half-width but a tight localized blob. Footprint
compactness separates them cleanly (403 has 12 channels above 25% of peak,
everything else <= 10). **The threshold is calibrated against one expert
judgement on one probe, not derived**, and is documented as such in the code.

### Status
Blocked only on GPU availability. `bash run_all_v2.sh` runs 9 configurations x
4 replicates (~70 min), refuses to start while Trodes is on the GPU, and is
resumable. Then
`python run_v2_comparison.py --compare <configs>` reports recall and precision
per tier pooled across replicates, the spread ACROSS replicates, per-replicate
tables, merge errors, and a Spearman correlation of recall against each
measured site property -- which is what will finally separate "hard because the
neighbour is loud" from "hard because the neighbourhood is busy".


## 5aj. The benchmark as actually run: 33 hand-reviewed units, remote pairs

Sections 5ah and 5ai describe a 49-unit selection that was SUPERSEDED before
any run. Two corrections from Gil replaced it, and this section is the one to
trust.

### Kilosort's label is not a substitute for looking at the unit
Gil had been reviewing units by hand in the unit review tool
(`scripts/build_review_html.py <session_id>`, which had never been run for this
session). Those verdicts live in the published artifact's database, not in any
file in the repo, which is why nothing here was reading them:
`claude.use('db')` -> `db.collection('verdicts')` on
https://claude.ai/code/artifact/74dec064-c86a-46a3-8c03-f112ee4ecb05
Now exported to `outputs/manual_verdicts_20260916_110311.csv`.

**198 units labelled: 111 good, 74 mua, 13 noise.** Checking the 49-unit
selection against them:

| Gil's verdict | count of the 49 |
|---|---|
| good | 23 |
| mua | 12 |
| **noise** | **4**  (127, 134, 195, 251) |
| not yet reviewed | 10 |

**Unit 461, labelled mua, had been placed as an EASY-tier source.** The
benchmark would have been scoring recall on units its own operator calls noise.
This is the same failure as the first benchmark's mua sources (5ab), caught a
second time by the same means: asking.

It cuts both ways -- **12 of the pool are `KSLabel == 'mua'`**, rescued by the
review. Neither label dominates the other; the manual one is simply made with
more information (ACG and waveform, not just ContamPct).

### Also: `cluster_group.tsv` is NOT curation
It is a byte-identical copy of `cluster_KSLabel.tsv` (0 of 472 rows differ).
`cluster_info.tsv` has a different `group` column with 24 `noise` labels, but
that file is written by
`scripts/build_cluster_info_from_classification_session.py` -- an automated
classification, not Phy curation. **No Phy curation exists for this session.**

### Donor filters: a different question from the verdict
`good` says the unit is a real isolated neuron. A DONOR must additionally
yield clean snippets and a usable footprint, and a unit can be genuinely good
while failing either:
- **contamination <= 10%.** Snippets are drawn at random from the cluster, so a
  43%-contaminated cluster (unit 51) donates 43% wrong waveforms and the
  "ground truth" stops being true.
- **>= 3 channels above 25% of peak.** The pair tier needs two footprints to
  overlap; units 99 and 61 are single-channel, which makes that test vacuous.

Both run BEFORE the independence search. Filtering afterwards discards the
probe location along with the unit; doing it in the right order gained 4 units
(29 -> 33).

### The pair tier was pairing each unit with its NEIGHBOUR
Gil raised this and it was a real bug. The pool is sorted by peak channel and
the code took `order[idx]` and `order[idx+1]`, so pairs came out **51 um apart
at similarity 0.197**. Two clusters that close could be halves of one oversplit
neuron -- in which case MERGING them is arguably correct and the tier scores
backwards. v1's 150 um rule existed for exactly this; relaxing the pool-wide
filter to 40 um when the CCG test arrived was right for every other tier (those
units go to separate sites and are never tested against each other) and wrong
for this one.

The partner is now SEARCHED FOR: `>= 150 um` and similarity `<= 0.05`. All
eight pairs now sit **181-242 um apart at similarity 0.000**. The CCG test is
kept alongside -- distance is a structural guarantee, the CCG test is
statistical and can lack power on low-rate units.

### Final composition
| | value |
|---|---|
| donor units | **33**, every one hand-labelled `good` |
| contamination | 0.0 - 9.9% |
| placements | 52 across 4 replicates |
| ground-truth spikes | 15,086 |
| per tier | 12 easy / 12 hard / 12 collision / 16 pair |
| quiet-tier sites | 23-70 uV largest neighbour |
| hard-tier sites | 223-466 uV largest neighbour |
| pair donor separation | 181-242 um at similarity 0.000 |
| units in >1 tier | 18 of 33 (within-unit tier comparison) |

### Two operational traps hit today
1. **Trodes and Kilosort cannot share this GPU.** During extraction Trodes held
   5,900 MiB of 6,144 at 100% utilisation. Kilosort still STARTS (it needs only
   ~2.8 GB) but is compute-starved: 0 of 60 batches after 4 minutes, against
   ~2 minutes for a whole run. `run_all_v2.sh` refuses to start if Trodes is on
   the GPU; `FORCE=1` overrides, which is correct once Trodes is merely open and
   idle (599 MiB / 37% is a UI, not a workload). Check disk I/O to tell
   extracting from idle -- 0 MB/s read means finished.
2. **`git add -A` from a SUBDIRECTORY stages the whole repository.** One such
   call swept a 2.6 GB `.bin` and ~2 GB of parquet/pkl into a commit. GitHub
   hard-rejects files over 100 MB, so every push that day was impossible --
   and it HUNG rather than erroring, which looked exactly like a slow network.
   Diagnose with
   `git rev-list origin/<branch>..HEAD --objects | git cat-file --batch-check=...`
   The nine original commits are preserved on `backup-before-bigfile-fix`.

Scripts: `select_source_units.py` (now reads the manual verdicts),
`build_tiered_v2.py` (remote-pair search), `run_all_v2.sh`,
`run_v2_comparison.py`.


## 5ak. THE v2 RESULTS — all 36 runs, and the first statistically real win

All 9 configurations x 4 replicates completed (36/36, no failures, ~170 s
each). 468 unit-placements scored. `outputs/v2_scores.csv`,
`outputs/v2_paired_vs_vanilla.csv`, `analyse_v2_results.py`.

### FIRST: the noise floor is no longer zero
In round one `vanilla_repeat` reproduced `vanilla` exactly, which is what
licensed treating every difference as signal. On v2 it does **not**:

| tier | placements where two IDENTICAL runs disagree | largest disagreement |
|---|---|---|
| easy | 0 of 12 | — bit-identical |
| collision | 0 of 12 | — bit-identical |
| pair | 0 of 16 | — bit-identical |
| **hard** | **2 of 12** | **0.444 recall, 0.817 precision** |

Both unstable placements are on the hard tier at its loudest sites, and the
mechanism is visible:
- unit 278 (222.9 uV neighbour): same matched cluster (26) both runs, but
  recall 0.966 vs 0.522 and precision 0.183 vs 1.000 — the cluster absorbed a
  mass of resident spikes in one run and not the other.
- unit 297 (466.5 uV, the loudest site): matched cluster flips 123 -> 125.

So Kilosort is deterministic except where a clustering decision is genuinely
marginal, which is exactly what a loud neighbourhood produces. **Consequence:
easy/collision/pair differences are real signal; hard-tier differences below
~0.44 recall are not evidence of anything.** No hard-tier claim in this
section should be believed.

### The result: footprint-aware clustering is confirmed, with significance
Paired Wilcoxon against vanilla on the same 52 placements (paired because every
config sees identical placements; this removes the between-unit variance that
made round one's n=3 tier means unreadable).

**`footprint_cluster_strong` on the PAIR tier — the tier it was built for:**
recall **0.7205 -> 0.7977 (+7.7 points), 10 wins / 2 losses, p = 0.038**, on a
tier with a bit-identical noise floor. Precision also up (0.4863 -> 0.5554)
though not significantly. **This is the first statistically supported
improvement over stock Kilosort in the whole project.**

**`footprint_cluster` (weight 1.0) wins PRECISION:**
overall **0.5998 -> 0.6542 (+5.4 points), 26 wins / 12 losses, p = 0.010**;
on the pair tier **0.4863 -> 0.5906 (+10.4 points), p = 0.023**.

So the two weights do different jobs: weight 3.0 buys recall on closely-spaced
neurons, weight 1.0 buys precision broadly. Both beat stock; neither dominates.

**Everything else is neutral or harmful.** `amplitude_normalize` overall recall
-0.034 (p = 0.012) and `coarse_align_amp_norm` -0.036 (p = 0.023) are
significantly WORSE. The alignment family is indistinguishable from stock.

### Merge errors appear for the first time
| config | merged pairs |
|---|---|
| vanilla | **1 of 8** (units 285+331, replicate 1) |
| vanilla_repeat | 1 of 8 (same) |
| coarse_align_amp_norm | 1 of 8 |
| **both footprint configs** | **0 of 8** |
| all other configs | 0 of 8 |

Round one reported zero merge errors everywhere and concluded "Kilosort
shatters rather than merges". With pairs now genuinely remote (181-242 um,
similarity 0.000) the failure is detectable, and stock Kilosort does merge one
pair in eight. Both footprint configurations prevent it. One pair is a single
observation — not a result on its own, but it points the same way as the recall
and precision effects.

### What actually makes a placement hard — NOT the thing the hard tier tests
Spearman against vanilla recall across all 52 placements:

| predictor | rho |
|---|---|
| **donor contamination** | **-0.556** |
| site max neighbour amplitude | -0.176 |
| number of injected spikes | -0.179 |
| site noise (MAD) | -0.137 |
| site density | -0.019 |

**The donor's own contamination predicts recall three times better than any
property of where it was placed.** Site amplitude — the axis the whole hard
tier was designed around — is weak, and density is nil.

Caveat on direction of causation: this is partly a construction artefact.
Snippets are drawn at random from the donor cluster, so a contaminated donor
injects inconsistent waveforms, which Kilosort then splits. It is evidence that
donor purity dominates the benchmark, not proof that contamination makes real
neurons hard to sort. It argues for tightening the contamination cap below 10%
in any future round.

### Headline numbers (mean over 4 replicates, 52 placements)
RECALL by tier — vanilla / footprint_cluster_strong / footprint_cluster:
easy .883 / .868 / .862 | collision .832 / .850 / .805 |
hard .764 / .736 / .700 (unreliable) | **pair .720 / .798 / .730**
PRECISION by tier:
easy .765 / .726 / .867 | collision .594 / .571 / .572 |
hard .593 / .655 / .609 (unreliable) | **pair .486 / .555 / .591**
OVERALL recall: footprint_cluster_strong .812 > vanilla .794 > others
OVERALL precision: footprint_cluster .654 > vanilla .600

---

## 5al. Is the problem only splitting? No — and the answer is the same for every algorithm

Gil's question, twice: if a unit was split into three clusters, what are the
recall and precision of all three COMBINED? If the pieces are clean and
together hold ~100% of the spikes, the only thing to fix is merging. If
combining imports contamination, it is a different problem needing a different
fix. `analyse_split_vs_contamination.py`, extended to all eight configurations.

Method: a fragment is a cluster holding at least `max(15, 2%)` of the unit's
spikes. Matching is global one-to-one across ALL clusters, so no true spike can
be claimed twice — which is why best-cluster recall reads ~7 points lower here
than in the headline tables, where each cluster is matched independently.
Compare within a file, never across the two.

**Finding 1 — the premise does not hold. Units are barely split.** Mean
fragments per unit is 1.27–1.38 across every configuration, the maximum
anywhere is 3, and 67–77% of units have exactly ONE substantial cluster. There
is no three-way split to merge.

**Finding 2 — combining always trades down. 16 algorithm×tier cells, negative
in all 16.** Recall gain vs precision cost:

| algorithm | easy | collision | pair | hard |
|---|---|---|---|---|
| vanilla | +3.3 / −5.9 | +7.1 / −10.3 | +5.3 / −10.0 | +6.4 / −10.9 |
| footprint_cluster_strong | +3.2 / **−3.6** | +3.4 / −7.5 | +2.9 / −9.5 | +3.5 / −22.4 |
| footprint_cluster | +5.0 / −10.0 | +4.6 / −11.0 | +5.0 / −11.0 | +7.4 / −12.1 |
| coarse_then_align | +4.3 / −16.1 | +3.7 / −7.0 | +4.5 / −8.3 | +6.8 / −5.7 |

Best case anywhere: footprint_cluster_strong on easy, +3.2 recall for −3.6
precision. Still a loss, on the easiest tier.

**Finding 3 — a perfect merge tool converges every algorithm to the same
place.** Pooled combined recall/precision: vanilla 77.7/45.8,
footprint_cluster_strong 78.6/47.4, footprint_cluster 75.7/49.0,
coarse_then_align 75.8/44.7. Merging cannot separate the algorithms.

**Threshold discipline.** At `>= 5` spikes the fragment count rises to 2.46 and
combined precision falls to 0.340 — but the same counting against RANDOM times
invents **6.02 fragments from nothing**, so that threshold is measuring the
procedure, not the data. At `>= 15` chance gives 0.00 and the conclusion is
stable to `>= 30`. Fourth time the base-rate trap has been caught in this
project; it is now checked by default.

**Union ceiling: 94.4–95.8%** of injected spikes land in SOME cluster. So
detection is near-perfect and the loss is assignment — but not into a few clean
pieces. 14–22 spikes per 100 sit in thin scatter, a handful each across many
clusters, none large enough to be a fragment, and those clusters belong to
other neurons.

---

## 5am. The post-Kilosort correction pipeline, finally tested against truth — and it does not work

Gil's second question: run the post-hoc correction we built before (timing
alignment + whitening to find missed spikes, plus merge detection) on the v2
synthetic data, because post-hoc correction may be more efficient and accurate
than changing Kilosort itself. `test_post_hoc_on_v2.py`, 52 placements × 3
configurations, all 4 replicates.

**Why this run matters: the pipeline had been run twice before and both times
the headline had to be withdrawn, for a structural reason.** On real data a
recovered candidate cannot be checked — the only available test was "does some
other unit already have a spike here", which answers a different question. 5h
ended at "these look like collisions, not misses". 5p watched the
genuinely-new count fall 115 → 40, and for the cleanest unit 29 → 1, once the
frequency rule was fixed. Neither could say whether a recovered spike truly
belonged to the unit. The v2 benchmark removes that limitation completely.

**The pipeline under test, strictly truth-blind** (sees Kilosort's output and
the raw voltage, never the truth): (1) template convergence — rebuild from the
cluster's own empirical mean waveform, probe frequency by the timing criterion
`f0*|W|`, wavelet-align and re-average to a fixed point (3.7 iterations mean,
f0 787–5364 Hz); (2) AR(4) temporal whitening from the quietest spike-free
stretch on that channel; (3) full-session matched filter with the rebuilt
template over all 120 s — the earlier runs scanned only ±150 samples around
existing spikes, which at these firing rates cannot reach most misses; (4) two
gates, both calibrated on the cluster's OWN real spikes: whitened fit R² and
footprint similarity.

**A 20-sample sign error was found and fixed before any result was read.**
`fftconvolve(..., mode="same")` indexes the template window's centre, so the
event sits at `i - N//2 + NT0MIN`; I had the sign backwards, putting every
candidate 20 samples late. It recovered exactly zero spikes while the gates
still passed at 86%, because `align_snippet`'s ±25 coarse search walked back to
the real event — a silent failure that looked like a finding. Caught by
validating the convention against known truth times rather than reasoning about
it: median offset is now +0.0 samples with 100% of detected spikes inside ±3.
This is the fifth silent failure in this project that produced plausible output;
the lesson is unchanged and now applied by default — validate conventions
against truth, do not derive them.

**THE RESULT (vanilla): recovery precision 2.3%.** 11,027 candidates accepted
across 52 placements; **259 were real missed spikes of the unit.**

| tier | missed | accepted | recovered | chance | recall before→after | precision before→after |
|---|---|---|---|---|---|---|
| easy | 34.1 | 143.6 | 4.0 | 0.03 | .883 → .897 | .765 → .739 |
| collision | 49.2 | 207.9 | 4.0 | 0.08 | .832 → .845 | .594 → .539 |
| pair | 81.2 | 334.1 | 5.4 | 0.19 | .720 → .739 | .486 → .440 |
| hard | 68.7 | 121.9 | 6.3 | 0.10 | .764 → .786 | .593 → .551 |

Pooled: recall .794 → .811 (**+1.7**), precision .600 → .557 (**−4.3**).
Same shape for footprint_cluster_strong (.812 → .831, .621 → .552) and
coarse_then_align (.771 → .787, .584 → .529).

**It is finding real signal — 23× chance — but not this unit's spikes.** The
chance control (same accepted set against random times at the same rate) gives
0.1%, so 2.3% is genuinely above chance and the matched filter works. But **of
the 10,768 false accepts, 10,070 (93.5%) coincide with a spike Kilosort already
filed under a DIFFERENT cluster.** The tool is recovering real spikes belonging
to other neurons. That is 5h's and 5p's conclusion, now quantified against
truth instead of inferred: a SPECIFICITY failure, not a detection failure.

**WHAT KIND of spike did it recover?** (`analyse_recovery_breakdown.py`, added
after Gil asked whether the tool pulled back spikes that were never detected or
spikes misfiled under another unit — the 5am headline collapsed the two.) Every
spike absent from a unit's best cluster is either MISFILED (some other cluster
has a spike within tolerance; detection worked, attribution did not) or
UNDETECTED (no cluster anywhere has one).

| config | missed | misfiled | undetected | recovered misfiled | recovered undetected |
|---|---|---|---|---|---|
| vanilla | 3122 | 2480 (79.4%) | 642 (20.6%) | 212 (8.5%) | 47 (7.3%) |
| footprint_cluster_strong | 2852 | 2085 (73.1%) | 767 (26.9%) | 262 (12.6%) | 25 (3.3%) |
| coarse_then_align | 3473 | 2625 (75.6%) | 848 (24.4%) | 208 (7.9%) | 34 (4.0%) |

So the answer is **both, and mostly re-attribution**: of vanilla's 259
recoveries, 82% were spikes Kilosort had already detected and filed elsewhere,
18% were genuinely new detections. The tool does work in both modes — it is not
only finding noise, and it is not only moving spikes around — but it reaches
only ~8% of what is available in either category, at 2.3% precision.

Two things worth keeping from this table. First, **the 79/21 misfiled-to-
undetected split independently reproduces 5ad's 84% clustering-failure finding**
on different data and a different method, which is the strongest corroboration
that result has. Second, **footprint_cluster_strong recovers noticeably more
misfiled spikes (12.6% vs 8.5%) and fewer undetected ones (3.3% vs 7.3%)** —
exactly the signature of a method that is better at attribution and no better
at detection, and one more reason the open lead is footprint-first attribution.

**Gate sweep — no threshold rescues it.** Recovery precision / share of all
misses recovered, vanilla: matched filter alone 1.6% / 15.0%; footprint gate
only 2.7% / 10.2%; whitened R² only 1.6% / 11.0%; both 2.3% / 8.3%; both with
R² at the median of real spikes 3.5% / 6.9%. The footprint gate is the one
doing real work — it roughly doubles precision at every setting, consistent
with 5q's AUC 0.999 — but doubling 1.6% is not useful. The ceiling with no gate
at all recovers only 15–19% of misses.

**The merge detector fails too, and for a diagnosable reason.** First test ever
run against a known answer: same-neuron pairs are a unit's fragment vs its best
cluster (truth: MERGE), different-neuron pairs are the two units of a pair-tier
placement (truth: DO NOT MERGE). Sensitivity is **0.00–0.11** across all three
configs and all three thresholds — it essentially never says merge for a true
fragment. The reason is in the numbers: the merged refractory violation rate
for TRUE fragments (0.066–0.076) is barely below that of genuinely DIFFERENT
neurons (0.076–0.123), because the input clusters are already contaminated —
the main cluster alone violates at 0.051–0.060. The refractory test needs clean
inputs to have any headroom, and these clusters do not provide it.

**CONCLUSION, answering Gil's question directly: post-hoc correction as built
is NOT more efficient than fixing Kilosort.** It buys +1.7 recall for −4.3
precision, the same trade-down that merging offers (5al), for far more
machinery. Both fail for the identical underlying reason: on this probe,
anything that scores as a unit's spike on its own channel is overwhelmingly
likely to be a neighbour's real spike. That is one problem, and it is a
per-spike attribution problem.

**Honest limits of this test.** (1) The search is a single-channel matched
filter with footprint used only as a post-hoc gate; 5q's finding was that
footprint similarity should be the DISCRIMINATOR. A footprint-first search is
untested and is the obvious next variant. (2) Candidates were capped at the top
800 per placement by matched-filter score (28,908–29,499 refined in total);
clusters on loud channels had tens of thousands above threshold, so the sweep's
"no gate" row is the best 800 guesses, not all of them. (3) The hard tier
remains unreliable (noise floor 0.444 recall).

Artifacts: `outputs/v2_post_hoc_correction.csv` (per-placement),
`outputs/v2_post_hoc_candidates.csv` (every refined candidate with its mf/R²/
footprint score and whether it was a real missed spike — sweepable offline),
`outputs/v2_merge_detector_truth.csv`.

---

## 5an. First physiology: reward and cue-end responses on the full session

Gil asked for the sync pipeline to be run on the full sorted session and for
neurons responding to reward and to the end of the cue stimulus, split by
choice side, with incomplete trials removed.

**Pipeline run** (`sync_pipeline` at `D:\Gil\sync_pipeline`, output in
`...20260916_110311.kilosort\synced_20261002`). Sync is clean: **21,422 /
21,422 Bpod state events matched (100%)**, local-interpolation held-out error
**median 0.000 ms / max 43 ms**, `gaps=[]` so all 953 trials are `sync_valid`.
The README's critical ephys-offset fix engaged — row 0 of `probe1.dat` is
absolute sample 2,625,402 = **87.5134 s**, which would silently have shifted
every spike by that much. `sync_status` reads `needs_review`, which is expected
and not a problem: that flag describes the GLOBAL affine fit (363.9 ms median
residual from real Bpod clock drift), while the actual conversion uses local
interpolation.

One snag worth recording: the cached `.pkl` beside the Bpod `.mat` was written
by a numpy 2.x environment and cannot be unpickled under this machine's numpy
1.26 (`ModuleNotFoundError: numpy._core.numeric`). Fixed without touching Gil's
files by staging a scratch `rat_root` holding a copy of the `.mat` and the
`rat_metadata.json`, so the pipeline rebuilt its own cache.

**Trials and units.** 953 trials, **478 completed and sync-valid (50%)** —
`sync_valid & TrialCompleted`. The other 475 are genuine incompletes (never
poked centre, broke fixation, early withdrawal, no side poke). Units are the
**111 clusters Gil labelled `good` in the review tool**, NOT Phy and NOT
KSLabel — note `units.parquet`'s own `quality_label` column is KSLabel and
would have given 189 units, so using it would have been the wrong curation.

**The two events, taken from `TASK_TIMELINE.md`.** Reward = start of
`water_L`/`water_R`, the valve opening (291 trials; 153 right, 138 left). NOT
`rewarded_Lin`/`rewarded_Rin`, which are the Step-11 waiting states named after
the correct side and re-entered thousands of times by the grace loop. Cue end =
end of `stimulus_delivery`, verified identical to the start of `wait_Sin` on
all 479 trials because `AuditoryStimulusTime - MinSampleAud` = 0.35 - 0.35 = 0 s
this session.

### Results (`analyse_reward_and_cue_responses.py`)

| event | responsive (FDR<0.05) | substantial (abs mod >= 0.10) | up / down | side-selective |
|---|---|---|---|---|
| end of cue | 54/111 (49%) | **44/111 (40%)** | 27 / 27 | 45 (41%) |
| reward delivery | 30/111 (27%) | **21/111 (19%)** | 21 / 9 | 15 (14%) |
| give-up, no water (control) | 13/111 (12%) | 10/111 (9%) | 6 / 7 | 18 (16%) |

Significance alone overstates this — with 478 paired trials the test detects
very small shifts — so the substantial column is the one to quote.

**Strongest reward responses are large and sharply time-locked:** unit 35
(ch345, 6.20 mm) goes **19.0 -> 80.1 Hz** at valve open; unit 250 **32.4 ->
13.8 Hz** (a decrease); unit 131 **27.0 -> 41.1 Hz**. Onsets sit on the valve
instant, not on the choice poke a median 1.61 s earlier.

**Cue-end responses split hard by side.** Unit 368 (ch77) fires **0.6 Hz on
left-choice trials and 14.5 Hz on right** in the 150 ms after cue offset, from
a 1.2 Hz baseline. 45 of 111 units differ between left and right choices there,
against 15 at reward — side information is carried at the choice point, not at
the outcome.

**Reward responses look reward-specific.** All 21 substantially
reward-responsive units were re-tested at the give-up moment with the same
windows: **18 show no response there at all, and the remaining 3 (250, 319,
310) respond in the OPPOSITE direction.** None responds the same way.

**Overlap:** 32 units cue-end only, 8 reward only, 22 both, 49 neither.

### Two honest limits on the cue-end result
1. **Cue offset, side-lights-on and movement onset are the same instant in this
   task.** `MT` is measured from `wait_Sin`'s start and its median is 0.253 s.
   The test window was cut to 0.15 s to close before the 10th-percentile
   movement time (0.205 s), but no window can separate "responds to the
   stimulus ending" from "responds to initiating a choice" given this design.
2. **CORRECTED (see 5ao): I first wrote that "a matched reward-omission
   control does not exist". That was wrong, and Gil corrected it.** The
   omission condition is real and is a designed part of the task: of the 478
   completed trials, 71 are ERRORS, 291 are CORRECT+REWARDED and **116 are
   CORRECT+OMISSION** (67 of them catch trials). On those the rat chose
   correctly and no water came, and because it cannot tell them from a
   rewarded trial until the water fails to arrive, **how long it waits before
   giving up is a readout of its confidence**. What genuinely does not exist
   is a *delivered* negative outcome at a matched time — all 187 unrewarded
   trials end in `skipped_feedback` because `GUI.CatchError=1` stretches the
   incorrect wait to 20 s so it is never sat out. So the give-up moment is not
   a passive omission marker, it is the behavioural report itself, which is
   why the reward-specificity result above stays suggestive rather than
   decisive.

Artifacts: `outputs/session_20260916_reward_cue_responses.csv` (per unit per
event), `outputs/reward_cue_figs/{reward,cue_end}_top_units.png` (raster + PSTH,
split left/right).

---

## 5ao. Decoding DV: the population carries the rat's DECISION, not the stimulus

Gil asked whether DV can be decoded from the good units, at which stage, and by
which method — expecting (correctly) that it would not be decodable before the
stimulus. He also corrected my trial taxonomy, which is recorded first because
it changes how the behaviour is read.

### The taxonomy, corrected
The 478 completed trials are **71 ERRORS**, **291 CORRECT+REWARDED** and **116
CORRECT+OMISSION**. I had previously written that "a matched reward-omission
control does not exist" (5an, now corrected in place). It does: on omission
trials the rat chose correctly and no water came, and since it cannot tell
those from a rewarded trial until the water fails to arrive, **how long it
waits is a confidence report**. What genuinely does not exist is a *delivered*
negative outcome at a matched time.

The 116 must be split before anything is measured: **67 are catch trials**
(wait stretched to 20 s — the designed confidence probe) and **49 are ordinary
trials the rat abandoned early**, which wait a median of **0.83 s** and are a
completely different population.

### The confidence behaviour reproduces, on the matched comparison
Catch trials and error trials are both stretched to 20 s (`GUI.CatchError=1`),
so they are directly comparable:

| group | n | median wait |
|---|---|---|
| CORRECT + catch | 67 | **7.14 s** |
| ERROR | 71 | **6.25 s** |
| CORRECT, normal delay, abandoned | 49 | 0.83 s |

Correct > error, one-sided Mann-Whitney **p = 0.0029**. On catch trials the
wait scales with difficulty (**rho = +0.330, p = 0.0064**, monotonic across
quartiles 6.36 / 7.06 / 6.85 / 8.05 s); on error trials it does **not**
(rho = +0.027, p = 0.83). That last contrast is the one worth having — when
the rat was wrong, its persistence stops tracking the true stimulus strength.
Pooling the 49 abandoned trials back in drops it to p = 0.10, which is why the
split matters.

### The decoding (`decode_dv.py`, `outputs/dv_decoding_*.csv`)
111 good units, 478 completed trials, 200 ms sliding windows, 5-fold CV
repeated 5x, **every number against its own permutation null** (never an
assumed chance level — fifth time that discipline has mattered here).

**Before the stimulus: nothing, across every method.** DV R2 = −0.009 (ridge,
p=0.25), −0.073 (random forest, p=0.86); |DV| p=0.47–0.80; side AUC 0.47–0.48;
choice AUC 0.51. A clean negative control, confirming the pipeline does not
manufacture signal.

**DV becomes decodable ~100 ms after stimulus onset and peaks at R2 = 0.568.**
But the rat's CHOICE rises in lockstep and slightly ahead of it:

| window centre from stim onset | DV R2 | correct-side AUC | choice AUC |
|---|---|---|---|
| 0.00 | −0.010 | 0.444 | 0.498 |
| 0.10 | 0.054 | 0.606 | 0.672 |
| 0.20 | 0.266 | 0.731 | 0.796 |
| 0.35 (cue end) | 0.456 | 0.824 | 0.952 |
| 0.55 (peak) | **0.568** | 0.901 | **0.998** |

**THREE INDEPENDENT PROOFS THAT THIS IS CHOICE, NOT STIMULUS:**

1. **The variance ceiling.** Knowing the rat's choice perfectly explains
   **R2 = 0.533** of DV (because sign(DV) is the correct side and the rat is
   right on 85.1% of trials). The decoder reached 0.564 — essentially nothing
   beyond the choice. Knowing sign(DV) perfectly would give 0.709, which it
   never approaches.
2. **The error-trial test, the decisive one.** Train sign(DV) on the 407
   correct trials, test on the 71 errors, where choice and correct side
   DISAGREE. AUC = **0.002 / 0.001 / 0.003 / 0.030** from cue end onward — not
   at chance but *inverted*, i.e. it predicts the choice just as well on error
   trials. It was never reading the stimulus.
3. **Nonlinearity adds nothing.** Random forest 0.240 vs ridge 0.251 during the
   stimulus; 0.578 vs 0.571 at cue end; linear SVR worse than both. No hidden
   structure the linear decoder missed.

### The one genuinely non-motor signal: |DV| (difficulty)
Difficulty is independent of which side was correct, so it cannot be a motor
readout. Tested with ridge AND random forest:

| epoch | ridge R2 | RF R2 | RF p |
|---|---|---|---|
| pre-stimulus | −0.023 | −0.045 | 0.47 |
| **during stimulus** | **−0.018** | **−0.027** | **0.17** |
| cue end -> choice | +0.039 | **+0.091** | 0.008 |
| at choice poke | +0.011 | +0.060 | 0.008 |
| after choice (wait) | +0.083 | +0.068 | 0.008 |
| late wait | +0.011 | +0.035 | 0.008 |

**Difficulty carries no information while the stimulus is playing — not even
nonlinearly — and appears only after the choice**, peaking ~450–500 ms into the
waiting period before decaying to chance by ~1 s. Random forest beats ridge
here (0.091 vs 0.039 at cue end), so the difficulty code is somewhat nonlinear,
which is the one place a nonlinear decoder earned its cost.

**Interpretation.** That timing is exactly where the confidence behaviour sits.
Difficulty also predicts how long the rat waits (rho=+0.33 on catch trials) and
whether it was correct (mean |DV| 0.533 on correct vs **0.208** on errors). So
the post-choice difficulty signal reads as **the rat's confidence state**, not
a sensory representation.

### The limit that no analysis here can get past
Choice is decodable at AUC 0.672 within 100 ms of stimulus onset and 0.796 by
200 ms. **The decision is forming while the tone is still playing**, so no time
window in this task isolates sensory coding from the developing choice.
Separating them needs a design where the choice is withheld or delayed.

Artifacts: `outputs/dv_decoding_timecourse.csv`, `dv_decoding_epochs.csv`,
`dv_decoding_error_generalisation.csv`, `dv_decoding_absdv_rf.csv`,
`outputs/reward_cue_figs/dv_decoding_timecourse.png`.

---

## 5ap. Leaving time, and the clean version of the DV/choice test

Gil asked two things: whether the leaving time can be decoded (and whether it
sharpens closer to the leave, using long-wait no-reward trials, errors and
omissions kept apart), and whether my 5ao claim really amounts to "DV and
choice are confounded, so we may just be decoding choice". The second is
exactly right, and is now shown directly rather than by inference.

### (A) Decoding DV WITHIN a single choice — and a correction to 5ao
5ao used an error-trial generalisation test. The cleaner test is to decode DV
among trials that share one choice, where choice is constant by construction so
nothing decodable can be motor (`decode_leaving_time.py`, part A).

| epoch | subset | DV R2 | \|DV\| R2 |
|---|---|---|---|
| stimulus | all 478 | **+0.216** | −0.024 (p=0.75) |
| stimulus | left choice (208) | −0.046 (p=0.68) | −0.050 (p=0.70) |
| stimulus | right choice (270) | **+0.062** | **+0.077** |
| cue end → choice | all 478 | **+0.564** | +0.020 |
| cue end → choice | left choice | +0.066 | −0.012 (p=0.14) |
| cue end → choice | right choice | **+0.207** | **+0.187** |

**The main conclusion of 5ao stands and is now exact:** conditioning on choice
destroys most of the signal — 0.564 → 0.066/0.207 at cue end, 0.216 →
−0.046/+0.062 during the stimulus. The large R2 was the choice.

**But 5ao overstated one thing and it needs correcting.** I wrote that
difficulty "carries no information while the stimulus is playing". That was
based on pooling all trials. **Within right-choice trials, |DV| IS decodable
during the stimulus (R2 = +0.077, p = 0.005)**, and more strongly at cue end
(+0.187). Pooling hid it, which is what happens if the code is choice-dependent
— opposite-signed contributions cancel. So stimulus information IS present
during the stimulus; it is just small, and organised per choice.

Two cautions on that. Within a choice group DV and |DV| are near-redundant
(choosing right, correct trials carry large |DV| of one sign and errors small
|DV| of the other), so this is substantially "will this be an error", which is
still genuine stimulus information since errors happen on hard trials
(mean |DV| 0.208 vs 0.533). And it is **asymmetric — right-choice trials
only**. That is the contralateral side for this animal (`hemisphere: "L"`,
ML −2.45), which is the expected direction for striatum, but n is also larger
on the right (270 vs 208), so power is not controlled. Worth a proper test.

### (B1) How much longer will the rat stay? NOT decodable.
Rewarded trials are excluded (the valve ends them, not the rat) along with the
49 abandoned trials; errors and catch trials kept separate, per Gil.

**The confound that would otherwise fake this entire result:** a fixed window
at lag T applied to every trial measures post-departure activity on trials that
already ended, so a "decoder" would really be detecting whether the rat is
still in the port. Every window therefore includes only trials still waiting
when it closes, and n is reported per row — which is also why long-wait trials
are the only well-posed regime, as Gil anticipated.

Result: **cross-validated R2 is negative at every lag from 0 to 4 s, in both
groups** (catch −0.20 to −0.38, error −0.14 to −0.31), against permutation
nulls near −0.03, p = 0.54–0.87. Not a hint of signal. R2 falling *below* the
shuffled null is the ridge selecting a weaker penalty on apparent structure
that then fails to generalise — the signature of fitting noise.

**Honest limit:** n is 64–71 per group and the SD of remaining wait is only
1.6–1.9 s, so this is a low-powered negative, not a proof of absence.

### (B2) Is the leave IMMINENT? Yes — strongly.
Different question, and it works. Time points every 250 ms through the wait;
from a 300 ms window, will the rat go within the next 500 ms? Cross-validated
by trial (GroupKFold, no trial in both train and test); the null shuffles which
trial got which leave time, which **preserves elapsed time as a predictor** —
so chance here is 0.77, not 0.5.

| group | trials | time points | AUC | chance (95th) | p |
|---|---|---|---|---|---|
| correct, no water (catch) | 67 | 1878 | **0.923** | 0.766 (0.811) | 0.005 |
| error | 69 | 1715 | **0.901** | 0.769 (0.809) | 0.005 |

**The dissociation is the result:** there is no graded countdown — nothing
predicts *how much longer* — but the moment of leaving is strongly flagged
about half a second ahead, well beyond what elapsed time alone gives. Both
groups behave the same way, so this is not specific to being right or wrong.

**Interpretation to hold loosely:** leaving is a movement, so a signal 500 ms
ahead of it is at least partly motor preparation. Distinguishing "decided to
quit" from "about to withdraw" needs an independent readout of the decision,
which this task does not provide.

Artifacts: `outputs/dv_within_choice.csv`, `leaving_time_decoding.csv`,
`leaving_imminent.csv`, `outputs/reward_cue_figs/leaving_time_decoding.png`.

---

## 5aq. Stimulus-period dynamics  [STIMULUS DESCRIPTION CORRECTED IN 5as]

> **Read 5as before this section.** The stimulus is NOT a tone cloud. It is a
> deterministic morph of two natural sounds indexed by AuditoryOmega, which IS
> saved for every trial. The decoding results below are unaffected, but the
> description of what the rat heard is wrong.

Gil asked how choice/DV evolve during the stimulus, and whether the signal
tracks the individual evidence events — "some clicks are evidence for the left
and some for the right".

### The stimulus is a tone cloud, and the sequence was not saved
This session is **not** the click-train variant (`ClickTask = 0`,
`AuditoryTrial = 1`, `TaskType = 3`). It is the tone cloud: 30 ms tones from 18
frequencies spanning 200 Hz–20 kHz (`Aud_ToneDuration = 0.03`,
`Aud_nFreq = 18`), overlapping by 2/3 (`Aud_ToneOverlap = 0.6667`), so a tone
starts every ~10 ms and the 350 ms stimulus carries **~35 of them**, each above
or below `CategoryBoundary` and therefore evidence for one side. The question is
the right one for this task.

**But the realized sequence is not recorded.** `Custom.AudSound` has 957 entries
and only **four** are non-empty — indices 953–956, the pre-generated look-ahead
trials that were never played (953 ran). It is a rolling buffer for upcoming
trials, not a log. Nothing else carries per-tone identity or timing:
`AudFracHigh` is a single 2-element array and `DV` is one number per trial.

**I got this wrong once before concluding it.** My first check read only the
first three trials, found empty arrays, and I wrote that the sequence "was not
saved anywhere". The size histogram then showed four entries of 128,251 samples
— real waveforms. Only on checking *which* trials did it turn out they are the
unplayed look-ahead ones. The conclusion survived; the reasoning behind it would
not have.

**To make this analysis possible on future sessions** the protocol must log, per
trial, the tone frequencies and onset times (or the RNG seed that generated
them). That is a protocol change, not an analysis one, and it is cheap.

### What the dynamics actually look like
`decode_stimulus_dynamics.py`, 100 ms windows stepped 25 ms (the earlier 200 ms
windows were wider than half the stimulus and smeared exactly this period).

| time from stimulus onset | choice AUC | DV R2 |
|---|---|---|
| −0.15 to 0.00 (pre) | 0.48–0.52, all p>0.2 | ~−0.012, all n.s. |
| +0.075 | **0.586 (p=0.010)** — first significant | n.s. |
| +0.125 | 0.636 | +0.048 |
| +0.150 | 0.720 | +0.118 |
| +0.250 | 0.826 | +0.256 |
| +0.300 (stimulus ends 0.35) | **0.881** | +0.320 |
| +0.550 (after the poke) | 0.995 | +0.543 |

Choice becomes decodable **~75–100 ms after stimulus onset — about 7–10 tones
in** — and climbs smoothly to **AUC 0.88 by the time the stimulus ends**, which
is still ~250 ms before the median choice poke at +0.603 s. **The decision is
essentially complete at stimulus offset**, which is the mechanistic reason no
window in this task separates sensory coding from the choice (5ao).

### The accumulation proxy: it behaves like evidence integration
Without the tone stream, the closest available test is whether the choice signal
rises FASTER when the net evidence is stronger. Trials split into |DV| tertiles:

| evidence | n | mean \|DV\| | AUC≥0.60 | AUC≥0.70 | peak within stimulus |
|---|---|---|---|---|---|
| weak | 159 | 0.13 | +0.225 s | +0.250 s | 0.804 |
| medium | 160 | 0.45 | +0.150 s | +0.200 s | 0.906 |
| strong | 159 | 0.83 | **+0.100 s** | **+0.125 s** | **0.938** |

**Perfectly monotonic: the choice signal appears 125 ms earlier on strong-
evidence trials than on weak ones, and reaches a higher plateau.** That is the
signature of accumulation — more evidence per unit time crosses the bound
sooner.

Two cautions kept with it. Easy trials also produce more consistent behaviour,
so better decodability there is not by itself proof of integration. And the
per-tertile curves are noisy before the stimulus (n=159 each, values wandering
0.48–0.64), so the onset times carry real uncertainty; the "first window that
*stays* above threshold" rule was used precisely to stop that noise setting the
answer.

Artifacts: `outputs/stimulus_dynamics.csv`,
`outputs/reward_cue_figs/stimulus_dynamics.png`.

---

## 5ar. Validating the stimulus-dynamics result: two more sessions, split-half, and the pre-stimulus leak explained

Gil's challenge: the 5aq result is very strong for one session. He also asked
for a within-session check comparing trials where evidence for the correct side
arrives early vs late in the stimulus, and separately flagged that older
sessions might not be correctly synchronised.

**Gil's synchronisation worry was right, and it mattered.** Both other sorted
sessions had `synced` folders predating the ephys-offset fix (made Sep 2 and
Sep 16; the fix landed Sep 17). Both were reprocessed from scratch. The
recovered offsets: **20260901_085606 = 25.79 s**, **20260911_100049 = 28.8 s**.
Using the old folders would have silently destroyed the alignment. Both
reprocessed cleanly — 15174/15174 and 23153/23153 events matched, held-out
median 0.000 ms, no gaps.

**The early-vs-late-evidence analysis remains impossible** for the reason in
5aq: it needs per-tone timing, and `Custom.AudSound` holds only a rolling
buffer of unplayed look-ahead trials. No analysis recovers it.

### Cross-session replication (same code, `decode_stimulus_dynamics.py`)
The two validation sessions have no hand review, so they use KSLabel `good` —
a weaker curation, making it a harder test.

| session | curation | trials | units | pre-stim DV | DV first significant |
|---|---|---|---|---|---|
| 20260916_110311 | hand-curated | 478 | 111 | n.s. (p≥0.34) | +0.100 s |
| 20260901_085606 | KSLabel | 419 | 112 | n.s. (p≥0.72) | +0.075 s |
| 20260911_100049 | KSLabel | 565 | 183 | n.s. (p≥0.089) | +0.075 s |

**The accumulation proxy replicates in all three, on both measures:**

| session | peak choice AUC in stimulus (weak/med/strong) | AUC≥0.70 onset (weak/med/strong) |
|---|---|---|
| 20260916_110311 | 0.804 / 0.906 / **0.938** | 0.250 / 0.200 / **0.125** s |
| 20260901_085606 | 0.799 / 0.884 / **0.952** | 0.325 / 0.125 / **0.100** s |
| 20260911_100049 | 0.857 / 0.943 / **0.964** | 0.225 / 0.150 / **0.100** s |

Monotonic in all six comparisons. Stronger evidence produces an earlier and
higher choice signal, in three independent sessions with two different curation
methods.

### Split-half within each session (`validate_stimulus_dynamics.py`)
Independent fits on the first and second half of trials — a fluke of one subset,
or an artefact of slow drift, would differ between halves.

**Late-stimulus decoding is significant in all 6 half-sessions** (choice AUC
0.761–0.928, DV R² +0.271 to +0.331, every p = 0.005). Early-stimulus DV is
significant in 5 of 6. Pre-stimulus DV is non-significant in all 6. The result
does not depend on which half of a session you look at.

### THE PRE-STIMULUS LEAK, AND ITS CAUSE
The validation sessions decode the rat's CHOICE above chance *before* the
stimulus (0.60–0.63 and 0.54–0.60), where 20260916 is clean (0.48–0.52). That
would contaminate any onset estimate for choice, so it had to be explained.

**It is choice history, and the test is decisive:**

| session | decode CURRENT choice | decode PREVIOUS choice | repeat rate | current, *within* previous-choice groups |
|---|---|---|---|---|
| 20260916_110311 | 0.502 (n.s.) | **0.702** (p=0.005) | 54.9% | 0.453 |
| 20260901_085606 | 0.598 (p=0.005) | **0.717** (p=0.005) | 64.4% | 0.532 |
| 20260911_100049 | 0.538 (n.s.) | **0.695** (p=0.005) | 55.1% | 0.480 |

Pre-stimulus activity carries the PREVIOUS trial's choice at AUC ~0.70 in all
three sessions. Because the rat repeats its last choice on 55–64% of trials,
that alone predicts the current choice — and **conditioning on the previous
choice collapses current-choice decoding to chance (0.45–0.53) in every
session**. The leak is history, not a sorting or sync artefact, and the session
with the weakest repeat rate (20260916, 54.9%) is the one with no leak.

**This does not touch the DV result.** Pre-stimulus DV is at chance in all three
sessions and in all six halves. Only the *choice* onset estimate is affected,
and only in the two sessions with strong choice history.

**A finding in its own right:** a persistent representation of the previous
trial's choice, AUC ~0.70 in the 100 ms before the next stimulus, replicated
across three sessions. Not something this project set out to look for.

Artifacts: `outputs/stimulus_dynamics_{session}.csv`,
`validation_split_half.csv`, `validation_prestim_history.csv`,
`outputs/reward_cue_figs/stimulus_dynamics_{session}.png`.

---

## 5as. CORRECTION: the stimulus is two natural sounds, it IS fully recorded, and 5aq described it wrongly

Gil: "can't you find the original .mat bpod file and understand where the
stimulus is? It should contain all the trials information. this task uses two
natural sounds for the stimulus." He was right on every count.

### What 5aq got wrong
5aq described the stimulus as a **tone cloud** — 30 ms tones from 18
frequencies, ~35 per trial — and concluded the per-tone sequence "was not
saved". That description came from reading the `Aud_*` GUI settings without
checking which stimulus variant was actually selected. **It is wrong.**

`AuditoryStimulusType` is a popupmenu whose options are
`['Clicks', 'Freqs', 'Natural']`, and this session's value is **3 = Natural**.
The `Aud_nFreq` / `Aud_ToneDuration` / `Aud_ToneOverlap` parameters belong to
the `AudFreq` GUI panel — the *Freqs* variant — and are inert here. The real
stimulus is set by the `AudNatural` panel: `soundA = frogs2.wav`,
`soundB = Passer_Montanus.wav` (a tree sparrow), from the protocol's
`naturalsounds165` folder.

### The stimulus IS fully recorded, contrary to 5aq
`Custom.AuditoryOmega` is saved for **every trial** (956 distinct values in
[0,1]) and **`DV = 2*omega - 1` exactly** — median absolute difference 0.00e+00,
99.8% of trials matching to 1e-6 (the two exceptions are the StartEasyTrials at
indices 0-1). So the stimulus parameter was never missing. 5aq's claim that
"we know the NET evidence and not the stream that produced it" was the right
sentence about the wrong task.

### The sound is a DETERMINISTIC function of omega — measured, not assumed
Four real stimulus waveforms are in the file (indices 953-956, the unplayed
look-ahead trials; 128,251 samples = 0.668 s at 192 kHz, longer than the 0.35 s
actually played). Comparing them pairwise:

| pair | delta-omega | raw waveform r | spectrogram r |
|---|---|---|---|
| 953 vs 955 | **0.0087** | **0.9999** | 0.9998 |
| 953 vs 956 | 0.327 | 0.857 | 0.934 |
| 954 vs 956 | 0.296 | 0.849 | 0.951 |
| 953 vs 954 | 0.622 | 0.455 | 0.832 |

Two trials whose omega differs by 0.009 produce waveforms correlating at
**0.9999**, and similarity falls off smoothly with delta-omega. There is **no
random component**: the sound is a continuous, deterministic morph between two
fixed recordings, indexed by omega alone.

### What this means for Gil's early-vs-late question
**It is not answerable, but for a completely different and better reason than
5aq gave.** Not missing data — *the task has no such variability by design*.
Evidence strength is one scalar per trial, and the sound for a given omega is
always the same sound, so there are no trials where evidence for the correct
side "arrived early" versus "late" to compare against each other. A click train
has that structure; a deterministic morph does not.

**But a real analysis path does exist, and it needs no protocol change.** The
morph has genuine within-trial spectrotemporal structure (band power swings
~15 dB across a trial, inherited from the natural sounds). With the two source
.wav files one could compute, at each moment, how much the mixture locally
resembles sound A versus sound B — a *time-resolved local evidence* signal,
identical across trials sharing an omega — and regress neural activity against
it. That is the closest thing this task permits to the click-train analysis,
and it requires only frogs2.wav and Passer_Montanus.wav from the rig PC's
protocol folder, which is not on any drive mounted here.

### What survives from 5aq and 5ar
Everything quantitative. None of the decoding depended on the stimulus
description being right: the analyses used `DV` and `ChoiceLeft`, which are
unaffected. The replication across three sessions, the split-half results, and
the choice-history explanation of the pre-stimulus leak all stand. Only the
prose describing what the rat heard was wrong, and only in 5aq.

**Process note.** This is the second time in two sections that reading a subset
and generalising produced a confident wrong statement — first the `AudSound`
"all empty" claim, now the stimulus identity. Both were caught by Gil rather
than by me. The pattern is the same: inspecting a few entries or a few fields
without checking which configuration was actually in force.


---

## 5at. The stimulus fully reconstructed — and why early-vs-late is impossible by design, not by omission

Gil pressed the point: "I didn't understand whether you were able to find the
stimulus vector and perform the analysis for early vs late evidence. Each trial
should have its own sound stimulus. Some are easy some are hard (that is the
DV)." Both halves are now settled, by measurement rather than argument.

### The stimulus vector: found, and exactly reconstructible
The 'Natural' stimulus is a **linear superposition of two fixed sounds**:

    x(t; w) = w * A(t) + (1 - w) * B(t)

with A = frogs2.wav, B = Passer_Montanus.wav, and w = `AuditoryOmega`, saved
for every trial (`DV = 2w - 1` exactly).

Two of the four saved waveforms determine A and B per sample by solving a 2x2
system. Predicting the remaining two from w alone:

| held-out trial | omega | r | normalised RMSE |
|---|---|---|---|
| 955 | 0.8648 | **1.000000** | 0.0000 |
| 956 | 0.5469 | **1.000000** | 0.0000 |

So **the exact played waveform of all 953 trials is recoverable** from data
already in hand — the rig's `.wav` files are no longer needed. Gil was right
that the file contains all the trial information; it just stores the stimulus
as one parameter rather than as audio.

### Why early-vs-late has no referent: the mixing ratio is constant
Fitting the mixture ratio independently in 10 ms frames returns the trial's own
w in **every frame, with SD = 0.0000**:

| trial | true w | per-frame w (median, IQR, SD) |
|---|---|---|
| 953 | 0.8735 | 0.8735, 0.8735-0.8735, 0.0000 |
| 954 | 0.2511 | 0.2511, 0.2511-0.2511, 0.0000 |
| 955 | 0.8648 | 0.8648, 0.8648-0.8648, 0.0000 |
| 956 | 0.5469 | 0.5469, 0.5469-0.5469, 0.0000 |

The ratio is fixed for the whole 350 ms. Two trials sharing a w are the same
sound sample-for-sample. **There is therefore no trial on which evidence for
the correct side arrived early and none on which it arrived late** — the
variability the comparison requires does not exist in the design. This is now a
measured property of the stimulus, not an inference from missing data, and it
supersedes both earlier explanations (5aq's "not saved", 5as's "deterministic
morph").

Gil's framing is exactly right and is the reason: each trial has its own sound,
and its difficulty is DV — but difficulty lives in the *ratio*, a single number
held constant in time, not in a temporal sequence of evidence events.

### The within-trial version that IS possible — and is inconclusive
Differentiating the mixture with respect to w gives `dx/dw = A - B`, so the
instantaneous evidence rate is governed by `(A - B)^2`: moments where the two
recordings differ are informative, moments where they coincide are not. That
profile is fixed across trials, which is why it cannot split trials — but it
can be tested *within* the trial. Integration predicts the choice signal should
grow fastest while informative sound is arriving.

Measured over the played 350 ms in 25 ms bins: `(A-B)^2` power varies **53x**
between the least and most informative bin, spectral difference 11.2-18.5 dB.
Against the increment in choice AUC per step:

| evidence measure | Spearman | Pearson |
|---|---|---|
| `(A-B)^2` power | rho = +0.459, p = 0.114 | r = +0.361, p = 0.226 |
| spectral difference | rho = -0.036, p = 0.908 | r = -0.001, p = 0.998 |

**Inconclusive, and reported as such.** Only 13 windows fit inside the
stimulus, so power is very low; and the two ways of quantifying
"informativeness" disagree with each other (Spearman rho = -0.357 between them),
which means the operationalisation is not robust enough to carry a claim either
way. The positive trend on the power measure is the direction integration
predicts, and that is all that can honestly be said.

Artifacts: `outputs/stimulus_evidence_profile.csv`,
`outputs/reward_cue_figs/stimulus_evidence_profile.png`,
`post_ks_correction/stimulus_evidence_profile.py`.


---

## 5au. The mixture rule, verified exactly — and a loudness confound it exposes

Gil asked how the mixture is actually computed and how it was worked out. Five
properties, each measured (`verify_mixture.py`).

### The rule
    x(t) = omega * A(t) + (1 - omega) * B(t)

A plain weighted sum of two fixed waveforms. `omega` = `Custom.AuditoryOmega`,
one value per trial, constant for the whole 350 ms. A and B are the omega=1 and
omega=0 endpoints; the settings label them `soundA = frogs2.wav` and
`soundB = Passer_Montanus.wav`, which is the natural reading but cannot be
confirmed without the files themselves.

1. **Exact to double precision.** Solving for A and B from trials 953/954 and
   predicting the held-out two: **max absolute error 2.2e-16 and 1.7e-16** on
   signals of range 1.76 and 1.14. Not "a good fit" — machine epsilon.

2. **Exactly two sources, not three.** The SVD of the four waveforms gives
   singular values `[1, 0.465, 0, 0]`. They span a strictly 2-dimensional
   space, so no third component exists.

3. **omega -> side is deterministic.** Trials with DV > +0.5 have
   `LeftRewarded = 1` on **100.0%**; DV < −0.5 on **0.0%**. High omega (more of
   sound A) means LEFT is correct. (This also corrects a sloppy check in 5as
   that compared a boolean against a float array and printed "False".)

4. **No power normalisation.** Measured RMS matches the plain-weighted-sum
   prediction to five decimals on all four trials.

### The confound that falls out of (4), and it touches 5aq/5ar
Because the two sources partially cancel, a plain weighted sum is **quietest
near omega = 0.5** — exactly where the trial is hardest:

| trial | omega | \|DV\| | RMS |
|---|---|---|---|
| 956 | 0.547 | 0.09 (hardest) | **0.0697** |
| 954 | 0.251 | 0.50 | 0.0769 |
| 955 | 0.865 | 0.73 | 0.0862 |
| 953 | 0.874 | 0.75 (easiest) | **0.0869** |

That is a ~25% RMS span, about **2 dB louder on easy trials than hard ones**.
So `|DV|` is confounded with loudness in this task by construction.

**What that does to the accumulation result (5aq, replicated in 5ar).** The
finding was that the choice signal rises earlier and higher on strong-evidence
trials. Strong-evidence trials are also ~2 dB louder, so part of that ordering
could be a loudness effect rather than evidence integration. Two things keep it
from sinking the result: 2 dB is a small change against the dynamic range of
auditory responses, and the effect replicated across three sessions with the
same direction and similar magnitude. But it is a real alternative explanation
that the earlier sections did not know about and did not control, and it should
be stated whenever that result is quoted.

**It is also testable**, now that the stimulus is exactly reconstructible:
regress the decoding onset against loudness and |DV| jointly across trials,
since RMS is a known deterministic function of omega. Not yet done.

### How this was found, in order
1. Noticed `AuditoryStimulusType = 3` and read its popupmenu strings —
   `['Clicks','Freqs','Natural']` — which showed the `Aud_*` tone settings I had
   been reading belong to a variant this session never ran (5as).
2. Found `AuditoryOmega` present on every trial and verified `DV = 2*omega - 1`
   exactly, so the stimulus parameter was never missing.
3. Found four real waveforms in `AudSound` (the unplayed look-ahead trials) and
   noticed two of them had nearly equal omega and correlated at r = 0.9999 —
   evidence of a deterministic rule rather than a random draw.
4. Hypothesised a linear superposition, which is falsifiable: two waveforms at
   known omega determine A and B per sample, leaving the other two as a genuine
   held-out test. It passed at machine precision.
5. Fitted omega independently in 10 ms frames to ask whether the ratio moves
   within a trial. It does not (SD = 0.0000), which is what makes the
   early-vs-late comparison impossible by design (5at).

Artifacts: `verify_mixture.py` and `recover_sources.py` in the scratchpad,
`post_ks_correction/stimulus_evidence_profile.py`.




---

## 5av. The external proposal audited against our own code, and three of its four surviving ideas measured dead

Gil brought a detailed implementation specification for a Kilosort4 refinement
package (overcomplete representation, nuisance-aware waveform families, robust
reliability-aware learning) and asked two things: can we try all of it, and
does any of its *implementation* detail improve something we already tried.

5av answers the second by measurement. Section 1 is the code audit, section 2
the four experiments it motivated. The short version: of the four items I had
put on a keep-list after reading the proposal, **three are now measured and
dead, one survives untested**, and the audit also retired one concern I had
raised from code reading alone.

### 1. What the audit found by reading our own code

| plan item | our code | audit verdict |
|---|---|---|
| §4 families `m(t;a,τ,β)`, `q = −t·s′(t)` | `nuisance_model.fit_nuisance_prealigned` | **already built**, and ahead of the plan on identifiability (below) |
| §5.A.1 whitening applied to data AND templates | `test_post_hoc_on_v2.py` whitens only the post-hoc R² gate, never the matched filter | **real gap** → tested, no effect (E3) |
| §5.A.2 score the full informative footprint | footprint used only as a post-hoc gate; `multichannel_template` exists but is unused by the pipeline | **real gap** → tested, actively harmful (E3) |
| §6 one joint regularized scorer, not separate votes | three independent gates (mf, R², fp) | **real gap** → tested, worse than the gates (E1) |
| §6 competing hypotheses across units | one unit at a time, "is this mine?" | **real gap** → tested, 2× precision for 7× less recall (E2) |
| §8.A graph hook, identity features for the kNN graph only | footprint columns appended to `Xd` inside `cluster()` | **genuinely different, still untested** |
| §2 noise from representative blocks, not a quiet window | AR(4) from one 2 s min-spike-count window | tested, **exactly zero effect** (E3) |
| pin the baseline (Sept 2026 refractory-veto fix) | installed **4.0.24** | does not apply to us — veto is live |

**On §4 our implementation is ahead of the proposal.** The plan specifies the
same first-order family and then enforces identifiability by orthogonalising a
learned deformation against `span{s, s′, q}` in the whitened metric.
`nuisance_model.py` already documents why that is not the real problem: `s′`
and `q` are nearly parallel for real spikes (16–30°, cos 0.86–0.96) because
`⟨s′,q⟩ = −∫t·s′(t)²dt` vanishes only for time-symmetric derivative energy,
which real fast-rise/slow-fall spikes do not have. Gram–Schmidt does **not**
fix it — done correctly with back-substitution it reproduces the identical
fitted numbers, because OLS has one unique solution; the ambiguity is in what
the data can distinguish, not in the arithmetic. Our fix was to drop `s′`
entirely and supply τ from the independent phase-based estimator. That was
found the right way: τ and β came out correlated at r=0.77 on burst spikes and
**more strongly, r=0.97, on rested isolated spikes** that should show no
stretch, which ruled out physiology and pointed at basis geometry.

**Two verified facts about the installed source.** `clustering_qr.cluster()`
does `if kn is None: kn, M = neigh_mat(...)` and then uses `M` unconditionally,
so passing `kn` raises `UnboundLocalError` — the plan's implementation trap is
real and present in 4.0.24. And `swarmsplitter.split` reaches
`refractoriness()` with `meta=st0` passed at both live call sites, so the
disabled-veto bug the plan warns about is not in our baseline. Also confirmed:
`run_one` is dead code (never called, and would crash unpacking four returns
into three), and the splitter receives the **unaugmented** `Xd`, so our
footprint patch never reached the bimodality test.

### 2. E1 — one joint scorer instead of three gates: REFUTED

`audit_joint_scorer.py`, on the exact candidate table the 2.3% run produced
(87,607 candidates, all three configs). Ridge-logistic on (mf, R², fp),
unit-grouped CV so no unit trains and tests together.

| config | candidates | base rate | AUC mf | AUC R² | AUC fp | AUC joint (OOF) |
|---|---|---|---|---|---|---|
| vanilla | 28,908 | 1.72% | 0.517 | 0.521 | 0.528 | **0.486** |
| footprint_cluster_strong | 29,200 | 1.86% | 0.533 | 0.525 | 0.524 | **0.446** |
| coarse_then_align | 29,499 | 2.09% | 0.546 | 0.503 | 0.483 | **0.394** |

Recovery precision at matched acceptance count (vanilla, n = 11,027): current
two gates **2.35%**, mf alone 1.57%, R² alone 1.53%, fp alone 1.90%, joint
scorer **1.56%**. Every other config the same shape.

**Inside the candidate pool all three scores are at chance and the joint
scorer does not generalise across units at all.** The failure was never that
we combined evidence badly — there is no evidence left to combine once the
matched filter has proposed.

**This also puts 5q's AUC 0.999 in its place.** That was unit 440 vs unit 439,
one well-separated pair. Against the real mixture of everything a matched
filter proposes, footprint similarity scores **0.528**. The headline number
was a best case, not a representative one, and this is the first time it has
been measured against the real impostor population.

### 3. E2 — argmax over competing units: works correctly, and finds nothing

`audit_argmax_attribution.py`, rep0 vanilla, 12 placements, 5,098 candidates
(124 real, 2.43%). Each candidate's observed spatial energy vector is scored
against the target unit's footprint **and** against every cluster peaking
within 120 µm (2–28 competitors, median ~13), all on the **same channel set** —
the plan's "same union of channels across competing explanations".

| rule | accepted | real | precision |
|---|---|---|---|
| shipped two gates | 1,541 | 49 | 3.18% |
| **argmax over units** | 110 | 7 | **6.36%** |
| argmax AND gates | 73 | 4 | 5.48% |

Attribution doubles precision and costs 7× the recoveries. The number that
explains why:

**mean footprint similarity of a candidate to the target 0.8759, to its best
competitor 0.9757.**

An average candidate matches some neighbour's footprint *better than* the
target's. So argmax is not failing — it is working, and correctly concluding
that the candidates belong to other neurons. There is nothing for an
attribution rule to rescue. (Note also how compressed cosine between
non-negative energy vectors is: 0.88 for the right unit. Rule 1 again — a
score near 0.9 sounds like a match and is near the chance level here.)

Building the expected footprints on unfiltered raw voltage — a domain mismatch
the shipped pipeline really does contain — moves precision 6.36% → 8.33% on
84 accepts. Too few to mean anything, and pointing the harmless way.

### 4. E3 — the proposal stage, where the binding constraint actually is

`audit_proposal_stage.py`, rep0, 7 placements with missed spikes, **template
held fixed** so only the filter changes. Endpoint is what matters here: of the
unit's genuinely missed spikes, how many does a 800-candidate pool *contain*?
Everything downstream can only lose from there.

| variant | pool | captured of 441 missed | purity | per-placement vs A |
|---|---|---|---|---|
| **A single-channel, unwhitened (shipped)** | 5,257 | **115 (26.1%)** | 2.19% | — |
| B whitened, AR(4) from shipped quiet window | 4,924 | 101 (22.9%) | 2.05% | 2 better, 1 worse, 4 tied |
| C whitened, AR(4) from representative blocks | 4,924 | 101 (22.9%) | 2.05% | 2 better, 1 worse, 4 tied |
| D multichannel over footprint, unwhitened | 5,600 | 36 (8.2%) | 0.64% | **0 better, 4 worse, 3 tied** |
| E multichannel, whitened | 5,600 | 48 (10.9%) | 0.86% | 1 better, 1 worse, 5 tied |

**§5.A.1 whitening the detection statistic: no effect.** Applied consistently
to trace and template as the plan specifies, it is 2 better / 1 worse / 4 tied;
the pooled −14 spikes is one placement (unit 371, 72 → 56). This was the
genuine code gap I found by reading — whitening had only ever touched the R²
gate — and closing it changes nothing.

**B and C are identical on all 7 placements.** The noise-estimation flaw I
raised from code reading — AR(4) fitted on one 2 s window selected for having
the fewest spikes, which the plan warns underestimates noise — has **exactly
zero** measurable effect. A code-reading concern retired by measurement.

**§5.A.2 scoring the full footprint: consistently and substantially worse**,
26.1% → 8.2%, the only consistent direction in the table. Same mechanism 5q
diagnosed for the concatenated fit (0.897 vs 0.999): the sum is dominated by
the loudest channels, which is where neighbours look alike, and the
low-amplitude channels contribute noise no template explains. Whitening
recovers most of the loss (10.9%) without beating one channel.

Note `RADIUS_UM = 60` yields **10 channels** on this probe, so "full
informative footprint" here means 10, not a wide patch.

### 5. What this leaves

- **§6 is finished, both halves.** The joint scorer is worse than our gates;
  argmax attribution is correct and finds almost nothing, because the pool is
  genuinely other neurons' spikes (0.976 vs 0.876). **This retires the
  footprint-first attribution lead** that 6.10 has carried as the one open
  sorting idea. Three measurements pointed there; a fourth, direct one says
  the pool it would operate on does not contain the spikes.
- **§5.A is finished.** Neither whitening nor full-footprint scoring improves
  the proposal, and the proposal is the ceiling: the best variant captures
  **26.1%** of missed spikes, so even a perfect downstream scorer recovers at
  most a quarter of them.
- **§4 is already built**, and our version solved an identifiability problem
  the plan's version walks into.
- **§8.A, the graph hook, is the only genuinely new idea still standing.** Our
  footprint patch augments `Xd` inside `cluster()`, so footprint reaches both
  `kmeans_plusplus` seeding and `neigh_mat`; the plan's version touches the
  neighbour graph only. Worth doing because it is aimed at the measured
  failure (84% misfiling, 25.7% of all injected spikes) and because the
  over-seeding hypothesis (5aa: `nclust=200`, hard-coded, slicing a unimodal
  distribution) makes seeding and graph plausibly opposite in sign. Needs
  patching `neigh_mat` to return a consistent `(kn, M)` pair, not passing
  `kn`. Requires the GPU, so **Trodes must be closed** (mistake 4).

Artifacts: `outputs/audit_argmax_attribution.csv`,
`outputs/audit_proposal_stage.csv`; scripts `audit_joint_scorer.py`,
`audit_argmax_attribution.py`, `audit_proposal_stage.py` in the scratchpad.

**Honest limits.** E2 and E3 are one replicate and 7–12 placements, with 124
and 441 real spikes respectively; only D's penalty is consistent across
placements. E1 is all 4 replicates and 87,607 candidates and is the solid one.
E2 and E3 both use energy-cosine footprints and a single fixed template, so
they test the plan's *principles* with our best available discriminator, not
its full §6 machinery (signed multichannel whitened residuals with joint
refitting of neighbours). What makes the fuller version unpromising is not the
machinery but the 0.976-vs-0.876 fact: the candidates are other neurons'
spikes, which is a property of the pool, not of the scorer.


---

## 5aw. The graph hook built and run — the proposal's one surviving idea works, and it is the project's best sorting result

5av left exactly one idea from the external proposal standing: its §8.A
clustering hook, which puts identity features into the **kNN graph only** and
leaves the rest of clustering on PC features. Our own footprint patch appends
to `Xd` before `cluster()` is called and therefore moves two stages at once.
Built, run on all 4 replicates, and it is the first thing from the proposal
that helps — and it beats our own version of the same idea.

### Why the two stages are the whole story
`clustering_qr.cluster()` consults its feature matrix in exactly two places and
nowhere else:

    neigh_mat(Xd)        -> the (kn, M) graph; after this Xd is never read
                            again, because the iterative assignment loop is
                            purely graph-based
    kmeans_plusplus(Xg)  -> the initial seeding of nclust=200 centres

Three new configs in `ks_patches.py` hook them independently:
`footprint_graph` / `footprint_graph_strong` patch `neigh_mat` (graph only,
weights 1.0/3.0); `footprint_seed` patches `kmeans_plusplus` (seeding only).
Identical scaling to the combined patch, so all three are comparable.

Patching `neigh_mat` is **required**, not stylistic: `cluster()` does
`if kn is None: kn, M = neigh_mat(...)` and then uses `M` unconditionally, so
passing `kn` raises `UnboundLocalError`. The proposal flagged this trap and it
is real in the installed 4.0.24.

### THE RESULT — hard tier excluded, because it is not measurable
Paired Wilcoxon on identical placements, easy + collision + pair only
(noise floor on those three tiers is **bit-identical**, 0 of 40 placements
differing between two identical runs; the hard tier differs on 2 of 12 by up to
0.444 recall and is excluded throughout).

| config | stage | recall | precision |
|---|---|---|---|
| `footprint_cluster` | graph+seed, w1 | −0.0104 p=0.925 | **+0.0657 p=0.009** |
| `footprint_cluster_strong` | graph+seed, w3 | **+0.0320 p=0.015** | +0.0094 p=0.273 |
| `footprint_graph` | graph only, w1 | −0.0150 p=0.283 | +0.0517 p=0.252 |
| **`footprint_graph_strong`** | **graph only, w3** | **+0.0480 p=0.001** | −0.0332 p=0.245 |
| `footprint_seed` | seeding only, w1 | −0.0099 p=0.418 | +0.0470 p=0.086 |

**`footprint_graph_strong` gains +4.8 recall points at p=0.001** — half again
the effect of the previous best (`footprint_cluster_strong`, +3.2, p=0.015) and
an order of magnitude smaller p. It is **better in all four replicates** on
pair-tier recall (0.992 / 0.727 / 0.562 / 0.942 against vanilla's 0.965 /
0.551 / 0.500 / 0.866), so this is not one window. Overall recall 0.8173 is
the highest any configuration has reached, and mean fragments fall from 3.08
to 2.79.

**Precision is the cost, and it is not significant.** −0.0332 at p=0.245, so it
cannot be called a real loss, but the direction is consistent and the rule
stands: never quote the recall without it. The two weights still do different
jobs — w3 buys recall, w1 is where precision lives — and for precision the
**combined** patch at w1 remains the best thing we have (+0.0657, p=0.009),
unchanged by any of this.

### The decomposition: the graph does the work, the seeding does nothing
Paired tests of each half against the combination:

- **Removing the seeding half** (`footprint_graph*` vs `footprint_cluster*`) is
  **not significant on anything**, either metric, either weight. The seeding
  contributes nothing measurable.
- **Removing the graph half** (`footprint_seed` vs `footprint_cluster`) costs
  precision significantly: −0.0152 p=0.012 overall, −0.0505 p=0.049 on the
  pair tier. The graph half is load-bearing.
- **Seeding-only on its own does nothing** for recall (−0.0099, p=0.418).
- **The halves are sub-additive**: on pair-tier recall at w3, graph alone
  **+0.0854**, seeding alone +0.0284, their sum +0.1138, the combination
  actually measured **+0.0772**. Showing the footprint to the seeding slightly
  *cancels* the graph gain.

**This is what 5aa's over-seeding hypothesis predicts.** `nclust=200` is
hard-coded and absent from `parameters.py`, and an over-seeded kmeans++ places
boundaries along the direction of greatest within-unit spread. Giving the
seeding an extra 10 footprint dimensions hands it more directions to cut along,
which is why seeding-only buys no recall and why it drags on the graph-only
gain. Giving the *graph* those dimensions instead changes which spikes are
neighbours without adding cut directions. The two stages really do pull in
opposite directions, exactly the reason the proposal gave for separating them.

### The one real cost: merge-error protection is lost
Pair-tier merge errors (correct answer is TWO clusters, 8 pairs):

| config | merged |
|---|---|
| vanilla / vanilla_repeat | 1 of 8 |
| **both `footprint_cluster` configs** | **0 of 8** |
| `footprint_graph_strong` | 1 of 8 |
| `footprint_seed` | 1 of 8 |

The combined patch prevented the one merge stock Kilosort makes; neither half
alone does. One pair is a single observation and nothing should be built on it,
but it is the only axis on which the combined patch beats graph-only, and it is
the axis where a false win is most dangerous.

### Verdict
The proposal was right about this one, and specific enough to be actionable:
the narrow hook beats the broad one, for the mechanistic reason it gave. After
a long run of negative results this is a genuine, replicated, statistically
supported improvement to Kilosort on this data — **+4.8 recall points, p=0.001,
better in all four replicates, on tiers with a bit-identical noise floor.**

**Honest limits.** 52 placements, 4 replicates, one session, one animal. The
hard tier is excluded and no claim here covers loud neighbourhoods. The
recall gain comes with a non-significant precision cost and the loss of a
single-observation merge protection, so `footprint_graph_strong` is the right
choice when recall matters and `footprint_cluster` when precision does —
neither dominates. Everything is still judged against Kilosort's own
thresholds and PC basis, both tuned on unmodified data.

Artifacts: `outputs/v2_scores.csv` (12 configs, 624 placements),
`outputs/graph_vs_seed_paired.csv`, `outputs/v2_merge_errors.csv`,
`outputs/graph_hook_sweep.log`; code in `ks_patches.py`
(`_patched_neigh_mat`, `_patched_kmeans_plusplus`) and
`analyse_graph_vs_seed.py`.


---

## 5ax. A sorter that is not Kilosort, built and measured on our own benchmark — five iterations, one important negative

Gil asked whether the loop could be run iteratively to look for genuinely new
sorting methods, not necessarily involving Kilosort, tested on the hybrid
benchmark we built to evaluate Kilosort. It can, and it was: five iterations,
each one's design chosen from the previous one's measurement rather than from
intuition. The sorter loses to Kilosort end to end. The valuable output is not
the sorter but two measurements it made possible, one of which independently
confirms 5av's ceiling using a completely different instrument.

**First, a correction to the record.** Earlier sessions described this
benchmark as 13 units. 13 is the count *per replicate*. Across the four
replicates the benchmark contains **33 distinct source neurons in 52
placements**, with 18 neurons appearing in more than one replicate and
**rotated through tiers** between them — unit 331 is hard in rep0, pair in
rep1 and collision in rep3. That rotation is what makes tier effects
measurable within neuron rather than only across neurons, and it is a
stronger design than "13 units" suggests.

### Why a second sorter was worth building
Every result from 5a to 5aw is a modification of Kilosort4 judged against
Kilosort's own detection threshold, its own PC basis and its own `nclust=200`
seeding — all three tuned on unmodified data, which 5aw flagged as a standing
confound. And the one real win in the whole project (5aw: footprint in the kNN
graph, +4.8 recall at p=0.001, net flat on F1) says the across-channel energy
split carries identity information Kilosort's features do not use. LOCDEN was
the attempt to build a sorter around that fact from the start.

**The benchmark needed no modification to accept it.** `run_v2_comparison.score()`
reads only `spike_times.npy` and `spike_clusters.npy` from
`hybrid_v2_rep{r}/ks_{name}/`. Anything that writes those two files is scored
by the identical matching rule, the identical 10-sample tolerance and the
identical fragment definition as all 12 Kilosort configurations. That is worth
recording as a property of the harness: it was sorter-agnostic all along.

Two facts verified before writing code rather than assumed: truth times are
**trough samples**, and Kilosort's reported times agree with them to a median
of 0 and a 5–95% range of ±1 sample, so a new sorter must report troughs or
fail the matcher for a bookkeeping reason. And the probe is effectively
**one-dimensional** — 2 staggered columns spanning 48 µm in x, 192 rows at
20 µm — so a localizer here estimates depth, not 3-D position.

### Iteration 1 — per-channel threshold detection: dead, and the reason is the data
Classic architecture: bandpass + CAR, per-channel threshold on robust
z-scores, spatiotemporal dedup, point-source localization, global HDBSCAN.

Detection recall of injected spikes: **0.214 against a measured chance rate of
0.009.**

The first number computed was 0.556, and it was nearly reported. It was almost
entirely base rate: it allowed a match to any event **anywhere on the probe**,
and with 90,664 events over 3.6 M samples a random time has an event within
±10 samples with probability **0.415**. Restricting the match to within 40 µm
of the destination channel gives 0.214 against chance 0.009. Rule 1 again, and
this time it moved a headline by a factor of 2.6.

**Why no threshold can work here.** Measured directly, the injected units'
spike-triggered-average trough on their own destination channel is a median of
**3.06 noise SDs**, range **0.64 to 5.29**. Unit 371's deterministic signal is
0.64 SD — *below* the single-sample noise. Only 11.7% of injected spikes reach
z = −5 anywhere near their destination. A single-channel threshold detector is
the wrong instrument for this population, and the reason Kilosort reaches 0.88
recall is that whitened multichannel template matching integrates over ~10
channels and ~60 samples.

### Iteration 2 — whitened multichannel matched filter: a 4× better detector
Three additions, each aimed at the measurement above: spatial whitening (ZCA
over 384 channels), temporal matching against a **self-bootstrapped** basis
(the top 5 PCs of the 8,000 loudest waveforms iteration 1 detected — large
spikes teach the shape, the shape finds small ones), and pooling of the
projection energy over ±5 channels in depth order. Under noise the pooled
statistic is χ²(33).

| detector | events/s/chan | detection recall | chance |
|---|---|---|---|
| iteration 1, per-channel threshold | 1.97 | 0.214 | 0.009 |
| iteration 2, whitened matched filter | 4.02 | **0.949** | 0.015 |
| iteration 2, more headroom | 8.04 | **0.978** | 0.026 |

Over the full 120 s at the chosen operating point: **0.863**. Basis quality
mattered: PCs from all 90,664 waveforms explained 24/18/16% of variance and
gave recall 0.949; PCs from the loudest 8,000 explained **52/21/11%** and gave
0.978.

**Occlusion, not threshold, is the binding constraint.** With the
local-maximum footprint at ±10 samples × ±5 channels, recall saturates at
**0.476 no matter how low the threshold goes** — a quiet spike beside a loud
one is never a local maximum, so no threshold can reach it. Shrinking the
footprint to ±3 × ±1 lifted recall to 0.978 at the same event rate. This is
the clearest statement yet of why the collision and pair tiers are hard, and
it is a property of the detector, not of the clustering.

### Iterations 3 and 4 — the clustering stage is where it loses
With a detector at 0.978 detection recall, end-to-end performance against
stock Kilosort on the identical 13 rep0 placements:

| tier | recall LOCDEN | recall KS | precision LOCDEN | precision KS |
|---|---|---|---|---|
| easy | 0.511 | **1.000** | 0.191 | **0.781** |
| collision | 0.601 | **0.961** | 0.221 | **0.933** |
| pair | 0.522 | **0.965** | 0.089 | **0.749** |
| hard | 0.439 | 0.574 | 0.051 | 0.365 |
| **ALL** | **0.519** | **0.882** | **0.134** | **0.710** |

The entire loss is in clustering, and two causes were visible in the output:
**under-clustering** (134 clusters where Kilosort finds 216 — a single HDBSCAN
over a 3.8 mm probe finds broad density modes, not neurons, which is exactly
the 0.134 precision) and **too few features** (3 PCs of the peak channel only,
against Kilosort's ~10 channels × 6 features).

Iteration 4 fixed both in the obvious way — clustering in 8-channel depth
bands with an 8-channel context margin, the L2-normalised 20-channel footprint
entering as PCs alongside fitted depth, and a cross-band merge requiring
footprint cosine ≥ 0.97 **and** waveform cosine ≥ 0.95 **and** depth within
15 µm (all three, because 5av measured footprint alone as a weak
discriminator). It did not help: 156 band clusters, **0 merges**, and the
refractory rule removed **62.1%** of all events.

**Why, measured.** Mean **3.35 detections per real spike** (median 3, p90 6,
max 14). The small local-maximum footprint that cured occlusion necessarily
admits multiple detections of one spike. The two failure modes trade off
directly against each other, and the only thing that resolves both at once is
**iterative template subtraction** — detect the strongest event, subtract its
fitted waveform, re-detect. That is the core of Kilosort, SpyKING CIRCUS and
YASS, and it is what LOCDEN does not have. That is the honest explanation of
the gap, and it is an engineering gap, not a conceptual one.

### Iteration 5 — the result worth keeping: 5av's ceiling confirmed by a different instrument
The detector is strong even though the sorter is not, so the question worth
asking is whether it finds the spikes **Kilosort** misses. 5av had established
the ceiling on post-hoc recovery: with the template held fixed, the best
candidate-proposal variant captured **26.1%** of a unit's genuinely missed
spikes in an **800-candidate** pool, and neither of the external proposal's
two recommended changes beat it.

Raw answer: **0.843 of 445 missed spikes captured, chance 0.015.** Three times
the ceiling.

**That number is a pool-size artifact and does not survive.** 5av's pool was
800 candidates per unit; the LOCDEN events within 40 µm of a unit over 120 s
number **10,353**. Held at matched pool size, ranking candidates by the
detection statistic:

| pool size | capture | chance |
|---|---|---|
| 200 | 0.018 | 0.003 |
| 400 | 0.034 | 0.001 |
| **800 (5av's size)** | **0.070** | 0.001 |
| 1,600 | 0.083 | 0.001 |
| 3,200 | 0.164 | 0.003 |
| all 10,353 | 0.843 | 0.014 |

Capture scales almost exactly **linearly** with pool size: 800/10353 × 0.843 =
0.065 predicted by an uninformative ranking, 0.070 measured. **The detection
statistic carries essentially no information about which candidates are the
missed spikes.** On the non-hard tiers at 800 candidates, capture is **0.243
against 5av's 0.261** — indistinguishable, on 74 missed spikes.

This is the valuable finding. 5av's 26.1% ceiling was measured with one family
of single-channel matched filters, which left open the possibility that a more
sensitive detector would break it. A completely different detector — spatially
whitened, 5-component learned temporal basis, 11-channel pooling, 4× the
detection sensitivity on injected spikes — hits **the same wall at the same
height**. The ceiling is not about detector sensitivity. It is about *ranking*,
and 5av already measured the ranking scores at chance (AUC 0.48–0.55 over
87,607 candidates). Two independent routes, same conclusion: **post-hoc
recovery of Kilosort's missed spikes is dead, and it is dead for a reason that
does not depend on the proposal stage.**

Note also that 371 of rep0's 445 missed spikes belong to the **hard** tier,
which is not measurable in this benchmark (two identical Kilosort runs disagree
there by up to 0.444 recall), so the non-hard rows are the ones carrying weight
and they rest on 74 spikes. That is thin, and it is the main limit on this
section.

### Verdict
Running the loop on genuinely new methods works, and the harness supports it
for free. What it produced:

- **A detection stage that is competitive and cheap.** 0.978 in-window /
  0.863 full-recording detection recall at a Kilosort-comparable event rate,
  CPU-only in ~5 minutes per replicate — so unlike every Kilosort run in this
  project it does **not** require the GPU, which means no Trodes conflict.
- **A quantitative reason the hard tiers are hard** that is independent of
  clustering: occlusion pins detection recall at 0.476 for any
  physically-sized local-maximum footprint, and the cure costs 3.35 duplicate
  detections per spike. Only deconvolution resolves both.
- **Independent confirmation of 5av's ceiling** with a 4×-more-sensitive
  detector, which closes off the "we just need better candidate proposal"
  escape route that 5av could not rule out.
- **A corrected description of the benchmark**: 33 distinct neurons, 52
  placements, units rotated through tiers.

What it did not produce is a better sorter: **0.519/0.134 against Kilosort's
0.882/0.710**, and the gap is the deconvolution stage. Recommendation on the
sorting question is unchanged — `footprint_cluster` remains the best
configuration measured (F1 +0.044, p=0.002).

**Honest limits.** One replicate (rep0) for every end-to-end number; the
iteration-2 sweep is one 10 s window; the matched-pool comparison rests on 74
non-hard missed spikes; no LOCDEN result is replicated across the four
replicates the way the Kilosort configurations are. Nothing here should be
quoted as a four-replicate result, and the sorter's loss to Kilosort is large
enough that replication would not change the conclusion.

Artifacts: `outputs/locden_vs_ks_misses.csv`,
`outputs/locden_matched_pool.csv`; code in `sorter_locden.py`
(`detect`, `detect2`, `localize`), `cluster_locden.py`, `cluster_locden2.py`,
`diag_locden_detection.py`, `diag_injected_amplitude.py`,
`diag_detect2_sweep.py`, `diag_locden_vs_ks_misses.py`,
`diag_locden_matched_pool.py`. Event caches live in
`hybrid_v2_rep{r}/ks_locden_cache/` (gitignored, regenerable).

---

## 5ay. Four third-party sorters on our benchmark, a time-convention error corrected, and the deconvolution question answered without building it

Gil asked for three things — iterative template subtraction, a multi-axis
wavelet-aligned feature representation, and a general think about what else is
worth trying — and mentioned a MountainSort environment somewhere on D. That
last remark was the most valuable part of the message.

`D:\conda_envs\si_ms5` holds SpikeInterface 0.104.8 with five **installed**
sorters (`mountainsort5`, `spykingcircus2`, `tridesclous2`, `lupin`, `simple`)
and `isosplit6` importable on its own. Two consequences, one of which
retires a question that would otherwise have cost several hundred lines.

### 1. A TIME-CONVENTION ERROR THAT WAS CORRUPTING EVERY EXTERNAL COMPARISON

5ax recorded that "truth times are **trough samples**, and Kilosort agrees to
a median of 0 and ±1 sample." The second half is true. **The first half is
wrong**, and the agreement was over-read.

Measured directly, as the spike-triggered average on each unit's destination
channel at the truth times: the **physical trough sits a median of +13 samples
AFTER the truth time** (range +8 to +17 across the 13 rep0 units; unit 331 is
inverted at −1). Truth times are the `nt0min=20` index of the 61-sample
extraction window. Kilosort uses that same `nt0min` convention, which is
exactly *why* truth and Kilosort agree at 0 — and that agreement never implied
either sits at the trough.

**Consequence.** A sorter that reports the real trough is ~13 samples outside
the scorer's 10-sample tolerance and scores near zero for pure bookkeeping:

| sorter | scored naively | at its own alignment |
|---|---|---|
| `tridesclous2` | 0.276 / 0.492 | **0.562 / 0.900** |
| `mountainsort5` | 0.089 / 0.221 | **0.323 / 0.691** |
| `lupin` | 0.101 / 0.203 | **0.689 / 0.778** |

MountainSort5 was about to be written down at 0.089 recall. Lupin would have
been written down at 0.101 instead of 0.689 — a factor of **6.8**.

**Is sweeping the shift cheating?** It uses the ground truth, so it needs a
justification rather than a wave-through. It is a one-integer-per-sorter
*convention calibration*, not parameter tuning: it cannot invent spikes,
cannot change which clusters exist, and cannot trade recall against precision
— a wrong shift lowers **both**, as every sweep shows. And the optimum is a
broad plateau (tridesclous2: −16 through −10 within 0.003 recall; lupin: −16
through −12 within 0.001), so it identifies a convention rather than fitting
noise. Both columns are reported everywhere so the size of the correction
stays visible. A truth-free alternative was built first
(`run_external_sorters.realign`, per-cluster template trough from the
recording alone) and it correctly returns offset 0 for MountainSort5 —
confirming MS5 really is at the trough and that the mismatch is on the
benchmark's side, not the sorter's.

### 2. THE FIRST EXTERNAL REFERENCE POINT THIS PROJECT HAS EVER HAD

Every sorting number from 5a to 5ax compares Kilosort to our own
modifications of Kilosort. Nothing external has ever said whether 0.882 recall on
these placements is good, average or poor.

rep0, 13 placements, identical matching rule, each sorter at its own best
alignment:

| sorter | spikes | clusters | recall | precision |
|---|---|---|---|---|
| **`footprint_cluster` (ours)** | 233k | 238 | **0.909** | **0.832** |
| vanilla Kilosort4 | 224k | 216 | 0.882 | 0.710 |
| **`lupin`** (template matching) | 178k | 244 | 0.689 | 0.778 |
| `locden_iso` (ours, 5ay below) | 149k | 76 | 0.604 | 0.212 |
| `tridesclous2` | — | 190 | 0.562 | **0.900** |
| `mountainsort5` | 23k | 48 | 0.323 | 0.691 |

**Kilosort4 is a strong baseline on this data, not a convenient one.** Three
independent third-party sorters, two of them template-matching, all score
below it on recall. Our `footprint_cluster` beats every one of them on both
metrics.

**But precision headroom is real and externally demonstrated.**
`tridesclous2` reaches **0.900** precision and `lupin` **0.778**, against
vanilla Kilosort's 0.710 — at much lower recall in both cases. So the
contamination Kilosort carries is removable in principle; nobody has to take
0.710 as a floor.

`spykingcircus2` could not be run: it needs the standalone `hdbscan` package
for its `iterative_hdbscan` clustering stage, which si_ms5 lacks. Not
installed, to avoid disturbing a working environment. Its first failure was a
different missing dependency (`torch`, for dredge motion correction); that one
was fixed by `apply_motion_correction=False`, which is also the **matched**
setting, since every Kilosort run here uses `nblocks=0`.

### 3. THE DECONVOLUTION QUESTION, ANSWERED WITHOUT BUILDING IT

5ax identified iterative template subtraction as the one component LOCDEN
lacks and the only remaining structural idea on sorting, and quantified the
tension that makes it necessary: an occlusion-safe detection footprint pins
recall at 0.476, and the cure costs 3.35 detections per real spike. The
obvious next step was to build it. `lupin` makes that unnecessary as a first
step, because it *is* template matching, and it is mature.

Per tier, `lupin` against vanilla Kilosort:

| tier | lupin recall | KS recall | lupin precision | KS precision |
|---|---|---|---|---|
| easy | 0.930 | 1.000 | **0.937** | 0.782 |
| **collision** | **0.596** | **0.961** | 0.703 | 0.933 |
| pair | 0.874 | 0.965 | **0.910** | 0.749 |
| hard | 0.295 | 0.574 | 0.516 | 0.365 |

**The collision tier is where a deconvolution sorter should win, and it is
where lupin loses worst** — 0.596 against Kilosort's 0.961. The collision tier
is built from spikes placed deliberately 4–25 samples from a resident
neuron's spike; resolving exactly that overlap is what subtraction is for. On
easy and pair, where overlap is not the challenge, lupin's precision is
*better* than Kilosort's (0.937 and 0.910 against 0.782 and 0.749).

So the expected value of writing our own deconvolution engine is **much lower
than 5ax assumed**. A mature implementation at defaults does not beat
Kilosort here and is specifically weakest on the overlap tier. That does not
prove subtraction cannot work — lupin is untuned and this is one replicate —
but it removes the presumption that it is the obvious next build, for the
price of one background job instead of a few hundred lines.

### 4. GIL'S MULTI-AXIS REPRESENTATION: the biggest jump LOCDEN has had

25 **named physical** axes rather than three peak-channel PCs — trough depth,
peak before and after, peak-to-trough amplitude and duration (the classic
striatal fast-spiking vs projection-neuron discriminator), trough half-width,
repolarization and recovery slope, asymmetry, energy, trough area, zero
crossings, skew, kurtosis, per-event best-match frequency `f0`, spectral
centroid, fitted depth/distance/amplitude/fit-quality, footprint spread and
dorsoventral asymmetry, channels above half-maximum, and propagation slope.
PCs were avoided deliberately: they spend variance on whatever varies most,
including sub-sample jitter, which 5ad measured Kilosort wasting a component
on.

Clustered with **ISO-SPLIT** instead of HDBSCAN:

| LOCDEN variant | recall | precision |
|---|---|---|
| v1, HDBSCAN, 3 peak-channel PCs | 0.519 | 0.134 |
| v2, band-wise HDBSCAN + footprint | 0.538 | 0.172 |
| **ISO-SPLIT + 25 multi-axis** | **0.604** | **0.212** |

Pair-tier recall 0.713. Still far behind Kilosort, but the first two
improvements to LOCDEN's clustering that both metrics agree on.

**ISO-SPLIT was verified before being trusted**, and it is well matched to
this project's diagnosed failure. 5aa found `nclust=200` hard-coded and absent
from `parameters.py`; 5aw found that giving that seeding extra dimensions
*cancels* part of the graph gain — the signature of an over-seeded k-means
slicing unimodal units. ISO-SPLIT takes no k and no minimum cluster size, and
splits only where a projection fails a non-parametric unimodality test. On
**200,000 points of pure Gaussian noise it returns ONE cluster, not 200** —
the exact opposite failure mode — recovers five unequal well-separated blobs
at ARI 1.000 in 0.06 s, and is ~35× faster than the HDBSCAN it replaces
(4.6 s for 200k×10 against 170 s for 150k×6). It does call `abort()` on
exactly-duplicate rows, which matters because several of the new axes are
integer-valued, so the features get a jitter far below their own scale to
break ties.

### 5. THE WAVELET HALF OF GIL'S IDEA DOES NOT WORK, AND THE TRUTH FILE PROVED IT

The benchmark builder drew a sub-sample shift for every injected spike from
U(−0.5, 0.5) and **stored it** in the truth file as `shifts`. So the alignment
has a per-spike correct answer, which makes this a direct accuracy test.

On 1,240 matched injected spikes:

| | value |
|---|---|
| correlation with the known shift | **−0.061** |
| RMSE of the wavelet estimate | **1.402** samples |
| RMSE of answering zero for every spike | **0.292** samples |

It is **4.8× worse than no alignment at all**, and no better on the loudest
half (corr −0.065). 61.6% of estimates saturated at the clamp. It is now off
by default, with the measurement written beside the switch.

**This does not contradict the earlier wavelet work, and the reason is in our
own docstring.** `coarse_then_fine_shift` states that phase-only correction is
trustworthy only within half an oscillation cycle of the true spike time,
reached by a **coarse template match first**, with a reference phase
**calibrated per unit**. At this point in a from-scratch sorter there are no
units, so there is no template to calibrate against and no coarse stage to
run. Phase-only alignment was being used outside its validated operating
range.

**The fix is an ordering, not a parameter.** Cluster first on the axes that
are insensitive to sub-sample timing (amplitude, footprint, fitted position,
|W| frequency), then align inside each cluster against that cluster's own
template with a calibrated reference phase, then recompute the shape axes and
re-cluster. That is an alternating scheme with the same shape as the
deconvolution loop, and all the machinery for it already exists.

### 6. WHAT THIS LEAVES, with the dead ends marked

Checked against the log rather than proposed from intuition:

- **Merging fragments — DEAD, at the oracle level.** 5al tested merging
  knowing the true fragments and it trades down in **all 16** algorithm × tier
  cells (+3 to +7 recall for −4 to −22 precision); 67–77% of units have
  exactly one substantial cluster. Any ISO-SPLIT-based merge is strictly worse
  than that oracle, so this must not be revisited.
- **Post-hoc spike recovery — DEAD from both ends** (5av: ranking scores at
  chance over 87,607 candidates; 5ax: a 4.6×-more-sensitive detector hits the
  same 26.1% ceiling).
- **Own deconvolution engine — DEPRIORITISED** by §3 above.
- **SPLITTING Kilosort's mixed clusters with a unimodality test — the live
  lead.** Untested, and it is the direction merging failed in rather than the
  same one. Externally demonstrated headroom: tridesclous2 reaches 0.900
  precision and lupin 0.778 against vanilla Kilosort's 0.710. `isocut6` gives
  a principled one-dimensional test, and it runs post-hoc on saved output with
  no GPU, so Trodes can stay open.
- **Alternating align → cluster → realign**, per §5.
- **Consensus across sorters.** Six-plus independent outputs now exist on
  identical data; keeping only spikes two or more sorters agree on targets
  precision directly and costs almost nothing to test.
- **Extending the benchmark past 33 donor neurons** — more replicates with
  fresh donors, and a drift-containing variant.

**Honest limits.** Every number here is rep0 and 13 placements. All four
third-party sorters ran at or near defaults, so none of these is that
sorter's best achievable result and none should be quoted as a verdict on the
software — `mountainsort5`'s defaults in particular contain the two settings
this project has already measured to be wrong for this data
(`detect_threshold` 5.5 when the injected units average 3.06 σ, and a 0.5 ms
exclusion radius, the same occlusion trap that pinned our own detector at
0.476). A tuned MountainSort5 run was still in flight when this was written.
The alignment correction rests on a ground-truth sweep, justified in §1 but
not free of it.

Artifacts: `outputs/external_sorters_rep0.log`,
`outputs/external_sorters_rep0b.log`, `outputs/wavelet_alignment_truth.csv`;
code `run_external_sorters.py`, `features_locden.py`, `cluster_isosplit.py`,
`diag_wavelet_alignment_truth.py`, `score_all_fair.py`.
## 5az. What recall and precision actually measure here, a per-unit alignment that the Kilosort control validates, and a LOCDEN result retracted

> **READ 5ba FIRST. Every absolute number in this section is rep0 only, and
> rep0 is the easiest of the four replicates.** Stock Kilosort scores
> 0.882/0.710 here but 0.794/0.600 pooled over four replicates, which is what
> the Shamir Sorting Audit reports. The claim below that footprint_cluster
> beats stock in 6 of 8 tier cells is a rep0 artifact: pooled, its recall is
> 2.3 points LOWER than stock and its win is precision only (+5.4, p=0.010).
> What survives from this section is the definition of the metrics, the timing
> measurements, the per-unit alignment method and its control, and the two
> retractions.

Gil asked three questions about the 5ay table that turned out to matter more
than the table: how recall and precision are computed, whether the numbers
were easy-tier only, and whether 238 clusters for 13 injected units is itself
a bad sign. Answering them properly exposed one methodological flaw of my own
and killed one result I had reported an hour earlier.

### What the two numbers actually cover

From `run_v2_comparison.score()` (lines 90-142), for each of the 13 injected
units:

  * collect every cluster with a spike within TOLERANCE = 10 samples
    (0.33 ms) of any of that unit's true spikes;
  * keep **the single cluster with the most matches**, matched greedily
    one-to-one so one detected spike cannot satisfy two true spikes;
  * `recall = matched / that unit's true spike count`;
  * `precision = matched / all spikes in that one cluster`;
  * the headline figure is the **mean over the 13 units**.

So **the metrics describe 13 clusters out of 238. The other ~225 are never
scored at all** -- measured, not assumed:

| sorter | injected units | clusters emitted | clusters the metrics touch |
|---|---|---|---|
| footprint_cluster | 13 | 238 | 13 |
| vanilla KS4 | 13 | 216 | 13 |
| tridesclous2 | 13 | 185 | 13 |
| mountainsort5 | 13 | 48 | 11 |

**238 clusters for 13 units is not inflation.** The benchmark is Shamir's real
striatal recording with 13 synthetic units added on top; the other ~225
clusters are resident neurons that were already in the tissue, have no ground
truth, and therefore cannot be scored. This is the same point as 5ag, restated
because the 5ay table invites the opposite reading.

Two consequences worth keeping in front of anyone reading these tables:

  * **"precision" is the purity of one cluster, not a false-positive rate.**
    A matched cluster that also swallowed a resident neuron scores low. That
    is a fair penalty -- a cluster you would accept as one unit is not one --
    but it is not "X% of all spikes are wrong".
  * **recall is credited against one cluster only.** A unit split across two
    clusters is credited for the larger. Deliberate, since a split unit is a
    failure for the user; splits are carried separately as `n_fragments`.

### Per tier, which 5ay never printed

rep0, 13 placements, each sorter at its own best alignment (recall/precision):

| sorter | easy (n=3) | collision (n=3) | pair (n=4) | hard (n=3) |
|---|---|---|---|---|
| **footprint_cluster** | **0.986/1.000** | **0.954/1.000** | **0.971/0.895** | **0.711/0.411** |
| vanilla KS4 | 1.000/0.781 | 0.961/0.933 | 0.965/0.749 | 0.576/0.365 |
| lupin | 0.930/0.937 | 0.613/0.788 | 0.874/0.910 | 0.298/0.521 |
| tridesclous2 | 0.723/0.965 | 0.610/0.862 | 0.519/0.907 | 0.409/**0.866** |
| mountainsort5 | 0.632/0.992 | 0.303/0.658 | 0.255/0.742 | 0.133/0.335 |

`footprint_cluster` beats stock Kilosort in 6 of 8 tier cells; both losses are
trivial (easy recall 0.986 vs 1.000, collision recall 0.954 vs 0.961). It
remains the one configuration in this project that is genuinely better than
stock Kilosort, at F1 +0.044, p = 0.002 on non-hard tiers over all four
replicates (5aw).

### The flaw in 5ay's own fair-scoring method

5ay justified sweeping one global integer shift per sorter as a "convention
calibration", partly on the grounds that the optimum was a broad plateau.
`diag_timing_jitter.py` tested that claim per spike rather than per sorter:

| sorter | median offset | per-unit medians | per-spike MAD | within +-2 of own median |
|---|---|---|---|---|
| footprint_cluster | +0 | +0..+0 | 0.0 | 99% |
| vanilla KS4 | +0 | +0..+0 | 0.0 | 99% |
| lupin | +18 | +9..+20 | 1.0 | 87% |
| tridesclous2 | +12 | +7..+17 | 4.0 | 28% |
| mountainsort5 | +14 | +4..+17 | 3.0 | 39% |
| locden_iso | +9 | +5..+15 | 3.0 | 39% |

**The offset is not one constant per sorter -- it varies per unit, by 10 to 13
samples.** That is not a sorter property: it is the per-unit trough position
of the donor waveforms, measured in 5ay at +8..+17. One global integer shift
cannot bring a 13-sample spread inside a 10-sample tolerance, so 5ay's shifted
numbers were an under-correction for exactly the sorters they were meant to
rescue. The plateau argument held only for Kilosort, whose plateau is flat at
0.908-0.910 across shifts -8..+6 -- fifteen samples wide because the tolerance
is +-10, the signature of a single clean convention. `locden_iso`'s profile is
ragged by comparison (0.739 at -20, 0.890 at -14, 0.806 at -4, 0.604 at 0),
because wide per-spike jitter means each candidate shift captures a different
subset of spikes.

`score_per_unit_fair.py` therefore scores with a **per-unit** offset. The
shift search is a vectorised offset histogram; the reported score at the
chosen shift is the project's own exact greedy `_match`, so the numbers stay
comparable to every other table in this log. **The control that makes it
readable: Kilosort shares the benchmark's convention, so per-unit freedom
should buy it nothing.**

| sorter | global shift (5ay) | per-unit | gain |
|---|---|---|---|
| vanilla KS4 | 0.882/0.710 | 0.882/0.710 | **+0.000/+0.000** |
| footprint_cluster | 0.909/0.832 | 0.911/0.832 | **+0.002/+0.000** |
| tridesclous2 | 0.562/0.900 | 0.562/0.900 | +0.000/+0.000 |
| mountainsort5 | 0.323/0.691 | 0.325/0.687 | +0.002/-0.004 |
| lupin | 0.689/0.778 | 0.694/0.799 | +0.005/+0.021 |
| locden_iso | 0.890/0.241 | 0.920/0.284 | +0.030/+0.043 |

The control holds: **+0.000 for Kilosort and +0.002 for footprint_cluster**, so
the selection bias of taking a maximum over 33 shifts is about 0.002 and
anything above that is real. The worry that drove the rebuild was mostly
wrong, though: tridesclous2 and mountainsort5 gained essentially nothing,
meaning a global shift had already absorbed their per-unit spread. Only
`locden_iso`, the sorter with the worst timing, gained materially.

Two bugs were caught in building it, both worth recording because the first
produced plausible-looking output. (1) `a.shift` silently resolved to
`DataFrame.shift`, the method, rather than a column -- hence the column is now
`off`. (2) The offsets are `(detected - true)`, so the corrective shift is
`-off`, but the search tested `|off - s| <= tol`. With the sign wrong and ties
broken by scan order, `footprint_cluster` reported a +0-offset sorter as
needing -9 while still scoring 0.908, because -9 is inside the tolerance. The
fix was the sign plus **breaking ties toward s = 0**, without which a
correctly-aligned sorter reports an arbitrary shift.

**Cross-checked against an independent implementation.** The first attempt at
per-unit scoring was a brute-force version that called the exact greedy
`_match` at every one of the 33 shifts for every candidate cluster -- correct
but slow enough to need a background run. It finished after the vectorised
scorer had already been written, which makes it a free check on it, and **all
twelve headline figures and all twenty-four per-tier cells agree to three
decimals**. The two differ only in the *offset* they report: the brute-force
version has no tie-break, so scanning from -24 it reports an arbitrary member
of the tolerance plateau (lupin -24, tridesclous2 -18) where the vectorised
version reports the plateau's centre. The scores are identical because every
shift inside the plateau scores the same -- which is precisely what made the
earlier sign bug dangerous: a wrong offset with a right-looking score.

### RETRACTED: locden_iso's recall is a merging artifact

At per-unit alignment `locden_iso` scores recall **0.920**, the highest of any
sorter tested, with hard-tier recall 0.837 against stock Kilosort's 0.576.
Reported as such, that would be the project's most striking result. It does
not survive its own controls.

**Recall does not penalise merging.** A cluster holding three neurons contains
nearly all of each one's spikes. `locden_iso` emits 76 clusters where Kilosort
emits 238, so `diag_recall_chance_and_bloat.py` measured both controls: a
chance rate from permuting cluster labels while preserving every cluster size
and every spike time, and the bloat factor of the matched cluster.

| sorter | recall | CHANCE recall | matched-cluster bloat | clusters |
|---|---|---|---|---|
| footprint_cluster | 0.911 | 0.072 | **1.00x** | 238 |
| vanilla KS4 | 0.882 | 0.073 | 1.00x | 216 |
| lupin | 0.694 | 0.068 | 1.11x | 244 |
| tridesclous2 | 0.562 | 0.087 | 1.09x | 185 |
| mountainsort5 | 0.325 | 0.092 | 1.02x | 48 |
| **locden_iso** | **0.920** | **0.237** | **8.16x** | 76 |

`locden_iso`'s chance baseline is **3.3x everyone else's**, and its matched
clusters are **8.16x larger than the part that belongs to the unit**, against
1.00x for both Kilosort configurations. Its recall is bought by
under-clustering, not by finding spikes Kilosort misses. **The claim that
LOCDEN's recall is at or above Kilosort's is withdrawn.** The honest statement
is that LOCDEN is still badly under-clustered, which its precision of 0.284
already said; the recall number added nothing.

This is also the cleanest demonstration in the project of why the standing
rule exists that every "X is near Y" statistic must carry its measured chance
rate. The same rule killed the raw 0.843 capture figure in 5ax, by pool size,
and it killed this one, by cluster size.

### What stands after all of it

  * `footprint_cluster` at **0.911/0.832** vs stock Kilosort's 0.882/0.710 is
    the project's one real improvement, now with a per-tier breakdown and an
    alignment method validated by its own null control.
  * **tridesclous2's precision is the live headroom**: 0.862-0.965 in every
    tier, including **0.866 on hard where ours is 0.411**. It finds far fewer
    spikes, but what it finds is clean. This is the strongest evidence yet
    that Kilosort's contamination is removable, and it is what the
    cluster-splitting lead targets.
  * **lupin's numbers are fair** -- MAD 1.0, 87% of spikes within +-2 samples,
    so a global shift genuinely fixes it. The 5ay deconvolution conclusion,
    which rested on lupin's collision tier, therefore stands unchanged.
  * **`ms5_tuned` is worse than untuned**, 0.212/0.217 against 0.323/0.691,
    collapsing on collision to 0.066/0.004. The tuning in 5ay's ALIAS dict was
    chosen from our own occlusion measurement and was simply wrong; untuned
    MountainSort5 defaults stand.

Artifacts: `outputs/per_tier_rep0.csv`, `outputs/per_unit_fair_rep0.csv`,
`outputs/per_unit_fair_<sorter>_rep0.csv`, `outputs/timing_jitter_rep0.csv`;
code `score_per_unit_fair.py`, `diag_timing_jitter.py`,
`diag_recall_chance_and_bloat.py`.

All of 5az is **rep0 only**, 13 placements. The footprint_cluster-vs-vanilla
comparison is the exception: that one is pooled over four replicates (5aw).

## 5ba. CORRECTION: 5az's tables are rep0 only, rep0 is the easiest replicate, and the audit's mediocre numbers are the correct ones

Gil asked why 5az reports roughly 0.9 recall and 0.9 precision on every tier
but hard, while the Shamir Sorting Audit reports mediocre numbers for the same
tests. The answer is that **the audit is right and 5az was reported badly.**
Every 5az table is rep0 only, rep0 is the easiest of the four replicates, and
I led with those point estimates instead of the pooled figures.

### The audit's numbers reproduce exactly

The audit reports stock Kilosort4 per tier as easy 88/76, collision 83/59,
pair 72/49, with the hard tier deliberately omitted as not reproducible.
Re-scoring all four replicates confirms it to the digit: easy .883/.765,
collision .832/.594, pair .720/.486, pooled **.794/.600** over 52 placements.
Nothing is wrong with the audit.

### rep0 is the easiest window, by a lot

Stock Kilosort4, same code, same 13 placements per replicate:

| replicate | recall | precision | easy | collision | pair |
|---|---|---|---|---|---|
| **rep0** | **0.882** | **0.710** | 1.00/0.78 | 0.96/0.93 | 0.97/0.75 |
| rep1 | 0.674 | 0.518 | 0.92/0.81 | 0.59/0.50 | 0.55/0.30 |
| rep2 | 0.764 | 0.529 | 0.65/0.59 | 1.00/0.63 | 0.50/0.21 |
| rep3 | 0.855 | 0.642 | 0.96/0.88 | 0.77/0.32 | 0.87/0.69 |
| **pooled** | **0.794** | **0.600** | .883/.765 | .832/.594 | .720/.486 |

rep0 is the best replicate on both metrics. **Reporting it alone inflates
recall by 8.8 points and precision by 11.0.** Worse, the per-tier spread
across replicates is enormous -- pair recall runs 0.97, 0.55, 0.50, 0.87 and
collision precision runs 0.93, 0.50, 0.63, 0.32 -- because a tier within one
replicate is **3 or 4 units**. That is precisely the n=3 problem that motivated
building the four-replicate v2 benchmark in the first place (5ab); 5az walked
straight back into it.

### The substantive error: footprint_cluster's recall is not an improvement

5az said footprint_cluster "beats stock Kilosort in 6 of 8 tier cells" at
0.911/0.832 against 0.882/0.710, and called it the project's one real
improvement. Pooled over the four replicates:

| | recall | precision | easy | collision | pair | hard |
|---|---|---|---|---|---|---|
| stock KS4 | **0.794** | 0.600 | .883/.765 | .832/.594 | .720/.486 | .764/.593 |
| footprint_cluster | 0.771 | **0.654** | .862/.867 | .805/.572 | .730/.591 | .700/.609 |
| delta | **-2.3** | **+5.4** | -2.1/+10.2 | -2.7/-2.2 | +1.0/+10.5 | -6.4/+1.6 |

**Pooled recall is 2.3 points LOWER than stock, not higher**, and the tally is
4 of 8 cells rather than 6 of 8, with every win on precision and recall losing
in three of four tiers. This matches the audit's own row for this
configuration -- "Footprint-aware clustering x1: -2.3 recall, +5.4 precision,
precision 60->65, p=0.010" -- which has been right the whole time.

So the configuration is still worth having, but **its win is precision and
only precision.** The F1 figure of +0.044 at p=0.002 that 5ay and 5az both
led with is real on non-hard tiers, and it is also the wrong summary: F1
averaged away a recall loss, which is exactly what the standing rule against
quoting a single combined number exists to prevent. The best-recall
configuration remains `footprint_graph_strong` (5aw, +4.8 recall, p=0.001).

### What survives from 5az, and what does not

Still valid, because they are internal to rep0 and were cross-checked:

  * the definition of what recall and precision measure -- 13 clusters out of
    238, one best-matching cluster per injected unit;
  * the per-spike timing measurements and the per-unit alignment method, with
    its Kilosort null control at +0.000;
  * the LOCDEN retraction (chance recall 0.237, clusters 8.16x bloated);
  * `ms5_tuned` being worse than untuned MountainSort5.

Not valid as stated:

  * **every absolute number in 5az's tier tables.** They are rep0 and must be
    labelled rep0 wherever they appear.
  * **the four external sorters' figures carry the same inflation**, since all
    of them were run on rep0 only. tridesclous2's precision of 0.862-0.965 per
    tier -- the one live lead coming out of 5ay and 5az -- is a rep0 number and
    is NOT yet established. Running it on rep1-3 is the decisive test and is
    CPU-only, so it is not blocked by Trodes. **RESOLVED the same day in 5bb:
    it ran, and the precision lead SURVIVES pooling -- 0.851 vs stock's 0.600,
    +0.252 at p = 4.27e-05, higher in all four replicates.** lupin,
    mountainsort5 and LOCDEN remain rep0-only.

### The rule this breaks, written down so it is not broken again

The project already requires reporting recall and precision separately, per
tier, **with spread**. A single replicate's tier cell is 3-4 units and has no
spread to report, so **no tier-level number from one replicate should ever be
quoted as a result.** Single-replicate runs are for comparing sorters to each
other under identical conditions, which is what the external comparison
legitimately did; the moment a number is presented as this project's
performance, it must be the four-replicate pooled figure with its per-replicate
range.

## 5bb. tridesclous2 on all four replicates: the precision lead is real, p = 4e-05, and it names the headroom the audit has been pointing at

5ba ended by saying tridesclous2's per-tier precision was the one live lead
and was **not** established, because it had only ever run on rep0 and rep0 is
the easiest window. It has now run on rep1-3 as well (CPU-only, 241-243 s per
replicate, 56.8k-64.6k spikes, 172-197 units). The lead survives, and pooled
it is larger than rep0 suggested.

### Pooled over 4 replicates, 52 placements

| | recall | precision | easy | collision | pair | hard |
|---|---|---|---|---|---|---|
| stock KS4 | **0.794** | 0.600 | .883/.765 | .832/.594 | .720/.486 | .764/.593 |
| footprint_cluster | 0.771 | 0.654 | .862/.867 | .805/.572 | .730/.591 | .700/.609 |
| **tridesclous2** | 0.392 | **0.851** | .445/**.942** | .347/**.801** | .405/**.824** | .369/**.848** |

Paired Wilcoxon on the same 52 placements, the project's standard test:

  * **precision +0.252, p = 4.27e-05**, better in **34 of 52** placements;
  * recall **-0.401, p = 2.59e-09**, better in only 6 of 52.

It is higher on precision in **all four replicates** -- 0.900 vs 0.710,
0.710 vs 0.518, 0.908 vs 0.529, 0.887 vs 0.642 -- so no single window carries
it. Note that rep1, the hardest window, drags tridesclous2 down too (0.710),
which is the right behaviour: the replicates differ in genuine difficulty, not
in some artifact.

**The pair tier is the result.** The audit's own commentary on it is "on the
pair tier, half of what Kilosort calls one neuron is another neuron", and it
is the worst cell in the whole benchmark at 0.486. tridesclous2 scores
**0.824 there, +0.337, p = 8.98e-03** on 16 placements. Whatever makes
Kilosort merge neighbouring neurons is **not forced by this data.**

### The control, this time pooled rather than rep0 only

tridesclous2 is scored with 33 shifts of per-unit alignment freedom and stock
Kilosort at a fixed 0, so the comparison needs its null. 5az ran that control
on rep0 only; pooled over four replicates, giving per-unit freedom to a sorter
that already shares the benchmark's convention is worth:

| | recall gain | precision gain |
|---|---|---|
| stock KS4 | +0.0051 | +0.0020 |
| footprint_cluster | +0.0058 | +0.0027 |

**+0.002 of precision against the +0.252 being claimed**, two orders of
magnitude smaller. The comparison is not rigged by the alignment.

### The sharpest number, and another thing rep0 hid

Matched-cluster bloat -- how many times larger the matched cluster is than the
part of it that belongs to the injected unit -- pooled over four replicates:

| sorter | bloat |
|---|---|
| **stock KS4** | **1.71x** |
| footprint_cluster | 1.15x |
| tridesclous2 | 1.06x |

**Stock Kilosort's matched clusters are 71% larger than the unit they
represent, while tridesclous2's are 6% larger on the same recordings.** On
rep0 alone Kilosort measured 1.00x, so 5az reported Kilosort's clusters as
essentially pure -- another thing the easiest replicate hid, and a direct
quantification of the merging the audit describes.

This is what makes the splitting lead concrete rather than hopeful: the
contamination is **merging, it is large, it is measured on all four
replicates, and a different algorithm avoids it on identical data.** Splitting
operates on exactly that and does not have to touch recall.

### The honest limit on this result

tridesclous2 buys its purity by being conservative: 0.392 recall, finding
roughly half the spikes Kilosort does, better on recall in 6 of 52 placements.
**It is therefore not evidence that 0.79 recall and 0.85 precision are
simultaneously achievable** -- it is evidence that the purity is attainable at
all, and that the specific failure is merging rather than noise. The
asymmetry that matters for the lead: a merge can in principle be undone after
the fact by splitting, whereas an undetected spike cannot be recovered after
the fact -- which 5am and 5ax already established from both ends.

Artifacts: `outputs/allreps_per_unit_tridesclous2.csv`,
`outputs/allreps_per_unit_vanilla.csv`,
`outputs/allreps_per_unit_footprint_cluster.csv`; code
`score_tdc_all_replicates.py`, `test_tdc_vs_stock.py`.

## 5bc. Why rep0 is the easy replicate: not the window, not the amplitude, and nothing done differently — it drew cleaner donor neurons

Gil asked whether rep0 might have been *built* differently, and whether it
appears in the audit at all. Both answers are measurable and neither excuses
5az.

### Is rep0 in the audit? Yes, as one quarter of it

The audit states its benchmark as "52 placements x 4 replicates" and "13
placements per recording, 4 recordings from different windows of the session,
~290 spikes each". So rep0 is **inside every audit number**, contributing 13
of the 52 placements, and the audit never breaks it out -- the string "rep0"
does not occur in it. It is not a separate or earlier test. 5az simply quoted
one quarter of the audit's own data as though it were the whole thing.

### Nothing was done differently

  * All four replicate directories (`hybrid.bin`, `hybrid_truth.npz`,
    `manifest.csv`) were written **within the same minute**, 2026-10-01
    14:48-14:49: one generator run, one code version.
  * The **destination sites are literally reused** across replicates -- the
    same channels appear in several manifests with identical measured
    properties (e.g. channel 71 is 20.66 uV noise and 333.8 uV largest
    neighbour in all four).
  * Consequently the site statistics are matched by construction: site noise
    17.59-17.97 uV, largest neighbour 107.9-113.3 uV, residents 19.5-21.9,
    density 8481-9793.
  * Ground-truth spike counts are matched: 3755, 3787, 3765, 3779.
  * Windows are evenly spaced -- `bg_start` at 2000 s, 4000 s, 6000 s and
    8000 s into the session, 120 s each -- and **rep1, the worst, is the
    second one**, so this is not progressive drift.

### It is not the window, and this is the decisive test

18 donors appear in more than one replicate, which is exactly what the
rotation design was for. Comparing stock Kilosort on **the same neurons** in
different replicates:

| comparison | same neurons | recall there | recall here | delta |
|---|---|---|---|---|
| rep0 vs rep2 | 6 | 0.977 | 0.999 | **-0.022** |
| rep0 vs rep3 | 7 | 0.800 | 0.830 | **-0.029** |
| rep1 vs rep3 | 6 | 0.664 | 0.795 | -0.131 |

**On identical neurons rep0 is marginally WORSE, not better.** Its whole
apparent advantage vanishes once neuron identity is held fixed. The rep0
window is not easier to sort.

### It is not amplitude either

The obvious guess -- rep0 drew louder, more separable spikes -- is wrong, and
in the wrong direction:

| | STA trough (uV) | SNR | stock recall |
|---|---|---|---|
| rep0 | 476 | **26.99 (lowest)** | **0.882 (best)** |
| rep1 | 514 | 28.92 | 0.674 |
| rep2 | 723 | **41.93 (highest)** | 0.764 |
| rep3 | 526 | 29.89 | 0.855 |

rep0 has the **smallest** spikes of the four and still scores best, while rep2
has spikes 1.5x larger and scores worse. Across all 52 placements SNR does not
predict recall at all (Spearman rho -0.178, p = 0.21) nor precision
(rho -0.002, p = 0.99). This independently reproduces 5au's loudness finding
by a different route.

### What it is: donor cleanliness

| predictor | vs recall | vs precision |
|---|---|---|
| **donor contamination %** | **rho -0.555, p < 1e-4** | **rho -0.678, p < 1e-4** |
| SNR | -0.178, p 0.21 | -0.002, p 0.99 |
| STA trough | -0.202, p 0.15 | +0.028, p 0.84 |
| site noise | -0.136, p 0.34 | +0.039, p 0.78 |

Donor contamination -- the donor unit's **own** refractory violation measured
in the original recording, before any injection -- is the only strong
predictor of either metric, and rep0 drew the cleanest donors: **mean 1.45%
with 8 of 13 at exactly zero**, against rep1's 2.79% with only 4 at zero.

Regressing each metric on contamination alone (R2 = 0.415 recall, 0.444
precision) and comparing replicate means of the residuals:

| | rep0 | rep1 | rep2 | rep3 | spread |
|---|---|---|---|---|---|
| raw recall | 0.882 | 0.674 | 0.764 | 0.855 | 0.208 |
| contamination-adjusted | 0.845 | 0.718 | 0.754 | 0.858 | **0.139** |
| raw precision | 0.710 | 0.518 | 0.529 | 0.642 | 0.192 |
| contamination-adjusted | 0.657 | 0.581 | 0.515 | 0.646 | **0.142** |

Contamination explains **42% of the recall gap and 48% of the precision gap**
between rep0 and the other three. The rest is neuron identity with nothing
further to attribute it to: **rep0 and rep1 share only 1 donor of 13**, so
they are nearly independent draws of 13 neurons from a 33-neuron pool, and 13
is a small sample of a population the project has repeatedly measured as
dominated by unit identity rather than tier (5ab, 5ag).

### The caveat this raises about the benchmark itself

If a donor's own contamination predicts the score this strongly, then part of
what the benchmark charges to the sorter is **the ground truth being impure**:
a donor with 9.9% refractory contamination was already a slight mixture of two
neurons in the original recording, so its injected spikes are not one
neuron's. This cannot be the whole story -- contamination tops out near 10%,
which cannot produce a pair-tier precision of 0.486 -- but it is a real
modulator and it was not controlled for in the tier design, which stratified
on the *destination* site rather than on donor quality. A future benchmark
should either stratify on donor contamination or restrict to donors at 0%.

Artifacts: `outputs/donor_amplitude_by_rep.csv`; code
`diag_why_rep0_is_easy.py`, `diag_explain_rep_gap.py`.

## 5bd. Manual curation made part of the dataset: stability windows, dropped-packet gaps, and merges without rewriting the clustering  (2026-10-15)

Nothing physiological here. This section records a tooling chain built on 2026-10-15
that changes what the analysis dataset *contains*, so it belongs in the log: three
manual-curation facts that previously lived only in a human's head (or not at all) are
now columns in `synced/`.

### 5bd.1 Trial times are available before sync, which made the timeline panel possible

The review tool needed a per-unit firing-rate curve across the whole session with trial
markers on it, and the obvious route (read `trials.parquet`) would have made the review
tool depend on the sync pipeline having already run. It turns out not to be necessary.

**The DIO TTL code is the Bpod state number, so code 1 = each trial's first state =
trial start.** Verified against `synced_20261002/trials.parquet`: 954 code-1 markers
against 953 Bpod trials, and the ordered difference to `trial_start_trodes` is
**0.0 ms at both the median and the maximum**. Code 0 is the inter-trial reset and is
useless for this (median −2147 ms, max |13991| ms); codes 2 and 3 sit at +49.9 ms and
+549.9 ms.

Also pinned down, because it is a recurring source of error: **three clocks**.
`spike_times.npy` holds row indices into `probe1.dat`; `<session>.timestamps.dat` maps
rows to absolute Trodes samples (an 87.5134 s offset for 20260916_110311, and *not*
constant — about 15 s of drift over 3 h); DIO timestamps are already absolute. Every
range this work saves is in **absolute Trodes seconds**.

`scripts/build_session_timeline_data.py` memmaps `timestamps.dat` and emits base64
uint16 bin counts, trial markers, a DIO code census and recording gaps — 4–5 s for 472
units and 25.2 M spikes.

### 5bd.2 A negative result worth keeping: the amplitude trace is not a stability signal

The first version of the timeline panel carried a per-unit amplitude trace from
`amplitudes.npy`, on the theory that a unit drifting off the electrode would show a
falling amplitude. Gil doubted it; the data settled it. **Units 163 and 38 both collapse
from ~50 Hz to ~1 Hz while amplitude stays flat at 14–15.**

The reason is selection bias: `amplitudes.npy` is Kilosort's scaling factor for spikes
it *already assigned* to the cluster. As a unit fades, the spikes that still clear
threshold are the big ones, so the mean amplitude of what survives barely moves even as
the count collapses. **Amplitude cannot see a unit leaving; the rate can.** The trace
was removed.

### 5bd.3 A real 711 s dropout in 20261014_101945, verified four independent ways

Checking the gap machinery against yesterday's session turned up an actual dropout:
**711.4097 s (11 min 51 s), from 7682.2975 to 8393.7072 s, 76.4 % of the way through**,
sample 230468924 → 251811215. Gil doubted it, so it was verified four ways that do not
share a failure mode:

1. `probe1.dat` has 300,833,603 rows against a 322,175,918-sample span.
2. The `.rec` file's mtime is **13:19:01**, matching the counter-derived last sample at
   13:18:49 — and *not* the 13:06:58 a contiguous recording would have ended at.
3. DIO trial markers have a median interval of 11.43 s with exactly one of **716.20 s**,
   starting at t = 7677.7 s.
4. Raw data is real on both sides (sd 1029 and 1281) and both streams resume at the same
   instant.

Two smaller glitches (24 samples and 1 sample) sit below the 0.05 s threshold.
20260916_110311 and 20261013_091625 are perfectly contiguous.

This is now detected automatically: `units_spikes.find_recording_gaps` reads
`timestamps.dat`'s own counter, and `trials.parquet` gains `ephys_missing_s` and
`overlaps_recording_gap`. **Flagged, never removed** — the behaviour for those trials is
perfectly good, so dropping the rows would corrupt behaviour-only analyses, and
`spikes.parquet` has no rows for such a trial either way, which makes a gap trial
otherwise indistinguishable from a genuinely silent neuron. That indistinguishability is
the entire reason the flag exists.

### 5bd.4 Per-unit stability windows now reach the dataset

The review tool gained a timeline panel for marking which stretches of the session each
unit was actually present for. Those marks become
`<kilosort4_folder>/unit_trial_selection.json` (via
`scripts/write_unit_trial_selection.py`) and then `unit_trial_selection.parquet`, sparse
in units and dense in trials: a unit that appears has an explicit answer per trial, a
unit that does not keeps every trial. A trial counts as inside a kept window iff its
**start** falls inside it, applied once by the pipeline.

Applied to 20260916_110311: unit 163 keeps [4500, 12690.39] (9.04 vs 0.50 Hz, 580/953
trials), unit 38 keeps [700, 8900] (48.73 vs 4.00 Hz, 639/953). Rasters confirm the holes
are gone and responses sharpen — unit 163's cue peak 16 → 23 Hz; unit 38's cue 30 → 40
against 42 → 55 Hz, reward 33 → 44 against 52 → 67.

**The non-obvious part, and the thing that confused matters for a while: 314 excluded
trials cost unit 38 only 4 % of its spikes.** Those trials were nearly empty — they
contributed almost nothing to the numerator while sitting in the denominator of every
PSTH. The effect on a *rate* is far larger than the effect on a spike count, which is
exactly why the filter is worth having.

Relevant to 5bc: **units 163 and 38 are both audit donors** (of 33), and
`build_tiered_v2.py` draws its 120-snippet pool across the whole session, so a drifting
donor injects a blend of present and absent stretches into the hybrid ground truth.
That is a concrete mechanism for the donor-quality effect 5bc identified.

### 5bd.5 Merges recorded in a manifest, not written into the clustering

Replacing Phy's merge step raised the question of what a merge should *produce*. Phy
rewrites `spike_clusters.npy`: the parents stop existing and ids get renumbered, so
every unit id recorded anywhere else silently starts pointing at a different neuron.
The decision was to leave the clustering immutable and record the decision instead
(`sync_pipeline/unit_merges.py`, `scripts/write_unit_merges.py`, HOW_TO_RUN §6c).

Consequences, in order of how much they matter here:

- **Merging does not invalidate a stability selection.** The clustering fingerprint
  still matches because the clustering did not change, so the two curation passes are
  independent and can be done in either order.
- Parent ids stay valid forever; un-merging is deleting a line; two competing manifests
  can be compared by processing the session twice.
- Merged ids start at 10000, so a synthetic unit is recognisable on sight.
- **The cost:** Phy, bombcell and SpikeInterface read `spike_clusters.npy`, so after a
  merge they show the parents, unmerged. The merged view exists only in `synced/`.

Stability windows combine as the **union** of the parents', **frozen at merge time**,
with each parent's own ranges kept as provenance — a merged unit already signed off
cannot change because someone re-marked a parent afterwards.

**The merged unit's quality label is set by hand, and that is a second pass (added
2026-10-16).** The first version inherited the parents' label and fell back to `mua`
when they disagreed, which is the common case for a drift pair — so the rule that was
meant to be conservative would in practice have labelled most real merges `mua` and
quietly dropped them out of every `quality_label == "good"` analysis. The fix is not a
better rule: a merge cannot be judged before it exists, since its waveform, ACG and
rate are properties of the combination. So a manifest entry may carry
`"quality_label"`, which overrides everything, and `write_unit_merges.py --score
10000=good` sets it in place afterwards without disturbing the frozen `kept_ranges`.
`unit_merges.quality_label_source` records `manual` / `inherited` /
`conservative_mua`, which makes "nobody has looked at this merged unit yet" a query
rather than something to remember. Verified on the real session: scoring 38+163 `good`
puts unit 10000 in the good-unit set, where the inherited rule had it as `mua`.

Verified end to end with a throwaway 38+163 manifest (since removed — a smoke test, not
a real merge decision). Six invariants, all holding: the `.npy`'s sha1 is unchanged; 38
and 163 are still in the clustering and 10000 is not; the merged unit has exactly
795,573 spikes against the parents' 681,429 + 114,144; **every other unit's spike count
is identical to the un-merged run**; and the frozen union 700–12690.4 s keeps 894 of 953
trials against 639 and 580 separately — a union keeps *more* trials than either parent,
as it must.

### 5bd.7 The merge tool, and what it decides on (2026-10-16)

The manifest format settled first and the UI followed.
`scripts/build_merge_candidates.py` exports every pair within 60 µm — for
20260916_110311, 448 non-noise units of 472 and **2790 pairs**, ~2 min, 9.6 MB — and
the suggestions are shown **inside the review tool**, per unit, under the session
timeline (https://claude.ai/code/artifact/74dec064-c86a-46a3-8c03-f112ee4ecb05). A standalone queue over the same
data and the same `merge_verdicts` database also exists for bulk work
(https://claude.ai/artifact/DLDrEjL5Sn74TvBuMnmdcW); a decision in either appears in the other.

Putting the list in the unit view is the point, not a detail: a merge is judged while
you are looking at the unit, against the unit. But on a dense probe nearly every unit
has *some* cluster within 60 µm, so an unfiltered list marks everything and says
nothing. Only candidates with a readable refractory dip or a drift signature are shown
and counted (the `N⇄` chip in the unit list, and a filter); the rest stay one click
away. **For unit 233 that is 2 worth a look of 7 neighbours.**

The ordering principle, which is the substance of the tool: **the cross-correlogram
decides, everything else is circumstantial.** One neuron cannot fire inside its own
refractory period, so if two clusters are one neuron the CCG has a dip at zero lag.
Reported as a ratio of observed counts in ±1.5 ms over what the correlogram itself
predicts for a window that size, **always printed with the expected count**, so that a
ratio of 0.0 computed where 0.7 counts were expected is visibly not a measurement.
Thresholds used for colouring only (they decide nothing): ≤0.45 consistent with one
neuron, ≥0.8 two neurons, ≥1.5 a zero-lag *peak*, i.e. synchrony or a double-detected
spike rather than identity.

**Which baseline predicts that window is not a detail, and the first version got it
wrong — see 5bd.8.**

### 5bd.8 The dip must be measured against its own shoulders (2026-10-16)

Gil, on first reading the tool: *"the dip_ratio sounds very similar to kilosort's
problematic metric — can't you use the violation ratio comparing the shoulders vs the
dip?"* He was right, and it was not a stylistic point.

The first version normalised the refractory window by the **far flanks** (|lag| ≥ 10
ms). That is the same construction as the contamination metrics that misbehave: it
assumes the correlogram is flat out there. Any CCG with broad structure — bursting,
shared input, slow co-modulation — breaks that assumption, and the ratio then reports
a dip that does not exist. Measured instead against the **2–5 ms shoulders either side
of the refractory window**, and required to be unlikely under Poisson chance at the
count actually observed (p ≤ 0.01), on this session:

| | far-flank ratio | shoulders + Poisson |
|---|---|---|
| pairs classed "one neuron" | **30** | **13** (9 kept, 21 dropped, 4 newly found) |
| pairs "worth a look" | 328 | 311 |

59 of the 2790 pairs have a 2–5 ms shoulder disagreeing with the ≥10 ms flank by
≥1.5× — precisely the cases where the old baseline was invalid. The casualty is the
pair this log previously called the textbook merge:

- **233+236** — far-flank ratio 0.310, which read as a clean dip. Against its own
  shoulders it is **0.606**, with 10 observed against 16.5 expected, **p = 0.06**.
  Shoulders and flanks disagree 1.95×. Not evidence of one neuron. The footprint
  agrees: the two mean waveforms have visibly different shapes on all ten channels.
- **246+248** survives — ratio 0.241 against shoulders, 1560 observed against 6478
  expected, p < 0.0001, tilt 1.05×. Still carrying the caution that merging raises
  refractory violations to 0.879% from 0.445/0.368.

Both baselines are now shown side by side in the tool, with the disagreement called
out, because when they differ neither number stands on its own. Two further changes
from the same conversation:

- **the firing rate of each unit sits above the merged ACG.** With a lopsided pair the
  merged train is almost entirely the fast unit's spikes, so the merged violation
  *percentage* barely moves however wrong the merge is. Above 5:1 the panel says so
  outright, and the number it leads with instead is the **count** of refractory
  violations the interleaving creates, which rate asymmetry cannot dilute.
- **the waveform footprint across channels** — both units' mean waveform on every
  channel either covers, each trace at its real position on the probe, one µV scale
  throughout. The data always had this (10 channels per unit with their µm
  positions); only the single shared channel was being drawn. One neuron has one
  footprint, and 233+236 fails that test on sight.

Spliced in by `scripts/patch_review_tool_merge_v2.py` (12 exact anchors, applied on
top of `patch_review_tool_merge_panel.py`); review tool now at Version 5.

### 5bd.9 No verdict had ever been saveable, and what similarity is worth (2026-10-16)

Gil made one rejection "just to verify it works and everything is documented." It
was not in the database, and neither was anything else: `merge_verdicts` was empty.

The cause had been there since the panel was first spliced in. The verdict buttons
carried their payload as `data-mv=' + JSON.stringify(JSON.stringify(patch))`, which
writes backslash-escaped quotes into an HTML attribute. HTML has no backslash
escaping, so the attribute ended at the first inner quote:

```html
<button class="mbtn" data-pair="8_11" data-mv="{\" verdict\":\"merge\"}"="">MERGE</button>
```

`dataset.mv` came back as the two characters `{\`, `JSON.parse` threw, and the click
died there — silently, with no visible change to the button. Replaced with two plain
attributes (`data-mvf`, `data-mvv`) that need no quoting. Verified by clicking
SEPARATE in a served copy and reading `mergeVerdicts` back. **Any verdict entered
before Version 7 is lost and has to be redone.**

`trial_selections` was unaffected — it writes through a different path — which is why
`db` looked healthy and the fault stayed hidden.

**On Kilosort's template similarity.** Gil remembered it being useful at high values
and asked why it had been dismissed. Measured on this session, he is right and the
blanket dismissal was too strong. Of 2554 pairs with a readable correlogram,
**0.51%** show a one-neuron dip — that is the chance rate everything below is
measured against:

| template similarity | pairs | one-neuron dip | vs chance |
|---|---|---|---|
| 0.00–0.50 | 2051 | 0.2% | 0.5× |
| 0.50–0.80 | 418 | 1.4% | 2.8× |
| 0.80–0.90 | 57 | 1.8% | 3.4× |
| 0.90–0.95 | 15 | **6.7%** | **13.1×** |
| ≥0.95 | 17 | 0.0% | — |

So similarity carries real information. What the top band shows is subtler than
"useless": all 17 pairs at ≥0.95 have waveform r between 0.88 and 0.97 and a partial
dip (0.25–0.80), and **not one has enough coincident spikes to settle the question**.
Similarity finds the pairs; it cannot close them, and at the very top it selects
precisely for pairs the correlogram cannot adjudicate — where the footprint and the
session timelines are the only evidence.

Those pairs were being hidden behind *show the other N neighbours*. They now surface
as **twin templates**: similarity ≥ 0.95 and waveform r ≥ 0.85 with a thin or partial
CCG. 11 such pairs across 22 units; "worth a look" goes 311 → 321. The best of them
is **167+168** — similarity 0.968, waveform r 0.971, 0 µm apart, merged violations
0.015%, rate correlation −0.59 with **zero** simultaneous activity. A textbook drift
pair that the dip test alone would never have surfaced, and 233+236 returns here too,
honestly labelled a partial dip rather than proven.

Two usability changes at Gil's request: the merge panel now sits at the **bottom** of
the unit view (examine the unit, then decide what to merge into it), and each
candidate row has an **open ↗** button that jumps to that unit, with a back chip at
the top of the detail that walks the trail back.

Spliced by `scripts/patch_review_tool_merge_v3.py`; review tool at Version 7.

### 5bd.10 The review tool is reproducible again (2026-10-16)

Gil asked whether a future session could regenerate the review tool with all of this
in it. The honest answer at the time was **no**: `build_review_html.py` produced the
base page and knew nothing about merges, while the merge panel existed only as three
patch scripts applied to a *published artifact read*, which bakes in this session's
data and depends on remembering the order.

Fixed with `scripts/build_review_tool.py <session_id>`, which runs
`build_review_html.py` and then the three patches in sequence into
`outputs/unit_review_tool_<sid>.html`. Verified by running the chain against a clean
generator build: the result is identical to the published page apart from the
publisher's own `<!doctype>` skeleton and two reworded comments. The published
artifact is now republished **from that generated file** (Version 8), so the repo is
the source of truth rather than the artifact.

Two repairs on the way: `patch_review_tool_merge_panel.py` ignored its destination
argument and always wrote to a hard-coded scratch path, and
`patch_review_tool_merge_v2.py` had been corrupted by an in-place edit that turned
`\n` escapes into real newlines inside string literals (the Bash-heredoc mangling
trap again — write Python with the editor, not a heredoc).

### 5bd.11 Three supports, each for a case the CCG alone gets wrong

- the **merged ACG** — the combined train's own autocorrelogram, the nearest thing to
  seeing the merged unit before it exists, with the combined refractory-violation
  percentage against each parent's;
- **raw mean waveforms** from `probe1.dat` in microvolts on one shared scale, on the
  channels the two units share, plus the footprint down the probe (one depth, one
  neuron). Deliberately not normalised: a normalised overlay hides a 40 µV / 400 µV
  mismatch, which is a different neuron;
- the **session timelines**, for the drift pair — one unit fading out as the other
  fades in. Those two are almost never present together, so their CCG is empty *by
  construction* and the timelines are the only evidence there is. Flagged
  automatically as rate correlation ≤ −0.1 with simultaneous-activity overlap ≤ 0.35.

Kilosort's `similar_templates.npy` is displayed but never used to rank a decision: it
compares templates, so two different neurons with similar waveform shapes score high.
It only chooses which pairs are worth computing a CCG for.

First look at the real session: the top of the queue is 246+248 (dip 0.23, 124,869
coincident spikes, merged violations 0.879% against 0.445/0.368 apart — a dip, but
merging costs refractory cleanliness) and 233+236 (dip 0.31, waveform r 0.95,
similarity 0.989, merged violations 0.096%), which is the textbook shape of a real
merge. **Nothing has been adjudicated yet** — the verdicts are Gil's to make, in the
artifact's own `merge_verdicts` database.

### 5bd.6 What this means for anyone reading `synced/`

Three filters now exist and they are not interchangeable:

```python
trials = trials[trials.sync_valid]                 # clock trustworthy        (per trial)
trials = trials[~trials.overlaps_recording_gap]    # neural data exists       (per trial)
# then, as a JOIN, never as a trial filter:
drop   = sel.loc[~sel.selected, ["unit_id", "trial_id"]]   # unit was present (per pair)
```

Merges need no change to an analysis at all — a merged unit is just a unit.
`session_meta.json` always states what a given dataset was built with
(`unit_trial_selection.status`, `unit_merges.status`, `recording_gaps_checked`), so none
of it has to be remembered. `scripts/read_synced_with_selection.py <synced_folder>` is a
runnable worked example.

**And the step that is easy to miss: none of this reaches a dataset until the pipeline is
re-run for that session.** A `synced/` folder built before a curation file existed does
not contain it and never gains it retroactively. `synced_20261002` has no curation in it;
`synced_20261015` is the first output with both tables.

`sync_pipeline` also went under version control for the first time (it had none — the
pre-change state was recovered from `D:\Gil\files\sync_pipeline.zip`, which matched
byte-for-byte except the edited files). Test suite went 31 → 79 passing over this work.

---

## 6. WHERE THINGS STAND  (current as of 2026-10-15 — read this first)

Self-contained. Everything needed to resume with no memory of the conversation.

### 6.1 The one-paragraph version
This project built a post-Kilosort correction module (wavelet sub-sample
alignment, temporal whitening, spatial footprint), then a hybrid ground-truth
benchmark, then seven flag-gated modifications to Kilosort4 itself. **All of
that is finished and the verdict is negative**: five of seven modifications do
nothing or hurt, merging a unit's fragments trades down in all 16 cells tested,
and the post-hoc correction pipeline recovers spikes at 2.3% precision. They
all fail for one measured reason — 93.5% of what a unit's template matches on
its own channel is a NEIGHBOUR's real spike. The problem is per-spike
attribution. **5av then closed the attribution lead too**: argmax over
competing units works correctly and finds almost nothing, and the proposal
stage itself captures only 26.1% of missed spikes, so the proposal - not the
scorer - is the ceiling. **5aw** then found the one real win in the whole
project — the footprint in Kilosort's kNN graph — but the combined measure
settles it: `footprint_cluster` beats stock Kilosort on F1 (+0.044, p=0.002)
while the newer graph-only hook is net flat (−0.002, p=0.984), its +4.8 recall
cancelled by its precision cost. **5ax** then built a sorter that is not
Kilosort at all (whitened multichannel matched-filter detection, point-source
localization, HDBSCAN) and measured it on the same benchmark: a competitive,
CPU-only detection stage (0.978 detection recall vs 0.214 for a plain
threshold) but a decisive end-to-end loss (0.519/0.134 vs 0.882/0.710),
because it lacks iterative template subtraction. Its one lasting contribution
is that its 4×-more-sensitive detector hits 5av's 26.1% recovery ceiling at
exactly the same height, which closes the "better candidate proposal" escape
route. Work then
moved to **physiology**: the sync pipeline was run on session 20260916_110311,
and reward responses, cue-end side selectivity, a confidence readout, DV
decoding and leaving-time decoding were all measured, with the main
stimulus-period result replicated across three sessions.

### 6.2 IMMEDIATE STATE
Nothing is running. No background jobs, no copies, no Kilosort runs pending.
Everything described below is complete, committed and pushed on branch
`post-ks-correction`.

**Three things a new session should know about the harness.** (1)
`run_v2_comparison.score()` reads only `spike_times.npy` and
`spike_clusters.npy` from `hybrid_v2_rep{r}/ks_{name}/`, so **any** sorter —
not just Kilosort — can be dropped into the benchmark and scored by the
identical rule, with no change to the harness.

(1b) **WHAT RECALL AND PRECISION ACTUALLY MEAN HERE (5az).** Per injected
unit, the scorer keeps the ONE cluster holding the most of that unit's spikes
(greedy one-to-one, ±10 samples) and reports `recall = matched / true` and
`precision = matched / all spikes in that cluster`; the headline is the mean
over the 13 units. So **the metrics score 13 clusters out of ~238 — the other
~225 are never scored**, because the benchmark is Shamir's REAL recording with
13 synthetic units added and the rest are resident neurons with no ground
truth. A high cluster count is therefore not inflation. Two corollaries:
"precision" is the purity of one cluster, NOT a false-positive rate; and
recall credits only the largest fragment, with splits carried separately in
`n_fragments`. Critically, **recall does not penalise merging** — see the
LOCDEN retraction in 6.x and 5az, where 8.16x-bloated clusters produced the
highest recall of any sorter tested. (2) **THE TIME CONVENTION, and
5ax got this wrong.** 5ax said "truth times are trough samples and Kilosort
agrees to ±1 sample". The agreement is real; the conclusion was not. 5ay
measured the spike-triggered average at the truth times directly: the
**physical trough is a median of +13 samples AFTER the truth time** (range +8
to +17 over 13 rep0 units). Truth times are the `nt0min=20` index of the
61-sample extraction window, and Kilosort uses that same convention, which is
*why* they agree — it never implied either was at the trough. **Any sorter
reporting the real trough is ~13 samples outside the scorer's 10-sample
tolerance and will score near zero for pure bookkeeping.** This cost nothing
only because it was caught: lupin reads 0.101 recall unshifted and **0.689**
at its own alignment, a factor of 6.8.

**Use `score_per_unit_fair.py`, not `score_all_fair.py`.** 5az measured that
the offset is not one constant per sorter but **varies per unit by 10–13
samples** (locden_iso +5..+15, tridesclous2 +7..+17, mountainsort5 +4..+17),
because the trough position relative to `nt0min` is a property of each donor
waveform. One global integer shift cannot bring a 13-sample spread inside a
10-sample tolerance, so `score_all_fair.py` under-corrects. The per-unit
scorer is validated by its own null control: Kilosort shares the convention
and gains **+0.000**, footprint_cluster **+0.002**, so the selection bias of
maximising over 33 shifts is ~0.002 and real gains sit above it. In practice
only locden_iso moved materially (+0.030); tridesclous2 and mountainsort5
gained nothing, so 5ay's global-shift table was not as wrong as feared.

**LOCDEN runs on the CPU only**, so unlike every Kilosort run in this project
it is not blocked by Trodes holding the GPU (mistake 4).

### 6.3 THE SORTING VERDICT (benchmark work, sections 5ag–5am)

**The benchmark.** 33 hand-reviewed donor units, 52 placements, 15,086
ground-truth spikes, 4 replicates × 120 s from different windows of the
session, 9 configurations. Real recorded spike snippets (not template copies)
injected into real voltage at known times. Four tiers by the amplitude of the
largest resident neuron on the destination channel: **easy** 23–70 µV,
**collision** (quiet site, spikes placed 4–25 samples from a resident's spike),
**pair** (two remote donors, 181–242 µm apart, similarity 0.000, placed 3
channels apart on quiet sites), **hard** 223–466 µV.

**Stock Kilosort4, recall / precision:** easy .883/.765 · collision .832/.594 ·
pair .720/.486 · overall .794/.600.
**These pooled figures are the project's canonical numbers** — they are what
the Shamir Sorting Audit reports, and rep0 alone reads .882/.710, which 5az
wrongly quoted as a result (5ba). **Never quote one replicate.**
**Why the replicates differ (5bc):** not the window and not amplitude — on
the SAME donor neurons rep0 is marginally worse than rep2 and rep3, and rep0
has the LOWEST SNR of the four while scoring best. The driver is **donor
contamination**, the donor's own refractory violation before injection
(Spearman −0.555 on recall, −0.678 on precision, R² ≈ 0.42–0.44, explaining
42–48% of the rep0 gap). rep0 drew the cleanest donors, 8 of 13 at exactly
zero vs rep1's 4 of 13. **Caveat this raises:** part of what the benchmark
charges to the sorter is the ground truth being impure, and the tier design
stratified on the destination site, never on donor quality. A future benchmark
should stratify on contamination or restrict to 0% donors.
**The hard tier is NOT measurable** — two identical runs disagree on 2 of its 12
placements by up to 0.444 recall. Never quote a hard-tier number.

**Only the footprint configs beat stock.** Best recall is now
`footprint_graph_strong` (5aw, graph-only hook): **+4.8 recall points,
p=0.001** on the measurable tiers, better in all four replicates, overall
recall 0.8173 — the project's strongest sorting result. Best precision is
`footprint_cluster` (+6.6, p=0.009). Previous best recall was
`footprint_cluster_strong` (+3.2, p=0.015). Amplitude normalisation is
actively harmful (p=0.012). The graph/seed decomposition says the **kNN graph
does all the work and the kmeans++ seeding does nothing**, and the two are
sub-additive — consistent with the over-seeding hypothesis (5aa). Cost: the
graph-only config loses the combined patch's merge-error protection
(1 of 8 pairs merged, same as stock, vs 0 of 8).

**Merging fragments does not help (5al).** Units average 1.27–1.38 substantial
clusters, max 3, and 67–77% have exactly one — there is no three-way split to
merge. Combining trades down in **all 16** algorithm × tier cells: +3 to +7
recall for −4 to −22 precision. A perfect merge tool converges every algorithm
to ~77/46.

**Post-hoc correction does not help (5am).** Template convergence + AR(4)
whitening + full-session matched filter + R²/footprint gates, 52 placements:
11,027 accepted candidates, **259 were real** (2.3%; chance 0.1%). Pooled
+1.7 recall for −4.3 precision. Of 10,768 false accepts, **10,070 (93.5%)
coincide with a spike Kilosort filed under a different cluster**. The merge
detector scored 0.00–0.11 sensitivity because the input clusters already
violate refractory at 0.051–0.060.

**What it recovered, split (5am):** of vanilla's 259 correct recoveries, 82%
were spikes taken back from another unit and 18% genuinely new detections, from
pools of 2,480 misfiled / 642 undetected. That **79/21 split independently
reproduces the earlier 84%-misfiling finding** by a different route.

### 6.4 THE PHYSIOLOGY (sections 5an–5au)

**Pipeline.** `D:\Gil\sync_pipeline`, output in
`F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced_20261002`.
21,422/21,422 events matched, local-interpolation held-out error median
0.000 ms / max 43 ms, no gaps. The ephys-offset fix recovered **87.51 s**.
`sync_status: needs_review` is EXPECTED and fine — it describes the global
affine fit against real Bpod drift, not the local interpolation actually used.

**Trial taxonomy (478 completed, sync-valid, of 953):** 71 ERROR,
291 CORRECT+REWARDED, **116 CORRECT+OMISSION** — of which **67 catch trials**
(wait stretched to 20 s, the designed confidence probe) and 49 ordinary trials
abandoned early (median wait 0.83 s, a different population, must be excluded).

**Confidence replicates the classic result.** Catch 7.14 s vs error 6.25 s
(one-sided p=0.0029); on catch trials wait scales with difficulty
(rho=+0.330, p=0.0064) and on error trials it does not (rho=+0.027, p=0.83).

**Responses** (111 review-tool `good` units; note `units.parquet`'s own
`quality_label` is KSLabel and would give 189 — wrong curation):
reward delivery (`water_L`/`water_R` onset) 21/111 substantial, unit 35 goes
19→80 Hz; cue end 44/111 substantial and **45 side-selective vs only 15 at
reward** — side is carried at the choice point, not the outcome. All 21 reward
units re-tested at the give-up moment: 18 show nothing, 3 go the opposite way.

**DV decoding: the population carries the CHOICE, not the stimulus (5ao).**
DV R² peaks at 0.568, but knowing the rat's choice alone explains R²=0.533;
a sign(DV) decoder trained on correct trials scores **AUC 0.002 on error
trials** (inverted, i.e. it predicts the choice); random forest matches ridge.
Pre-stimulus is clean in this session across all methods.

**Leaving time: no countdown, but the departure is flagged (5ap).** Remaining
wait is NOT decodable at any lag 0–4 s (R² negative throughout). "Will the rat
leave within 500 ms" IS: AUC 0.923 (catch) / 0.901 (error) against a
permutation null of **0.77** (the null preserves elapsed time).

**Stimulus-period dynamics, replicated 3× (5aq, 5ar).** Choice becomes
decodable ~75–100 ms after stimulus onset and reaches AUC 0.88 by stimulus
offset — a quarter-second before the rat moves, so **the decision is complete
when the sound stops**. The accumulation proxy (stronger evidence → earlier,
higher choice signal) is **monotonic in all six comparisons across three
sessions**, and late-stimulus decoding is significant in all six half-sessions.

**The pre-stimulus leak is choice history (5ar).** The two validation sessions
decode current choice before the stimulus (AUC 0.60/0.54). Pre-stimulus
activity carries the PREVIOUS trial's choice at AUC ~0.70 in all three
sessions; the rat repeats on 55–64% of trials; conditioning on the previous
choice collapses current-choice decoding to chance (0.45–0.53) everywhere.
A real finding in its own right. Does not touch DV, which is at chance
pre-stimulus in every session and every half.

### 6.5 THE STIMULUS — fully solved (5as, 5at, 5au)

**x(t) = ω·A(t) + (1−ω)·B(t)**, a plain weighted sum of two fixed natural
sounds (`soundA=frogs2.wav`, `soundB=Passer_Montanus.wav`), with
ω = `Custom.AuditoryOmega` saved for every trial and **DV = 2ω − 1 exactly**.

- Held-out prediction error **2.2e-16** (machine epsilon). SVD of the four
  saved waveforms = `[1, 0.465, 0, 0]` → exactly two sources.
- **A and B were recovered** from the 4 waveforms in `AudSound` (indices
  953–956, the unplayed look-ahead trials). The rig's `.wav` files are NOT
  needed — every trial's exact waveform is reconstructible from its ω.
- **ω is constant within a trial** (per-frame SD = 0.0000). High ω → LEFT
  correct (100.0% vs 0.0%).
- **This is NOT a click train and NOT a tone cloud.** `AuditoryStimulusType=3`
  selects `Natural` from `['Clicks','Freqs','Natural']`; the `Aud_*` tone
  settings belong to the Freqs variant and are inert.

**Therefore an early-vs-late evidence comparison is impossible BY DESIGN**, not
for want of data: every trial delivers one fixed ratio for 350 ms, and two
trials sharing ω are the same sound sample-for-sample. Answering that question
needs a stimulus whose evidence varies within the trial.

**LOUDNESS CONFOUND, uncontrolled, state it whenever the accumulation result is
quoted.** No power normalisation, so the mixture is quietest near ω=0.5 —
exactly where trials are hardest. RMS 0.0697 at |DV|=0.09 vs 0.0869 at
|DV|=0.75: **~2 dB louder on easy trials**. Part of the accumulation ordering
could be loudness. Testable and NOT yet done: regress decoding onset on
loudness and |DV| jointly, since RMS is a known function of ω.

### 6.6 HOW RESULTS MUST BE REPORTED (Gil has said this twice)
1. **Recall AND precision together, never one alone.** They move in opposite
   directions here.
2. **Per tier, never only pooled.** Tier effects have opposite signs.
3. **Spread across replicates**, not just the mean.
4. **Merge errors separately** for the pair tier (correct answer is TWO
   clusters).
5. **Precision here measures merging with resident neurons, not a noise rate.**
   80.6% of what counts against precision is a resident's REAL spikes (chance
   3.9%). Never say "half of Kilosort's spikes are noise".
6. **Chance is measured, never assumed** — permutation null for every decoding
   number. The leaving-time null is 0.77, not 0.5.
7. **Plain language first**, then the table.

### 6.7 WHERE EVERYTHING LIVES
- Code: `D:\Gil\spike_sorting_agent\post_ks_correction\`
- Git: branch `post-ks-correction`,
  https://github.com/ZurGil/striatal-spike-sorting.git
- Sessions (all three reprocessed with the timing fix):
  `F:\Gil\Shamir\20260916_110311.rec\...\synced_20261002` (hand-curated),
  and in the session scratchpad `synced_20260901_085606` (offset 25.79 s),
  `synced_20260911_100049` (offset 28.8 s) — **both reprocessed because their
  on-disk `synced` folders predate the ephys-offset fix and are WRONG.**
- Benchmark: `D:\Gil\spike_sorting_agent\hybrid_v2_rep{0,1,2,3}\`
- Envs: `phy2_ky` (analysis), `kilosort4` (CUDA)
- **Manual verdicts**: `outputs/manual_verdicts_20260916_110311.csv`
  (111 good / 74 mua / 13 noise). The ONLY human judgement for this session —
  `cluster_group.tsv` is a byte-identical copy of `cluster_KSLabel.tsv`.
- **Trodes ↔ Kilosort: `ntrode = 1466 − ks_channel`** (verified, not assumed).
- Bpod `.pkl` caches on Z: were written under numpy 2.x and will NOT unpickle
  under this machine's numpy 1.26. Workaround: stage a scratch `rat_root` with
  a copy of the `.mat` and no `.pkl`.
- Report artifact: https://claude.ai/code/artifact/b9a94a61-e21b-4642-88de-dc24f082519b
  Standalone copies: `outputs/Shamir_Sorting_Audit.{html,pdf}` (self-contained,
  figures embedded; the PDF is the one to put in Slack).

### 6.8 KEY ANALYSIS SCRIPTS
`run_v2_comparison.py` (score) · `analyse_v2_results.py` (paired tests, noise
floor) · `analyse_split_vs_contamination.py` (merging) ·
`test_post_hoc_on_v2.py` (post-hoc correction vs truth) ·
`analyse_recovery_breakdown.py` (misfiled vs undetected) ·
`analyse_reward_and_cue_responses.py` · `decode_dv.py` ·
`decode_leaving_time.py` · `decode_stimulus_dynamics.py`
(takes `--sync/--units/--tag`, runs on any session) ·
`validate_stimulus_dynamics.py` · `stimulus_evidence_profile.py` ·
`recover_sources.py` / `verify_mixture.py` (the stimulus solution).

### 6.9 MISTAKES NOT TO REPEAT
1. **The base-rate trap — committed FOUR times.** Every "is X near Y" statistic
   needs its chance rate in the same table.
2. **Silent failures that produce plausible output — five so far.** The worst
   was a 20-sample sign error in `fftconvolve(mode="same")` indexing that
   recovered exactly zero spikes while the gates still passed at 86%. **Validate
   conventions against known truth; do not derive them.**
3. **Reading a subset and generalising — twice in two sections, both caught by
   Gil.** First claiming `AudSound` was "all empty" from three trials when four
   of 957 are full; then describing the stimulus as a tone cloud from the
   `Aud_*` values without checking which variant was selected. **Check which
   configuration is in force before reading parameter values.**
4. **Trodes must be CLOSED for Kilosort**, not merely idle — it holds the GPU at
   37% and Kilosort hangs silently rather than erroring.
5. **Don't trust an existing `synced` folder** — check it postdates the
   ephys-offset fix, or reprocess.
6. **`git add -A` from a subdirectory** once staged 2.6 GB and hung the push.

### 6.10 WHAT HAS NOT BEEN DONE
- **RETIRED 2026-10-09 (5av): footprint-first attribution.** It was the one
  open lead for months; it is now measured and dead. Argmax over every nearby
  unit on a shared channel set doubles recovery precision (3.18% → 6.36%) and
  costs 7× the recoveries, because a candidate's mean footprint similarity is
  **0.976 to its best competitor vs 0.876 to the target** — the pool genuinely
  is other neurons' spikes. 5q's AUC 0.999 was one well-separated pair; against
  the real impostor mixture footprint similarity scores **0.528**.
- **DONE 2026-10-09 (5aw): the clustering graph hook.** Built as
  `footprint_graph` / `footprint_graph_strong` / `footprint_seed` and run on
  all 4 replicates. It worked: +4.8 recall points at p=0.001. The note below
  is kept for the implementation detail.
- **The clustering GRAPH HOOK — was the only open lead on sorting.** Our
  footprint patch appends columns to `Xd` inside `cluster()`, so footprint
  reaches `kmeans_plusplus` seeding AND `neigh_mat`. Build the narrower
  version: identity features for the neighbour graph ONLY, PC features for
  seeding and everything downstream, then ablate the two apart. Aimed at the
  measured failure (84% misfiling, 25.7% of all injected spikes) and at the
  over-seeding hypothesis (5aa). Implementation: patch `neigh_mat` to return a
  consistent `(kn, M)`; do NOT pass `kn` to `cluster()` — `M` is then unbound
  (verified in 4.0.24). Needs the GPU, so Trodes must be CLOSED.
- **RETIRED 2026-10-12 (5ax): "a better candidate proposal will break the
  26.1% ceiling."** 5av left this open, because its ceiling was measured with
  one family of single-channel matched filters. A completely different
  detector — spatially whitened, 5-component learned temporal basis,
  11-channel pooling, 4× the detection sensitivity on injected spikes
  (0.978 vs 0.214) — captures **0.243** of Kilosort's non-hard missed spikes
  at 5av's 800-candidate pool size against 5av's **0.261**. Same wall, same
  height. Capture scales linearly with pool size (0.070 at 800, 0.843 at
  10,353), so the detection statistic has no ranking power either. Post-hoc
  recovery is closed from both ends.
- **RETRACTED 2026-10-12 (5az): "LOCDEN's recall matches or beats
  Kilosort's".** At per-unit alignment locden_iso scores recall 0.920, the
  highest of anything tested, hard-tier 0.837 against stock Kilosort's 0.576.
  It fails both its controls: **chance recall 0.237** (vs ~0.07 for every
  other sorter) and **matched clusters 8.16x larger** than the part belonging
  to the unit (vs 1.00x for both Kilosort configs). Recall does not penalise
  merging — 76 clusters where Kilosort emits 238 means each LOCDEN cluster is
  several neurons, and a cluster holding three neurons contains nearly all of
  each one's spikes. The recall figure added nothing its precision of 0.284
  had not already said. LOCDEN remains badly under-clustered.
- **MEASURED DEAD 2026-10-12 (5az): the `ms5_tuned` parameters.** Tuned
  MountainSort5 (detect_threshold 4.0, detect_time_radius_msec 0.2,
  scheme2_detect_channel_radius 75, chosen from our own occlusion
  measurement) scores **0.212/0.217 against untuned 0.323/0.691**, collapsing
  on collision to 0.066/0.004. Untuned defaults stand; the ALIAS entry is
  kept only so the result is not re-derived.
- **DEPRIORITISED 2026-10-12 (5ay): building our own deconvolution engine.**
  5ax called this the only remaining structural idea. `lupin` — a mature
  template-matching sorter, installed in `D:\conda_envs\si_ms5` — settles the
  premise for the cost of one background job: it scores **0.689 / 0.778**
  against Kilosort's 0.882 / 0.710, and on the **collision** tier, the one
  built from deliberately overlapping spikes and therefore the one
  subtraction exists to fix, it gets **0.596 against Kilosort's 0.961**. A
  mature implementation does not beat Kilosort here and is weakest exactly
  where it should be strongest. Not disproven (lupin is untuned, one
  replicate), but no longer the obvious build. The 5ax reasoning below is
  kept because the tension it measured is real.
- **NEW LEAD 2026-10-12 (5ay, strengthened by 5az), the strongest one: SPLIT
  Kilosort's mixed
  clusters with a unimodality test.** Untested, and it is the direction
  merging failed in rather than the same one (5al killed merging at the
  oracle level). **The headroom is now measured on all four replicates (5bb),
  not demonstrated on one:** stock Kilosort's matched clusters are **1.71x
  larger** than the unit they represent, pooled, where tridesclous2 manages
  **1.06x** on identical recordings; and on the pair tier, the benchmark's
  worst cell, tridesclous2 holds **0.824 precision against stock's 0.486**
  (+0.337, p = 9e-03). The contamination is merging, it is large, and another
  algorithm avoids it on the same data. The headroom is externally
  demonstrated: `tridesclous2`
  reaches **0.900** precision and `lupin` **0.778** against vanilla
  Kilosort's **0.710**, so the contamination Kilosort carries is removable in
  principle. `isocut6` from `isosplit6` gives a principled 1-D test, and it
  runs post-hoc on saved output with **no GPU**, so Trodes can stay open.
- **NEW LEAD (5ay): consensus across sorters.** Six-plus independent outputs
  now exist on identical data (Kilosort configs, lupin, tridesclous2,
  mountainsort5, LOCDEN). **Note after 5ba: every external sorter figure is
  rep0 only, and rep0 is the easiest replicate (stock Kilosort 0.882/0.710
  there vs 0.794/0.600 pooled), so tridesclous2's precision lead is not yet
  established -- it needs rep1-3, which is CPU-only work.** **Note after 5az:
  do NOT enter LOCDEN into a
  consensus as a recall source.** Its apparent 0.920 recall is a merging
  artifact (clusters 8.16x bloated, chance recall 0.237 against ~0.07 for
  everyone else) and its per-spike timing is the worst measured (MAD 3
  samples, only 39% of spikes within ±2 of their own unit's median), so
  matching it to another sorter's times is loose. tridesclous2 and lupin are
  the sound partners: both are 1.1x bloated with precise timing.
  Keeping only spikes two or more sorters agree on
  targets precision directly and is nearly free to test.
- **NEW LEAD (5ay): alternating align -> cluster -> realign.** 5ay measured
  that phase-only wavelet alignment without a per-unit template is at chance
  (corr -0.061 with the known injected shifts, RMSE 1.402 vs 0.292 for
  answering zero). The fix is an ordering: cluster on jitter-insensitive axes,
  align inside each cluster against its own template with a calibrated
  reference phase, recompute shape axes, re-cluster.
- **Iterative template subtraction — the tension 5ax measured, kept for
  reference.** A local-maximum footprint large enough to avoid
  duplicates (±10 samples × ±5 channels) pins detection recall at **0.476 no
  matter how low the threshold goes**, because a quiet spike beside a loud one
  is never a local maximum; shrinking it to ±3 × ±1 reaches 0.978 but admits
  **3.35 detections per real spike** (median 3, max 14). Only detect-subtract-
  redetect resolves both at once, and it is the one component LOCDEN lacks and
  Kilosort has. This is a substantial build, not a patch.
- **The loudness/|DV| disentangling** described in 6.5.
- 10 units Gil has not reviewed (24, 102, 126, 144, 205, 236, 280, 289, 359,
  387).
- The 5aw and 5ax results are not in the Shamir audit artifact / HTML / PDF.
- Temporal whitening inside Kilosort. Ceiling was already 4.8% (5ad); 5av then
  applied it to the post-hoc detection statistic exactly as the external
  proposal specifies (trace AND template), and found **no effect** - 2 better,
  1 worse, 4 tied across 7 placements. Deprioritised.

### 6.11 HONEST LIMITS
One animal. Three sessions for the stimulus-dynamics result, one for everything
else physiological. Benchmark patches are judged against Kilosort's own
thresholds and learned PC basis, both tuned on unmodified data, so the
measurement is "does switching this flag on help as Kilosort currently stands".
Absolute precision is low for the reason in 6.6 item 5; only RELATIVE
comparisons are meaningful. The cue-end result cannot be separated from
movement onset — they are the same instant in this task. The reward-specificity
result rests on a comparison that is not a matched omission.

### 5bd.12 20261016 archived to Z, and the real reason the 1015 size looked right (2026-10-20)

`F:` was at **57.8 GiB free of 1863** with two sessions on it, and
`20261016_103511` — the only session not in the Z archive — was still local only.
Copied it to `Z:\Gil\Shamir_1\20261016_103511.rec`: 11 files, 769,909,160,572 bytes
(717.0 GiB), 1:48:19 at 118 MB/s, robocopy reporting 0 failed and 0 mismatched.
(robocopy exit code **1** means "one or more files copied successfully" — success, not
failure.)

Verified three independent ways before declaring it safe:

1. **Paths and sizes** — the sorted `(size, relpath)` lists from F and Z are identical,
   11 files each, 769,909,160,572 bytes on both sides.
2. **Content** — 32 x 1 MiB blocks spread evenly across each of the three big files
   plus the final partial block, sha256 compared across F and Z; full sha256 on the
   other eight files. All match.
3. **Internal completeness of the archived `.rec`** — 255,101-byte XML header,
   404,816,827,880 bytes of payload, which is **472,916,855 whole 856-byte packets
   with zero trailing bytes**, and 472,916,855 x 384 x 2 = 363,200,144,640 =
   `probe1.dat` exactly (timestamps.dat is 4 x packets + its own 607-byte header).
   So the archived recording accounts for every sample the extraction contains.
   4.38 h of recording.

**Correction to 5bd.? / the 1015 post-mortem.** I had attributed the misleading
destination size on 2026-10-15 to robocopy's `/J` (unbuffered I/O) preallocating the
file. That is wrong, and it mattered because it implied the trap was avoidable by
dropping a flag. This copy was started **without** `/J` and the destination still
showed its full 404,817,082,981 bytes within seconds. robocopy preallocates
regardless. **A size match in Explorer is never evidence that a copy has finished,
under any flags.**

What does work: run robocopy **without** `/NP` so its log carries real percentages,
and read progress from the log (translating its carriage returns to newlines and
taking the last percentage). That gives an honest number. Measured F -> Z throughput
is ~113-118 MB/s, so a ~700 GiB session takes ~1.8 h.

**Consequence for freeing space.** Kilosort reads `probe1.dat`, not the `.rec`. With
both `.rec` files now verified on Z, the ~678 GiB they occupy on F is the space to
reclaim — it leaves everything sorting and syncing need in place locally, and the
`.rec` is only required to re-extract. Not done; Gil's call.

### 5bd.13 Gil's whole review is in the dataset, and it is checked rather than assumed (2026-10-20)

Gil finished reviewing 20260916_110311 and asked for every decision to be integrated,
with **tests proving each kind of verdict reached the dataset**. Done: `synced/` is
rebuilt and all nine checks pass.

What he had decided, read straight out of the artifact database:

| collection | docs | content |
|---|---|---|
| `verdicts` | 240 | good 114 / mua 113 / noise 13; **132 differ** from the automatic label (good->mua 100, mua->good 25, mua->noise 7) |
| `merge_verdicts` | 10 | 8 merge, 2 separate -> **6 groups** after transitive chaining |
| `trial_selections` | 20 | 17 real windows (units 0, 104, 111 have empty ranges = keep everything) |

240 verdicts against 472 units is not a half-finished review: **no unit the pipeline
called GOOD lacks a verdict.** The 232 without one are 214 automatic MUA and 18
automatic NOISE. He judged everything the pipeline proposed as good, which is the pass
that matters, and the rest keep their automatic label.

Result: `units.parquet` 433 rows = 472 - 31 noise - 14 merge parents + 6 merged units,
labels good 111 / mua 322.

#### The frozen union was wrong for mixed parents

Worth recording because it is silent, costly, and the tests could not see it.

Merged unit 10004 = 262 + 263 + 264 came out restricted to **27 of 953 trials**. Only
263 had a stability window (2699.7-3041.3 s); 262 and 264 had none.
`write_unit_merges.py` unioned only the parents that *had* windows, so the merged unit
inherited 263's. But a parent with no window was never restricted -- it is present for
the whole session -- and the pipeline's own rule says "only stretches where BOTH
parents were absent are dropped". An unrestricted parent is absent nowhere, so the
union is everything. 97% of that unit's trials were being discarded.

Every one of the nine checks passed on that output, because T7 compares the merged unit
against the frozen union in the manifest -- the same wrong value on both sides. The
tests verify that verdicts ARRIVE; they do not verify that the merge semantics are
right. Found by reading the output table, not by running the tests.

Fixed with `freeze_union()`, its own documented function: mixed restricted and
unrestricted parents now yield the full session window, and `unrestricted_parents` is
written into the manifest so the reason is visible rather than looking like a bug. It
cannot be expressed as an empty `kept_ranges`, because the pipeline (correctly) refuses
that combination as indistinguishable from a manifest that forgot its parents -- so
`write_unit_trial_selection.py` now carries `session_ephys_window` (87.5-12690.4 s,
from the review tool's own export) and the merge writer raises if it is absent rather
than quietly narrowing. 10004 now keeps 953/953.

10001 = 167 + 168 was right by accident: both parents had windows and they union to the
whole session anyway. That is itself the point of the merge -- 167 early, 168 late is
the drift-pair signature, so the merged unit being present throughout is the correct
consequence of judging them one cell.

#### Merged ids were renumbering themselves

Ids were allocated by position, so adding a merge of lower-numbered units pushed every
existing one up. It had already happened: 233+236 was 10001 on 2026-10-16 and became
10002 once four new merges sorted ahead of it. Nothing downstream had consumed the old
ids, but `--score 10001=good` would have landed on a different pair of cells. Ids are
now pinned to their **parent set** and carried across re-runs; a retired merge keeps its
id reserved rather than letting it be handed to other neurons. Verified by re-running:
all six reused, none reallocated.

#### scripts/verify_verdicts_in_dataset.py

Nine checks, deliberately sharing no code with the writers -- it recomputes the merge
grouping itself, so a bug in the writer's union-find shows up as a disagreement instead
of being reproduced on both sides. T1 fingerprints, T2 good/mua labels, T3 noise
exclusions, T4 unjudged units untouched, T5 merges, T6 separates respected, T7
stability windows (trial by trial), T8 every verdict accounted for, T9 referential
integrity. Wired into `sync_session.py`, which now raises and stamps `verified: false`
if it fails.

**Proved the checks bite** -- a test that only ever passes is worthless. Six negative
controls, each detected: a conflicting good/mua verdict (unit 11 mua->good), an
unapplied noise verdict, a missing merge, an unapplied stability window, verdicts
stamped with a wrong clustering sha1, and a separate-verdict pair merged anyway. (The
first attempt was a dud: "flipping" unit 51 to good when its verdict is already good
changed nothing and passed, correctly.)

#### Three things that blocked the run, all real

1. `sync_session.py` could never have run the pipeline: it invoked
   `python -m sync_pipeline.process_session` with cwd set to *this* repo, where
   `sync_pipeline` is not importable -- it is a sibling checkout, not an installed
   package. Now resolved explicitly, with `SYNC_PIPELINE_DIR` as an override.
2. No Bpod file found. The behaviour (`bpod/`, `rat_metadata.json`) lives with the
   archive on `Z:\Gil\Shamir_1` while the ephys works from `F:`, so `--rat_root` is
   required for this rat. Now a documented flag, recorded in the stamp.
3. `ModuleNotFoundError: numpy._core.numeric` -- the Bpod `.pkl` cache on Z was written
   on 2026-09-17 by a **numpy 2.x** interpreter, and no environment on this machine has
   numpy 2.x any more. Moved it to `.pkl.numpy2-stale` and let the loader rebuild it
   from the `.mat`, which is the source of truth.

Point 3 has a consequence beyond this run: **the September `synced/` output was produced
by an environment that no longer exists here**, so it was not reproducible. This rebuild
ran under `phy2_ky` (numpy 1.26.4, pandas 2.2.2, pyarrow 16.1.0), the only environment
with both pyarrow and a working scipy. Pinning the pipeline's environment is unfinished
work, independent of verdicts.

Gil's verdicts are also now committed at `outputs/review_verdicts/20260916_110311/`, so
his review no longer exists only inside the artifact database.

Still open: merged unit **10002** (233+236) is unscored and inherits `mua` -- a
placeholder, not a verdict. `--score 10002=good` once he has looked at it.

### 5bd.14 Today's export came within ~16 minutes of dying of a full disk (2026-10-21)

Gil asked whether Trodes had finished extracting the day's recording and for the session
to be copied to Z when it did. It had not finished, and checking turned up something
more urgent: **it was not going to finish.**

Session `20261021_111842`, recorded 11:18 to 16:06. The `.rec` is 442,655,652,202 bytes
(412 GiB), 517,120,791 packets, 4.79 h at 30 kHz. `trodesexport -dio -kilosort` started
at 16:06:38 and wrote at 2.75-2.95 GB/min, measured twice.

| 18:03 | bytes |
|---|---|
| `probe1.dat` so far | 340,221,935,616 |
| `probe1.dat` final | 397,148,767,488 |
| still to write, with `timestamps.dat` | ~57,400,000,000 |
| F free | **47,296,000,000** |

Short by about 10 GB -- a silent disk-full failure at roughly 97%, after two and a half
hours of export, somewhere around 18:21.

**The arithmetic worth remembering:** the export products are a fixed fraction of the
`.rec`, because a packet is 856 bytes of which 768 are the 384 int16 channels.
`probe1.dat` is 768/856 = 89.7% of the `.rec` and `timestamps.dat` is 4/856 = 0.5%, so
**an export needs ~91% of the `.rec`'s size free, and `trodesexport` does not check
before it starts.** Those ratios were confirmed against the finished September export
rather than assumed: its `probe1.dat` / 768 gives a whole number of packets, and that
same count reproduces its `.rec` length to within one packet and its `timestamps.dat`
length exactly, given a 609-byte stamp header.

#### What freed the space, and the thing that was nearly wrong

Gil's answer was that the September session folder could go if it was fully on Z. It was
not. A file-by-file comparison found **4 files on F that Z lacked and 9 whose sizes
differed** -- every one of them the verdict integration from 2026-10-20/21: the rewritten
`cluster_info.tsv` with its `.preverdicts` backup, `unit_merges.json` and
`unit_merges.parquet`, `unit_trial_selection.*`, and the regenerated `synced/` parquets.
Z still held the September versions. Those 13 files went up first and were verified by
full sha256, so the archive's `synced/` is now the one that matches his review.

What was then deleted was **only** `F:\Gil\Shamir\20260916_110311.rec\20260916_110311.rec`
-- 301 GiB, verified as an identical size plus 33 x 1 MiB sampled blocks hashing equal on
both sides before anything was removed. That freed 345 GiB, ten times what the export
needed, and left September's `probe1.dat`, DIO and `kilosort4`/`synced` outputs on F, so
the session is still sortable and analysable locally without pulling 290 GiB back off Z.
Deleting the whole folder, which Gil had authorised, would have cost that for no gain.

A size match was deliberately not treated as sufficient for the `.rec`: that is the exact
mistake of 2026-10-15 (section 5bd.12), and this file is irreplaceable.

#### `scripts/archive_session_when_exported.py`

New, and the standing answer to "copy it when it's done". It waits for the export, then
**checks that the export is complete from the data itself** before trusting it:
`probe1.dat` must be a whole number of 384-channel samples, `timestamps.dat` must agree
on that packet count, and both must agree with the count implied by the `.rec`'s own
length. A disk-full export fails the first test and a truncated one fails the third, so
a partial export can never be copied into the archive and recorded as a session. Only
then does it copy and verify -- identical path and size lists, then sampled blocks hashed
on both sides for the three big files. It deletes nothing; freeing F stays a human
decision.

It also waits on the process being gone **and** the files having stopped growing for two
polls, because a dead `trodesexport` is not the same as a finished one.

#### The check cried wolf once, on a perfectly good export

Its first real run refused to copy, reporting `timestamps.dat is not a whole number of
4-byte stamps past its 609-byte header`. The export was flawless; the check was wrong.
Both Trodes headers are **variable-length text**, and both had been hard-coded here
from the September files: the `timestamps.dat` header is 609 bytes in September and
**612** in October, and the `.rec` header 255,102 and **255,106**. `header_len()` now
finds the end marker (`<End settings>`, `</Configuration>`) and measures from it, which
reproduces both sessions' real values.

The cost was two minutes, because the script refuses rather than copying something
doubtful — the failure mode worth having. And with the headers measured, the three
packet counts agree *exactly* instead of to within a packet, which is a stronger
statement than the version with the bug could ever make.

#### The export, and the archive

The export finished at 18:23, 2 h 16 m after it started, landing on exactly the
`probe1.dat` size predicted at 17:56 — 397,148,767,488 bytes — with all three counts at
**517,120,791 packets**, 4.79 h at 30 kHz. Without the space freed it would have died
around 18:21.

The copy ran 18:28 to 20:27, 1 h 58 m for 784.1 GiB at 118 MB/s, robocopy exit 1 with
11 of 11 files copied and none failed. Verified four ways:

1. identical path and size lists, 11 files, 841,873,149,455 bytes;
2. 33 x 1 MiB sampled blocks per big file hashing equal;
3. **51 more 1 MiB blocks at different offsets, re-read with `FILE_FLAG_NO_BUFFERING`**
   — because step 2 ran seconds after the write and could have been served from the
   local SMB cache rather than from the share, which would have made it a check of
   nothing;
4. full sha256 on the eight small files, and the archived `.rec` read back from Z as
   internally complete: 255,106-byte header, 517,120,791 whole packets, **zero**
   trailing bytes, and 517,120,791 x 384 x 2 = the archived `probe1.dat` exactly.

Mid-copy, Z showed `20261021_111842.rec` at its full 442.66 GB *and* carrying its
original 16:06 timestamp while a third of the session had not crossed the wire. Explorer
would have called it done. That is the October 15 trap in miniature, and the reason real
progress was read off robocopy's own `ReadTransferCount` instead.

The session is still on F. Nothing was deleted.

### 5bd.15 The 2026-10-23 session, archived unattended -- and a watcher that lied (2026-10-23)

Gil asked for a standing task: copy today's session from F to Z once Trodes finished
extracting, without him having to come back and ask. `20261023_083635` is a **4.56 h**
recording, the longest so far.

**How it was set up.** `scripts/archive_session_when_exported.py 20261023_083635
--pid 21568`, started at 13:31 with `Start-Process` so it was **not** a child of the
agent session and would survive the terminal closing. Log in
`outputs/archive_20261023_083635.log`.

Space was checked first, because of what happened on the 21st: the `.rec` was
421,166,963,754 bytes (392.3 GiB), so the export needed ~354 GiB and F had 629.65 GiB
free. No risk this time. Stating it beats assuming it.

**Result**, entirely unattended:

| | |
|---|---|
| export finished | 15:19:34, `probe1.dat` = 377,869,196,544 bytes |
| packets | probe1 = timestamps = rec = **492,017,183**, all three exactly equal |
| headers | rec 255,106 / timestamps 612 -- measured, not assumed |
| copy | 745.99 GiB in 112.8 min, 118.4 MB/s, robocopy exit 1 (success) |
| archived at | `Z:\Gil\Shamir_1\20261023_083635.rec` |

**A watcher that reported success without ever waiting.** The job meant to notify on
completion was written as

```bash
while tasklist /FI "PID eq 8556" /NH | grep -q 8556; do sleep 300; done
```

Git Bash rewrites slash-prefixed arguments as paths, so `/FI` arrived as
`C:/Program Files/Git/FI`, `tasklist` errored, `grep` matched nothing, the loop body ran
**zero times**, and the job exited 0 within seconds announcing that the archiver had
finished -- about a copy that had not started. A liveness check that can only return
false turns a waiter into an instant success report. The replacement asserts it can see
the pid *before* entering the loop and aborts loudly otherwise. `MSYS_NO_PATHCONV=1`,
PowerShell, or `//FI` all avoid the conversion. The archiver itself was never affected:
it calls `tasklist` from Python under Windows, not through Bash.

**The script's own "verified" was overstating its evidence.** Step 4 reads Z *seconds*
after robocopy stops writing, and the local SMB client can answer those reads from its
own cache -- so on its own it confirms what was **sent**, not what **landed**. On the
21st that gap was closed by hand. Doing it by hand twice is a sign it belongs in the
code, so it is now `scripts/verify_archive.py <session_id>`, called as step 5 and also
runnable standalone later without redoing a two-hour copy:

1. full sha256 on every file that is not one of the three big ones -- 8 files today,
   all identical;
2. sampled 1 MiB blocks on the three big files opened with `FILE_FLAG_NO_BUFFERING`
   (via `CreateFileW` through ctypes, then `msvcrt.open_osfhandle`), at **different**
   offsets from step 4, so it covers blocks nothing has looked at yet -- 54 blocks
   today, every one identical;
3. the archived `.rec` judged on its own terms, read from Z: header measured at 255,106,
   **492,017,183 whole packets, zero trailing bytes**, and 492,017,183 x 384 x 2 =
   377,869,196,544 = the archived `probe1.dat` exactly. This is the only check that does
   not take F's copy as the reference.

If the cache cannot be bypassed it exits **6** and says the big files are not
independently confirmed, rather than quietly downgrading to a cached read and still
printing "verified".

The session is still on F. Nothing was deleted -- that stays a human decision.

### 5bd.16 The channel map changed today, and the date it changed was measurable (2026-10-26)

Gil changed the Trodes channel map for `20261026_113546` and wanted it recorded, but had
forgotten when the previous map was last used. That does not have to be remembered:
every `.rec` carries its whole configuration in its text header, including the 384
`<SpikeChannel>` entries that **are** the channel map -- each one's `hwChan` and its
`coord_ml` / `coord_ap` / `coord_dv` position in micrometres. So the map every session
ever recorded used is recoverable from the session itself.

`scripts/channel_map_history.py` fingerprints that map per session and groups sessions by
fingerprint. The fingerprint deliberately covers only the ntrode ids, `hwChan` and the
three coordinates -- not thresholds, filters, colours or display state, so changing a
threshold does not read as changing the map.

**The answer: there is only one previous map, and it is all of them.**

| map | sessions | `coord_dv` |
|---|---|---|
| A | **28**, `20260901_085606` .. `20261023_083635` | 920..4740 um, 192 steps |
| B | 1, `20261026_113546` | 100..3920 um, 192 steps |

Fingerprint `8e66a88e1f06` for every session from September 1 through October 23, and
`cdfe5ac929cf` for today. Nothing to hunt for: the "previous channel map" is the one used
for the entire dataset to date.

**What actually differs.** Two things, and only one of them matters:

- **A uniform 820 um depth re-reference.** Row for row, every `coord_dv` is 820 um
  smaller in B. Same 192 pads at the same 20 um pitch, same `coord_ap` exactly, same
  `coord_ml` values -- the origin of the depth axis moved. A spans 920..4740, B spans
  100..3920 (3840 um = 192 x 20 um, the probe's full span, in both).
- **The staggered ML columns are swapped.** Row 0 was `ml=2426`, now `ml=2442`; the set
  `{2426, 2442, 2458, 2474}` is unchanged, the assignment to rows is offset. 16 um.

Checked two ways that agree. Indexed by **hardware channel**, `coord_ml` and `coord_ap`
are identical for all 384 and `coord_dv` offsets are only `{0, 3840}` -- 82 channels wrap
by the probe's full span, 302 do not, because the ntrode list is **rotated by 82
positions** (ids A 1083..1466, B 1001..1384). Indexed by **exported row**, which is what
`probe1.dat` follows, the `coord_dv` offset is a single value: 820 um.

Confirmed independently in the file Kilosort actually reads. The exported
`channelmap_probe1.dat` is a 636-byte text header plus 384 x `<ml int32><dv int32><ap
int32>`; parsed, it gives row-by-row `dv` differences of exactly `{820}`, `ap` identical,
`ml` differing by the stagger swap. Its sha256 differs between the two sessions, as it
must.

**Why this matters.** `depth_mm = implant_depth_mm - phy_depth/1000`, and `phy_depth`
comes from those pad coordinates. An 820 um shift in the coordinates is a **0.82 mm shift
in every unit's reported depth**, silently -- exactly the class of error already flagged
for the four arithmetic `rat_metadata` fields. Units from map B are not depth-comparable
with the 28 sessions on map A without correcting for it. Which origin is the *correct*
one is Gil's call; the record states the offset rather than guessing.

Written to `Z:\Gil\Shamir_1\channel_map_history.json` and `.md`, beside
`rat_metadata.json`, and regenerable at any time with `--write`. The record is derived,
not hand-maintained, so it cannot drift out of date.

**Bonus, noticed in that header:** `Headstage_serialnum: 01803 00185`.
`rat_metadata.json` carries `headstage_serial_number: null` for all three rats with the
note "pending". It is recoverable from any exported session and worth filling in.

### 5bd.17 Two more sessions archived, and the drive that is now the constraint

`20261027_090722` and `20261028_093206` are on Z and verified. Both were asked for after
the export had already finished, which is the invocation worth remembering: run
`archive_session_when_exported.py <sid>` with **no `--pid`** and it waits on size
stability alone (2 polls, ~2 min) instead of watching a process that no longer exists.
Passing a dead pid is now a refusal rather than an instant false "complete" -- see
5bd.15.

| session | files | size | copy | packets | trailing | duration | map |
|---|---|---|---|---|---|---|---|
| `20261027_090722` | 11 | 685.63 GiB | 103.6 min | 452,205,765 | 0 | 4.19 h | B |
| `20261028_093206` | 11 | 777.91 GiB | 117.5 min | 513,070,777 | 0 | 4.75 h | B |

Both at 118 MB/s, both with `probe1 = timestamps = rec` packet counts agreeing exactly,
both verified twice with step 5 (the cache-bypassed pass) running automatically: 8 small
files by full sha256, 54 x 1 MiB blocks read from the device, and the archived `.rec`
judged complete on its own terms from Z. For 1028: header 255,039, and
513,070,777 x 384 x 2 = 394,038,356,736 = the archived `probe1.dat` exactly. 4.75 h is the
longest session recorded so far. Nothing was deleted from F by the archiver -- it never
deletes.

`20261028` is also on **channel map B**, so the new map now covers three consecutive
sessions (1026, 1027, 1028) and the switch described in 5bd.16 looks deliberate rather
than a one-off. `channel_map_history.json/.md` were regenerated to say so: 2 maps across
31 sessions, A = 28, B = 3. None of the three has been sorted, so the 820 um depth-origin
question is still a clean decision.

**The real finding is the space.** F is down to **298.3 GiB free** with both of those
sessions still on it. Today's `.rec` alone is 409 GiB, and an export needs ~91% of the
`.rec` size, which `trodesexport` does not check before starting -- the failure mode is a
silent disk-full 97% of the way through a 2 h export, which nearly happened on 10-21.
So the next recording cannot be exported until something is cleared. Both 1027 and 1028
are verified on Z and either one is enough; the advice stands that deleting just the
`.rec` beats deleting the session folder, because `probe1.dat`, the DIO files and the
timestamp offsets are regenerable from the archived `.rec` and the `.rec` is not
regenerable from anything. Deletion stays a human decision.

### 5bd.18 Reading the Bpod sync word out of a live recording, and catching Din1 die

Gil was debugging the Bpod-to-MCU connection and wanted to know whether the sync signal
being recorded was intact -- "the numbers run as they should, and not some weird numbers."
The obstacle is that `sync_pipeline/dio_decoder.py` reads `trodesexport`-ed
`Din{1..6}.dat` files, which do not exist until the recording is closed and exported,
which is no use while you are standing at the rig.

The word is already in the `.rec`. A Trodes packet is a `0x55` sentinel, then each
available `<Device>`'s bytes in `packetOrderPreference` order, then a uint32 timestamp,
then 384 int16 -- so for this configuration (`Controller_DIO` 1 B, `Multiplexed` 8 B,
`neuropixels_LFP` 66 B, `SysClock` 8 B) the DIO byte is **byte 1 of 856** and the
timestamp is at byte 84. All of it measured from the header, with two self-checks before
the byte is believed: the sentinel on every packet, and a timestamp that increments by
exactly 1. `scripts/check_bpod_dio_in_rec.py` does this, read-only, while Trodes writes.

**Weak evidence, and it did not convince.** The first pass reported that all six lines
toggled, that `Din7` was silent so nothing had shifted by a pin, that the values were a
subset of a known-good session's, and that landmark durations matched. Gil's answer was
"i'm not sure you convinced me" -- correctly, because none of that names anything.

**The Bpod file settles it.** `SessionData.RawData.OriginalStateNamesByNumber` says what
each state *number* is called and `OriginalStateTimestamps` how long it really lasted, so
every value on the wire can be named and bounded. 32 distinct values, every one a real
state of `Dual2AFC_nat`, every duration inside the protocol's own range: code 1
`PreITI_start` at 49.90 ms against 50.00, code 2 `preITI` at 500.00, code 3 `wait_Cin`
out to 20000.30 ms against a 20000.00 ms timeout, code 40 `ITI` at 1000.00. A swapped
pair of pins cannot survive that test, because the durations would land on the wrong
numbers.

Then `scripts/compare_dio_word_to_bpod.py` compares *sequences*: Bpod's
`OriginalStateData` is the list of state numbers it ran per trial, so the codes between
one `PreITI_start` and the following `ITI` must equal that list exactly. The recording
started long after the Bpod session, so the trial offset is found by matching rather than
assumed -- it located the window's first trial as Bpod trial **354 of 377** unaided.
Trials 354..363 match **state-for-state**, up to 38 states in a trial, and every state
onset sits **0.1-0.3 ms** from Bpod's own. That is the evidence the first pass could not
give.

**Two things I had wrong, both corrected.** The sub-millisecond codes are not line skew:
this protocol has eleven genuine pass-through states lasting **0.10 ms**, which is 3
samples at 30 kHz, and that is exactly what they measure on the wire in both sessions.
And the first and last state in a read window are truncated by the window, so their
lengths were being compared as if real and flagged as impossible.

**What it found.** `Din1` is stuck high. It flickered from about Bpod trial 364 and
latched HIGH for good at **3453.9193 s** into the recording (11:57:58.919), after which
it shows zero transitions over 120 s while `Din2..Din6` keep working. Every value is then
OR'd with bit 0 -- 2 reads as 3, 6 as 7, 16 as 17, 20 as 21, 28 as 29, 40 as **41**,
which is not a state at all. The four non-matching trials in the window differ from Bpod
*only* in bit 0, which is the whole diagnosis in one line.

The consequence is not recoverable in software: with bit 0 jammed high, `PreITI_start`
(1) cannot be told from no-state (0), so the trial-start markers `sync.py` aligns on are
simply absent. Bpod trials 1..~363 of 377 are usable.

**Worth keeping:** an unconnected input on this controller reads **HIGH** -- `Din8`
demonstrates it all session, nothing wired to it, high with zero transitions. So a line
that has come loose is indistinguishable from one stuck high, and a stuck-high reading
points at the connection before the Bpod. Also, the Bpod clock drifts ~2300-2900 ppm
against Trodes (2300 ppm on 20260916, 2911 ppm today) -- normal, and what `sync.py` fits;
within a trial the clocks agree to a few tenths of a millisecond.

Nothing in `sync_pipeline` was modified. Both scripts are read-only and import
`dio_decoder` unchanged, only to decode the comparison session.

**Correction to 5bd.18, measured later the same day.** That entry says an unconnected
input on this controller reads HIGH, citing `Din8`. `Din7` sits constantly **LOW** with
nothing wired to it either, measured over 300 s of `20261029_123356` -- so there is no
single idle level to infer from and the claim is withdrawn. It does not change the `Din1`
diagnosis, which rests on 0 transitions while the other five lines worked, every value
odd, and 41 appearing. The part of that reasoning that does hold is the **chatter before
the latch**: a logic output that fails tends to fail at once, a contact losing grip
flickers first.

Two further measurements from the post-repair recording, both now in
`docs/bpod_mcu_sync_wiring.md` for whoever works on the connector next. There is **no
measurable skew between the six lines**: over 238 word changes, transitions of 1, 2, 3,
4, 5 and 6 bits all land inside a single 33.3 us sample, and no intermediate value ever
persisted for 1-2 samples, so the word needs no strobe and can be read at any sample.
And `trodesexport` writes **only** `Din1..Din6` -- a signal moved onto `Din7` or `Din8`
would never be exported and the sync pipeline would never see it, which matters if the
wiring is about to be rearranged.

The repair was verified against Bpod: trials 519-524 of `Session2` matched
state-for-state, including a 30-state and a 20-state trial, every onset within
0.13-0.30 ms. Gil discarded both of the day's recordings afterwards.

### 5bd.19 Laying the base for sorting on compute1/compute2 (2026-11-04)

Gil wants Kilosort moved onto the lab compute nodes. The nodes have not been
requested yet -- he is about to ask Sarah -- so this entry is the local half: what was
set up here, and what the measurements say the plan should be.

#### Why it is worth doing, in the order the reasons actually matter

The obvious argument is the GPU, and it is the weakest of the three. This box has an
**RTX 2060 with 6.4 GB** (compute capability 7.5), which is the low end for
384-channel KS4 but has in fact been sorting sessions all along.

The real argument is the interlock. **This PC cannot record and sort at the same
time** -- Trodes and Kilosort both want the GPU and the failure mode is a silent hang,
not an error. So every sort currently costs a recording slot. Sorting on a node
removes that entirely, which is worth more than the speedup.

#### What was set up, and what was deliberately left alone

A dedicated ed25519 key, **no passphrase**, at `C:\Users\Adam\.ssh\id_ed25519_compute`
(fingerprint `SHA256:x6hiYyUqDtfn14nQy8cJdqiO6X9ukPRKIDti0BdrFno`), with its ACL cut
down to `Adam` alone via `icacls /inheritance:r`. There was no `.ssh` directory on this
machine at all before today.

`~/.ssh/config` now carries `compute1` and `compute2` blocks with `HostName` and `User`
as obvious placeholders, verified to parse with `ssh -G`. They are left resolvable-to-
nothing on purpose: a plausible-but-wrong hostname would be worse than an error that
names itself. The blocks set `ServerAliveInterval 30` / `ServerAliveCountMax 10`,
because a sort is hours of silence and an idle timeout would take the job with it, and
`Compression no`, because the payload is already-dense binary. Worth recording that
**Windows OpenSSH does not support `ControlMaster` multiplexing**, so every command
opens its own session and the keepalives carry more weight here than on Linux.

The `ssh-agent` service was left **Disabled**. With no passphrase it is not needed, and
not altering a system service is the better default. The passphrase decision is flagged
for Gil rather than made for him -- the key file alone is currently enough to log in.

#### The measurement that decides the whole plan

Kilosort does not need the `.rec`. It needs the `trodesexport` products, and **those
are already on the archive share**. Confirmed by listing
`Z:\Gil\Shamir_1\20261021_111842.rec\20261021_111842.kilosort\`:

| file | size |
|---|---|
| `20261021_111842.probe1.dat` | 369.87 GiB |
| `20261021_111842.timestamps.dat` | 1.93 GiB |
| `20261021_111842.channelmap_probe1.dat` | <1 MiB |

The share is `\172.20.22.203\NeuroPixels`, 42.3 TB free, 32 archived sessions, and the
share host sits on the **same /24 as this PC** (`172.20.22.115/24`). So the question
"can the nodes mount that share?" is not a detail -- it is the difference between never
copying anything and pushing 270-370 GiB per session at the measured 113-118 MiB/s,
which is **7-9 hours per session** with this PC occupied throughout. At that cost,
remote sorting would be worse than local for all but the biggest jobs. It is now
question 5 of the list for Sarah, and the one to press on.

F currently holds exactly one `probe1.dat` (`20260916_110311`, 270.4 GiB), which is the
other half of the same point: the local copies get deleted, so the share is where the
sortable data lives.

#### The environment to reproduce, read out of the env that works

From the `kilosort4` conda env on this machine: Python 3.9.19, **kilosort 4.0.24**,
torch 2.4.0 (CUDA build 11.8), numpy 1.26.4, scipy 1.13.1. The base conda python is
3.8.5 and has no kilosort or torch at all, so `kilosort4` is the only env that matters
here.

`kilosort` is pinned at 4.0.24 and should stay pinned: every sorted session in this
project came from it, and moving it would make new output incomparable to the existing
results without a deliberate re-run. The torch CUDA build is **not** in that category
and may have to move up for a current-gen card.

Two project-specific traps written into the doc so the remote setup does not walk into
them: the KS4 thresholds are **per session** (`Th_universal` 9.0 / `Th_learned` 8.0 on
20260916, but 9.2 / 7.0 on 20260910 and 20260901), and there are **two channel maps**
in play differing by a silent 820 um depth shift (map A through 2026-10-23, map B from
2026-10-26). Either one carried across wrongly produces output that looks fine and is
wrong.

#### Deliverables

- **`docs/remote_kilosort_setup.md`** -- the plan plus a forwardable request list,
  split into "needed to connect at all" (4 items) and "needed before the first real
  run" (8 items), so Gil can send the short list now. Carries the public key to
  install, the `ssh-copy-id` equivalent for Windows, the no-root conda recipe, and a
  "what is NOT known" section stating plainly that nothing about the nodes has been
  verified.
- **`scripts/survey_compute_node.sh`** -- read-only, piped in over ssh so it leaves
  nothing behind (`ssh compute1 'bash -s' < scripts/survey_compute_node.sh`). It
  measures GPU, driver, scheduler, containers, scratch, quota, and whether the node can
  reach `172.20.22.203:445`, which answers questions 5-12 without a second round trip
  to Sarah. Syntax-checked with `bash -n`; not yet run against anything, since there is
  nothing to run it against.

The ordered plan ends on the step that is easy to skip and shouldn't be: sort a session
that has **already** been sorted here and compare the output, before moving the queue
across. The point of the exercise is results that belong in the same analysis as the
existing ones, and a remote setup that quietly produces different output is not a faster
pipeline, it is a second pipeline.

### 5bd.20 The remote plan, corrected: WashU RIS, Docker, and Kilosort on CPU (2026-11-04)

Gil corrected three assumptions in 5bd.19, and each one changed the design. The
compute is **WashU RIS Compute1**, an LSF + Docker cluster -- not generic SSH nodes.
It **cannot reach the lab share**, so the "nothing is ever copied" plan is dead and
this PC becomes the thing that copies. And GPU availability is **uncertain**: the plan
is CPU, with throughput from running several sessions at once. Free to use while
compute1 lasts, which he was told is until July. The stated goal is to leave the
Neuropixels PC free for Trodes.

Also worth recording: RIS's own docs show the compute1 pages migrating to a **Compute2**
documentation space, so "compute1/compute2" are platform generations, not two machines.
That retrospectively explains Unmukt's "like doker or something" -- RIS runs every job
in a container, so an image is mandatory, not a convenience.

#### Kilosort4 on CPU, measured instead of guessed

The whole plan turns on one unknown, so it got measured rather than estimated. The
project already had a GPU result for the 120 s x 384 ch fixture in `hybrid_small/`, from
2026-09-30 with the same thresholds: **131.69 s total** on the RTX 2060, 1.10x realtime.
Re-running the identical fixture on CPU with 8 threads via a new
`scripts/benchmark_kilosort_cpu.py`:

The spike-detection stage ran at a steady **~31 s per batch over 60 batches, ~30 min**,
against **65.11 s** for that same stage on the GPU. That is a **28x** ratio on the stage
that dominated the GPU run, which puts the CPU total near **30x realtime**.

Scaled up: a 4.5 h session is **5-6 days** of CPU wall clock on 8 cores, against ~5 h
on this PC's GPU. The honest framing is therefore throughput, not speed -- about
**1 session/day** from ~5 concurrent CPU jobs, against 4-5/day on the local GPU that
blocks recording completely. For clearing the 32-session backlog that is a good trade:
a month of unattended free sorting that never touches the rig. For keeping up with daily
recording it is roughly break-even.

Which makes **"is any GPU queue available?"** the highest-leverage question in the whole
exercise -- 30x dominates every other optimisation -- and it has replaced the
share-mount question from 5bd.19 at the top of the list for Sarah.

Trodes was open (PID 11344, since 10-30) when this was run. Checked first that no `.rec`
was being written and that Trodes had used only **7.6 s of CPU in five days**, i.e. open
and idle, before loading 8 cores. A `device='cpu'` run never initialises CUDA, so the
documented Trodes/Kilosort GPU interlock does not apply -- but the GPU half of the
comparison came from the September **log** rather than a re-run, precisely because of it.

#### The storage arithmetic, and a question hiding inside it

Measured a real KS4 output folder for the first time: **19.18 GiB, 733 files**, of which
`pc_features.npy` is 5.64 GiB. Against a 252-370 GiB input (mean ~325), so ~345 GiB in
flight per session and **2 TB holds about 5 sessions**.

But that may be the wrong number to plan against. RIS splits `storage1` (persistent,
quota'd) from `scratch1` (large, purged). The *outputs* are only 19 GiB, so 2 TB of
storage1 would hold **100 sessions of results**; it is the inputs that are enormous and
they belong on scratch. If the 2 TB is storage1 and scratch is separate, concurrency is
set by scratch and could be far above 5. Added as an explicit question rather than
assuming the pessimistic reading.

#### Two things that would have bitten us

**The probe map cannot be generated on the node.** `make_probe_map.py` -- which lives at
`D:\Gil\make_probe_map.py`, outside the repo and untracked anywhere -- imports the Trodes
python reader from a local Trodes install. No Trodes on a compute node, so the map must
be built on Windows and travel with the data. Gil flagged this himself ("dont forget the
code that generate the map"). Rather than vendor SpikeGadgets' reader into a pushed
Docker image, `scripts/stage_session_to_ris.ps1` generates the map here, sanity-checks
it, and refuses to copy if that step failed. The `channelmap_probe1.dat` travels too so
the map can be audited later without coming back to this PC.

**Sequencing.** If the review pipeline is to run on RIS, it has to run *while the input
is still staged* -- waveform extraction needs `probe1.dat`, not just the KS4 output.
Purge the input and decide later you want waveforms, and it is another 325 GiB copy. So
the unit of work is stage -> sort -> build review data -> copy back -> purge, not
"sort everything then review everything."

#### Built

`docker/Dockerfile.kilosort4` (CPU, pins **asserted at build time** so a job cannot
discover a wrong kilosort six hours in; no ENTRYPOINT and `/bin/sh` present as LSF
requires; GPU variant documented), `scripts/run_kilosort4_session.py` (the existing
runner with paths and thresholds as arguments -- it **refuses** to fall back to CPU when
`--device cuda` was asked for, since a silent fallback turns a 5 h job into a 6 day one
unnoticed, refuses a probe whose channel count disagrees, and writes
`run_provenance.json`), `scripts/ris_submit_kilosort.sh` (bsub template with
`LSF_DOCKER_VOLUMES`, `span[hosts=1]`, a `kinit` check), `scripts/stage_session_to_ris.ps1`,
and the benchmark. All four syntax-checked; none run against RIS, because there is no RIS
access yet.

The **Kerberos** detail is in the submit script because it is the kind of thing that
only shows up at hour 30: an expiring ticket breaks a running job's access to RIS
storage, and a multi-day sort is well past the default lifetime.

The thresholds being per session and the two channel maps differing by a silent 820 um
depth shift are both now enforced in code rather than documented and hoped for -- the
thresholds as required arguments with no default, the map by printing the y-range that
is the only thing distinguishing map A from map B by eye.

### 5bd.21 RIS: free until July 2027, GPUs confirmed, and the resume point (2026-11-04)

Closing this out pending access, which Sarah has already requested for Gil. Everything
below is the state to resume from.

#### The cost question, answered from RIS's own page

Gil asked what GPU time would cost, and fairly challenged whether my "not big usage"
claim was measured against a neuroscience lab or a CS one -- it was implicitly against
machine-learning workloads, which was the wrong frame to leave unstated. It is moot:
RIS's compute page says Compute1 "will remain operational at no cost until that date",
the date being its end-of-life **1 July 2027**. There is no per-hour charge for CPU or
GPU, so availability is the only currency. That also confirms what Gil was told about it
being free "until july".

Two deadlines came with it that nobody had flagged. **1 January 2027**: Compute1 stops
accepting new General Access users -- about eight weeks out, so Sarah's request needs to
land before it. **1 July 2027**: all workloads off. The long-term home is **Compute2**,
which is billed in Standard Billing Units against a ~$2,800/yr faculty subsidy with 5 TB
of free storage, so the steady one-session-a-day load will eventually cost something and
the 30-session backlog should run on Compute1 while it does not.

#### GPUs are there, and the walltime worry was unfounded

RIS publishes an inventory of **242 V100/A100/H100/A40 GPUs, 100+ in the general pool**.
Against that, our ask is tiny: **3.3 GB of peak VRAM** (read out of the September run
log) and ~4.5 GPU-h per session. Since the platform is free, asking for GPU is a
question about queue time, not budget -- so the recommendation flipped to "ask for it",
having been "don't bother" one turn earlier when the only known workload was a
no-rush backlog.

And the **`general` queue allows 28-day jobs**. I had been treating a 5-6 day CPU sort
as something the queue might refuse; it fits with room to spare. The 24 h cap I had seen
belongs to `general-interactive`, which only ever applies to interactive sessions.

#### The storage split, now that the 2 TB is understood

It is Gil's practical slice of the lab's shared **5 TB free faculty allocation**, not a
separate grant, and results go back to Z. Since KS4 output is only **19.18 GiB** per
session -- measured today from a real output folder, 733 files with `pc_features.npy` at
5.64 GiB -- all 30 sessions of results are **570 GB** and fit the 2 TB easily. It is
purely the ~325 GiB inputs that need somewhere else, which is why "can the inputs live
on `scratch1`?" is the one question that decides concurrency. Never the `.rec`: 412 GiB
that Kilosort does not read.

#### What can be answered without asking anyone

`scripts/survey_ris.sh` replaced the generic node survey, because the right questions are
LSF's (`bqueues`, `lshosts -gpu`, `bhosts -gpu`, `busers`, `df`, `klist`) rather than the
client host's hardware, which is irrelevant -- RIS login nodes exist to submit, not to
compute. It covers **eight of the ten** questions in
`docs/ris_questions_for_unmukt.md`. The detail worth remembering: RIS writes its own
`RIS_cleanup_policy` and `RIS_Usage_Report.txt` *into* the scratch directory, so the
purge window can be read rather than asked about.

Human-only: whether the allocation lands before the 1 Jan 2027 cutoff, and Unmukt's
experience of what breaks in practice. Written up in plain language with a glossary
(node, core, queue, walltime, container, storage1 vs scratch1, Kerberos, Globus) because
Gil said he was not familiar enough with the terms to know what to ask.

#### Where the CPU benchmark got to

Left running at the point of handover: the 8-thread run on the 120 s fixture was on its
**second** detection pass (KS4 does two), 43/60 at a steady 31 s/batch. The conclusion
does not depend on its completion -- the 28x stage ratio that gives ~30x realtime and
~5.2 days/session is already measured -- and with GPUs confirmed on Compute1 the CPU
figure is now the fallback case rather than the plan. `benchmark_kilosort_cpu.py` should
be re-run **on the node** regardless; the number that matters is the node's.

Trodes was open throughout (PID 11344, since 10-30). Verified no `.rec` was being written
and that it had consumed **7.6 s of CPU in five days** before loading 8 cores, and took
the GPU half of the comparison from the September log rather than re-running it, since a
GPU run with Trodes open hangs silently.

#### 5bd.21a The CPU benchmark finished: 34.54x, and two cautions (2026-11-04)

The 8-thread run on the 120 s x 384 ch fixture completed at **4144.5 s = 34.54x
realtime**, against the GPU's 131.69 s (1.10x) on the same fixture and thresholds. So
**31.5x slower overall** -- my 28x estimate from the spike-detection stage alone was
close but slightly optimistic. Summary kept at
`outputs/kilosort_cpu_benchmark_8threads.json`.

Per mean 4.13 h session that is **5.95 days** on 8 cores (4.3 days for the shortest
session here, 6.9 for the longest). Re-running the throughput arithmetic with the
measured number moves one conclusion: at 5 concurrent the capacity is **0.84
sessions/day**, so 5 concurrent now **falls behind** a 1/day arrival rate where my
earlier estimate had it break even. Keeping up needs **5.9 jobs running continuously**,
so 8 concurrent (1.35/day) is the first configuration with real margin. The backlog of
30 takes 35.7 days at 5 concurrent, 22.3 at 8, 17.8 at 10.

**Caution 1: CPU and GPU do not agree exactly.** CPU found **206 units / 218,586
spikes**; the GPU found **210 units / 228,725 spikes** -- about 4.4% fewer spikes on
CPU. Expected, given different floating-point paths and KS4's stochastic steps, but it
matters for the planned validation: "re-sort a session already sorted here and compare"
must **allow a few percent of difference**. A small discrepancy is not evidence of a
broken remote setup; only a large divergence would be. Worth having measured this before
the comparison rather than during it.

**Caution 2: do not size the LSF memory request from this benchmark.** Peak RSS was
**7.89 GB**, but for *120 seconds* of recording. KS4 holds per-spike and per-template
arrays that grow with the **number of detected spikes**, and this fixture produced 218k
spikes where a 4 h session produces tens of millions. What we know empirically is that
full sessions sort on this box, which has 64 GB. So the submit script asks for **64 GB**
and now carries a comment saying the benchmark figure is a floor rather than an
estimate -- otherwise someone reads 7.89 GB, requests 16 GB, and loses a five-day job to
an out-of-memory kill at hour 100.

### 5bd.22 The rewired Bpod sync, checked against 953 known-good trials (2026-11-05)

Gil rewired the Bpod-to-MCU sync and recorded `test_synch` -- a few trials, nothing
else -- to find out whether the state word is arriving intact before committing the
wiring. The question was narrow: do the numbers coming off the six lines look like real
trials?

They do. Verdict first: **the connection works**, including Din1, which had died
mid-session on 2026-10-29 and is the reason any of this was in doubt.

`F:\Gil\Shamir\test_synch.rec` is 99.1 s, 2,972,923 packets, 856 bytes each, 384
channels at 30 kHz. Five complete trials plus a partial one at each end.

**The standard check refused to run, correctly.** `scripts/check_bpod_dio_in_rec.py`
stops if the packet timestamp does not increment by exactly 1, because a wrong packet
offset would otherwise produce a confident wrong answer about the wiring. It found 1,643
breaks and bailed. That self-check earned its keep, but here it was a false alarm about
layout: the 0x55 sentinel is clean on **every one of the 2.97M packets**, and the breaks
are all positive, all small (2-12 samples), and all confined to one burst between
packet 325,526 and 327,727 -- **t = 10.85 s to about 11.3 s**. Total loss is 12,574
samples, **419 ms**, in one half-second hiccup; the other 98.6 s is contiguous. That is
a Trodes acquisition dropout, not a parsing error and not a sync problem.

So I decoded it in a separate scratch script rather than loosening the check script's
guard, and took **time from the packet timestamp rather than the packet index**, so the
dropout cannot fabricate a long-held state. Nothing in `sync_pipeline/` was touched; its
`dio_decoder` was imported read-only for the comparison below.

**The trial structure** (code 40 = ITI, which ends a trial):

```
trial 1  40 0 1 2 3 5 6 8 10 11 | 14 15 25 26 27 38 37
trial 2  40 0 1 2 3 5 6 8 10 11 | 14 15 25 26 27 38 37
trial 3  40 0 1 2 3 5 6 8 10 11 | 14 15 19 20 29 31
trial 4  40 0 1 2 3 5 6 8 10 11 | 12 13 16 17 28 30
trial 5  40 0 1 2 3 5 6 8 10 11 | 14 15 19 20 29 31
```

The first ten states are identical across all five trials and the divergence begins
exactly at the point a choice should branch, producing three distinct outcome paths.

**The check that actually settles it** is not internal consistency but agreement with a
session recorded when the wiring was known good. Decoding `20260916_110311` (953 trials)
gives a reference set of state values and per-value durations. Against it:

- **No novel values.** Every number `test_synch` put on the wire appears in those 953
  trials. Nothing illegal, nothing above 40.
- **The hardware-timed states match to the sample**: code 1 at 49.90 ms on all six
  occurrences (known good 49.87-49.90, n=954), code 2 at 500.00-500.03 (identical range,
  n=954), code 8 at 350.00-350.03, code 28 at 84.60 (known good 84.60-84.63, n=138),
  code 40 at 999.97-1000.03 (n=953). Even the 0.10 ms pass-through states (10, 12-15,
  37, 38) land in their known ranges.

That combination is what rules out a **swapped pin** specifically, which was the real
worry: a swap remaps bits, so legal-looking numbers still come out, but the 50 ms state
would arrive on some *other* number. Instead 49.90 ms landed on code 1 six times out of
six and 1000 ms on code 40 six out of six.

**Two deviations, both benign:**

- **Code 29 was 73.70 ms, both times.** The known-good session has it at 87.30-87.33 ms
  across 153 occurrences -- tight enough that 73.70 is clearly a different number, not
  noise. Its sibling code 28 matched exactly. One state's fixed duration changing while
  the other 25 stay sample-exact cannot come from wiring; it is a protocol-side change,
  most likely a **valve recalibration or a changed reward amount**. Flagged for Gil to
  confirm he meant it.
- **Code 26 ran 10.3 s and 11.0 s** against a known-good max of 5.4 s. It is an
  animal-paced waiting state and there was no animal in the box, so a long wait is the
  expected result of a bench test, not a fault.

Din7 and Din8 remain at zero transitions, so **Din7 is still free for the camera strobe**
as designed in `docs/flir_camera_sync_design.md`. This recording does not test the
Din7-export hypothesis recorded in `docs/bpod_mcu_sync_wiring.md`, since Din7 had no
transitions here either -- that still needs a recording with a real pulse on it.

### 5bd.23 A reader that names the pin the camera strobe landed on (2026-11-05)

The camera PC reached the point of wiring the GPIO to Trodes, which runs straight into
the one thing nobody knows: **which physical pin of the 20-pin controller connector is
Din7.** The pin-walk procedure in `docs/bpod_mcu_sync_wiring.md` would measure it, but
there is a cheaper route now that there is a camera emitting a known signal -- wire a
best guess, record a minute, and let the recording say where the pulses went.

`scripts/find_camera_strobe_in_rec.py` does that. It reads all eight DIO bits out of the
`.rec` and asks which one looks like a free-running frame strobe, then names it. The
discriminator is deliberately two-sided: **a tight period AND a tight pulse width**. A
period alone is not enough, because a behavioural state line can be regular for stretches;
requiring the width to be near-constant as well is what separates a strobe from a state
word. Period is checked against `--fps` and width against `--exposure-ms`, since on
`ExposureActive` the pulse width *is* the exposure time.

It reports three things the wiring test actually needs:

- **which `Din`**, and a loud warning if the answer is Din1-Din6, because that means the
  camera is sitting on a Bpod pin and that recording's behavioural decode is corrupted --
  the only outcome of a wrong guess that is worse than nothing happening at all;
- **polarity**, by checking whether the measured width is the exposure or the frame period
  *minus* the exposure. The latter is an inverted line, which is a one-line fix and
  otherwise looks like a plausible signal;
- **pulse count against the camera's own frame count** (`--frames`), where 0 or +1 is
  expected (the last exposure can fire as the recording script stops, before that frame
  reaches the container) and anything else is a finding -- more pulses than frames means
  frames exposed but not written, fewer means pulses were missed.

Unlike `check_bpod_dio_in_rec.py` this one does **not** abort on a timestamp
discontinuity. That guard is right for the Bpod check, but a Trodes dropout is a real and
survivable event -- §5bd.22 measured one at 419 ms -- and refusing to look is worse than
looking carefully. So time comes from the packet **timestamp** rather than the packet
index, and any interval spanning a gap is **held out** of the period and width statistics
instead of silently widening them. The sentinel check is kept, because that is the one
that actually detects a wrong byte offset.

**Smoke-tested against a recording with no strobe in it**, which is the test that matters
for a detector: run on `test_synch`, it correctly reported no strobe rather than inventing
one, correctly classified all six Bpod lines as state-word bits (their periods are tens of
thousands of samples against the 500 a 60 fps strobe would give), showed Din7/Din8 idle,
and held out the 419 ms dropout.

**Exposure, corrected.** I warned the camera side that an exposure close to the frame
period could merge consecutive pulses and make my pulse count read low -- which would
mimic dropped frames, the exact failure this scheme exists to detect. That hazard is real
but it belongs to **auto-exposure**, which was already disabled there. At a fixed 15 ms
the low gap is 1.667 ms = **50 samples**, ample. So I walked the recommendation back:
going 15 -> 10 ms costs a third of the light and about 3.5 dB more gain, and gain is
sensor noise that DeepLabCut pays for. Keep 10 ms only if gain stays low; otherwise 15 ms
with lower gain is the better trade, and extra IR illumination beats both.

Nothing in `sync_pipeline/` is imported or modified by any of this.

### 5bd.24 Camera settings locked, and the strobe geometry written down (2026-11-05)

The camera PC settled its parameters and they are now recorded in
`docs/flir_camera_sync_design.md` §9, so verifying a recording is a lookup rather than a
re-derivation each time.

**Final: 1280×1024 Mono8, 60 fps nominal / 59.9944 measured, exposure 15.0 ms fixed, gain
12.9 dB fixed, both Auto off, GPIO Line1 = `ExposureActive`.**

The exposure question resolved against my first recommendation, correctly. Measured cost
of 10 ms instead of 15 was **gain 12.9 → 16.4 dB** — which is the 3.52 dB that losing a
third of the light predicts, so the measurement and the arithmetic agree. Since the
merging hazard belonged to auto-exposure (already off) and a fixed 15 ms still leaves a
**50-sample** low gap at 30 kHz, there was no reason to pay 3.5 dB of sensor noise for
margin that was not needed. The video feeds DeepLabCut, which pays for noise. The rule
worth keeping: **IR illumination first, exposure second, gain last.**

Worth noting the sustained hour-long test had itself been run at 15 ms, so the shipped
configuration is the validated one and the exposure decision needed no re-test. The
camera-side frame audit also came back clean on the stronger version of the check — all
215,971 CSV rows parsed, **zero FrameID gaps**, inter-frame interval 16.6617–16.6748 ms
with no outliers, timestamp span 3599.838 s. I re-derived it: 215,970 intervals at
16.66819 ms matches their 16.668232 ms mean, which puts the true rate at 59.9944 fps, and
at that rate a full 3600.000 s would yield 215,980 frames — so the 9-frame shortfall is
0.150 s of startup latency, exactly what they described. The naive "29 frames missing" was
my own assumption of exactly 60.000 fps. Nothing lost.

**Expected strobe geometry at 30 kHz** (now in §9): period **500 samples**, width **450**
(= 15 ms), low gap **50**, `frac high` **≈0.90**, **~3600 pulses in 60 s**. Three
signatures to recognise: a width of ~50 instead of 450 means the line is **inverted**
(count is unaffected — one rising edge per frame either way, so it is a polarity fix, not
lost data); a width matching neither suggests the signal is `ExposureEnd` or
`FrameTriggerWait` rather than `ExposureActive`; a pulse count *below* frame count means
pulses were missed rather than frames.

**The Din7 pin is no longer a blocker**, and §8 has been amended to say so. The pin-walk
procedure in `docs/bpod_mcu_sync_wiring.md` is no longer the only route now that there is
a camera emitting a known 60 fps signal: wire a guess, record 60 s, and
`find_camera_strobe_in_rec.py` names the line it arrived on. The Trodes-first invariant
does not apply to that test — it only matters for the "first pulse = video frame 1"
property in real sessions — but the start order gets recorded rather than assumed.

### 5bd.25 Code 29 explained, and the camera wiring state before compaction (2026-11-06)

**Code 29 is closed.** Gil confirmed the 73.70 ms it showed in `test_synch`, against
87.30–87.33 ms across 153 occurrences of the known-good session: **the reward value in one
port was changed during calibration.** So code 29 is a **valve-open time**, and its
duration is expected to move whenever valves are recalibrated.

Worth keeping as a rule for future comparisons: when a new recording is checked against a
known-good one, a **single** fixed-duration state moving while its siblings stay
sample-exact means **protocol or hardware calibration, not wiring**. A wiring fault cannot
change one state's duration selectively — it would relabel numbers, not retime one of
them. That is what let code 29 be set aside as benign on 2026-11-05 before Gil confirmed
it, and the confirmation is what turns the inference into a fact.

With that, **`test_synch` is fully accounted for**: every value matched the known-good set,
every hardware-timed state matched to the sample, the one duration change is explained, and
the only other anomaly was a 419 ms Trodes acquisition dropout at t = 10.85 s (§5bd.22).
The rewired Bpod connection is good.

**Camera state at the point Gil compacted the session.** He asked me to stop messaging the
camera-PC agent ("you burn token for vain") and to park the file-transfer problem — a flash
drive covers it for now and he is pursuing the network route himself. Both noted; the
durable state is in `docs/flir_camera_sync_design.md`, `docs/bpod_mcu_sync_wiring.md` and
memory.

Three things were settled in that exchange and one was not:

- **Settled, and it is the next action: `Din7` is disabled by default at the MCU** and the
  workspace file cannot enable it — neither our `.trodesconf` nor Trodes'
  `DeviceProfiles.xml` has any enable attribute in the schema. It is a GUI/MCU setting for
  Gil. **It must happen before the test recording**, because with Din7 off a null result
  cannot be told apart from a wrong pin.
- **Settled: the MCU-side pinout.** `Din7` is the **top row, third column** of a header
  labelled `5V 8 7 6 5 4 3 2 1 GND` left to right. The column immediately right is **Din6,
  a live Bpod line** — err left onto Din8.
- **Settled: the isolation topology.** The pull-up for the opto output must be powered from
  the **MCU** side, via a 2.2 k/4.7 k divider off the header's 5 V; pulling up from the
  camera's own 3.3 V would bond the grounds through the very part chosen to isolate them,
  invisibly.
- **Not settled: camera-side pin numbers**, and I stopped asserting them. I was wrong twice
  from unsourced recall (pin 6 for the 3.3 V supply; pin 2/1 for Line1/OPTO_GND). FLIR's
  docs give pin 4 = Line1 and pin 5 = opto ground **for cased models** — but FLIR fits two
  connectors with different pinouts, and our notes record the *board-level* JST part for a
  camera whose `-C` suffix implies the cased Hirose. Looking at the connector settles it.
  The camera agent's **software-toggle test** identifies the pair by function and makes the
  question moot, which is the better route regardless.

### 5bd.26 — Din7 cannot be enabled by us: it is a factory configuration

Gil asked what he should do on the Trodes side to enable Din7 — open Trodes, load an old
session, or just have the program running. The answer turned out to be **none of those**,
and the reason is worth recording because it closes a line of investigation rather than
advancing it.

The MCU manual page Gil photographed says digital inputs 7 and 8 are off by default. The
sentence that matters is the one just past the edge of the photo. SpikeGadgets' **Control
Units manual** (p.12, "Inputs/Outputs") and their **System manual** (same section) both
carry it verbatim:

> "By default, only Din pins 1-6 are turned on for digital event logging. **If requested,**
> this connector can also be configured for additional digital lines, including up to eight
> outputs with custom behavior (hardware synchronization, etc)."

and under the pin figure:

> "Douts 1-8 are turned off by default but can be enabled for custom applications."

"If requested" means requested from SpikeGadgets. This is an MCU firmware/hardware
configuration, not a setting exposed to the user, and four independent checks agree:

1. **Trodes 2.4.2 has exactly one MCU dialog.** Extracting the UI strings from `Trodes.exe`
   recovers the menu tree — the dialog is reached at `Settings → Headstage → MCU
   settings...`, and the only widgets inside it are `Sampling Rate (kHz)` and `Set Sampling
   Rate`. Nothing about digital lines. (The binary also carries `MCU Settings (None
   detected)`, which confirms the dialog needs live hardware to populate — relevant only if
   there had been anything in it to set.)
2. **No command-line setter exists.** `mcu_utils/windows/` ships `get_mcu_settings.exe`,
   `get_mcu_version.exe`, `set_rfchannel.exe`, `set_samplingrate_mcu.exe` and
   `recordSync.exe`. There is a getter for settings and setters for the radio channel and
   the sampling rate; there is no digital-input setter.
3. **The schema has no enable flag** (established earlier, §5bd.23). `DeviceProfiles.xml`
   declares `Controller_Din1` through `Controller_Din8` on the single DIO byte, all with
   `dataType="digital"` and no enable attribute; the session workspace matches. Notably the
   `Controller_DIO` profile declares no Douts at all, while the `ECU` profile declares 32 —
   consistent with the MCU's outputs being disabled at the factory too.
4. **The measurement agrees.** Din7 constant 0, Din8 constant 1, zero transitions across
   both a 300 s and a 99 s recording.

So the action is an **email to SpikeGadgets** asking them to enable Din7 (and Din8, since
asking for both costs nothing) for digital event logging. They will want the MCU serial and
firmware. Firmware comes free from any recording header — `controllerFirmwareVersion="3.18"`
in `test_synch.rec`. **The serial does not:** that header reads
`controllerSerial="65535 65535"`, which is the 0xFFFF unset sentinel, so the serial has to
be read off the sticker on the unit or from Trodes' `Help → About Hardware` with the MCU
connected. That is the one thing opening Trodes is good for here, and it takes a minute.

**One alternative worth raising in the same email**, because it could remove the firmware
change altogether: the manual describes the front-panel **Aux 2** jack as one that
*"receives pulses used for syncing frames on an Allied Vision GigE camera."* That is already
a camera frame-sync input. What is **not** known is where those pulses are stored in the
`.rec` and whether the path is usable by a camera that is not an Allied Vision GigE unit.
Worth a question, not worth an assumption.

**There is no free input to borrow instead.** Din1–Din6 all carry the Bpod state word, and
the codes run to 40, so the word needs all six bits. This rig has no ECU (the workspace
devices are `Controller_DIO`, `Multiplexed`, `neuropixels_LFP`, `SysClock`); an ECU would
bring 32 digital inputs, which is the heavyweight fallback if SpikeGadgets will not
reconfigure the MCU.

**Consequence for sequencing.** Wiring Din7 now is harmless and the wiring plan is unchanged
— same pin, same divider, same isolation rules. But a test recording made before the input
is enabled teaches nothing, because a null result cannot distinguish "wrong pin" from "input
disabled". The strobe test waits on SpikeGadgets' reply, not on the soldering.

### 5bd.27 — Popper's configuration, and where the 820 um channel-map offset comes from

Gil is building the Trodes configuration for Popper and asked two things of the archive:
the implant depth, and which probe port the Neuropixels is connected to.

**Depth: 7.8 mm.** `Z:\Gil\Popper\rat_metadata.json`, `implant_depth_mm = 7.8`, positive,
brain surface down to probe tip. For context, Shamir_1 is 7.5 and Begin_1 is 8.0. Popper's
probe is a *different physical probe* from Shamir's — serial `19108309652` against Shamir's
`19108309381` — so nothing about Shamir's hardware setup should be assumed to carry over.

**The probe port is not recorded anywhere, and that is worth stating plainly rather than
guessing at.** Three places were checked:

- `Z:\Gil\Popper` holds only `bpod\`, `probe_marker_label.jpg` and `rat_metadata.json`.
  There are no Popper recordings at all, so there is no header to read it out of.
- The `rat_metadata.json` schema has no port or slot field, on any rat.
- The `.rec` header and the `.trodesconf` have no such field either: grepping the
  `test_synch` header for `port`, `slot` and `probeSerial` returns **zero** hits.

What Trodes *does* have is a **logical** probe index, not a physical port. The strings in
`Trodes.exe` show the Neuropixels custom source GUI offers "Number of probes", "Probe
selection", "Probe %1", `probeIndex`, and "Copy settings from this probe to all other
probes". The workspace stores one `<SourceOptions name="NeuroPixels1">` block per probe, and
`20260901_084658_shamir.trodesconf` has **exactly one**, which is why the exports are
`probe1.dat` and `channelmap_probe1.dat`. So for a single-probe implant Trodes calls it
**Probe 1** regardless of which connector it physically sits in. A physical port label has
to be read off the hardware, or seen in `Source -> Specific source control -> Open custom
source GUI for neuropixels` with the probe connected.

**Din7 needs no workspace change.** `Controller_Din7` is already declared in
`20260901_084658_shamir.trodesconf` with `bit="6"`, `input="1"`, alongside all eight bits.
Every bit of the Din7 problem was MCU-side (§5bd.26); the config file was never the obstacle.

**The incidental finding, which matters more than the questions asked.** Counting the
workspace's `channelsOn` selection: 960 entries, **384 on, contiguous, sites 82 through
465**. Those 82 leading off-sites are 41 rows at the Neuropixels 1.0 20 um row pitch =
**820 um** — which is exactly the uniform `coord_dv` offset between channel map A and map B
recorded in `Z:\Gil\Shamir_1\channel_map_history.md` (A spans 920..4740 um, B spans
100..3920 um, "per exported row: coord_dv offsets [820] um"). The history file also notes the
row order is "rotated by 82 positions" — the same 82.

So the long-open "820 um depth origin" question is not two arbitrary maps: **one map
measures depth from the probe tip and the other from the first *selected* site**, and the
820 um is the selected bank's offset up the shank. That does not by itself say which
convention is correct, but it does mean the choice is made when the channel selection is
made, and it propagates silently into every unit through
`depth_mm = implant_depth_mm - phy_depth/1000`. **Decide the convention before Popper's
first recording**, not after thirty sessions exist under it.

**One sign-convention trap in the workspace**, so it does not get "fixed" by mistake:
`ProbeLocation` reads `1.2 2.45 0` = AP, ML, angle, with ML **positive**, while
`rat_metadata.json` uses `implant_ML_mm = -2.45` for the same left-hemisphere implant.
Popper has the same coordinates (ML -2.45, AP 1.2, angle 0), so `ProbeLocation` copies
across unchanged.

Still outstanding on Popper: `headstage_serial_number` is `null`. The Shamir recordings
report `headstageSerial="01803 00185"`; whether that is the headstage Popper will run on is
Gil's to confirm before it goes in the file.

### 5bd.28 — The first camera-sync wiring attempt failed, and why

The Din7 strobe test was run on 2026-11-06 with `test_camera.rec` and produced nothing:
255 s recorded, **Din7 constant 0, zero transitions**, while the camera was confirmed
streaming. Din8 sat at its usual floating 1 and Din1-Din6 were idle (Bpod not running).

The fault is now fully diagnosed, and it was in the MCU-side divider, not the camera.
Measuring resistance between the three detached jumper terminals gave:

| between | measured | expected |
|---|---|---|
| red (5V) ↔ purple (Din7) | **2.2 k** | 2.2 k &nbsp;✔ |
| blue (GND) ↔ purple (Din7) | **0** | 4.7 k &nbsp;✘ |
| red ↔ blue | **2.2 k** | 6.9 k &nbsp;✘ |

So **purple and blue are the same node**: the 4.7 k is bridged across and Din7 is tied
directly to ground. Almost certainly the solder bridge on the perfboard ran one pad too far
and caught the 4.7 k's ground-side pad along with the three junction pads. The pull-up half
was built correctly; only the pull-down leg was shorted out.

That single fault explains every observation: Din7 held hard at 0 V cannot be moved by an
open-drain output, so the camera's pulses had nowhere to go, and the line looked exactly as
it had before any wiring existed.

**Three things worth keeping from this.**

1. **Everything was potted in silicone before anything was measured.** That turned a
   two-minute solder fix into a lost session and made the junction unreachable — the
   diagnosis had to be done entirely from the three jumper ends. The order must be: solder,
   measure the three resistances, measure 3.4 V on the output, *then* insulate.
2. **The three-terminal resistance test is the right diagnostic** and should be reused. It
   characterises the whole assembly from outside without disturbing the MCU: red↔purple
   2.2 k, blue↔purple 4.7 k, red↔blue 6.9 k. Any 0 means a bridge is too long; a swapped
   pair shows up as 4.7/2.2 instead of 2.2/4.7.
3. **A dead input and a shorted input look identical from the `.rec`.** Din7 read constant 0
   before the wiring and constant 0 after it, so the recording alone could never have
   distinguished "input disabled", "wrong pin" and "node shorted to ground". Only the meter
   separated them. This is the same ambiguity flagged in §5bd.26, and it argues for a
   bench check of the divider output before any future test recording is made.

**Also confirmed along the way:** the camera session states that `ExposureActive` asserts
(opto conducting, line pulled low) **during** the exposure and releases between frames, so
with the camera idle Din7 should rest **high** via the pull-up. That settles the polarity
question — the expected signature stays as recorded: ~15 ms low, ~1.67 ms high, one falling
edge per frame.

**One incidental finding in `test_camera.rec`:** a timestamp gap at t = 44.894 s losing
1,574,221 samples (**52.5 s**). It falls well before the camera ran, so it did not affect
this test, but it is a much larger dropout than the 419 ms seen in `test_synch` (§5bd.22)
and is worth watching for in real sessions.

Next: rebuild the divider on fresh perfboard, verify by the three-resistance test and the
3.4 V output check, insulate only after both pass, then repeat the strobe recording.

### 5bd.29 — The camera strobe works perfectly, and Din7/Din8 are provably dead (2026-11-09)

The rebuilt divider works. The camera signal is clean. Din7 and Din8 on this MCU are not
live digital inputs, and we now have the control experiment to prove it rather than infer it.

**The rebuild.** Gil rebuilt the board on fresh perfboard after the first attempt was ruined
by a solder bridge (§5bd.28). Three copper bars, joined only through the two resistors:
top = RED + 2.2k top leg; middle = PURPLE + 2.2k bottom leg + 4.7k top leg + camera WHITE;
bottom = 4.7k bottom leg + camera BLUE + BLUE jumper to GND. Within a bar the order of the
soldered items is irrelevant -- it is one electrical node. Only which bar each leg lands on
matters, and keeping the middle and bottom bars from touching is the whole game.

**Measured, with purple out of the header:**

| condition | node voltage | meaning |
|---|---|---|
| camera connector unplugged | 3.1 V | divider unloaded, opto absent |
| camera connected, not streaming | 0.364 V (stable) | opto conducting |
| camera connected, streaming | 0 to ~2.0 V, wandering | meter averaging a 60 Hz square wave; wanders with auto-exposure |

3.1 V rather than the predicted 3.43 V. Still a clean HIGH (threshold ~2 V) and safely under
the 5 V absolute maximum, so not worth chasing.

**Polarity is the opposite of what the camera session described.** The line rests LOW and goes
HIGH during exposure, not the reverse. Probably a line-inverter setting or a non-default
`Line Source` on Line1. It changes nothing for the wiring, but it has one consequence worth
remembering: an idle camera and a disconnected wire both read 0, so a flat line in the `.rec`
does not by itself distinguish "camera not running" from "wiring failed".

**The signal, measured in the file.** `F:\Gil\camera_test.rec` with purple on Din4, Bpod
connector removed, camera free-running in SpinView:

```
Din4   2298 rises   frac high 0.5988   period 500.0 samples (500.0-501.0)
                                        width  299.0 samples (299.0-300.0)
       implied frame rate 60.0000 fps, 0 ppm from nominal
       pulse width 9.97 ms median (9.97-10.00 ms)
```

Period exactly 500 samples at 30 kHz with a one-sample spread. 2298 pulses in 38.3 s. This is
a better sync signal than the Bpod state word.

**The control experiment.** The decisive step, and the one that should have come first. With
nothing proven about the logging chain, "lines 7-8 are disabled" and "our board does not
deliver what we think it does" are indistinguishable. So purple was walked along the top row
one column at a time, letting Trodes name each pin:

| pin Trodes reported | pulses seen |
|---|---|
| Din4 | yes |
| Din5 | yes |
| Din6 | yes |
| Din7 | **no, flat 0** |
| Din8 | **no, flat 1** |

Identical signal, identical board, identical camera, moved one column at a time. The enabled
range stops at Din6. This also confirms the physical pin mapping is exactly
`5V 8 7 6 5 4 3 2 1 GND` left to right on the top row, which had been assumed but never
verified against Trodes' own labels.

**A second, independent sign.** With the Bpod connector removed, every unconnected working
input floats to 1 -- Din1, 2, 3, 5, 6 and Din8 all read constant 1. Din7 reads constant 0.
Din7 does not behave like any live input on the header.

**Three lessons.**

1. *Din7 appearing on the Trodes screen proves nothing about the hardware.* The workspace
   declares `Controller_Din1..Din8` with `input="1"` regardless of what the controller
   actually has, so the on-screen line exists whether or not the pin does. This cost most of
   a day: we treated "I found Din7 and it shows up" as evidence the line was enabled. It was
   not. §5bd.26's original conclusion -- that only SpikeGadgets can enable it -- was right
   after all, and the retraction in that section was itself wrong.
2. *Build the positive control before the experiment, not after.* Every hour spent on
   "is it the board or the pin?" would have been answered in two minutes by putting the same
   signal on a known-good input first.
3. *A multimeter cannot read a 60 Hz line, and its averaging is not noise.* The reading that
   wandered between 0 and 2 V was the correct answer to a question the meter cannot ask.
   Hz mode, or a recording, is the only honest readout.

**Next:** Slack SpikeGadgets to ask how lines 7 and 8 get enabled -- firmware we can flash, or
does the unit go back. Restore the Bpod wire to Din6. A fallback worth designing while we
wait: run the signal the other way, a Bpod- or Trodes-visible pulse into the camera's Line0
(opto-isolated input, black wire) stamped into each frame's chunk data. Same alignment,
opposite direction, no new MCU input needed.

### 5bd.30 — SpikeGadgets' answer: the camera input is Aux2, and Din8 is already wired to it (2026-11-09)

Daria at SpikeGadgets replied the same afternoon. Two statements, both important:

1. **Din7 and Din8 are used by the system** — they are not free header pins and will not be
   made free. The walk-the-signal experiment in §5bd.29 was reading pins that are internally
   committed.
2. **Din8 is already configured in the MCU to take a camera pulse**, routed from the **Aux2
   RJ45 port on the front of the MCU**, not from the Digital I/O header. SpikeGadgets ship a
   cable for the Allied Vision Manta/Mako; for any other camera you make your own adapter.

**Aux2 RJ45 pinout, as given:**

| RJ45 pin | function |
|---|---|
| 1 | **digital input** — this is the line that reaches Din8 |
| 4 | **+5 V**, supplied by the MCU |
| 5 | **ground** |

**Hard constraint: "not input more than 3.3V pulses."** Our current divider is 2.2k/4.7k off
5 V, nominally 3.43 V and measured 3.1 V. That is at or slightly over the stated limit.
Change the top resistor so the output lands comfortably inside it — 4.7k/4.7k gives
5.03 x 0.5 = **2.52 V**, still well above any plausible V_IH (CMOS 0.7 x 3.3 = 2.31 V,
TTL 2.0 V) and with real headroom under 3.3 V. Gil has 30 of each resistor, so this is a
one-component swap.

**This retro-explains a measurement from §5bd.29.** With nothing connected, Din7 read a
constant **0** while Din8 read a constant **1**. Every working header input floats to 1. Din8
floating high is exactly what an enabled input sitting behind an unconnected Aux2 pin 1
should do; Din7 held at 0 is what a line committed to an internal function looks like. The
two lines were never equivalent, and the difference was visible in the data the whole time.

**That the MCU supplies 5 V on pin 4 at all is the tell.** You only need a rail next to a
digital input if the expected source is open-collector and needs an external pull-up — which
is precisely what the FLIR Line1 OPTOOUT is. The circuit SpikeGadgets expect is the circuit
we already built.

**Plan:** keep the perfboard, move its three wires from the Digital I/O header to an RJ45
plug. A cut Ethernet cable is the simplest adapter; under T568B, pin 1 = white/orange,
pin 4 = blue, pin 5 = white/blue. Verify the cable's wiring with a meter before trusting the
colour code. Restore the Bpod wire to Din6 — the header is no longer involved at all. The
strobe will then appear on **Din8** in the `.rec`.

**Open questions for Daria:** (a) the logic threshold on Aux2 pin 1, and whether the MCU
already provides an internal pull-up there — if it does, our divider may be unnecessary or
may need re-sizing against it; (b) confirmation of the RJ45 pin-numbering reference in the
image she sent. She also asked what camera we use: FLIR Blackfly S BFS-U3-13Y3M-C, USB3.

### 5bd.31 — The MCU already contains the pull-up: the divider board is not needed (2026-11-10)

Daria's follow-up, with a board layout image: **"The signal runs through a 200 Ohm and then 10k
pullup on the MCU side, not sure what the threshold is, I think 2.7V but [Magnus] will know
this."**

So inside the MCU, Aux2 pin 1 passes through a 200 ohm series resistor and is pulled up to
+3.3 V through 10 k before reaching the logic input that becomes Din8. That is precisely the
circuit the perfboard was built to supply. It was there all along, and the 5 V on pin 4 is
for cameras that need something else, not for us.

**The wiring is now two wires and no components:**

| camera wire | Aux2 contact |
|---|---|
| WHITE (Line1, ExposureActive, open-drain) | 1 |
| BLUE (opto GND) | 5 |

Contact 4 (+5 V) stays unused. Levels: 3.3 V between frames when the opto releases, ~0.2 V
while exposing. The camera sinks 3.3/10200 = **0.32 mA**, nothing against its 25 mA rating.

**Two ways to get this wrong, both computed rather than guessed:**

1. *Leave the whole board connected.* Our Thevenin source is 3.43 V behind 2.2k||4.7k =
   1.499 k; theirs is 3.3 V behind 10.2 k. The node settles at
   `(3.43/1.499 + 3.3/10.2) / (1/1.499 + 1/10.2) = 3.41 V` -- **above their stated 3.3 V
   maximum**, and pushing current back into the MCU's own rail.
2. *Remove only the red wire and leave the 4.7 k to ground.* Worse. It divides against their
   internal 10.2 k: `3.3 x 4700/14900 = 1.04 V`. Below any plausible threshold -- the line
   would never register a pulse. A partial teardown is more dangerous than none.

The board comes out of the signal path entirely.

**On the threshold.** Daria put it at roughly 2.7 V but was explicit that she was unsure and
referred it to a colleague. It does not block anything: driven directly the line swings the
full 0 to 3.3 V. Worth recording that the divider's measured 3.1 V would have left only
0.4 V of margin above a 2.7 V threshold -- so the direct connection is not merely simpler, it
is the better circuit. Had we built the Aux2 adapter around the divider without asking
question 1, we would have shipped something marginal and never known why it was flaky.

**The lesson, and it is the same one as §5bd.29 in a different costume.** Both failures came
from building before asking what was already there. First we assumed a header pin was wired
to the line that bears its name; then we assumed an input had no pull-up because our camera's
output needed one. One question to the manufacturer, asked before any soldering, would have
replaced the entire perfboard with two wires.

**Still open:** Magnus to confirm the threshold. Then build the adapter, restore the Bpod wire
to Din6, and record. Expect the strobe on **Din8**, resting high and dropping ~10 ms per frame.

### 5bd.32 — Our MCU has no RJ45: Aux1/Aux2 are round connectors (2026-11-10)

Before building anything, Gil photographed the MCU front panel. There is **no RJ45 on this
unit**. The front carries two **round multi-pin sockets engraved 1 and 2**, a card slot, two
buttons and a power LED.

Daria's PCB image showed a staggered 2x4 pad layout with mounting posts, which is an RJ45
footprint. So her unit and ours are different hardware revisions. The electrical design behind
the connector is probably unchanged -- 200 ohm in series, 10 k pull-up to 3.3 V -- but **the
pin numbering does not carry across**: contact 1/4/5 on an RJ45 says nothing about which pin
is which on a round 8-pin connector.

**Nothing is to be unsoldered or cut until SpikeGadgets confirm the connector.** Asked:
whether the round socket marked 2 is Aux2, its pin numbering (input / 5 V / ground), the
connector type so the mating plug can be ordered, and whether the 200 ohm / 10 k arrangement
is the same on this revision. Photo sent.

**Do not probe the socket blind** to find out. The pins are recessed and closely spaced, and
slipping between the 5 V and ground pins shorts the MCU supply. Once the pinout is confirmed,
the meter check is still worth doing -- find the pin pair reading 5 V, then the pin sitting at
3.3 V -- as verification, not as discovery.

**State of play, so this can be picked up cold:**

- The divider board is built, correct, and **not needed** (§5bd.31). It stays intact and out
  of the signal path. Camera WHITE and BLUE are still soldered to it; unsolder, never cut.
- The camera signal itself is fully proven: 60.0000 fps, 0 ppm, 2298 pulses in 38.3 s
  (§5bd.29). Nothing about the camera side is in question.
- Final wiring, once the connector is known: camera WHITE (Line1) to the input pin, camera
  BLUE (opto GND) to ground. No 5 V, no components.
- The pulses will land in the **Din8** column of the `.rec`.
- Rear header: Din7 and Din8 are used internally and unusable. **The Bpod wire still needs
  putting back on Din6.**
- Open with SpikeGadgets: the connector questions above, plus the input threshold (~2.7 V per
  Daria, unconfirmed, referred to Magnus Karlsson).

### 5bd.33 — RIS access granted, surveyed end to end (2026-11-10)

Charlotte Rouse granted access: *"zur has been granted access to compute-akepecs &
compute2-akepecs."* WUSTL key **zur**. First interactive login created the home directory;
the ed25519 key prepared on 2026-11-04 was installed and `~/.ssh/config` filled in, so
`ssh compute1` now works non-interactively from this PC. The read-only survey
(`scripts/survey_ris.sh`) ran clean, plus four targeted follow-ups. Nothing was submitted.

**Groups:** `compute-akepecs`, `compute-akepecs-t1`, `compute2-akepecs`, `compute2`,
`storage2-akepecs-gil_zur-rw`, `storage1/2/3`. Both platform allocations, and a personal
read-write slice on storage2. The 1 Jan 2027 new-user cutoff is cleared.

**Cluster:** IBM Spectrum LSF 10.1.0.12, cluster `compute1-lsf`.

| queue | walltime | per-user job limit | state now |
|---|---|---|---|
| `general` | **40320 min = 28 days** | 500 | **19765 jobs, 16251 PENDING, 3514 running** |
| `general1k` | 40320 min = 28 days | 1000 | **completely empty** |
| `general-interactive` | 1440 min = 24 h | 64 | 388 jobs, 175 pending |

28 days confirmed, so a 5-6 day CPU sort fits with room. `zur` has **no personal MAX cap**,
so concurrency is bounded by the queue limit, not by policy.

**The congestion is the new risk, and it is severe.** 16251 jobs pending on `general` against
3514 running. Throughput maths assuming jobs start promptly is now suspect. `general1k` has
the same 28-day limit, double the per-user cap, and *zero* jobs on it -- that asymmetry is
either an opportunity or a trap, and it is worth one question to Unmukt before relying on it.

**GPUs:** host group `general-gpu`, 17 hosts. Mostly Tesla V100 (SXM2 and PCIE), plus compute
capability 8.0 (A100), 8.6 (A40/A10) and 9.0 (H100) elsewhere in the cluster. At the time of
the survey six GPU hosts were **completely idle** -- roughly 24 V100s doing nothing. KS4 needs
3.3 GB VRAM, so any of them is enormous overkill. Request with `-R 'gpuhost' -gpu "num=1"`.

**Node size:** largest hosts are 128-core AMD EPYC with 1.9 TB RAM. Kilosort cannot span
hosts, so one node's core count is the ceiling -- and 128 is far above anything we need.

**Storage -- the question that decided concurrency is answered YES.**

| | path | state |
|---|---|---|
| scratch | `/scratch1/fs1/akepecs/zur` | **writable**, 58 TB free on the filesystem |
| permanent | `/storage2/fs1/akepecs/Active/Gil_Zur` | **writable**, currently empty |
| storage1 | `/storage1/fs1/akepecs` | exists but not readable by us |

A 1 GiB write test to scratch ran at **4.0 GB/s**. Unmukt already parks **325 GB** in
`/scratch1/fs1/akepecs/unmukt`, which is exactly one session's input -- so the precedent for
staging a whole `probe1.dat` on scratch is already set in this very allocation. **Storage no
longer limits how many sessions run at once.**

`mmlsquota` cannot see these filesystems from the client host, so the per-fileset quota is
still unknown. The daily `RIS_cleanup_policy.akepecs.<date>` files in the scratch root are
plain lists of doomed paths and do **not** state the purge window; that still needs asking.

**Two operational findings not in the plan:**

1. **Key-based login obtains no Kerberos ticket** -- `klist` reports no credentials cache at
   all. Despite that, reads and writes to both `/scratch1` and `/storage2` worked. So the
   Kerberos worry recorded earlier may not apply to these mounts, but it has only been tested
   from the submission host, not from inside a job on an exec node.
2. **No docker client on the submission host** (`docker: command not found`), and the system
   python is 3.6.8 with no conda. The container is mandatory and must either be pulled from a
   registry at submit time or built through RIS's own mechanism -- we cannot `docker build`
   here. That is a real gap in the plan: `docker/Dockerfile.kilosort4` exists but there is no
   path yet from that file to an image the cluster can run.

**Next, in order:** settle how the image gets built/published; then the validation sort of
`20260916_110311` on RIS compared against the output on F; only then the backlog. Gil's nod
is still outstanding for the first real submission and for the batch of 30.

### 5bd.34 — The Kilosort4 container is built, pushed and proven on a cluster GPU (2026-11-10)

The image blocker from §5bd.33 is cleared, and not the way the docs suggested.

**The route.** RIS's own `esub.docker_build` builds on the cluster and pushes to Docker Hub
(WashU has no private registry). Unmukt builds locally with Docker Desktop instead -- that is
the documented path, but it is the wrong one here: **Docker Desktop must not go on the
Neuropixels PC**, which is the acquisition rig, and which already cannot run Trodes and
Kilosort at once ([[project-trodes-kilosort-gpu-conflict]]). Building on the cluster installs
nothing locally.

**The option names are not in the public docs** -- the Atlassian pages were partly
unreachable. They came from running the builder's own `--help` through LSF:
`--login_only`, `--login-username`, `--login-password`, `--push-url`, `-f/--file`, `-t/--tag`.
The push URL is an *esub* argument: `-a 'docker_build(<url>)'`, not a `bsub` flag.

**Built:** `zurgil/kilosort4:4.0.24-cu121`, public, digest
`sha256:f79d0ab9ccf323e555e81da2046022691e1220c5e8ab9204500c7062b203bbe8`.
Job 705043, **dispatched in 1 second, done in 10 minutes**, 4 cores / 16 GB (it used 54 MB).

Build-time assertion fired and passed:

    pins OK: 4.0.24 2.4.0+cu121 cuda 12.1 1.26.4 1.13.1

Same kilosort version as every session already sorted here, which is what makes the planned
re-sort comparison meaningful rather than a version experiment.

**Proven on a real GPU.** Job 705302, `-R 'gpuhost' -gpu "num=1"` on the `general` queue:

    KS 4.0.24 torch 2.4.0+cu121 cuda_available True device Tesla V100-SXM2-32GB

A V100 with 32 GB against KS4's measured 3.3 GB requirement.

**Design decision worth keeping.** The image contains no lab code at all -- the scripts are
mounted from `/storage2/fs1/akepecs/Active/Gil_Zur/code` via `LSF_DOCKER_VOLUMES`. Two
consequences: the image can be public, so the cluster needs no stored credential to pull it
mid-run; and editing a script costs nothing, where baking it in would cost a rebuild and a
push every time.

**A credential mistake, recorded so it is not repeated.** The builder's
`--login_password_stdin` does *not* suppress terminal echo despite its name, so Gil's Docker
Hub password appeared on screen. It was reset immediately. The replacement was an access
token, which then also ended up in the transcript and was revoked and reissued. **Use an
access token, never the account password, and expect it to be visible** -- the entire reason
a token is right here is that revoking one is two clicks.

**Unmukt's answers to the four questions:**

1. *Images* -- build locally with Docker Desktop and publish a public tag. Superseded by the
   cluster-build route above, for the acquisition-PC reason.
2. *Queue waits* -- "Most of them are rubbish; some have been pending for a year because the
   resource requested is just not possible. Typically a few minutes to a day; faster if CPU
   only." Both our jobs dispatched in ~1 s against 16251 pending, which corroborates it. The
   caveat is that **GPU jobs are the slower case**, and both of ours were short.
3. *`general1k`* -- he misread the question, but volunteered something better: **he had RIS
   create his own queue with no job limits**. Worth pursuing for the 30-session batch.
   (`general1k` itself is `USERS: f.qichen`, so it was never available.)
4. *Scratch purge window* -- he does not know either, but **runs a daemon that syncs off
   scratch continuously**, which answers the practical question: do not trust scratch to hold
   anything. Copy each session's results out as soon as its sort finishes, rather than at the
   end of the batch. He is sharing the script.

**Next:** stage one session to scratch and re-sort `20260916_110311`, comparing against the
KS4 output already on F. Expect a few percent difference in spike count between GPU runs on
different hardware; a large divergence means something is wrong. Gil's go-ahead is still
outstanding for that first real submission and for the batch of 30.

### 5bd.35 — A sort-vs-sort comparison, and the self-check that caught two bugs in it (2026-11-10)

To judge a cluster re-sort against the sort already on F we need a comparison, and the
honest framing matters: neither sort is ground truth. Both are estimates of the same
recording, so "accuracy" is meaningless and the only question is agreement — did the
second run reproduce the first? Agreement is directional, so every matched pair gets
**both** recall (did the test sort find the reference unit's spikes) and precision (is
the test cluster pure), tiered by the reference sort's KSLabel. A unit the test run
split scores high precision and low recall; a merge scores the reverse. One number
cannot distinguish them, so `scripts/compare_two_sorts.py` never prints one, and counts
splits and merges on their own lines.

The matching took three attempts. The self-check — running the reference sort against
*itself*, where every number must come out exactly 1.000 — is what caught the first two.

**Attempt 1** walked both spike trains in step for every cluster pair. Correct, but that
is 472×472 Python-level loops over 25 M spikes. It does not finish.

**Attempt 2** was fast and wrong: for each reference spike, find the nearest test spike
anywhere in the recording and credit its cluster. Measured on this session, **74.65% of
spikes have another spike within 20 samples** — across 384 channels that is simply what
dense extracellular data looks like. So "nearest spike in the recording" is usually a
different cell firing at nearly the same moment. It scored the sort against itself at
**0.963**. Had the self-check not run first, that 3.7% would have been read as a real
GPU-vs-CPU difference and chased into Kilosort.

**Attempt 3, in now,** is two-stage. Stage 1 is a cheap upper bound: for each reference
spike take the *window* of all test spikes within ±tol and tally their clusters — pairs,
not nearest, so no real partner is ever missed. Stage 2 takes the dozen clusters that
bound says are plausible and counts them exactly. Exact because `searchsorted` brackets
each reference spike between two spikes *of that candidate cluster*, and the nearer of
that pair is necessarily the nearest in the whole train — so the cluster's internal
spacing is irrelevant. An intermediate version warned about internal spacing; that
warning was wrong and was replaced by a check for the failure that can actually occur,
two reference spikes claiming one test spike (which would push precision above 1).

The self-check also found a second, quieter bug: `kilosort_original/` holds the pristine
arrays but **no `cluster_KSLabel.tsv`** — the labels live in the parent phy directory. So
every unit was being reported as unlabelled and the tiering was silently empty. It now
looks one directory up and prints which file the labels came from.

Self-check result: **1.000 median, IQR and minimum on both recall and precision across
all 472 units** (187 good, 285 mua), no splits, no merges, nothing unmatched on either
side. Two minutes on 25 M spikes. Default tolerance is ±10 samples (0.33 ms); two runs of
the same algorithm on the same data align to within a sample or two, so that is generous.

### 5bd.36 — Getting a GPU on compute1 is not a matter of asking for one (2026-11-10)

The first real submission, job 715494 (`20260916_110311`, Th_universal 9.0, Th_learned
8.0) was queued at 20:15 and sat in PEND with:

    Job's requirements for reserving resource (ngpus_physical) not satisfied: 4 hosts

while `bhosts -gpu` reported every GPU on those hosts completely idle. Idle and available
are different things here. What the survey turned up:

- **The lab has a purchased GPU entitlement.** LSF exposes it as the host resource
  `akepecs_t1_gpus`, value 1 on each of six V100 hosts (exec-195 to 198, 209, 210). This
  is the condo model — the Kepecs lab (akepecs = Adam Kepecs, our own lab) bought capacity. It also answers the cost question: we
  are spending an allocation that already exists, not running up a new per-hour charge.
- **Gil is in four groups**, not one: `compute-akepecs`, `compute-akepecs-t1`,
  `compute2-akepecs`, `storage2-akepecs-gil_zur-rw`. The `-t1` suffix is the subscription
  identity and is a different group from the one I had been submitting under.
- **Those six subscription hosts sit in host group `subscription`, served by a queue of
  the same name** — on whose USERS list `compute-akepecs-t1` does appear. But that queue
  is `Closed:Inact`, with zero jobs. So on compute1 the lab's own subscription GPUs are
  not reachable at all right now.
- **The `general` host group does contain GPUs**: 8 hosts with TeslaV100_SXM2_32GB, 4
  with NVIDIA A40, 2 with A100_SXM4_40GB. So GPU work via `general` is possible.
- Claiming `rusage[akepecs_t1_gpus=1]` did **not** fix it (job 715613, same error), so the
  entitlement was not the blocker — the `gmodel=TeslaV100_SXM2_32GB` constraint is the
  likelier culprit, since it pins the job to exactly the card family that mostly lives in
  the closed subscription group.
- `general` itself is heavily loaded: **15,146 jobs pending, 3,206 running**, JL/U 500,
  fairshare with backfill. PRIO 1, against subscription's PRIO 9.

Job 715643 is a deliberately cheap probe of the remaining hypothesis — 1 core, 4 GB,
10 min walltime, `-gpu "num=1"` with **no model constraint** — running `nvidia-smi` and a
`torch.cuda.is_available()` check in our own image. If a plain GPU request schedules, the
`gmodel` pin was the whole problem and the fix is to drop it. Lesson recorded for its own
sake: a ten-minute probe is the right way to test a scheduler hypothesis, not a 48-hour
job that fails to start.

**Still open:** whether the `subscription` queue is closed permanently or by maintenance,
and whether **compute2** (Gil also has `compute2-akepecs`) is where the lab's capacity
actually runs — Charlotte's grant named both clusters, and compute2's client hostname has
never been confirmed.

---

## 5bd.37 — The cluster sort is validated. Decision: no run-to-run control.

Job **715803** sorted `20260916_110311` on a Tesla V100-SXM2-32GB in **02:41:24**, against
roughly 5.4 h for the same session on the Trodes PC — and without occupying the acquisition
machine at all. 174 units came out with good refractory periods.

The job's own log printed its map at run time rather than us trusting the staging step:

    384 channels | yc 920.0..4740.0 um | xc 2426.0..2474.0 um

That is map A, correct for a September session (map B begins 2026-10-26). This is Gil's
standing requirement — verify the map on the data the sort actually read, not on the file
we think we uploaded — satisfied inside the run.

**Agreement with the September sort on F** (`kilosort_original/`, i.e. before any of Gil's
curation was applied — the parent phy dir's `cluster_info.tsv` has had verdicts in it since
October and is not a fair reference):

| | cluster | Trodes PC |
|---|---|---|
| spikes | 25,348,358 | 25,229,992 |
| clusters | 449 | 472 |

+0.47% on spike count. Per tier, using the OPEN matching (recall = fraction of the
reference unit's spikes found; precision = purity of the test cluster that holds them):

- **good, n=187** — recall median 0.987 (IQR 0.969–0.996, min 0.148, >=0.90 in 170/187);
  precision median 0.982 (IQR 0.913–0.994, min 0.001, >=0.90 in 144/187)
- **mua, n=285** — recall median 0.965 (IQR 0.928–0.981, min 0.033, >=0.90 in 230/285);
  precision median 0.965 (IQR 0.909–0.980, >=0.90 in 219/285)

Weighted by spikes rather than by unit: **95.05%** of all 25,229,992 reference spikes sit in
units matched at recall >= 0.80.

Structurally the residual is merging, not loss: 10 splits, **55 merges**, 29 reference units
with no counterpart holding 3.08% of spikes, and zero units with no overlapping cluster at
all. 449 clusters instead of 472 is the same fact seen from the other side. Disagreement
concentrates almost entirely in small units — of 50 reference units under 1,000 spikes only
10 agree, while 191/211 agree in the 20k–100k band and 63/68 above 100k.

**The control we are not running.** Kilosort4 is nondeterministic run to run and we have
never measured its own spread, so strictly speaking the 55 merges cannot be attributed to
the cluster rather than to Kilosort being run a second time. The clean experiment is a
second cluster sort of this same session scored against this one, at 2:41 of free compute.
I recommended it; **Gil declined on 2026-11-11** — "i dont see any reason for another test,
it works." That is a reasonable call on the evidence: 95% of spike mass reproduced, the
residual is in units too small to carry analysis weight, and the direction of the residual
(merging of small units) is the known signature of a reclustering rather than of a broken
environment. Recording the decision so that nobody later reads the uncertainty as an
oversight. If a specific session's cluster sort ever looks wrong, this control is the first
thing to run.

**Verdict: the cluster is cleared for production sorting.** What remains before the batch of
30 is logistics, not validation — staging (data + the right probe JSON per session), the map
A/B boundary at 2026-10-26, streaming results off scratch, and the scratch purge window,
which is still unknown.

---

## 5bd.38 — Probe maps for the whole batch, and the archive bundle that lets F be cleared

Two jobs today, both prerequisites for sorting the remaining sessions on the cluster.

### The channel map comes from the session, not from the calendar

I had planned to pick map A or map B per session by date, with the boundary at
2026-10-26. That works but it is a rule applied to the data from outside, and the
failure it protects against -- an 820 um depth error that produces a completely
plausible-looking sort -- is exactly the kind that nothing downstream would catch.

It is also unnecessary. Every session carries its own `channelmap_probe1.dat`, written
by Trodes at acquisition, and the geometry is read from that. `make_probe_maps_batch.py`
derives each map from the session's own file and uses the date **only to check the
answer**, never to produce it. It groups the sessions by date and asserts the two groups
have different geometry and that each is internally consistent.

Result over 31 Shamir sessions:

| group | yc range | sessions |
|---|---|---|
| through 2026-10-23 | 920.0 – 4740.0 um | 27 |
| from 2026-10-26 | 100.0 – 3920.0 um | 3 |

920 − 100 = **820 um**, exactly the documented shift, and the split falls exactly on the
documented date. The map story is now confirmed from the raw acquisition files rather
than believed from a note.

Stronger still: for the four sessions that already had a map, the regenerated file is
identical in all five fields (`chanMap`, `xc`, `yc`, `kcoords`, `n_chan`) to the one
used for the original sorts. The md5 differs only because of JSON whitespace. So the
generator reproduces exactly what was used before.

`20261015_094934` has no `channelmap_probe1.dat` and no exported `.probe1.dat` -- it
survives only as a `.rec` and needs the old export route. It is the one session of 31
that cannot be staged directly from the archive.

Two things were needed to import the Trodes reader at all, both recorded because they
will bite again: it calls `np.set_printoptions(threshold='nan')` at module level, which
has raised `TypeError` since numpy 1.13; and it is a *script*, not a library -- at import
it runs `readTrodesExtractedDataFile(argv[1])`, so importing it with any command line
makes it try to open your first argument as a Trodes file. Both are worked around during
the import and restored immediately after.

### The review bundle: making a session self-contained on Z

The verdicts for a session do not live in the repo or the dataset. They live in the
review tool's own database on claude.ai. That is fine while the page is reachable and
useless as an archive, which matters now that F is to be cleared.

So the database was exported and placed on Z beside the data, at
`<session>.kilosort/review/`: the published page, the `data/merge_candidates.json` it
fetches, the three verdict collections as plain files, and the inputs needed to rebuild
the page from scratch. A README states where the verdicts already are in the dataset and
warns about a real trap -- `cluster_group.tsv` and `cluster_KSLabel.tsv` are byte-identical
to each other, because they are Kilosort's own output and curation never touched them.
The decisions are in the `group` column of `cluster_info.tsv` (114 good / 327 mua /
31 noise, against 189 / 259 / 24 before curation).

The export was checked against the 2026-10-20 one document by document: verdicts 240,
merge_verdicts 10, trial_selections 20, no field differences in any shared document and
no documents on either side only. Nothing has changed since October, so the dump is
current.

### Verifying before deleting

`verify_archive_before_delete.py` is the gate. Every file on F must exist on Z at the
same size and matching content: 767 files hashed in full, and the 9 files over 200 MB
compared by 64 sampled 1 MiB blocks plus first and last, read on both sides. A full read
of 293 GiB over the network would take hours and would not change the decision -- sampling
catches truncation, partial copies, a wrong file and a stale version, which are the
failures that actually occur, and the script says so rather than claiming more.

Result: **776 files, 293.0 GiB, everything present on Z and matching.** Z additionally
holds the 301.4 GiB `.rec`, which F no longer has, and the new review bundle.

So F holds nothing that Z does not. The deletion is safe to make; it is Gil's to call.

### 5bd.38a — The archived review tool now shows the verdicts it archives

Serving the bundled page off Z, Gil's reaction was immediate and correct: every unit
rendered un-reviewed, and his first thought was that the verdicts had not been saved.

They had. The cross-check, run entirely against Z: all **240** stored verdicts match
the `group` column of `cluster_info.tsv` on Z, zero differences, zero units missing --
and his 114 `good` calls are exactly the 114 goods in the dataset. The decisions were
made between 2026-09-17 15:07 and 2026-10-01 14:39, and the timestamps are in the file.

The fault was the page. `initDb` does `claude.use('db')`, and on a copy with no
database it returns early with the status text "local only (db unavailable)" and leaves
every verdict unset. The data was beside it the whole time in `verdicts/`.

`patch_review_tool_offline_verdicts.py` replaces that early return with a loader that
fetches the three exported collections and fills exactly the globals the snapshot
handlers fill (`verdicts`, `timeSel`/`timeSelDocs`, `mergeVerdicts`), then re-renders.
It refuses unless all three exports exist and parse, asserts its anchor appears exactly
once, and refuses to patch twice. The page is left read-only and says so in a banner,
because `setVerdict` already guards its write with `if(db)` -- so clicking a verdict on
the archived copy changes the screen and nothing else, which is a trap worth naming
rather than leaving silent.

Two lessons, both general:

1. **An archive that cannot display what it archives will be read as an empty archive.**
   The files were correct, verified and in the right place, and the thing still looked
   like data loss. For a bundle whose entire purpose is to outlive the tool that made
   it, "the data is there if you open the JSON" is not good enough.
2. **Serve the bundle, never double-click it.** `fetch()` of a relative path is blocked
   for `file://` pages, so opening the page from disk gives units but no verdicts and a
   dead merge panel -- indistinguishable from a broken archive. The README now says this
   and gives the one-line http.server command.

### 5bd.38b — F cleared of 20260916_110311

Deleted `F:\Gil\Shamir\20260916_110311.rec` on 2026-11-11 at Gil's instruction, after
he had opened the archived review tool from Z and confirmed his verdicts were there.

Gates passed before the delete, in order:

1. Full content verification earlier the same session -- 776 files, 293.0 GiB, 767
   hashed in full and the 9 files over 200 MB compared by sampled blocks on both
   sides. Everything present on Z and matching.
2. A metadata re-check immediately before deleting: same 776 files, same byte total,
   nothing on F modified that day, every file confirmed present on Z at matching size.
   Content was not re-read -- Trodes was recording and nothing had written to F since
   the full check, so a second 293 GiB read would have cost acquisition bandwidth to
   re-prove an unchanged fact.
3. Gil's own inspection of the archived review tool, served from Z.

280.9 GiB freed; F went from 1,113 to 1,394 GiB free. Z retains the 301.4 GiB `.rec`
that F never had, the full 293.1 GiB kilosort subtree, and the review bundle. Verdicts
re-read from Z after the delete: 240, of which 114 good, plus 10 merge decisions.

Worth noting for the batch: F is now effectively empty of Shamir sessions, and since
30 of 31 sessions already have an exported `.probe1.dat` on Z, the batch does not need
F at all. The old "copy the .rec down, re-export, upload" route is only needed for
`20261015_094934`.

### 5bd.39 — Is it the same neuron tomorrow? What the field does, and what we should save now

Gil asked the general question while the upload waits: how do you tell whether a unit
in session A is the same neuron as a unit in session B, is there anything we can do,
and how have other labs attacked it. Writing the answer down because one part of it is
time-sensitive -- it changes what the cluster job has to save, and the cheap moment to
save it is while the raw `.dat` is on scratch.

#### The honest framing

Identity cannot be proven. There is no measurement that distinguishes "the same cell"
from "a different cell that happens to sit 8 um away with a similar spike". So the
question is never "is this the same neuron" but "what is the false-match rate of my
decision rule, at what recall". Same discipline as the sorter comparison: a rule, a
positive control, a negative control, and recall *and* precision reported per tier.

The positive control is the part that makes this tractable, and it is the trick the
whole literature now rests on: **split a single session's spikes into two halves and
treat the halves as two recordings.** Every unit then has a partner that is the same
neuron by construction, and the distribution of the similarity score over those pairs
is the within-neuron distribution. Pairs of *different* units in the same session give
the between-neuron distribution. The threshold comes from those two distributions, not
from judgement, and the error rates fall out of the overlap.

#### What actually carries the signal

On a 384-site Neuropixels (ours: 4 columns, 16 um x-pitch, 20 um rows, 920-4740 um)
a neuron's **spatial waveform footprint** is close to a fingerprint. A unit lights up
10-20 sites; the shape of the amplitude decay across those sites, the centroid position
to a few um, and the waveform shape on each site together are far more specific than
anything available in the tetrode era. That is the signal. In order of usefulness:

1. Footprint position (centroid from the amplitude distribution) -- strongest single cue.
2. Amplitude decay profile across sites -- distinguishes two neurons at the same depth.
3. Waveform shape on the peak sites -- weakest alone, useful in combination.
4. Firing rate, ISI shape, autocorrelogram -- weak, and they move with brain state.

And the trap: **functional similarity must not be used for matching.** Matching on
tuning or PSTH similarity guarantees you find stable tuning, which is usually the
thing you wanted to measure. The field's convention is to match on the electrical
signature only, and then use functional agreement as *validation* of the matcher.
This matters for us specifically, because the sessions are a learning series.

#### Drift is the whole difficulty

The same neuron looks different across sessions mostly because the probe moved relative
to the brain, not because the neuron changed. So every working method estimates a
session-to-session shift (often a rigid vertical offset of tens of um, sometimes
depth-dependent) and applies it before comparing positions. Skip the registration and
the matcher fails in the honest direction -- it just stops matching anything.

Our own version of this hazard is bigger than drift: **the 820 um map change between
2026-10-23 and 2026-10-26.** Across that boundary we must know whether the probe was
physically advanced or only the channel map file was relabelled. Labels only -- undo
the 820 um and the sessions are comparable. Probe advanced -- they are different cells
and no matcher should be run across the boundary at all. Gil knows which; it needs to
be written down before anybody matches across that date.

#### How other labs have attacked it

Two families, plus a validation-only third.

**Family 1 -- sort separately, match afterwards.** The mainstream.

- Tetrode/array era: waveform-shape correlation plus ISI histogram similarity, with
  the key methodological move being to compare the across-day similarity distribution
  against a within-day one (Tolias et al. 2007, J Neurophysiol; Dickey et al. 2009;
  Fraser & Schwartz 2012; Eleryan et al. 2014 built a feature database over many days).
  Their ceiling was low because 4 channels carry almost no spatial information.
- Neuropixels 2.0 (Steinmetz et al. 2021, Science) tracked units over weeks using the
  waveform footprint plus firing-pattern similarity, and reported stability over months.
- **UnitMatch** (van Beest, Bimbard, Carandini et al., Nature Methods 2024) is the
  current reference. Post-hoc on the sorted output, uses only each unit's average
  waveform, naive Bayes classifier over footprint-derived features, gives a
  *probability* per cross-session pair, calibrated by exactly the split-half control
  above. It processes all sessions jointly, so a unit that vanishes for a session and
  comes back can still be tracked. Reported ~30% of units held for >100 days; day-to-day
  matching is far better than that number suggests.
- Earth Mover's Distance method (eLife reviewed preprint 92495) -- physical distance
  plus a waveform distance, applied to consecutive session pairs.
- **DANT** (bioRxiv 2025 / Patterns 2026) -- alternates density-based clustering in
  feature space with probe-motion estimation from the provisional matches, re-registers,
  repeats. Aimed at freely-moving work, which is us.
- The calcium-imaging field solved the analogous problem earlier and more rigorously
  (CellReg, Sheintuch et al. 2017): an explicit probabilistic registration model that
  returns a per-pair probability rather than a yes/no. Worth reading for the framing
  even though the measurement is unrelated.

**Family 2 -- sort the sessions jointly.** Concatenate two or more sessions and sort
them as one recording; identity is then given by construction, no matching needed.
Used by a number of labs and genuinely appealing, but: the drift correction in KS is
built for within-session drift, not for a gap of a day with the animal unplugged; a
bad concatenation silently splits or merges across the seam; and it would discard the
per-session curation Gil has already done. Not for us now, but it is the thing to
reach for if matching turns out to be hopeless across the longer gaps.

**Family 3 -- functional fingerprints as validation only.** Cross-correlograms with
other simultaneously recorded units, or tuning similarity. Good for checking a matcher,
never for building one.

#### What this means for our pipeline

Three concrete consequences.

**1. The cluster job must save raw split-half mean waveforms. This is the time-sensitive
part.** KS4 already writes `templates.npy` (44 MB here -- 472 units x 61 samples x 384
channels), which is a footprint per unit and is enough for a first-pass matcher. But KS
templates live in whitened, drift-corrected space and are fit from a shared basis, so
they are not the unit's true average waveform; UnitMatch and its relatives want raw
snippet averages, and they want them **split into two halves** so the within-neuron
calibration is available. Extracting those needs the raw `.dat`. Doing it on the cluster
while the file is already on scratch costs minutes and a few hundred MB; doing it later
means re-uploading ~250 GB per session. So it goes into the same job as Kilosort, for
every session in the batch, starting with the one about to run.

Size: 472 units x 2 halves x 82 samples x 384 channels x float32 is about 119 MB per
session, 3.7 GB for the whole rat. Trimmable by keeping only sites near the peak.

**2. A free feasibility test exists right now, with no upload.** Three sessions are
already sorted on Z -- 20260901, 20260911, 20260916 -- and 20260915 is the one going to
the cluster next. That gives a **one-day pair (15th/16th September)**, a 5-day pair, a
10-day pair and a 15-day pair, all on the same channel map, all from the same chronic
implant. `spike_positions.npy`, `amplitudes.npy`, `templates.npy` and
`whitening_mat_inv.npy` are all sitting on Z for the sorted ones, so a
position-plus-footprint matcher with split-half calibration can be built and scored
without reading a single raw file. If the one-day pair does not match cleanly, nothing
further is worth building.

**3. The payoff is the verdict burden, not just the science.** 240 hand verdicts per
session across 31 sessions is not a thing anyone is going to do. A calibrated matcher
turns it into "carry the verdict across, review what is new or uncertain" -- which also
means the matcher's precision directly sets how much wrong labelling gets propagated,
so the per-tier precision number is the one that matters, not recall.

One ordering constraint: **merges must be resolved before matching.** A merged pair is
one neuron with one footprint, and matching the pre-merge clusters separately would ask
the matcher to decide between two halves of the same cell.

#### 5bd.39a — the map question answered, and four things to settle before the batch sorts

Gil, 2026-11-11: **"it was only map changing."** The probe was never advanced. So the
820 um between map A and map B is a bookkeeping difference between two descriptions of
one unchanged geometry, and the entire 2026-09-01 .. 2026-10-28 series sits in a single
coordinate frame. Cross-session tracking is therefore possible over the whole two
months, not just within each map era. One of the two maps is simply wrong about where
the probe sits in the brain; which one is open, but it is a rigid offset applied to
everything and can be settled later without re-sorting anything.

He then asked what should be done before the sessions are sorted. Four things, and two
of them are confounds rather than improvements.

**(a) One pair of detection thresholds for the whole batch.** `run_kilosort4_session.py`
makes `Th_universal`/`Th_learned` required arguments with no default, precisely because
they have been per session so far: 9.0/8.0 on 20260916, 9.2/7.0 on 20260901 and
20260910. That was the right call for one-off sorts and is the wrong call for a series.
Detection thresholds set how many units are found and how big their footprint
amplitudes are, so letting them vary session to session puts a non-biological trend
straight into a learning series, and it perturbs the very features a matcher scores on.
The cluster makes the fix cheap: a 3-point sweep on two sessions is six jobs, parallel
and free, and it measures how much the yield actually moves. Needs Gil's input on how
the current values were chosen before anything is fixed.

**(b) One Kilosort version.** The RIS container is pinned to **4.0.24**; the three
sessions already sorted on Z (20260901, 20260911, 20260916) ran **4.1.7**. The size of
the difference is already measured, which is the useful part: cluster 4.0.24 against
local 4.1.7 on the same session, same thresholds, gave 99.8% recall and 99.2% precision
on the matched units. So the version split is small enough not to be alarming, but the
batch should still be internally uniform, with the legacy sorts either redone on the
cluster or kept plainly separate rather than mixed in.

**(c) Every session's map expressed in one frame — but still derived per session.**
Tempting conclusion from "the probe never moved" would be to feed one shared probe JSON
to all 31 sessions. That would be wrong: the row order of `probe1.dat` differs between
the two eras, because the ntrode list is rotated by 82, so a shared file would
mis-assign channels. The correct operation is per session derivation from its own
`channelmap_probe1.dat` (which `make_probe_maps_batch.py` already does) followed by a
re-expression of map-B sessions into map A's frame: +820 um in dv, and undo the 16 um
swap of the staggered ML columns. Check afterwards that the SET of (xc, yc) pairs is
identical across all 31; if it is not, stop. A uniform offset preserves every
inter-channel distance, so this changes the reported depths and not the clustering --
but doing it before the sorts means the coordinates are right in the files forever,
with no post-hoc correction for anyone to forget.

**(d) Raw split-half mean waveforms, written by the sort job.** Repeated from 5bd.39
because it is the only item whose cost is irreversible: on the cluster, while the `.dat`
is on scratch, it is minutes and about 120 MB per session; afterwards it is a 250 GB
re-upload per session. Save the mean waveform over all 384 channels for each half of
each unit's spikes, plus the per-half spike count and centroid, and keep `ops.npy` for
its drift estimate.

One thing deliberately NOT done: re-sorting 20260916 in place. Its 240 verdicts are
keyed to that sort's cluster ids. A cluster re-sort of it goes in its own directory and
the verdicts are carried over with `compare_two_sorts.py`, which is the same machinery
that produced the agreement numbers in (b).

#### 5bd.39b — thresholds settled: Kilosort4 stock defaults, 9.0/8.0, for all 31 sessions

Gil, 2026-11-11: "use the default settings of kilosor, which is i think 9.0/8.0". He is
right, and it was worth checking rather than agreeing: `kilosort.parameters`
`DEFAULT_SETTINGS` in 4.0.24 gives `Th_universal = 9`, `Th_learned = 8` exactly. So the
batch runs on stock detection thresholds, which is the best possible answer for a
series -- uniform by construction, and nobody has to defend a hand-tuned number later.

Consequences worth noting:

- **20260916 already used 9.0/8.0**, so the curated session and its 240 verdicts are
  already on the batch setting. That is lucky rather than planned.
- **20260901 and 20260910 used 9.2/7.0**, which are NOT the defaults. Those two sorts
  are off-batch and should be redone on the cluster before they are used in any
  cross-session comparison. Until then, the only session on Z that is both sorted and
  batch-consistent is 20260916.
- The thresholds stay required arguments in `run_kilosort4_session.py`; the submit
  script is where 9.0/8.0 now gets passed for every session. A default in the runner
  would hide exactly the kind of drift we just removed.
- No threshold sweep will be run. It was offered as a way to measure the sensitivity
  and is unnecessary once the answer is "stock defaults for everything".

Gil also said items (c) and (d) of 5bd.39a were not clear. Both are code changes on our
side with nothing for him to do, which is the part that had not been said plainly. The
lesson is the same one as the archived review tool: a correct explanation that leaves
the reader unsure whether they are being asked for something has not done its job.

#### 5bd.39c — where the verdicts fit, and Gil's concatenation idea becomes the ground truth

Two questions from Gil on 2026-11-11, both of which change the plan.

**Where do the verdicts fit?** UnitMatch runs strictly *after* Kilosort, on the sorted
output. It never re-sorts, never touches spike times, never renumbers clusters. Its
input is each unit's average waveform plus the channel positions; its output is a table
of cross-session pairs with a probability. So the verdicts are not an input and are not
at risk. What the matcher produces is a *mapping between cluster ids across sessions*,
and propagating the verdicts along that mapping is our own step, not UnitMatch's.

Three design points for that step:

1. Apply merges first. A merged pair is one neuron with one footprint
   ([[project-merge-tool-semantics]]).
2. Restrict matching to units worth tracking -- Gil's good and mua -- rather than all
   472 clusters. Fewer candidates means fewer false matches, and the noise units are
   not worth following.
3. **A disagreement is data, not an error.** If a unit is good on day 1 and its match
   looks like noise on day 2, that is a fact about the recording, so the propagated
   label must stay distinguishable from a hand-made one. Verdicts carry provenance:
   hand-made, or inherited from session X at probability p.

**Concatenation.** Gil asked whether people don't concatenate part of the previous and
next sessions with the current one before sorting. They do -- it is the joint-sorting
family from 5bd.39, and the partial-overlap variant he describes is a real technique for
giving the sorter a bridge so one cell keeps one id across the seam. Assessed honestly,
it is a poor fit as our *production* route:

- The seam is a genuine discontinuity -- the animal was unplugged and the probe settles
  differently -- and KS4 estimates drift over the whole file (`nblocks=1`), so a step
  change at the seam gets absorbed as drift and can smear templates on both sides.
- One compromise template is fit to both days, so a cell whose waveform genuinely
  changed is isolated *worse* in each session than if that session were sorted alone.
- Every cluster id changes, which detaches the 240 verdicts.
- Overlapping pairs means sorting most of the data twice: compute is free on the
  cluster, storage is not.

But it is an excellent *check*, and that is the better use of the idea. A joint sort of
two sessions yields units that span the seam **by construction**, which is a same-neuron
label we did not have to assume. So:

> Sort 20260915 + 20260916 concatenated, once, as a second opinion. Units spanning the
> seam give the matcher's recall; pairs the matcher claims that the joint sort keeps
> separate are candidate false positives, giving precision.

Two independent methods on one pair, from Gil's own suggestion, and nearly free because
both sessions are staged anyway. It is a second opinion rather than truth -- a joint
sort can also merge two different cells across the seam -- so the numbers it gives are
an estimate with its own error, not a gold standard. Said plainly now so the figure is
not over-read later.

Also: the localhost:8765 server for the archived review tool was stopped, along with the
leftover verdict-comparison task. Bringing the review tool back is one command, recorded
in the bundle's README.

### 5bd.40 — 20260915 sorted on the cluster and reviewed end-to-end, and the waveforms rescued off scratch before it was cleaned

The first session taken through the whole pipeline with Kilosort running on the RIS
cluster and the post-sort half running here. 447 units, published review tool,
everything on Z. Details worth keeping, in the order they bit.

**The orphaned upload.** The scp that had been running since 15:10 was not dead. Killing
the background task killed only the PowerShell wrapper; `scp.exe` kept going as an orphan
and had crawled to 4 MB/s — about four more hours for the remaining 61 GB. scp has no
resume, so the obvious fix throws away 23 GB. Instead: kill the orphan, truncate the
destination back to a clean 1 MiB boundary 16 MiB below where it stopped (in case the kill
left a torn block), and append the remainder with `dd bs=1M skip=N | ssh compute1 "cat >> dst"`.
71 MB/s, 14 minutes. The pipeline was proved on a 10 MiB chunk first — md5 equal on both
ends — before 57 GiB went through it. Verified afterwards three ways: exact size, the size
divides evenly into 110,207,963 samples of 384 channels, and md5 on three 10 MiB chunks
including one straddling the splice.

Lesson for the next upload: `[killed]` in a task log says the wrapper died, not that the
work stopped. Check for the real process.

**The sort.** LSF 741761, exec-357, container `zurgil/kilosort4:4.0.24-cu121`, one GPU,
41m58s for 1.02 h of data — realtime factor 0.686. Stock thresholds 9.0/8.0 per the
batch-wide decision. 447 units, 222 with good refractory periods. The job's own log printed
`yc 920.0..4740.0` before starting, which is the check that earns its keep: that is map A,
the correct frame for September, and map B would have read 1740–5560.

**What was nearly thrown away.** Scratch still held the 290 GB raw copy of 20260916 from the
night before, and that session had never had split-half waveforms extracted because the
extractor was only written today. Cleaning scratch as routine housekeeping would have turned
minutes of compute into a 290 GB re-upload, on the single worst session to lose it on: the
curated one, the anchor for everything cross-session. So `ris_submit_waveforms.sh` now runs
the extractor alone against a sort that already exists, taking the Kilosort directory as an
explicit argument rather than deriving it from the session id.

That argument is not pedantry. 20260916 has two sorts: the curated 4.1.7 one from September
that Gil's verdicts are keyed to, and the cluster's own 4.0.24 one from 2026-11-10. They
differ by about 118,000 spikes, so their unit numbering does not correspond, and pointing the
extractor at the wrong one yields a file that describes different units without saying so.
Three checks tied the September sort to the file on scratch: same byte count
(290,370,270,720), same gain in the `.rec` header, and its last spike lands **13 samples from
the end of the file**. The last one is what actually settles it — a sort run against a
different export would not sit flush against this file's end.

Result: 20260916 has waveforms for all 472 units keyed to the curated numbering, and
20260915 has them for all 447, both with no unit below the spike floor. The one-day pair the
matcher needs now exists on both sides.

**Why the local half ran from F and not Z.** The first pipeline batch, run straight off Z,
took 8 minutes — 45 batches would be six hours. It is not compute: the recovery sweep scans
thousands of small windows per unit (7,741 candidates for a single unit in batch 0), and
small random reads over SMB are latency-bound. The same batch from F took **1m15s**, 6.4x
faster. Staging the `.kilosort` folder plus DIO and timestampoffset is 85 GB and 14 minutes,
which pays for itself in the first three batches.

**Parallelism, and the race that makes it non-trivial.** `run_pipeline` writes
`agent_report.json` and `cluster_<label>.tsv` at FIXED names, and `run_pipeline_batch_session.py`
archives them immediately afterwards. Two batches sharing a session root therefore race, and
the loser files the other batch's units under its own batch number — silently, no error, a
report that looks complete and describes the wrong units. The fix is one session root per
worker. The 79 GB raw file is hardlinked rather than copied (same NTFS volume, read-only to
the pipeline, verified: link count 5 on one inode), so four roots cost 2.1 GB each instead of
79 GB. Consolidation needed no change at all because it globs `batch_*_report.json`, so the
per-worker archives gather back together by copying them into one place.

Four workers, 60–90 s per batch each — the same as one batch alone, so the speedup is real
and not divided throughput. 43 batches in about 11 minutes, 29 GB RAM still free.

**The heredoc bit again.** The first launch of all four workers failed instantly. The runner
had been written through a bash heredoc, which halved the doubled backslashes, turning
`\\${SESSION}` into `\${SESSION}` — an *escaped dollar* — so every batch was handed a root
path containing the literal text `${SESSION}`. This is the hazard already recorded in memory,
and it still got me, because the script looked right in the editor. Two changes: the runner is
written with the Write tool and uses forward slashes throughout (Windows Python accepts them),
and per-batch stdout goes to a log file instead of `/dev/null`. The `/dev/null` was the worse
mistake — it hid the real error behind a bare exit code and cost a round trip to diagnose
something the first line of output would have shown.

**Results.** 447 units, consolidated 447 records with 0 duplicates, 434 tier2 / 10 tier3 /
3 tier1. The `cluster_*.tsv` files carry 437 rows rather than 447 because the 10 tier3 units
get per-unit records and no score columns; that matched 20260911 (8 of 466 the same way)
before it was accepted as normal, which is the only reason it did not look like data loss.

bombcell raw: 18 GOOD / 105 MUA / 107 NOISE / 217 NON-SOMA. After the project's calibrated
thresholds: 72 GOOD, 157 GOOD+NON-SOMA, 85 MUA, 126 MUA+NON-SOMA, 7 NOISE. The raw 18 is not
a number to quote anywhere; it is what uncalibrated bombcell says about striatal data.

Session: 3691 s, 314 trials from DIO code 1, no dropped-packet gaps anywhere between first
and last sample, median unit mean rate 2.62 Hz. 150 merge candidates within 60 µm.

**Published** at https://claude.ai/artifact/QreYaMjGtujiE17m2k9zXf with
`capabilities = {"db": {}, "downloads": true}` and `data/merge_candidates.json` alongside —
both confirmed present on the live version after publishing, because a page published without
the db capability looks fine and cannot save a single verdict.

**On Z.** The sort, the waveforms, the full `agent_processed` and `kilosort_original`, and a
`review/` bundle in the same layout as 20260916's: the page, its data file, and every input
needed to rebuild it. `verdicts/` is deliberately empty and the README says so in its first
line, because this session has not been reviewed yet — an archive that looks finished and
contains no decisions is worse than one that admits what is missing.

**Still open.** Nothing has been deleted from RIS. Scratch holds 330 GB across the two
sessions and is 87% full; both sessions' waveforms are now safely on storage2 and Z, so the
raw copies are finally free to go — but that is Gil's call to make, and the list goes to him
first. The `kilosort4/` outputs on storage2 stay regardless: 6.5 GB a session, and they are
the result.

### 5bd.41 — Clearing RIS, and the one result that was only there (2026-11-11)

Gil asked whether the results were all on Z and whether there was any reason to keep
files on RIS. Answering it properly turned up one thing that was not on Z at all, and
a transfer measurement that changes how every future fetch should be done.

**The one result that existed only on RIS.** `storage2/.../kilosort4/20260916_110311`
was the cluster's OWN sort of that session — Kilosort 4.0.24, 2026-11-10/11, 449 units,
25,348,358 spikes — and it had never been fetched. Z holds the curated 4.1.7 sort
instead, 472 units and 25,229,992 spikes, which is the one Gil's verdicts are keyed to.
So "all the results are on Z" was true of everything except the file that would have
been lost by acting on it. Checking before deleting is the whole reason the question
was worth answering slowly.

It is now at
`...20260916_110311.kilosort/kilosort4_cluster_4.0.24_NOT_CURATED/`, under Gil's
instruction to make very clear it is a spare. Two files do that: the folder's own
`README_FIRST_SPARE_SORT_DO_NOT_USE_BY_DEFAULT.txt`, and `WHICH_SORT_IS_WHICH.txt` one
level up, where someone browsing the session will actually meet it. Both lead with the
same point: the sort of record is `kilosort4/`, it carries hand-made per-unit verdicts
in `cluster_info.tsv`, and those verdicts are the most expensive object in the folder
because they are hours of human curation that no amount of GPU reproduces. And both
spell out the trap — the two sets of unit ids do not correspond, 118,366 spikes and 23
units apart, so unit 137 in one is not unit 137 in the other, and anything carried
across by id is silently wrong while looking entirely plausible. Why keep it at all:
the other 30 sessions are being sorted with 4.0.24, so this session's sort of record is
the odd one out of the 31, and if a batch-wide analysis ever needs one Kilosort version
throughout, this is the copy that belongs in it. Re-making it costs a 290 GB upload and
2.7 h of GPU, which is why it is worth 6.5 GB of archive.

**scp is the slow path on this link, and it is not close.** The fetch started as `scp`
and managed **1.5 MB/s** — 1.1 GB in twelve minutes, which is an hour for 6.5 GB. Two
measured tests after killing it: `ssh compute1 "dd if=... bs=1M count=1024" > /dev/null`
moved 1 GiB in **6.0 s** (179 MB/s, pure link), and the same into Z took **15.0 s**
(71 MB/s, link plus SMB write). A streaming
`ssh compute1 'tar cf - .' | tar xf -` of the whole directory averaged 11.5 MB/s —
dragged down by per-file SMB writes, but still 8x scp. The upload of the next session,
`dd | ssh "cat >"`, is running at **98 MB/s**. The rule from this: scp has no business
on this link in either direction. It has now cost an hour today and four hours
yesterday.

**Nothing was deleted until it had been checked file by file.** For the spare sort:
25/25 files identical in size to the cluster copy, md5 equal on `spike_times.npy`,
`spike_clusters.npy`, `templates.npy` and `run_provenance.json`, and the last spike
lands 13 samples from the end of the raw file — the same signature that tied the
curated sort to the scratch copy yesterday. For 20260915_103710: all 28 files present
on Z at identical sizes, md5 equal on four sort files and both waveform files. For
`ks_curated_z` and `waveforms_curated_z`: md5 equal on all five files against Z.

Then deleted, all of it ours and nobody else's (`stat -c %U` checked `zur` on every
path before `rm`, and `/scratch1/fs1/akepecs/zur/` contained only our two
directories): **352 GB of scratch** (80 GB + 272 GB of raw uploads) and **9.1 GB of
storage2** (both kilosort4 outputs, ks_curated_z, waveforms_curated_z). RIS is down to
**1.8 MB** — `code/`, `build/` and `logs/`. `code/` has to stay: every `bsub` points
the container at a script inside it, so clearing it would break the next sort rather
than save space.

**A bug in the registry, from writing it through a shell.** Five strings in the
20260915_103710 entry read `Shamir_1\x8260915_103710.rec`. The mechanism: `\20260915`
was written through something that interprets escapes, `\202` was consumed as an octal
escape producing byte 0x82, and `\review` lost its `\r` to a carriage return the same
way. Repaired by putting the consumed characters back — and then, because a repaired
path that is merely plausible is worth nothing, each of the five was checked against
the disk with `os.path.exists`. All five exist.

This is the fourth instance of the same hazard, and nothing about it is new: the
project's own note already says that a **quoted** heredoc halves doubled backslashes
here and already records `\202` being read back as an octal escape, which is exactly
what damaged `WORKFLOW.md` once before. The repair attempt then walked into it a fifth
time — `"\\202"` written through `<<'PYEOF'` arrived in the file as `"\202"`, so the
first pass reported zero matches and looked like a logic error rather than a quoting
one, until `cat -A` showed the file. The rule was already right and was simply not
followed: a file containing backslashes gets written with the Write tool, never through
a heredoc of any kind, quoted or not.

**Next session chosen and verified: 20260915_084812.** It is the same-day sibling of
the session just sorted — 08:48 against 10:37, both about an hour, back to back — which
makes it the tightest cross-session pair in the whole batch: wider than a within-session
split-half and narrower than the 1-day 15/16 pair, which is exactly the calibration
point the matcher has nothing for. It is also the cheapest remaining upload, 98 GB
against the 250-397 GB that is typical for the other 28.

The pre-upload checks Gil asked to be standing: `probe1.dat` is 98,188,818,432 bytes =
127,850,024 samples of 384 int16 channels = 71.0 min, and divides evenly by 768;
`timestamps.dat` present; DIO present with all six Din channels and Din6 carrying
7,137 bytes, so the Bpod sync word is complete; gain read from this session's own `.rec`
header as 0.018311105685598315, matching the sibling rather than assumed from the
default. The channel map is the strongest version of the comparison he asked for — the
staged map is **byte-identical** to the map the 20260915_103710 sort actually used, so
it is map A (yc 920-4740, xc 2426-2474) by identity and not by inference. Map B would
have read yc 1740-5560 and sorted the whole session 820 um deep.

### 5bd.42 — 2026-09-15 is one session, not two, and the data says what interrupted it (2026-11-11)

Gil asked whether 20260915_084812 was another session from the same day, and said that
when that happens it is usually because something forced him to stop Trodes and start
again — so they should probably be sorted together. He also asked whether Bpod had one
session that day or two. Both questions have measured answers, and the second one turned
out to point straight at the cause.

**The two recordings.** Part 1 `20260915_084812` runs 08:48:12 to 09:59:14 (71.0 min,
127,850,024 samples). Part 2 `20260915_103710` runs 10:37:10 to 11:38:24 (61.2 min,
110,207,963 samples). The gap between them is **37 min 56 s**.

**Bpod has two sessions that day, one per recording.**
`Shamir01_Dual2AFC_nat_Sep15_2026_Session1.mat` was written at 09:59 and `_Session2.mat`
at 11:39, matching the two recordings' end times to the minute. 2026-09-15 is the ONLY
date in `Z:\Gil\Shamir_1\bpod` with a Session2, which independently says this is the only
interrupted day in the Shamir batch.

**The probe did not move, so joining is sound.** Spike-band (300-6000 Hz, after CAR)
per-channel RMS fingerprints across all 384 channels, part 1 at 35 min against part 2 at
30 min: **r = 0.9157**. The within-recording controls are 0.9275 (part 1, 1 min vs 35 min)
and 0.8328 (part 2, start vs end), and the best depth lag within part 2 is 0 rows. So
across the 38-minute gap the channels see the same thing as well as they do inside a
single recording. The objections recorded in 5bd.39 against joint sorting were written for
cross-day concatenation, where the animal is unplugged and the probe re-settles; none of
them survive a 38-minute interruption within one day.

The first attempt at this measurement was wrong and worth recording as a method note. Run
on unfiltered data it gave r = 0.0197 across the gap, which reads as "the probe moved" —
but the within-session control was also 0.05, so the measure itself was broken.
Unfiltered RMS is dominated by LFP and slow drift, which changes with brain state between
recordings and carries no channel fingerprint. Band-pass to the spike band first, and
always compute the within-recording control before interpreting the across-gap number:
without it an r is uninterpretable in either direction.

**What forced the restart, found in the data.** At **t = 4150 s (69.17 min)** into part 1,
82 of the 384 channels jump above three times the healthy spike-band noise level and stay
there for the remaining 112 s. Narrowed to the second: r against a healthy reference is
0.92 at 4149 s and 0.11 at 4150 s. The affected channels *alternate* — 10, 12, 14 … 32,
then 59, 61, 63 … 105, then 154, 156, 158 … — spanning yc 1300-4640 µm, with no
saturation and a normal total variance. Alternating channels over most of the probe is a
connector, ground or reference signature, not biology. Part 2 at 30 min has **zero** such
channels and percentiles identical to part 1's healthy stretch (5/50/95 of 708/826/1004
against 709/822/1041), so whatever was done in those 38 minutes fixed it.

So the join trims part 1 at **4145 s**, five seconds before the fault, dropping 116.7 s.
Trimming rather than keeping matters because Kilosort fits whitening and drift over the
whole file: 112 s of 82 bad channels would have degraded the other 2.16 h rather than only
itself.

**What Gil asked for instead, and why concatenation beats it.** He asked whether part 2
could be run assuming part 1's units already exist, so it continues with the same units.
That is a real technique — SpikeInterface's `sortingcomponents.matching` (`wobble`,
`circus-omp-svd`) takes fixed templates and finds their spikes in new data. It is the
worse choice here for one reason: frozen templates make part 2 blind to any unit that was
not in part 1, and after a connector fault and 38 minutes some units will have appeared or
changed. A joint sort learns templates over both halves, lets the drift correction see the
seam, and still delivers the thing he actually wanted — one unit id spanning both
recordings. It also keeps the session comparable to the other 29, which all go through
plain KS 4.0.24. Template matching stays available as a cross-check.

**The joint file.** 180,140,515,584 bytes = 234,557,963 samples = **2.172 h**, seam at
sample **124,350,000**. Built on scratch with `head -c` for the trimmed part 1 and
`dd | ssh "cat >>"` for part 2. The append dropped once — `Connection reset by peer` after
39.4 GB, no orphan this time — and was resumed by truncating to a clean 1 MiB boundary
32 MiB below the drop and continuing with `dd skip=37560`, at 93.6 MB/s, with ssh
keepalives added so a quiet moment cannot kill it again. Verified by exact total size,
divisibility by 768, and 16 MiB md5 against the sources on Z at five places: part 1 start,
the last block before the seam, the first block after it, a block straddling the resume
splice, and the tail. All five equal.

Provenance is in `outputs/concat_20260915_084812_103710.json`, which self-checks: the two
parts' sample and byte counts sum to the total, and the total divides by 768. It also
records the rule for mapping a joint spike back to its own part and clock, which is the
thing that will be needed weeks from now.

**Submitted** as LSF 759350, RUN on compute1-exec-360, stock 9.0/8.0, and the job's own log
printed `yc 920.0..4740.0` — map A, correct for September.

~~A side benefit worth noticing: the waveform extractor splits each unit's spikes into
two halves by time, and for this session the halves fall either side of the seam. So a
unit's two half-waveforms are a same-neuron pair across a 38-minute gap **by
construction** — exactly the calibration point between a within-session split-half and
the 1-day 15/16 pair that the matcher had nothing for.~~

**CORRECTED 2026-11-12, see 5bd.43.** The claim above is wrong and was checked rather
than left standing. The extractor splits each unit's spikes into two EQUAL-COUNT halves
(`np.array_split`), so the cut is at that unit's own median spike time, which has nothing
to do with the seam. Measured on the finished sort: median split point 3871.6 s against a
seam at 4145.0 s, the split falls inside part 1 for 296 of 465 units (64 %), and **zero**
units have half 1 entirely in part 1 with half 2 entirely in part 2 — for a typical unit
half 2 is the last 7 % of part 1 plus all of part 2. The pair is still worth having, and
is now obtainable rather than imagined: `extract_unit_waveforms.py` gained
`--split-at-sample`, so re-running it with the seam as a fixed cut gives
before-the-break against after-the-break per unit.

**Gil's standing rule, now mechanical.** He asked that from now on every date with
multiple recordings be examined first, and noted there might be more than two on a day.
`scripts/find_multi_recording_days.py` sweeps every rat on Z and reports each such date
with durations, the real-time gap and that date's Bpod files, plus a WORKFLOW.md STAGE 0
section with the three measurements that settle it. The sweep found **four** dates:
Shamir_1 2026-09-15 (now joined), and **Begin_1 2026-08-03 (97 min gap), 2026-08-13
(86 min) and 2026-08-17 (4 min)**. The Begin_1 dates are unexamined and that rat is not in
the current batch; two of those durations are estimated from `.rec` size because the
recordings were never exported, and Begin_1's `.rec`-to-`.probe1.dat` ratio ranges
1.1146-2.1507 across its exported sessions, so those two gaps are triage numbers and not
measurements. The 4-minute one is short under any ratio in that range and is almost
certainly a crash and restart.

One correction to our own bookkeeping that falls out of this: Shamir_1 has 31 `.rec`
folders over **30 distinct dates**. The batch is 30 sessions recorded as 31 files, and
everything that says "31 sessions" should say 30.

**Open, flagged by Gil and not solved here.** The sync stage was written for one recording
per session: `scripts/sync_session.py` and `sync_pipeline.process_session` take one
`--rec` and one `--bpod_file`. The shape of the fix, recorded so it is not re-derived —
unit ids are global, trial times are per part. Split the joint `spike_times` at the seam,
then run the existing per-recording sync twice, once per `.rec` with its own DIO and its
own Bpod file, carrying the same unit numbering into both. The result is two trial tables
sharing one unit space, kept separate with a `part` column or concatenated with trials
renumbered and the 37m56s gap recorded rather than closed. What must not happen is either
half being synced against the wrong Bpod file, or anything interpolating across the gap.

### 5bd.43 — The joint sort came back, and 462 of 465 units span both recordings (2026-11-12)

LSF 759350 finished in 1.595 h wall, 0.734 x realtime, inside the 0.69-0.77 band the
two earlier cluster sorts measured. **465 units (216 good / 249 mua), 16,681,732 spikes
over 2.172 h = 2134 sp/s.** Kilosort 4.0.24, stock Th_universal=9 / Th_learned=8,
do_CAR, map A (`yc 920.0..4740.0`, byte-identical to the map the standalone
20260915_103710 sort used). The split-half waveform extractor ran in the same job:
108.7 MB, all 465 units, microvolts; one unit (433) has fewer than 20 spikes in a half
and was written with its true counts, 17/17.

**Fetched without scp, as the rule now says.** Everything but `pc_features.npy` by
streaming tar (58 s), then `pc_features.npy` (4.0 GB) by `dd` over ssh (56 s). All 28
files present at identical names and sizes; md5 equal on `spike_times`,
`spike_clusters`, `amplitudes`, `templates`, `pc_features`, `cluster_KSLabel` and
`unit_waveforms.npz`. The 15,360-byte `du` difference is the remote directory inodes,
not data — worth noting because it looks like a mismatch and is not.

**The sort behaves as one recording.** `spike_times` is monotonic, min 1, max
234,557,910 — 53 samples from the end of the file, so nothing was lost off the tail.
Part 1 (below sample 124,350,000) holds 9,283,188 spikes = 2240 sp/s; part 2 holds
7,398,544 = 2014 sp/s. Population rate in 10 s bins either side of the seam: 2215 sp/s
in part 1's last bin, 2056 sp/s in part 2's first, against a bin-to-bin scatter of
2056-2456 sp/s over the surrounding two minutes. No step at the seam.

**And the thing the join was for, measured: 462 of the 465 units fire in BOTH
recordings — 99.4 %, including 214 of the 216 good units.** Only unit 73 is
part-1-only; only 151 and 346 are part-2-only. Over those 462 units the median
log2(rate in part 2 / rate in part 1) is -0.03, with the 10th and 90th percentiles at
-1.50 and +1.30 and 72 % inside a factor of two. So the units are not merely present on
both sides of the gap, they fire at the same rate on both sides.

**What joining bought, stated against the alternative.** The standalone sort of part 2
alone found 447 units (222 good / 225 mua), 7,900,062 spikes over 1.02 h. The joint sort
finds 465 over 2.172 h, 462 of them spanning both halves. Joining did NOT inflate the
unit count — it is the same population watched for twice as long, which is what should
happen and is itself more evidence the probe did not move. Had we sorted the halves
apart we would have had ~900 unit ids for ~465 neurons, each neuron's spikes split
across two files, and no way to know which id in one file was which in the other.
Part 2's spike count inside the joint sort is 6.3 % below its standalone count, which is
ordinary between two sorts that learn their templates independently.

**Bookkeeping on Z, since a joint session has no natural home.** The results went to
`Z:\Gil\Shamir_1\20260915_084812_103710_concat.rec\20260915_084812_103710_concat.kilosort\`
— deliberately the same shape as a real session folder, so every `--rec`-scoped script
works on it unchanged. It is not a Trodes folder and there is no `.rec` file inside, so
a script that needs one fails loudly, which is the right behaviour until the sync
problem is solved. Three documents were written so nobody meets these folders without
knowing which sort is the record: a `README_THIS_IS_A_JOINT_SESSION.txt` in the concat
folder (the seam rule, the trim, what is in here, what is still open), and a
`WHICH_SORT_IS_WHICH.txt` in each part — part 1's saying it must never be sorted alone
and recording the 116.7 s trim, part 2's saying its standalone sort is SUPERSEDED and
that the two sets of unit ids do not correspond. The 180 GB joint raw file is not kept
on Z: it is reproducible exactly from the two parts with two commands, which the README
gives.

**The local raw file was rebuilt, not downloaded.** The post-KS pipeline memory-maps the
raw file, so it has to exist locally. Rather than pull 180 GB off the cluster, part 1's
first 95,500,800,000 bytes were read off Z and part 2's already-staged local copy
appended — after spot-checking that local copy md5-equal to Z's at five 16 MiB offsets
first, which is the check that would have caught a stale or torn staging copy. The four
parallel worker roots then get the result by **hardlink**, not copy; that is how
`par\w1..w4` have shared one 84 GB raw file all along (link count 5 on the file), and it
is the only reason four workers fit on F at all.

**Two scripts stopped hardcoding a session id.** `setup_pipeline_workers.ps1` and
`run_pipeline_workers.sh` both had `SESSION=20260915_103710` baked in. That is the kind
of default that quietly runs one session's whole pipeline against another session's raw
file, so the session id is now a required parameter with no default on both. The setup
script also now refuses a raw file whose size is not a multiple of 768 (2 bytes x 384
channels) — the same guard `ris_submit_session.sh` has, because a truncated raw file
sorts successfully and is quietly wrong — and treats `timestamps.dat` as optional,
since a joint session has one per part and none for the whole.

**The timeline step needed real new code, and the reason is a clock.** The review tool's
firing-rate panel reads `session_timeline_<sid>.json`, which the plain builder writes on
the absolute Trodes clock via `timestamps.dat`. A joint session has one such file per
part and none for the whole, because the sample counter restarts when Trodes restarts —
and the two parts' absolute seconds *overlap*, so "absolute Trodes seconds" is not a
well-defined axis here. `scripts/build_session_timeline_data_concat.py` reads each
part's own `timestamps.dat` and `.DIO`, converts both to seconds elapsed within that
part, lays the parts end to end, and declares `clock: "concat_elapsed_seconds"`. It
imports the plain builder's payload-offset, gap-finding and timestamp-map functions
rather than copying them, and it takes the seam and the part list from the concatenation
provenance JSON only — never from the folder name.

Two display decisions worth recording because both alternatives mislead exactly where
someone would look hardest. The 37m56s real gap is **not** drawn as empty bins: 38
minutes of blank would eat a third of the panel and read as "the cell went silent"
rather than "nothing was recorded", so it is reported in the JSON and the seam is put on
an exact bin boundary instead. And each non-final part contributes only whole bins —
part 1 spans 4145.0 s, which is 207 bins of 20 s plus 5 s left over, and those 5 s
(~11 k of 16.7 M spikes, 0.07 %) are left out of the display and counted in the output.
A short 208th bin would draw a 4x dip right at the seam; folding its spikes into bin 207
would inflate that bin by a quarter.

**The clock name is load-bearing, and is now checked rather than assumed.** This is the
part that would have bitten us weeks later. The review tool stamps its own axis
(`ephys_t0_s`/`ephys_t1_s`) onto every per-unit kept-time-range selection it exports,
and `write_unit_trial_selection.py` wrote those ranges out as
`clock: trodes_absolute_seconds` for `sync_pipeline` to apply with
`inclusion_rule: trial_start_inside_range`. Hand it ranges measured on
`concat_elapsed_seconds` under that label and every trial-inclusion decision for the
session goes to the wrong trials — silently, because all the numbers stay plausible.
So `write_unit_trial_selection.py` now reads the clock out of the session's own timeline
file instead of assuming it, carries that file's `clock_mapping` into the output, and
refuses to write a selection whose clock `sync_pipeline` cannot apply. `--force_clock`
writes it anyway with the true clock name stamped, so the hand work is kept rather than
lost and `sync_pipeline`'s own check — which already refuses any clock but
`trodes_absolute_seconds`, and which was not modified — rejects it instead of
misapplying it. It also now refuses to run with no timeline file at all, since then
there is no record of what the ranges were drawn against; `--no_timeline_check` is the
escape hatch. Checked against both existing sessions (20260915_103710, 20260916_110311):
both still read `trodes_absolute_seconds` and still pass.

Nothing in `sync_pipeline` was touched. The split-at-the-seam work Gil flagged is still
open and still deferred until after the verdicts; what changed is that the deferral is
now enforced by a check instead of remembered by whoever reads the registry.

**A claim of mine from yesterday was wrong, and the check is the entry.** I wrote in
5bd.42, and told Gil, that the waveform extractor's two halves fall either side of the
seam for this session, so every unit would be a same-neuron pair across the 38-minute gap
by construction. It does not work that way. The extractor splits each unit's spikes into
two EQUAL-COUNT halves, so the cut is at that unit's own median spike time, not at the
seam. On the finished sort: median split point 3871.6 s against a seam at 4145.0 s, the
split falls inside part 1 for 296 of the 465 units (64 %), and **zero** units have half 1
entirely in part 1 with half 2 entirely in part 2. For a typical unit, half 2 is the last
7 % of part 1 plus all of part 2. So `waveforms/unit_waveforms.npz` from this job is an
ordinary within-session split-half pair and must be read as one. 5bd.42's paragraph is
struck through in place, the registry carries the correction, and the README on Z now
leads with it, because "the halves straddle the seam" is exactly the kind of thing that
would be believed six months from now without being re-derived.

The pair is still worth having — it is the calibration point between a within-session
split-half and the 1-day 15/16 pair the matcher has nothing for — and it is now one flag
away instead of imagined. `extract_unit_waveforms.py` gained `--split-at-sample N`, which
cuts at a fixed sample index instead of each unit's median, so `--split-at-sample
124350000` against the joint raw file gives before-the-break against after-the-break
averages per unit. The costs are recorded in the flag's own help and in the output
metadata: the halves are deliberately unequal (55/45 at this seam), and a unit that fell
silent on one side gets an empty half, reported in `units_below_min_spikes` like any other
thin half. That run is queued behind the local raw-file rebuild, which is what it needs.

### 5bd.44 — The review pipeline moves to the cluster, where it was supposed to be all along (2026-11-12)

Gil, on seeing that 20260915's post-KS half was running here: *"didnt we agree that
you'll run the review tool pipline also on the cluster right after the kilosort? this
should be part of the same pipline"*. He is right, and 5bd.20 says it in our own words:
**the unit of work is stage -> sort -> build review data -> copy back -> purge**, not
"sort everything then review everything". Only the first two steps were ever built.

**What that cost, concretely.** The joint 2026-09-15 session came home as a rebuilt
180 GB raw file, four agent-pipeline workers were started on a 64 GB machine, all four
were reaped for system memory pressure, and the remaining raw-bound step (bombcell) then
had to wait behind a live Popper recording writing to the same disk. None of that is a
bug in anything -- it is the consequence of running raw-bound work on the acquisition
rig when a free cluster was holding the same bytes.

**Which steps actually had to move, and the test for it: does it read the raw .dat?**

    agent pipeline              memory-maps probe1.dat
    bombcell                    reads raw waveforms
    raw_cluster_points          detects spikes in the raw
    raw_separation_metric       detects spikes in the raw

plus the two cheap steps that sit between them in the dependency chain (corrected
classification, which bombcell feeds and the raw_* metrics consume, and the Trodes
channel lookup). The display half -- session timeline, merge candidates, review HTML --
stays on this PC: it needs the DIO and the Bpod file, it is seconds of CPU, and it ends
in a published artifact.

**Two images, because they cannot share an interpreter.** kilosort 4.0.24 pins
python 3.9 / numpy 1.26.4; bombcell 0.89 and pandas 3.0.5 need python 3.11 / numpy 2.4.6.
Both pins are load-bearing -- every session already processed used them, and a bombcell
that computes a metric differently moves every verdict downstream without announcing it,
because `corrected_classification_session.py` thresholds those metrics against values
calibrated on 20260901_085606. So `docker/Dockerfile.postks` pins exactly what
`D:\conda_envs\bombcell` holds, read off it today, and asserts those pins at build time
the same way the Kilosort image does.

Two jobs rather than two steps in one job, for a second reason: the sort needs a GPU host
and nothing in the post-KS half touches a GPU. One job would hold a V100 idle through
bombcell, on the resource the cluster is actually short of. So `ris_submit_all.sh`
submits both and chains the second with `-w "done(<job1>)"`.

**The portability this needed, and what was deliberately NOT touched.** 77 scripts under
`scripts/` spell out `D:\Gil\spike_sorting_agent\outputs`. Five of them have to run on
Linux; the other ~70 are display-side and were left exactly as they are rather than
swept through for a portability they do not need. `striatal_agent/paths.py` reads
`SPIKE_AGENT_OUTPUTS` and falls back to the old path, so nothing on this PC changes.
Proof rather than assertion: `build_trodes_channel_lookup_session.py` re-run after the
refactor writes a **byte-identical** CSV to the one from before it.

**One input could not be made portable.** That same channel-lookup script reads the
Trodes SpikeNTrode XML out of the `.rec` file's header, and scratch holds `probe1.dat`,
never the 300 GB `.rec`. Same shape of problem as the probe map in 5bd.20, and the same
answer: `stage_session_to_ris.ps1` extracts the `.trodesconf` here and ships it, because
what the step needs cannot exist on a compute node.

**Status.** Code written, syntax-checked, pushed to `code/` on storage2 together with
`striatal_agent/`. The image build is submitted-but-blocked: pushing to Docker Hub is a
public publication and needs Gil's go-ahead, exactly as `ris_build_image.sh` says for
the Kilosort image. The one command, for when he gives it:

    ssh compute1
    bsub -G compute-akepecs -q general -n 4 -R 'rusage[mem=16GB] span[hosts=1]' \
         -o /storage2/fs1/akepecs/Active/Gil_Zur/build_postks/build.%J.log \
         -a 'docker_build(docker.io)' \
         -- --tag zurgil/spike-postks:bc0.89-py311 \
            /storage2/fs1/akepecs/Active/Gil_Zur/build_postks

**First real use, when the image exists.** 20260915_084812_103710_concat's raw file is
STILL on scratch (168 GiB, confirmed today), so its post-KS half can run there instead of
waiting for the rig -- and since the same session was just scored locally, its cluster
result is a free end-to-end correctness check: same inputs, same code, same unit scores
expected.

### 5bd.45 — The joint session's review tool is live, and the cluster half proved itself (2026-11-12)

**https://claude.ai/artifact/9wCjhAQncHPDuSYijKwgMb** — 465 units, the first review tool
built for a joint session and the first whose raw-bound half ran on RIS.

**The correctness check that mattered.** 20260915_084812_103710_concat had already been
scored on this PC, so running it again on the cluster was a free end-to-end comparison.
LSF 762947, 53 minutes, and the consolidated agent report came back **identical**: median
6.14476495185639 % spikes recovered, the same 155-unit red-flag list, the same tier split
(459 tier2 / 1 tier1 / 5 tier3). Same inputs, same code, different machine, same science.

**bombcell on the joint session:** 74 GOOD, 148 GOOD+NON-SOMA, 97 MUA, 142 MUA+NON-SOMA,
4 NOISE. Against bombcell's own original labels the corrected classification promotes 65
of its NOISE calls to GOOD, which is the same pattern every session has shown and the
reason the correction exists.

**Two steps failed, for one reason, and it is a joint-session reason.**
`build_trodes_channel_lookup_session.py` reads the SpikeNTrode XML out of the `.rec`
header; a concatenated session has no `.rec` at all, so there was nothing to read, and
`raw_separation_metric_session.py` then failed because it reads that lookup's CSV. Fixed
by extracting the config from **part 2's** `.rec` (255 KB, 384 SpikeNTrode entries) and
dropping it in as the concat session's `.trodesconf` -- the two parts of one day share a
config. Re-run as LSF 763728: 384/384 channels matched, then 465/465 units in 2462 s.
`ris_submit_postks.sh` now checks for that file up front and names the two steps that
will fail without it.

**The timeline, which was the step to be careful with.** 207 bins from part 1 and 184
from part 2, 391 of 20 s, seam exactly on a bin boundary. **670 trial markers** (356 +
314). No dropped-packet gaps in either part. Part 1's 5.000 s sub-bin tail drops 10,719
spikes (0.115 %) from the display only. The 37m56s break is in the JSON as
`real_time_gap_seconds`, not drawn as empty bins.

**Assembled and published.** merge_review_data reported `cluster_points=0, separation=0,
burst=0, trodes_channel=0` missing -- every field populated for all 465 units. 150 merge
candidates, the strongest 311/312 (0 um apart, similarity 0.974, CCG dip 0.185, waveform
r 0.964). 465 `raw_traces` documents pushed to the artifact db in 39 batches of 12 (12 is
the ceiling: 25 serialises to 1.92 MB against a 1 MiB request limit).

**Two things learned about the build that are not about this session.** The post-KS image
passed every build-time check and then died on a real node -- numba could not write
bombcell's `@njit(cache=True)` cache into read-only site-packages -- so a green build is
not evidence the image runs, and step 3 of the build recipe is not optional. And the
build job exits 101 even when the push succeeded, because `docker_build(docker.io)` makes
the builder tag and push a bogus `docker.io/library/docker.io` first; judge it by a
`digest:` line in the log, never by `bjobs`.

## 5bd.46 — Popper 20261112: the first session put through the whole cluster pipeline from the day it was recorded, and the map check that nearly passed for the wrong reason  (2026-11-12)

Gil asked for today's session to be copied to Z and, if the export was finished, taken
through the whole route in one go: upload, Kilosort, review tool, results back to Z. This
is the first time the pipeline has been run on a session the same day it was recorded
rather than on a backlog session, which is what §5bd.20's unit-of-work principle was
written for.

**The session.** Popper `20261112_082028`, recorded 08:20:28 to 13:03, 4 h 43 m. Trodes
finished exporting at 15:21. Complete and self-consistent before anything was moved:
`probe1.dat` is 391,485,372,672 bytes = exactly 509,746,579 samples of 384-channel int16,
and `timestamps.dat` agrees on that same count. Only `Trodes.exe` itself was running, no
exporter, and the sizes had been stable for minutes.

**THE MAP CHECK, AND WHY THE FIRST VERSION OF IT WAS NOT GOOD ENOUGH.** Gil's standing
rule is to compare the resulting channels against the map already used for the session
before committing a sort to them. The first check compared the exported
`channelmap_probe1.dat` — and found today's 4,608-byte pad-location body byte-identical
to 20261110's and 20261111's, which looked like a clean pass.

It was the right answer from the wrong evidence. That file carries only COORDINATES. A
change in which *hardware channel* feeds which ntrode would not necessarily move a single
byte of it, and that is precisely the change that sorts a session on the wrong sites while
every number still looks plausible. Gil then said the thing that forced the better check:
*"notice that this session should have different channel map from the other sessions."*

Redone against the fingerprint `channel_map_history.py` uses — per ntrode, the nested
`SpikeChannel`'s `hwChan` **plus** `coord_ml`/`coord_ap`/`coord_dv` — read out of each
recording's own header:

| session | fingerprint | `coord_dv` span |
|---|---|---|
| Popper 20261112 (today) | `2e360c196cae03cc` | 620..4440 um |
| Popper 20261111 | `2e360c196cae03cc` | 620..4440 |
| Popper 20261110 | `2e360c196cae03cc` | 620..4440 |
| Popper 20261109 | `2e360c196cae03cc` | 620..4440 |
| Popper trodesconf on Z (20261026) | `2e360c196cae03cc` | 620..4440 |
| Shamir 20260915 / 20261023 ("map A") | `ad898e43216224a6` | 920..4740 |
| Shamir 20261028 ("map B") | `c4f75a4452027f93` | 100..3920 |

So Gil was right and the labels in `channel_map_history.md` are Shamir-only: **Popper has
a third map**, 300 um off map A and 520 um off map B. Both are the silent-depth-shift kind
of error. Within Popper the map has been stable since the Oct 26 config — all four Popper
days share one fingerprint — which is what cross-session comparison of these sessions
needs. The map that went to the sort was built from TODAY's own channel file (384 channels,
yc 620..4440, matching today's header exactly) and verified md5-identical on the cluster,
`470c05375ea263b449c292a17fc94b62`.

Lesson worth keeping: **the exported pad-location file is not sufficient evidence of a
channel map.** Fingerprint the `.rec` header, hwChan included. `channel_map_history.py` is
hardcoded to Shamir, which is why no written record existed for Popper to check against.

**Two transfers at once, which is why this took 38 minutes and not three hours.** The
archive (F to Z) and the upload (F to scratch) both read from F but bottleneck on different
things — SMB write versus the WashU link — so they were pointed at DIFFERENT files: the
436 GB `.rec`, which the cluster does not need, went to Z while the 391 GB `probe1.dat`
went to the cluster. Measured: the upload ran at **173 MB/s** sustained, 2,267 s for
391 GB, far better than the 98 MB/s of §5bd's earlier `dd | ssh` run; robocopy dropped to
~40 MB/s while sharing F's read bandwidth and recovered to ~118 MB/s once the upload
finished. F's read ceiling is therefore around 210 MB/s and it was the real constraint.

Verified before anything was queued on it: both sides exactly 391,485,372,672 bytes,
`dd` exit 0, and **nine spread 1 MiB blocks identical**, including the last MiB, which is
the one that catches a truncated transfer.

**A sizing bug caught before it cost a GPU.** `ris_submit_postks.sh` asked for a fixed
112 GB, from a measurement on `20260915_084812_103710_concat`: 10 units per worker,
2.17 h, 16-20 GB RSS each. That number is not a constant — a batch process holds the raw
snippets for its units' spikes, so it scales with session DURATION as well as units. At
4.72 h, 10 units per worker wants ~40 GB each, 160 GB for four, against a request of 112.
The job would have been reaped hours in, *after* the GPU half had already been paid for.

Both knobs now derive from the session's own length, the raw file serving as the clock
(384ch x int16 x 30 kHz). The invariant is **unit-hours per worker**: 10 x 2.17 = 21.7 is
the product that measured 16-20 GB, so `BATCH = 22 / hours`, clamped to [2, 10]. The
property to preserve when editing this: it returns BATCH=10 for the session it was
measured on, and gives 4 for today. `PER_PROC_GB` is deliberately 32 and not 20 because
above 2.2 h the linear scaling is INFERRED, not measured, so the headroom covers being
wrong by half. Memory is not scarce here — 37 of 41 exec hosts carry >=502 GB (`lshosts`),
so 144 GB costs nothing in queue time while an OOM costs the whole job.

**Queued 16:16.** Job **769960** (sort + split-half waveforms, GPU, stock thresholds
Th_universal=9.0 / Th_learned=8.0 for batch uniformity) RUN on compute1-exec-358; job
**769961** (post-KS, 5 cores, 144 GB, no GPU, 4 workers x 4 units) PEND on
`done(769960)`, which is the correct state and not a stuck job.

**Two gaps found while looking around, neither blocking today.**

1. `20261109_092303` is on F only — 238 GB, raw `.rec`, never exported and never archived
   to Z. It is the one hole in the recent run, and it shares Popper's map.
2. `Z:\Gil\Popper\bpod\` is **empty**. The sort and the review tool do not need it —
   `build_session_timeline_data.py` takes trial markers straight from the DIO, by design,
   "no Bpod, no sync" — but the sync stage does, for today and for every Popper session.

**The map history now exists for Popper too, which is why the gap above was invisible.**
`channel_map_history.py` was hardcoded to Shamir, so "map A" and "map B" in
`channel_map_history.md` had always meant Shamir's maps with nothing saying so — and that
is exactly the ambiguity that made the first check here look like a pass. The script now
takes `--rat` (Shamir / Popper / Begin, the same explicit F-vs-Z folder mapping
`archive_session_when_exported.py` uses, because the two drives spell the animals
differently), and the written record says in its own first lines that **labels are per
rat**: Popper's map A is not Shamir's map A.

Written to `Z:\Gil\Popper\channel_map_history.{json,md}`: **one map across all four Popper
sessions**, `coord_dv` 620..4440 um in 192 steps, fingerprint `55698a66f4e9`. Shamir
re-runs unchanged — 2 maps over 31 sessions, 28 + 3, same spans as the existing record.

One fix the run itself forced: a session being archived at that moment holds an exclusive
write lock on its destination, so reading Z's copy of `20261112_082028.rec` failed with
`PermissionError` while robocopy had it open. `find_recs()` now keeps BOTH copies of a
session and tries them in turn, archive first; F's copy read fine and the header is
identical either way. Worth having permanently — the natural time to run this script is
right after a session lands, which is exactly when its archive copy is locked.

### 5bd.47 — A job that reported DONE having done nothing, and the 2 GB file that caused it (2026-11-12)

Job **769961** (post-KS for 20261112_082028) went DONE at 21:02:08 after running from
02:57:34 to 02:59:13Z — under two minutes for a six-step pipeline whose fastest step
alone takes forty. `bjobs` said `Successfully completed`. Nothing had been computed.

**Root cause, and it is a staging mistake, not a cluster one.** `timestamps.dat` was never
uploaded to scratch. I staged the session by hand-picking files instead of using
`stage_session_to_ris.ps1`, whose `$files` list already contains it, and reasoned that
the cluster would not need it because sync runs locally off Z. But `discover_session()`
(striatal_agent/session.py:71) refuses to describe a session without it, and every one of
the six steps starts by describing the session. So all 111 batch workers died on the same
`FileNotFoundError` inside 45 s, `kilosort_original` was therefore never built, bombcell
then found no `spike_templates.npy`, and the remaining steps fell behind it.

**Why LSF still said DONE, which is the part worth remembering.** `run_step` in
`ris_run_postks.sh` returns 0 on failure by design, so one broken step cannot strand the
five that do not depend on it. The cost of that design is that a wholly-failed job is
indistinguishable from a successful one in `bjobs`. **Never judge a post-KS job by its LSF
state — read the log's last block.** The honest tell here was arithmetic: the job was far
too fast.

**Fixed where it can't recur.** `ris_submit_postks.sh` now REFUSES to submit without
`<sid>.timestamps.dat` beside the raw file, accepting a `concat_*.json` instead for a
joined session (which legitimately has no single timestamps file, and proves itself with
its concatenation provenance). Refuse, not warn: the file is 2 GB against a 391 GB raw
file, and its absence breaks all six steps at once. The message prints the `dd` command
that ships it.

**The recovery cost nothing but wall clock, because the raw file was still on scratch.**
timestamps.dat uploaded (2,038,986,923 bytes, md5 `845b010fa8b2f10366c0048ea9d3fdfc`
identical both sides), the 111 stale batch logs cleared so they could not be mistaken for
fresh output, and the sort itself untouched — 442 units, 29,735,992 spikes, both steps
including the split-half waveforms. Resubmitted unchained as **776207**.

**776207, 93 minutes, all six steps.** 111 batches of 4 dispatched over 33 minutes (the
health signal: the failed run dispatched all 111 in 45 s), 442 of 442 units covered with
0 duplicates, median 3.35 % / mean 7.52 % spikes recovered, 117 red-flag units, tiers
434 tier2 / 8 tier3, classes 342 tan-like / 70 bursty-MSN-like / 22 tonic-FSI-like.
bombcell: 17 GOOD, 110 MUA, 52 NOISE, 263 NON-SOMA; the corrected classification turns
that into 35 GOOD, 120 GOOD+NON-SOMA, 105 MUA, 179 MUA+NON-SOMA, 3 NOISE — promoting 11
of bombcell's 52 NOISE calls to GOOD, the same pattern every session shows. 384/384
channels matched to a Trodes ntrode. raw cluster points 442 units in 969 s, raw separation
442 in 2100 s.

### 5bd.48 — Popper 20261112 is reviewable, and the whole session is on Z (2026-11-12)

**https://claude.ai/artifact/M54x4TFQzUXJZJWwMcycSG** — 442 units. The full request Gil
made this morning is now done: archived to Z, uploaded, sorted, scored, review tool
built, results back on Z.

**Brought home without scp, as the rule says.** The cluster `outputs/` (401 MB) came by
streaming tar in 3 s; the sort itself — `kilosort4/` with `kilosort_original/` and
`agent_processed/` inside it, plus `waveforms/` — in one tar of **1,547 files /
24,309,341,086 bytes**, verified by inventory (every name and size equal, the apparent
diff being only `sort` collation) and by md5 on the eight files the science rests on:
spike_times, spike_clusters, amplitudes, templates, similar_templates,
channel_positions, agent_report.json and unit_waveforms.npz.

**One missing file stopped the local half, and it is the same class of mistake as
yesterday's.** `compute_raw_trace_snippets_session.py` calls `discover_session()`, which
also wants `<sid>_custom_neuropixels_map.json` — a file the stage script generates on
the way up and which therefore existed only on scratch. Copied down with the
`.trodesconf` beside it, md5 equal both sides. The lesson is the same one the
timestamps.dat guard records: `discover_session()` is the gate every step passes
through, so its whole file list has to travel with the session, both directions.

**The local build, in order, all six steps:** review data (442 units, ACGs), session
timeline (**850 bins of 20 s, 1087 trial markers from DIO code 1, no dropped-packet
gaps**, 4.72 h, median unit rate 1.90 Hz), burst/ISI metrics (284 flat, 93 excluded as
high-rate, 17 dip-then-rise, 17 facilitation, 11 decay, 8 rise-then-dip, 12 insufficient),
the merge into `pipeline_review_data` reporting **cluster_points=0, separation=0, burst=0,
trodes_channel=0 missing** — every field populated for all 442 — then 150 merge candidates
and the page itself (14.15 MiB of the 16 MiB limit). 442 `raw_traces` documents pushed to
the artifact database in 37 batches of 12.

**The strongest merge candidate is 133+136:** 0 um apart, similarity 0.883, waveform
r 0.887, and a **CCG dip of exactly 0.0** — not a shallow trough, an empty refractory
window. 232+234 (dip 0.275, sim 0.944) and 381+383 (dip 0.304, sim 0.987, 40 um) follow.

**A unit mismatch worth writing down before it bites a third time.**
`build_merge_candidates.py` defaults `--uv_per_bit` to 0.195, which is the SpikeGLX
figure for Neuropixels 1.0 at gain 500. Our Trodes exports are
**0.018311105685598315 uV per LSB, read from each session's own .rec header** — the value
the cluster job and the corrected classification use. The default is 10.6x too large.
It changes no decision and no shape (both units of a pair scale alike) but it makes the
merge panel's uV axis disagree with every other panel. 20260916_110311's merge tool was
built on the default; the joint session and this one pass the real gain. The flag's help
text now says all of this instead of naming a probe.

**On Z, beside the sort, in the shape 20260916_110311 established.**
`20261112_082028.kilosort/` now holds `kilosort4/` (22.543 GiB, 1545 files, copied at
116 MB/s, 0 failed), `waveforms/`, `outputs/` (the bombcell result directory, the
corrected classification, the raw metrics, the Trodes lookup and all five cluster job
logs — 769960's sort, 769961's failure and 776207's rerun), and `review/` with the page,
its `data/merge_candidates.json`, the seven `inputs/` files needed to rebuild it, and a
`verdicts/` directory holding only a NOTE. **The verdicts do not exist yet** and the
README says so in its second paragraph: they live in the page's database until Gil
reviews, and `cluster_info.tsv` is where they then get applied, never
`cluster_group.tsv`.

**Still open on this session.** Gil's review of the 442 units, then the three verdict
collections exported into `review/verdicts/` and applied. Sync needs
`Z:\Gil\Popper\bpod\`, which is still empty. And `/scratch1/fs1/akepecs/zur/20261112_082028`
(391 GB) can now be purged — every step that needs the raw file has run — but that is
Gil's call, not mine.

---

## 5bd.49 The merge panel was showing 150 of 2728 pairs, and the cap cost sixteen seconds

Gil asked how far apart units 55 and 70 are in 20261112_082028. They are 80 um — same
column on the probe (`xc` 2474 um for both), peak channels 317 and 309, `yc` 1280 and
1360 um, so four site-rows along the shank. Outside the 60 um pairing radius, so they
were never merge candidates. That part was fine. What the question exposed was not.

**Neither unit appeared in ANY of the 150 analysed pairs, and geometry was not the
reason.** 2728 pairs in this session are genuinely within 60 um of one another. The
analysis looked at 150, because `--max_pairs` truncates the shortlist after ranking by
template similarity. The cut lands at similarity 0.826. Unit 55 has twelve partners
inside 60 um and unit 70 has twelve, including units on the *same peak channel* — 0 um
— and not one of those pairs had been computed. 55+57 ranked 243rd at similarity 0.749;
68+70 ranked 298th at 0.700. Of the 317 pairs in this session that share a peak
channel, only 101 had been analysed.

**The metadata hid it.** `n_pairs_in_range` was recorded as `len(cands)` *after* the
truncation, so the archived JSON claimed 150 pairs were in range when the true number
was 2728. A unit absent from the merge panel reads as "checked, nothing found"; it
actually meant "never compared". Fixed in efa33f8: keep the true count before
truncating, report `n_pairs_in_range` and `n_pairs_analysed` separately, record
`max_pairs` beside them, and log a warning naming how many pairs are being dropped and
what their absence does and does not mean.

**The cap was obsolete, and measuring it took a minute.** Estimates were given before
measuring — "20x, overnight", then "a few hours" — and both were wrong by orders of
magnitude. Measured:

| work | measured |
|---|---|
| CCG for all 2728 co-located pairs | **15.9 s** (5.8 ms/pair; median 63k spikes/pair, max 1.38M) |
| raw waveform reads, all 439 non-noise units | **~31 s** (0.07 s/unit, 6 units sampled across the probe) |
| the real rebuild, end to end | **171 s** |

`correlogram()` does two `searchsorted` calls and a ragged gather rather than an outer
difference. Its own docstring says it: *"a unit here can have 400k spikes and this runs
150 times, so the loop version is the difference between seconds and an hour."* Someone
hit the slow version, vectorised it, and the cap from the slow days was never lifted.
It had been discarding 94% of the co-located pairs to save sixteen seconds.

The payload estimate was wrong too — 44 MB predicted, 9.6 MB actual. A pair is 1.2 KB;
the file's bulk is the per-*unit* waveform and rate timeline, 15 KB each. 2728 pairs and
439 units fit inside the 16 MB artifact limit with room to spare. **Lesson, the same one
as §5bd.47 in a different costume: measure before quoting a number. Every one of the
four estimates here was wrong, and the measurement cost less than the explanation.**

### What full coverage actually found

Re-ranked with the live panel's own statistics (`dipStats` from
`patch_review_tool_merge_v2.py`: refractory count against what the 2-5 ms shoulders
predict, tested against Poisson chance, `DIP_P = 0.01`):

- **907 pairs pass p < 0.01**, of which **865 have similarity below 0.826** — invisible
  under the old cut.
- 118 pairs have shoulders predicting under one count, so no test is possible at all.
- The strongest, backed by real counts: **129+136** (0 um, 111 observed vs 296.5
  expected, p < 1e-5, sim 0.951, wf_r 0.884), **195+196** (899 vs 2510, p < 1e-5),
  **381+383** (101 vs 221, p < 1e-5), **131+135** (14 vs 33, p = 0.00016), **129+133**
  (4 vs 12, p = 0.0076). The 129/133/136 triplet wants looking at as a group.

**Two claims in §5bd.48 were wrong and are corrected here.** 133+136 was reported as the
headline merge — "CCG dip of exactly zero, the one piece of real evidence that two
clusters are one neuron". Against its own shoulders it is 0 observed against 4.00
expected, **p = 0.0183, which does not pass the panel's own bar**. Suggestive, not
decided. 232+234, listed as a runner-up, is 2 against 2.5 expected, p = 0.54 — nothing.
Both readings came from the far-flank `ccg.dip_ratio` field, which v2 exists to
deprecate: it is the same construction as the contamination metrics that misbehave, and
any CCG with broad structure invalidates it. **Do not read `ccg.dip_ratio` out of the
JSON. The number that counts is computed in the page.**

### And the answer to Gil's actual question

None of units 55's or 70's co-located pairs show a significant dip:

| pair | dist | refractory count | expected | dip | p | reading |
|---|---|---|---|---|---|---|
| 55+57 | 0 um | 8 | 10.00 | 0.800 | 0.333 | flat — two neurons |
| 68+70 | 0 um | 21 | 19.00 | 1.105 | 0.726 | flat — two neurons |
| 55+60 | 0 um | 11 | 3.50 | 3.143 | 0.9997 | a *peak*: synchrony, not identity |
| 55+59 | 0 um | 0 | 0.00 | n/a | n/a | no test possible, ever |

So they are not fragments of their neighbours, and that is a real answer that could not
be obtained before. 55+60's zero-lag peak is the textbook case the panel warns about —
common input or one spike detected twice, never a merge.

Gil's proposal — compute the CCG whenever both units have a clear autocorrelogram —
is sound in spirit but would have excluded exactly these pairs. 55 and 70 both have
*perfectly* clean ACGs (dip 0.000, measurable). It is their partners that fail: 57 at
0.538, 68 at 0.522, 60 at 0.994 (flat — contaminated), and 59 unmeasurable on 266
spikes. The pairs most worth testing are often one clean fragment beside one messy blob,
so requiring both to be clean throws them away. With full coverage costing a minute the
question is moot: compute every co-located pair, and let ACG quality be context shown
beside the pair rather than an admission test.

### Done to the live session, mid-review, without losing anything

Gil was already reviewing when this landed, so order mattered:

1. **Verdicts dumped first**, before any change: 72 unit verdicts, 7 merge verdicts, 6
   trial selections, saved locally and to
   `Z:\...\review\verdicts\snapshot_2026-11-13_1017_mid_review\` (85 files).
2. **Checked that regeneration cannot orphan them.** `pair_id` is `"%d_%d" % (a, b)`,
   built from the unit numbers, not a list index — so growing 150 to 2728 leaves every
   key pointing at the same pair. All 7 of his merge-verdict pairs are present in the
   new set. Unit verdicts are keyed by unit id and untouched.
3. Rebuilt with `--max_pairs 3000`, rebuilt the page through all five patches
   (14,832,172 bytes, 14.15 MiB), published to the **same URL** as Version 2 so the
   database carried over; capabilities `db` + `downloads` carried forward.
4. **Verified after publishing:** verdict 255_260 still reads back at version 2, and the
   new build's `clustering.spike_clusters_sha1` is
   `5bc217d99f3e8c29e72eb21c23420d70f56556e6` — identical to the sha pinned inside his
   merge verdicts, so none of them went stale.

A republish reloads the page in the reviewer's browser. The database survives it, but
the view resets, so this is worth saying out loud before doing it to someone mid-review.

**Still open.** Gil's review continues on the 442 units with 907 significant pairs now
visible. `Z:\Gil\Popper\bpod\` is still empty, blocking sync. The archived
`merge_candidates` JSON on Z still carries the old 150-pair build and the misleading
metadata field; it should be refreshed when the review settles. And
`/scratch1/fs1/akepecs/zur/20261112_082028` (391 GB) remains purgeable on Gil's word.

## 5bd.50 Gil's 20261112 decisions taken out of the browser and put on Z, with the four units a merge leaves unlabelled  (2026-11-13)

Verdicts live in the review page's own database on claude.ai. A database is not an
archive: it has no copy, no history anyone else can read, and no relationship to the
sort it describes except the sha1 stamped inside each merge verdict. So Gil asked for
everything he had decided to be written somewhere durable, ready for the Bpod sync and
the analysis after it. This is what came out and what it was checked against.

Pulled from https://claude.ai/artifact/M54x4TFQzUXJZJWwMcycSG, all three decision
collections:

    verdicts            260 documents    good 140, mua 120, noise 0
    merge_verdicts       31 documents    merge 27, separate 3, unsure 1
    trial_selections     18 documents    16 with windows, 2 recorded empty

The page writes a fourth collection name, `raw_traces`, but only ever reads it
(`db.collection('raw_traces').doc(...).get()` at one site, no write anywhere) -- it is
the raw-snippet payload, not a decision. Three collections is the whole decision state.

CHECKED AGAINST THE SORT, not just counted:

  * every unit id named by a verdict is one of the 442 clusters in
    kilosort4/spike_clusters.npy. No orphans.
  * all 31 merge verdicts carry spike_clusters_sha1
    5bc217d99f3e8c29e72eb21c23420d70f56556e6, which is that file's actual sha1. Every
    pair was judged against the clustering still on disk, so the frozen unions still
    mean what they meant when Gil clicked.
  * every selection window lies inside the recording. The first check said eleven
    windows ran past the end, which was the check being wrong, not the data: the
    windows are in ABSOLUTE Trodes seconds, and this session's clock starts at
    first_timestamp 312134 = 10.4045 s, not at zero. 17001.957 - 10.404 = 16991.553 s,
    and 16991.553 x 30000 = 509,746,578 samples against a last spike at 509,746,562.
    They fit exactly. A window check that assumes a zero origin will condemn every
    correct selection in this project.

WHAT IS NOT DECIDED, said plainly because a saved file reads as finished:

  * 182 of the 442 clusters have no verdict. 260 do.
  * Four units sit in a pair Gil marked MERGE but carry no verdict of their own: 32
    (in 32+33), 185 (in 184+185), and 402 and 403 (in 402+403, where NEITHER unit is
    labelled). Whatever a merged unit inherits, it should not be inheriting silence,
    so these four want a verdict before the sync writes cluster_info.tsv.
  * Units 185 and 403 have a trial selection with zero windows, which the page records
    as "the whole session" rather than as nothing. Harmless, worth knowing.
  * No cluster was called `noise`. The page offers good / mua / noise (`data-v=`), so
    the vocabulary was there and unused -- Gil is putting the bad ones in `mua`.

Written to Z, beside the sort, every file hashed on both sides:

    Z:\Gil\Popper\20261112_082028.rec\20261112_082028.kilosort\review\verdicts\
        verdicts.json  merge_verdicts.json  trial_selections.json
        manual_verdicts.csv          the unit verdicts flat, for reading by eye
        MANIFEST.txt                 counts, the checks above, sha256 of each file
        snapshot_2026-11-13_1346_260_units\   309 files, one JSON per decision
        snapshot_2026-11-13_1017_mid_review\  this morning's 85, left untouched
        NOTE_superseded_2026-11-12.txt        the old "NO VERDICTS YET" note

The old NOTE.txt said this session had never been reviewed, which stopped being true
at 10:17. It is renamed rather than deleted: it is the only record of what the bundle
claimed on the day it was written.

NOT APPLIED. `scripts/sync_session.py 20261112_082028` is the only thing that should
write these into cluster_info.tsv, and it should run when the review is finished --
or knowingly, with 182 units unlabelled.
