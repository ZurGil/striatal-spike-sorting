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


## 6. WHERE THINGS STAND  (current as of 2026-09-30 — read this first)

### The one-paragraph version
The module reads Kilosort4 output plus raw voltage and scores candidate
spikes against a unit's template, with three corrections layered on: wavelet
sub-sample alignment (1b), temporal noise whitening (pillar 3 / step A), and
multi-channel spatial footprint (1c). It is DIAGNOSTIC ONLY — nothing has
ever written to a Kilosort or Phy file. The work moved from "find spikes
Kilosort missed" to "figure out why a unit's own template cannot reject its
neighbour's spikes." Footprint gating is now the answer to the second
question and has been run inside a real detection scan: on the cleanest unit
it removed 470 of 470 contaminating events at a cost of 0.8 points of recall.
The cost of that is that almost nothing survives as a genuinely new spike.

### Which session
**20260916_110311**. Kilosort output at
`F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4`,
raw at `...\20260916_110311.probe1.dat` (290 GB, 378M samples, 210 min).
Sections 1–5g used the earlier session 20260901_085606 (on D:), whose
curation labels were flagged as untrustworthy — that is why the switch
happened, and why its merge candidates are not being re-litigated.
Constants are identical across both: nt0min=20, nt=61, fs=30000, 384
channels. `GAIN_TO_UV = 0.018311105685598315` is carried over and NOT
confirmed for this session (no gain field in its ops.npy) — harmless for
R²/cosine/AUC, which are scale-invariant, wrong for any microvolt claim.

### What is built and validated
- `nuisance_model.py` — amplitude+stretch fit. Use `fit_nuisance_prealigned`
  (2-parameter); the 3-parameter version's timing and stretch regressors are
  near-collinear and cannot be separated (section 1). Takes an optional
  `whitening_matrix`.
- `wavelet_features.py` — per-unit probe frequency selection
  (`select_probe_frequency`, criterion="timing", f_hi=8000, grid +
  bracketed golden-section refinement, edge warning), coarse+fine alignment
  (`coarse_then_fine_shift`).
- `noise_whitening.py` — AR(4) temporal noise model per channel to whitening
  operator. Real noise is strongly correlated (lag-1 0.72 to -0.09 after
  whitening); ~3x better spike-vs-noise discriminability.
- `spatial_footprint.py` — per-channel amplitude footprint, cosine
  similarity, plus `multichannel_template` / `multichannel_snippet_concat` /
  `block_diagonal_whitening`.
- `fixed_point_acceleration.py` — Aitken (scalar) and Anderson (vector)
  acceleration for the template-rebuild fixed point. NOT for the frequency
  search, which is a maximization.
- `tests.py` — **31 tests, all passing**. Run `python tests.py` from inside
  `post_ks_correction/`.

### The findings that matter most
1. **Single-channel scoring was the binding limitation, and footprint
   gating fixes it — where the two units' footprints actually differ.**
   Unit 440's detections went 474 to 1 with 470 of 470 contaminating events
   from unit 439 removed, at a cost of 0.8 points of recall. (5q, 5s)
2. **The footprint bar must be calibrated per pair, not at a fixed
   percentile.** Where an impostor sits relative to the target's own score
   distribution is a property of the pair. A bar at the target's 5th
   percentile eliminated unit 439 entirely but let unit 412 through almost
   untouched, even though 412 is separable at AUC 0.827. (5s)
3. **The frequency-selection rule was wrong and made alignment harmful.**
   Choosing f0 by argmax|W| ignores that the same f0 sets timing precision
   (one cycle = fs/f0 samples, so any phase error is multiplied by
   fs/(2*pi*f0)). At ~800 Hz a 10-degree phase error becomes a full sample.
   The correct rule is argmax f0*|W(f0)|. Fixing it took spurious shift on
   correctly-placed spikes from median 1.01 samples (53% moved >1 sample) to
   0.37 (1.3%). (5o, 5r)
4. **Detection recall is not this dataset's problem; cluster precision is.**
   Every "new spike" count has shrunk each time the tooling got more honest:
   115 to 40 with the frequency fix, and the surviving 37 on unit 408 do not
   survive a properly-calibrated footprint gate either. Across ~150
   candidates on six validated units, not one has held up. (5n, 5p, 5s)
5. **Cosine similarity overstates spatial agreement and should never be
   quoted alone.** 408/412: cosine 0.767 but Pearson r 0.481 (p=0.16).
   440/439: cosine 0.589 but Pearson r -0.068. All-positive amplitude
   vectors are never far apart in angle. (5m, 5t)
6. **The rebuild loop has a noise floor around 6e-4 under plain iteration.**
   Anderson (depth=3) is 1.9-5.3x faster at a reachable tolerance and
   reaches residuals plain iteration never does. Set tol=1e-3; 1e-6 is below
   the sampling noise floor of a 250-spike average. (5u)

### What to do next, in priority order
1. **Rerun the gated scan with per-pair calibrated bars** (5s's design
   conclusion) across more units, and report detection counts at a stated
   contamination target (e.g. "bar chosen to remove 90% of the dominant
   impostor") rather than at a fixed percentile. This is the run that
   produces defensible final numbers.
2. **Decide what the deliverable actually is.** Given finding 4, the module's
   value is not "extra spikes" but "which clusters are contaminated by which
   neighbour, and how separable they are." A per-unit contamination report
   — dominant impostor, footprint AUC against it, achievable
   recall/contamination operating point — is a more honest product than a
   missed-spike list, and every piece needed to build it now exists.
3. **The hybrid-ground-truth validation experiment.** Still the only thing
   that would establish real performance rather than diagnostics: inject
   real deformation into real background and compare against Kilosort at a
   matched false-positive rate.
4. **Still never built:** burst-history amplitude-recovery curve and its
   gating, robust (Huber) estimation, the learned deformation direction d_k,
   and automatic split-candidate detection across a whole session.

### Known caveats to carry forward
- Units 440/439 and 408/412 both have near-identical waveforms on their
  shared channel (0.972, 0.991) and are NOT merges. Shape agreement on one
  channel is weak evidence; the footprint decides.
- The refractory/CCG test is a VETO ONLY. Non-significance is never support
  for a merge (5l). Both pairs above pass the refractory test and are still
  rejected.
- Footprint comparison is alignment-insensitive by construction
  (peak-to-trough), but the 1c tests align both populations with the
  target's template and this was never separately controlled.
- Units 302 and 31 are unusable for detection work: their own real spikes
  score at or below the noise floor.
- "No nearby unit has this event" means no unit within 60 um has a spike
  within 15 samples. It does not prove the event belongs to the unit tested.
- Spike density is high enough (41M spikes / 3 h in the older session) that a
  random timepoint is within 15 samples of SOME cluster's spike 98% of the
  time. Any "already detected elsewhere" statistic MUST be restricted to
  spatially relevant units or it is meaningless.
- `outputs/pipeline_review_data.json` (source of the `sep_vs_noise` metric)
  is from an EARLIER curation stage; some unit_ids in it no longer exist in
  the final `spike_clusters.npy`. Always filter against the final data.
- Unit IDs are per-session and arbitrary.

### Git
Branch `post-ks-correction` in `D:\Gil\spike_sorting_agent`, remote `origin`
= https://github.com/ZurGil/striatal-spike-sorting.git. Authentication works
as of 2026-09-30 (Git Credential Manager, credential stored).
