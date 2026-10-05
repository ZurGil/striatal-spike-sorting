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

## 5aq. Stimulus-period dynamics, and why the per-tone analysis is impossible here

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


## 6. WHERE THINGS STAND  (current as of 2026-10-01 17:10 — read this first)

This section is self-contained. Everything needed to resume with no memory of
the conversation is here: what exists, what is running, what to run next, and
how to report it.

### 6.1 The one-paragraph version
This project built a post-Kilosort correction module (wavelet sub-sample
alignment, temporal whitening, spatial footprint), then a hybrid ground-truth
benchmark, then seven flag-gated modifications to Kilosort4 itself. The
benchmark is the durable result. Round one (13 units) said six of seven
modifications are worthless and one — footprint-aware clustering — wins on the
hardest case. Its headline finding is diagnostic: **84% of Kilosort's failures
here are spikes it DETECTED and then filed under the wrong cluster**, not
spikes it missed. Round one was too small (n=3 per tier) and its sources were
chosen by Kilosort's own labels, so **a rebuilt v2 benchmark now exists** —
33 hand-reviewed units, 52 placements, 15,086 ground-truth spikes over four
replicates. Round two has RUN (5ak), and two follow-up analyses answered the
question the benchmark was built for (5al, 5am): **neither merging fragments
nor post-hoc spike recovery helps.** Both buy a couple of points of recall and
cost four to ten points of precision, and both fail for one shared reason —
93.5% of what a unit's own template matches on its own channel is a
NEIGHBOUR'S real spike. The problem is per-spike attribution. Footprint
similarity is the only signal measured to address it (5q, AUC 0.999 vs 0.749;
5ak, the project's only statistically supported wins), and it has only ever
been used as a clustering weight or a post-hoc gate — never as the
discriminator in the detection path, which is the open lead.

### 6.2 THE IMMEDIATE STATE (what is happening right now)

**A 730 GB copy is in progress.**
`F:\Gil\Shamir\20261001_094335.rec` → `Z:\Gil\Shamir_1\20261001_094335.rec`
Started 16:58:45 at ~113 MB/s, ETA ~1.8 h (so ~18:45). Launched with
`robocopy /E /J /R:3 /W:10` — deliberately NOT `/MIR`, which deletes at the
target. Destination did not previously exist; Z: has ~50 TB free.
NOTE: Gil said "Z:\Gil\Shamir" but that folder does not exist — every prior
session lives in `Z:\Gil\Shamir_1\` under the same `<date>_<time>.rec` pattern,
so that is where it went.
**On completion: verify file count (11) and total size (730.33 GB) against the
source before declaring success.**

**Trodes is CLOSED** (17:10, after confirming 0 bytes of disk I/O over 12 s and
the source file untouched for 3 h). GPU is free: ~500 MiB / ~20%.

**THE SWEEP IS DONE — see section 5ak for the results.** All 36 runs completed
18:49-20:31, no failures. The copy finished and verified first (11 files,
730.33 GB, exact byte match). Headline: `footprint_cluster_strong` improves
pair-tier recall 0.720 -> 0.798 (p = 0.038) and `footprint_cluster` improves
overall precision 0.600 -> 0.654 (p = 0.010) — the first statistically
supported wins in the project. Everything else is neutral or worse. **The hard
tier is NOT reliable**: two identical runs disagree on 2 of its 12 placements,
by up to 0.44 recall.

**THE STALL, and why Trodes was closed.** Three separate attempts to run
Kilosort froze at exactly the same point — `spikedetect` logging "Detecting
spikes...", then nothing for 9+ minutes against a ~2 minute whole-run time.
Trodes was open each time. It holds the GPU at 37% even when completely idle,
and the GPU pinned at 100% during each attempt. Kilosort does START (it needs
only ~2.8 GB of the 6 GB) so this presents as a hang rather than an error.
**Trodes being "idle" is not sufficient — it must be closed.**

### 6.3 WHAT TO RUN NEXT — the whole instruction

**The sweep and both follow-up analyses are DONE. The open lead is one thing:**
a footprint-FIRST search in the detection path. Every measurement points at it
and nothing has tried it. 5q measured footprint similarity separating two
same-channel neurons at AUC 0.999 where the single-channel score manages 0.749;
5ak's only statistically supported wins are both footprint configs; 5am showed
the footprint gate roughly doubles recovery precision at every threshold — but
in all three it is a weight or an after-the-fact filter, never the thing that
decides which neuron a spike belongs to. The test to write scores each
candidate event against EVERY nearby unit's footprint and assigns it to the
best, instead of asking one unit at a time "is this mine?". `test_post_hoc_on_v2.py`
is the harness to extend — it already holds the truth-blind structure, the
chance control, and the per-candidate score table.

Analyses already run (do not repeat):
```bash
python analyse_v2_results.py                              # 5ak: paired tests, noise floor
python analyse_split_vs_contamination.py                  # 5al: merging fragments
python test_post_hoc_on_v2.py vanilla footprint_cluster_strong coarse_then_align   # 5am: ~25 min
```

To re-run the sweep itself from scratch:
```bash
cd D:/Gil/spike_sorting_agent/post_ks_correction
bash run_all_v2.sh          # ~70 min for all 36 runs
```
- Refuses to start if Trodes is on the GPU or < 4 GB free. `FORCE=1` overrides
  — but do NOT force past a live Trodes again; that is what caused the stalls.
- **Resumable**: skips any run whose `spike_times.npy` already exists, so it can
  be stopped and restarted freely. Nothing is lost by killing it.
- One OS process per (config, replicate), because `ks_patches.enable()` swaps a
  module-level function and would otherwise leak between configurations.

Then score:
```bash
python run_v2_comparison.py --compare vanilla vanilla_repeat subsample_align \
  coarse_then_align amplitude_normalize align_and_amp_norm \
  coarse_align_amp_norm footprint_cluster footprint_cluster_strong
```
Writes `outputs/v2_scores.csv` and `outputs/v2_merge_errors.csv`.

### 6.4 THE NINE CONFIGURATIONS BEING TESTED
All are flag-gated in `ks_patches.py`; `enable(name)` swaps a function at
runtime, `disable()` restores, the installed Kilosort package is never
modified. No flag = stock Kilosort by construction.

| config | what it changes | where |
|---|---|---|
| `vanilla` | nothing — stock Kilosort4 | — |
| `vanilla_repeat` | nothing — **control**, must reproduce `vanilla` exactly | — |
| `subsample_align` | sub-sample aligns each spike before the PCA projection | `spikedetect.run` |
| `coarse_then_align` | integer sliding alignment, then the above | `spikedetect.run` |
| `amplitude_normalize` | divides each snippet by its own magnitude | `spikedetect.run` |
| `align_and_amp_norm` | sub-sample align + amplitude normalize | `spikedetect.run` |
| `coarse_align_amp_norm` | coarse + fine align + amplitude normalize | `spikedetect.run` |
| `footprint_cluster` | appends normalized per-channel footprint to the clustering features, weight 1.0 | `clustering_qr.cluster` |
| `footprint_cluster_strong` | same, weight 3.0 | `clustering_qr.cluster` |

`vanilla_repeat` is not padding: in round one it reproduced `vanilla` exactly,
which is what established that the run-to-run noise floor is zero and every
difference is real signal. **If it does NOT reproduce vanilla this time, stop
and investigate before interpreting anything else.**

### 6.5 HOW RESULTS MUST BE REPORTED
Gil has been explicit about this, twice. Follow it exactly.

1. **Recall AND precision together, never one alone.** A percentage-only table
   was rejected as "not informative enough". They routinely move in OPPOSITE
   directions here — in round one `subsample_align` traded 4 points of recall
   for 19 of precision on collisions.
2. **Per tier, never only pooled.** easy / hard / collision / pair. Tier effects
   have opposite signs and an average hides them.
3. **Spread across replicates**, not just the mean. This is the whole reason
   for four replicates. Round one's easy tier scored 0.30 / 0.45 / 0.86 on its
   three units, so its tier means were close to meaningless.
4. **Merge errors separately** for the pair tier. Correct behaviour there is TWO
   clusters; a merge looks like excellent recall and must not be allowed to read
   as success.
5. **Interpret precision correctly (section 5ag).** 80.6% of what counts against
   precision is a resident neuron's REAL spikes, against a 3.9% chance rate.
   Counting those as real takes overall precision 0.4669 → 0.8591. So precision
   here measures **merging with resident neurons, not a noise rate** — a genuine
   isolation failure, but not the one the raw number implies. Never report
   "half of Kilosort's spikes are noise".
6. **Use the recorded site properties.** `run_v2_comparison.py --compare` prints
   a Spearman correlation of recall against `site_max_neighbor_uv`,
   `site_density`, `site_noise_uv` and `contam_pct`. This finally separates
   "hard because the neighbour is LOUD" from "hard because the neighbourhood is
   BUSY" — the open question behind the hard tier's design.
7. **Plain language.** Gil asks for plain-language summaries whenever results
   get dense. Lead with what it means, then the table.

Round one's numbers, for comparison (13 units, 3,449 spikes):
RECALL vanilla easy .9303 / collision .8746 / noisy .8037 / pair .5696 /
overall .7773. PRECISION .5383 / .5887 / .2696 / .4701 / .4669.
Only `footprint_cluster_strong` beat stock: pair recall .5696→.6644 AND
precision .4701→.5188 together, finishing essentially tied overall
(.7657/.4699). Merge errors: ZERO for every config including vanilla.

### 6.6 THE v2 BENCHMARK (what is about to be measured)
Built by `build_tiered_v2.py <rep>` for rep 0-3 →
`D:\Gil\spike_sorting_agent\hybrid_v2_rep{0,1,2,3}\`.

| | value |
|---|---|
| donor units | **33**, every one hand-labelled `good` by Gil |
| contamination | 0.0 - 9.9% |
| placements | 52 across 4 replicates (different session windows) |
| ground-truth spikes | 15,086 |
| per tier | 12 easy / 12 hard / 12 collision / 16 pair |
| quiet-tier sites | 23-70 uV largest neighbour |
| hard-tier sites | 223-466 uV largest neighbour |
| pair donor separation | 181-242 um at similarity 0.000 |
| units in >1 tier | 18 of 33 — enables WITHIN-unit tier comparison |

Tiers (the design rule: hard placements are good as long as they are
intentional and recorded):
- **easy** — quiet site, injection times guarded 75 samples from every resident
- **hard** — loud site (large resident neighbours), times still guarded
- **collision** — QUIET site, times deliberately 4-25 samples from a resident
  spike. Quiet on purpose: the challenge must be temporal separation only.
- **pair** — two donors from REMOTE probe locations placed 3 channels apart at
  a quiet site. Correct behaviour is TWO clusters.

Source selection (`select_source_units.py`): pool = Gil's manual `good`
verdicts ONLY → ≥300 spikes → donor filters (≤10% contamination, ≥3 channels
above 25% of peak) → mutual independence (≥40 um, similarity ≤0.20, no
cross-correlogram refractory dip).

### 6.7 WHERE EVERYTHING LIVES
- Code: `D:\Gil\spike_sorting_agent\post_ks_correction\`
- Git: branch `post-ks-correction`,
  https://github.com/ZurGil/striatal-spike-sorting.git (auth works, pushes fast)
- Session: `F:\Gil\Shamir\20260916_110311.rec\...\kilosort4` (210 min, 384 ch)
- Benchmark: `D:\Gil\spike_sorting_agent\hybrid_v2_rep{0,1,2,3}\`
- Kilosort being patched:
  `C:\Users\Adam\anaconda3\envs\kilosort4\Lib\site-packages\kilosort`
- Envs: `phy2_ky` (analysis, no GPU), `kilosort4` (runs Kilosort, CUDA)
- **Manual verdicts**: `outputs/manual_verdicts_20260916_110311.csv` (198 units:
  111 good, 74 mua, 13 noise), exported from the review artifact's database at
  https://claude.ai/code/artifact/74dec064-c86a-46a3-8c03-f112ee4ecb05
  (`claude.use('db')` → `db.collection('verdicts')`). **This is the only human
  judgement that exists for this session** — `cluster_group.tsv` is a
  byte-identical copy of `cluster_KSLabel.tsv` (no Phy curation), and the
  `group` column in `cluster_info.tsv` is written by an automated script.
- Review tool: `python scripts/build_review_html.py 20260916_110311`
- **Trodes ↔ Kilosort channel map: `ntrode = 1466 - ks_channel`** (verified
  against pad coordinates, not assumed). Gil reads channels in Trodes numbers.

### 6.8 THE FINDINGS THAT MATTER MOST
1. **84% of failures are misfiling, not missed detection.** Of 3,449 injected
   spikes, vanilla found 69.5%, **detected-but-misfiled 25.7%**, never detected
   4.8%. Detection-stage work (whitening, thresholds) has a ceiling of 4.8%;
   clustering-stage work addresses 25.7%. This should drive all prioritisation.
2. **Footprint-aware clustering works on closely-spaced neurons.** Pair tier
   recall AND precision both up at once, dose-responsive in the weight.
3. **It fails where spikes overlap in time** — a colliding spike's per-channel
   energy is contaminated, so weighting a corrupted descriptor hurts. Helps in
   SPACE, hurts in TIME.
4. **Amplitude normalization is actively harmful** everywhere.
5. **Timing work is close to exhausted.** Sub-sample alignment measures to 0.035
   samples RMS against ground truth (8x better than integer) and still does not
   help sorting.
6. **Kilosort does not MERGE nearby neurons here — it SHATTERS them.** Zero
   merge errors in every configuration including vanilla. This contradicts the
   premise much of the earlier footprint work was built on.
7. **Collision closeness predicts nothing** — recall is flat as the neighbouring
   spike moves from 4 to 25 samples away.

### 6.9 MISTAKES NOT TO REPEAT
- **`templates.npy` is NOT in microvolts.** Kilosort normalizes it; multiplying
  by GAIN_TO_UV gives "0.2 uV spikes". Use real averaged snippets from the raw
  file. (Its relative profile ACROSS channels is still valid — fine for
  counting footprint channels.)
- **Base rates — committed THREE times.** Any "is X near Y" statistic must be
  restricted to spatially relevant units AND print its chance rate in the same
  table. Unrestricted versions claimed 170 of 210 clusters were "fragments" of
  every unit, that 98% of random timepoints were near a spike, and that 96.2% of
  contamination was pre-existing (true answer, restricted to 60 um: 80.6%
  against 3.9% chance).
- **Kilosort's label is not a substitute for looking at the unit.** A 49-unit
  selection built on `KSLabel=='good'` + ContamPct contained 4 units Gil calls
  NOISE, one of them placed as an easy-tier source. It cuts both ways: 12 pool
  units are `KSLabel=='mua'` and the manual review rescued them.
- **A silent placement failure looks exactly like a smaller benchmark.** Four
  separate pair/tier placement bugs each produced a plausible dataset and no
  error. Assert that every tier got its full count.
- **F-ratios on fragments are partly circular** — they identify the AXIS of a
  split, not its cause.
- **Synthetic tests using identical copies flatter everything.** The
  multi-frequency ambiguity flag passed synthetically and failed on real spikes
  (AUC 0.62) because real spikes vary.
- **Measure the quantity the patch actually uses** — a coarse-stage bug was
  "diagnosed" from final output times when the patch operates on an earlier
  internal index.
- `r.shift` on a pandas row returns the DataFrame method, not the column.
- **`git add -A` from a SUBDIRECTORY stages the whole repository.** One such call
  swept a 2.6 GB `.bin` and ~2 GB of parquet/pkl into a commit; GitHub rejects
  files > 100 MB, so the push HUNG rather than erroring and looked like a slow
  network for hours. Originals preserved on `backup-before-bigfile-fix`.
- **Trodes and Kilosort cannot share this 6 GB GPU** (see 6.2).

### 6.10 WHAT HAS NOT BEEN DONE
- **Footprint-first attribution — the one open lead.** See 6.3. Score each
  candidate against every nearby unit's footprint and assign to the best,
  rather than one unit at a time. Untried.
- **Temporal whitening inside Kilosort** — built and validated on real noise
  (lag-1 0.72 → -0.09, ~3x discriminability) but never made a patch. By finding
  1 its ceiling is 4.8%, so it is the lowest-value remaining idea.
- Footprint weighting applied only to non-colliding spikes — the obvious
  refinement, blocked by having no working collision detector (AUC 0.62).
- Burst-history amplitude recovery, Huber robust estimation, learned deformation
  direction d_k, automatic split detection across a session.
- **10 units Gil has not yet reviewed** (24, 102, 126, 144, 205, 236, 280, 289,
  359, 387). Not needed for the current run; would enlarge the pool if labelled.

### 6.11 HONEST LIMITS
One session, one animal, one 120 s window per replicate (four different
windows). Patches are judged against Kilosort's thresholds and learned PC
basis, both tuned on unmodified data — a fully fair test of the underlying
ideas might require retuning `Th_universal`/`Th_learned` alongside, which has
not been done. The measurement is "does switching this flag on help as Kilosort
currently stands". Absolute precision is low across the board for the reason in
6.5 item 5; only RELATIVE comparisons between configurations are meaningful.
