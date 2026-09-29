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
