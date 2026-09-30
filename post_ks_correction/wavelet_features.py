"""
Pillar 1b -- complex wavelet ("phase-tolerant matched oscillation detector").

WHAT THIS IS (concrete, not just formula):
A wavelet here is a short, artificial oscillating test-waveform -- like a
brief burst of a sine wave that fades in and fades out -- that we slide
along the raw voltage trace and compare against, the same way a matched
filter compares the trace against a copy of the spike's own shape. The
difference: this probe oscillates at a chosen frequency rather than trying
to mimic the spike's exact waveform, and it is COMPLEX-valued, meaning it
is a cosine-shaped probe and a sine-shaped probe (90 degrees offset from
each other) bundled together into one complex number at every time point.

    psi(t) = exp(i * 2*pi*f0*t) * window(t)

  f0      -- probe frequency in Hz (how fast the test-oscillation wiggles)
  window  -- a smooth bump (Gaussian here) that fades psi to ~0 away from
             its center, so the probe only "listens" to a short local
             stretch of the trace (this is what makes it "localized in
             time", as opposed to an infinite sine wave)
  i       -- the imaginary unit; exp(i*theta) = cos(theta) + i*sin(theta),
             so psi really is (cosine probe) + i*(sine probe) in one object

Sliding it along the trace and reading off two numbers per candidate time:

    W(t0) = sum_tau  x(tau) * conj(psi(tau - t0))

  |W(t0)|         -- MAGNITUDE: "how strongly does the trace look like this
                     oscillation here", independent of exact alignment.
                     Shift-tolerant -- good for coarse "roughly here"
                     localization, unlike a rigid matched filter whose score
                     can nearly flip sign from a single sample of misalignment.
  angle(W(t0))    -- PHASE: exactly where inside one oscillation cycle the
                     best match sits. If the true spike time is off from t0
                     by a small amount delta (samples), the phase rotates by
                     a PREDICTABLE amount:
                         delta_phi = -omega * delta,   omega = 2*pi*f0
                     so you can invert this: delta = -delta_phi / omega, and
                     use it to correct your timing guess to sub-sample
                     precision. This ONLY works within about half an
                     oscillation period of the true time -- phase wraps
                     around every 2*pi, so beyond that range you cannot tell
                     "off by a little" from "off by a little the other way".
                     Use magnitude for coarse localization FIRST, phase only
                     for FINE correction once already close.

Multiple scales (f0, window width) are needed because of a hard tradeoff: a
narrow time-window pins down WHEN precisely but is vague about WHAT
frequency; a wide window is the opposite (this is the same time-frequency
tradeoff as in a spectrogram / short-time Fourier transform -- no single
window width is simultaneously good at both).

RECOMMENDED USAGE: call coarse_then_fine_shift, not sub_sample_shift_from_phase
directly. Phase alone only works within the narrow half-cycle window described
above (confirmed: mean error jumps from ~0.008ms to ~0.51ms crossing that
boundary, tested +/-20 samples). coarse_then_fine_shift removes that limit by
actually doing what this docstring says -- coarse whole-sample localization
first, phase-based fine correction second -- which was described here from
the start but never wired together and tested end-to-end until this project's
conversation history pushed on it. The lower-level pieces
(wavelet_transform_at, sub_sample_shift_from_phase, calibrate_reference_phase)
remain here as building blocks / for diagnostics, not as the recommended
top-level call.
"""
import numpy as np


def make_morlet(f0_hz, fs, n_cycles=5.0):
    """Build one complex Gaussian-windowed oscillating probe (a "Morlet
    wavelet"): psi(t) = exp(i*2*pi*f0*t) * gaussian_window(t).

    f0_hz    -- probe frequency in Hz (pick this near the dominant
                frequency content of a real extracellular spike's fast
                phase, typically ~1-3 kHz for the sharp Na+ deflection)
    fs       -- sample rate in Hz
    n_cycles -- how many oscillation cycles fit inside the window's
                effective width (this is the time-vs-frequency-resolution
                knob: more cycles = better frequency selectivity, worse
                timing precision, and vice versa)

    Returns (t_samples, psi) -- psi is complex, same length as t_samples,
    centered at t=0.
    """
    period_s = 1.0 / f0_hz
    half_width_s = n_cycles * period_s / 2.0
    half_width_samples = int(np.ceil(half_width_s * fs))
    t_samples = np.arange(-half_width_samples, half_width_samples + 1)
    t_s = t_samples / fs
    sigma_s = half_width_s / 2.5  # Gaussian envelope: ~2.5 sigma to the edge
    window = np.exp(-0.5 * (t_s / sigma_s) ** 2)
    psi = np.exp(1j * 2 * np.pi * f0_hz * t_s) * window
    return t_samples, psi


def wavelet_transform_at(x, psi, t0_index):
    """W(t0) = sum_tau x(tau) * conj(psi(tau - t0)) at ONE candidate index
    t0_index into x. x and psi both real-sample-indexed 1D arrays; psi is
    centered (its own index 0 is the middle of the returned array from
    make_morlet). Returns one complex number.

    Implemented directly (not via np.convolve) so the "slide psi along x,
    conjugate-multiply, sum" definition above is literally what runs --
    this is the reference/teaching implementation. A scan across many
    t0_index values should use scan_wavelet_transform below (FFT-based),
    not a Python loop over this function, for speed.
    """
    half = len(psi) // 2
    lo, hi = t0_index - half, t0_index - half + len(psi)
    if lo < 0 or hi > len(x):
        return np.nan + 1j * np.nan
    segment = x[lo:hi]
    return np.sum(segment * np.conj(psi))


def scan_wavelet_transform(x, psi):
    """W(t0) for EVERY t0 in x, via FFT-based correlation (equivalent to
    sliding psi across x and conjugate-multiplying+summing at every
    position, just done efficiently). Returns a complex array the same
    length as x (edges where psi extends past x are set to nan+nan*1j).
    """
    n = len(x)
    half = len(psi) // 2
    # correlate(x, psi) with 'full' mode, then slice to 'same' alignment,
    # matches the direct sliding-window definition above at every valid t0
    corr = np.correlate(x.astype(np.complex128), psi, mode="full")
    # np.correlate with complex args already conjugates the second argument
    start = len(psi) - 1 - half
    out = corr[start:start + n]
    out = out.astype(np.complex128)
    out[:half] = np.nan + 1j * np.nan
    out[n - (len(psi) - half - 1):] = np.nan + 1j * np.nan
    return out


def calibrate_reference_phase(s0, psi, fs, align_index=None):
    """Phase is only exactly 0 at zero shift for a PURE TONE probe compared
    against itself. A real spike is not a pure tone -- it has its own
    asymmetric shape -- so W(t0) at the correct, unshifted alignment has
    some fixed nonzero phase that depends on the template's shape, not on
    any real misalignment. Left uncorrected, this shows up as a constant
    offset error in every recovered shift (this is exactly what the first
    version of this module's self-test found: a consistent ~-0.3 to -0.5
    sample bias across every test case).

    Fix: compute this "shape phase" ONCE per (template, probe) pair, by
    running the unshifted template s0 through the exact same wavelet
    transform, and store it as a reference to subtract from every future
    phase measurement on a real spike from this unit.

    align_index: the sample index WITHIN s0 that all shift measurements are
    relative to -- i.e. Kilosort's own nt0min (the detected-spike-time
    sample), NOT necessarily the array midpoint. This MUST match whatever
    convention the caller uses when later measuring a real spike's shift,
    or the calibration silently introduces a large, wrong constant offset
    (this happened during development here: using n//2 instead of the
    real nt0min produced a ~10-sample spurious offset -- caught by
    comparing the calibrated-vs-uncalibrated recovery plot against the
    perfect-recovery diagonal, not by inspection of the code).
    Defaults to n//2 only if you have verified that IS your alignment
    convention; pass it explicitly otherwise.
    """
    n = len(s0)
    if align_index is None:
        align_index = n // 2
    half = len(psi) // 2
    pad = half + 5
    trace = np.zeros(n + 2 * pad)
    center = pad + align_index
    trace[pad:pad + n] = s0
    w_ref = wavelet_transform_at(trace, psi, center)
    return float(np.angle(w_ref))


def sub_sample_shift_from_phase(phase_at_t0, f0_hz, fs, reference_phase=0.0):
    """Invert delta_phi = -omega*delta for the sub-sample timing correction
    delta (in samples), AFTER removing the template's own zero-shift phase
    offset (reference_phase, from calibrate_reference_phase -- pass 0.0 only
    if you have already verified your probe/template pair has ~0 shape
    phase, which is not true in general).

    phase_at_t0 is angle(W(t0)) in radians as returned by np.angle. The
    subtraction is done via complex division (not plain subtraction) so the
    result is correctly re-wrapped to (-pi, pi] -- plain subtraction of two
    angles near +-pi can jump by 2*pi and silently corrupt the estimate.

    This function does NOT resolve the half-cycle ambiguity itself (see
    module docstring): only call it once a magnitude-based coarse
    localization has you within half an oscillation period of the true
    spike time."""
    delta_phi = np.angle(np.exp(1j * (phase_at_t0 - reference_phase)))
    omega = 2 * np.pi * f0_hz
    delta_samples = -delta_phi / omega * fs
    return delta_samples


def _golden_section_max(func, lo, hi, tol_hz=5.0, max_iter=60):
    """Maximize a 1-D function on [lo, hi] by golden-section search.

    ONLY valid when the function is unimodal on that interval -- which is
    why this is never called on the full frequency range. Measured on 20
    real units, f0*|W(f0)| has a median of 2 local maxima across 300-8000Hz
    (max 5), and only 10/20 units are strictly unimodal, so a bracket-and-
    narrow search run over the whole range would sometimes converge onto
    the wrong peak. It is called only to refine INSIDE a bracket the coarse
    grid has already localized the global peak to.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(max_iter):
        if b - a < tol_hz:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = func(d)
    return (a + b) / 2.0


def select_probe_frequency(template, fs, n_cycles=3.0, nt0min=None,
                            f_lo=300.0, f_hi=8000.0, n_grid=60, criterion="timing",
                            refine=True, warn_at_edge=True):
    """Choose this unit's probe frequency f0.

    criterion="strength" -- argmax |W(f0)|. Picks whichever frequency the
        template matches most strongly. This was the rule used everywhere in
        this project until it was found to be actively harmful, and WHY it
        is harmful is worth stating, because it is not obvious:

        The fine stage converts a measured phase into a time shift via
        delta = -delta_phi/omega, i.e.

            timing_error  ~=  phase_error * (fs/f0) / 360   [samples per degree]

        so the cycle length (fs/f0) is a straight multiplier on any phase
        measurement error. At f0=800Hz one cycle spans ~37 samples, so a
        10-degree phase error -- easily produced by ordinary noise on a real
        spike -- becomes a FULL SAMPLE of spurious shift. Measured on real
        unit-440 spikes with this criterion: median spurious shift 0.72
        samples, 34% of known-correctly-placed spikes moved by more than a
        sample, and template fit got WORSE than doing no alignment at all
        (mean R2 0.780 -> 0.737).

    criterion="timing" (default) -- argmax f0*|W(f0)|. Minimizes the
        expected TIMING error rather than maximizing match strength. The
        phase error scales roughly as 1/|W(f0)| (less signal at that
        frequency, noisier phase), so

            timing_error  ~  fs / (f0 * |W(f0)|)

        which is minimized by maximizing f0*|W(f0)|. This is a genuine
        optimum, not "pick a higher frequency": push f0 too high and |W|
        collapses, making the phase noisier faster than the shorter cycle
        helps. Measured on the same real spikes: median spurious shift
        drops 0.72 -> 0.33 samples, spikes moved >1 sample drop 34% -> 2.8%,
        and alignment goes from harmful to mildly helpful
        (mean R2 0.780 -> 0.787).

        Side benefit: |W(f0)| alone is broad and flat across low
        frequencies, so argmax lands in the same plateau for many different
        units -- which is why several units with visibly different waveform
        widths kept selecting the identical f0 earlier in this project.
        Weighting by f0 sharpens the peak and gives genuinely per-unit
        frequencies.

    SCAN RANGE MATTERS, and the old default was silently truncating.
    f_hi was 4000Hz (an arbitrary value inherited from early in this
    project) until it was measured across 20 real units: under the timing
    criterion 6/20 units selected the 4000Hz endpoint itself, i.e. their
    real optimum was above the bound and the "choice" was just the edge.
    Widening to 8000Hz put 0/20 at the edge and left 80% of units'
    selections unchanged -- only the truncated minority moved (to
    4033-5589Hz). Those higher frequencies were verified to genuinely align
    better on real spikes, not just score better: unit 39's mean fit went
    0.347 (at 1500Hz) -> 0.465 (4000Hz) -> 0.473 (5589Hz, plateau), against
    a no-alignment baseline of 0.459 -- i.e. below the bound alignment was
    HURTING and above it it helps.

    refine: after the coarse grid locates the global peak, refine inside
        the bracketing interval by golden-section. The grid alone resolves
        f0 only to its own spacing (~130Hz at these defaults). Refinement is
        confined to the bracket precisely because the function is NOT
        globally unimodal (see _golden_section_max).

    warn_at_edge: print a warning if the selection lands on the first or
        last grid point, which means the scan range -- not the data -- is
        determining the answer.

    Returns (f0_hz, f_grid, scores) -- scores is the criterion actually
    maximized on the coarse grid, exposed so a caller can inspect how
    peaked the choice is.
    """
    template = np.asarray(template, dtype=np.float64)
    n = len(template)
    if nt0min is None:
        nt0min = n // 2
    f_grid = np.linspace(f_lo, f_hi, n_grid)
    pad = int(np.ceil(n_cycles * fs / (2 * f_grid.min()))) + 20
    trace = np.zeros(n + 2 * pad)
    center = pad + nt0min
    trace[center - nt0min: center - nt0min + n] = template

    def score_at(f):
        mag = abs(wavelet_transform_at(trace, make_morlet(f, fs, n_cycles)[1], center))
        if criterion == "strength":
            return mag
        if criterion == "timing":
            return mag * f
        raise ValueError(f"unknown criterion {criterion!r} (use 'timing' or 'strength')")

    scores = np.array([score_at(f) for f in f_grid])
    i = int(np.nanargmax(scores))
    f0 = float(f_grid[i])

    if warn_at_edge and (i == 0 or i == len(f_grid) - 1):
        print(f"  [select_probe_frequency] WARNING: selected f0={f0:.0f}Hz is at the edge of the "
              f"scan range [{f_lo:.0f}, {f_hi:.0f}] -- the range, not the data, is setting this. "
              f"Widen it.")

    if refine and 0 < i < len(f_grid) - 1:
        f0 = float(_golden_section_max(score_at, f_grid[i - 1], f_grid[i + 1]))

    return f0, f_grid, scores


def coarse_then_fine_shift(trace, center_index, template, psi, f0_hz, fs,
                            reference_phase=0.0, search_radius=25, nt0min=None,
                            coarse_step=None, safety_fraction=0.5):
    """THE RECOMMENDED way to get a spike's sub-sample timing -- use this,
    not sub_sample_shift_from_phase alone.

    Phase-only correction (sub_sample_shift_from_phase by itself) is only
    trustworthy within +/- half an oscillation cycle of the true spike
    time (a few tenths of a millisecond -- see the accuracy-vs-frequency
    numbers worked out in this project's conversation history). Outside
    that narrow window it doesn't degrade gracefully, it silently returns
    a completely wrong answer (confirmed: mean error jumped from ~0.008ms
    to ~0.51ms crossing that boundary, tested across a +/-20 sample true
    range). This function removes that limitation by doing what the
    module docstring always said to do, but which was never actually
    wired together and tested end-to-end until prompted to in this
    conversation: coarse whole-sample localization FIRST (which has no
    wraparound ambiguity, because it only ever compares whole numbers),
    THEN phase-based fine correction on the small leftover fractional
    part -- which is now guaranteed to be within the safe window, because
    the coarse step already got you to within one sample.

    Measured performance (synthetic spike, 5% noise, +/-20 sample true
    shift range, n_cycles=3 probe): mean error 0.0020ms inside the old
    phase-only safe zone, 0.0039ms outside it -- both far better than
    phase-only ever achieved, and stable across the WHOLE tested range,
    not just a narrow sliver of it.

    KNOWN OPEN QUESTION, not yet checked: behavior at low SNR. The coarse
    matched-filter search could in principle lock onto the wrong integer
    sample if a noise fluctuation scores higher than the true position by
    chance -- a different, more gradual failure mode than phase-only's
    hard cliff, but not yet measured. Worth checking before trusting this
    on real, heavily degraded burst-tail spikes specifically.

    Parameters
    ----------
    trace : full raw (or already-filtered) voltage array containing the
        spike, long enough that center_index +/- (search_radius + len(psi)//2)
        stays in-bounds.
    center_index : the nominal/candidate detected-spike sample index within
        trace (e.g. Kilosort's own reported spike time) -- the coarse
        search scans integer offsets around this.
    template : the unit's own rested waveform snippet (same convention as
        nuisance_model.py's s0 -- typically length 61, aligned so sample
        nt0min is the detected-spike-time sample).
    psi : probe array, from make_morlet.
    reference_phase : from calibrate_reference_phase, computed once per
        (template, psi) pair -- see that function's docstring.
    search_radius : how many samples either side of center_index the
        coarse stage searches. Sets the outer range this function can
        correct (this is now a cheap, adjustable design choice, not a
        physics-derived limit the way the old phase-only safe zone was) --
        but does not make it unlimited: search much wider than the real
        spike snippet and the matched-filter score stops carrying real
        signal.
    nt0min : alignment sample within template (defaults to len(template)//2
        if not given -- pass this dataset's real Kilosort nt0min explicitly
        whenever known, the same way calibrate_reference_phase requires it).
    coarse_step : spacing (samples) between candidate shifts the coarse
        stage actually tries. Defaults (see safety_fraction below) to a
        SPARSE step derived from this unit's own half-cycle safe-zone
        width, f_s/(2*f0_hz) -- NOT exhaustive (step=1) checking of every
        sample. This was a real, verified-on-real-data optimization added
        in this project's conversation history: the coarse stage's only
        job is landing the fine stage's input somewhere inside the safe
        zone, not finding the exact best integer sample -- so the step
        size can be as large as that zone's own width and still hand the
        fine stage a trustworthy starting point. Measured on real spikes,
        two units at very different f0: sparse vs. exhaustive coarse
        picks sometimes land on genuinely different integers (differing
        by several samples), but the FINAL total_shift after the fine
        stage still agrees to within ~0.15 samples either way, for an
        8.5x speedup at low f0 (739Hz) and 2.0x at high f0 (2808Hz) --
        low-f0 units have a wider safe zone, so they tolerate bigger
        jumps and see a bigger speedup; high-f0 units need finer steps
        and see less speedup. Pass coarse_step=1 to force the old
        exhaustive behavior.
    safety_fraction : only used to compute the default coarse_step -- the
        step is safety_fraction times the half-cycle width, so the
        worst-case leftover error (up to one full step) still lands
        comfortably inside the safe zone rather than right at its
        less-accurate edge. 0.5 is what was actually tested; not
        re-verified at other values.

    Returns
    -------
    dict(coarse_shift, fine_shift, total_shift, coarse_score)
    total_shift is the number to use; coarse_shift/fine_shift/coarse_score
    are exposed for diagnostics (e.g. flagging a suspiciously weak
    coarse_score as a possible low-SNR failure per the open question above).
    """
    n = len(template)
    if nt0min is None:
        nt0min = n // 2
    template_norm = template / (np.linalg.norm(template) + 1e-12)

    if coarse_step is None:
        half_cycle_samples = fs / (2 * f0_hz)
        coarse_step = max(1, int(half_cycle_samples * safety_fraction))

    candidate_shifts = list(range(-search_radius, search_radius + 1, coarse_step))
    if candidate_shifts[-1] != search_radius:
        candidate_shifts.append(search_radius)  # always cover the requested edge

    best_score, best_shift = -np.inf, 0
    for candidate_shift in candidate_shifts:
        lo = center_index - nt0min + candidate_shift
        hi = lo + n
        if lo < 0 or hi > len(trace):
            continue
        window = trace[lo:hi]
        score = np.dot(window, template_norm)
        if score > best_score:
            best_score, best_shift = score, candidate_shift

    fine_center = center_index + best_shift
    half = len(psi) // 2
    if fine_center - half < 0 or fine_center + half >= len(trace):
        # can't evaluate the probe here -- return coarse-only rather than crash
        return dict(coarse_shift=best_shift, fine_shift=0.0,
                     total_shift=float(best_shift), coarse_score=best_score)

    w = wavelet_transform_at(trace, psi, fine_center)
    fine_shift = sub_sample_shift_from_phase(np.angle(w), f0_hz, fs, reference_phase=reference_phase)
    return dict(coarse_shift=best_shift, fine_shift=fine_shift,
                total_shift=best_shift + fine_shift, coarse_score=best_score)


if __name__ == "__main__":
    # This used to hold an ad-hoc printed self-test (shift-tolerance,
    # phase recovery, half-cycle wraparound, narrow-vs-wide localization).
    # It's been superseded by tests.py, which checks the same claims (and
    # more, including coarse_then_fine_shift) as permanent pass/fail
    # assertions instead of numbers someone has to eyeball -- see that
    # file's own docstring for why. Run that instead:
    print("See tests.py for this module's validated behavior:")
    print("  python tests.py")
