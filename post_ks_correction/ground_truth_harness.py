"""
Hybrid ground truth: real noise, real spike variability, KNOWN answers.

WHY THIS EXISTS. Every number this project has produced is one configuration
compared against another. Nothing has ever been checked against a known
truth, and that gap has now cost us four times: the "new spikes" that
evaporated twice, the footprint gate reading I got backwards, and the
multi-frequency ambiguity flag that passed synthetic tests and then failed
completely on real spikes (RESEARCH_LOG 5w).

THE LESSON FROM THAT LAST FAILURE, BUILT INTO THIS DESIGN. The synthetic test
for the ambiguity flag passed with a huge margin and meant nothing, because
in that test every "clean" spike was an IDENTICAL copy of the template. The
only thing that could disturb the phase was the collision I had put there
myself. Real spikes from one neuron vary in amplitude and shape from firing
to firing, and that variation disturbs the phase just as much. So:

    A harness that injects identical template copies will flatter us
    exactly the same way. It MUST carry real spike-to-spike variability.

THE TRICK THAT MAKES THE TRUTH EXACT ANYWAY. Using real snippets seems to
destroy the ground truth, because a real snippet's own sub-sample offset is
unknown. It does not, if the question is asked as a DIFFERENCE. Take one real
snippet with unknown intrinsic offset e. Inject copies of it shifted by known
amounts d_1, d_2, ... Any estimator worth having must satisfy

    estimate(d_j) - estimate(d_0)  =  d_j - d_0

and the unknown e cancels exactly. So we get rigorous, absolute accuracy
numbers while keeping every bit of real waveform variability. No assumption
that a real spike resembles its template is needed anywhere.

WHAT THE BACKGROUND IS. Real recorded voltage from the same session and the
same channel, at times chosen so no unit within a given radius has a spike
nearby -- so the local background is genuinely quiet, while still being real
noise with its real temporal correlation structure (lag-1 ~0.72, the thing
noise_whitening.py models) rather than Gaussian white noise.

HONEST LIMIT. Injected spikes are added linearly on top of background. Real
extracellular spikes also sum linearly to good approximation, so this is
fair, but it does mean the harness cannot test anything about the recording
hardware or the spike-generation process itself -- only about what our
estimators do with a known signal in real noise.
"""
import numpy as np
from scipy.interpolate import interp1d


def find_quiet_positions(spike_times, spike_clusters, relevant_units,
                          window_start, window_end, n_positions,
                          guard_samples=120, min_separation=400, rng=None):
    """Pick sample positions whose neighbourhood contains no spike from any
    unit in `relevant_units`, so an injected spike sits on genuinely quiet
    real background.

    guard_samples    -- required clearance from any relevant unit's spike
    min_separation   -- required clearance between chosen positions, so
                        injected spikes never interfere with each other
    """
    rng = np.random.default_rng() if rng is None else rng
    mask = np.isin(spike_clusters, list(relevant_units))
    busy = np.sort(spike_times[mask])

    lo = int(window_start + guard_samples + min_separation)
    hi = int(window_end - guard_samples - min_separation)
    chosen = []
    # walk candidate positions in random order, keep those that are clear
    cand = rng.permutation(np.arange(lo, hi, max(min_separation // 2, 1)))
    for c in cand:
        if len(chosen) >= n_positions:
            break
        i = np.searchsorted(busy, c)
        near = np.inf
        if i < len(busy):
            near = min(near, abs(busy[i] - c))
        if i > 0:
            near = min(near, abs(busy[i - 1] - c))
        if near <= guard_samples:
            continue
        if chosen and np.min(np.abs(np.asarray(chosen) - c)) < min_separation:
            continue
        chosen.append(int(c))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def extract_real_snippets(trace, trace_offset, spike_samples, nt=61, nt0min=20,
                           max_snippets=None, rng=None):
    """Pull a unit's REAL spike waveforms out of the recording. These carry
    the genuine amplitude and shape variability that makes synthetic tests
    misleading -- that is the entire point of using them.

    trace_offset : the absolute sample index that trace[0] corresponds to.
    Returns (snippets, used_samples) with snippets of shape (n, nt).
    """
    rng = np.random.default_rng() if rng is None else rng
    s = np.asarray(spike_samples, dtype=np.int64)
    if max_snippets is not None and len(s) > max_snippets:
        s = s[rng.permutation(len(s))[:max_snippets]]
    out, used = [], []
    for sp in s:
        i = int(sp) - trace_offset - nt0min
        if i < 0 or i + nt > len(trace):
            continue
        out.append(np.asarray(trace[i:i + nt], dtype=np.float64))
        used.append(int(sp))
    if not out:
        return np.zeros((0, nt)), np.zeros(0, dtype=np.int64)
    return np.stack(out), np.asarray(used, dtype=np.int64)


def shift_waveform(waveform, delta, pad=8):
    """Shift a waveform by `delta` samples (positive = later) using cubic
    interpolation, with edge padding so the interpolation never extrapolates
    into the region we keep.

    This is the operation that defines the ground truth, so it has to be
    accurate: `test_shift_waveform_is_accurate_and_invertible` checks that
    shifting by +d then -d returns the original.
    """
    w = np.asarray(waveform, dtype=float)
    n = len(w)
    xp = np.arange(-pad, n + pad)
    wp = np.concatenate([np.full(pad, w[0]), w, np.full(pad, w[-1])])
    ip = interp1d(xp, wp, kind="cubic", bounds_error=False, fill_value=0.0)
    return ip(np.arange(n) - delta)


def inject(trace, waveform, at_index, nt0min=20, amplitude_scale=1.0):
    """Add one waveform into a trace IN PLACE, so that the waveform's
    `nt0min` sample lands exactly on `at_index`. Returns True if it fitted.

    Linear addition is the right model: extracellular potentials from
    different cells superpose. See the module docstring's honest-limit note.
    """
    w = np.asarray(waveform, dtype=float) * float(amplitude_scale)
    i = int(at_index) - int(nt0min)
    if i < 0 or i + len(w) > len(trace):
        return False
    trace[i:i + len(w)] += w
    return True


def build_hybrid_trace(background, positions, snippets, shifts,
                        nt0min=20, amplitude_scales=None):
    """Assemble one hybrid trace plus its truth table.

    background       -- real voltage, COPIED (not modified)
    positions        -- integer sample index for each injected spike
    snippets         -- (n, nt) real waveforms, one per position
    shifts           -- (n,) known sub-sample shift applied to each
    amplitude_scales -- optional (n,) extra scaling

    Returns (trace, truth) where truth is a list of dicts with the position,
    the known shift, which snippet was used, and the scale -- everything
    needed to grade an estimator.
    """
    trace = np.array(background, dtype=np.float64, copy=True)
    if amplitude_scales is None:
        amplitude_scales = np.ones(len(positions))
    truth = []
    for k, (p, d, sc) in enumerate(zip(positions, shifts, amplitude_scales)):
        w = shift_waveform(snippets[k], float(d))
        ok = inject(trace, w, int(p), nt0min=nt0min, amplitude_scale=float(sc))
        if ok:
            truth.append(dict(index=k, position=int(p), true_shift=float(d),
                              amplitude_scale=float(sc)))
    return trace, truth


def build_shift_ladder(background, position, snippet, shifts, nt0min=20,
                        separation=400):
    """The DIFFERENCE design, which is what makes real snippets usable as
    ground truth.

    One real snippet, injected repeatedly at `shifts`, each copy far enough
    from the next that they cannot interfere. The snippet's own unknown
    intrinsic offset `e` is identical in every copy, so

        estimate(d_j) - estimate(d_0) = d_j - d_0

    holds regardless of what `e` is, and an estimator can be graded
    absolutely without ever knowing it.

    Returns (trace, positions, shifts) where positions[j] holds the copy
    shifted by shifts[j].
    """
    shifts = np.asarray(shifts, dtype=float)
    positions = np.asarray([int(position) + j * int(separation)
                            for j in range(len(shifts))], dtype=np.int64)
    trace = np.array(background, dtype=np.float64, copy=True)
    kept_p, kept_d = [], []
    for p, d in zip(positions, shifts):
        w = shift_waveform(snippet, float(d))
        if inject(trace, w, int(p), nt0min=nt0min):
            kept_p.append(int(p))
            kept_d.append(float(d))
    return trace, np.asarray(kept_p, dtype=np.int64), np.asarray(kept_d)


def grade_shift_ladder(estimates, true_shifts, reference_index=None):
    """Grade a ladder of estimates against the known shifts, cancelling the
    snippet's unknown intrinsic offset.

    reference_index : which rung to treat as the baseline. Default is the
    rung whose true shift is closest to zero.

    Returns dict with:
      errors        -- (n,) per-rung error after cancelling the offset
      rms           -- root-mean-square error in samples
      max_abs       -- worst single error
      slope         -- regression slope of estimate on true shift. 1.0 means
                       the estimator tracks real shifts at the right GAIN;
                       a slope of e.g. 0.6 means it systematically
                       under-corrects, which an RMS alone would hide.
      implied_bias  -- the recovered intrinsic offset (diagnostic only)
    """
    est = np.asarray(estimates, dtype=float)
    tru = np.asarray(true_shifts, dtype=float)
    ok = np.isfinite(est) & np.isfinite(tru)
    est, tru = est[ok], tru[ok]
    if len(est) < 2:
        return dict(errors=np.array([]), rms=np.nan, max_abs=np.nan,
                    slope=np.nan, implied_bias=np.nan, n=len(est))
    j = (int(np.argmin(np.abs(tru))) if reference_index is None
         else int(reference_index))
    err = (est - est[j]) - (tru - tru[j])
    slope = float(np.polyfit(tru, est, 1)[0])
    return dict(errors=err, rms=float(np.sqrt(np.mean(err ** 2))),
                max_abs=float(np.max(np.abs(err))), slope=slope,
                implied_bias=float(np.mean(est - tru)), n=len(est))


def match_detections(detected_samples, true_positions, tolerance=5):
    """Greedy one-to-one matching of detections to injected spikes.

    One-to-one matters: without it, a detector that fires three times around
    one true spike would be credited with three hits.

    Returns dict(n_true, n_detected, n_hit, n_missed, n_false_positive,
                 recall, precision, matched pairs).
    """
    det = np.sort(np.asarray(detected_samples, dtype=np.int64))
    tru = np.sort(np.asarray(true_positions, dtype=np.int64))
    used_det = np.zeros(len(det), dtype=bool)
    pairs = []
    for t in tru:
        if len(det) == 0:
            break
        d = np.abs(det - t)
        d[used_det] = np.iinfo(np.int64).max
        i = int(np.argmin(d))
        if d[i] <= tolerance:
            used_det[i] = True
            pairs.append((int(t), int(det[i])))
    n_hit = len(pairs)
    n_true, n_det = len(tru), len(det)
    return dict(n_true=n_true, n_detected=n_det, n_hit=n_hit,
                n_missed=n_true - n_hit,
                n_false_positive=n_det - n_hit,
                recall=(n_hit / n_true) if n_true else np.nan,
                precision=(n_hit / n_det) if n_det else np.nan,
                pairs=pairs)
