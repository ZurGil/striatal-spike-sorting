"""
Pillar 2 -- nuisance-aware identity: treat amplitude/timing/burst-broadening
as legitimate states of one neuron, not identity differences.

Model: m_k(t; a, tau, beta) = a * s_k0((t - tau) / (1 + beta))
  a     -- amplitude scale (burst-decrement / facilitation)
  tau   -- sub-sample timing offset
  beta  -- waveform stretch (burst-related broadening)

First-order Taylor expansion around tau=0, beta=0 (small-shift regime) gives
a fixed 3-vector linear basis:

  m_k(t) ~= a*s_k0(t) - a*tau*s_k0'(t) + a*beta*q_k(t)

  where s_k0' is the ordinary time-derivative of the rested template, and
  q_k(t) = -t * s_k0'(t) (chain rule on the stretch term, evaluated at
  beta=0). Fitting (a, tau, beta) per spike is then ordinary linear
  regression of the observed snippet against the fixed design matrix
  [s_k0, s_k0', q_k] -- fast, closed-form, stable even at low SNR, unlike
  a direct nonlinear fit of the original model.

Regressor geometry (all three pairs now checked numerically -- see
tests.py):
  <s_k0, s_k0'> ~= 0   (integral of a total derivative; true whenever the
                         waveform starts/ends near baseline)
  <s_k0, q_k>   ~= 0.5 * energy(s_k0)   (via integration by parts on the
                         q_k = -t*s_k0' term -- NOT orthogonal, and joint
                         regression handles that correlation correctly;
                         do not treat these as independent votes)

  *** SEVERE, KNOWN PROBLEM, found via a real-data integration test in
  this project's conversation history: <s_k0', q_k> is close to
  PARALLEL, not orthogonal -- cos(angle) ~= 0.91 (only ~24 degrees apart)
  for this project's real template shape. s_k0' drives tau, q_k drives
  beta, so this near-collinearity means the regression CANNOT reliably
  separate timing shift from stretch: noise or any real model-misfit
  gets ambiguously split between the tau and beta coefficients. This was
  discovered because tau and beta came out strongly correlated (r=0.77)
  on real burst spikes -- and, critically, EVEN MORE strongly correlated
  (r=0.97) on a control sample of ISOLATED, non-burst, fully-rested
  spikes from the same unit, which should show ~no real stretch at all.
  That isolated-spike result is what rules out "real burst physiology"
  as the (sole) explanation and points at this basis geometry instead.
  CONSEQUENCE: individual tau and beta values from fit_nuisance should
  NOT be trusted as cleanly separated quantities until this is fixed
  (e.g. by re-orthogonalizing q_k against s_k0' via Gram-Schmidt, or by
  reparameterizing). The amplitude a and the overall fit quality
  (r_squared) are NOT implicated by this specific problem and remain
  usable as-is. NOT YET FIXED -- next concrete task for this module.
  ***

Recovering (a, tau, beta) from the fitted coefficients (c0, c1, c2) of
[s_k0, s_k0', q_k]:
  a    = c0
  tau  = -c1 / c0
  beta =  c2 / c0
(undefined / unstable as c0 -> 0 -- see MIN_AMPLITUDE_FOR_TAU_BETA below)

Deliberately NOT included yet (per design doc, deferred to a follow-up):
a 4th learned-deformation vector d_k. If ever added, it must be orthogonal
to span{s_k0, s_k0', q_k} -- which a least-squares residual already is, for
free -- and only fit from moderate/high-SNR burst followers whose
reproducibility has been checked across independent held-out burst sets.
"""
import numpy as np

MIN_AMPLITUDE_FOR_TAU_BETA = 1e-6  # below this, a is noise-dominated; tau/beta undefined


def build_basis(s0, dt=1.0):
    """s0: 1D array, the rested template on one channel (or concatenated
    across a footprint of channels -- the math is agnostic to that, as long
    as s0/s0'/q are built consistently over the same sample axis).

    dt: sample spacing (1.0 = basis in sample units; pass 1/fs for seconds).
    Sample units are fine and preferred internally -- tau then comes out in
    samples, convert to ms only when reporting.

    Returns (s0, s0_prime, q) as three arrays of the same length as s0.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    n = len(s0)
    t = (np.arange(n) - n // 2) * dt  # centered time axis; origin placement
    # only shifts what tau=0 means, not the math -- callers should center on
    # the template's own alignment sample (nt0min) for tau to mean "offset
    # from Kilosort's detected spike time".
    s0_prime = np.gradient(s0, dt)
    q = -t * s0_prime
    return s0, s0_prime, q


def check_basis_geometry(s0, s0_prime, q):
    """Numerically verify the two orthogonality/overlap claims above.
    Returns a dict of measured vs. predicted values -- call this once per
    new template shape as a sanity check, not per spike."""
    energy_s0 = float(np.dot(s0, s0))
    ortho_s0_sprime = float(np.dot(s0, s0_prime))
    overlap_s0_q = float(np.dot(s0, q))
    predicted_overlap = 0.5 * energy_s0
    return dict(
        energy_s0=energy_s0,
        s0_dot_sprime=ortho_s0_sprime,
        s0_dot_sprime_relative=ortho_s0_sprime / energy_s0 if energy_s0 > 0 else np.nan,
        s0_dot_q_measured=overlap_s0_q,
        s0_dot_q_predicted=predicted_overlap,
        s0_dot_q_relative_error=(
            abs(overlap_s0_q - predicted_overlap) / predicted_overlap
            if predicted_overlap != 0 else np.nan),
    )


def fit_nuisance(x, s0, s0_prime, q):
    """Fit one observed spike snippet x (same length as s0) against the
    fixed 3-vector basis [s0, s0_prime, q] via ordinary least squares.

    Returns dict with:
      a, tau, beta     -- recovered nuisance parameters (tau, in whatever
                           units dt was given to build_basis; NaN if
                           |a| < MIN_AMPLITUDE_FOR_TAU_BETA)
      coeffs            -- raw (c0, c1, c2) regression coefficients
      residual          -- x - fit, same length as x
      r_squared         -- fraction of x's variance explained by the fit
                           (a poor fit here means this spike doesn't belong
                           to this family at all, regardless of the fitted
                           values -- check this before trusting a/tau/beta)
    """
    x = np.asarray(x, dtype=np.float64)
    design = np.column_stack([s0, s0_prime, q])
    coeffs, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    c0, c1, c2 = coeffs
    fit = design @ coeffs
    residual = x - fit

    ss_res = float(np.dot(residual, residual))
    x_centered = x - x.mean()
    ss_tot = float(np.dot(x_centered, x_centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if abs(c0) < MIN_AMPLITUDE_FOR_TAU_BETA:
        tau, beta = np.nan, np.nan
    else:
        tau = -c1 / c0
        beta = c2 / c0

    return dict(a=float(c0), tau=float(tau), beta=float(beta),
                coeffs=tuple(float(c) for c in coeffs),
                residual=residual, r_squared=r_squared)


def fit_nuisance_prealigned(x, s0, q):
    """THE RECOMMENDED fit when the snippet has ALREADY been sub-sample
    aligned by an independent method (pillar 1b's coarse_then_fine_shift)
    before this is called -- fits only (a, beta), 2 parameters, dropping
    s0_prime (and therefore tau) from the regression entirely.

    WHY: fit_nuisance's 3-vector basis [s0, s0_prime, q] has a severe,
    unfixable-by-reformulation problem -- s0_prime (drives tau) and q
    (drives beta) are nearly PARALLEL for real spike shapes (checked on 3
    real units in this project's conversation history: 16-30 degrees
    apart, cos 0.86-0.96, never near the 90 degrees needed for clean
    separation). This happens because <s0_prime, q> = -integral(t *
    s0_prime(t)^2 dt), which is only ~0 if the template's derivative
    ENERGY is symmetric in time around the alignment point -- real spikes
    aren't (fast rise, slower fall), so this is a near-unavoidable
    property of realistic waveforms, not a fixable quirk of one template.
    Gram-Schmidt-style re-orthogonalization does NOT fix this: done
    correctly (with proper back-substitution to recover the original
    parameters), it reproduces the EXACT same fitted numbers as the
    original 3-vector regression, because ordinary least squares has one
    unique best answer regardless of which numerically-equivalent method
    computes it. The ambiguity is in what the DATA can distinguish (a
    long, thin ridge of near-equally-good (tau,beta) pairs), not in the
    arithmetic used to find a point on that ridge.

    Discovered via: tau and beta came out correlated on real burst spikes
    (r=0.77) -- AND MORE STRONGLY on a control sample of isolated,
    non-burst, fully-rested spikes from the same unit (r=0.97), which
    should show ~no real stretch. That ruled out real burst physiology as
    the explanation and pointed at this basis geometry instead.

    THE FIX: pillar 1b already provides an independent, validated timing
    estimate via a completely different principle (oscillation phase, not
    template-derivative regression) -- it doesn't share this collinearity
    at all. So: align the snippet with coarse_then_fine_shift first (as
    the integration pipeline already does), then fit ONLY (a, beta) here,
    using a 2-vector basis with s0_prime dropped entirely. s0 and q still
    overlap somewhat (<s0,q> ~= 0.5*energy(s0), established earlier) but
    that is a real, quantified, MODERATE overlap -- nowhere near the 91%+
    collinearity that was actually causing the problem.

    Returns dict with a, beta, coeffs=(c0,c2), residual, r_squared. No tau
    (by design -- supply it externally from pillar 1b instead of asking
    this fit to re-derive it).
    """
    x = np.asarray(x, dtype=np.float64)
    design = np.column_stack([s0, q])
    coeffs, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    c0, c2 = coeffs
    fit = design @ coeffs
    residual = x - fit

    ss_res = float(np.dot(residual, residual))
    x_centered = x - x.mean()
    ss_tot = float(np.dot(x_centered, x_centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    beta = c2 / c0 if abs(c0) >= MIN_AMPLITUDE_FOR_TAU_BETA else np.nan

    return dict(a=float(c0), beta=float(beta), coeffs=(float(c0), float(c2)),
                residual=residual, r_squared=r_squared)


def synthesize_deformed_spike(s0, a=1.0, tau=0.0, beta=0.0, dt=1.0, noise_sigma=0.0, rng=None):
    """Ground-truth generator for validation: apply the FULL (non-linearized)
    model m(t) = a*s0((t-tau)/(1+beta)) via interpolation, not the linear
    approximation -- so fitting this and recovering (a,tau,beta) is a
    genuine test of the linearization's validity, not a tautology."""
    rng = rng or np.random.default_rng(0)
    s0 = np.asarray(s0, dtype=np.float64)
    n = len(s0)
    t = (np.arange(n) - n // 2) * dt
    u = (t - tau) / (1.0 + beta)
    deformed = np.interp(u, t, s0, left=0.0, right=0.0)
    out = a * deformed
    if noise_sigma > 0:
        out = out + rng.normal(0, noise_sigma, size=n)
    return out


if __name__ == "__main__":
    # Synthetic self-test -- no real data, no disk I/O, safe to run any time
    # (including mid-recording). Validates: (1) the two basis-geometry claims
    # numerically, (2) that fit_nuisance recovers known (a,tau,beta) from a
    # spike generated by the FULL nonlinear model, across a range of SNRs
    # and deformation magnitudes representative of a real burst.
    rng = np.random.default_rng(42)
    fs = 30000.0
    dt_samples = 1.0
    n = 61
    nt0min = 20

    # a plausible biphasic extracellular spike shape (fast Na+ peak, slower
    # K+ trough), built from a difference of two Gaussians -- not from real
    # data, purely synthetic, just needs realistic timescales/shape
    t_samples = np.arange(n) - nt0min
    template = (1.0 * np.exp(-0.5 * (t_samples / 2.2) ** 2)
                - 0.55 * np.exp(-0.5 * ((t_samples - 7) / 5.0) ** 2))
    template -= template[:5].mean()  # baseline near 0 at the edges

    s0, s0_prime, q = build_basis(template, dt=dt_samples)

    geom = check_basis_geometry(s0, s0_prime, q)
    print("=== basis geometry check ===")
    for k, v in geom.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    ortho_ok = abs(geom["s0_dot_sprime_relative"]) < 0.05
    overlap_ok = geom["s0_dot_q_relative_error"] < 0.15
    print(f"  s0 ~orthogonal to s0': {'PASS' if ortho_ok else 'FAIL'}")
    print(f"  s0.q ~= 0.5*energy(s0): {'PASS' if overlap_ok else 'FAIL'}")

    print("\n=== recovery test across (a, tau, beta, SNR) grid ===")
    true_amp_uv = 150.0  # a representative rested-spike peak-to-trough scale
    cases = [
        dict(a=1.00, tau=0.0, beta=0.00, noise_frac=0.05),
        dict(a=0.85, tau=0.3, beta=0.05, noise_frac=0.05),
        dict(a=0.60, tau=-0.4, beta=0.15, noise_frac=0.05),
        dict(a=0.60, tau=-0.4, beta=0.15, noise_frac=0.25),  # low-SNR burst tail
        dict(a=0.35, tau=0.6, beta=0.30, noise_frac=0.35),   # deep attenuation
    ]
    header = f"{'true a':>8}{'true tau':>10}{'true beta':>11}{'noise':>8} | {'fit a':>8}{'fit tau':>9}{'fit beta':>10}{'R2':>7}"
    print(header)
    print("-" * len(header))
    all_close = True
    for c in cases:
        noise_sigma = c["noise_frac"] * true_amp_uv
        x = synthesize_deformed_spike(template * true_amp_uv, a=c["a"], tau=c["tau"],
                                       beta=c["beta"], dt=dt_samples,
                                       noise_sigma=noise_sigma, rng=rng)
        # basis is built on the UNSCALED template (a is fit, not baked in)
        result = fit_nuisance(x, s0 * true_amp_uv, s0_prime * true_amp_uv, q * true_amp_uv)
        print(f"{c['a']:8.2f}{c['tau']:10.2f}{c['beta']:11.2f}{c['noise_frac']:8.2f} | "
              f"{result['a']:8.2f}{result['tau']:9.2f}{result['beta']:10.2f}{result['r_squared']:7.3f}")
        # loose tolerances -- this is a linearization, exact recovery isn't
        # expected at large tau/beta or low SNR, just qualitatively sane
        if c["noise_frac"] <= 0.05:
            all_close &= abs(result["a"] - c["a"]) < 0.1
            all_close &= abs(result["tau"] - c["tau"]) < 0.5
            all_close &= abs(result["beta"] - c["beta"]) < 0.1

    print(f"\nHigh-SNR recovery within tolerance: {'PASS' if all_close else 'FAIL'}")
