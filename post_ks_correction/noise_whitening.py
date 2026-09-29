"""
Pillar 3 / design-doc step A -- temporal noise whitening.

WHAT THIS ADDRESSES: every fit and matched-filter score built so far
(nuisance_model.py, wavelet_features.py, spatial_footprint.py) implicitly
treats every timepoint's noise as independent and equal-variance -- i.e.
white noise. Real extracellular background noise is NOT white: it has been
through a high-pass filter and reflects a mix of real physical noise
sources, so nearby timepoints on the same channel are correlated (checked
below, on real data). A plain matched filter or ordinary-least-squares fit
that ignores this is leaving real, exploitable structure on the table --
this is the `Sigma^-1` (precision matrix) piece of pillar 1a's
`score = s^T Sigma^-1 x`, which nothing built so far actually uses (every
existing fit is implicitly `Sigma = identity`).

NOTE ON SCOPE: Kilosort4 already does its own whitening, but that is a
SPATIAL whitening (its ZCA transform decorrelates each channel against its
nearest ~32 neighboring channels, at a single time sample). It does NOT
address correlation across nearby TIME SAMPLES on one channel, which is
what this module builds -- a separate, complementary gap, not a duplicate
of something Kilosort already does.

APPROACH: fit a low-order autoregressive (AR) model to a real, spike-free
stretch of background noise on a unit's own channel -- a small number of
parameters (regularized, per the design doc), not a full empirical
covariance matrix (which would need far more real noise data than is
practical to estimate all ~61*61/2 free entries of a 61-sample window
without just re-fitting noise). From the fitted AR model, derive the
autocovariance the process implies over a spike-length window, then a
whitening operator W such that W @ noise has ~identity covariance. Applying
W to both a template and an observed snippet before comparing them turns an
ordinary least-squares fit into the noise-aware version.
"""
import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_triangular


def fit_ar_model(noise_trace, order=4):
    """Fit a causal AR(order) model to real background noise via the
    Yule-Walker equations (standard, closed-form, no iterative optimizer).

    noise_trace : 1D array, a real, spike-free (or at least
        spike-dominated-by-noise) stretch of one channel's filtered voltage.
        Longer is more stable -- this project uses ~1-2 real seconds
        (30000-60000 samples) elsewhere for noise-floor characterization
        (see demo_derivation_and_reliability.py); the same scale is used
        here.
    order : AR model order (p). Low by design (the "regularized" model the
        design doc asks for) -- default 4 is a reasonable starting point
        for filtered extracellular background, not tuned per-channel yet.

    Returns (phi, sigma2):
      phi    -- length-`order` array of AR coefficients
                (x[t] = phi[0]*x[t-1] + ... + phi[p-1]*x[t-p] + eps[t])
      sigma2 -- innovation (residual) variance
    """
    x = np.asarray(noise_trace, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n <= order * 4:
        raise ValueError(f"noise_trace too short ({n} samples) for a stable AR({order}) fit")

    # biased empirical autocorrelation, standard for Yule-Walker
    acf = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(order + 1)])

    R = np.array([[acf[abs(i - j)] for j in range(order)] for i in range(order)])
    r = acf[1:order + 1]
    phi = np.linalg.solve(R, r)
    sigma2 = acf[0] - np.dot(phi, r)
    if sigma2 <= 0:
        raise ValueError(f"AR fit produced non-positive innovation variance ({sigma2:.4g}) "
                          f"-- noise_trace may be too short, too structured, or order too high")
    return phi, sigma2


def ar_autocovariance(phi, sigma2, n):
    """The autocovariance function gamma[0..n-1] implied by an AR(p) model
    with coefficients `phi` and innovation variance `sigma2`, derived from
    the Yule-Walker relations (not simulated) -- exact for a stationary AR
    process, checked against the closed-form AR(1) result
    (gamma[k] = sigma2/(1-phi^2) * phi^k) in tests.py.

    gamma[0] and gamma[1..p] are solved together as one linear system
    (they're coupled); gamma[p+1..n-1] then follow from a pure recursion
    using only already-known values -- no further unknowns appear once you
    have the first p+1 lags.
    """
    phi = np.asarray(phi, dtype=np.float64)
    p = len(phi)
    size = p + 1
    A = np.zeros((size, size))
    b = np.zeros(size)

    # k=0: gamma[0] - sum_i phi_i*gamma[i] = sigma2
    A[0, 0] = 1.0
    for i in range(1, p + 1):
        A[0, i] -= phi[i - 1]
    b[0] = sigma2

    # k=1..p: gamma[k] - sum_i phi_i*gamma[|k-i|] = 0
    for k in range(1, p + 1):
        A[k, k] += 1.0
        for i in range(1, p + 1):
            lag = abs(k - i)
            A[k, lag] -= phi[i - 1]
        b[k] = 0.0

    gamma = list(np.linalg.solve(A, b))
    for k in range(p + 1, n):
        gamma.append(sum(phi[i - 1] * gamma[k - i] for i in range(1, p + 1)))
    return np.array(gamma[:n])


def whitening_operator(gamma, ridge=1e-6):
    """Build the n x n whitening matrix W from an autocovariance sequence
    gamma[0..n-1]: forms the Toeplitz covariance Sigma, Cholesky-factors it
    (Sigma = L L^T), and returns W = L^-1, so that for noise v ~ (0, Sigma),
    W @ v has ~identity covariance (W Sigma W^T = L^-1 L L^T L^-T = I).

    ridge : small fraction of the process variance added to the diagonal
        before factoring, purely for numerical safety against a
        near-singular covariance from a noisy real-data AR fit -- not a
        modeling choice, just standard Cholesky jitter.
    """
    n = len(gamma)
    Sigma = toeplitz(gamma)
    Sigma = Sigma + ridge * gamma[0] * np.eye(n)
    L = cholesky(Sigma, lower=True)
    W = solve_triangular(L, np.eye(n), lower=True)
    return W


def build_whitening_from_noise(noise_trace, n, order=4, ridge=1e-6):
    """Convenience wrapper: real noise trace -> AR fit -> autocovariance ->
    whitening operator, in one call. Returns (W, phi, sigma2) -- phi/sigma2
    exposed for diagnostics (e.g. checking the fitted process is stationary:
    all roots of 1 - phi_1*z - ... - phi_p*z^p outside the unit circle)."""
    phi, sigma2 = fit_ar_model(noise_trace, order=order)
    gamma = ar_autocovariance(phi, sigma2, n)
    W = whitening_operator(gamma, ridge=ridge)
    return W, phi, sigma2
