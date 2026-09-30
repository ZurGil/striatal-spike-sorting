"""
STAGE 3: acceleration for the template-rebuild fixed-point loop.

WHERE THIS APPLIES, AND WHERE IT DOES NOT. Two separate iterative things
happen in this project and only one of them is a fixed point:

  * The template rebuild IS a fixed point. It computes
        T_{k+1} = G(T_k),  G = "align every spike to T_k, then average"
    and stops when T stops moving. This is a Picard iteration, and
    Aitken/Anderson-style extrapolation is exactly the right tool.

  * The probe-frequency search is NOT a fixed point. It maximizes
    f0*|W(f0)| over f0. Aitken's Delta-squared method accelerates
    convergent SEQUENCES, so applying it to a maximization is a category
    error -- that search is handled by a coarse grid plus bracketed
    golden-section refinement (wavelet_features.select_probe_frequency),
    and the grid stage is there because the criterion is not unimodal for
    every unit (only 10 of 20 tested units were strictly unimodal).

This module supplies the accelerators for the first case.

WHY BOTHER: the plain iteration converges linearly, so each step cuts the
error by roughly the same factor. With ~250 spikes per iteration and a
disk read plus wavelet alignment for each, an iteration is expensive
enough that halving the iteration count is worth having. More importantly,
under the OLD (broken) frequency rule this loop was observed to drift
rather than settle -- f0 walking 802 -> 739 -> 676 Hz with the amplitude
decaying 8% -- so having a principled convergence criterion and a
stopping-versus-drifting distinction matters for more than just speed.
"""
import numpy as np


def aitken_delta2(x0, x1, x2):
    """Aitken's Delta-squared extrapolation for a SCALAR sequence.

    Given three consecutive iterates of a linearly-convergent sequence,
    estimate the limit. If the sequence behaves like
        x_k - L ~ C * r^k
    then the three iterates determine r and L exactly, and this returns L.

    Returns x2 unchanged when the denominator is degenerate (the sequence
    has already stopped moving, or moves non-monotonically), which is the
    safe fallback -- never extrapolate off a numerically meaningless
    difference.
    """
    denom = (x2 - x1) - (x1 - x0)
    if not np.isfinite(denom) or abs(denom) < 1e-14:
        return x2
    return x2 - (x2 - x1) ** 2 / denom


def aitken_sequence(seq):
    """Apply aitken_delta2 across a whole scalar sequence, returning the
    accelerated sequence (length len(seq) - 2)."""
    seq = np.asarray(seq, dtype=float)
    if len(seq) < 3:
        return np.asarray([], dtype=float)
    return np.array([aitken_delta2(seq[i], seq[i + 1], seq[i + 2])
                     for i in range(len(seq) - 2)])


class AndersonAccelerator:
    """Anderson acceleration for a VECTOR fixed-point iteration x = G(x).

    Aitken's method is scalar. A template is a 61-sample vector (or a
    concatenation across channels), so the vector generalization is what
    the rebuild loop actually needs. Anderson acceleration is that
    generalization: instead of extrapolating along one sequence, it forms
    the linear combination of the last `depth` iterates whose residual is
    smallest in least-squares terms.

    The step, in the residual-difference form actually implemented:
        g_k = G(x_k) - x_k                       (residual)
        F   = [dg_{k-m+1} ... dg_k]              (residual differences)
        E   = [dG_{k-m+1} ... dG_k]              (G-value differences)
        gamma = argmin || g_k - F gamma ||       (least squares)
        x_{k+1} = G(x_k) - E gamma
    with `beta` optionally damping toward the plain Picard step.

    depth=0 reproduces the plain iteration exactly, which makes the
    comparison in the demo an honest one -- the same code path runs both.

    Usage:
        acc = AndersonAccelerator(depth=3)
        x = x0
        for k in range(max_iter):
            gx = G(x)
            if np.linalg.norm(gx - x) < tol: break
            x = acc.step(x, gx)
    """

    def __init__(self, depth=3, beta=1.0, reg=1e-10):
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth = int(depth)
        self.beta = float(beta)
        self.reg = float(reg)
        self._x = []   # past iterates
        self._g = []   # past residuals G(x)-x
        self._G = []   # past G(x) values

    def reset(self):
        self._x, self._g, self._G = [], [], []

    def step(self, x, gx):
        """One accelerated step. `x` is the current iterate, `gx` is G(x).
        Returns the next iterate."""
        x = np.asarray(x, dtype=float).ravel()
        gx = np.asarray(gx, dtype=float).ravel()
        g = gx - x

        self._x.append(x.copy())
        self._g.append(g.copy())
        self._G.append(gx.copy())
        # keep depth+1 points so we can form `depth` differences
        keep = self.depth + 1
        if len(self._x) > keep:
            self._x = self._x[-keep:]
            self._g = self._g[-keep:]
            self._G = self._G[-keep:]

        m = len(self._x) - 1
        if self.depth == 0 or m < 1:
            # plain Picard step, damped by beta
            return x + self.beta * g

        F = np.column_stack([self._g[i + 1] - self._g[i] for i in range(m)])
        E = np.column_stack([self._G[i + 1] - self._G[i] for i in range(m)])

        # regularized least squares: (F'F + reg*I) gamma = F' g
        FtF = F.T @ F
        scale = np.trace(FtF) / max(m, 1)
        if not np.isfinite(scale) or scale <= 0:
            return x + self.beta * g
        try:
            gamma = np.linalg.solve(FtF + self.reg * scale * np.eye(m), F.T @ g)
        except np.linalg.LinAlgError:
            return x + self.beta * g
        if not np.all(np.isfinite(gamma)):
            return x + self.beta * g

        x_next = gx - E @ gamma
        if self.beta != 1.0:
            x_next = (1.0 - self.beta) * (x - self._x[0] * 0.0) + self.beta * x_next
        if not np.all(np.isfinite(x_next)):
            return x + self.beta * g
        return x_next


def iterate_to_fixed_point(G, x0, depth=3, tol=1e-8, max_iter=100,
                            norm=None, return_history=False):
    """Run x = G(x) to convergence, optionally Anderson-accelerated.

    G     : callable mapping a vector to a vector.
    x0    : starting vector.
    depth : Anderson history depth. 0 = plain Picard iteration.
    tol   : stop when ||G(x) - x|| / max(||x||, 1) falls below this.
    norm  : optional custom residual norm, callable(gx, x) -> float.

    Returns (x, n_G_evaluations) or, with return_history=True,
    (x, n_G_evaluations, residual_history).
    """
    acc = AndersonAccelerator(depth=depth)
    x = np.asarray(x0, dtype=float).ravel()
    hist = []
    n_eval = 0
    for _ in range(max_iter):
        gx = np.asarray(G(x), dtype=float).ravel()
        n_eval += 1
        res = (norm(gx, x) if norm is not None
               else np.linalg.norm(gx - x) / max(np.linalg.norm(x), 1.0))
        hist.append(float(res))
        if res < tol:
            x = gx
            break
        x = acc.step(x, gx)
    if return_history:
        return x, n_eval, hist
    return x, n_eval
