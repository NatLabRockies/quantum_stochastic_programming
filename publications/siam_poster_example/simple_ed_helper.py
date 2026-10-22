"""simple_ed_helper.py

Optimization helpers for the two-stage stochastic energy dispatch problem.

TwoStageOptHelper encapsulates the first-stage cost and the second-stage
phi interpolation so that scipy optimizers can treat the first-stage
decision variable x as a continuous quantity.  The caches (phi_cache,
phi_grad_cache) are built in the main script and injected at construction
time, keeping the helper stateless with respect to quantum circuit calls.
"""
from __future__ import annotations

import math
import numpy as np


class TwoStageOptHelper:
    """Callable helpers for the two-stage stochastic objective.

    Parameters
    ----------
    c_x : list
        First-stage cost coefficients [linear, optional quadratic].
    n_y : int
        Number of wind turbine qubits (upper bound on w_d).
    interp_mode : str
        Interpolation mode for phi(w_d) between integer knots:
          'floor'    — piecewise-constant  phi_cache[floor(w_d)]
          'linear'   — linear interpolation between adjacent integer knots
          'hermite'  — cubic Hermite spline (requires phi_grad_cache)
          'parabola' — global quadratic fit (degree 2) to all cached values and gradients
          'cubic'    — global cubic fit    (degree 3) to all cached values and gradients
          'quartic'  — global quartic fit  (degree 4) to all cached values and gradients
          'sextic'   — global sextic fit   (degree 6) to all cached values and gradients
    phi_cache : dict
        Mapping {w_d (int): phi_value (float)} for w_d in 0..n_y.
    phi_grad_cache : dict
        Mapping {w_d (int): dphi/d(w_d) (float)} for w_d in 0..n_y.
          Required for 'hermite', 'parabola', 'cubic', 'quartic', and 'sextic' modes; unused otherwise.
    """

    def __init__(self, c_x: list, n_y: int, interp_mode: str,
                 phi_cache: dict, phi_grad_cache: dict):
        self.c_x            = c_x
        self.n_y            = n_y
        self.interp_mode    = interp_mode
        self.phi_cache      = phi_cache
        self.phi_grad_cache = phi_grad_cache
        _poly_degree = {'parabola': 2, 'cubic': 3, 'quartic': 4, 'sextic': 6}
        self._poly_coeffs: np.ndarray | None = None
        self._poly_deriv_coeffs: np.ndarray | None = None
        if interp_mode in _poly_degree:
            self._poly_coeffs = self._fit_poly(_poly_degree[interp_mode])
            self._poly_deriv_coeffs = np.polyder(self._poly_coeffs)

    # ------------------------------------------------------------------
    # Polynomial fit (shared by 'parabola' and 'quartic')
    # ------------------------------------------------------------------

    def _fit_poly(self, degree: int) -> np.ndarray:
        """Fit a polynomial of the given degree to all cached values and gradients.

        The combined least-squares system stacks value rows
          [x_i^n, x_i^(n-1), ..., x_i, 1]        → phi_cache[x_i]
        and gradient rows
          [n*x_i^(n-1), (n-1)*x_i^(n-2), ..., 1, 0] → phi_grad_cache[x_i]
        and solves via np.linalg.lstsq.  Returns coefficients in descending
        order compatible with np.polyval / np.polyder.
        """
        xs = sorted(self.phi_cache.keys())
        n  = degree
        rows_v = [[xi ** (n - j) for j in range(n + 1)] for xi in xs]
        rhs_v  = [self.phi_cache[xi] for xi in xs]
        rows_g = [[(n - j) * xi ** (n - j - 1) if j < n else 0.0
                   for j in range(n + 1)]
                  for xi in xs if xi in self.phi_grad_cache]
        rhs_g  = [self.phi_grad_cache[xi]
                  for xi in xs if xi in self.phi_grad_cache]
        A = np.array(rows_v + rows_g, dtype=float)
        b = np.array(rhs_v  + rhs_g,  dtype=float)
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs  # shape (degree+1,) descending

    # ------------------------------------------------------------------
    # First-stage cost
    # ------------------------------------------------------------------

    def fs_cost(self, x_val: float) -> float:
        """First-stage cost: c_x[0]*x + c_x[1]*x^2 (quadratic if len > 1)."""
        c = self.c_x[0] * x_val
        if len(self.c_x) > 1:
            c += self.c_x[1] * x_val ** 2
        return c

    def fs_cost_grad(self, x_val: float) -> float:
        """Derivative of first-stage cost w.r.t. x."""
        g = self.c_x[0]
        if len(self.c_x) > 1:
            g += 2.0 * self.c_x[1] * x_val
        return g

    # ------------------------------------------------------------------
    # Second-stage phi interpolation
    # ------------------------------------------------------------------

    def phi_interp(self, w_d_float: float) -> float:
        """Interpolated second-stage cost at continuous wind demand w_d_float.

        Clamps w_d_float to [0, n_y] before interpolating.
        """
        w = max(0.0, min(float(w_d_float), float(self.n_y)))
        k = int(math.floor(w))
        if self.interp_mode in ('parabola', 'cubic', 'quartic', 'sextic'):
            assert self._poly_coeffs is not None
            return float(np.polyval(self._poly_coeffs, w))
        if k >= self.n_y:                              # at or past upper boundary
            return self.phi_cache.get(self.n_y, float('inf'))
        if self.interp_mode == 'floor':
            return self.phi_cache.get(k, float('inf'))
        t  = w - k                                     # fractional part in [0, 1]
        p0 = self.phi_cache.get(k,     float('inf'))
        p1 = self.phi_cache.get(k + 1, float('inf'))
        if self.interp_mode == 'linear':
            return (1.0 - t) * p0 + t * p1
        # cubic Hermite: h00*p0 + h10*m0 + h01*p1 + h11*m1
        m0 = self.phi_grad_cache.get(k,     0.0)
        m1 = self.phi_grad_cache.get(k + 1, 0.0)
        t2 = t * t;  t3 = t2 * t
        return ((2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0
                + (-2*t3 + 3*t2)*p1  + (t3 - t2)*m1)

    def phi_interp_deriv(self, w_d_float: float) -> float:
        """Derivative of phi_interp w.r.t. w_d_float (d(phi)/d(w_d)).

        Used to form the gradient of the total objective w.r.t. x via the
        chain rule: d(phi)/dx = d(phi)/d(w_d) * d(w_d)/dx = -d(phi)/d(w_d).
        Returns 0 for 'floor' mode (derivative is zero almost everywhere).
        """
        w = max(0.0, min(float(w_d_float), float(self.n_y)))
        k = int(math.floor(w))
        if self.interp_mode in ('parabola', 'cubic', 'quartic', 'sextic'):
            assert self._poly_deriv_coeffs is not None
            return float(np.polyval(self._poly_deriv_coeffs, w))
        if k >= self.n_y or self.interp_mode == 'floor':
            return 0.0
        t  = w - k
        p0 = self.phi_cache.get(k,     0.0)
        p1 = self.phi_cache.get(k + 1, 0.0)
        if self.interp_mode == 'linear':
            return p1 - p0
        m0 = self.phi_grad_cache.get(k,     0.0)
        m1 = self.phi_grad_cache.get(k + 1, 0.0)
        t2 = t * t
        return ((6*t2 - 6*t)*p0   + (3*t2 - 4*t + 1)*m0
                + (-6*t2 + 6*t)*p1 + (3*t2 - 2*t)*m1)

    # ------------------------------------------------------------------
    # Total two-stage objective
    # ------------------------------------------------------------------

    def quantum_obj(self, x_val: float, d_val: int) -> float:
        """Total objective: first-stage cost + phi_interp(d_val - x_val)."""
        return self.fs_cost(x_val) + self.phi_interp(d_val - x_val)

    def quantum_obj_and_grad(self, x_val: float, d_val: int):
        """Total objective and its gradient w.r.t. x (for L-BFGS-B).

        Returns (objective, gradient_array) where gradient_array has shape (1,).
        Chain rule: d(phi)/dx = -d(phi)/d(w_d)  since w_d = d - x.
        """
        w_d_float = d_val - x_val
        obj  = self.fs_cost(x_val) + self.phi_interp(w_d_float)
        grad = self.fs_cost_grad(x_val) - self.phi_interp_deriv(w_d_float)
        return obj, np.array([grad])
