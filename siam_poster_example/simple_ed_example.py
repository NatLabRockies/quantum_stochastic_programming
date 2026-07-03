#!/usr/bin/env python3
"""
this script focuses on the multiplexed-oracle MLQAE layer.does not
replace DQA state prep with an annealing circuit yet.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from typing import Iterable
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time as _time

try:
    import cudaq
except ImportError as exc:
    raise RuntimeError("cudaq is required to run this script.") from exc

# Path setup to match simple_ed scripts

_QISKIT_SP = "/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages"
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qiskit_impl")
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

from binary_optimizer import BinaryNestedOptimizer

## UC Parameters ##

RUN_GROVER_IDENTITY_CHECKS = False

N_X = 1
C_X = [4.0]
X0 = [6.0]

N_Y = 4
D = 8
C_Y = [[c0, c1] for c0, c1 in zip(np.linspace(2.0, 3.0, N_Y), 1e-3 * np.ones(N_Y))]
C_R = 10.0

## Derived Parameters ##
N_T_MIN = 32
N_T = max(N_T_MIN, N_Y * N_Y)
# N_T = 4
W_D = int(D - sum(X0))
assert W_D > 0, f"W_D={W_D} <= 0: x0 covers full demand."

C_Y_EFF = [float(c[0] + c[1]) if isinstance(c, list) else float(c) for c in C_Y]

# Uniform xi distribution over all 2^n_y scenarios.
PDF = {
    tuple(int(v) for v in ("{0:0" + str(N_Y) + "b}").format(i)): 1.0 / (2 ** N_Y)
    for i in range(2 ** N_Y)
}
SCENARIOS = list(PDF.keys())
S = len(SCENARIOS)

# Multiplexed index register encodes k in [0, 2^N_IDX_Q - 1].
# We use:
#   k=0              - objective phi component
#   k=1..N_Y         - gradient components
#   remaining ks     - padded constants (do not use)
N_Q_Y = N_Y            # Number of stochastic turbines
N_SCEN_Q = N_Y         # 2^N_SCEN_Q = number of scenarios for uniform xi
N_IDX_Q = 3            # 8 indices, enough for 1 + N_Y = 5 quantities
N_IDX = 1 << N_IDX_Q
N_USED_IDX = 1 + N_Y
N_SYS = N_SCEN_Q + N_IDX_Q + 1  # + ancilla
N_FULL = N_Q_Y + N_SCEN_Q + N_IDX_Q + 1  # + ancilla

# ── Optimization Sweep configuration ───────────────────────────────────────────────────
# Extend D_VALUES to sweep demand levels, e.g. list(range(6, 11))
# D_VALUES    = [d]                               # extend to sweep, e.g. list(range(6, 11))
# D_VALUES    = list(range(1,13))
D_VALUES    = list(np.arange(1.0, 12.5, 0.5))
# D_VALUES    = list(np.arange(7.0, 8.5, 1.0))
OPT_METHODS = ['bounded', 'COBYLA', 'Nelder-Mead', 'L-BFGS-B', 'BFGS']
# OPT_METHODS = ['COBYLA', 'L-BFGS-B']
# bounded     — minimize_scalar with Brent (1-D, tolerates non-smooth objectives)
# COBYLA      — derivative-free, handles inequality constraints directly
# Nelder-Mead — derivative-free simplex (bounds enforced by clipping)
# L-BFGS-B    — gradient-based with box bounds; requires linear or hermite mode
# BFGS        — gradient-based, unconstrained; result clipped to [x_lo, x_hi]

# Interpolation mode for phi(w_d) between integer grid points:
#   'floor'    — piecewise-constant (original behaviour, phi = phi_cache[floor(w_d)])
#   'linear'   — linear interpolation between adjacent integer knots
#   'hermite'  — cubic Hermite spline using phi_cache values + phi_grad_cache derivatives
#   'parabola' — global quadratic fit (least-squares) to all cached values + gradients
#   'cubic'    — global cubic fit    (least-squares) to all cached values + gradients
#   'quartic'  — global quartic fit  (least-squares) to all cached values + gradients
#   'sextic'   — global sextic fit   (least-squares) to all cached values + gradients
SWEEP_INTERP_MODES = ['linear', 'hermite', 'parabola', 'cubic', 'quartic', 'sextic']   # modes to compare in the sweep table
# SWEEP_INTERP_MODES = ['linear', 'hermite']



#  Classical UC helpers and per-scenario multiplexed table



def _wind_scenario_cost(y: tuple[int, ...], xi: tuple[int, ...]) -> float:
    """Scenario cost Q(y, xi) consistent with cudaq_impl._wind_scenario_cost."""
    cost = 0.0
    for j in range(N_Y):
        if y[j] == 1:
            if xi[j] == 1:
                c_j = C_Y[j]
                if isinstance(c_j, list):
                    cost += c_j[0] + c_j[1]
                else:
                    cost += c_j
            else:
                cost += C_R
    return float(cost)


def _best_y_for_xi(xi: tuple[int, ...]) -> tuple[int, ...]:
    """Brute-force feasible y (Hamming weight W_D) and return argmin Q(y, xi)."""
    best_cost = float("inf")
    best_y: tuple[int, ...] | None = None
    for active in itertools.combinations(range(N_Y), W_D):
        y = [0] * N_Y
        for j in active:
            y[j] = 1
        y_t = tuple(y)
        c = _wind_scenario_cost(y_t, xi)
        if c < best_cost:
            best_cost = c
            best_y = y_t
    if best_y is None:
        raise RuntimeError("No feasible y found for scenario.")
    return best_y


def _scenario_gradient_component(j: int, y_star: tuple[int, ...], xi: tuple[int, ...]) -> float:
    """Gradient piece matching simple_ed/cudaq_impl semantics.

    For xi[j]=1 and quadratic c_y[j]=[c0,c1], gradient is c0 + 2*c1*y[j].
    For xi[j]=0, gradient is c_r.
    """
    if xi[j] == 1:
        c_j = C_Y[j]
        if isinstance(c_j, list):
            return float(c_j[0] + 2.0 * c_j[1] * y_star[j])
        return float(c_j)
    return float(C_R)


def build_per_scenario_table() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build F_used[s, k] and padded F_full[s, k] for multiplexed oracle.

    F_used columns:
      k=0       -> scenario objective min_y Q(y, xi)
      k=1..N_Y  -> per-turbine gradient component
    """
    f_used = np.zeros((S, N_USED_IDX), dtype=float)

    for s_idx, xi in enumerate(SCENARIOS):
        y_star = _best_y_for_xi(xi)
        f_used[s_idx, 0] = _wind_scenario_cost(y_star, xi)
        for j in range(N_Y):
            f_used[s_idx, 1 + j] = _scenario_gradient_component(j, y_star, xi)

    f_min = float(f_used.min())
    f_max = float(f_used.max())

    # Pad to full index space; unused indices pinned to f_min so g=0.
    f_full = np.full((S, N_IDX), f_min, dtype=float)
    f_full[:, :N_USED_IDX] = f_used
    return f_used, f_full, f_min, f_max


def build_oracle_angles(f_full: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """Affine map f -> g in [0,1], then theta = 2*asin(sqrt(g))."""
    span = f_max - f_min if f_max > f_min else 1.0
    g = np.clip((f_full - f_min) / span, 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(g))
    return theta.reshape(-1)


def build_exact_oracle_thetas(norm: float) -> list[float]:
    """Pre-compute exact oracle angles for all (y, xi) states with sum(y)==W_D.

    addr = (y_int << N_Y) | xi_int using MSB-first bit convention:
      y_int  bit b = (y_int  >> (N_Y-1-b)) & 1  -> y_reg[b]
      xi_int bit b = (xi_int >> (N_Y-1-b)) & 1  -> xi_reg[b]

    Returns a flat list of length 2^(2*N_Y) = 256.
    norm must satisfy norm >= max possible total cost (e.g. W_D * C_R).
    States with sum(y) != W_D (infeasible) get angle 0 (oracle doesn't rotate).
    """
    n_states = 1 << (2 * N_Y)
    thetas = [0.0] * n_states
    for y_int in range(1 << N_Y):
        y = tuple((y_int >> (N_Y - 1 - b)) & 1 for b in range(N_Y))
        if sum(y) != W_D:
            continue
        for xi_int in range(1 << N_Y):
            xi = tuple((xi_int >> (N_Y - 1 - b)) & 1 for b in range(N_Y))
            cost = _wind_scenario_cost(y, xi)
            addr = (y_int << N_Y) | xi_int
            thetas[addr] = 2.0 * math.asin(math.sqrt(min(cost / norm, 1.0)))
    return thetas


def _build_exact_thetas_wd(w_d_val: int, norm: float) -> list[float]:
    """Build exact oracle angles for Hamming-weight-w_d_val states.

    Generalises build_exact_oracle_thetas to accept an arbitrary w_d value
    instead of the module-level constant W_D.  Used in optimization_test to
    pre-compute the exact-oracle angle tables for every feasible wind-demand
    level w_d = 1 .. N_Y.
    """
    n_states = 1 << (2 * N_Y)
    thetas = [0.0] * n_states
    for y_int in range(1 << N_Y):
        y = tuple((y_int >> (N_Y - 1 - b)) & 1 for b in range(N_Y))
        if sum(y) != w_d_val:
            continue
        for xi_int in range(1 << N_Y):
            xi = tuple((xi_int >> (N_Y - 1 - b)) & 1 for b in range(N_Y))
            cost = _wind_scenario_cost(y, xi)
            addr = (y_int << N_Y) | xi_int
            thetas[addr] = 2.0 * math.asin(math.sqrt(min(cost / norm, 1.0)))
    return thetas


def classical_truth() -> tuple[float, np.ndarray]:
    """Return classical phi(w_d) and gradient vector consistent with this table."""
    bno = BinaryNestedOptimizer(C_X, C_Y_EFF, C_R, PDF, D, is_uniform=True)
    exp_vals = bno.brute_force_wind_demand_expectation_values()
    phi = float(exp_vals[W_D])

    # Expectation of our scenario gradient table across uniform scenarios.
    _, f_full, _, _ = build_per_scenario_table()
    grad = np.mean(f_full[:, 1 : 1 + N_Y], axis=0)
    return phi, grad



#  CUDA-Q kernel building blocks



@cudaq.kernel
def _ry_gate(theta: float, q: cudaq.qubit):
    ry(theta, q)


@cudaq.kernel
def _z_gate(q: cudaq.qubit):
    z(q)


@cudaq.kernel
def _apply_a(oracle_thetas: list[float], scen: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    # Uniform scenario and index superpositions, consistent with simple_ed's uniform xi.
    for q in scen:
        h(q)
    for q in idx:
        h(q)

    # Multiplexed oracle: address = (scenario_bits << N_IDX_Q) | index_bits.
    for addr in range(1 << (N_SCEN_Q + N_IDX_Q)):
        # Wrap controls to select only the desired basis address.
        for b in range(N_SCEN_Q):
            bit = (addr >> (N_IDX_Q + (N_SCEN_Q - 1 - b))) & 1
            if bit == 0:
                x(scen[b])
        for b in range(N_IDX_Q):
            bit = (addr >> (N_IDX_Q - 1 - b)) & 1
            if bit == 0:
                x(idx[b])

        cudaq.control(
            _ry_gate,
            [scen[0], scen[1], scen[2], scen[3], idx[0], idx[1], idx[2]],
            oracle_thetas[addr],
            anc,
        )

        # Uncompute wrap.
        for b in range(N_IDX_Q):
            bit = (addr >> (N_IDX_Q - 1 - b)) & 1
            if bit == 0:
                x(idx[b])
        for b in range(N_SCEN_Q):
            bit = (addr >> (N_IDX_Q + (N_SCEN_Q - 1 - b))) & 1
            if bit == 0:
                x(scen[b])


@cudaq.kernel
def _apply_a_dagger(oracle_thetas: list[float], scen: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    # Explicit inverse of _apply_a.
    # A = H_all * Oracle(thetas), so A† = Oracle(-thetas) * H_all.
    # Step 1: apply oracle blocks in reverse order with negated angles.
    for addr in range((1 << (N_SCEN_Q + N_IDX_Q)) - 1, -1, -1):
        for b in range(N_SCEN_Q):
            bit = (addr >> (N_IDX_Q + (N_SCEN_Q - 1 - b))) & 1
            if bit == 0:
                x(scen[b])
        for b in range(N_IDX_Q):
            bit = (addr >> (N_IDX_Q - 1 - b)) & 1
            if bit == 0:
                x(idx[b])

        cudaq.control(
            _ry_gate,
            [scen[0], scen[1], scen[2], scen[3], idx[0], idx[1], idx[2]],
            -oracle_thetas[addr],
            anc,
        )

        for b in range(N_IDX_Q):
            bit = (addr >> (N_IDX_Q - 1 - b)) & 1
            if bit == 0:
                x(idx[b])
        for b in range(N_SCEN_Q):
            bit = (addr >> (N_IDX_Q + (N_SCEN_Q - 1 - b))) & 1
            if bit == 0:
                x(scen[b])

    # Step 2: H† = H on all registers (H is self-inverse).
    for q in idx:
        h(q)
    for q in scen:
        h(q)


@cudaq.kernel
def _apply_s_chi(k: int, idx: cudaq.qview, anc: cudaq.qubit):
    for b in range(N_IDX_Q):
        bit = (k >> (N_IDX_Q - 1 - b)) & 1
        if bit == 0:
            x(idx[b])

    cudaq.control(_z_gate, [idx[0], idx[1], idx[2]], anc)

    for b in range(N_IDX_Q):
        bit = (k >> (N_IDX_Q - 1 - b)) & 1
        if bit == 0:
            x(idx[b])


@cudaq.kernel
def _apply_s0(scen: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    for q in scen:
        x(q)
    for q in idx:
        x(q)
    x(anc)

    cudaq.control(_z_gate, [scen[0], scen[1], scen[2], scen[3], idx[0], idx[1], idx[2]], anc)

    x(anc)
    for q in idx:
        x(q)
    for q in scen:
        x(q)


@cudaq.kernel
def _apply_q(oracle_thetas: list[float], k: int, scen: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    _apply_s_chi(k, idx, anc)
    _apply_a_dagger(oracle_thetas, scen, idx, anc)
    _apply_s0(scen, idx, anc)
    _apply_a(oracle_thetas, scen, idx, anc)


@cudaq.kernel
def mlqae_state_kernel(oracle_thetas: list[float], k: int, m_power: int):
    q = cudaq.qvector(N_SYS)
    scen = q[0:N_SCEN_Q]
    idx = q[N_SCEN_Q : N_SCEN_Q + N_IDX_Q]
    anc = q[N_SYS - 1]

    _apply_a(oracle_thetas, scen, idx, anc)
    for _ in range(m_power):
        _apply_q(oracle_thetas, k, scen, idx, anc)


# MLQAE utilities



def p_good_exact(oracle_thetas: np.ndarray, k: int, m_power: int) -> float:
    sv = np.array(cudaq.get_state(
        mlqae_state_kernel,
        [float(v) for v in oracle_thetas],
        int(k),
        int(m_power),
    ))

    prob = 0.0
    for i in range(1 << N_SYS):
        p = abs(sv[i]) ** 2
        if p < 1e-14:
            continue
        b = format(i, f"0{N_SYS}b")[::-1]

        # idx bits are qubits [N_SCEN_Q .. N_SCEN_Q+N_IDX_Q-1].
        idx_bits = b[N_SCEN_Q : N_SCEN_Q + N_IDX_Q]
        idx_int = int(idx_bits, 2)
        anc = int(b[N_SYS - 1])

        if idx_int == k and anc == 1:
            prob += p
    return float(prob)


def mlqae_estimate(oracle_thetas: np.ndarray,
                   k: int,
                   schedule: Iterable[int] = (0, 1, 2, 4, 8, 16),
                   shots: int = 2000,
                   seed: int = 0) -> tuple[float, float]:
    """Return (theta_hat, a_hat) for index k using MLQAE likelihood fitting."""
    schedule = tuple(int(v) for v in schedule)
    rng_local = np.random.default_rng(seed)

    p_true = np.array([p_good_exact(oracle_thetas, k, m_power) for m_power in schedule])
    hits = rng_local.binomial(shots, p_true)

    m_arr = np.asarray(schedule, dtype=float)
    h_arr = hits.astype(float)
    n_shots = float(shots)

    def neg_ll_scalar(theta: float) -> float:
        p = np.sin((2.0 * m_arr + 1.0) * theta) ** 2
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -float(np.sum(h_arr * np.log(p) + (n_shots - h_arr) * np.log(1.0 - p)))

    m_max = max(schedule)
    n_lobes = 2 * m_max + 1
    grid_size = max(4000, 200 * n_lobes)
    theta_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, grid_size)

    p_grid = np.sin(np.outer(2.0 * m_arr + 1.0, theta_grid)) ** 2
    p_grid = np.clip(p_grid, 1e-12, 1.0 - 1e-12)
    ll_grid = (h_arr[:, None] * np.log(p_grid) + (n_shots - h_arr)[:, None] * np.log(1.0 - p_grid)).sum(axis=0)

    best_i = int(np.argmax(ll_grid))
    lo = theta_grid[max(0, best_i - 1)]
    hi = theta_grid[min(grid_size - 1, best_i + 1)]

    res = minimize_scalar(
        neg_ll_scalar,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6},
    )
    theta_hat = float(res.x)
    return theta_hat, float(np.sin(theta_hat))


def quantum_expectations_mlqae(oracle_thetas: np.ndarray,
                               f_min: float,
                               f_max: float,
                               schedule: Iterable[int],
                               shots: int,
                               seed_base: int = 0) -> np.ndarray:
    """Decode MLQAE estimates for k=0..N_USED_IDX-1 back to f-units."""
    span = f_max - f_min if f_max > f_min else 1.0
    out = np.zeros(N_USED_IDX)

    for k in range(N_USED_IDX):
        _, a_hat = mlqae_estimate(
            oracle_thetas,
            k,
            schedule=schedule,
            shots=shots,
            seed=seed_base + k,
        )
        e_g_k = N_IDX * (a_hat ** 2)
        out[k] = f_min + span * e_g_k
    return out


def check_sin_squared_identity(oracle_thetas: np.ndarray, k: int, schedule: Iterable[int]) -> float:
    a0 = math.sqrt(max(0.0, min(1.0, p_good_exact(oracle_thetas, k, 0))))
    theta0 = math.asin(a0)

    err = 0.0
    for m_power in schedule:
        pred = math.sin((2 * m_power + 1) * theta0) ** 2
        meas = p_good_exact(oracle_thetas, k, m_power)
        err = max(err, abs(pred - meas))
    return err


# =============================================================================
# DQA building blocks: Dicke state prep, cost operator, XY mixer, and their
# adjoints.  Consumed by _a_op_exact_multi and _a_op_exact_multi_dagger.
# =============================================================================


@cudaq.kernel
def ccry(theta: float,
         ctrl1: cudaq.qubit, ctrl2: cudaq.qubit, target: cudaq.qubit):
    """Doubly-controlled RY via cudaq.control (correct for all input states).

    Qiskit counterpart: qc.append(RYGate(theta).control(2), [ctrl1, ctrl2, target])
    used in ExpValFun_functions.dicke_state_circuit() and
    single_oracle_sin_inconstraint().
    Using cudaq.control avoids relative-phase errors from manual decompositions.
    Ref: CUDA-Q docs — https://nvidia.github.io/cuda-quantum/
    """
    cudaq.control(_ry_gate, [ctrl1, ctrl2], theta, target)


@cudaq.kernel
def _mcry(theta: float, ctrls: cudaq.qview, target: cudaq.qubit):
    """Multi-controlled RY(theta) on target, gated by all qubits in ctrls.

    Generalises cry (1 control) and ccry (2 controls) to arbitrary fan-in.
    Qiskit counterpart: RYGate(theta).control(len(ctrls)) used inside
    dicke_state_circuit() for the j-th SCS layer.
    """
    cudaq.control(_ry_gate, ctrls, theta, target)

# -----------------------------------------------------------------------------
# PDF state (xi register) — uniform distribution: H^⊗n
# Qiskit counterpart: dist_prep.pdf_initialize() with is_uniform=True
#   (binary_optimizer.BinaryNestedOptimizer.pdf_initialize, is_uniform=True branch)
#   Qiskit: qc.h(i) for i in range(num_wind_vars)
# -----------------------------------------------------------------------------
@cudaq.kernel
def pdf_init_uniform(xi: cudaq.qview):
    """Prepare uniform superposition on the xi (pdf) register: H^⊗n.

    Qiskit counterpart: BinaryNestedOptimizer.pdf_initialize() (is_uniform=True)
    in binary_optimizer.py and pdf_initialize() in dist_prep.py.
    Both apply H gates to every qubit in the xi register.
    """
    for q in xi:
        h(q)


@cudaq.kernel
def dicke_state(n: int, k: int, angles: list[float], y: cudaq.qview):
    """General Dicke state |D_n^k⟩ preparation (Bärtschi & Eidenbenz 2019).

    Prepares the uniform superposition over all n-qubit bit-strings with
    exactly k ones.  Pass pre-computed angles from dicke_state_angles(n, k).

    Qiskit counterpart: ExpValFun_functions.dicke_state_circuit(args) and
    BinaryNestedOptimizer.dicke_state_circuit(weight) in binary_optimizer.py,
    generalised to arbitrary n=args['n_y'] and k=args['w_d'].

    Algorithm (Bärtschi & Eidenbenz 2019, Fig. 3):
      1. Initialise |0^{n-k} 1^k⟩ by flipping the k rightmost qubits.
      2. Phase 1 — (n-k) SCS(n-i, k) steps on qubits y[n-k-1-i .. n-1-i]
         for i = 0 .. n-k-1.
      3. Phase 2 — (k-1) SCS(k-i, k-1-i) steps on qubits y[0 .. k-1-i]
         for i = 0 .. k-2.
    Each SCS(m, l) step applies, for j = 1 .. l:
      cx(q[l-j], q[l]);  (j-controlled) RY(theta_j) ctrls=q[l-j+1..l], tgt=q[l-j];  cx(q[l-j], q[l])
    where theta_j = 2*arccos(sqrt(j/m)) (pre-computed by dicke_state_angles).

    For n=4, k=2 this produces an equivalent circuit to dicke_state_n4_k2.
    """
    # ---- Step 1: initialise ------------------------------------------------
    for i in range(k):
        x(y[n - 1 - i])

    angle_idx = 0

    # ---- Phase 1: (n-k) SCS steps across the full register -----------------
    for step in range(n - k):
        start = n - k - 1 - step        # leftmost qubit index for this SCS
        for j in range(1, k + 1):
            tgt = start + k - j          # target qubit index
            cx(y[tgt], y[start + k])
            _mcry(angles[angle_idx], y[tgt + 1 : start + k + 1], y[tgt])
            cx(y[tgt], y[start + k])
            angle_idx += 1

    # ---- Phase 2: (k-1) SCS steps at the left end of the register ----------
    for step in range(k - 1):
        l = k - 1 - step                # excitation count for this SCS
        for j in range(1, l + 1):
            tgt = l - j
            cx(y[tgt], y[l])
            _mcry(angles[angle_idx], y[tgt + 1 : l + 1], y[tgt])
            cx(y[tgt], y[l])
            angle_idx += 1


@cudaq.kernel
def cost_operator(gamma: float,
                  c_y :list[float], c_r: float, cost_norm: float,
                  y: cudaq.qview, xi: cudaq.qview):
    scale = gamma / cost_norm
    for k in range(len(y)):
        cr1(c_y[k] * scale, xi[k], y[k])
        x(xi[k])
        cr1(c_r * scale, xi[k], y[k])
        x(xi[k])
    ## end for
    return


@cudaq.kernel
def fswap_power(beta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    """Partial SWAP (SWAP^beta) on two qubits via XX+YY decomposition
    plus the odd-parity phase needed to match Qiskit's SwapGate().power(beta).

    Qiskit counterpart: SwapGate().power(amplitude) applied per pair (qj, qk)
    inside ExpValFun_functions.demand_constraint_preserving_mixer().

    SWAP^β matrix (computational basis |00⟩,|01⟩,|10⟩,|11⟩):
      [[1, 0,            0,            0          ],
       [0, cos(βπ/2),    i·sin(βπ/2),  0          ],
       [0, i·sin(βπ/2),  cos(βπ/2),    0          ],
       [0, 0,            0,            e^(iβπ/2)  ]]

    The |11⟩ diagonal element e^(iβπ/2) is the odd-parity phase.
    Rxx+Ryy alone gives e^(0)=1 for |11⟩, which is wrong.
    The extra CX–Rz(angle)–CX block adds the phase only when both qubits are |1⟩.
    """
    angle = beta * math.pi / 2.0

    # Rxx(angle)
    h(q0)
    h(q1)
    cx(q0, q1)
    rz(angle, q1)
    cx(q0, q1)
    h(q0)
    h(q1)

    # Ryy(angle)
    rx(math.pi / 2.0, q0)
    rx(math.pi / 2.0, q1)
    cx(q0, q1)
    rz(angle, q1)
    cx(q0, q1)
    rx(-math.pi / 2.0, q0)
    rx(-math.pi / 2.0, q1)

    # Odd-parity phase: adds e^(iβπ/2) to the |11⟩ component only,
    # matching Qiskit's SwapGate().power(beta).
    cx(q0, q1)
    rz(angle, q1)
    cx(q0, q1)


@cudaq.kernel
def mixer(beta: float, y: cudaq.qview):
    for j in range(len(y)):
        for k in range(j + 1, len(y)):
            fswap_power(beta, y[j], y[k])
        # end for k
    # end for j
    return


# -- Adjoint kernels for DQA layers (used by _a_op_exact_multi_dagger) --


@cudaq.kernel
def cost_operator_dagger(gamma: float,
                         c_y: list[float], c_r: float, cost_norm: float,
                         y: cudaq.qview, xi: cudaq.qview):
    """Adjoint of cost_operator: reverse loop order, negate CR1 angles.
    Gate sequence per iteration in forward: cr1(theta1), x, cr1(theta2), x.
    Adjoint per iteration (reversed):       x, cr1(-theta2), x, cr1(-theta1).
    """
    scale = gamma / cost_norm
    for kk in range(len(y)):
        k = len(y) - 1 - kk          # reverse loop
        x(xi[k])
        cr1(-c_r * scale, xi[k], y[k])
        x(xi[k])
        cr1(-c_y[k] * scale, xi[k], y[k])


@cudaq.kernel
def mixer_dagger(beta: float, y: cudaq.qview):
    """Adjoint of mixer: reverse pair order, apply fswap_power(-beta).
    SWAP^beta adjoint = SWAP^{-beta}; also reverse the (j,k) application order.
    Forward pairs: (0,1),(0,2),...,(n-2,n-1).
    Reversed pairs: (n-2,n-1),...,(0,2),(0,1).
    """
    for ji in range(len(y)):
        j = len(y) - 1 - ji
        for ki in range(len(y) - 1 - j):
            k = len(y) - 1 - ki
            fswap_power(-beta, y[j], y[k])


@cudaq.kernel
def dicke_state_dagger(n: int, k: int, angles: list[float], y: cudaq.qview):
    """Adjoint of dicke_state: reverse SCS sequence, negate all RY angles.

    Uses the same angles list as the forward kernel.  Each forward angle
    angles[i] is negated and the execution order is reversed:
      - Phase 2 SCS steps executed before Phase 1 (reversed phase order)
      - Within each phase, steps and j-indices run in reverse
    The mapping to the reversed-negated angle is: -angles[total - 1 - angle_idx]
    where angle_idx increments 0..total-1 through the dagger execution order.

    Dagger loop structure derived from the forward structure:
      Forward Phase 1 (n-k steps): start = n-k-1-step (decreasing)
      Dagger Phase 1: start = s (increasing, s=0..n-k-1), real_j = k..1
      Forward Phase 2 (k-1 steps): l = k-1-step (decreasing)
      Dagger Phase 2: l = s+1 (increasing, s=0..k-2), jj in range(l)
    Gate structure per SCS triplet: cx, mcry(-angle), cx (CX is self-adjoint).
    """
    total = (n - k) * k + k * (k - 1) // 2
    angle_idx = 0

    # Phase 2 dagger first (was last in forward)
    # Dagger step s maps to forward step (k-2-s), giving l = s+1
    for s in range(k - 1):
        l = s + 1
        for jj in range(l):
            tgt = jj
            cx(y[tgt], y[l])
            _mcry(-angles[total - 1 - angle_idx], y[tgt + 1 : l + 1], y[tgt])
            cx(y[tgt], y[l])
            angle_idx += 1

    # Phase 1 dagger second (was first in forward)
    # Dagger step s maps to forward step (n-k-1-s), giving start = s
    for s in range(n - k):
        start = s
        for jj in range(k):
            real_j = k - jj          # j runs k..1 (reversed vs forward j=1..k)
            tgt = start + k - real_j
            cx(y[tgt], y[start + k])
            _mcry(-angles[total - 1 - angle_idx], y[tgt + 1 : start + k + 1], y[tgt])
            cx(y[tgt], y[start + k])
            angle_idx += 1

    # Un-init: undo the forward X flips (X is self-adjoint)
    for i in range(k):
        x(y[n - 1 - i])


# =============================================================================
# Shared infrastructure: _apply_s_chi selects index k; _apply_s0_full reflects
# about |0⋯0⟩ over the full yx_reg+idx+anc register.
# =============================================================================


@cudaq.kernel
def _apply_s0_full(yx_reg: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    """S_0 reflection for DQA: flips all of yx_reg (y+xi, 8 qubits), idx, and anc."""
    for q in yx_reg:
        x(q)
    for q in idx:
        x(q)
    x(anc)
    cudaq.control(_z_gate, [yx_reg[0], yx_reg[1], yx_reg[2], yx_reg[3],
                             yx_reg[4], yx_reg[5], yx_reg[6], yx_reg[7],
                             idx[0], idx[1], idx[2]], anc)
    x(anc)
    for q in idx:
        x(q)
    for q in yx_reg:
        x(q)

# =============================================================================
# Exact multiplexed pipeline: _a_op_exact_multi replaces the approximate k=0
# sequential oracle with a 256-iter 11-ctrl-RY loop.  Gradients (k=1..N_Y) use
# the same exact single-turbine blocks.  Correct for any W_D.
# =============================================================================


# ---------------------------------------------------------------------------
# Exact multiplexed oracle: exact per-(y,xi) RY for k=0 + exact gradient
# blocks for k=1..N_Y.  Correct for any W_D with no angle accumulation.
# Requires extra parameter exact_thetas (256 pre-computed k=0 angles).
# ---------------------------------------------------------------------------
@cudaq.kernel
def _a_op_exact_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                       cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                       n_y: int, oracle_cy: list[float], oracle_cr: float,
                       oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                       exact_thetas: list[float],
                       qubits: cudaq.qview, idx: cudaq.qview, ancilla: cudaq.qubit):
    """Multiplexed A operator with exact k=0 oracle: no angle accumulation for any W_D.

    k=0: 256-iter loop of 11-ctrl-RY gates (idx + y + xi), exact for W_D>1.
    k=1..4: same exact gradient blocks as _a_op_multi (already W_D-agnostic).
    """
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    dicke_state(n_y, w_d, dicke_angles, y)
    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)
    h(idx[0]); h(idx[1]); h(idx[2])
    # k=0 (000): EXACT per-(y,xi) oracle — 11-qubit controlled RY, 256 iterations
    x(idx[0]); x(idx[1]); x(idx[2])
    for addr in range(1 << (2 * N_Y)):           # compile-time unrolled (256)
        y_int  = addr >> N_Y
        xi_int = addr & ((1 << N_Y) - 1)
        for b in range(N_Y):
            if not ((y_int  >> (N_Y - 1 - b)) & 1):
                x(y[b])
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi[b])
        cudaq.control(_ry_gate, [idx[0], idx[1], idx[2],
                                  y[0], y[1], y[2], y[3],
                                  xi[0], xi[1], xi[2], xi[3]],
                      exact_thetas[addr], ancilla)
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi[b])
        for b in range(N_Y):
            if not ((y_int  >> (N_Y - 1 - b)) & 1):
                x(y[b])
    x(idx[0]); x(idx[1]); x(idx[2])
    # k=1..4: exact gradient blocks (identical to _a_op_multi)
    x(idx[0]); x(idx[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cy_hi[0], ancilla)
    x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cy_lo[0], ancilla)
    x(y[0]); x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[0]], oracle_cr, ancilla)
    x(xi[0]); x(idx[0]); x(idx[1])
    x(idx[0]); x(idx[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cy_hi[1], ancilla)
    x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cy_lo[1], ancilla)
    x(y[1]); x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[1]], oracle_cr, ancilla)
    x(xi[1]); x(idx[0]); x(idx[2])
    x(idx[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cy_hi[2], ancilla)
    x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cy_lo[2], ancilla)
    x(y[2]); x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[2]], oracle_cr, ancilla)
    x(xi[2]); x(idx[0])
    x(idx[1]); x(idx[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cy_hi[3], ancilla)
    x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cy_lo[3], ancilla)
    x(y[3]); x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[3]], oracle_cr, ancilla)
    x(xi[3]); x(idx[1]); x(idx[2])


@cudaq.kernel
def _a_op_exact_multi_dagger(dicke_angles: list[float], c_y: list[float], c_r: float,
                              cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                              n_y: int, oracle_cy: list[float], oracle_cr: float,
                              oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                              exact_thetas: list[float],
                              qubits: cudaq.qview, idx: cudaq.qview, ancilla: cudaq.qubit):
    """Adjoint of _a_op_exact_multi: oracle†(k=4..1 same as _a_op_multi, k=0 exact)."""
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    # 1. Gradient daggers k=4..1 (identical to _a_op_multi_dagger)
    x(idx[1]); x(idx[2])
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[3]], -oracle_cr, ancilla)
    x(xi[3]); x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cy_lo[3], ancilla)
    x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cy_hi[3], ancilla)
    x(idx[1]); x(idx[2])
    x(idx[0])
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[2]], -oracle_cr, ancilla)
    x(xi[2]); x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cy_lo[2], ancilla)
    x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cy_hi[2], ancilla)
    x(idx[0])
    x(idx[0]); x(idx[2])
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[1]], -oracle_cr, ancilla)
    x(xi[1]); x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cy_lo[1], ancilla)
    x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cy_hi[1], ancilla)
    x(idx[0]); x(idx[2])
    x(idx[0]); x(idx[1])
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[0]], -oracle_cr, ancilla)
    x(xi[0]); x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cy_lo[0], ancilla)
    x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cy_hi[0], ancilla)
    x(idx[0]); x(idx[1])
    # 2. k=0 dagger: exact 256-iter loop with negated angles (blocks commute)
    x(idx[0]); x(idx[1]); x(idx[2])
    for addr in range(1 << (2 * N_Y)):
        y_int  = addr >> N_Y
        xi_int = addr & ((1 << N_Y) - 1)
        for b in range(N_Y):
            if not ((y_int  >> (N_Y - 1 - b)) & 1):
                x(y[b])
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi[b])
        cudaq.control(_ry_gate, [idx[0], idx[1], idx[2],
                                  y[0], y[1], y[2], y[3],
                                  xi[0], xi[1], xi[2], xi[3]],
                      -exact_thetas[addr], ancilla)      # negated
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi[b])
        for b in range(N_Y):
            if not ((y_int  >> (N_Y - 1 - b)) & 1):
                x(y[b])
    x(idx[0]); x(idx[1]); x(idx[2])
    # 3. H† = H on idx
    h(idx[0]); h(idx[1]); h(idx[2])
    # 4. DQA layers†
    for ii in range(n_steps * 2):
        i = n_steps * 2 - 1 - ii
        if i % 2 == 0:
            cost_operator_dagger(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer_dagger(thetas[i], y)
    # 5. PDF†
    pdf_init_uniform(xi)
    # 6. Dicke†
    dicke_state_dagger(n_y, w_d, dicke_angles, y)


@cudaq.kernel
def _apply_full_q_exact_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                               cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                               n_y: int, oracle_cy: list[float], oracle_cr: float,
                               oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                               exact_thetas: list[float],
                               k: int, yx_reg: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    """Grover iterate using _a_op_exact_multi."""
    _apply_s_chi(k, idx, anc)
    _a_op_exact_multi_dagger(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                              oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo,
                              exact_thetas, yx_reg, idx, anc)
    _apply_s0_full(yx_reg, idx, anc)
    _a_op_exact_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                       oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo,
                       exact_thetas, yx_reg, idx, anc)


@cudaq.kernel
def dqa_mlqae_state_kernel_exact_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                                         cost_norm: float, w_d: int, thetas: list[float],
                                         n_steps: int, n_y: int,
                                         oracle_cy: list[float], oracle_cr: float,
                                         oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                                         exact_thetas: list[float], k: int, m_power: int):
    """Prepare Q^m_power · A_exact_multi |0⟩ — exact phi and gradients for any W_D."""
    q = cudaq.qvector(N_FULL)
    yx_reg = q[0 : N_Q_Y + N_SCEN_Q]
    idx = q[N_Q_Y + N_SCEN_Q : N_Q_Y + N_SCEN_Q + N_IDX_Q]
    anc = q[N_FULL - 1]
    _a_op_exact_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                       oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo,
                       exact_thetas, yx_reg, idx, anc)
    for _ in range(m_power):
        _apply_full_q_exact_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                                    oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo,
                                    exact_thetas, k, yx_reg, idx, anc)


# =============================================================================
# Python wrappers: sampled P(idx==k, anc==1) and MLQAE estimation
# for the exact multiplexed pipeline (dqa_mlqae_state_kernel_exact_multi).
# =============================================================================


def dqa_p_good_exact_full(dicke_angles: list, c_y: list, c_r: float,
                           cost_norm: float, w_d: int, thetas: list, n_steps: int,
                           n_y: int, oracle_cy: list, oracle_cr: float,
                           oracle_cy_hi: list, oracle_cy_lo: list,
                           exact_thetas: list, k: int, m_power: int,
                           shots: int = 10000) -> float:
    """P(idx==k, anc==1) estimated via cudaq.sample on dqa_mlqae_state_kernel_exact_multi.

    Bit-string convention: cudaq.sample returns bitstrings in little-endian order
    where bits[q] is the measured value of qubit q — consistent with the
    format(i, "0Nb")[::-1] convention used in the statevector path.
    No reversal is needed.
    """
    result = cudaq.sample(
        dqa_mlqae_state_kernel_exact_multi,
        [float(v) for v in dicke_angles],
        [float(v) for v in c_y],
        float(c_r),
        float(cost_norm),
        int(w_d),
        [float(v) for v in thetas],
        int(n_steps),
        int(n_y),
        [float(v) for v in oracle_cy],
        float(oracle_cr),
        [float(v) for v in oracle_cy_hi],
        [float(v) for v in oracle_cy_lo],
        [float(v) for v in exact_thetas],
        int(k),
        int(m_power),
        shots_count=shots,
    )
    idx_start = N_Q_Y + N_SCEN_Q
    good  = 0
    total = 0
    for bits, count in result.items():
        total += count
        # bits[q] = qubit q directly (little-endian, no reversal needed)
        if int(bits[idx_start : idx_start + N_IDX_Q], 2) == k and int(bits[N_FULL - 1]) == 1:
            good += count
    return float(good) / total if total > 0 else 0.0


def dqa_mlqae_estimate_exact_full(dicke_angles: list, c_y: list, c_r: float,
                                    cost_norm: float, w_d: int, thetas: list, n_steps: int,
                                    n_y: int, oracle_cy: list, oracle_cr: float,
                                    oracle_cy_hi: list, oracle_cy_lo: list,
                                    exact_thetas: list,
                                    k: int,
                                    schedule: Iterable[int] = (0, 1, 2, 4, 8, 16),
                                    shots: int = 2000,
                                    seed: int = 0) -> tuple[float, float]:
    """MLQAE estimate using the exact multiplexed kernel (all k, any W_D)."""
    schedule = tuple(int(v) for v in schedule)
    p_true = np.array([dqa_p_good_exact_full(
        dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
        oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, exact_thetas, k, m_power,
        shots=shots)
        for m_power in schedule])
    hits = np.round(p_true * shots).astype(int)
    m_arr = np.asarray(schedule, dtype=float)
    h_arr = hits.astype(float)
    n_shots = float(shots)
    def neg_ll_scalar(theta: float) -> float:
        p = np.sin((2.0 * m_arr + 1.0) * theta) ** 2
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -float(np.sum(h_arr * np.log(p) + (n_shots - h_arr) * np.log(1.0 - p)))
    m_max = max(schedule)
    grid_size = max(4000, 200 * (2 * m_max + 1))
    theta_grid = np.linspace(1e-6, np.pi / 2 - 1e-6, grid_size)
    p_grid = np.sin(np.outer(2.0 * m_arr + 1.0, theta_grid)) ** 2
    p_grid = np.clip(p_grid, 1e-12, 1.0 - 1e-12)
    ll_grid = (h_arr[:, None] * np.log(p_grid) + (n_shots - h_arr)[:, None] * np.log(1.0 - p_grid)).sum(axis=0)
    best_i = int(np.argmax(ll_grid))
    res = minimize_scalar(
        neg_ll_scalar,
        bounds=(theta_grid[max(0, best_i - 1)], theta_grid[min(grid_size - 1, best_i + 1)]),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(res.x), float(np.sin(res.x))


def quantum_expectations_dqa_mlqae_exact(dicke_angles: list, c_y: list, c_r: float,
                                          cost_norm: float, w_d: int, thetas: list, n_steps: int,
                                          n_y: int, oracle_cy: list, oracle_cr: float,
                                          oracle_cy_hi: list, oracle_cy_lo: list,
                                          exact_thetas: list,
                                          f_min: float, f_max: float,
                                          schedule: Iterable[int],
                                          shots: int,
                                          seed_base: int = 0) -> np.ndarray:
    """Decode exact-multi MLQAE estimates for k=0..N_USED_IDX-1 back to f-units."""
    span = f_max - f_min if f_max > f_min else 1.0
    out = np.zeros(N_USED_IDX)
    for k in range(N_USED_IDX):
        _, a_hat = dqa_mlqae_estimate_exact_full(
            dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
            oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, exact_thetas, k,
            schedule=schedule, shots=shots, seed=seed_base + k,
        )
        out[k] = f_min + span * N_IDX * (a_hat ** 2)
    return out


def dqa_check_sin_squared_identity_exact(dicke_angles: list, c_y: list, c_r: float,
                                          cost_norm: float, w_d: int, thetas: list,
                                          n_steps: int, n_y: int,
                                          oracle_cy: list, oracle_cr: float,
                                          oracle_cy_hi: list, oracle_cy_lo: list,
                                          exact_thetas: list, k: int,
                                          schedule: Iterable[int]) -> float:
    """Sin² identity check using the exact multiplexed kernel (all k non-trivial)."""
    a0 = math.sqrt(max(0.0, min(1.0, dqa_p_good_exact_full(
        dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
        oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, exact_thetas, k, 0))))
    theta0 = math.asin(a0)
    err = 0.0
    for m_power in schedule:
        pred = math.sin((2 * m_power + 1) * theta0) ** 2
        meas = dqa_p_good_exact_full(
            dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
            oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, exact_thetas, k, m_power)
        err = max(err, abs(pred - meas))
    return err


def dicke_state_angles(n: int, k: int) -> List[float]:
    """Pre-compute SCS angles for |D_n^k⟩ (Bärtschi & Eidenbenz 2019).

    Returns a flat list of angles consumed by dicke_state() in order:
      - (n-k)*k angles for the first (n-k) SCS steps (phase 1),
      - k*(k-1)//2 angles for the remaining (k-1) SCS steps (phase 2).
    Each angle theta_j = 2*arccos(sqrt(j/m)) for SCS(m, l) at position j.

    Example (n=4, k=2) — matches the inline constants in dicke_state_n4_k2:
      [2*acos(sqrt(1/4)), pi/2, 2*acos(sqrt(1/3)), 2*acos(sqrt(2/3)), pi/2]
    """
    angles: List[float] = []
    # Phase 1: (n-k) steps of SCS(n-i, k) for i = 0 .. n-k-1
    for i in range(n - k):
        m = n - i
        for j in range(1, k + 1):
            angles.append(2.0 * math.acos(math.sqrt(j / m)))
    # Phase 2: (k-1) steps of SCS(k-i, k-1-i) for i = 0 .. k-2
    for i in range(k - 1):
        m = k - i
        l = k - 1 - i
        for j in range(1, l + 1):
            angles.append(2.0 * math.acos(math.sqrt(j / m)))
    return angles


# =============================================================================
# Pure DQA: prepare the DQA approximate-optimal state on y+xi registers only
# (no idx / ancilla qubits, no MLQAE wrapper) and evaluate the objective and
# gradient expectation values directly from the statevector.
# =============================================================================

@cudaq.kernel
def dqa_state_kernel(dicke_angles: list[float], c_y: list[float], c_r: float,
                     cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                     n_y: int):
    """DQA approximate-optimal state on the y + xi registers (no idx / ancilla).

    Prepares |D_{n_y}^{w_d}> on y and H^{n_y} on xi, then applies n_steps
    interleaved cost-operator (gamma) / XY-mixer (beta) layers.
    """
    q  = cudaq.qvector(N_Q_Y + N_SCEN_Q)
    y  = q[0 : N_Q_Y]
    xi = q[N_Q_Y : N_Q_Y + N_SCEN_Q]
    dicke_state(n_y, w_d, dicke_angles, y)
    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)


def dqa_expectation_values(dicke_angles: list, c_y: list, c_r: float,
                           cost_norm: float, w_d: int, thetas: list,
                           n_steps: int, n_y: int,
                           shots: int = 10000) -> tuple[float, np.ndarray]:
    """Compute objective and gradient expectation values via cudaq.sample on dqa_state_kernel.

    Returns (phi, grad) where:
      phi     = sum_{y,xi} p(y,xi) * Q(y,xi)
      grad[j] = sum_{y,xi} p(y,xi) * dQ/dy_j(y,xi)
    with p(y,xi) estimated from cudaq.sample shot counts.
    Bit-string convention: cudaq.sample returns bitstrings in little-endian order
    where bits[q] is the measured value of qubit q.  No reversal is needed.
    """
    result = cudaq.sample(
        dqa_state_kernel,
        [float(v) for v in dicke_angles],
        [float(v) for v in c_y],
        float(c_r),
        float(cost_norm),
        int(w_d),
        [float(v) for v in thetas],
        int(n_steps),
        int(n_y),
        shots_count=shots,
    )
    phi   = 0.0
    grad  = np.zeros(n_y)
    total = sum(result.values())
    for bits, count in result.items():
        # bits[q] = qubit q directly (little-endian, no reversal needed)
        y_tup  = tuple(int(bits[j])       for j in range(n_y))
        xi_tup = tuple(int(bits[n_y + j]) for j in range(n_y))
        w = count / total
        phi += w * _wind_scenario_cost(y_tup, xi_tup)
        for j in range(n_y):
            grad[j] += w * _scenario_gradient_component(j, y_tup, xi_tup)
    return phi, grad

#### Running functions ####

def point_test(schedule: tuple[int, ...], shots: int, trials: int):

    t0_total = _time.perf_counter()

    t0 = _time.perf_counter()
    f_used, f_full, f_min, f_max = build_per_scenario_table()
    oracle_thetas = build_oracle_angles(f_full, f_min, f_max)
    phi_classical, grad_classical = classical_truth()
    truth = np.concatenate([[phi_classical], grad_classical])
    print(f"[setup] classical table + truth: {_time.perf_counter() - t0:.3f}s")
    
    print("\nUC-consistent multiplexed MLQAE (CUDA-Q)")
    print(f"target={cudaq.get_target().name} | schedule={schedule} | shots={shots}")
    print(f"n_y={N_Y}, d={D}, x0={X0}, w_d={W_D}, c_r={C_R}")
    print(f"c_y_eff={[round(v, 4) for v in C_Y_EFF]}")
    print(f"n_steps_dqa={N_T} | cost_norm={max(C_Y_EFF + [C_R]):.4f} | exact_norm={W_D * C_R:.4f}")
    print("\n" + "=" * 60)

    if RUN_GROVER_IDENTITY_CHECKS:
        t0 = _time.perf_counter()
        print("\nStep A: exact Grover-power identity checks")
        for k in range(N_USED_IDX):
            err = check_sin_squared_identity(oracle_thetas, k, schedule)
            label = "phi" if k == 0 else f"grad[{k-1}]"
            print(f"  k={k} ({label}): max |P_m - sin^2((2m+1)theta)| = {err:.3e}")
        print(f"[Step A] {_time.perf_counter() - t0:.3f}s")

    t0 = _time.perf_counter()
    print("\nStep B: MLQAE recovery vs simple_ed-consistent classical truth")
    print(f"  classical [phi, grad_0..grad_{N_Y-1}] = {np.round(truth, 6)}")

    max_abs = 0.0
    for trial in range(trials):
        est = quantum_expectations_mlqae(
            oracle_thetas=oracle_thetas,
            f_min=f_min,
            f_max=f_max,
            schedule=schedule,
            shots=shots,
            seed_base=100 * trial,
        )
        abs_err = np.abs(est - truth)
        max_abs = max(max_abs, float(abs_err.max()))

        print(f"\ntrial {trial}:")
        print(f"  MLQAE estimate [phi, grad_0..grad_{N_Y-1}] = {np.round(est, 6)}")
        print(f"  abs error per component                    = {np.round(abs_err, 6)}")

    print(f"\nMAX absolute error across trials/components: {max_abs:.3e}")
    print("Reference analytical grad under uniform xi:")
    analytical_grad = [0.5 * C_Y_EFF[j] + 0.5 * C_R for j in range(N_Y)]
    print(f"  {np.round(analytical_grad, 6)}")
    print(f"[Step B] {_time.perf_counter() - t0:.3f}s")
    print("\n" + "=" * 60)

    # --- DQA parameters ---
    # cost_norm_dqa: normalisation for DQA cost/mixer layers (per-turbine max cost).
    # exact_norm:    normalisation for the exact oracle (W_D turbines * max single cost).
    # thetas_dqa:    linear annealing schedule gamma/beta pairs for DQA layers.
    t0 = _time.perf_counter()
    dicke_angles = list(dicke_state_angles(N_Y, W_D))
    cost_norm_dqa = float(max(C_Y_EFF + [C_R]))
    n_steps_dqa = N_T
    thetas_dqa: list[float] = []
    for t in range(n_steps_dqa):
        thetas_dqa.append(float(t / n_steps_dqa))
        thetas_dqa.append((1 - float(t / n_steps_dqa)) / math.pi)
    # Exact oracle: norm = W_D * C_R  (max total cost = W_D turbines all on recourse)
    exact_norm = float(W_D * C_R)
    exact_thetas = build_exact_oracle_thetas(exact_norm)
    print(f"\n[DQA setup] {_time.perf_counter() - t0:.3f}s")
    print(f"cost_norm_dqa={cost_norm_dqa:.2f} | exact_norm={exact_norm:.2f} | n_steps={n_steps_dqa}")

    # --- Exact multiplexed oracle: oracle angles normalised by exact_norm for uniform decode ---
    t0 = _time.perf_counter()
    oracle_cy_exact  = [2.0 * math.asin(math.sqrt(min(c / exact_norm, 1.0))) for c in C_Y_EFF]
    oracle_cr_exact  = 2.0 * math.asin(math.sqrt(min(C_R / exact_norm, 1.0)))
    oracle_cy_hi_exact = [2.0 * math.asin(math.sqrt(min((C_Y[j][0] + 2.0*C_Y[j][1]) / exact_norm, 1.0)))
                          for j in range(N_Y)]
    oracle_cy_lo_exact = [2.0 * math.asin(math.sqrt(min(C_Y[j][0] / exact_norm, 1.0)))
                          for j in range(N_Y)]
    # Decode: out[k] = exact_norm * N_IDX * a_hat^2  for all k (uniform norm)
    f_min_exact_multi = 0.0
    f_max_exact_multi = exact_norm
    print(f"[exact multi setup] {_time.perf_counter()-t0:.3f}s")

    # --- Pure DQA: direct statevector expectation values (no MLQAE) ---
    t0 = _time.perf_counter()
    print("\n" + "=" * 60)
    print("Pure DQA: objective and gradient expectation values")
    print(f"  n_steps={n_steps_dqa} | cost_norm={cost_norm_dqa:.2f}")
    print(f"  classical [phi, grad_0..grad_{N_Y-1}] = {np.round(truth, 6)}")
    max_abs_dqa = 0.0
    for trial in range(trials):
        t_trial = _time.perf_counter()
        phi_dqa, grad_dqa = dqa_expectation_values(
            dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
            thetas_dqa, n_steps_dqa, N_Y, shots=shots,
        )
        dqa_est = np.concatenate([[phi_dqa], grad_dqa])
        dqa_err = np.abs(dqa_est - truth)
        max_abs_dqa = max(max_abs_dqa, float(dqa_err.max()))
        print(f"\ntrial {trial} ({_time.perf_counter()-t_trial:.2f}s):")
        print(f"  DQA  [phi, grad_0..grad_{N_Y-1}] = {np.round(dqa_est, 6)}")
        print(f"  abs error per component           = {np.round(dqa_err, 6)}")
    print(f"\nMAX absolute error across trials/components: {max_abs_dqa:.3e}")
    print(f"[Pure DQA] {_time.perf_counter() - t0:.3f}s")

    # --- DQA/MLQAE ---
    print("\n" + "=" * 60)
    print("Exact multiplexed oracle (exact phi + gradients, any W_D)")
    print(f"exact_norm={exact_norm:.2f} | oracle_cr_exact={oracle_cr_exact:.4f}")

    if RUN_GROVER_IDENTITY_CHECKS:
        t0 = _time.perf_counter()
        print("\nStep A (exact multi): Grover-power identity checks (all k)")
        for k in range(N_USED_IDX):
            t_k = _time.perf_counter()
            err = dqa_check_sin_squared_identity_exact(
                dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
                thetas_dqa, n_steps_dqa, N_Y,
                oracle_cy_exact, oracle_cr_exact, oracle_cy_hi_exact, oracle_cy_lo_exact,
                exact_thetas, k, schedule)
            label = "phi" if k == 0 else f"grad[{k-1}]"
            print(f"  k={k} ({label}): max |P_m - sin^2((2m+1)theta)| = {err:.3e}  ({_time.perf_counter()-t_k:.2f}s)")
        print(f"[Step A (exact multi)] {_time.perf_counter()-t0:.3f}s")

    t0 = _time.perf_counter()
    print("\nStep B (exact multi): MLQAE all components (exact phi + gradients)")
    print(f"  classical [phi, grad_0..grad_{N_Y-1}] = {np.round(truth, 6)}")
    max_abs_exact_multi = 0.0
    for trial in range(trials):
        t_trial = _time.perf_counter()
        est_exact = quantum_expectations_dqa_mlqae_exact(
            dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
            thetas_dqa, n_steps_dqa, N_Y,
            oracle_cy_exact, oracle_cr_exact, oracle_cy_hi_exact, oracle_cy_lo_exact,
            exact_thetas, f_min_exact_multi, f_max_exact_multi,
            schedule=schedule, shots=shots, seed_base=100 * trial,
        )
        abs_err_exact = np.abs(est_exact - truth)
        max_abs_exact_multi = max(max_abs_exact_multi, float(abs_err_exact.max()))
        print(f"\ntrial {trial} ({_time.perf_counter()-t_trial:.2f}s):")
        print(f"  exact-multi [phi, grad_0..grad_{N_Y-1}] = {np.round(est_exact, 6)}")
        print(f"  abs error per component                 = {np.round(abs_err_exact, 6)}")
    print(f"\nMAX absolute error across trials/components: {max_abs_exact_multi:.3e}")
    print(f"[Step B (exact multi)] {_time.perf_counter()-t0:.3f}s")

    print(f"\n[TOTAL] {_time.perf_counter() - t0_total:.3f}s")


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
          Required for 'hermite', 'parabola', 'cubic', 'quartic', and 'sextic' modes.
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

    def _fit_poly(self, degree: int) -> np.ndarray:
        """Least-squares polynomial fit to all cached values and gradients."""
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
        return coeffs

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

    def phi_interp(self, w_d_float: float) -> float:
        """Interpolated second-stage cost at continuous wind demand w_d_float."""
        w = max(0.0, min(float(w_d_float), float(self.n_y)))
        k = int(math.floor(w))
        if self.interp_mode in ('parabola', 'cubic', 'quartic', 'sextic'):
            assert self._poly_coeffs is not None
            return float(np.polyval(self._poly_coeffs, w))
        if k >= self.n_y:
            return self.phi_cache.get(self.n_y, float('inf'))
        if self.interp_mode == 'floor':
            return self.phi_cache.get(k, float('inf'))
        t  = w - k
        p0 = self.phi_cache.get(k,     float('inf'))
        p1 = self.phi_cache.get(k + 1, float('inf'))
        if self.interp_mode == 'linear':
            return (1.0 - t) * p0 + t * p1
        m0 = self.phi_grad_cache.get(k,     0.0)
        m1 = self.phi_grad_cache.get(k + 1, 0.0)
        t2 = t * t;  t3 = t2 * t
        return ((2*t3 - 3*t2 + 1)*p0 + (t3 - 2*t2 + t)*m0
                + (-2*t3 + 3*t2)*p1  + (t3 - t2)*m1)

    def phi_interp_deriv(self, w_d_float: float) -> float:
        """Derivative of phi_interp w.r.t. w_d_float."""
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

    def quantum_obj(self, x_val: float, d_val: float) -> float:
        """Total objective: first-stage cost + phi_interp(d_val - x_val)."""
        return self.fs_cost(x_val) + self.phi_interp(d_val - x_val)

    def quantum_obj_and_grad(self, x_val: float, d_val: float):
        """Total objective and gradient w.r.t. x (for L-BFGS-B / BFGS)."""
        w_d_float = d_val - x_val
        obj  = self.fs_cost(x_val) + self.phi_interp(w_d_float)
        grad = self.fs_cost_grad(x_val) - self.phi_interp_deriv(w_d_float)
        return obj, np.array([grad])


def optimization_test(schedule: tuple[int, ...], shots: int):
    """Optimization sweep using DQA and DQA-MLQAE pipelines.

    Pre-computes phi_cache / phi_grad_cache for both pipelines via shot-based
    DQA sampling and exact-multiplexed-oracle MLQAE, then sweeps over demand
    levels (D_VALUES), interpolation modes (SWEEP_INTERP_MODES), and
    optimization methods (OPT_METHODS).  Results are saved to a CSV file.
    """
    import csv as _csv_mod

    t0_total = _time.perf_counter()

    # ── DQA parameter setup ───────────────────────────────────────────────────
    cost_norm_dqa = float(max(C_Y_EFF + [C_R]))
    n_steps_dqa   = N_T
    thetas_dqa: list[float] = []
    for t in range(n_steps_dqa):
        thetas_dqa.append(float(t / n_steps_dqa))
        thetas_dqa.append((1.0 - float(t / n_steps_dqa)) / math.pi)

    _c0        = [C_Y[j][0] if isinstance(C_Y[j], list) else C_Y[j] for j in range(N_Y)]
    phi_grad_0 = float(np.mean([0.5 * (_c0[j] + C_R) for j in range(N_Y)]))
    print(f"DQA setup: cost_norm={cost_norm_dqa:.2f}, n_steps={n_steps_dqa}")
    print(f"Analytical grad(w_d=0) = {phi_grad_0:.4f}")

    # ── Pre-compute phi caches for DQA and DQA-MLQAE ─────────────────────────
    phi_cache_dqa:        dict = {0: 0.0}
    phi_grad_cache_dqa:   dict = {0: phi_grad_0}
    phi_cache_mlqae:      dict = {0: 0.0}
    phi_grad_cache_mlqae: dict = {0: phi_grad_0}

    print(f"\nPre-computing phi_cache (DQA and DQA-MLQAE, w_d = 1 .. {N_Y}) ...")
    t0_cache = _time.perf_counter()
    for wd in range(1, N_Y + 1):
        t0_wd         = _time.perf_counter()
        da            = list(dicke_state_angles(N_Y, wd))
        exact_norm_wd = float(wd * C_R)

        # ── Pure DQA: cudaq.sample on dqa_state_kernel ──
        t0 = _time.perf_counter()
        phi_wd, grad_wd = dqa_expectation_values(
            da, C_Y_EFF, C_R, cost_norm_dqa, wd,
            thetas_dqa, n_steps_dqa, N_Y, shots=shots,
        )
        phi_cache_dqa[wd]      = float(phi_wd)
        phi_grad_cache_dqa[wd] = float(np.mean(grad_wd))
        print(f"  [DQA]       w_d={wd}: phi={phi_wd:.4f}  mean_grad={np.mean(grad_wd):.4f}"
              f"  ({(_time.perf_counter()-t0)*1e3:.1f} ms)")

        # ── DQA-MLQAE: exact multiplexed oracle via dqa_mlqae_state_kernel_exact_multi ──
        t0 = _time.perf_counter()
        oracle_cy_wd    = [2.0 * math.asin(math.sqrt(min(c / exact_norm_wd, 1.0)))
                           for c in C_Y_EFF]
        oracle_cr_wd    = 2.0 * math.asin(math.sqrt(min(C_R / exact_norm_wd, 1.0)))
        oracle_cy_hi_wd = [2.0 * math.asin(math.sqrt(
                               min((C_Y[j][0] + 2.0 * C_Y[j][1]) / exact_norm_wd, 1.0)))
                           for j in range(N_Y)]
        oracle_cy_lo_wd = [2.0 * math.asin(math.sqrt(
                               min(C_Y[j][0] / exact_norm_wd, 1.0)))
                           for j in range(N_Y)]
        exact_thetas_wd = _build_exact_thetas_wd(wd, exact_norm_wd)
        est             = quantum_expectations_dqa_mlqae_exact(
            da, C_Y_EFF, C_R, cost_norm_dqa, wd, thetas_dqa, n_steps_dqa, N_Y,
            oracle_cy_wd, oracle_cr_wd, oracle_cy_hi_wd, oracle_cy_lo_wd,
            exact_thetas_wd, 0.0, exact_norm_wd, schedule, shots,
        )
        phi_cache_mlqae[wd]      = float(est[0])
        phi_grad_cache_mlqae[wd] = float(np.mean(est[1:]))
        print(f"  [DQA-MLQAE] w_d={wd}: phi={est[0]:.4f}  mean_grad={np.mean(est[1:]):.4f}"
              f"  ({(_time.perf_counter()-t0)*1e3:.1f} ms)")
        print(f"  [w_d={wd} total] {(_time.perf_counter()-t0_wd):.3f}s")

    print(f"[cache] phi_cache build complete: {_time.perf_counter()-t0_cache:.3f}s")

    pipeline_caches = {
        'dqa':       (phi_cache_dqa,   phi_grad_cache_dqa),
        'dqa_mlqae': (phi_cache_mlqae, phi_grad_cache_mlqae),
    }

    # ── Optimization sweep ────────────────────────────────────────────────────
    col_w  = 12
    header = (f"{'pipeline':<{col_w}} {'interp':<10} {'d':<5} {'method':<14} "
              f"{'x*':>8} {'w_d*':>6} {'fs_obj':>10} {'ss_obj':>10} {'obj*':>10} "
              f"{'nfev':>6} {'nit':>5} {'conv':>5} {'ms':>8}")
    print(f"\n{header}")
    print('-' * len(header))

    sweep_results: list[dict] = []
    t0_sweep = _time.perf_counter()

    for pipeline_name, (pc, pgc) in pipeline_caches.items():
        t0_pipeline = _time.perf_counter()
        for _sweep_mode in SWEEP_INTERP_MODES:
            t0_mode    = _time.perf_counter()
            _sh        = TwoStageOptHelper(C_X, N_Y, _sweep_mode, pc, pgc)
            _qobj      = _sh.quantum_obj
            _qobj_grad = _sh.quantum_obj_and_grad
            _phi_sw    = _sh.phi_interp

            for d_val in D_VALUES:
                _xlo = float(max(0.0, d_val - N_Y))
                _xhi = float(d_val)
                _x0  = (_xlo + _xhi) / 2.0

                # Classical reference for this demand level
                _d_int  = int(math.floor(d_val))
                _pdf_d  = {
                    tuple(int(v) for v in ('{0:0' + str(N_Y) + 'b}').format(i)): 1.0 / 2**N_Y
                    for i in range(2**N_Y)
                }
                _bno_d   = BinaryNestedOptimizer(C_X, C_Y_EFF, C_R, _pdf_d, _d_int, is_uniform=True)
                _exp_d   = _bno_d.brute_force_wind_demand_expectation_values()
                _xr_d    = list(range(int(_xlo), _d_int + 1))
                _obj_cl  = [_sh.fs_cost(float(xv)) + _exp_d[_d_int - xv] for xv in _xr_d]
                cl_x_s   = float(_xr_d[int(np.argmin(_obj_cl))]) if _obj_cl else float('nan')
                cl_obj_s = float(min(_obj_cl)) if _obj_cl else float('nan')

                for method in OPT_METHODS:
                    _t0m = _time.perf_counter()
                    try:
                        if method == 'bounded':
                            res = minimize_scalar(
                                lambda xv: _qobj(float(xv), d_val),
                                bounds=(_xlo, _xhi),
                                method='bounded',
                                options={'xatol': 1e-6},
                            )
                            x_s, obj_s, ok = float(res.x), float(res.fun), res.success
                            nfev = getattr(res, 'nfev', None)
                            nit  = None
                        elif method == 'COBYLA':
                            res = minimize(
                                lambda xv: _qobj(float(xv[0]), d_val),
                                [_x0],
                                method='COBYLA',
                                constraints=[
                                    {'type': 'ineq', 'fun': lambda xv: xv[0] - _xlo},
                                    {'type': 'ineq', 'fun': lambda xv: _xhi - xv[0]},
                                ],
                                options={'maxiter': 500, 'rhobeg': 0.5},
                            )
                            x_s, obj_s, ok = float(res.x[0]), float(res.fun), res.success
                            nfev = getattr(res, 'nfev', None)
                            nit  = getattr(res, 'nit', None)
                        elif method == 'Nelder-Mead':
                            res = minimize(
                                lambda xv: _qobj(float(np.clip(xv[0], _xlo, _xhi)), d_val),
                                [_x0],
                                method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8},
                            )
                            x_s  = float(np.clip(res.x[0], _xlo, _xhi))
                            obj_s, ok = float(res.fun), res.success
                            nfev = getattr(res, 'nfev', None)
                            nit  = getattr(res, 'nit', None)
                        elif method == 'L-BFGS-B':
                            if _sweep_mode == 'floor':
                                raise ValueError(
                                    "L-BFGS-B requires a smooth objective; "
                                    "use 'linear', 'hermite', or a polynomial interp mode.")
                            res = minimize(
                                lambda xv: _qobj_grad(float(xv[0]), d_val),
                                [_x0],
                                method='L-BFGS-B',
                                jac=True,
                                bounds=[(_xlo, _xhi)],
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
                            )
                            x_s, obj_s, ok = float(res.x[0]), float(res.fun), res.success
                            nfev = getattr(res, 'nfev', None)
                            nit  = getattr(res, 'nit', None)
                        elif method == 'BFGS':
                            if _sweep_mode == 'floor':
                                raise ValueError(
                                    "BFGS requires a smooth objective; "
                                    "use 'linear', 'hermite', or a polynomial interp mode.")
                            res = minimize(
                                lambda xv: _qobj_grad(
                                    float(np.clip(xv[0], _xlo, _xhi)), d_val),
                                [_x0],
                                method='BFGS',
                                jac=True,
                                options={'maxiter': 500, 'gtol': 1e-8},
                            )
                            x_s  = float(np.clip(res.x[0], _xlo, _xhi))
                            obj_s, ok = float(res.fun), res.success
                            nfev = getattr(res, 'nfev', None)
                            nit  = getattr(res, 'nit', None)
                        else:
                            raise ValueError(f"Unknown method: {method!r}")

                        wd_s = d_val - x_s
                        fs_s = _sh.fs_cost(x_s)
                        ss_s = _phi_sw(wd_s)
                        t_ms = (_time.perf_counter() - _t0m) * 1e3
                        sweep_results.append(dict(
                            pipeline=pipeline_name,
                            interp_mode=_sweep_mode,
                            d=d_val, method=method,
                            x_star=x_s, wd_star=wd_s,
                            fs_obj=fs_s, ss_obj=ss_s, obj_star=obj_s,
                            cl_x_star=cl_x_s, cl_obj_star=cl_obj_s,
                            nfev=nfev, nit=nit,
                            time_ms=t_ms, success=ok,
                        ))
                        _nfev_s = str(nfev) if nfev is not None else '--'
                        _nit_s  = str(nit)  if nit  is not None else '--'
                        _conv_s = 'yes' if ok else 'NO'
                        print(
                            f"{pipeline_name:<{col_w}} {_sweep_mode:<10} {d_val:<5} {method:<14} "
                            f"{x_s:>8.3f} {wd_s:>6.3f} {fs_s:>10.4f} {ss_s:>10.4f} "
                            f"{obj_s:>10.4f} {_nfev_s:>6} {_nit_s:>5} {_conv_s:>5} {t_ms:>8.1f}"
                        )
                    except Exception as exc:
                        print(f"{pipeline_name:<{col_w}} {_sweep_mode:<10} {d_val:<5} "
                              f"{method:<14} ERROR: {exc}")

            print(f"[sweep] {pipeline_name} / {_sweep_mode}: "
                  f"{len(D_VALUES)} demand levels in {_time.perf_counter()-t0_mode:.3f}s")

        print(f"[sweep] pipeline '{pipeline_name}' complete: "
              f"{_time.perf_counter()-t0_pipeline:.3f}s")

    print(f"[sweep] all pipelines complete: {_time.perf_counter()-t0_sweep:.3f}s")

    # ── Save results to CSV ───────────────────────────────────────────────────
    _csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'simple_ed_example_opt_sweep.csv')
    if sweep_results:
        _fieldnames = list(sweep_results[0].keys())
        with open(_csv_path, 'w', newline='') as _f:
            _w = _csv_mod.DictWriter(_f, fieldnames=_fieldnames)
            _w.writeheader()
            _w.writerows(sweep_results)
        print(f"\nSweep results saved to {_csv_path}")

    print(f"\n[TOTAL optimization_test] {_time.perf_counter() - t0_total:.3f}s")


####  Main ####

def _parse_schedule(text: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("schedule cannot be empty")
    if min(vals) < 0:
        raise ValueError("schedule values must be non-negative")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="UC-consistent multiplexed MLQAE (CUDA-Q)")
    parser.add_argument("--target", type=str, default="nvidia", help="CUDA-Q target (qpp-cpu or nvidia)")
    parser.add_argument("--schedule", type=str, default="0,1,2,4,8,16", help="Comma-separated Grover powers")
    parser.add_argument("--shots", type=int, default=4000, help="Shots per Grover power for MLQAE likelihood data")
    parser.add_argument("--trials", type=int, default=1, help="Number of MLQAE repeated trials")
    parser.add_argument("--mode", type=str, default="point",
                        choices=["point", "opt"],
                        help="'point' runs point_test; 'opt' runs optimization_test sweep")
    args = parser.parse_args()

    schedule = _parse_schedule(args.schedule)

    try:
        cudaq.set_target(args.target)
    except Exception as exc:
        print(f"WARNING: failed to set target '{args.target}' ({exc}); using qpp-cpu.")
        cudaq.set_target("qpp-cpu")

    if args.mode == 'opt':
        optimization_test(schedule, args.shots)
    else:
        point_test(schedule, args.shots, args.trials)

if __name__ == "__main__":
    main()
