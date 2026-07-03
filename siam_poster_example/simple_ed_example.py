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
from scipy.optimize import minimize_scalar
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

# from cudaq_impl import *

RNG = np.random.default_rng(0)

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


#### Full DQA / MLQAE pipeline ####

# -----------------------------------------------------------------------------
# SCS gadget for Dicke state preparation (Bärtschi & Eidenbenz 2019)
# Each SCS(n, k) acts on k+1 qubits: [q_{ni-k} .. q_{ni-1}, q_{ni}]
# Qiskit counterpart: the inner SCS() closure in:
#   ExpValFun_functions.dicke_state_circuit() and
#   BinaryNestedOptimizer.dicke_state_circuit() in binary_optimizer.py
#   Qiskit uses scs.cx / scs.cry / scs.append(RYGate(...).control(2))
# -----------------------------------------------------------------------------
@cudaq.kernel
def scs_2(theta_half: float, q0: cudaq.qubit, q1: cudaq.qubit):
    """SCS gadget for k=1 (2 qubits).  Angle = 2*arccos(sqrt(1/n)).

    Qiskit counterpart: SCS() closure inside dicke_state_circuit().
    Implements: cx(ni-1, ni); cry(angle, ni, ni-1); cx(ni-1, ni)
    """
    cx(q0, q1)
    cry(theta_half, q1, q0)  # controlled-RY: control=q1, target=q0
    cx(q0, q1)


@cudaq.kernel
def scs_3(theta1: float, theta2: float,
          q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit):
    """SCS gadget for k=2 (3 qubits).
    theta1 = 2*arccos(sqrt(1/n)), theta2 = 2*arccos(sqrt(2/n))."""
    # First layer (l=1 in the reference)
    cx(q1, q2)
    cry(theta1, q2, q1)
    cx(q1, q2)
    # Second layer (l=2)
    cx(q0, q2)
    ccx(q2, q1, q0)           # ccry approximation: use Toffoli + RY pattern
    # NOTE: CUDA-Q does not have a native CCRy; decompose as:
    #   cry(theta/2, q1, q0); cx(q2, q1); cry(-theta/2, q1, q0); cx(q2, q1)
    ccx(q2, q1, q0)           # undo Toffoli to restore ancilla -- placeholder
    cx(q0, q2)


# # Cleaner CCRy decomposition used in Dicke state preparation
# # Qiskit counterpart: RYGate().control(2) applied via qc.append()
# #   in both dicke_state_circuit() and single_oracle_sin_inconstraint().
# #   CUDA-Q has no native CCRy gate; use cudaq.control(kernel, [controls], args)
# #   which compiles to a multiply-controlled version of the sub-kernel.
# @cudaq.kernel
# def _ry_gate(theta: float, q: cudaq.qubit):
#     """Single-qubit RY helper — used by ccry via cudaq.control."""
#     ry(theta, q)


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


@cudaq.kernel
def oracle_sin(oracle_cy: list[float], oracle_cr: float,
               y: cudaq.qview, xi: cudaq.qview, ancilla: cudaq.qubit):
    # Exact QAE encoding: angles pre-computed as 2*arcsin(sqrt(cost/norm))
    # so that P(ancilla=1) = cost/norm.  Call site must pass the pre-computed
    # angles; math.asin/sqrt are not available inside @cudaq.kernel.
    # Valid for w_d=1 (one turbine fires per basis state, no angle accumulation).
    for k in range(len(y)):
        ccry(oracle_cy[k], y[k], xi[k], ancilla)
        x(xi[k])
        ccry(oracle_cr, y[k], xi[k], ancilla)
        x(xi[k])
    ## end for
    return


@cudaq.kernel
def _a_op(dicke_angles: list[float],
          c_y: list[float], c_r: float, cost_norm: float, w_d: int,
          thetas: list[float], n_steps: int, n_y: int,
          oracle_cy: list[float], oracle_cr: float,
          qubits: cudaq.qview, ancilla: cudaq.qubit):
    """A operator: U_opt followed by F_sin oracle on ancilla.

    Prepares:  A|0⟩ = sqrt(1-a)|ψ_bad⟩|0⟩ + sqrt(a)|ψ_good⟩|1⟩
    where a = E[cost]/norm (exact encoding, valid for w_d=1).

    oracle_cy / oracle_cr: pre-computed angles [2*arcsin(sqrt(c/norm))].
    Qiskit counterpart: BernoulliA.__init__() — appends op then oracle.
    system qubits: qubits[0..2*n_y-1] (y register then xi register)
    ancilla: separate qubit
    """
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]

    # U_opt: Dicke state + PDF + DQA layers
    dicke_state(n_y, w_d, dicke_angles, y)
    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)

    # F_sin oracle: encodes E[cost] as Pr[ancilla=|1⟩] using exact encoding.
    oracle_sin(oracle_cy, oracle_cr, y, xi, ancilla)


@cudaq.kernel
def oracle_sin_dagger(oracle_cy: list[float], oracle_cr: float,
                      y: cudaq.qview, xi: cudaq.qview, ancilla: cudaq.qubit):
    """Adjoint of oracle_sin: reverse loop order, negate pre-computed CCRY angles.
    Gate sequence per iteration in forward: ccry(theta1), x, ccry(theta2), x.
    Adjoint per iteration (reversed):       x, ccry(-theta2), x, ccry(-theta1).
    Angles pre-computed as 2*arcsin(sqrt(cost/norm)) to match oracle_sin.
    """
    for kk in range(len(y)):
        k = len(y) - 1 - kk          # reverse loop
        x(xi[k])
        ccry(-oracle_cr, y[k], xi[k], ancilla)
        x(xi[k])
        ccry(-oracle_cy[k], y[k], xi[k], ancilla)


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

@cudaq.kernel
def _a_op_dagger(dicke_angles: list[float],
                 c_y: list[float], c_r: float, cost_norm: float, w_d: int,
                 thetas: list[float], n_steps: int, n_y: int,
                 oracle_cy: list[float], oracle_cr: float,
                 qubits: cudaq.qview, ancilla: cudaq.qubit):
    """Manual adjoint of _a_op.

    A = dicke_state · pdf_init · DQA_layers · oracle_sin
    A† = oracle_sin† · DQA_layers† · pdf_init† · dicke_state†

    oracle_cy / oracle_cr: pre-computed exact QAE angles [2*arcsin(sqrt(c/norm))].
    Replaces cudaq.adjoint(_a_op, ...) which fails in CUDA-Q 0.14.2 with:
      RuntimeError: could not autogenerate the adjoint of a kernel
    """
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]

    # 1. F_sin oracle†
    oracle_sin_dagger(oracle_cy, oracle_cr, y, xi, ancilla)

    # 2. DQA layers† (reverse order, dagger of each layer)
    for ii in range(n_steps * 2):
        i = n_steps * 2 - 1 - ii    # i runs from last index down to 0
        if i % 2 == 0:
            cost_operator_dagger(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer_dagger(thetas[i], y)

    # 3. PDF uniform† = H^⊗n (H is self-adjoint)
    pdf_init_uniform(xi)

    # 4. Dicke state†
    dicke_state_dagger(n_y, w_d, dicke_angles, y)


# ---------------------------------------------------------------------------
# Exact oracle (CUDA-Q port of Qiskit exact_oracle)
# One 8-qubit-controlled RY per (y, xi) basis state: no angle accumulation.
# Correct for any W_D.  Gate count O(2^(2*N_Y)) = 256 for N_Y=4.
# addr = (y_int << N_Y) | xi_int, MSB-first convention.
# ---------------------------------------------------------------------------
@cudaq.kernel
def exact_oracle_k0(oracle_thetas: list[float], y_reg: cudaq.qview,
                    xi_reg: cudaq.qview, anc: cudaq.qubit):
    """Exact oracle for phi (k=0): single 8-ctrl-RY per (y, xi) state.

    Hardcoded for N_Y=4 (256 addresses).  oracle_thetas[addr] pre-computed as
    2*arcsin(sqrt(Q(y,xi)/norm)) for sum(y)==W_D, 0 for infeasible states.
    P(anc=1) = Q(y,xi)/norm exactly for any W_D; no sequential-angle bias.
    """
    for addr in range(1 << (2 * N_Y)):   # 256 iterations; unrolled at compile time
        y_int  = addr >> N_Y
        xi_int = addr & ((1 << N_Y) - 1)
        # Wrap y: flip y_reg[b] where y_int bit b is 0 (MSB-first)
        for b in range(N_Y):
            if not ((y_int >> (N_Y - 1 - b)) & 1):
                x(y_reg[b])
        # Wrap xi
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi_reg[b])
        # 8-qubit controlled RY: fires only when system == (y_int, xi_int)
        cudaq.control(_ry_gate, [y_reg[0], y_reg[1], y_reg[2], y_reg[3],
                                  xi_reg[0], xi_reg[1], xi_reg[2], xi_reg[3]],
                      oracle_thetas[addr], anc)
        # Uncompute xi wrap
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi_reg[b])
        # Uncompute y wrap
        for b in range(N_Y):
            if not ((y_int >> (N_Y - 1 - b)) & 1):
                x(y_reg[b])


@cudaq.kernel
def exact_oracle_k0_dagger(oracle_thetas: list[float], y_reg: cudaq.qview,
                           xi_reg: cudaq.qview, anc: cudaq.qubit):
    """Adjoint of exact_oracle_k0: identical structure with negated angles.

    Blocks select orthogonal subspaces so they commute; adjoint = negate angles.
    """
    for addr in range(1 << (2 * N_Y)):
        y_int  = addr >> N_Y
        xi_int = addr & ((1 << N_Y) - 1)
        for b in range(N_Y):
            if not ((y_int >> (N_Y - 1 - b)) & 1):
                x(y_reg[b])
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi_reg[b])
        cudaq.control(_ry_gate, [y_reg[0], y_reg[1], y_reg[2], y_reg[3],
                                  xi_reg[0], xi_reg[1], xi_reg[2], xi_reg[3]],
                      -oracle_thetas[addr], anc)          # negated
        for b in range(N_Y):
            if not ((xi_int >> (N_Y - 1 - b)) & 1):
                x(xi_reg[b])
        for b in range(N_Y):
            if not ((y_int >> (N_Y - 1 - b)) & 1):
                x(y_reg[b])


@cudaq.kernel
def _a_op_exact_k0(dicke_angles: list[float], c_y: list[float], c_r: float,
                    cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                    n_y: int, exact_thetas: list[float],
                    qubits: cudaq.qview, ancilla: cudaq.qubit):
    """A operator with exact oracle: DQA state prep + exact_oracle_k0.  Correct for any W_D."""
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    dicke_state(n_y, w_d, dicke_angles, y)
    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)
    exact_oracle_k0(exact_thetas, y, xi, ancilla)


@cudaq.kernel
def _a_op_exact_k0_dagger(dicke_angles: list[float], c_y: list[float], c_r: float,
                           cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                           n_y: int, exact_thetas: list[float],
                           qubits: cudaq.qview, ancilla: cudaq.qubit):
    """Adjoint of _a_op_exact_k0."""
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    exact_oracle_k0_dagger(exact_thetas, y, xi, ancilla)
    for ii in range(n_steps * 2):
        i = n_steps * 2 - 1 - ii
        if i % 2 == 0:
            cost_operator_dagger(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer_dagger(thetas[i], y)
    pdf_init_uniform(xi)
    dicke_state_dagger(n_y, w_d, dicke_angles, y)


# ---------------------------------------------------------------------------
# Option B: multiplexed oracle — H on idx, then per-k oracle blocks
# Hardcoded for N_Y=4, N_IDX_Q=3, N_USED_IDX=5.
# Bit layout for wrap pattern (MSB first): k=0→000, k=1→001, k=2→010,
#   k=3→011, k=4→100.  For each k, flip idx bits that are 0 so |k>→|111>.
# ---------------------------------------------------------------------------
@cudaq.kernel
def _a_op_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                n_y: int, oracle_cy: list[float], oracle_cr: float,
                oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                qubits: cudaq.qview, idx: cudaq.qview, ancilla: cudaq.qubit):
    """A operator with multiplexed oracle: H on idx + per-k oracle conditioned on idx.

    k=0 encodes total cost (oracle_sin); k=j+1 encodes gradient of turbine j.
    Three cases for gradient: (xi=1,y=1)->oracle_cy_hi[j], (xi=1,y=0)->oracle_cy_lo[j],
    (xi=0)->oracle_cr.  Amplitude diluted by 1/sqrt(N_IDX).
    """
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    # DQA state prep
    dicke_state(n_y, w_d, dicke_angles, y)
    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)
    # Uniform superposition over index register
    h(idx[0]); h(idx[1]); h(idx[2])
    # k=0 (000): total cost — wrap all 3 idx bits, apply all 4 turbines
    x(idx[0]); x(idx[1]); x(idx[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cy[0], ancilla)
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cr, ancilla)
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cy[1], ancilla)
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cr, ancilla)
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cy[2], ancilla)
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cr, ancilla)
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cy[3], ancilla)
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cr, ancilla)
    x(xi[3])
    x(idx[0]); x(idx[1]); x(idx[2])
    # k=1 (001): gradient turbine 0 — flip idx[0], idx[1]
    # (xi[0]=1,y[0]=1): oracle_cy_hi[0];  (xi[0]=1,y[0]=0): oracle_cy_lo[0];  (xi[0]=0): oracle_cr
    x(idx[0]); x(idx[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cy_hi[0], ancilla)
    x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], oracle_cy_lo[0], ancilla)
    x(y[0])
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[0]], oracle_cr, ancilla)
    x(xi[0])
    x(idx[0]); x(idx[1])
    # k=2 (010): gradient turbine 1 — flip idx[0], idx[2]
    x(idx[0]); x(idx[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cy_hi[1], ancilla)
    x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], oracle_cy_lo[1], ancilla)
    x(y[1])
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[1]], oracle_cr, ancilla)
    x(xi[1])
    x(idx[0]); x(idx[2])
    # k=3 (011): gradient turbine 2 — flip idx[0]
    x(idx[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cy_hi[2], ancilla)
    x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], oracle_cy_lo[2], ancilla)
    x(y[2])
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[2]], oracle_cr, ancilla)
    x(xi[2])
    x(idx[0])
    # k=4 (100): gradient turbine 3 — flip idx[1], idx[2]
    x(idx[1]); x(idx[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cy_hi[3], ancilla)
    x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], oracle_cy_lo[3], ancilla)
    x(y[3])
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[3]], oracle_cr, ancilla)
    x(xi[3])
    x(idx[1]); x(idx[2])
    # k=5,6,7: padding — no oracle applied


@cudaq.kernel
def _a_op_multi_dagger(dicke_angles: list[float], c_y: list[float], c_r: float,
                       cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                       n_y: int, oracle_cy: list[float], oracle_cr: float,
                       oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                       qubits: cudaq.qview, idx: cudaq.qview, ancilla: cudaq.qubit):
    """Adjoint of _a_op_multi: oracle†(k=4..0) → H†(idx) → DQA†layers → H†(xi) → Dicke†."""
    y  = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]
    # 1. Oracle† — reverse k order, negate angles, reverse per-turbine order within each k
    # k=4 dagger (100 → flip idx[1], idx[2]); reverse of forward k=4 block
    x(idx[1]); x(idx[2])
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[3]], -oracle_cr, ancilla)
    x(xi[3])
    x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cy_lo[3], ancilla)
    x(y[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cy_hi[3], ancilla)
    x(idx[1]); x(idx[2])
    # k=3 dagger (011 → flip idx[0])
    x(idx[0])
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[2]], -oracle_cr, ancilla)
    x(xi[2])
    x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cy_lo[2], ancilla)
    x(y[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cy_hi[2], ancilla)
    x(idx[0])
    # k=2 dagger (010 → flip idx[0], idx[2])
    x(idx[0]); x(idx[2])
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[1]], -oracle_cr, ancilla)
    x(xi[1])
    x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cy_lo[1], ancilla)
    x(y[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cy_hi[1], ancilla)
    x(idx[0]); x(idx[2])
    # k=1 dagger (001 → flip idx[0], idx[1])
    x(idx[0]); x(idx[1])
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], xi[0]], -oracle_cr, ancilla)
    x(xi[0])
    x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cy_lo[0], ancilla)
    x(y[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cy_hi[0], ancilla)
    x(idx[0]); x(idx[1])
    # k=0 dagger (000 → flip all idx), reverse turbine order j=3..0
    x(idx[0]); x(idx[1]); x(idx[2])
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cr, ancilla)
    x(xi[3])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[3], xi[3]], -oracle_cy[3], ancilla)
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cr, ancilla)
    x(xi[2])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[2], xi[2]], -oracle_cy[2], ancilla)
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cr, ancilla)
    x(xi[1])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[1], xi[1]], -oracle_cy[1], ancilla)
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cr, ancilla)
    x(xi[0])
    cudaq.control(_ry_gate, [idx[0], idx[1], idx[2], y[0], xi[0]], -oracle_cy[0], ancilla)
    x(idx[0]); x(idx[1]); x(idx[2])
    # 2. H† = H on idx (H is self-adjoint)
    h(idx[0]); h(idx[1]); h(idx[2])
    # 3. DQA layers†
    for ii in range(n_steps * 2):
        i = n_steps * 2 - 1 - ii
        if i % 2 == 0:
            cost_operator_dagger(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer_dagger(thetas[i], y)
    # 4. PDF† = H^⊗xi (self-adjoint)
    pdf_init_uniform(xi)
    # 5. Dicke state†
    dicke_state_dagger(n_y, w_d, dicke_angles, y)


@cudaq.kernel
def _apply_full_q_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                         cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                         n_y: int, oracle_cy: list[float], oracle_cr: float,
                         oracle_cy_hi: list[float], oracle_cy_lo: list[float],
                         k: int, yx_reg: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    """Grover iterate Q = S_chi · A_multi† · S_0 · A_multi."""
    _apply_s_chi(k, idx, anc)
    _a_op_multi_dagger(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                        oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, yx_reg, idx, anc)
    _apply_s0_full(yx_reg, idx, anc)
    _a_op_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                 oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, yx_reg, idx, anc)


@cudaq.kernel
def dqa_mlqae_state_kernel_multi(dicke_angles: list[float], c_y: list[float], c_r: float,
                                   cost_norm: float, w_d: int, thetas: list[float],
                                   n_steps: int, n_y: int, oracle_cy: list[float],
                                   oracle_cr: float, oracle_cy_hi: list[float],
                                   oracle_cy_lo: list[float], k: int, m_power: int):
    """Prepare Q^m_power · A_multi |0> — all k values give non-trivial amplitudes."""
    q = cudaq.qvector(N_FULL)
    yx_reg = q[0 : N_Q_Y + N_SCEN_Q]
    idx = q[N_Q_Y + N_SCEN_Q : N_Q_Y + N_SCEN_Q + N_IDX_Q]
    anc = q[N_FULL - 1]
    _a_op_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                 oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, yx_reg, idx, anc)
    for _ in range(m_power):
        _apply_full_q_multi(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                             oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, k, yx_reg, idx, anc)


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


@cudaq.kernel
def _apply_full_q(dicke_angles: list[float], c_y: list[float], c_r: float,
                  cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                  n_y: int, oracle_cy: list[float], oracle_cr: float,
                  k: int, yx_reg: cudaq.qview, idx: cudaq.qview, anc: cudaq.qubit):
    """Grover iterate Q = S_chi · A† · S_0 · A using _a_op/_a_op_dagger."""
    _apply_s_chi(k, idx, anc)
    _a_op_dagger(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                 oracle_cy, oracle_cr, yx_reg, anc)
    _apply_s0_full(yx_reg, idx, anc)
    _a_op(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
          oracle_cy, oracle_cr, yx_reg, anc)


@cudaq.kernel
def dqa_mlqae_state_kernel(dicke_angles: list[float], c_y: list[float], c_r: float,
                            cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                            n_y: int, oracle_cy: list[float], oracle_cr: float,
                            k: int, m_power: int):
    """Prepare Q^m_power · A |0> using DQA state prep (_a_op) and full Grover iterate."""
    q = cudaq.qvector(N_FULL)
    yx_reg = q[0 : N_Q_Y + N_SCEN_Q]   # combined y (0..N_Q_Y-1) + xi (N_Q_Y..N_Q_Y+N_SCEN_Q-1)
    idx = q[N_Q_Y + N_SCEN_Q : N_Q_Y + N_SCEN_Q + N_IDX_Q]
    anc = q[N_FULL - 1]

    _a_op(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
          oracle_cy, oracle_cr, yx_reg, anc)
    for _ in range(m_power):
        _apply_full_q(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                      oracle_cy, oracle_cr, k, yx_reg, idx, anc)


@cudaq.kernel
def _apply_s0_yx(yx_reg: cudaq.qview, anc: cudaq.qubit):
    """S_0 reflection about |0...0⟩ for 8-qubit yx_reg + ancilla (no idx register)."""
    for q in yx_reg:
        x(q)
    x(anc)
    cudaq.control(_z_gate, [yx_reg[0], yx_reg[1], yx_reg[2], yx_reg[3],
                             yx_reg[4], yx_reg[5], yx_reg[6], yx_reg[7]], anc)
    x(anc)
    for q in yx_reg:
        x(q)


@cudaq.kernel
def _apply_full_q_exact_k0(dicke_angles: list[float], c_y: list[float], c_r: float,
                             cost_norm: float, w_d: int, thetas: list[float], n_steps: int,
                             n_y: int, exact_thetas: list[float],
                             yx_reg: cudaq.qview, anc: cudaq.qubit):
    """Grover iterate Q = S_chi(k=0) · A_exact† · S_0 · A_exact.

    S_chi(k=0) = Z on ancilla (flips phase of |good⟩ = anc=1 states).
    S_0 acts on yx_reg (8 qubits) + ancilla.
    """
    z(anc)                                       # S_chi for k=0
    _a_op_exact_k0_dagger(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                           exact_thetas, yx_reg, anc)
    _apply_s0_yx(yx_reg, anc)                    # S_0
    _a_op_exact_k0(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                    exact_thetas, yx_reg, anc)


@cudaq.kernel
def dqa_mlqae_state_kernel_exact_k0(dicke_angles: list[float], c_y: list[float], c_r: float,
                                      cost_norm: float, w_d: int, thetas: list[float],
                                      n_steps: int, n_y: int, exact_thetas: list[float],
                                      m_power: int):
    """Prepare Q^m_power · A_exact |0⟩ for k=0 phi estimation (no idx register)."""
    n_qubits = N_Q_Y + N_SCEN_Q + 1    # y + xi + ancilla
    q = cudaq.qvector(n_qubits)
    yx_reg = q[0 : N_Q_Y + N_SCEN_Q]
    anc = q[N_Q_Y + N_SCEN_Q]
    _a_op_exact_k0(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                    exact_thetas, yx_reg, anc)
    for _ in range(m_power):
        _apply_full_q_exact_k0(dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
                                 exact_thetas, yx_reg, anc)


def dqa_p_good_exact(dicke_angles: list, c_y: list, c_r: float,
                     cost_norm: float, w_d: int, thetas: list, n_steps: int,
                     n_y: int, oracle_cy: list, oracle_cr: float,
                     k: int, m_power: int) -> float:
    """Exact P(idx==k, anc==1) from the DQA state kernel statevector."""
    sv = np.array(cudaq.get_state(
        dqa_mlqae_state_kernel,
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
        int(k),
        int(m_power),
    ))

    prob = 0.0
    idx_start = N_Q_Y + N_SCEN_Q
    for i in range(1 << N_FULL):
        p = abs(sv[i]) ** 2
        if p < 1e-14:
            continue
        b = format(i, f"0{N_FULL}b")[::-1]
        idx_int = int(b[idx_start : idx_start + N_IDX_Q], 2)
        anc_bit = int(b[N_FULL - 1])
        if idx_int == k and anc_bit == 1:
            prob += p
    return float(prob)


def dqa_p_good_exact_multi(dicke_angles: list, c_y: list, c_r: float,
                            cost_norm: float, w_d: int, thetas: list, n_steps: int,
                            n_y: int, oracle_cy: list, oracle_cr: float,
                            oracle_cy_hi: list, oracle_cy_lo: list,
                            k: int, m_power: int) -> float:
    """Exact P(idx==k, anc==1) using the multiplexed DQA kernel (all k valid)."""
    sv = np.array(cudaq.get_state(
        dqa_mlqae_state_kernel_multi,
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
        int(k),
        int(m_power),
    ))
    prob = 0.0
    idx_start = N_Q_Y + N_SCEN_Q
    for i in range(1 << N_FULL):
        p = abs(sv[i]) ** 2
        if p < 1e-14:
            continue
        b = format(i, f"0{N_FULL}b")[::-1]
        idx_int = int(b[idx_start : idx_start + N_IDX_Q], 2)
        anc_bit = int(b[N_FULL - 1])
        if idx_int == k and anc_bit == 1:
            prob += p
    return float(prob)


def dqa_p_good_exact_phi(dicke_angles: list, c_y: list, c_r: float,
                          cost_norm: float, w_d: int, thetas: list, n_steps: int,
                          n_y: int, exact_thetas: list, m_power: int) -> float:
    """Exact P(anc==1) using the exact k=0 oracle.  No angle accumulation for any W_D."""
    n_sys = N_Q_Y + N_SCEN_Q + 1   # 9 qubits; no idx register
    sv = np.array(cudaq.get_state(
        dqa_mlqae_state_kernel_exact_k0,
        [float(v) for v in dicke_angles],
        [float(v) for v in c_y],
        float(c_r),
        float(cost_norm),
        int(w_d),
        [float(v) for v in thetas],
        int(n_steps),
        int(n_y),
        [float(v) for v in exact_thetas],
        int(m_power),
    ))
    prob = 0.0
    for i in range(1 << n_sys):
        p = abs(sv[i]) ** 2
        if p < 1e-14:
            continue
        b = format(i, f"0{n_sys}b")[::-1]
        if int(b[n_sys - 1]) == 1:   # ancilla is last qubit
            prob += p
    return float(prob)


def dqa_mlqae_estimate_exact_phi(dicke_angles: list, c_y: list, c_r: float,
                                   cost_norm: float, w_d: int, thetas: list, n_steps: int,
                                   n_y: int, exact_thetas: list, exact_norm: float,
                                   schedule: Iterable[int] = (0, 1, 2, 4, 8, 16),
                                   shots: int = 2000,
                                   seed: int = 0) -> float:
    """MLQAE estimate of E_DQA[Q(y,xi)] using the exact oracle.  Returns decoded phi.

    Decode: P(anc=1) = E[Q/exact_norm], so phi_hat = exact_norm * a_hat^2.
    exact_norm should be >= W_D * max_single_cost (e.g. W_D * C_R).
    """
    schedule = tuple(int(v) for v in schedule)
    rng_local = np.random.default_rng(seed)

    p_true = np.array([dqa_p_good_exact_phi(dicke_angles, c_y, c_r, cost_norm, w_d,
                                              thetas, n_steps, n_y, exact_thetas, m_power)
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
    a_hat = math.sin(float(res.x))
    return exact_norm * (a_hat ** 2)   # P(anc=1) = E[Q]/exact_norm  =>  E[Q] = exact_norm*a_hat^2


def dqa_mlqae_estimate(dicke_angles: list, c_y: list, c_r: float,
                       cost_norm: float, w_d: int, thetas: list, n_steps: int,
                       n_y: int, oracle_cy: list, oracle_cr: float,
                       oracle_cy_hi: list, oracle_cy_lo: list,
                       k: int,
                       schedule: Iterable[int] = (0, 1, 2, 4, 8, 16),
                       shots: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """Return (theta_hat, a_hat) for index k using MLQAE with DQA state prep."""
    schedule = tuple(int(v) for v in schedule)
    rng_local = np.random.default_rng(seed)

    p_true = np.array([dqa_p_good_exact_multi(dicke_angles, c_y, c_r, cost_norm, w_d,
                                               thetas, n_steps, n_y, oracle_cy, oracle_cr,
                                               oracle_cy_hi, oracle_cy_lo,
                                               k, m_power) for m_power in schedule])
    # hits = rng_local.binomial(shots, p_true)   # stochastic; replace with line below for noiseless
    hits = np.round(p_true * shots).astype(int)  # noiseless: exact expected counts

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


def quantum_expectations_dqa_mlqae(dicke_angles: list, c_y: list, c_r: float,
                                    cost_norm: float, w_d: int, thetas: list, n_steps: int,
                                    n_y: int, oracle_cy: list, oracle_cr: float,
                                    oracle_cy_hi: list, oracle_cy_lo: list,
                                    f_min: float, f_max: float,
                                    schedule: Iterable[int],
                                    shots: int,
                                    seed_base: int = 0) -> np.ndarray:
    """Decode DQA-MLQAE estimates for k=0..N_USED_IDX-1 back to f-units."""
    span = f_max - f_min if f_max > f_min else 1.0
    out = np.zeros(N_USED_IDX)

    for k in range(N_USED_IDX):
        _, a_hat = dqa_mlqae_estimate(
            dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
            oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, k,
            schedule=schedule,
            shots=shots,
            seed=seed_base + k,
        )
        e_g_k = N_IDX * (a_hat ** 2)
        out[k] = f_min + span * e_g_k
    return out


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


def dqa_check_sin_squared_identity(dicke_angles: list, c_y: list, c_r: float,
                                    cost_norm: float, w_d: int, thetas: list,
                                    n_steps: int, n_y: int, oracle_cy: list,
                                    oracle_cr: float, oracle_cy_hi: list,
                                    oracle_cy_lo: list, k: int,
                                    schedule: Iterable[int]) -> float:
    """Same sin^2 identity check as check_sin_squared_identity using the multiplexed kernel.

    All k=0..N_USED_IDX-1 give non-trivial results with the multiplexed oracle.
    """
    a0 = math.sqrt(max(0.0, min(1.0, dqa_p_good_exact_multi(
        dicke_angles, c_y, c_r, cost_norm, w_d, thetas, n_steps, n_y,
        oracle_cy, oracle_cr, oracle_cy_hi, oracle_cy_lo, k, 0))))
    theta0 = math.asin(a0)
    err = 0.0
    for m_power in schedule:
        pred = math.sin((2 * m_power + 1) * theta0) ** 2
        meas = dqa_p_good_exact_multi(dicke_angles, c_y, c_r, cost_norm, w_d,
                                       thetas, n_steps, n_y, oracle_cy, oracle_cr,
                                       oracle_cy_hi, oracle_cy_lo, k, m_power)
        err = max(err, abs(pred - meas))
    return err


#  Main

def _parse_schedule(text: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("schedule cannot be empty")
    if min(vals) < 0:
        raise ValueError("schedule values must be non-negative")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="UC-consistent multiplexed MLQAE (CUDA-Q)")
    parser.add_argument("--target", type=str, default="qpp-cpu", help="CUDA-Q target (qpp-cpu or nvidia)")
    parser.add_argument("--schedule", type=str, default="0,1,2,4,8,16", help="Comma-separated Grover powers")
    parser.add_argument("--shots", type=int, default=4000, help="Shots per Grover power for MLQAE likelihood data")
    parser.add_argument("--trials", type=int, default=1, help="Number of MLQAE repeated trials")
    args = parser.parse_args()

    schedule = _parse_schedule(args.schedule)

    try:
        cudaq.set_target(args.target)
    except Exception as exc:
        print(f"WARNING: failed to set target '{args.target}' ({exc}); using qpp-cpu.")
        cudaq.set_target("qpp-cpu")

    t0_total = _time.perf_counter()

    t0 = _time.perf_counter()
    f_used, f_full, f_min, f_max = build_per_scenario_table()
    oracle_thetas = build_oracle_angles(f_full, f_min, f_max)
    phi_classical, grad_classical = classical_truth()
    truth = np.concatenate([[phi_classical], grad_classical])
    print(f"[setup] classical table + truth: {_time.perf_counter() - t0:.3f}s")

    print("\nUC-consistent multiplexed MLQAE (CUDA-Q)")
    print(f"target={cudaq.get_target().name} | schedule={schedule} | shots={args.shots}")
    print(f"n_y={N_Y}, d={D}, x0={X0}, w_d={W_D}, c_r={C_R}")
    print(f"c_y_eff={[round(v, 4) for v in C_Y_EFF]}")

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
    for trial in range(args.trials):
        est = quantum_expectations_mlqae(
            oracle_thetas=oracle_thetas,
            f_min=f_min,
            f_max=f_max,
            schedule=schedule,
            shots=args.shots,
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

    # --- DQA parameters (n_steps=0: Dicke-state prep only, no variational layers) ---
    t0 = _time.perf_counter()
    dicke_angles = list(dicke_state_angles(N_Y, W_D))
    cost_norm_dqa = float(max(C_Y_EFF + [C_R]))
    oracle_cy_dqa = [2.0 * math.asin(math.sqrt(min(c / cost_norm_dqa, 1.0))) for c in C_Y_EFF]
    oracle_cr_dqa = 2.0 * math.asin(math.sqrt(min(C_R / cost_norm_dqa, 1.0)))
    # Gradient angles: quadratic cost means grad depends on y[j] when xi[j]=1
    #   (xi=1, y=1): grad = c0 + 2*c1  -> oracle_cy_hi[j]
    #   (xi=1, y=0): grad = c0          -> oracle_cy_lo[j]
    oracle_cy_hi_dqa = [2.0 * math.asin(math.sqrt(min((C_Y[j][0] + 2.0 * C_Y[j][1]) / cost_norm_dqa, 1.0)))
                        for j in range(N_Y)]
    oracle_cy_lo_dqa = [2.0 * math.asin(math.sqrt(min(C_Y[j][0] / cost_norm_dqa, 1.0)))
                        for j in range(N_Y)]
    n_steps_dqa = N_T
    thetas_dqa: list[float] = []
    for t in range(n_steps_dqa):
        thetas_dqa.append(float(t / n_steps_dqa))
        thetas_dqa.append((1 - float(t / n_steps_dqa)) / math.pi)
    # Multiplexed oracle: H on idx dilutes amplitude by 1/sqrt(N_IDX).
    # Decode: out[k] = f_min + span * N_IDX * a_hat^2  with span = cost_norm
    #   = cost_norm * N_IDX * (E[f_k/cost_norm] / N_IDX) = E[f_k]  ✓
    f_min_dqa = 0.0
    f_max_dqa = cost_norm_dqa  # span = cost_norm (not cost_norm/N_IDX)
    print(f"\n[DQA setup] {_time.perf_counter() - t0:.3f}s")

    print("\n" + "=" * 60)
    print("DQA pipeline (Dicke-state prep, n_steps=0)")
    print(f"cost_norm={cost_norm_dqa:.4f} | oracle_cr={oracle_cr_dqa:.4f}")
    print(f"oracle_cy={[round(v, 4) for v in oracle_cy_dqa]}")

    if RUN_GROVER_IDENTITY_CHECKS:
        t0 = _time.perf_counter()
        print("\nStep A (DQA): Grover-power identity checks (multiplexed oracle, all k)")
        for k in range(N_USED_IDX):
            t_k = _time.perf_counter()
            err = dqa_check_sin_squared_identity(
                dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
                thetas_dqa, n_steps_dqa, N_Y, oracle_cy_dqa, oracle_cr_dqa,
                oracle_cy_hi_dqa, oracle_cy_lo_dqa, k, schedule)
            label = "phi" if k == 0 else f"grad[{k-1}]"
            print(f"  k={k} ({label}): max |P_m - sin^2((2m+1)theta)| = {err:.3e}  ({_time.perf_counter()-t_k:.2f}s)")
        print(f"[Step A (DQA)] {_time.perf_counter() - t0:.3f}s")

    t0 = _time.perf_counter()
    print("\nStep B (DQA): MLQAE recovery vs classical truth (all components)")
    print(f"  classical [phi, grad_0..grad_{N_Y-1}] = {np.round(truth, 6)}")
    max_abs_dqa = 0.0
    for trial in range(args.trials):
        t_trial = _time.perf_counter()
        est_dqa = quantum_expectations_dqa_mlqae(
            dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
            thetas_dqa, n_steps_dqa, N_Y, oracle_cy_dqa, oracle_cr_dqa,
            oracle_cy_hi_dqa, oracle_cy_lo_dqa,
            f_min_dqa, f_max_dqa,
            schedule=schedule,
            shots=args.shots,
            seed_base=100 * trial,
        )
        abs_err_dqa = np.abs(est_dqa - truth)
        max_abs_dqa = max(max_abs_dqa, float(abs_err_dqa.max()))
        print(f"\ntrial {trial} ({_time.perf_counter()-t_trial:.2f}s):")
        print(f"  DQA-MLQAE [phi, grad_0..grad_{N_Y-1}] = {np.round(est_dqa, 6)}")
        print(f"  abs error per component               = {np.round(abs_err_dqa, 6)}")
    print(f"\nMAX absolute error across trials/components: {max_abs_dqa:.3e}")
    print(f"[Step B (DQA)] {_time.perf_counter() - t0:.3f}s")

    # --- Exact oracle for phi (correct for any W_D, no angle accumulation) ---
    t0 = _time.perf_counter()
    exact_norm = float(W_D * C_R)   # upper bound: W_D turbines all on recourse
    exact_thetas = build_exact_oracle_thetas(exact_norm)
    print(f"\n[exact oracle build] {_time.perf_counter()-t0:.3f}s")
    print(f"exact_norm={exact_norm:.2f}  (W_D={W_D} * C_R={C_R})")

    t0 = _time.perf_counter()
    print("\nStep B (exact phi, any W_D): E_DQA[Q(y,xi)] vs classical phi")
    print(f"  classical phi = {truth[0]:.6f}")
    for trial in range(args.trials):
        t_trial = _time.perf_counter()
        phi_exact = dqa_mlqae_estimate_exact_phi(
            dicke_angles, C_Y_EFF, C_R, cost_norm_dqa, W_D,
            thetas_dqa, n_steps_dqa, N_Y, exact_thetas, exact_norm,
            schedule=schedule, shots=args.shots, seed=100 * trial,
        )
        abs_err = abs(phi_exact - truth[0])
        print(f"\ntrial {trial} ({_time.perf_counter()-t_trial:.2f}s):")
        print(f"  exact-oracle phi estimate = {phi_exact:.6f}")
        print(f"  abs error (phi)           = {abs_err:.6f}")
    print(f"[Step B (exact phi)] {_time.perf_counter() - t0:.3f}s")

    print(f"\n[TOTAL] {_time.perf_counter() - t0_total:.3f}s")


if __name__ == "__main__":
    main()
