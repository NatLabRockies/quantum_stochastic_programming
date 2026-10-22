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



# 1) UC parameters 

RNG = np.random.default_rng(0)

N_X = 1
C_X = [4.0]
X0 = [6]

N_Y = 4
D = 8
C_Y = [[c0, c1] for c0, c1 in zip(np.linspace(0.1, 1.0, N_Y), 1e-3 * np.ones(N_Y))]
C_R = 10.0

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
N_SCEN_Q = N_Y         # 2^N_SCEN_Q = number of scenarios for uniform xi
N_IDX_Q = 3            # 8 indices, enough for 1 + N_Y = 5 quantities
N_IDX = 1 << N_IDX_Q
N_USED_IDX = 1 + N_Y
N_SYS = N_SCEN_Q + N_IDX_Q + 1  # + ancilla



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
        options={"xatol": 1e-10},
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
    parser.add_argument("--trials", type=int, default=3, help="Number of MLQAE repeated trials")
    args = parser.parse_args()

    schedule = _parse_schedule(args.schedule)

    try:
        cudaq.set_target(args.target)
    except Exception as exc:
        print(f"WARNING: failed to set target '{args.target}' ({exc}); using qpp-cpu.")
        cudaq.set_target("qpp-cpu")

    f_used, f_full, f_min, f_max = build_per_scenario_table()
    oracle_thetas = build_oracle_angles(f_full, f_min, f_max)

    phi_classical, grad_classical = classical_truth()
    truth = np.concatenate([[phi_classical], grad_classical])

    print("UC-consistent multiplexed MLQAE (CUDA-Q)")
    print(f"target={cudaq.get_target().name} | schedule={schedule} | shots={args.shots}")
    print(f"n_y={N_Y}, d={D}, x0={X0}, w_d={W_D}, c_r={C_R}")
    print(f"c_y_eff={[round(v, 4) for v in C_Y_EFF]}")

    print("\nStep A: exact Grover-power identity checks")
    for k in range(N_USED_IDX):
        err = check_sin_squared_identity(oracle_thetas, k, schedule)
        label = "phi" if k == 0 else f"grad[{k-1}]"
        print(f"  k={k} ({label}): max |P_m - sin^2((2m+1)theta)| = {err:.3e}")

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


if __name__ == "__main__":
    main()
