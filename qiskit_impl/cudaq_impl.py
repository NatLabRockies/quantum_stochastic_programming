"""
CUDA-Q port of the two-stage stochastic optimization QAE algorithm.

Mirrors the Qiskit reference in:
  - ExpValFun_functions.py     (circuit primitives)
  - qae.py                     (QAE_Optimizer)
  - binary_optimizer.py        (execute_* methods)

CUDA-Q kernel restrictions (enforced at compile time):
  - Only a restricted Python subset is valid inside @cudaq.kernel.
  - No NumPy/SciPy inside kernels — compute angles in Python FIRST, pass as
    float parameters.
  - Kernel-to-kernel calls are allowed; the callee must also be @cudaq.kernel.
  - For parameterized angle lists use a list[float] argument with fixed length.

GPU target (H100/sm_90):
  cudaq.set_target('nvidia')              # single-GPU statevector (up to ~30 qubits)
  cudaq.set_target('nvidia', option='mgpu')  # multi-GPU >30 qubits (needs MPI)
  cudaq.set_target('nvidia', option='mqpu')  # parallel circuit execution per GPU
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np

try:
    import cudaq
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    print("WARNING: cudaq not found.  Install via `pip install cudaq` or "
          "`module load qiskit/aer-gpu` on Kestrel.")


# =============================================================================
# CUDA-Q kernel primitives
# =============================================================================

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


# Cleaner CCRy decomposition used in Dicke state preparation
# Qiskit counterpart: RYGate().control(2) applied via qc.append()
#   in both dicke_state_circuit() and single_oracle_sin_inconstraint().
#   CUDA-Q has no native CCRy gate; use cudaq.control(kernel, [controls], args)
#   which compiles to a multiply-controlled version of the sub-kernel.
@cudaq.kernel
def _ry_gate(theta: float, q: cudaq.qubit):
    """Single-qubit RY helper — used by ccry via cudaq.control."""
    ry(theta, q)


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


# -----------------------------------------------------------------------------
# Dicke state |D_n^k⟩ — uniform superposition over n-bit strings with k ones
# Qiskit counterpart: ExpValFun_functions.dicke_state_circuit(args)
#   and BinaryNestedOptimizer.dicke_state_circuit(weight) in binary_optimizer.py
#   Both build the same SCS sequence and return a gate via .to_gate().
# -----------------------------------------------------------------------------
@cudaq.kernel
def dicke_state_n4_k2(y: cudaq.qview):
    """Dicke state |D_4^2⟩ on 4 qubits (wind_demand=2, n_y=4).

    Qiskit counterpart: ExpValFun_functions.dicke_state_circuit(args)
    called with args['n_y']=4, args['w_d']=2, and
    BinaryNestedOptimizer.dicke_state_circuit(weight=2) in binary_optimizer.py.
    The SCS angle patterns are identical; CUDA-Q inlines them as floats
    because kernel code cannot call math outside the compiled subset.

    Hardcoded for n=4, k=2 matching the example notebook.
    For general (n, k), compute SCS angles in Python and pass as list[float].
    Angles:
      SCS(4,2): theta1=2*arccos(sqrt(1/4)), theta2=2*arccos(sqrt(2/4))
      SCS(3,2): theta1=2*arccos(sqrt(1/3)), theta2=2*arccos(sqrt(2/3))
      SCS(2,1): theta=2*arccos(sqrt(1/2))
    """
    # Initialize: flip the k=2 rightmost bits
    x(y[3])
    x(y[2])

    # n-k=2 applications of SCS going left
    # SCS(4, 2) on qubits [y[1], y[2], y[3]]  (local: 0->y[1], 1->y[2], 2->y[3])
    # Precomputed: 2*acos(sqrt(1/4)) ≈ 2.094, 2*acos(sqrt(2/4)) = pi/2
    # math.acos/math.sqrt are function calls not supported inside @cudaq.kernel;
    # angles are constants so we inline their numeric values here.
    cx(y[2], y[3])                              # cx(local1, local2)
    cry(2.0943951023931953, y[3], y[2])         # cry(t1_42, local2, local1) -- ctrl=y[3]
    cx(y[2], y[3])                              # cx(local1, local2)
    cx(y[1], y[3])                              # cx(local0, local2)
    ccry(1.5707963267948966, y[3], y[2], y[1])  # ccry(pi/2, local2, local1, local0)
    cx(y[1], y[3])                              # cx(local0, local2)

    # SCS(3, 2) on qubits [0,1,2]
    # Precomputed: 2*acos(sqrt(1/3)), 2*acos(sqrt(2/3))
    cx(y[1], y[2])
    cry(1.9106332362490186, y[2], y[1])   # 2*acos(sqrt(1/3))
    cx(y[1], y[2])
    cx(y[0], y[2])
    ccry(1.2309594173407747, y[2], y[1], y[0])  # 2*acos(sqrt(2/3))
    cx(y[0], y[2])

    # Last k-1=1 application of SCS
    # SCS(2,1) on qubits [0,1]
    # Precomputed: 2*acos(sqrt(1/2)) = pi/2
    cx(y[0], y[1])
    cry(1.5707963267948966, y[1], y[0])   # 2*acos(sqrt(1/2)) = pi/2
    cx(y[0], y[1])


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


@cudaq.kernel
def _mcry(theta: float, ctrls: cudaq.qview, target: cudaq.qubit):
    """Multi-controlled RY(theta) on target, gated by all qubits in ctrls.

    Generalises cry (1 control) and ccry (2 controls) to arbitrary fan-in.
    Qiskit counterpart: RYGate(theta).control(len(ctrls)) used inside
    dicke_state_circuit() for the j-th SCS layer.
    """
    cudaq.control(_ry_gate, ctrls, theta, target)


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


# # -----------------------------------------------------------------------------
# # Cost operator U_q(gamma) — CP gates encoding second-stage costs as phases
# # Phase kick: e^{i*gamma*c_y/norm} when y[j]=1 AND xi[j]=1 (wind available)
# #             e^{i*gamma*c_r/norm} when y[j]=1 AND xi[j]=0 (recourse)
# # In CUDA-Q: CP(theta) = cr1(theta) (controlled-Phase); equivalent to Qiskit's
# #   qc.cp(theta, ctrl, tgt)
# # Qiskit counterpart: ExpValFun_functions.cost_operator(amplitude, args)
# #   and BinaryNestedOptimizer.cost_operator(amplitude, constraint_amplitude, norm)
# # -----------------------------------------------------------------------------
# @cudaq.kernel
# def cost_operator_n4(gamma: float,
#                      c_y0: float, c_y1: float, c_y2: float, c_y3: float,
#                      c_r: float, cost_norm: float,
#                      y: cudaq.qview, xi: cudaq.qview):
#     """Cost phase operator for n_y=4 turbines.

#     Qiskit counterpart: ExpValFun_functions.cost_operator(amplitude, args)
#     in ExpValFun_functions.py.
#     Qiskit uses qc.cp(amplitude*cost/cost_norm, q_pdf, q_w) for the
#     operational cost and X+CP+X for the recourse cost.
#     CUDA-Q replaces qc.cp(...) with cr1(...) (controlled-Phase gate).
#     """
#     # List construction inside @cudaq.kernel is not supported;
#     # unroll all four turbines explicitly.
#     scale = gamma / cost_norm
#     cr1(c_y0 * scale, xi[0], y[0])
#     x(xi[0])
#     cr1(c_r * scale, xi[0], y[0])
#     x(xi[0])
#     cr1(c_y1 * scale, xi[1], y[1])
#     x(xi[1])
#     cr1(c_r * scale, xi[1], y[1])
#     x(xi[1])
#     cr1(c_y2 * scale, xi[2], y[2])
#     x(xi[2])
#     cr1(c_r * scale, xi[2], y[2])
#     x(xi[2])
#     cr1(c_y3 * scale, xi[3], y[3])
#     x(xi[3])
#     cr1(c_r * scale, xi[3], y[3])
#     x(xi[3])

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


# -----------------------------------------------------------------------------
# XY (demand-constraint-preserving) mixer U_d(beta)
# SWAP^beta = exp(-i beta pi/2 (XX+YY)/2)
# Decomposition: Rxx(beta*pi/2) + Ryy(beta*pi/2) via native CUDA-Q gates
# Qiskit counterpart: ExpValFun_functions.demand_constraint_preserving_mixer()
#   Qiskit uses qc.append(SwapGate().power(amplitude/layers), [qj, qk])
#   CUDA-Q has no SwapGate().power(); decomposed here as Rxx+Ryy.
# -----------------------------------------------------------------------------
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


# # Qiskit counterpart: the outer loop in demand_constraint_preserving_mixer()
# #   for qj in args['y_reg']: for qk in args['y_reg'][qj+1:]: SwapGate().power(...)
# @cudaq.kernel
# def mixer_n4(beta: float, y: cudaq.qview):
#     """XY mixer for n_y=4: applies partial SWAP to all pairs (j, k).

#     Qiskit counterpart: ExpValFun_functions.demand_constraint_preserving_mixer(
#         amplitude, args) — the double loop over y_reg pairs.
#     """
#     for j in range(4):
#         for k in range(j + 1, 4):
#             fswap_power(beta, y[j], y[k])


@cudaq.kernel
def mixer(beta: float, y: cudaq.qview):
    for j in range(len(y)):
        for k in range(j + 1, len(y)):
            fswap_power(beta, y[j], y[k])
        # end for k
    # end for j
    return


# # -----------------------------------------------------------------------------
# # Oracle F_sin — encodes normalized cost as ancilla rotation amplitude
# # CCRY(theta) rotates ancilla when y[j]=1 AND xi[j]=1 (or xi[j]=0 flipped)
# # Qiskit counterpart: ExpValFun_functions.single_oracle_sin_inconstraint(args)
# #   Qiskit: qc.append(RYGate(theta_cy).control(2), [q, pdf_reg[q], ancilla])
# #           qc.x(pdf_reg[q]); same with theta_cr; qc.x(pdf_reg[q])
# #   CUDA-Q replaces RYGate(...).control(2) with our ccry() kernel.
# # -----------------------------------------------------------------------------
# @cudaq.kernel
# def oracle_sin_n4(c_y0: float, c_y1: float, c_y2: float, c_y3: float,
#                   c_r: float, norm: float,
#                   y: cudaq.qview, xi: cudaq.qview, ancilla: cudaq.qubit):
#     """F_sin oracle for n_y=4 turbines.

#     Qiskit counterpart: ExpValFun_functions.single_oracle_sin_inconstraint(args)
#     in ExpValFun_functions.py.
#     Qiskit appends RYGate(theta).control(2) (i.e., CCRY) for each turbine;
#     CUDA-Q uses the ccry() kernel defined above as a drop-in replacement.
#     Gate angles pi*c_y/norm and pi*c_r/norm are identical in both versions.

#     Rotates ancilla qubit by angles proportional to per-turbine costs so that:
#       Pr[ancilla=|1>] ≈ normalized_expected_cost
#     """
#     # List construction inside @cudaq.kernel is not supported;
#     # unroll all four turbines explicitly.
#     # scale = pi/norm is a float-float division — valid in kernel scope.
#     scale = math.pi / norm
#     # turbine 0
#     ccry(c_y0 * scale, y[0], xi[0], ancilla)
#     x(xi[0])
#     ccry(c_r * scale, y[0], xi[0], ancilla)
#     x(xi[0])
#     # turbine 1
#     ccry(c_y1 * scale, y[1], xi[1], ancilla)
#     x(xi[1])
#     ccry(c_r * scale, y[1], xi[1], ancilla)
#     x(xi[1])
#     # turbine 2
#     ccry(c_y2 * scale, y[2], xi[2], ancilla)
#     x(xi[2])
#     ccry(c_r * scale, y[2], xi[2], ancilla)
#     x(xi[2])
#     # turbine 3
#     ccry(c_y3 * scale, y[3], xi[3], ancilla)
#     x(xi[3])
#     ccry(c_r * scale, y[3], xi[3], ancilla)
#     x(xi[3])


@cudaq.kernel
def oracle_sin(c_y: list[float], c_r: float, norm: float,
               y: cudaq.qview, xi: cudaq.qview, ancilla: cudaq.qubit):
    scale = math.pi / norm
    for k in range(len(y)):
        ccry(c_y[k] * scale, y[k], xi[k], ancilla)
        x(xi[k])
        ccry(c_r * scale, y[k], xi[k], ancilla)
        x(xi[k])
    ## end for
    return


# =============================================================================
# QAE kernels — Quantum Amplitude Estimation
# Qiskit counterpart: BinaryNestedOptimizer.implemented_qae() in binary_optimizer.py
# =============================================================================

@cudaq.kernel
def _a_op(dicke_angles: list[float],
          c_y: list[float], c_r: float, cost_norm: float, w_d: int,
          thetas: list[float], n_steps: int, n_y: int,
          qubits: cudaq.qview, ancilla: cudaq.qubit):
    """A operator: U_opt followed by F_sin oracle on ancilla.

    Prepares:  A|0⟩ = sqrt(1-a)|ψ_bad⟩|0⟩ + sqrt(a)|ψ_good⟩|1⟩
    where a ≈ E[Q]/norm (normalized expected cost).

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

    # F_sin oracle: encodes E[Q] as Pr[ancilla=|1⟩]
    oracle_sin(c_y, c_r, cost_norm, y, xi, ancilla)


@cudaq.kernel
def _s_chi(ancilla: cudaq.qubit):
    """S_χ: phase flip on |ψ_good⟩ — reflects about ancilla=|0⟩ subspace.

    Applies -1 phase to all states where ancilla=|1⟩:
        S_χ = I - 2|ψ_good⟩⟨ψ_good| ≡ X·Z·X on ancilla (up to global phase).

    Qiskit counterpart: the 'S_psi0' block in implemented_qae():
        qc.x(ancilla); qc.cz(..., ancilla); qc.x(ancilla)  (phase on |1⟩)
    """
    z(ancilla)


@cudaq.kernel
def _s0(n_y: int, qubits: cudaq.qview, ancilla: cudaq.qubit):
    """S_0: phase flip of |00...0⟩ on system+ancilla register.

    Implements I - 2|0⟩⟨0| via X-flip all, multi-controlled Z, X-flip back.

    Qiskit counterpart: the 'S_0' block in implemented_qae():
        for q in all_qubits: qc.x(q)
        qc.h(last); qc.mcx(controls, last); qc.h(last)   <- multi-controlled Z
        for q in all_qubits: qc.x(q)
    """
    # Flip all qubits so |0...0⟩ maps to |1...1⟩
    for i in range(2 * n_y):
        x(qubits[i])
    x(ancilla)
    # Multi-controlled Z: fire only when all are |1⟩ (= original |0...0⟩ state)
    # Decompose as H · MCX · H on ancilla with qubits as controls
    h(ancilla)
    cudaq.control(_mcx_helper, qubits, ancilla)
    h(ancilla)
    # Unflip
    for i in range(2 * n_y):
        x(qubits[i])
    x(ancilla)


@cudaq.kernel
def _mcx_helper(target: cudaq.qubit):
    """Single-qubit X — used as the target of cudaq.control for MCX in S_0."""
    x(target)


@cudaq.kernel
def _grover_iterate(dicke_angles: list[float],
                    c_y: list[float], c_r: float, cost_norm: float, w_d: int,
                    thetas: list[float], n_steps: int, n_y: int,
                    qubits: cudaq.qview, ancilla: cudaq.qubit):
    """One Grover iterate: Q = A · S_0 · A† · S_χ

    Qiskit counterpart: the inner block of the QPE loop in implemented_qae():
        S_psi0 (flip ancilla=|1⟩) -> A†  -> S_0 (flip |0...0⟩) -> A

    Note: CUDA-Q cudaq.adjoint() compiles the exact unitary inverse of a kernel.
    """
    # S_χ: phase flip on |good⟩ (ancilla = |1⟩)
    _s_chi(ancilla)
    # A†: inverse of A operator
    cudaq.adjoint(_a_op, dicke_angles, c_y, c_r, cost_norm, w_d,
                  thetas, n_steps, n_y, qubits, ancilla)
    # S_0: phase flip of |0...0⟩
    _s0(n_y, qubits, ancilla)
    # A: reapply A operator
    _a_op(dicke_angles, c_y, c_r, cost_norm, w_d,
          thetas, n_steps, n_y, qubits, ancilla)


# # -----------------------------------------------------------------------------
# # Full DQA ansatz kernel (alternating_operator_ansatz in CUDA-Q)
# # Theta list: [gamma_0, beta_0, gamma_1, beta_1, ...]
# # For 4 timesteps: 8 angles total
# # Qiskit counterpart: ExpValFun_functions.alternating_operator_ansatz(args)
# #   Qiskit builds a QuantumCircuit, appends initial_state_circuit and
# #   pdf_circuit gates, then loops over args['Theta'] alternating between
# #   cost_operator_circuit (even i) and mixer_operator_circuit (odd i).
# #   This kernel does exactly the same sequence using the CUDA-Q primitives above.
# # -----------------------------------------------------------------------------
# @cudaq.kernel
# def dqa_ansatz_n4(
#         c_y0: float, c_y1: float, c_y2: float, c_y3: float,
#         c_r: float, cost_norm: float,
#         thetas: list[float],   # [gamma_0, beta_0, gamma_1, beta_1, ...]
#         n_steps: int):
#     """DQA alternating operator ansatz for n_y=4, n_xi=4.

#     Qiskit counterpart: ExpValFun_functions.alternating_operator_ansatz(args)
#     in ExpValFun_functions.py.
#     Qiskit structure:
#       qc.append(initial_state_circuit(args).to_gate(), y_reg)   <- dicke_state_n4_k2
#       qc.append(pdf_circuit(args).to_gate(),          pdf_reg)  <- pdf_init_uniform
#       for i, theta in enumerate(Theta):
#           if i % 2 == 0: cost_operator_circuit(theta, args)     <- cost_operator_n4
#           else:          mixer_operator_circuit(theta, args)     <- mixer_n4

#     Total qubits: n_y + n_xi = 8.
#     Layout: y[0..3], xi[0..3].
#     """
#     qubits = cudaq.qvector(8)
#     y  = qubits[0:4]
#     xi = qubits[4:8]

#     # Initial state
#     dicke_state_n4_k2(y)
#     pdf_init_uniform(xi)

#     # Alternating layers
#     for i in range(n_steps * 2):
#         if i % 2 == 0:
#             cost_operator_n4(thetas[i], c_y0, c_y1, c_y2, c_y3, c_r, cost_norm,
#                              y, xi)
#         else:
#             mixer_n4(thetas[i], y)


@cudaq.kernel
def dqa_ansatz(dicke_angles: list[float],
               c_y: list[float], c_r: float, cost_norm: float, w_d: int,
               thetas: list[float],
               n_steps: int, n_y: int):
    qubits = cudaq.qvector(2*n_y)
    y = qubits[0:n_y]
    xi = qubits[n_y:2*n_y]

    # dicke_state_n4_k2(y)
    dicke_state(n_y, w_d, dicke_angles, y)

    pdf_init_uniform(xi)
    for i in range(n_steps * 2):
        if i % 2 == 0:
            cost_operator(thetas[i], c_y, c_r, cost_norm, y, xi)
        else:
            mixer(thetas[i], y)
        ## end if
    ## end for
    return


# =============================================================================
# Python-level runner (mirrors BinaryNestedOptimizer.execute_optimizer)
# =============================================================================

class CudaqQAEOptimizer:
    """GPU-accelerated QAE optimizer using CUDA-Q.

    Qiskit counterpart: BinaryNestedOptimizer in binary_optimizer.py.
    Method mapping:
      __init__               <- BinaryNestedOptimizer.__init__
      sample_ansatz          <- execute_optimizer(qc, num_meas=N)
      estimate_expected_value<- execute_optimizer() + process_expectation_value_optimizer()
      _wind_scenario_cost    <- wind_scenario_cost() in binary_optimizer.py
      benchmark_vs_qiskit    <- (new) wraps both and times them

    Uses @cudaq.kernel functions instead of Qiskit QuantumCircuits.

    Usage::

        cudaq.set_target('nvidia')   # or 'qpp-cpu' for CPU fallback
        opt = CudaqQAEOptimizer(c_x=[3.], c_y=[0.4, 0.5, 0.7, 1.],
                                c_r=10., n_y=4, w_d=2, cost_norm=5.)
        phi_est = opt.estimate_expected_value(thetas, shots=1000)
    """

    def __init__(self, c_x: list, c_y: list, c_r: float,
                 n_y: int = 4, n_xi: int = 4, w_d: int = 2, cost_norm: float = 5.0):
        if not _CUDAQ_AVAILABLE:
            raise RuntimeError("cudaq not installed.")
        # if n_y != 4:
        #     raise NotImplementedError("Generic n_y requires parameterized kernels; "
        #                               "currently only n_y=4 is hardcoded.")
        self.c_y = list(c_y)
        self.c_r = float(c_r)
        self.n_y = n_y
        self.n_xi = n_xi
        self.w_d = w_d
        self.cost_norm = cost_norm
        self.dang = dicke_state_angles(n_y, w_d)
        return

    def sample_ansatz(self, thetas: list, shots: int = 4096) -> dict:
        """Sample the DQA ansatz and return bitstring probabilities.

        Args:
            thetas: Alternating [gamma, beta, gamma, beta, ...] angles.
            shots:  Number of measurement shots.

        Returns:
            Dict {bitstring: probability} over 2*n_y qubits (y+xi register).
        """
        n_steps = len(thetas) // 2
        counts = cudaq.sample(
            dqa_ansatz,
            self.dang,
            self.c_y,
            self.c_r, self.cost_norm,
            self.w_d,
            thetas, n_steps, self.n_y,
            shots_count=shots)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def get_statevector(self, thetas: list) -> np.ndarray:
        """Return the full complex amplitude vector for the DQA ansatz.

        Uses cudaq.get_state() — no measurement, no sampling noise.
        Requires a statevector-capable target: 'qpp-cpu' or 'nvidia'.

        Args:
            thetas: Alternating [gamma, beta, ...] DQA angles.

        Returns:
            Complex numpy array of length 2**(2*n_y).
        """
        n_steps = len(thetas) // 2
        state = cudaq.get_state(
            dqa_ansatz,
            self.dang,
            self.c_y, self.c_r, self.cost_norm,
            self.w_d, thetas, n_steps, self.n_y)
        return np.array(state)

    def estimate_expected_value(self, thetas: list, wind_demand: int,
                                shots: int = 4096) -> float:
        """Estimate E[Q(x, xi)] via DQA sampling and classical post-processing.

        Mirrors BinaryNestedOptimizer.execute_optimizer +
        process_expectation_value_optimizer().

        Args:
            thetas:       DQA angles [gamma_0, beta_0, ...].
            wind_demand:  k — required Hamming weight of y (= d - x).
            shots:        Number of measurement shots.

        Returns:
            Float — estimated expected second-stage cost.
        """
        probs = self.sample_ansatz(thetas, shots=shots)
        expectation = 0.0
        for bstr, prob in probs.items():
            # CUDA-Q bitstring convention: qubit 0 = leftmost character.
            # Kernel layout: y[0..n_y-1] = chars 0..n_y-1,
            #                xi[0..n_y-1] = chars n_y..2*n_y-1.
            # No reversal needed — c_y[j] correctly pairs with y[j] and xi[j].
            y_bits  = [int(b) for b in bstr[:self.n_y]]
            xi_bits = [int(b) for b in bstr[self.n_y:]]
            cost = self._wind_scenario_cost(y_bits, xi_bits, wind_demand)
            expectation += cost * prob
        return expectation

    def estimate_expected_value_sv(self, thetas: list, wind_demand: int) -> float:
        """Exact expected second-stage cost via statevector simulation (no shots).

        Calls get_statevector() to obtain the full amplitude vector, then
        computes E[Q] = sum_i |alpha_i|^2 * Q(y_i, xi_i) over all 2**(2*n_y)
        basis states.  Exact up to floating-point precision; no sampling noise.

        Requires a statevector-capable target: 'qpp-cpu' or 'nvidia'.

        Args:
            thetas:      DQA angles [gamma_0, beta_0, ...].
            wind_demand: k — required Hamming weight of y (= d - x).

        Returns:
            Exact expected second-stage cost.
        """
        sv = self.get_statevector(thetas)
        n_total = 2 * self.n_y
        expectation = 0.0
        for i in range(1 << n_total):
            prob = abs(sv[i]) ** 2
            if prob < 1e-14:
                continue
            # State-vector index i: qubit j = bit j of i (qubit 0 = LSB).
            # Reversing the binary string places qubit j at char position j,
            # matching the CUDA-Q sample() convention (qubit 0 = leftmost).
            bstr = format(i, f'0{n_total}b')[::-1]
            y_bits  = [int(b) for b in bstr[:self.n_y]]
            xi_bits = [int(b) for b in bstr[self.n_y:]]
            cost = self._wind_scenario_cost(y_bits, xi_bits, wind_demand)
            expectation += cost * prob
        return expectation

    def build_cost_hamiltonian(self) -> 'cudaq.SpinOperator':
        """Build the cost function as a CUDA-Q SpinOperator in Pauli Z form.

        Expresses Q(y, xi) = sum_j [c_y[j]*n_{y_j}*n_{xi_j} + c_r*n_{y_j}*(1-n_{xi_j})]
        in terms of Pauli Z operators via the number-operator identity n_j = (I - Z_j)/2:

          n_{y_j} * n_{xi_j}       = (1/4)(I - Z_{y_j} - Z_{xi_j} + Z_{y_j}*Z_{xi_j})
          n_{y_j} * (1-n_{xi_j})   = (1/4)(I - Z_{y_j} + Z_{xi_j} - Z_{y_j}*Z_{xi_j})

        So per turbine j:
          coeff_I      = (c_y[j] + c_r) / 4
          coeff_Zy     = -(c_y[j] + c_r) / 4
          coeff_Zxi    = (c_r - c_y[j]) / 4
          coeff_ZyZxi  = (c_y[j] - c_r) / 4

        Qubit layout matches the kernel: y[0..n_y-1] at indices 0..n_y-1,
        xi[0..n_y-1] at indices n_y..2*n_y-1.

        Returns:
            cudaq.SpinOperator representing the full cost Hamiltonian.
        """
        h = None
        for j in range(self.n_y):
            y_idx  = j
            xi_idx = self.n_y + j

            coeff_I     = (self.c_y[j] + self.c_r) / 4.0
            coeff_Zy    = -(self.c_y[j] + self.c_r) / 4.0
            coeff_Zxi   = (self.c_r - self.c_y[j]) / 4.0
            coeff_ZyZxi = (self.c_y[j] - self.c_r) / 4.0

            terms = (coeff_I     * cudaq.spin.i(y_idx)
                   + coeff_Zy    * cudaq.spin.z(y_idx)
                   + coeff_Zxi   * cudaq.spin.z(xi_idx)
                   + coeff_ZyZxi * cudaq.spin.z(y_idx) * cudaq.spin.z(xi_idx))

            h = terms if h is None else h + terms
        return h

    def estimate_expected_value_observe(self, thetas: list) -> float:
        """Exact expected second-stage cost via cudaq.observe() (no shots).

        Constructs the cost function as a Pauli Z SpinOperator via
        build_cost_hamiltonian(), then evaluates <psi|H|psi> exactly using
        cudaq.observe().  No sampling noise; exact up to floating-point precision.

        This is the most direct CUDA-Q approach: the expectation value is computed
        in a single pass through the statevector without enumerating basis states.

        Requires a statevector-capable target: 'qpp-cpu' or 'nvidia'.

        Args:
            thetas: DQA angles [gamma_0, beta_0, ...].

        Returns:
            Exact expected second-stage cost.
        """
        h = self.build_cost_hamiltonian()
        n_steps = len(thetas) // 2
        result = cudaq.observe(
            dqa_ansatz, h,
            self.dang, self.c_y, self.c_r, self.cost_norm,
            self.w_d, thetas, n_steps, self.n_y)
        return result.expectation()
    
    def execute_qae(self, thetas: list, m: int, shots: int = 8192) -> dict:
        """Run Quantum Amplitude Estimation and return QPE readout distribution.

        Implements the canonical QAE circuit (Brassard et al. 2002):
          1. Apply A = U_opt · F_sin once to prepare sqrt(a)|ψ_good⟩|1⟩ + ...
          2. For each QPE qubit i (i=0..m-1): apply controlled-Q^(2^i)
          3. Inverse QFT on QPE register
          4. Measure QPE qubits -> integer b -> a ≈ sin²(bπ/2^m)

        Qiskit counterpart: BinaryNestedOptimizer.implemented_qae() +
        execute_qae() in binary_optimizer.py.

        Total qubits: m (QPE) + 2*n_y (system) + 1 (ancilla) = m + 2*n_y + 1

        Args:
            thetas: DQA angles [gamma_0, beta_0, ...].
            m:      Number of QPE estimate qubits (error ~ π/2^m).
            shots:  Measurement shots.

        Returns:
            Dict {b_bitstring: probability} over all 2^m QPE outcomes.
        """
        n_steps = len(thetas) // 2
        n_total = m + 2 * self.n_y + 1   # QPE + system + ancilla

        @cudaq.kernel
        def qae_circuit(dicke_angles: list[float],
                        c_y: list[float], c_r: float, cost_norm: float, w_d: int,
                        thetas: list[float], n_steps: int, n_y: int, m_qpe: int):
            all_qubits = cudaq.qvector(m_qpe + 2 * n_y + 1)
            qpe_reg  = all_qubits[0:m_qpe]
            sys_reg  = all_qubits[m_qpe:m_qpe + 2 * n_y]
            anc      = all_qubits[m_qpe + 2 * n_y]

            # Step 1: H on all QPE qubits
            for i in range(m_qpe):
                h(qpe_reg[i])

            # Step 2: Apply A once (unconditionally)
            _a_op(dicke_angles, c_y, c_r, cost_norm, w_d,
                  thetas, n_steps, n_y, sys_reg, anc)

            # Step 3: Controlled-Q^(2^i) for each QPE qubit i
            # Unroll: apply _grover_iterate controlled on qpe_reg[i], 2^i times
            for i in range(m_qpe):
                reps = 1 << i   # 2^i
                for _ in range(reps):
                    cudaq.control(_grover_iterate,
                                  [qpe_reg[i]],
                                  dicke_angles, c_y, c_r, cost_norm, w_d,
                                  thetas, n_steps, n_y, sys_reg, anc)

            # Step 4: Inverse QFT on QPE register
            # CUDA-Q has no built-in IQFT; implement manually.
            # IQFT on m qubits: reverse order, then H + controlled-Rz(-π/2^k) pattern.
            # Reverse qubit order
            for i in range(m_qpe // 2):
                swap(qpe_reg[i], qpe_reg[m_qpe - 1 - i])
            # QFT† layers
            for i in range(m_qpe):
                h(qpe_reg[i])
                for j in range(i + 1, m_qpe):
                    angle = -math.pi / (1 << (j - i))
                    cr1(angle, qpe_reg[j], qpe_reg[i])

        counts = cudaq.sample(
            qae_circuit,
            self.dang, self.c_y, self.c_r, self.cost_norm,
            self.w_d, thetas, n_steps, self.n_y, m,
            shots_count=shots)

        # Marginalize: keep only the m QPE qubits (leftmost m chars in CUDA-Q convention)
        qpe_counts: dict = {}
        total = sum(counts.values())
        for bstr, cnt in counts.items():
            b = bstr[:m]   # QPE qubits are allocated first -> leftmost chars
            qpe_counts[b] = qpe_counts.get(b, 0) + cnt
        return {b: v / total for b, v in qpe_counts.items()}

    def process_qae_result(self, qpe_counts: dict, m: int) -> float:
        """Convert QPE readout distribution to estimated expected cost.

        Post-processing (mirrors execute_qae caller in binary_optimizer.py):
            b_str = argmax p(b)
            b_int = int(b_str, 2)
            a_tilde   = sin²(b_int · π / 2^m)
            phi_tilde = a_tilde · cost_norm

        Args:
            qpe_counts: Dict {b_bitstring: probability} from execute_qae().
            m:          Number of QPE qubits (must match execute_qae call).

        Returns:
            Float — estimated expected second-stage cost.
        """
        b_str = max(qpe_counts, key=qpe_counts.get)
        b_int = int(b_str, 2)
        a_tilde = math.sin(b_int * math.pi / (1 << m)) ** 2
        return a_tilde * self.cost_norm

    def estimate_expected_value_qae(self, thetas: list, m: int,
                                    shots: int = 8192) -> float:
        """End-to-end QAE expected cost estimate.

        Combines execute_qae() and process_qae_result() into a single call.
        Achieves O(1/2^m) error vs O(1/sqrt(shots)) for classical sampling,
        with quadratic quantum speedup in the number of oracle calls.

        Qiskit counterpart: execute_qae() + argmax post-processing
        in binary_optimizer.py.

        Args:
            thetas: DQA angles [gamma_0, beta_0, ...].
            m:      QPE qubits — error scales as π·norm/2^m.
            shots:  Measurement shots.

        Returns:
            Float — QAE-estimated expected second-stage cost.
        """
        qpe_counts = self.execute_qae(thetas, m=m, shots=shots)
        return self.process_qae_result(qpe_counts, m=m)

    def _wind_scenario_cost(self, y: list, xi: list, wind_demand: int) -> float:
        """Second-stage cost Q(y, xi) for one scenario.

        Qiskit counterpart: BinaryNestedOptimizer.wind_scenario_cost()
        in binary_optimizer.py.  Logic is identical.
        """
        cost = 0.0
        for j in range(self.n_y):
            if y[j] == 1:
                if xi[j] == 1:
                    cost += self.c_y[j]           # operational cost
                else:
                    cost += self.c_r              # recourse penalty
        return cost

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def benchmark_vs_qiskit(self, thetas: list, wind_demand: int,
                             shots: int = 4096,
                             qiskit_optimizer=None) -> dict:
        """Time CUDA-Q vs Qiskit execution and report results.

        Args:
            thetas:            DQA angles.
            wind_demand:       k = d - x.
            shots:             Shots per method.
            qiskit_optimizer:  A BinaryNestedOptimizer instance (optional).

        Returns:
            Dict with keys 'cudaq_time', 'qiskit_time', 'cudaq_phi', 'qiskit_phi'.
        """
        # CUDA-Q
        t0 = time.perf_counter()
        phi_cudaq = self.estimate_expected_value(thetas, wind_demand, shots=shots)
        t_cudaq = time.perf_counter() - t0

        result = {'cudaq_time': t_cudaq, 'cudaq_phi': phi_cudaq,
                  'qiskit_time': None, 'qiskit_phi': None}

        if qiskit_optimizer is not None:
            from binary_optimizer import BinaryNestedOptimizer  # local import
            # Build the Qiskit adiabatic circuit and time it
            t1 = time.perf_counter()
            # Use the existing adiabatic_evolution_circuit + execute_optimizer flow
            n_steps = len(thetas) // 2
            qc = qiskit_optimizer.adiabatic_evolution_circuit(
                wind_demand=wind_demand,
                time=n_steps,
                time_steps=n_steps,
                norm=self.cost_norm)
            counts = qiskit_optimizer.execute_optimizer(qc, num_meas=shots)
            phi_qiskit = qiskit_optimizer.process_expectation_value_optimizer(
                wind_demand, counts)
            t_qiskit = time.perf_counter() - t1
            result.update({'qiskit_time': t_qiskit, 'qiskit_phi': phi_qiskit})

        return result
