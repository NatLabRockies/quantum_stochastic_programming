"""Shared circuit builders for the binary-wind stochastic optimization model.

Local layout is y[0:n], xi[n:2*n], followed by the oracle ancilla.
Builders return circuits; compatibility APIs decide when to convert to gates.
"""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate, SwapGate


def dicke_state_circuit(n_qubits: int, weight: int) -> QuantumCircuit:
    """Prepare equal amplitudes on weight-k bitstrings using the SCS construction.

    Reference: Baertschi and Eidenbenz, https://arxiv.org/abs/1904.07358.
    """
    qc = QuantumCircuit(n_qubits, name="DickeState")
    if weight == 0:
        return qc

    def scs(n: int, k: int) -> Gate:
        block = QuantumCircuit(k + 1)
        block.cx(k - 1, k)
        block.cry(2 * np.arccos(np.sqrt(1 / n)), k, k - 1)
        block.cx(k - 1, k)
        for level in range(2, k + 1):
            block.cx(k - level, k)
            block.append(
                RYGate(2 * np.arccos(np.sqrt(level / n))).control(2),
                [k, k - level + 1, k - level],
            )
            block.cx(k - level, k)
        return block.to_gate()

    for i in range(weight):
        qc.x(n_qubits - i - 1)
    for i in range(n_qubits - weight):
        qc.append(
            scs(n_qubits - i, weight),
            list(range(n_qubits - i - weight - 1, n_qubits - i)),
        )
    for i in range(weight - 1, 0, -1):
        qc.append(scs(i + 1, i), list(range(i + 1)))
    return qc


def cost_operator(
    amplitude: float,
    wind_costs: Sequence[float],
    recourse_cost: float,
    norm: float,
    pdf_qubits: Sequence[int],
) -> QuantumCircuit:
    """Encode available-wind costs and unavailable-wind recourse as CP phases."""
    qc = QuantumCircuit(2 * len(wind_costs), name=f"$U_q({np.round(amplitude, 3)})$")
    for qubit, cost in enumerate(wind_costs):
        pdf_qubit = pdf_qubits[qubit]
        qc.cp(amplitude * cost / norm, pdf_qubit, qubit)
        qc.x(pdf_qubit)
        qc.cp(amplitude * recourse_cost / norm, pdf_qubit, qubit)
        qc.x(pdf_qubit)
    return qc


def cost_scenario_operator(
    amplitude: float,
    wind_costs: Sequence[float],
    recourse_cost: float,
    norm: float,
    scenario: Sequence[int],
) -> QuantumCircuit:
    """Encode one fixed scenario as phases on the decision register."""
    qc = QuantumCircuit(len(wind_costs))
    for qubit, cost in enumerate(wind_costs):
        value = recourse_cost if scenario[qubit] == 0 else cost
        qc.p(amplitude * value / norm, qubit)
    return qc


def demand_constraint_preserving_mixer(
    amplitude: float, y_qubits: Sequence[int]
) -> QuantumCircuit:
    """Apply fractional SWAPs in pair order, preserving decision Hamming weight."""
    qc = QuantumCircuit(len(y_qubits), name=f"$U_d({np.round(amplitude, 3)})$")
    for first in y_qubits:
        for second in y_qubits[first + 1:]:
            qc.append(SwapGate().power(amplitude), [first, second])
    return qc


def cost_oracle(
    wind_costs: Sequence[float],
    recourse_cost: float,
    norm: float,
    y_qubits: Sequence[int],
    pdf_qubits: Sequence[int],
    *,
    encoding: Literal["linear", "arcsine"],
    scale: float = 1.0,
    inverse: bool = False,
) -> QuantumCircuit:
    """Encode additive turbine contributions as controlled ancilla rotations.

    Linear angles are pi*scale*cost/norm. Arcsine angles are
    2*arcsin(sqrt(cost/norm)); these encode cost/norm exactly only when one
    selected turbine contributes per basis state. The caller chooses encoding.
    """
    if encoding not in ("linear", "arcsine"):
        raise ValueError(f"Unsupported oracle encoding: {encoding}")
    if encoding == "arcsine" and scale != 1.0:
        raise ValueError("scale is supported only for linear oracle encoding")
    n = len(wind_costs)
    qc = QuantumCircuit(2 * n + 1, name="F")
    ancilla = 2 * n
    sign = -1 if inverse else 1
    for qubit in y_qubits:
        if encoding == "arcsine":
            theta_cost = sign * 2 * np.arcsin(np.sqrt(wind_costs[qubit] / norm))
            theta_recourse = sign * 2 * np.arcsin(np.sqrt(recourse_cost / norm))
        else:
            factor = sign * np.pi * scale / norm
            theta_cost = wind_costs[qubit] * factor
            theta_recourse = recourse_cost * factor
        qc.append(
            RYGate(theta_cost).control(2), [qubit, pdf_qubits[qubit], ancilla]
        )
        qc.x(pdf_qubits[qubit])
        qc.append(
            RYGate(theta_recourse).control(2), [qubit, pdf_qubits[qubit], ancilla]
        )
        qc.x(pdf_qubits[qubit])
    return qc


def alternating_operator_ansatz(
    n_y: int,
    y_qubits: Sequence[int],
    pdf_qubits: Sequence[int],
    angles: Sequence[float],
    initial_state: QuantumCircuit,
    pdf_state: QuantumCircuit,
    cost_builder: Callable[[float], QuantumCircuit],
    mixer_builder: Callable[[float], QuantumCircuit],
) -> QuantumCircuit:
    """Prepare decision/scenario states, then alternate cost and mixer layers."""
    qc = QuantumCircuit(2 * n_y)
    qc.append(initial_state.to_gate(), list(y_qubits))
    qc.append(pdf_state.to_gate(), list(pdf_qubits))
    for i, theta in enumerate(angles):
        if i % 2 == 0:
            qc.append(cost_builder(theta).to_gate(), list(y_qubits) + list(pdf_qubits))
        else:
            qc.append(mixer_builder(theta).to_gate(), list(y_qubits))
    return qc
