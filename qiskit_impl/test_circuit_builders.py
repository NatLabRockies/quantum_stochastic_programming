"""Deterministic characterization of the two existing Qiskit circuit APIs."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Statevector

# qae.py currently imports its siblings as top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qiskit_impl import ExpValFun_functions as circuits
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
from qiskit_impl.dist_prep import pdf_initialize
from qiskit_impl.qae import QAE_Optimizer


ATOL = 1e-10


def make_optimizer(n, *, pdf=None, uniform=True):
    if pdf is None:
        pdf = {
            tuple((index >> qubit) & 1 for qubit in range(n)): 1 / 2**n
            for index in range(2**n)
        }
    return BinaryNestedOptimizer(
        [0.5], [0.2 + 0.1 * qubit for qubit in range(n)],
        0.8, pdf, n, is_uniform=uniform,
    )


def circuit_args(optimizer):
    return {
        "n_y": optimizer.num_wind_vars,
        "y_reg": optimizer.wind_qubits,
        "pdf_reg": optimizer.pdf_qubits,
        "c_y": optimizer.wind_costs,
        "c_r": optimizer.recourse_cost,
        "cost_norm": 2.0,
        "norm": 2.0,
        "pdf": optimizer.pdf,
        "uniform": optimizer.is_uniform,
        "w_d": 1,
        "Theta": [0.3, 0.7, 0.8, 0.2],
        "m": 2,
        "gateset": False,
        "cost_operator_circuit": circuits.cost_operator,
        "mixer_operator_circuit": circuits.demand_constraint_preserving_mixer,
        "initial_state_circuit": circuits.dicke_state_circuit,
        "pdf_circuit": pdf_initialize,
        "oracle_circuit": circuits.single_oracle_sin_inconstraint,
    }


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize(
    ("n", "weight"),
    [(n, weight) for n in range(1, 5) for weight in range(n + 1)],
)
def test_dicke_state_matches_definition(api, n, weight):
    optimizer = make_optimizer(n)
    circuit = (
        circuits.dicke_state_circuit({"n_y": n, "w_d": weight})
        if api == "functional"
        else optimizer.dicke_state_circuit(weight)
    )
    amplitude = 1 / math.sqrt(math.comb(n, weight))
    expected = Statevector([
        amplitude if index.bit_count() == weight else 0
        for index in range(2**n)
    ])
    assert Statevector.from_instruction(circuit).equiv(expected, atol=ATOL)


@pytest.mark.parametrize(
    ("api", "distribution"),
    [
        ("functional", "uniform"),
        ("functional", "asymmetric"),
        ("functional", "sparse"),
        ("optimizer", "uniform"),
        *[
            pytest.param(
                "optimizer", distribution,
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=QiskitError,
                    reason="Nonuniform Initialize includes reset; to_gate() rejects it.",
                ),
            )
            for distribution in ("asymmetric", "sparse")
        ],
    ],
)
def test_pdf_state_and_qubit_order(api, distribution):
    # Tuple entries correspond to qubits 0, 1, unlike displayed bitstrings.
    probabilities = {
        "uniform": [0.25, 0.25, 0.25, 0.25],
        "asymmetric": [0.1, 0.2, 0.3, 0.4],
        "sparse": [0.0, 0.7, 0.0, 0.3],
    }[distribution]
    pdf = {
        (0, 0): probabilities[0],
        (1, 0): probabilities[1],
        (0, 1): probabilities[2],
        (1, 1): probabilities[3],
    }
    optimizer = make_optimizer(
        2, pdf=pdf, uniform=distribution == "uniform"
    )
    circuit = (
        pdf_initialize(circuit_args(optimizer))
        if api == "functional"
        else optimizer.pdf_initialize()
    )
    expected = Statevector(np.sqrt(probabilities))
    assert Statevector.from_instruction(circuit).equiv(expected, atol=ATOL)


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("amplitude", [0.0, 0.37, -0.6])
def test_cost_operator_phases(api, amplitude):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    circuit = (
        circuits.cost_operator(amplitude, args)
        if api == "functional"
        else optimizer.cost_operator(amplitude, args["c_r"], args["cost_norm"])
    )
    phases = []
    for index in range(16):
        cost = sum(
            ((index >> qubit) & 1)
            * (wind_cost if (index >> (qubit + 2)) & 1 else args["c_r"])
            for qubit, wind_cost in enumerate(args["c_y"])
        )
        phases.append(np.exp(1j * amplitude * cost / args["cost_norm"]))
    np.testing.assert_allclose(
        Operator(circuit).data, np.diag(phases), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("scenario", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_single_scenario_cost_phases(api, scenario):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    amplitude = 0.43
    args.update({
        "scenario": scenario,
        "wind_costs": optimizer.wind_costs,
        "constraint_amplitude": optimizer.recourse_cost,
    })
    circuit = (
        circuits.cost_scenario_operator(amplitude, args)
        if api == "functional"
        else optimizer.cost_scenario_operator(
            scenario, amplitude, args["c_r"], args["cost_norm"]
        )
    )
    phases = [
        np.exp(1j * amplitude / args["cost_norm"] * sum(
            ((index >> qubit) & 1)
            * (cost if scenario[qubit] else args["c_r"])
            for qubit, cost in enumerate(args["c_y"])
        ))
        for index in range(4)
    ]
    np.testing.assert_allclose(
        Operator(circuit).data, np.diag(phases), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("amplitude", [0.0, 0.37, 1.0, -0.4])
def test_mixer_operator_and_hamming_weight(api, amplitude):
    optimizer = make_optimizer(3)
    circuit = (
        circuits.demand_constraint_preserving_mixer(
            amplitude, circuit_args(optimizer)
        )
        if api == "functional"
        else optimizer.demand_constraint_preserving_mixer(amplitude)
    )
    # SWAP**a = P_symmetric + exp(i*pi*a) * P_antisymmetric.
    expected = np.eye(8, dtype=complex)
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        swap = np.zeros((8, 8))
        for index in range(8):
            different = ((index >> first) ^ (index >> second)) & 1
            output = index ^ ((1 << first) | (1 << second)) if different else index
            swap[output, index] = 1
        symmetric = (np.eye(8) + swap) / 2
        antisymmetric = (np.eye(8) - swap) / 2
        expected = (
            symmetric + np.exp(1j * np.pi * amplitude) * antisymmetric
        ) @ expected
    actual = Operator(circuit).data
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    for row in range(8):
        for column in range(8):
            if row.bit_count() != column.bit_count():
                assert abs(actual[row, column]) < ATOL


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("weight", [1, 2])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("scale", [0.6, 1.0])
def test_oracle_rotation_encoding(api, weight, inverse, scale):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    args["w_d"] = weight
    if api == "functional":
        circuit = circuits.single_oracle_sin_inconstraint(args, inverse=inverse)
    else:
        circuit = optimizer.single_oracle_sin_inconstraint(
            scale, args["norm"], inverse=inverse
        )

    expected = np.zeros((32, 32))
    for basis in range(16):
        angle = 0.0
        for qubit, wind_cost in enumerate(args["c_y"]):
            if (basis >> qubit) & 1:
                cost = wind_cost if (basis >> (qubit + 2)) & 1 else args["c_r"]
                if api == "functional" and weight == 1:
                    angle += 2 * np.arcsin(np.sqrt(cost / args["norm"]))
                else:
                    factor = scale if api == "optimizer" else 1.0
                    angle += np.pi * factor * cost / args["norm"]
        if inverse:
            angle = -angle
        cosine, sine = np.cos(angle / 2), np.sin(angle / 2)
        expected[basis, basis] = cosine
        expected[basis + 16, basis] = sine
        expected[basis, basis + 16] = -sine
        expected[basis + 16, basis + 16] = cosine
    np.testing.assert_allclose(
        Operator(circuit).data, expected, atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("weight", [1, 2])
def test_oracle_inverse_is_identity(api, weight):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    args["w_d"] = weight
    if api == "functional":
        forward = circuits.single_oracle_sin_inconstraint(args)
        inverse = circuits.single_oracle_sin_inconstraint(args, inverse=True)
    else:
        forward = optimizer.single_oracle_sin_inconstraint(0.6, args["norm"])
        inverse = optimizer.single_oracle_sin_inconstraint(
            0.6, args["norm"], inverse=True
        )
    np.testing.assert_allclose(
        Operator(forward.compose(inverse)).data, np.eye(32), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("api", ["functional", "optimizer"])
def test_dqa_initialization_and_layer_order(api):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    steps = 2
    angles = [
        angle
        for step in range(steps)
        for angle in (step / steps, (1 - step / steps) / np.pi)
    ]
    args["Theta"] = angles
    if api == "functional":
        circuit = circuits.alternating_operator_ansatz(args)
    else:
        circuit = optimizer.adiabatic_evolution_circuit(
            wind_demand=1, time=4, time_steps=steps, norm=args["cost_norm"]
        )
    dicke = np.array([0, 1, 1, 0]) / np.sqrt(2)
    pdf = np.ones(4) / 2
    expected = Statevector(np.kron(pdf, dicke))
    for cost_angle, mixer_angle in zip(angles[::2], angles[1::2]):
        expected = expected.evolve(
            circuits.cost_operator(cost_angle, args)
        ).evolve(
            circuits.demand_constraint_preserving_mixer(mixer_angle, args),
            qargs=[0, 1],
        )
    assert Statevector.from_instruction(circuit).equiv(expected, atol=ATOL)


@pytest.mark.parametrize("api", ["functional", "optimizer"])
@pytest.mark.parametrize("peak", [0, 1, 2, 4])
def test_qae_known_amplitude_distribution(api, peak):
    optimizer = make_optimizer(1)
    args = circuit_args(optimizer)
    args["m"] = 3
    preparation = QuantumCircuit(2)
    oracle = QuantumCircuit(3)
    oracle.ry(2 * np.pi * peak / 8, 2)
    if api == "functional":
        circuit = QAE_Optimizer(args).implemented_qae(
            preparation, oracle, preparation.inverse(), oracle.inverse()
        )
    else:
        circuit = optimizer.implemented_qae(
            preparation, oracle, preparation.inverse(), oracle.inverse(),
            args["m"], args["norm"],
        )
    expected = np.zeros(8)
    expected[peak] += 0.5
    expected[-peak % 8] += 0.5
    actual = Statevector.from_instruction(circuit).probabilities([0, 1, 2])
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


def test_qae_assembly_apis_agree_for_nontrivial_inputs():
    optimizer = make_optimizer(1)
    args = circuit_args(optimizer)
    preparation = QuantumCircuit(2)
    preparation.ry(0.41, 0)
    preparation.h(1)
    preparation.cp(0.27, 0, 1)
    oracle = QuantumCircuit(3)
    oracle.cry(0.63, 0, 2)
    oracle.cry(-0.19, 1, 2)
    modular = QAE_Optimizer(args).implemented_qae(
        preparation, oracle, preparation.inverse(), oracle.inverse()
    )
    integrated = optimizer.implemented_qae(
        preparation, oracle, preparation.inverse(), oracle.inverse(),
        args["m"], args["norm"],
    )
    assert Operator(modular).equiv(Operator(integrated), atol=ATOL)


@pytest.mark.parametrize("gateset", [False, ["u", "cx"]])
def test_compile_qae_wires_modular_builders(gateset):
    optimizer = make_optimizer(1)
    args = circuit_args(optimizer)
    args["gateset"] = gateset
    compiled = QAE_Optimizer(args).compile_qae_circuit()
    preparation = circuits.alternating_operator_ansatz(args)
    oracle = circuits.single_oracle_sin_inconstraint(args)
    expected = optimizer.implemented_qae(
        preparation, oracle, preparation.inverse(), oracle.inverse(),
        args["m"], args["norm"],
    )
    assert Statevector.from_instruction(compiled).equiv(
        Statevector.from_instruction(expected), atol=ATOL
    )
