"""Deterministic checks of shared builders and their compatibility APIs."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Statevector

from qiskit_impl import ExpValFun_functions as circuits
from qiskit_impl import circuits as shared
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
from qiskit_impl.dist_prep import build_pdf_circuit, pdf_initialize
from qiskit_impl.qae import QAE_Optimizer, build_qae_circuit


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


@pytest.mark.parametrize("api", ["functional", "optimizer", "shared"])
@pytest.mark.parametrize(
    ("n", "weight"),
    [(n, weight) for n in range(1, 5) for weight in range(n + 1)],
)
def test_dicke_state_matches_definition(api, n, weight):
    optimizer = make_optimizer(n)
    if api == "shared":
        circuit = shared.dicke_state_circuit(n, weight)
    elif api == "functional":
        circuit = circuits.dicke_state_circuit({"n_y": n, "w_d": weight})
    else:
        circuit = optimizer.dicke_state_circuit(weight)
    amplitude = 1 / math.sqrt(math.comb(n, weight))
    expected = Statevector([
        amplitude if index.bit_count() == weight else 0
        for index in range(2**n)
    ])
    assert Statevector.from_instruction(circuit).equiv(expected, atol=ATOL)


@pytest.mark.parametrize("api", ["functional", "shared", "optimizer"])
@pytest.mark.parametrize("distribution", ["uniform", "asymmetric", "sparse", "deterministic"])
def test_pdf_state_and_qubit_order(api, distribution):
    # Tuple entries correspond to qubits 0, 1, unlike displayed bitstrings.
    probabilities = {
        "uniform": [0.25, 0.25, 0.25, 0.25],
        "asymmetric": [0.1, 0.2, 0.3, 0.4],
        "sparse": [0.0, 0.7, 0.0, 0.3],
        "deterministic": [0.0, 1.0, 0.0, 0.0],
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
    if api == "shared":
        circuit = build_pdf_circuit(2, pdf, uniform=distribution == "uniform")
    elif api == "functional":
        circuit = pdf_initialize(circuit_args(optimizer))
    else:
        circuit = optimizer.pdf_initialize()
    expected = Statevector(np.sqrt(probabilities))
    assert Statevector.from_instruction(circuit).equiv(expected, atol=ATOL)


@pytest.mark.parametrize(
    "pdf",
    [
        {(0, 0): 0.1, (1, 0): 0.2, (0, 1): 0.3, (1, 1): 0.4},
        {(1, 0): 0.7, (1, 1): 0.3},
        {(1, 0): 1.0},
    ],
)
def test_nonuniform_pdf_is_unitary_and_controllable(pdf):
    circuit = build_pdf_circuit(2, pdf, uniform=False)
    gate = circuit.to_gate()
    amplitudes = np.zeros(4)
    for scenario, probability in pdf.items():
        amplitudes[scenario[0] + 2 * scenario[1]] = np.sqrt(probability)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit).data, amplitudes, atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(
        Operator(circuit.compose(circuit.inverse())).data,
        np.eye(4), atol=ATOL, rtol=0,
    )
    for control in (0, 1):
        controlled = QuantumCircuit(3)
        if control:
            controlled.x(0)
        controlled.append(gate.control(), [0, 1, 2])
        expected = np.zeros(8)
        if control:
            expected[1::2] = amplitudes
        else:
            expected[0] = 1
        np.testing.assert_allclose(
            Statevector.from_instruction(controlled).data,
            expected, atol=ATOL, rtol=0,
        )


def test_nonuniform_dqa_and_qae_match_analytic_expectations():
    optimizer = make_optimizer(
        1, pdf={(0,): 0.25, (1,): 0.75}, uniform=False
    )
    args = circuit_args(optimizer)
    dqa = circuits.alternating_operator_ansatz(args)
    integrated_dqa = optimizer.adiabatic_evolution_circuit(
        wind_demand=1, time=4, time_steps=2, norm=args["cost_norm"]
    )
    expected_cost = 0.25 * args["c_r"] + 0.75 * args["c_y"][0]
    for preparation in (dqa, integrated_dqa):
        state = Statevector.from_instruction(preparation)
        np.testing.assert_allclose(
            state.probabilities(), [0, 0.25, 0, 0.75], atol=ATOL, rtol=0
        )
        assert optimizer.process_expectation_value_optimizer(
            1, state.probabilities_dict()
        ) == pytest.approx(expected_cost, abs=ATOL)

    # For demand one the arcsine oracle encodes expected cost/norm exactly.
    probability = expected_cost / args["norm"]
    theta = np.arcsin(np.sqrt(probability))
    size = 2**args["m"]
    readout = np.arange(size)
    expected = np.zeros(size)
    for phase in (theta / np.pi, -theta / np.pi):
        differences = phase - readout / size
        amplitudes = np.mean(
            np.exp(2j * np.pi * np.outer(differences, np.arange(size))), axis=1
        )
        expected += np.abs(amplitudes)**2 / 2
    qae = QAE_Optimizer(args).compile_qae_circuit()
    actual = Statevector.from_instruction(qae).probabilities(list(range(args["m"])))
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


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


@pytest.mark.parametrize("api", ["functional", "optimizer", "optimizer_arcsine"])
@pytest.mark.parametrize("weight", [1, 2])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("scale", [0.6, 1.0])
def test_oracle_rotation_encoding(api, weight, inverse, scale):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    args["w_d"] = weight
    if api == "functional":
        circuit = circuits.single_oracle_sin_inconstraint(args, inverse=inverse)
    elif api == "optimizer_arcsine":
        circuit = optimizer.single_oracle_asin_inconstraint(args["norm"], inverse=inverse)
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
                if (api == "functional" and weight == 1) or api == "optimizer_arcsine":
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


@pytest.mark.parametrize("api", ["functional", "optimizer", "optimizer_arcsine"])
@pytest.mark.parametrize("weight", [1, 2])
def test_oracle_inverse_is_identity(api, weight):
    optimizer = make_optimizer(2)
    args = circuit_args(optimizer)
    args["w_d"] = weight
    if api == "functional":
        forward = circuits.single_oracle_sin_inconstraint(args)
        inverse = circuits.single_oracle_sin_inconstraint(args, inverse=True)
    elif api == "optimizer_arcsine":
        forward = optimizer.single_oracle_asin_inconstraint(args["norm"])
        inverse = optimizer.single_oracle_asin_inconstraint(args["norm"], inverse=True)
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


@pytest.mark.parametrize("api", ["functional", "optimizer", "shared"])
@pytest.mark.parametrize("peak", [0, 1, 2, 4])
def test_qae_known_amplitude_distribution(api, peak):
    optimizer = make_optimizer(1)
    args = circuit_args(optimizer)
    args["m"] = 3
    preparation = QuantumCircuit(2)
    oracle = QuantumCircuit(3)
    oracle.ry(2 * np.pi * peak / 8, 2)
    if api == "shared":
        circuit = build_qae_circuit(
            preparation, oracle, preparation.inverse(), oracle.inverse(),
            args["n_y"], args["m"],
        )
    elif api == "functional":
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


@pytest.mark.parametrize("weight", [0, 1, 2])
def test_dicke_compatibility_return_types(weight):
    optimizer = make_optimizer(2)
    assert isinstance(shared.dicke_state_circuit(2, weight), QuantumCircuit)
    assert isinstance(circuits.dicke_state_circuit({"n_y": 2, "w_d": weight}), QuantumCircuit)
    expected_type = QuantumCircuit if weight == 0 else Gate
    assert isinstance(optimizer.dicke_state_circuit(weight), expected_type)


def test_uniform_pdf_adapter_does_not_require_explicit_probabilities():
    actual = Statevector.from_instruction(pdf_initialize({"n_y": 2, "uniform": True}))
    assert actual.equiv(Statevector([0.5, 0.5, 0.5, 0.5]), atol=ATOL)


def test_modular_ansatz_retains_custom_factories_and_register_mapping():
    args = circuit_args(make_optimizer(1))
    args.update({
        "y_reg": [1],
        "pdf_reg": [0],
        "Theta": [0.4, -0.7, 0.2],
        "initial_state_circuit": lambda _: QuantumCircuit(1),
        "pdf_circuit": lambda _: QuantumCircuit(1),
    })

    def cost(theta, _):
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.cp(theta, 0, 1)
        return qc

    def mixer(theta, _):
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        return qc

    args.update({"cost_operator_circuit": cost, "mixer_operator_circuit": mixer})
    expected = QuantumCircuit(2)
    expected.rx(0.4, 1)
    expected.cp(0.4, 1, 0)
    expected.ry(-0.7, 1)
    expected.rx(0.2, 1)
    expected.cp(0.2, 1, 0)
    assert Operator(circuits.alternating_operator_ansatz(args)).equiv(
        Operator(expected), atol=ATOL
    )


def test_oracle_default_demand_still_uses_arcsine_encoding():
    args = circuit_args(make_optimizer(2))
    del args["w_d"]
    actual = circuits.single_oracle_sin_inconstraint(args)
    expected = shared.cost_oracle(
        args["c_y"], args["c_r"], args["norm"], args["y_reg"], args["pdf_reg"],
        encoding="arcsine",
    )
    np.testing.assert_allclose(
        Operator(actual).data, Operator(expected).data, atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("invalid_encoding", ["unknown"])
def test_oracle_rejects_invalid_encoding_and_arcsine_scale(invalid_encoding):
    args = circuit_args(make_optimizer(1))
    with pytest.raises(ValueError, match="Unsupported oracle encoding"):
        shared.cost_oracle(
            args["c_y"], args["c_r"], args["norm"], [0], [1],
            encoding=invalid_encoding,
        )
    with pytest.raises(ValueError, match="scale"):
        shared.cost_oracle(
            args["c_y"], args["c_r"], args["norm"], [0], [1],
            encoding="arcsine", scale=0.5,
        )
