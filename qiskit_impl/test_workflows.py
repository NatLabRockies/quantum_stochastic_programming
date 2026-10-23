"""Regression checks for the migrated active Qiskit workflows."""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from experiments.qiskit_sweep import run_sweep
from qiskit_impl import ExpValFun_functions as compatibility
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
from qiskit_impl.dist_prep import pdf_initialize
from qiskit_impl.qae import QAE_Optimizer


@pytest.mark.parametrize("n_y", [3, 4])
def test_exact_sweep_preserves_dictionary_workflow(n_y):
    """Exercise demand-one arcsine and demand-two linear encodings."""
    options = SimpleNamespace(
        n_y=[n_y], timesteps=2, qpe_qubits=2, backend="exact", shots=128,
    )
    result, = run_sweep(options)
    wind_demand = n_y - 2
    costs = list(np.linspace(0.1, 1.0, n_y))
    pdf = {
        tuple(int(bit) for bit in f"{index:0{n_y}b}"): 1 / 2**n_y
        for index in range(2**n_y)
    }
    args = {
        "n_y": n_y, "c_y": costs, "c_r": 10.0, "pdf": pdf,
        "y_reg": list(range(n_y)), "pdf_reg": list(range(n_y, 2 * n_y)),
        "Theta": [0.0, 1 / math.pi, 0.5, 0.5 / math.pi],
        "w_d": wind_demand, "cost_norm": wind_demand * 10.0 / n_y,
        "norm": wind_demand * 10.0, "uniform": True, "m": 2, "gateset": False,
        "cost_operator_circuit": compatibility.cost_operator,
        "mixer_operator_circuit": compatibility.demand_constraint_preserving_mixer,
        "initial_state_circuit": compatibility.dicke_state_circuit,
        "pdf_circuit": pdf_initialize,
        "oracle_circuit": compatibility.single_oracle_sin_inconstraint,
    }
    optimizer = BinaryNestedOptimizer([3.0], costs, 10.0, pdf, n_y, True)
    dqa = compatibility.alternating_operator_ansatz(args)
    probabilities = Statevector.from_int(0, 2**(2 * n_y)).evolve(dqa).probabilities_dict()
    expected_dqa = optimizer.process_expectation_value_optimizer(wind_demand, probabilities)
    qae = QAE_Optimizer(args).compile_qae_circuit()
    readout = optimizer.execute_qae(qae, args["m"])
    expected_qae = sum(
        np.sin(int(key, 2) * np.pi / 2**args["m"])**2 * args["norm"] * probability
        for key, probability in readout.items()
    )
    assert result["w_d"] == wind_demand
    assert result["dqa_phi"] == pytest.approx(expected_dqa, abs=1e-10)
    assert result["qae_phi"] == pytest.approx(expected_qae, abs=1e-10)
    assert result["dqa_time_s"] >= 0
    assert result["qae_time_s"] >= 0


@pytest.mark.parametrize("name", ["qae_example.ipynb", "qae_example_backends.ipynb"])
def test_notebook_code_parses(name):
    notebook = json.loads(Path(__file__).with_name(name).read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(
                line for line in cell["source"] if not line.lstrip().startswith("%")
            )
            ast.parse(source, filename=f"{name}:cell-{index}")
