"""Independent numerical checks of the classical problem definition."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from classical import (
    expected_recourse_cost,
    first_stage_objective,
    scenario_cost,
    solve_scenario,
    wind_demand_costs,
)


WIND_COSTS = [1.0, 3.0]
RECOURSE = 10.0
PDF = {(0, 0): 0.1, (1, 0): 0.2, (0, 1): 0.3, (1, 1): 0.4}


@pytest.mark.parametrize(
    ("decision", "scenario", "demand", "expected"),
    [
        ((0, 0), (1, 1), 0, 0),
        ((1, 0), (1, 0), 1, 1),
        ((1, 0), (0, 1), 1, 10),
        ((1, 1), (1, 0), 2, 11),
        ((1, 1), (1, 1), 1, 4),
        ((0, 0), (1, 1), 2, 20),
    ],
)
def test_fixed_decision_cost(decision, scenario, demand, expected):
    assert scenario_cost(
        decision, scenario, demand, WIND_COSTS, RECOURSE
    ) == expected


@pytest.mark.parametrize(
    ("demand", "scenario", "expected"),
    [
        (0, (1, 1), (0, (0, 0))),
        (1, (0, 0), (10, (0, 1))),
        (1, (1, 0), (1, (1, 0))),
        (1, (0, 1), (3, (0, 1))),
        (1, (1, 1), (1, (1, 0))),
        (2, (1, 1), (4, (1, 1))),
        (3, (1, 1), (float("inf"), ())),
    ],
)
def test_optimal_decision_and_tie_breaking(demand, scenario, expected):
    assert solve_scenario(demand, scenario, WIND_COSTS, RECOURSE) == expected


def test_large_costs_do_not_use_artificial_upper_bound():
    assert solve_scenario(1, (1, 1), [100, 200], 1) == (100, (1, 0))


def test_expected_cost_is_expectation_of_scenario_minima():
    # For demand one, costs by scenario are 10, 1, 3, 1.
    # For demand two they are 20, 11, 13, 4.
    assert wind_demand_costs(WIND_COSTS, RECOURSE, PDF, 2) == pytest.approx(
        [0, 2.5, 9.7]
    )
    assert expected_recourse_cost(1, WIND_COSTS, RECOURSE, PDF) == pytest.approx(2.5)
    fixed_decision_costs = [
        sum(p * scenario_cost(y, xi, 1, WIND_COSTS, RECOURSE) for xi, p in PDF.items())
        for y in [(0, 1), (1, 0)]
    ]
    assert min(fixed_decision_costs) > 2.5


def test_first_stage_objective_and_feasible_optimum():
    surface = first_stage_objective(4, WIND_COSTS, RECOURSE, PDF, 3)
    assert surface == pytest.approx({3: 12, 2: 10.5, 1: 13.7, 0: float("inf")})
    assert min(surface, key=surface.get) == 2


def test_zero_probability_and_infeasible_demands():
    pdf = {(0, 0): 0.0, (1, 1): 1.0}
    assert wind_demand_costs(WIND_COSTS, RECOURSE, pdf, 3) == [
        0, 1, 4, float("inf")
    ]


@pytest.mark.parametrize("demand", [-1, 1.5])
def test_invalid_demand_is_rejected(demand):
    with pytest.raises(ValueError, match="nonnegative integer"):
        wind_demand_costs(WIND_COSTS, RECOURSE, PDF, demand)


@pytest.mark.parametrize(
    "pdf",
    [
        {},
        {(0, 0): 0.5},
        {(0,): 1.0},
        {(0, 2): 1.0},
        {(0, 0): -0.1, (1, 1): 1.1},
        {(0, 0): float("nan")},
    ],
)
def test_invalid_pdf_is_rejected(pdf):
    with pytest.raises(ValueError):
        expected_recourse_cost(1, WIND_COSTS, RECOURSE, pdf)


def test_invalid_decision_is_rejected():
    with pytest.raises(ValueError, match="binary entry"):
        scenario_cost((1,), (1, 0), 1, WIND_COSTS, RECOURSE)


@pytest.mark.parametrize("cost", [float("nan"), float("inf")])
def test_nonfinite_costs_are_rejected(cost):
    with pytest.raises(ValueError, match="finite"):
        solve_scenario(1, (1, 1), [cost, 3], RECOURSE)
    with pytest.raises(ValueError, match="finite"):
        solve_scenario(1, (1, 1), WIND_COSTS, cost)
    with pytest.raises(ValueError, match="finite"):
        first_stage_objective(cost, WIND_COSTS, RECOURSE, PDF, 2)


def test_package_runs_without_quantum_or_numerical_libraries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
class BlockQuantumImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'qiskit', 'qiskit_aer', 'cudaq', 'numpy', 'scipy',
            'qiskit_impl', 'cudaq_impl',
        }:
            raise ImportError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockQuantumImports())
from classical import first_stage_objective
assert first_stage_objective(4, [1], 10, {(1,): 1.0}, 1) == {1: 4, 0: 1}
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
