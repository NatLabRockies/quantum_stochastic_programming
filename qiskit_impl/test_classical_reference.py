"""Compatibility checks for the optimizer's classical facade."""

import pytest

from classical import (
    first_stage_objective,
    scenario_cost,
    solve_scenario,
    wind_demand_costs,
)
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer


@pytest.fixture
def optimizer():
    return BinaryNestedOptimizer(
        [4], [1, 3], 10,
        {(0, 0): 0.1, (1, 0): 0.2, (0, 1): 0.3, (1, 1): 0.4},
        2, is_uniform=False,
    )


def test_existing_optimizer_matches_hand_calculated_reference(optimizer):
    assert optimizer.brute_force_wind_demand_expectation_values() == pytest.approx(
        [0, 2.5, 9.7]
    )
    assert optimizer.brute_force_energy_surface() == pytest.approx(
        {2: 8, 1: 6.5, 0: 9.7}
    )
    for demand in range(3):
        for scenario in optimizer.pdf:
            expected = solve_scenario(
                demand, scenario, optimizer.wind_costs, optimizer.recourse_cost
            )
            assert optimizer.min_scenario(demand, scenario) == expected
            cost, decision = expected
            assert optimizer.wind_scenario_cost(decision, scenario, demand) == cost
            assert scenario_cost(
                decision, scenario, demand,
                optimizer.wind_costs, optimizer.recourse_cost,
            ) == cost


def test_reference_tables_match_optimizer(optimizer):
    assert optimizer.brute_force_wind_demand_expectation_values() == wind_demand_costs(
        optimizer.wind_costs, optimizer.recourse_cost, optimizer.pdf, optimizer.demand
    )
    assert optimizer.brute_force_energy_surface() == first_stage_objective(
        optimizer.gas_costs[0], optimizer.wind_costs, optimizer.recourse_cost,
        optimizer.pdf, optimizer.demand,
    )


def test_ground_state_reference_assembly_is_unchanged(optimizer):
    # Characterize the existing indexing independently of circuit construction.
    import numpy as np

    expected = np.zeros(16)
    expected[[4, 10, 5, 11]] = np.sqrt([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(optimizer.prep_gs(1), expected)


def test_optimizer_infeasible_demands_have_infinite_cost(optimizer):
    optimizer.demand = 3
    assert optimizer.min_scenario(3, (1, 1)) == (float("inf"), ())
    assert optimizer.brute_force_wind_demand_expectation_values()[-1] == float("inf")
    assert optimizer.brute_force_energy_surface()[0] == float("inf")
    with pytest.raises(ValueError, match="infeasible"):
        optimizer.prep_gs(3)


def test_optimizer_does_not_clip_large_costs(optimizer):
    optimizer.wind_costs = [100, 200]
    optimizer.recourse_cost = 1
    assert optimizer.min_scenario(1, (1, 1)) == (100, (1, 0))
