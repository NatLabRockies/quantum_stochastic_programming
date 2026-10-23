"""Brute-force references for binary wind decisions and one gas generator.

Decision and scenario entries are in turbine order, not displayed-qubit order.
Ties retain the lexicographically first feasible decision. Infeasible wind
demands have infinite cost and no decision.
"""

from collections.abc import Mapping, Sequence
from itertools import product
import math


def _validate_inputs(
    wind_costs: Sequence[float],
    recourse_cost: float,
    wind_demand: int,
) -> None:
    if not isinstance(wind_demand, int) or wind_demand < 0:
        raise ValueError("wind_demand must be a nonnegative integer")
    if not math.isfinite(recourse_cost) or any(
        not math.isfinite(cost) for cost in wind_costs
    ):
        raise ValueError("Costs must be finite")


def _validate_bits(bits: Sequence[int], n: int) -> None:
    if len(bits) != n or any(bit not in (0, 1) for bit in bits):
        raise ValueError("Decisions and scenarios must contain one binary entry per turbine")


def _validate_pdf(
    pdf: Mapping[tuple[int, ...], float], n: int
) -> None:
    for scenario, probability in pdf.items():
        _validate_bits(scenario, n)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("PDF probabilities must be finite and nonnegative")
    if not math.isclose(sum(pdf.values()), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("PDF probabilities must sum to one")


def scenario_cost(
    wind_output: Sequence[int],
    scenario: Sequence[int],
    wind_demand: int,
    wind_costs: Sequence[float],
    recourse_cost: float,
) -> float:
    """Evaluate a fixed decision, including backup purchases for any shortfall.

    Unlike the constrained minimizer, this evaluator does not require the
    decision's Hamming weight to equal wind demand.
    """
    _validate_inputs(wind_costs, recourse_cost, wind_demand)
    _validate_bits(wind_output, len(wind_costs))
    _validate_bits(scenario, len(wind_costs))
    output = [decision * available for decision, available in zip(wind_output, scenario)]
    return float(
        sum(cost * power for cost, power in zip(wind_costs, output))
        + max(0, wind_demand - sum(output)) * recourse_cost
    )


def solve_scenario(
    wind_demand: int,
    scenario: Sequence[int],
    wind_costs: Sequence[float],
    recourse_cost: float,
) -> tuple[float, tuple[int, ...]]:
    """Minimize cost over decisions with exactly wind_demand selected turbines.

    Returns (infinity, ()) if wind_demand exceeds the turbine count.
    Equal-cost decisions are resolved in lexicographic order.
    """
    _validate_inputs(wind_costs, recourse_cost, wind_demand)
    _validate_bits(scenario, len(wind_costs))
    if wind_demand > len(wind_costs):
        return math.inf, ()
    minimum = math.inf
    decision: tuple[int, ...] = ()
    for candidate in product((0, 1), repeat=len(wind_costs)):
        if sum(candidate) == wind_demand:
            cost = scenario_cost(
                candidate, scenario, wind_demand, wind_costs, recourse_cost
            )
            if cost < minimum:
                minimum, decision = cost, candidate
    return minimum, decision


def expected_recourse_cost(
    wind_demand: int,
    wind_costs: Sequence[float],
    recourse_cost: float,
    pdf: Mapping[tuple[int, ...], float],
) -> float:
    """Return E_scenario[min_decision cost], including wind operating costs.

    This is scenario-wise optimization, not minimization of expected cost for
    one fixed decision. Infeasible demands return infinity.
    """
    _validate_inputs(wind_costs, recourse_cost, wind_demand)
    _validate_pdf(pdf, len(wind_costs))
    if wind_demand > len(wind_costs):
        return math.inf
    return sum(
        probability * solve_scenario(
            wind_demand, scenario, wind_costs, recourse_cost
        )[0]
        for scenario, probability in pdf.items()
        if probability > 0
    )


def wind_demand_costs(
    wind_costs: Sequence[float],
    recourse_cost: float,
    pdf: Mapping[tuple[int, ...], float],
    demand: int,
) -> list[float]:
    """Return exact second-stage costs for wind demands 0 through demand."""
    _validate_inputs(wind_costs, recourse_cost, demand)
    return [
        expected_recourse_cost(wind_demand, wind_costs, recourse_cost, pdf)
        for wind_demand in range(demand + 1)
    ]


def first_stage_objective(
    gas_cost: float,
    wind_costs: Sequence[float],
    recourse_cost: float,
    pdf: Mapping[tuple[int, ...], float],
    demand: int,
) -> dict[int, float]:
    """Return {gas_level: gas_level * gas_cost + second_stage_cost}.

    Models a single linear-cost gas generator, as in the original reference.
    """
    if not math.isfinite(gas_cost):
        raise ValueError("gas_cost must be finite")
    costs = wind_demand_costs(wind_costs, recourse_cost, pdf, demand)
    return {
        demand - wind_demand: (demand - wind_demand) * gas_cost + value
        for wind_demand, value in enumerate(costs)
    }
