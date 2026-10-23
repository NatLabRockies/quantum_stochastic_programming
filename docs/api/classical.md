# Classical reference computations

The `classical` package uses only the Python standard library and does not
import Qiskit, Aer, CUDA-Q, NumPy, or the quantum implementation packages.

Each binary scenario and decision tuple is indexed by turbine. The constrained
solver selects exactly `wind_demand` turbines, charges operating costs only for
available selected turbines, and purchases backup power for any shortfall.
The expectation is taken over **scenario-wise optimal decisions**.

The first-stage objective assumes a single linear-cost gas generator. Invalid
inputs raise `ValueError`; wind demand above the turbine count instead has
infinite cost and an empty optimal decision. Zero-probability scenarios do not
contribute to expectations. Ties retain the lexicographically first decision.

The classical methods on `BinaryNestedOptimizer` remain available as wrappers.
Unlike their old implementation, they do not use an artificial finite cost
bound. Reference wavefunction construction remains in the quantum layer,
including its existing indexing convention, and rejects infeasible demands.

::: classical.reference.scenario_cost

::: classical.reference.solve_scenario

::: classical.reference.expected_recourse_cost

::: classical.reference.wind_demand_costs

::: classical.reference.first_stage_objective
