# Shared circuit builders

These typed functions are the canonical circuit implementations used by both
the optimizer methods and the dictionary-based compatibility API.

The builders remain specific to binary wind decisions and scenarios. Their
local layout is `y[0:n_y]`, `xi[n_y:2*n_y]`, with an additional final ancilla for
cost oracles. They return `QuantumCircuit` objects, not gates; register mapping
and existing return types are preserved by the compatibility adapters.
This cleanup does not introduce arbitrary-width registers or new encodings.

Active scripts and notebooks use these functions directly. Use explicit package
imports, rather than adding `qiskit_impl/` itself to `sys.path`:

```python
from qiskit_impl import circuits
from qiskit_impl.dist_prep import build_pdf_circuit

dqa = circuits.alternating_operator_ansatz(
    n_y, y_reg, pdf_reg, angles,
    circuits.dicke_state_circuit(n_y, wind_demand),
    build_pdf_circuit(n_y, pdf, uniform=True),
    lambda angle: circuits.cost_operator(angle, wind_costs, recourse_cost, cost_norm, pdf_reg),
    lambda angle: circuits.demand_constraint_preserving_mixer(angle, y_reg),
)
```

The original dictionary adapters and public optimizer methods remain available;
their removal is not part of this cleanup.

Oracle encoding is explicit:

- `linear`: rotations of `pi * scale * cost / norm`, as used by the optimizer's
  sin oracle.
- `arcsine`: rotations of `2 * arcsin(sqrt(cost / norm))`. These encode the
  normalized cost exactly only when one selected turbine contributes.

The dictionary sin-oracle adapter retains its original rule: demand one
(including an omitted `w_d`) uses arcsine encoding; other demands use linear
encoding. The optimizer's separately named arcsine method always uses arcsine
encoding. No automatic change is made to either API's behavior.

::: qiskit_impl.circuits
