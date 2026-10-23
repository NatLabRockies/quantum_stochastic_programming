# Shared circuit builders

These typed functions are the canonical circuit implementations used by both
the optimizer methods and the dictionary-based compatibility API.

The builders remain specific to binary wind decisions and scenarios. Their
local layout is `y[0:n_y]`, `xi[n_y:2*n_y]`, with an additional final ancilla for
cost oracles. They return `QuantumCircuit` objects, not gates; register mapping
and existing return types are preserved by the compatibility adapters.
This cleanup does not introduce arbitrary-width registers or new encodings.

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
