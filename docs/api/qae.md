# `qae` — Standalone QAE Circuit Builder

Assembles the canonical Quantum Amplitude Estimation (QAE) circuit from an `args`
dictionary. Logic is separated from `binary_optimizer` so the QAE circuit can be
reused with different cost/mixer functions. Both existing QAE assembly methods
delegate to `build_qae_circuit`, preserving the readout-first layout and
`2*n_y` system width. The shared builder accepts circuits or gates and does
not transpile or measure; the compatibility class retains optional transpilation.

::: qiskit_impl.qae.build_qae_circuit

---

## `QAE_Optimizer`

::: qiskit_impl.qae.QAE_Optimizer
