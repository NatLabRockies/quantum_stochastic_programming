# `qae` — Standalone QAE Circuit Builder

Assembles the canonical Quantum Amplitude Estimation (QAE) circuit from an `args`
dictionary. Logic is separated from `binary_optimizer` so the QAE circuit can be
reused with different cost/mixer functions.

---

## `QAE_Optimizer`

::: qiskit_impl.qae.QAE_Optimizer
