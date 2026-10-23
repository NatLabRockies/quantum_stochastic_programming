# `qae` — Standalone QAE Circuit Builder

`build_qae_circuit` assembles the canonical Quantum Amplitude Estimation (QAE)
circuit from preparation and oracle circuits and their inverses. Logic is
separated from `binary_optimizer` so QAE can be reused with different
cost/mixer functions. Both existing QAE assembly methods
delegate to `build_qae_circuit`, preserving the readout-first layout and
`2*n_y` system width. The shared builder accepts circuits or gates and does
not transpile or measure; the compatibility class retains optional transpilation.

Active examples use the shared builder directly:

```python
from qiskit_impl.qae import build_qae_circuit

qae = build_qae_circuit(
    dqa, oracle, dqa.inverse(), oracle.inverse(), n_y, m
)
```

The dictionary-based `QAE_Optimizer`, the optimizer's assembly method, and
the PDF helper re-exports in this module remain available for compatibility.
New callers should import PDF functions from `qiskit_impl.dist_prep` directly.

::: qiskit_impl.qae.build_qae_circuit

---

## `QAE_Optimizer`

::: qiskit_impl.qae.QAE_Optimizer
