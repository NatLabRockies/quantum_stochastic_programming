# GPU Porting Guide

The current code runs on the CPU Aer statevector simulator.
The NREL HPC module `qiskit/aer-gpu` exposes `AerSimulator` with `device='GPU'`
using NVIDIA cuStateVec (cuQuantum) for GPU-accelerated simulation.

---

## Three-Tier Strategy

### Tier 1 — Drop-in backend swap (immediate, ~1 line per location)

Replace every `Aer.get_backend(...)` call:

```python
# BEFORE
from qiskit_aer import Aer
simulator = Aer.get_backend('statevector_simulator')
# or
simulator = Aer.get_backend('aer_simulator_statevector')

# AFTER
from qiskit_aer import AerSimulator
simulator = AerSimulator(method='statevector', device='GPU')
```

Locations marked with `# GPU-SWAP TIER 1` in the codebase:

- `qiskit_impl/binary_optimizer.py` → `execute_optimizer()`, `execute_optimizer_oracle()`, `execute_qae()`
- `QuantumExpectedValueFunctionProject/dense_optimizer.py` → all `solveAnnealing*()` methods

Expected speedup: **5–50×** for circuits with $n \geq 20$ qubits. Risk: **zero** — pure backend swap.

---

### Tier 2 — Transpile once + `ParameterVector` binding

Inside DQA time-step loops, sub-circuits are currently rebuilt with numeric angles each iteration.
The fix — demonstrated in `bayesianQC/optimize_10epoch_performance.py`:

```python
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Build symbolic template ONCE
s = Parameter('s')
cost_tmpl = cost_operator_symbolic(s, ...)

# Transpile ONCE to GPU target
pm = generate_preset_pass_manager(optimization_level=1, backend=gpu_simulator)
cost_t = pm.run(cost_tmpl)

# Inside loop: only bind numbers (cheap)
for t in range(time_steps):
    s_val = (dt * t) / time
    qc.append(cost_t.assign_parameters({s: s_val}), qubits)
```

Locations marked with `# PARAM-TRANSPILE` in the codebase.

---

### Tier 3 — cuStateVec (>20 qubits)

```python
simulator = AerSimulator(
    method='statevector', device='GPU',
    cuStateVec_enable=True,     # NVIDIA cuStateVec via cuquantum-cu12
    blocking_enable=True,        # auto-chunk if VRAM insufficient
    blocking_qubits=23,
)
```

---

## Benchmarking Plan

1. Run `brute_force_energy_surface()` as classical baseline
2. For `n_y` in [3, 4, 5, 6, 7, 8]: time `execute_optimizer()` on CPU vs GPU
3. For `m_qae` in [3, 4, 5, 6, 7]: time `execute_qae()` on CPU vs GPU
4. Plot speedup vs `n_qubits` and vs circuit depth
5. Report: max speedup, crossover point (where GPU beats CPU)
