# `cudaq_impl` — CUDA-Q Port

GPU-accelerated CUDA-Q implementation of the DQA+QAE pipeline.
Mirrors the Qiskit reference in `binary_optimizer.py` and `ExpValFun_functions.py`,
replacing Qiskit `QuantumCircuit` objects with `@cudaq.kernel` primitives that
run natively on NVIDIA GPUs via cuStateVec.

---

## Optimizer Class

::: qiskit_impl.cudaq_impl.CudaqQAEOptimizer

---

## Kernel Primitives

### `pdf_init_uniform`

::: qiskit_impl.cudaq_impl.pdf_init_uniform

---

### `ccry`

::: qiskit_impl.cudaq_impl.ccry

---

### `dicke_state_n4_k2`

::: qiskit_impl.cudaq_impl.dicke_state_n4_k2

---

### `cost_operator_n4`

::: qiskit_impl.cudaq_impl.cost_operator_n4

---

### `fswap_power`

::: qiskit_impl.cudaq_impl.fswap_power

---

### `mixer_n4`

::: qiskit_impl.cudaq_impl.mixer_n4

---

### `oracle_sin_n4`

::: qiskit_impl.cudaq_impl.oracle_sin_n4

---

### `dqa_ansatz_n4`

::: qiskit_impl.cudaq_impl.dqa_ansatz_n4
