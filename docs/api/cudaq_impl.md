# `cudaq_impl` — CUDA-Q Port

GPU-accelerated CUDA-Q implementation of the DQA+QAE pipeline.
Mirrors the Qiskit reference in `binary_optimizer.py` and `ExpValFun_functions.py`,
replacing Qiskit `QuantumCircuit` objects with `@cudaq.kernel` primitives that
run natively on NVIDIA GPUs via cuStateVec.

---

## Optimizer Class

::: cudaq_impl.cudaq_impl.CudaqQAEOptimizer

---

## Kernel Primitives

### `pdf_init_uniform`

::: cudaq_impl.cudaq_impl.pdf_init_uniform

---

### `ccry`

::: cudaq_impl.cudaq_impl.ccry

---

### `dicke_state_n4_k2`

::: cudaq_impl.cudaq_impl.dicke_state_n4_k2

---

### `cost_operator`

::: cudaq_impl.cudaq_impl.cost_operator

---

### `fswap_power`

::: cudaq_impl.cudaq_impl.fswap_power

---

### `mixer`

::: cudaq_impl.cudaq_impl.mixer

---

### `oracle_sin`


::: cudaq_impl.cudaq_impl.oracle_sin

---

### `dqa_ansatz`

::: cudaq_impl.cudaq_impl.dqa_ansatz
