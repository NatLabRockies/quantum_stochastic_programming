# Quantum Stochastic Programming

Quantum computing algorithms for **two-stage stochastic optimization**, with a focus on the Unit Commitment (UC) problem in power systems.
The algorithms combine **Discrete Quantum Annealing (DQA)** with **Quantum Amplitude Estimation (QAE)** to compute expected-value objective functions over a probability distribution of wind-power scenarios.

Based on: [arXiv 2402.15029](https://arxiv.org/abs/2402.15029) — *"Quantum algorithms for the two-stage stochastic unit commitment problem"*

---

## Problem Description

The **Unit Commitment** problem decides which gas generators to commit before uncertain wind power is realised (first-stage decision $x$), then dispatches all generators optimally for each realised scenario (second-stage recourse $y$).

$$\min_{x,y} \; c_x^\top x + \mathbb{E}_\xi\!\left[\min_r c_r r(\xi) \right] \quad \text{s.t.} \quad \sum_j y_j(\xi) + r(\xi) = d, \quad y_j(\xi) \le \xi_j$$

The quantum algorithm evaluates $\mathbb{E}[C(x,\xi)]$ via QAE in $O(1/\epsilon)$ circuit evaluations rather than the $O(1/\epsilon^2)$ shots needed classically.

---

## Repository Structure

Two implementations exist:

| | `qiskit_impl/` | `QuantumExpectedValueFunctionProject/` |
|---|---|---|
| API style | Functional primitives + `args` dicts | Class-based OOP |
| Encoding | Binary only | Binary or Unary |
| Qiskit version | 2.x (current) | Originally 0.x/1.x (needs import fixes) |
| Status | Active development | Legacy / reference |

---

## Quick Start

```python
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
import numpy as np

# Problem: 2 wind turbines, total demand = 2 MW (both turbines needed).
# Each turbine produces exactly 1 MW when ON and wind is available.
# The algorithm decides: which turbines to commit (y) given a gas backup (x)?

# Probability distribution over wind scenarios: keys are (ξ_0, ξ_1) binary tuples.
# ξ_j = 1: wind is blowing for turbine j (it can generate); ξ_j = 0: no wind.
pdf = {(0,0): 0.25,   # no wind for either turbine
       (0,1): 0.25,   # wind only for turbine 1
       (1,0): 0.25,   # wind only for turbine 0
       (1,1): 0.25}   # wind available for both turbines

bno = BinaryNestedOptimizer(
    [0.4],            # gas_costs:     cost of committing the gas generator ($/MW)
                      #                 1 gas generator here; it can cover 0, 1, or 2 MW.
    [0.05, 0.10],     # wind_costs:    dispatch cost per turbine when wind is available ($/MWh)
                      #                 turbine 0 = $0.05/MWh (cheaper), turbine 1 = $0.10/MWh
    1.0,              # recourse_cost: penalty per MW of unmet demand (expensive backup) ($/MWh)
    pdf,              # pdf:           scenario probability distribution (must sum to 1.0)
    demand=2,         # demand:        total MW that must be covered each period
    is_uniform=True,  # is_uniform:    if True, override pdf weights with uniform 1/N
)
# With demand=2: each scenario dispatches both turbines (y=[1,1]).
# If wind fails for one (ξ_j=0), that turbine produces 0 MW → recourse covers the gap.
# Tradeoff: commit gas (certain but costly at $0.40) vs rely on wind (cheap but uncertain).

# --- Step 1: Classical brute-force reference ---
# Computes φ(x) = E_ξ[min_y q(y,ξ)] for ALL values of x (0, 1, 2).
# Returns a dict/array indexed by x: ev_true[x] = expected second-stage cost when gas covers x MW.
ev_true = bno.brute_force_wind_demand_expectation_values()
print("Classical φ(x) for all x:", ev_true)
# Total cost for each x: c_x * x + φ(x) = 0.4*x + ev_true[x]  → pick x with minimum total cost.

# --- Step 2: DQA quantum estimate of φ(x) for one specific x ---
#
# 'wind_demand' is the MW the wind turbines must cover = demand - x.
# It encodes the first-stage decision x implicitly:
#   wind_demand=2  →  x=0  (gas off, wind covers all 2 MW)
#   wind_demand=1  →  x=1  (gas covers 1 MW, wind covers the remaining 1 MW)
#   wind_demand=0  →  x=2  (gas covers everything, no wind needed)
#
# To find the optimal x, run the circuit once per candidate and pick min 0.4*x + φ̃(x).
# Here we evaluate x=0 (wind_demand=2) as an example.
# (Spoiler: x=1 is optimal for this problem — see the Algorithm Overview for the full comparison.)
qc = bno.adiabatic_evolution_circuit(
    wind_demand=2,    # encodes x=0: wind turbines must cover all 2 MW of demand
    time=100,         # total annealing time T (longer → closer to ground state)
    time_steps=4,     # number of Trotter steps p (circuit depth ∝ p)
    norm=10,          # normalization factor for Hamiltonian coefficients
)
counts = bno.execute_optimizer(qc, num_meas=4096)
ev_dqa = bno.process_expectation_value_optimizer(wind_demand=2, counts=counts)
print(f"DQA estimate of φ(x=0):  {ev_dqa:.4f}  (classical: {ev_true[0]:.4f})")
```

---

## Installation

### NREL HPC (GPU-enabled)

```bash
module load qiskit/aer-gpu
# Python 3.12, Qiskit 2.2.3, qiskit-aer-gpu 0.15.1 (CUDA 12.4, H100)
```

### Local

```bash
pip install qiskit>=2.0 qiskit-aer qiskit-algorithms matplotlib scipy numpy
```

---

## Running the Code

### Start here (active implementation)

**`qiskit_impl/qae_example.ipynb`** — the main end-to-end notebook. Walks through the full algorithm (DQA + QAE) using the current `qiskit_impl/` codebase. Run this first.

```bash
cd qiskit_impl
jupyter notebook qae_example.ipynb
```

### Legacy / reference notebooks (original paper)

These are in `QuantumExpectedValueFunctionProject/` and use an older Qiskit API (may need import fixes):

| Notebook | Purpose |
|---|---|
| `four_binary_turbines_uniform_distribution_tutorial.ipynb` | Step-by-step tutorial: 4 turbines, uniform wind distribution |
| `ExpValFun_QAOA_QAE_computations.ipynb` | Reproduces the numerical results from the paper |
| `ExpValFun_QAOA_QAE_figures.ipynb` | Generates the paper's figures from saved computation data |
| `simple_UC_model.ipynb` | Minimal classical unit commitment reference |
| `SED_classical_solution.ipynb` | Classical SED (stochastic economic dispatch) baseline |

### Convergence experiment (script)

```bash
cd QuantumExpectedValueFunctionProject
python experiment_convergence.py
```
Runs DQA convergence sweeps over annealing depth and time steps, producing the data behind Fig. 3 of the paper.

---

## Navigation

- **[Algorithm Overview](algorithm.md)** — How DQA and QAE work step by step
- **[API Reference](api/binary_optimizer.md)** — Full class and function documentation
- **[GPU Porting Guide](gpu_porting.md)** — Three-tier strategy for H100 acceleration
- **[References](references.md)** — Papers and further reading
