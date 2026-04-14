# quantum_stochastic_programming

Quantum computing algorithms for **two-stage stochastic optimization**, with a focus on the Unit Commitment (UC) problem in power systems.  The algorithms combine **Discrete Quantum Annealing (DQA)** with **Quantum Amplitude Estimation (QAE)** to compute expected-value objective functions over a probability distribution of wind-power scenarios, which is the key subroutine needed for first-stage hedged decision-making.

Based on: [arXiv 2402.15029](https://arxiv.org/abs/2402.15029) — *"Quantum algorithms for the two-stage stochastic unit commitment problem"*

---

## Table of Contents

1. [Problem description](#1-problem-description)
2. [Repository structure](#2-repository-structure)
3. [Module reference — `qiskit_impl/`](#3-module-reference--qiskit_impl)
4. [Module reference — `QuantumExpectedValueFunctionProject/`](#4-module-reference--quantumexpectedvaluefunctionproject)
5. [Installation and environment](#5-installation-and-environment)
6. [Quick-start usage](#6-quick-start-usage)
7. [Notebooks](#7-notebooks)
8. [GPU porting strategy (branch `gpuOptim`)](#8-gpu-porting-strategy-branch-gpuoptim)
9. [Fault-tolerant resource estimation](#9-fault-tolerant-resource-estimation)
10. [Development conventions](#10-development-conventions)
11. [References](#11-references)

---

## 1. Problem description

The **Unit Commitment** problem decides which gas generators to commit before uncertain wind power is realised (first-stage decision), then dispatches all generators optimally for each realised scenario (second-stage recourse).

Mathematically:

$$\min_{x,y} \; c_x^\top x + \mathbb{E}_\xi\!\left[\min_r c_r r(\xi) \right] \quad \text{s.t.} \quad \sum_j y_j(\xi) + r(\xi) = d, \quad y_j(\xi) \le \xi_j$$

- **First-stage**: binary gas commitment $x \in \{0,1\}^{n_g}$
- **Second-stage recourse**: real-valued $r(\xi) \ge 0$ to cover unmet demand at cost $c_r$
- **Uncertainty**: wind scenario $\xi \sim p(\xi)$ (encoded as a quantum PDF register)

The quantum algorithm evaluates $\mathbb{E}[C(x,\xi)]$ via QAE in $O(1/\epsilon)$ circuit evaluations rather than the $O(1/\epsilon^2)$ shots needed classically.

---

## 2. Repository structure

```
quantum_stochastic_programming/
│
├── qiskit_impl/                        # Modular Qiskit 2.x implementation (primary)
│   ├── binary_optimizer.py             # Core DQA+QAE solver (BinaryNestedOptimizer)
│   ├── qae.py                          # Standalone QAE circuit builder (QAE_Optimizer)
│   ├── ExpValFun_functions.py          # Pure-function circuit primitives
│   ├── dist_prep.py                    # PDF encoding circuits
│   ├── resource_estimator.py           # Fault-tolerant resource estimation
│   └── qae_example.ipynb              # Example notebook
│
├── QuantumExpectedValueFunctionProject/ # Original implementation (C. Rotello)
│   ├── optimizer_utils.py             # VariableRegister + PowerSystem_1Bus
│   ├── dense_optimizer.py             # Optimizer_Dense — scenario-superposition solver
│   ├── expanded_optimizer.py          # Optimizer_Expanded — scenario-expanded solver
│   ├── binary_optimizer.py            # Legacy BinaryNestedOptimizer
│   ├── experiment_convergence.py      # Experiment runner script
│   ├── test_dense_optimizer.py        # Unit tests for dense_optimizer
│   ├── test_expanded_optimizer.py     # Unit tests for expanded_optimizer
│   ├── test_optimizer_utils.py        # Unit tests for optimizer_utils
│   ├── four_binary_turbines_uniform_distribution_tutorial.ipynb  # Tutorial
│   ├── ExpValFun_QAOA_QAE_computations.ipynb                    # Computation notebook
│   ├── ExpValFun_QAOA_QAE_figures.ipynb                         # Figure generation
│   ├── simple_UC_model.ipynb          # Minimal UC example
│   └── SED_classical_solution.ipynb   # Classical reference solution
│
├── cuda-q_mac_install_notes.md        # CUDA-Q environment notes
├── twoStageStocOptQ.code-workspace    # VS Code multi-root workspace
└── README.md
```

**Two implementations exist** because the code evolved:

| | `qiskit_impl/` | `QuantumExpectedValueFunctionProject/` |
|---|---|---|
| API style | Functional primitives + `args` dicts | Class-based OOP |
| Encoding | Binary only | Binary or Unary |
| QAE integration | QFTGate-based canonical QAE | Multiple annealing solve strategies |
| Qiskit version | 2.x (qiskit_aer separated) | Originally 0.x/1.x (needs import fixes) |
| Status | Active development | Legacy / reference |

---

## 3. Module reference — `qiskit_impl/`

### `binary_optimizer.py`

The main entry point for the DQA+QAE pipeline.

#### `class BernoulliA(QuantumCircuit)`
Wraps a DQA state-preparation circuit $\mathcal{A}$ into the Bernoulli amplitude estimation form.  Used by `execute_qae()`.

| Argument | Type | Description |
|---|---|---|
| `uopt` | `QuantumCircuit` | State-preparation unitary (DQA output) |
| `oracle` | `QuantumCircuit` | Phase-marking oracle |
| `n` | `int` | Number of wind qubits |

#### `class BinaryNestedOptimizer`

The core solver.  Encodes the power system as a quantum circuit and drives the full DQA → QAE pipeline.

```python
bno = BinaryNestedOptimizer(
    gas_costs   = [0.4],          # first-stage generator costs
    wind_costs  = [0.05, 0.10],   # wind turbine recourse cost coefficients
    recourse_cost = 1.0,          # penalty for unmet demand
    pdf         = {(0,0):0.25, (0,1):0.25, (1,0):0.25, (1,1):0.25},
    demand      = 1,              # total wind demand (sum of y_j)
    is_uniform  = False,          # True → use uniform distribution shortcut
)
```

**Circuit-building methods:**

| Method | Returns | Description |
|---|---|---|
| `dicke_state_initialize(weight)` | `QuantumCircuit` | Prepares $\|D_n^k\rangle$ — the Dicke state with `weight` excitations (feasible initial state) |
| `pdf_initialize()` | `QuantumCircuit` | Encodes $p(\xi)$ into the PDF register amplitudes |
| `cost_operator(amplitude, constraint_amplitude, norm)` | `QuantumCircuit` | Phase operator $e^{-i s H_C}$ over all scenarios |
| `cost_scenario_operator(scenario, amplitude, constraint_amplitude, norm)` | `QuantumCircuit` | Single-scenario phase operator |
| `demand_constraint_preserving_mixer(amplitude)` | `QuantumCircuit` | XY-mixer $U_d(\beta)$ that preserves $\sum y_j = k$ (Hamming weight) |
| `adiabatic_evolution_circuit(wind_demand, time, time_steps, norm)` | `QuantumCircuit` | Full DQA circuit $U_\text{DQA}(T)$: Dicke init + $T$ Trotter steps |
| `adiabatic_evolution_scenario_circuit(wind_demand, time, time_steps, scenario, norm)` | `QuantumCircuit` | Single-scenario DQA circuit |
| `implemented_qae(op, oracle, op_inv, oracle_inv, m, norm)` | `QuantumCircuit` | Canonical QAE circuit (Brassard 2002, Fig. 2 of paper) |
| `canonical_qae(Uopt, oracle, m, c, norm)` | `QuantumCircuit` | Convenience wrapper for canonical QAE |

**Oracle methods** (mark the good subspace for QAE):

| Method | Description |
|---|---|
| `exact_oracle(ydemand, norm, inverse)` | Exact phase-kickback oracle |
| `single_oracle_sin_inconstraint(c, norm, inverse)` | $\sin$-encoded in-constraint oracle |
| `single_oracle_asin_inconstraint(norm, inverse)` | $\arcsin$-encoded oracle |
| `grover_operator_sin_inconstraint(power, c, norm)` | Grover operator $Q^p$ for sin oracle |
| `grover_reflections_sin_inconstraint(m, c, norm)` | $m$ Grover reflections |

**Execution methods:**

| Method | Description |
|---|---|
| `execute_optimizer(qc, num_meas)` | Run DQA circuit on `statevector_simulator`, return measurement counts |
| `execute_optimizer_oracle(qc, num_meas)` | Run oracle circuit on `aer_simulator_statevector` |
| `execute_qae(qc, m, num_meas)` | Run canonical QAE circuit with $m$ evaluation qubits, return amplitude estimate |

**Classical utilities:**

| Method | Description |
|---|---|
| `gen_all_wind_outputs()` | Enumerate all feasible wind assignments |
| `wind_scenario_cost(wind_output, scenario, wind_demand)` | Evaluate cost for one assignment + scenario |
| `brute_force_wind_demand_expectation_values()` | Classical brute-force expected value for all demands (reference) |
| `brute_force_energy_surface()` | Full energy landscape (for small instances) |
| `prep_gs(wind_demand)` | Prepare the exact ground-state statevector (for testing) |
| `process_expectation_value_optimizer(wind_demand, counts)` | Convert measurement counts to expected cost |

---

### `qae.py`

Standalone QAE circuit builder, separated from the optimizer logic.

#### `class QAE_Optimizer`

Assembles the canonical QAE circuit from an `args` dict.

```python
args = {
    'n_y'   : 4,           # number of wind qubits
    'm'     : 3,           # number of evaluation (ancilla) qubits
    'pdf'   : pdf_dict,    # scenario probability distribution
    'y_reg' : [0,1,2,3],   # qubit indices for wind register
    'pdf_reg': [4,5,6,7],  # qubit indices for PDF register
    'Theta' : theta,       # QAE target angle
    'w_d'   : wind_demand, # Hamming weight constraint
    'norm'  : 10.0,        # cost normalization
    'uniform': False,      # use uniform PDF shortcut
    'gateset': 'default',
    'cost_operator_circuit'  : cost_op_fn,   # callable -> QuantumCircuit
    'mixer_operator_circuit' : mixer_op_fn,  # callable -> QuantumCircuit
}

qae_opt = QAE_Optimizer(args)
qc = qae_opt.compile_qae_circuit()      # full QAE circuit
qc = qae_opt.implemented_qae(op, oracle, op_inv, oracle_inv)  # lower-level
```

---

### `ExpValFun_functions.py`

Pure functions returning `QuantumCircuit` objects.  No class state — designed for use as callables passed via `args` dicts.

| Function | Signature | Description |
|---|---|---|
| `demand_constraint_preserving_mixer` | `(amplitude, args)` | XY mixer; preserves $\sum y_j$ = `w_d` |
| `cost_operator` | `(amplitude, args)` | Full stochastic cost phase operator |
| `cost_scenario_operator` | `(amplitude, args)` | Single-scenario cost phase operator |
| `single_oracle_sin_inconstraint` | `(args, inverse=False)` | $\sin$-encoded phase oracle |
| `dicke_state_circuit` | `(args)` | Dicke state preparation $\|D_n^k\rangle$ |
| `alternating_operator_ansatz` | `(args)` | Full DQA alternating-operator ansatz for one Trotter step |

---

### `dist_prep.py`

Circuits that encode probability distributions into quantum amplitudes.

| Function | Description |
|---|---|
| `pdf_initialize(args)` | Encode arbitrary discrete PDF $p(\xi)$ into register amplitudes |
| `make_pdf_uniform(n)` | Uniform distribution over $2^n$ scenarios |
| `make_pdf_skewnormal(n)` | Skew-normal distribution, $n$ wind turbines |
| `make_normal_distribution_circuit(args)` | Gaussian amplitude encoding |
| `make_variational_distribution_circuit(args)` | Variational (trainable) amplitude encoding |

---

### `resource_estimator.py`

Fault-tolerant resource estimation using **pytket** (gate counting) and **qualtran** (Beverland et al. surface-code model).

#### `class Resource_Estimator`

```python
args = {
    'num_algo_qubits' : 10,
    'circuit'         : pytket_circuit,
    'num_meas'        : 1000,
    'error_budget'    : 1e-3,
    'architecture'    : 'superconducting',   # or 'ion', 'majorana'
    'optimistic'      : False,
    # Magic state factory params:
    'R'   : 2,     # distillation rounds
    'Q_0' : 1e-3,  # physical qubit error rate
    'P_r' : 1e-6,  # target logical Clifford error
    'F_copies': 1,
}

re = Resource_Estimator(args)
result = re.bev_resource_estimator()
# result keys: num_physical_qubits, num_logical_qubits, T_count, T_depth,
#              clifford_count, rotation_count, wall_clock_time, ...
```

**Module-level helpers:**

| Function | Description |
|---|---|
| `get_circuit_metrics(circuit)` | Extract gate counts (T, Clifford, rotation) from a pytket `Circuit` |
| `get_factory_params(R, Q_0, P_r, d, c, M, tau_d, C, error_budget)` | Compute magic-state factory parameters |
| `qasm_to_clifford_and_t(qc, ...)` | Convert Qiskit `QuantumCircuit` → Clifford+T decomposition via Solovay-Kitaev |

---

## 4. Module reference — `QuantumExpectedValueFunctionProject/`

### `optimizer_utils.py`

Foundation utilities shared by all optimizers.

#### `class VariableRegister`

Stores an integer variable as a qubit register.  Handles binary and unary encodings, and generates Hamiltonian sub-circuits for arithmetic constraints.

```python
reg = VariableRegister(max_value=3, encoding='binary')  # 2-qubit register
```

**Key methods:**

| Method | Description |
|---|---|
| `numberOperator(amplitude)` | Phase operator $e^{i \alpha \hat{n}}$ (diagonal in number basis) |
| `squaredOperator(amplitude)` | Phase operator $e^{i \alpha \hat{n}^2}$ |
| `productOperator(other, amplitude)` | Two-register product $e^{i \alpha \hat{n}_A \hat{n}_B}$ |
| `swapOperator(other, amplitude)` | SWAP-network mixing between two registers |
| `lessThanOperator(other, amplitude)` | Phase if `self < other` |
| `lessThanValue(value, amplitude)` | Phase if `self < value` |
| `getValue(bstr)` | Decode measurement bitstring to integer |
| `setValue(integer)` | Encode integer as circuit preparation |

#### `class PowerSystem_1Bus`

Stores the power-system problem specification.

```python
system = PowerSystem_1Bus(
    gas_costs         = [0.4],
    wind_costs        = [0.05, 0.10],
    decision_levels   = 4,
    undersatisfied_cost = 1.0,
    demand            = 2,
    pdf               = {(0,1): 0.5, (1,0): 0.5},
    normalization     = None,
)
```

**Key methods:**

| Method | Description |
|---|---|
| `cobylaSolve(discrete, gas_values)` | Classical COBYLA solver (reference) |
| `getFirstStageCosts(gas_decisions)` | Evaluate first-stage cost for a gas commitment |
| `plotMeasurementsVExpectedCost(measured_decisions)` | Plot optimizer output vs. classical optimum |
| `normalize(val)` / `unNormalize(val)` | Cost normalization helpers |

---

### `dense_optimizer.py` — `class Optimizer_Dense`

Solves the UC problem by keeping scenarios in **superposition** in the PDF register, measuring gas decisions only.  This is the approach that makes QAE applicable.

```python
from optimizer_utils import PowerSystem_1Bus
from dense_optimizer import Optimizer_Dense

system = PowerSystem_1Bus(...)
opt = Optimizer_Dense(system, encoding='binary')
```

**Annealing solve methods** — all return a `counts` dict `{bitstring: probability}`:

| Method | Schedule | Notes |
|---|---|---|
| `solveAnnealing(time, method, num_meas, penalty)` | Linear, single-phase | QUBO or quadratic penalty; all-to-one transpile |
| `solveAnnealingAlternating(total_time, time_steps, ...)` | Alternating operator | QAOA-style; supports `phase='PEN'/'COST'` and `mixer='X'/'SWAP'` |
| `solveAnnealingDenseStochHam(total_time, t2, t1_steps, t2_steps, ...)` | Two-phase stochastic | Phase 1: annealing; Phase 2: stochastic Hamiltonian |
| `solveAnnealingDenseStochHam2Layers(t1_time, t1_steps, t2_time, t2_steps, ...)` | Two-phase layered | Like above with explicit layer separation |
| `solveAnnealingStochHam(total_time, t_steps, ...)` | Stochastic Hamiltonian | Full stochastic Hamiltonian throughout |
| `solveECAnnealing(total_time, gamma, ...)` | EC (ensemble copy) | Multiple copies with weight $\gamma$ |
| `solveThreePhaseAnnealing(total_time, ...)` | Three-phase | Annealing → PDF projection → re-annealing |
| `solveThreePhaseAnnealing(...)` | Three-phase | Phase 1: DQA; Phase 2: fixed; Phase 3: sampling with PDF reinit |
| `solveTwoPhaseAnnealing(total_time, time_2, ...)` | Two-phase sampling | Phase 1: build state; Phase 2: measure-and-reinit loop |

**Circuit-building helpers:**

| Method | Description |
|---|---|
| `priceOperator(amp)` | Gas cost phase operator |
| `scenarioOperator(amp)` | Scenario (wind) cost phase operator |
| `penaltyOperator(amp, penalty)` | Demand-constraint penalty |
| `initializePDF(inverse)` | Prepare (or un-prepare) the PDF register |
| `dickeState()` | Prepare Dicke state on decision qubits |
| `setVariables(vars, vals)` | Initialize a register to specific values |

**Post-processing:**

| Method | Description |
|---|---|
| `calcCostPerScenario(counts)` | Compute expected cost per scenario from measurements |
| `calcGasExpectationValue(counts)` | Expected value of gas commitment |
| `getGasCounts(counts)` | Marginalise over scenarios, return gas-only counts |
| `getDecision(gas_decisions)` | Extract optimal gas commitment from measurement histogram |

---

### `expanded_optimizer.py` — `class Optimizer_Expanded`

Alternative formulation: each scenario gets its own set of wind+slack qubits in the register (no superposition over scenarios).  Simpler to understand, but exponentially larger register.

```python
opt = Optimizer_Expanded(system, encoding='binary')
```

---

### `experiment_convergence.py`

Script for convergence experiments.  Run directly:

```bash
python experiment_convergence.py --n 4 --m 3
```

Generates QAE convergence data and saves to pickle for downstream analysis notebooks.

---

## 5. Installation and environment

### NREL HPC (recommended — GPU-enabled)

```bash
module load qiskit/aer-gpu
# Provides: Python 3.11, Qiskit 2.2.3, qiskit-aer-gpu 0.15.1 (CUDA 12.4, H100),
#           qiskit-algorithms 0.4.0, matplotlib 3.10.8
```

### Local / portable install

```bash
pip install qiskit>=2.0 qiskit-aer qiskit-algorithms matplotlib scipy numpy
# For resource estimation:
pip install pytket qualtran
```

> **Note**: `QuantumExpectedValueFunctionProject/` code was written for an older Qiskit API.  Before running it, apply these import fixes:
> ```python
> # OLD                                           NEW
> from qiskit import Aer                    →   from qiskit_aer import Aer
> from qiskit.tools.visualization import … →   from qiskit.visualization import …
> from qiskit.extensions import Initialize  →   from qiskit.circuit.library import Initialize
> ```

---

## 6. Quick-start usage

### Minimal end-to-end example (`qiskit_impl`)

```python
import numpy as np
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer

# Define a 2-turbine problem with uniform scenarios
n = 2
pdf = {(0,0):0.25, (0,1):0.25, (1,0):0.25, (1,1):0.25}
cy  = np.linspace(0.01, 0.10, n)    # wind cost coefficients
cr  = 1.0                            # recourse cost
cx  = [0.4]                          # dummy gas cost

bno = BinaryNestedOptimizer(cx, cy, cr, pdf, demand=1, is_uniform=True)

# 1. Build the DQA state-preparation circuit
qc = bno.adiabatic_evolution_circuit(wind_demand=1, time=100, time_steps=4, norm=10)

# 2. Execute the DQA circuit and extract expectation value
counts = bno.execute_optimizer(qc, num_meas=8192)
exp_val = bno.process_expectation_value_optimizer(wind_demand=1, counts=counts)
print(f"Expected cost (DQA):           {exp_val:.4f}")

# 3. Classical brute-force reference
ev_true = bno.brute_force_wind_demand_expectation_values()
print(f"Expected cost (brute force):   {ev_true[1]:.4f}")

# 4. Full QAE pipeline
oracle  = bno.single_oracle_sin_inconstraint(c=0.5, norm=10)
orc_inv = bno.single_oracle_sin_inconstraint(c=0.5, norm=10, inverse=True)
uopt    = bno.adiabatic_evolution_circuit(wind_demand=1, time=100, time_steps=4, norm=10)
qae_qc  = bno.canonical_qae(uopt, oracle, m=3, c=0.5, norm=10)
amplitude = bno.execute_qae(qae_qc, m=3, num_meas=8192)
print(f"QAE amplitude estimate:        {amplitude:.4f}")
```

### Dense optimizer (`QuantumExpectedValueFunctionProject`)

```python
import sys
sys.path.insert(0, 'QuantumExpectedValueFunctionProject')
from optimizer_utils import PowerSystem_1Bus
from dense_optimizer import Optimizer_Dense

system = PowerSystem_1Bus(
    gas_costs=[0.4], wind_costs=[0.05, 0.10],
    decision_levels=4, undersatisfied_cost=1.0,
    demand=2,
    pdf={(0,1):0.5, (1,0):0.5},
)

opt = Optimizer_Dense(system, encoding='binary')
counts = opt.solveAnnealingAlternating(
    total_time=10, time_steps=5, penalty=1, num_meas=2000,
    init_cond='XGS', mixer='X', phase='PEN',
)
decision = opt.getDecision(opt.getGasCounts(counts))
print("Gas commitment decision:", decision)
```

---

## 7. Notebooks

| Notebook | Location | Purpose |
|---|---|---|
| `qae_example.ipynb` | `qiskit_impl/` | Step-by-step walkthrough of DQA + canonical QAE |
| `four_binary_turbines_uniform_distribution_tutorial.ipynb` | `QuantumExpectedValueFunctionProject/` | Tutorial: 4 turbines, uniform PDF |
| `ExpValFun_QAOA_QAE_computations.ipynb` | `QuantumExpectedValueFunctionProject/` | Full computation pipeline for QAOA+QAE |
| `ExpValFun_QAOA_QAE_figures.ipynb` | `QuantumExpectedValueFunctionProject/` | Publication-quality figures |
| `simple_UC_model.ipynb` | `QuantumExpectedValueFunctionProject/` | Simplest possible UC model |
| `SED_classical_solution.ipynb` | `QuantumExpectedValueFunctionProject/` | Classical stochastic ED reference |

On NREL HPC:
```bash
module load qiskit/aer-gpu
jupyter lab
```

---

## 8. GPU porting strategy (branch `gpuOptim`)

Branch `gpuOptim` contains detailed comments marking every location that needs changing to run on the H100 GPU.  **No code is changed** — only comments.

### TIER 1 — Drop-in backend swap (immediate, ~1 line per location)

Replace every `Aer.get_backend(...)` call with:

```python
from qiskit_aer import AerSimulator
simulator = AerSimulator(method='statevector', device='GPU')
```

Marked with `# GPU-SWAP TIER 1` in:
- `qiskit_impl/binary_optimizer.py` — `execute_optimizer()`, `execute_optimizer_oracle()`, `execute_qae()`
- `QuantumExpectedValueFunctionProject/dense_optimizer.py` — all 8 `solveAnnealing*()` methods

### TIER 2 — Transpile once + `ParameterVector` binding

Inside DQA time-step loops, sub-circuits are currently rebuilt with numeric angles each iteration.  The fix — demonstrated in `bayesianQC/optimize_10epoch_performance.py`:

```python
# Build symbolic template ONCE
from qiskit.circuit import Parameter
s = Parameter('s')
cost_tmpl = cost_operator_symbolic(s, ...)     # symbolic angles

# Transpile ONCE to GPU target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=1, backend=gpu_simulator)
cost_t = pm.run(cost_tmpl)

# Inside loop: only bind numbers (cheap)
for t in range(time_steps):
    s_val = (dt * t) / time
    qc.append(cost_t.assign_parameters({s: s_val}), qubits)
```

Marked with `# PARAM-TRANSPILE` in `binary_optimizer.py` and `dense_optimizer.py`.

### TIER 3 — cuStateVec (>20 qubits)

```python
simulator = AerSimulator(
    method='statevector', device='GPU',
    cuStateVec_enable=True,           # NVIDIA cuStateVec via cuquantum-cu12
    blocking_enable=True,             # auto-chunk if VRAM insufficient
    blocking_qubits=23,
)
```

---

## 9. Fault-tolerant resource estimation

```python
from pytket import Circuit as TKCircuit
from qiskit_impl.resource_estimator import Resource_Estimator, qasm_to_clifford_and_t
from qiskit import QuantumCircuit

# Build the circuit you want to estimate
qc = QuantumCircuit(10)
# ... add gates ...

# Convert to pytket (for gate counting)
from qiskit.qasm2 import dumps
import pytket.extensions.qiskit as tkqk
tk_circ = tkqk.qiskit_to_tk(qc)

args = {
    'num_algo_qubits' : qc.num_qubits,
    'circuit'         : tk_circ,
    'num_meas'        : 1000,
    'error_budget'    : 1e-3,
    'architecture'    : 'superconducting',
    'optimistic'      : False,
    'R': 2, 'Q_0': 1e-3, 'P_r': 1e-6, 'F_copies': 1,
}
result = Resource_Estimator(args).bev_resource_estimator()
print(result)
```

Based on the Beverland et al. (2022) surface-code resource model ([arXiv:2211.07629](https://arxiv.org/abs/2211.07629)).

---

## 10. Development conventions

- Each `file.py` has a corresponding `test_file.py`.  Run tests with `python test_file.py`.
- In-development functions live as `_TEST_functionname()` inside the source file, called from `main()`.  They are moved to `test_file.py` when the feature is complete.
- Experiments go in Jupyter notebooks that import from the source modules — not in source modules themselves.
- Branch `nilesh/comments` — adds explanatory comments and Qiskit 2.x compatibility fixes.
- Branch `gpuOptim` — adds GPU porting strategy comments (no code changes).

---

## 11. References

1. Sævarsson et al., *"Quantum algorithms for the two-stage stochastic unit commitment problem"*, arXiv:2402.15029 (2024)
2. Brassard et al., *"Quantum amplitude amplification and estimation"*, AMS Contemporary Mathematics 305 (2002)
3. Beverland et al., *"Assessing requirements to scale to practical quantum advantage"*, arXiv:2211.07629 (2022)
4. Farhi et al., *"A Quantum Approximate Optimization Algorithm"*, arXiv:1411.4028 (2014)
5. Bartschi & Eidenbenz, *"Deterministic Preparation of Dicke States"*, FCT 2019
