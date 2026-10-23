# Quantum Stochastic Programming

This repository contains research implementations of quantum algorithms for
two-stage stochastic optimization, with a focus on unit commitment under
uncertain wind generation. The active Python implementation combines Discrete
Quantum Annealing (DQA) with Quantum Amplitude Estimation (QAE).

The work is based on
[“Quantum algorithms for the two-stage stochastic unit commitment problem”](https://arxiv.org/abs/2402.15029).

## Start here

- **Run an example:** open [`qiskit_impl/qae_example.ipynb`](qiskit_impl/qae_example.ipynb).
- **Read the algorithm overview:** [`docs/algorithm.md`](docs/algorithm.md).
- **Browse the API docs:** start at [`docs/index.md`](docs/index.md).
- **Use the legacy implementation:** see [`legacy/QuantumExpectedValueFunctionProject/`](legacy/QuantumExpectedValueFunctionProject/README.md); it targets older Qiskit APIs and is retained as reference code.

## Repository layout

| Path | Contents |
|---|---|
| [`qiskit_impl/`](qiskit_impl/) | Actively maintained Qiskit implementation, tests, and notebooks |
| [`cudaq_impl/`](cudaq_impl/) | CUDA-Q implementation and tests |
| [`classical/`](classical/) | Quantum-independent exact reference costs and brute-force solvers |
| [`experiments/`](experiments/) | Qiskit and CUDA-Q sweeps, benchmark runners, plotting tools, HPC launchers, and notebooks |
| [`legacy/QuantumExpectedValueFunctionProject/`](legacy/QuantumExpectedValueFunctionProject/) | Older class-based implementation and original notebooks |
| [`docs/`](docs/) | MkDocs documentation, algorithm guide, and API reference |
| [`publications/`](publications/) | Publication figures, poster sources, and study reports with their results |
| [`publications/noise_study/`](publications/noise_study/) | Noise-study report, result data, and generated figure |
| [`publications/parallelisation/`](publications/parallelisation/) | Parallelisation report, benchmark data, and generated figures |
| [`publications/siam_poster_example/`](publications/siam_poster_example/) | SIAM example code and its Julia project environment |

The active and legacy implementations have different APIs. New Python
development should generally use `qiskit_impl/`.

## Installation

Use Python 3.11 or newer and install the core Qiskit dependencies:

```bash
python -m pip install "qiskit>=2.0" qiskit-aer numpy scipy matplotlib
```

The Qiskit CPU simulator is suitable for small examples. Aer GPU execution
requires a compatible NVIDIA GPU, CUDA installation, and GPU-enabled Aer build;
the repository's simulator helper falls back to CPU when GPU execution is
unavailable.

CUDA-Q support is optional and requires a CUDA-Q installation compatible with
your platform and target. For GPU execution, configure CUDA-Q for the available
NVIDIA hardware. Tests select the `qpp-cpu` target for portable CPU validation.

For the SIAM example only, install Julia and activate the project from its
directory:

```bash
cd publications/siam_poster_example
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

This small example compares an exact classical reference with a sampled DQA
estimate. `wind_demand` is the number of wind units required after the
first-stage gas commitment.

```python
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer

n_wind = 2
pdf = {
    (0, 0): 0.25,
    (0, 1): 0.25,
    (1, 0): 0.25,
    (1, 1): 0.25,
}

optimizer = BinaryNestedOptimizer(
    gas_costs=[0.4],
    wind_costs=[0.05, 0.10],
    recourse_cost=1.0,
    pdf=pdf,
    demand=n_wind,
    is_uniform=True,
)

# Classical expected costs for each wind-demand value.
classical_values = optimizer.brute_force_wind_demand_expectation_values()

# DQA estimate for wind_demand=1 (one unit of demand is met by gas).
circuit = optimizer.adiabatic_evolution_circuit(
    wind_demand=1,
    time=100,
    time_steps=4,
    norm=10,
)
counts = optimizer.execute_optimizer(circuit, num_meas=4096)
dqa_estimate = optimizer.process_expectation_value_optimizer(1, counts)

print("Classical expected costs:", classical_values)
print("DQA estimate at wind_demand=1:", dqa_estimate)
```

## Tests

Run the standalone classical reference tests from the repository root:

```bash
python -m pytest classical/test_reference.py -v
```

The classical package uses only the Python standard library; its tests require
pytest but no quantum SDKs. Run the Qiskit and classical-compatibility suites with:

```bash
python -m pytest qiskit_impl/test_classical_reference.py qiskit_impl/test_circuit_builders.py qiskit_impl/test_gpu_optimizer.py -v
```

The circuit-builder tests use exact statevectors and operators to characterize
both Qiskit APIs before consolidation, including their different oracle
encodings. Two nonuniform-PDF optimizer cases are strict expected failures:
the current initializer contains reset instructions that cannot be converted
to a gate. These tests must be updated when that limitation is fixed.

The CUDA-Q tests are optional and run when CUDA-Q is installed:

```bash
python -m pytest cudaq_impl/test_cudaq_impl.py -v
```

The legacy tests target older Qiskit interfaces; failures under current Qiskit
versions do not indicate a regression in the active implementation.

## Classical references without quantum libraries

```python
from classical import wind_demand_costs, first_stage_objective

pdf = {(0, 0): 0.1, (1, 0): 0.2, (0, 1): 0.3, (1, 1): 0.4}
costs = wind_demand_costs([1.0, 3.0], 10.0, pdf, demand=2)
objective = first_stage_objective(4.0, [1.0, 3.0], 10.0, pdf, demand=2)
# costs: [0.0, 2.5, 9.7]; objective: {2: 8.0, 1: 6.5, 0: 9.7}
```

These references optimize the turbine decision separately for each wind
scenario, then take the expectation. They are not costs of a single fixed
decision or of a sampled quantum distribution. The optimizer's existing
classical methods delegate to these functions.

Wind demands exceeding the turbine count now return infinite cost (and an
empty optimal decision), rather than the old artificial finite sentinel.
Feasible costs are no longer clipped by that sentinel. Ties still choose the
lexicographically first decision. See the [classical API](docs/api/classical.md).

## Experiments and results

Experiment scripts are in [`experiments/`](experiments/). The implementation
code is organized separately in [`qiskit_impl/`](qiskit_impl/) and
[`cudaq_impl/`](cudaq_impl/). The
Qiskit and CUDA-Q sweeps accept command-line options; by default they print a
summary without saving files or opening a plot:

```bash
python experiments/qiskit_sweep.py --n-y 4 6 --backend exact
python experiments/cuda_q_sweep.py --n-y 4 6 --target qpp-cpu --shots 1024
```

Use `--save-results results.csv` and/or `--save-plot results.png` to request
files, or `--show-plot` to display a plot. See `--help` for options. Some
experiments require CUDA-Q, GPU, or HPC configuration and can be computationally
intensive. CSV and PNG files are ignored by Git by default; use `git add -f
<file>` if you intentionally want to version a generated result.

Publication results and figures are stored alongside their reports under
[`publications/`](publications/). Local virtual environments, caches, and
generated MkDocs output are not project dependencies.

## Documentation site

The docs use MkDocs Material and mkdocstrings. To build the site locally:

```bash
python -m pip install mkdocs mkdocs-material "mkdocstrings[python]"
python -m mkdocs build
```

Useful pages:

- [Algorithm overview](docs/algorithm.md)
- [GPU porting guide](docs/gpu_porting.md)
- [References](docs/references.md)
- [API: Qiskit optimizer](docs/api/binary_optimizer.md)
- [API: CUDA-Q implementation](docs/api/cudaq_impl.md)

## Citation

If you use this work, cite the associated paper:

> Sævarsson et al., “Quantum algorithms for the two-stage stochastic unit
> commitment problem,” 2024. [arXiv:2402.15029](https://arxiv.org/abs/2402.15029).

Additional references are listed in [`docs/references.md`](docs/references.md).
