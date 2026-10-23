# Copilot Instructions: Quantum Stochastic Programming

These instructions apply only to the `quantum_stochastic_programming` repository.
Treat that directory as the repository root; do not assume its parent workspace
directories are part of the project.

## Project Overview

This research project implements quantum algorithms for two-stage stochastic
optimization, especially unit commitment in power systems. Its main algorithm is
Discrete Quantum Annealing (DQA) combined with Quantum Amplitude Estimation
(QAE), based on [arXiv 2402.15029](https://arxiv.org/abs/2402.15029).

## Repository Layout

- `qiskit_impl/`: actively maintained Qiskit implementation, tests, and notebooks.
- `cudaq_impl/`: CUDA-Q implementation and tests.
- `experiments/`: Qiskit and CUDA-Q sweeps, benchmark runners, plotting tools,
  HPC launchers, and the noise-study notebook.
- `legacy/QuantumExpectedValueFunctionProject/`: legacy class-based implementation;
  treat it as reference code because it targets older Qiskit APIs.
- Run experiment scripts from the repository root; results are written only
  when explicitly requested.
- `publications/`: poster source, PDF, and publication figures.
- `publications/siam_poster_example/`: SIAM poster example and supporting code.
- `docs/`: MkDocs documentation and API reference.

Use repository-relative paths in scripts and documentation. Keep generated
outputs and machine-specific environment files out of version control unless
they are intentional reproducibility inputs or publication assets.

## Environment and Dependencies

The primary implementation uses Python, Qiskit, Qiskit Aer, NumPy, SciPy, and
Matplotlib. CUDA-Q and GPU support are optional and depend on the target
environment. The Julia environment under
`publications/siam_poster_example/Project.toml` is for the SIAM poster's
classical optimization example, not the primary Python implementation.

Use the repository's configured Python environment. Do not install dependencies
or alter environment manifests unless needed for the requested change.

## Tests and Validation

From the repository root, run the primary CPU/GPU-fallback suite with:

```bash
python -m pytest qiskit_impl/test_gpu_optimizer.py -v
```

The CUDA-Q tests are in `cudaq_impl/test_cudaq_impl.py`; run them only when the
configured environment supports CUDA-Q. `experiments/run_mgpu_smoke.py` is an
HPC/GPU smoke-test script, not a pytest test module.

The legacy tests may fail collection under current Qiskit versions because the
legacy implementation imports APIs removed from Qiskit. Do not treat those
tests as validation of the active implementation.

For documentation changes, update relevant pages under `docs/` and keep
`mkdocs.yml` navigation and API references consistent. Build locally with
`mkdocs build` when MkDocs dependencies are available.

## Development Conventions

- Follow nearby code patterns; use snake_case for Python names and four-space
  indentation.
- Document public Python APIs with Google-style docstrings.
- Keep changes focused and preserve existing behavior unless the request calls
  for a behavior change.
- When moving files, update imports, notebooks, scripts, documentation, and
  workflows that refer to their old paths.
