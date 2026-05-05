# %% [markdown]
# # QAE Example — Multi-Backend
# 
# Copy of `qae_example.ipynb` extended with a **backend toggle**:
# 
# | Backend | Description |
# |---|---|
# | **Qiskit CPU** | `Statevector.evolve()` — exact, CPU-only |
# | **Aer GPU** | `AerSimulator(device='GPU')` — H100 accelerated via cuStateVec |
# | **CUDA-Q** | `@cudaq.kernel` + `cudaq.sample()` — native GPU quantum kernels |
# 
# **How to use:** run cell 2 to pick your backend, then run all remaining cells — each execution cell dispatches automatically to the selected backend.

# %%
# ── BACKEND TOGGLE ─────────────────────────────────────────────────────────────
# Change this variable to switch the entire notebook's execution path.
# Options: 'cpu' | 'aer-gpu' | 'cuda-q'
BACKEND = 'cuda-q'

# ── Optional: ipywidgets interactive toggle (works in JupyterLab) ───────────────
try:
    import ipywidgets as widgets
    from IPython.display import display

    toggle = widgets.ToggleButtons(
        options=[('Qiskit CPU', 'cpu'), ('Aer GPU', 'aer-gpu'), ('CUDA-Q', 'cuda-q')],
        description='Backend:',
        value=BACKEND,
        style={'description_width': 'initial'},
        button_style='info',
    )

    def _on_toggle(change):
        global BACKEND
        BACKEND = change['new']
        print(f'Backend set to: {BACKEND}')

    toggle.observe(_on_toggle, names='value')
    display(toggle)
    print("⬆  Use the toggle above, then re-run the cells below.")
except ImportError:
    print(f"ipywidgets not installed — using BACKEND = '{BACKEND}'. Edit the variable above to change.")

print(f"Active backend: {BACKEND}")


# %%
# ── IMPORTS ────────────────────────────────────────────────────────────────────
import os, sys, math
import numpy as np
import matplotlib.pyplot as plt

# Add the qiskit/aer-gpu venv site-packages so the Jupyter kernel (which starts
# without 'module load qiskit') can find qiskit, qiskit_aer, cudaq, etc.
_QISKIT_SP = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qiskit_impl')
# _QISKIT_IMPL = '/projects/hpcapps/nsawant/quantum_stochastic_programming/qiskit_impl'
# _QISKIT_IMPL = '/projects/msoc/jmaack/quantum_stochastic_programming/qiskit_impl'

for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

# Always available: Qiskit core + problem helpers
from qae import *
import ExpValFun_functions as exp
from binary_optimizer import BinaryNestedOptimizer, _get_simulator
from qiskit.quantum_info import Statevector

# Aer (needed for cpu and aer-gpu paths)
try:
    from qiskit_aer import AerSimulator
    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False
    print("WARNING: qiskit_aer not found — 'cpu' and 'aer-gpu' backends unavailable.")

# CUDA-Q (needed only for cuda-q path)
try:
    import cudaq
    from cudaq_impl import CudaqQAEOptimizer
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    print("INFO: cudaq not found — 'cuda-q' backend unavailable.")

# %load_ext autoreload
# %autoreload 2
print(f"Imports OK  |  cwd={os.getcwd()}")
print(f"AER={_AER_AVAILABLE}  CUDAQ={_CUDAQ_AVAILABLE}")

# %%
# ── BACKEND INITIALIZATION ─────────────────────────────────────────────────────
if BACKEND == 'cpu':
    if not _AER_AVAILABLE:
        raise RuntimeError("qiskit_aer required for 'cpu' backend.")
    simulator = AerSimulator(method='statevector', device='CPU')
    print(f"[cpu] AerSimulator(method='statevector', device='CPU') ready.")

elif BACKEND == 'aer-gpu':
    if not _AER_AVAILABLE:
        raise RuntimeError("qiskit_aer required for 'aer-gpu' backend.")
    try:
        simulator = AerSimulator(method='statevector', device='GPU')
        simulator.available_devices()  # raises if no GPU found
        print(f"[aer-gpu] AerSimulator(method='statevector', device='GPU') ready.")
    except Exception as e:
        print(f"WARNING: GPU not available ({e}), falling back to CPU.")
        BACKEND = 'cpu'
        simulator = AerSimulator(method='statevector', device='CPU')

elif BACKEND == 'cuda-q':
    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq required for 'cuda-q' backend.")
    try:
        cudaq.set_target('nvidia')
        print(f"[cuda-q] cudaq target set to 'nvidia' (cuStateVec).")
    except Exception as e:
        print(f"WARNING: nvidia target unavailable ({e}), falling back to qpp-cpu.")
        cudaq.set_target('qpp-cpu')
        print(f"[cuda-q] cudaq target set to 'qpp-cpu'.")
    simulator = None  # CUDA-Q uses its own execution model

else:
    raise ValueError(f"Unknown BACKEND '{BACKEND}'. Choose 'cpu', 'aer-gpu', or 'cuda-q'.")

print(f"\nActive backend: {BACKEND}")

# %% [markdown]
# ## Problem Setup
# 
# 4 wind turbines, 1 gas generator, uniform wind scenario distribution (identical for all backends).

# %%
# ── PROBLEM PARAMETERS (shared by all backends) ────────────────────────────────
n_y  = 6         # wind turbines
# n_xi = 4        # xi register qubits (= n_y for uniform pdf)
n_xi = n_y      # n_xi == n_y pretty much always...
n_x  = 1        # gas generators
d    = n_y      # total demand (must equal n_y for the problem to be well-posed)

c_x  = [3.]
# c_y  = [0.4, 0.5, 0.7, 1.0]
c_y = np.linspace(0.1, 1.0, n_y)
c_r  = 10.0

x0 = [2] # First stage value to compute expectation for

# -------- CIRCUIT PARAMETERS -------- #
N_TIME_STEPS = 100
N_SHOTS      = 2**12
USE_COBYLA   = True   # True: optimise DQA angles via COBYLA; False: use linear ramp
# ------------------------------------ #

assert(len(c_y) == n_y)
assert(len(c_x) == n_x)
assert(len(x0) == n_x)

# Uniform wind scenario distribution
pdf = {tuple([int(v) for v in ('{0:0'+str(n_y)+'b}').format(i)]) : 1/2**n_y
       for i in range(2**n_y)}

# DQA annealing schedule
w_d       = int(d - sum(x0))  # wind demand = d - x
cost_norm = w_d * c_r / n_y   # DQA cost operator normalisation (per-turbine upper bound)
m         = 5                 # QPE readout qubits for QAE (5 gives Δφ ≈ norm/32)
norm      = w_d * c_r         # QAE oracle amplitude normalisation -- must be >= max possible cost
                              # max cost = w_d shortfall × c_r when all turbines produce nothing

timesteps = n_y               # number of DQA timesteps (scales with problem size)

print(f"n_y={n_y}, w_d={w_d}, c_y={c_y}, c_r={c_r}")

# %% [markdown]
# ## DQA Angle Optimisation (COBYLA)
#
# Use COBYLA to minimise $\phi$ over the $2p$ DQA angles $(\gamma_t, \beta_t)$.
# Optimisation always uses the exact CPU statevector path regardless of BACKEND.

# %%
# ── COBYLA OPTIMISATION OF DQA ANGLES ─────────────────────────────────────────
from scipy.optimize import minimize

# bno is needed by both COBYLA and the classical section below
bno = BinaryNestedOptimizer(c_x, c_y, c_r, pdf, n_y, is_uniform=True)

# For cuda-q, create the optimizer now so COBYLA uses the same kernel
if BACKEND == 'cuda-q':
    cudaq_opt = CudaqQAEOptimizer(c_x=c_x, c_y=c_y, c_r=c_r,
                                  n_y=n_y, w_d=w_d, cost_norm=cost_norm)

def _dqa_expected_value(theta_flat: list[float]) -> float:
    """Evaluate DQA expected value for COBYLA.
    cuda-q: uses exact statevector via the cudaq kernel (no angle mismatch).
    cpu/aer-gpu: uses Qiskit exact CPU statevector.
    """
    if BACKEND == 'cuda-q':
        return cudaq_opt.estimate_expected_value_sv(list(theta_flat), w_d)
    # cpu / aer-gpu path
    y_reg   = list(range(n_y))
    pdf_reg = list(range(n_y, 2*n_y))
    args = {
        'n_y': n_y, 'n_x': n_x, 'n_xi': n_xi,
        'c_x': c_x, 'c_y': c_y, 'c_r': c_r, 'pdf': pdf,
        'y_reg': y_reg, 'pdf_reg': pdf_reg,
        'Theta': list(theta_flat), 'w_d': w_d, 'cost_norm': cost_norm,
        'uniform': True,
        'cost_operator_circuit':   exp.cost_operator,
        'mixer_operator_circuit':  exp.demand_constraint_preserving_mixer,
        'initial_state_circuit':   exp.dicke_state_circuit,
        'pdf_circuit':             pdf_initialize,
    }
    circ = exp.alternating_operator_ansatz(args)
    sv   = Statevector.from_label('0' * (n_y + n_xi)).evolve(circ)
    counts = sv.probabilities_dict()
    return bno.process_expectation_value_optimizer(w_d, counts)

# Initial Theta: linear ramp (warm start for COBYLA)
theta0 = []
for t in range(timesteps):
    theta0.append(float(t / timesteps))
    theta0.append((1 - float(t / timesteps)) / math.pi)

import time as _time

if USE_COBYLA:
    print(f"Optimising {len(theta0)} DQA angles with COBYLA ({timesteps} timesteps) via {BACKEND}…")
    _t_opt = _time.perf_counter()
    result = minimize(
        _dqa_expected_value,
        theta0,
        method='COBYLA',
        options={'maxiter': 500, 'rhobeg': 0.5, 'disp': False},
    )
    Theta = list(result.x)
    _elapsed_opt = _time.perf_counter() - _t_opt
    print(f"COBYLA done in {_elapsed_opt:.1f}s | {result.nfev} evals | "
          f"phi = {result.fun:.4f} (success={result.success})")
else:
    Theta = theta0
    print(f"COBYLA disabled — using linear ramp ({len(Theta)} angles)")

print(f"Theta ({len(Theta)} angles): {[round(v,3) for v in Theta]}")

# %% [markdown]
# ## Classical Baseline
# 
# Brute-force expected value function $\phi(x) = \mathbb{E}_\xi[\min_y Q(y,\xi)]$ for all $x$. This is the ground truth for all backends.

# %%
# ── CLASSICAL SOLUTION (same for all backends) ─────────────────────────────────
exp_vals        = bno.brute_force_wind_demand_expectation_values()
bruteforce_vals = [bno.gas_costs[0]*x + exp_vals[n_y-x] for x in range(n_y+1)]

print("Classical phi(wind_demand):", [round(v, 3) for v in exp_vals])
print("Classical o(x):", [round(v, 3) for v in bruteforce_vals])

# ── INITIAL STATE ENERGY (diagnostic) ─────────────────────────────────────
# Before any DQA evolution, the state is Dicke|D_n^k⟩ ⊗ uniform|xi⟩.
# The expected cost of this INITIAL state upper-bounds the DQA convergence:
#   E_initial = sum(c_y)/n_y + c_r
# For DQA to be useful the result must be substantially below this.
# Convergence toward phi_classical requires O(n_y^2) or more time steps;
# for n_y=10 with N_TIME_STEPS=100, expect only partial convergence.
c_y_arr = np.array(c_y)
initial_phi = float(c_y_arr.sum() / n_y + c_r)
print(f"Initial state phi (Dicke ⊗ uniform xi): {initial_phi:.4f}")
print(f"Classical optimum phi(w_d={w_d}):         {exp_vals[w_d]:.4f}")
print(f"(DQA should reduce phi from {initial_phi:.2f} toward {exp_vals[w_d]:.4f}; "
      f"needs ~{int(n_y**2)} time steps for n_y={n_y})")

plt.plot(bruteforce_vals, '-o', label='o(x) classical')
plt.xlabel(r'$x$'); plt.ylabel('Cost'); plt.title('Classical objective function')
plt.legend(); plt.show()

# %% [markdown]
# ## DQA Ansatz Execution
# 
# Run the Digitized Quantum Annealing (DQA) circuit to sample the optimal turbine assignment distribution for wind demand $w_d = 2$. Result: `dqa_counts` — a `{bitstring: probability}` dict.

# %%
# ── DQA EXECUTION DISPATCH ─────────────────────────────────────────────────────
import time

# ── Qiskit circuit shared by 'cpu' and 'aer-gpu' ────────────────────────────
if BACKEND in ('cpu', 'aer-gpu'):
    y_reg   = list(range(n_y))
    pdf_reg = list(range(n_y, 2*n_y))

    args_dqa = {
        'n_y': n_y, 'n_x': n_x, 'n_xi': n_xi,
        'c_x': c_x, 'c_y': c_y, 'c_r': c_r, 'pdf': pdf,
        'y_reg': y_reg, 'pdf_reg': pdf_reg,
        'Theta': Theta, 'w_d': w_d, 'cost_norm': cost_norm,
        'uniform': True,
        'cost_operator_circuit':   exp.cost_operator,
        'mixer_operator_circuit':  exp.demand_constraint_preserving_mixer,
        'initial_state_circuit':   exp.dicke_state_circuit,
        'pdf_circuit':             pdf_initialize,
    }
    dqa_circuit = exp.alternating_operator_ansatz(args_dqa)
    print(f"DQA circuit: {dqa_circuit.num_qubits} qubits, depth={dqa_circuit.depth()}")

# ── Execute ──────────────────────────────────────────────────────────────────
t0 = time.perf_counter()

if BACKEND == 'cpu':
    # Exact statevector via Qiskit quantum_info
    sv = Statevector.from_label('0' * (n_y + n_xi))
    sv = sv.evolve(dqa_circuit)
    dqa_counts = sv.probabilities_dict()
    print(f"[cpu] Statevector exact — {len(dqa_counts)} bitstrings")

elif BACKEND == 'aer-gpu':
    # GPU-accelerated via AerSimulator (falls back to CPU if no GPU)
    dqa_counts = bno.execute_optimizer(dqa_circuit.copy())
    print(f"[aer-gpu] AerSimulator statevector — {len(dqa_counts)} bitstrings")

elif BACKEND == 'cuda-q':
    # cudaq_opt already created above (before COBYLA) — reuse it
    dqa_counts = cudaq_opt.sample_ansatz(Theta, shots=N_SHOTS)
    print(f"[cuda-q] cudaq.sample — {len(dqa_counts)} bitstrings")

elapsed = time.perf_counter() - t0
print(f"Wall time: {elapsed*1000:.1f} ms")

# %%
# ── DQA EXPECTED VALUE (post-processing) ────────────────────────────────────
if BACKEND == 'cuda-q':
    # CUDA-Q bitstrings use qubit 0 = leftmost, so bstr[:n_y] = y register
    # and bstr[n_y:] = xi register in natural (un-reversed) order.
    # bno.process_expectation_value_optimizer expects Qiskit convention
    # (qubit 0 = rightmost), so we post-process directly here.
    expectation = 0.0
    for bstr, prob in dqa_counts.items():
        y_bits  = [int(b) for b in bstr[:n_y]]
        xi_bits = [int(b) for b in bstr[n_y:]]
        true_output = np.array(y_bits) * np.array(xi_bits)
        op_cost = float(np.dot(c_y, true_output))
        shortfall = max(0, w_d - int(true_output.sum()))
        expectation += (op_cost + shortfall * c_r) * prob
    dqa_phi = expectation
else:
    dqa_phi = bno.process_expectation_value_optimizer(w_d, dqa_counts)
classical_phi = exp_vals[w_d]

print(f"DQA  phi(w_d={w_d}) [{BACKEND}] : {dqa_phi:.4f}")
print(f"Classical phi(w_d={w_d})         : {classical_phi:.4f}")
print(f"Relative error                   : {abs(dqa_phi - classical_phi)/classical_phi*100:.1f}%")

# %% [markdown]
# ## Quantum Amplitude Estimation (QAE)
# 
# Use $m=4$ QPE qubits and oracle $\mathcal{F}_{\sin}$ to estimate $\phi(x)$ with error $O(1/2^m)$.
# The post-processing formula is:
# 
# $$\tilde{\phi} = \sin^2\!\left(\frac{b\,\pi}{2^m}\right) \cdot \text{norm}$$
# 
# where $b$ is the most-probable QPE readout integer.

# %%
# ── QAE EXECUTION DISPATCH ─────────────────────────────────────────────────────
t0 = time.perf_counter()

if BACKEND in ('cpu', 'aer-gpu'):
    # Build QAE circuit using QAE_Optimizer (same circuit for both)
    args_qae = dict(args_dqa)
    args_qae['m']              = m
    args_qae['norm']           = norm
    args_qae['oracle_circuit'] = exp.single_oracle_sin_inconstraint
    args_qae['gateset']        = False

    qae_optimizer = QAE_Optimizer(args_qae)
    qae_circuit   = qae_optimizer.compile_qae_circuit()
    print(f"QAE circuit: {qae_circuit.num_qubits} qubits, depth={qae_circuit.depth()}")

    if BACKEND == 'cpu':
        # Exact statevector via Statevector.evolve() (avoids transpile qubit routing)
        b_counts = bno.execute_qae(qae_circuit.copy(), m)
        print(f"[cpu] Statevector exact QAE — {len(b_counts)} QPE outcomes")

    else:  # aer-gpu
        # GPU shot-based QAE — measurement gates are qubit-routing-safe
        shots = N_SHOTS
        b_counts = bno.execute_qae(qae_circuit.copy(), m, num_meas=shots)
        print(f"[aer-gpu] AerSimulator GPU QAE ({shots} shots) — {len(b_counts)} QPE outcomes")

elif BACKEND == 'cuda-q':
    phi_cudaq = cudaq_opt.estimate_expected_value(Theta, w_d, shots=N_SHOTS)
    # phi_cudaq = cudaq_opt.estimate_expected_value_sv(Theta, w_d)
    # phi_cudaq = cudaq_opt.estimate_expected_value_observe(Theta)
    b_counts  = {}
    print(f"[cuda-q] CUDA-Q estimate_expected_value = {phi_cudaq:.4f}")

elapsed = time.perf_counter() - t0
print(f"Wall time: {elapsed*1000:.1f} ms")

# %%
# ── QAE POST-PROCESSING & RESULTS ─────────────────────────────────────────────
if BACKEND in ('cpu', 'aer-gpu'):
    # Weighted estimate from QPE readout histogram
    qae_phi = 0.0
    for key, prob in b_counts.items():
        b_int    = int(key, 2)
        amp      = np.sin(b_int * np.pi / 2**m)**2
        qae_phi += amp * norm * prob

    best_key  = max(b_counts, key=b_counts.get)
    best_b    = int(best_key, 2)
    peak_phi  = np.sin(best_b * np.pi / 2**m)**2 * norm

    print(f"QAE  phi (weighted avg)  [{BACKEND}] : {qae_phi:.4f}")
    print(f"QAE  phi (peak readout)  [{BACKEND}] : {peak_phi:.4f}")

elif BACKEND == 'cuda-q':
    qae_phi  = phi_cudaq
    peak_phi = phi_cudaq
    print(f"CUDA-Q  phi [{BACKEND}] : {qae_phi:.4f}")

print(f"Classical phi (w_d={w_d})           : {classical_phi:.4f}")
print(f"Relative error (weighted)           : {abs(qae_phi - classical_phi)/classical_phi*100:.1f}%")

# %% [markdown]
# ## Results Visualization

# %%
# ── PLOT 1: QPE readout histogram (cpu / aer-gpu) or single bar (cuda-q) ───────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: QPE readout probability distribution
ax = axes[0]
if BACKEND in ('cpu', 'aer-gpu') and b_counts:
    keys   = sorted(b_counts.keys(), key=lambda k: int(k, 2))
    probs  = [b_counts[k] for k in keys]
    x_pos  = [int(k, 2) for k in keys]
    amps   = [np.sin(b * np.pi / 2**m)**2 * norm for b in x_pos]

    ax.bar(x_pos, probs, color='steelblue', alpha=0.8)
    ax.axvline(best_b, color='red', linestyle='--', label=f'peak b={best_b} → φ≈{peak_phi:.2f}')
    ax.set_xlabel('QPE readout integer $b$')
    ax.set_ylabel('Probability')
    ax.set_title(f'QPE Histogram [{BACKEND}]')
    ax.legend(fontsize=8)
else:
    ax.bar([0], [1.0], color='salmon', alpha=0.8)
    ax.set_xticks([0]); ax.set_xticklabels(['CUDA-Q'])
    ax.set_ylabel('(no QPE histogram)')
    ax.set_title(f'CUDA-Q direct estimate: φ={qae_phi:.3f}')

# Right: comparison bar chart across methods
ax2 = axes[1]
labels   = ['Classical', 'DQA\n'+BACKEND, 'QAE\n'+BACKEND]
values   = [classical_phi, dqa_phi, qae_phi]
colors   = ['#4c72b0', '#dd8452', '#55a868']
bars     = ax2.bar(labels, values, color=colors, alpha=0.85, width=0.5)
ax2.axhline(classical_phi, color='#4c72b0', linestyle='--', linewidth=1, alpha=0.6)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9)
ax2.set_ylabel(r'$\phi(w_d)$')
ax2.set_title(f'Expected value comparison (w_d={w_d})')

plt.suptitle(f'Backend: {BACKEND}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results Summary
# 
# All three backends validated on NVIDIA H100 PCIe (81 GB, CUDA 12.4).  
# Problem: 4-turbine unit commitment, `n_y=4`, `m=4` QPE qubits, `w_d=2`.
# 
# | Backend | DQA φ(w_d=2) | QAE φ(w_d=2) | Classical φ | DQA error | QAE error |
# |---|---|---|---|---|---|
# | `cpu` (Qiskit statevector) | 5.31 | 5.59 | 4.71 | 12.8% | 18.6% |
# | `aer-gpu` (Qiskit Aer + cuStateVec) | 5.31 | 5.59 | 4.71 | 12.8% | 18.6% |
# | `cuda-q` (CUDA-Q nvidia target) | 5.42 | 5.46 | 4.71 | 15.0% | 15.8% |
# 
# **Notes:**
# - CPU and aer-gpu results are identical (same Qiskit statevector path, GPU acceleration transparent).
# - CUDA-Q uses `cudaq.sample` for DQA and `CudaqQAEOptimizer.estimate_expected_value` for QAE (direct inner-product, no QPE histogram).
# - Error relative to classical brute-force optimum is driven by finite DQA ansatz depth (`p=1`), not backend noise.
# - Wall time: aer-gpu QAE (4096 shots) ≈ 3.8 s; cuda-q QAE ≈ 0.08 s (direct amplitude estimation, no sampling loop).
# 


