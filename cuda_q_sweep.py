# cuda_q_sweep.py
#
# Sweeps n_y (qubit count) using only the CUDA-Q backend.
# Runs DQA + QAE for each n_y and plots phi estimates and wall times.
# No classical validation — CUDA-Q only.

import os, sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qiskit_impl')
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

from qae import *
from binary_optimizer import BinaryNestedOptimizer
import cudaq
from cudaq_impl import CudaqQAEOptimizer

# ── CUDA-Q TARGET ──────────────────────────────────────────────────────────────
try:
    cudaq.set_target('nvidia')
    print("[cuda-q] target set to 'nvidia' (cuStateVec).")
except Exception as e:
    print(f"WARNING: nvidia target unavailable ({e}), falling back to qpp-cpu.")
    cudaq.set_target('qpp-cpu')

# ── SWEEP PARAMETERS ───────────────────────────────────────────────────────────
N_Y_VALUES = [4, 6, 8, 10, 12]   # n_y values to sweep (must be >= 3 with x0=[2])
N_SHOTS    = 2**12
USE_COBYLA = False
m          = 5                    # QPE readout qubits

# Fixed first-stage parameters
n_x = 1
c_x = [3.]
c_r = 10.0
x0  = [2]    # first-stage gas commitment; w_d = n_y - sum(x0) scales with n_y

# ── RESULTS STORAGE ────────────────────────────────────────────────────────────
results = []

# ── MAIN SWEEP LOOP ────────────────────────────────────────────────────────────
for n_y in N_Y_VALUES:
    print(f"\n{'='*60}")
    print(f"n_y = {n_y}")
    print(f"{'='*60}")

    n_xi      = n_y
    d         = n_y
    c_y       = list(np.linspace(0.1, 1.0, n_y))
    timesteps = n_y
    w_d       = int(d - sum(x0))        # wind demand; scales as n_y grows
    norm      = w_d * c_r               # QAE amplitude normalisation
    cost_norm = w_d * c_r / n_y         # DQA cost operator normalisation

    if w_d <= 0:
        print(f"  Skipping n_y={n_y}: w_d={w_d} <= 0 (x0={x0} covers full demand).")
        continue

    pdf = {
        tuple(int(v) for v in ('{0:0' + str(n_y) + 'b}').format(i)): 1 / 2**n_y
        for i in range(2**n_y)
    }

    print(f"  c_y={[round(v,2) for v in c_y]}, w_d={w_d}, norm={norm:.1f}")

    # ── DQA ANGLES ──────────────────────────────────────────────────────────
    cudaq_opt = CudaqQAEOptimizer(
        c_x=c_x, c_y=c_y, c_r=c_r,
        n_y=n_y, w_d=w_d, cost_norm=cost_norm,
    )

    theta0 = []
    for t in range(timesteps):
        theta0.append(float(t / timesteps))
        theta0.append((1 - float(t / timesteps)) / math.pi)

    if USE_COBYLA:
        print(f"  Optimising {len(theta0)} angles with COBYLA …")
        opt_result = minimize(
            lambda th: cudaq_opt.estimate_expected_value_sv(list(th), w_d),
            theta0,
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.5, 'disp': False},
        )
        Theta = list(opt_result.x)
        print(f"  COBYLA done | phi={opt_result.fun:.4f} | {opt_result.nfev} evals")
    else:
        Theta = theta0
        print(f"  Linear ramp ({len(Theta)} angles, COBYLA disabled)")

    # ── DQA EXECUTION ────────────────────────────────────────────────────────
    t0         = time.perf_counter()
    dqa_counts = cudaq_opt.sample_ansatz(Theta, shots=N_SHOTS)
    dqa_time   = time.perf_counter() - t0

    expectation = 0.0
    for bstr, prob in dqa_counts.items():
        y_bits      = [int(b) for b in bstr[:n_y]]
        xi_bits     = [int(b) for b in bstr[n_y:]]
        true_output = np.array(y_bits) * np.array(xi_bits)
        op_cost     = float(np.dot(c_y, true_output))
        shortfall   = max(0, w_d - int(true_output.sum()))
        expectation += (op_cost + shortfall * c_r) * prob
    dqa_phi = expectation

    print(f"  DQA  phi(w_d={w_d}) = {dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms)")

    # ── QAE EXECUTION ────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    qae_phi = cudaq_opt.estimate_expected_value(Theta, w_d, shots=N_SHOTS)
    qae_time = time.perf_counter() - t0

    print(f"  QAE  phi(w_d={w_d}) = {qae_phi:.4f}  ({qae_time*1e3:.1f} ms)")

    results.append({
        'n_y':      n_y,
        'w_d':      w_d,
        'dqa_phi':  dqa_phi,
        'qae_phi':  qae_phi,
        'dqa_time': dqa_time,
        'qae_time': qae_time,
    })

# ── SUMMARY TABLE ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'n_y':>6} {'w_d':>5} {'DQA phi':>10} {'QAE phi':>10} {'DQA ms':>10} {'QAE ms':>10}")
print(f"{'-'*6} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for r in results:
    print(
        f"{r['n_y']:>6} {r['w_d']:>5} {r['dqa_phi']:>10.4f} {r['qae_phi']:>10.4f} "
        f"{r['dqa_time']*1e3:>10.1f} {r['qae_time']*1e3:>10.1f}"
    )

# ── PLOTS ──────────────────────────────────────────────────────────────────────
n_y_vals  = [r['n_y']           for r in results]
dqa_phis  = [r['dqa_phi']       for r in results]
qae_phis  = [r['qae_phi']       for r in results]
dqa_times = [r['dqa_time']*1e3  for r in results]
qae_times = [r['qae_time']*1e3  for r in results]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.plot(n_y_vals, dqa_phis, '-o', label='DQA φ', color='steelblue')
ax.plot(n_y_vals, qae_phis, '-s', label='QAE φ', color='salmon')
ax.set_xlabel('$n_y$ (turbine qubits)')
ax.set_ylabel(r'$\phi(w_d)$')
ax.set_title('Expected value vs qubit count')
ax.legend()
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(n_y_vals, dqa_times, '-o', label='DQA wall time', color='steelblue')
ax2.plot(n_y_vals, qae_times, '-s', label='QAE wall time', color='salmon')
ax2.set_xlabel('$n_y$ (turbine qubits)')
ax2.set_ylabel('Wall time (ms)')
ax2.set_title('Wall time vs qubit count')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('CUDA-Q $n_y$ sweep (nvidia target)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('cuda_q_sweep_results.png', dpi=150)
plt.show()
print("Plot saved to cuda_q_sweep_results.png")
