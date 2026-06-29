# simple_ed_cuda_q.py
#
# Single-instance CUDA-Q run for a fixed n_y.
# Runs DQA + QAE and compares both to the classical brute-force optimum.

import os, sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qiskit_impl')
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

from qae import *
from binary_optimizer import BinaryNestedOptimizer
import cudaq
from cudaq_impl import CudaqQAEOptimizer

# ── CUDA-Q TARGET ──────────────────────────────────────────────────────────────
CUDAQ_TARGET_MODE = os.getenv('CUDAQ_TARGET_MODE', 'qpp-cpu').strip().lower()

try:
    if CUDAQ_TARGET_MODE == 'mgpu':
        cudaq.set_target('nvidia', option='mgpu')
        print("[cuda-q] target set to 'nvidia' (mgpu).")
    else:
        cudaq.set_target('nvidia')
        print("[cuda-q] target set to 'nvidia' (single GPU).")
except Exception as e:
    print(f"WARNING: nvidia target unavailable ({e}), falling back to qpp-cpu.")
    cudaq.set_target('qpp-cpu')

# ── PARAMETERS ─────────────────────────────────────────────────────────────────
# Quantum circuits parameters
N_SHOTS     = 2**16  # DQA shot count (only used when USE_STATEVECTOR=False)
USE_STATEVECTOR = False  # True:  exact statevector probabilities (no shot noise)
                        # False: shot-based sampling with N_SHOTS shots
m_qpe    = 5         # QPE readout qubits; estimation error ~ π·norm/2^m_qpe

# Fixed first-stage parameters
n_x  = 1
c_x  = [4.0]
x0   = [6]   # first-stage gas commitment
# n_x  = 2
# c_x  = [4., 5.]
# x0   = [5., 1.]

# Second-stage parameters
n_y    = 4         # number of wind turbine qubits
n_xi   = n_y
d      = 8
# c_y    = list(np.linspace(0.1, 1.0, n_y))
c_y = [[c0, c1] for c0, c1 in zip(np.linspace(0.1, 1.0, n_y), 1e-3*np.ones(n_y))]
c_r    = 10.0



# Derived quantities
c_y_eff = [sum(c_y[i]) for i in range(n_y)]
timesteps = max(10, n_y * n_y)
w_d       = int(d - sum(x0))     # wind demand
norm      = w_d * c_r            # QAE amplitude normalisation
# cost_norm = w_d * c_r / n_y     # DQA cost operator normalisation

x_min = d - n_y

assert w_d > 0, f"w_d={w_d} <= 0: x0={x0} covers full demand, nothing for wind turbines to do."

pdf = {
    tuple(int(v) for v in ('{0:0' + str(n_y) + 'b}').format(i)): 1 / 2**n_y
    for i in range(2**n_y)
}

n_qubits_dqa = 2 * n_y
n_qubits_qae = m_qpe + 2 * n_y + 1
# print(f"n_y={n_y}, w_d={w_d}, c_y={[round(v,2) for v in c_y]}, norm={norm:.1f}")
print(f"n_y={n_y}, w_d={w_d}, m_qpe={m_qpe}, norm={norm:.1f}")
print(f"Qubits: DQA={n_qubits_dqa}, QAE={n_qubits_qae}")

# ── CLASSICAL BASELINE ─────────────────────────────────────────────────────────
bno          = BinaryNestedOptimizer(c_x, c_y_eff, c_r, pdf, d, is_uniform=True)
exp_vals     = bno.brute_force_wind_demand_expectation_values()
classical_phi = exp_vals[w_d]
obj_surface  = [bno.gas_costs[0] * x + exp_vals[d - x] for x in range(x_min, d + 1)]

print(f"\nClassical phi(wind_demand): {[round(v, 3) for v in exp_vals]}")
print(f"Classical phi(w_d={w_d}): {classical_phi:.4f}")

# ── DQA ANGLE SETUP ────────────────────────────────────────────────────────────
cudaq_opt = CudaqQAEOptimizer(
    c_x=c_x, c_y=c_y, c_r=c_r,
    n_y=n_y, w_d=w_d, norm=norm,
)

theta0 = []
for t in range(timesteps):
    theta0.append(float(t / timesteps))
    theta0.append((1 - float(t / timesteps)) / math.pi)

Theta = theta0

# if USE_COBYLA:
#     print(f"\nOptimising {len(theta0)} angles with COBYLA …")
#     opt_result = minimize(
#         lambda th: cudaq_opt.estimate_expected_value_sv(list(th), w_d),
#         theta0,
#         method='COBYLA',
#         options={'maxiter': 500, 'rhobeg': 0.5, 'disp': False},
#     )
#     Theta = list(opt_result.x)
#     print(f"COBYLA done | phi={opt_result.fun:.4f} | {opt_result.nfev} evals")
# else:
#     Theta = theta0
#     print(f"\nLinear ramp ({len(Theta)} angles, COBYLA disabled)")

# ── DQA EXECUTION ──────────────────────────────────────────────────────────────
print(f"\nRunning DQA ({'statevector' if USE_STATEVECTOR else f'{N_SHOTS} shots'}) …")
t0 = time.perf_counter()
if USE_STATEVECTOR:
    dqa_phi = cudaq_opt.estimate_expected_value_sv(Theta, w_d)
else:
    dqa_phi = cudaq_opt.estimate_expected_value(Theta, w_d, shots=N_SHOTS)
dqa_time = time.perf_counter() - t0
dqa_mode = 'sv' if USE_STATEVECTOR else f'{N_SHOTS} shots'
print(f"DQA  phi(w_d={w_d}) = {dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms, {dqa_mode})")

t_grad = time.perf_counter()
dqa_grad = cudaq_opt.estimate_expected_gradient(Theta, w_d, shots=N_SHOTS)
t_grad = time.perf_counter() - t_grad
print(f"DQA  grad(w_d={w_d}) = {[round(g, 4) for g in dqa_grad]}  ({t_grad*1e3:.1f} ms)")

# Classical check 1: exact statevector gradient (no shot noise)
t_grad_sv = time.perf_counter()
dqa_grad_sv = cudaq_opt.estimate_expected_gradient_sv(Theta, w_d)
t_grad_sv = time.perf_counter() - t_grad_sv
print(f"DQA  grad_sv(w_d={w_d}) = {[round(g, 4) for g in dqa_grad_sv]}  ({t_grad_sv*1e3:.1f} ms, exact SV)")

# Classical check 2: analytical bound under uniform xi (linear costs only)
# E_xi[dQ/dy_j] = 0.5*c_y_eff[j] + 0.5*c_r  (independent of DQA angles)
analytical_grad = [0.5 * c_y_eff[j] + 0.5 * c_r for j in range(n_y)]
print(f"Analytical grad (uniform xi) = {[round(g, 4) for g in analytical_grad]}")

# ── QAE EXECUTION ──────────────────────────────────────────────────────────────
print(f"\nRunning QAE ({'statevector' if USE_STATEVECTOR else f'{N_SHOTS} shots'}) …")
t0        = time.perf_counter()
qae_shots = None if USE_STATEVECTOR else N_SHOTS
qae_phi   = cudaq_opt.estimate_expected_value_qae(Theta, m=m_qpe, shots=qae_shots)
qae_time  = time.perf_counter() - t0
qae_mode  = 'sv' if USE_STATEVECTOR else f'{N_SHOTS} shots'
print(f"QAE  phi(w_d={w_d}) = {qae_phi:.4f}  ({qae_time*1e3:.1f} ms, {qae_mode})")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"{'Method':<20} {'phi(w_d)':>10} {'error':>8} {'time (ms)':>12}")
print(f"{'-'*50}")
print(f"{'Classical':<20} {classical_phi:>10.4f} {'—':>8} {'—':>12}")
print(f"{'DQA (CUDA-Q)':<20} {dqa_phi:>10.4f} {abs(dqa_phi-classical_phi)/classical_phi*100:>7.1f}% {dqa_time*1e3:>12.1f}")
print(f"{'QAE (CUDA-Q)':<20} {qae_phi:>10.4f} {abs(qae_phi-classical_phi)/classical_phi*100:>7.1f}% {qae_time*1e3:>12.1f}")
print(f"{'='*50}")

# ── PLOTS ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: bar chart comparing classical, DQA, QAE
ax = axes[0]
labels = ['Classical', f'DQA\n(CUDA-Q)', f'QAE\n(CUDA-Q)']
values = [classical_phi, dqa_phi, qae_phi]
colors = ['#4c72b0', '#dd8452', '#55a868']
bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
ax.axhline(classical_phi, color='#4c72b0', linestyle='--', linewidth=1, alpha=0.6, label='Classical optimum')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel(r'$\phi(w_d)$')
ax.set_title(f'Expected value comparison ($w_d={w_d}$, $n_y={n_y}$)')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

# Right: classical full objective surface o(x)
ax2 = axes[1]
x_vals = list(range(x_min, d + 1))
ax2.plot(x_vals, obj_surface, '-o', color='#4c72b0', label='Classical $o(x)$')
ax2.axvline(sum(x0), color='gray', linestyle='--', alpha=0.6, label=f'$x_0={x0[0]}$')
ax2.set_xlabel('Gas commitment $x$')
ax2.set_ylabel('Objective $o(x)$')
ax2.set_title('Classical objective surface')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'CUDA-Q single run — $n_y={n_y}$, nvidia target', fontsize=12, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_ed_cuda_q_results.png')
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Plot saved to {out_png}")
