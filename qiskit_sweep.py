# qiskit_sweep.py
#
# Sweeps n_y (qubit count) using the Qiskit CPU statevector backend.
# Runs DQA + QAE for each n_y and plots phi estimates and wall times.
# Mirrors cuda_q_sweep.py but dispatches via Qiskit Statevector / AerSimulator.

import os, sys, math, time
import numpy as np
import matplotlib.pyplot as plt

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qiskit_impl')
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

from qiskit.quantum_info import Statevector
from qae import QAE_Optimizer, pdf_initialize
import ExpValFun_functions as exp
from binary_optimizer import BinaryNestedOptimizer

# ── SWEEP PARAMETERS ───────────────────────────────────────────────────────────
N_Y_VALUES = range(4, 18, 2)   # n_y values to sweep
N_SHOTS    = 2**14             # shots for AerSimulator QAE (not used for exact SV)
USE_EXACT_SV = True            # True: exact statevector (no shots); False: AerSimulator shots

# Fixed first-stage parameters
n_x = 1
c_x = [3.]
c_r = 10.0
x0  = [2]   # first-stage gas commitment; w_d = n_y - sum(x0) scales with n_y
m   = 5     # QPE readout qubits for QAE

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
    timesteps = n_y**2
    w_d       = int(d - sum(x0))
    norm      = w_d * c_r
    cost_norm = w_d * c_r / n_y

    if w_d <= 0:
        print(f"  Skipping n_y={n_y}: w_d={w_d} <= 0 (x0={x0} covers full demand).")
        continue

    pdf = {
        tuple(int(v) for v in ('{0:0' + str(n_y) + 'b}').format(i)): 1 / 2**n_y
        for i in range(2**n_y)
    }

    print(f"  c_y={[round(v,2) for v in c_y]}, w_d={w_d}, norm={norm:.1f}")

    bno = BinaryNestedOptimizer(c_x, c_y, c_r, pdf, n_y, is_uniform=True)

    # ── DQA ANGLES (linear ramp) ─────────────────────────────────────────────
    theta0 = []
    for t in range(timesteps):
        theta0.append(float(t / timesteps))
        theta0.append((1 - float(t / timesteps)) / math.pi)
    Theta = theta0
    print(f"  Linear ramp ({len(Theta)} angles)")

    # ── BUILD SHARED DQA CIRCUIT ─────────────────────────────────────────────
    y_reg   = list(range(n_y))
    pdf_reg = list(range(n_y, 2 * n_y))

    args_dqa = {
        'n_y': n_y, 'n_x': n_x, 'n_xi': n_xi,
        'c_x': c_x, 'c_y': c_y, 'c_r': c_r, 'pdf': pdf,
        'y_reg': y_reg, 'pdf_reg': pdf_reg,
        'Theta': Theta, 'w_d': w_d, 'cost_norm': cost_norm,
        'uniform': True,
        'cost_operator_circuit':  exp.cost_operator,
        'mixer_operator_circuit': exp.demand_constraint_preserving_mixer,
        'initial_state_circuit':  exp.dicke_state_circuit,
        'pdf_circuit':            pdf_initialize,
    }
    dqa_circuit = exp.alternating_operator_ansatz(args_dqa)
    print(f"  DQA circuit: {dqa_circuit.num_qubits} qubits, depth={dqa_circuit.depth()}")

    # ── DQA EXECUTION ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sv = Statevector.from_label('0' * (n_y + n_xi)).evolve(dqa_circuit)
    dqa_counts = sv.probabilities_dict()
    dqa_time   = time.perf_counter() - t0

    dqa_phi = bno.process_expectation_value_optimizer(w_d, dqa_counts)
    print(f"  DQA  phi(w_d={w_d}) = {dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms)")

    # ── BUILD QAE CIRCUIT ────────────────────────────────────────────────────
    args_qae = dict(args_dqa)
    args_qae['m']              = m
    args_qae['norm']           = norm
    args_qae['oracle_circuit'] = exp.single_oracle_sin_inconstraint
    args_qae['gateset']        = False

    qae_optimizer = QAE_Optimizer(args_qae)
    qae_circuit   = qae_optimizer.compile_qae_circuit()
    print(f"  QAE circuit: {qae_circuit.num_qubits} qubits, depth={qae_circuit.depth()}")

    # ── QAE EXECUTION ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if USE_EXACT_SV:
        b_counts = bno.execute_qae(qae_circuit.copy(), m)
    else:
        b_counts = bno.execute_qae(qae_circuit.copy(), m, num_meas=N_SHOTS)
    qae_time = time.perf_counter() - t0

    # Weighted estimate from QPE readout histogram
    qae_phi = 0.0
    for key, prob in b_counts.items():
        b_int    = int(key, 2)
        amp      = np.sin(b_int * np.pi / 2**m)**2
        qae_phi += amp * norm * prob

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
n_y_vals  = [r['n_y']          for r in results]
dqa_phis  = [r['dqa_phi']      for r in results]
qae_phis  = [r['qae_phi']      for r in results]
dqa_times = [r['dqa_time']*1e3 for r in results]
qae_times = [r['qae_time']*1e3 for r in results]

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

backend_label = 'Qiskit exact statevector (CPU)' if USE_EXACT_SV else f'Qiskit AerSimulator ({N_SHOTS} shots)'
plt.suptitle(f'Qiskit $n_y$ sweep — {backend_label}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('qiskit_sweep_results.png', dpi=150)
plt.show()
print("Plot saved to qiskit_sweep_results.png")
