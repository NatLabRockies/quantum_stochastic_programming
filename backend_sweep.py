# backend_sweep.py
#
# Sweeps n_y (qubit count) across multiple backends:
#   qiskit       — Qiskit exact statevector (CPU)
#   qpp-cpu      — CUDA-Q qpp-cpu (single-threaded CPU)
#   nvidia       — CUDA-Q cuStateVec (single GPU)
#   nvidia-mqpu  — CUDA-Q multi-QPU simulation (multi-GPU)
#   mgpu         — CUDA-Q cuStateVec multi-GPU (single large SV)
#
# DQA is run for each (n_y, backend) pair using a linear-ramp angle schedule.
# Results are collected into a combined timing/accuracy table and plots.

import csv
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

from qae import pdf_initialize
import ExpValFun_functions as exp
from binary_optimizer import BinaryNestedOptimizer
from qiskit_aer import AerSimulator
from qiskit import transpile as qiskit_transpile
import cudaq
from cudaq_impl import CudaqQAEOptimizer

# ── SWEEP PARAMETERS ───────────────────────────────────────────────────────────
MAX_N_Y = 16
MAX_CPU_N_Y = 9
# MAX_SINGLE_GPU_N_Y = 16
N_Y_VALUES = range(3, MAX_N_Y + 1) # n_y values to sweep
N_SHOTS    = 2**14                 # shots for CUDA-Q shot-based backends

# Backends to benchmark. Comment out any that are unavailable on your system.
BACKENDS = [
    'qiskit-cpu',    # Qiskit AerSimulator, device=CPU
    'qiskit-gpu',    # Qiskit AerSimulator, device=GPU (skipped if no CUDA device)
    'qpp-cpu',       # CUDA-Q reference CPU simulator
    'nvidia',        # CUDA-Q cuStateVec single-GPU
    'nvidia-mqpu',   # CUDA-Q multi-QPU (parallelises over multiple GPUs)
    # 'mgpu',          # CUDA-Q cuStateVec multi-GPU (one large statevector)
]

# Fixed first-stage parameters
n_x = 1
c_x = [3.]
c_r = 10.0
x0  = [2]   # first-stage gas commitment; w_d = n_y - sum(x0) scales with n_y

# ── RESULTS STORAGE ────────────────────────────────────────────────────────────
# results[backend] = list of dicts with keys: n_y, w_d, dqa_phi, dqa_time
results = {b: [] for b in BACKENDS}

# ── HELPER: try to activate a CUDA-Q target ───────────────────────────────────
# Maps backend label -> (target_name, option) for cudaq.set_target()
_CUDAQ_TARGET_MAP = {
    'qpp-cpu':      ('qpp-cpu',  None),
    'nvidia':       ('nvidia',   None),
    'nvidia-mqpu':  ('nvidia',   'mqpu'),
    'mgpu':         ('nvidia',   'mgpu'),
}

def try_set_cudaq_target(backend: str) -> bool:
    """Returns True if the CUDA-Q target was set successfully."""
    if backend not in _CUDAQ_TARGET_MAP:
        print(f"  [WARNING] Unknown CUDA-Q backend '{backend}'.")
        return False
    target, option = _CUDAQ_TARGET_MAP[backend]
    try:
        if option:
            cudaq.set_target(target, option=option)
        else:
            cudaq.set_target(target)
        return True
    except Exception as e:
        print(f"  [WARNING] CUDA-Q target '{backend}' unavailable: {e}")
        return False

# ── MAIN SWEEP ─────────────────────────────────────────────────────────────────
for n_y in N_Y_VALUES:
    print(f"\n{'='*65}")
    print(f"n_y = {n_y}")
    print(f"{'='*65}")

    n_xi      = n_y
    d         = n_y
    c_y       = list(np.linspace(0.1, 1.0, n_y))
    timesteps = n_y
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

    print(f"  c_y={[round(v, 2) for v in c_y]}, w_d={w_d}, norm={norm:.1f}")

    # Linear-ramp angles (shared across all backends)
    theta0 = []
    for t in range(timesteps):
        theta0.append(float(t / timesteps))
        theta0.append((1 - float(t / timesteps)) / math.pi)
    Theta = theta0
    print(f"  Linear ramp ({len(Theta)} angles)")

    # ── Shared Qiskit circuit objects (reused for qiskit backend and
    #    any Qiskit-based fallback) ─────────────────────────────────────────
    y_reg   = list(range(n_y))
    pdf_reg = list(range(n_y, 2 * n_y))
    bno     = BinaryNestedOptimizer(c_x, c_y, c_r, pdf, n_y, is_uniform=True)

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

    # CPU backends that should be skipped beyond MAX_CPU_N_Y
    _CPU_BACKENDS = {'qiskit-cpu', 'qpp-cpu'}

    for backend in BACKENDS:
        if backend in _CPU_BACKENDS and n_y > MAX_CPU_N_Y:
            print(f"\n  --- backend: {backend} --- [SKIPPED: n_y={n_y} > MAX_CPU_N_Y={MAX_CPU_N_Y}]")
            results[backend].append({
                'n_y': n_y, 'n_qubits': 2 * n_y, 'w_d': w_d,
                'dqa_phi': None, 'dqa_time': None,
            })
            continue

        print(f"\n  --- backend: {backend} ---")

        dqa_phi, dqa_time = None, None

        # ── Qiskit AerSimulator shot-based sampling ────────────────────────────
        if backend in ('qiskit-cpu', 'qiskit-gpu'):
            device = 'CPU' if backend == 'qiskit-cpu' else 'GPU'
            try:
                sim = AerSimulator(method='statevector', device=device,
                                   max_parallel_shots=0)
                if device == 'GPU':
                    # Probe GPU availability before building the circuit
                    sim.available_devices()
                dqa_circuit = exp.alternating_operator_ansatz(args_dqa)
                dqa_circuit.measure_all()
                dqa_circuit = qiskit_transpile(dqa_circuit, sim)
                t0     = time.perf_counter()
                result = sim.run(dqa_circuit, shots=N_SHOTS).result()
                dqa_time = time.perf_counter() - t0
                raw_counts = result.get_counts(0)
                dqa_counts = {k: v / N_SHOTS for k, v in raw_counts.items()}
                dqa_phi    = bno.process_expectation_value_optimizer(w_d, dqa_counts)
                print(f"    DQA phi={dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms)")
            except Exception as e:
                print(f"    [ERROR] {backend} backend failed: {e}")

        # ── CUDA-Q backends ────────────────────────────────────────────────
        else:
            if not try_set_cudaq_target(backend):
                results[backend].append({
                    'n_y': n_y, 'n_qubits': 2 * n_y, 'w_d': w_d,
                    'dqa_phi': None, 'dqa_time': None,
                })
                continue

            try:
                cudaq_opt = CudaqQAEOptimizer(
                    c_x=c_x, c_y=c_y, c_r=c_r,
                    n_y=n_y, w_d=w_d, cost_norm=cost_norm,
                )
                t0 = time.perf_counter()
                if backend == 'nvidia-mqpu':
                    dqa_phi = cudaq_opt.estimate_expected_value_async(
                        Theta, w_d, shots=N_SHOTS)
                else:
                    dqa_phi = cudaq_opt.estimate_expected_value(
                        Theta, w_d, shots=N_SHOTS)
                dqa_time = time.perf_counter() - t0
                print(f"    DQA phi={dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms)")
            except Exception as e:
                print(f"    [ERROR] CUDA-Q backend '{backend}' run failed: {e}")

        results[backend].append({
            'n_y':      n_y,
            'n_qubits': 2 * n_y,
            'w_d':      w_d,
            'dqa_phi':  dqa_phi,
            'dqa_time': dqa_time,
        })

# ── SUMMARY TABLE ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
col = 12
header = f"{'n_y':>5} {'n_qubits':>8} {'w_d':>4}"
for b in BACKENDS:
    label = b[:col]
    header += f"  {label+' phi':>{col}}  {label+' ms':>{col}}"
print(header)
print('-' * len(header))

# Collect all n_y values that appear in any backend's results
all_n_y = sorted({r['n_y'] for b in BACKENDS for r in results[b]})
for n_y in all_n_y:
    # find w_d from any backend that has this n_y
    w_d_val = next(
        (r['w_d'] for b in BACKENDS for r in results[b] if r['n_y'] == n_y),
        '?'
    )
    n_qubits_val = 2 * n_y
    row = f"{n_y:>5} {n_qubits_val:>8} {w_d_val:>4}"
    for b in BACKENDS:
        rec = next((r for r in results[b] if r['n_y'] == n_y), None)
        phi_s = f"{rec['dqa_phi']:>{col}.4f}"  if (rec and rec['dqa_phi']  is not None) else f"{'N/A':>{col}}"
        ms_s  = f"{rec['dqa_time']*1e3:>{col}.1f}" if (rec and rec['dqa_time'] is not None) else f"{'N/A':>{col}}"
        row  += f"  {phi_s}  {ms_s}"
    print(row)

# ── PLOTS ──────────────────────────────────────────────────────────────────────
COLORS  = ['steelblue', 'darkorange', 'seagreen', 'orchid', 'firebrick']
MARKERS = ['o', 's', '^', 'D', 'v']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax_phi  = axes[0]
ax_time = axes[1]

for idx, backend in enumerate(BACKENDS):
    recs = [r for r in results[backend] if r['dqa_phi'] is not None]
    if not recs:
        continue
    xs     = [r['n_qubits']      for r in recs]
    phis   = [r['dqa_phi']       for r in recs]
    times  = [r['dqa_time']*1e3  for r in recs]
    color  = COLORS[idx % len(COLORS)]
    marker = MARKERS[idx % len(MARKERS)]

    ax_phi.plot(xs, phis,  f'-{marker}', label=backend, color=color)
    ax_time.plot(xs, times, f'-{marker}', label=backend, color=color)

ax_phi.set_xlabel('Total qubits ($2 n_y$)')
ax_phi.set_ylabel(r'$\phi(w_d)$')
ax_phi.set_title('DQA expected value vs qubit count')
ax_phi.legend(fontsize=8)
ax_phi.grid(True, alpha=0.3)

ax_time.set_xlabel('Total qubits ($2 n_y$)')
ax_time.set_ylabel('Wall time (ms)')
ax_time.set_title('DQA wall time vs qubit count')
ax_time.set_yscale('log')
ax_time.legend(fontsize=8)
ax_time.grid(True, alpha=0.3, which='both')

plt.suptitle('Backend $n_y$ sweep — DQA comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('backend_sweep_results.png', dpi=150)
plt.show()
print("Plot saved to backend_sweep_results.png")

# ── CSV OUTPUT ─────────────────────────────────────────────────────────────────
csv_path = 'backend_sweep_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    header = ['n_y', 'n_qubits', 'w_d']
    for b in BACKENDS:
        header += [f'{b}_dqa_phi', f'{b}_dqa_ms']
    writer.writerow(header)
    # Rows
    for n_y in all_n_y:
        w_d_val = next(
            (r['w_d'] for b in BACKENDS for r in results[b] if r['n_y'] == n_y),
            ''
        )
        row = [n_y, 2 * n_y, w_d_val]
        for b in BACKENDS:
            rec = next((r for r in results[b] if r['n_y'] == n_y), None)
            phi = f"{rec['dqa_phi']:.6f}"  if (rec and rec['dqa_phi']  is not None) else ''
            ms  = f"{rec['dqa_time']*1e3:.3f}" if (rec and rec['dqa_time'] is not None) else ''
            row += [phi, ms]
        writer.writerow(row)
print(f"Results saved to {csv_path}")
