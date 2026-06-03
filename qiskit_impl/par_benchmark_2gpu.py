"""
2-GPU single-node mqpu benchmark for n_y = 14,16,18,20.
256 shots split across 2 virtual QPUs via cudaq.sample_async.

Run on an allocated node:
    ssh x3103c0s9b0n0 "nohup /path/to/python par_benchmark_2gpu.py > /tmp/par_bench_2gpu.log 2>&1 &"
"""
import os, sys, math, time, json

_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

import numpy as np
import cudaq
from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

N_SHOTS   = 256
TIMESTEPS = 20
P1, P2    = 0.0001, 0.001
C_R, C_X  = 10.0, [3.]
NY_LIST   = [14]   # n_y>=16 (>=32 qubits) OOM on H100

cudaq.set_target('nvidia', option='mqpu')
N_QPUS = cudaq.num_available_gpus()
noise_model = build_depolarizing_noise_model(p1=P1, p2=P2)

print(f'\n── 2-GPU mqpu benchmark ──')
print(f'   target={cudaq.get_target().name}  QPUs={N_QPUS}  N_SHOTS={N_SHOTS}  ny={NY_LIST}\n')

def make_opt(ny):
    c_y       = list(np.linspace(0.1, 1.0, ny))
    w_d       = ny - 2
    cost_norm = w_d * C_R / ny
    theta0    = []
    for t in range(TIMESTEPS):
        theta0.append(float(t / TIMESTEPS))
        theta0.append((1 - float(t / TIMESTEPS)) / math.pi)
    return CudaqQAEOptimizer(c_x=C_X, c_y=c_y, c_r=C_R, n_y=ny, w_d=w_d,
                             cost_norm=cost_norm, noise_model=noise_model), theta0

results = {}
for ny in NY_LIST:
    opt, theta0 = make_opt(ny)
    try:
        t0 = time.perf_counter()
        opt.sample_ansatz_mqpu(theta0, total_shots=N_SHOTS, n_qpus=N_QPUS)
        dt = time.perf_counter() - t0
        results[ny] = round(dt, 2)
        print(f'  n_y={ny:2d}  qubits={2*ny:2d}  {dt:.1f}s', flush=True)
    except Exception as e:
        print(f'  n_y={ny:2d}  FAIL: {e}', flush=True)

print(f"\n{'n_y':>4}  {'qubits':>6}  {'2-GPU(s)':>10}")
print('-' * 26)
for ny in NY_LIST:
    if ny in results:
        print(f'{ny:>4}  {2*ny:>6}  {results[ny]:>10.1f}')

fname = os.path.join(_QISKIT_IMPL, 'par_benchmark_2gpu_results.json')
with open(fname, 'w') as f:
    json.dump({'n_shots': N_SHOTS, 'n_gpus': N_QPUS,
               'timings': {str(ny): results[ny] for ny in NY_LIST if ny in results}}, f, indent=2)
print(f'\nResults -> {fname}')
