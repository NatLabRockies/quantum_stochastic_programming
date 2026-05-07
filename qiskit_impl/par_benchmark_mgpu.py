"""
Single-process multi-GPU (nvidia mgpu) DQA benchmark.

cuStateVec shards the statevector across all visible GPUs automatically.
n_y=16 → 32 qubits → 32 GB fp32 → 8 GB/GPU  (4 GPUs)
n_y=17 → 34 qubits → 137 GB fp32 → 34 GB/GPU (4 GPUs)

Launch with the wrapper (sets GTL preload):
    srun --jobid=<ID> -N1 --ntasks=1 --gpus-per-node=4 \\
        /kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl/run_mgpu.sh \\
        par_benchmark_mgpu.py 2>&1 | tee /kfs3/scratch/nsawant/bench_mgpu.log

NOTE: run_mgpu.sh exec's python3 directly, passing all args through.
"""
import os, sys, math, time, json

_SP      = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_IMPL    = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'
for _p in [_SP, _IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_IMPL)

_CRAY_LIBMPI = '/opt/cray/pe/mpich/8.1.28/ofi/gnu/10.3/lib/libmpi.so'
os.environ.setdefault('CUDAQ_MPI_COMM_LIB',
    f'{_SP}/distributed_interfaces/libcudaq_distributed_interface_mpi.so')
os.environ.setdefault('CUDAQ_MGPU_LIB_MPI', _CRAY_LIBMPI)
os.environ.setdefault('CUDAQ_MGPU_COMM_PLUGIN_TYPE', 'MPICH')

import numpy as np
import cudaq
from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

cudaq.set_target('nvidia', option='mgpu,fp32')
n_visible = cudaq.num_available_gpus()
print(f"Target: {cudaq.get_target().name}  visible GPUs: {n_visible}  cudaq {cudaq.__version__}")

N_SHOTS   = 256
TIMESTEPS = 20
P1, P2    = 0.0001, 0.001
C_R       = 10.0
C_X       = [3.]
NY_LIST   = [16, 17]

noise_model = build_depolarizing_noise_model(p1=P1, p2=P2)

def make_opt(ny, noisy=False):
    c_y       = list(np.linspace(0.1, 1.0, ny))
    w_d       = ny - 2
    cost_norm = w_d * C_R / ny
    theta0    = []
    for t in range(TIMESTEPS):
        theta0.append(float(t / TIMESTEPS))
        theta0.append((1 - float(t / TIMESTEPS)) / math.pi)
    opt = CudaqQAEOptimizer(
        c_x=C_X, c_y=c_y, c_r=C_R, n_y=ny, w_d=w_d,
        cost_norm=cost_norm,
        noise_model=noise_model if noisy else None)
    return opt, theta0

results = {}

print(f"\n── mgpu DQA benchmark ──  shots={N_SHOTS}  timesteps={TIMESTEPS}\n")

for ny in NY_LIST:
    n_q = 2 * ny
    fp32_gb = (2**n_q * 8) / 1e9
    print(f"=== n_y={ny}  ({n_q} qubits, {fp32_gb:.0f} GB fp32, ~{fp32_gb/n_visible:.0f} GB/GPU) ===")

    for label, noisy in [('noiseless', False), ('noisy', True)]:
        opt, theta0 = make_opt(ny, noisy=noisy)
        try:
            t0 = time.perf_counter()
            probs = opt.sample_ansatz(theta0, shots=N_SHOTS)
            elapsed = time.perf_counter() - t0
            top = max(probs, key=probs.get)
            print(f"  {label:10s}: {elapsed:7.1f} s  top={top}  p={probs[top]:.3f}")
            results[f'ny{ny}_{label}'] = {'elapsed_s': elapsed, 'top': top, 'p': probs[top]}
        except Exception as e:
            print(f"  {label:10s}: FAILED — {e}")
            results[f'ny{ny}_{label}'] = {'error': str(e)}

out = '/kfs3/scratch/nsawant/bench_mgpu_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")
print(json.dumps(results, indent=2))

cudaq.mpi.finalize()
