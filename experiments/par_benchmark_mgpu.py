"""
Single-process multi-GPU (nvidia mgpu) DQA benchmark.

cuStateVec shards the statevector across all visible GPUs automatically.
n_y=16 → 32 qubits → 32 GB fp32 → 8 GB/GPU  (4 GPUs)
n_y=17 → 34 qubits → 137 GB fp32 → 34 GB/GPU (4 GPUs)

Launch from the repository root after configuring any site-specific MPI/GPU environment:
    srun --jobid=<ID> -N1 --ntasks=1 --gpus-per-node=4 \\
        ./run_mgpu.sh par_benchmark_mgpu.py 2>&1 | tee bench_mgpu.log

NOTE: run_mgpu.sh selects the interpreter via PYTHON and forwards all args.
"""
import os, sys, math, time, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parents[1] / 'publications' / 'parallelisation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set these CUDA-Q MPI variables in the launch environment when required by the
# local CUDA-Q and MPI installations.
if os.environ.get('CUDAQ_MGPU_LIB_MPI'):
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
        norm=cost_norm * ny,
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

out = RESULTS_DIR / 'bench_mgpu_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")
print(json.dumps(results, indent=2))

cudaq.mpi.finalize()
