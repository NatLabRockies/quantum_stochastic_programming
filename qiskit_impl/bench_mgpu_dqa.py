"""
mgpu DQA benchmark — n_y = 16, 17, 18 across all available GPUs.

Each MPI rank drives one GPU; cuStateVec distributes the fp32 statevector
across all ranks via GPU-aware MPI. Configure any site-specific MPI/GPU
environment before launching. From qiskit_impl/:

    srun -N 2 --ntasks-per-node=4 --gpus-per-node=4 \\
        ./run_mgpu.sh \\
        bench_mgpu_dqa.py [--ny 16 17 18] [--shots 256] [--noisy] 2>&1 | tee bench_mgpu_dqa.log
"""
import sys, os, time, math, json, argparse
from pathlib import Path

# ── path setup (before any cudaq import) ─────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
QISKIT_IMPL_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set CUDA-Q MPI library variables in the launch environment when the installed
# CUDA-Q and MPI distribution requires explicit library configuration.
if os.environ.get('CUDAQ_MGPU_LIB_MPI'):
    os.environ.setdefault('CUDAQ_MGPU_COMM_PLUGIN_TYPE', 'MPICH')

import cudaq
import numpy as np

# Use SLURM_PROCID to identify rank before cudaq.mpi is available
rank = int(os.environ.get('SLURM_PROCID', 0))
ntasks = int(os.environ.get('SLURM_NTASKS', 1))

cudaq.set_target('nvidia', option='mgpu,fp32')

from qiskit_impl.cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ny',    nargs='+', type=int, default=[16, 17, 18])
parser.add_argument('--shots', type=int,            default=256)
parser.add_argument('--noisy', action='store_true')
args = parser.parse_args()

TIMESTEPS = 20
P1, P2    = 0.0001, 0.001
C_R       = 10.0
C_X       = [3.]
N_SHOTS   = args.shots

noise_model = build_depolarizing_noise_model(p1=P1, p2=P2) if args.noisy else None

def make_opt(ny):
    c_y       = list(np.linspace(0.1, 1.0, ny))
    w_d       = ny - 2
    cost_norm = w_d * C_R / ny
    theta0    = []
    for t in range(TIMESTEPS):
        theta0.append(float(t / TIMESTEPS))
        theta0.append((1 - float(t / TIMESTEPS)) / math.pi)
    opt = CudaqQAEOptimizer(c_x=C_X, c_y=c_y, c_r=C_R, n_y=ny, w_d=w_d,
                            norm=cost_norm * ny, noise_model=noise_model)
    return opt, theta0

if rank == 0:
    mode = 'NOISY trajectory' if args.noisy else 'noiseless'
    print(f'\n── mgpu DQA benchmark ({mode}) ──')
    print(f'   GPUs={ntasks}  shots={N_SHOTS}  ny={args.ny}')
    print(f'   target={cudaq.get_target().name}\n', flush=True)

results = {}

for ny in args.ny:
    n_qubits = 2 * ny
    mem_total = (2**n_qubits * 8) / 1e9
    mem_per_gpu = mem_total / ntasks

    if rank == 0:
        print(f'n_y={ny}: {n_qubits} qubits  fp32={mem_total:.1f} GB total  '
              f'{mem_per_gpu:.1f} GB/GPU ... ', end='', flush=True)

    opt, theta0 = make_opt(ny)

    t0 = time.perf_counter()
    try:
        _ = opt.sample_ansatz(theta0, shots=N_SHOTS)
        elapsed = time.perf_counter() - t0
        if rank == 0:
            print(f'{elapsed:.1f} s', flush=True)
        results[ny] = {'elapsed_s': elapsed, 'n_qubits': n_qubits,
                       'n_gpus': ntasks, 'shots': N_SHOTS, 'noisy': args.noisy,
                       'mem_total_gb': mem_total, 'mem_per_gpu_gb': mem_per_gpu}
    except Exception as e:
        if rank == 0:
            print(f'FAILED — {e}', flush=True)
        results[ny] = {'error': str(e)}

if rank == 0:
    out = QISKIT_IMPL_DIR / f'bench_mgpu_dqa_{"noisy" if args.noisy else "ideal"}_gpu{ntasks}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out}')

cudaq.mpi.finalize()
