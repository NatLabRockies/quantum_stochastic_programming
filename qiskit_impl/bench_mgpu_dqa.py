"""
mgpu DQA benchmark — n_y = 16, 17, 18 across all available GPUs.

Each MPI rank drives one GPU; cuStateVec distributes the fp32 statevector
across all ranks via GPU-aware MPI.  Must be launched via run_mgpu.sh so
that Cray GTL is preloaded before Python starts:

    srun -N 2 --ntasks-per-node=4 --gpus-per-node=4 \\
        /kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl/run_mgpu.sh \\
        bench_mgpu_dqa.py [--ny 16 17 18] [--shots 256] [--noisy] 2>&1 | tee bench_mgpu_dqa.log
"""
import sys, os, time, math, json, argparse

# ── path setup (before any cudaq import) ─────────────────────────────────────
_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_IMPL = os.path.dirname(os.path.abspath(__file__))
for _p in [_SP, _IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_IMPL)

# ── cudaq MPI env (must be before import cudaq) ───────────────────────────────
os.environ.setdefault('CUDAQ_MPI_COMM_LIB',
    f'{_SP}/distributed_interfaces/libcudaq_distributed_interface_mpi.so')
os.environ.setdefault('CUDAQ_MGPU_LIB_MPI',
    '/opt/cray/pe/mpich/8.1.28/ofi/gnu/10.3/lib/libmpi.so')
os.environ.setdefault('CUDAQ_MGPU_COMM_PLUGIN_TYPE', 'MPICH')

import cudaq
import numpy as np

# Use SLURM_PROCID to identify rank before cudaq.mpi is available
rank = int(os.environ.get('SLURM_PROCID', 0))
ntasks = int(os.environ.get('SLURM_NTASKS', 1))

cudaq.set_target('nvidia', option='mgpu,fp32')

from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

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
                            cost_norm=cost_norm, noise_model=noise_model)
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
    out = f'/kfs3/scratch/nsawant/bench_mgpu_dqa_{"noisy" if args.noisy else "ideal"}_gpu{ntasks}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out}')

cudaq.mpi.finalize()
