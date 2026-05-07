"""
Multi-node parallelisation benchmark using mpi4py + CUDA-Q single-GPU per rank.

Each MPI rank owns one GPU (set via SLURM_LOCALID -> CUDA_VISIBLE_DEVICES).
Rank 0 times the parallel shot sampling and gathers results via MPI.

Launch from within a salloc (2 nodes, 2 GPUs each = 4 ranks total):

    srun --jobid=<JOBID> -N 2 --ntasks-per-node=2 --gpus-per-node=2 \
        /nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/bin/python \
        par_benchmark_mpi.py 2>&1 | tee /tmp/par_bench_mpi.log
"""
import os, sys, math, time, json

# GPU binding MUST happen before any CUDA/cudaq import
_local_id = int(os.environ.get('SLURM_LOCALID', 0))
os.environ['CUDA_VISIBLE_DEVICES'] = str(_local_id)

_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

import numpy as np
from mpi4py import MPI
import cudaq
from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

cudaq.set_target('nvidia')   # single GPU per rank

# Identity check (serialised)
comm.Barrier()
for r in range(size):
    if rank == r:
        print(f'  rank {rank}: node={MPI.Get_processor_name()}  '
              f'GPU={os.environ["CUDA_VISIBLE_DEVICES"]}  '
              f'cudaq_gpus={cudaq.num_available_gpus()}', flush=True)
    comm.Barrier()

N_SHOTS   = 256
TIMESTEPS = 20
P1, P2    = 0.0001, 0.001
C_R, C_X  = 10.0, [3.]
NY_LIST   = [16]   # n_y=16: 32 qubits, fp32 = 32 GB/trajectory — fits H100 80 GB

noise_model = build_depolarizing_noise_model(p1=P1, p2=P2)

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
    print(f'\n── 4-GPU (2 nodes x 2 GPUs) mpi4py benchmark ──')
    print(f'   ranks={size}  N_SHOTS={N_SHOTS}  ny={NY_LIST}\n')

results_4gpu = {}

for ny in NY_LIST:
    opt, theta0 = make_opt(ny)

    # Divide shots across ranks (round-robin remainder)
    base      = N_SHOTS // size
    my_shots  = base + (1 if rank < (N_SHOTS % size) else 0)

    # All ranks run their local GPU simulation in parallel
    comm.Barrier()
    t0 = time.perf_counter()
    local_probs = opt.sample_ansatz(theta0, shots=my_shots)
    comm.Barrier()
    dt = time.perf_counter() - t0   # wall time = slowest rank

    # Convert probs -> raw counts, gather on rank 0
    local_raw = {k: round(v * my_shots) for k, v in local_probs.items()}
    all_dicts = comm.gather(local_raw, root=0)

    if rank == 0:
        merged = {}
        for d in all_dicts:
            for bitstr, cnt in d.items():
                merged[bitstr] = merged.get(bitstr, 0) + cnt
        results_4gpu[ny] = round(dt, 2)
        print(f'  n_y={ny:2d}  qubits={2*ny:2d}  {dt:.1f}s', flush=True)

if rank == 0:
    print(f"\n{'n_y':>4}  {'qubits':>6}  {'4-GPU(s)':>10}")
    print('-' * 28)
    for ny in NY_LIST:
        if ny in results_4gpu:
            print(f"{ny:>4}  {2*ny:>6}  {results_4gpu[ny]:>10.1f}")

    fname = os.path.join(_QISKIT_IMPL, 'par_benchmark_mpi_results.json')
    with open(fname, 'w') as f:
        json.dump({'n_shots': N_SHOTS, 'n_gpus': size,
                   'timings': {str(ny): results_4gpu[ny]
                               for ny in NY_LIST if ny in results_4gpu}}, f, indent=2)
    print(f'\nResults -> {fname}')

MPI.Finalize()
