"""
Multi-GPU memory and target verification for n_y = 16 (32 qubits).

Tests:
  fp32  -- nvidia (complex<float>, ~32 GB, single GPU)   should fit H100
  fp64  -- nvidia-fp64 (complex<double>, ~64 GB, 1 GPU)  might OOM
  mgpu  -- nvidia target with mgpu,fp32 option (2 GPUs via cuStateVec+MPI)

Usage (on Kestrel — must use wrapper to set GTL preload BEFORE python starts):
    srun --jobid=<ID> -N1 --ntasks=2 --gpus-per-node=2 \
        /kfs3/scratch/nsawant/run_mgpu.sh test_mgpu.py mgpu

    # single-GPU modes can run without the wrapper:
    srun ... --ntasks=1 --gpus-per-node=1 python test_mgpu.py fp32
    srun ... --ntasks=1 --gpus-per-node=1 python test_mgpu.py fp64

Cray-MPICH GTL notes:
    mgpu uses cuStateVec's MPI inter-GPU swap, which passes GPU pointers to
    MPI_Isend/Irecv. Cray MPICH's default SHM path calls process_vm_readv on
    those pointers and crashes. Fix = GPU-aware MPI via GTL:
      LD_PRELOAD=.../libmpi_gtl_cuda.so  MPICH_GPU_SUPPORT_ENABLED=1
    Plus libcudart must be findable: LD_LIBRARY_PATH=/nopt/cuda/12.4/lib64:...
    All three are set in run_mgpu.sh.
"""
import time, sys, os

_SP = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
if _SP not in sys.path:
    sys.path.insert(0, _SP)

_MPI_PLUGIN = f'{_SP}/distributed_interfaces/libcudaq_distributed_interface_mpi.so'
os.environ.setdefault('CUDAQ_MPI_COMM_LIB', _MPI_PLUGIN)

# mgpu uses cuStateVec's own MPI comm plugin, separately from CUDAQ_MPI_COMM_LIB.
# By default it looks for 'libmpi.so'. On Cray we must point it explicitly.
_CRAY_LIBMPI = '/opt/cray/pe/mpich/8.1.28/ofi/gnu/10.3/lib/libmpi.so'
os.environ.setdefault('CUDAQ_MGPU_LIB_MPI', _CRAY_LIBMPI)
# Tell cuStateVec to use MPICH-compatible plugin mode
os.environ.setdefault('CUDAQ_MGPU_COMM_PLUGIN_TYPE', 'MPICH')

import cudaq

MODE = sys.argv[1] if len(sys.argv) > 1 else 'fp32'
N_SHOTS = 64
N = 32  # 2 * n_y for n_y=16

print(f"Mode={MODE}  cudaq {cudaq.__version__}")

if MODE == 'fp32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudaq.set_target('nvidia')
    print(f"Target: {cudaq.get_target().name}  (complex<float>, ~32 GB)")

elif MODE == 'fp64':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudaq.set_target('nvidia-fp64')
    print(f"Target: {cudaq.get_target().name}  (complex<double>, ~64 GB)")

elif MODE == 'mgpu':
    # mgpu internally initialises MPI via cuStateVec — do NOT call cudaq.mpi.initialize()
    # first or it will conflict. Let set_target handle MPI init.
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    cudaq.set_target('nvidia', option='mgpu,fp32')
    print(f"Target: {cudaq.get_target().name}  (mgpu fp32, 2 GPUs)")

else:
    sys.exit(f"Unknown mode: {MODE}")


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        cx(q[i], q[i + 1])
    mz(q)


print(f"Simulating {N}-qubit GHZ, {N_SHOTS} shots ...")
t0 = time.perf_counter()
result = cudaq.sample(ghz, N, shots_count=N_SHOTS)
elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f} s  most_probable={result.most_probable()}")

if MODE == 'mgpu':
    cudaq.mpi.finalize()
