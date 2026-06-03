#!/bin/bash
# Wrapper for cudaq nvidia-mgpu on Kestrel H100 nodes.
# Must be launched via srun so LD_PRELOAD takes effect on compute nodes (not login):
#   srun -N1 --ntasks=2 --gpus-per-node=2 ./run_mgpu.sh <script.py> [args...]
#
# Required for Cray MPICH GPU-aware transport (GTL):
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=/nopt/cuda/12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=/opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_cuda.so

PYTHON=/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/bin/python3
exec "$PYTHON" "$@"
