#!/bin/bash
# run_mpi_bench.sh — launch multi-node CUDA-Q mqpu benchmark
# Must be run inside an existing salloc with >= 2 nodes, 2 GPUs each.
#
# Usage:
#   bash run_mpi_bench.sh
#   (or: source it from within the srun terminal)

PYTHON=/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/bin/python
SCRIPT=/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl/par_benchmark_mpi.py
LOG=/tmp/par_bench_mpi.log

echo "Nodes in allocation: $SLURM_JOB_NODELIST"
echo "Total GPUs: $((SLURM_NNODES * 2))"
echo "Launching with srun: $SLURM_NNODES nodes, 2 tasks/node, 1 GPU/task"
echo ""

srun \
  --nodes=2 \
  --ntasks-per-node=2 \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  $PYTHON $SCRIPT 2>&1 | tee $LOG

echo ""
echo "Log saved → $LOG"
