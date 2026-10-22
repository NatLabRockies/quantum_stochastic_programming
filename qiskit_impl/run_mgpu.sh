#!/bin/bash
# Optional launcher wrapper; configure MPI/GPU-specific library variables in the
# caller's environment when the CUDA-Q installation requires them.
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$@"
