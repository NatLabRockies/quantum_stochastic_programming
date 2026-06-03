"""
Template: validate a CUDA-Q gate kernel against a reference unitary.

Usage
-----
Replace `reference_unitary` with the matrix from your source framework
(Qiskit, Cirq, PennyLane, …) and `my_cudaq_kernel` with your kernel.
Run with the `qpp-cpu` target (no GPU needed).
"""
import cudaq
import numpy as np

cudaq.set_target("qpp-cpu")

# 1. Reference unitary from your source framework (2-qubit example)
#    e.g. from Qiskit:  np.array(SomeGate().to_matrix())
#    e.g. from Cirq:    cirq.unitary(cirq.SomeGate())
reference_unitary = np.eye(4)  # replace with your gate matrix

# 2. CUDA-Q kernel under test
@cudaq.kernel
def my_cudaq_kernel(param: float, q0: cudaq.qubit, q1: cudaq.qubit):
    # ... your implementation ...
    pass

# 3. Extract CUDA-Q unitary
param_value = 0.5   # use a non-trivial parameter value
cudaq_unitary = np.array(cudaq.get_unitary(my_cudaq_kernel, param_value))

# 4. Compare (ignore global phase by comparing absolute values)
assert cudaq_unitary.shape == reference_unitary.shape, "Shape mismatch"
assert np.allclose(np.abs(cudaq_unitary), np.abs(reference_unitary), atol=1e-6), (
    f"Gate unitary mismatch!\n"
    f"CUDA-Q:\n{cudaq_unitary}\n"
    f"Reference:\n{reference_unitary}"
)
print("Gate validation PASSED")

# 5. Bit-ordering check: prepare |01> and assert correct bitstring
@cudaq.kernel
def prep_01():
    q = cudaq.qvector(2)
    x(q[1])  # flip qubit 1 → state |01> (q0=0, q1=1)

counts = cudaq.sample(prep_01, shots_count=100)
most_common = max(counts, key=counts.__getitem__)
# In CUDA-Q big-endian convention: q0 is leftmost → expect "01"
assert most_common == "01", (
    f"Bit-ordering mismatch: got '{most_common}', expected '01'. "
    "Check qubit index ↔ bitstring position for this CUDA-Q version."
)
print("Bit-ordering check PASSED")
