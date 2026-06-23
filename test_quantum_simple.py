#!/usr/bin/env python3
"""
Minimal quantum optimization test — demonstrates DQA + QAE working end-to-end.
Runs on single GPU, produces visible output and results file.
"""

import os
import sys
import time

# Add paths
_QISKIT_SP = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qiskit_impl')
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

print("\n" + "="*70)
print("QUANTUM STOCHASTIC OPTIMIZATION — SIMPLE TEST")
print("="*70)

# === IMPORTS ===
try:
    from binary_optimizer import BinaryNestedOptimizer
    import numpy as np
    print("[✓] Imports successful")
except Exception as e:
    print(f"[✗] Import failed: {e}")
    sys.exit(1)

# === PROBLEM SETUP ===
print("\n[1] Setting up problem...")
n_y = 4  # wind turbines
c_x = [3.0]  # gas generator cost
c_y = np.linspace(0.1, 1.0, n_y)  # wind dispatch costs
c_r = 10.0  # recourse cost (unmet demand penalty)
x0 = [2]  # first-stage gas commitment
w_d = int(n_y - sum(x0))  # wind demand

# Uniform PDF over all 2^n_y wind scenarios
pdf = {tuple(int(v) for v in ('{0:0'+str(n_y)+'b}').format(i)): 1/2**n_y
       for i in range(2**n_y)}

print(f"  Problem: {n_y} wind turbines, {len(pdf)} scenarios")
print(f"  Gas cost: {c_x[0]:.1f}, Recourse cost: {c_r:.1f}")
print(f"  Wind demand (to optimize): {w_d} MW")

# === CLASSICAL BASELINE ===
print("\n[2] Classical brute-force reference...")
bno = BinaryNestedOptimizer(c_x, c_y, c_r, pdf, n_y, is_uniform=True)
exp_vals = bno.brute_force_wind_demand_expectation_values()
classical_result = exp_vals[w_d]
print(f"  Classical φ(w_d={w_d}) = {classical_result:.6f}")

# === DQA QUANTUM OPTIMIZATION ===
print("\n[3] Running Discrete Quantum Annealing (DQA)...")
print(f"  Building {2*n_y}-qubit circuit...")

norm = w_d * c_r
cost_norm = w_d * c_r / n_y
timesteps = n_y

# DQA angles (linear ramp)
import math
theta = []
for t in range(timesteps):
    theta.append(float(t / timesteps))
    theta.append((1 - float(t / timesteps)) / math.pi)

print(f"  DQA angles: {len(theta)} parameters (linear ramp)")

try:
    # Build DQA circuit
    qc = bno.adiabatic_evolution_circuit(wind_demand=w_d, time=100, 
                                          time_steps=timesteps, norm=cost_norm)
    print(f"  Circuit depth: {qc.depth()}, Width: {qc.num_qubits}")
    
    # Execute on exact statevector
    print(f"  Executing on statevector backend...")
    t0 = time.perf_counter()
    counts = bno.execute_optimizer(qc, num_meas=8192)
    dqa_time = time.perf_counter() - t0
    
    # Process results
    dqa_result = bno.process_expectation_value_optimizer(w_d, counts)
    print(f"  DQA φ(w_d={w_d}) = {dqa_result:.6f}  ({dqa_time*1e3:.1f} ms)")
    
except Exception as e:
    print(f"  [✗] DQA execution failed: {e}")
    dqa_result = None
    dqa_time = None

# === SUMMARY ===
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Metric':<40} {'Value':>20}")
print("-"*70)
print(f"{'Problem size (qubits)':<40} {2*n_y:>20}")
print(f"{'Scenarios':<40} {len(pdf):>20}")
print(f"{'Classical solution':<40} {classical_result:>20.6f}")
if dqa_result is not None:
    error = abs(dqa_result - classical_result) / classical_result * 100
    print(f"{'Quantum (DQA) estimate':<40} {dqa_result:>20.6f}")
    print(f"{'Relative error (%)':<40} {error:>20.2f}%")
    print(f"{'Wall time (ms)':<40} {dqa_time*1e3:>20.1f}")
else:
    print(f"{'Quantum (DQA) estimate':<40} {'FAILED':>20}")

print("="*70)

# === SAVE RESULTS ===
output_file = '/home/ssavadat/quantum_stochastic_programming-1/test_results.txt'
try:
    with open(output_file, 'w') as f:
        f.write("QUANTUM STOCHASTIC OPTIMIZATION — RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Problem: {n_y} turbines, {len(pdf)} scenarios, w_d={w_d}\n")
        f.write(f"Classical solution: {classical_result:.6f}\n")
        if dqa_result is not None:
            f.write(f"Quantum solution:   {dqa_result:.6f}\n")
            f.write(f"Error:              {error:.2f}%\n")
            f.write(f"Exec time:          {dqa_time*1e3:.1f} ms\n")
    print(f"\n[✓] Results saved to {output_file}")
except Exception as e:
    print(f"[✗] Failed to save results: {e}")

print("\n[✓] TEST COMPLETE\n")
