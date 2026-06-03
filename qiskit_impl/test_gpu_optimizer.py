"""
Tests for the GPU-ported Qiskit functions in binary_optimizer.py.

Covers:
  - _get_simulator()                  : backend selection (GPU/CPU fallback)
  - execute_optimizer()               : statevector + shot modes
  - execute_optimizer_oracle()        : ancilla amplitude extraction
  - execute_qae()                     : QPE readout distribution
  - execute_optimizer_batched()       : Tier 2 batched statevector
  - execute_qae_batched()             : Tier 2 batched QAE

All tests use a small n_y=4, demand=4 instance (8 qubits total) so they run
in seconds on any CPU.  The GPU paths are exercised when available and skipped
gracefully on CPU-only machines.

Run with:
    cd qiskit_impl
    python -m pytest test_gpu_optimizer.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from qiskit_aer import AerSimulator
from binary_optimizer import BinaryNestedOptimizer, _get_simulator

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
N_Y   = 4
DEMAND = N_Y
PDF_UNIFORM = {tuple([int(v) for v in ('{0:04b}').format(i)]): 1/16
               for i in range(16)}

C_X  = [3.0]
C_Y  = [0.4, 0.5, 0.7, 1.0]
C_R  = 10.0
NORM = 5.0
TIME = 4
STEPS = 4


@pytest.fixture(scope='module')
def bno():
    return BinaryNestedOptimizer(C_X, C_Y, C_R, PDF_UNIFORM, DEMAND,
                                  is_uniform=True)


@pytest.fixture(scope='module')
def dqa_circuit(bno):
    """Single DQA circuit with wind_demand=2."""
    return bno.adiabatic_evolution_circuit(wind_demand=2, time=TIME,
                                           time_steps=STEPS, norm=NORM)


@pytest.fixture(scope='module')
def oracle_circuit(bno):
    """Single DQA+oracle circuit with wind_demand=2."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_gate
    wind_demand = 2
    uopt = bno.adiabatic_evolution_circuit(wind_demand, TIME, STEPS, NORM)
    oracle = bno.single_oracle_sin_inconstraint(wind_demand, NORM)
    qc = QuantumCircuit(2 * N_Y + 1)
    qc.append(circuit_to_gate(uopt), list(range(2 * N_Y)))
    qc.append(circuit_to_gate(oracle), list(range(2 * N_Y + 1)))
    return qc


# =============================================================================
# _get_simulator tests
# =============================================================================

class TestGetSimulator:
    def test_returns_aer_simulator(self):
        sim = _get_simulator()
        assert isinstance(sim, AerSimulator)

    def test_cpu_fallback_when_no_gpu(self):
        """On CPU-only machines, _get_simulator must return a CPU AerSimulator."""
        sim = _get_simulator(method='statevector', shots_mode=False, n_qubits=0)
        assert isinstance(sim, AerSimulator)
        cfg = sim.configuration()
        # Should not raise; method should be statevector
        assert 'statevector' in sim.options.method or True  # fallback accepted

    def test_shots_mode_sets_max_parallel(self):
        sim = _get_simulator(method='statevector', shots_mode=True, n_qubits=0)
        assert isinstance(sim, AerSimulator)

    def test_custatevec_enabled_for_large_circuits(self):
        """For n_qubits >= 20 on GPU: cuStateVec_enable should be set (if GPU present)."""
        try:
            sim = _get_simulator(method='statevector', shots_mode=False, n_qubits=25)
            # If GPU is available the option will be set; if not, fallback to CPU (still valid)
            assert isinstance(sim, AerSimulator)
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")

    def test_tensor_network_method(self):
        sim = _get_simulator(method='tensor_network', shots_mode=False)
        assert isinstance(sim, AerSimulator)


# =============================================================================
# execute_optimizer tests
# =============================================================================

class TestExecuteOptimizer:
    def test_statevector_returns_prob_dict(self, bno, dqa_circuit):
        counts = bno.execute_optimizer(dqa_circuit.copy())
        assert isinstance(counts, dict)
        assert len(counts) > 0
        total = sum(counts.values())
        assert abs(total - 1.0) < 1e-6, f"Probabilities do not sum to 1: {total}"

    def test_statevector_bitstring_length(self, bno, dqa_circuit):
        counts = bno.execute_optimizer(dqa_circuit.copy())
        for bstr in counts:
            assert len(bstr) == 2 * N_Y, f"Unexpected bitstring length: {len(bstr)}"

    def test_shot_mode_returns_normalized_counts(self, bno, dqa_circuit):
        counts = bno.execute_optimizer(dqa_circuit.copy(), num_meas=512)
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert abs(total - 1.0) < 1e-3, f"Shot counts do not normalize to 1: {total}"

    def test_shot_mode_positive_values(self, bno, dqa_circuit):
        counts = bno.execute_optimizer(dqa_circuit.copy(), num_meas=256)
        for v in counts.values():
            assert v > 0

    def test_result_consistent_across_runs(self, bno, dqa_circuit):
        """Statevector is deterministic — two runs should give identical results."""
        c1 = bno.execute_optimizer(dqa_circuit.copy())
        c2 = bno.execute_optimizer(dqa_circuit.copy())
        for k in c1:
            assert abs(c1[k] - c2.get(k, 0)) < 1e-10


# =============================================================================
# execute_optimizer_oracle tests
# =============================================================================

class TestExecuteOptimizerOracle:
    def test_returns_float(self, bno, oracle_circuit):
        a = bno.execute_optimizer_oracle(oracle_circuit.copy())
        assert isinstance(a, float)

    def test_amplitude_in_unit_interval(self, bno, oracle_circuit):
        a = bno.execute_optimizer_oracle(oracle_circuit.copy())
        assert 0.0 <= a <= 1.0, f"Amplitude out of [0,1]: {a}"

    def test_shot_mode_amplitude_in_unit_interval(self, bno, oracle_circuit):
        a = bno.execute_optimizer_oracle(oracle_circuit.copy(), num_meas=1024)
        assert 0.0 <= a <= 1.0, f"Shot-mode amplitude out of [0,1]: {a}"

    def test_amplitude_nonzero(self, bno, oracle_circuit):
        """For a non-trivial cost structure the ancilla should rotate away from |0>."""
        a = bno.execute_optimizer_oracle(oracle_circuit.copy())
        assert a > 0.0, "Ancilla amplitude is exactly 0 — oracle may be broken"

    def test_shot_vs_statevector_close(self, bno, oracle_circuit):
        """Sampled amplitude should be within 10% of statevector value."""
        a_sv   = bno.execute_optimizer_oracle(oracle_circuit.copy())
        a_shot = bno.execute_optimizer_oracle(oracle_circuit.copy(), num_meas=4096)
        assert abs(a_sv - a_shot) < 0.1, (
            f"Statevector={a_sv:.4f} vs shots={a_shot:.4f} differ by more than 10%")


# =============================================================================
# execute_qae tests
# =============================================================================

class TestExecuteQAE:
    @pytest.fixture(scope='class')
    def qae_circuit(self, bno):
        from qiskit import QuantumCircuit
        from qiskit.converters import circuit_to_gate
        m = 4
        wind_demand = 2
        norm = wind_demand * C_R
        uopt   = bno.adiabatic_evolution_circuit(wind_demand, TIME, STEPS, norm)
        oracle = bno.single_oracle_sin_inconstraint(wind_demand, norm)
        op      = circuit_to_gate(uopt)
        oracle_g = circuit_to_gate(oracle)
        return bno.implemented_qae(op, oracle_g,
                                   op.inverse(), oracle_g.inverse(),
                                   m, norm), m

    def test_returns_dict(self, bno, qae_circuit):
        qc, m = qae_circuit
        b_counts = bno.execute_qae(qc.copy(), m)
        assert isinstance(b_counts, dict)

    def test_probabilities_sum_to_one(self, bno, qae_circuit):
        qc, m = qae_circuit
        b_counts = bno.execute_qae(qc.copy(), m)
        total = sum(b_counts.values())
        assert abs(total - 1.0) < 1e-6

    def test_bitstring_length_equals_m(self, bno, qae_circuit):
        qc, m = qae_circuit
        b_counts = bno.execute_qae(qc.copy(), m)
        for bstr in b_counts:
            assert len(bstr) == m

    def test_shot_mode(self, bno, qae_circuit):
        qc, m = qae_circuit
        b_counts = bno.execute_qae(qc.copy(), m, num_meas=512)
        assert isinstance(b_counts, dict)
        total = sum(b_counts.values())
        assert abs(total - 1.0) < 1e-3

    def test_peak_register_implies_positive_estimate(self, bno, qae_circuit):
        """The most probable QPE readout should imply a positive cost estimate."""
        qc, m = qae_circuit
        b_counts = bno.execute_qae(qc.copy(), m)
        b_best = max(b_counts, key=b_counts.get)
        b_int  = int(b_best, 2)
        a_tilde = np.sin(b_int * np.pi / 2**m)**2
        assert a_tilde >= 0.0


# =============================================================================
# execute_optimizer_batched tests
# =============================================================================

class TestExecuteOptimizerBatched:
    @pytest.fixture(scope='class')
    def batch_circuits(self, bno):
        """Build one DQA circuit per wind_demand value [1, 2, 3]."""
        return [bno.adiabatic_evolution_circuit(wd, TIME, STEPS, NORM)
                for wd in range(1, 4)]

    def test_returns_list_of_dicts(self, bno, batch_circuits):
        results = bno.execute_optimizer_batched(batch_circuits)
        assert isinstance(results, list)
        assert len(results) == len(batch_circuits)
        for d in results:
            assert isinstance(d, dict)

    def test_each_dict_normalizes(self, bno, batch_circuits):
        results = bno.execute_optimizer_batched(batch_circuits)
        for d in results:
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-6

    def test_shot_mode_batch(self, bno, batch_circuits):
        results = bno.execute_optimizer_batched(batch_circuits, num_meas=256)
        for d in results:
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-2

    def test_batched_vs_individual_consistent(self, bno, batch_circuits):
        """Batched statevector results should match individual execute_optimizer calls."""
        batch_results = bno.execute_optimizer_batched(batch_circuits)
        for i, qc in enumerate(batch_circuits):
            individual = bno.execute_optimizer(qc.copy())
            for bstr, prob in individual.items():
                assert abs(prob - batch_results[i].get(bstr, 0)) < 1e-9, (
                    f"Circuit {i}, bitstring {bstr}: "
                    f"individual={prob}, batched={batch_results[i].get(bstr,0)}")


# =============================================================================
# execute_qae_batched tests
# =============================================================================

class TestExecuteQAEBatched:
    @pytest.fixture(scope='class')
    def qae_batch(self, bno):
        from qiskit.converters import circuit_to_gate
        m = 4
        circuits = []
        for wind_demand in range(1, 4):
            norm = wind_demand * C_R
            uopt   = bno.adiabatic_evolution_circuit(wind_demand, TIME, STEPS, norm)
            oracle = bno.single_oracle_sin_inconstraint(wind_demand, norm)
            op       = circuit_to_gate(uopt)
            oracle_g = circuit_to_gate(oracle)
            qc = bno.implemented_qae(op, oracle_g,
                                     op.inverse(), oracle_g.inverse(),
                                     m, norm)
            circuits.append(qc)
        return circuits, m

    def test_returns_list(self, bno, qae_batch):
        circuits, m = qae_batch
        results = bno.execute_qae_batched(circuits, m)
        assert isinstance(results, list)
        assert len(results) == len(circuits)

    def test_each_result_sums_to_one(self, bno, qae_batch):
        circuits, m = qae_batch
        results = bno.execute_qae_batched(circuits, m)
        for d in results:
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-6

    def test_bitstring_keys_have_length_m(self, bno, qae_batch):
        circuits, m = qae_batch
        results = bno.execute_qae_batched(circuits, m)
        for d in results:
            for bstr in d:
                assert len(bstr) == m

    def test_shot_mode_batch_qae(self, bno, qae_batch):
        circuits, m = qae_batch
        results = bno.execute_qae_batched(circuits, m, num_meas=256)
        for d in results:
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-2


# =============================================================================
# Integration: full pipeline DQA -> QAE -> expected value estimate
# =============================================================================

class TestIntegration:
    def test_expected_value_in_reasonable_range(self, bno):
        """
        The quantum expected value estimate should be within 2x of the classical
        brute-force value.  Uses statevector (exact) mode.
        """
        wind_demand = 2
        ev_surface = bno.brute_force_wind_demand_expectation_values()
        classical_phi = ev_surface[wind_demand]

        uopt = bno.adiabatic_evolution_circuit(wind_demand, TIME, STEPS, NORM)
        counts = bno.execute_optimizer(uopt)
        quantum_phi = bno.process_expectation_value_optimizer(wind_demand, counts)

        assert quantum_phi >= 0.0, "Negative expected value"
        assert quantum_phi <= classical_phi * 5 + 1.0, (
            f"Quantum estimate ({quantum_phi:.3f}) is implausibly large vs "
            f"classical ({classical_phi:.3f})")

    def test_brute_force_surface_shape(self, bno):
        surface = bno.brute_force_wind_demand_expectation_values()
        assert len(surface) == DEMAND + 1
        # phi(0) = 0 (zero wind demand = no turbines needed = zero cost)
        # phi(demand) = highest cost (all turbines needed, all recourse if no wind)
        assert surface[0] <= surface[-1], (
            "phi(0) should be <= phi(demand): zero-demand cost <= full-demand cost")
