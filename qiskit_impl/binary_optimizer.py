
import sys
from qiskit_impl import circuits as circuit_builders
from qiskit_impl.dist_prep import build_pdf_circuit
from qiskit_impl.qae import build_qae_circuit
from classical import (
    first_stage_objective,
    scenario_cost,
    solve_scenario,
    wind_demand_costs,
)

import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.library.standard_gates import RYGate
try:
    from qiskit_algorithms import EstimationProblem  # optional: pip install qiskit-algorithms
except ImportError:
    EstimationProblem = None  # canonical_qae() will be unavailable without qiskit-algorithms
from qiskit.converters import circuit_to_gate

# ---------------------------------------------------------------------------
# GPU backend helper
# ---------------------------------------------------------------------------
def _get_simulator(method: str = 'statevector', shots_mode: bool = False,
                   n_qubits: int = 0) -> object:
    """Return an AerSimulator targeting GPU if available, else CPU.

    Args:
        method: Aer simulation method ('statevector' or 'matrix_product_state').
        shots_mode: If True, add max_parallel_shots=0 for GPU shot parallelism.
        n_qubits: Circuit width hint.  When >= 20 on GPU, enables cuStateVec
                  (NVIDIA cuQuantum) for large-statevector acceleration.
    """
    try:
        options = {'method': method, 'device': 'GPU'}
        if shots_mode:
            options['max_parallel_shots'] = 0
        # Tier 3: enable cuStateVec for large circuits (>= 20 qubits)
        if method == 'statevector' and n_qubits >= 20:
            options['cuStateVec_enable'] = True
        sim = AerSimulator(**options)
        # Probe GPU availability — raises if no CUDA device found
        sim.available_devices()
        if sys.platform == 'darwin':
            raise RuntimeError('Mac OS!')
        ## end if
        return sim
    except Exception:
        # Fall back to CPU statevector
        options = {'method': method, 'device': 'CPU'}
        return AerSimulator(**options)
# ---------------------------------------------------------------------------


# Let's turn our problem into a bernouli estimation problem
class BernoulliA(QuantumCircuit):
    """A circuit representing the Bernoulli A operator."""

    def __init__(self, uopt, oracle, n):
        super().__init__(2*n+1)  # circuit on 1 qubit
        self.system_qubits = list(range(n))
        self.pdf_qubits = list(range(n, 2*n))
        self.ancilla_qubit = 2*n

        self.append(uopt, self.system_qubits + self.pdf_qubits)
        self.append(oracle, self.system_qubits + self.pdf_qubits + [self.ancilla_qubit])

def sample_wvfn(probability_dict):
    ''' given a probability vector (ie wvfn squared), return one measurement '''
    i = 0.
    rv = random.random()
    for bstr, val in probability_dict.items():
        i+=val 
        if i >= rv:
            return bstr



# NOTE: for now, we just assume one gas generator with integer output
class BinaryNestedOptimizer:
    """
    Quantum + classical solver for the two-stage stochastic unit commitment (UC) problem.

    Problem structure
    -----------------
    First stage  (x) : choose how much power the gas generator provides
                       (integer, 0 <= x <= demand).  Cost = x * gas_cost.
    Second stage (y) : given x and a *realized* wind scenario xi, choose which
                       wind turbines to turn on (y in {0,1}^n_y, sum(y) = demand - x).
                       Cost = sum_j c_y[j]*y[j]*xi[j]  +  recourse*(shortfall)
    Objective        : minimize  gas_cost(x) + E_xi[ min_y q(y, xi) ]

    Quantum qubit layout (used consistently across ALL circuits)
    ------------------------------------------------------------
    wind_qubits  [0 .. n_y-1]           : y register  -- binary turbine ON/OFF decisions
    pdf_qubits   [n_y .. 2*n_y-1]       : xi register  -- wind scenario, loaded with PDF amplitudes
    ancilla      [2*n_y]                : oracle ancilla for QAE
                                          |0> ->  sqrt(1-q?)|0> + sqrtq?|1>   (cost encoded as amplitude)
    estimate     [2*n_y+1 .. +m]        : m QPE readout qubits  (QAE output: integer b)

    The DQA (Discrete Quantum Annealing) circuit evolves the system from
    the Dicke state (uniform superposition over feasible y) toward the
    optimal state by interpolating between the XY mixer and the cost operator.
    """
    def __init__(self, gas_costs: list[float], wind_costs: list[float], recourse_cost: float, pdf: dict[tuple[int], float], demand: int, is_uniform: bool):
        self.gas_costs = gas_costs
        self.num_gas_vars = len(gas_costs)
        self.wind_costs = wind_costs
        self.num_wind_vars = len(wind_costs)
        self.recourse_cost = recourse_cost
        self.pdf = pdf
        self.is_uniform = is_uniform
        self.demand = demand

        assert(np.isclose(sum(pdf.values()), 1.))
        for scenario,value in pdf.items():
            assert(len(scenario) == len(wind_costs))

        self.all_wind_outputs_list = None

        # qc stuff
        self.wind_qubits = list(range(self.num_wind_vars))
        self.pdf_qubits = [self.num_wind_vars+i for i in self.wind_qubits]

    ## Helpers
    def gen_all_wind_outputs(self):
        ''' create a list of all bitstrings possible for the wind bitstrings
        '''
        bstrs = []
        for i in range(2**self.num_wind_vars):
            bstr = ('{0:0' + str(self.num_wind_vars) + 'b}').format(i)
            bstrs.append(self.bstr_to_output(bstr))
        self.all_wind_outputs_list = bstrs
    
    def wind_scenario_cost(self, wind_output: list[int], scenario: tuple[int], wind_demand: int) -> float:
        """
        Given a list of wind outputs and the corresponding scenario, return the cost of that scenario.

        Compute second-stage cost  q(y, xi)  for one turbine decision and one wind scenario.

        For each turbine j:
          effective_output[j] = wind_output[j] AND scenario[j]
            -> turbine j generates power only if it is ON (y[j]=1) AND wind blows (xi[j]=1)
          cost contribution   = wind_costs[j] * effective_output[j]

        If total effective output < wind_demand:
          shortfall = wind_demand - sum(effective_output)
          recourse  = shortfall * recourse_cost   <- expensive backup power purchase

        This is q(x, y, xi) from Eq. (2) of the paper.
        """
        return scenario_cost(
            wind_output, scenario, wind_demand, self.wind_costs, self.recourse_cost
        )

    def bstr_to_output(self, bstr):
        ''' convert a bitstring to the list of integers '''
        return [int(i) for i in bstr]

    def prep_gs(self, wind_demand):
        """
        Create a statevector which is the minima for each scenario.

        Build the ideal "ground-state" wavefunction |psi*> -- the state perfect DQA would produce.

        For each scenario xi (with probability Pr[xi]):
          1. Solve  y*(xi) = argmin_{sum(y)=wind_demand} q(y, xi)  classically
          2. Place amplitude  sqrtPr[xi]  on the basis state  |y*(xi)>|xi>

        The resulting statevector satisfies:
          <psi*| H_Q |psi*>  =  E_xi[ min_y q(y,xi) ]  =  phi(x)   (Eq. 20 of paper)

        This is the benchmark target.  Real DQA produces an approximation
        (residual temperature delta > 0 means some amplitude leaks to sub-optimal states).
        """
        if wind_demand < 0 or wind_demand > self.num_wind_vars:
            raise ValueError("Cannot prepare a reference state for infeasible wind demand")
        wvfn = np.zeros(2**(2*self.num_wind_vars))
        # use the min_scenario to get the best decision for each scenario
        for scenario,pr in self.pdf.items():
            sc_cost, sc_decision = self.min_scenario(wind_demand, scenario)
            bstr = ''.join([str(b) for b in scenario[::-1] + sc_decision[::-1]])
            wvfn[int(bstr[::-1],2)] = np.sqrt(pr)
        return wvfn


    ## Classical solvers
    def min_scenario(self, wind_demand, scenario):
        """
        Get the minimization for a single wind scenario.

        Classical brute-force second-stage solver for ONE fixed wind scenario xi.

        Enumerate all turbine decisions y with sum(y) == wind_demand (the feasible set),
        evaluate q(y, xi) for each, and return the cheapest one:

            y*(xi) = argmin_{y : sum(y)=wind_demand}  q(y, xi)

        Returns: (min_cost, optimal_turbine_decision_tuple)
        Infeasible wind demand returns (infinity, ()).
        """
        return solve_scenario(
            wind_demand, scenario, self.wind_costs, self.recourse_cost
        )

    def brute_force_wind_demand_expectation_values(self):
        """Compute the classical exact expected cost for every possible wind demand.

        For each integer `wind_demand` in `[0, demand]`, enumerates all feasible
        turbine decisions and scenarios to compute:

        $$\\phi(k) = \\mathbb{E}_{\\xi}[\\min_{|y|=k} q(y, \\xi)]$$

        This is the classical brute-force ground truth used to benchmark DQA and QAE.

        Returns:
            List of floats `[phi(0), phi(1), ..., phi(demand)]`.
            Demands above the turbine count have infinite cost.
        """
        return wind_demand_costs(
            self.wind_costs, self.recourse_cost, self.pdf, self.demand
        )

    def brute_force_energy_surface(self):
        """Compute the full first-stage objective function over all gas levels.

        For each gas commitment level `x` in `[0, demand]`:

        $$o(x) = x \\cdot c_x + \\phi(\\text{demand} - x)$$

        where `phi` is from `brute_force_wind_demand_expectation_values()`.

        Returns:
            Dict `{gas_level: objective_value}` — the classical optimal is `min(o.values())`.
        """
        return first_stage_objective(
            self.gas_costs[0], self.wind_costs, self.recourse_cost,
            self.pdf, self.demand,
        )


    ## quantum helpers
    def dicke_state_initialize(self, weight):
        """Compatibility wrapper around `dicke_state_circuit()`.

        Returns a gate that prepares $|D_n^k\\rangle$ on `num_wind_vars` qubits
        with `weight` excitations. Delegates to `dicke_state_circuit()`.

        Args:
            weight: Integer $k$ — number of turbines to turn on (Hamming weight).

        Returns:
            Gate for nonzero weight, or a QuantumCircuit for weight zero.
        """
        return self.dicke_state_circuit(weight)
    
    def dicke_state_circuit(self, weight):
        """
        Prepare the Dicke state. Reference: https://arxiv.org/pdf/1904.07358.pdf

        Prepare the Dicke state  |D_n^k>  with n = num_wind_vars qubits and k = weight ones.

        The Dicke state is a uniform superposition over ALL bitstrings with exactly k ones:

            |D_n^k> = 1/sqrtC(n,k)  sum_{|y|=k} |y>

        This is the constraint-respecting initial state for DQA: every basis state in
        the superposition already satisfies sum(y) = wind_demand.

        Construction uses the recursive SCS (Splitting-Coherence-Splitting) gadget from:
            Baertschi & Eidenbenz, arXiv:1904.07358  (Fig. 1)

        SCS(n, k) acts on k+1 qubits and "moves" one unit of weight downward:
          - CX + doubly-controlled RY: redistribute amplitude between |1...1 0> and |1...0 1>
          - Applied (n-k) times top-down, then (k-1) times bottom-up

        Gate count: O(n*k)  with no ancilla, depth O(n).
        """
        qc = circuit_builders.dicke_state_circuit(self.num_wind_vars, weight)
        if weight == 0:
            return qc
        return qc.to_gate()

    def pdf_initialize(self):
        """Build a circuit that encodes the PDF as amplitudes on the xi register.

        Prepares the state $|\\mathcal{P}\\rangle = \\sum_\\xi \\sqrt{\\Pr[\\xi]}|\\xi\\rangle$
        so that measuring the xi register gives scenario $\\xi$ with probability $\\Pr[\\xi]$.

        Two cases:

        - `is_uniform=True`: $n$ Hadamard gates — $O(n)$ circuit depth.
        - `is_uniform=False`: unitary Qiskit `StatePreparation` — $O(2^n)$ gates after transpile.

        Returns:
            Gate object (via `.to_gate()`) acting on `num_wind_vars` qubits.
        """
        return build_pdf_circuit(
            self.num_wind_vars, self.pdf, uniform=self.is_uniform
        ).to_gate()

    def cost_operator(self, amplitude, constraint_amplitude, norm):
        """
        Our cost operator.
        `constraint_amplitude` will be recourse for the constraint-preserving mixer,
        and a penalty weight otherwise.

        Build the cost (phase) operator  U_q(gamma) = exp(-i gamma H_C).

        For each turbine j, applies TWO controlled-phase (CP) gates:

          CP(gamma . c_y[j] / norm)   on [pdf_qubit_j, wind_qubit_j]
            -> adds phase  gamma.c_y[j]/norm  when BOTH y[j]=1 AND xi[j]=1
              (turbine ON and wind available -> incurs operational cost)

          X(pdf_qubit_j)  ->  CP(gamma . recourse / norm)  ->  X(pdf_qubit_j)
            -> adds phase  gamma.recourse/norm  when y[j]=1 AND xi[j]=0
              (turbine ON but no wind -> incurs recourse penalty)

        The CP gate  CP(theta)|11> = e^(itheta)|11>  implements  exp(i.theta.|1><1| (x) |1><1|),
        which is the diagonal cost Hamiltonian acting as a phase kick on computational basis states.

        `amplitude`            = gamma (annealing parameter, 0..1)
        `constraint_amplitude` = recourse cost coefficient
        `norm`                 = normalization factor (prevents phase wrapping)
        Corresponds to U_q in Eq. (42) of the paper.
        """
        return circuit_builders.cost_operator(
            amplitude, self.wind_costs, constraint_amplitude, norm, self.pdf_qubits
        )

    def demand_constraint_preserving_mixer(self, amplitude):
        """
        Mixing operator which preserves the demand constraint.

        Build the XY mixer  U_d(beta) = exp(-i beta H_XY)  that preserves Hamming weight.

        For every pair of qubits (j, k):  applies  SWAP^beta  (a fractional SWAP gate).

        SWAP^beta  interpolates between identity (beta=0) and a full SWAP (beta=1):

            SWAP^beta = exp(-i beta pi/2 . (XX + YY)/2)

        The XX+YY part of the Hamiltonian is the XY model -- it moves excitations
        between qubits without changing the total number of |1>s.  This ensures
        that mixing only connects states with the SAME sum(y) = wind_demand.

        Combined with the Dicke state initializer, this keeps the entire DQA
        evolution within the feasible subspace  {y : sum(y) = wind_demand}.
        Corresponds to U_d (Eq. 39) of the paper.
        """
        return circuit_builders.demand_constraint_preserving_mixer(
            amplitude, self.wind_qubits
        )

    def cost_scenario_operator(self, scenario, amplitude, constraint_amplitude, norm):
        """Build the single-scenario cost (phase) operator for a fixed wind scenario.

        Unlike `cost_operator` (which loops over all scenarios in superposition),
        this version acts only on the y-register and encodes the cost for one
        specific scenario xi deterministically.

        For each turbine j:

        - If `scenario[j] == 0` (no wind): apply `P(amplitude * recourse / norm)` on qubit j.
        - If `scenario[j] == 1` (wind):    apply `P(amplitude * c_y[j] / norm)` on qubit j.

        Used in `adiabatic_evolution_scenario_circuit()` for single-scenario DQA.

        Args:
            scenario: Tuple of 0/1 values encoding the wind scenario xi.
            amplitude: Annealing parameter gamma (0 to 1).
            constraint_amplitude: Recourse cost coefficient c_r.
            norm: Cost normalization factor (prevents phase wrapping).

        Returns:
            QuantumCircuit acting on `num_wind_vars` qubits.
        """
        return circuit_builders.cost_scenario_operator(
            amplitude, self.wind_costs, constraint_amplitude, norm, scenario
        )

    def adiabatic_evolution_circuit(self, wind_demand, time, time_steps, norm):
        """
        Adiabatic evolution; currently only using the constraint-preserving mixer.

        Build the Discrete Quantum Annealing (DQA) circuit  U_DQA(T).

        Implements the alternating-operator ansatz (QAOA-style) with a LINEAR
        annealing schedule  s(t) = t/T  interpolating from mixer-dominated (t=0)
        to cost-dominated (t=T):

            U_DQA = Prod_{t=0}^{T-1}  U_q(s(t))  .  U_d(1 - s(t))

        At t=0: s=0 -> full mixing, no cost -> system stays in Dicke state (uniform)
        At t=T: s=1 -> full cost,  no mixing -> system is "frozen" near cost minimum

        The adiabatic theorem guarantees that if T is large enough relative to
        the inverse spectral gap, the final state is close to the ground state
        of H_C (i.e., the optimal turbine assignment for each scenario).

        Initialization:
          - Dicke state |D_n^{wind_demand}> on wind_qubits
          - PDF state |P> on pdf_qubits  (encodes scenario probabilities)

        Parameters
        ----------
        wind_demand : integer k  (sum(y) must equal this)
        time        : total annealing time T (scales schedule)
        time_steps  : number of Trotter steps (circuit depth)
        norm        : cost normalization (keeps phases in [0, 2pi])
        """
        qc = QuantumCircuit(self.num_wind_vars*2, name='Uopt')
        
        qc.append(self.dicke_state_initialize(wind_demand), self.wind_qubits)
        qc.append(self.pdf_initialize(), self.pdf_qubits)
        for t in range(time_steps):
            dt = time/time_steps
            s = np.float64((dt*t)/time)

            qc.append(circuit_to_gate(self.cost_operator(s, self.recourse_cost, norm)), self.wind_qubits + self.pdf_qubits)
            qc.append(circuit_to_gate(self.demand_constraint_preserving_mixer((1-s)/np.pi)), self.wind_qubits)
        return qc

    def adiabatic_evolution_scenario_circuit(self, wind_demand, time, time_steps, scenario, norm):
        ''' adiabatic evolution; constraint-preserving mixer and only a specific scenario
        '''
        qc = QuantumCircuit(self.num_wind_vars, name='Uopt({})'.format(scenario))

        qc.append(self.dicke_state_initialize(wind_demand), self.wind_qubits)
        qc.barrier()
        for t in range(time_steps):
            dt = time/time_steps
            s = (dt*t)/time

            qc.append(self.cost_scenario_operator(scenario, s, self.recourse_cost, norm), self.wind_qubits)
            qc.append(self.demand_constraint_preserving_mixer(1/np.pi*(1-s)), self.wind_qubits)
        qc.barrier()
        return qc  
    

    #### Set aside a suite of QAE algorithms
    ####
    def implemented_qae(self, op, oracle, op_inv, oracle_inv, m, norm):
        """
        Build the canonical Quantum Amplitude Estimation (QAE) circuit.
        (Brassard et al. 2002, also Fig. 2 of the paper)

        Goal: estimate  a = E_xi[q?(y,xi)]  (the normalized expected cost)
              with error O(1/2^m) using 2^m Grover applications.

        Qubit layout:
          [0 .. m-1]              : m QPE estimate qubits (readout)
          [m .. m+2*n_y-1]        : system qubits (wind + pdf)
          [m+2*n_y]               : ancilla qubit

        Circuit structure:
          1. H(x)m on estimate qubits  ->  superposition |0>+|1>+...+|2^m-1>
          2. A  = oracle . op  applied once on system + ancilla
             Prepares:  sqrt(1-a)|psi_bad>|0> + sqrta|psi_good>|1>
          3. Controlled-Grover repetitions: for i=0..m-1, controlled on estimate qubit i:
               - S_psi0 : flip phase of ancilla=|1> states  (X.CZ.X on ancilla)
               - Adag   : invert oracle then invert op
               - S_0  : flip phase of |00...0>  (X-flip all, MCZ, X-flip back)
               - A    : reapply op then oracle
             Each iteration of j applies this block 2^i times, encoding theta_a
             as a binary fraction in the QPE phase register.
          4. IQFT on estimate qubits  ->  phase -> readable integer b
          Measurement: b -> ? = sin^2(b.pi/2^m) -> phi?(x) = ?.norm
        """
        return build_qae_circuit(
            op, oracle, op_inv, oracle_inv, self.num_wind_vars, m
        )

    def canonical_qae(self, Uopt, oracle, m, c, norm):
        ''' use qiskit tools to setup estimator NOTE still need to setup '''
        A = BernoulliA(circuit_to_gate(Uopt),circuit_to_gate(oracle),self.num_wind_vars)

        problem = EstimationProblem(
                    state_preparation=A,  # A operator
                    objective_qubits=[2*self.num_wind_vars],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
                )
        return problem

        

    def exact_oracle(self, ydemand, norm, inverse=False):
        """
        Compute the exact oracle -- obviously untractable, but helpful for actually determining accuracy.

        Build the exact oracle  F_exact  (Eq. 44 of paper).

        For every (y, xi) basis state that satisfies sum(y) == ydemand:

            F_exact |y>|xi>|0>_anc  =  |y>|xi> . (sqrt(1-q?)|0> + sqrtq?|1>)

        where  q? = q(y,xi)/norm  is the normalized cost.
        After this operation  Pr[ancilla = |1>] = q?.

        Implementation -- "bit-flip sandwich" for each basis state:
          1. Flip X on every qubit where the target bit pattern is 0
             -> now ALL control qubits are |1> for this basis state only
          2. Apply multi-controlled RY(2.arcsin(sqrtq?)) on the ancilla
             -> fires *only* when all controls are |1> = when system is in this exact state
          3. Flip X back on the same qubits  (uncompute)

        This requires O(2^{2n_y}) gates and is impractical at scale, but gives
        the exact result for small n_y.  Used only as a convergence benchmark.
        """
        # let's grab a control operator for each bitstring
        qc = QuantumCircuit(2*self.num_wind_vars+1)
        ancilla = self.num_wind_vars*2
        all_wind_bstr_outputs = {''.join([str(b) for b in output]): output for output in self.all_wind_outputs_list}
        for output_bstr, output in all_wind_bstr_outputs.items():
            if sum([int(b) for b in output_bstr]) != ydemand:
                continue
            for scenario,pr in self.pdf.items():
                scenario_bstr = ''.join([str(s) for s in scenario])
                bstr = scenario_bstr + output_bstr
                cost = self.wind_scenario_cost(output[::-1], scenario[::-1], ydemand)
                theta = 2*np.arcsin(np.sqrt(cost/norm))
                if inverse:
                    theta *= -1
                qc.append(RYGate(theta).control(num_ctrl_qubits=2*self.num_wind_vars, ctrl_state=bstr), self.wind_qubits+self.pdf_qubits+[ancilla])
        return qc


    def single_oracle_smallangle_inconstraint(self, c, norm):


        0

    def single_oracle_sin_inconstraint(self, c, norm, inverse=False):
        """
        Compute the oracle onto an ancilla qubit -- assuming our states are in-constraint,
        using the sin approximation.

        Build the practical sin-approximation oracle  F_sin  (Eq. 45 of paper).

        Uses the small-angle approximation  sin(theta) ~= theta  to decompose the oracle
        into O(n_y) doubly-controlled RY gates instead of O(2^n) multi-controlled ones.

        For each turbine j, applies TWO CCRY (doubly-controlled RY) pairs:

          CCRY(pi.c_y[j]/norm)  controlled on [wind_qubit_j, pdf_qubit_j] -> ancilla
            -> fires when y[j]=1 AND xi[j]=1  (turbine ON, wind available)
            -> rotates ancilla by  pi.c_y[j]/norm  (encodes operational cost)

          X(pdf_qubit_j)  then  CCRY(pi.c_r/norm)  then  X(pdf_qubit_j)
            -> fires when y[j]=1 AND xi[j]=0  (turbine ON, no wind)
            -> rotates ancilla by  pi.c_r/norm  (encodes recourse cost)

        The approximation works because costs are a SUM over independent turbines,
        so each turbine's contribution can be added independently to the ancilla.
        Error grows with the magnitude of individual cost angles.

        `c`    = cost normalization scalar  (controls scale = pi.c/norm)
        `norm` = maximum cost  (prevents ancilla from over-rotating past pi/2)
        """
        return circuit_builders.cost_oracle(
            self.wind_costs, self.recourse_cost, norm,
            self.wind_qubits, self.pdf_qubits,
            encoding="linear", scale=c, inverse=inverse,
        )

    def grover_operator_sin_inconstraint(self, power, c, norm):


        return 0

    def grover_reflections_sin_inconstraint(self, m, c, norm):
        return 0

       

    def single_oracle_asin_inconstraint(self, norm, inverse=False):
        """Build the arcsin oracle using exact RY angles (more accurate than sin oracle).

        Applies CCRY rotations with angles $2\\arcsin(\\sqrt{c/\\text{norm}})$ instead
        of the small-angle approximation $\\pi c/\\text{norm}$ used by
        `single_oracle_sin_inconstraint()`.

        For each turbine j:

        - `theta_cy = 2*arcsin(sqrt(c_y[j]/norm))` — exact rotation for operational cost
        - `theta_cr = 2*arcsin(sqrt(c_r/norm))`    — exact rotation for recourse cost

        More accurate than `F_sin` at the cost of a different approximation error
        profile (arcsin vs. linear). Requires `c_y/norm <= 1` and `c_r/norm <= 1`.

        Args:
            norm: Maximum cost; all costs must satisfy `c <= norm`.
            inverse: If `True`, negate all rotation angles (returns $\\mathcal{F}^\\dagger$).

        Returns:
            QuantumCircuit on `2*num_wind_vars + 1` qubits (system + ancilla).
        """
        return circuit_builders.cost_oracle(
            self.wind_costs, self.recourse_cost, norm,
            self.wind_qubits, self.pdf_qubits,
            encoding="arcsine", inverse=inverse,
        )

    def grover_operator_asin_inconstraint(self, power, norm):


        return 0

    def grover_reflections_asin_inconstraint(self, m, norm):
        return 0



    def execute_optimizer(self, qc, num_meas=None):
        """Execute the DQA circuit and return measurement results.

        Two modes:

        - `num_meas=None`: Run with the statevector simulator (exact). Returns a
          probability dictionary `{bitstring: probability}`.
        - `num_meas=N`: Append `measure_all()` and run N shots. Returns a
          normalized count dictionary `{bitstring: count/N}`.

        !!! tip "GPU Tier 1"
            Replace `Aer.get_backend('statevector_simulator')` with
            `AerSimulator(method='statevector', device='GPU')` for H100 speedup.

        Args:
            qc: The DQA `QuantumCircuit` to execute.
            num_meas: Number of measurement shots, or `None` for exact statevector.

        Returns:
            Dict mapping bitstrings to probabilities (exact) or normalized counts (sampled).
        """
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            # GPU-TIER1: AerSimulator with GPU device (falls back to CPU automatically)
            simulator = _get_simulator(method='statevector', shots_mode=False,
                                       n_qubits=qc.num_qubits)
            qc.save_statevector()  # required in Aer 0.15+ to store SV in result
            qc = transpile(qc, simulator)
            result = simulator.run(qc).result()
            statevector = result.get_statevector(0)  # use index; circuit object key invalid after transpile in Qiskit 2.x
            return statevector.probabilities_dict()
        # use counts
        else:
            qc.measure_all()
            # GPU-TIER1+2: GPU + max_parallel_shots for batched shot execution
            simulator = _get_simulator(method='statevector', shots_mode=True,
                                       n_qubits=qc.num_qubits)
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=num_meas).result()
            counts = result.get_counts(0)  # use index; circuit object key invalid after transpile in Qiskit 2.x
            counts = {key:value/num_meas for key,value in counts.items()}
            return counts
    
    def execute_optimizer_oracle(self, qc, num_meas=None):
        """Execute the DQA + oracle circuit and return the ancilla amplitude.

        Runs the circuit that is DQA followed by the oracle, then marginalizes
        over the ancilla qubit to recover the amplitude $a = \\Pr[\\text{ancilla}=|1\\rangle]$.
        This $a$ is the normalized expected cost $\\bar{\\phi}(x)$.

        Two modes:

        - `num_meas=None`: Exact statevector — sums probability of all states with ancilla=`1`.
        - `num_meas=N`: Sampled — returns `counts['1'] / N`.

        !!! tip "GPU Tier 1"
            Replace `Aer.get_backend('aer_simulator_statevector')` with
            `AerSimulator(method='statevector', device='GPU')`.

        Args:
            qc: Circuit comprising DQA followed by the oracle (ancilla is MSB).
            num_meas: Number of shots, or `None` for exact statevector.

        Returns:
            Float in [0, 1] — the amplitude on the ancilla `|1⟩` state.
        """
        # if no number of measurements is specified, try to use statevector
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            # GPU-TIER1: AerSimulator with GPU device (falls back to CPU automatically)
            simulator = _get_simulator(method='statevector', shots_mode=False,
                                       n_qubits=qc.num_qubits)
            qc.save_statevector()  # required in Aer 0.15+ to store SV in result
            qc = transpile(qc, simulator)
            result = simulator.run(qc).result()
            statevector = result.get_statevector(0)  # use index; circuit object key invalid after transpile in Qiskit 2.x
            pd = statevector.probabilities_dict()
            a = 0.
            for key,value in pd.items():
                if key[0] == '1':
                    a += value
            return a
        # use counts
        else:
            qc.measure_all()  # measure all qubits; ancilla is MSB (key[0])
            # GPU-TIER1+2: GPU + max_parallel_shots for batched oracle evaluations
            simulator = _get_simulator(method='statevector', shots_mode=True,
                                       n_qubits=qc.num_qubits)
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=num_meas).result()
            counts = result.get_counts(0)  # use index; circuit object key invalid after transpile in Qiskit 2.x
            # ancilla is the highest-index qubit = MSB (leftmost char) in Qiskit bitstrings
            return sum(v for k, v in counts.items() if k[0] == '1') / num_meas


    def execute_qae(self, qc, m, num_meas=None):
        """Execute the full QAE circuit and return the QPE readout distribution.

        Runs the canonical QAE circuit (built by `implemented_qae()`) and returns
        measurement outcomes of the $m$ QPE estimate qubits.

        Post-processing (done by caller):

        ```python
        b_str = max(b_counts, key=b_counts.get)   # most probable readout
        b_int = int(b_str, 2)
        a_tilde   = np.sin(b_int * np.pi / (2**m))**2
        phi_tilde = a_tilde * norm
        ```

        Two modes:

        - `num_meas=None`: Exact statevector, marginalizes over the system qubits.
        - `num_meas=N`: Sampled, applies measurement gates on the $m$ estimate qubits only.

        !!! tip "GPU Tier 1 + Tier 3"
            Replace backend with `AerSimulator(method='statevector', device='GPU')`.
            For `m >= 7`, also enable `cuStateVec_enable=True`.

        Args:
            qc: The full QAE `QuantumCircuit` (from `implemented_qae()`).
            m: Number of QPE estimate qubits (determines resolution: error ~ 1/2^m).
            num_meas: Number of shots, or `None` for exact statevector.

        Returns:
            Dict `{b_bitstring: probability}` over all $2^m$ QPE outcomes.
        """
        # if no number of measurements is specified, try to use statevector
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            # Use Statevector.evolve() directly: avoids transpile qubit routing
            # which would silently reorder logical qubits and corrupt the marginalisation.
            # GPU-TIER1 for large circuits: switch to shots mode by providing num_meas.
            from qiskit.quantum_info import Statevector as _SV
            sv = _SV.from_label('0' * qc.num_qubits)
            sv = sv.evolve(qc)
            pd = sv.probabilities_dict()
            b_counts = {}
            for key, value in pd.items():
                # QPE qubits at circuit indices 0..m-1 = last m chars in bitstring
                reg = key[-m:]
                b_counts[reg] = b_counts.get(reg, 0) + value
            return b_counts
        # use counts
        else:
            # Shots mode: measure only the QPE register (qubits 0..m-1) into m classical bits.
            # GPU-TIER1+2: GPU + max_parallel_shots for shot execution.
            # NOTE: QPE qubits are at circuit indices 0..m-1 in implemented_qae(); measuring
            # them via dedicated classical bits avoids qubit routing ambiguity.
            n_sys = 2 * self.num_wind_vars + 1
            qc_meas = QuantumCircuit(n_sys + m, m)
            qc_meas.append(qc, list(range(n_sys + m)))
            qc_meas.measure(list(range(m)), list(range(m)))  # QPE qubits 0..m-1
            simulator = _get_simulator(method='statevector', shots_mode=True,
                                       n_qubits=qc_meas.num_qubits)
            qc_meas = transpile(qc_meas, simulator)
            result = simulator.run(qc_meas, shots=num_meas).result()
            counts = result.get_counts(0)  # use index; circuit object key invalid after transpile in Qiskit 2.x
            counts = {key: value / num_meas for key, value in counts.items()}
            return counts


    # ---------------------------------------------------------------------------
    # Tier 2 batched executors
    # ---------------------------------------------------------------------------
    def execute_optimizer_batched(self, circuits: list, num_meas: int = None) -> list:
        """GPU-batched variant of execute_optimizer for multiple circuits at once.

        Submits all circuits in a single `simulator.run()` call, which amortises
        GPU kernel launch overhead and maximises H100 occupancy.

        Args:
            circuits: List of DQA QuantumCircuits (one per x-value).
            num_meas: Shots per circuit, or ``None`` for exact statevector.

        Returns:
            List of probability/count dicts, one per input circuit.
        """
        simulator = _get_simulator(method='statevector', shots_mode=(num_meas is not None))
        if num_meas is None:
            # save_statevector() must be added before transpile in Aer 0.15+
            sv_circuits = [qc.copy() for qc in circuits]
            for qc in sv_circuits:
                qc.save_statevector()
            transpiled = [transpile(qc, simulator) for qc in sv_circuits]
            job = simulator.run(transpiled)
            results = job.result()
            return [results.get_statevector(i).probabilities_dict()  # use index; circuit object key invalid after transpile in Qiskit 2.x
                    for i in range(len(transpiled))]
        else:
            meas_circuits = []
            for qc in circuits:
                qc_copy = qc.copy()
                qc_copy.measure_all()
                meas_circuits.append(transpile(qc_copy, simulator))
            job = simulator.run(meas_circuits, shots=num_meas)
            results = job.result()
            return [{k: v / num_meas for k, v in results.get_counts(i).items()}
                    for i in range(len(meas_circuits))]

    def execute_qae_batched(self, circuits: list, m: int, num_meas: int = None) -> list:
        """GPU-batched variant of execute_qae for multiple QAE circuits at once.

        Args:
            circuits: List of full QAE QuantumCircuits (one per x-value).
            m: Number of QPE estimate qubits.
            num_meas: Shots per circuit, or ``None`` for exact statevector.

        Returns:
            List of QPE outcome dicts ``{b_bitstring: probability}``.
        """
        method = 'matrix_product_state' if m >= 7 else 'statevector'
        simulator = _get_simulator(method=method, shots_mode=(num_meas is not None))

        if num_meas is None:
            # Use Statevector.evolve() directly: avoids transpile qubit routing
            # which would corrupt the QPE register marginalisation.
            from qiskit.quantum_info import Statevector as _SV
            out = []
            for qc in circuits:
                sv = _SV.from_label('0' * qc.num_qubits)
                sv = sv.evolve(qc)
                pd = sv.probabilities_dict()
                b_counts: dict = {}
                for key, value in pd.items():
                    reg = key[-m:]
                    b_counts[reg] = b_counts.get(reg, 0) + value
                out.append(b_counts)
            return out
        else:
            meas_circuits = []
            n_sys = 2 * self.num_wind_vars + 1
            for qc in circuits:
                qc_meas = QuantumCircuit(n_sys + m, m)
                qc_meas.append(qc, list(range(n_sys + m)))
                qc_meas.measure(list(range(m)), list(range(m)))  # QPE qubits 0..m-1
                meas_circuits.append(transpile(qc_meas, simulator))
            job = simulator.run(meas_circuits, shots=num_meas)
            results = job.result()
            return [{k: v / num_meas for k, v in results.get_counts(i).items()}
                    for i in range(len(meas_circuits))]
    # ---------------------------------------------------------------------------

    # post-processors
    def process_expectation_value_optimizer(self, wind_demand, counts):
        """Convert DQA measurement counts to the classical expected cost.

        Post-processes the output of `execute_optimizer()` into an expected cost
        estimate by evaluating `wind_scenario_cost()` for each measured bitstring
        and weighting by its probability:

        $$\\tilde{\\phi}(x) = \\sum_{(y, \\xi)} p(y, \\xi) \\cdot q(y, \\xi)$$

        where the sum is over all bitstrings in `counts` and $p$ is the
        normalized measurement frequency.

        Args:
            wind_demand: Integer $k$ — required Hamming weight of turbine decisions.
            counts: Dict `{bitstring: probability}` from `execute_optimizer()`.
                Qiskit bitstring layout (qubit 0 = rightmost character):
                ``[xi_{n-1} ... xi_0 | y_{n-1} ... y_0]``
                so the first half (left) is the reversed xi register and
                the second half (right) is the reversed y register.

        Returns:
            Float — estimated expected second-stage cost $\\tilde{\\phi}(x)$.
        """
        expectation_value = 0.
        for bstr, val in counts.items():
            output = self.bstr_to_output(bstr)
            n = len(output) // 2
            # Qiskit convention: qubit k appears at position (2n-1-k) from the left.
            # First half  = qubits 2n-1..n  = xi[n-1..0]  → reverse to get xi[0..n-1]
            # Second half = qubits n-1..0   = y[n-1..0]   → reverse to get y[0..n-1]
            wind_output = output[n:][::-1]   # y in natural qubit order
            pdf_output  = output[:n][::-1]   # xi in natural qubit order
            expectation_value += self.wind_scenario_cost(wind_output, pdf_output, wind_demand) * val
        return expectation_value
