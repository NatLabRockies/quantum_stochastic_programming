'''
created by csagal on 7/2/2024
Script for running quantum optimization algorithms
'''
# Package Import
from typing import List, Dict, Tuple

# Qiskit Import
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.circuit.library import QFTGate

# File Import
from dist_prep import *
from ExpValFun_functions import alternating_operator_ansatz

'''
Class for compiling Quantum Expectation Value function algorithm

args should contain the following:

{

# Variables
args['n_y']
args['m']
args['pdf']
args['y_reg']
args['pdf_reg']
args['Theta']
args['w_d']
args['norm']
args['uniform']
args['gateset']

# Functions
args['cost_operator_circuit']
args['mixer_operator_circuit']
args['initial_state_circuit']
args['pdf_circuit']
args['oracle_circuit']

# Also include any other variables your functions need!

}

'''

class QAE_Optimizer:
    """
    Assembles and runs the canonical Quantum Amplitude Estimation (QAE) algorithm
    for the two-stage stochastic optimization problem.

    The QAE_Optimizer takes a user-supplied dictionary `args` that provides:
      - Problem dimensions  (n_y, m, y_reg, pdf_reg)
      - Circuit factories    (cost_operator_circuit, mixer_operator_circuit,
                              initial_state_circuit, pdf_circuit, oracle_circuit)
      - Annealing angles    (Theta list, w_d wind demand, norm)
      - Misc flags          (uniform PDF, gateset for transpilation)

    The main output is `compile_qae_circuit()` which returns the full QAE
    QuantumCircuit ready to run on a simulator or hardware.
    """

    def __init__(self, args:dict):

        self.args = args
        self.m = args['m']
        self.pdf = args['pdf']
        self.y_reg = args['y_reg']
        self.pdf_reg = args['pdf_reg']

        self.cost_operator_circuit = args['cost_operator_circuit'] 
        self.mixer_operator_circuit = args['mixer_operator_circuit']
        self.initial_state_circuit = args['initial_state_circuit'] 
        self.pdf_circuit = args['pdf_circuit']

        self.n_y = args['n_y']
        self.oracle_circuit = args['oracle_circuit'] 
        self.Theta = args['Theta']

        self.gateset = args['gateset'] # If user wants an abstract gateset, set this to 'False'

    def compile_qae_circuit(self) -> QuantumCircuit:
        '''
        Generates and returns a Qiskit QuantumCircuit object to run the QAE algorithm
        on the user-specified problem.

        Assemble the full QAE circuit for the user-specified problem.

        Steps:
          1. Build the DQA ansatz circuit  op  = alternating_operator_ansatz(args)
             (Dicke-state init → alternating cost + mixer layers at angles Theta)
          2. Build the oracle circuit  oracle  (encodes cost as ancilla amplitude)
          3. Combine into full QAE via implemented_qae()

        Returns a QuantumCircuit with layout:
          [0..m-1]       : m QPE estimate qubits (readout after IQFT)
          [m..m+2n_y-1]  : system qubits (y-register + pdf-register)
          [m+2n_y]       : ancilla qubit
        '''
        op = alternating_operator_ansatz(self.args)
        op_inv = op.inverse()
        oracle = self.oracle_circuit(self.args)
        oracle_inv = oracle.inverse()

        qae_circuit = self.implemented_qae(op, oracle, op_inv, oracle_inv)

        return qae_circuit

    def implemented_qae(self, op, oracle, op_inv, oracle_inv) -> QuantumCircuit:
        """
        Takes QuantumCircuit objects and returns a QuantumCircuit object representing the QAE algorithm.

        Build the canonical QAE circuit (Brassard et al. 2002, Fig. 2 of paper).

        Qubit layout:
          [0..m-1]      : m estimate (QPE) qubits
          [m..m+2n_y-1] : system qubits
          [m+2n_y]      : ancilla qubit  (ancilla index = m + 2*n_y)

        Circuit in order:
          1. H^{\otimes m} on estimate qubits
             → creates superposition so QPE can probe all powers of Q simultaneously

          2. op  on system qubits  (DQA circuit — prepares optimal superposition)
             oracle  on system + ancilla  (encodes cost as Pr[ancilla=|1⟩])

          3. For each estimate qubit i (0..m-1), repeat 2^i times:
               a) S_ψ0: X(ancilla) · CZ(estimate_i, ancilla) · X(ancilla)
                  → phase-flip on ancilla=|0⟩ states (marks the "bad" subspace)
               b) A^{-1}: oracle_inv.control(1) then op_inv.control(1)
                  → uncompute A, controlled on estimate qubit i
               c) S_0: X all system+ancilla qubits, controlled-MCZ, X back
                  → phase-flip on |0...0⟩ (the zero-reflection for Grover)
               d) A: op.control(1) then oracle.control(1)
                  → reapply A, controlled on estimate qubit i
             Together (a)-(d) implement one application of the Grover operator Q,
             controlled on estimate qubit i.  The 2^i repetitions encode the
             eigenphase 2*arcsin(sqrt(a)) as a binary fraction in the QPE register.

          4. IQFT on estimate qubits  →  phase → integer b

        Post-processing (done outside this function):
          a_tilde   = sin^2(b * pi / 2^m)
          phi_tilde = a_tilde * norm
        """
        ancilla = self.m + 2*self.n_y 
        qc = QuantumCircuit(self.m + 2*self.n_y  + 1) 
        for i in range(self.m):
            qc.h(i)
        qc.append(op, list(range(self.m, self.m + 2*self.n_y )))
        qc.append(oracle, list(range(self.m, self.m + 2*self.n_y  + 1)))
        for i in range(self.m):
            for j in range(2**i):
                qc.x(ancilla)
                qc.cz(i,ancilla)
                qc.x(ancilla)
                # invert
                # A^-1
                qc.append(oracle_inv.control(1), [i]+list(range(self.m, ancilla+1)))
                qc.append(op_inv.control(1), [i]+list(range(self.m, ancilla)))
                # |000...><000...|
                for q in range(self.m,ancilla):
                    qc.x(q)
                qc.x(ancilla)
                qc.append(ZGate().control(2*self.n_y  + 1), [i]+list(range(self.m, ancilla+1)))
                for q in range(self.m,ancilla):
                    qc.x(q)
                qc.x(ancilla)
                # A
                qc.append(op.control(1), [i]+list(range(self.m, ancilla)))
                qc.append(oracle.control(1), [i]+list(range(self.m, ancilla+1)))
        qc.append(QFTGate(self.m).inverse(), list(range(self.m)))

        if self.gateset:
            qc = transpile(qc, basis_gates=self.gateset)

        return qc
