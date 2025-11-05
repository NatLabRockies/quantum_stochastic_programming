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
from qiskit.circuit.library import QFT

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
        Generates and returns qiskit QuantumCircuit object to run QAE algorithm on user-specified problem
        '''
        op = alternating_operator_ansatz(self.args)
        op_inv = op.inverse()
        oracle = self.oracle_circuit(self.args)
        oracle_inv = oracle.inverse()

        qae_circuit = self.implemented_qae(op, oracle, op_inv, oracle_inv)

        return qae_circuit

    def implemented_qae(self, op, oracle, op_inv, oracle_inv) -> QuantumCircuit:
        '''
        Takes QuantumCircuit objects and returns QuantumCircuit object representing the QAE algorithm
        '''
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
        qc.append(QFT(self.m, inverse=True), list(range(self.m)))

        if self.gateset:
            qc = transpile(qc, basis_gates=self.gateset)

        return qc