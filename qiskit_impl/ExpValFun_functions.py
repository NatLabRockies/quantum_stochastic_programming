'''
created by csagal on 9/6/2024
Script for running quantum optimization algorithms
'''
# Package Import
import numpy as np
from typing import List, Dict, Tuple

# Qiskit Import
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile
from qiskit.circuit.library.standard_gates import SwapGate, RYGate

'''
Functions for constructing the Expectation Value Function algorithm circuit

args should be the same as the args in qae.py, but should also contain the required parameters in the function used.

'''

# Functions from binary optimizer for testing
def demand_constraint_preserving_mixer(amplitude:float, args:dict) -> QuantumCircuit:
    """
    Mixing operator which preserves the demand constraint.

    XY mixer  U_d(β)  that preserves the Hamming weight (= wind demand constraint).

    For every pair (j, k) of wind qubits, applies SWAP^β:
      SWAP^β = exp(-i βπ/2 · (XX+YY)/2)

    The XX+YY Hamiltonian moves excitations between qubits without changing their
    total count, so sum(y) = wind_demand is preserved throughout DQA evolution.
    This makes the Dicke-state initial condition self-consistent: the entire
    trajectory stays within the feasible subspace.

    Eq. (39) of the paper.  amplitude = β (decreases from 1 to 0 during annealing).
    """
    qc = QuantumCircuit(len(args['y_reg']), name=r'$U_d({})$'.format(np.round(amplitude, 3)))
    layers = 1
    for layer in range(layers):
        for qj in args['y_reg']:
            for qk in args['y_reg'][qj+1:]:
                qc.append(SwapGate().power(amplitude/layers), [qj, qk])
    return qc

def cost_operator(amplitude:float, args:dict) -> QuantumCircuit:
    """
    Our cost operator.
    `constraint_amplitude` will be recourse for the constraint-preserving mixer,
    and a penalty weight otherwise.

    Phase (cost) operator  U_q(γ)  encoding second-stage costs as quantum phases.

    For each wind turbine j, applies two CP (controlled-phase) gates:

      CP(γ · c_y[j] / cost_norm)   on [pdf_reg[j], y_reg[j]]
        → phase e^{iγc_y/norm} when BOTH y[j]=|1⟩ AND ξ[j]=|1⟩
          (turbine ON  +  wind available  →  operational cost c_y[j])

      X(pdf_reg[j])  then  CP(γ · c_r / cost_norm)  then  X(pdf_reg[j])
        → phase e^{iγc_r/norm} when y[j]=|1⟩ AND ξ[j]=|0⟩
          (turbine ON  +  no wind  →  recourse cost c_r)

    The CP gate acts as  exp(iθ |1⟩⟨1| ⊗ |1⟩⟨1|),  i.e., it encodes the
    diagonal cost Hamiltonian H_C as a phase on computational basis states.

    Corresponds to U_q in Eq. (42) of the paper.
    `amplitude` = γ annealing parameter;  `args['cost_norm']` prevents phase wrapping.
    """
    qc = QuantumCircuit(args['n_y']*2, name=r'$U_q({})$'.format(np.round(amplitude, 3)))
    for q_w,cost in enumerate(args['c_y']):
        q_pdf = args['pdf_reg'][q_w]
        qc.cp(amplitude*cost/args['cost_norm'], q_pdf, q_w)

        qc.x(q_pdf)
        qc.cp(amplitude*args['c_r']/args['cost_norm'], q_pdf, q_w)
        qc.x(q_pdf)
    return qc

def cost_scenario_operator(amplitude:float, args:dict) -> QuantumCircuit:
    qc = QuantumCircuit(len(args['y_reg']))
    for q_w, cost in enumerate(args['wind_costs']):
        if args['scenario'][q_w] == 0:
            qc.p(amplitude*args['constraint_amplitude']/args['cost_norm'], q_w)
        else:
            qc.p(amplitude*cost/args['cost_norm'], q_w)
    return qc

def single_oracle_sin_inconstraint(args:dict, inverse=False) -> QuantumCircuit:
    """
    Compute the oracle onto an ancilla qubit — assuming our states are in-constraint,
    using the sin approximation.

    Practical oracle  F_sin  (Eq. 45 of paper) using the sin-approximation.

    For each turbine j, applies two CCRY (doubly-controlled-RY) rotations on the ancilla:

      CCRY(π·c_y[j]/norm)  on [y_reg[j], pdf_reg[j]] → ancilla
        → fires when turbine j is ON (y[j]=|1⟩) AND wind is available (ξ[j]=|1⟩)
        → rotates ancilla by  π·c_y/norm  (encodes operational cost as amplitude)

      X(pdf_reg[j])  →  CCRY(π·c_r/norm)  →  X(pdf_reg[j])
        → fires when y[j]=|1⟩ AND ξ[j]=|0⟩  (recourse scenario)
        → rotates ancilla by  π·c_r/norm

    After F_sin:  Pr[ancilla=|1⟩] ≈ q̄(y,ξ) = q(y,ξ)/norm  (normalized cost).

    The approximation  sin(θ)≈θ  is valid when angles are small (c/norm << 1).
    Efficient: requires only  2·n_y  two-qubit-controlled RY gates instead of O(2^n).
    """
    qc = QuantumCircuit(args['n_y']*2 + 1, name='F')
    ancilla = args['n_y']*2
    scale = np.pi*1/args['norm']
    if inverse:
        scale *= -1
    for q in args['y_reg']:
        theta_cy = args['c_y'][q] * scale 
        theta_cr = args['c_r'] * scale

        qc.append(RYGate(theta_cy).control(2), [q, args['pdf_reg'][q], ancilla])

        qc.x(args['pdf_reg'][q])
        qc.append(RYGate(theta_cr).control(2), [q, args['pdf_reg'][q], ancilla])
        qc.x(args['pdf_reg'][q])
    return qc

def dicke_state_circuit(args:dict) -> QuantumCircuit:
    """
    Reference: https://arxiv.org/pdf/1904.07358.pdf

    Prepare the Dicke state  |D_n^k⟩  on n = args['n_y'] qubits with k = args['w_d'] ones.

    The Dicke state is a UNIFORM superposition over all n-bit strings with exactly k ones:

        |D_n^k⟩ = 1/√C(n,k)  Σ_{|y|=k} |y⟩

    This is the initial state for DQA: every term already satisfies the demand
    constraint sum(y) = wind_demand.  The XY mixer then keeps the state within
    this feasible subspace throughout the annealing.

    Uses the SCS (Splitting-Coherence-Splitting) construction from
    Bärtschi & Eidenbenz, arXiv:1904.07358.  Gate count O(n*k), no ancilla.
    """
    qc = QuantumCircuit(args['n_y'], name='DickeState')
    if args['w_d'] == 0:
        return qc
    def SCS(n,k):
        scs = QuantumCircuit(k+1)
        ni = k#n-k+1
        scs.cx(ni-1, ni)
        scs.cry(2*np.arccos(np.sqrt(1/n)), ni, ni-1)
        scs.cx(ni-1, ni)
        for l in range(2,k+1):
            scs.cx(ni-l, ni)
            scs.append(RYGate(2*np.arccos(np.sqrt(l/n))).control(2), [ni, ni-l+1, ni-l])
            #scs.ccry(2*np.arccos(np.sqrt(l/n)), ni, ni-l+1, ni-l)
            scs.cx(ni-l, ni)
        return scs.to_gate()
    for i in range(args['w_d']):# range(self.num_wind_vars - weight - 1, self.num_wind_vars):
        qc.x(args['n_y']-i-1)
    # n-k applications
    for i in range(args['n_y'] - args['w_d']):
        qc.append(SCS(args['n_y']-i,args['w_d']), list(range(args['n_y']-i-args['w_d']-1,args['n_y']-i)))
    # last k
    for i in range(args['w_d']-1,0,-1):
        #print(i+1, i)
        qc.append(SCS(i+1, i), list(range(i+1)))
    #print(qc)
    return qc

def alternating_operator_ansatz(args:dict) -> QuantumCircuit:
    """
    Creates alternating ansatz circuit given circuits for cost operator and mixer operator.
    Input:  args - dict containing num_qubits, gate used, graph edge list, angle (Parameter),
            cost_operator_circuit (Function), initial_state (QuantumCircuit),
            mixer_operator_circuit (Function), cost_qubits, ancilla_qubits, Theta (list).
    Output: QuantumCircuit

    Build the DQA (Discrete Quantum Annealing) ansatz circuit  U_DQA.

    This is the QAOA-style alternating-operator ansatz:

        U_DQA = [initial_state] ⊗ [pdf_state]
                · Π_i  (cost_op or mixer_op)(Theta[i])

    Layout:
      y_reg   (wind turbines)  : initialized with Dicke state |D_n^{w_d}⟩
      pdf_reg (wind scenarios) : initialized with PDF state |ξ⟩ encoding Pr[ξ]

    Then alternating layers indexed by Theta (the variational angles):
      even i : cost_operator_circuit(Theta[i])   on y_reg + pdf_reg
               (encodes second-stage cost as phase)
      odd  i : mixer_operator_circuit(Theta[i])  on y_reg only
               (mixes turbine decisions while preserving demand constraint)

    The angles Theta are set by the linear annealing schedule  s(t) = t/T,
    making this a parameterized quantum circuit that approximates adiabatic evolution.
    """
    qc = QuantumCircuit(args['n_y']*2)

    qc.append(args['initial_state_circuit'](args).to_gate(), args['y_reg'])
    qc.append(args['pdf_circuit'](args).to_gate(), args['pdf_reg'])
    # Need function to initialize pdf qubits
    for i, theta in enumerate(args['Theta']):

        if i % 2 == 0:
            qc.append(args['cost_operator_circuit'](theta, args).to_gate(), args['y_reg'] + args['pdf_reg'])
        else:
            qc.append(args['mixer_operator_circuit'](theta, args).to_gate(), args['y_reg'])

    return qc