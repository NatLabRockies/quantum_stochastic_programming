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
    ''' mixing operator which preserves the demand constraint
    '''
    qc = QuantumCircuit(len(args['y_reg']), name=r'$U_d({})$'.format(np.round(amplitude, 3)))
    layers = 1
    for layer in range(layers):
        for qj in args['y_reg']:
            for qk in args['y_reg'][qj+1:]:
                qc.append(SwapGate().power(amplitude/layers), [qj, qk])
    return qc

def cost_operator(amplitude:float, args:dict) -> QuantumCircuit:
    ''' our cost operator
        constraint_amplitude will be recourse for the constraint-preserving mixer, and a penalty weight otherwise
    '''
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
    ''' compute the oracle ont an ancilla qubit - assuming our states are in-constraint, using the sin approx
    '''
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
    ''' https://arxiv.org/pdf/1904.07358.pdf '''
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
    '''
    Creates alternating ansatz circuit given circuits for cost operator and mixer operator
    Input: args - dict containing the following: num_qubits - int, gate used - str, graph edge list - list, 
    angle - Parameter, cost_operator_circuit - Function that returns a QuantumCircuit, 
    initial_state - QuantumCircuit
    mixer_operator_circuit - Function that returns a QuantumCircuit, cost_qubits - list, ancilla_qubits - list, Theta - list
    Output: QuantumCircuit
    '''
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