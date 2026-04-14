'''
created by csagal on 7/2/2024
File for generating probability distributions
'''

# Package Import
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Qiskit Import
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.circuit.library import Initialize
#from qiskit_finance.circuit.library import NormalDistribution

'''
Functions to generate circuits that encode specific probability distributions into quantum states

args should be the same as the args in qae.py, but should also contain the required parameters in the function used.

'''

def make_normal_distribution_circuit(args:dict) -> QuantumCircuit:
    mu = args['mu']  # Can be changed as needed NOTE: qiskit NormalDistribution centers the dist at num_qubits/2 when mu=0
    sigma = args['sigma'] # Can be changed as needed

    qc = NormalDistribution(args['n_y'], mu=mu, sigma=sigma)

    return qc

def make_variational_distribution_circuit(args:dict) -> QuantumCircuit:

    qc = QuantumCircuit(args['n_y'])
    
    for i in range(int(len(args['variational_angles'])/args['n_y'])):
        for qubit in range(args['n_y']):
            qc.ry(args['variational_angles'][i+qubit], qubit)
        for i in range(args['n_y']-1):
            qc.cx(i, i+1)

    return qc


def make_pdf_skewnormal(n:int) -> QuantumCircuit:
    """
    Generates a probability density for a skew normal distribution.
    Input: system size n
    Output: dictionary containing PDF

    Build a discrete PDF approximating a skew-normal distribution over n-qubit bitstrings.

    Strategy:
      - Map each Hamming weight s = sum(bitstring) to a normal-distribution probability.
      - Divide that probability equally among all bitstrings with that weight  (C(n,s) of them).

    Returns: dict  {(b0,b1,...,bn-1): probability}  summing to 1.
    """
    dist = np.arange(-1-int(n/2), int(n/2)+n%2)
    prob = scipy.stats.norm.pdf(dist, scale=1)
    prob = prob / prob.sum()

    pdf = {}
    for xi in range(2**n):
        bstr = ('{0:0'+str(n)+'b}').format(xi)
        sample_tuple = tuple([int(i) for i in bstr])
        s = sum(sample_tuple)
        size = math.comb(n,s)
        pdf[sample_tuple] = prob[s]/size 
    return pdf

def make_pdf_uniform(n:int) -> QuantumCircuit:
    """
    Generates a probability density for a uniform distribution.
    Input: system size n - int
    Output: dictionary containing PDF - dict

    Build a discrete uniform PDF over all 2^n bitstrings on n qubits.

    Every bitstring (including ones that violate wind-demand) gets equal probability 1/2^n.
    This is the simplest baseline; in the quantum circuit it corresponds to
    n Hadamard gates  H^{\otimes n}  on the pdf register.

    Returns: dict  {(b0,b1,...,bn-1): 1/2^n}.
    """
    #vec = np.zeros(2**n)
    pdf = {}
    for i in range(2**n):
        bstr = ('{0:0'+str(n)+'b}').format(i)
        sample_tuple = tuple([int(i) for i in bstr])
        pdf[sample_tuple] = 1/2**n
    return pdf

def pdf_initialize(args:dict) -> QuantumCircuit:
    """
    Create the pdf as a wavefunction.

    Build a quantum circuit that encodes the PDF  Pr[ξ]  as amplitudes on the ξ register.

    The prepared state is:
        |ξ⟩ = Σ_ξ  √Pr[ξ]  |ξ⟩

    so that measuring the ξ register gives outcome ξ with probability Pr[ξ].
    This is Eq. (17) of the paper  (|ξ⟩ = Π_j H_j  for uniform, or Initialize for general).

    Two cases:
      uniform == True   : n Hadamard gates  (H^{\otimes n})  — efficient O(n) circuit
      uniform == False  : Qiskit Initialize (arbitrary statevector) — O(2^n) gates after transpile
    """
    qc = QuantumCircuit(args['n_y'], name=r'$|\mathcal{P}\rangle$')
    # if np.isclose(list(args['pdf'].values()), [1/2**args['num_wind_vars']]*2**args['num_wind_vars']).all():
    if args['uniform'] == True:
        for i in range(args['n_y']):
            qc.h(i)
    else:
        wvfn = np.zeros(2**args['n_y'])
        for scenario, pr in args['pdf'].items():
            bstr = ''.join([str(i) for i in scenario])
            wvfn[int(bstr[::-1], 2)] = np.sqrt(pr)
        qc.append(Initialize(wvfn), list(range(args['n_y'])))
        #qc.initialize(wvfn)#, list(range(self.))) 
        #simulator = Aer.get_backend('statevector_simulator')         
        gates = ['u', 'p', 'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'cx']
        qc = transpile(qc, basis_gates=gates)
    #qc = circuit_to_gate(qc)
    return qc