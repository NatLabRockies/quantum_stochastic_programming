''' experiment_convergence.py
    CRotello Dec 2023
    A python script to run binary optimizer experiments
    Experiment: 
'''

from binary_optimizer import BinaryNestedOptimizer
import math
import scipy
import numpy as np
#import json
import pickle
import argparse
import copy
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
from qiskit.compiler import transpile
from qiskit.algorithms import AmplitudeEstimation
from qiskit.extensions import Initialize
from qiskit.converters import circuit_to_gate
from qiskit.quantum_info import Statevector


def make_pdf_skewnormal(n):
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

def make_pdf_uniform(n):
    #vec = np.zeros(2**n)
    pdf = {}
    for i in range(2**n):
        bstr = ('{0:0'+str(n)+'b}').format(i)
        sample_tuple = tuple([int(i) for i in bstr])
        pdf[sample_tuple] = 1/2**n
    return pdf


def experiment_sampling_makewvfn(pdf, n, m, spec_d=None, is_uniform=False):
    ''' For a dataset (w/ high variance in Q) get the different convergences in M
        For a single demand output from the first stage, unless demand is not specified
    '''    
    results = {}
    results['pdf'] = pdf
    results['wd'] = spec_d
    # costs
    #cy = [0.08, 0.015, 0.1, 0.05, 0.03, 0.15, 0.06, 0.2, 0.02, 0.01, 0.13, 0.17,]
    #cy = cy[:n]
    #cy = [.10]*n
    cy = np.linspace(0.01, 0.2, n)
    cr = 1.
    cx = [0.4,] # dummy
    bno = BinaryNestedOptimizer(cx, cy, cr, pdf, n, is_uniform=is_uniform)
    e = bno.brute_force_wind_demand_expectation_values()
    results['model'] = bno
    results['true_expectation_values'] = e
    results['cy'] = cy
    results['data'] = {}

    for d in range(1, n+1):
        # Odd way to do this - if we specify a second-stage demand, this loop will break after one run
        if spec_d is not None:
            d = spec_d
        gs = bno.prep_gs(d)
        wd_results = {}
        wd_results['ground_state'] = gs

        ###
        ### Test the expectation value
        qc = QuantumCircuit(2*n)
        qc.prepare_state(gs, list(range(2*n)))
        counts = bno.execute_optimizer(qc)
        exp = bno.process_expectation_value_optimizer(d, counts)
        wd_results['ground_state_expectation_value'] = exp
        ###

        norm = d*cr

        # State prep
        gs_qc = QuantumCircuit(2*n)
        gs_qc.prepare_state(gs)
        gs_op_inverse = gs_qc.inverse()

        # Oracle
        oracle = bno.exact_oracle(d, norm, inverse=False)
        oracle_inverse = bno.exact_oracle(d, norm, inverse=True)

        ### Test the oracle
        qc = QuantumCircuit(2*n+1)
        qc.append(gs_qc, list(range(2*n)))
        qc.append(oracle, list(range(2*n+1)))
        sv = Statevector.from_label('0'*(2*n+1))
        sv = sv.evolve(qc)
        pd = sv.probabilities_dict(qargs=[2*n])
        wd_results['oracle_state'] = sv

        ### Do canonical QAE
        qc = bno.implemented_qae(gs_qc, oracle, gs_op_inverse, oracle_inverse, m, norm)

        sv = Statevector.from_label('0'*(2*n+m+1))
        sv = sv.evolve(qc)
        pd = sv.probabilities_dict(qargs=list(range(m)))
        wd_results['qae_state'] = sv

        results['data'][d] = wd_results

        if spec_d is not None:
            break
    return results


def experiment_sampling_optimize(pdf, t, n, m, spec_d=None, is_uniform=False, exact_oracle=True):
    ''' For a dataset (w/ high variance in Q) get the different convergences in M
        For a single demand output from the first stage, unless demand is not specified
        Do the optimizer to a certain time
    '''    
    results = {}
    results['pdf'] = pdf
    results['wd'] = spec_d
    # costs
    #cy = [0.08, 0.015, 0.1, 0.05, 0.03, 0.15, 0.06, 0.2, 0.02, 0.01, 0.13, 0.17,]
    cy = np.linspace(0.01, 0.2, n)
    #cy = cy[:n]
    #cy = [1.0]*n
    cr = 1.
    cx = [0.4,] # dummy
    bno = BinaryNestedOptimizer(cx, cy, cr, pdf, n, is_uniform=is_uniform)
    e = bno.brute_force_wind_demand_expectation_values()
    results['true_expectation_values'] = e
    results['cy'] = cy
    results['model'] = bno
    results['data'] = {}

    for d in range(0, n+1):
        print('wind demand', d)
        # Odd way to do this - if we specify a second-stage demand, this loop will break after one run
        if spec_d is not None:
            d = spec_d
        gs = bno.prep_gs(d)
        wd_results = {}
        wd_results['ground_state'] = gs

        ###
        ### Test the expectation value
        qc = QuantumCircuit(2*n)
        qc.prepare_state(gs, list(range(2*n)))
        counts = bno.execute_optimizer(qc)
        exp = bno.process_expectation_value_optimizer(d, counts)
        wd_results['ground_state_expectation_value'] = exp
        ###

        norm = d*cr

        # State prep 
        Uopt = bno.adiabatic_evolution_circuit(d, t, t, norm=1)
        #print(Statevector.from_label('0'*n).evolve(bno.dicke_state_circuit(2)).probabilities_dict())
        #exit(1)
        Uopt_inverse = Uopt.inverse()
        #Uopt = Uopt.to_gate()
        #Uopt_inverse = Uopt_inverse.to_gate()
        #gs_qc = QuantumCircuit(2*n)
        #gs_qc.prepare_state(gs)
        #gs_op_inverse = gs_qc.inverse()

        ## TODO WARNING NOTE the oracle is perfect for the exactly prepared wavefunction, 
        #### but has some hidden error for the qaoa prepared wavefunction (flipping bitstrings fixes)

        # Oracle
        if exact_oracle:
            oracle = copy.deepcopy(bno.exact_oracle(d, norm, inverse=False))
            oracle_inverse = copy.deepcopy(bno.exact_oracle(d, norm, inverse=True))
        else:
            c = 1#np.pi
            oracle = bno.single_oracle_sin_inconstraint(c, norm, inverse=False)
            oracle_inverse = bno.single_oracle_sin_inconstraint(c, norm, inverse=True)
            #oracle = bno.single_oracle_asin_inconstraint(norm, inverse=False)
            #oracle_inverse = bno.single_oracle_asin_inconstraint(norm, inverse=True)

        ### Just plain wavefunction
        sv = Statevector.from_label('0'*(2*n))
        sv = sv.evolve(qc)
        wd_results['qaoa_state'] = sv

        ### Test the oracle
        qc = QuantumCircuit(2*n+1)
        qc.append(Uopt, list(range(2*n)))
        qc.append(oracle, list(range(2*n+1)))
        sv = Statevector.from_label('0'*(2*n+1))
        sv = sv.evolve(qc)
        #pd = sv.probabilities_dict(qargs=[2*n])
        wd_results['oracle_state'] = sv

        ### Do canonical QAE
        qc = bno.implemented_qae(Uopt, oracle, Uopt_inverse, oracle_inverse, m, norm)

        sv = Statevector.from_label('0'*(2*n+m+1))
        sv = sv.evolve(qc)
        #pd = sv.probabilities_dict(qargs=list(range(m)))
        wd_results['qae_state'] = sv

        results['data'][d] = wd_results

        if spec_d is not None:
            break
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nbits', type=int) # number of variables in the wind system
    parser.add_argument('-m', '--mbits', type=int) # number of readout bits for QAE
    parser.add_argument('-t', '--time', type=int) # annealing time. if not specified, we automatically construct the wavefunction
    parser.add_argument('-d', '--wdemand', type=int) # the amount of demand the second stage is responsible for. If none, we run over the full range
    parser.add_argument('--uniform', action='store_true') # hit this if we want a uniform pdf
    parser.add_argument('--approx', action='store_true') # hit this if we want a non exact oracle
    # TODO specify distribution?

    args = parser.parse_args()
    n = args.nbits
    m = args.mbits
    t = args.time
    w = args.wdemand
    is_unipdf = args.uniform
    exact_oracle = not args.approx

    if not is_unipdf:
        pdf = make_pdf_skewnormal(n)
    else:
        pdf = make_pdf_uniform(n)

    # if we assume a perfect optimization
    if t is None:
        fname = 'sampling_data/sampling_uniform={}_n={}_m={}.pkl'.format(is_unipdf,n,m)
        results = experiment_sampling_makewvfn(pdf, n, m, spec_d=w, is_uniform=is_unipdf)
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
    else:
        fname = 'sampling_data/sampling_t={}_exactoracle={}_uniform={}_n={}_m={}.pkl'.format(t,exact_oracle,is_unipdf,n,m)
        results = experiment_sampling_optimize(pdf, t, n, m, spec_d=w, is_uniform=is_unipdf, exact_oracle=exact_oracle)
        with open(fname, 'wb') as f:
            pickle.dump(results, f)

if __name__=='__main__':
    main()