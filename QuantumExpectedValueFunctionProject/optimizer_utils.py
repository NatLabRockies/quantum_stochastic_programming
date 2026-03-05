#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 08:37:56 2023

@author: crotello
"""

import numpy as np
import copy
import scipy #as sp
from scipy.optimize import LinearConstraint, Bounds, minimize, shgo, milp
import matplotlib.pyplot as plt
#import seaborn as sns
from fractions import Fraction
from decimal import Decimal
from math import gcd

from qiskit import QuantumCircuit, Aer, BasicAer#, ClassicalRegister, QuantumRegister
#from qiskit import BasicAer, Aer
from qiskit.compiler import transpile
#from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator#, Pauli
#from qiskit.extensions import UnitaryGate
#from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library.standard_gates import RXGate, XGate, CXGate, CSwapGate
from qiskit.circuit.library.standard_gates import CCZGate, HGate, SwapGate, iSwapGate, CCXGate, CZGate



def int_to_bstr_N(i,N):
    if N == 0:
        return ''
    else:
        return (('{0:0' + str(N) + 'b}').format(i))[::-1]




def plot_counts(counts, nqubits):
    xdat = []
    ydat = []
    shots = sum(counts.values())
    for i in range(2**nqubits):
        #bstr = ''.join(list(('{0:0' + str(nqubits) + 'b}').format(i))[::-1])
        bstr = ('{0:0' + str(nqubits) + 'b}').format(i)
        xdat.append(bstr)
        if bstr in counts.keys():
            ydat.append(counts[bstr]/shots)
        else:
            ydat.append(0)
    plt.bar(xdat, ydat)
    
def plot_statevector(vec, nqubits):
    xdat = []
    ydat = []
    for i,val in enumerate(vec):
        bstr = ('{0:0' + str(nqubits) + 'b}').format(i)
        xdat.append(bstr)
        ydat.append(np.abs(val)**2)
    plt.bar(xdat,ydat)





class VariableRegister:
    ''' class VariableRegister: 
            Use this object to store an integer variable in a register of qubits, and to perform interactions 
            between registers of qubits. We use a register of qubits 's', which represent binary variables 'y' 
            to encode the integer variable 'v'. Depending on the encoding, this integer variable 'v' is created 
            by a linear combination of binary variables 'sum b*y', and 'b' depends on the type of encoding.
    '''
    def __init__(self, max_value, encoding='binary'):
        try:
            assert(encoding in ['binary', 'unary'])
        except:
            print("ERROR: Encoding type must be 'binary' or 'unary', got '{}'".format(encoding))
        self.encoding = encoding
        self.max_value = max_value
        if self.max_value > 0:
            self.width = int(np.log2(self.max_value)) + 1 if self.encoding == 'binary' else self.max_value
        elif self.max_value == 0:
            self.width = 0
        else:
            print("ERROR: max_value must be non-negative")
        
        if 2**self.width - 1 > self.max_value and self.encoding == 'binary':
            print("WARNING: Possible variable range ({}) is larger than variable maximum value ({})".format(2**self.width-1, self.max_value))
        
    def __str__(self):
        return '{} encoded variable, 0<=v<={}, width={}'.format(self.encoding, self.max_value, self.width)
    
    
   # less than op impls 
    def __unaryLessThanOperator(self, other, amplitude):
        ''' __unaryLessThanOperator
            If this > other, return e^i(this - other). Otherwise, Identity
        '''
        # TODO: implement with a real circuit
        qc = QuantumCircuit(self.width + other.width)
        length = self.width+other.width
        dim = 2**(length)
        op = np.zeros(shape=(dim,dim), dtype=np.complex128)
        for i in range(dim):
            bstr = ''.join(list(int_to_bstr_N(i, length)))
            my_bstr = bstr[:self.width]
            other_bstr = bstr[self.width:]
            my_val = self.getValue(my_bstr)
            other_val = self.getValue(other_bstr)
            if my_val > other_val:
                op[i,i] = np.exp(1j*amplitude*(my_val-other_val)) 
            else:
                op[i,i] = 1.
        qc.append(Operator(op), range(length))
        return qc
    
    def __binaryLessThanOperator(self, other, amplitude):
        ''' __binaryLessThanOperator
            If this > other, return e^i(this - other). Otherwise, Identity
        '''
        # TODO: implement with a real circuit
        qc = QuantumCircuit(self.width + other.width)
        length = self.width+other.width
        dim = 2**(length)
        op = np.zeros(shape=(dim,dim), dtype=np.complex128)
        for i in range(dim):
            bstr = ''.join(list(int_to_bstr_N(i, length)))
            my_bstr = bstr[:self.width]
            other_bstr = bstr[self.width:]
            my_val = self.getValue(my_bstr[::-1])
            other_val = other.getValue(other_bstr[::-1])
            if my_val > other_val:
                op[i,i] = np.exp(1j*amplitude*(my_val-other_val))
            else:
                op[i,i] = 1.
        qc.append(Operator(op), range(length))
        return qc


    # less than a value
    def __binaryLessThanValue(self, value, amplitude): 
        ''' __binaryLessThanValue
        '''
        # TODO: implement with a real circuit
        qc = QuantumCircuit(self.width)
        dim = 2**self.width
        op = np.zeros(shape=(dim,dim), dtype=np.complex128)
        for i in range(dim):
            bstr = ''.join(list(int_to_bstr_N(i, self.width)))
            my_val = self.getValue(bstr[::-1])
            if my_val > value:
                op[i,i] = np.exp(1j*amplitude*(my_val-value))
            else:
                op[i,i] = 1.
        qc.append(Operator(op), range(self.width))
        return qc

    # number op impls
    def __binaryNumberOperator(self, amplitude):
        ''' __binaryNumberOperator
            Return amplitude * v in Little Endian
        '''
        qc = QuantumCircuit(self.width)
        for i in range(self.width):
            qc.p(amplitude * 2**i, i)
        return qc
        
    def __unaryNumberOperator(self, amplitude):
        ''' __unaryNumberOperator
            Return amplitude * v encoded as a unary operator
        '''
        qc = QuantumCircuit(self.width)
        for i in range(self.width):
            qc.p(amplitude, i)
        return qc
    
    
    # swap op impls
    def __unarySwapOperator(self, other, amplitude):
        ''' __unarySwapOperator
            Return integer encoding as a unary operator
            For now, the swap interactions are all-to-all, single Trotter step
        ''' 
        qc = QuantumCircuit(self.width+other.width, name='swap({})'.format(np.round(amplitude)))
        for i in range(self.width):
            for j in range(self.width, self.width+other.width):
                qc.append(SwapGate().power(amplitude), [i,j])
        return qc
    
    def __binarySwapOperator(self, other, amplitude):
        ''' __binarySwapOperator
            Return integer swap operation for the binary encoding
        '''
        ''' kind of works
        length = self.width+other.width
        qc = QuantumCircuit(length)
        # TODO: find a real circuit
        dim = 2**length
        ham = np.zeros(shape=(dim,dim), dtype=np.complex128)
        for i in range(dim):
            # decode first number
            bstr_i = int_to_bstr_N(i, length)[::-1]
            my_int_i = self.getValue(bstr_i[:self.width])
            other_int_i = other.getValue(bstr_i[self.width:])
            for j in range(dim):
                # decode second number
                bstr_j = int_to_bstr_N(j, length)[::-1]
                my_int_j = self.getValue(bstr_j[:self.width])
                other_int_j = other.getValue(bstr_j[self.width:])
                if my_int_i + other_int_i == my_int_j + other_int_j and i!=j:
                    ham[i,j] = np.abs(my_int_i - my_int_j)*self.max_value#/self.max_value #1.
        #ham[0,0] = 1.
        #ham[dim-1,dim-1] = 1.
        #ham /= sp.linalg.norm(ham)
        op = sp.linalg.expm(-1j *np.pi/2* amplitude * ham)
        qc.append(Operator(op), list(range(length)))
        '''
        length = self.width+other.width
        ancilla = length
        qc = QuantumCircuit(length)
        # TODO: account for different register width, remove ancilla
        assert(self.width == other.width)
        for i in range(self.width):
            j = i+self.width
            #qc.append(SwapGate().power(amplitude), [i,j]) # swap bits of equal value
            # promote two bits of equal value - manual matrix, bit order (i+1),i,j
            op = [[1,0,0,0,0,0,0,0],#111
                  [0,1,0,0,0,0,0,0],#110
                  [0,0,1,0,0,0,0,0],#101
                  [0,0,0,np.cos(1/2*amplitude*np.pi/2),1j*np.sin(1/2*amplitude*np.pi/2),0,0,0],#100 - 011
                  [0,0,0,1j*np.sin(1/2*amplitude*np.pi/2),np.cos(1/2*amplitude*np.pi/2),0,0,0],#011 - 100
                  [0,0,0,0,0,1,0,0],#010
                  [0,0,0,0,0,0,1,0],#001
                  [0,0,0,0,0,0,0,1]]#000
            if i < self.width-1:
                qc.append(Operator(op), [i,j,i+1])
                qc.append(Operator(op), [i,j,j+1])
            # swap bits of equal significance
            qc.append(SwapGate().power(amplitude), [i,j]) # swap bits of equal value
        return qc
    
    # product of ops impl
    def __unaryProductOperator(self, other, amplitude):
        ''' __unaryProductOperator
        '''
        qc = QuantumCircuit(self.width + other.width)
        for j in range(self.width):
            for k in range(self.width, self.width + other.width):
                qc.cp(amplitude, j, k)
        return qc
    
    def __binaryProductOperator(self, other, amplitude):
        ''' __binaryProductOperator
        '''
        qc = QuantumCircuit(self.width + other.width)
        for j in range(self.width):
            for k in range(other.width):
                qc.cp(2**j * 2**k * amplitude, j, k+self.width)
        return qc
    
    # Oracle implementation
    def __unaryOracleLessThanOperator(self, other, amplitude):
        qc = QuantumCircuit(self.width + other.width + 1)
        length = self.width+other.width+1
        ancilla = length - 1
        dim = 2**(length)
        op = np.zeros(shape=(dim,dim), dtype=np.complex128)
        for i in range(dim):
            bstr = ''.join(list(int_to_bstr_N(i, length)))
            my_bstr = bstr[:self.width]
            other_bstr = bstr[self.width:]
            my_val = self.getValue(my_bstr)
            other_val = self.getValue(other_bstr)
            if my_val > other_val:
                op[i,i] = np.exp(1j*amplitude*(my_val-other_val)) 
            else:
                op[i,i] = 1.
        qc.append(Operator(op), range(length))
        return qc

    
    # squared op impl
    def __binarySquaredOperator(self, amplitude):
        ''' __binarySquaredOperator
        '''
        qc = QuantumCircuit(self.width)
        for j in range(self.width):
            for k in range(j+1,self.width):
                qc.cp(2*2**j * 2**k * amplitude, j, k)
        return qc
        
    def __unarySquaredOperator(self, amplitude):
        ''' __unarySquaredOperator
        '''
        qc = QuantumCircuit(self.width)
        for j in range(self.width):
            for k in range(j+1,self.width):
                qc.cp(2*amplitude, j, k)
        return qc
        
        
    # callers    
    def swapOperator(self, other, amplitude):
        ''' swapOperator
            Return a circuit which does an amplitude * "integer swap" between two registers.
        '''
        if self.encoding == 'binary':
            if other.encoding != 'binary':
                print("ERROR: I have encoding '{}' and do not support SWAP with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__binarySwapOperator(other, amplitude)
        elif self.encoding == 'unary':
            if other.encoding != 'unary':
                print("ERROR: I have encoding '{}' and do not support SWAP with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__unarySwapOperator(other, amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)
    
    def numberOperator(self, amplitude):
        ''' numberOperator
            Return a circuit which does amplitude*v using the corresponding encoding.
        '''
        if self.encoding == 'binary':
            return self.__binaryNumberOperator(amplitude)
        elif self.encoding == 'unary':
            return self.__unaryNumberOperator(amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)
            
    def productOperator(self, other, amplitude):
        ''' productOperator
            Return a circuit which, given two registers, computes amplitude*this*other
        '''
        if self.encoding == 'binary':
            if other.encoding != 'binary':
                print("ERROR: I have encoding '{}' and do not support N_i*N_j with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__binaryProductOperator(other, amplitude)
        elif self.encoding == 'unary':
            if other.encoding != 'unary':
                print("ERROR: I have encoding '{}' and do not support N_i*N_j with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__unaryProductOperator(other, amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)
            
    def squaredOperator(self, amplitude):
        ''' squaredOperator
            Return a circuit which does amplitude*this*this
        '''
        if self.encoding == 'binary':
            return self.__binarySquaredOperator(amplitude)
        elif self.encoding == 'unary':
            return self.__unarySquaredOperator(amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)
        
    def lessThanOperator(self, other, amplitude):
        ''' lessThanOperator
            Return a circuit which does phase amplitude*(this-other) if this > other, otherwise 1
        '''
        if self.encoding == 'binary':
            if other.encoding != 'binary':
                print("ERROR: I have encoding '{}' and do not support LTPhase with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__binaryLessThanOperator(other, amplitude)
        elif self.encoding == 'unary':
            if other.encoding != 'unary':
                print("ERROR: I have encoding '{}' and do not support LTPhase with a register of encoding '{}'".format(self.encoding, other.encoding))
                exit(1)
            return self.__unaryLessThanOperator(other, amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)

    def oracleLessThanOperator(self, other, amplitude):
        ''' 
            Return a circuit which rotates an ancilla by (this-other) if this > other
        '''
        if self.encoding == 'binary':
            print("ERROR: oracleLessThanOperator not implemented for encoding 'binary'")
            exit(1)
        elif self.encoding == 'unary':
            if other.encoding != 'unary':
                print("ERROR: I have encoding {} but other has encoding {}".format('unary', 'binary'))
            return self.__unaryOracleLessThanOperator(other, amplitude)
        else:
            print("ERROR: Unimplemented")
            exit(1)
    
    def lessThanValue(self, value, amplitude):
        ''' lessThanValue
            Return a circuit which does phase amplitude*(this-value) if this > value, otherwise Identity
        '''
        if self.encoding == 'binary':
            return self.__binaryLessThanValue(value, amplitude)
        elif self.encoding == 'unary':
            print("ERROR, unimplemented lessThanValue for unary encoding")
            # TODO do we need to implement this?
        else:
            print("ERROR: Unimplemented")
            exit(1)
            
    def getValue(self, bstr):
        ''' Given a register's bitstring, return the integer it encodes
        '''
        assert(len(bstr) == self.width)
        if self.encoding == 'binary':
            v = 0
            revbstr = bstr[::-1] # reverse the bitstring to do Little Endian encoding
            for i in range(self.width):
                v += int(revbstr[i])*2**i
            return v
        elif self.encoding == 'unary':
            return sum([int(b) for b in bstr])
        else:
            print("ERROR: Unimplemented")
            exit(1)
            
    def setValue(self, integer):
        ''' Given an integer, return a circuit which creates that integer
        '''
        assert(integer <= self.max_value)
        qc = QuantumCircuit(self.width)
        if self.encoding == 'unary':
            for i in range(integer):
                qc.x(i)
        elif self.encoding == 'binary':
            bstr = int_to_bstr_N(integer, self.width)
            for i,val in enumerate(bstr):
                if val == '1':
                    qc.x(i)
        return qc    
        
        
            
            
class PowerSystem_1Bus:
    ''' class PowerSystem_1Bus:
            Use this object to define an all-to-all lossless power grid (single bus approximation)
            
            
            Notes below are for the optimizer
            
            Define several ways to solve the optimization problem, enumerated here:
                - Simultaneously by encoding with the probability distribution
                    - Constraint violation by slack variable
                    - Constraint never violated
                    
                - Expansion of different scenarios; 
                
            NOTE: discretization and costs must follow this constraint: cost*(levels-1) = max_cost
    '''    
    def __init__(self, gas_costs, wind_costs, decision_levels, undersatisfied_cost, demand, pdf, normalization=None):
        ''' ctor.
            initialize the power system with the costs of each variable, levels, and pdf
        '''
        assert(np.isclose(sum(pdf.values()),1.))
        
        # number of each generator type and cost
        self.num_gas_generators = len(gas_costs)
        self.num_wind_turbines = len(wind_costs)
        self.gas_costs = gas_costs
        self.wind_costs = wind_costs
        self.undersatisfied_cost = undersatisfied_cost
        
        self.normalization = normalization
        if normalization is not None: # sometimes, we want to normalize the operator amplitudes; this parameter is something like (max cost, max op amplitude)
            obj = normalization[0]
            su2 = normalization[1]
            for i in range(len(self.gas_costs)):
                self.gas_costs[i] *= su2/obj
            for i in range(len(self.wind_costs)):
                self.wind_costs[i] *= su2/obj
            self.undersatisfied_cost *= su2/obj
        
        # discretization
        # TODO: make this variable specific
        self.decision_levels = decision_levels
        # demand
        self.demand = demand
        
        # probability distriution function, or scenarios
        self.pdf = pdf
        self.scenarios = list(self.pdf.keys()) # index is scenario 'number'

        # decompose pdf into samples
        '''
        self.sample_list = []
        self.sample_hist = {}
        denominators = [Fraction(Decimal(str(self.pdf[scene]))).denominator for scene in self.scenarios] # iterate over 'scenarios' to preserve order of vars
        # get least common multiple of the Pr's denominators
        lcm = 1 
        for denom in denominators:
            lcm = lcm*denom//gcd(lcm,denom)
        # convert Pr to frequency of occurency in SAA
        for scene, prob in self.pdf.items():
            reps = prob * lcm
            self.sample_hist[scene] = int(reps)
            for _ in range(int(np.around(reps))):
                self.sample_list.append(scene)
        '''
        #print(self.sample_hist)
        # expanded costs
        self.variable_costs = copy.deepcopy(self.gas_costs)
        for scenario, amp in self.pdf.items():
            [self.variable_costs.append(amp*w) for w in self.wind_costs]
            self.variable_costs.append(amp*self.undersatisfied_cost)
            
        
    def __str__(self):
        s = 'PowerSystem_1Bus\n'
        s += '\tGas generators: ' + str(self.num_gas_generators)
        s += '\n\tWind turbines: ' + str(self.num_wind_turbines) 
        s += '\n\tc_x = ' + str(self.gas_costs) + '\n\tc_w = ' + str(self.wind_costs)
        s += '\n\tc_y- = ' + str(self.undersatisfied_cost)
        s += '\n\tDemand: ' + str(self.demand)
        s += '\n\tDecision levels (per generator): ' + str(self.decision_levels)
        s += '\n\tProbability distribution function: ' + str(self.pdf)
        s += '\n\tScenarios:'
        for i,scenario in enumerate(self.scenarios):
            s += '\n\t\t{}: {}'.format(i, scenario)
        if self.normalization is not None:
            s += '\n\tNormalization: ' + str(self.normalization[0]) + '->' + str(self.normalization[1])
        else:
            s += '\n\tNormalization: False'
        s+='\n'
        return s
    
    def _price(self, x):
        cost = sum([self.variable_costs[i]*x[i] for i,_ in enumerate(x)])
        return cost

    def normalize(self, val):
        return val * self.normalization[1] / self.normalization[0]

    def unNormalize(self, val):
        return val * self.normalization[0] / self.normalization[1]

    def getFirstStageCosts(self, gas_decisions):
        ''' Given a list of decisions over the gas variables, return the cost and wind decision variables for each
        '''
        decision_to_cost = {}
        decision_to_decision = {}
        for gas_dec in gas_decisions:
            result = self.cobylaSolve(gas_values=gas_dec)
            if result.fun is not None:
                decision_to_cost[gas_dec] = result.fun
            decision_to_decision[gas_dec] = result.x
        return decision_to_cost,decision_to_decision
    
    def cobylaSolve(self, discrete=True, gas_values=None):
        ''' Use cobyla to solve the optimization problem. 
            Either solve the entire two-stage problem via direct expansion or, 
            given a gas decision, solve _only_ the second stage 
        '''
        num_vars = self.num_gas_generators + len(self.scenarios)*(self.num_wind_turbines + 1)
        gas_upper_bounds = [self.decision_levels - 1] * self.num_gas_generators
        
        # Get the second stage upper bounds
        second_stage_upper_bounds = []
        for scenario_id, scenario in enumerate(self.scenarios):
            for wind in scenario:
                second_stage_upper_bounds.append(wind)
            second_stage_upper_bounds.append(self.demand)
        if gas_values is None:
            bounds = Bounds([0.]*num_vars, gas_upper_bounds + second_stage_upper_bounds)
        else:
            assert(len(gas_values) == self.num_gas_generators)
            gas_values = list(gas_values)
            other_zeros = [0.]*(num_vars - self.num_gas_generators)
            bounds = Bounds(gas_values + other_zeros, gas_values + second_stage_upper_bounds)
                                        
        # Set the linear constraint
        A = []
        for scenario_id, scenario in enumerate(self.scenarios):
            constraint_row = [1]*self.num_gas_generators + [0]*scenario_id*(self.num_wind_turbines + 1)
            for wind in scenario:
                constraint_row.append(1)
            constraint_row.append(1)
            constraint_row += [0]*(num_vars - len(constraint_row))
            A.append(constraint_row)
        lin_constraint = LinearConstraint(A, self.demand, self.demand)

        # QUICK AND DIRTY force non-anticipativity among wind
        #B = []
        #for w in range(self.num_wind_turbines):
        #    for scenario_i, scenario in enumerate(self.scenarios):
        #        for scenario_j, scenario in enumerate(self.scenarios):
        #            if scenario_j <= scenario_i:
        #                continue
        #            constraint_row = [0]*self.num_gas_generators + [0]*scenario_i*(self.num_wind_turbines + 1)
        #            # si
        #            constraint_row += [0]*w + [1] + [0]*(self.num_wind_turbines-w)
        #            constraint_row += [0]*(scenario_j-scenario_i-1)*(self.num_wind_turbines + 1)
        #            # sj
        #            constraint_row += [0]*w + [-1] + [0]*(self.num_wind_turbines-w)
        #            constraint_row += [0]*(num_vars - len(constraint_row))
        #            #print(num_vars, len(constraint_row), constraint_row)
        #            B.append(constraint_row)
        #nonant_constraint = LinearConstraint(B, 0, 0)
                
        
        # Optimize
        # If we use discrete variables 
        if discrete:
            #res = milp(self.variable_costs, constraints=[lin_constraint, nonant_constraint], bounds=bounds, integrality=1)
            res = milp(self.variable_costs, constraints=[lin_constraint,], bounds=bounds, integrality=1)
        else:
            # TODO: initial condition 
            x0 = [0] * num_vars
            x0[0] = self.demand
            res = minimize(self._price, x0, constraints=[lin_constraint], bounds=bounds)
        
        return res
 

    def plotMeasurementsVExpectedCost(self, measured_decisions, pdf=False):
        ''' plotMeasurementsVExpectedCost 
            Given a dictionary of gas generator measured results, plot the measurement count vs. expected cost of that decision
            It is the quantum optimizer's job to create a histogram of gas decisions.
        '''
        #plt.cla()
        first_index = 2 if not pdf else 3
        ex_costs,_ = self.getFirstStageCosts(measured_decisions.keys())

        #plt.figure()
        plt.figure().set_figheight(10)

        plt.subplot(first_index, 1, 1)
        measured_decisions = {k:measured_decisions[k] for k in sorted(measured_decisions.keys())}
        plt.bar([str(a) for a in measured_decisions.keys()], measured_decisions.values())
        plt.ylabel(r"Pr$(x)$")
        plt.xlabel(r"$x$")

        plt.subplot(first_index, 1, 2)
        ex_costs = {k:ex_costs[k] for k in sorted(ex_costs.keys())}
        if self.normalization is not None:
            #plt.bar([str(a) for a in val_of_decisions.keys()], [system.normalization[0]/system.normalization[1]*v for v in val_of_decisions.values()])
            #plt.bar([str(a) for a in ex_costs.keys()], [v for v in ex_costs.values()])
            plt.bar([str(a) for a in ex_costs.keys()], [self.unNormalize(v) for v in ex_costs.values()])
        else:
            plt.bar([str(a) for a in ex_costs.keys()], ex_costs.values())
        plt.ylabel(r'$\sum_g c_gx_g + \mathbb{E}_\xi [L(x,\xi)]$')
        plt.xlabel(r"$x$")

        if pdf:
            plt.subplot(3, 1, 3)
            pdf = {k:self.pdf[k] for k in sorted(self.pdf.keys())}
            plt.bar([str(a) for a in pdf.keys()], pdf.values())
            plt.ylabel(r"Pr$(\xi)$")
            plt.xlabel(r"$\xi$")
        plt.tight_layout()
        return plt

    
    
   
def main():
    0

if __name__=='__main__':
    main()          
            
            
            