#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:43:41 2023

@author: crotello
"""

import optimizer_utils 

import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, Aer
from qiskit.compiler import transpile
from qiskit.tools.visualization import plot_histogram


class Optimizer_Expanded:
    def __init__(self, system, encoding):
        ''' ctor. 
            Take the PowerSystem we will be solving, reserve variables and qubits
            
            Order the variables as gas, wind+slack(scenario1), wind+slack(scenario2), wind+slack(scenario3)...
        '''
        assert(encoding == 'binary' or encoding == 'unary')
        self.system = system
        self.encoding = encoding
        self.num_scenarios = len(system.pdf.keys())
        self.num_variables = system.num_gas_generators + self.num_scenarios*system.num_wind_turbines + self.num_scenarios
        # get cost for each variable in this problem encoding
        self.variable_costs = system.gas_costs
        for i,_ in enumerate(system.scenarios):
            [self.variable_costs.append(cost) for cost in system.wind_costs]
            self.variable_costs.append(system.undersatisfied_cost)
        
        # declare variables
        # assign varids, expect order gas, wind(+slackscenario1), wind+slack(scenario2), ... 
        self.varids = list(range(self.num_variables))
        # Limit variable register max values
        # declare gas variables
        self.variables =[optimizer_utils.VariableRegister(system.decision_levels-1, encoding) for _ in range(system.num_gas_generators)]
        # declare a set of wind variables and the slack variable for each scenario
        for scenario in system.scenarios:
            # the wind turbines for this scenario
            for w in range(system.num_wind_turbines):
                # we can only use a hard-cap for unary encodings
                #if encoding == 'unary':
                    self.variables.append(optimizer_utils.VariableRegister(scenario[w], encoding))
                #elif encoding == 'binary':
                #    # get the register-max for the scenario

            # slack variable for that scenario - the slack variable is always the last variable in the scenario
            self.variables.append(optimizer_utils.VariableRegister(self.system.demand, encoding))
        # map gas to varids
        self.gas_to_varids = {i : i for i in range(system.num_gas_generators)}
        # map scenario to varids
        self.scenario_to_varids = {i : list(range((system.num_wind_turbines+1)*i + system.num_gas_generators, 
                                                  (system.num_wind_turbines+1)*i + system.num_gas_generators + system.num_wind_turbines + 1))
                                   for i in range(self.num_scenarios)}
        # reserve qubits
        self.num_qubits = sum([var.width for var in self.variables])
        self.varid_to_qubits = {}
        qubit = 0
        for i,reg in enumerate(self.variables):
            self.varid_to_qubits[i] = list(range(qubit, qubit+reg.width))
            qubit += reg.width
   

    def __str__(self):
        s = 'Optimizer_Expanded\n'
        s += '\t#Variables: ' + str(self.num_variables) + '\n'
        s += '\t#Qubits: ' + str(self.num_qubits) + '\n'
        s += '\tVariables: \n'
        for varid,var in enumerate(self.variables):
            s += '\t\tVar({}): Q={}, c={} \n'.format(varid, self.varid_to_qubits[varid], self.variable_costs[varid])
        s += '\tScenarios: \n'
        for scenarioid, varids in self.scenario_to_varids.items():
            s += '\t\tScen({}): vars={}, xi={} \n'.format(scenarioid, varids, self.system.scenarios[scenarioid])
        return s
        
    def solveAnnealing(self, time, method='QUBO', num_meas=10_000, penalty=1):
        ''' solveAnnealing
            Solve the optimization problem with an annealing routine, specify if we use 
            a Dicke state and constraint preserving mixer or QUBO with a penalty Hamiltonian
        '''
        if self.system.normalization is not None and penalty is not None:
            #penalty *= self.system.normalization[1]/self.system.normalization[0]
            penalty = self.system.normalize(penalty)
        demand = self.system.demand
        # NOTE this will require a bit more work
        if self.system.normalization is not None:# and penalty is not None:
            demand = self.system.normalize(demand)
        # warning
        #if phase!='PEN' and penalty is not None:
        #    print("WARNING: specified a penalty but the penalty cost Hamiltonian is not used")
        #if penalty is None and phase == 'PEN':
        #    print("ERROR: if PEN is specified (penalty Hamiltonian) we need a penalty specified")
        #    exit(1)

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        if method == 'QUBO':
            for qubit in range(self.num_qubits):
                qc.h(qubit)
        else:
            print("Unimplemented annealing solution method: {}".format(method))
            return 0
        
        for t in range(time):
            f = (t+1)/(time+1)
            ####
            # cost operator
            ####
            # cost operator - gas
            for _,varid in self.gas_to_varids.items():
                reg = self.variables[varid]
                cost = self.variable_costs[varid]
                qc.append(reg.numberOperator(f*cost), self.varid_to_qubits[varid])
            # cost operator - second stage
            for scenario_id,varids in self.scenario_to_varids.items():
                # get the probability of this scenario
                pr = self.system.pdf[self.system.scenarios[scenario_id]]
                #print(scenario_id)
                # apply cost operator to each set of variables in this scenario
                for varid in varids:
                    reg = self.variables[varid]
                    cost = self.variable_costs[varid]
                    #print(varid,cost)
                    qc.append(reg.numberOperator(pr*f*cost), self.varid_to_qubits[varid])
            qc.barrier()
            
            # cost operator - penalty term
            # each scenario has an equlity constraint enforced with quadratic penalty
            # additionally, binary encodings need to enforce stochastic inequality constraints with a penalty
            for scenario_id, varids in self.scenario_to_varids.items():
                gas_varids = list(self.gas_to_varids.values())
                all_varids = gas_varids + varids
                # penalty constraint
                for j,varid_j in enumerate(all_varids):
                    # -2*gamma*d*f * sum_j(N_j)
                    qc.append(self.variables[varid_j].numberOperator(-2 * penalty * demand * f), 
                              self.varid_to_qubits[varid_j])#[::-1])
                    # gamma*f * sum_j(N_j*N_j)
                    qc.append(self.variables[varid_j].squaredOperator(penalty * f), 
                              self.varid_to_qubits[varid_j])#[::-1])
                    # 2*gamma*f * sum_j<k(N_j*N_k)
                    for varid_k in all_varids[j+1:]:
                        qc.append(self.variables[varid_j].productOperator(self.variables[varid_k], 2*penalty*f),
                                  #(self.varid_to_qubits[varid_k]+self.varid_to_qubits[varid_j])[::-1])
                                  #(self.varid_to_qubits[varid_k][::-1]+self.varid_to_qubits[varid_j][::-1]))
                                  self.varid_to_qubits[varid_j]+self.varid_to_qubits[varid_k])
                                  #self.varid_to_qubits[varid_k] + self.varid_to_qubits[varid_j])
                # stochastic inequality constraint
                continue
                if self.encoding == 'binary':
                    for j,varid_j in enumerate(varids[:-1]):
                        # NOTE we assume j is that variables position within the pdf
                        qc.append(self.variables[varid_j].lessThanValue(self.system.scenarios[scenario_id][j], 
                                                                        self.system.undersatisfied_cost*f),
                                                                        #penalty*f),
                                  self.varid_to_qubits[varid_j])#[::-1])
            #break
            qc.barrier()
            ####
            # mixing operator
            ####
            if method == 'QUBO':
                for i in range(self.num_qubits):
                    qc.rx(1-f, i)
            else:
                print("Unimplemented mixing operator for method: {}".format(method))
        qc.measure(list(range(self.num_qubits)), list(range(self.num_qubits)))
        
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)
        #print(qc)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        return counts

    def bstrToVars(self, bstr):
        bstr = bstr[::-1]
        i = 0
        values = []
        for varid,variable in enumerate(self.variables):
            substr = bstr[i:i+variable.width]
            values.append(variable.getValue(substr))
            i+=variable.width
        return values
    
    def countsBstrToVars(self, counts):
        hist = {}
        for key, values in counts.items():
            hist[tuple(self.bstrToVars(key))] = values
        return hist
    
    def getGasCounts(self, counts):
        hist = self.countsBstrToVars(counts)
        gas_hist = {}
        for key,value in hist.items():
            gas = key[:len(self.gas_to_varids.keys())]
            if gas not in gas_hist.keys():
                gas_hist[gas] = 0.
            gas_hist[gas] += value
        return gas_hist


    

def _TEST_expandedPowerSystem1Bus_penalty2_unary_hardmax():
    ## Test the PowerSystem_1Bus 
    print("Testing Optimizer_Expanded.solveAnnealing for fewer scenarios, different slack costs")
    ### 1 Gas, 1 Wind, 2 scenarios
    print("\ttest costs [x=3,w=1,y=10] xi=[0.5,0,0,0.5] d=3")
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            #pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
            pdf={tuple([3]): 0.5, tuple([0]): 0.5}
        )
    ### OPTIMAL choice x=3, w0=w3=0
    opt = Optimizer_Expanded(system, 'unary')
    counts = opt.solveAnnealing(6, method='QUBO', penalty=2)
    h = opt.countsBstrToVars(counts)
    a = []
    b = []
    for s,v in h.items():
        if v > 50:
            a.append(str(s))
            b.append(v/1_000)
    plt.bar(a,b)
    plt.xlabel("Decisions")
    plt.ylabel("Pr(decision)")
    plt.show()
    decision = max(zip(h.values(), h.keys()))[1]
    assert(decision == (3,0,0,0,0))
    
    ### 1 Gas, 1 Wind, 1 scenarios and 4 levels
    print("\ttest costs [x=3,w=1,y=10] xi=[0,1,0,0] d=2")
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=2, 
            pdf={tuple([1]): 1.}
        )
    ### OPTIMAL choice x=1, w0=0, w1=w2=w3=1
    opt = Optimizer_Expanded(system, 'unary')
    counts = opt.solveAnnealing(10, method='QUBO', penalty=2)
    h = opt.countsBstrToVars(counts)
    #plt.bar([str(s) for s in h.keys()], list(h.values()))
    decision = max(zip(h.values(), h.keys()))[1]
    # a = []
    # b = []
    # for s,v in h.items():
    #     if v > 200:
    #         a.append(str(s))
    #         b.append(v)
    # plt.bar(a,b)
    #print(decision, h[tuple([1,1,0,1,0,1,0,0,1])])
    assert(decision == (1,1,0))
    
    ### 1 Gas, 2 Wind, 2 scenarios
    print("\ttest costs [x=3,w0=1,w1=1,y=10] xi=[0.5,0,0,0.5] d=3")
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1,1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            #pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
            pdf={tuple([2,0]): 0.5, tuple([0,3]): 0.5}
        )
    ### OPTIMAL choice x=3, w0=w3=0
    opt = Optimizer_Expanded(system, 'unary')
    print(opt)
    counts = opt.solveAnnealing(30, method='QUBO', penalty=2)
    h = opt.countsBstrToVars(counts)
    a = []
    b = []
    for s,v in h.items():
        if v > 200:
            a.append(str(s))
            b.append(v)
    plt.bar(a,b)
    plt.show()
    decision = max(zip(h.values(), h.keys()))[1]
    #assert(decision == (1,0,0,3,0 ,0,0,3,0))
    print("\tSuccess!")
    
    

def _TEST_expandedPowerSystem1Bus_1Scenario():
    ## Test the PowerSystem_1Bus 
    ### 1 Gas, 1 Wind, 4 scenarios and 4 levels
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=20, demand=3, 
            pdf={tuple([3]): 1.}
        )
    print(system)
    ### OPTIMAL choice x=1, w0=0, w1=w2=w3=1
    opt = Optimizer_Expanded(system, 'unary')
    print("qubits", opt.num_qubits)
    counts = opt.solveAnnealing(20, method='QUBO')
    #plot_histogram(counts)
    #plt.bar(counts.keys(), counts.values())
    a = []
    b = []
    for key,value in counts.items():
        if value > 2_00:
            a.append(str(opt.bstrToVars(key)))
            b.append(value)
    plt.bar(a,b)


def _TEST_expandedPowerSystem1Bus_penalty2_binary():
    ## Test the PowerSystem_1Bus 
    ### 1 Gas, 1 Wind, 4 scenarios and 4 levels
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[1], wind_costs=[0.01], decision_levels=4,
            undersatisfied_cost=10, demand=3,
            #pdf={(0,): 0.25, (1,): 0.25, (2,): 0.25, (3,): 0.25},
            pdf={(0,): 0.5, (3,): 0.5},
            normalization=None,#(np.pi,1)
        )
    opt = Optimizer_Expanded(system, 'binary')
    #print("qubits", opt.num_qubits)
    print(opt)
    counts = opt.solveAnnealing(15, method='QUBO', penalty=2)
    h = opt.getGasCounts(counts)
    h = opt.countsBstrToVars(counts)
    decision = max(zip(h.values(), h.keys()))[1]
    print("decision:", decision)
    print(system.cobylaSolve().x)
    a = []
    b = []
    for s,v in h.items():
        #if v > 200:
            a.append(str(s))
            b.append(v)
    plt.bar(a,b)
    plt.xticks(rotation=45)
    plt.show()

def main():
    #_TEST_expandedPowerSystem1Bus_1Scenario()
    #_TEST_expandedPowerSystem1Bus_penalty2_unary_hardmax()
    _TEST_expandedPowerSystem1Bus_penalty2_binary()

if __name__=="__main__":
    main()