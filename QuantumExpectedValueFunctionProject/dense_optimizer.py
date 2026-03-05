#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:07:10 2023

@author: crotello
"""

import optimizer_utils 

import matplotlib.pyplot as plt
from itertools import permutations
from sympy.utilities.iterables import multiset_permutations

from qiskit import QuantumCircuit, Aer
from qiskit.compiler import transpile
from qiskit.tools.visualization import plot_histogram
from qiskit.extensions import Initialize
import numpy as np

class Optimizer_Dense:
    def __init__(self, system, encoding, slack_register=False):
        ''' ctor. 
            Take the PowerSystem we will be solving, reserve variables and qubits
            
            Order the variables as gas, wind, slack, pdf
        '''
        assert(encoding == 'binary' or encoding == 'unary')
        self.system = system
        self.encoding = encoding
        self.num_scenarios = len(system.pdf.keys())
        if slack_register:
            self.num_decision_variables = system.num_gas_generators + system.num_wind_turbines + 1
        else:
            self.num_decision_variables = system.num_gas_generators + system.num_wind_turbines
        self.num_pdf_variables = system.num_wind_turbines
        
        # Get cost for each variable in this problem encoding
        if slack_register:
            self.variable_costs = system.gas_costs + system.wind_costs + [system.undersatisfied_cost]
        else:
            self.variable_costs = system.gas_costs + system.wind_costs #+ [system.undersatisfied_cost]
        # Normalize the costs
        #self.normalization = normalization
        #if normalization is not None:
        #    mvar = max(self.variable_costs)
        #    print(mvar)
        
        # Declare variables
        # Assign varids, expect order gas, wind, slack
        self.decision_varids = list(range(self.num_decision_variables))
        # Reserve gas,wind variables
        self.decision_variables = [optimizer_utils.VariableRegister(system.decision_levels-1, encoding) 
                                   for _ in range(system.num_gas_generators + system.num_wind_turbines)]
        # Slack variable
        if slack_register:
            self.decision_variables.append(optimizer_utils.VariableRegister(system.demand, encoding))
        # Gas varids
        self.gas_varids = self.decision_varids[:system.num_gas_generators]
        # Wind varids
        self.wind_varids = self.decision_varids[system.num_gas_generators : system.num_gas_generators + system.num_wind_turbines]
        
        # Declare pdf variables
        self.pdf_varids = list(range(self.num_decision_variables, self.num_decision_variables+system.num_wind_turbines))
        # Reserve pdf variables
        self.pdf_variables = {i: optimizer_utils.VariableRegister(system.decision_levels-1, encoding) for i in self.pdf_varids}

        
        # reserve qubits
        self.num_qubits = sum([var.width for var in self.decision_variables + list(self.pdf_variables.values())])
        self.varid_to_qubits = {}
        qubit = 0
        # decision qubits
        for i,reg in enumerate(self.decision_variables):
            self.varid_to_qubits[i] = list(range(qubit, qubit+reg.width))
            qubit += reg.width
        # pdf qubits
        for i in self.pdf_varids:
            reg = self.pdf_variables[i]
            self.varid_to_qubits[i] = list(range(qubit, qubit+reg.width))
            qubit += reg.width
            
        self.decision_qubits = [q for varid in self.decision_varids for q in self.varid_to_qubits[varid]]  
        self.pdf_qubits = [q for varid in self.pdf_varids for q in self.varid_to_qubits[varid]]

    
    def __str__(self):
        s = 'Optimizer_Dense\n'
        s += '\t#Decision Variables: ' + str(self.num_decision_variables) + '\n'
        s += '\t#PDF Variables: ' + str(self.num_pdf_variables) + '\n'
        s += '\t#Qubits: ' + str(self.num_qubits) + '\n'
        s += '\tDecision Variables: \n'
        for varid,var in enumerate(self.decision_variables):
            s += '\t\tVar({}): Q={}, c={} \n'.format(varid, self.varid_to_qubits[varid], self.variable_costs[varid])
        s += '\tPDF Variables: \n'
        for varid,var in self.pdf_variables.items():
            s += '\t\tVar({}): Q={} \n'.format(varid, self.varid_to_qubits[varid])
        s += '\tScenarios \n'
        for scenario_id, scenario in enumerate(self.system.scenarios):
            s += '\t\tScen({})={}, Pr({})={}\n'.format(scenario_id, scenario, scenario_id, self.system.pdf[scenario])
        #s += '\tNormalization = {}'.format(self.normalization)
        return s
        
    ####
    # Phase/Price/Cost Operators
    ####
    def priceOperator(self, amp, scenario_weight=1.):
        ''' The price of x in each register
        '''
        qc = QuantumCircuit(len(self.decision_qubits))
        for varid in self.decision_varids:
            reg = self.decision_variables[varid]
            cost = self.variable_costs[varid]
            amp_varid = amp
            #if varid in self.wind_varids:
            amp_varid *= scenario_weight
            qc.append(reg.numberOperator(cost*amp_varid), self.varid_to_qubits[varid])
        return qc
    
    def scenarioOperator(self, amp):
        ''' The price of wind w > weather xi
        '''
        qc = QuantumCircuit(self.num_qubits)
        for i,varid in enumerate(self.wind_varids):
            pdf_varid = self.pdf_varids[i]
            wind_reg = self.decision_variables[varid]
            pdf_reg = self.pdf_variables[pdf_varid]
            qc.append(wind_reg.lessThanOperator(pdf_reg, self.system.undersatisfied_cost*amp), 
                      self.varid_to_qubits[varid]+self.varid_to_qubits[pdf_varid])
        return qc
        
    def penaltyOperator(self, amp, penalty):
        ''' The price of (x+w+y-d)^2
        '''
        qc = QuantumCircuit(len(self.decision_qubits))
        for varid_j in self.decision_varids:
            reg_j = self.decision_variables[varid_j]
            qc.append(reg_j.numberOperator((-2 * penalty * self.system.demand)* amp), self.varid_to_qubits[varid_j])
            qc.append(reg_j.squaredOperator((penalty) * amp), self.varid_to_qubits[varid_j])
            for varid_k in self.decision_varids[varid_j+1:]:
                reg_k = self.decision_variables[varid_k]
                qc.append(reg_j.productOperator(reg_k, (2 * penalty)* amp), 
                          self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
        return qc
        
    def solveAnnealing(self, time, method='QUBO', num_meas=10_000, penalty=1):
        ''' solveAnnealing
            *DEPRECATED* need to update this
            Solve the optimization problem with an annealing routine, specify if we use 
            a Dicke state and constraint preserving mixer or QUBO with a penalty Hamiltonian
        '''
        qc = QuantumCircuit(self.num_qubits + 1, self.num_qubits)
        if method == 'QUBO':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        else:
            print("Unimplemented annealing solution method: {}".format(method))
            return 0
        qc.append(self.initializePDF(), self.pdf_qubits)
        #demand = self.system.demand
        for t in range(1,time+1):
            f = t/time
            ####
            # cost operator
            ####
            # cost operator
            # for varid in self.decision_varids:
            #     reg = self.decision_variables[varid]
            #     cost = self.variable_costs[varid]
            #     qc.append(reg.numberOperator(cost*f), self.varid_to_qubits[varid])
            qc.append(self.priceOperator(f), self.decision_qubits)
            
            # scenario operator 
            # for i,varid in enumerate(self.wind_varids):
            #     pdf_varid = self.pdf_varids[i]
            #     wind_reg = self.decision_variables[varid]
            #     pdf_reg = self.pdf_variables[pdf_varid]
            #     qc.append(wind_reg.lessThanOperator(pdf_reg, self.system.undersatisfied_cost*f), 
            #     #qc.append(wind_reg.lessThanOperator(pdf_reg, self.system.undersatisfied_cost), 
            #               self.varid_to_qubits[varid]+self.varid_to_qubits[pdf_varid])
            #               #self.varid_to_qubits[pdf_varid]+self.varid_to_qubits[varid])
            qc.append(self.scenarioOperator(f), range(self.num_qubits))
            
            # cost operator - penalty term
            # for varid_j in self.decision_varids:
            #     reg_j = self.decision_variables[varid_j]
            #     qc.append(reg_j.numberOperator((-2 * penalty * self.system.demand)* f), self.varid_to_qubits[varid_j])
            #     qc.append(reg_j.squaredOperator((penalty) * f), self.varid_to_qubits[varid_j])
            #     for varid_k in self.decision_varids[varid_j+1:]:
            #         reg_k = self.decision_variables[varid_k]
            #         qc.append(reg_j.productOperator(reg_k, (2 * penalty)* f), 
            #                   self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
            qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            qc.barrier()
            
            ####
            # mixing operator
            ####
            if method == 'QUBO':
                for i in self.decision_qubits:#range(self.num_qubits):
                    qc.rx(1-f, i)
            else:
                print("Unimplemented mixing operator for method: {}".format(method))
        qc.measure(list(range(self.num_qubits)), list(range(self.num_qubits)))
        
        # Transpile for simulator
        simulator = Aer.get_backend('statevector_simulator')
        print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        print('getting results...')
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts


    def solveAnnealingAlternating(self, total_time, time_steps, penalty=1, num_meas=1_000, init_cond='XGS', mixer='X', phase='PEN'):
        ''' solveAnnealingAlternating
            The same dense->serial seperation of labor, but we start a different annealing schedule at layer 2
        '''
        #if self.system.normalization 
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial condition
        if init_cond == 'XGS':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        elif init_cond == 'DICKE':
            qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0

        qc.append(self.initializePDF(), self.pdf_qubits)
        for j in range(time_steps):
            dt_1 = total_time/time_steps
            f = (dt_1*j + 1)/(total_time + 1)
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator number 2 - deterministic
                qc.append(self.initializePDF(inverse=True), self.pdf_qubits)
                for sample, pr in self.system.pdf.items():
                    qc_set_var = self.setVariables(self.pdf_variables, sample)
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    qc.append(self.scenarioOperator(.5*pr*f), range(self.num_qubits))
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                qc.append(self.initializePDF(), self.pdf_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(.5*f), range(self.num_qubits))
            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qubits_j = self.varid_to_qubits[varid_j]
                        qubits_k = self.varid_to_qubits[varid_k]
                        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        #qc.append(self.initializePDF(inverse=True), self.pdf_qubits)
        #for q in self.pdf_qubits:
        #    qc.reset(q)

        
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator_statevector')#aer_simulator_matrix_product_state')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts

    def solveAnnealingDenseStochHam2Layers(self, t1_time, t1_steps, t2_time, t2_steps, penalty=1, num_meas=1_000, init_cond='XGS', mixer='X', phase='PEN'):
        ''' solveAnnealingStochHam2Layers
            The same dense->serial seperation of labor, but we start a different annealing schedule at layer 2
        '''
        #if self.system.normalization 
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial condition
        if init_cond == 'XGS':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        elif init_cond == 'DICKE':
            qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0

        qc.append(self.initializePDF(), self.pdf_qubits)
        for j in range(t1_steps):
            dt_1 = t1_time/t1_steps
            f = (dt_1*j + 1)/(t1_time + 1)
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(f), range(self.num_qubits))

            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qubits_j = self.varid_to_qubits[varid_j]
                        qubits_k = self.varid_to_qubits[varid_k]
                        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        qc.append(self.initializePDF(inverse=True), self.pdf_qubits)
        #for q in self.pdf_qubits:
        #    qc.reset(q)

        # Phase 2
        for j in range(t2_steps):
            dt_2 = (t2_time)/(t2_steps) 
            f = (dt_2 + 1)/(t2_time + 1) 
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                for sample, pr in self.system.pdf.items():
                    qc_set_var = self.setVariables(self.pdf_variables, sample)
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    qc.append(self.scenarioOperator(pr*f), range(self.num_qubits))
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    #for varid_j,var_j in enumerate(self.decision_variables):
                    #    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                    #        qubits_j = self.varid_to_qubits[varid_j]
                    #        qubits_k = self.varid_to_qubits[varid_k]
                    #        qc.append(var_j.swapOperator(var_k, pr/np.pi*(1-f)), qubits_j + qubits_k)

            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qubits_j = self.varid_to_qubits[varid_j]
                        qubits_k = self.varid_to_qubits[varid_k]
                        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator_statevector')#aer_simulator_matrix_product_state')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts
        

    def solveAnnealingDenseStochHam(self, total_time, t2, t1_steps, t2_steps, 
                                    penalty=1, num_meas=1_000, reset_pdf=False,
                                    init_cond='XGS', mixer='X', phase='PEN'):
        ''' solveAnnealingStochHam
        '''
        #if self.system.normalization 
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial condition
        if init_cond == 'XGS':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        elif init_cond == 'DICKE':
            qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0

        qc.append(self.initializePDF(), self.pdf_qubits)
        for j in range(t1_steps):
            dt_1 = t2/t1_steps
            f = (dt_1*j + 1)/(total_time + 1)
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(f), range(self.num_qubits))

            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qubits_j = self.varid_to_qubits[varid_j]
                        qubits_k = self.varid_to_qubits[varid_k]
                        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        if reset_pdf:
            for q in self.pdf_qubits:
                qc.reset(q)
        else:
            qc.append(self.initializePDF(inverse=True), self.pdf_qubits)


        # Phase 2
        for j in range(t2_steps):
            dt_2 = (total_time-t2)/(t2_steps) # NOTE not sure about this form exactly
            f = (dt_2*(j+1) + (t2 - dt_1) + 1)/(total_time + 1) 
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                for sample, pr in self.system.pdf.items():
                    qc_set_var = self.setVariables(self.pdf_variables, sample)
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    qc.append(self.scenarioOperator(pr*f), range(self.num_qubits))
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    #for varid_j,var_j in enumerate(self.decision_variables):
                    #    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                    #        qubits_j = self.varid_to_qubits[varid_j]
                    #        qubits_k = self.varid_to_qubits[varid_k]
                    #        qc.append(var_j.swapOperator(var_k, pr/np.pi*(1-f)), qubits_j + qubits_k)

            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qubits_j = self.varid_to_qubits[varid_j]
                        qubits_k = self.varid_to_qubits[varid_k]
                        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator_statevector')#aer_simulator_matrix_product_state')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts

    def solveAnnealingStochHam(self, total_time, t_steps, penalty=1, num_meas=1_000, init_cond='XGS', mixer='X', phase='PEN'):
        ''' solveAnnealingStochHam
        '''
        #if self.system.normalization 
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Initial condition
        if init_cond == 'XGS':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        elif init_cond == 'DICKE':
            qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0

        for j in range(t_steps):
            dt = (total_time+1)/t_steps
            f = (dt*j + 1)/(total_time + 1)

            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                # TODO stochastic
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                for sample, pr in self.system.pdf.items():
                    qc_set_var = self.setVariables(self.pdf_variables, sample)
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)
                    qc.append(self.scenarioOperator(pr*f), range(self.num_qubits))
                    qc.append(qc_set_var, range(self.num_qubits))#self.pdf_qubits)

                    for varid_j,var_j in enumerate(self.decision_variables):
                        for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                            qubits_j = self.varid_to_qubits[varid_j]
                            qubits_k = self.varid_to_qubits[varid_k]
                            qc.append(var_j.swapOperator(var_k, pr/np.pi*(1-f)), qubits_j + qubits_k)

            else:
                print("Unimplemented phase operator: {}".format(phase))

            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                0
                #for varid_j,var_j in enumerate(self.decision_variables):
                #    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                #        qubits_j = self.varid_to_qubits[varid_j]
                #        qubits_k = self.varid_to_qubits[varid_k]
                #        qc.append(var_j.swapOperator(var_k, 1/np.pi*(1-f)), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator_statevector')#aer_simulator_matrix_product_state')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts

    def solveECAnnealing(self, total_time, gamma, penalty=1, num_meas=1_000, init_cond='XGS', mixer='X', phase='PEN'):
        ''' solveECAnnealing

        '''
        #if self.system.normalization 
        copies = 2
        qc = QuantumCircuit(copies*self.num_qubits, self.num_qubits)

        # Initial condition
        if init_cond == 'XGS':
            # TODO: fix for copies
            for c in range(copies):
                for qubit in self.decision_qubits:#range(self.num_qubits):
                    qc.h(c*self.num_qubits+qubit)
        elif init_cond == 'DICKE':
            for c in range(copies):
                qc.append(self.dickeState(), [c*self.num_qubits+q for q in self.decision_qubits])
                #qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0
        
        for c in range(copies):
            qc.append(self.initializePDF(), [c*self.num_qubits+q for q in self.pdf_qubits])

        for t in range(total_time):
            f = (t+1)**2/(total_time+1)**2

            ####
            # phase operator
            ####
            if phase == 'PEN':
                # TODO: fix for copies
                for c in range(copies):
                    # price
                    qc.append(self.priceOperator(f), [c*self.num_qubits+q for q in self.decision_qubits])
                    # scenario operator 
                    qc.append(self.scenarioOperator(f), [c*self.num_qubits+q for q in range(self.num_qubits)])
                    # penalty term
                    qc.append(self.penaltyOperator(f, penalty), [c*self.num_qubits+q for q in self.decision_qubits])
            elif phase == 'COST':
                for c in range(copies):
                    # price
                    qc.append(self.priceOperator(f), [c*self.num_qubits+q for q in self.decision_qubits])
                    # scenario operator 
                    qc.append(self.scenarioOperator(f), [c*self.num_qubits+q for q in range(self.num_qubits)])
            else:
                print("Unimplemented phase operator: {}".format(phase))

            ###
            # Non-anticipativity constraint
            ###
            for c_j in range(copies):
                for c_k in range(copies):
                    if c_j != c_k:
                    #for q in self.gas
                        for q in [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]:
                            qc.rzz(gamma*f, c_k*self.num_qubits+q, c_j*self.num_qubits+q)
            qc.barrier()

            ####
            # mixing operator
            ####
            if mixer == 'X':
                # TODO: fix for copies
                for c in range(copies):
                    for i in self.decision_qubits:
                        qc.rx(1-f, c*self.num_qubits+i)
            elif mixer == 'SWAP':
                for c in range(copies):
                    for varid_j,var_j in enumerate(self.decision_variables):
                        for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                            qubits_j = [c*self.num_qubits+q for q in self.varid_to_qubits[varid_j]]
                            qubits_k = [c*self.num_qubits+q for q in self.varid_to_qubits[varid_k]]
                            qc.append(var_j.swapOperator(var_k, 1-f), qubits_j + qubits_k)
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator_statevector')#aer_simulator_matrix_product_state')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts

    def solveThreePhaseAnnealing(self, total_time, time_1_steps, time_2, time_3, time_3_steps, num_permutations, samples_repeats=1, num_meas=10_000, penalty=None,
                               init_cond='XGS', mixer="X", phase='PEN'):
        ''' solveThreePhaseAnnealing
            Use an annealing routine; in the first phase, optimize for a given sample. In the second phase, exchange this for 
            other samples. In the third phase, prepare the PDF and sample the gas.
            Repeat for a few permutations of the order of the samples
        '''
        if self.system.normalization is not None and penalty is not None:
            #penalty *= self.system.normalization[1]/self.system.normalization[0]
            penalty = self.system.normalize(penalty)
        # warning
        if phase!='PEN' and penalty is not None:
            print("WARNING: specified a penalty but the penalty cost Hamiltonian is not used")
        if penalty is None and phase == 'PEN':
            print("ERROR: if PEN is specified (penalty Hamiltonian) we need a penalty specified")
            exit(1)


        # Take a few different permutations of the samples
        # TODO: get a better random sample of permutations
        #all_permutations = list(multiset_permutations(self.system.sample_list))#[:num_permutations]
        #if num_permutations is not None:
        #    permutations_list = all_permutations[::int(len(all_permutations)/num_permutations)]
        #else:
        #    permutations_list = all_permutations
        permutations_list = [self.system.sample_list[::-1],]
        #permutations_list = [list(self.system.pdf.keys()),] # pdf impl
        all_counts = {tuple(permutation): None for permutation in permutations_list}
        for p_i, permutation in enumerate(permutations_list):
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)

            # Initial condition
            if init_cond == 'XGS':
                for qubit in self.decision_qubits:#range(self.num_qubits):
                    qc.h(qubit)
            elif init_cond == 'DICKE':
                qc.append(self.dickeState(), self.decision_qubits)
            else:
                print("Unimplemented initial state: {}".format(init_cond))
                return 0
            #qc.append(self.initializePDF(), self.pdf_qubits)
            sample_idx = 0
            qc_set_var = self.setVariables(self.pdf_variables, permutation[sample_idx])
            qc.append(qc_set_var, list(range(self.num_qubits)))
            display = False

            ###
            # Phase 1
            ###
            dt_1 = time_2/time_1_steps
            for j in range(time_1_steps):
                f = (dt_1*j + 1)/(total_time + 1)
                a = 1
                if display:
                    print("phase 1", f, j, time_2)
                ####
                # phase operator
                ####
                if phase == 'PEN':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    # scenario operator 
                    qc.append(self.scenarioOperator(f), range(self.num_qubits))
                    # penalty term
                    qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
                elif phase == 'COST':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    #qc.append(self.priceOperator(f,self.system.sample_hist[permutation[sample_idx]]), self.decision_qubits)
                    # scenario operator 
                    qc.append(self.scenarioOperator(a*f), range(self.num_qubits))
                    #qc.append(self.scenarioOperator(
                    #          #self.system.sample_hist[permutation[sample_idx]]/(len(self.system.sample_list)*1)*f), 
                    #          self.system.sample_hist[permutation[sample_idx]]*f), 
                    #          range(self.num_qubits))
                else:
                    print("Unimplemented phase operator: {}".format(phase))
                qc.barrier()

                ####
                # mixing operator
                ####
                if mixer == 'X':
                    for i in self.decision_qubits:
                        qc.rx(1-f, i)
                elif mixer == 'SWAP':
                    for varid_j,var_j in enumerate(self.decision_variables):
                        for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                            qc.append(var_j.swapOperator(var_k, 1-f), self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
                else:
                    print("Unimplemented mixing operator: {}".format(mixer))
                qc.barrier()

            ###
            # Phase 2
            ###
            # undo previous sample
            qc_set_var = self.setVariables(self.pdf_variables, permutation[sample_idx])
            qc.append(qc_set_var, list(range(self.num_qubits)))
            # this phase needs M-1 steps
            dt_2 = (time_3 - time_2)/(len(permutation)*samples_repeats - 1)
            #for j in range(len(permutation)-1):
            for j in range(1, len(permutation)*samples_repeats):
                sample_idx += 1
                sample_idx %= len(permutation)
                f = (dt_2*(j-1) + time_2 + 1)/(total_time + 1)
                a = 1/(len(permutation))*(1+1*(j+1)/len(permutation)*samples_repeats)
                if display:
                    print("phase 2", f, j, time_2, time_3, 'sample weight', a)
                qc_set_var = self.setVariables(self.pdf_variables, permutation[sample_idx])
                qc.append(qc_set_var, list(range(self.num_qubits)))
                ####
                # phase operator
                ####
                if phase == 'PEN':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    # scenario operator 
                    #qc.append(self.scenarioOperator((1+1*(j+1)/len(permutation))*f), range(self.num_qubits))
                    qc.append(self.scenarioOperator(f), range(self.num_qubits))
                    # penalty term
                    qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
                elif phase == 'COST':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    #qc.append(self.priceOperator(f, 
                    #                             self.system.sample_hist[permutation[sample_idx]]*(1+1*(j+1)/len(permutation))), 
                    #          self.decision_qubits)
                    # scenario operator 
                    qc.append(self.scenarioOperator(a*f), range(self.num_qubits))
                    #qc.append(self.scenarioOperator(
                    #          #self.system.sample_hist[permutation[sample_idx]]/(len(self.system.sample_list)*1)*(1+1*(j+1)/len(permutation))*f), 
                    #          self.system.sample_hist[permutation[sample_idx]]*(1+1*(j+1)/len(permutation))*f), 
                    #          range(self.num_qubits))
                else:
                    print("Unimplemented phase operator: {}".format(phase))
                qc.barrier()

                ####
                # mixing operator
                ####
                if mixer == 'X':
                    for i in self.decision_qubits:
                        qc.rx(1-f, i)
                elif mixer == 'SWAP':
                    for varid_j,var_j in enumerate(self.decision_variables):
                        for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                            qc.append(var_j.swapOperator(var_k, 1-f), self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
                else:
                    print("Unimplemented mixing operator: {}".format(mixer))
                # Undo the sample
                qc_set_var = self.setVariables(self.pdf_variables, permutation[sample_idx])
                qc.append(qc_set_var, list(range(self.num_qubits)))
                qc.barrier()

            ###
            # Phase 3
            ###
            # Optimize with the entire PDF
            qc.append(self.initializePDF(), self.pdf_qubits)
            for j in range(time_3_steps):
                dt_3 = (total_time-time_3)/time_3_steps
                f = (dt_3*j + time_3 + 1)/(total_time + 1)
                if display:
                    print('phase 3', f, j, time_3, total_time)
                ####
                # phase operator
                ####
                if phase == 'PEN':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    # scenario operator 
                    qc.append(self.scenarioOperator(f), range(self.num_qubits))
                    # penalty term
                    qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
                elif phase == 'COST':
                    # price
                    qc.append(self.priceOperator(f), self.decision_qubits)
                    # scenario operator 
                    qc.append(self.scenarioOperator(2*f), range(self.num_qubits))
                else:
                    print("Unimplemented phase operator: {}".format(phase))
                qc.barrier()

                ####
                # mixing operator
                ####
                if mixer == 'X':
                    for i in self.decision_qubits:
                        qc.rx(1-f, i)
                elif mixer == 'SWAP':
                    for varid_j,var_j in enumerate(self.decision_variables):
                        for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                            qc.append(var_j.swapOperator(var_k, 1-f), self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
                else:
                    print("Unimplemented mixing operator: {}".format(mixer))
                # Undo the sample
                qc_set_var = self.setVariables(self.pdf_variables, permutation[sample_idx])
                qc.append(qc_set_var, list(range(self.num_qubits)))
                qc.barrier()
            gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
            qc.measure(gas_qubits, gas_qubits)
            # Transpile for simulator
            simulator = Aer.get_backend('aer_simulator')
            qc = transpile(qc, simulator)
    
            # Run and get statevector
            result = simulator.run(qc, shots=num_meas).result()
            counts = result.get_counts(qc)
            counts = {key:value/num_meas for key,value in counts.items()}
            all_counts[tuple(permutation)] = counts
        return all_counts

    
    def solveTwoPhaseAnnealing(self, total_time, time_2, meas_layers, num_meas=10_000, penalty=None,
                               init_cond='XGS', mixer="X", phase='PEN'):
        ''' solveTwoPhaseAnnealing
            Use an annealing routine, but fill the second phase with measurements and slower time steps
            Provide a few keyword arguments
        '''
        # check normalization on penalty ham
        if self.system.normalization is not None and penalty is not None:
            penalty *= self.system.normalization[1]/self.system.normalization[0]
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # warning
        if phase!='PEN' and penalty is not None:
            print("WARNING: specified a penalty but the penalty cost Hamiltonian is not used")
        if penalty is None and phase == 'PEN':
            print("ERROR: if PEN is specified (penalty Hamiltonian) we need a penalty specified")
            exit(1)

        if init_cond == 'XGS':
            for qubit in self.decision_qubits:#range(self.num_qubits):
                qc.h(qubit)
        elif init_cond == 'DICKE':
            qc.append(self.dickeState(), self.decision_qubits)
        else:
            print("Unimplemented initial state: {}".format(init_cond))
            return 0
        qc.append(self.initializePDF(), self.pdf_qubits)
        
        phase_1_duration = time_2
        for t in range(phase_1_duration):
            f = (t+1)/(total_time+1)
            #print("phase 1", t, f, phase_1_duration)
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
            else:
                print("Unimplemented phase operator: {}".format(phase))
            qc.barrier()
            
            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qc.append(var_j.swapOperator(var_k, 1-f), self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
                
        phase_2_duration = total_time - time_2
        for mlayer in range(meas_layers):
            dt = phase_2_duration/meas_layers
            f = (time_2 + dt*mlayer + 1)/(total_time+1)
            #print('phase 2', mlayer, dt, f, phase_2_duration)
            
            #if phase =='COST':
            #    # price
            #    qc.append(self.priceOperator(f/2), self.decision_qubits)
            #    # scenario operator 
            #    qc.append(self.scenarioOperator(f/2), range(self.num_qubits))

            # measure
            #if mlayer % 2 == 1:
            #qc.measure([q for varid in self.wind_varids for q in self.varid_to_qubits[varid]], [q for varid in self.wind_varids for q in self.varid_to_qubits[varid]])
            #    qc.measure([q for varid in self.gas_varids for q in self.varid_to_qubits[varid]], [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]])
            #else:
            #    qc.measure(self.pdf_qubits, self.pdf_qubits)
            #for q in [q for varid in self.wind_varids for q in self.varid_to_qubits[varid]]:
            #    qc.h(q)
            # reset
            #qc.reset(self.pdf_qubits)
            qc.append(self.initializePDF(), self.pdf_qubits)
            
            ####
            # phase operator
            ####
            if phase == 'PEN':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(f), range(self.num_qubits))
                # penalty term
                qc.append(self.penaltyOperator(f, penalty), self.decision_qubits)
            elif phase == 'COST':
                # price
                qc.append(self.priceOperator(f), self.decision_qubits)
                # scenario operator 
                qc.append(self.scenarioOperator(2.*f), range(self.num_qubits))
            else:
                print("Unimplemented phase operator: {}".format(phase))
            qc.barrier()
            
            ####
            # mixing operator
            ####
            if mixer == 'X':
                for i in self.decision_qubits:
                    qc.rx(1-f, i)
            elif mixer == 'SWAP':
                for varid_j,var_j in enumerate(self.decision_variables):
                    for varid_k,var_k in enumerate(self.decision_variables[:varid_j]):
                        qc.append(var_j.swapOperator(var_k, 1-f), self.varid_to_qubits[varid_j] + self.varid_to_qubits[varid_k])
            else:
                print("Unimplemented mixing operator: {}".format(mixer))
            qc.barrier()
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        qc.measure(gas_qubits, gas_qubits)
        #print(qc)
        # Transpile for simulator
        simulator = Aer.get_backend('aer_simulator')
        #print('transpiling...')
        qc = transpile(qc, simulator)
    
        # Run and get statevector
        #print('getting results...')
        result = simulator.run(qc, shots=num_meas).result()
        counts = result.get_counts(qc)
        counts = {key:value/num_meas for key,value in counts.items()}
        return counts

    def setVariables(self, vars, vals):
        ''' setVariables
            Given a list of variables and list of values, set the variables to those values
        '''
        assert(len(vars) == len(vals))
        qc = QuantumCircuit(self.num_qubits)
        for i,var in enumerate(vars):
            qc.append(self.pdf_variables[var].setValue(vals[i]), self.varid_to_qubits[var])
        return qc

    
    def initializePDF(self, inverse=False):
        ''' initializePDF
            Reset the pdf registers and initialize them in the PDF
        '''
        qc = QuantumCircuit(len(self.pdf_qubits))
        for i in range(len(self.pdf_qubits)):
            qc.reset(i)
        if self.encoding == 'unary':
            vector_scenario = np.zeros(shape=2**len(self.pdf_qubits), dtype=np.float128)
            # NOTE degeneracy will make this weird; for now, we just use one of the states
            for scenario, amp in self.system.pdf.items():
                # make a bitstring which represents this scenario
                global_bstr = ''
                for idx,value in enumerate(scenario):
                    bstr_value = '1'*value + '0'*(self.system.decision_levels-value-1)#(self.varid_to_levels[self.pdf_to_varid[idx]]-value-1) # TODO: account for degeneracy here
                    global_bstr += bstr_value
                global_bstr = ''.join(list(global_bstr)[::-1])
                # turn into an index and store in vector scenario
                vector_scenario[int(global_bstr, 2)] = np.sqrt(amp)
            if not inverse:
                qc.initialize(vector_scenario, range(len(self.pdf_qubits)))
            else:
                qc.append(Initialize(vector_scenario).gates_to_uncompute(), range(len(self.pdf_qubits)))
        elif self.encoding == 'binary':
            vector_scenario = np.zeros(shape=2**len(self.pdf_qubits))
            # TODO: real circuit here
            for scenario, amp in self.system.pdf.items():
                global_bstr = ''
                # NOTE: in future, double check we append from the right direction
                for idx,value in enumerate(scenario):
                    var_bstr = optimizer_utils.int_to_bstr_N(value, self.pdf_variables[idx+self.num_decision_variables].width)#len(self.varid_to_qubits[self.pdf_to_varid[idx]]))
                    global_bstr += var_bstr
                vector_scenario[int(global_bstr[::-1], 2)] = np.sqrt(amp)
            if not inverse:
                qc.initialize(vector_scenario, range(len(self.pdf_qubits)))
            else:
                qc.append(Initialize(vector_scenario).gates_to_uncompute(), range(len(self.pdf_qubits)))
        else:
            print("ERROR: Not implemented initialize PDF for encoding,", self.encoding)
        return qc

    def dickeState(self):
        ''' dickeState
            Create a uniform superposition of states which satisfy N|x> = d|x> 
        '''
        qc = QuantumCircuit(len(self.decision_qubits))
        if self.encoding == 'unary':
            # TODO: implement dicke state algo
            vec = np.zeros(2**len(self.decision_qubits))
            for val in range(2**len(self.decision_qubits)):
                bstr = optimizer_utils.int_to_bstr_N(val, len(self.decision_qubits))
                if sum([int(val) for val in bstr]) == self.system.demand:
                    vec[val] = 1
            vec /= np.sqrt(sum(vec))
            qc.initialize(vec, self.decision_qubits)
        else:
            print("ERROR: Not implemented Dicke state for encoding,", self.encoding)
        return qc
    

    ''' data processing \/
    '''
    def calcCostPerScenario(self, counts):
        ''' calcCostPerScenario
            Given a set of measurements from the dense encoding, without the mid-circuit measurements,
            calculate the gas/wind decisions for each weather scenario to benchmark how close the optimizer is 
            to getting the optimal gas/wind decision for each scenario
            This does NOT properly benchmark the two-phase optimization
            Either use post-processed cost function OR use cost function from cost Hamiltonian
        '''
        scenario_expval = {} # map each cost to the expected value
        counts_vars = self.countsBstrToVars(counts)
        # for each situation in the processed results
        for sitch, amp in counts_vars.items():
            # determine which weather scenario it belongs to
            scenario = sitch[-self.system.num_wind_turbines:]
            
            # compute the cost - if wind turbines exceed available wind, consider undermet cost
            #   use computed slack variable, not measured slack variable
            cost = 0
            gas = sitch[:self.system.num_gas_generators]
            cost += sum([self.system.gas_costs[i]*gas[i] for i in range(self.system.num_gas_generators)])
            wind = list(sitch[self.system.num_gas_generators:self.system.num_gas_generators+self.system.num_wind_turbines])
            # adjust wind outputs to match output scenario
            for i,wout in enumerate(wind):
                if wout > scenario[i]:
                    wind[i] = scenario[i]
            cost += sum([self.system.wind_costs[i]*wind[i] for i in range(self.system.num_wind_turbines)])
            generated_power = sum(wind)+sum(gas)
            if generated_power < self.system.demand:
                cost += self.system.undersatisfied_cost*(self.system.demand-generated_power)
            if scenario not in scenario_expval.keys():
                scenario_expval[scenario] = 0
            scenario_expval[scenario] += amp*cost#/self.system.pdf[scenario]
        return scenario_expval

    def calcGasExpectationValue(self, counts):
        ''' Given the counts from an algorithm run, calculate the expectation value <gas + E>
        '''
        #measured_decisions = self.countsBstrToGas(counts)
        measured_decisions = self.getGasCounts(counts)
        val_of_decisions,_ = self.system.getFirstStageCosts(measured_decisions.keys())
        exp_val = 0
        for decision,amp in measured_decisions.items():
            exp_val += amp * val_of_decisions[decision]
        return exp_val
    
    def getGasCounts(self, counts):
        gas_decisions = {}
        gas_qubits = [q for varid in self.gas_varids for q in self.varid_to_qubits[varid]]
        for key,value in counts.items():
            gas_bstr = key[::-1][:max(gas_qubits)+1]
            #print(key, gas_bstr)
            gas_decision = []
            i = 0
            for varid in self.gas_varids:
                variable = self.decision_variables[varid]
                var_substr = gas_bstr[i:i+variable.width]
                gas_decision.append(variable.getValue(var_substr[::-1]))
                i += variable.width
            #print(gas_decision, gas_bstr, key)
            gas_decision = tuple(gas_decision)
            if gas_decision not in gas_decisions.keys():
                gas_decisions[gas_decision] = 0
            gas_decisions[gas_decision] += value
        return gas_decisions
    
    def bstrToVars(self, bstr):
        bstr = bstr[::-1]
        i = 0
        values = []
        for varid,variable in enumerate(self.decision_variables + list(self.pdf_variables.values())):
            substr = bstr[i:i+variable.width]
            values.append(variable.getValue(substr))
            i+=variable.width
        return values
    
    def getVarsCounts(self, counts):
        hist = {}
        for key, values in counts.items():
            #hist[tuple(self.bstrToVars(key))] = values
            output = tuple(self.bstrToVars(key))
            if output not in hist.keys():
                hist[output] = 0
            hist[output] += values
        return hist

    def getDecision(self, gas_decisions):
        ''' getDecision
            Given the gas decisions results, choose the most common gas output
        '''
        sorted_decisions = sorted(gas_decisions.items(), key=lambda x: x[1])
        return sorted_decisions[-1]

    





def _TEST_twoVars_fourLevels_unary_classicalsamples():
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3],
            wind_costs=[0.5],
            decision_levels=4, undersatisfied_cost=10, demand=3,
            pdf={(0,):0.1, (1,):0.2, (2,):0.3, (3,):0.4},
                #{tuple([0]): 0.15, tuple([1]): 0.6, tuple([2]): 0.15, tuple([3]): 0.1,},
            normalization=(20,1)
    )
    opt = Optimizer_Dense(system, 'unary')
    num_perms = 5
    perm_counts = opt.solveThreePhaseAnnealing(total_time=10, time_1_steps=2, time_2=2, time_3=9, time_3_steps=2,
                                          num_permutations=num_perms, 
                                          num_meas=1_00, init_cond='DICKE', mixer='SWAP', phase='COST')
    combined_counts = {}
    for perm,counts in perm_counts.items():
        print(perm)
        gas_decisions = opt.getGasCounts(counts)
        gas_decision, freq = opt.getDecision(gas_decisions)
        system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
        plt.savefig("test_dense_optimizer_classical_figs/{}_{}.png".format(num_perms, perm))
        #plt.show()
        for key,val in counts.items():
            if key not in combined_counts.keys():
                combined_counts[key] = 0
            combined_counts[key] += val
        plt.cla()
    gas_decisions = opt.getGasCounts(combined_counts)
    gas_decision, freq = opt.getDecision(gas_decisions)
    system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
    plt.savefig("test_dense_optimizer_classical_figs/all.png")   
    plt.cla()

    
def main():
    _TEST_twoVars_fourLevels_unary_classicalsamples()
    
if __name__=='__main__':
    main()