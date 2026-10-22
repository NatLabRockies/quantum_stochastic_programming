
import numpy as np
import scipy
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit#, Aer
from qiskit_aer import Aer
from qiskit.circuit.library import Initialize
from qiskit.compiler import transpile
#from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library.standard_gates import SwapGate, RYGate, ZGate
from qiskit.circuit.library import QFT
from qiskit_algorithms import EstimationProblem
#from qiskit.primitives import Sampler
#from qiskit.algorithms import AmplitudeEstimation, FasterAmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation
from qiskit.converters import circuit_to_gate
from qiskit.quantum_info import Statevector



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



# NOTE: for not, we just assume one gas generator with integer output
class BinaryNestedOptimizer:
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
        #return bstrs
    
    def wind_scenario_cost(self, wind_output: list[int], scenario: tuple[int], wind_demand: int) -> float:
        ''' given a list of wind outputs and the corresponding scenario, return the cost of that scenario
        '''
        true_output = np.array(wind_output)*np.array(scenario)
        cost = sum(np.array(self.wind_costs) * true_output)
        #print(true_output, wind_output, scenario)
        if sum(true_output) < wind_demand:
            cost += (wind_demand - sum(true_output))*self.recourse_cost
        return cost

    def bstr_to_output(self, bstr):
        ''' convert a bitstring to the list of integers '''
        return [int(i) for i in bstr]

    def prep_gs(self, wind_demand):
        ''' create a statevector which is the minima for each scenario '''
        wvfn = np.zeros(2**(2*self.num_wind_vars))
        # use the min_scenario to get the best decision for each scenario
        for scenario,pr in self.pdf.items():
            sc_cost, sc_decision = self.min_scenario(wind_demand, scenario)
            #print(scenario, sc_decision)
            bstr = ''.join([str(b) for b in scenario[::-1] + sc_decision[::-1]])
            #print(bstr)
            wvfn[int(bstr[::-1],2)] = np.sqrt(pr)
        return wvfn


    ## Classical solvers
    def min_scenario(self, wind_demand, scenario):
        ''' get the minimization for a single wind scenario '''
        min_wind_cost = self.recourse_cost*self.demand+1
        min_wind_decision = []
        # if we have not made a list of wind outputs, make the list
        if self.all_wind_outputs_list is None:
            self.gen_all_wind_outputs()
        for wind_output in self.all_wind_outputs_list:
            if sum(wind_output) == wind_demand:
                cost = self.wind_scenario_cost(wind_output, scenario, wind_demand)
                # NOTE: if we want to track all solns, including degenerate ones, this is where
                if cost < min_wind_cost:
                    min_wind_decision = wind_output
                    min_wind_cost = cost
        return min_wind_cost,tuple(min_wind_decision)

    def brute_force_wind_demand_expectation_values(self):
        ''' solve for the energy surface of expected cost as a function of demand on the wind turbines
        '''
        demand_surface = []
        # check each integer demand which we could expect the wind turbines to fulfill
        for wind_demand in range(self.demand+1):
            # brute force the expectation value
            expectation_val = 0.
            for scenario,pr in self.pdf.items():
                # for each output that satisfies the demand            
                min_wind_cost,min_wind_decision = self.min_scenario(wind_demand, scenario)
                expectation_val += pr*min_wind_cost
            demand_surface.append(expectation_val)
        return demand_surface

    def brute_force_energy_surface(self):
        ''' go through each gas output, and get it's objective function
        '''
        obj_x = {}
        wind_demand_surface = self.brute_force_wind_demand_expectation_values()
        for wind_demand,expectation_value in enumerate(wind_demand_surface):
            gas_level = self.demand-wind_demand
            obj_x[gas_level] = gas_level*self.gas_costs[0] + expectation_value
        return obj_x


    ## quantum helpers
    def dicke_state_initialize(self, weight):
        ''' Create a dicke state with 'initialize' function '''
        qc = QuantumCircuit(self.num_wind_vars, name=r"$|D_n^{d-y}\rangle$")
        ## What happens if classical init. cond?
        #for i in range(weight):
        #    qc.x(i)

        #wvfn = np.zeros(2**self.num_wind_vars)
        #all_bstrs = [''.join([str(b) for b in output]) for output in self.all_wind_outputs_list]
        #for bstr in all_bstrs:
        #    if sum(self.bstr_to_output(bstr)) == weight:
        #        wvfn[int(bstr, 2)] = 1.
        #qc.append(Initialize(wvfn/np.sqrt(sum(wvfn))), list(range(self.num_wind_vars)))
        ##qc.initialize(wvfn/np.sqrt(sum(wvfn)))
        ##simulator = Aer.get_backend('statevector_simulator')
        #gates = ['u', 'p', 'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'cx']
        #qc = transpile(qc, basis_gates=gates)
        #qc = circuit_to_gate(qc)
        qc = self.dicke_state_circuit(weight)
        return qc
    
    def dicke_state_circuit(self, weight):
        ''' https://arxiv.org/pdf/1904.07358.pdf '''
        qc = QuantumCircuit(self.num_wind_vars, name='DickeState')
        if weight == 0:
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
        for i in range(weight):# range(self.num_wind_vars - weight - 1, self.num_wind_vars):
            qc.x(self.num_wind_vars-i-1)
        # n-k applications
        for i in range(self.num_wind_vars - weight):
            qc.append(SCS(self.num_wind_vars-i,weight), list(range(self.num_wind_vars-i-weight-1,self.num_wind_vars-i)))
        # last k
        for i in range(weight-1,0,-1):
            #print(i+1, i)
            qc.append(SCS(i+1, i), list(range(i+1)))
        #print(qc)
        return qc.to_gate()

    def pdf_initialize(self):
        ''' Create the pdf as a wavefunction '''
        qc = QuantumCircuit(self.num_wind_vars, name=r'$|\mathcal{P}\rangle$')
        if self.is_uniform:#np.isclose(list(self.pdf.values()), [1/2**self.num_wind_vars]*2**self.num_wind_vars).all():
            for i in range(self.num_wind_vars):
                qc.h(i)
        else:
            wvfn = np.zeros(2**self.num_wind_vars)
            for scenario, pr in self.pdf.items():
                bstr = ''.join([str(i) for i in scenario])
                wvfn[int(bstr[::-1], 2)] = np.sqrt(pr)
            qc.append(Initialize(wvfn), list(range(self.num_wind_vars)))
            #qc.initialize(wvfn)#, list(range(self.))) 
            #simulator = Aer.get_backend('statevector_simulator')         
            gates = ['u', 'p', 'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'cx']
            qc = transpile(qc, basis_gates=gates)
        #qc = circuit_to_gate(qc)
        return qc.to_gate()

    def cost_operator(self, amplitude, constraint_amplitude, norm):
        ''' our cost operator
            constraint_amplitude will be recourse for the constraint-preserving mixer, and a penalty weight otherwise
        '''
        qc = QuantumCircuit(self.num_wind_vars*2, name=r'$U_q({})$'.format(np.round(amplitude, 3)))
        for q_w,cost in enumerate(self.wind_costs):
            q_pdf = self.pdf_qubits[q_w]
            qc.cp(amplitude*cost/norm, q_pdf, q_w)

            qc.x(q_pdf)
            qc.cp(amplitude*constraint_amplitude/norm, q_pdf, q_w)
            qc.x(q_pdf)
        return qc

    def demand_constraint_preserving_mixer(self, amplitude):
        ''' mixing operator which preserves the demand constraint
        '''
        qc = QuantumCircuit(self.num_wind_vars, name=r'$U_d({})$'.format(np.round(amplitude, 3)))
        layers = 1
        for layer in range(layers):
            for qj in self.wind_qubits:
                for qk in self.wind_qubits[qj+1:]:
                    qc.append(SwapGate().power(amplitude/layers), [qj, qk])
        return qc

    def cost_scenario_operator(self, scenario, amplitude, constraint_amplitude, norm):
        qc = QuantumCircuit(self.num_wind_vars)
        for q_w, cost in enumerate(self.wind_costs):
            if scenario[q_w] == 0:
                qc.p(amplitude*constraint_amplitude/norm, q_w)
            else:
                qc.p(amplitude*cost/norm, q_w)
        return qc

    def adiabatic_evolution_circuit(self, wind_demand, time, time_steps, norm):
        ''' adiabatic evolution; currently only using the constraint-preserving mixer 
        '''
        qc = QuantumCircuit(self.num_wind_vars*2, name='Uopt')
        
        #qc.append(circuit_to_gate(self.dicke_state_initialize(wind_demand)), self.wind_qubits)
        #qc.append(circuit_to_gate(self.pdf_initialize()), self.pdf_qubits)
        qc.append(self.dicke_state_initialize(wind_demand), self.wind_qubits)
        qc.append(self.pdf_initialize(), self.pdf_qubits)
        for t in range(time_steps):
            #dt = (time+1)/time_steps
            #s = (dt*t + 1)/(time + 1)
            dt = time/time_steps
            s = np.float64((dt*t)/time)

            qc.append(circuit_to_gate(self.cost_operator(s, self.recourse_cost, norm)), self.wind_qubits + self.pdf_qubits)
            qc.append(circuit_to_gate(self.demand_constraint_preserving_mixer((1-s)/np.pi)), self.wind_qubits)
            #qc.append(self.cost_operator(s, self.recourse_cost, norm), self.wind_qubits + self.pdf_qubits)
            #qc.append(self.cost_operator(s, self.recourse_cost, norm), self.wind_qubits + self.pdf_qubits)
        return qc

    def adiabatic_evolution_scenario_circuit(self, wind_demand, time, time_steps, scenario, norm):
        ''' adiabatic evolution; constraint-preserving mixer and only a specific scenario
        '''
        qc = QuantumCircuit(self.num_wind_vars, name='Uopt({})'.format(scenario))

        qc.append(self.dicke_state_initialize(wind_demand), self.wind_qubits)
        qc.barrier()
        for t in range(time_steps):
            #dt = (time+1)/time_steps
            #s = (dt*t + 1)/(time + 1)
            dt = time/time_steps
            s = (dt*t)/time

            qc.append(self.cost_scenario_operator(scenario, s, self.recourse_cost, norm), self.wind_qubits)
            qc.append(self.demand_constraint_preserving_mixer(1/np.pi*(1-s)), self.wind_qubits)
        qc.barrier()
        return qc  
    

    #### Set aside a suite of QAE algorithms
    ####
    def implemented_qae(self, op, oracle, op_inv, oracle_inv, m, norm):
        ancilla = m+2*self.num_wind_vars
        qc = QuantumCircuit(2*self.num_wind_vars+1+m)
        for i in range(m):
            qc.h(i)
        qc.append(op, list(range(m, m+2*self.num_wind_vars)))
        qc.append(oracle, list(range(m, m+2*self.num_wind_vars+1)))
        for i in range(m):
            for j in range(2**i):
                qc.x(ancilla)
                qc.cz(i,ancilla)
                qc.x(ancilla)
                # invert
                # A^-1
                qc.append(oracle_inv.control(1), [i]+list(range(m, ancilla+1)))
                qc.append(op_inv.control(1), [i]+list(range(m, ancilla)))
                # |000...><000...|
                for q in range(m,ancilla):
                    qc.x(q)
                qc.x(ancilla)
                qc.append(ZGate().control(2*self.num_wind_vars+1), [i]+list(range(m, ancilla+1)))
                for q in range(m,ancilla):
                    qc.x(q)
                qc.x(ancilla)
                # A
                qc.append(op.control(1), [i]+list(range(m, ancilla)))
                qc.append(oracle.control(1), [i]+list(range(m, ancilla+1)))
        qc.append(QFT(m, inverse=True), list(range(m)))
        #print(qc)
        return qc

    def canonical_qae(self, Uopt, oracle, m, c, norm):
        ''' use qiskit tools to setup estimator NOTE still need to setup '''
        #oracle = self.single_oracle_sin_inconstraint(c, norm)
        #oracle = self.exact_oracle(ydemand,norm)
        #print(oracle)
        A = BernoulliA(circuit_to_gate(Uopt),circuit_to_gate(oracle),self.num_wind_vars)

        problem = EstimationProblem(
                    state_preparation=A,  # A operator
                    objective_qubits=[2*self.num_wind_vars],  # the "good" state Psi1 is identified as measuring |1> in qubit 0
                )
        return problem
        #sampler = Sampler(options={'shots':50})
        #mle = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=m,
        #                                           sampler=sampler)
        #mle_result = mle.estimate(problem)
        #return mle_result.estimation

        #fae = FasterAmplitudeEstimation(delta=0.01,
        #                                maxiter=m,
        #                                sampler=sampler)
        #fae_result = fae.estimate(problem)
        #return fae_result.estimation
        
        #ae = AmplitudeEstimation(
        #            num_eval_qubits=m,  
        #            sampler=sampler,
        #        )
        #ae_result = ae.estimate(problem)
        #return ae_result.estimation, ae_result.mle

    def exact_oracle(self, ydemand, norm, inverse=False):
        ''' compute the exact oracle - obviously untractable, but helpful for actually determining accuracy 
        '''
        # let's grab a control operator for each bitstring
        qc = QuantumCircuit(2*self.num_wind_vars+1)
        #Fdiag = [0.]*2**(self.num_wind_vars*2)
        ancilla = self.num_wind_vars*2
        #all_wind_bstr_outputs = {bstr: self.bstr_to_output(bstr) for bstr in self.all_wind_bitstrings()}
        all_wind_bstr_outputs = {''.join([str(b) for b in output]): output for output in self.all_wind_outputs_list}
        for output_bstr, output in all_wind_bstr_outputs.items():
            if sum([int(b) for b in output_bstr]) != ydemand:
                continue
            for scenario,pr in self.pdf.items():
                scenario_bstr = ''.join([str(s) for s in scenario])
                #bstr = (output_bstr + scenario_bstr)
                bstr = scenario_bstr + output_bstr
                cost = self.wind_scenario_cost(output[::-1], scenario[::-1], ydemand)
                #print(bstr, cost, self.wind_costs)
                #bstr = bstr[::-1]
                theta = 2*np.arcsin(np.sqrt(cost/norm))
                if inverse:
                    theta *= -1
                qc.append(RYGate(theta).control(num_ctrl_qubits=2*self.num_wind_vars, ctrl_state=bstr), self.wind_qubits+self.pdf_qubits+[ancilla])
        #print(qc)
        return qc


    def single_oracle_smallangle_inconstraint(self, c, norm):
    #    ''' compute the oracle onto an ancilla qubit - using smallangle approximations '''
    #    qc = QuantumCircuit(self.num_wind_vars*2 + 1, name='Fsmall')
    #    ancilla = self.num_wind_vars*2
    #    scale = np.pi*c/norm
    #    #scale = 2*c*np.pi/norm
    #    qc.ry(np.pi/2, ancilla) # R(pi/4)
    #    qc.ry(-c*2, ancilla) # R(-c)
    #    for q in self.wind_qubits:
    #        theta_cy = self.wind_costs[q]*scale
    #        theta_cr = self.recourse_cost*scale

    #        qc.append(RYGate(theta_cy).control(2), [q, self.pdf_qubits[q], ancilla])

    #        qc.x(self.pdf_qubits[q])
    #        qc.append(RYGate(theta_cr).control(2), [q, self.pdf_qubits[q], ancilla])
    #        qc.x(self.pdf_qubits[q])
    #    return qc
        0

    def single_oracle_sin_inconstraint(self, c, norm, inverse=False):
        ''' compute the oracle ont an ancilla qubit - assuming our states are in-constraint, using the sin approx
        '''
        qc = QuantumCircuit(self.num_wind_vars*2 + 1, name='F')
        ancilla = self.num_wind_vars*2
        scale = np.pi*c/norm
        if inverse:
            scale *= -1
        for q in self.wind_qubits:
            theta_cy = self.wind_costs[q] * scale 
            theta_cr = self.recourse_cost * scale

            qc.append(RYGate(theta_cy).control(2), [q, self.pdf_qubits[q], ancilla])

            qc.x(self.pdf_qubits[q])
            qc.append(RYGate(theta_cr).control(2), [q, self.pdf_qubits[q], ancilla])
            qc.x(self.pdf_qubits[q])
        return qc

    def grover_operator_sin_inconstraint(self, power, c, norm):
    #    ''' do the grover operator to a power for the arcsin+in constraint oracle '''
    #    qc = QuantumCircuit(self.num_wind_vars*2 + 2, name="F^{}".format(power))
    #    ancilla = self.num_wind_vars*2
    #    control = ancilla + 1
    #    scale = np.pi*c/norm
    #    for q in self.wind_qubits:
    #        theta_cy = -2*power * self.wind_costs[q] * scale
    #        theta_cr = -2*power * self.recourse_cost * scale

    #        qc.append(RYGate(theta_cy).control(3), [control, q, self.pdf_qubits[q], ancilla])

    #        qc.x(self.pdf_qubits[q])
    #        qc.append(RYGate(theta_cr).control(3), [control, q, self.pdf_qubits[q], ancilla])
    #        qc.x(self.pdf_qubits[q])
    #    return qc
        0

    def grover_reflections_sin_inconstraint(self, m, c, norm):
    #    ''' Create a circuit which does the grover reflections with the asin+constraint oracle
    #        Assume the system is in the state |psi_opt>(1-a|0> + a|1>
    #    '''
    #    qc = QuantumCircuit(self.num_wind_vars*2 + 1 + m, name="QAE")
    #    ancilla = self.num_wind_vars*2
    #    for m_j in range(m):
    #        qm =  ancilla+1+m_j
    #        qc.h(qm)
    #        qc.append(self.grover_operator_sin_inconstraint(2**m_j, c, norm), self.wind_qubits+self.pdf_qubits+[ancilla]+[qm])
    #    return qc
        0

       

    def single_oracle_asin_inconstraint(self, norm, inverse=False):
        ''' compute the oracle onto an ancilla qubit - assuming our states are all in-constraint, using the arcsin approximate
        '''
        qc = QuantumCircuit(self.num_wind_vars*2 + 1, name='F')
        ancilla = self.num_wind_vars*2
        for q in self.wind_qubits:
            cq = self.wind_costs[q]/norm
            cr = self.recourse_cost/norm
            #theta_cy = cq/norm*np.pi#2*np.arcsin(np.sqrt(cq))
            #theta_cr = cr/norm*np.pi#2*np.arcsin(np.sqrt(cr))
            theta_cy = 2*np.arcsin(np.sqrt(cq))
            theta_cr = 2*np.arcsin(np.sqrt(cr))
            if inverse:
                theta_cy *= -1
                theta_cr *= -1

            qc.append(RYGate(theta_cy).control(2), [q, self.pdf_qubits[q], ancilla])
            #qc.cry(theta_cy, q, ancilla)

            qc.x(self.pdf_qubits[q])
            qc.append(RYGate(theta_cr).control(2), [q, self.pdf_qubits[q], ancilla])
            qc.x(self.pdf_qubits[q])
        return qc
        0

    def grover_operator_asin_inconstraint(self, power, norm):
    #    ''' do the grover operator to a power for the arcsin+in constraint oracle '''
    #    qc = QuantumCircuit(self.num_wind_vars*2 + 2, name="F^{}".format(power))
    #    ancilla = self.num_wind_vars*2
    #    control = ancilla + 1
    #    for q in self.wind_qubits:
    #        cq = self.wind_costs[q]/norm
    #        cr = self.recourse_cost/norm
    #        theta_cy = -2*power * 2*np.arcsouin(np.sqrt(cq))
    #        theta_cr = -2*power * 2*np.arcsin(np.sqrt(cr))

    #        qc.append(RYGate(theta_cy).control(3), [control, q, self.pdf_qubits[q], ancilla])

    #        qc.x(self.pdf_qubits[q])
    #        qc.append(RYGate(theta_cr).control(3), [control, q, self.pdf_qubits[q], ancilla])
    #        qc.x(self.pdf_qubits[q])
    #    return qc
        0

    def grover_reflections_asin_inconstraint(self, m, norm):
    #    ''' Create a circuit which does the grover reflections with the asin+constraint oracle
    #        Assume the system is in the state |psi_opt>(1-a|0> + a|1>
    #    '''
    #    qc = QuantumCircuit(self.num_wind_vars*2 + 1 + m, name="QAE")
    #    ancilla = self.num_wind_vars*2
    #    for m_j in range(m):
    #        qm =  ancilla+1+m_j
    #        qc.h(qm)
    #        qc.append(self.grover_operator_asin_inconstraint(2**m_j, norm), self.wind_qubits+self.pdf_qubits+[ancilla]+[qm])
    #    return qc
        0


    # builders
    # TODO: add control parameters for different algorithm flows
    #def build_uopt_oracle(self):

    # executors
    def execute_optimizer(self, qc, num_meas=None):
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            simulator = Aer.get_backend('statevector_simulator')
            #job = execute(circ, backend)
            qc = transpile(qc, simulator)
            result = simulator.run(qc).result()
            statevector = result.get_statevector(qc)
            return statevector.probabilities_dict()
        # use counts
        else:
            qc.measure_all()
            simulator = Aer.get_backend('aer_simulator_statevector')
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=num_meas).result()
            counts = result.get_counts(qc)
            counts = {key:value/num_meas for key,value in counts.items()}
            return counts
    
    def execute_optimizer_oracle(self, qc, num_meas=None):
        ''' given the optimizer followed by the oracle, get the amplitude of the ancilla register
        '''
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            simulator = Aer.get_backend('statevector_simulator')
            #job = execute(circ, backend)
            qc = transpile(qc, simulator)
            result = simulator.run(qc).result()
            statevector = result.get_statevector(qc)
            pd = statevector.probabilities_dict()
            a = 0.
            for key,value in pd.items():
                if key[0] == '1':
                    a += value
            return a
        # use counts
        else:
            #qc.measure_all()
            simulator = Aer.get_backend('aer_simulator_statevector')
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=num_meas).result()
            counts = result.get_counts(qc)
            return counts['1'] / num_meas if '1' in counts.keys() else 0


    def execute_qae(self, qc, m, num_meas=None):
        ''' assume we have the full qae circuit
            specify how many sample qubits there are
        '''
        # if no number of measurements is specified, try to use statevector
        if num_meas is None:
            simulator = Aer.get_backend('statevector_simulator')
            #job = execute(circ, backend)
            qc = transpile(qc, simulator)
            result = simulator.run(qc).result()
            statevector = result.get_statevector(qc)
            pd = statevector.probabilities_dict()
            b_counts = {}
            for key,value in pd.items():
                reg = key[:m]
                if reg not in b_counts.keys():
                    b_counts[reg] = 0
                b_counts[reg] += value
            return b_counts
        # use counts
        else:
            #qc.measure_all()
            qc_meas = QuantumCircuit(2*self.num_wind_vars+1+m, m)
            qc_meas.append(qc, list(range(2*self.num_wind_vars+1+m)))
            qc_meas.measure(list(range(2*self.num_wind_vars+1, 2*self.num_wind_vars+1+m)), list(range(m)))
            simulator = Aer.get_backend('aer_simulator_statevector')
            qc_meas = transpile(qc_meas, simulator)
            result = simulator.run(qc_meas, shots=num_meas).result()
            counts = result.get_counts(qc_meas)
            counts = {key:value/num_meas for key,value in counts.items()}
            return counts


    # post-processors
    def process_expectation_value_optimizer(self, wind_demand, counts):
        ''' post-process the counts returned from only the optimizer to get the expectation value
        '''
        expectation_value = 0.
        for bstr,val in counts.items():
            output = self.bstr_to_output(bstr)
            wind_output = output[:int(len(output)/2)]
            pdf_output = output[int(len(output)/2):]
            expectation_value += self.wind_scenario_cost(wind_output, pdf_output, wind_demand)*val
        return expectation_value


def main():

    bno = BinaryNestedOptimizer([1,], [0.1, 0.2, 0.15, .1, .1], 10, {(0,0,0,0,0): 0.5, (0,1,1,0,1):0.5}, 3)
    demand_surface = bno.brute_force_wind_demand_expectation_values()

    q = bno.dicke_state_circuit(3)
    print(q)
    sv = Statevector.from_label('0'*5)
    sv = sv.evolve(q)
    print(sv.probabilities_dict())
    return
    # should be 0, 5.075 
    # second val is cr*.5 + c_2*.5 (since gen2 is the cheapest available in scenario 011)
    print("expectation value surface")
    print(demand_surface)
    obj_x = bno.brute_force_energy_surface()
    print("objective function surface")
    print(obj_x)

    print("QC!!!")
    print("adiabatic evolution circuit")
    wind_demand = 1
    Uopt = bno.adiabatic_evolution_circuit(wind_demand, time=100, time_steps=4, norm=10)
    print(Uopt)
    opt_counts = bno.execute_optimizer(Uopt, num_meas=10_000)
    print(opt_counts)
    opt_expectation_value = bno.process_expectation_value_optimizer(wind_demand, opt_counts)
    print(opt_expectation_value)


    print("A = F(Uopt o I)")
    qc = QuantumCircuit(7)
    Uopt = bno.adiabatic_evolution_circuit(wind_demand, 100, 4, 10)
    qc.append(Uopt, list(range(6)))
    F = bno.single_oracle_asin_inconstraint(wind_demand*10)
    qc.append(F, list(range(7)))
    print(qc)


    bno.grover_reflections_asin_inconstraint(m=4, norm=wind_demand*10)

if __name__=='__main__':
    main()