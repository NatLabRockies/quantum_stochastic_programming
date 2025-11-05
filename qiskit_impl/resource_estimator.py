'''
created by csagal on 8/6/2024
Script for quantum resource estimation
'''
# Package Import
import numpy as np
from pytket.circuit import Circuit, OpType

# qiskit imports
from qiskit import transpile
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev

# qualtran imports
from qualtran.surface_code import PhysicalParameters
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary
from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost
from qualtran.surface_code import QECScheme
from qualtran.surface_code import FastDataBlock
from qualtran.surface_code import beverland_et_al_model

'''
args should contain the following:

{

args['num_algo_qubits'] - # qubits in the algorithm : int
args['circuit'] - # pytket circuit object : Circuit
args['num_meas'] - # measurement gates used : int
args['error_budget'] - accepted error threshold for algorithm : float
args['architecture'] - ion, superconducting, or majorana : str
args['optimistic'] - optimistic outlook:bool

magic state factory parameters
args['R'] - # distillation rounds : int
args['Q_0'] - physical qubit error rate : float
args['P_r'] - desired logical clifford gate error rate
args['F_copies'] - # copies of magic state factory

}
'''

class Resource_Estimator:

    def __init__(self, args:dict):

        self.args = args

    def bev_resource_estimator(self) -> dict:
        '''
        Performs resource estimation on quantum circuit and returns statistics
        Inspired by https://arxiv.org/pdf/2211.07629 and using code from https://qualtran.readthedocs.io/en/latest/surface_code/beverland_et_al_model.html

        Input: dict of arguments containing information about circuit of interest
        
        Output: dict containing fault-tolerant resource estimation metrics
        '''

        re_dict = get_circuit_metrics(self.args['circuit'])

        bev_dict = {}

        # Create algorithm summary
        qd_alg = AlgorithmSummary(
        n_algo_qubits = self.args['num_algo_qubits'],
        n_logical_gates = GateCounts(
            rotation = re_dict['number of 1-qubit gates'],
            measurement = self.args['num_meas'],
        ),
        n_rotation_layers = re_dict['circuit depth']
    )
        # Get # logical qubits
        logical_qubits = FastDataBlock.get_n_tiles(n_algo_qubits=qd_alg.n_algo_qubits)

        # Calculate the minimum number of logical time steps
        error_budget = self.args['error_budget']
        c_min = beverland_et_al_model.minimum_time_steps(
            error_budget = error_budget,
            alg = qd_alg,
            rotation_model = BeverlandEtAlRotationCost,
        )

        # Get the # of T operations
        t_operations = beverland_et_al_model.t_states(
            error_budget = error_budget,
            alg = qd_alg,
            rotation_model = BeverlandEtAlRotationCost
        )

        # QEC Scheme

        qec = QECScheme.make_beverland_et_al()
        beverland_phys_params = PhysicalParameters.make_beverland_et_al(self.args['architecture'], optimistic_err_rate=self.args['optimistic'])

        # Get Code distance
        d = beverland_et_al_model.code_distance(
            error_budget = error_budget,
            time_steps = c_min,
            alg = qd_alg,
            qec_scheme = qec,
            physical_error = beverland_phys_params.physical_error,
        )

        # Runtime in seconds
        t_s = d * beverland_phys_params.cycle_time_us * 1e-6 * c_min

        tau_d = d * beverland_phys_params.cycle_time_us
        # Get number of factories and factory qubits
        F, n_D = get_factory_params(self.args['R'], self.args['Q_0'], self.args['P_r'], d, self.args['F_copies'], t_operations, tau_d, c_min, error_budget)

        distillation_qubits = F * n_D
        q = distillation_qubits + logical_qubits * qec.physical_qubits(d)
        percent_distillation_qubits = round(distillation_qubits / q * 100, 1)

        # Print estimate
        print('Beverland Resource Estimate')
        print('---------------------------')

        print('Logical Qubits =', logical_qubits)
        print('Minimum Number of Logical Timesteps = %e' % c_min)
        print('# T operations needed = %e' % t_operations)
        print(f'{d = }')
        print(f'algorithm run time of {t_s:g} seconds')
        print('total number of physical qubits:', q)
        print('percentage of distillation qubits: {}%'.format(percent_distillation_qubits))        

        bev_dict['Logical Qubits'] = logical_qubits
        bev_dict['Minimum Number of Logical Timesteps'] = c_min
        bev_dict['# T Operations Needed'] = t_operations
        bev_dict['Code Distance'] = d
        bev_dict['Runtime (seconds)'] = t_s
        bev_dict['Total Number of Physical Qubits'] = q
        bev_dict['Percentage of Distillation Qubits'] = percent_distillation_qubits

        return bev_dict
    
'''
Helper Functions
'''
def get_circuit_metrics(circuit:Circuit) -> dict:
    '''
    Takes a circuit and returns gate counts, circuit depth, etc. for resource estimation purposes

    Input: Pytket Circuit
    Output: dictionary containing metrics
    '''
    # Check that we have a pytket circuit
    try:
        assert type(circuit) == Circuit

        re_dict = {}
    
        re_dict['total gate count'] = circuit.n_1qb_gates() + circuit.n_2qb_gates()
        re_dict['number of 1-qubit gates'] = circuit.n_1qb_gates()
        re_dict['number of 2-qubit gates'] = circuit.n_2qb_gates()
        re_dict['number of T gates'] = circuit.n_gates_of_type(OpType.T) + circuit.n_gates_of_type(OpType.Tdg)
        re_dict['number of 1-qubit Clifford rotations'] = re_dict['number of 1-qubit gates'] - re_dict['number of T gates']
        re_dict['circuit depth'] = circuit.depth()
        re_dict['2-qubit gate circuit depth'] = circuit.depth_2q()
        re_dict['1-qubit gate circuit depth'] = circuit.depth() - circuit.depth_2q()

        # Print results
        print('Circuit Metrics')
        print('---------------')

        for key, value in re_dict.items():
            print(f'{key}: {value}')
        print('\n')
        return re_dict

    except AssertionError:
        print('Not a pytket Circuit object. Please recompile to pytket and try again.')

def get_factory_params(R:int, Q_0:float, P_r:float, d:int, c:int, M:int, tau_d:float, C:int, error_budget:float, M_D=1):
    ''' 
    Calculates n(D) and F for 15-1 space-efficient logical distillation factories
    Inputs
    --------------------
    R - # Distillation rounds
    Q_0 - Physical qubit error rate
    P_r - logical Clifford gate error rate
    d - Code distance
    M - # T operations in circuit
    tau_d - Code distance cycle time (seconds)
    C - Logical timesteps of algorithm
    M_D - # T states generated from distillation
    error_budget - Acceptable error from algorithm

    Outputs
    --------------------
    F - # Factories
    n_D - # Physical Qubits
    ''' 
    P_T_list = [Q_0]
    P_T_D = Q_0
    P_list = [P_r]
    t = tau_d*C


    for round in range(R):
        P_T_list.append(35 * P_T_list[round - 1]**3 + 7.1 * P_list[round - 1])
        P_list.append(1 - 15 * P_T_list[round - 1] - 356 * P_list[round - 1])
    
    P_T_D = P_T_list[R]

    if P_T_D > error_budget/3:
        print('Error rate of factory too high, adjust parameters and try again')
        return 0
    
    else:

        tau_D = R * 13 * tau_d

        F = int(np.ceil(M * tau_D / (M_D * t)))

        n_D = max([c[r] * 2*d**2 for r in range(R)])

        return F, n_D

'''
Tools to move circuits between qiskit and pytket
'''
def qasm_to_clifford_and_t(qc, basic_approx_depth=3, recursion_degree=1):
    '''
    Takes qiskit circuit and decomposes it to gates in the group Clifford + T using Solovay-Kitaev Decomposition
    From https://quantumcomputing.stackexchange.com/questions/16006/how-can-i-find-a-cliffordt-approximation-of-an-arbitrary-one-qubit-gate-in-qisk
    '''
    # basic_approx_depth - size of basic circuits pool - as the RAM size I will to give
    # recursion_degree=1 -> best aprox from pool
    # bigger recursion_degree -> use only basic aproximations
    # recursion_degree - choose by resolution wanted
    qc = transpile(qc,basis_gates=["cx","u3"])
    basis = ["x", "y", "z", "s", "sdg", "t", "tdg", "z", "h"]
    approx = generate_basic_approximations(basis, depth=basic_approx_depth)
    skd = SolovayKitaev(recursion_degree, basic_approximations=approx)
    qc = skd(qc)

    return qc