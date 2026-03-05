#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:57:51 2023

@author: crotello
"""

# -*- coding: utf-8 -*-
from optimizer_utils import *


import matplotlib.pyplot as plt
import matplotlib

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, Aer
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library.standard_gates import RXGate, XGate, CXGate, CSwapGate
from qiskit.circuit.library.standard_gates import CCZGate, HGate, SwapGate, iSwapGate, CCXGate, CZGate


## Test VariableRegister
def _TEST_swapOperator():
    print("Testing VariableRegister.swapOperator(...) function")
    
    ## Test unary swaps - full swap
    print("\t\tunary swap, full...", end='')
    var1 = VariableRegister(3, 'unary')
    var2 = VariableRegister(3, 'unary')
    qc = QuantumCircuit(6,6)
    qc.append(var1.setValue(2), [0,1,2])
    qc.append(var1.swapOperator(var2, 1), [0,1,2,3,4,5])
    qc.measure([0,1,2,3,4,5], [0,1,2,3,4,5])
    
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=1_000).result()
    counts = result.get_counts()
    assert(len(counts) == 1)
    bstr = list(counts.keys())[0]
    bstr1 = bstr[3::]
    bstr2 = bstr[:3]
    assert(var1.getValue(bstr1) == 0)
    assert(var2.getValue(bstr2) == 2)
    print("success")
    
    ## Test unary swaps - no swap
    print("\t\tunary swap, none...",end='')
    var1 = VariableRegister(3, 'unary')
    var2 = VariableRegister(3, 'unary')
    qc = QuantumCircuit(6,6)
    qc.append(var1.setValue(2), [0,1,2])
    qc.append(var1.swapOperator(var2, 0), [0,1,2,3,4,5])
    qc.measure([0,1,2,3,4,5], [0,1,2,3,4,5])
    
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=1_000).result()
    counts = result.get_counts()
    assert(len(counts) == 1)
    bstr = list(counts.keys())[0]
    bstr1 = bstr[3::]
    bstr2 = bstr[:3]
    assert(var1.getValue(bstr1) == 2)
    assert(var2.getValue(bstr2) == 0)
    print("success")
    
    
    ## Test binary swaps - full swap - one
    print("\t\tbinary swap, full, |0>|1>...",end='')
    print()
    var1 = VariableRegister(3, 'binary')
    var2 = VariableRegister(3, 'binary')
    qc = QuantumCircuit(4,4)
    qc.append(var1.setValue(1), [0,1])
    qc.append(var1.swapOperator(var2, 1), [0,1,2,3])
    qc.measure([0,1,2,3], [0,1,2,3])
    
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=1_000).result()
    counts = result.get_counts()
    print(counts)
    print(qc)
    assert(len(counts) == 1)
    bstr = list(counts.keys())[0]
    bstr1 = bstr[2:]
    bstr2 = bstr[:2]
    assert(var1.getValue(bstr1) == 0)
    assert(var2.getValue(bstr2) == 1)
    print("success")
    
    
    ## Test binary swaps - full swap - two
    print("\t\tbinary swap, full, |0>|2>...",end='')
    print()
    var1 = VariableRegister(3, 'binary')
    var2 = VariableRegister(3, 'binary')
    qc = QuantumCircuit(4,4)
    qc.append(var1.setValue(2), [0,1])
    qc.append(var1.swapOperator(var2, 1), [0,1,2,3])
    qc.measure([0,1,2,3], [0,1,2,3])
    
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    print(qc)
    result = simulator.run(qc, shots=1_000).result()
    counts = result.get_counts()
    print(counts)
    assert(len(counts) == 1)
    bstr = list(counts.keys())[0]
    bstr1 = bstr[2:]
    bstr2 = bstr[:2]
    assert(var1.getValue(bstr1) == 0)
    assert(var2.getValue(bstr2) == 2)
    print("success")
    print("\tSuccess!")
    


def _TEST_getValue():
    print("Testing VariableRegister.getValue(...) function")
    ## Test unary values
    var = VariableRegister(7, 'unary')
    five = '1111100'
    assert(var.getValue(five) == 5)
    one = '0001000'
    assert(var.getValue(one) == 1)
    ## Test binary values
    var = VariableRegister(7, 'binary')
    five = '101'
    assert(var.getValue(five) == 5)
    one = '001'
    assert(var.getValue(one) == 1)
    print("\tSuccess!")
 


def _TEST_numberOperator():
    print("Testing VariableRegister.numberOperator(...) function")
    
    ## Test unary values
    # 2
    var = VariableRegister(7, 'unary')
    qc = var.setValue(2)
    qc.append(var.numberOperator(1), list(range(7)))
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi  = result.get_statevector(qc)
    assert(np.isclose(psi[3], np.exp(1j*2)))
    
    # 5 div 3
    var = VariableRegister(7, 'unary')
    qc = var.setValue(5)
    qc.append(var.numberOperator(1/3), list(range(7)))
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi  = result.get_statevector(qc)
    assert(np.isclose(psi[int('011111',2)], np.exp(1j*5/3)))
    
    ## Test binary values
    # 7
    var = VariableRegister(7, 'binary')
    qc = var.setValue(7)
    qc.append(var.numberOperator(1), list(range(3)))
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi  = result.get_statevector(qc)
    assert(np.isclose(psi[7], np.exp(1j*7)))
    
    # 12 amp .77
    var = VariableRegister(15, 'binary')
    qc = var.setValue(12)
    qc.append(var.numberOperator(0.77), list(range(4)))
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi  = result.get_statevector(qc)
    assert(np.isclose(psi[12], np.exp(1j*12*0.77)))
    print("\tSuccess!")
 


def _TEST_productOperator():
    print("Testing VariableRegister.productOperator(...) function")
    
    ## Test unary values
    # 2*3
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(2), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(3), [4,5,6,7])
    qc.append(var1.productOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01110011',2)], np.exp(1j*6)))
    
    # 3*3, div 7
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(3), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(3), [4,5,6,7])
    qc.append(var1.productOperator(var2, 1/7), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01110111',2)], np.exp(1j*9/7)))
        
    ## Test binary values
    # 8*7
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(8), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(7), [4,5,6,7])
    qc.append(var1.productOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01111000',2)], np.exp(1j*8*7)))
    
    # 15*13, div -3
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(15), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(13), [4,5,6,7])
    qc.append(var1.productOperator(var2, -1/3), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('11011111',2)], np.exp(-1j*15*13/3)))
    print("\tSuccess!")



def _TEST_lessThanOperator():
    print("Testing VariableRegister.lessThanOperator(...) function")
    
    ## Test unary values
    # 2,3 -> 0
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(2), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(3), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01110011',2)], np.exp(1j*0)))
    
    # 1,1 -> 0
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(1), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(1), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('00010001',2)], np.exp(1j*0)))
    
    # 3,1, -> 2
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(3), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(1), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('00010111',2)], np.exp(1j*2)))
    
    # amp=1/4,(3,0), -> 3*1/4
    qc = QuantumCircuit(8)
    var1 = VariableRegister(4, 'unary')
    qc.append(var1.setValue(3), [0,1,2,3])
    var2 = VariableRegister(4, 'unary')
    qc.append(var2.setValue(0), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1/4), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator') 
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('00000111',2)], np.exp(1j*3/4)))
        
    ## Test binary values
    # 7,7 -> 0
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(7), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(7), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01110111',2)], np.exp(1j*0)))
    
    # 2,14 -> 0
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(2), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(14), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator') 
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('11100010',2)], np.exp(1j*0)))
    
    # 1/256 * 13,2 -> 11 / 256
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(13), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(2), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1/256), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('00101101',2)], np.exp(1j*11/256)))
    
    # 7,6 -> 1
    qc = QuantumCircuit(8)
    var1 = VariableRegister(15, 'binary')
    qc.append(var1.setValue(7), [0,1,2,3])
    var2 = VariableRegister(15, 'binary')
    qc.append(var2.setValue(6), [4,5,6,7])
    qc.append(var1.lessThanOperator(var2, 1), list(range(8)))
    
    backend = BasicAer.get_backend('statevector_simulator') 
    result = backend.run(transpile(qc, backend)).result()
    psi = result.get_statevector(qc)
    assert(np.isclose(psi[int('01100111',2)], np.exp(1j*1)))
    print("\tSuccess!")
    




## Test PowerSystem_1Bus
def _TEST_makeSystem():
    print("Testing PowerSystem_1Bus")
    ## Test the PowerSystem_1Bus
    system = PowerSystem_1Bus(
            gas_costs=[4,4,3,3], wind_costs=[1,1,1,1], decision_levels=4,
            undersatisfied_cost=8, demand=6, pdf={tuple([3,3,3,3]): 0.75, tuple([0,0,0,0]): 0.25}
        )

    print(system.variable_costs)
    assert(system.variable_costs == [4,4,3,3,
                                     .75,.75,.75,.75,8*.75,
                                     .25,.25,.25,.25,8*.25])
    assert(system.demand == 6)
    assert(system.scenarios == [(3,3,3,3), (0,0,0,0)])
    print("\tSuccess!")
   
    

def _TEST_normalization():
    print("Testing PowerSystem_1Bus.normalization ")
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]) :1},
            normalization = (12, np.pi)
        )
    assert(system.gas_costs == [np.pi/4] and system.wind_costs == [np.pi/12] and np.isclose(system.undersatisfied_cost, 10/12 * np.pi))
    
    system = PowerSystem_1Bus(
            gas_costs=[3,6], wind_costs=[1,.5], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 1.},
            normalization = (12, np.pi/2)
        )
    assert(np.isclose(system.gas_costs, [np.pi/8, np.pi/4]).all())
    assert(np.isclose(system.wind_costs, [np.pi/24, np.pi/48]).all())
    assert(np.isclose(system.undersatisfied_cost, 10/24*np.pi))
    print("\tSuccess!")



def _TEST_price():
    print("Testing PowerSystem_1Bus._price()")
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
        )
    cost = system._price([2,1,0,1,0,1,0,0,1])
    assert(cost == 7.9)
    
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 0.5, tuple([0]): 0.5}
        )
    cost = system._price([3,0,0,0,0])
    assert(cost == 9)
    print("\tSuccess!")
    


def _TEST_cobylaSolve():
    print("Testing PowerSystem_1Bus.cobylaSolve()")
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
        )
    res = system.cobylaSolve()
    assert(list(res.x) == [2,1,0,1,0,1,0,0,1])
    
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 0.5, tuple([0]): 0.5}
        )
    res = system.cobylaSolve()
    assert(list(res.x) == [3,0,0,0,0])
    print("\tSuccess!")



def _TEST_cobylaSolveSubproblems():
    print("Testing PowerSystem_1Bus.getFirstStageCosts()")
    system = PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
        )
    cost,decisions = system.getFirstStageCosts([tuple([2])])
    res = system.cobylaSolve()
    assert((decisions[tuple([2])] == res.x).all())
    assert(cost[tuple([2])] == res.fun)
    print("\tSuccess!")





def main():
    # PowerSystem_1Bus tests
    _TEST_makeSystem()
    _TEST_normalization()
    _TEST_price()
    _TEST_cobylaSolve()
    _TEST_cobylaSolveSubproblems()

    # VariableRegister tests
    _TEST_getValue()
    _TEST_numberOperator()
    _TEST_productOperator()
    _TEST_lessThanOperator()
    #_TEST_swapOperator() # TODO: make this work

if __name__=='__main__':
    main()