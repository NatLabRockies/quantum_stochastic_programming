#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:39:43 2023

@author: crotello
"""

import expanded_optimizer as eo
import optimizer_utils

def _TEST_optimizersetupunary_expandedPowerSystem1Bus():
    print("Testing the Expanded optimizer setup for PowerSystem 1 Bus, unary encoding")
    ## Test the PowerSystem_1Bus 
    ### 1 Gas, 1 Wind, 4 scenarios and 4 levels
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,2], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=2, 
            pdf={tuple([3]): 0.4, tuple([2]): 0.3, tuple([1]): 0.2, tuple([0]): 0.1}
        )
    #print(system)
    ### OPTIMAL choice x=1, w0=0, w1=w2=w3=1
    opt = eo.Optimizer_Expanded(system, 'unary')
    assert(opt.num_variables == 6)
    assert(opt.variable_costs == [3,2,1,1,1,1])
    assert(opt.num_qubits == 18)
    assert(opt.varid_to_qubits == {0:[0,1,2], 1:[3,4,5], 2:[6,7,8], 3:[9,10,11], 4:[12,13,14], 5:[15,16,17]})
    print("Success!")

def main():
    _TEST_optimizersetupunary_expandedPowerSystem1Bus()

if __name__=="__main__":
    main()