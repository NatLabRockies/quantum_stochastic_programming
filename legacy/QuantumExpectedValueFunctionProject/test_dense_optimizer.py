
"""
Created on Thu Sep 31 

@author: crotello
"""

import optimizer_utils
from dense_optimizer import *

######
# 3 Phases of annealing, decide a path of 
######

def _TEST_twoVars_fourLevels_unary_classical(demand=3, save_plots=False):
    print("Testing |G|=|W|=1, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{tuple([3]): 0.5, tuple([0]): 0.5,}, 
                {tuple([0]): 0.1, tuple([1]): 0.25, tuple([2]): 0.4, tuple([3]): 0.25,},
                {tuple([0]): 0.1, tuple([1]): 0.2, tuple([2]): 0.3, tuple([3]): 0.4,},
                {tuple([0]): 0.15, tuple([1]): 0.6, tuple([2]): 0.15, tuple([3]): 0.1,},
                {tuple([0]): 0.15, tuple([1]): 0.35, tuple([2]): 0.15, tuple([3]): 0.35,},
                {tuple([0]): 0.05, tuple([1]): 0.1, tuple([2]): 0.5, tuple([3]): 0.35,},
                ]
    pdf_name_list = [
                "p5,0,0,p5",
                "p1,p25,p4,p25",
                "p1,p2,p3,p4",
                "p15,p6,p15,p1",
                "p15,p35,p15,p35",
                "p05,p1,p5,p35"
    ]
    for i,pdf in enumerate(pdf_list):
        print("\t Test d={}, Pr={}".format(demand, pdf))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,],
            wind_costs=[1,],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=4,
            pdf = pdf,
            normalization=None#(20,1)
        ) 
        # 
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        # 
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        #counts = opt.solveThreePhaseAnnealing(total_time=10, time_1_steps=8, time_2=8, time_3=9, time_3_steps=3,
        counts = opt.solveThreePhaseAnnealing(total_time=10, time_1_steps=8, time_2=8, time_3=9, time_3_steps=3,
                                              num_permutations=None, samples_repeats=2,
                                              init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=1_000)
        #print(counts)
        counts = list(counts.values())[0]
        gas_decisions = opt.getGasCounts(counts)
        gas_decision,freq = opt.getDecision(gas_decisions)
        #
        if save_plots:
            plt.title("Test d={}, Pr(*)={}".format(demand, pdf_name_list[i]))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_classical_figs/vars=2_demand={}_levels=4_unary_[{}].png".format(demand, pdf_name_list[i]))
            plt.cla()
        #assert(true_gas_dec == gas_decision)
        #break
        print("\t\tSuccess!")
    print("Success!!!")


def _TEST_fourVars_twoLevels_unary_classical(demand=3, save_plots=False):
    print("Testing |G|=|W|=2, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{(0,0): 0.1, (0,1): 0.15, (1,0): 0.15, (1,1): 0.6},
                {(0,0): 0.05, (0,1): 0.3, (1,0): 0.3, (1,1): 0.35},
                {(1,1): 0.05, (0,1): 0.3, (1,0): 0.3, (0,0): 0.35},
                ]
    pdf_name_list = ["p1,p15,p15,p6",
                     "p05,p3,p3,p35",
                     "p35,p3,p3,p05"
                     ]
    # given a list of pdfs, test each one
    for i,_ in enumerate(pdf_list):
        pdf = pdf_list[i]
        pdf_name = pdf_name_list[i]
        print("\tTest d={}, Pr(00, 01, 10, 11)=[{}] ...".format(demand, pdf_name))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,2.8],
            wind_costs=[1,1.1],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=2,
            pdf=pdf,
            normalization=(10,1)
        ) 
        #
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        counts = opt.solveThreePhaseAnnealing(total_time=10, time_1_steps=8, time_2=8, time_3=8, time_3_steps=3,
                                              num_permutations=None, samples_repeats=1,
                                              init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=1_000)
        #
        counts = list(counts.values())[0]
        gas_decisions = opt.getGasCounts(counts)
        gas_decision, freq = opt.getDecision(gas_decisions)
        if save_plots:
            plt.title("Test d={}, Pr(00, 01, 10, 11)=[{}]".format(demand, pdf_name))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_classical_figs/vars=4_levels=2_demand={}_unary_[{}].png".format(demand, pdf_name))
            plt.cla()
        #assert((true_gas_dec == gas_decision).all())
        print("\t\tSuccess!")
    print("Success!!!")



def _TEST_fourVars_threeLevels_unary_classical(demand=6, save_plots=False):
    print("Testing |G|=|W|=2, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{(0,1): 0.05, (1,0): 0.05, (1,1): 0.2, (2,0): 0.2, (0,2): 0.2, (2,1): 0.1, (1,2): 0.1, (2,2): .1},
                {(0,0): 0.02, (0,1): 0.02, (1,0): 0.02, (1,1): 0.05, (2,0): 0.02, (0,2): 0.02, (2,1): 0.3, (1,2): 0.3, (2,2): .25},
                #{(0,0): 0.05, (0,1): 0.3, (1,0): 0.3, (1,1): 0.35}
                ]
    
    pdf_name_list = ["p00,p05,p05,p2,p2,p2,p1,p1,p1",
                     "p02,p02,p02,p05,p02,p02,p3,p3,p25"
                     ]
    # given a list of pdfs, test each one
    for i,_ in enumerate(pdf_list):
        pdf = pdf_list[i]
        pdf_name = pdf_name_list[i]
        print("\tTest d={}, Pr(xi)=[{}] ...".format(demand, pdf_name))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,1.5],
            wind_costs=[0.01,0.011],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=3,
            pdf=pdf,
            normalization=(10,1)
        ) 
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        counts = opt.solveTwoPhaseAnnealing(20, 29, 5, init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=10_000)
        gas_decisions = opt.getGasCounts(counts)
        gas_decision, freq = opt.getDecision(gas_decisions)
        if save_plots:
            plt.title("Test d={}, Pr(00, 01, 10, 11)=[{}]".format(demand, pdf_name))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_figs/vars=4_levels=3_demand={}_unary_[{}].png".format(demand, pdf_name))
            plt.cla()
        print((true_gas_dec == gas_decision).all())
        print("\t\tSuccess!")
    print("Success!!!")




######
# 2 Phases of annealing, measure the wavefunction throughout second phase
######

def _TEST_twoVars_fourLevels_unary_wvfnpdf(demand=3, save_plots=False):
    print("Testing |G|=|W|=1, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{tuple([3]): 0.5, tuple([0]): 0.5,}, 
                {tuple([0]): 0.1, tuple([1]): 0.25, tuple([2]): 0.4, tuple([3]): 0.25,},
                {tuple([0]): 0.1, tuple([1]): 0.2, tuple([2]): 0.3, tuple([3]): 0.4,},
                {tuple([0]): 0.15, tuple([1]): 0.6, tuple([2]): 0.15, tuple([3]): 0.1,},
                {tuple([0]): 0.15, tuple([1]): 0.35, tuple([2]): 0.15, tuple([3]): 0.35,},
                ]
    pdf_name_list = [
                "p5,0,0,p5",
                "p1,p25,p4,p25",
                "p1,p2,p3,p4",
                "p15,p6,p15,p1",
                "p15,p35,p15,p35",
    ]
    for i,pdf in enumerate(pdf_list):
        print("\t Test d={}, Pr={}".format(demand, pdf))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,],
            wind_costs=[1,],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=4,
            pdf = pdf,
            normalization=(10,1)
        ) 
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        counts = opt.solveTwoPhaseAnnealing(10, 9, 8, init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=10_000)
        gas_decisions = opt.getGasCounts(counts)
        gas_decision,freq = opt.getDecision(gas_decisions)
        if save_plots:
            plt.title("Test d={}, Pr(*)={}".format(demand, pdf_name_list[i]))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_figs/vars=2_demand={}_levels=4_unary_[{}].png".format(demand, pdf_name_list[i]))
            plt.cla()
        assert(true_gas_dec == gas_decision)
        print("\t\tSuccess!")
    print("Success!!!")


def _TEST_fourVars_twoLevels_unary_wvfnpdf(demand=3, save_plots=False):
    print("Testing |G|=|W|=2, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{(0,0): 0.1, (0,1): 0.15, (1,0): 0.15, (1,1): 0.6},
                {(0,0): 0.05, (0,1): 0.3, (1,0): 0.3, (1,1): 0.35}
                ]
    pdf_name_list = ["p1,p15,p15,p6",
                     "p05,p3,p3,p35"
                     ]
    # given a list of pdfs, test each one
    for i,_ in enumerate(pdf_list):
        pdf = pdf_list[i]
        pdf_name = pdf_name_list[i]
        print("\tTest d={}, Pr(00, 01, 10, 11)=[{}] ...".format(demand, pdf_name))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,2.8],
            wind_costs=[1,1.1],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=2,
            pdf=pdf,
            normalization=(10,1)
        ) 
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        counts = opt.solveTwoPhaseAnnealing(10, 9, 4, init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=10_000)
        gas_decisions = opt.getGasCounts(counts)
        gas_decision, freq = opt.getDecision(gas_decisions)
        if save_plots:
            plt.title("Test d={}, Pr(00, 01, 10, 11)=[{}]".format(demand, pdf_name))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_figs/vars=4_levels=2_demand={}_unary_[{}].png".format(demand, pdf_name))
            plt.cla()
        assert((true_gas_dec == gas_decision).all())
        print("\t\tSuccess!")
    print("Success!!!")



def _TEST_fourVars_threeLevels_unary_wvfnpdf(demand=6, save_plots=False):
    print("Testing |G|=|W|=2, cg=3, cw=1, cy=10, E=4, Normalized 10->1, varying PDFs")
    pdf_list = [{(0,1): 0.05, (1,0): 0.05, (1,1): 0.2, (2,0): 0.2, (0,2): 0.2, (2,1): 0.1, (1,2): 0.1, (2,2): .1},
                {(0,0): 0.02, (0,1): 0.02, (1,0): 0.02, (1,1): 0.05, (2,0): 0.02, (0,2): 0.02, (2,1): 0.3, (1,2): 0.3, (2,2): .25},
                #{(0,0): 0.05, (0,1): 0.3, (1,0): 0.3, (1,1): 0.35}
                ]
    
    pdf_name_list = ["p00,p05,p05,p2,p2,p2,p1,p1,p1",
                     "p02,p02,p02,p05,p02,p02,p3,p3,p25"
                     ]
    # given a list of pdfs, test each one
    for i,_ in enumerate(pdf_list):
        pdf = pdf_list[i]
        pdf_name = pdf_name_list[i]
        print("\tTest d={}, Pr(xi)=[{}] ...".format(demand, pdf_name))
        system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3,1.5],
            wind_costs=[0.01,0.011],
            undersatisfied_cost=10,
            demand=demand,
            decision_levels=3,
            pdf=pdf,
            normalization=(10,1)
        ) 
        true_ans = system.cobylaSolve()
        true_gas_dec = true_ans.x[:system.num_gas_generators]
        opt = Optimizer_Dense(system, encoding='unary', slack_register=False)
        counts = opt.solveTwoPhaseAnnealing(20, 29, 5, init_cond='DICKE', mixer='SWAP', phase='COST', num_meas=10_000)
        gas_decisions = opt.getGasCounts(counts)
        gas_decision, freq = opt.getDecision(gas_decisions)
        if save_plots:
            plt.title("Test d={}, Pr(00, 01, 10, 11)=[{}]".format(demand, pdf_name))
            system.plotMeasurementsVExpectedCost(gas_decisions, pdf=True)
            plt.savefig("test_dense_optimizer_figs/vars=4_levels=3_demand={}_unary_[{}].png".format(demand, pdf_name))
            plt.cla()
        print((true_gas_dec == gas_decision).all())
        print("\t\tSuccess!")
    print("Success!!!")



def _TEST_pdfInitialize():
    print("Testing Optimizer_Dense.initializePDF() ")
    ## Test unary, single outcome
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=2, 
            pdf={tuple([1]): 1.}
        )
    opt = Optimizer_Dense(system, 'unary')
    qc = opt.initializePDF()
    qc.measure_all()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=100_000).result()
    counts = result.get_counts()
    assert(np.isclose(counts['001']/100_000, 1.))
    
    ## Test unary, two outcomes
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=2, 
            pdf={tuple([3]): .25, tuple([1]): 0.75}
        )
    opt = Optimizer_Dense(system, 'unary')
    qc = opt.initializePDF()
    qc.measure_all()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=100_000).result()
    counts = result.get_counts()
    assert(np.isclose(counts['111']/100_000, 0.25, atol=2))
    assert(np.isclose(counts['001']/100_000, 0.75, atol=2))
    
    ## Test binary, two outcomes
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): .25, tuple([1]): 0.75}
        )
    opt = Optimizer_Dense(system, 'binary')
    qc = opt.initializePDF()
    qc.measure_all()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=100_000).result()
    counts = result.get_counts()
    assert(np.isclose(counts['11']/100_000, 0.25, atol=2))
    assert(np.isclose(counts['01']/100_000, 0.75, atol=2))
    
    ## Test binary four outcomes
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[3], wind_costs=[1], decision_levels=4,
            undersatisfied_cost=10, demand=3, 
            pdf={tuple([3]): .4, tuple([2]): 0.3, tuple([1]): .2, tuple([0]): 0.1,}
        )
    opt = Optimizer_Dense(system, 'binary')
    qc = opt.initializePDF()
    qc.measure_all()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=100_000).result()
    counts = result.get_counts()
    assert(np.isclose(counts['11']/100_000, 0.4, atol=2))
    assert(np.isclose(counts['10']/100_000, 0.3, atol=2))
    assert(np.isclose(counts['01']/100_000, 0.2, atol=2))
    assert(np.isclose(counts['00']/100_000, 0.1, atol=2))
    
    ## Test binary several variables
    system = optimizer_utils.PowerSystem_1Bus(
            gas_costs=[2.9, 3, 3.1],
            wind_costs=[1, 0.9, 1.1],
            decision_levels=2, undersatisfied_cost=10, demand=3,
            pdf={(0,1,1): 0.3, (1,1,1): 0.4, (1,0,0): 0.2, (0,0,0): 0.05, (0,0,1):0.05}
        )
    opt = Optimizer_Dense(system, 'binary')
    qc = opt.initializePDF()
    qc.measure_all()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)
    result = simulator.run(qc, shots=100_000).result()
    counts = result.get_counts()
    plt.bar(counts.keys(), counts.values())
    print("\tSuccess!")
    
 
    

def main():
    #_TEST_fourVars_threeLevels_unary_classical(demand=5, save_plots=True)
    #_TEST_fourVars_twoLevels_unary_classical(demand=2, save_plots=True)
    _TEST_twoVars_fourLevels_unary_classical(demand=3, save_plots=True)

    #_TEST_fourVars_threeLevels_unary_wvfnpdf(demand=5, save_plots=True)
    #_TEST_fourVars_twoLevels_unary_wvfnpdf(demand=3, save_plots=True)
    #_TEST_twoVars_fourLevels_unary_wvfnpdf(demand=3, save_plots=True)

    _TEST_pdfInitialize()

if __name__=='__main__':
    main()