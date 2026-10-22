# Solving two-stage stochastic optimization problems on a quantum processor

This repository holds the code to solve two-stage stochastic optimization on a quantum computer. 

## Code organization
Each file has a corresponding Python test script; i.e. to test code in `file.py` run the script `test_file.py`. For in-vitro features, tests are contained within that file and run from main: if I am working on adding function `a()` to `file.py`, I will have the function `_TEST_a()` within `file.py` _instead_ of `test_file.py`, and will move it to `test_file.py` when the feature is complete.

Specific experiments should be implemented in a Jupyter Notebook and import relevant modules from this repository.

## File descriptions
`optimizer_utils.py`
+ `class VariableRegister` Stores code for interfacing with a collection of qubits as a single unary or binary variable. Note: does _not_ store specific qubit indicies; optimizers are in charge of this.
+ `class PowerSystem_1Bus` Stores variable costs, demand, decision levels (discretization), and the PDF. In the future, this will also hold samples instead of a PDF.

`expanded_optimizer.py`
+ `class Optimizer_Expanded` Solves the optimization problem with wind/slack variables expanded for each scenario, with amplitude weighted by the probability.
    + Implements quadratic penalty constraint.
    + Currently only implements Annealing-inspired algorithm, but straightforward to implement general QAOA.
    + Measurement basis vectors give a decision for the gas generators and wind turbines for each weather scenario.

`denser_optimizer.py`
+ `class Optimizer_Dense` Solves the optimization problem by using some kind of superposition over scenarios to get the best first-stage decision.
    + Implements quadratic penalty constraint and constraint-preserving mixers
    + Only implements Annealing-inspired algorithm. Unclear how to do a variational algorithm, but likely Quantum Amplitude Estimation (QAE) is necessary - has been used for Value-at-Risk calculations, which are isomorphic.
    + Measurement basis vectors are a single scenario and the gas/wind we would have chosen. Only measure gas to get first-stage decisions.
    + PDFs are used in two ways:
        + Per-scenario samples by optimizing with the PDF, then form a single 'hedged' decision by measuring and re-initializing the PDF trepeatedly towards the end of the annealing.
        + Same as above, but instead of re-initializing and sampling the PDF at the end of the optimization, we run through samples of our PDF at the end of optimization. **NOTE unimplemented here**