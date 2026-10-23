"""Compatibility adapters for the original dictionary-based circuit API.

New code should use the typed builders in qiskit_impl.circuits.
"""

from qiskit import QuantumCircuit

from qiskit_impl import circuits


def demand_constraint_preserving_mixer(amplitude: float, args: dict) -> QuantumCircuit:
    return circuits.demand_constraint_preserving_mixer(amplitude, args["y_reg"])


def cost_operator(amplitude: float, args: dict) -> QuantumCircuit:
    return circuits.cost_operator(
        amplitude, args["c_y"], args["c_r"], args["cost_norm"], args["pdf_reg"]
    )


def cost_scenario_operator(amplitude: float, args: dict) -> QuantumCircuit:
    return circuits.cost_scenario_operator(
        amplitude, args["wind_costs"], args["constraint_amplitude"],
        args["cost_norm"], args["scenario"],
    )


def single_oracle_sin_inconstraint(args: dict, inverse=False) -> QuantumCircuit:
    return circuits.cost_oracle(
        args["c_y"], args["c_r"], args["norm"], args["y_reg"], args["pdf_reg"],
        encoding="arcsine" if args.get("w_d", 1) == 1 else "linear",
        inverse=inverse,
    )


def dicke_state_circuit(args: dict) -> QuantumCircuit:
    return circuits.dicke_state_circuit(args["n_y"], args["w_d"])


def alternating_operator_ansatz(args: dict) -> QuantumCircuit:
    return circuits.alternating_operator_ansatz(
        args["n_y"], args["y_reg"], args["pdf_reg"], args["Theta"],
        args["initial_state_circuit"](args), args["pdf_circuit"](args),
        lambda theta: args["cost_operator_circuit"](theta, args),
        lambda theta: args["mixer_operator_circuit"](theta, args),
    )
