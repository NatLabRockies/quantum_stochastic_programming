"""Run a Qiskit DQA/QAE sweep over the number of wind-turbine qubits."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
QISKIT_IMPL_DIR = REPO_ROOT / "qiskit_impl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QISKIT_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(QISKIT_IMPL_DIR))

from qiskit_impl import ExpValFun_functions as exp
from qiskit_impl.binary_optimizer import BinaryNestedOptimizer
from qiskit_impl.qae import QAE_Optimizer, pdf_initialize


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-y",
        type=int,
        nargs="+",
        default=[4, 6],
        metavar="N",
        help="wind-turbine qubit counts to run",
    )
    parser.add_argument(
        "--backend",
        choices=("aer", "exact"),
        default="aer",
        help="QAE readout execution mode; DQA uses an exact statevector",
    )
    parser.add_argument("--shots", type=int, default=2**14, help="QAE shots for --backend aer")
    parser.add_argument("--qpe-qubits", type=int, default=5, help="QPE readout register size")
    parser.add_argument(
        "--timesteps",
        type=int,
        help="DQA layers per problem size (default: n_y squared)",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        metavar="CSV",
        help="write result rows to this CSV file",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        metavar="PNG",
        help="save the summary plot to this path",
    )
    parser.add_argument("--show-plot", action="store_true", help="display the summary plot")
    args = parser.parse_args()
    if any(n_y < 3 for n_y in args.n_y):
        parser.error("every --n-y value must be at least 3")
    if args.shots < 1 or args.qpe_qubits < 1:
        parser.error("--shots and --qpe-qubits must be positive")
    if args.timesteps is not None and args.timesteps < 1:
        parser.error("--timesteps must be positive")
    return args


def run_sweep(args):
    results = []
    c_x = [3.0]
    c_r = 10.0
    x0 = [2]

    for n_y in args.n_y:
        print(f"\n{'=' * 60}\nn_y = {n_y}\n{'=' * 60}")

        n_x = 1
        n_xi = n_y
        d = n_y
        c_y = list(np.linspace(0.1, 1.0, n_y))
        timesteps = args.timesteps or n_y**2
        w_d = int(d - sum(x0))
        norm = w_d * c_r
        cost_norm = w_d * c_r / n_y
        pdf = {
            tuple(int(v) for v in f"{i:0{n_y}b}"): 1 / 2**n_y
            for i in range(2**n_y)
        }

        print(f"  c_y={[round(v, 2) for v in c_y]}, w_d={w_d}, norm={norm:.1f}")
        bno = BinaryNestedOptimizer(c_x, c_y, c_r, pdf, n_y, is_uniform=True)
        theta = [
            value
            for t in range(timesteps)
            for value in (t / timesteps, (1 - t / timesteps) / math.pi)
        ]
        print(f"  Linear ramp ({len(theta)} angles)")

        y_reg = list(range(n_y))
        pdf_reg = list(range(n_y, 2 * n_y))
        args_dqa = {
            "n_y": n_y,
            "n_x": n_x,
            "n_xi": n_xi,
            "c_x": c_x,
            "c_y": c_y,
            "c_r": c_r,
            "pdf": pdf,
            "y_reg": y_reg,
            "pdf_reg": pdf_reg,
            "Theta": theta,
            "w_d": w_d,
            "cost_norm": cost_norm,
            "uniform": True,
            "cost_operator_circuit": exp.cost_operator,
            "mixer_operator_circuit": exp.demand_constraint_preserving_mixer,
            "initial_state_circuit": exp.dicke_state_circuit,
            "pdf_circuit": pdf_initialize,
        }
        dqa_circuit = exp.alternating_operator_ansatz(args_dqa)
        print(f"  DQA circuit: {dqa_circuit.num_qubits} qubits, depth={dqa_circuit.depth()}")

        start = time.perf_counter()
        state = Statevector.from_label("0" * (n_y + n_xi)).evolve(dqa_circuit)
        dqa_counts = state.probabilities_dict()
        dqa_time = time.perf_counter() - start
        dqa_phi = bno.process_expectation_value_optimizer(w_d, dqa_counts)
        print(f"  DQA phi(w_d={w_d}) = {dqa_phi:.4f} ({dqa_time * 1e3:.1f} ms)")

        args_qae = dict(args_dqa)
        args_qae.update(
            {
                "m": args.qpe_qubits,
                "norm": norm,
                "oracle_circuit": exp.single_oracle_sin_inconstraint,
                "gateset": False,
            }
        )
        qae_circuit = QAE_Optimizer(args_qae).compile_qae_circuit()
        print(f"  QAE circuit: {qae_circuit.num_qubits} qubits, depth={qae_circuit.depth()}")

        start = time.perf_counter()
        if args.backend == "exact":
            b_counts = bno.execute_qae(qae_circuit.copy(), args.qpe_qubits)
        else:
            b_counts = bno.execute_qae(
                qae_circuit.copy(), args.qpe_qubits, num_meas=args.shots
            )
        qae_time = time.perf_counter() - start
        qae_phi = sum(
            np.sin(int(key, 2) * np.pi / 2**args.qpe_qubits) ** 2 * norm * prob
            for key, prob in b_counts.items()
        )
        print(f"  QAE phi(w_d={w_d}) = {qae_phi:.4f} ({qae_time * 1e3:.1f} ms)")
        results.append(
            {
                "n_y": n_y,
                "w_d": w_d,
                "dqa_phi": dqa_phi,
                "qae_phi": qae_phi,
                "dqa_time_s": dqa_time,
                "qae_time_s": qae_time,
            }
        )
    return results


def print_summary(results):
    print(f"\n{'n_y':>6} {'w_d':>5} {'DQA phi':>10} {'QAE phi':>10} {'DQA ms':>10} {'QAE ms':>10}")
    print(f"{'-' * 6} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for result in results:
        print(
            f"{result['n_y']:>6} {result['w_d']:>5} "
            f"{result['dqa_phi']:>10.4f} {result['qae_phi']:>10.4f} "
            f"{result['dqa_time_s'] * 1e3:>10.1f} {result['qae_time_s'] * 1e3:>10.1f}"
        )


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def make_plot(results, backend, shots, qpe_qubits, save_path, show):
    n_y_vals = [result["n_y"] for result in results]
    dqa_phis = [result["dqa_phi"] for result in results]
    qae_phis = [result["qae_phi"] for result in results]
    dqa_times = [result["dqa_time_s"] * 1e3 for result in results]
    qae_times = [result["qae_time_s"] * 1e3 for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(n_y_vals, dqa_phis, "-o", label="DQA phi", color="steelblue")
    axes[0].plot(n_y_vals, qae_phis, "-s", label="QAE phi", color="salmon")
    axes[0].set_xlabel("$n_y$ (turbine qubits)")
    axes[0].set_ylabel(r"$\phi(w_d)$")
    axes[0].set_title("Expected value vs qubit count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_y_vals, dqa_times, "-o", label="DQA wall time", color="steelblue")
    axes[1].plot(n_y_vals, qae_times, "-s", label="QAE wall time", color="salmon")
    axes[1].set_xlabel("$n_y$ (turbine qubits)")
    axes[1].set_ylabel("Wall time (ms)")
    axes[1].set_title("Wall time vs qubit count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    backend_label = (
        "Qiskit exact statevector"
        if backend == "exact"
        else f"Qiskit AerSimulator ({shots} shots)"
    )
    fig.suptitle(
        f"Qiskit $n_y$ sweep — {backend_label}, {qpe_qubits} QPE qubits",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    results = run_sweep(args)
    if not results:
        print("No sweep results were produced.")
        return
    print_summary(results)
    if args.save_results:
        save_results(results, args.save_results)
    if args.save_plot or args.show_plot:
        make_plot(
            results,
            args.backend,
            args.shots,
            args.qpe_qubits,
            args.save_plot,
            args.show_plot,
        )


if __name__ == "__main__":
    main()
