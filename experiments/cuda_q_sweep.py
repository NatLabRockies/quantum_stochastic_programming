"""Run a CUDA-Q DQA/QAE sweep over the number of wind-turbine qubits."""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
QISKIT_IMPL_DIR = REPO_ROOT / "qiskit_impl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QISKIT_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(QISKIT_IMPL_DIR))

from cudaq_impl import CudaqQAEOptimizer
import cudaq


def parse_args():
    default_target = os.getenv("CUDAQ_TARGET_MODE", "nvidia").strip().lower()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-y",
        type=int,
        nargs="+",
        default=list(range(4, 32, 2)),
        metavar="N",
        help="wind-turbine qubit counts to run",
    )
    parser.add_argument(
        "--target",
        choices=("nvidia", "mgpu", "qpp-cpu"),
        default=default_target,
        help="CUDA-Q simulator target (or set CUDAQ_TARGET_MODE)",
    )
    parser.add_argument("--shots", type=int, default=2**14, help="shots per estimate")
    parser.add_argument(
        "--qpe-qubits",
        type=int,
        default=5,
        help="QPE readout qubits when --qae is enabled",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="DQA layers per problem size (default: n_y)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="optimize each DQA angle schedule with COBYLA instead of using a linear ramp",
    )
    parser.add_argument(
        "--qae",
        action="store_true",
        help="also run QAE (can be much more expensive than DQA)",
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
    if args.target == "mgpu":
        cudaq.set_target("nvidia", option="mgpu")
    else:
        cudaq.set_target(args.target)
    print(f"[cuda-q] target: {cudaq.get_target().name}")

    results = []
    c_x = [3.0]
    c_r = 10.0
    x0 = [2]

    for n_y in args.n_y:
        print(f"\n{'=' * 60}\nn_y = {n_y}\n{'=' * 60}")
        d = n_y
        c_y = list(np.linspace(0.1, 1.0, n_y))
        timesteps = args.timesteps or n_y
        w_d = int(d - sum(x0))
        norm = w_d * c_r

        print(f"  c_y={[round(v, 2) for v in c_y]}, w_d={w_d}, norm={norm:.1f}")
        print(f"  Qubits: DQA={2 * n_y}, QAE={args.qpe_qubits + 2 * n_y + 1}")
        optimizer = CudaqQAEOptimizer(
            c_x=c_x,
            c_y=c_y,
            c_r=c_r,
            n_y=n_y,
            w_d=w_d,
            norm=norm,
        )
        theta = [
            value
            for t in range(timesteps)
            for value in (t / timesteps, (1 - t / timesteps) / math.pi)
        ]
        if args.optimize:
            print(f"  Optimizing {len(theta)} angles with COBYLA...")
            optimization = minimize(
                lambda angles: optimizer.estimate_expected_value_sv(list(angles), w_d),
                theta,
                method="COBYLA",
                options={"maxiter": 500, "rhobeg": 0.5, "disp": False},
            )
            theta = list(optimization.x)
            print(
                f"  COBYLA done | phi={optimization.fun:.4f} "
                f"| {optimization.nfev} evaluations"
            )
        else:
            print(f"  Linear ramp ({len(theta)} angles)")

        dqa_phi = dqa_time = qae_phi = qae_time = None
        start = time.perf_counter()
        dqa_phi = optimizer.estimate_expected_value(theta, w_d, shots=args.shots)
        dqa_time = time.perf_counter() - start
        print(f"  DQA phi(w_d={w_d}) = {dqa_phi:.4f} ({dqa_time * 1e3:.1f} ms)")

        if args.qae:
            start = time.perf_counter()
            qae_phi = optimizer.estimate_expected_value_qae(
                theta, m=args.qpe_qubits, shots=args.shots
            )
            qae_time = time.perf_counter() - start
            print(f"  QAE phi(w_d={w_d}) = {qae_phi:.4f} ({qae_time * 1e3:.1f} ms)")

        results.append(
            {
                "n_y": n_y,
                "w_d": w_d,
                "dqa_phi": dqa_phi,
                "dqa_time_s": dqa_time,
                "qae_phi": qae_phi,
                "qae_time_s": qae_time,
            }
        )
    return results


def print_summary(results, run_qae):
    columns = ["n_y", "w_d", "DQA phi", "DQA ms"]
    if run_qae:
        columns.extend(("QAE phi", "QAE ms"))
    print("\n" + " ".join(f"{column:>10}" for column in columns))
    print("-" * (11 * len(columns)))
    for result in results:
        row = [
            f"{result['n_y']:>10}",
            f"{result['w_d']:>10}",
            f"{result['dqa_phi']:>10.4f}",
            f"{result['dqa_time_s'] * 1e3:>10.1f}",
        ]
        if run_qae:
            row.extend(
                (
                    f"{result['qae_phi']:>10.4f}",
                    f"{result['qae_time_s'] * 1e3:>10.1f}",
                )
            )
        print(" ".join(row))


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("n_y", "w_d", "dqa_phi", "dqa_time_s", "qae_phi", "qae_time_s")
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def make_plot(results, args):
    n_y_vals = [result["n_y"] for result in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(
        n_y_vals, [result["dqa_phi"] for result in results],
        "-o", label="DQA phi", color="steelblue",
    )
    if args.qae:
        axes[0].plot(
            n_y_vals, [result["qae_phi"] for result in results],
            "-s", label="QAE phi", color="salmon",
        )
    axes[0].set_xlabel("$n_y$ (turbine qubits)")
    axes[0].set_ylabel(r"$\phi(w_d)$")
    axes[0].set_title("Expected value vs qubit count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        n_y_vals, [result["dqa_time_s"] * 1e3 for result in results],
        "-o", label="DQA wall time", color="steelblue",
    )
    if args.qae:
        axes[1].plot(
            n_y_vals, [result["qae_time_s"] * 1e3 for result in results],
            "-s", label="QAE wall time", color="salmon",
        )
    axes[1].set_xlabel("$n_y$ (turbine qubits)")
    axes[1].set_ylabel("Wall time (ms)")
    axes[1].set_title("Wall time vs qubit count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"CUDA-Q $n_y$ sweep ({args.target} target, {args.shots} shots)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    if args.save_plot:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_plot, dpi=150)
        print(f"Plot saved to {args.save_plot}")
    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    results = run_sweep(args)
    if not results:
        print("No sweep results were produced.")
        return
    print_summary(results, args.qae)
    if args.save_results:
        save_results(results, args.save_results)
    if args.save_plot or args.show_plot:
        make_plot(results, args)


if __name__ == "__main__":
    main()
