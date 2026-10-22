# simple_ed_cuda_q.py
#
# Single-instance CUDA-Q run for a fixed n_y.
# Runs DQA + QAE and compares both to the classical brute-force optimum.

import os, sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qiskit_impl')
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

from qae import *
from binary_optimizer import BinaryNestedOptimizer
import cudaq
from cudaq_impl import CudaqQAEOptimizer
from simple_ed_helper import TwoStageOptHelper

# ── CUDA-Q TARGET ──────────────────────────────────────────────────────────────
CUDAQ_TARGET_MODE = os.getenv('CUDAQ_TARGET_MODE', 'qpp-cpu').strip().lower()

try:
    if CUDAQ_TARGET_MODE == 'mgpu':
        cudaq.set_target('nvidia', option='mgpu')
        print("[cuda-q] target set to 'nvidia' (mgpu).")
    else:
        cudaq.set_target('nvidia')
        print("[cuda-q] target set to 'nvidia' (single GPU).")
except Exception as e:
    print(f"WARNING: nvidia target unavailable ({e}), falling back to qpp-cpu.")
    cudaq.set_target('qpp-cpu')

# ── PARAMETERS ─────────────────────────────────────────────────────────────────
# Quantum circuits parameters
N_SHOTS     = 2**16  # DQA shot count (only used when USE_STATEVECTOR=False)
USE_STATEVECTOR = False  # True:  exact statevector probabilities (no shot noise)
                        # False: shot-based sampling with N_SHOTS shots
m_qpe    = 5         # QPE readout qubits; estimation error ~ π·norm/2^m_qpe

# Fixed first-stage parameters
n_x  = 1
c_x  = [4.0]
x0   = [6]   # first-stage gas commitment
# n_x  = 2
# c_x  = [4., 5.]
# x0   = [5., 1.]

# Second-stage parameters
n_y    = 4         # number of wind turbine qubits
n_xi   = n_y
d      = 8
# c_y    = list(np.linspace(0.1, 1.0, n_y))
c_y = [[c0, c1] for c0, c1 in zip(np.linspace(2.0, 3.0, n_y), 1e-3*np.ones(n_y))]
c_r    = 10.0

# Derived quantities
c_y_eff = [sum(c_y[i]) for i in range(n_y)]
timesteps = max(10, n_y * n_y)
w_d       = int(d - sum(x0))     # wind demand
norm      = w_d * c_r            # QAE amplitude normalisation
# cost_norm = w_d * c_r / n_y     # DQA cost operator normalisation

x_min = d - n_y

assert w_d > 0, f"w_d={w_d} <= 0: x0={x0} covers full demand, nothing for wind turbines to do."

pdf = {
    tuple(int(v) for v in ('{0:0' + str(n_y) + 'b}').format(i)): 1 / 2**n_y
    for i in range(2**n_y)
}

n_qubits_dqa = 2 * n_y
n_qubits_qae = m_qpe + 2 * n_y + 1
# print(f"n_y={n_y}, w_d={w_d}, c_y={[round(v,2) for v in c_y]}, norm={norm:.1f}")
print(f"n_y={n_y}, w_d={w_d}, m_qpe={m_qpe}, norm={norm:.1f}")
print(f"Qubits: DQA={n_qubits_dqa}, QAE={n_qubits_qae}")

# ── CLASSICAL BASELINE ─────────────────────────────────────────────────────────
bno          = BinaryNestedOptimizer(c_x, c_y_eff, c_r, pdf, d, is_uniform=True)
exp_vals     = bno.brute_force_wind_demand_expectation_values()
classical_phi = exp_vals[w_d]
obj_surface  = [bno.gas_costs[0] * x + exp_vals[d - x] for x in range(x_min, d + 1)]

print(f"\nClassical phi(wind_demand): {[round(v, 3) for v in exp_vals]}")
print(f"Classical phi(w_d={w_d}): {classical_phi:.4f}")

# ── DQA ANGLE SETUP ────────────────────────────────────────────────────────────
cudaq_opt = CudaqQAEOptimizer(
    c_x=c_x, c_y=c_y, c_r=c_r,
    n_y=n_y, w_d=w_d, norm=norm,
)

theta0 = []
for t in range(timesteps):
    theta0.append(float(t / timesteps))
    theta0.append((1 - float(t / timesteps)) / math.pi)

Theta = theta0
dqa_phi = 0.0
qae_phi = 0.0

# # ── DQA EXECUTION ──────────────────────────────────────────────────────────────
# print(f"\nRunning DQA ({'statevector' if USE_STATEVECTOR else f'{N_SHOTS} shots'}) …")
# t0 = time.perf_counter()
# if USE_STATEVECTOR:
#     dqa_phi = cudaq_opt.estimate_expected_value_sv(Theta, w_d)
# else:
#     dqa_phi = cudaq_opt.estimate_expected_value(Theta, w_d, shots=N_SHOTS)
# dqa_time = time.perf_counter() - t0
# dqa_mode = 'sv' if USE_STATEVECTOR else f'{N_SHOTS} shots'
# print(f"DQA  phi(w_d={w_d}) = {dqa_phi:.4f}  ({dqa_time*1e3:.1f} ms, {dqa_mode})")

# t_grad = time.perf_counter()
# dqa_grad = cudaq_opt.estimate_expected_gradient(Theta, w_d, shots=N_SHOTS)
# t_grad = time.perf_counter() - t_grad
# print(f"DQA  grad(w_d={w_d}) = {[round(g, 4) for g in dqa_grad]}  ({t_grad*1e3:.1f} ms)")

# # Classical check 1: exact statevector gradient (no shot noise)
# t_grad_sv = time.perf_counter()
# dqa_grad_sv = cudaq_opt.estimate_expected_gradient_sv(Theta, w_d)
# t_grad_sv = time.perf_counter() - t_grad_sv
# print(f"DQA  grad_sv(w_d={w_d}) = {[round(g, 4) for g in dqa_grad_sv]}  ({t_grad_sv*1e3:.1f} ms, exact SV)")

# # # Classical check 2: analytical bound under uniform xi (linear costs only)
# # # E_xi[dQ/dy_j] = 0.5*c_y_eff[j] + 0.5*c_r  (independent of DQA angles)
# # analytical_grad = [0.5 * c_y_eff[j] + 0.5 * c_r for j in range(n_y)]
# # print(f"Analytical grad (uniform xi) = {[round(g, 4) for g in analytical_grad]}")

# # ── QAE EXECUTION ──────────────────────────────────────────────────────────────
# print(f"\nRunning QAE ({'statevector' if USE_STATEVECTOR else f'{N_SHOTS} shots'}) …")
# t0        = time.perf_counter()
# qae_shots = None if USE_STATEVECTOR else N_SHOTS
# qae_phi   = cudaq_opt.estimate_expected_value_qae(Theta, m=m_qpe, shots=qae_shots)
# qae_time  = time.perf_counter() - t0
# qae_mode  = 'sv' if USE_STATEVECTOR else f'{N_SHOTS} shots'
# print(f"QAE  phi(w_d={w_d}) = {qae_phi:.4f}  ({qae_time*1e3:.1f} ms, {qae_mode})")

# # ── SUMMARY ────────────────────────────────────────────────────────────────────
# print(f"\n{'='*50}")
# print(f"{'Method':<20} {'phi(w_d)':>10} {'time (ms)':>12}")
# print(f"{'-'*50}")
# print(f"{'Classical':<20} {classical_phi:>10.4f} {'—':>12}")
# print(f"{'DQA (CUDA-Q)':<20} {dqa_phi:>10.4f} {dqa_time*1e3:>12.1f}")
# print(f"{'QAE (CUDA-Q)':<20} {qae_phi:>10.4f} {qae_time*1e3:>12.1f}")
# print(f"{'='*50}")

# ── SCIPY OPTIMIZATION: optimal x over demand levels and methods ──────────────
# Two-stage objective (mirrors poster_example_jump.ipynb JuMP/Ipopt model):
#   o(x, d) = c_x[0]*x + [c_x[1]*x^2]  +  phi_dqa(floor(d - x))
# x is treated as a continuous first-stage variable; w_d = floor(d - x) maps
# it to the nearest integer wind demand for the Dicke-state DQA circuit.
# The phi cache avoids re-running the same circuit for repeated w_d values.
from scipy.optimize import minimize, minimize_scalar

# ── Sweep configuration ────────────────────────────────────────────────────────
# Extend D_VALUES to sweep demand levels, e.g. list(range(6, 11))
# D_VALUES    = [d]                               # extend to sweep, e.g. list(range(6, 11))
# D_VALUES    = list(range(1,13))
D_VALUES    = list(np.arange(1.0, 12.5, 0.5))
OPT_METHODS = ['bounded', 'COBYLA', 'Nelder-Mead', 'L-BFGS-B', 'BFGS']
# bounded     — minimize_scalar with Brent (1-D, tolerates non-smooth objectives)
# COBYLA      — derivative-free, handles inequality constraints directly
# Nelder-Mead — derivative-free simplex (bounds enforced by clipping)
# L-BFGS-B    — gradient-based with box bounds; requires linear or hermite mode
# BFGS        — gradient-based, unconstrained; result clipped to [x_lo, x_hi]

# Interpolation mode for phi(w_d) between integer grid points:
#   'floor'    — piecewise-constant (original behaviour, phi = phi_cache[floor(w_d)])
#   'linear'   — linear interpolation between adjacent integer knots
#   'hermite'  — cubic Hermite spline using phi_cache values + phi_grad_cache derivatives
#   'parabola' — global quadratic fit (least-squares) to all cached values + gradients
#   'cubic'    — global cubic fit    (least-squares) to all cached values + gradients
#   'quartic'  — global quartic fit  (least-squares) to all cached values + gradients
#   'sextic'   — global sextic fit   (least-squares) to all cached values + gradients
INTERP_MODE = 'quartic'
SWEEP_INTERP_MODES = ['linear', 'hermite', 'quartic']   # modes to compare in the sweep table

# ── Pre-compute phi_dqa(w_d) for all feasible wind demands ────────────────────
print("\nPre-computing DQA phi(w_d) cache …")
_th_c = []
for _ts in range(timesteps):
    _th_c.append(float(_ts / timesteps))
    _th_c.append((1 - float(_ts / timesteps)) / math.pi)
phi_cache: dict = {0: 0.0}   # w_d=0: all demand met by gas, no second-stage cost
# At w_d=0 the Dicke state is |0...0>, so y[j]=0 for all j.
# dQ/dy_j|_{y=0} = c0_j (if xi[j]=1) or c_r (if xi[j]=0), where c0_j is the
# linear cost coefficient.  The quadratic term 2*c1*y[j] vanishes at y[j]=0.
# With uniform xi: E[dQ/dy_j|y=0] = 0.5*(c0_j + c_r).
_c0 = [c_y[j][0] if isinstance(c_y[j], list) else c_y[j] for j in range(n_y)]
phi_grad_cache: dict = {0: float(np.mean([0.5 * (_c0[j] + c_r) for j in range(n_y)]))}
# phi_grad_cache: dict = {}
print(f"  grad(w_d=0) = {phi_grad_cache[0]:.4f}  (analytical, uniform xi)")
for _wd in range(1, n_y + 1):
    _tc = time.perf_counter()
    _opt_c = CudaqQAEOptimizer(c_x=c_x, c_y=c_y, c_r=c_r,
                                n_y=n_y, w_d=_wd, norm=_wd * c_r)

    # phi_cache[_wd] = _opt_c.estimate_expected_value(_th_c, _wd, shots=N_SHOTS)
    phi_cache[_wd] = _opt_c.estimate_expected_value_sv(_th_c, _wd)
    print(f"  phi(w_d={_wd}) = {phi_cache[_wd]:.4f}  "
        f"({(time.perf_counter()-_tc)*1e3:.1f} ms)")

    _tg = time.perf_counter()
    # _grad = _opt_c.estimate_expected_gradient(_th_c, _wd, shots=N_SHOTS)
    _grad = _opt_c.estimate_expected_gradient_sv(_th_c, _wd)
    phi_grad_cache[_wd] = np.mean(_grad)
    print(f"  grad(w_d={_wd}) = {phi_grad_cache[_wd]:.4f}  "
          f"({(time.perf_counter()-_tg)*1e3:.1f} ms)")


# ── Sweep over interpolation modes, demand levels, and optimization methods ───
print(f"\n{'interp':<10} {'d':<5} {'method':<14} {'x*':>8} {'w_d*':>5} {'fs_obj':>10} "
      f"{'ss_obj':>10} {'obj*':>10} {'nfev':>6} {'nit':>5} {'conv':>5} {'ms':>8}")
print('-' * 103)

sweep_results = []
for _sweep_mode in SWEEP_INTERP_MODES:
    _sh            = TwoStageOptHelper(c_x, n_y, _sweep_mode, phi_cache, phi_grad_cache)
    _qobj          = _sh.quantum_obj
    _qobj_grad     = _sh.quantum_obj_and_grad
    _phi_sw        = _sh.phi_interp

    for d_val in D_VALUES:
        _xlo = float(max(0, d_val - n_y))
        _xhi = float(d_val)
        _x0  = (_xlo + _xhi) / 2.0
        # _x0 = _xlo
        # _x0 = _xhi

        # Classical reference for this demand level (use floor for integer-indexed lookups)
        _d_int  = int(math.floor(d_val))
        _pdf_d  = {tuple(int(v) for v in ('{0:0'+str(n_y)+'b}').format(i)): 1/2**n_y
                   for i in range(2**n_y)}
        _bno_d  = BinaryNestedOptimizer(c_x, c_y_eff, c_r, _pdf_d, _d_int, is_uniform=True)
        _exp_d  = _bno_d.brute_force_wind_demand_expectation_values()
        _xr_d   = list(range(int(_xlo), _d_int + 1))
        _obj_cl = [_sh.fs_cost(float(xv)) + _exp_d[_d_int - xv] for xv in _xr_d]
        cl_x_s  = _xr_d[int(np.argmin(_obj_cl))]
        cl_obj_s = min(_obj_cl)

        for method in OPT_METHODS:
            _t0m = time.perf_counter()
            try:
                if method == 'bounded':
                    res = minimize_scalar(
                        lambda xv: _qobj(float(xv), d_val),
                        bounds=(_xlo, _xhi),
                        method='bounded',
                        options={'xatol': 1e-6},
                    )
                    x_s, obj_s, ok = float(res.x), float(res.fun), res.success
                    nfev = getattr(res, 'nfev', None)
                    nit  = None   # minimize_scalar has no nit
                elif method == 'COBYLA':
                    res = minimize(
                        lambda xv: _qobj(float(xv[0]), d_val),
                        [_x0],
                        method='COBYLA',
                        constraints=[
                            {'type': 'ineq', 'fun': lambda xv: xv[0] - _xlo},
                            {'type': 'ineq', 'fun': lambda xv: _xhi - xv[0]},
                        ],
                        options={'maxiter': 500, 'rhobeg': 0.5},
                    )
                    x_s, obj_s, ok = float(res.x[0]), float(res.fun), res.success
                    nfev = getattr(res, 'nfev', None)
                    nit  = getattr(res, 'nit', None)
                elif method == 'Nelder-Mead':
                    res = minimize(
                        lambda xv: _qobj(
                            float(np.clip(xv[0], _xlo, _xhi)), d_val),
                        [_x0],
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8},
                    )
                    x_s  = float(np.clip(res.x[0], _xlo, _xhi))
                    obj_s, ok = float(res.fun), res.success
                    nfev = getattr(res, 'nfev', None)
                    nit  = getattr(res, 'nit', None)
                elif method == 'L-BFGS-B':
                    if _sweep_mode == 'floor':
                        raise ValueError(
                            "L-BFGS-B requires a smooth objective; "
                            "set INTERP_MODE to 'linear', 'hermite', 'parabola', 'cubic', 'quartic', or 'sextic'.")
                    res = minimize(
                        lambda xv: _qobj_grad(float(xv[0]), d_val),
                        [_x0],
                        method='L-BFGS-B',
                        jac=True,
                        bounds=[(_xlo, _xhi)],
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8},
                    )
                    x_s, obj_s, ok = float(res.x[0]), float(res.fun), res.success
                    nfev = getattr(res, 'nfev', None)
                    nit  = getattr(res, 'nit', None)
                elif method == 'BFGS':
                    if _sweep_mode == 'floor':
                        raise ValueError(
                            "BFGS requires a smooth objective; "
                            "set INTERP_MODE to 'linear', 'hermite', 'parabola', 'cubic', 'quartic', or 'sextic'.")
                    res = minimize(
                        lambda xv: _qobj_grad(
                            float(np.clip(xv[0], _xlo, _xhi)), d_val),
                        [_x0],
                        method='BFGS',
                        jac=True,
                        options={'maxiter': 500, 'gtol': 1e-8},
                    )
                    x_s  = float(np.clip(res.x[0], _xlo, _xhi))
                    obj_s, ok = float(res.fun), res.success
                    nfev = getattr(res, 'nfev', None)
                    nit  = getattr(res, 'nit', None)
                else:
                    raise ValueError(f"Unknown method: {method!r}")

                wd_s   = d_val - x_s
                fs_s   = _sh.fs_cost(x_s)
                ss_s   = _phi_sw(wd_s)
                t_ms = (time.perf_counter() - _t0m) * 1e3
                sweep_results.append(dict(interp_mode=_sweep_mode,
                                          d=d_val, method=method, x_star=x_s,
                                          wd_star=wd_s, fs_obj=fs_s, ss_obj=ss_s,
                                          obj_star=obj_s,
                                          nfev=nfev, nit=nit,
                                          time_ms=t_ms, success=ok))
                _nfev_s = str(nfev) if nfev is not None else '—'
                _nit_s  = str(nit)  if nit  is not None else '—'
                _conv_s = 'yes' if ok else 'NO'
                print(f"{_sweep_mode:<10} {d_val:<5} {method:<14} {x_s:>8.3f} {wd_s:>5.3f} "
                      f"{fs_s:>10.4f} {ss_s:>10.4f} {obj_s:>10.4f} "
                      f"{_nfev_s:>6} {_nit_s:>5} {_conv_s:>5} {t_ms:>8.1f}")
            except Exception as exc:
                print(f"{_sweep_mode:<10} {d_val:<5} {method:<14} ERROR: {exc}")

# ── SAVE SWEEP RESULTS ────────────────────────────────────────────────────────
import csv as _csv
_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_ed_cuda_q_sweep.csv')
if sweep_results:
    _fieldnames = list(sweep_results[0].keys())
    with open(_csv_path, 'w', newline='') as _f:
        _w = _csv.DictWriter(_f, fieldnames=_fieldnames)
        _w.writeheader()
        _w.writerows(sweep_results)
    print(f"Sweep results saved to {_csv_path}")

# Instantiate helper for plot generation (uses INTERP_MODE)
_helper     = TwoStageOptHelper(c_x, n_y, INTERP_MODE, phi_cache, phi_grad_cache)
_fs_cost    = _helper.fs_cost
_phi_interp = _helper.phi_interp

# ── PLOTS ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: bar chart comparing classical, DQA, QAE
ax = axes[0]
labels = ['Classical', f'DQA\n(CUDA-Q)', f'QAE\n(CUDA-Q)']
values = [classical_phi, dqa_phi, qae_phi]
colors = ['#4c72b0', '#dd8452', '#55a868']
bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
ax.axhline(classical_phi, color='#4c72b0', linestyle='--', linewidth=1, alpha=0.6, label='Classical optimum')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel(r'$\phi(w_d)$')
ax.set_title(f'Expected value comparison ($w_d={w_d}$, $n_y={n_y}$)')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

# Right: classical vs quantum objective surface o(x) with scipy optima
ax2 = axes[1]
x_vals_plot = list(range(x_min, d + 1))
ax2.plot(x_vals_plot, obj_surface, '-o', color='#4c72b0', label='Classical $o(x)$')
q_obj_plot = [_fs_cost(float(xv)) + phi_cache.get(max(0, min(d - xv, n_y)), 0)
              for xv in x_vals_plot]
ax2.plot(x_vals_plot, q_obj_plot, '-s', color='#dd8452', label='Quantum DQA $o(x)$ (floor)')
# Smooth interpolated surface for d = 8 (all modes except floor)
if INTERP_MODE != 'floor' and d in D_VALUES:
    _x_dense = np.linspace(x_min, d, 300)
    _obj_smooth = [_fs_cost(float(xv)) + _phi_interp(d - float(xv))
                   for xv in _x_dense]
    _smooth_colors = {'hermite': '#2ca02c', 'parabola': '#9467bd',
                      'cubic': '#8c564b', 'quartic': '#d62728', 'sextic': '#e377c2'}
    ax2.plot(_x_dense, _obj_smooth, '-',
             color=_smooth_colors.get(INTERP_MODE, 'gray'), linewidth=1.5,
             label=f'Quantum DQA $o(x)$ ({INTERP_MODE})')
# Mark scipy optima for the current demand level d (INTERP_MODE results only)
_method_colors = {'bounded': '#e377c2', 'COBYLA': '#8c564b',
                  'Nelder-Mead': '#17becf', 'L-BFGS-B': '#9467bd', 'BFGS': '#bcbd22'}
for _r in [r for r in sweep_results if r['d'] == d and r['interp_mode'] == INTERP_MODE]:
    ax2.axvline(_r['x_star'], linestyle=':', alpha=0.8,
                color=_method_colors.get(_r['method'], 'gray'),
                label=f"{_r['method']} $x^*$={_r['x_star']:.2f}")
ax2.set_xlabel('Gas commitment $x$')
ax2.set_ylabel('Objective $o(x)$')
ax2.set_title(f'Objective surface ($d={d}$): classical vs quantum')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.suptitle(f'CUDA-Q single run — $n_y={n_y}$, nvidia target', fontsize=12, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_ed_cuda_q_results.png')
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Plot saved to {out_png}")






