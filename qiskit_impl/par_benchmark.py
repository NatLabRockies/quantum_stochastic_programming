"""
Extended parallelisation benchmark: n_y = 4..20 (step 2).
Measures 1-GPU vs 2-GPU mqpu noisy sampling wall time.

Uses N_SHOTS=256 (timing study only — physics accuracy not the goal here).
Skips any n_y that exceeds TIMEOUT_S seconds on 1-GPU.

Run on an allocated node with 2 GPUs:
  /nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/bin/python par_benchmark.py
"""
import os, sys, math, time, json
_QISKIT_SP   = '/nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/lib/python3.11/site-packages'
_QISKIT_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'
for _p in [_QISKIT_SP, _QISKIT_IMPL]:
    if _p not in sys.path: sys.path.insert(0, _p)
os.chdir(_QISKIT_IMPL)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cudaq
from cudaq_impl import CudaqQAEOptimizer, build_depolarizing_noise_model

N_SHOTS   = 256
TIMESTEPS = 20
P1, P2    = 0.0001, 0.001
C_R, C_X  = 10.0, [3.]
NY_LIST   = list(range(4, 22, 2))   # [4,6,8,10,12,14,16,18,20]
TIMEOUT_S = 300                      # skip if 1-GPU takes longer

noise_model = build_depolarizing_noise_model(p1=P1, p2=P2)

def make_opt(ny, nm):
    c_y  = list(np.linspace(0.1, 1.0, ny))
    w_d  = ny - 2
    cost_norm = w_d * C_R / ny
    theta0 = []
    for t in range(TIMESTEPS):
        theta0.append(float(t / TIMESTEPS))
        theta0.append((1 - float(t / TIMESTEPS)) / math.pi)
    opt = CudaqQAEOptimizer(c_x=C_X, c_y=c_y, c_r=C_R, n_y=ny, w_d=w_d,
                            cost_norm=cost_norm, noise_model=nm)
    return opt, theta0

# ── 1-GPU baseline ─────────────────────────────────────────────────────────────
print(f'\n── 1-GPU baseline  (nvidia, N_SHOTS={N_SHOTS}) ──')
cudaq.set_target('nvidia')
t1_gpu = {}
for ny in NY_LIST:
    opt, theta0 = make_opt(ny, noise_model)
    t0 = time.perf_counter()
    try:
        opt.sample_ansatz(theta0, shots=N_SHOTS)
        dt = time.perf_counter() - t0
        t1_gpu[ny] = dt
        print(f'  n_y={ny:2d}  qubits={2*ny:2d}  {dt:.1f}s', flush=True)
        if dt > TIMEOUT_S:
            print(f'  → skipping n_y>{ny} (exceeded {TIMEOUT_S}s)')
            break
    except Exception as e:
        print(f'  n_y={ny:2d}  FAIL: {e}')
        break

# ── 2-GPU mqpu ─────────────────────────────────────────────────────────────────
print(f'\n── 2-GPU mqpu  (nvidia --mqpu, N_SHOTS={N_SHOTS}) ──')
cudaq.set_target('nvidia', option='mqpu')
N_QPUS = cudaq.num_available_gpus()
print(f'   Virtual QPUs: {N_QPUS}')
t2_gpu = {}
for ny in list(t1_gpu.keys()):   # only test sizes that succeeded on 1-GPU
    opt, theta0 = make_opt(ny, noise_model)
    t0 = time.perf_counter()
    try:
        opt.sample_ansatz_mqpu(theta0, total_shots=N_SHOTS, n_qpus=N_QPUS)
        dt = time.perf_counter() - t0
        t2_gpu[ny] = dt
        print(f'  n_y={ny:2d}  qubits={2*ny:2d}  {dt:.1f}s', flush=True)
    except Exception as e:
        print(f'  n_y={ny:2d}  FAIL: {e}')

# ── SAVE JSON ─────────────────────────────────────────────────────────────────
NY_done = sorted(set(t1_gpu) & set(t2_gpu))
results = {ny: {'t1_s': round(t1_gpu[ny], 2),
                't2_s': round(t2_gpu[ny], 2),
                'speedup': round(t1_gpu[ny]/t2_gpu[ny], 3),
                'efficiency_pct': round(t1_gpu[ny]/t2_gpu[ny]/N_QPUS*100, 1)}
           for ny in NY_done}
with open('par_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults → par_benchmark_results.json')

# ── 3-PANEL PLOT ──────────────────────────────────────────────────────────────
NY    = NY_done
t1    = np.array([t1_gpu[ny] for ny in NY])
t2    = np.array([t2_gpu[ny] for ny in NY])
su    = t1 / t2
eff   = su / N_QPUS * 100
qbits = [2*ny for ny in NY]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Wall time
ax = axes[0]
x, bw = np.arange(len(NY)), 0.32
b1 = ax.bar(x - bw/2, t1, width=bw, color='steelblue', alpha=0.85, label='1 GPU (baseline)')
b2 = ax.bar(x + bw/2, t2, width=bw, color='tomato',    alpha=0.85, label=f'{N_QPUS} GPU mqpu')
for rect, v in zip(list(b1)+list(b2), list(t1)+list(t2)):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.3,
            f'{v:.0f}s' if v >= 10 else f'{v:.1f}s',
            ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=8)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title(f'Noisy sampling wall time\n({N_SHOTS} shots, p₂={P2})', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

# Panel 2: Speedup
ax = axes[1]
colors = ['seagreen' if s >= 1 else 'tomato' for s in su]
bars = ax.bar(x, su, color=colors, alpha=0.85, width=0.5)
for rect, v in zip(bars, su):
    yoff = 0.02 if v >= 1 else -0.08
    va = 'bottom' if v >= 1 else 'top'
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+yoff,
            f'{v:.2f}×', ha='center', va=va, fontsize=9)
ax.axhline(1.0,     color='k',    lw=1.2, ls='--', label='No speedup (1×)')
ax.axhline(N_QPUS,  color='grey', lw=0.8, ls=':',  label=f'Ideal ({N_QPUS}×)')
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=8)
ax.set_ylabel('Speedup  $T_\\mathrm{1GPU} / T_\\mathrm{2GPU}$', fontsize=11)
ax.set_title(f'{N_QPUS}-GPU mqpu speedup\n(green = faster, red = slower)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

# Panel 3: Parallel efficiency
ax = axes[2]
colors_e = ['seagreen' if e >= 80 else ('gold' if e >= 50 else 'tomato') for e in eff]
bars = ax.bar(x, eff, color=colors_e, alpha=0.85, width=0.5)
for rect, v in zip(bars, eff):
    ax.text(rect.get_x()+rect.get_width()/2, max(rect.get_height()+1, 5),
            f'{v:.0f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(100, color='grey', lw=0.8, ls=':', label='Ideal 100%')
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=8)
ax.set_ylabel('Parallel efficiency  (%)', fontsize=11)
ax.set_title('Parallel efficiency\n(green≥80%, yellow 50–80%, red<50%)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(bottom=0)

fig.suptitle(
    f'CUDA-Q mqpu Shot Splitting: {N_QPUS}-GPU Parallelisation  ·  '
    f'TIMESTEPS=20  ·  {N_SHOTS} shots  ·  p₁={P1}, p₂={P2}  ·  H100',
    fontsize=11, y=1.01)

plt.tight_layout()
out = os.path.join(_QISKIT_IMPL, 'parallelisation_study.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Plot → {out}')

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'n_y':>4}  {'qubits':>6}  {'1-GPU(s)':>9}  {'2-GPU(s)':>9}  "
      f"{'Speedup':>8}  {'Efficiency':>11}")
print('─' * 58)
for ny in NY:
    r = results[ny]
    print(f"{ny:>4}  {2*ny:>6}  {r['t1_s']:>9.1f}  {r['t2_s']:>9.1f}  "
          f"{r['speedup']:>8.2f}×  {r['efficiency_pct']:>10.0f}%")
