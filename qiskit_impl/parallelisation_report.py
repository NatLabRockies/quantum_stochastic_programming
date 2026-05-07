"""
Generate parallelisation report: timings, speedup, and efficiency plots.

Data sources
------------
- /tmp/timings_1gpu.json  : 1-GPU baseline timings (single cudaq.sample call)
- noise_study_results.json: 2-GPU mqpu timings embedded in the mqpu run log

Run on the allocated node after both baseline and mqpu runs are complete:
  /nopt/nrel/apps/gpu_stack/software/qiskit/aer-gpu/venv/bin/python parallelisation_report.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── DATA ──────────────────────────────────────────────────────────────────────
# 1-GPU baseline (from /tmp/timings_1gpu.json)
baseline = {
    4:  {'ideal_ms': 715,    'noisy_ms': 1482},
    6:  {'ideal_ms': 610,    'noisy_ms': 3578},
    8:  {'ideal_ms': 1060,   'noisy_ms': 6500},
    10: {'ideal_ms': 1712,   'noisy_ms': 127472},
}

# 2-GPU mqpu run (from tee /tmp/noise_study_mqpu.log)
mqpu_2gpu = {
    4:  {'ideal_ms': 718,   'noisy_ms': 3190},
    6:  {'ideal_ms': 600,   'noisy_ms': 6233},
    8:  {'ideal_ms': 1066,  'noisy_ms': 10678},
    10: {'ideal_ms': 1734,  'noisy_ms': 64755},
}

NY      = [4, 6, 8, 10]
N_SHOTS = 2048
N_QPUS  = 2

# Derived
t1   = np.array([baseline[ny]['noisy_ms']  for ny in NY])
t2   = np.array([mqpu_2gpu[ny]['noisy_ms'] for ny in NY])
ti1  = np.array([baseline[ny]['ideal_ms']  for ny in NY])
ti2  = np.array([mqpu_2gpu[ny]['ideal_ms'] for ny in NY])

speedup    = t1 / t2                        # >1 means mqpu is faster
efficiency = speedup / N_QPUS * 100         # % of ideal N_QPUS× speedup
overhead   = t2 - t1 / N_QPUS              # extra time vs perfect scaling (ms)

# Projected 4-GPU times (linear extrapolation from measured overhead)
# overhead ≈ t_dispatch + t_gather; treat as constant per ny
t4_proj    = t1 / 4 + overhead             # optimistic: overhead unchanged

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

x   = np.arange(len(NY))
bw  = 0.32
ny_labels = [f'$n_y={n}$' for n in NY]

# ── Panel 1: Absolute noisy eval wall time ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
b1 = ax.bar(x - bw/2, t1/1000,  width=bw, color='steelblue', alpha=0.85, label='1 GPU (baseline)')
b2 = ax.bar(x + bw/2, t2/1000,  width=bw, color='tomato',    alpha=0.85, label='2 GPU (mqpu)')
for rect, v in zip(list(b1)+list(b2), list(t1/1000)+list(t2/1000)):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.3,
            f'{v:.1f}s', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(ny_labels, fontsize=9)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title('Noisy eval wall time\n(2048 shots total)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

# ── Panel 2: Speedup ──────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
colors_su = ['tomato' if s < 1 else 'seagreen' for s in speedup]
bars = ax.bar(NY, speedup, color=colors_su, alpha=0.85, width=1.2)
for rect, v in zip(bars, speedup):
    ax.text(rect.get_x()+rect.get_width()/2,
            rect.get_height() + (0.02 if v >= 1 else -0.08),
            f'{v:.2f}×', ha='center',
            va='bottom' if v >= 1 else 'top', fontsize=9)
ax.axhline(1.0, color='k',      lw=1.2, ls='--', label='No speedup (1×)')
ax.axhline(N_QPUS, color='grey', lw=1.0, ls=':',  label=f'Ideal {N_QPUS}×')
ax.set_xlabel('$n_y$ (turbines)', fontsize=11)
ax.set_ylabel('Speedup  $T_1 / T_2$', fontsize=11)
ax.set_title(f'Speedup: 2-GPU mqpu vs 1-GPU\n(red = slowdown, green = speedup)', fontsize=11)
ax.set_xticks(NY); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

# ── Panel 3: Parallel efficiency ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
colors_eff = ['tomato' if e < 50 else ('gold' if e < 80 else 'seagreen') for e in efficiency]
bars = ax.bar(NY, efficiency, color=colors_eff, alpha=0.85, width=1.2)
for rect, v in zip(bars, efficiency):
    ypos = max(rect.get_height() + 1, 5)
    ax.text(rect.get_x()+rect.get_width()/2, ypos,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(100, color='grey', lw=1.0, ls=':', label='Ideal 100%')
ax.set_xlabel('$n_y$ (turbines)', fontsize=11)
ax.set_ylabel('Parallel efficiency  (%)\n= speedup / N_GPUs × 100', fontsize=10)
ax.set_title('Parallel efficiency\n(green≥80%, yellow 50–80%, red<50%)', fontsize=11)
ax.set_xticks(NY); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(bottom=0)

# ── Panel 4: Overhead breakdown ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
compute_time = t1 / N_QPUS              # ideal compute portion per GPU
overhead_abs = np.maximum(0, t2 - compute_time)  # all extra time = dispatch+gather
b1 = ax.bar(NY, compute_time/1000,  width=1.2, color='steelblue', alpha=0.85,
            label='Compute (ideal share)')
b2 = ax.bar(NY, overhead_abs/1000,  width=1.2, color='orange', alpha=0.85,
            label='Overhead (dispatch + gather)', bottom=compute_time/1000)
ax.set_xlabel('$n_y$ (turbines)', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
ax.set_title('2-GPU time decomposition\ncompute vs overhead', fontsize=11)
ax.set_xticks(NY); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
for i, ny in enumerate(NY):
    ov = overhead_abs[i]/1000
    ax.text(ny, (compute_time[i]+overhead_abs[i])/1000 + 0.2,
            f'+{ov:.1f}s', ha='center', va='bottom', fontsize=8, color='saddlebrown')

# ── Panel 5: Projected 4-GPU times ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
b1 = ax.bar(x - bw,   t1/1000,      width=bw, color='steelblue', alpha=0.85, label='1 GPU (measured)')
b2 = ax.bar(x,        t2/1000,      width=bw, color='tomato',    alpha=0.85, label='2 GPU (measured)')
b3 = ax.bar(x + bw,   t4_proj/1000, width=bw, color='seagreen',  alpha=0.85,
            label='4 GPU (projected)', hatch='//')
for rect, v in zip(list(b3), t4_proj/1000):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.3,
            f'~{v:.0f}s', ha='center', va='bottom', fontsize=7.5, color='darkgreen')
ax.set_xticks(x); ax.set_xticklabels(ny_labels, fontsize=9)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title('Projected 4-GPU wall time\n(dashed = projected, overhead assumed constant)', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.4)

# ── Panel 6: Shots per second per n_y ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
sps_1 = N_SHOTS / (t1 / 1000)
sps_2 = N_SHOTS / (t2 / 1000)
b1 = ax.bar(x - bw/2, sps_1, width=bw, color='steelblue', alpha=0.85, label='1 GPU')
b2 = ax.bar(x + bw/2, sps_2, width=bw, color='tomato',    alpha=0.85, label='2 GPU (mqpu)')
ax.set_xticks(x); ax.set_xticklabels(ny_labels, fontsize=9)
ax.set_ylabel('Shots per second', fontsize=11)
ax.set_title('Noisy simulation throughput\n(shots / second)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

fig.suptitle(
    f'CUDA-Q mqpu Shot Splitting: Parallelisation Study\n'
    f'DQA circuit · TIMESTEPS=20 · {N_SHOTS} shots · p₁={0.0001} p₂={0.001} · '
    f'2× H100 (single node)',
    fontsize=12, y=1.01)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'parallelisation_study.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Plot saved → {out}')

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print()
print(f"{'n_y':>4}  {'T_1gpu(s)':>10}  {'T_2gpu(s)':>10}  {'Speedup':>8}  "
      f"{'Efficiency':>11}  {'Overhead(s)':>12}  {'T_4gpu_proj(s)':>15}")
print('─' * 80)
for i, ny in enumerate(NY):
    print(f"{ny:>4}  {t1[i]/1000:>10.1f}  {t2[i]/1000:>10.1f}  {speedup[i]:>8.2f}×  "
          f"{efficiency[i]:>10.0f}%  {overhead_abs[i]/1000:>12.1f}  "
          f"{t4_proj[i]/1000:>15.0f}")

print()
print('Key observations:')
for i, ny in enumerate(NY):
    if speedup[i] < 1:
        print(f'  n_y={ny:2d}: SLOWDOWN {1/speedup[i]:.2f}× — overhead ({overhead_abs[i]/1000:.1f}s) '
              f'> compute gain ({(t1[i]-t2[i])/1000:.1f}s saved)')
    else:
        print(f'  n_y={ny:2d}: SPEEDUP  {speedup[i]:.2f}× — noisy eval dominated by circuit cost, '
              f'overhead ({overhead_abs[i]/1000:.1f}s) small relative to compute')
