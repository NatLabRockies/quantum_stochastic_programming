"""
Extended parallelisation report: n_y = 4..14 (step 2).
Data from par_benchmark.py run on 2x H100 (Kestrel), 256 shots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'

# Measured timings (256 shots, 2x H100, TIMESTEPS=20, p1=0.0001, p2=0.001)
DATA = {
    # ny: (t_1gpu_s, t_2gpu_s)
    4:  (2.2,   2.6),
    6:  (3.9,   5.4),
    8:  (6.5,  10.4),
    10: (46.9,  24.9),
    12: (69.2,  36.1),
    14: (808.1, 393.7),
}
N_SHOTS = 256
N_QPUS  = 2

NY    = sorted(DATA)
t1    = np.array([DATA[ny][0] for ny in NY])
t2    = np.array([DATA[ny][1] for ny in NY])
su    = t1 / t2
eff   = su / N_QPUS * 100

# 3-Panel figure
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
x, bw = np.arange(len(NY)), 0.32

# Panel 1: Wall time (log scale)
ax = axes[0]
b1 = ax.bar(x - bw/2, t1, width=bw, color='steelblue', alpha=0.85, label='1 GPU (baseline)')
b2 = ax.bar(x + bw/2, t2, width=bw, color='tomato',    alpha=0.85, label='2 GPU mqpu')
for rect, v in zip(list(b1)+list(b2), list(t1)+list(t2)):
    label = f'{v:.0f}s' if v >= 10 else f'{v:.1f}s'
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()*1.05,
            label, ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=9)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title(f'Noisy sampling wall time\n({N_SHOTS} shots, TIMESTEPS=20)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_yscale('log'); ax.set_ylim(bottom=0.5)

# Panel 2: Speedup
ax = axes[1]
colors = ['seagreen' if s >= 1 else 'tomato' for s in su]
bars = ax.bar(x, su, color=colors, alpha=0.85, width=0.5)
for rect, v in zip(bars, su):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.03,
            f'{v:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(1.0,    color='k',    lw=1.5, ls='--', label='Break-even (1x)')
ax.axhline(N_QPUS, color='grey', lw=1.0, ls=':',  label=f'Ideal ({N_QPUS}x)')
ax.axvspan(2.5, 3.5, alpha=0.08, color='gold', label='Crossover zone')
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=9)
ax.set_ylabel('Speedup  T_1GPU / T_2GPU', fontsize=11)
ax.set_title('2-GPU mqpu speedup\n(green = faster, red = slower)', fontsize=11)
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, 2.4)

# Panel 3: Parallel efficiency
ax = axes[2]
colors_e = ['seagreen' if e >= 80 else ('gold' if e >= 50 else 'tomato') for e in eff]
bars = ax.bar(x, eff, color=colors_e, alpha=0.85, width=0.5)
for rect, v in zip(bars, eff):
    ax.text(rect.get_x()+rect.get_width()/2, max(rect.get_height()+1, 6),
            f'{v:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(100, color='grey', lw=1.0, ls=':', label='Ideal 100%')
ax.axhline(80,  color='gold', lw=0.8, ls='--', alpha=0.6, label='80% threshold')
ax.set_xticks(x)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY], fontsize=9)
ax.set_ylabel('Parallel efficiency (%)', fontsize=11)
ax.set_title('Parallel efficiency\n(green>=80%, yellow 50-80%, red<50%)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, 120)

fig.suptitle(
    f'CUDA-Q mqpu Shot Splitting: 2-GPU Parallelisation  '
    f'TIMESTEPS=20  {N_SHOTS} shots  p1=0.0001, p2=0.001  H100 (Kestrel)',
    fontsize=11, y=1.02)

plt.tight_layout()
out = os.path.join(SCRATCH_IMPL, 'parallelisation_study.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {out}')

# Summary table
print(f"\n{'n_y':>4}  {'qubits':>6}  {'1-GPU(s)':>9}  {'2-GPU(s)':>9}  "
      f"{'Speedup':>8}  {'Efficiency':>11}  {'Overhead(s)':>12}")
print('-' * 72)
for ny in NY:
    t1s, t2s = DATA[ny]
    s = t1s / t2s
    e = s / N_QPUS * 100
    oh = t2s - t1s / N_QPUS
    print(f"{ny:>4}  {2*ny:>6}  {t1s:>9.1f}  {t2s:>9.1f}  "
          f"{s:>8.2f}x  {e:>10.0f}%  {oh:>12.1f}s")
