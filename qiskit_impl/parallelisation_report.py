"""
Parallelisation report: 1-GPU vs 2-GPU (single-node mqpu) vs 4-GPU (multi-node mpi4py).
Data from par_benchmark.py, par_benchmark_2gpu.py, par_benchmark_mpi.py on Kestrel H100s.
All runs: 256 shots, TIMESTEPS=20, p1=0.0001, p2=0.001.

Memory limit: n_y>=16 (>=32 qubits) needs >=68.7 GB/trajectory -> OOM on H100 (80 GB).
Practical ceiling: n_y=14 (28 qubits, 4.3 GB/trajectory).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH_IMPL = '/kfs3/scratch/nsawant/quantum_stochastic_programming/qiskit_impl'

# ny -> (t_1gpu, t_2gpu, t_4gpu)  None = not run / OOM
# n_y=4-12: 1-GPU & 2-GPU from par_benchmark.py (warm GPU, nvidia mqpu target)
# n_y=14: 1-GPU from par_benchmark.py (warm); 2-GPU/4-GPU cold-start reruns
# n_y>=16: OOM on H100 (68.7 GB/trajectory for 32 qubits)
DATA = {
    4:  (2.2,   2.6,   None),
    6:  (3.9,   5.4,   None),
    8:  (6.5,  10.4,   None),
    10: (46.9,  24.9,  12.0),
    12: (69.2,  36.1,  19.3),
    14: (808.1, 283.2, 144.3),
}
N_SHOTS = 256

NY_ALL = sorted(DATA)
NY_MPI = [ny for ny in NY_ALL if DATA[ny][2] is not None]

t1_all = np.array([DATA[ny][0] for ny in NY_ALL])
t2_all = np.array([DATA[ny][1] for ny in NY_ALL])
su2_all = t1_all / t2_all
eff2_all = su2_all / 2 * 100

t1_mpi = np.array([DATA[ny][0] for ny in NY_MPI])
t2_mpi = np.array([DATA[ny][1] for ny in NY_MPI])
t4_mpi = np.array([DATA[ny][2] for ny in NY_MPI])
su2_mpi = t1_mpi / t2_mpi
su4_mpi = t1_mpi / t4_mpi
eff2_mpi = su2_mpi / 2 * 100
eff4_mpi = su4_mpi / 4 * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Wall time (log scale, all n_y)
ax = axes[0]
xa = np.arange(len(NY_ALL))
bw = 0.27
b1 = ax.bar(xa - bw,   t1_all, width=bw, color='steelblue', alpha=0.85, label='1 GPU (baseline)')
b2 = ax.bar(xa,        t2_all, width=bw, color='tomato',    alpha=0.85, label='2 GPU mqpu (single-node)')
for i, ny in enumerate(NY_ALL):
    t4 = DATA[ny][2]
    if t4 is not None:
        ax.bar(xa[i] + bw, t4, width=bw, color='seagreen', alpha=0.85,
               label='4 GPU mpi4py (multi-node)' if i == NY_ALL.index(NY_MPI[0]) else '')
for rect, v in zip(list(b1)+list(b2), list(t1_all)+list(t2_all)):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()*1.06,
            f'{v:.0f}s' if v >= 10 else f'{v:.1f}s', ha='center', va='bottom', fontsize=6.5)
for i, ny in enumerate(NY_ALL):
    t4 = DATA[ny][2]
    if t4 is not None:
        ax.text(xa[i] + bw, t4*1.06, f'{t4:.0f}s', ha='center', va='bottom', fontsize=6.5)
# OOM annotation
ax.annotate('n_y\u226516\nOOM\n(68 GB/traj)', xy=(len(NY_ALL)-0.5, 500),
            ha='left', va='center', fontsize=8, color='red', style='italic')
ax.set_yscale('log'); ax.set_ylim(bottom=0.5)
ax.set_xticks(xa)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_ALL], fontsize=9)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title(f'Noisy sampling wall time\n({N_SHOTS} shots, TIMESTEPS=20)', fontsize=11)
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, axis='y', alpha=0.4)

# Panel 2: Speedup (n_y = 10..14 where all 3 configs measured)
ax = axes[1]
xm = np.arange(len(NY_MPI))
bw2 = 0.35
ax.bar(xm - bw2/2, su2_mpi, width=bw2, color='tomato',   alpha=0.85, label='2-GPU speedup')
ax.bar(xm + bw2/2, su4_mpi, width=bw2, color='seagreen', alpha=0.85, label='4-GPU speedup')
for i, (s2, s4) in enumerate(zip(su2_mpi, su4_mpi)):
    ax.text(xm[i] - bw2/2, s2+0.05, f'{s2:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(xm[i] + bw2/2, s4+0.05, f'{s4:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(2, color='tomato',   lw=1.0, ls=':', alpha=0.6, label='Ideal 2x')
ax.axhline(4, color='seagreen', lw=1.0, ls=':', alpha=0.6, label='Ideal 4x')
ax.set_xticks(xm)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_MPI], fontsize=10)
ax.set_ylabel('Speedup  T_1GPU / T_nGPU', fontsize=11)
ax.set_title('2-GPU vs 4-GPU speedup\n(vs 1-GPU baseline)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, max(su4_mpi) * 1.25)

# Panel 3: Parallel efficiency
ax = axes[2]
ax.bar(xm - bw2/2, eff2_mpi, width=bw2, color='tomato',   alpha=0.85, label='2-GPU eff.')
ax.bar(xm + bw2/2, eff4_mpi, width=bw2, color='seagreen', alpha=0.85, label='4-GPU eff.')
for i, (e2, e4) in enumerate(zip(eff2_mpi, eff4_mpi)):
    ax.text(xm[i] - bw2/2, e2+1, f'{e2:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(xm[i] + bw2/2, e4+1, f'{e4:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', lw=0.8, ls=':', label='Ideal 100%')
ax.axhline(80,  color='gold', lw=0.8, ls='--', alpha=0.7, label='80% threshold')
ax.set_xticks(xm)
ax.set_xticklabels([f'$n_y$={ny}\n({2*ny}q)' for ny in NY_MPI], fontsize=10)
ax.set_ylabel('Parallel efficiency (%)', fontsize=11)
ax.set_title('Parallel efficiency\n(n_y = 10..14)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0, max(eff4_mpi) * 1.25)

fig.suptitle(
    'CUDA-Q Shot-Splitting Parallelisation  |  '
    f'{N_SHOTS} shots  |  p1=0.0001, p2=0.001  |  H100 (Kestrel)  |  '
    'Max: n_y=14 (28 qubits) — n_y\u226516 OOM',
    fontsize=10, y=1.02)

plt.tight_layout()
out = os.path.join(SCRATCH_IMPL, 'parallelisation_study.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {out}')

# ── Figure 2: strategy comparison at n_y=14 (noisy) and n_y=16 (mgpu only) ──
#
# n_y=14 noisy (256 shots):
#   1-GPU baseline  808.1 s
#   2-GPU mqpu      283.2 s
#   4-GPU mpi4py    144.3 s   ← fastest (shot-splitting)
#   8-GPU mgpu      198.4 s   ← statevector sharding, sequential shots
#
# n_y=16: single-GPU strategies OOM for noisy (68.7 GB/trajectory > H100 80 GB)
#   8-GPU mgpu noiseless   12.6 s   ← only viable approach

DATA14 = {
    '1-GPU\nbaseline':     808.1,
    '2-GPU\nmqpu':         283.2,
    '4-GPU\nmpi4py':       144.3,
    '8-GPU\nmgpu\n(noisy)': 198.4,
}
LABELS14 = list(DATA14.keys())
TIMES14  = list(DATA14.values())
COLORS14 = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5),
                           gridspec_kw={'width_ratios': [1.5, 1]})

# --- Left panel: n_y=14 all 4 strategies ---
ax = axes2[0]
xs = np.arange(len(LABELS14))
bars = ax.bar(xs, TIMES14, color=COLORS14, alpha=0.85, width=0.55, zorder=3)
for bar, t in zip(bars, TIMES14):
    su = 808.1 / t
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{t:.0f} s\n({su:.2f}×)', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(808.1, color='steelblue', lw=1.0, ls=':', alpha=0.5, label='1-GPU baseline')
ax.set_xticks(xs)
ax.set_xticklabels(LABELS14, fontsize=10)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title('$n_y = 14$  (28 qubits)  — NOISY trajectory\n256 shots, p₁=0.0001, p₂=0.001', fontsize=11)
ax.set_ylim(0, max(TIMES14) * 1.3)
ax.grid(True, axis='y', alpha=0.4, zorder=0)
ax.legend(fontsize=8)

# Annotate key finding
ax.annotate('mpi4py (shot-split)\nfastest: 144.3 s\n5.60× speedup',
            xy=(2, 144.3), xytext=(2.6, 500),
            arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.5),
            fontsize=8.5, color='seagreen', fontweight='bold',
            ha='center')

# --- Right panel: n_y=16, strategy availability ---
ax = axes2[1]

strategies_16 = [
    ('1-GPU\nbaseline',       None, 'OOM\n(68.7 GB/traj)'),
    ('2-GPU\nmqpu',           None, 'OOM'),
    ('4-GPU\nmpi4py',         None, 'OOM'),
    ('8-GPU\nmgpu\n(ideal)',  12.6, None),
]
xs16 = np.arange(len(strategies_16))
colors16 = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

for i, (lbl, t, oom) in enumerate(strategies_16):
    if t is not None:
        b = ax.bar(i, t, color=colors16[i], alpha=0.85, width=0.55, zorder=3)
        ax.text(i, t + 0.4, f'{t:.1f} s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        # Draw a hatched "OOM" bar for visual clarity
        ax.bar(i, 15, color=colors16[i], alpha=0.25, width=0.55, hatch='//', zorder=3)
        ax.text(i, 16, oom, ha='center', va='bottom', fontsize=9,
                color='red', style='italic', fontweight='bold')

ax.set_xticks(xs16)
ax.set_xticklabels([s[0] for s in strategies_16], fontsize=10)
ax.set_ylabel('Wall time (s)', fontsize=11)
ax.set_title('$n_y = 16$  (32 qubits)  — noiseless only\n'
             '256 shots  |  noisy = OOM on single H100\n'
             '(68.7 GB/traj > 80 GB)', fontsize=10)
ax.set_ylim(0, 50)
ax.grid(True, axis='y', alpha=0.4, zorder=0)

fig2.suptitle(
    'Strategy comparison: noisy n_y=14 vs mgpu n_y=16  |  '
    '256 shots  |  H100 80 GB (Kestrel)',
    fontsize=10, y=1.01)

plt.tight_layout()
out2 = os.path.join(SCRATCH_IMPL, 'parallelisation_mgpu_comparison.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {out2}')

# Summary table
print(f"\n{'n_y':>4}  {'qubits':>6}  {'1-GPU(s)':>9}  {'2-GPU(s)':>9}  "
      f"{'4-GPU(s)':>9}  {'Su_2x':>7}  {'Su_4x':>7}  {'Eff_2x':>7}  {'Eff_4x':>7}")
print('-' * 85)
for ny in NY_ALL:
    t1s, t2s, t4s = DATA[ny]
    su2 = t1s/t2s; e2 = su2/2*100
    if t4s is not None:
        tag = ' †' if ny == 14 else ''
        su4 = t1s/t4s; e4 = su4/4*100
        print(f'{ny:>4}  {2*ny:>6}  {t1s:>9.1f}  {t2s:>9.1f}  '
              f'{t4s:>9.1f}  {su2:>7.2f}x  {su4:>7.2f}x  {e2:>6.0f}%  {e4:>6.0f}%{tag}')
    else:
        print(f'{ny:>4}  {2*ny:>6}  {t1s:>9.1f}  {t2s:>9.1f}  '
              f'{"—":>9}  {su2:>7.2f}x  {"—":>7}  {e2:>6.0f}%  {"—":>6}')
print('† n_y=14: 1-GPU warm (808 s); 2-GPU & 4-GPU cold-start — speedup includes warm-up benefit')
print('  n_y>=16 (>=32 qubits): 68.7 GB/trajectory exceeds H100 HBM -> OOM')
