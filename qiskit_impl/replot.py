"""Regenerate noise_study_accuracy.png from saved JSON (no GPU needed)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

data    = json.load(open('noise_study_results.json'))
ny_vals = sorted(int(k) for k in data)

ref_v   = [data[str(ny)]['classical'] for ny in ny_vals]
phi_i   = [data[str(ny)]['ideal']     for ny in ny_vals]
phi_n   = [data[str(ny)]['noisy']     for ny in ny_vals]
P1      = data[str(ny_vals[0])]['p1']
P2      = data[str(ny_vals[0])]['p2']
N_SHOTS = data[str(ny_vals[0])]['n_shots']

abs_gap   = [phi_n[i] - phi_i[i]  for i in range(len(ny_vals))]
dqa_gap   = [phi_i[i] - ref_v[i]  for i in range(len(ny_vals))]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Left: absolute phi
ax = axes[0]
ax.plot(ny_vals, ref_v, 'k--o', lw=2, ms=8, label='Classical optimal')
ax.plot(ny_vals, phi_i, 'b-s',  lw=2, ms=8, label='DQA ideal (noiseless, GPU observe)')
ax.plot(ny_vals, phi_n, 'r-^',  lw=2, ms=8, label=f'DQA noisy (p1={P1}, p2={P2})')
ax.set_xlabel('$n_y$ (turbines)', fontsize=12)
ax.set_ylabel('Expected second-stage cost  $\\phi$', fontsize=12)
ax.set_title('Expected cost vs system size', fontsize=13)
ax.set_xticks(ny_vals); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

# Middle: absolute noise-induced cost increase
ax = axes[1]
bars = ax.bar(ny_vals, abs_gap, color='tomato', alpha=0.85, width=1.2)
for rect, val in zip(bars, abs_gap):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('$n_y$ (turbines)', fontsize=12)
ax.set_ylabel(r'Absolute noise impact  $\Delta\phi = \phi_{noisy} - \phi_{ideal}$', fontsize=10)
ax.set_title(f'Noise-induced cost increase\n(p1={P1}, p2={P2}, {N_SHOTS} shots)', fontsize=13)
ax.set_xticks(ny_vals); ax.grid(True, axis='y', alpha=0.4)

# Right: stacked bar decomposition
ax = axes[2]
b1 = ax.bar(ny_vals, ref_v,    width=1.2, color='steelblue', alpha=0.85, label='Classical optimal')
b2 = ax.bar(ny_vals, dqa_gap,  width=1.2, color='gold',      alpha=0.85, label='DQA approx. gap',
            bottom=ref_v)
b3 = ax.bar(ny_vals, abs_gap,  width=1.2, color='tomato',    alpha=0.85, label='Noise impact',
            bottom=[ref_v[i]+dqa_gap[i] for i in range(len(ny_vals))])
for i, (rect, val) in enumerate(zip(b3, abs_gap)):
    top = ref_v[i] + dqa_gap[i] + val
    ax.text(rect.get_x()+rect.get_width()/2, top + 0.2,
            f'+{val:.1f}', ha='center', va='bottom', fontsize=8, color='darkred')
ax.set_xlabel('$n_y$ (turbines)', fontsize=12)
ax.set_ylabel('Expected second-stage cost  $\\phi$', fontsize=12)
ax.set_title('Cost decomposition\nclassical + DQA gap + noise impact', fontsize=13)
ax.set_xticks(ny_vals); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('noise_study_accuracy.png', dpi=150, bbox_inches='tight')
print('Saved noise_study_accuracy.png')

print(f"\n{'ny':>4}  {'Delta_phi':>10}  {'DQA_gap':>10}")
for i, ny in enumerate(ny_vals):
    print(f"{ny:>4}  {abs_gap[i]:>10.2f}  {dqa_gap[i]:>10.2f}")
