import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frames = np.arange(0, 180)
base = 0.11 + 0.04 * np.sin(frames / 7.0) + 0.02 * np.cos(frames / 12.0)
curve = base.copy()
curve[30:60] += 0.45 * np.exp(-((frames[30:60] - 45.0) / 8.5) ** 2)
curve[120:150] += 0.38 * np.exp(-((frames[120:150] - 135.0) / 8.0) ** 2)
threshold = 0.25

plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(12, 6.8), dpi=170)
fig.patch.set_facecolor('#111827')
ax.set_facecolor('#111827')

ax.plot(frames, curve, color='#8ec5ff', linewidth=2.2, label='Rec MSE Curve')
ax.axhline(threshold, color='#ff5c5c', linestyle='--', linewidth=2.2, label='Threshold T = 0.25')
ax.axvspan(30, 60, color='red', alpha=0.18)
ax.axvspan(120, 150, color='red', alpha=0.18)

ax.set_xlim(0, 180)
ax.set_ylim(0, 0.8)
ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.linspace(0, 0.7, 8))
ax.grid(color='white', alpha=0.12, linewidth=0.8)
ax.set_xlabel('Video Frame Index', fontsize=13, color='white', labelpad=12)
ax.set_ylabel('Reconstruction MSE (Lrec)', fontsize=13, color='white', labelpad=12)
ax.tick_params(axis='both', colors='white', labelsize=11)

for spine in ax.spines.values():
    spine.set_color('#e5e7eb')
    spine.set_linewidth(1.0)

ax.text(44, 0.34, 'Biker Anomaly', color='white', fontsize=13, ha='center', va='center', weight='bold')
ax.text(134, 0.34, 'Skateboarder Anomaly', color='white', fontsize=13, ha='center', va='center', weight='bold')

fig.text(0.06, 0.92, 'Baseline Mean', color='#f9fafb', fontsize=12, ha='left')
fig.text(0.17, 0.92, '0.11', color='#f9fafb', fontsize=18, fontweight='bold', ha='left')
fig.text(0.33, 0.92, 'Decision T', color='#f9fafb', fontsize=12, ha='left')
fig.text(0.48, 0.92, '0.25', color='#f9fafb', fontsize=18, fontweight='bold', ha='left')
fig.text(0.63, 0.92, 'Peak Error (F65)', color='#f9fafb', fontsize=12, ha='left')
fig.text(0.80, 0.92, '0.58', color='#ff6b6b', fontsize=18, fontweight='bold', ha='left')
fig.text(0.88, 0.92, 'Peak Error (F148)', color='#f9fafb', fontsize=12, ha='left')
fig.text(0.995, 0.92, '0.49', color='#ff6b6b', fontsize=18, fontweight='bold', ha='right')

legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=11, handlelength=2.5)
for text in legend.get_texts():
    text.set_color('white')

plt.subplots_adjust(bottom=0.22, top=0.82, left=0.09, right=0.98)
out = 'paper/figures/anomaly_threshold_plot.png'
plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')
