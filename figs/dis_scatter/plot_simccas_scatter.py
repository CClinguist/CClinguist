import numpy as np
import matplotlib.pyplot as plt
import os
import re


result_path = 'dis_scatter/dis_log_sim'
num_known_ccas = ['13', '15']
file_map = {
    'pccLatency2.txt': 'WS7',
    'pccLatency3.txt': 'WS8',
}

color_dict = {
    'WS7': '#FF6666', 
    'WS8': '#FF9900',  
}

bg_colors = ['#D6EAF8', '#D6EAF8']  
margin = 2.45


dis_all = {}
for cca in num_known_ccas:
    dis_all[cca] = {}
    dir_path = os.path.join(result_path, cca)
    for fname, sample_label in file_map.items():
        full_path = os.path.join(dir_path, fname)
        if not os.path.exists(full_path):
            continue
        dis = []
        with open(full_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                match = re.search(r'\(([\d.]+)', line)
                if match:
                    dis_val = float(match.group(1))
                    dis.append(dis_val)
        dis_all[cca][sample_label] = dis


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
plt.subplots_adjust(wspace=0.7)

for idx, cca in enumerate(num_known_ccas):
    ax = axes[idx]
    data = dis_all[cca]
    sorted_items = sorted(data.items(), key=lambda x: x[0])


    ax.axhspan(-2, len(sorted_items) * 4 - 2, facecolor=bg_colors[idx], alpha=0.3, zorder=0)


    all_x = []
    for i, (label, distances) in enumerate(sorted_items):
        y_pos = i * 4
        y_jitter = np.random.uniform(0, 0, size=len(distances)) + y_pos
        ax.scatter(
            distances,
            y_jitter,
            color=color_dict.get(label, 'gray'),
            s=150,
            alpha=1,
            edgecolors='k',
            linewidth=0.3
        )
        all_x.extend(distances)

  
    if idx == 0:  
        x_min, x_max = 0, max(all_x) + 0.5  
    else:  
        x_min, x_max = min(all_x) - 0.1, max(all_x) + 0.1  

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-2, len(sorted_items) * 4 - 2)
    ax.set_yticks([i * 4 for i in range(len(sorted_items))])
    ax.set_yticklabels([label for label, _ in sorted_items], fontsize=22)
    ax.invert_yaxis()

    ax.set_xlabel('Distance', fontsize=24)
    ax.tick_params(axis='x', labelsize=22)
    ax.axvline(x=margin, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('scatter_simccas_2panel.png', dpi=300)
plt.show()