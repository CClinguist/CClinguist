import numpy as np
import matplotlib.pyplot as plt
import os
import re

result_paths = {
    'dis_scatter/dis_log_0': {
        'Samples1': 'WS2',
        'Samples2': 'WS4',
        'Samples3': 'WS6'
    },
    'dis_scatter/dis_log_1': {
        'Samples1': 'WS1',
        'Samples2': 'WS3',
        'Samples3': 'WS5'
    }
}

file_name_map = {
    'pccLatency.txt': 'Samples1',
    'astraea.txt': 'Samples2',
    'pccLoss.txt': 'Samples3'
}


color_dict = {
    'WS1': '#E41A1C', 
    'WS2': '#377EB8',  
    'WS3': '#4DAF4A', 
    'WS4': '#984EA3', 
    'WS5': '#FF7F00',  
    'WS6': '#FFD300'   
}


bg_colors = ['#D6EAF8', '#D6EAF8', '#D6EAF8', '#D6EAF8']


num_known_ccas = ['12', '13', '14', '15']
dis_all = {key: {} for key in num_known_ccas}


for path, sample_name_map in result_paths.items():
    for cca in num_known_ccas:
        dir_path = os.path.join(path, cca)
        for fname, sample_key in file_name_map.items():
            full_path = os.path.join(dir_path, fname)
            if not os.path.exists(full_path):
                continue
            dis = []
            with open(full_path, 'r') as f:
                lines = f.readlines()[2:]
                for line in lines:
                    match = re.search(r'\(([\d.]+)', line)
                    if match:
                        dis_val = float(match.group(1))
                        dis.append(dis_val)
            sample_name = sample_name_map[sample_key]
            dis_all[cca].setdefault(sample_name, []).extend(dis)


fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
#fig.suptitle('Distance Distribution of Unknown CCAs', fontsize=16)
plt.subplots_adjust(wspace=1.5)  


for idx, cca in enumerate(num_known_ccas):
    ax = axes[idx]
    data = dis_all[cca]


    sorted_items = sorted(data.items(), key=lambda x: int(x[0][2:]))


    ax.axhspan(-2, len(sorted_items)*4 - 2, facecolor=bg_colors[idx], alpha=0.3, zorder=0)



    all_x = []
    for i, (sample_label, distances) in enumerate(sorted_items):
        y_jitter = np.random.uniform(0, 0, size=len(distances)) + i * 4  

        ax.scatter(
            distances,
            y_jitter,
            color=color_dict.get(sample_label, 'gray'),
            alpha=1,
            label=sample_label,
            s=50,
            edgecolors='k',
            linewidth=0.3
        )
        all_x.extend(distances)


    if all_x:
        x_min, x_max = min(all_x), max(all_x)
    else:
        x_min, x_max = 0, 1

    if idx == 0:
        ax.set_xlim(0, x_max + 0.5) 
    elif idx == 1:
        ax.set_xlim(x_min - 0.5, x_max + 0.2) 
    elif idx == 2:
        ax.set_xlim(x_min - 0.9, x_max + 0.5)  
    else:
        ax.set_xlim(x_min - 0.5, x_max + 0.5) 



    ax.axvline(x=2.45, color='black', linestyle='--', linewidth=1)


    ax.set_ylim(-2, len(sorted_items) * 4 - 2)
    ax.set_yticks([i * 4 for i in range(len(sorted_items))])

    ax.set_yticklabels([k for k, _ in sorted_items], fontsize=12)

    ax.invert_yaxis()

    ax.set_xlabel('Distance', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)


    ax.grid(axis='y', linestyle='--', alpha=0.3)


    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('scatter_unknown_ccas_1x4_final.png', dpi=300)
plt.show()
