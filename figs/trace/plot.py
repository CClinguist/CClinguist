import pandas as pd
import matplotlib.pyplot as plt

def plot_x_trace(ax, trace, alo_name, color, show_xlabel=False):
    ax.plot(trace[:, 0], trace[:, 1], label=f'{alo_name}', color=color, linewidth=5)
    if show_xlabel:
        ax.set_xlabel('Time (s)', fontsize=52)
    ax.set_ylabel('BQDF', fontsize=52)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis='both', which='major', labelsize=48)
    ax.legend(loc='upper right', fontsize=52)


df_aliyun = pd.read_csv('aliyun.csv')

trace_aliyun = df_aliyun.to_numpy()
trace_aliyun = trace_aliyun[trace_aliyun[:, 0] <= 10]

fig, ax = plt.subplots(figsize=(12, 8)) 


plot_x_trace(ax, trace_aliyun, 'Alibaba Cloud', '#1f77b4', show_xlabel=True)

plt.tight_layout()

plt.show()

fig.savefig('ccprints.png')
