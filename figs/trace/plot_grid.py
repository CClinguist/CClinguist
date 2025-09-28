import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_x_trace(ax, trace, alo_name, color, show_xlabel=False, show_ylabel=False):
    trace = trace[trace[:, 0] <= 8.2]
    ax.plot(trace[:, 0], trace[:, 1], label=f'{alo_name}', color=color, linewidth=3)
    if show_xlabel:
        ax.set_xlabel('Time (s)', fontsize=34)
    if show_ylabel:
        ax.set_ylabel('BQDF', fontsize=34)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=32)


df_ecn_dctcp = pd.read_csv('dctcpecn.csv')
df_loss_cubic = pd.read_csv('cubicecn.csv')
df_ecn_bbrv2 = pd.read_csv('bbr2.csv')
df_loss_bbr = pd.read_csv('bbr.csv')

trace_ecn_dctcp = df_ecn_dctcp.to_numpy()
trace_loss_cubic = df_loss_cubic.to_numpy()
trace_ecn_bbrv2 = df_ecn_bbrv2.to_numpy()
trace_loss_bbr = df_loss_bbr.to_numpy()


fig = plt.figure(figsize=(20, 10)) 
gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.25) 


axs_left = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
plot_x_trace(axs_left[0], trace_ecn_dctcp, 'ECN-based DCTCP', '#009E73', 
             show_xlabel=False, show_ylabel=True)
plot_x_trace(axs_left[1], trace_loss_cubic, 'ECN-based CUBIC', '#D95319', 
             show_xlabel=True, show_ylabel=True)


axs_right = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
plot_x_trace(axs_right[0], trace_ecn_bbrv2, 'ECN-based BBR-V2', '#1f77b4', 
             show_xlabel=False, show_ylabel=False)
plot_x_trace(axs_right[1], trace_loss_bbr, 'BBR', '#ff7f0e', 
             show_xlabel=True, show_ylabel=False)


handles, labels = [], []
for ax in axs_left + axs_right:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

fig.legend(handles, labels, loc='upper center', fontsize=30, ncol=4, 
           bbox_to_anchor=(0.5, 1), columnspacing=1, handletextpad=0.4)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])  


fig.savefig('CCprints.png', bbox_inches='tight', dpi=500)
plt.show()