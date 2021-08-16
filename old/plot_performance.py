import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy, load_results

import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')

log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

"""
We explore the model's performance now.
"""
# Organising the logged results in to a clean format for plotting
x, y = ts2xy(load_results(log_dir), 'timesteps')  

window_size = 5
y_smooth =  np.convolve(y, np.ones(window_size), mode='same') / window_size

plt.close()
plt.plot(x,y,c='b', alpha=0.2,label='non-smooth')
plt.plot(x,y_smooth, c='k', lw=2,label='smooth')
plt.legend()
#plt.ylim([-300, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.tight_layout()
plt.savefig('performance_plot.png')

plt.close()
plt.plot(x, np.cumsum(y)/(np.arange(len(y))+1), c='k')
#plt.ylim([-300, 300])
plt.xlabel('Timesteps')
plt.ylabel('Episode Rewards')
plt.tight_layout()
plt.savefig('performance_plot_cum.png')

