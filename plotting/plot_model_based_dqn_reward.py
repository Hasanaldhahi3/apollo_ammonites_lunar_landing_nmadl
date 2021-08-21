import re
import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from stable_baselines3.common.results_plotter import ts2xy, load_results
from matplotlib import pyplot as plt
import seaborn
plt.style.use('default')
plt.style.use('fivethirtyeight')


def interpolate_timesteps(time, reward):
  y_interp = interp1d(time, reward)
  interp_rewards = y_interp(interp_times)
  return interp_rewards


def moving_average(a, n=3) :
  #ret = np.cumsum(a, dtype=float)
  #ret[n:] = ret[n:] - ret[:-n]
  #return ret[n - 1:] / n
  # xxxxx
  #   ^^^
  #  ^^^
  # ^^^
  start_ids = np.arange(0, len(a)-n+1, 1)
  a_seg = np.array([a[x:x+n] for x in start_ids])
  return a_seg.mean(axis=1)


# Load hyperparameter search data

#data_dirs = glob.glob("../tmp/effect_buffersize/gym-*")
#ids = np.argsort([float(re.search('gym-buffer_size(\d+)-m', os.path.basename(x)).group(1)) for x in data_dirs])

data_dirs = glob.glob("../tmp/effect_model/gym-*")
ids = np.argsort([float(re.search('gym-buffer_size(\d+)-b', os.path.basename(x)).group(1)) for x in data_dirs])

#data_dirs = glob.glob("../tmp/effect_learning_rate/gym-*")
#ids = np.argsort([float(re.search('gym-learning_rate([.\d]+)', os.path.basename(x)).group(1)) for x in data_dirs])

#data_dirs = glob.glob("../tmp/effect_gamma/gym-*")
#ids = np.argsort([float(re.search('gym-gamma([.\d]+)', os.path.basename(x)).group(1)) for x in data_dirs])

data_dirs = [data_dirs[x] for x in ids]

configs = []
times = []
episode_rewards = []
for dd in data_dirs:
    configs.append(os.path.basename(dd)[4:])
    x, y = ts2xy(load_results(dd), 'episodes')
    times.append(x)
    episode_rewards.append(y)

# Round to the largest starting time and shortest end time, to compare in same time intervals.
#max_times = (min([max(t) for t in times]) // 10**4) * 10**4
#min_times = ((max([min(t) for t in times]) // 100) + 1) * 100
# Interpolate to equal timesteps
#interp_times = np.arange(min_times, max_times, 100)
#interp_rewards = [interpolate_timesteps(time, reward) for time, reward in zip(times, episode_rewards)]
# Moving average to smooth out fluctuations
w_size = 100
smooth_rewards = [moving_average(trace, w_size) for trace in episode_rewards]
smooth_times = [trace[w_size - 1:] for trace in times]
# Crop the time to get same size
#time_cropped = interp_times[w_size - 1:]
# Total reward in last time steps
final_rewards = [x[-1] for x in smooth_rewards]

for i,j in zip(configs, final_rewards):
    print(i,j)


# Plot results
plt.close()
colors = plt.cm.viridis(np.arange(len(episode_rewards))/len(episode_rewards))
fig = plt.figure(figsize=[12,6])

ax1 = plt.subplot(111)

for i in range(len(smooth_rewards)):
  ax1.plot(smooth_times[i]+100000, smooth_rewards[i], color=colors[i], label=configs[i].replace('-',' '), lw=2)
ax1.legend(frameon=False, loc='lower right')#, ncol=2)

ax1.set_xlabel('Episode')
ax1.set_title(f'Rewards MA({w_size})')

ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,5))
ax1.set_xlim(100,2000)
#ax1.set_ylim(-300, 200)
seaborn.despine()

plt.tight_layout()
fig_dir = '../tmp/figures/'
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, 'fig_model_based_dqn_reward.png'), bbox_inches='tight', orientation='landscape' )
