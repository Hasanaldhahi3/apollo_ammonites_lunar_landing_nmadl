from scipy.interpolate import interp1d
import numpy as np
import os
from matplotlib import pyplot as plt

plt.style.use('default')
plt.style.use('fivethirtyeight')

def interpolate_timesteps(time, reward):
  y_interp = interp1d(time, reward)
  interp_rewards = y_interp(interp_times)
  return interp_rewards

def moving_average(a, n=3) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

# Load hyperparameter search data
data_dir = "/tmp/hyperparamtuning/"
outfile = np.load(os.path.join(data_dir, 'hyper-search-output.npz'), allow_pickle=True)
# /tmp/hyperparamtuning/hyper-search-output.npz
# os.path.join(data_dir, 'hyper-search-output.npz')

times = outfile['times']
lr_array = outfile['lr_array']
episode_rewards = outfile['episode_rewards']
nn_layers_array = outfile['nn_layers_array']

# Round to the largest starting time and shortest end time, to compare in same time intervals.
max_epoch_times = (min([max(t) for t in times]) // 10**4) * 10**4
min_epoch_times = ((max([min(t) for t in times]) // 100) + 1) * 100
print(min_epoch_times)
# Interpolate to equal timesteps
interp_times = np.arange(min_epoch_times, max_epoch_times, 100)
interp_rewards = [interpolate_timesteps(time, reward) for time, reward in zip(times, episode_rewards)]
# Moving average to smooth out fluctuations
w_size = 500
smooth_rewards = [moving_average(trace, w_size) for trace in interp_rewards]
# Crop the time to get same size
time_cropped = interp_times[w_size - 1:]
# Total reward in last time steps
final_reward = [np.mean(reward[-500:]) for reward in interp_rewards]

# Plot results
fig = plt.figure(figsize=[15, 3])
# fig = plt.figure(figsize=(11.69,8.27))

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

colors = plt.cm.Dark2(np.arange(len(episode_rewards)))

for i_trace in np.arange(len(times)):
  ax2.plot(time_cropped, smooth_rewards[i_trace], color=colors[i_trace])
  ax1.plot(interp_times, interp_rewards[i_trace], color=colors[i_trace], alpha=0.5)

bplot = ax3.bar(np.arange(len(final_reward)),final_reward, 
                color=colors)

layer_labels = ['-'.join(map(str, nn_layers)) for nn_layers in nn_layers_array]
ax3.set_xticks(np.arange(len(final_reward)))
ax3.set_xticklabels(layer_labels, Rotation=90)

ax1.set_xlabel('Timesteps')
ax1.set_title('Episode Rewards')
ax2.set_xlabel('Timesteps')
ax2.set_title('Episode Rewards MA(5000)')
ax3.set_xlabel('Layer architecture')
ax3.set_title('Mean(Last 5000 episode rewards)')

ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,5))
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,5))
ax1.set_ylim(-300, 300)
ax2.set_ylim(-300, 300)

for ax in [ax1, ax2, ax3]:
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

fig_dir = '/tmp/figures/'
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, 'fig_reward_hyperparamtuning_lr_0.001_32-256_1or2layers.pdf'), bbox_inches='tight', orientation='landscape' )
