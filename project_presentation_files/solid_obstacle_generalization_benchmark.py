# Imports
import io
import os
import glob
import base64

import torch
import stable_baselines3
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback

import gym
from gym import spaces
from gym.wrappers import Monitor

# Custom class imports
from gym.envs.registration import register
from apollo_lander import ApolloLander

def wrap_env(env, subdir):
    env = Monitor(env, './video/' + subdir, force = True)
    return env

def make_obstacle_env(obstacle_params=[-0.5, 4.0, 0.5]):
    """
    Create environment
    We register our custom environment using the register() function imported from
    gym, we can then use the gym.make method to be able to monitor the environment
    and export movies

    obstacle_params: array with position and size of obstacle, all units in meters
    obstacle_params[0]: obs_x, a number between -15 and +15 (centered at goal)
    obstacle_params[1]: obs_y, a number between 0 and 20 (0 is at goal)
    obstacle_params[2]: obs_radius, a number between 0 and 30
    """
    # Remove the environment if it was already registered
    for env in gym.envs.registration.registry.env_specs.copy():
        if 'ApolloLanderSolidObstacle-v0' in env:
            # print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    # Register our environment
    register(
        id='ApolloLanderSolidObstacle-v0',
        entry_point="apollo_lander:ApolloLanderSolidObstacle", # Where our class is located
        kwargs={'obstacle_params' : obstacle_params}, # We can define the pos of an obstacle
        max_episode_steps=1000, # max_episode_steps / FPS = runtime seconds
        reward_threshold=200,
    )
    env = gym.make('ApolloLanderSolidObstacle-v0')
    return env


obs_x_array = np.array([-2, 0, 2])
obs_y_array = np.array([3, 5, 7])
obs_radius_array = np.array([0.75, 1.0])

n_x_pos = len(obs_x_array)
n_y_pos = len(obs_y_array)
n_radius = len(obs_radius_array)
n_episodes = 1

total_reward_array = np.zeros((n_radius, n_x_pos, n_y_pos, n_episodes)) 

for i_obs, obs_radius in enumerate(obs_radius_array):
    for j_obs, obs_x in enumerate(obs_x_array):
        for k_obs, obs_y in enumerate(obs_y_array):
            # print("obs_radius: " + str(obs_radius) + " obs_x: " + str(obs_x) + " obs_y: " + str(obs_y))
            # print("i_obs: " + str(i_obs) + " j_obs: " + str(j_obs) + " k_obs: " + str(k_obs))
            env = make_obstacle_env(obstacle_params=[obs_x, obs_y, obs_radius])

            # model = DQN.load("./DQN_no_obstacle/DQN_no_obstacle_model.zip",
            model = DQN.load("./DQN_solid_obstacle/DQN_solid_obstacle_model.zip",
                              env=env) # Load model
            """
            We try the pretrained model here
            """
            for l_episode in np.arange(n_episodes):
              env = wrap_env(env, "video_radius_" + str(obs_radius) + "_x_" + str(obs_x) + "_y_" + str(obs_y) + "_episode_" + str(l_episode))
              observation = env.reset()
              total_reward = 0
              while True:
                  env.render()
                  action, states = model.predict(observation, deterministic=True)
                  observation, reward, done, info = env.step(action)
                  total_reward += reward
                  if done:
                      break;
              total_reward_array[i_obs, j_obs, k_obs, l_episode] = total_reward
              env.close()
              print(total_reward)

data_dir = "./obstacle_benchmark/"
os.makedirs(data_dir, exist_ok=True)

# Save the data
filepath = os.path.join(data_dir, 'reward_obstacle_map_pretrainedModel_noObstacle')
data = {"obs_x_array": obs_x_array, "obs_y_array": obs_y_array, 
      "obs_radius_array": obs_radius_array, "total_reward_array": total_reward_array}
np.savez(filepath, **data, allow_pickle=True)

# Plot the data

plt.figure(figsize=(10, 5))
for i_radius, obs_radius in enumerate(obs_radius_array):
  plt.subplot(1, n_radius, i_radius + 1)
  mean_reward = total_reward_array[i_radius,:,:,:].mean(axis=2).squeeze()
  max_abs = np.max(np.abs(mean_reward[:]))
  # transpose matrix for plotting
  plt.imshow(mean_reward.T, vmin=-max_abs, vmax=+max_abs, cmap='RdBu', 
    extent=(obs_x_array[0], obs_x_array[-1], obs_y_array[0], obs_y_array[-1]))
  plt.colorbar()
  plt.title('Average reward over 10 episodes \n (obstacle radius %f)' % obs_radius)
  plt.xlabel('obstacle x')
  plt.ylabel('obstacle y')

plt.show()

