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
        if 'ApolloLander-v0' in env:
            # print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    # Register our environment
    register(
        id="ApolloLander-v0",
        entry_point="apollo_lander:ApolloLander", # Where our class is located
        kwargs={'obstacle_params' : obstacle_params}, # We can define the pos of an obstacle
        max_episode_steps=1000, # max_episode_steps / FPS = runtime seconds
        reward_threshold=200,
    )
    env = gym.make('ApolloLander-v0')
    return env


n_x_pos = 10
n_y_pos = 10
n_radius = 5

obs_x_array = np.linspace(-10, 10, n_x_pos)
obs_y_array = np.linspace(0, 20, n_y_pos)
obs_radius_array = np.linspace(0.1, 1, n_radius)

n_episodes = 100

total_reward_array = np.zeros((n_radius, n_x_pos, n_y_pos, n_episodes)) 

for i_obs, obs_radius in enumerate(obs_radius_array):
    for j_obs, obs_x in enumerate(obs_x_array):
        for k_obs, obs_y in enumerate(obs_y_array):
            
            env = make_obstacle_env(obstacle_params=[obs_x, obs_y, obs_radius])

            model = DQN.load("./dqn_lunar_1.0e-03_256-256",
                             env=env) # Load model

            """
            We try the pretrained model here
            """
            # test_env = wrap_env(gym.make("LunarLander-v2"))
            for l_episode in np.arange(n_episodes):
              observation = env.reset()
              total_reward = 0
              while True:
                  action, states = model.predict(observation, deterministic=False)
                  observation, reward, done, info = env.step(action)
                  total_reward += reward
                  if done:
                      break;
              total_reward_array[i_obs, j_obs, k_obs, l_episode] = total_reward
            del env

  data_dir = "./tmp/obstacle_benchmark/"
  os.makedirs(data_dir, exist_ok=True)

# Save the data
  filepath = os.path.join(data_dir, 'reward_obstacle_map_pretrainedModel_noObstacle')
  data = {"obs_x_array": obs_x_array, "obs_y_array": obs_y_array, 
          "obs_radius_array": obs_radius_array, "total_reward_array": total_reward_array}
  np.savez(filepath, **data, allow_pickle=True)

# Plot the data

plt.figure(figsize=(25, 5))
for i_radius, obs_radius in enumerate(obs_radius_array):
  plt.subplot(1, 5, i_radius + 1)
  mean_reward = total_reward_array[i_radius,:,:,:].min(axis=2).squeeze()
  max_abs = np.max(np.abs(mean_reward[:]))
  plt.imshow(mean_reward, vmin=-max_abs, vmax=+max_abs, cmap='RdBu')
  plt.colorbar()
  plt.title('Average reward over 10 episodes \n (obstacle radius %f)' % obs_radius)
