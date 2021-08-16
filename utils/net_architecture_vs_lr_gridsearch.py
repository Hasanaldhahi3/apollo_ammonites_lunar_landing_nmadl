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
from model_based_dqn import ModelBasedDQN


''' Define the function to create the model and to loop over hyperparamgrid
this takes only a learning rate and network architecture, but more parameters can be defined
in a different gridsearch function
'''
def make_model(nn_layers=[64, 64], learning_rate=0.0001, discount_factor=0.99, 
               exploration_initial_eps=1, exploration_fraction = 0.5, 
               log_dir="/tmp/gym/"):
  
  # log_dir = "/tmp/gym/"
  os.makedirs(log_dir, exist_ok=True)

  # Create environment
  env = gym.make('LunarLander-v2')
  #You can also load other environments like cartpole, MountainCar, Acrobot. Refer to https://gym.openai.com/docs/ for descriptions.
  #For example, if you would like to load Cartpole, just replace the above statement with "env = gym.make('CartPole-v1')". 

  env = stable_baselines3.common.monitor.Monitor(env, log_dir )

  callback = EvalCallback(env,log_path = log_dir, deterministic=True) #For evaluating the performance of the agent periodically and logging the results.
  policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                      net_arch=nn_layers)

  model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs, 
              learning_rate=learning_rate,
              batch_size=1,  #for simplicity, we are not doing batch update.
              buffer_size=1, #size of experience of replay buffer. Set to 1 as batch update is not done
              learning_starts=1, #learning starts immediately!
              gamma=discount_factor, #discount facto. range is between 0 and 1. 
              tau = 1,  #the soft update coefficient for updating the target network
              target_update_interval=1, #update the target network immediately.
              train_freq=(1,"step"), #train the network at every step.
              max_grad_norm = 10, #the maximum value for the gradient clipping
              exploration_initial_eps = exploration_initial_eps, #initial value of random action probability
              exploration_fraction = exploration_fraction, #fraction of entire training period over which the exploration rate is reduced
              gradient_steps = 1, #number of gradient steps 
              seed = 1, #seed for the pseudo random generators
              verbose=0) #Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1. 
  return model, callback

def layers_lr_grid_search(lr_array, nn_layers_array):
  time_steps = 500000
  log_interval = 10
  # times = np.zeros((len(lr_array), len(nn_layers_array), time_steps))
  # episode_rewards = np.zeros((len(lr_array), len(nn_layers_array), time_steps))
  # times = np.zeros((len(lr_array), len(nn_layers_array), time_steps))
  # episode_rewards = np.zeros((len(lr_array), len(nn_layers_array), time_steps))
  # Because the output time steps are of different sizes, we use lists.
  times = []
  episode_rewards = []
  for i_lr, lr in enumerate(lr_array):
    for j_nn, nn_layers in enumerate(nn_layers_array):
      log_dir = "/tmp/gym/log_%0.1e_%s/" % (lr, '-'.join(map(str, nn_layers)))
      print(log_dir)
      model, callback = make_model(learning_rate=lr, nn_layers=nn_layers, log_dir=log_dir)
      model.learn(total_timesteps=time_steps, 
                  log_interval=log_interval, callback=callback)
      model.save("dqn_lunar_%0.1e_%s" % (lr, '-'.join(map(str, nn_layers))))
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      print(x)
      # times[i_lr, j_nn, :] =  x
      # episode_rewards[i_lr, j_nn, :] = y
      times.append(x)
      episode_rewards.append(y)
      del model
  
  data_dir = "/tmp/hyperparamtuning/"
  os.makedirs(data_dir, exist_ok=True)

  # file = open(os.path.join(data_dir, 'hyper-search-output.json'), 'w+')
  filepath = os.path.join(data_dir, 'hyper-search-output')
  data = {"times": times, "episode_rewards": episode_rewards, 
          "lr_array": lr_array, "nn_layers_array": nn_layers_array}
  np.savez(filepath, **data, allow_pickle=True)
  # json.dump(data, file)
  return

# Testing a single learning rate over architectures to avoid colab limits.
lr_array = [0.001]
# For more powerful computers use these
# lr_array = [0.001, 0.0005, 0.0001, 0.00001]

# Recommended sizes according to some sample param ranges in stable_baselines docs
nn_layers_array = [[32], [64], [128], [256], [32, 32], [64, 64], [128, 128], [256, 256]]

# Run the hyperparameter search
layers_lr_grid_search(lr_array, nn_layers_array)
# Models will be saved to current directory or home
# logs of each training in separate folders ""/tmp/gym/log_*"
# from logs the times and rewards are stored together in "/tmp/hyperparamtuning/hyper-search-output.npz"
