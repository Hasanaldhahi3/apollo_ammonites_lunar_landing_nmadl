"""
Minimum working example to train a Lunar Lander Agent using a DQN algorithm

Apollo Ammonites - NMA Deep Learning 2021
Team members:
    J. A. Moreno-Larios
    ADD YOURSELVES HERE

Requirements:
    xvfb - X-display for headless systems
    python-opengl - for movie rendering
    ffmpeg - for movie rendering
    stable-baseline3 - RNN framework
    box2d-py
    gym - Scenes
    pyvirtualdisplay
    matplotlib
    numpy
    - Note that there will be additional dependencies. pip, anaconda or 
        miniconda should be able to install them.
    - For miniconda, be sure to execure 
    'conda config --add channels conda-forge' to be able to 
    install software on an isolated environment
    - To have CUDA support, you'll need to run 
        conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
        https://pytorch.org/get-started/locally/

"""

# Imports
import io
import os
import glob
import torch
import base64
import stable_baselines3

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env

import gym
from gym import spaces
from apollo_lander import ApolloLander
from gym.wrappers import Monitor
from gym.envs.registration import register

# Plotting/Video functions
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay

display = Display(visible=0, size=(1400, 900))
display.start()

"""
We will get the reward values for different tests using our pretrained model
"""
def wrap_env(env, subdir):
    env = Monitor(env, './video/' + subdir, force = True)
    return env

"""
Create environment
"""
log_dir = "./tmp/gym/stable"
os.makedirs(log_dir, exist_ok=True)
env = gym.make('LunarLander-v2')
env = stable_baselines3.common.monitor.Monitor(env, log_dir )
model = DQN.load("./model_stable_avg_reward_300.zip",
                 env=env) # Load model

"""
We try the pretrained model here
"""
number_of_iterations = 500
total_rewards = [0] * number_of_iterations # Creates a list full of zeros
for index in range(0, number_of_iterations):
    observation = env.reset()
    while True:
        action, states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        total_rewards[index] += reward
        if done:
            break;
    print("Test number #" + str(index + 1) + " reward: " + str(total_rewards[index]))
env.close()

total_rewards = np.array(total_rewards)
average_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
print("Average reward: " + str(average_reward))
print("Reward standard deviation: " + str(std_reward))

"""
Rendering
"""
test_env = wrap_env(gym.make("LunarLander-v2"), "after_training_stable")
observation = test_env.reset()
total_reward = 0
while True:
  test_env.render()
  action, _states = model.predict(observation, deterministic=True)
  observation, reward, done, info = test_env.step(action)
  total_reward += reward
  if done:
    break;
print(total_reward)
test_env.close()