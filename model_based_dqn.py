#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:49:49 2021 EST

@author: haoqisun
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN
from gym.envs.box2d.lunar_lander import *


class ModelBasedDQN(DQN):
    """
    We modify DQN class to make it model based
    where is the model is a physical model,
    taken from gym.envs.box2d.lunar_lander.LunarLander.step()
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L266
    """
    def _generate_from_model(self, Nsample=1, random_seed=None):
        """
        based on https://github.com/DLR-RM/stable-baselines3/blob/v1.1.0/stable_baselines3/common/off_policy_algorithm.py#L510
        
        Randomly sample state and action,
        run the model one step
        save (state, action, new state, reward) to buffer
        repeat above `Nsample` times
        """
        np.random.seed(random_seed)
        for _ in range(Nsample):
            # randomly sample a state by final state after running the model `Kinit` steps with random actions
            self.model.reset()
            Kinit = np.random.randint(1,1000)
            for i in range(Kinit):
                action = self.model.action_space.sample()
                self.model.step(action)
                
            # randomly sample an action
            action = self.model.action_space.sample()
            
            # run model one step
            new_obs, reward, done, infos = self.model(action)

            #save (state, action, new state, reward) to buffer
            self.replay_buffer_model.add(obs, new_obs, buffer_action, reward, done, infos)
                
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # add data generated from model
        #TODO Not implemented: self.learn_model_from_memory()
        #TODO Not implemented: self._generate_from_model(self.replay_buffer.size())
        #TODO Not implemented: self.replay_buffer_both  = combine_buffer(self.replay_buffer, self.replay_buffer_model)
        self.replay_buffer_both  = self.replay_buffer #TODO to be removed
        
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer_both.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
