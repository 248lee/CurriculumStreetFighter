# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

ABS_STATE_DIR = os.getcwd() + '/states/'
# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False, load_state_name="", enemy=0):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.finish_reward_coeff = 3

        self.total_timesteps = 0

        self.welfare_length = 50

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(200, 256, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering

        self.load_state_name = load_state_name
        self.enemy = enemy

    def seed(self, arg1):
        pass

    def close(self):
        self.env.close()
    
    def _stack_observation(self):
        result = np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return result

    def reset(self, seed=None):
        random.seed(seed)
        if self.load_state_name == "":
            s = random.randint(1, 20)
            if self.enemy == 0:
                self.env.load_state(ABS_STATE_DIR + 'Champion_RyuVSSagat_D3_' + str(s) + '.state')
            elif self.enemy == 1:
                self.env.load_state(ABS_STATE_DIR + 'Champion_RyuVSSagat_D4_' + str(s) + '.state')
            elif self.enemy == 2:
                self.env.load_state(ABS_STATE_DIR + 'Champion_RyuVSSagat_D5_' + str(s) + '.state')
            else:
                self.env.load_state(ABS_STATE_DIR + 'WRONG ENEMY.state')
        else:
            print("STATE DIR:", ABS_STATE_DIR + self.load_state_name)
            self.env.load_state(ABS_STATE_DIR + self.load_state_name)

        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0

        self.welfare_countdown = self.welfare_length
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        observation, info = self.env.reset()
        for _ in range(self.num_frames):
            # new_observation = pan_int_obs(observation)
            # self.frame_stack.append(new_observation)
            # self.frame_stack.append(observation)
            self.frame_stack.append((observation))
            

        return self._stack_observation(), info

    def step(self, action):
        custom_done = False

        obs, _reward, _done, _trunc, info = self.env.step(action)
        # obs = pan_int_obs(obs)
        self.frame_stack.append(obs)
        # self.frame_stack.append(obs[::2, ::2, :])# This is the old way of downsampling the image. Now we use avgpool in the neural network

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, _trunc, info = self.env.step(action)
            # obs = pan_int_obs(obs)
            self.frame_stack.append(obs)
            # self.frame_stack.append(obs[::2, ::2, :]) # This is the old way of downsampling the image. Now we use avgpool in the neural network
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        self.total_timesteps += self.num_step_frames
        if self.welfare_countdown == 0:
            custom_done = True
            # Game is over and player loses.
            if curr_player_health < 0:
                custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                    # If the opponent also has negative health points, it's a even game and the reward is +1.

            # Game is over and player wins.
            elif curr_oppont_health < 0:
                # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                    # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

                # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
                # 35097 -> 89 sec.  26901 -> 69 sec.
                # countdown_multiplier = min(max((35907 - info['round_countdown']) / (35907 - 26901), 0), 1.0)  # keep the multiplier between 0 and 1
                custom_reward = (self.full_hp * (curr_player_health + 1) / (self.full_hp + 1)) * self.finish_reward_coeff
            else:
                custom_reward = 0
        # If the fighting ends, wait for welfare
        elif curr_player_health < 0 or curr_oppont_health < 0 or info['round_countdown'] == 0:                
            # waiting for the welfare
            custom_reward = 0
            self.welfare_countdown -= 1
            custom_done = False
        # If the fighting is still going on
        else:
            # if info['agent_y'] <= 180:
            #     damage_faced_reward_multiplier = 3.5269
            # else:
            damage_faced_reward_multiplier = 2.5269
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health) * damage_faced_reward_multiplier
            
            if custom_reward == 0:
                # if info['isDown'] == 0:
                #     custom_reward = 0.5
                # else:
                custom_reward = 1
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
                     
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), (0.001) * custom_reward, custom_done, _trunc, info # reward normalization
    
class HoducanStreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False, load_state_name="", enemy=0):
        super(HoducanStreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 15

        self.reward_coeff = 3.0

        self.finish_reward_coeff = 3

        self.total_timesteps = 0

        self.welfare_length = 50

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(200, 256, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering

        self.load_state_name = load_state_name
        self.enemy = enemy

    def seed(self, arg1):
        pass

    def close(self):
        self.env.close()
    
    def _stack_observation(self):
        result = np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return result

    def reset(self, seed=None):
        observation, info = self.env.reset()
        random.seed(seed)
        if self.load_state_name == "":
            s = random.randint(1, 16)
            if self.enemy == 0:
                self.env.load_state('Champion.Level12.ChunVSRyu_D5_' + str(s) + '.state')
            else:
                self.env.load_state('Champion.Level12.ChunVSRyu_D7_' + str(s) + '.state')
        else:
            self.env.load_state(self.load_state_name)

        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0

        self.welfare_countdown = self.welfare_length
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            # new_observation = pan_int_obs(observation)
            # self.frame_stack.append(new_observation)
            # self.frame_stack.append(observation)
            self.frame_stack.append((observation))
            

        return self._stack_observation(), info

    def step(self, action):
        custom_done = False
        actions = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
                    np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float),
                    np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=float),
                    np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float)]
        obs, _reward, _done, _trunc, info = self.env.step(actions[0])
        # obs = pan_int_obs(obs)
        self.frame_stack.append(obs)
        # self.frame_stack.append(obs[::2, ::2, :])# This is the old way of downsampling the image. Now we use avgpool in the neural network

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for tt in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, _trunc, info = self.env.step(actions[tt + 1])
            # obs = pan_int_obs(obs)
            self.frame_stack.append(obs)
            # self.frame_stack.append(obs[::2, ::2, :]) # This is the old way of downsampling the image. Now we use avgpool in the neural network
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        self.total_timesteps += self.num_step_frames
        if self.welfare_countdown == 0:
            custom_done = True
            # Game is over and player loses.
            if curr_player_health < 0:
                custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                    # If the opponent also has negative health points, it's a even game and the reward is +1.

            # Game is over and player wins.
            elif curr_oppont_health < 0:
                # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                    # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

                # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
                # 35097 -> 89 sec.  26901 -> 69 sec.
                # countdown_multiplier = min(max((35907 - info['round_countdown']) / (35907 - 26901), 0), 1.0)  # keep the multiplier between 0 and 1
                custom_reward = (self.full_hp * (curr_player_health + 1) / (self.full_hp + 1)) * self.finish_reward_coeff
            else:
                custom_reward = 0
        # If the fighting ends, wait for welfare
        elif curr_player_health < 0 or curr_oppont_health < 0 or info['round_countdown'] == 0:                
            # waiting for the welfare
            custom_reward = 0
            self.welfare_countdown -= 1
            custom_done = False
        # If the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health) * 2.5269
            if custom_reward == 0:  # and info['agent_y'] >= 150:
                custom_reward = 2
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
                     
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), (0.001) * custom_reward, custom_done, _trunc, info # reward normalization
    
class TransferStreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(TransferStreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(200, 256, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering

    def seed(self, arg1):
        pass

    def close(self):
        self.env.close()
    
    def _stack_observation(self):
        result = np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return result

    def reset(self, seed=None, state='Champion.Level12.RyuVsBison_1.state'):
        observation, info = self.env.reset()
        self.env.load_state(state)
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            # new_observation = pan_int_obs(observation)
            # self.frame_stack.append(new_observation)
            # self.frame_stack.append(observation)
            self.frame_stack.append((observation))
            

        return self._stack_observation(), info

    def step(self, action):
        custom_done = False

        obs, _reward, _done, _trunc, info = self.env.step(action)
        # obs = pan_int_obs(obs)
        self.frame_stack.append(obs)
        # self.frame_stack.append(obs[::2, ::2, :])# This is the old way of downsampling the image. Now we use avgpool in the neural network

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, _trunc, info = self.env.step(action)
            # obs = pan_int_obs(obs)
            self.frame_stack.append(obs)
            # self.frame_stack.append(obs[::2, ::2, :]) # This is the old way of downsampling the image. Now we use avgpool in the neural network
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        
        self.total_timesteps += self.num_step_frames
        
        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                   # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
                     
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), 0.001 * custom_reward, custom_done, _trunc, info # reward normalization
    



shape = (100, 128)
x = np.linspace(0, 1, shape[0])
y = np.linspace(0, 1, shape[1])
x_i = np.linspace(0, 1, shape[0] * 2) # therefore, the shape of the interpolated kernel must be even, because of * 2
y_i = np.linspace(0, 1, shape[1] * 2)
x_i, y_i = np.meshgrid(x_i, y_i)
points = np.vstack([x_i.ravel(), y_i.ravel()]).T
def pan_int_obs(observation):
    '''pan'''
    observation = observation[::2, ::2, :]
    interp = RegularGridInterpolator((x, y), observation)
    z_i = interp(points)
    z_i = z_i.reshape((x_i.shape[0], x_i.shape[1], observation.shape[2]))
    z_i = np.transpose(z_i, [1, 0, 2])
    return z_i
    '''end_pan'''
    