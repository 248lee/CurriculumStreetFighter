import math
import cv2
import gymnasium as gym
import numpy as np

side_length_each_stage = [(0, 0), (80, 80), (160, 160), (160, 160), (160, 160)]
attack_buttons = [0, 1, 8, 9, 10, 11]

class StreetFighterPPOWrapper(gym.Wrapper):
    def __init__(self, env, now_stage, reset_round=True, rendering=False, num_frame_steps=3):
        gym.Wrapper.__init__(self, env)
        self.now_stage = now_stage
        self.reward_coeff = 3.0

        self.total_timesteps = 0
        self.num_step_frames = num_frame_steps
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
        obs, _info = self.reset()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs.shape[0], obs.shape[1], obs.shape[2]), dtype=np.uint8)

    def reset(self, **kwargs):
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        self.env.reset()
        obs, re, _, _, info = self.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        return obs, info

    def step(self, action):
        custom_done = False
        # encode the action from MultiBinary(8) to MultiBinary(12)
        obs, _reward, _done, _trun, info = self.env.step(action)

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()

        obs = []
        for i in range(self.num_step_frames - 1):
            # Keep the button pressed for (num_step_frames - 1) frames.
            image_data, _reward, _done, trun, info = self.env.step(action)
            if self.rendering:
                self.env.render()
            if i == self.num_step_frames // 2:
                # Middle Observation processing
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
                if self.now_stage != -1:
                    input_sidelength = side_length_each_stage[self.now_stage]
                    image_data = cv2.resize(image_data, (input_sidelength[0], input_sidelength[1]))
                obs.append(image_data.T)
        # Last Observation processing
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        if self.now_stage != -1:
            input_sidelength = side_length_each_stage[self.now_stage]
            image_data = cv2.resize(image_data, (input_sidelength[0], input_sidelength[1]))
        obs.append(image_data.T)
        obs = np.array(obs)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        round_countdown = info['round_countdown']
        
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

        elif round_countdown == 0:
            if curr_player_health > curr_oppont_health:
                custom_reward = 1
            else:
                custom_reward = -1
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            if custom_reward == 0:
                is_attacking = False
                for index in attack_buttons:
                    is_attacking = is_attacking or (action[index] == 1)
                if is_attacking:
                    custom_reward = 0
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
             
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001

        
        return obs, 0.001 * custom_reward, custom_done, trun, info # reward normalization
    