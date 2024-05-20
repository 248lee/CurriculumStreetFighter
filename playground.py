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

import os
import time 

import retro
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
import torch as th
import torch.nn as nn
import numpy as np
from network_structures import Stage2CustomFeatureExtractorCNN
from collections import OrderedDict

RESET_ROUND = False  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

MODEL_NAME = r"transferred_model" # Specify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.

# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

RANDOM_ACTION = False
NUM_EPISODES = 30 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"trained_models/"

relu = nn.ReLU()
maxpool = nn.MaxPool2d(2, stride=2)

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsBison_test")()
# model = PPO("CnnPolicy", env)

if not RANDOM_ACTION:
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[], vf=[]),
        features_extractor_class=Stage2CustomFeatureExtractorCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO(
        "CnnPolicy", 
        env,
        policy_kwargs=policy_kwargs
    )

    frame_kernels = [OrderedDict() for i in range(3)]
    for i in range(3):
        frame_kernels[i]['weight'] = (2**i) * th.ones([32, 1, 8, 8])
    model.policy.features_extractor.cnn_stage2_sub1_input_1[0].load_state_dict(frame_kernels[0], strict=False)
    model.policy.features_extractor.cnn_stage2_sub2_input_1[0].load_state_dict(frame_kernels[0], strict=False)
    model.policy.features_extractor.cnn_stage2_sub3_input_1[0].load_state_dict(frame_kernels[0], strict=False)
    model.policy.features_extractor.cnn_stage2_sub4_input_1[0].load_state_dict(frame_kernels[0], strict=False)

    model.policy.features_extractor.cnn_stage2_sub1_input_2[0].load_state_dict(frame_kernels[1], strict=False)
    model.policy.features_extractor.cnn_stage2_sub2_input_2[0].load_state_dict(frame_kernels[1], strict=False)
    model.policy.features_extractor.cnn_stage2_sub3_input_2[0].load_state_dict(frame_kernels[1], strict=False)
    model.policy.features_extractor.cnn_stage2_sub4_input_2[0].load_state_dict(frame_kernels[1], strict=False)

    model.policy.features_extractor.cnn_stage2_sub1_input_3[0].load_state_dict(frame_kernels[2], strict=False)
    model.policy.features_extractor.cnn_stage2_sub2_input_3[0].load_state_dict(frame_kernels[2], strict=False)
    model.policy.features_extractor.cnn_stage2_sub3_input_3[0].load_state_dict(frame_kernels[2], strict=False)
    model.policy.features_extractor.cnn_stage2_sub4_input_3[0].load_state_dict(frame_kernels[2], strict=False)

    merge_weight = th.zeros([96, 4, 1, 1])
    merge_weight[0,:,:,:] = 1 / 4
    merge_weight[1,:,:,:] = 1 / 8
    merge_weight[2,:,:,:] = 1 / 16
    merge_kernel = OrderedDict()
    merge_kernel['weight'] = merge_weight
    model.policy.features_extractor.merge_conv.load_state_dict(merge_kernel, strict=False)


obs, _info = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False
    
    if RESET_ROUND:
        obs, info = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)

            
        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        
        if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
            done = True

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory += 1

    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        while info['enemy_hp'] < 0 or info['agent_hp'] < 0:
        # Inter scene transition. Do nothing.
            obs, reward, done, trunc, info = env.step([0] * 12)
            env.render()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))