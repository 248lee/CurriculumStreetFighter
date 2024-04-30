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
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
    print(model.policy)
    top_kernels = model.get_parameters()['policy']['features_extractor.cnn_stage1.0.weight']
    print(top_kernels[0].shape) # (3, 8, 8)
    m = nn.Conv2d(3, 1, 8, padding='same', bias=False)
    target = 21
    m.load_state_dict({'weight': top_kernels[target].unsqueeze(0)}) # 4: shadow, 15: jumping enemy, 21: feat, 25: shadow
    avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    new_kernels = model.get_parameters()['policy']['features_extractor.cnn_stage2.0.weight']
    M = nn.Conv2d(3, 3, 8, groups=3, padding='same', bias=False)
    print("conv2 one kernel shape", M.state_dict()['weight'].shape) # [3, 1, 8, 8]
    M.load_state_dict({'weight': new_kernels[target * 3 : target * 3 + 3, :, :, :]})


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

            draw_obs = th.from_numpy(obs.transpose(2, 0, 1)).float()
            with th.no_grad():
                draw_obs_s1 = avg_pool(draw_obs)
            print(draw_obs_s1.shape)
            # Create a figure and axis objects
            fig, axes = plt.subplots(nrows=1, ncols=3)

            # Display the first image on the left
            axes[0].imshow(th.Tensor.numpy(draw_obs_s1).transpose(1, 2, 0) / 255)
            axes[0].set_title('Before Conv')
            with th.no_grad():
                draw_obs_s1 = m(draw_obs_s1)
                draw_obs_s1 = relu(draw_obs_s1)
            # Display the second image on the right
            pic = axes[1].imshow(th.Tensor.numpy(draw_obs_s1).transpose(1, 2, 0) / 255)
            axes[1].set_title('After Conv')

            with th.no_grad():
                draw_obs_s2 = M(draw_obs)
                draw_obs_s2 = relu(draw_obs_s2)
                draw_obs_s2 = maxpool(draw_obs_s2)
                draw_obs_s2 = m(draw_obs_s2)
                draw_obs_s2 = relu(draw_obs_s2)
            pic = axes[2].imshow(th.Tensor.numpy(draw_obs_s2).transpose(1, 2, 0) / 255)
            fig.colorbar(pic, ax=axes[2])
            axes[2].set_title('After 2 Convs')
            plt.show()
        # for i in range(3):
        #     plt.imshow(obs[:, :, i], cmap='gray')
        #     plt.show()
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