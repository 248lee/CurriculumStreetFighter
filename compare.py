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

RESET_ROUND = True  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

MODEL_NAME1 = r"ppo_ryu_john_value_regression_6000000_steps" # Specify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.
MODEL_NAME2 = r"transferred_model"
MODEL_NAME3 = r"ppo_ryu_john_honda_comes_lowres_11002432_steps"

# Model notes:
# ppo_ryu_2000000_steps_updated: Just beginning to overfit state, generalizable but not quite capable.
# ppo_ryu_2500000_steps_updated: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_ryu_3000000_steps_updated: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_ryu_7000000_steps_updated: Overfitted, dominates first round but not generalizable. 

NUM_EPISODES = 1 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"trained_models/"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING, load_state_name="Champion.Level12.RyuVsBison_test.state")
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsHonda_7.state")()
# model = PPO("CnnPolicy", env)

if MODEL_NAME1 != None:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME1), env=env)
    print(model.policy)

if MODEL_NAME2 != None:
    model2 = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME2), env=env)
    model3 = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME3), env=env)

obs, _info = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")
values = []
value2s = []
value3s = []

for _ in range(num_episodes):
    done = False
    
    if RESET_ROUND:
        obs, info = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()
        value = None
        value2 = None
        if MODEL_NAME2 != None:
            obs_tensor, _ = model2.policy.obs_to_tensor(obs)
            value2 = th.squeeze(model2.policy.predict_values(obs_tensor), 0)
            value3 = th.squeeze(model3.policy.predict_values(obs_tensor), 0)
        if MODEL_NAME1 == None:
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            value = th.squeeze(model.policy.predict_values(obs_tensor), 0)
            obs, reward, done, trunc, info = env.step(action)

        values.append(value.cpu().detach().numpy()[0])
        value2s.append(value2.cpu().detach().numpy()[0])
        value3s.append(value3.cpu().detach().numpy()[0])
        # for i in range(3):
        #     plt.imshow(obs[:, :, i], cmap='gray')
        #     plt.show()
        if reward != 0:
            total_reward += reward
            # print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
        
        # if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
        #     done = True

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
# if RANDOM_ACTION:
#     print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
print("Average reward for {}: {}".format(MODEL_NAME1, episode_reward_sum/num_episodes))
print("Average reward for {}: {}".format(MODEL_NAME2, episode_reward_sum/num_episodes))

plt.plot(range(len(values)), values)
plt.plot(range(len(value2s)), value2s)
plt.plot(range(len(value3s)), value3s)
plt.show()