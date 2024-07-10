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
import sys

import retro
from vppo import VPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
import torch as th


from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from value_transfer_policy import TransferActorCriticPolicy
from network_structures import WhiteFeatureExtractorCNN

NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            render_mode='rgb_array'  
        )
        env = StreetFighterCustomWrapper(env, enemy=1)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init



def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = (SubprocVecEnv([make_env(game, state="Champion.Level12.RyuVsBison", seed=i) for i in range(NUM_ENV)]))

    # Set linear schedule for learning rate
    lr_schedule = linear_schedule(1e-4, 5e-5)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    clip_range_schedule = linear_schedule(0.2, 0.02)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    
    custom_objects = {
    "learning_rate": lr_schedule,
    "clip_range": clip_range_schedule,
    "n_steps": 512,
    "n_epochs": 4,
    "gamma": 0.94,
    "batch_size": 512,
    "tensorboard_log": "logs",
    "verbose": 1
    }
    model = VPPO.load('trained_models/ppo_chun_vs_ryu_john_final.zip', device="cuda", custom_objects=custom_objects, env=env)

    # input("Press ENTER to continue...")
    # Set the save directory
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models/ppo_ryu_7000000_steps.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    #     "n_steps": 512
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set up callbacks
    # Note that 1 timesetp = 6 frame
    checkpoint_interval = 31250 * 2 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    ExperimentName = "value_transfer"
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix=ExperimentName)

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    print('start training')
    model.learn(
        total_timesteps=int(500000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
        callback=[checkpoint_callback],#, stage_increase_callback]
        progress_bar=True,
        tb_log_name=ExperimentName,
    )
    env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, ExperimentName + "_final.zip"))

if __name__ == "__main__":
    main()