"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)
from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

import retro

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, now_stage, state=None, max_episode_steps=None, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StreetFighterCustomWrapper(env)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    #env = WarpFrame(env, 50, 50) # resolution of the game
    env = ClipRewardEnv(env)
    return env

class CustomFeatureExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(self.cnn_stage1(
                th.as_tensor(observation_space.sample()[None]).float()
            )).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(self.cnn_stage1(observations)))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="StreetFighterIISpecialChampionEdition-Genesis")
    parser.add_argument("--state", default="Round1")
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, now_stage=1, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env

    # te = make_env()
    # from stable_baselines3.common.env_checker import check_env
    # check_env(te)
    # input()
    venv = (VecFrameStack(SubprocVecEnv([make_env] * 12), n_stack=2))
    # venv = (VecFrameStack(DummyVecEnv([make_env]), n_stack=4))
    # venv.reset()
    # obs, re, _, _ = venv.step([venv.action_space.sample()])
    # print(obs.shape)
    # input()
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[96], vf=[32]),
        features_extractor_class=CustomFeatureExtractorCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=1e-05,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.8,
        clip_range=0.15,
        tensorboard_log='./logs/sf_tensorboard'
    )
    # model2 = PPO.load('testModel.zip')
    # model.set_parameters(model2.get_parameters())
    print(model.policy)
    input()
    model.learn(
        total_timesteps=2000000,
        progress_bar=True,
        tb_log_name='sf2_1S'
    )
    model.save('testModel.zip')
    model.learning_rate = 1e-06
    model.learn(
        total_timesteps=8000000,
        progress_bar=True,
        tb_log_name='sf2_1S'
    )
    model.save('testModel.zip')


if __name__ == "__main__":
    main()
