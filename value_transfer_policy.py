from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from network_structures import CustomFeatureExtractorCNN, Stage2CustomFeatureExtractorCNN, Stage3CustomFeatureExtractorCNN


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        observation_space: spaces.Box,
    ):
        super().__init__()


        # Policy network
        self.j_policy_net = CustomFeatureExtractorCNN(observation_space)
        # Value network
        self.j_value_net = CustomFeatureExtractorCNN(observation_space)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.j_policy_net.latent_output_shape[1]
        self.latent_dim_vf = self.j_value_net.latent_output_shape[1]

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.j_policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.j_value_net(features)


class TransferActorCriticPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.observation_space)
