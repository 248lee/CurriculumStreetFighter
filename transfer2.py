from stable_baselines3 import PPO
from trppo import TRPPO
import retro
from street_fighter_custom_wrapper import TransferStreetFighterCustomWrapper
from network_structures import Stage2CustomFeatureExtractorCNN
import numpy as np
import torch as th

env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis", 
            state="Champion.Level12.RyuVsBison", 
            use_restricted_actions=retro.Actions.FILTERED, 
            render_mode='rgb_array'  
        )
env = TransferStreetFighterCustomWrapper(env)

value_model = TRPPO.load('trained_models/value_transferring_policy_regression_4000000_steps.zip', env=env)

policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[], vf=[]),
        features_extractor_class=Stage2CustomFeatureExtractorCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
model2 = PPO(
    "CnnPolicy", 
    env,
    policy_kwargs=policy_kwargs
)


model2.policy.features_extractor.load_state_dict(value_model.policy.mlp_extractor.j_value_net.state_dict())
model2.policy.action_net.load_state_dict(value_model.policy.action_net.state_dict())
model2.policy.value_net.load_state_dict(value_model.policy.value_net.state_dict())

model2.save('trained_models/transferred_model2.zip')
# printing the norm of the weightings
check_model = model2.get_parameters()['policy']
print(np.linalg.norm(check_model['features_extractor.cnn_stage2.0.weight'].cpu().numpy()))
print(np.linalg.norm(check_model['features_extractor.cnn_stage1.0.weight'].cpu().numpy()))
print(np.linalg.norm(check_model['features_extractor.cnn.0.weight'].cpu().numpy()))

