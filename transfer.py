from train import CustomFeatureExtractorCNN, Stage2CustomFeatureExtractorCNN
from stable_baselines3 import PPO
import retro
from street_fighter_custom_wrapper import TransferStreetFighterCustomWrapper, StreetFighterCustomWrapper
from kernel_operations import john_bilinear
from network_structures import conv_stage1_kernels, conv_stage2_kernels
import torch as th
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from stable_baselines3.common.preprocessing import preprocess_obs

env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis", 
            state="Champion.Level12.RyuVsBison", 
            use_restricted_actions=retro.Actions.FILTERED, 
            render_mode='rgb_array'  
        )
env = TransferStreetFighterCustomWrapper(env)
model = PPO.load('trained_models/ppo_ryu_john_Dmall_long_rew_final.zip', env=env)
policy = model.policy
movie_obs = []
movie_label = []
movie_action = []
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.distributions import BernoulliDistribution
for _ in range(2):
    for i in range(1, 33): # 32 episodes
        env.reset(state='Champion.Level12.RyuVsBison_{}.state'.format(i))
        print('BATTLE:', i)
        done = False
        obs, info = env.reset()
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs_tensor, _ = policy.obs_to_tensor(obs)
            with th.no_grad():
                prob = policy.get_distribution(obs_tensor).distribution.probs
                prob = th.squeeze(prob, 0)  # shape [1, 12] -> [12]
                value = th.squeeze(policy.predict_values(obs_tensor), 0)
                label = th.cat((prob, value))  # combine value and probs to a new label, shape [13]
                movie_label.append(label)
            # print(movie_label[-1].shape)
            # input("hello there")
            obs_tensor = th.squeeze(obs_tensor, 0)  # reduce the dimension
            movie_obs.append(obs_tensor)
            movie_action.append(action)

            obs, reward, done, trunc, info = env.step(action)
            if reward != 0:
                total_reward += reward
                print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))
            
            if info['enemy_hp'] < 0 or info['agent_hp'] < 0:
                done = True
        if info['enemy_hp'] < 0:
            print("Victory!")
        else:
            print('Lose...')
        print("Total reward: {}\n".format(total_reward))
env.close()
ordered_dict_of_params = model.get_parameters()['policy']
itr = iter(ordered_dict_of_params.items())
old_top_kernel = next(itr)[1] # This gets the top item of the dict, which is the top kernel
old_top_bias = next(itr)[1] # This gets the second item of the dict, which is the top bias
interpolated_kernel = john_bilinear(old_top_kernel, old_top_bias, conv_stage2_kernels)
# class TrainingSetInputGenerator(nn.Module):
#     def __init__(self):
#         super(TrainingSetInputGenerator, self).__init__()
#         self.conv = nn.Conv2d(interpolated_kernel.shape[1], interpolated_kernel.shape[0], kernel_size=interpolated_kernel.shape[2], stride=1, padding='same', bias=False)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, stride=2)
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         return x
# class TrainingSetGroundGenerator(nn.Module):
#     def __init__(self):
#         super(TrainingSetGroundGenerator, self).__init__()
#         self.avg = nn.AvgPool2d(2, stride=2)
#         self.conv = nn.Conv2d(old_top_kernel.shape[1], old_top_kernel.shape[0], kernel_size=old_top_kernel.shape[2], stride=1, padding='same')
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.flatten = nn.Flatten(start_dim=0, end_dim=-1) # garbage pytorch, if you don't specify the start_dim, it only deals with the batched input instead of pure input
#     def forward(self, x):
#         x = self.avg(x)
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         return x

# ts_input_generator = TrainingSetInputGenerator().cuda()
# tmp = ts_input_generator.state_dict()
# tmp['conv.weight'] = th.from_numpy(interpolated_kernel)
# ts_input_generator.load_state_dict(tmp)

# ts_ground_generator = TrainingSetGroundGenerator().cuda()
# tmp = ts_ground_generator.state_dict()
# tmp['conv.weight'] = (old_top_kernel)
# tmp['conv.bias'] = (old_top_bias)
# ts_ground_generator.load_state_dict(tmp)

# training_set_input = []
# training_set_ground = []
# for mf in tqdm(movie_obs):
#     # mf = np.transpose(mf, [2, 0, 1])
#     # mf = th.from_numpy(mf).cuda().to(th.float32)
#     mf = preprocess_obs(mf, env.observation_space, normalize_images=True)
#     with th.no_grad():
#         mf = ts_input_generator(mf)
#     training_set_input.append(mf)
# if th.cuda.get_device_name(None) == 'NVIDIA GeForce GTX 1080 Ti': # 1080 Ti is too weak to operate this. CPU is needed
#     training_set_input_cpu = []
#     for i in range(len(training_set_input)):
#         training_set_input_cpu.append(training_set_input[i].cpu()) # copy it back to the cpu
#     del(training_set_input) # remove it from the gpu
#     training_set_input = training_set_input_cpu # rename

# for mo in tqdm(movie_obs):
#     mo = np.transpose(mo, [2, 0, 1])
#     mo = th.from_numpy(mo).cuda().to(th.float32)
#     mo = mo / 255
#     with th.no_grad():
#         mo = ts_ground_generator(mo)
#     training_set_ground.append(mo)
# del(movie_obs)
# if th.cuda.get_device_name(None) == 'NVIDIA GeForce GTX 1080 Ti': # 1080 Ti is too weak to operate this. CPU is needed
#     training_set_ground_cpu = []
#     for i in range(len(training_set_ground)):
#         training_set_ground_cpu.append(training_set_ground[i].cpu()) # copy it back to the cpu
#     del(training_set_ground) # remove it from the gpu
#     training_set_ground = training_set_ground_cpu

# print('input shape', training_set_input[0].shape)
# print('ground shape', training_set_ground[0].shape)

from train import make_env
env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis", 
            state="Champion.Level12.RyuVsBison", 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
env = StreetFighterCustomWrapper(env)
policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[], vf=[]),
        features_extractor_class=Stage2CustomFeatureExtractorCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler
lr_schedule = linear_schedule(7e-5, 2.5e-6)
clip_range_schedule = linear_schedule(0.15, 0.025)
model2 = PPO(
    "CnnPolicy", 
    env,
    device="cuda", 
    verbose=1,
    n_steps=384,
    batch_size=384,
    n_epochs=4,
    gamma=0.94,
    learning_rate=lr_schedule,
    clip_range=clip_range_schedule,
    tensorboard_log="logs",
    policy_kwargs=policy_kwargs
)
model2 = PPO(
    "CnnPolicy", 
    env,
    policy_kwargs=policy_kwargs
)
print(model2.policy)

old_params = model.get_parameters()['policy']
from collections import OrderedDict
old_params_toload = OrderedDict()
old_params_toload['policy'] = OrderedDict()
i = 0
for key, value in old_params.items():
    if "cnn_stage1" in key:
        i += 1
        continue
    print(i, key)
    old_params_toload['policy'][key] = value
    i += 1

model2.set_parameters(old_params_toload, exact_match=False)
new_params = model2.get_parameters()['policy']
new_params_toload = OrderedDict()
new_params_toload['policy'] = OrderedDict()
for key, value in new_params.items():
    if "cnn_stage2.0.weight" in key:
        new_params_toload['policy'][key] = th.from_numpy(interpolated_kernel)
    # elif "bn" in key:
    #     if key.rsplit('.', 1)[-1] in bn_dict:
    #         new_params_toload['policy'][key] = bn_dict[key.rsplit('.', 1)[-1]] # get each parameters of bn
    # elif "cnn_stage1.0.weight" in key:
    #     new_params_toload['policy'][key] = trained_conv_weight
    # elif "cnn_stage1.0.bias" in key:
    #     new_params_toload['policy'][key] = trained_conv_bias
    else:
        new_params_toload['policy'][key] = value
model2.set_parameters(new_params_toload, exact_match=False)

obs, info = env.reset()
obs_tensor, _ = model2.policy.obs_to_tensor(obs)
# print(policy.get_distribution(obs_tensor).distribution.probs.shape)
# input("hello here")

print(model2.get_parameters()['policy']['features_extractor.cnn_stage1.0.weight'][0][0])
from kernel_operations import transfer
transferred_policy = transfer(model2.policy, movie_obs, movie_label)

model2.policy = transferred_policy
print("=======")
print(model2.get_parameters()['policy']['features_extractor.cnn_stage1.0.weight'][0][0])

model2.save('trained_models/transferred_model.zip')
# printing the norm of the weightings
check_model = model2.get_parameters()['policy']
print(np.linalg.norm(check_model['features_extractor.cnn_stage2.0.weight'].cpu().numpy()))
print(np.linalg.norm(check_model['features_extractor.cnn_stage1.0.weight'].cpu().numpy()))
print(np.linalg.norm(check_model['features_extractor.cnn.0.weight'].cpu().numpy()))
print(np.linalg.norm(check_model['features_extractor.cnn.3.weight'].cpu().numpy()))
