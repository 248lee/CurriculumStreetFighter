import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

conv_stage1_kernels = 32
conv_stage2_kernels = 384

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
        self.avg = nn.AvgPool2d(2, stride=2)
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(n_input_channels, conv_stage1_kernels, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(conv_stage1_kernels, conv_stage1_kernels, kernel_size=8, stride=1, padding='same'), # (32, 100, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # (32, 50, 64)
            nn.Conv2d(conv_stage1_kernels, 64, kernel_size=4, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # (64, 25, 32)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # (64, 12, 16)
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(self.cnn_stage1(self.avg(
                th.as_tensor(observation_space.sample()[None]).float()
            ))).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(self.cnn_stage1(self.avg(observations))))

class Stage2CustomFeatureExtractorCNN(BaseFeaturesExtractor):
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
        # Define each sub-network
        self.cnn_stage2_sub1_input_1 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 32, H, W)
        )

        self.cnn_stage2_sub2_input_1 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub3_input_1 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub4_input_1 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        # Define each sub-network
        self.cnn_stage2_sub1_input_2 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub2_input_2 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub3_input_2 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub4_input_2 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        # Define each sub-network
        self.cnn_stage2_sub1_input_3 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub2_input_3 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub3_input_3 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )

        self.cnn_stage2_sub4_input_3 = nn.Sequential(
            nn.Conv2d(1, conv_stage2_kernels // 12, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # Output: (conv_stage2_kernels // 12, H, W)
        )
        # Define a 1x1 convolutional layer to merge the outputs of the four sub-networks
        self.merge_conv = nn.Conv2d(in_channels=384, out_channels=96, kernel_size=1, groups=96)
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(32 * 3, conv_stage1_kernels, groups=32, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(conv_stage1_kernels, conv_stage1_kernels, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(conv_stage1_kernels, 64, kernel_size=4, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            input_sub = th.as_tensor(observation_space.sample()[None]).float()
            output_sub1_input_1 = self.cnn_stage2_sub1_input_1(input_sub[:,0:1])
            output_sub2_input_1 = self.cnn_stage2_sub2_input_1(input_sub[:,0:1])
            output_sub3_input_1 = self.cnn_stage2_sub3_input_1(input_sub[:,0:1])
            output_sub4_input_1 = self.cnn_stage2_sub4_input_1(input_sub[:,0:1])
            output_sub1_input_2 = self.cnn_stage2_sub1_input_2(input_sub[:,1:2])
            output_sub2_input_2 = self.cnn_stage2_sub2_input_2(input_sub[:,1:2])
            output_sub3_input_2 = self.cnn_stage2_sub3_input_2(input_sub[:,1:2])
            output_sub4_input_2 = self.cnn_stage2_sub4_input_2(input_sub[:,1:2])
            output_sub1_input_3 = self.cnn_stage2_sub1_input_3(input_sub[:,2:3])
            output_sub2_input_3 = self.cnn_stage2_sub2_input_3(input_sub[:,2:3])
            output_sub3_input_3 = self.cnn_stage2_sub3_input_3(input_sub[:,2:3])
            output_sub4_input_3 = self.cnn_stage2_sub4_input_3(input_sub[:,2:3])
            print(output_sub1_input_1.shape)
            print(output_sub1_input_1)
            print(output_sub2_input_1.shape)
            print(output_sub2_input_1)
            print(output_sub3_input_1.shape)
            print(output_sub4_input_1.shape)
            print(output_sub1_input_2.shape)
            print(output_sub2_input_2.shape)
            print(output_sub3_input_2.shape)
            print(output_sub4_input_2.shape)
            print(output_sub1_input_3.shape)
            print(output_sub2_input_3.shape)
            print(output_sub3_input_3.shape)
            print(output_sub4_input_3.shape)
            alternating_outputs = []
            for i in range(output_sub1_input_1.size(1)):
                alternating_outputs.extend([output_sub1_input_1[:,i], output_sub2_input_1[:,i],output_sub3_input_1[:,i], output_sub4_input_1[:,i],output_sub1_input_2[:,i], output_sub2_input_2[:,i],output_sub3_input_2[:,i], output_sub4_input_2[:,i],output_sub1_input_3[:,i], output_sub2_input_3[:,i],output_sub3_input_3[:,i], output_sub4_input_3[:,i]])
            print(alternating_outputs[0].shape)
            # 將列表轉換為張量，並在通道維度上進行堆疊
            concatenated_output = th.stack(alternating_outputs, dim=1)
            print(concatenated_output.shape)
            # Merge the outputs using the 1x1 convolutional layer
            merged_output = self.merge_conv(concatenated_output)
            n_flatten = self.cnn(self.cnn_stage1(
                merged_output
            )).shape[1]
            # print("obs shape", (th.as_tensor(observation_space.sample()[None]).float()).shape)
            # print("cnn2 shape", (self.cnn_stage2(th.as_tensor(observation_space.sample()[None]).float())).shape)
            # input()

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, input_sub: th.Tensor) -> th.Tensor:
        # Forward pass through each sub-network
        output_sub1_input_1 = self.cnn_stage2_sub1_input_1(input_sub[:,0:1])
        output_sub2_input_1 = self.cnn_stage2_sub2_input_1(input_sub[:,0:1])
        output_sub3_input_1 = self.cnn_stage2_sub3_input_1(input_sub[:,0:1])
        output_sub4_input_1 = self.cnn_stage2_sub4_input_1(input_sub[:,0:1])
        output_sub1_input_2 = self.cnn_stage2_sub1_input_2(input_sub[:,1:2])
        output_sub2_input_2 = self.cnn_stage2_sub2_input_2(input_sub[:,1:2])
        output_sub3_input_2 = self.cnn_stage2_sub3_input_2(input_sub[:,1:2])
        output_sub4_input_2 = self.cnn_stage2_sub4_input_2(input_sub[:,1:2])
        output_sub1_input_3 = self.cnn_stage2_sub1_input_3(input_sub[:,2:3])
        output_sub2_input_3 = self.cnn_stage2_sub2_input_3(input_sub[:,2:3])
        output_sub3_input_3 = self.cnn_stage2_sub3_input_3(input_sub[:,2:3])
        output_sub4_input_3 = self.cnn_stage2_sub4_input_3(input_sub[:,2:3])
        # Concatenate outputs along channel dimension
        alternating_outputs = []
        for i in range(output_sub1_input_1.size(1)):
          alternating_outputs.extend([output_sub1_input_1[:,i], output_sub2_input_1[:,i],output_sub3_input_1[:,i], output_sub4_input_1[:,i],output_sub1_input_2[:,i], output_sub2_input_2[:,i],output_sub3_input_2[:,i], output_sub4_input_2[:,i],output_sub1_input_3[:,i], output_sub2_input_3[:,i],output_sub3_input_3[:,i], output_sub4_input_3[:,i]])

        # 將列表轉換為張量，並在通道維度上進行堆疊
        concatenated_output = th.stack(alternating_outputs, dim=1)
        #concatenated_output = th.cat((output_sub1_input_1, output_sub2_input_1, output_sub3_input_1, output_sub4_input_1,output_sub1_input_2, output_sub2_input_2, output_sub3_input_2, output_sub4_input_2,output_sub1_input_3, output_sub2_input_3, output_sub3_input_3, output_sub4_input_3), dim=0)
        # Merge the outputs using the 1x1 convolutional layer
        merged_output = self.merge_conv(concatenated_output)

        return self.linear(self.cnn(self.cnn_stage1(merged_output)))