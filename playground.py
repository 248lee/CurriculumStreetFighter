import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(in_features=40, out_features=50)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.linear2 = nn.Linear(in_features=50, out_features=25)
        self.bn2 = nn.BatchNorm1d(num_features=25)
        self.linear3 = nn.Linear(in_features=25, out_features=1)

    def forward(self, input):
        out_list = []
        y = F.relu(self.bn1(self.linear1(input)))
        out_list.append(y)
        y = F.relu(self.bn2(self.linear2(y)))
        out_list.append(y)
        y = self.linear3(y)
        out_list.append(y)
        return out_list

model = Network()
x = torch.randn(10, 40)
output_list = model(x)

# Compute gradients for each output separately
output_list[0].mean().backward(retain_graph=True)
output_list[1].mean().backward(retain_graph=True)
output_list[2].mean().backward()

# Check gradients for each parameter
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
