from sklearn.cluster import KMeans
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import torch.nn as nn
import torch as th
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2

conv_stage3_kernels = 64
conv_stage2_kernels = 64
conv_stage1_kernels = 64
conv_kernels = 64

def john_bilinear(oarr, obias, new_num_of_kernels):
  oarr = oarr.cpu().numpy()
  obias = obias.cpu().numpy()
  num_of_kernels = oarr.shape[0]
  interpolated_kernels = []
  for i in range(num_of_kernels):
    num_of_channels = oarr[i].shape[0]
    interpolated_piece = []
    for j in range(num_of_channels):
      old_kernel = oarr[i][j]
      # print(old_kernel)
      # print(old_kernel)
      # x = np.linspace(0, 1, old_kernel.shape[0])
      # y = np.linspace(0, 1, old_kernel.shape[1])

      # interp = RegularGridInterpolator((x, y), old_kernel)

      # x_i = np.linspace(0, 1, old_kernel.shape[0] * 2) # therefore, the shape of the interpolated kernel must be even, because of * 2
      # y_i = np.linspace(0, 1, old_kernel.shape[1] * 2)
      # x_i, y_i = np.meshgrid(x_i, y_i)
      # points = np.vstack([x_i.ravel(), y_i.ravel()]).T
      # z_i = interp(points)
      # z_i = z_i.reshape(x_i.shape)
      z_i = cv2.resize(old_kernel, (old_kernel.shape[0] * 2, old_kernel.shape[1] * 2), interpolation=cv2.INTER_CUBIC)
      interpolated_piece.append(z_i)
    interpolated_piece = np.array(interpolated_piece)
    interpolated_kernels.append(interpolated_piece)

  # Start Cut the Kernels
  cut_kernels = []
  for ik in interpolated_kernels: # for each (4, 6, 6) kernel
    print(ik.shape[1] / 2)
    for x in range(0, ik.shape[1], ik.shape[1] // 2): # x will be 0 or 3
      for y in range(0, ik.shape[2], ik.shape[2] // 2): # y will be 0 or 3
        cut_pieces = [] # collect all channels
        for i in range(ik.shape[0]): # iterate through channels, aka [0, 1, 2, 3]
          cut_piece = np.zeros((ik[i].shape[0] // 2, ik[i].shape[1] // 2))
          for j in range(0, ik[i].shape[0] // 2): # iterate through the side of the kernel, aka [0, 1, 2]
            for k in range(0, ik[i].shape[1] // 2): # iterate through the side of the kernel, aka [0, 1, 2]
              cut_piece[j][k] = ik[i][j + x][k + y] # fill the piece. Remember to add the offset x and y
          cut_pieces.append(cut_piece)
        cut_pieces = np.array(cut_pieces) # one kernel has finished cutting! Ready to push? GO!!!
        cut_kernels.append(cut_pieces)
  print((interpolated_kernels[5][0]))
  print('==================================')
  print((cut_kernels[20][0]))
  print((cut_kernels[21][0]))
  print((cut_kernels[22][0]))
  print((cut_kernels[23][0]))
  print('===============================')
  # print(z_i)
  ls = []
  for i in range(len(cut_kernels)):
    ls.append(cut_kernels[i].reshape(-1))
  # print((cut_kernels[23]))
  # print(ls[23])
  # ls = np.array(ls)
  # print(ls.shape)
  kmeans = KMeans(n_clusters=new_num_of_kernels,n_init='auto',random_state=10,max_iter=1000)
  kmeans.fit(ls)
  result = kmeans.cluster_centers_
  result = result.reshape((result.shape[0], cut_kernels[0].shape[0], cut_kernels[0].shape[1], cut_kernels[0].shape[2]))
  print(result.shape)
  return result

# class TransferModel(nn.Module):
#   def __init__(self, input_shape, num_of_filters) -> None:
#     super(TransferModel, self).__init__()
#     num_of_t_input_channels = input_shape[0]
#     self.bn = nn.BatchNorm2d(num_of_t_input_channels, affine=False)
#     self.conv = nn.Conv2d(num_of_t_input_channels, num_of_filters, kernel_size=8, stride=1, padding='same')
#     self.relu = nn.ReLU()
#     self.pool = nn.MaxPool2d(2, stride=2)
#     self.flatten = nn.Flatten()

#   def forward(self, x):
#     x = self.bn(x)
#     x = self.conv(x)
#     x = self.relu(x)
#     x = self.pool(x)
#     x = self.flatten(x)
#     return x

class TransferDataset(Dataset):
    def __init__(self, data_list, label_list):
        """
        Args:
            data_list (list of torch.Tensor): List of input data tensors
            label_list (list): List of corresponding labels
        """
        assert len(data_list) == len(label_list), "Data and label lists must have the same length"
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label


def transfer(stage2_policy, training_set_inputs, training_set_grounds):
  loss_fn = nn.MSELoss()
  # trans_model = TransferModel(training_set_inputs[0].shape, num_of_filters).cuda()
  trans_model = stage2_policy
  optimizer = th.optim.Adam(trans_model.parameters(), lr=1e-06)
  EPOCH = 60 
  REPORT_DUR = 4
  training_dataset = TransferDataset(data_list=training_set_inputs, label_list=training_set_grounds)
  training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)

  # Freeze the layers except cnn_stage1
  for name, param in trans_model.named_parameters():
    if not (("cnn_stage1" in name) or ("bn" in name)):
      param.requires_grad = False

  for e in range(EPOCH):
    with tqdm(total=len(training_loader)) as pbar:
      running_loss = 0
      last_loss = 0
      for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        prob = trans_model.get_distribution(inputs).distribution.probs
        value = trans_model.predict_values(inputs)
        outputs = th.cat((prob, value), dim=1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Print the gradients to check
        # for name, param in trans_model.named_parameters():
        #   print(f"{name} | {param.grad}")
        # input()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        pbar.set_description('EPOCH {},  batch {} loss: {}'.format(e, i, avg_loss))
        pbar.update()
    # pbar.clear()
    
  # Unfreeze the layers
  # Freeze the layers except cnn_stage2
  for name, param in trans_model.named_parameters():
    if (("cnn_stage2" in name)):
      param.requires_grad = False
    else:
      param.requires_grad = True

  print(training_set_grounds[76])
  # Fine Tune
  with th.no_grad():
    training_set_grounds_finetune = []
    for i in range(len(training_set_grounds)):
      multiplier = 0.77
      for j in range(12):
        training_set_grounds[i][j] = (training_set_grounds[i][j] - 0.5) * multiplier + 0.5
      training_set_grounds_finetune.append(training_set_grounds[i])  # e.g. if the prob is 1, it will be converted into 0.5+0.5*multiplier
  training_dataset_finetune = TransferDataset(data_list=training_set_inputs, label_list=training_set_grounds_finetune)
  training_loader_finetune = DataLoader(training_dataset_finetune, batch_size=8, shuffle=True)
  print(training_set_grounds_finetune[76])
  EPOCH = 15
  for e in range(EPOCH):
    with tqdm(total=len(training_loader_finetune)) as pbar:
      running_loss = 0
      last_loss = 0
      for i, data in enumerate(training_loader_finetune):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        prob = trans_model.get_distribution(inputs).distribution.probs
        value = trans_model.predict_values(inputs)
        outputs = th.cat((prob, value), dim=1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Print the gradients to check
        # for name, param in trans_model.named_parameters():
        #   print(f"{name} | {param.grad}")
        # input()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        pbar.set_description('EPOCH {},  batch {} loss: {}'.format(e, i, avg_loss))
        pbar.update()
  
  return trans_model