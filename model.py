import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# Convolution Model
class DQN(nn.Module):
    # def __init__(self, lr, input_dim, num_neurons, num_actions):
    def __init__(self, state_dims, action_dims, num_neurons, lr, batch_size):
        super(DQN, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.num_neurons = num_neurons
        self.batch_size = batch_size

        # Convolution Layers
        self.conv1 = nn.Conv1d(state_dims, state_dims * 2, kernel_size=2, groups=state_dims)
        self.conv2 = nn.Conv1d(state_dims * 2, state_dims * 4, kernel_size=2)
        self.conv3 = nn.Conv1d(state_dims * 4, state_dims * 8, kernel_size=2, groups=state_dims)
        self.conv4 = nn.Conv1d(state_dims * 8, state_dims * 16, kernel_size=2)
        self.conv_flat = nn.Flatten()

        # Dense Layers
        # With two convolutions change 256 input to 416
        self.fc1 = nn.Linear(448, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, self.action_dims)
        # self.fc3 = nn.Linear(4, self.action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Convert stonk state into tensor, for convolutions
        if x.shape == torch.Size([self.batch_size, self.state_dims]):
            x = torch.Tensor(x).view(1, self.state_dims, self.batch_size)
        else:
            x = torch.Tensor(x).view(self.batch_size, self.state_dims, self.batch_size)

        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(torch.relu(self.conv2(x)), 3)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool1d(torch.relu(self.conv4(x)), 3)
        x = self.conv_flat.forward(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        actions = torch.softmax(self.fc5(x), dim=1)

        return actions


# # FC Model
# class DQN(nn.Module):
#     # def __init__(self, lr, input_dim, num_neurons, num_actions):
#     def __init__(self, state_dims, action_dims, num_neurons, lr, batch_size):
#         super(DQN, self).__init__()
#         self.lr = lr
#         self.state_dims = state_dims
#         self.action_dims = action_dims
#         self.num_neurons = num_neurons
#         self.batch_size = batch_size
#
#         # Dense Layers
#         self.fc1 = nn.Linear(self.state_dims, self.num_neurons)
#         self.fc2 = nn.Linear(self.num_neurons, 2 * self.num_neurons)
#         self.fc3 = nn.Linear(2 * self.num_neurons, self.num_neurons)
#         self.fc4 = nn.Linear(self.num_neurons, round(self.num_neurons / 2))
#         self.fc5 = nn.Linear(round(self.num_neurons / 2), self.action_dims)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         self.loss = nn.MSELoss()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.to(self.device)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = F.leaky_relu(self.fc3(x))
#         x = F.leaky_relu(self.fc4(x))
#         actions = F.softmax(self.fc5(x))
#         return actions
