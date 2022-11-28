import random
import torch
import torch.nn.functional as F
import numpy as np

from replay_memory import ReplayMemory
from model import DQN


class TradingAgent:
    def __init__(self, env, device, n_neurons, epsilon_max, epsilon_min, epsilon_decay,
                 memory_capacity, batch_size, discount=0.99, lr=1e-3):
        self.env = env
        self.observaiton_space = env.observation_space
        self.observation_space_size = env.observation_space_size
        self.action_space = env.action_space
        self.action_space_size = env.action_space_size

        self.device = device
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random_len = 3

        self.discount = discount
        self.lr = lr
        self.n_neurons = n_neurons

        self.memory_capacity = memory_capacity
        self.replay_memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.count = 0

        self.policy_network = DQN(self.observation_space_size[1], self.action_space_size[0], n_neurons, lr, batch_size)
        self.target_network = DQN(self.observation_space_size[1], self.action_space_size[0], n_neurons, lr, batch_size)
        self.target_network.eval()
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state):
        epsilon = max(self.epsilon, self.epsilon_min)
        if np.random.sample() < epsilon:
            return random.sample(self.action_space, 1)[0]

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action = torch.argmax(self.policy_network(state))

        return action.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batchsize):
        if len(self.replay_memory) < batchsize:
            return

        states, actions, next_states, rewards, dones = self.replay_memory.sample(batchsize, self.device)

        q_pred = self.policy_network.forward(states).gather(1, actions.view(-1, 1))
        q_target = self.target_network.forward(next_states).max(dim=1).values
        q_target[dones] = 0.0
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)

        self.policy_network.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_network.optimizer.step()

        # actions = actions.reshape((-1, 1))
        # rewards = rewards.reshape((-1, 1))
        # dones = dones.reshape((-1, 1))
        #
        # predicted_qs = self.policy_network(states)
        # predicted_qs.gather(1, actions)
        #
        # target_qs = self.target_network(next_states)
        # target_qs = torch.max(target_qs, dim=1).values
        # target_qs = target_qs.reshape((-1, 1))
        # target_qs[dones] = 0.0
        # y_js = rewards + self.discount * target_qs
        #
        # loss = F.mse_loss(predicted_qs, y_js)
        # self.policy_network.optimizer.zero_grad()
        # loss.backward()
        # self.policy_network.optimizer.step()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    def load(self, filename):
        self.policy_network.load_state_dict(torch.load(filename))
        self.policy_network.eval()
