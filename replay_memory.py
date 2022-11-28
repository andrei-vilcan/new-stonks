import random
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

        self.ind = 0

    def store(self, states, actions, next_states, rewards, dones):
        if len(self.states) < self.capacity:
            self.states.append(states)
            self.actions.append(actions)
            self.next_states.append(next_states)
            self.rewards.append(rewards)
            self.dones.append(dones)
        else:
            self.states[self.ind] = states
            self.actions[self.ind] = actions
            self.next_states[self.ind] = next_states
            self.rewards[self.ind] = rewards
            self.dones[self.ind] = dones

        self.ind = (self.ind + 1) % self.capacity

    def sample(self, batchsize, device):
        indices_to_sample = random.sample(range(len(self.states)), k=batchsize)

        states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.dones)[indices_to_sample]).to(device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.states)