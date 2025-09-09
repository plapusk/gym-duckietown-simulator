import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gym
import requests
import matplotlib.pyplot as plt
from gym_duckietown.envs import DuckietownEnv


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)
