import numpy as np
import events as e
import pickle
import time
import os
from datetime import datetime

from .callbacks import state_to_features, ACTIONS

# The Deep Part
import torch
import torch.nn as nnimport torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


"""
Test Agent Uno (Tau):

Using a Deep Q-Learning Network, roughly as sketched in the pytorch documentation under
https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py,

we will implement the following steps:

    Replay Memory: To store the Transitions our agent observes and reuse this data as part of the 
    training process.


"""
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000

class DQN(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim); 

    """
    Network architecture:

    We will experiment with different layer architectures and sizes,
    using ReLU activation functions throughout the network. The next 
    lines dynamically adjust the dimension size to the network layers.
    """
        super(DQN, self).__init__()

        # Define layers:
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for dim in layer_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # Pass through all hidden layers with ReLU
        for layer in self.layers:
            x = F.relu(layer(x))

        # Output layer
        return self.output_layer(x)
        

def setup_training(self):
    pass

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    pass

def end_of_round(self, last_game_state, last_action, events):
    pass

def train_dqn()

def reward_from_events(events) -> int:
    pass
