import numpy as np

import torch
import torch.nn as nn
import math
import torch.nn.functional as F



def fanin_init(size, fanin=None):
"""
Xavier-like initialization for linear layers.

Args:
    size (tuple): shape of the weight tensor
    fanin (int, optional): number of input units; if not provided, defaults to size[0]

Returns:
    torch.Tensor: initialized weights
"""
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
"""
Actor network for deterministic policy (used in DDPG).
Maps states to continuous actions or thresholds.
"""
    def __init__(self, nb_inputs, nb_outputs, hidden, init_w=5e-1):
    """
    Args:
        nb_inputs (int): dimensionality of input state
        nb_outputs (int): dimensionality of output action
        hidden (list): list of hidden layer sizes
        init_w (float): initialization range for output layer
    """
        super(Actor, self).__init__()
        self.fc = nn.ModuleList()

        # Build fully connected layers
        for layer in range(len(hidden)+1):
            if layer == 0:
                self.fc.append(nn.Linear(nb_inputs, hidden[0]))
            elif layer == len(hidden):
                self.fc.append(nn.Linear(hidden[layer-1], nb_outputs))
            else:
                self.fc.append(nn.Linear(hidden[layer-1], hidden[layer]))
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
    """
    Custom weight initialization.
    """
        for layer in range(len(self.fc)):
            if layer == len(self.fc)-1:
                self.fc[layer].weight.data.uniform_(-init_w, init_w)
            else:
                self.fc[layer].weight.data = fanin_init(self.fc[layer].weight.data.size())

    def forward(self, x):
    """
    Forward pass through the actor network.
    
    Args:
        x (torch.Tensor): input state
    
    Returns:
        torch.Tensor: action or threshold value
    """
        out = x
        for layer in range(len(self.fc)):
            out = self.fc[layer](out)
            if layer < len(self.fc)-1:
                out = self.relu(out)
        return out

class Critic(nn.Module):
"""
Critic network for DDPG.
Maps (state, action) pairs to Q-values.
    """
    def __init__(self, nb_inputs, nb_actions, hidden, init_w=5e-1):
    """
    Args:
        nb_inputs (int): dimensionality of state input
        nb_actions (int): dimensionality of action input
        hidden (list): list of hidden layer sizes
        init_w (float): initialization range for output layer
    """
        super(Critic, self).__init__()
        self.fc=nn.ModuleList()

        # Build fully connected layers
        for layer in range(len(hidden)+1):
            if layer == 0:
                self.fc.append(nn.Linear(nb_inputs, hidden[0]))
            elif layer == math.floor(len(hidden)/2):
            # Concatenate action at this layer
                self.layer_num_for_action = layer
                self.fc.append(nn.Linear(hidden[layer-1]+nb_actions, hidden[layer]))
            elif layer == len(hidden):
            # Final output layer: Q-value
                self.fc.append(nn.Linear(hidden[layer-1], 1))
            else:
                self.fc.append(nn.Linear(hidden[layer-1], hidden[layer]))
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
    """
    Custom weight initialization.
    """
        for layer in range(len(self.fc)):
            if layer == len(self.fc)-1:
                self.fc[layer].weight.data.uniform_(-init_w, init_w)
            else:
                self.fc[layer].weight.data = fanin_init(self.fc[layer].weight.data.size())

    def forward(self, xs):
    """
    Forward pass through the critic network.

    Args:
        xs (tuple): (state, price, action)
            - state: torch.Tensor of states
            - price: torch.Tensor of price signal
            - action: torch.Tensor of actions

    Returns:
        torch.Tensor: estimated Q-values
    """
        x, price, a = xs
        out = torch.cat([x, price], -1)
        for layer in range(len(self.fc)):
            if layer == self.layer_num_for_action:
            # Insert action input at designated layer
                out = self.fc[layer](torch.cat([out, a], -1))
            else:
                out = self.fc[layer](out)
            if layer < len(self.fc)-1:
                out = self.relu(out)
        return out

