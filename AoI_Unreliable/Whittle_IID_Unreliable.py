
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


class Whittle_IID_Unreliable(object):
    # nb_arms: number of arms
    # state_dims: a list of state dimensions, one for each arm
    # action_dims: a list of action space dimensions, one for each arm
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, nb_arms, budget, state_dims, action_dims, hidden, args):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.s_t = []
        self.a_t = []
        self.p = []
        # Create Actor and Critic Networks, one for each arm
        for arm in range(nb_arms):
            self.p.append( 0.2 + 0.6/nb_arms*arm )
            self.s_t.append(None)  # Most recent state
            self.a_t.append(None)  # Most recent action

        # Hyper-parameters
        self.discount = args.discount
        self.is_training = True

    def update_policy(self):
        return

    def eval(self):
        return

    def cuda(self):
        return

    def observe(self, r_t, s_t1, done):
        for arm in range(self.nb_arms):
            self.s_t[arm] = s_t1[arm]

    def random_action(self):
        return self.select_action(self.s_t)   # This policy don't do random action

    def select_action(self, s_t, decay_epsilon=True):
        indices = []
        for arm in range(self.nb_arms):
            sum1 = 0
            for k in range(1, 21):
                sum1 += (s_t[arm][0] + k)*((1 - self.p[arm])**(k-1))
            sum1 *= self.p[arm] * self.p[arm] * s_t[arm][0]
            sum2 = self.p[arm] * s_t[arm][0] * (s_t[arm][0]+1) / 2
            indices.append(sum1+sum2)
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(
            sort_indices[self.nb_arms - 1] - 2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        return actions

    def cal_index(self, s_t, decay_epsilon=True):
        indices = []
        for arm in range(self.nb_arms):
            sum1 = 0
            for k in range(1, 21):
                sum1 += (s_t[arm] + k) * ( (1 - self.p[arm]) ** (k-1) )
            sum1 *= self.p[arm] * self.p[arm] * s_t[arm]
            sum2 = self.p[arm] * s_t[arm] * (s_t[arm]+1) / 2
            indices.append(s_t[arm])
        return indices


    def reset(self, obs):
        self.s_t = obs

