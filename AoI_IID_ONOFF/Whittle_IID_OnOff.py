
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()


class Whittle_IID_OnOff(object):
    """
    Class implementing a Whittle Index policy for the i.i.d. On-Off channel model.

    This is a multi-armed bandit setting with:
    - An Age of Information (AoI)-like state
    - On-Off channel for each arm
    - A fixed budget of arms to activate per step

    Whittle index is calculated analytically based on the state and reliability of each arm.
    """
    def __init__(self, state_dims, action_dims, hidden, cfg):
        """
        Initialize the Whittle policy.

        Args:
            nb_arms (int): number of arms
            budget (int): max number of arms to activate per step
            state_dims (list): dimensionality of state space for each arm
            action_dims (list): dimensionality of action space for each arm
            hidden (list): number of hidden units per layer (not used here)
            args: additional args with hyperparameters like discount factor
        """
        self.nb_arms = cfg['nb_arms']
        self.budget = cfg['budget']
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.s_t = []  # Most recent observed states for each arm
        self.a_t = []  # Most recent selected actions for each arm
        self.p = []    # Channel reliability for each arm

        # Initialize channel reliability `p` for each arm linearly between [0.2, 0.8]
        for arm in range(nb_arms):
            self.p.append( 0.2 + 0.6/nb_arms*arm )
            self.s_t.append(None)
            self.a_t.append(None)

        self.discount = cfg['discount'] # Discount factor for future rewards
        self.is_training = True         # Whether this policy is in training mode

    def update_policy(self):
        """
        Placeholder: No training is performed in this analytic policy.
        """
        return

    def eval(self):
        """
        Placeholder: No evaluation logic needed.
        """
        return

    def cuda(self):
        """
        Placeholder: No use of GPU in this policy.
        """
        return

    def observe(self, r_t, s_t1, done):
        """
        Observe new state for each arm. Ignores rewards and done flag.

        Args:
            r_t (list): reward (not used)
            s_t1 (list): list of next states for each arm
            done (bool): terminal flag (not used)
        """
        for arm in range(self.nb_arms):
            self.s_t[arm] = s_t1[arm]

    def random_action(self):
        """
        Returns action selected based on current policy (not truly random).
        """
        return self.select_action(self.s_t)

    def select_action(self, s_t, decay_epsilon=True):
        """
        Select actions based on computed Whittle indices.

        Args:
            s_t (list): current state for each arm
            decay_epsilon (bool): unused, kept for API compatibility

        Returns:
            actions (list): binary list indicating whether to activate each arm
        """
        indices = []
        # Compute Whittle index for each arm based on its AoI and channel state
        for arm in range(self.nb_arms):
            index = 0
            index += s_t[arm][0] ** 2 / 2             # (x^2)/2
            index -= s_t[arm][0] / 2                  # -x/2
            index += s_t[arm][0] / (1 - self.p[arm])  # + x/(1-p)
            if s_t[arm][1] == 0:                      # If channel is Off, index is set to 0
                index = 0
            indices.append(index)

        # Sort indices in descending order
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)

        # Handle edge case when budget == nb_arms by adding a sentinel index
        sort_indices.append(
            sort_indices[self.nb_arms - 1] - 2)

        actions = []
        # Activate arms with top-k highest indices (where k = budget)
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        return actions

    def cal_index(self, s_t, decay_epsilon=True):
        """
        (Unused) Calculates alternative index formula, could be used for debugging or analysis.

        Args:
            s_t (list): state (AoI) values for each arm
            decay_epsilon (bool): unused

        Returns:
            indices (list): calculated indices for each arm (currently just s_t[arm])
        """
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
        """
        Resets the internal state of the agent to the initial observation.

        Args:
            obs (list): initial states for each arm
        """
        self.s_t = obs

