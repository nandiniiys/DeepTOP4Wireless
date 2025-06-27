# This file implements the Deep Threshold-Optimal Policy (DeepTOP)


import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class LPQL(object):
    # nb_arms: number of arms
    # state_dims: a list of state dimensions, one for each arm
    # action_dims: a list of action space dimensions, one for each arm
    # hidden: a list of number of neurons in each hidden layer
    def __init__(self, nb_arms, budget, state_dims, action_dims, state_sizes, action_sizes, hidden, args):
        self.nb_arms = nb_arms
        self.budget = budget
        self.state_dims = state_dims
        self.action_dims = action_dims

        if args.seed > 0:
            self.seed(args.seed)

        self.critics = []
        self.critic_targets = []
        self.critic_optims = []
        self.memories = []
        self.random_processes = []
        self.s_t = []
        self.initial_state = []
        self.a_t = []
        # Create Actor and Critic Networks, one for each arm
        for arm in range(nb_arms):
            self.critics.append(
                Critic(self.state_dims[arm] + 1, 1, hidden))  # input is state and lambda, output is Q value
            self.critic_targets.append(Critic(self.state_dims[arm] + 1, 1, hidden))
            self.critic_optims.append(Adam(self.critics[arm].parameters(), lr=args.rate))

            hard_update(self.critic_targets[arm], self.critics[arm])

            # Create replay buffer
            self.memories.append(SequentialMemory(limit=args.rmsize, window_length=args.window_length))
            self.random_processes.append(
                OrnsteinUhlenbeckProcess(size=action_dims[arm], theta=args.ou_theta, mu=args.ou_mu,
                                         sigma=args.ou_sigma))
            self.s_t.append(None)  # Most recent state
            self.initial_state.append(None)
            self.a_t.append(None)  # Most recent action

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        #
        self.epsilon = 1.0
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

    def update_policy(self):
        for arm in range(self.nb_arms):
            # Sample batch
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memories[arm].sample_and_split(self.batch_size)

            price_batch = []
            next_action_batch = []
            net_reward_batch = []
            for i in range(self.batch_size):
                price_batch.append(np.random.uniform(-1., 1.))
                net_reward_batch.append(reward_batch[i] - price_batch[i] * action_batch[i])
                if self.critic_targets[arm](
                        [torch.FloatTensor(next_state_batch[i]), torch.FloatTensor([price_batch[i]]),
                         torch.FloatTensor([1])]) \
                        > self.critic_targets[arm](
                    [torch.FloatTensor(next_state_batch[i]), torch.FloatTensor([price_batch[i]]),
                     torch.FloatTensor([0])]):
                    next_action_batch.append(1)
                else:
                    next_action_batch.append(0)

            # convert all batches to tensors
            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.FloatTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)
            terminal_batch = torch.FloatTensor(terminal_batch)
            price_batch = torch.FloatTensor(price_batch).unsqueeze(dim=-1)
            next_action_batch = torch.FloatTensor(next_action_batch).unsqueeze(dim=-1)
            net_reward_batch = torch.FloatTensor(net_reward_batch)

            # Prepare for the target q batch
            next_q_values = self.critic_targets[arm]([next_state_batch, price_batch, next_action_batch])

            target_q_batch = net_reward_batch + self.discount * next_q_values

            # Critic update
            self.critics[arm].zero_grad()

            q_batch = self.critics[arm]([state_batch, price_batch, action_batch])

            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            self.critic_optims[arm].step()

            # Target update
            soft_update(self.critic_targets[arm], self.critics[arm], self.tau)

    def eval(self):
        for arm in range(self.nb_arms):
            self.critics[arm].eval()
            self.critic_targets[arm].eval()

    def cuda(self):
        for arm in range(self.nb_arms):
            self.critics[arm].cuda()
            self.critic_targets[arm].cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            for arm in range(self.nb_arms):
                self.memories[arm].append(self.s_t[arm], self.a_t[arm], r_t[arm], done[arm])
                self.s_t[arm] = s_t1[arm]

    def random_action(self):
        indices = []
        for arm in range(self.nb_arms):
            indices.append(np.random.uniform(-1., 1.))
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(-2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        return actions

    def J_calculate(self, price):
        J_sum = price * self.budget / (1 - self.discount)
        for arm in range(self.nb_arms):
            if self.critics[arm].forward([torch.FloatTensor(self.initial_state[arm]), torch.FloatTensor([price]),
                                          torch.FloatTensor([1])]) > \
                    self.critics[arm].forward([torch.FloatTensor(self.initial_state[arm]), torch.FloatTensor([price]),
                                               torch.FloatTensor([0])]):
                J_sum = J_sum + self.critics[arm].forward([torch.FloatTensor(self.initial_state[arm]),
                                                           torch.FloatTensor([price]), torch.FloatTensor([1])]).item()
            else:
                J_sum = J_sum + self.critics[arm].forward([torch.FloatTensor(self.initial_state[arm]),
                                                           torch.FloatTensor([price]), torch.FloatTensor([0])]).item()
        return J_sum

    def select_action(self, s_t, decay_epsilon=True):
        current_price = -1
        current_J = self.J_calculate(current_price)
        next_price = current_price + 0.01
        next_J = self.J_calculate(next_price)
        while next_J < current_J:
            current_price = next_price
            current_J = next_J
            next_price = next_price + 0.01
            next_J = self.J_calculate(next_price)

        indices = []
        for arm in range(self.nb_arms):
            indices.append(self.critics[arm].forward([torch.FloatTensor(self.s_t[arm]), torch.FloatTensor([current_price]),
                                                      torch.FloatTensor([1])]).item() - \
                           self.critics[arm].forward(
                               [torch.FloatTensor(self.s_t[arm]), torch.FloatTensor([current_price]),
                                torch.FloatTensor([0])]).item()
                           )
        sort_indices = indices.copy()
        sort_indices.sort(reverse=True)
        sort_indices.append(sort_indices[self.nb_arms - 1] - 2)  # Create an additional item to handle the case when budget = nb_arms
        actions = []
        for arm in range(self.nb_arms):
            if indices[arm] > sort_indices[self.budget]:
                actions.append(1)
                self.a_t[arm] = 1
            else:
                actions.append(0)
                self.a_t[arm] = 0
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return actions

    def reset(self, obs):
        self.s_t = obs
        self.initial_state = obs
        for arm in range(self.nb_arms):
            self.random_processes[arm].reset_states()

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)


'''TODOS:

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
'''
