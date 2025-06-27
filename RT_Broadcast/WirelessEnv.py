'''
Environment to calculate the Whittle index values as a deep reinforcement
learning environment modelled after the OpenAi Gym API.
This is an AoI setting
The state x is the current AoI
The channel reliability is p
The reward of state x = -x
Transition prob is given by:
If activated: move to 1 w.p. p; move to x+1 w.p. 1-p
If not activated: move to x+1
'''

import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces


# from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal. requires python 3.6.

class RT_Multicast_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    This is the setting of multicasting real-time traffic 
    The delay = the period = d
    The channel reliability is p
    There are N users
    The reward is -1*(number of deadline violation)
    '''

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, seed, p, d, N):
        super(RT_Multicast_Env, self).__init__()
        self.seed = seed
        self.p = p
        self.d = d
        self.N = N
        self.myRandomPRNG = random.Random(self.seed)
        self.observationSize = 1
        self.time_to_deadline = self.d
        self.remaining_user = self.N
        self.action_space = spaces.Discrete(2)
        '''self.observation_space = spaces.Discrete(N)'''



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        self.time_to_deadline -= 1
        reward = 0
        if action == 1:
            self.remaining_user -= np.random.binomial(self.remaining_user, self.p)
        if self.time_to_deadline == 0:
            reward -= self.remaining_user
            self.remaining_user = self.N
            self.time_to_deadline = self.d
        nextState = [self.time_to_deadline, self.remaining_user]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.time_to_deadline = self.d
        self.remaining_user = self.N
        initialState = np.array([self.time_to_deadline, self.remaining_user], dtype=np.intc)

        return initialState


#########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 50
N = 100
OptX = 20
OptY = 50
env = gridEnv(seed = SEED, N = N, OptX = OptX, OptY = OptY)

observation = env.reset()


x = np.array([1,0,0,0,1])
x = np.tile(x, 100)
n_steps = np.size(x)

for step in range(n_steps):
    nextState, reward = env.step(x[step])
    print(f'action: {x[step]} nextstate: {nextState}  reward: {reward}')
    print("---------------------------------------------------------")
'''
