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

class AoIEnv_IID_OnOff(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    This is an AoI setting with i.i.d. unreliable channel
    The state has x and on_off
    The state x is the current AoI
    on_off = 1 if the channel is On
    The reward of state x = -x
    Transition prob is given by:
    If activated: move to 1 if the channel is On; move to x+1 otherwise
    If not activated: move to x+1
    The prob. of going from on to off is p
    The prob. of going from off to on is 1-p
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

    def __init__(self, seed, p):
        super(AoIEnv_IID_OnOff, self).__init__()
        self.seed = seed
        self.p = p
        self.myRandomPRNG = random.Random(self.seed)
        self.observationSize = 1
        self.X = 1
        self.on_off = 1
        self.action_space = spaces.Discrete(2)
        '''self.observation_space = spaces.Discrete(N)'''



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        reward = (-1)*self.X
        nextX = self.X + 1
        if nextX > 100:
            nextX = 100
        next_on_off = self.on_off
        if action == 1:
            if self.on_off == 1:
                nextX = 1
        if self.on_off == 1:
            if random.uniform(0, 1.0) < self.p:
                next_on_off = 0
        if self.on_off == 0:
            if random.uniform(0, 1.0) < 1 - self.p:
                next_on_off = 1
        nextState = [nextX, next_on_off]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        self.X = nextState[0]
        self.on_off = nextState[1]
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.X = 0
        self.on_off = 1
        initialState = np.array([self.X, self.on_off], dtype=np.intc)

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

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    This is a test environment that represents stock trading.
    The state has price which is the current price of the stock.
    If activated, we sell the stock otherwise we hold.
    When activated, the reward becomes price else 0.
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

    def __init__(self):
        super(StockTradingEnv, self).__init__()
        self.price = random.uniform(0, 100.0)
        self.action_space = spaces.Discrete(2)



    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        reward = 0
        if action == 1:
            reward = self.price

        nextPrice = random.uniform(0, 100.0)
            
        nextState = [nextPrice]
        return nextState, reward

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        self.price = nextState[0]
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.price = random.uniform(0, 100.0)
        initialState = np.array([self.price], dtype=np.intc)

        return initialState

#########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''

# env = StockTradingEnv()

# observation = env.reset()


# x = np.array([1,0,0,0,1])
# x = np.tile(x, 100)
# n_steps = np.size(x)

# for step in range(n_steps):
#     nextState, reward, _, _ = env.step(x[step])
#     print(f'action: {x[step]} nextstate: {nextState}  reward: {reward}')
#     print("---------------------------------------------------------")

