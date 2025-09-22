import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces

class AoIEnv_OnOff_Penalty(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, seed, p):
        super(AoIEnv_OnOff_Penalty, self).__init__()
        self.seed = seed
        self.p = p
        self.myRandomPRNG = random.Random(self.seed)
        self.X = 1                # Initial AoI
        self.on_off = 1           # Channel starts in 'On' state

        # Observation/state consists of [AoI, channel state]
        self.observationSize = 2
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([100, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

    def _calRewardAndState(self, action):
        """
        Reward is a function of AoI.
        We want to test the theory that since transmitting even the channel is off doesn't 
          have a reward penalty, the network learns to favor AoI a lot more than the channel quality.
        In this environment, if we attempt to transmit even when the channel is off, the reward decreases
          by a factor of 2.

        """
        reward = -1 * self.X

        nextX = self.X + 1


        if nextX > 100:
            nextX = 100


        next_on_off = self.on_off


        if action == 1:
            if self.on_off == 1:   # Only succeed if channel is On
                nextX = 1          # AoI reset to 1
            else:
                reward = reward * 2

        if self.on_off == 1:
            if self.myRandomPRNG.uniform(0, 1.0) < self.p:
                next_on_off = 0  # Transition from On → Off
        else:
            if self.myRandomPRNG.uniform(0, 1.0) < (1 - self.p):
                next_on_off = 1  # Transition from Off → On

        nextState = [nextX, next_on_off]
        return nextState, reward

    def step(self, action):
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        self.X = nextState[0]
        self.on_off = nextState[1]
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        self.X = 0
        self.on_off = 1
        initialState = np.array([self.X, self.on_off], dtype=np.float32)

        return initialState

