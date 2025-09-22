"""
Environment for computing Whittle index values in an Age of Information (AoI) setting.
Follows the OpenAI Gym interface for reinforcement learning environments.

Context:
- This models an unreliable channel that can be On or Off.
- The state includes:
    - x: current AoI (Age of Information)
    - on_off: channel state (1 = On, 0 = Off)

System Dynamics:
- Reward: r(x) = -x (the lower the AoI, the better)
- Channel transitions:
    - On → Off with probability `p`
    - Off → On with probability `1 - p`
- Action = 1 (transmit):
    - If channel is On → AoI resets to 1
    - If channel is Off → AoI increases by 1
- Action = 0 (do not transmit):
    - AoI increases by 1
"""

import gym
import math
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from gym import spaces

class AoIEnv_IID_OnOff(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, seed, p):
        """
        Initialize the AoI environment with a given random seed and channel reliability `p`.

        Args:
            seed (int): Seed for reproducible randomness
            p (float): Probability of transition from On → Off (and hence Off → On with 1-p)
        """
        super(AoIEnv_IID_OnOff, self).__init__()
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

        # Action space: 0 = do nothing, 1 = attempt to transmit
        self.action_space = spaces.Discrete(2)

    def _calRewardAndState(self, action):
        """
        Computes the reward and next state given the action taken.

        Args:
            action (int): 0 = no transmission, 1 = transmit if possible

        Returns:
            nextState (list): [next AoI, next channel state]
            reward (float): computed as -AoI
        """
        # Reward is negative AoI (we want AoI to be small)
        reward = -1 * self.X
        # Default AoI increase if no reset
        nextX = self.X + 1

        # Cap AoI to a maximum of 100
        if nextX > 100:
            nextX = 100

        # Initialize next channel state as current
        next_on_off = self.on_off

        # If attempting transmission
        if action == 1:
            if self.on_off == 1:   # Only succeed if channel is On
                nextX = 1          # AoI reset to 1

        # Update channel state (Markov transition)
        if self.on_off == 1:
            if self.myRandomPRNG.uniform(0, 1.0) < self.p:
                next_on_off = 0  # Transition from On → Off
        else:
            if self.myRandomPRNG.uniform(0, 1.0) < (1 - self.p):
                next_on_off = 1  # Transition from Off → On

        nextState = [nextX, next_on_off]
        return nextState, reward

    def step(self, action):
        """
        Executes one step in the environment.

        Args:
            action (int): 0 = do nothing, 1 = try to transmit

        Returns:
            nextState (list): updated state [AoI, channel state]
            reward (float): reward = -AoI
            done (bool): always False (no terminal state in this environment)
            info (dict): extra info (empty in this environment)
        """
        assert action in [0, 1]

        nextState, reward = self._calRewardAndState(action)
        self.X = nextState[0]
        self.on_off = nextState[1]
        done = False
        info = {}

        return nextState, reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state at the beginning of an episode.

        Returns:
            initialState (np.array): [AoI = 0, channel state = On]
        """
        self.X = 0
        self.on_off = 1
        initialState = np.array([self.X, self.on_off], dtype=np.float32)

        return initialState


#########################################################################################
# The below block is for testing/validation and is commented out.
# It was used to test the correctness of next state and reward computation logic.

"""
# Sample test code for validating environment behavior

SEED = 50
N = 100
OptX = 20
OptY = 50
env = gridEnv(seed = SEED, N = N, OptX = OptX, OptY = OptY)

observation = env.reset()

x = np.array([1, 0, 0, 0, 1])  # Example action sequence
x = np.tile(x, 100)
n_steps = np.size(x)

for step in range(n_steps):
    nextState, reward = env.step(x[step])
    print(f'action: {x[step]} nextstate: {nextState}  reward: {reward}')
    print("---------------------------------------------------------")
"""

