import gym
import random
import numpy as np
from gym import spaces


class TestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    Test environment simulating a simplified stock trading scenario.
    
    Environment Details:
    - State: [price], where price is a randomly generated stock price.
    - Action:
        0 = hold (do nothing)
        1 = sell (reward is current price)
    - Transition:
        - If action is 1 (sell), receive reward equal to current price.
        - If action is 0, reward is 0.
        - After every step, price is randomly resampled from [0, 100].
    """

    def __init__(self):
        """
        Initializes the environment by setting an initial random stock price
        and defining a discrete action space with 2 actions: hold or sell.
        """
        super(TestEnv, self).__init__()
        self.price = random.uniform(0, 100.0)
        self.action_space = spaces.Discrete(2)  # 0 = hold, 1 = sell

    def _calRewardAndState(self, action):
        """
        Computes the reward and next state given an action.
        
        Args:
            action (int): 0 = hold, 1 = sell
        
        Returns:
            nextState (list): next state containing new stock price
            reward (float): reward based on action
        """
        reward = 0
        if action == 1:
            reward = self.price  # Reward is current price if we sell

        nextPrice = random.uniform(0, 100.0)
        nextState = [nextPrice]
        return nextState, reward

    def step(self, action):
        """
        Standard Gym step function to execute the given action.

        Args:
            action (int): 0 = hold, 1 = sell
        
        Returns:
            nextState (list): new state after action
            reward (float): reward from taking the action
            done (bool): always False (no terminal state)
            info (dict): additional info (empty)
        """
        assert action in [0, 1]
        nextState, reward = self._calRewardAndState(action)
        self.price = nextState[0]
        done = False
        info = {}
        return nextState, reward, done, info

    def reset(self):
        """
        Resets the environment for a new episode.
        
        Returns:
            initialState (np.array): newly initialized state with a random price
        """
        self.price = random.uniform(0, 100.0)
        initialState = np.array([self.price], dtype=np.intc)
        return initialState


#########################################################################################
"""
For environment validation purposes, the below code checks if the next state and reward
match expectations given a predefined sequence of dummy actions.
"""

# env = TestEnv()
# observation = env.reset()

# x = np.array([1, 0, 0, 0, 1])
# x = np.tile(x, 100)
# n_steps = np.size(x)

# for step in range(n_steps):
#     nextState, reward, _, _ = env.step(x[step])
#     print(f'action: {x[step]} nextstate: {nextState}  reward: {reward}')
#     print("---------------------------------------------------------")
