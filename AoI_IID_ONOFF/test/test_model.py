import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
from model import Actor, Critic

class TestModel(unittest.TestCase):
    def setUp(self):
        self.x_dim = 3                                # features for state
        self.price_dim = 1                            # features for price
        self.state_dim = self.x_dim + self.price_dim  # must match actor input
        self.action_dim = 2
        self.hidden = [8, 16, 16, 8]

    def test_actor_output_shape(self):
        actor = Actor(self.state_dim, self.action_dim, self.hidden)
        dummy_input = torch.randn(1, self.state_dim)
        output = actor(dummy_input)
        self.assertEqual(output.shape, (1, self.action_dim))

    def test_critic_output_shape(self):
        critic = Critic(self.state_dim, self.action_dim, self.hidden)

        x = torch.randn(1, self.x_dim)              # e.g., 3
        price = torch.randn(1, self.price_dim)      # e.g., 1
        a = torch.randn(1, self.action_dim)         # e.g., 2

        output = critic((x, price, a))
        self.assertEqual(output.shape, (1, 1))




