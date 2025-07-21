import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import patch, MagicMock
import traceback
from train import train

class TestTrain(unittest.TestCase):
    @patch("train.DeepTOP_RMAB")
    def test_train_function_runs(self, MockAgent):
        # --- Mock agent with all required methods ---
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = [1, 0]
        mock_agent.random_action.return_value = [1, 0]
        mock_agent.update.return_value = None
        mock_agent.eval.return_value = None
        mock_agent.reset.return_value = None
        mock_agent.actor = MagicMock()
        mock_agent.critic = MagicMock()
        MockAgent.return_value = mock_agent

        # --- Full mock config (cfg) ---
        cfg = {
            # General
            'mode': 'train',
            'seed': 42,
            'output': 'output/deeptop_run',
            'resume_path': None,
            'use_wandb': False,
            'run_note': 'default_test_env_run',
            'env_type': 'test_env',

            # Agent
            'agent_policy': 0,
            'nb_arms': 2,
            'budget': 1,

            # Training
            'train_iter': 5,
            'reset_iter': 2,
            'max_episode_length': 500,
            'warmup': 10,
            'discount': 0.95,
            'bsize': 128,
            'epsilon': 100000,

            # Replay Memory
            'rmsize': 1000,
            'window_length': 1,

            # Optimizer
            'rate': 0.0003,
            'prate': 0.00005,
            'tau': 0.005,

            # Exploration (OU noise)
            'ou_theta': 0.2,
            'ou_sigma': 0.3,
            'ou_mu': 0.0,

            # Checkpointing
            'checkpoint_every': 100,
        }

        # --- Mock environment list ---
        mock_env = MagicMock()
        mock_env.reset.return_value = [0, 0]
        mock_env.step.return_value = ([0, 0], 0.0, False, {})
        envs = [mock_env, mock_env]
        state_dims = [2, 2]
        action_dims = [2, 2]
        run_dir = "/tmp"
        logger = MagicMock()

        try:
            train(cfg, envs, state_dims, action_dims, run_dir, logger)
            ran_successfully = True
        except Exception:
            traceback.print_exc()
            ran_successfully = False

        self.assertTrue(ran_successfully)
