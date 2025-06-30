import sys
import os
import unittest
import draccus
import yaml


# Ensure project root is in sys.path for module resolution
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        with open('run_configs/test.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        self.assertIsInstance(cfg, dict)
        self.assertIn('nb_arms', cfg)