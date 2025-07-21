import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from env_registry import make_env


class TestEnvRegistry(unittest.TestCase):
    def test_make_env_aoi_iid_onoff_valid(self):
        env = make_env('aoi_iid_onoff', seed=42, p=0.5)
        self.assertIsNotNone(env)

    def test_make_env_test_env_valid(self):
        env = make_env('test_env', seed=42)
        self.assertIsNotNone(env)

    def test_make_env_invalid(self):
        with self.assertRaises(ValueError):
            make_env('nonexistent_env')