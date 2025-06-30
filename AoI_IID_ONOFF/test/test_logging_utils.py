import sys
import os
import unittest
from logging_utils import setup_logger

class TestLoggingUtils(unittest.TestCase):
    def test_logger_creation(self):
        log_dir = 'test_logs'
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logger(log_dir)
        self.assertTrue(hasattr(logger, 'info'))