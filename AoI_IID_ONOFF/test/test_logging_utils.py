import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import tempfile
from logging_utils import setup_logger, custom_warning_format

class TestLoggingUtils(unittest.TestCase):
    def test_custom_warning_format(self):
        msg = "This is a test warning"
        category = UserWarning
        filename = __file__  # this test file
        lineno = 42

        result = custom_warning_format(msg, category, filename, lineno)
        rel_filename = os.path.relpath(filename)
        expected = f"{rel_filename}:{lineno}: UserWarning: {msg}\n"

        self.assertEqual(result, expected)

    def test_logger_creation_and_output(self):
        with tempfile.TemporaryDirectory() as log_dir:
            logger = setup_logger(log_dir)
            self.assertTrue(hasattr(logger, 'info'))

            # Write a test log message
            test_message = "This is a test log message."
            logger.info(test_message)

            # Check if any log file was created
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            self.assertTrue(log_files, "No log files were created by logger.")

            # Check the log file contains the test message
            log_path = os.path.join(log_dir, log_files[0])
            with open(log_path, "r") as f:
                log_content = f.read()
                self.assertIn(test_message, log_content)

            # Cleanup (optional redundancy; TemporaryDirectory auto-deletes)
            if os.path.exists(log_path):
                os.remove(log_path)
