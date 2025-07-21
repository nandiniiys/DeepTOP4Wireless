import logging
import os
from datetime import datetime
import warnings
import sys
import csv
import json

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    rel_filename = os.path.relpath(filename)
    return f"{rel_filename}:{lineno}: {category.__name__}: {message}\n"

def setup_logger(log_dir, log_filename="train.log"):
    """
    Configures and returns a logger that logs both to file and console.

    Args:
        log_dir (str): Path to directory where logs should be saved.
        log_filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("aoi_iid_onoff_logger")
    logger.setLevel(logging.DEBUG)
    warnings.formatwarning = custom_warning_format

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

