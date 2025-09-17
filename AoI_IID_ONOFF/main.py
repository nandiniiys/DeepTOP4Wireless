import os
import time
import random
import numpy as np
from datetime import datetime
import torch
import draccus
import csv
import yaml
import traceback
import argparse

from env_registry import initialize_envs
from logging_utils import setup_logger
from train import train

# ---------------------------
# Config Definition via Draccus
# ---------------------------

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

if __name__ == '__main__':
    try:
        cfg = load_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{cfg['env_type']}_deeptop"
        run_dir = os.path.join(cfg['output'], run_id)
        os.makedirs(run_dir, exist_ok=True)

        logger = setup_logger(run_dir)
        logger.info("Logger initialized.")

        with open(os.path.join(run_dir, 'used_config.yaml'), 'w') as f:
            yaml.dump(cfg, f)

        logger.info("Setting random seeds.")
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        logger.info("Initializing environments.")
        envs, state_dims, action_dims = initialize_envs(cfg)

        logger.info("Starting training loop.")
        train(cfg, envs, state_dims, action_dims, run_dir, logger)

    except Exception as e:
        crash_path = os.path.join(run_dir, 'crash_log.txt')
        with open(crash_path, 'w') as f:
            f.write(traceback.format_exc())
        logger.error("Training crashed. See crash_log.txt for details.")