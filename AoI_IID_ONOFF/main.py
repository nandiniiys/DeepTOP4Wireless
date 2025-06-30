import os
import time
import random
import numpy as np
from datetime import datetime
import torch
import draccus
import wandb
import csv
import yaml
import traceback
import argparse

from env_registry import make_env, env_registry
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

def initialize_envs(cfg):
    """
    Initializes a list of environments using the centralized environment registry
    based on the provided configuration. Each arm receives its own instance of the environment.

    Returns:
        envs (list): List of initialized environments.
        state_dims (list): State dimension for each arm.
        action_dims (list): Action space dimension for each arm.
    """
    envs = []
    state_dims = []
    action_dims = []

    for i in range(cfg['nb_arms']):
        env = make_env(cfg['env_type'], seed=cfg['seed'] + i * 1000, p=0.2 + 0.6 / cfg['nb_arms'] * i)
        state_dim, action_dim = env_registry["aoi_iid_onoff"]["dims"]()
        state_dims.append(state_dim)
        action_dims.append(action_dim)
        envs.append(env)

    return envs, state_dims, action_dims

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

        if cfg['use_wandb']:
            wandb.init(project="rmab-aoi", config=cfg, name=run_id, notes=cfg['run_note'], dir=run_dir)

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
        if cfg['use_wandb']:
            wandb.alert(title="Training Crash", text=f"Run {run_id} crashed. Check crash_log.txt.")
