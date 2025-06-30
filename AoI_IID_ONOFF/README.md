# DeepTOP: Learning optimum threshold policies for RMAB Problems with Deep Reinforcement Learning

## Overview

This directory implements **DeepTOP**, a deep reinforcement learning framework designed to learn the **optimum threshold policy** for **Restless Multi-Armed Bandit (RMAB)** problems under **Age of Information (AoI)** settings.

It supports evaluation across different simulated environments. DeepTOP uses actor-critic neural networks to approximate the optimal index policy while logging performance, saving checkpoints, and allowing reproducibility through configuration files.

## Key Concepts

### Markov Decision Problem (MDP)

A simple way to model decision-making in situations where outcomes are partly random and partly under your control. Imagine you're playing a game where you're in a certain situation (called a *state*), and you can choose from a few possible moves (called *actions*). Each move leads you to a new situation and gives you a certain number of points (called a *reward*). The key idea is that what happens next depends only on where you are right now and what move you choose—*not* on how you got there. The goal is to find the best strategy (called a *policy*) for picking moves that will get you the most points over time.


### Restless Multi-Armed Bandit (RMAB)

A Restless Multi-Armed Bandit is like playing multiple slot machines (called *arms*), where each one changes over time—even when you’re not playing it. At each turn, you can only choose a few arms to play, and each one gives you a reward that depends on its current condition (called a *state*). But unlike regular slot machines, these arms keep evolving whether you play them or not. The challenge is to decide which arms to play at each step to get the most rewards over time, even though you can't watch or control them all at once.


### Optimum Threshold Policy

An optimum threshold policy is a simple rule for making decisions where you take action only when a certain condition crosses a set limit, or *threshold*. Imagine a thermostat controlling whether the AC is turned on or not. You set a rule - a threshold - where "Turn on the AC whenever the temperature increases past 75 degrees Farhenheit".


### Age of Information (AoI)

A way to measure how fresh or up-to-date the information you have is. Imagine you're tracking the temperature with a sensor that sends updates to your phone. AoI is like a timer that starts ticking the moment you receive an update—it tells you how long it’s been since the last piece of data arrived. The higher the AoI, the older the information. So, in systems where staying current is important (like monitoring health or weather), keeping the AoI low means you’re always working with the latest info.


## Features

* Train a DeepTOP agent using a configurable actor-critic network
* Log training metrics to **WandB** or local CSV
* Save checkpoints and reload models at specific steps
* Implement and Test DeepTOP with custom environments

## Repository Structure (Relevant Files)

```
.
├── deeptop_run.sh
├── DeepTOP.py                      # DeepTOP Actor-Critic implementation
├── env_registry.py                 # Functions for Environment Registry management
├── logging_utils.py                # Functions to set up logging
├── main.py                         # Loads config, Sets up logging, Start training
├── memory.py                       # Experience replay buffers
├── model.py                        # Actor and Critic model architectures
├── plot_metric.py                  # Script to plot rewards/losses from logs
├── random_process.py               # Ornstein-Uhlenbeck exploration noise
├── README.md                       # You're here!
├── run_configs/                    # Folder for Draccus run configs
├── test/                           # Folder for unit tests
├── train.py                        # Training loop
├── util.py                         # Utility functions (e.g. update target nets)
├── Whittle_IID_OnOff.py            # TODO
└── WirelessEnv
    ├── TestEnv.py                  # Simple testbed environment
    └── WirelessEnv.py              # Environment for IID On-Off channel
├── checkpoints/                    # Saved checkpoints of the Actor-Critic networks
└── output/                         # Run logs and metrics
```

## Quick Start

> ⚠️ **Note**: The following instructions assume **Getting Started** from the root folder README have been completed.

### Run Training

```bash
python main.py --config config.yaml
```

### Visualize Logs

* **WandB**: [wandb.ai](https://wandb.ai/)
  * You will need to create a WandB account and link it to a local(forked) copy of this repo.
* **Local CSV**: Use `plot_metric.py` to generate plots.

## Configuration

This repo uses **Draccus** for structured configs. All experiment parameters can be configured via a YAML file. Example config files can be found in `./run_configs/`

## How to Add a New Environment

Add your class to `WirelessEnv/` in a separate python file, implemeting at minimum the methods:`_calRewardAndState`, `step` and `reset`, then register it using a new entry in the environment registry (TODO).

## Authors

Developed by Nandinii Yeleswarapu as part of the DeepTOP4 Wireless project, advised by Dr. I-Hong Hou at Texas A\&M.
