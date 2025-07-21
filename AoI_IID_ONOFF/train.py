from DeepTOP import DeepTOP_RMAB
from Whittle_IID_OnOff import Whittle_IID_OnOff
from copy import deepcopy
import numpy as np
import os
import csv
import wandb

def train(cfg, envs, state_dims, action_dims, run_dir, logger):
    """
    Executes the main training loop for the specified agent and environment configuration.
    Handles exploration, policy updates, logging, checkpointing, and environment resets.

    Args:
        cfg (dict): Configuration dictionary.
        envs (list): List of initialized environments.
        state_dims (list): Dimensions of state spaces.
        action_dims (list): Dimensions of action spaces.
        run_dir (str): Directory path for saving logs and checkpoints.
        logger (logging.Logger): Logger instance for outputting run information.
    """

    # Initialize agent
    hidden = [8, 16, 16, 8]
    if cfg['agent_policy'] == 0:
        agent = DeepTOP_RMAB(state_dims, action_dims, hidden, cfg)
    else:
        agent = Whittle_IID_OnOff(cfg['nb_arms'], cfg['budget'], state_dims, action_dims, hidden, cfg)

    if cfg['resume_path'] and hasattr(agent, 'load'):
        agent.load(cfg['resume_path'])
        logger.info(f"Resumed agent from {cfg['resume_path']}")

    states = [env.reset() for env in envs]
    agent.eval()
    agent.reset(states)

    num_step = 0
    cumulative_reward = 0
    activation_counter = np.zeros(cfg['nb_arms'])
    best_avg_reward = -float('inf')



    if not cfg['use_wandb']:
        csv_path = os.path.join(run_dir, 'training_log.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'avg_reward', 'actor_loss', 'critic_loss'] +
                            [f'arm_{i}_output' for i in range(cfg['nb_arms'])] +
                            [f'arm_{i}_activation' for i in range(cfg['nb_arms'])])

    for t in range(cfg['train_iter'] + 1):
        # Reset agent and environments periodically
        if t % cfg['reset_iter'] == 0 and t != 0:
            logger.info(f'Resetting agent at iteration {t}')
            num_step = 0
            states = [env.reset() for env in envs]
            agent.reset(states)

        agent.is_training = True
        num_step += 1

        # Exploration vs exploitation
        if num_step <= cfg['warmup'] or np.random.uniform() < 0.05:
            action = agent.random_action()
        else:
            action = agent.select_action(states)

        activation_counter += np.array(action)

        next_state, reward, done, info = [], [], [], []
        for i, env in enumerate(envs):
            s1, r, d, inf = env.step(action[i])
            next_state.append(s1)
            reward.append(r)
            done.append(d)
            info.append(inf)

        agent.observe(reward, next_state, done)

        # Policy update and logging
        if num_step > cfg['warmup']:
            cumulative_reward += sum(reward)
            actor_loss, critic_loss, actor_outputs = agent.update_policy()

            if (num_step - cfg['warmup']) % 100 == 0:
                avg_reward = cumulative_reward / 100

                log_dict = {
                    "step": t,
                    "avg_reward": avg_reward,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                }
                log_dict.update({f"arm_{i}_output": actor_outputs[i] for i in range(cfg['nb_arms'])})
                log_dict.update({f"arm_{i}_activation": activation_counter[i] / 100 for i in range(cfg['nb_arms'])})

                if cfg['use_wandb']:
                    wandb.log(log_dict)
                else:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([log_dict[k] for k in ['step', 'avg_reward', 'actor_loss', 'critic_loss'] +
                                         [f'arm_{i}_output' for i in range(cfg['nb_arms'])] +
                                         [f'arm_{i}_activation' for i in range(cfg['nb_arms'])]])

                # Save best model
                if avg_reward > best_avg_reward and hasattr(agent, 'save'):
                    best_avg_reward = avg_reward
                    best_path = os.path.join(run_dir, 'best_model')
                    os.makedirs(best_path, exist_ok=True)
                    agent.save(best_path)

                cumulative_reward = 0
                activation_counter = np.zeros(cfg['nb_arms'])

        # Periodic checkpoint
        if t % cfg['checkpoint_every'] == 0 and hasattr(agent, 'save'):
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{t}")
            os.makedirs(checkpoint_path, exist_ok=True)
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint at step {t} to {checkpoint_path}")

        states = deepcopy(next_state)


