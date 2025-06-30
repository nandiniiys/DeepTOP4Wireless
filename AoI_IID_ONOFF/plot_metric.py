import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path, output_dir="plots", nb_arms=2):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Plot: Average Reward vs Steps (Linear)
    plt.figure()
    plt.plot(df["step"], df["avg_reward"])
    plt.xlabel("Steps")
    plt.ylabel("Average Reward (per 100 steps)")
    plt.title("Average Reward vs Steps")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward_vs_steps.png"))

    # Plot: Actor Loss (Log Scale)
    plt.figure()
    plt.plot(df["step"], df["actor_loss"])
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss vs Steps (Log Scale)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "actor_loss_log.png"))

    # Plot: Critic Loss (Log Scale)
    plt.figure()
    plt.plot(df["step"], df["critic_loss"])
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss vs Steps (Log Scale)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "critic_loss_log.png"))

    # Plot: Arm Activation Frequencies
    for i in range(nb_arms):
        plt.figure()
        plt.plot(df["step"], df[f"arm_{i}_activation"])
        plt.xlabel("Steps")
        plt.ylabel("Activation Frequency (per 100 steps)")
        plt.title(f"Arm {i} Activation Frequency vs Steps")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"arm_{i}_activation.png"))

    # Plot: Actor Outputs per Arm
    for i in range(nb_arms):
        plt.figure()
        plt.plot(df["step"], df[f"arm_{i}_output"])
        plt.xlabel("Steps")
        plt.ylabel(f"Actor Output - Arm {i}")
        plt.title(f"Actor Output for Arm {i} vs Steps")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"arm_{i}_output.png"))

    print(f"✅ Saved all plots to '{output_dir}'")

if __name__ == "__main__":
    plot_metrics(
        csv_path="output/deeptop_run/training_log.csv",
        output_dir="plots",
        nb_arms=2  # Change this to match your run config
    )
