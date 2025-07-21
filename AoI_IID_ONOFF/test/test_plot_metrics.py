import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import tempfile
import pandas as pd
from plot_metrics import plot_metrics

class TestPlotMetrics(unittest.TestCase):
    def test_plot_metrics_runs(self):
        # Create temporary CSV file with minimal valid data
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metrics.csv")
            output_dir = os.path.join(tmpdir, "plots")

            # Dummy data with 2 arms
            df = pd.DataFrame({
                "step": [1, 2, 3, 4],
                "avg_reward": [1.0, 1.5, 1.7, 2.0],
                "actor_loss": [0.5, 0.4, 0.3, 0.2],
                "critic_loss": [0.6, 0.5, 0.4, 0.3],
                "arm_0_activation": [0.1, 0.2, 0.3, 0.4],
                "arm_1_activation": [0.9, 0.8, 0.7, 0.6],
                "arm_0_output": [0.3, 0.4, 0.5, 0.6],
                "arm_1_output": [0.7, 0.6, 0.5, 0.4],
            })
            df.to_csv(csv_path, index=False)

            # Run the function
            plot_metrics(csv_path, output_dir=output_dir, nb_arms=2)

            # Check all expected files exist
            expected_files = [
                "reward_vs_steps.png",
                "actor_loss_log.png",
                "critic_loss_log.png",
                "arm_0_activation.png",
                "arm_1_activation.png",
                "arm_0_output.png",
                "arm_1_output.png",
            ]
            for filename in expected_files:
                path = os.path.join(output_dir, filename)
                self.assertTrue(os.path.isfile(path), f"Missing plot: {filename}")
                if os.path.exists(path):
                    os.remove(path)
