import os
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yaml
import io

from model import Actor  # assumes model.py defines the Actor network
from main import initialize_envs  # assumes main.py defines this

# --- UI CONFIG ---
st.title("Actor Network Output vs State Visualizer")

# Step 1: Directory Upload
directory = st.text_input("Enter path to checkpoint directory (e.g. output/deeptop_run/...):")

if directory and os.path.isdir(directory):
    checkpoint_folders = sorted([
        f for f in os.listdir(directory)
        if f.startswith("checkpoint_") and os.path.isdir(os.path.join(directory, f))
    ])

    if not checkpoint_folders:
        st.warning("No checkpoints found in this directory.")
    else:
        # Step 2: Select Checkpoint
        selected_checkpoint = st.selectbox("Select a checkpoint:", checkpoint_folders)

        # Step 3: Enter State Space Dimensions
        used_config_path = os.path.join(directory, "used_config.yaml")
        if not os.path.exists(used_config_path):
            st.error("used_config.yaml not found in the selected directory.")
        else:
            with open(used_config_path, "r") as f:
                cfg = yaml.safe_load(f)

            _, state_dims, _ = initialize_envs(cfg)
            hidden = [8, 16, 16, 8]

            num_dims = max(state_dims)
            dim_names = [f"dim_{i}" for i in range(num_dims)]
            active_dim = st.selectbox("Select the active dimension:", dim_names)

            active_index = dim_names.index(active_dim)
            active_range = st.slider(f"Range for {active_dim}", min_value=0, max_value=200, value=(0, 100))

            fixed_values = []
            for i in range(num_dims):
                if i != active_index:
                    fixed_val = st.number_input(f"Fixed value for {dim_names[i]}", value=50)
                    fixed_values.append((i, fixed_val))

            # Step 4: Run Actor Networks (All Arms)
            if st.button("Run and Plot"):
                checkpoint_path = os.path.join(directory, selected_checkpoint)
                arm_checkpoints = sorted([
                    f for f in os.listdir(checkpoint_path)
                    if f.startswith("actor_arm") and f.endswith(".pt")
                ])

                if not arm_checkpoints:
                    st.error(f"No actor_arm<#>.pt files found in {checkpoint_path}")
                else:
                    x_vals = list(range(active_range[0], active_range[1]+1))
                    fig, ax = plt.subplots()

                    for arm_file in arm_checkpoints:
                        arm_path = os.path.join(checkpoint_path, arm_file)
                        arm_index = int(arm_file.split("actor_arm")[-1].split(".pt")[0])

                        actor = Actor(state_dims[arm_index], 1, hidden)
                        actor.load_state_dict(torch.load(arm_path, map_location="cpu"))
                        actor.eval()

                        y_vals = []
                        for x in x_vals:
                            state = np.zeros(state_dims[arm_index], dtype=np.float32)
                            if active_index < state.shape[0]:
                                state[active_index] = x
                            for i, val in fixed_values:
                                if i < state.shape[0]:
                                    state[i] = val
                            with torch.no_grad():
                                output = actor(torch.tensor(state).unsqueeze(0))
                            y_vals.append(output.item())

                        ax.plot(x_vals, y_vals, label=f"Arm {arm_index}")

                    ax.set_xlabel(active_dim)
                    ax.set_ylabel("Actor Output")
                    ax.set_title(f"Output vs {active_dim} for {selected_checkpoint}")
                    ax.legend()
                    st.pyplot(fig)

                    # Save and download button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf,
                        file_name=f"actor_output_{selected_checkpoint}.png",
                        mime="image/png"
                    )
else:
    st.info("Please enter a valid checkpoint directory path.")