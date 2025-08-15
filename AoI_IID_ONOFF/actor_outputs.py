import os
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yaml
import io

from model import Actor
from main import initialize_envs

st.title("Actor Network Output vs State Visualizer")

# Allow selectable directory upload
parent_dir = st.sidebar.text_input("Enter parent path to search for runs:", value="output/deeptop_run")

selected_dir = None
if os.path.isdir(parent_dir):
    subdirs = sorted([
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and os.path.exists(os.path.join(parent_dir, d, "used_config.yaml"))
    ])
    dir_labels = [os.path.basename(d) for d in subdirs]
    if subdirs:
        selected_label = st.selectbox("Select a run directory:", dir_labels)
        selected_dir = subdirs[dir_labels.index(selected_label)]
else:
    st.warning("Enter a valid parent directory containing run subfolders with used_config.yaml")


if selected_dir:
    checkpoint_folders = sorted([
        f for f in os.listdir(selected_dir)
        if f.startswith("checkpoint_") and os.path.isdir(os.path.join(selected_dir, f))
    ])

    if not checkpoint_folders:
        st.warning("No checkpoints found in this directory.")
    else:
        # load available checkpoints to select from
        selected_checkpoint = st.selectbox("Select a checkpoint:", checkpoint_folders)

        # get the config that was used for the run
        used_config_path = os.path.join(selected_dir, "used_config.yaml")
        if not os.path.exists(used_config_path):
            st.error("used_config.yaml not found in the selected directory.")
        else:
            with open(used_config_path, "r") as f:
                cfg = yaml.safe_load(f)

            # initialize the environment based on loaded configs to get Actor network params
            _, state_dims, _ = initialize_envs(cfg)
            hidden = [8, 16, 16, 8]

            num_dims = max(state_dims) # all the arms *should* have the same state dims but just choose highest value in case.
            
            # select which state field we want to vary to plot against
            dim_names = [f"dim_{i}" for i in range(num_dims)]
            active_dim = st.selectbox("Select the active dimension:", dim_names)

            # set range to plot for selected state field
            active_index = dim_names.index(active_dim)
            active_range = st.slider(f"Range for {active_dim}", min_value=0, max_value=200, value=(0, 100))

            # set fixed values for remaining state values to plot against
            fixed_values = []
            for i in range(num_dims):
                if i != active_index:
                    fixed_val = st.number_input(f"Fixed value for {dim_names[i]}", value=50.0, step=0.1, format="%.3f")
                    fixed_values.append((i, fixed_val))

            # choose plot mode
            plot_mode = st.selectbox("Plot Mode", ["Per Arm", "Average Across All Arms"])
            
            # run Actor Network based on selected configs
            if st.button("Run and Plot"):
                checkpoint_path = os.path.join(selected_dir, selected_checkpoint)
                arm_checkpoints = sorted([
                    f for f in os.listdir(checkpoint_path)
                    if f.startswith("actor_arm") and f.endswith(".pt")
                ])

                if not arm_checkpoints:
                    st.error(f"No actor_arm<#>.pt files found in {checkpoint_path}")
                else:
                    x_vals = list(range(active_range[0], active_range[1]+1))
                    fig, ax = plt.subplots()

                    all_outputs = []
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

                        all_outputs.append(np.array(y_vals))

                        if plot_mode == "Per Arm":
                            ax.plot(x_vals, y_vals, label=f"Arm {arm_index}")

                    if plot_mode == "Average Across All Arms":
                        mean_vals = np.mean(all_outputs, axis=0)
                        ax.plot(x_vals, mean_vals, label="Average", color="black")

                    ax.set_xlabel(active_dim)
                    ax.set_ylabel("Actor Output")
                    ax.set_title(f"Output vs {active_dim} for {selected_checkpoint} [{plot_mode}]")
                    ax.legend()
                    st.pyplot(fig)

                    # Save and download button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf,
                        file_name=f"actor_output_{selected_checkpoint}_{plot_mode.replace(' ', '_').lower()}.png",
                        mime="image/png"
                    )
else:
    st.info("Please enter a valid parent directory path containing run subfolders.")
