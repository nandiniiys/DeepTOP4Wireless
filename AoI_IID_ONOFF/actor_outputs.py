import os
import io
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from model import Actor
from main import initialize_envs

st.title("Actor Network Output vs State Visualizer")

# Allow selectable directory upload
parent_dir = st.sidebar.text_input(
    "Enter parent path to search for runs:", value="output/deeptop_run"
)

selected_dir = None
if os.path.isdir(parent_dir):
    subdirs = sorted([
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
        and os.path.exists(os.path.join(parent_dir, d, "used_config.yaml"))
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

            # all the arms *should* have the same state dims; choose highest just in case
            num_dims = max(state_dims)

            # select which state field we want to vary to plot against
            dim_names = [f"dim_{i}" for i in range(num_dims)]
            active_dim = st.selectbox("Select the active dimension:", dim_names)

            # set range to plot for selected state field
            active_index = dim_names.index(active_dim)
            active_range = st.slider(
                f"Range for {active_dim}", min_value=0, max_value=200, value=(0, 100)
            )

            # set fixed values for remaining state values to plot against
            fixed_values = []
            for i in range(num_dims):
                if i != active_index:
                    fixed_val = st.number_input(
                        f"Fixed value for {dim_names[i]}",
                        value=50.0, step=0.1, format="%.3f"
                    )
                    fixed_values.append((i, fixed_val))

            # choose plot mode
            plot_mode = st.selectbox("Plot Mode", ["Per Arm", "Average Across All Arms"])

            # y-axis display mode
            y_mode = st.selectbox(
                "Display y-axis as",
                ["Probability σ(logit) (recommended)", "Raw logit"],
                help=(
                    "Raw logit = unbounded score directly from the actor head.\n"
                    "Probability = σ(logit) = 1 / (1 + exp(-logit)).\n\n"
                    "Since all our environments are monotone and binary (activate vs don't), "
                    "the sigmoid view shows P(activate) in [0,1] and makes the learned threshold "
                    "(logit≈0 ⇒ P≈0.5) easier to interpret."
                )
            )
            use_prob = (y_mode.startswith("Probability"))

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
                    x_vals = list(range(active_range[0], active_range[1] + 1))
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
                                logit = actor(torch.tensor(state).unsqueeze(0)).item()
                            if use_prob:
                                y_vals.append(1.0 / (1.0 + np.exp(-logit)))  # σ(logit)
                            else:
                                y_vals.append(logit)

                        y_vals_np = np.array(y_vals, dtype=np.float32)
                        all_outputs.append(y_vals_np)

                        if plot_mode == "Per Arm":
                            ax.plot(x_vals, y_vals_np, label=f"Arm {arm_index}")

                    if plot_mode == "Average Across All Arms":
                        mean_vals = np.mean(all_outputs, axis=0)
                        ax.plot(x_vals, mean_vals, label="Average", color="black")

                    ax.set_xlabel(active_dim)
                    if use_prob:
                        ax.set_ylabel("P(activate) = σ(logit)")
                        ax.set_ylim(0.0, 1.0)
                    else:
                        ax.set_ylabel("Actor Output (raw logit)")
                    ax.set_title(f"Output vs {active_dim} for {selected_checkpoint} [{plot_mode}]")
                    ax.legend()
                    st.pyplot(fig)

                    # Plot explainer
                    st.caption(
                        "Y-axis explanation: In 'Raw logit' mode, you see the actor's unbounded score; "
                        "larger ⇒ more likely to activate. In 'Probability' mode, we apply σ(logit) to show "
                        "P(activate) in [0,1]. The threshold where the model is indifferent is at logit≈0 "
                        "(i.e., P≈0.5)."
                    )

                    # Save and download button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf,
                        file_name=f"actor_output_{selected_checkpoint}_{plot_mode.replace(' ', '_').lower()}_{'prob' if use_prob else 'logit'}.png",
                        mime="image/png"
                    )

                    # =========================
                    # DIMENSION SENSITIVITY (JSON)
                    # =========================
                    st.subheader("Dimension sensitivity (local, finite difference)")

                    # Choose the reference value for the swept dimension (midpoint by default)
                    x_ref = float((active_range[0] + active_range[1]) / 2)
                    eps = 1e-2  # small step; OK for features on ~[0,100]. Adjust if you normalize features.

                    # Build a dict of fixed values for all dims (use x_ref for the active dim)
                    fixed_map = {i: v for (i, v) in fixed_values}
                    fixed_map[active_index] = x_ref

                    # Average |∂f/∂dim_i| across arms (f = raw logit)
                    avg_abs = np.zeros(num_dims, dtype=np.float64)
                    cnt = np.zeros(num_dims, dtype=np.int64)

                    for arm_file in arm_checkpoints:
                        arm_path = os.path.join(checkpoint_path, arm_file)
                        arm_index = int(arm_file.split("actor_arm")[-1].split(".pt")[0])

                        d = state_dims[arm_index]
                        actor = Actor(d, 1, hidden)
                        actor.load_state_dict(torch.load(arm_path, map_location="cpu"))
                        actor.eval()

                        ref = np.zeros(d, dtype=np.float32)
                        for i in range(d):
                            ref[i] = float(fixed_map.get(i, 0.0))

                        # central finite difference on the raw logit
                        with torch.no_grad():
                            for i in range(d):
                                sp = ref.copy(); sp[i] += eps
                                sm = ref.copy(); sm[i] -= eps
                                yp = actor(torch.tensor(sp).unsqueeze(0)).item()
                                ym = actor(torch.tensor(sm).unsqueeze(0)).item()
                                g = (yp - ym) / (2.0 * eps)
                                avg_abs[i] += abs(g)
                                cnt[i] += 1

                    sens_dict = {f"dim_{i}": (float(avg_abs[i] / cnt[i]) if cnt[i] > 0 else None)
                                 for i in range(num_dims)}
                    st.json(sens_dict)

                    with st.expander("What is this 'sensitivity' and how is it measured?"):
                        st.markdown(
                            "- We report a **local finite-difference sensitivity** of the actor's raw logit "
                            "with respect to each input dimension at a reference state.\n"
                            "- Reference state = the fixed values you provided, with the swept dim set to the "
                            f"midpoint `x_ref = {x_ref:.3f}`.\n"
                            "- For each dim *i*, we compute a central difference:  "
                            r"`∂f/∂x_i ≈ [f(x+ε e_i) − f(x−ε e_i)] / (2ε)` with `ε = 1e-2`."
                            "\n- We take the absolute value and **average across all arms** that use that dim.\n"
                            "- Larger value ⇒ the actor's score is more sensitive to that feature *locally*. "
                            "Because features may live on different numeric scales, compare sensitivities after "
                            "normalizing inputs if you need apples-to-apples magnitudes."
                        )

else:
    st.info("Please enter a valid parent directory path containing run subfolders.")
