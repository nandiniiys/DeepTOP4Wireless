# actor_outputs.py
import os
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import Actor
from main import initialize_envs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


st.title("Actor Output vs State")

# ---- Run selection ----
parent_dir = st.sidebar.text_input(
    "Parent path (run subfolders with used_config.yaml):",
    value="output/deeptop_run",
)

selected_dir = None
if os.path.isdir(parent_dir):
    subdirs = sorted(
        [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
            and os.path.exists(os.path.join(parent_dir, d, "used_config.yaml"))
        ]
    )
    if subdirs:
        label = st.selectbox("Select a run directory:", [os.path.basename(d) for d in subdirs])
        selected_dir = subdirs[[os.path.basename(d) for d in subdirs].index(label)]
else:
    st.warning("Enter a valid parent directory.")
if not selected_dir:
    st.stop()

ckpt_folders = sorted(
    [
        f
        for f in os.listdir(selected_dir)
        if f.startswith("checkpoint_") and os.path.isdir(os.path.join(selected_dir, f))
    ]
)
if not ckpt_folders:
    st.warning("No checkpoints found.")
    st.stop()
selected_checkpoint = st.selectbox("Select a checkpoint:", ckpt_folders)

# ---- Config / dims ----
used_config_path = os.path.join(selected_dir, "used_config.yaml")
if not os.path.exists(used_config_path):
    st.error("used_config.yaml not found.")
    st.stop()

with open(used_config_path, "r") as f:
    cfg = yaml.safe_load(f)

_, state_dims, _ = initialize_envs(cfg)
hidden = [8, 16, 16, 8]

num_dims = max(state_dims)
dim_names = [f"dim_{i}" for i in range(num_dims)]

# ---- Sweep setup ----
active_dim = st.selectbox("Sweep dimension:", dim_names)
active_index = dim_names.index(active_dim)
active_range = st.slider(f"Range for {active_dim}", min_value=0, max_value=200, value=(0, 100))
x_vals = list(range(active_range[0], active_range[1] + 1))

fixed_values = []
for i in range(num_dims):
    if i != active_index:
        fixed_val = st.number_input(
            f"Fixed value for {dim_names[i]}", value=50.0, step=0.1, format="%.3f"
        )
        fixed_values.append((i, fixed_val))

# ---- Channel compare & y-mode ----
compare_onoff = st.sidebar.checkbox("Compare channel on vs off", value=True)
channel_dim_index = st.sidebar.number_input(
    "Channel bit index (0-based)", min_value=0, value=1, step=1, help="Use 1 for [AoI, on_off]."
)

y_mode = st.selectbox("Y-axis mode", ["Sigmoid P", "Raw logit"])
use_prob = (y_mode == "Sigmoid P")

# ---- Plot ----
if st.button("Run and Plot"):
    ckpt_path = os.path.join(selected_dir, selected_checkpoint)
    arm_ckpts = sorted(
        [f for f in os.listdir(ckpt_path) if f.startswith("actor_arm") and f.endswith(".pt")]
    )
    if not arm_ckpts:
        st.error("No actor_arm<#>.pt files found.")
        st.stop()

    fig, ax = plt.subplots()

    # consistent color per arm
    color_cycle = plt.get_cmap("tab20").colors  # 20 distinct colors
    n_colors = len(color_cycle)

    for arm_file in arm_ckpts:
        arm_index = int(arm_file.split("actor_arm")[-1].split(".pt")[0])
        d = state_dims[arm_index]

        actor = Actor(d, 1, hidden)
        actor.load_state_dict(torch.load(os.path.join(ckpt_path, arm_file), map_location="cpu"))
        actor.eval()

        color = color_cycle[arm_index % n_colors]

        if compare_onoff and channel_dim_index < d:
            logits_on, logits_off = [], []
            for x in x_vals:
                state_on = np.zeros(d, dtype=np.float32)
                state_off = np.zeros(d, dtype=np.float32)

                if active_index < d:
                    state_on[active_index] = x
                    state_off[active_index] = x

                for i, val in fixed_values:
                    if i < d:
                        state_on[i] = val
                        state_off[i] = val

                state_on[channel_dim_index] = 1.0
                state_off[channel_dim_index] = 0.0

                with torch.no_grad():
                    logits_on.append(actor(torch.tensor(state_on).unsqueeze(0)).item())
                    logits_off.append(actor(torch.tensor(state_off).unsqueeze(0)).item())

            y_on = sigmoid(np.array(logits_on)) if use_prob else np.array(logits_on)
            y_off = sigmoid(np.array(logits_off)) if use_prob else np.array(logits_off)

            # label only the "on" curve for legend color-by-arm; "off" hidden label
            ax.plot(x_vals, y_on, color=color, linestyle="-", label=f"Arm {arm_index}")
            ax.plot(x_vals, y_off, color=color, linestyle=":", label="_nolegend_")
        else:
            logits = []
            for x in x_vals:
                state = np.zeros(d, dtype=np.float32)
                if active_index < d:
                    state[active_index] = x
                for i, val in fixed_values:
                    if i < d:
                        state[i] = val
                with torch.no_grad():
                    logits.append(actor(torch.tensor(state).unsqueeze(0)).item())
            y_vals = sigmoid(np.array(logits)) if use_prob else np.array(logits)
            ax.plot(x_vals, y_vals, color=color, linestyle="-", label=f"Arm {arm_index}")

    ax.set_xlabel(active_dim)
    if use_prob:
        ax.set_ylabel("P(activate) = σ(logit)")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Actor Output (raw logit)")
    ax.set_title(f"Output vs {active_dim} — {selected_checkpoint}")

    # Legend 1: colors = arms (labels from 'on' curves)
    leg1 = ax.legend(title="Arms", ncol=2, loc="upper right")
    ax.add_artist(leg1)

    # Legend 2: line style meaning
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="On (channel=1)"),
        Line2D([0], [0], color="black", linestyle=":", lw=2, label="Off (channel=0)"),
    ]
    ax.legend(handles=style_handles, title="Channel", loc="lower right")

    st.pyplot(fig)

    # ---- Sensitivity JSON (finite difference on raw logits) ----
    st.subheader("Dimension sensitivity (finite difference on raw logit)")
    x_ref = float((active_range[0] + active_range[1]) / 2.0)
    eps = 1e-2

    fixed_map = {i: v for (i, v) in fixed_values}
    fixed_map[active_index] = x_ref

    avg_abs = np.zeros(num_dims, dtype=np.float64)
    cnt = np.zeros(num_dims, dtype=np.int64)

    for arm_file in arm_ckpts:
        arm_index = int(arm_file.split("actor_arm")[-1].split(".pt")[0])
        d = state_dims[arm_index]

        actor = Actor(d, 1, hidden)
        actor.load_state_dict(torch.load(os.path.join(ckpt_path, arm_file), map_location="cpu"))
        actor.eval()

        ref = np.zeros(d, dtype=np.float32)
        for i in range(d):
            ref[i] = float(fixed_map.get(i, 0.0))

        with torch.no_grad():
            for i in range(d):
                sp = ref.copy(); sp[i] += eps
                sm = ref.copy(); sm[i] -= eps
                yp = actor(torch.tensor(sp).unsqueeze(0)).item()
                ym = actor(torch.tensor(sm).unsqueeze(0)).item()
                g = (yp - ym) / (2.0 * eps)
                avg_abs[i] += abs(g)
                cnt[i] += 1

    sens_dict = {f"dim_{i}": (float(avg_abs[i] / cnt[i]) if cnt[i] > 0 else None) for i in range(num_dims)}
    st.json(sens_dict)
