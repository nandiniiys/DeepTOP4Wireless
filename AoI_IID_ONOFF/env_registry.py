# env_registry.py
import numpy as np
from gym import spaces
from WirelessEnv.TestEnv import TestEnv
from WirelessEnv.AoIEnv_IID_OnOff import AoIEnv_IID_OnOff
from WirelessEnv.AoIEnv_IID_ChannelPenalty import AoIEnv_IID_ChannelPenalty

env_registry = {
    "test_env": {
        "make": lambda **kwargs: TestEnv(),
    },
    "aoi_iid_onoff": {
        "make": lambda **kwargs: AoIEnv_IID_OnOff(seed=kwargs.get("seed", 1), p=kwargs.get("p", 0.5)),
    },
    "aoi_iid_channelpenalty": {
        "make": lambda **kwargs: AoIEnv_IID_ChannelPenalty(seed=kwargs.get("seed", 1), p=kwargs.get("p", 0.5)),
    }
}

def infer_dims(env):
    # State dim
    if hasattr(env, "observation_space") and env.observation_space is not None:
        shape = getattr(env.observation_space, "shape", None)
        state_dim = int(np.prod(shape)) if shape else 1
    else:
        s0 = env.reset()
        state_dim = int(np.prod(np.array(s0).shape))
    # Action dim
    if isinstance(env.action_space, spaces.Discrete):
        action_dim = 1
    else:
        action_dim = int(np.prod(env.action_space.shape))
    return state_dim, action_dim

def make_env(env_type: str, **kwargs):
    if env_type not in env_registry:
        raise ValueError(f"Unknown environment type: {env_type}")
    return env_registry[env_type]["make"](**kwargs)

def initialize_envs(cfg):
    envs, state_dims, action_dims = [], [], []
    for i in range(cfg["nb_arms"]):
        kwargs = dict(cfg.get("env_kwargs", {}))
        kwargs.update(seed=cfg["seed"] + i * 1000)
        if cfg["env_type"] != "test_env":
            p = 0.125 * (1 + i)
            p = max(1e-6, min(1 - 1e-6, p))
            kwargs["p"] = p

        env = make_env(cfg["env_type"], **kwargs)
        sdim, adim = infer_dims(env)
        envs.append(env); state_dims.append(sdim); action_dims.append(adim)
    return envs, state_dims, action_dims