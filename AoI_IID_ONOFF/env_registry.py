# env_registry.py

from WirelessEnv.TestEnv import TestEnv
from WirelessEnv.AoIEnv_IID_OnOff import AoIEnv_IID_OnOff

# A registry mapping environment names to both their constructors and dimension specs
env_registry = {
    "test_env": {
        "make": lambda **kwargs: TestEnv(),
        "dims": lambda: (1, 1),  # state_dim, action_dim
    },
    "aoi_iid_onoff": {
        "make": lambda **kwargs: AoIEnv_IID_OnOff(seed=kwargs.get("seed", 1), p=kwargs.get("p", 0.5)),
        "dims": lambda: (2, 1),
    },
}

def make_env(env_type: str, **kwargs):
    """
    Factory function to create an environment instance based on its type.

    Args:
        env_type (str): The key in env_registry indicating which environment to construct.
        **kwargs: Parameters passed to the environment constructor.

    Returns:
        An instance of the specified environment.

    Raises:
        ValueError: If the env_type is not registered.
    """
    if env_type not in env_registry:
        raise ValueError(f"Unknown environment type: {env_type}")
    return env_registry[env_type]["make"](**kwargs)
