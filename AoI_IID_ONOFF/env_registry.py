from WirelessEnv.AoIEnv_IID_OnOff import AoIEnv_IID_OnOff
from WirelessEnv.TestEnv import TestEnv

# Registry for environment classes
env_registry = {
    "aoi_iid_onoff": AoIEnv_IID_OnOff,
    "test_env": TestEnv,
}

def make_env(env_name, seed=None, **kwargs):
    """
    Factory function to create an environment instance from a registered name.

    Args:
        env_name (str): The key name of the environment to initialize.
        seed (int): Optional seed for the environment.
        **kwargs: Additional keyword arguments passed to the environment constructor.

    Returns:
        An instance of the requested environment.
    """
    env_cls = env_registry.get(env_name.lower())
    if env_cls is None:
        raise ValueError(f"Unknown environment type: {env_name}. Available: {list(env_registry.keys())}")

    if env_name.lower() == 'aoi_iid_onoff':
        return env_cls(seed=seed, **kwargs)
    else:
        return env_cls()
