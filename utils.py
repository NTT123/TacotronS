"""
Utility functions
"""
import pickle
from pathlib import Path

import pax
import toml


def load_config(config_file=Path("pyproject.toml")):
    """
    Load the project configurations
    """
    return toml.load(config_file)["tacotron"]


def load_ckpt(net: pax.Module, optim: pax.Module, path):
    """
    load checkpoint from disk
    """
    with open(path, "rb") as f:
        dic = pickle.load(f)
    net = net.load_state_dict(dic["model_state_dict"])
    optim = optim.load_state_dict(dic["optim_state_dict"])
    return dic["step"], net, optim


def save_ckpt(ckpt_dir: Path, prefix: str, step, net: pax.Module, optim: pax.Module):
    """
    save checkpoint to disk
    """
    obj = {
        "step": step,
        "model_state_dict": net.state_dict(),
        "optim_state_dict": optim.state_dict(),
    }
    with open(ckpt_dir / f"{prefix}_{step:07d}.ckpt", "wb") as f:
        pickle.dump(obj, f)
