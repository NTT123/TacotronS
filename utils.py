"""
Utility functions
"""
import pickle
from pathlib import Path

import pax
import toml

from tacotron import Tacotron


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
    if net is not None:
        net = net.load_state_dict(dic["model_state_dict"])
    if optim is not None:
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


def create_tacotron_model(config):
    """
    return a random initialized Tacotron model
    """
    return Tacotron(
        mel_dim=config["MEL_DIM"],
        attn_bias=config["ATTN_BIAS"],
        rr=config["RR"],
        max_rr=config["MAX_RR"],
        mel_min=config["MEL_MIN"],
        sigmoid_noise=config["SIGMOID_NOISE"],
        pad_token=config["PAD_TOKEN"],
        prenet_dim=config["PRENET_DIM"],
        attn_hidden_dim=config["ATTN_HIDDEN_DIM"],
        attn_rnn_dim=config["ATTN_RNN_DIM"],
        rnn_dim=config["RNN_DIM"],
        postnet_dim=config["POSTNET_DIM"],
        text_dim=config["TEXT_DIM"],
    )
