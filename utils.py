"""
Utility functions
"""
import os
import pickle
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import pax
import toml

from tacotron import Tacotron


def get_wav_files(data_dir: Path):
    """
    Get all *.wav files in the data directory.
    """
    files = sorted(data_dir.glob("*.wav"))
    random.Random(42).shuffle(files)
    return files


def load_config(config_file=Path("pyproject.toml")):
    """
    Load the project configurations.
    
    Override an attribute if it is in the environment variables.
    However, this is a hack and only a few attributes can be overridden.
    """
    config = toml.load(config_file)["tacotron"]

    if "MODEL_PREFIX" in os.environ:
        config["MODEL_PREFIX"] = os.environ["MODEL_PREFIX"]

    if "RR" in os.environ:
        config["RR"] = int(os.environ["RR"])

    if "TRAINING_STEPS" in os.environ:
        config["TRAINING_STEPS"] = int(os.environ["TRAINING_STEPS"])

    return config


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


def prepare_train_batch(batch, reduction_factor, random_start=True):
    """
    Prepare the mini-batch for training:
    - make sure that the sequence length is divisible by the reduce factor RR.
    - randomly select the start frame.
    """
    text, mel = batch
    N, L, D = mel.shape
    L = L // reduction_factor * reduction_factor
    mel = mel[:, :L]
    if random_start:
        idx = random.randint(0, reduction_factor - 1)
    else:
        idx = 0
    if reduction_factor > 1:
        mel = mel[:, idx : (idx - reduction_factor)]
    return text, mel


def bce_loss(logit, target):
    """
    return binary cross entropy loss
    """
    llh1 = jax.nn.log_sigmoid(logit) * target
    llh2 = jax.nn.log_sigmoid(-logit) * (1 - target)
    return -jnp.mean(llh1 + llh2)


def l1_loss(x, y):
    """
    compute the l1 loss
    """
    delta = x - y
    return jnp.mean(jnp.abs(delta), axis=-1)
