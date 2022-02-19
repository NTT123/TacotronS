from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pax

from tacotron import Tacotron
from text import english_cleaners
from utils import load_ckpt, load_config

parser = ArgumentParser(description="Convert text to melspectrogram")

parser.add_argument("-m", "--model", type=Path, required=True)
parser.add_argument("-t", "--text", type=str, required=True)
parser.add_argument("-a", "--alphabet-file", type=Path, required=True)
parser.add_argument("-o", "--output", type=Path, required=True)
args = parser.parse_args()

with open(args.alphabet_file, "r", encoding="utf-8") as f:
    alphabet = f.read().split("\n")


text = english_cleaners(args.text)
print("Input: ", text)

config = load_config()
text = text + config["PAD"] * 10
tokens = [alphabet.index(c) for c in text]
print("Tokens:", tokens)

net = Tacotron(
    mel_dim=config["MEL_DIM"],
    attn_bias=config["ATTN_BIAS"],
    rr=config["RR"],
    max_rr=config["MAX_RR"],
    mel_min=config["MEL_MIN"],
    sigmoid_noise=config["SIGMOID_NOISE"],
    pad_token=config["PAD_TOKEN"],
)

_, net, _ = load_ckpt(net, None, args.model)
net = net.eval()
net = jax.device_put(net)
inference_fn = pax.pure(lambda net, text: net.inference(text, max_len=1000))
tokens = jnp.array(tokens, dtype=jnp.int32)
mel = inference_fn(net, tokens[None])
mel = jax.device_get(mel)
np.save(args.output, mel)
