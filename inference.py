from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pax

from utils import create_tacotron_model, load_ckpt, load_config

parser = ArgumentParser(description="Convert text to melspectrogram")

parser.add_argument("-m", "--model", type=Path, required=True)
parser.add_argument("-t", "--text", type=str, required=True)
parser.add_argument("-a", "--alphabet-file", type=Path, required=True)
parser.add_argument("-o", "--output", type=Path, required=True)
args = parser.parse_args()

with open(args.alphabet_file, "r", encoding="utf-8") as f:
    alphabet = f.read().split("\n")


text = args.text
print("Input: ", text)

config = load_config()
assert config["PAD"] not in text
assert config["END_CHARACTER"] not in text
text = text + config["END_CHARACTER"] + config["PAD"] * 10
tokens = [alphabet.index(c) for c in text]
print("Tokens:", tokens)

net = create_tacotron_model(config)
_, net, _ = load_ckpt(net, None, args.model)
net = net.eval()
net = jax.device_put(net)
inference_fn = pax.pure(lambda net, text: net.inference(text, max_len=10000))
tokens = jnp.array(tokens, dtype=jnp.int32)
mel = inference_fn(net, tokens[None])
mel = jax.device_get(mel)
np.save(args.output, mel)
