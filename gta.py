"""
Generate ground truth-aligned (gta) dataset from trained model.

Usage:
    python gta.py --ckpt=ckpts/mono_tts_tpu_0120000.ckpt
"""


from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import pax
import tensorflow as tf
from tqdm.cli import tqdm

from tacotron import Tacotron
from utils import load_ckpt, load_config

config = load_config()
ATTN_BIAS = config["ATTN_BIAS"]
BATCH_SIZE = config["BATCH_SIZE"]
LOG_DIR = Path(config["LOG_DIR"])
CKPT_DIR = Path(config["CKPT_DIR"])
LR = config["LR"]
MAX_RR = config["MAX_RR"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
MODEL_PREFIX = config["MODEL_PREFIX"]
RR = config["RR"]
SIGMOID_NOISE = config["SIGMOID_NOISE"]
TEST_DATA_SIZE = config["TEST_DATA_SIZE"]
TF_DATA_DIR = config["TF_DATA_DIR"]
TF_GTA_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]
PAD_TOKEN = config["PAD_TOKEN"]
PRENET_DIM = config["PRENET_DIM"]
RNN_DIM = config["RNN_DIM"]
TEXT_DIM = config["TEXT_DIM"]


def prepare_batch(batch):
    """
    Prepare batch for gta data generation
    """
    text, mel = batch
    N, L, D = mel.shape
    L = L // RR * RR
    mel = mel[:, :L]
    return text, mel


@jax.jit
def generate_gta(net, batch):
    """
    Generate gta features
    """
    net = net.eval()
    text, mel = batch
    mel = mel.astype(jnp.float32)
    go_frame = net.go_frame(mel.shape[0])[:, None, :]
    input_mel = mel[:, (RR - 1) :: RR][:, :-1]
    input_mel = jnp.concatenate((go_frame, input_mel), axis=1)
    net, predictions = pax.purecall(net, input_mel, text)
    (_, predicted_mel_postnet, _) = predictions
    assert predicted_mel_postnet.shape == mel.shape
    mel_mask = mel > jnp.log(MEL_MIN) + 1e-5
    predicted_mel_postnet = jnp.where(mel_mask, predicted_mel_postnet, mel)
    return predicted_mel_postnet


def main():
    """
    create tf gta data
    """
    parser = ArgumentParser(description="generate gta dataset")
    parser.add_argument(
        "--ckpt", type=Path, required=True, help="Path to the checkpoint"
    )
    args = parser.parse_args()

    net = Tacotron(
        mel_dim=MEL_DIM,
        attn_bias=ATTN_BIAS,
        rr=RR,
        max_rr=MAX_RR,
        mel_min=MEL_MIN,
        sigmoid_noise=SIGMOID_NOISE,
        pad_token=PAD_TOKEN,
        prenet_dim=PRENET_DIM,
        rnn_dim=RNN_DIM,
        text_dim=TEXT_DIM,
    )

    _, net, _ = load_ckpt(net, None, args.ckpt)
    net = jax.device_put(net.eval())
    sample_batch = next(iter(tf.data.experimental.load(str(TF_DATA_DIR)).batch(1)))
    data_loader = tf.data.experimental.load(str(TF_DATA_DIR)).batch(1)
    output_signature = tf.TensorSpec(shape=sample_batch[1].shape, dtype=tf.float16)

    def gta_iterator():
        """
        Iterator of gta features
        """
        length = len(data_loader)
        for batch in tqdm(data_loader.as_numpy_iterator(), total=length):
            batch = prepare_batch(batch)
            batch = jax.device_put(batch)
            mel = jax.device_get(generate_gta(net, batch).astype(jnp.float16))
            yield mel

    dataset = tf.data.Dataset.from_generator(
        gta_iterator, output_signature=output_signature
    )
    tf.data.experimental.save(dataset, str(TF_GTA_DATA_DIR))
    print(f"Created a GTA tf dataset at '{TF_GTA_DATA_DIR}'")


if __name__ == "__main__":
    main()
