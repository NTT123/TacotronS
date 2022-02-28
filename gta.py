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

from utils import create_tacotron_model, load_ckpt, load_config

config = load_config()
RR = config["RR"]
TF_DATA_DIR = config["TF_DATA_DIR"]
TF_GTA_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]


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
    mel_mask = mel != 0
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

    net = create_tacotron_model(config)

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
