"""
Generate ground truth-aligned (gta) dataset from trained model.

Usage:
    python gta.py \
        --ckpt=ckpts/mono_tts_tpu_0120000.ckpt \
        --wave-dir=./wavs \
        --output-dir=./gta_mels
"""


from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pax
import tensorflow as tf
from tqdm.cli import tqdm

from utils import create_tacotron_model, get_wav_files, load_ckpt, load_config

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
    parser.add_argument(
        "--wave-dir", type=Path, required=True, help="Path to wave directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Path to output directory"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    net = create_tacotron_model(config)

    _, net, _ = load_ckpt(net, None, args.ckpt)
    net = jax.device_put(net.eval())
    data_loader = tf.data.experimental.load(str(TF_DATA_DIR)).batch(1)
    wav_files = get_wav_files(args.wave_dir)

    length = len(data_loader)

    for wav_file, batch in tqdm(
        zip(wav_files, data_loader.as_numpy_iterator()), total=length
    ):
        mel_file = args.output_dir / f"{wav_file.stem}.mel.npy"
        batch = prepare_batch(batch)
        batch = jax.device_put(batch)
        mel = jax.device_get(generate_gta(net, batch).astype(jnp.float16))
        np.save(mel_file, mel[0])


if __name__ == "__main__":
    main()
