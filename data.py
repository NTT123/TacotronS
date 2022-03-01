"""
Create tensorflow dataset
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import tensorflow as tf
from tqdm.cli import tqdm

from dsp import MelFilter
from utils import get_wav_files, load_config

config = load_config()
SAMPLE_RATE = config["SAMPLE_RATE"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
PAD_TOKEN = config["PAD_TOKEN"]
PAD = config["PAD"]
TF_DATA_DIR = Path(config["TF_DATA_DIR"])
SAMPLE_RATE = config["SAMPLE_RATE"]
WINDOW_LENGTH = SAMPLE_RATE * 500 // 10_000  # 50.0 ms
HOP_LENGTH = SAMPLE_RATE * 125 // 10_000  # 12.5 ms
# Extract log-melspectrogram features from a wav file
# - window size: 50 ms
# - hop length : 12.5 ms
mel_filter = MelFilter(
    sample_rate=SAMPLE_RATE,
    n_fft=2048,
    window_length=WINDOW_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=MEL_DIM,
    fmin=0,
    fmax=SAMPLE_RATE // 2,
    mel_min=MEL_MIN,
)
mel_filter = jax.jit(mel_filter)


def get_transcripts(wav_files):
    """
    Read all *.txt files that corresponding to *.wav files.
    """
    texts = []
    for file_path in wav_files:
        txt_path = file_path.with_suffix(".txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            assert PAD not in text
            texts.append(text)
    return texts


def get_alphabet(wav_files):
    """
    Return a list of all characters in the transcripts.

    The padding character is also included.
    """
    texts = get_transcripts(wav_files)
    alphabet = [PAD] + sorted(set("".join(texts)))
    return alphabet


def create_tf_data(data_dir: Path, output_dir: Path):
    """
    Create a tensorflow dataset
    """
    wav_files = get_wav_files(data_dir)
    texts = get_transcripts(wav_files)
    maxlen = max(map(len, texts))
    padded_texts = [l + PAD * (maxlen - len(l)) for l in texts]
    alphabet = get_alphabet(wav_files)
    assert PAD_TOKEN == alphabet.index(PAD)
    text_tokens = []
    for text in tqdm(padded_texts):
        o = []
        for c in text:
            o.append(alphabet.index(c))
        text_tokens.append(o)
    sorted_files = sorted(wav_files, key=os.path.getsize)
    y, _ = librosa.load(sorted_files[-1], sr=SAMPLE_RATE, res_type="soxr_hq")
    mel = mel_filter(y[None])[0].astype(jnp.float16)
    mel_shape = mel.shape
    max_wav_len = len(y)

    def data_generator():
        data = list(zip(text_tokens, wav_files))
        random.Random(42).shuffle(data)
        data = tqdm(data)
        for text, wav_file in data:
            wav, rate = librosa.load(wav_file, sr=SAMPLE_RATE, res_type="soxr_hq")
            assert rate == SAMPLE_RATE
            assert max_wav_len >= len(wav)
            wav = wav / max(1.0, np.max(np.abs(wav)))  # rescale to avoid overflow
            pads = [(0, max_wav_len - wav.shape[0])]
            mel_len = len(wav) // HOP_LENGTH - WINDOW_LENGTH // HOP_LENGTH + 1
            wav = np.pad(wav, pads, constant_values=0)
            mel = mel_filter(wav[None])[0].astype(jnp.float16)
            mel = mel.at[mel_len:].set(0)
            mel = jax.device_get(mel)
            text = np.array(text, dtype=np.int32)
            yield text, mel

    output_signature = (
        tf.TensorSpec(shape=[len(text_tokens[0])], dtype=tf.int32),
        tf.TensorSpec(shape=mel_shape, dtype=tf.float16),
    )
    dataset = tf.data.Dataset.from_generator(
        data_generator, output_signature=output_signature
    )
    tf.data.experimental.save(dataset, str(output_dir))

    # save the alphabet for inference mode
    with open(output_dir / "alphabet.txt", "w", encoding="utf-8") as file:
        for ch in alphabet:
            file.write(ch + "\n")


parser = ArgumentParser()
parser.add_argument("wav_dir", type=Path)
wav_dir = parser.parse_args().wav_dir

# prepare tensorflow dataset
print(f"Loading data from directory '{wav_dir}'")
create_tf_data(wav_dir, TF_DATA_DIR)

print(
    f"Created a tensorflow dataset at '{TF_DATA_DIR}'.\n\n"
    f"Run 'python train.py' to train your model."
)
