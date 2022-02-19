"""
Create tensorflow dataset
"""

import random
from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tqdm.cli import tqdm

from utils import load_config

config = load_config()
SAMPLE_RATE = config["SAMPLE_RATE"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
PAD_TOKEN = config["PAD_TOKEN"]
PAD = config["PAD"]
TF_DATA_DIR = Path(config["TF_DATA_DIR"])


def extract_mel(wav_file):
    """
    Extract log-melspectrogram features from a wav file

    - window size: 50 ms
    - hop length : 12.5 ms

    We use np.float16 to save memory.
    """
    # convert to the sample rate `SAMPLE_RATE` if needed
    y, rate = librosa.load(wav_file, sr=SAMPLE_RATE)
    assert rate == SAMPLE_RATE
    hop = int(12.5 * rate / 1000)
    window = int(50 * rate / 1000)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=rate,
        n_mels=MEL_DIM,
        n_fft=2048,
        hop_length=hop,
        win_length=window,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        fmin=0,
        fmax=rate // 2,
    )
    mel: np.ndarray = np.log(mel + MEL_MIN)
    return mel.astype(np.float16).T


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


def get_wav_files(data_dir: Path):
    """
    Get all *.wav files in the data directory.
    """
    files = sorted(data_dir.glob("*.wav"))
    random.Random(42).shuffle(files)
    return files


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

    mels = []
    for wav_file in tqdm(wav_files):
        mels.append(extract_mel(wav_file))

    max_mel_len = max(map(len, mels))
    pad_mels = []
    for mel in tqdm(mels):
        mel = np.pad(
            mel,
            [(0, max_mel_len - mel.shape[0]), (0, 0)],
            constant_values=np.log(MEL_MIN),
        )
        pad_mels.append(mel)

    del mels

    def data_generator():
        data = list(zip(text_tokens, pad_mels))
        for text, mel in data:
            text = np.array(text, dtype=np.int32)
            yield text, mel

    output_signature = (
        tf.TensorSpec(shape=[len(text_tokens[0])], dtype=tf.int32),
        tf.TensorSpec(shape=pad_mels[0].shape, dtype=tf.float16),
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
print(f"Loading data from the directory '{wav_dir}'")
create_tf_data(wav_dir, TF_DATA_DIR)

print(
    f"Created a tensorflow dataset at '{TF_DATA_DIR}'.\n\n"
    f"Run 'python train.py' to train your model."
)
