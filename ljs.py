"""
Preprocess LJSpeech dataset:
- Creating pairs of (wav file, txt file) inside the "wav_dir"
"""

import os
from pathlib import Path

from text import english_cleaners
from utils import load_config


def download_ljs_dataset(data_dir: Path):
    """
    Download ljs dataset and save it to disk
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "LJSpeech-1.1.tar.bz2"
    os.system(
        f"wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2  -O {data_file}"
    )
    os.system(f"tar xjf {data_file} -C {data_dir}")


config = load_config()
RAW_DATA_DIR = Path(config["RAW_DATA_DIR"])
wav_dir = RAW_DATA_DIR / "LJSpeech-1.1/wavs"

if not os.path.isdir(RAW_DATA_DIR):
    download_ljs_dataset(RAW_DATA_DIR)
    with open(RAW_DATA_DIR / "LJSpeech-1.1/metadata.csv", encoding="utf-8") as f:
        for line in f:
            ident, raw_text, normalized_text = line.strip().split("|")
            normalized_text = english_cleaners(normalized_text)
            txt_file = wav_dir / (ident + ".txt")
            with open(txt_file, "w", encoding="utf-8") as g:
                g.write(normalized_text)

print(
    f"Data is prepared in the directory {wav_dir}.\n"
    f"Run 'python data.py {wav_dir}' to create the tensorflow dataset."
)
