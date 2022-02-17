"""
Preprocess LJSpeech dataset:
- Creating pairs of (wav file, txt file) inside the "wav_dir"
"""

from pathlib import Path

import pooch
from pooch import Untar

from text import english_cleaners


def download_ljs_dataset():
    """
    Download ljs dataset and save it to disk
    """

    file_paths = pooch.retrieve(
        url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        known_hash="md5:c4763be9595ddfa79c2fc6eaeb3b6c8e",
        processor=Untar(),
        progressbar=True,
    )
    readme_file = Path(sorted(file_paths)[0])
    print(readme_file)
    assert readme_file.name == "README"
    return readme_file.parent


data_dir = download_ljs_dataset()
wav_dir = data_dir / "wavs"

with open(data_dir / "metadata.csv", encoding="utf-8") as f:
    for line in f:
        ident, raw_text, normalized_text = line.strip().split("|")
        normalized_text = english_cleaners(normalized_text)
        txt_file = wav_dir / (ident + ".txt")
        with open(txt_file, "w", encoding="utf-8") as g:
            g.write(normalized_text)

print(
    f"Data is prepared in the directory {wav_dir}.\n\n"
    f"Run 'python data.py {wav_dir}' to create the tensorflow dataset."
)
