# TacotronS
Tacotron with stepwise monotonic attention


Step 1: download and prepare raw dataset

    python ljs.py


Step 2: prepare tf dataset

    python data.py <wav_dir>


Step 3: train tacotron

    python train.py