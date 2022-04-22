# TacotronS
Tacotron with stepwise monotonic attention


Step 1: download and prepare raw dataset

    python ljs.py


Step 2: prepare tf dataset

    python data.py <wav_dir>


Step 3: train tacotron

For the first 50k training steps, use a reduction factor of 2 to learn a stable attention:

    TRAINING_STEPS=50000 RR=2 python train.py

After that:

    python train.py
