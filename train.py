import os
import random
import time
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import opax
import pax
import tensorflow as tf
from pax.experimental import apply_scaled_gradients
from tqdm.cli import tqdm

from tacotron import Tacotron
from utils import load_ckpt, load_config, save_ckpt

config = load_config()
ATTN_BIAS = config["ATTN_BIAS"]
BATCH_SIZE = config["BATCH_SIZE"]
LOG_DIR = Path(config["LOG_DIR"])
CKPT_DIR = Path(config["CKPT_DIR"])
LR = config["LR"]
MAX_RR = config["MAX_RR"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
if "MODEL_PREFIX" in os.environ:
    MODEL_PREFIX = os.environ["MODEL_PREFIX"]
else:
    MODEL_PREFIX = config["MODEL_PREFIX"]
RR = config["RR"]
SIGMOID_NOISE = config["SIGMOID_NOISE"]
TEST_DATA_SIZE = config["TEST_DATA_SIZE"]
TF_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]
PAD_TOKEN = config["PAD_TOKEN"]
PRENET_DIM = config["PRENET_DIM"]
RNN_DIM = config["RNN_DIM"]
TEXT_DIM = config["TEXT_DIM"]
ATTN_RNN_DIM = config["ATTN_RNN_DIM"]


def double_buffer(ds):
    """
    create a double-buffer iterator
    """
    batch = None

    for next_batch in ds:
        assert next_batch is not None
        next_batch = jax.device_put(next_batch)
        if batch is not None:
            yield batch
        batch = next_batch

    if batch is not None:
        yield batch


def make_data_loader(batch_size: int, split: str = "train"):
    """
    return a dataloader of mini-batches
    """
    tfdata = tf.data.experimental.load(str(TF_DATA_DIR))
    tfdata = tfdata.shuffle(len(tfdata), seed=42)
    if split == "train":
        tfdata = tfdata.skip(TEST_DATA_SIZE).cache()
        tfdata = tfdata.shuffle(len(tfdata), reshuffle_each_iteration=True)
    elif split == "test":
        tfdata = tfdata.take(TEST_DATA_SIZE).cache()
    tfdata = tfdata.batch(batch_size, drop_remainder=True)
    tfdata = tfdata.prefetch(tf.data.AUTOTUNE)
    return tfdata


def prepare_train_batch(batch, random_start=True):
    """
    Prepare the mini-batch for training:
    - make sure that the sequence length is divisible by the reduce factor RR.
    - randomly select the start frame.
    """
    text, mel = batch
    N, L, D = mel.shape
    L = L // RR * RR
    mel = mel[:, :L]
    if random_start:
        idx = random.randint(0, RR - 1)
    else:
        idx = 0
    if RR > 1:
        mel = mel[:, idx : (idx - RR)]
    return text, mel


def bce_loss(logit, target):
    """
    return binary cross entropy loss
    """
    llh1 = jax.nn.log_sigmoid(logit) * target
    llh2 = jax.nn.log_sigmoid(-logit) * (1 - target)
    return -jnp.mean(llh1 + llh2)


def l1_loss(x, y):
    """
    compute the l1 loss
    """
    delta = x - y
    return jnp.mean(jnp.abs(delta), axis=-1)


def loss_fn(net: Tacotron, batch, scaler=None):
    """
    training loss function
    """
    text, mel = batch
    mel = mel.astype(jnp.float32)
    go_frame = net.go_frame(mel.shape[0])[:, None, :]
    input_mel = mel[:, (RR - 1) :: RR][:, :-1]
    input_mel = jnp.concatenate((go_frame, input_mel), axis=1)
    stop_token = mel[..., 0] == 0
    net, predictions = pax.purecall(net, input_mel, text)
    (predicted_mel, predicted_mel_postnet, predicted_eos) = predictions

    eos_loss = bce_loss(predicted_eos, stop_token)
    post_net_loss = l1_loss(predicted_mel_postnet, mel)
    loss = (l1_loss(predicted_mel, mel) + post_net_loss) / 2
    mel_mask = mel[..., 0] != 0
    # per-frame mel loss
    loss = jnp.sum(loss * mel_mask) / jnp.sum(mel_mask)
    loss = loss + eos_loss * 1e-2
    if scaler is not None:
        loss = scaler.scale(loss)
    return loss, net


fast_loss_fn = jax.jit(loss_fn)


@jax.jit
def train_step(net: Tacotron, optim: pax.Module, scaler, batch):
    """
    one training step
    """
    (loss, net), grads = pax.value_and_grad(loss_fn, has_aux=True)(net, batch, scaler)
    loss = scaler.unscale(loss)
    net, optim, scaler = apply_scaled_gradients(net, optim, scaler, grads)
    return net, optim, scaler, loss


def eval_score(net: Tacotron, data_loader):
    """
    evaluate the model on the test set
    """
    losses = []
    net = net.eval()
    data_iter = double_buffer(data_loader.as_numpy_iterator())
    for batch in data_iter:
        batch = prepare_train_batch(batch, random_start=False)
        loss, _ = fast_loss_fn(net, batch)
        losses.append(loss)
    loss = sum(losses) / len(losses)
    return loss


def plot_attn(step, attn_weight):
    """
    plot attention weights
    """
    plt.figure(figsize=(15, 5))
    plt.matshow(
        jax.device_get(attn_weight), fignum=0, aspect="auto", interpolation="nearest"
    )
    plt.savefig(LOG_DIR / f"{MODEL_PREFIX}_attn_{step:07d}.png")
    plt.close()


def eval_inference(step: int, net: Tacotron, batch):
    """
    evaluate inference mode
    """
    test_text, test_mel = batch
    test_text = test_text[:1]
    test_mel = test_mel[:1]
    inference_fn = pax.pure(lambda net, text: net.inference(text, max_len=400))
    net = net.eval()
    predicted_mel = inference_fn(net, test_text[:1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    del fig
    L = predicted_mel.shape[1]
    ax[0].imshow(
        test_mel[0, :L].T.astype(jnp.float32), origin="lower", interpolation="nearest"
    )
    ax[0].set_title("ground truth")
    ax[1].imshow(
        predicted_mel[0].T.astype(jnp.float32), origin="lower", interpolation="nearest"
    )
    ax[1].set_title("prediction")
    plt.savefig(LOG_DIR / f"{MODEL_PREFIX}_mel_{step:07d}.png")
    plt.close()


def train(batch_size: int = BATCH_SIZE, lr: float = LR):
    """
    train tacotron model
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
        attn_rnn_dim=ATTN_RNN_DIM,
    )

    if USE_MP:
        scaler = jmp.DynamicLossScale(jnp.array(2**15, dtype=jnp.float32))
        net = net.apply(pax.experimental.default_mp_policy)
    else:
        scaler = jmp.NoOpLossScale()

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale(lr),
    ).init(net.parameters())

    data_loader = make_data_loader(batch_size, "train")
    test_data_loader = make_data_loader(batch_size, "test")

    last_step = -1
    files = sorted(CKPT_DIR.glob(f"{MODEL_PREFIX}_*.ckpt"))
    if len(files) > 0:
        print("loading", files[-1])
        last_step, net, optim = load_ckpt(net, optim, files[-1])
        net, optim = jax.device_put((net, optim))

    step = last_step
    losses = []
    start = time.perf_counter()
    start_epoch = (last_step + 1) // len(data_loader)
    test_batch = next(iter(test_data_loader.as_numpy_iterator()))
    test_batch = prepare_train_batch(test_batch)
    # initialize attn_log
    text, mel = test_batch
    N, L = text.shape
    N, T, D = mel.shape
    net = net.replace(attn_log=jnp.zeros((L, T // RR)))

    for epoch in range(start_epoch, 10000):
        losses = []
        data_iter = double_buffer(data_loader.as_numpy_iterator())
        for batch in tqdm(
            data_iter, total=len(data_loader), leave=False, desc=f"epoch {epoch}"
        ):
            batch = prepare_train_batch(batch)
            step = step + 1
            net, optim, scaler, loss = train_step(net, optim, scaler, batch)
            losses.append(loss)
        test_loss = eval_score(net, test_data_loader)
        loss = sum(losses) / len(losses)
        end = time.perf_counter()
        duration = end - start
        start = end
        print(
            "step {:07d}  epoch {:05d}  loss {:.3f}  test loss {:.3f}  gradscale {:.0f}  {:.2f}s".format(
                step, epoch, loss, test_loss, scaler.loss_scale, duration
            ),
            flush=True,
        )

        if epoch % 10 == 0:
            save_ckpt(CKPT_DIR, MODEL_PREFIX, step, net, optim)
            plot_attn(step, net.attn_log)
            eval_inference(step, net, test_batch)


if __name__ == "__main__":
    fire.Fire(train)
