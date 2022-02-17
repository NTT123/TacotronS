import random
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import opax
import pax
import tensorflow as tf
from pax.experimental import apply_scaled_gradients

from tacotron import Tacotron
from utils import load_ckpt, load_config, save_ckpt

config = load_config()
ATTN_BIAS = config["ATTN_BIAS"]
BATCH_SIZE = config["BATCH_SIZE"]
LOG_DIR = Path(config["LOG_DIR"])
LR = config["LR"]
MAX_RR = config["MAX_RR"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
MODEL_PREFIX = config["MODEL_PREFIX"]
RR = config["RR"]
SIGMOID_NOISE = config["SIGMOID_NOISE"]
TEST_DATA_SIZE = config["TEST_DATA_SIZE"]
TF_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]


LOG_DIR.mkdir(parents=True, exist_ok=True)


def double_buffer(ds):
    """
    create a doudble-buffer iterator
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


def loss_fn(net: Tacotron, batch, scaler=None):
    """
    training loss function
    """
    text, mel = batch
    mel = mel.astype(jnp.float32)
    input_mel = jnp.pad(
        mel[:, (RR - 1) :: RR][:, :-1],
        [(0, 0), (1, 0), (0, 0)],
        constant_values=jnp.log(MEL_MIN),
    )
    mel_mask = mel > jnp.log(MEL_MIN) + 1e-5
    stop_token = mel[..., 0] <= jnp.log(MEL_MIN) + 1e-5
    net, (predicted_mel, predicted_eos) = pax.purecall(net, input_mel, text)
    delta = (mel - predicted_mel) * mel_mask
    loss1 = jnp.mean(jnp.square(delta))
    loss2 = jnp.mean(jnp.abs(delta))
    eos_loss = bce_loss(predicted_eos, stop_token)
    loss = (loss1 + loss2) / 2 + eos_loss * 1e-2
    if scaler is None:
        scaler = jmp.NoOpLossScale()
    loss = scaler.scale(loss)
    return loss, net


fast_loss_fn = jax.jit(loss_fn)


@jax.jit
def train_step(
    net: Tacotron, optim: pax.Module, scaler, batch
) -> Tuple[Tacotron, pax.Module, jmp.DynamicLossScale, jnp.ndarray]:
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


def eval_inference(step, net, test_data_loader):
    """
    evaluate inference mode
    """
    test_batch = next(iter(test_data_loader.as_numpy_iterator()))
    test_batch = prepare_train_batch(test_batch)
    test_text, test_mel = test_batch
    test_text = test_text[:1]
    test_mel = test_mel[:1]
    inference_fn = pax.pure(lambda net, text: net.inference(text, max_len=400))
    predicted_mel = inference_fn(net.eval(), test_text[:1])
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
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


net = Tacotron(
    mel_dim=MEL_DIM,
    attn_bias=ATTN_BIAS,
    rr=RR,
    max_rr=MAX_RR,
    mel_min=MEL_MIN,
    sigmoid_noise=SIGMOID_NOISE,
)

if USE_MP:
    scaler = jmp.DynamicLossScale(jnp.array(2 ** 15, dtype=jnp.float32))
    net = net.apply(pax.experimental.default_mp_policy)
else:
    scaler = jmp.NoOpLossScale()

optim = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.scale_by_adam(),
    opax.scale(LR),
).init(net.parameters())

data_loader = make_data_loader(BATCH_SIZE, "train")
test_data_loader = make_data_loader(BATCH_SIZE, "test")


last_step = -1
files = sorted(Path("./ckpts").glob(f"{MODEL_PREFIX}_*.ckpt"))
if len(files) > 0:
    print("loading", files[-1])
    last_step, net, optim = load_ckpt(net, optim, files[-1])


step = last_step
losses = []
start = time.perf_counter()
start_epoch = (last_step + 1) // len(data_loader)
for epoch in range(start_epoch, 10000):
    losses = []
    data_iter = double_buffer(data_loader.as_numpy_iterator())
    for batch in data_iter:
        batch = prepare_train_batch(batch)
        step = step + 1
        last_step = step
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
        )
    )
    if epoch % 5 == 0:
        save_ckpt(step, net, optim)
        plot_attn(step, net.attn_log)
        eval_inference(step, net, test_data_loader)