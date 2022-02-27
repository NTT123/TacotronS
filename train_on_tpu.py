import os
import random
import time
from functools import partial
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import opax
import pax
import tensorflow as tf

from tacotron import Tacotron
from utils import load_ckpt, load_config, save_ckpt

# TPU setup
if "COLAB_TPU_ADDR" in os.environ:
    jax.tools.colab_tpu.setup_tpu()
DEVICES = jax.devices()
NUM_DEVICES = len(DEVICES)
STEPS_PER_CALL = 10
print("Devices:", DEVICES)

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
TEST_DATA_SIZE = 100  # no testing when training on TPU
TF_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]
PAD_TOKEN = config["PAD_TOKEN"]
PRENET_DIM = config["PRENET_DIM"]
RNN_DIM = config["RNN_DIM"]
TEXT_DIM = config["TEXT_DIM"]
ATTN_RNN_DIM = config["ATTN_RNN_DIM"]


def make_data_loader(batch_size: int, split: str = "train"):
    """
    return a dataloader of mini-batches
    """
    tfdata = tf.data.experimental.load(str(TF_DATA_DIR))
    tfdata = tfdata.shuffle(len(tfdata), seed=42)
    if split == "train":

        tfdata = tfdata.skip(TEST_DATA_SIZE).cache()
        L = len(tfdata)
        tfdata = tfdata.repeat()
        tfdata = tfdata.shuffle(L, reshuffle_each_iteration=True)
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


def batch_reshape(x, K):
    """
    add a new first dimension
    """
    N, *L = x.shape
    return jnp.reshape(x, (K, N // K, *L))


def _device_put_sharded(sharded_tree):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], DEVICES
    )


def pmap_double_buffer(ds):
    """
    create a double buffer iterator for jax.pmap training
    """
    batch = None
    for next_batch in ds:
        assert next_batch is not None
        next_batch = prepare_train_batch(next_batch)
        next_batch = jax.tree_map(partial(batch_reshape, K=NUM_DEVICES), next_batch)
        next_batch = _device_put_sharded(next_batch)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


def train_step(net: Tacotron, optim: pax.Module, batch):
    """
    one training step
    """
    (loss, net), grads = pax.value_and_grad(loss_fn, has_aux=True)(net, batch, None)
    grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, loss


@partial(jax.pmap, axis_name="i")
def train_multiple_step(net, optim, batch):
    """
    multiple training steps
    """

    def loop(prev, inputs):
        net, optim = prev
        batch = inputs
        net, optim, loss = train_step(net, optim, batch)
        return (net, optim), loss

    state = (net, optim)
    inputs = jax.tree_map(partial(batch_reshape, K=STEPS_PER_CALL), batch)
    state, output = jax.lax.scan(loop, state, inputs)
    net, optim = state
    loss = jnp.mean(output)
    return net, optim, loss


@jax.jit
def gta_prediction(net, batch):
    """
    GTA prediction
    """
    net = net.eval()
    text, mel = batch
    go_frame = net.go_frame(mel.shape[0])[:, None, :]
    input_mel = mel[:, (RR - 1) :: RR][:, :-1]
    input_mel = jnp.concatenate((go_frame, input_mel), axis=1)
    net, predictions = pax.purecall(net, input_mel, text)
    (_, predicted_mel_postnet, _) = predictions
    return mel, predicted_mel_postnet


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


def plot_prediction(step, net, batch):
    """
    plot mel prediction
    """
    eval_net = jax.tree_map(lambda x: x[0], net.eval())
    gt_mel, predicted_mel = gta_prediction(eval_net, batch)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    gt_mel = gt_mel[0].astype(jnp.float32).T
    ax[0].imshow(gt_mel, aspect="auto", origin="lower")
    ax[0].set_title("ground truth")
    predicted_mel = predicted_mel[0].astype(jnp.float32).T
    ax[1].imshow(predicted_mel, aspect="auto", origin="lower")
    ax[1].set_title("prediction")
    plt.savefig(LOG_DIR / f"{MODEL_PREFIX}_mels_{step:07d}.png")
    plt.close()


def train(batch_size: int = BATCH_SIZE, lr: float = LR):
    """
    train tacotron model
    """
    assert batch_size % NUM_DEVICES == 0
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

    def lr_decay(step):
        e = jnp.floor(step * 1.0 / 50_000)
        return jnp.exp2(-e) * lr

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale_by_schedule(lr_decay),
    ).init(net.parameters())

    last_step = -STEPS_PER_CALL
    files = sorted(CKPT_DIR.glob(f"{MODEL_PREFIX}_*.ckpt"))
    if len(files) > 0:
        print("loading", files[-1])
        last_step, net, optim = load_ckpt(net, optim, files[-1])
        net, optim = jax.device_put((net, optim))

    # initialize attn_log
    test_data_loader = make_data_loader(1, "test")
    test_batch = next(iter(test_data_loader.as_numpy_iterator()))
    test_batch = prepare_train_batch(test_batch, random_start=False)
    text, mel = test_batch
    N, L = text.shape
    N, T, D = mel.shape
    net = net.replace(attn_log=jnp.zeros((L, T // RR)))

    # replicate on multiple cores
    net, optim = jax.device_put_replicated((net, optim), DEVICES)

    step = last_step
    data_loader = make_data_loader(batch_size * STEPS_PER_CALL, "train")
    data_iter = pmap_double_buffer(data_loader.as_numpy_iterator())
    start = time.perf_counter()
    loss_sum = 0.0
    log_interval = 10
    for batch in data_iter:
        step = step + STEPS_PER_CALL
        net, optim, loss = train_multiple_step(net, optim, batch)
        loss_sum = loss_sum + loss

        if (step // STEPS_PER_CALL) % log_interval == 0:
            loss = jnp.mean(loss_sum).item() / log_interval
            loss_sum = 0.0
            end = time.perf_counter()
            duration = end - start
            start = end
            print(
                f"step {step:07d}  loss {loss:.3f}  LR {optim[-1].learning_rate[0]:.3e}  {duration:.2f}s",
                flush=True,
            )

        if step % 10_000 == 0:
            net_, optim_ = jax.tree_map(lambda x: x[0], (net, optim))
            save_ckpt(CKPT_DIR, MODEL_PREFIX, step, net_, optim_)
            plot_attn(step, net.attn_log[0])
            plot_prediction(step, net, test_batch)


if __name__ == "__main__":
    fire.Fire(train)
