import os
import random
import time
from functools import partial
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import opax
import pax
import tensorflow as tf

from tacotron import Tacotron
from utils import load_ckpt, load_config, save_ckpt

# TPU setup
if "COLAB_TPU_ADDR" in os.environ:
    jax.tools.colab_tpu.setup_tpu()
devices = jax.devices()
num_devices = len(devices)
steps_per_update = 20
print("Devices:", devices)

config = load_config()
ATTN_BIAS = config["ATTN_BIAS"]
BATCH_SIZE = config["BATCH_SIZE"]
LOG_DIR = Path(config["LOG_DIR"])
CKPT_DIR = Path(config["CKPT_DIR"])
LR = config["LR"]
MAX_RR = config["MAX_RR"]
MEL_DIM = config["MEL_DIM"]
MEL_MIN = config["MEL_MIN"]
MODEL_PREFIX = config["MODEL_PREFIX"]
RR = config["RR"]
SIGMOID_NOISE = config["SIGMOID_NOISE"]
TEST_DATA_SIZE = 100  # no testing when training on TPU
TF_DATA_DIR = config["TF_DATA_DIR"]
USE_MP = config["USE_MP"]
PAD_TOKEN = config["PAD_TOKEN"]


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


def l1_l2_loss(x, y):
    """
    compute the average of l1 and l2 losses
    """
    delta = x - y
    mse = jnp.mean(jnp.square(delta), axis=-1)
    mae = jnp.mean(jnp.abs(delta), axis=-1)
    return (mse + mae) / 2


def loss_fn(net: Tacotron, batch, scaler=None):
    """
    training loss function
    """
    text, mel = batch
    mel = mel.astype(jnp.float32)
    go_frame = net.go_frame(mel.shape[0])[:, None, :]
    input_mel = mel[:, (RR - 1) :: RR][:, :-1]
    input_mel = jnp.concatenate((go_frame, input_mel), axis=1)
    stop_token = mel[..., 0] <= jnp.log(MEL_MIN) + 1e-5
    net, predictions = pax.purecall(net, input_mel, text)
    (predicted_mel, predicted_mel_postnet, predicted_eos) = predictions

    eos_loss = bce_loss(predicted_eos, stop_token)
    post_net_loss = l1_l2_loss(predicted_mel_postnet, mel)
    loss = (l1_l2_loss(predicted_mel, mel) + post_net_loss) / 2
    mel_mask = mel[..., 0] > jnp.log(MEL_MIN) + 1e-5
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
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], devices
    )


def double_buffer(ds):
    """
    create a double buffer iterator
    """
    batch = None
    for next_batch in ds:
        assert next_batch is not None
        next_batch = prepare_train_batch(next_batch)
        next_batch = jax.tree_map(partial(batch_reshape, K=num_devices), next_batch)
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
    inputs = jax.tree_map(partial(batch_reshape, K=steps_per_update), batch)
    state, output = jax.lax.scan(loop, state, inputs)
    net, optim = state
    loss = jnp.mean(output)
    return net, optim, loss


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
    )

    def lr_decay(step):
        e = jnp.floor(step * 1.0 / 50_000)
        return jnp.exp2(-e) * lr

    optim = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale_by_adam(),
        opax.scale_by_schedule(lr_decay),
    ).init(net.parameters())

    data_loader = make_data_loader(batch_size * steps_per_update, "train")
    test_data_loader = make_data_loader(batch_size, "test")

    last_step = -steps_per_update
    files = sorted(CKPT_DIR.glob(f"{MODEL_PREFIX}_*.ckpt"))
    if len(files) > 0:
        print("loading", files[-1])
        last_step, net, optim = load_ckpt(net, optim, files[-1])
        net, optim = jax.device_put((net, optim))

    test_batch = next(iter(test_data_loader.as_numpy_iterator()))
    test_batch = prepare_train_batch(test_batch)

    if last_step < 0:
        # initialize attn_log
        N, L = test_batch[0].shape
        N, T, D = test_batch[1].shape
        net = net.replace(attn_log=jnp.zeros((L, T // RR)))
    net, optim = jax.device_put_replicated((net, optim), devices)

    step = last_step
    data_iter = double_buffer(data_loader.as_numpy_iterator())
    start = time.perf_counter()
    loss_sum = 0.0
    for batch in data_iter:
        step = step + steps_per_update
        net, optim, loss = train_multiple_step(net, optim, batch)
        loss_sum = loss_sum + loss

        if step % (steps_per_update * 10) == 0:
            loss = jnp.mean(loss_sum).item() / 10
            loss_sum = 0.0
            end = time.perf_counter()
            duration = end - start
            start = end
            print(
                f"step {step:07d}  loss {loss:.3f}  LR {optim[-1].learning_rate[0]:.3e}  {duration:.2f}"
            )

        if step % 10_000 == 0:
            net_, optim_ = jax.tree_map(lambda x: x[0], (net, optim))
            save_ckpt(CKPT_DIR, MODEL_PREFIX, step, net_, optim_)

        if step > 400_000:
            break


if __name__ == "__main__":
    fire.Fire(train)