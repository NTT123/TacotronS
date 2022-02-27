"""
Tacotron + stepwise monotonic attention
"""

import jax
import jax.numpy as jnp
import pax


def conv_block(in_ft, out_ft, kernel_size, activation_fn, use_dropout, dilation=1):
    """
    Conv >> Layernorm >> activation >> Dropout
    """
    f = pax.Sequential()
    f >>= pax.Conv1D(
        in_ft, out_ft, kernel_size, padding="SAME", with_bias=False, rate=dilation
    )
    f >>= pax.LayerNorm(out_ft, -1, True, True)
    if activation_fn is not None:
        f >>= activation_fn
    if use_dropout:
        f >>= pax.Dropout(0.5)
    return f


class HighwayBlock(pax.Module):
    """
    Highway block
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.fc = pax.Linear(dim, 2 * dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        t, h = jnp.split(self.fc(x), 2, axis=-1)
        t = jax.nn.sigmoid(t - 1.0)  # bias toward keeping x
        h = jax.nn.relu(h)
        x = x * (1.0 - t) + h * t
        return x


class BiLSTM(pax.Module):
    """
    Bidirectional LSTM
    """

    def __init__(self, dim):
        super().__init__()

        self.rnn_fwd = pax.LSTM(dim, dim)
        self.rnn_bwd = pax.LSTM(dim, dim)

    def __call__(self, x, reset_masks):
        N = x.shape[0]
        x_fwd = x
        x_bwd = jnp.flip(x, axis=1)
        x_fwd_states = self.rnn_fwd.initial_state(N)
        x_bwd_states = self.rnn_bwd.initial_state(N)
        x_fwd_states, x_fwd = pax.scan(
            self.rnn_fwd, x_fwd_states, x_fwd, time_major=False
        )

        reset_masks = jnp.flip(reset_masks, axis=1)
        x_bwd_states0 = x_bwd_states

        def rnn_reset_core(prev, inputs):
            x, reset_mask = inputs

            def reset_state(x0, xt):
                return jnp.where(reset_mask, x0, xt)

            state, _ = self.rnn_bwd(prev, x)
            state = jax.tree_map(reset_state, x_bwd_states0, state)
            return state, state.hidden

        x_bwd_states, x_bwd = pax.scan(
            rnn_reset_core, x_bwd_states, (x_bwd, reset_masks), time_major=False
        )
        x_bwd = jnp.flip(x_bwd, axis=1)
        x = jnp.concatenate((x_fwd, x_bwd), axis=-1)
        return x


class CHR(pax.Module):
    """
    Convs >> Highway net >> RNN
    """

    def __init__(self, dim):
        super().__init__()
        self.convs = [
            conv_block(dim, dim, 3, jax.nn.relu, False, dilation=2**i)
            for i in range(5)
        ]
        self.highway = pax.Sequential(
            HighwayBlock(dim), HighwayBlock(dim), HighwayBlock(dim), HighwayBlock(dim)
        )
        self.rnn = BiLSTM(dim)

    def __call__(self, x, x_mask):
        residual = x
        for f in self.convs:
            residual = f(residual * x_mask)
        x = x + residual
        x = self.highway(x)
        x = self.rnn(x * x_mask, reset_masks=1 - x_mask)
        return x * x_mask


class PreNet(pax.Module):
    """
    Linear >> relu >> dropout >> Linear >> relu >> dropout
    """

    def __init__(self, input_dim, hidden_dim, output_dim, always_dropout=True):
        super().__init__()
        self.fc1 = pax.Linear(input_dim, hidden_dim)
        self.fc2 = pax.Linear(hidden_dim, output_dim)
        self.rng_seq = pax.RngSeq()
        self.always_dropout = always_dropout

    def __call__(self, x, k1=None, k2=None):
        x = self.fc1(x)
        x = jax.nn.relu(x)
        if self.always_dropout or self.training:
            if k1 is None:
                k1 = self.rng_seq.next_rng_key()
            x = pax.dropout(k1, 0.5, x)
        x = self.fc2(x)
        x = jax.nn.relu(x)
        if self.always_dropout or self.training:
            if k2 is None:
                k2 = self.rng_seq.next_rng_key()
            x = pax.dropout(k2, 0.5, x)
        return x


class Tacotron(pax.Module):
    """
    Tacotron TTS model.

    It uses stepwise monotonic attention for robust attention.
    """

    def __init__(
        self,
        mel_dim: int,
        attn_bias,
        rr,
        max_rr,
        mel_min,
        sigmoid_noise,
        pad_token,
        prenet_dim=256,
        attn_hidden_dim=128,
        rnn_dim=512,
        postnet_dim=512,
        text_dim=256,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.rr = rr
        self.max_rr = max_rr
        self.mel_dim = mel_dim
        self.mel_min = mel_min
        self.sigmoid_noise = sigmoid_noise
        self.pad_token = pad_token
        self.prenet_dim = prenet_dim

        # encoder submodules
        self.encoder_embed = pax.Embed(256, text_dim)
        self.encoder_pre_net = PreNet(text_dim, 256, prenet_dim, always_dropout=True)
        self.encoder_cbhg = CHR(prenet_dim)

        # random key generator
        self.rng_seq = pax.RngSeq()

        # pre-net
        self.decoder_pre_net = PreNet(mel_dim, 256, prenet_dim, always_dropout=True)

        # decoder submodules
        self.attn_rnn = pax.LSTM(prenet_dim + prenet_dim * 2, rnn_dim)
        self.text_key_fc = pax.Linear(prenet_dim * 2, attn_hidden_dim, with_bias=True)
        self.attn_query_fc = pax.Linear(rnn_dim, attn_hidden_dim, with_bias=False)

        self.attn_V = pax.Linear(attn_hidden_dim, 1, with_bias=False)
        self.attn_V_weight_norm = jnp.array(1.0 / jnp.sqrt(attn_hidden_dim))
        self.attn_V_bias = jnp.array(attn_bias)
        self.attn_log = jnp.zeros((1,))

        self.decoder_rnn1 = pax.LSTM(rnn_dim, rnn_dim)
        self.decoder_rnn2 = pax.LSTM(rnn_dim, rnn_dim)
        # mel + end-of-sequence token
        self.output_fc = pax.Linear(rnn_dim, (mel_dim + 1) * max_rr, with_bias=True)

        # post-net
        self.post_net = pax.Sequential(
            conv_block(mel_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, mel_dim, 5, None, True),
        )

    parameters = pax.parameters_method("attn_V_weight_norm", "attn_V_bias")

    def encode_text(self, text: jnp.ndarray) -> jnp.ndarray:
        """
        Encode text to a sequence of real vectors
        """
        N, L = text.shape
        text_mask = (text != self.pad_token)[..., None]
        x = self.encoder_embed(text)
        x = self.encoder_pre_net(x)
        x = self.encoder_cbhg(x, text_mask)
        return x

    def go_frame(self, batch_size: int) -> jnp.ndarray:
        """
        return the go frame
        """
        return jnp.ones((batch_size, self.mel_dim)) * jnp.log(self.mel_min)

    def decoder_initial_state(self, N: int, L: int):
        """
        setup decoder initial state
        """
        attn_context = jnp.zeros((N, self.prenet_dim * 2))
        attn_pr = jax.nn.one_hot(
            jnp.zeros((N,), dtype=jnp.int32), num_classes=L, axis=-1
        )

        attn_state = (self.attn_rnn.initial_state(N), attn_context, attn_pr)
        decoder_rnn_states = (
            self.decoder_rnn1.initial_state(N),
            self.decoder_rnn2.initial_state(N),
        )
        return attn_state, decoder_rnn_states

    def monotonic_attention(self, prev_state, inputs, envs):
        """
        Stepwise monotonic attention
        """
        attn_rnn_state, attn_context, prev_attn_pr = prev_state
        x, attn_rng_key = inputs
        text, text_key = envs
        attn_rnn_input = jnp.concatenate((x, attn_context), axis=-1)
        attn_rnn_state, attn_rnn_output = self.attn_rnn(attn_rnn_state, attn_rnn_input)
        attn_query_input = jax.nn.relu(attn_rnn_output + attn_context)
        attn_query = self.attn_query_fc(attn_query_input)
        attn_hidden = jnp.tanh(attn_query[:, None, :] + text_key)
        score = self.attn_V(attn_hidden)
        score = jnp.squeeze(score, axis=-1)
        weight_norm = jnp.linalg.norm(self.attn_V.weight)
        score = score * (self.attn_V_weight_norm / weight_norm)
        score = score + self.attn_V_bias
        noise = jax.random.normal(attn_rng_key, score.shape) * self.sigmoid_noise
        pr_stay = jax.nn.sigmoid(score + noise)
        pr_move = 1.0 - pr_stay
        pr_new_location = pr_move * prev_attn_pr
        pr_new_location = jnp.pad(
            pr_new_location[:, :-1], ((0, 0), (1, 0)), constant_values=0
        )
        attn_pr = pr_stay * prev_attn_pr + pr_new_location
        attn_context = jnp.einsum("NL,NLD->ND", attn_pr, text)
        new_state = (attn_rnn_state, attn_context, attn_pr)
        return new_state, attn_rnn_output

    def zoneout_lstm(self, lstm_core, rng_key, zoneout_pr=0.1):
        """
        Return a zoneout lstm core.

        It will zoneout the new hidden states and keep the new cell states unchanged.
        """

        def core(state, x):
            new_state, _ = lstm_core(state, x)
            h_old = state.hidden
            h_new = new_state.hidden
            mask = jax.random.bernoulli(rng_key, zoneout_pr, h_old.shape)
            h_new = h_old * mask + h_new * (1.0 - mask)
            return pax.LSTMState(h_new, new_state.cell), h_new

        return core

    def decoder_step(
        self,
        attn_state,
        decoder_rnn_states,
        rng_key,
        mel,
        text,
        text_key,
        call_pre_net=False,
    ):
        """
        One decoder step
        """
        if call_pre_net:
            k1, k2, zk1, zk2, rng_key, rng_key_next = jax.random.split(rng_key, 6)
            mel = self.decoder_pre_net(mel, k1, k2)
        else:
            zk1, zk2, rng_key, rng_key_next = jax.random.split(rng_key, 4)
        attn_inputs = (mel, rng_key)
        attn_envs = (text, text_key)
        attn_state, attn_rnn_output = self.monotonic_attention(
            attn_state, attn_inputs, attn_envs
        )
        (_, attn_context, attn_pr) = attn_state
        (decoder_rnn_state1, decoder_rnn_state2) = decoder_rnn_states
        decoder_rnn1_input = attn_rnn_output + attn_context
        decoder_rnn1 = self.zoneout_lstm(self.decoder_rnn1, zk1)
        decoder_rnn_state1, decoder_rnn_output1 = decoder_rnn1(
            decoder_rnn_state1, decoder_rnn1_input
        )
        decoder_rnn2_input = decoder_rnn1_input + decoder_rnn_output1
        decoder_rnn2 = self.zoneout_lstm(self.decoder_rnn2, zk2)
        decoder_rnn_state2, decoder_rnn_output2 = decoder_rnn2(
            decoder_rnn_state2, decoder_rnn2_input
        )
        x = decoder_rnn1_input + decoder_rnn_output1 + decoder_rnn_output2
        x = jax.nn.relu(x)
        decoder_rnn_states = (decoder_rnn_state1, decoder_rnn_state2)
        return attn_state, decoder_rnn_states, rng_key_next, x, attn_pr[0]

    def inference(self, text, seed=42, max_len=1000):
        """
        text to mel
        """
        text = self.encode_text(text)
        text_key = self.text_key_fc(text)
        N, L, D = text.shape
        mel = self.go_frame(N)

        @jax.jit
        def step(attn_state, decoder_rnn_states, rng_key, mel):
            attn_state, decoder_rnn_states, rng_key, x, _ = self.decoder_step(
                attn_state,
                decoder_rnn_states,
                rng_key,
                mel,
                text,
                text_key,
                call_pre_net=True,
            )
            x = self.output_fc(x)
            N, D2 = x.shape
            x = jnp.reshape(x, (N, self.max_rr, D2 // self.max_rr))
            x = x[:, : self.rr, :]
            x = jnp.reshape(x, (N, self.rr, -1))
            mel = x[..., :-1]
            eos = x[..., -1]
            return attn_state, decoder_rnn_states, rng_key, (mel, eos)

        attn_state, decoder_rnn_states = self.decoder_initial_state(N, L)
        rng_key = jax.random.PRNGKey(seed)
        mels = []
        count = 0
        while True:
            count = count + 1
            attn_state, decoder_rnn_states, rng_key, (mel, eos) = step(
                attn_state, decoder_rnn_states, rng_key, mel
            )
            mels.append(mel)
            if eos[0, -1].item() > 0 or count > max_len:
                break

            mel = mel[:, -1, :]

        mels = jnp.concatenate(mels, axis=1)
        mel = mel + self.post_net(mel)
        return mels

    def decode(self, mel, text):
        """
        Attention mechanism + Decoder
        """
        text_key = self.text_key_fc(text)

        def scan_fn(prev_states, inputs):
            attn_state, decoder_rnn_states = prev_states
            x, rng_key = inputs
            attn_state, decoder_rnn_states, _, output, attn_pr = self.decoder_step(
                attn_state, decoder_rnn_states, rng_key, x, text, text_key
            )
            states = (attn_state, decoder_rnn_states)
            return states, (output, attn_pr)

        N, L, D = text.shape
        decoder_states = self.decoder_initial_state(N, L)
        rng_keys = self.rng_seq.next_rng_key(mel.shape[1])
        rng_keys = jnp.stack(rng_keys, axis=1)
        decoder_states, (x, attn_log) = pax.scan(
            scan_fn,
            decoder_states,
            (mel, rng_keys),
            time_major=False,
        )
        self.attn_log = attn_log
        del decoder_states
        x = self.output_fc(x)

        N, T2, D2 = x.shape
        x = jnp.reshape(x, (N, T2, self.max_rr, D2 // self.max_rr))
        x = x[:, :, : self.rr, :]
        x = jnp.reshape(x, (N, T2 * self.rr, -1))
        mel = x[..., :-1]
        eos = x[..., -1]
        return mel, eos

    def __call__(self, mel: jnp.ndarray, text: jnp.ndarray):
        text = self.encode_text(text)
        mel = self.decoder_pre_net(mel)
        mel, eos = self.decode(mel, text)
        return mel, mel + self.post_net(mel), eos
