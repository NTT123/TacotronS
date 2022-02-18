import jax
import jax.numpy as jnp
import pax


def conv_block(in_ft, out_ft, kernel_size, activation_fn, use_dropout):
    """conv + batchnorm + activation + dropout"""
    return pax.Sequential(
        pax.Conv1D(in_ft, out_ft, kernel_size, padding="SAME"),
        pax.BatchNorm1D(out_ft, True, True, 0.99),
        activation_fn if activation_fn is not None else pax.Identity(),
        pax.Dropout(0.5) if use_dropout else pax.Identity(),
        name="conv_block",
    )


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
    ):
        super().__init__()
        text_dim = rnn_dim
        self.text_dim = text_dim
        self.rr = rr
        self.max_rr = max_rr
        self.mel_dim = mel_dim
        self.mel_min = mel_min
        self.sigmoid_noise = sigmoid_noise
        self.pad_token = pad_token

        # encoder submodules
        self.encoder_embed = pax.Embed(256, text_dim)

        self.encoder_conv1 = conv_block(text_dim, text_dim, 5, jax.nn.relu, True)
        self.encoder_conv2 = conv_block(text_dim, text_dim, 5, jax.nn.relu, True)
        self.encoder_conv3 = conv_block(text_dim, text_dim, 5, jax.nn.relu, True)

        self.encoder_gru_fwd = pax.GRU(text_dim, text_dim // 2)
        self.encoder_gru_bwd = pax.GRU(text_dim, text_dim // 2)

        # pre-net
        self.pre_net_rng = pax.RngSeq()
        self.pre_net_fc1 = pax.Linear(mel_dim, prenet_dim)
        self.pre_net_fc2 = pax.Linear(prenet_dim, prenet_dim)

        # decoder submodules
        self.attn_rnn = pax.LSTM(prenet_dim + text_dim, rnn_dim)
        self.attn_rng = pax.RngSeq()
        self.text_key_fc = pax.Linear(text_dim, attn_hidden_dim)
        self.attn_query_fc = pax.Linear(rnn_dim, attn_hidden_dim, with_bias=False)

        self.attn_score_weight = jax.random.normal(
            pax.next_rng_key(), [attn_hidden_dim]
        )
        self.attn_score_weight_norm = jnp.array(1.0 / jnp.sqrt(attn_hidden_dim))
        self.attn_score_bias = jnp.array(attn_bias)
        self.attn_log = jnp.zeros((1,))

        self.decoder_rnn1 = pax.LSTM(rnn_dim, rnn_dim)
        self.decoder_rnn2 = pax.LSTM(rnn_dim, rnn_dim)
        # mel + end-of-sequence token
        self.output_fc = pax.Linear(rnn_dim, (mel_dim + 1) * max_rr)

        # post-net
        self.post_net = pax.Sequential(
            conv_block(mel_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, postnet_dim, 5, jax.nn.tanh, True),
            conv_block(postnet_dim, mel_dim, 5, None, True),
        )

    parameters = pax.parameters_method(
        "attn_score_weight", "attn_score_weight_norm", "attn_score_bias"
    )

    def encode_text(self, text: jnp.ndarray) -> jnp.ndarray:
        """
        Encode text to a sequence of real vectors
        """
        N, L = text.shape
        text_mask = (text != self.pad_token)[..., None]
        x = self.encoder_embed(text)
        x = self.encoder_conv1(x * text_mask)
        x = self.encoder_conv2(x * text_mask)
        x = self.encoder_conv3(x * text_mask)
        x_fwd = x
        x_bwd = jnp.flip(x, axis=1)
        x_fwd_states = self.encoder_gru_fwd.initial_state(N)
        x_bwd_states = self.encoder_gru_bwd.initial_state(N)
        x_fwd_states, x_fwd = pax.scan(
            self.encoder_gru_fwd, x_fwd_states, x_fwd, time_major=False
        )

        reset_masks = (text == self.pad_token)[..., None]
        reset_masks = jnp.flip(reset_masks, axis=1)
        x_bwd_states0 = x_bwd_states

        def gru_reset_core(prev, inputs):
            x, reset_mask = inputs

            def reset_state(x0, xt):
                return jnp.where(reset_mask, x0, xt)

            state, _ = self.encoder_gru_bwd(prev, x)
            state = jax.tree_map(reset_state, x_bwd_states0, state)
            return state, state.hidden

        x_bwd_states, x_bwd = pax.scan(
            gru_reset_core, x_bwd_states, (x_bwd, reset_masks), time_major=False
        )
        x_bwd = jnp.flip(x_bwd, axis=1)
        x = jnp.concatenate((x_fwd, x_bwd), axis=-1)
        x = x * text_mask
        return x

    def pre_net(self, mel: jnp.ndarray, rng_key1=None, rng_key2=None) -> jnp.ndarray:
        """
        Transform mel features into randomized features
        """
        x = self.pre_net_fc1(mel)
        x = jax.nn.relu(x)
        if rng_key1 is None:
            rng_key1 = self.pre_net_rng.next_rng_key()
        x = pax.dropout(rng_key1, 0.5, x)
        x = self.pre_net_fc2(x)
        x = jax.nn.relu(x)
        if rng_key2 is None:
            rng_key2 = self.pre_net_rng.next_rng_key()
        x = pax.dropout(rng_key2, 0.5, x)
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
        attn_context = jnp.zeros((N, self.text_dim))
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
        weight_norm = jnp.linalg.norm(self.attn_score_weight)
        score = jnp.dot(attn_hidden, self.attn_score_weight)
        score = score * (self.attn_score_weight_norm / weight_norm)
        score = score + self.attn_score_bias
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
            k1, k2, zk1, zk2, rng_key, rng_key_next = jax.random.split(rng_key, 6)
            mel = self.pre_net(mel, k1, k2)
            attn_inputs = (mel, rng_key)
            attn_envs = (text, text_key)
            attn_state, attn_rnn_output = self.monotonic_attention(
                attn_state, attn_inputs, attn_envs
            )
            (_, attn_context, _) = attn_state
            (decoder_rnn_state1, decoder_rnn_state2) = decoder_rnn_states
            decoder_rnn1_input = jax.nn.relu(attn_rnn_output + attn_context)
            decoder_rnn1 = self.zoneout_lstm(self.decoder_rnn1, zk1)
            decoder_rnn_state1, decoder_rnn_output1 = decoder_rnn1(
                decoder_rnn_state1, decoder_rnn1_input
            )
            decoder_rnn2_input = jax.nn.relu(decoder_rnn1_input + decoder_rnn_output1)
            decoder_rnn2 = self.zoneout_lstm(self.decoder_rnn2, zk2)
            decoder_rnn_state2, decoder_rnn_output2 = decoder_rnn2(
                decoder_rnn_state2, decoder_rnn2_input
            )
            x = jax.nn.relu(
                decoder_rnn1_input + decoder_rnn_output1 + decoder_rnn_output2
            )
            x = self.output_fc(x)
            N, D2 = x.shape
            x = jnp.reshape(x, (N, self.max_rr, D2 // self.max_rr))
            x = x[:, : self.rr, :]
            x = jnp.reshape(x, (N, self.rr, -1))
            mel = x[..., :-1]
            eos = x[..., -1]
            decoder_rnn_states = (decoder_rnn_state1, decoder_rnn_state2)
            return attn_state, decoder_rnn_states, rng_key_next, (mel, eos)

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

        def decode_loop(prev_states, inputs):
            attn_state, decoder_rnn_states = prev_states
            decoder_rnn_state1, decoder_rnn_state2 = decoder_rnn_states
            x, attn_rng_key, zoneout_rng_key = inputs

            attn_input = (x, attn_rng_key[0])
            attn_envs = (text, text_key)

            attn_state, attn_rnn_output = self.monotonic_attention(
                attn_state, attn_input, attn_envs
            )
            (_, attn_context, attn_pr) = attn_state

            # decode
            decoder_rnn1_input = jax.nn.relu(attn_rnn_output + attn_context)
            zoneout_rng_key = zoneout_rng_key[0]
            decoder_rnn1 = self.zoneout_lstm(self.decoder_rnn1, zoneout_rng_key[0])
            decoder_rnn_state1, decoder_rnn_output1 = decoder_rnn1(
                decoder_rnn_state1, decoder_rnn1_input
            )
            decoder_rnn2_input = jax.nn.relu(decoder_rnn1_input + decoder_rnn_output1)
            decoder_rnn2 = self.zoneout_lstm(self.decoder_rnn2, zoneout_rng_key[1])
            decoder_rnn_state2, decoder_rnn_output2 = decoder_rnn2(
                decoder_rnn_state2, decoder_rnn2_input
            )

            output = jax.nn.relu(
                decoder_rnn1_input + decoder_rnn_output1 + decoder_rnn_output2
            )
            decoder_rnn_states = (decoder_rnn_state1, decoder_rnn_state2)
            states = (attn_state, decoder_rnn_states)
            return states, (output, attn_pr[0])

        N, L, D = text.shape
        decoder_states = self.decoder_initial_state(N, L)
        attn_rng_keys = self.attn_rng.next_rng_key(mel.shape[1])
        attn_rng_keys = jnp.stack(attn_rng_keys)[None]
        zoneout_rng_keys = self.attn_rng.next_rng_key(mel.shape[1] * 2)
        zoneout_rng_keys = jnp.stack(zoneout_rng_keys)[None]
        zoneout_rng_keys = zoneout_rng_keys.reshape((1, mel.shape[1], 2, -1))
        decoder_states, (x, attn_log) = pax.scan(
            decode_loop,
            decoder_states,
            (mel, attn_rng_keys, zoneout_rng_keys),
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
        mel = self.pre_net(mel)
        mel, eos = self.decode(mel, text)
        return mel, mel + self.post_net(mel), eos
