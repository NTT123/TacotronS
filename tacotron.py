import jax
import jax.numpy as jnp
import pax


class Tacotron(pax.Module):
    """
    Tacotron TTS model.

    It uses stepwise monotonic attention for robust attention.
    """

    def __init__(self, mel_dim: int, attn_bias, rr, max_rr, mel_min, sigmoid_noise):
        super().__init__()
        attn_hidden_dim = 128
        prenet_dim = 256
        rnn_dim = 512
        text_dim = rnn_dim

        self.text_dim = text_dim
        self.rr = rr
        self.max_rr = max_rr
        self.mel_dim = mel_dim
        self.mel_min = mel_min
        self.sigmoid_noise = sigmoid_noise

        # encoder submodules
        self.encoder_embed = pax.Embed(256, text_dim)

        def conv_block():
            return pax.Sequential(
                pax.Conv1D(text_dim, text_dim, 5, padding="SAME"),
                pax.BatchNorm1D(text_dim, True, True, 0.99),
                jax.nn.relu,
                pax.Dropout(0.5),
                name="conv_block",
            )

        self.encoder_convs = pax.Sequential(conv_block(), conv_block(), conv_block())
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

        self.mel_rnn1 = pax.LSTM(rnn_dim, rnn_dim)
        self.mel_rnn2 = pax.LSTM(rnn_dim, rnn_dim)
        # mel + end-of-sequence token
        self.output_fc = pax.Linear(rnn_dim, (mel_dim + 1) * max_rr)

    parameters = pax.parameters_method(
        "attn_score_weight", "attn_score_weight_norm", "attn_score_bias"
    )

    def encode_text(self, text: jnp.ndarray) -> jnp.ndarray:
        """
        Encode text to a sequence of real vectors
        """
        N, L = text.shape
        x = self.encoder_embed(text)
        x = self.encoder_convs(x)
        x_fwd = x
        x_bwd = jnp.flip(x, axis=1)
        x_fwd_states = self.encoder_gru_fwd.initial_state(N)
        x_bwd_states = self.encoder_gru_bwd.initial_state(N)
        x_fwd_states, x_fwd = pax.scan(
            self.encoder_gru_fwd, x_fwd_states, x_fwd, time_major=False
        )
        x_bwd_states, x_bwd = pax.scan(
            self.encoder_gru_bwd, x_bwd_states, x_bwd, time_major=False
        )
        x_bwd = jnp.flip(x_bwd, axis=1)
        x = jnp.concatenate((x_fwd, x_bwd), axis=-1)
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
        states = (
            self.attn_rnn.initial_state(N),
            self.mel_rnn1.initial_state(N),
            self.mel_rnn2.initial_state(N),
            attn_context,
            attn_pr,
        )
        return states

    def monotonic_attention(
        self,
        attn_rnn_state,
        attn_context,
        x,
        text,
        text_key,
        attn_rng_key,
        prev_attn_pr,
    ):
        """
        Stepwise monotonic attention
        """
        attn_rnn_input = jnp.concatenate((x, attn_context), axis=-1)
        attn_rnn_state, attn_rnn_output = self.attn_rnn(attn_rnn_state, attn_rnn_input)
        attn_query_input = jax.nn.relu(attn_rnn_output + attn_context)
        attn_query = self.attn_query_fc(attn_query_input)
        attn_hidden = jnp.tanh(attn_query[:, None, :] + text_key)
        weight_norm = self.attn_score_weight_norm / jnp.linalg.norm(
            self.attn_score_weight
        )
        score = jnp.einsum(
            "NLD,D->NL", attn_hidden, self.attn_score_weight * weight_norm
        )
        score = score + self.attn_score_bias
        if self.training:
            score = (
                score
                + jax.random.normal(attn_rng_key, score.shape) * self.sigmoid_noise
            )
        pr_stay = jax.nn.sigmoid(score)
        pr_move = 1.0 - pr_stay
        pr_new_location = pr_move * prev_attn_pr
        pr_new_location = jnp.pad(
            pr_new_location[:, :-1], ((0, 0), (1, 0)), constant_values=0
        )
        attn_pr = pr_stay * prev_attn_pr + pr_new_location
        attn_context = jnp.einsum("NL,NLD->ND", attn_pr, text)
        return attn_rnn_state, attn_context, attn_pr, attn_rnn_output

    def inference(self, text, seed=42, max_len=1000):
        """
        text to mel
        """
        text = self.encode_text(text)
        text_key = self.text_key_fc(text)
        N, L, D = text.shape
        out = []
        mel = self.go_frame(N)

        @jax.jit
        def step(decoder_state, rng_key, mel):
            k1, k2, rng_key, rng_key_next = jax.random.split(rng_key, 4)
            mel = self.pre_net(mel, k1, k2)
            (
                attn_rnn_state,
                mel_rnn_state1,
                mel_rnn_state2,
                attn_context,
                prev_attn_pr,
            ) = decoder_state
            (
                attn_rnn_state,
                attn_context,
                attn_pr,
                attn_rnn_output,
            ) = self.monotonic_attention(
                attn_rnn_state, attn_context, mel, text, text_key, rng_key, prev_attn_pr
            )

            mel_rnn1_input = jax.nn.relu(attn_rnn_output + attn_context)
            mel_rnn_state1, mel_rnn_output1 = self.mel_rnn1(
                mel_rnn_state1, mel_rnn1_input
            )
            mel_rnn2_input = jax.nn.relu(mel_rnn1_input + mel_rnn_output1)
            mel_rnn_state2, mel_rnn_output2 = self.mel_rnn2(
                mel_rnn_state2, mel_rnn2_input
            )
            x = jax.nn.relu(mel_rnn1_input + mel_rnn_output1 + mel_rnn_output2)
            x = self.output_fc(x)
            N, D2 = x.shape
            x = jnp.reshape(x, (N, self.max_rr, D2 // self.max_rr))
            x = x[:, : self.rr, :]
            x = jnp.reshape(x, (N, self.rr, -1))
            mel = x[..., :-1]
            eos = x[..., -1]
            decoder_state = (
                attn_rnn_state,
                mel_rnn_state1,
                mel_rnn_state2,
                attn_context,
                attn_pr,
            )
            return decoder_state, rng_key_next, (mel, eos)

        decoder_state = self.decoder_initial_state(N, L)
        rng_key = jax.random.PRNGKey(seed)
        mels = []
        count = 0
        while True:
            count = count + 1
            decoder_state, rng_key, (mel, eos) = step(decoder_state, rng_key, mel)
            mels.append(mel)
            if eos[0, -1].item() > 0 or count > max_len:
                break

            mel = mel[:, -1, :]

        mels = jnp.concatenate(mels, axis=1)
        return mels

    def decode(self, mel, text):
        """
        Attention mechanism + Decoder
        """
        text_key = self.text_key_fc(text)

        def decode_loop(prev_states, inputs):
            (
                attn_rnn_state,
                mel_rnn_state1,
                mel_rnn_state2,
                attn_context,
                prev_attn_pr,
            ) = prev_states
            x, attn_rng_key = inputs

            (
                attn_rnn_state,
                attn_context,
                attn_pr,
                attn_rnn_output,
            ) = self.monotonic_attention(
                attn_rnn_state,
                attn_context,
                x,
                text,
                text_key,
                attn_rng_key[0],
                prev_attn_pr,
            )

            ## decode
            mel_rnn1_input = jax.nn.relu(attn_rnn_output + attn_context)
            mel_rnn_state1, mel_rnn_output1 = self.mel_rnn1(
                mel_rnn_state1, mel_rnn1_input
            )
            mel_rnn2_input = jax.nn.relu(mel_rnn1_input + mel_rnn_output1)
            mel_rnn_state2, mel_rnn_output2 = self.mel_rnn2(
                mel_rnn_state2, mel_rnn2_input
            )

            output = jax.nn.relu(mel_rnn1_input + mel_rnn_output1 + mel_rnn_output2)

            return (
                (attn_rnn_state, mel_rnn_state1, mel_rnn_state2, attn_context, attn_pr),
                (output, attn_pr[0]),
            )

        N, L, D = text.shape
        states = self.decoder_initial_state(N, L)
        attn_rng_keys = self.attn_rng.next_rng_key(mel.shape[1])
        attn_rng_keys = jnp.stack(attn_rng_keys)[None]
        states, (x, attn_log) = pax.scan(
            decode_loop, states, (mel, attn_rng_keys), time_major=False
        )
        self.attn_log = attn_log
        del states
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
        return self.decode(mel, text)