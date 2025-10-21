import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import functools


# TRANSFORMER
class EncoderBlock(nn.Module):
    hidden_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int
    dim_feedforward: int
    dropout_prob: float = 0.0

    def setup(self):
        # Attention layer
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(
                self.dim_feedforward,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=constant(0.0),
            ),
            nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=constant(0.0),
            ),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        if mask is not None:
            mask = jnp.repeat(
                nn.make_attention_mask(mask, mask), self.num_heads, axis=-3
            )
        attended = self.self_attn(
            inputs_q=x, inputs_kv=x, mask=mask, deterministic=deterministic
        )

        x = self.norm1(attended + x)
        x = x + self.dropout(x, deterministic=deterministic)

        # MLP part
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        x = self.norm2(feedforward + x)
        x = x + self.dropout(x, deterministic=deterministic)

        return x


class Embedder(nn.Module):
    hidden_dim: int
    activation: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        if self.activation:
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return x


class ScannedTransformer(nn.Module):

    hidden_dim: int
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float = 0
    deterministic: bool = True
    return_embeddings: bool = False

    def setup(self):
        self.encoders = [
            EncoderBlock(
                self.hidden_dim,
                self.transf_num_heads,
                self.transf_dim_feedforward,
                self.transf_dropout_prob,
            )
            for _ in range(self.transf_num_layers)
        ]

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        hs = carry
        embeddings, mask, done = x

        # reset hidden state and add
        hs = jnp.where(
            done[:, np.newaxis],  # batch_wize, 1,
            self.initialize_carry(
                *done.shape, self.hidden_dim
            ),  # batch_size, hidden_dim
            hs,  # batch size, hidden_dim
        )
        embeddings = jnp.concatenate(
            (
                hs[..., np.newaxis, :],  # batch size, 1, hidden_dim
                embeddings,
            ),
            axis=-2,
        )
        for layer in self.encoders:
            embeddings = layer(embeddings, mask=mask, deterministic=self.deterministic)
        hs = embeddings[..., 0, :]  # batch size, hidden_dim

        # as y return the entire embeddings if required (i.e. transformer mixer), otherwise only agents' hs embeddings
        if self.return_embeddings:
            return hs, embeddings
        else:
            return hs, hs

    @staticmethod
    def initialize_carry(*shape):
        return jnp.zeros(shape)


class PPOActorTransformer(nn.Module):
    action_dim: int
    hidden_size: int
    num_layers: int = 2
    num_heads: int = 8
    ff_dim: int = 128

    @nn.compact
    def __call__(self, hs, x, return_all_hs=False):

        ins, resets, avail_actions = x
        embeddings = Embedder(
            self.hidden_size,
        )(ins)

        last_hs, hidden_states = ScannedTransformer(
            hidden_dim=self.hidden_size,
            transf_num_layers=self.num_layers,
            transf_num_heads=self.num_heads,
            transf_dim_feedforward=self.ff_dim,
            deterministic=True,
            return_embeddings=False,
        )(hs, (embeddings, None, resets))

        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(hidden_states)

        unavail_actions = 1 - avail_actions
        action_logits = logits - (unavail_actions * 1e10)

        if return_all_hs:
            return last_hs, (hidden_states, action_logits)
        else:
            return last_hs, action_logits
