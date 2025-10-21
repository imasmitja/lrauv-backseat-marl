import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """PyTorch equivalent of JAX MultiHeadDotProductAttention"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_prob = dropout_prob

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Linear projections for Q, K, V (without bias to match JAX implementation)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

        # Xavier uniform initialization to match JAX
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Implements attention following the exact Flax MultiHeadDotProductAttention pattern.

        Flax uses: inputs_q=x, inputs_kv=x (self-attention)
        - Projects Q, K, V using DenseGeneral
        - Applies dot_product_attention with einsum patterns
        - Final output projection
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Project to Q, K, V using the loaded JAX weights
        # JAX DenseGeneral projects from (batch, seq, hidden) to (batch, seq, heads, head_dim)
        query = self.q_proj(x)  # (batch, seq, hidden) -> (batch, seq, hidden)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # Reshape to match JAX layout: (batch, seq, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply Flax dot_product_attention_weights logic exactly:
        # 1. Scale queries: query = query / jnp.sqrt(depth).astype(dtype)
        depth = self.head_dim
        query = query / torch.sqrt(
            torch.tensor(depth, dtype=query.dtype, device=query.device)
        )

        # 2. Compute attention weights: jnp.einsum('...qhd,...khd->...hqk', query, key)
        # JAX: (batch, q_len, num_heads, head_dim) x (batch, k_len, num_heads, head_dim) -> (batch, num_heads, q_len, k_len)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, key)

        # 3. Apply mask (if provided)
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask: (batch, seq) -> (batch, heads, seq, seq)
                mask = mask.unsqueeze(1).unsqueeze(1)
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            # Mask with large negative value
            attn_weights = attn_weights.masked_fill(mask == False, -1e10)

        # 4. Softmax: jax.nn.softmax(attn_weights).astype(dtype)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 5. Apply attention to values: jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        # (batch, num_heads, q_len, k_len) x (batch, k_len, num_heads, head_dim) -> (batch, q_len, num_heads, head_dim)
        attended = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)

        # 6. Reshape back to (batch, seq, hidden) for standard Linear output projection
        attended = attended.reshape(batch_size, seq_len, hidden_dim)
        output = self.out_proj(attended)

        return output


class EncoderBlock(nn.Module):
    """PyTorch equivalent of JAX EncoderBlock"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout_prob

        # Self-attention
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_prob)

        # Feed-forward network
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # Xavier uniform initialization to match JAX
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection and layer norm - EXACT JAX ORDER
        # JAX: attended = self.self_attn(...), x = self.norm1(attended + x), x = x + self.dropout(x, ...)
        attended = self.self_attn(x, mask)
        x = self.norm1(attended + x)
        # JAX dropout with deterministic=True has no effect, but there's still an addition!
        # JAX: x = x + self.dropout(x, deterministic=deterministic) which is x = x + x = 2*x
        x = x + x

        # Feed-forward network with residual connection and layer norm - EXACT JAX ORDER
        # JAX: feedforward = self.linear[0](x), feedforward = nn.relu(feedforward), feedforward = self.linear[1](feedforward)
        # JAX: x = self.norm2(feedforward + x), x = x + self.dropout(x, ...)
        feedforward = self.linear1(x)
        feedforward = F.relu(feedforward)
        feedforward = self.linear2(feedforward)

        x = self.norm2(feedforward + x)
        # JAX dropout with deterministic=True has no effect, but there's still an addition!
        # JAX: x = x + self.dropout(x, deterministic=deterministic) which is x = x + x = 2*x
        x = x + x

        return x


class Embedder(nn.Module):
    """PyTorch equivalent of JAX Embedder"""

    def __init__(self, input_dim: int, hidden_dim: int, activation: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.dense = nn.Linear(
            input_dim, hidden_dim
        )  # Note: input_dim should be provided separately
        if activation:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        # Orthogonal initialization to match JAX
        nn.init.orthogonal_(self.dense.weight, gain=np.sqrt(2))
        nn.init.constant_(self.dense.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        if self.activation:
            x = self.layer_norm(x)
            x = F.relu(x)
        return x


class ScannedTransformer(nn.Module):
    """PyTorch equivalent of JAX ScannedTransformer"""

    def __init__(
        self,
        hidden_dim: int,
        transf_num_layers: int,
        transf_num_heads: int,
        transf_dim_feedforward: int,
        transf_dropout_prob: float = 0.0,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transf_num_layers = transf_num_layers
        self.return_embeddings = return_embeddings

        # Create encoder layers
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim,
                    transf_num_heads,
                    transf_dim_feedforward,
                    transf_dropout_prob,
                )
                for _ in range(transf_num_layers)
            ]
        )

    def initialize_carry(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        hidden_state: torch.Tensor,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (batch_size, hidden_dim)
            embeddings: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - attention mask
            done: (batch_size,) - reset signal

        Returns:
            new_hidden_state: (batch_size, hidden_dim)
            output: (batch_size, hidden_dim) or full embeddings if return_embeddings=True
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Reset hidden state where done is True - matching JAX logic exactly
        if done is not None:
            reset_mask = done.unsqueeze(1)  # (batch_size, 1)
            hidden_state = torch.where(
                reset_mask,
                torch.zeros_like(hidden_state),  # Use zeros like JAX initialize_carry
                hidden_state,
            )

        # Concatenate hidden state as first token - this matches JAX exactly
        hidden_state_expanded = hidden_state.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        full_embeddings = torch.cat([hidden_state_expanded, embeddings], dim=1)

        # Apply transformer layers sequentially (mimicking JAX scan behavior)
        for encoder in self.encoders:
            full_embeddings = encoder(full_embeddings, mask)

        # Extract new hidden state (first token) - this matches JAX exactly
        new_hidden_state = full_embeddings[:, 0, :]  # (batch_size, hidden_dim)

        if self.return_embeddings:
            return new_hidden_state, full_embeddings
        else:
            # Return new_hidden_state as both outputs to match JAX scan signature
            return new_hidden_state, new_hidden_state


class PPOActorTransformer(nn.Module):
    """PyTorch equivalent of JAX PPOActorTransformer"""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 128,
        matrix_obs: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.matrix_obs = matrix_obs

        # Embedder for input features
        # For matrix_obs=True, input should be per-entity features (6 in this case)
        # For matrix_obs=False, input should be total flattened dimension
        embedder_input_dim = 6 if matrix_obs else input_dim
        self.embedder = nn.Linear(embedder_input_dim, hidden_size)
        self.embedder_norm = nn.LayerNorm(hidden_size)
        nn.init.orthogonal_(self.embedder.weight, gain=np.sqrt(2))
        nn.init.constant_(self.embedder.bias, 0.0)

        # Transformer
        self.transformer = ScannedTransformer(
            hidden_dim=hidden_size,
            transf_num_layers=num_layers,
            transf_num_heads=num_heads,
            transf_dim_feedforward=ff_dim,
            transf_dropout_prob=0.0,
            return_embeddings=False,
        )

        # Action head
        self.action_head = nn.Linear(hidden_size, action_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=np.sqrt(2))
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(
        self,
        hidden_state: torch.Tensor,
        observations: torch.Tensor,
        reset_done: torch.Tensor,
        avail_actions: torch.Tensor,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: (batch_size, hidden_size)
            observations: (1, batch_size, seq_len, input_dim) or (1, batch_size, input_dim) - with extra batch dim
            reset_done: (1, batch_size,) - with extra batch dim
            avail_actions: (batch_size, action_dim)

        Returns:
            new_hidden_state: (batch_size, hidden_size)
            action_logits: (batch_size, action_dim)
        """
        # print(f"Input observations shape: {observations.shape}")
        # print(f"Input reset_done shape: {reset_done.shape}")

        # Remove the extra batch dimension immediately to match JAX processing
        # JAX: ins has shape like (batch_size, seq_len, features)
        # PyTorch gets: (1, batch_size, seq_len, features)
        if observations.dim() == 4:  # (1, batch_size, seq_len, input_dim)
            observations = observations.squeeze(0)  # (batch_size, seq_len, input_dim)
        elif observations.dim() == 3:  # (1, batch_size, input_dim)
            observations = observations.squeeze(0)  # (batch_size, input_dim)

        if reset_done.dim() == 2:  # (1, batch_size)
            reset_done = reset_done.squeeze(0)  # (batch_size,)

        embeddings = self.embedder(observations)
        embeddings = self.embedder_norm(embeddings)
        embeddings = F.relu(embeddings)

        # Apply transformer
        new_hidden_state, transformer_output = self.transformer(
            hidden_state, embeddings, done=reset_done
        )

        # Generate action logits
        logits = self.action_head(transformer_output)

        # Mask unavailable actions
        unavail_actions = 1 - avail_actions
        action_logits = logits - (unavail_actions * 1e10)

        if return_all_hs:
            return new_hidden_state, (transformer_output, action_logits)
        else:
            return new_hidden_state, action_logits
