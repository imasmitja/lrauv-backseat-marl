import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Tuple, Optional
import math

# RNN Modules
class GRUCellWithReset(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
    
    def forward(self, rnn_state, ins, resets):
        # Handle reset conditions
        batch_size = resets.shape[0]
        rnn_state = torch.where(
            resets.unsqueeze(1),
            self.initialize_carry(batch_size),
            rnn_state
        )
        new_rnn_state = self.gru_cell(ins, rnn_state)
        return new_rnn_state, new_rnn_state
    
    def initialize_carry(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class ScannedRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCellWithReset(hidden_size)
    
    def forward(self, hidden, x):
        """Process sequence - manual scan implementation"""
        ins, resets = x
        batch_size, seq_len, feature_dim = ins.shape
        
        # Initialize output tensor
        outputs = []
        current_hidden = hidden
        
        # Process each timestep
        for t in range(seq_len):
            current_hidden, output = self.gru_cell(
                current_hidden, 
                ins[:, t], 
                resets[:, t]
            )
            outputs.append(output.unsqueeze(1))
        
        # Stack outputs along sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return current_hidden, outputs
    
    def initialize_carry(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class PPOActorRNN(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int = 512, num_layers: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layers
        self.input_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for l in range(num_layers):
            self.input_layers.append(nn.Linear(hidden_size, hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
        
        # RNN
        self.rnn = ScannedRNN(hidden_size)
        
        # Output layers
        self.actor_mean1 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean_ln = nn.LayerNorm(hidden_size)
        self.actor_mean2 = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.action_dim:
                # Orthogonal initialization with scale 0.01 for output layer
                nn.init.orthogonal_(module.weight, gain=0.01)
            elif module.out_features == self.hidden_size and module.in_features == self.hidden_size:
                # Orthogonal initialization with scale sqrt(2) for hidden layers
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            else:
                # Default orthogonal initialization
                nn.init.orthogonal_(module.weight, gain=1.0)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, hidden, x):
        obs, dones, avail_actions = x
        
        embedding = obs
        for l in range(self.num_layers):
            embedding = self.input_layers[l](embedding)
            embedding = self.layer_norms[l](embedding)
            embedding = F.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn(hidden, rnn_in)
        
        actor_mean = self.actor_mean1(embedding)
        actor_mean = self.actor_mean_ln(actor_mean)
        actor_mean = F.relu(actor_mean)
        
        actor_mean = self.actor_mean2(actor_mean)
        
        # Mask unavailable actions - convert boolean to float first
        unavail_actions = 1 - avail_actions.float()  # Convert to float before subtraction
        action_logits = actor_mean - (unavail_actions * 1e10)
        
        return hidden, action_logits
    
    def initialize_hidden(self, batch_size):
        return self.rnn.initialize_carry(batch_size)

class PQNRnn(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int = 512, num_layers: int = 4, 
                 norm_input: bool = False, norm_type: str = "layer_norm", dueling: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.norm_input = norm_input
        self.norm_type = norm_type
        self.dueling = dueling
        
        # Input normalization
        if norm_input:
            self.input_norm = nn.BatchNorm1d(hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for l in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            if norm_type == "layer_norm":
                self.norms.append(nn.LayerNorm(hidden_size))
            elif norm_type == "batch_norm":
                self.norms.append(nn.BatchNorm1d(hidden_size))
        
        # RNN
        self.rnn = ScannedRNN(hidden_size)
        
        # Output layers
        if dueling:
            self.advantage = nn.Linear(hidden_size, action_dim)
            self.value = nn.Linear(hidden_size, 1)
        else:
            self.q_vals = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, hidden, x, train: bool = False):
        x, dones, avail_actions = x
        
        if self.norm_input:
            x = self.input_norm(x)
        
        for l in range(self.num_layers):
            x = self.hidden_layers[l](x)
            if self.norm_type != "none":
                if self.norm_type == "batch_norm":
                    x = self.norms[l](x)
                else:
                    x = self.norms[l](x)
            x = F.relu(x)
        
        rnn_in = (x, dones)
        hidden, x = self.rnn(hidden, rnn_in)
        
        if self.dueling:
            adv = self.advantage(x)
            val = self.value(x)
            q_vals = val + adv - adv.mean(dim=-1, keepdim=True)
        else:
            q_vals = self.q_vals(x)
        
        # Mask unavailable actions - convert boolean to float first
        unavail_actions = 1 - avail_actions.float()  # Convert to float before subtraction
        q_vals = q_vals - (unavail_actions * 1e10)
        
        return hidden, q_vals
        
    def initialize_hidden(self, batch_size):
        return self.rnn.initialize_carry(batch_size)

# TRANSFORMER Modules
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x, mask=None):
        # Handle both 3D and 4D inputs
        if x.dim() == 4:
            # [batch_size, seq_len, num_entities, hidden_dim] -> flatten entities into sequence
            batch_size, seq_len, num_entities, hidden_dim = x.shape
            x = x.reshape(batch_size, seq_len * num_entities, hidden_dim)
            reshape_back = True
        else:
            reshape_back = False
            batch_size, seq_len, hidden_dim = x.shape
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        out = self.out_linear(out)
        
        if reshape_back:
            # Reshape back to [batch_size, seq_len, num_entities, hidden_dim]
            out = out.reshape(batch_size, seq_len, num_entities, hidden_dim)
        
        return out

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int, dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_prob)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        
    def forward(self, x, mask=None, deterministic=True):
        # Attention part
        attended = self.self_attn(x, mask)
        if not deterministic:
            attended = self.dropout1(attended)
        x = self.norm1(attended + x)
        
        # MLP part
        feedforward = self.linear1(x)
        feedforward = F.relu(feedforward)
        feedforward = self.linear2(feedforward)
        if not deterministic:
            feedforward = self.dropout2(feedforward)
        x = self.norm2(feedforward + x)
        
        return x

class Embedder(nn.Module):
    def __init__(self, hidden_dim: int, activation: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        if activation:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.linear.weight, gain=math.sqrt(2))
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.layer_norm(x)
            x = F.relu(x)
        return x

class ScannedTransformer(nn.Module):
    def __init__(self, hidden_dim: int, transf_num_layers: int, transf_num_heads: int, 
                 transf_dim_feedforward: int, transf_dropout_prob: float = 0, 
                 deterministic: bool = True, return_embeddings: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transf_num_layers = transf_num_layers
        self.return_embeddings = return_embeddings
        self.deterministic = deterministic
        
        self.encoders = nn.ModuleList([
            EncoderBlock(hidden_dim, transf_num_heads, transf_dim_feedforward, transf_dropout_prob)
            for _ in range(transf_num_layers)
        ])
    
    def forward(self, hs, x):
        embeddings, mask, done = x
        # embeddings shape: [batch_size, seq_len, num_entities, hidden_dim]
        # hs shape: [batch_size, seq_len, hidden_dim]
        
        batch_size, seq_len, num_entities, hidden_dim = embeddings.shape
        
        # Reset hidden state where done is True
        hs = torch.where(
            done.unsqueeze(-1),  # [batch_size, seq_len, 1]
            self.initialize_carry(batch_size, seq_len, hidden_dim),
            hs
        )
        
        # Process each sequence element independently
        all_outputs = []
        for t in range(seq_len):
            # Get current timestep embeddings and hidden state
            emb_t = embeddings[:, t]  # [batch_size, num_entities, hidden_dim]
            hs_t = hs[:, t].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Combine hidden state with embeddings
            combined = torch.cat([hs_t, emb_t], dim=1)  # [batch_size, num_entities+1, hidden_dim]
            
            # Process through transformer layers
            for layer in self.encoders:
                combined = layer(combined, mask, deterministic=self.deterministic)
            
            # Extract new hidden state (first position)
            new_hs_t = combined[:, 0]  # [batch_size, hidden_dim]
            
            if self.return_embeddings:
                all_outputs.append(combined.unsqueeze(1))
            else:
                all_outputs.append(new_hs_t.unsqueeze(1))
        
        # Stack outputs along sequence dimension
        if self.return_embeddings:
            # [batch_size, seq_len, num_entities+1, hidden_dim]
            outputs = torch.cat(all_outputs, dim=1)
            new_hs = outputs[:, :, 0]  # [batch_size, seq_len, hidden_dim]
            return new_hs, outputs
        else:
            # [batch_size, seq_len, hidden_dim]
            new_hs = torch.cat(all_outputs, dim=1)
            return new_hs, new_hs
    
    def initialize_carry(self, batch_size, seq_len, hidden_dim):
        return torch.zeros(batch_size, seq_len, hidden_dim)
    
class JAXCompatibleTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.encoders = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_entities, hidden_dim]
        # Process through each encoder layer
        for encoder in self.encoders:
            x = encoder(x, mask=mask, deterministic=True)
        return x

class PPOActorTransformer(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int = 512, num_layers: int = 2, 
                 num_heads: int = 8, ff_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Embedder
        self.embedder = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Use JAX-compatible transformer
        self.transformer = JAXCompatibleTransformer(
            hidden_dim=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.output_layer:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, hidden, x):
        obs, dones, avail_actions = x
        
        print(f"Transformer forward - hidden: {hidden.shape}, obs: {obs.shape}")
        
        # obs shape: [batch_size, seq_len, num_entities, obs_dim]
        batch_size, seq_len, num_entities, obs_dim = obs.shape
        
        # Flatten batch and sequence dimensions for processing
        obs_flat = obs.reshape(batch_size * seq_len, num_entities, obs_dim)
        
        # Apply embedder to all entities
        embeddings = self.embedder(obs_flat)  # [batch*seq, num_entities, hidden]
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Process through transformer
        transformer_out = self.transformer(embeddings)  # [batch*seq, num_entities, hidden]
        print(f"Transformer output shape: {transformer_out.shape}")
        
        # Use only the agent's embedding (entity 0) for action prediction
        agent_embeddings = transformer_out[:, 0, :]  # [batch*seq, hidden]
        
        # Reshape back to [batch_size, seq_len, hidden]
        agent_embeddings = agent_embeddings.reshape(batch_size, seq_len, self.hidden_size)
        print(f"Agent embeddings shape: {agent_embeddings.shape}")
        
        # Get logits
        logits = self.output_layer(agent_embeddings)  # [batch_size, seq_len, action_dim]
        print(f"Logits shape: {logits.shape}")
        
        # Action masking
        unavail_actions = 1 - avail_actions.float()
        action_logits = logits - (unavail_actions * 1e10)
        
        # Update hidden state (use the agent embedding as new hidden state)
        new_hidden = agent_embeddings  # [batch_size, seq_len, hidden]
        
        return new_hidden, action_logits
    
    def initialize_hidden(self, batch_size, seq_len=1):
        return torch.zeros(batch_size, seq_len, self.hidden_size)