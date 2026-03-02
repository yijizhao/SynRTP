# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphormerMultiheadAttention(nn.Module):
    """Multi-head attention adapted for Graphormer.

    Notes:
    - Input tensors use a nodes-first layout: (N_nodes, B_flat, hidden_dim).
    - Attention computations are performed as (B_flat, num_heads, N_nodes, N_nodes).
    - External `attention_bias` (e.g., edge/structure encodings) is added to scores.
    """

    def __init__(self, hidden_dim, num_heads, dropout_rate=0.0):
        super(GraphormerMultiheadAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # scaling factor for stable softmax in dot-product attention
        self.scaling = self.head_dim ** -0.5

        # linear projections for q, k, v and final output projection
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_bias, key_padding_mask=None, need_weights=False):

        N_nodes, B_flat, H_dim = x.shape 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (B_flat, num_heads, N_nodes, head_dim) for attention matmul
        q = q.view(N_nodes, B_flat, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(N_nodes, B_flat, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(N_nodes, B_flat, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # compute scaled dot-product attention scores
        # result shape: [B_flat, num_heads, N_nodes, N_nodes]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # add external biases (spatial/edge encodings). Must be broadcastable.
        attn_scores = attn_scores + attention_bias

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand_as(attn_scores)
            attn_scores.masked_fill_(key_padding_mask, float('-inf'))

        # normalize to probabilities and apply dropout to attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # weighted sum over values -> [B_flat, num_heads, N_nodes, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(N_nodes, B_flat, H_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weights
        return attn_output


class GraphormerEncoderLayer(nn.Module):
    """Single Graphormer encoder layer with pre-layernorm, attention and FFN."""

    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super(GraphormerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = GraphormerMultiheadAttention(hidden_dim, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_bias, key_padding_mask=None):
        """
        x: Tensor[N_nodes, B_flat, hidden_dim]
        attention_bias: broadcastable to attention score shape
        key_padding_mask: optional [B_flat, N_nodes]
        returns: Tensor[N_nodes, B_flat, hidden_dim]
        """
        # attention block with residual connection
        residual = x
        x_norm = self.norm1(x)
        # note: attn returns either Tensor or (Tensor, weights) depending on need_weights
        attn_output = self.attn(x_norm, attention_bias=attention_bias, key_padding_mask=key_padding_mask)[0]
        x = residual + self.dropout1(attn_output)

        # feed-forward block with residual connection
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_output)

        return x