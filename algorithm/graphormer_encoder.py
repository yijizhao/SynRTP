# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from algorithm.graphormer_modules import GraphormerEncoderLayer

class GraphormerEncoder(nn.Module):
    """Graphormer encoder stacking multiple encoder layers.

    Notes:
    - Input `x` uses nodes-first layout: (N_nodes, B_flat, hidden_dim).
    - This module adds degree embeddings and computes a combined
      attention bias (spatial + edge) per head: [B_flat, num_heads, N, N].
    """

    def __init__(self,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 dropout_rate,
                 max_shortest_path_len,
                 max_degree,
                 edge_path_feature_dim,
                 device):
        super(GraphormerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.in_degree_embedding = nn.Embedding(max_degree + 1, hidden_dim)
        self.out_degree_embedding = nn.Embedding(max_degree + 1, hidden_dim)

        self.spatial_bias_embedding = nn.Embedding(max_shortest_path_len + 2, num_heads)

        self.edge_bias_linear = nn.Linear(edge_path_feature_dim, num_heads)

        # encoder stack
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self,
                x,
                node_in_degree,
                node_out_degree,
                shortest_path_distances,
                shortest_path_edge_features_aggregated,
                key_padding_mask=None
               ):
        N_nodes, B_flat, hidden_dim = x.shape
        # lookup degree embeddings;
        in_degree_emb = self.in_degree_embedding(node_in_degree.permute(1, 0).contiguous()) # (B_flat, N_nodes, hidden_dim)
        out_degree_emb = self.out_degree_embedding(node_out_degree.permute(1, 0).contiguous()) # (B_flat, N_nodes, hidden_dim)

        in_degree_emb = in_degree_emb.permute(1, 0, 2).contiguous() # (N_nodes, B_flat, hidden_dim)
        out_degree_emb = out_degree_emb.permute(1, 0, 2).contiguous() # (N_nodes, B_flat, hidden_dim)
        x = x + in_degree_emb + out_degree_emb
        #  spatial_bias from shortest path distances
        spatial_bias = self.spatial_bias_embedding(shortest_path_distances) # (N,N,B_flat,num_heads)
        spatial_bias = spatial_bias.permute(2, 3, 0, 1).contiguous() # (B_flat, num_heads, N, N)

        # edge bias projected to num_heads and permuted to (B_flat, num_heads, N, N)
        edge_bias = self.edge_bias_linear(shortest_path_edge_features_aggregated)  # (N,N,B_flat,num_heads)
        edge_bias = edge_bias.permute(2, 3, 0, 1).contiguous()  # -> (B_flat, num_heads, N, N)

        combined_attention_bias = spatial_bias + edge_bias

        encoded_nodes = x
        for layer in self.layers:
            encoded_nodes = layer(encoded_nodes,
                                  attention_bias=combined_attention_bias,
                                  key_padding_mask=key_padding_mask)

        return encoded_nodes
