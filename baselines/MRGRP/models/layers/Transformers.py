import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import mlp
from .utility_funcs import normalize_adjacency_matrix
from .MultiHeadAttn import MultiHeadAttention, FeedForward
from .MultiGCN import MultiGCNLayer


    
class TransformerEncoder(nn.Module):
    def __init__(self, hp):
        super(TransformerEncoder, self).__init__()
        self.hp = hp
        hp_dropout_rate = 0.1
        self.dropout = nn.Dropout(hp_dropout_rate)

        # Encoder layers
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(hp) for _ in range(hp.num_blocks)])
        # Positional Encoding is usually implemented as a separate function or layer
        self.edge_layer = nn.Sequential(nn.Linear(8, hp.d_model), nn.ReLU())

    def positional_encoding(self, inputs, maxlen, masking=torch.tensor([1])):
        maxlen = maxlen.item()
        E = inputs.size(-1)
        N, T = inputs.size(0), inputs.size(1)

        # position indices
        position_ind = torch.arange(T).unsqueeze(0).repeat(N, 1)

        E_t = int(E) 
        maxlen_t = int(maxlen) 
        C = 10000  
        position_enc_values = [
            [pos / (C**((i - i % 2) / E_t)) for i in range(E_t)]
            for pos in range(maxlen_t)
        ]
        position_enc = torch.tensor(position_enc_values).to(inputs.device)
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        position_enc = position_enc.float()

        # lookup
        outputs = position_enc[position_ind]

        # masks
        if masking:
            outputs = torch.where(inputs == 0, inputs, outputs)

        return outputs.float()

    def forward(self, embeddings, src_masks, edge_fea, causality = torch.tensor([0])):
        """
        embeddings: Float tensor of shape (N, T, d_model)
        src_masks: Byte tensor of shape (N, T), where padding positions are marked with True
        """
        # Embedding scaling and positional encoding
        enc = embeddings * (self.hp.d_model ** 0.5)

        enc = enc + self.positional_encoding(enc, torch.tensor([self.hp.maxlen1]))  # Assuming positional_encoding is defined elsewhere
        enc = self.dropout(enc)

        semantic_sim = (normalize_adjacency_matrix(torch.matmul(enc, enc.transpose(1, 2)) / (self.hp.d_model ** 0.5))*(~src_masks.unsqueeze(1).repeat(1, enc.size(1), 1))).unsqueeze(-1)

        edge_fea = torch.cat([edge_fea, semantic_sim], dim=-1)

        edge_fea = self.edge_layer(edge_fea)
        # Pass through each encoder layer
        count = 0
        for layer in self.enc_layers:
            enc, edge_fea = layer(enc, src_masks, edge_fea, causality, count) # Residual connection
            count += 1

        return enc

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hp):
        super(TransformerEncoderLayer, self).__init__()
        hp_dropout_rate = torch.tensor([0.1])
        self.self_attn = MultiHeadAttention(hp.d_model, hp.num_heads, hp_dropout_rate)  # Assuming MultiHeadAttention is defined elsewhere
        self.feed_forward = FeedForward(hp.d_model, hp.d_ff, hp_dropout_rate)  # Assuming FeedForward is defined elsewhere
        self.gcn = MultiGCNLayer(hp.d_model)
    def forward(self, x, src_mask, edge_fea, causality, count):
        adj = ~src_mask.unsqueeze(1).repeat(1, x.size(1), 1)
        adj = ~src_mask.unsqueeze(-1)*adj
        adj = normalize_adjacency_matrix(adj.float())

        if count >0:
            x, edge_fea = self.gcn(x, edge_fea, adj)
        x = self.self_attn(q=x, k=x, v=x, mask=src_mask, causality= causality, edge_fea = edge_fea) + x

        x = self.feed_forward(x) 
        return x, edge_fea