import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=torch.tensor([0.1])):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.dropout = nn.Dropout(dropout_rate.item())

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V, key_masks, edge_fea, causality=torch.tensor([0]), dropout_rate=torch.tensor([0])):

        d_k = Q.size(-1)

        # dot product
        outputs = torch.matmul(Q, K.transpose(-2, -1))  # (N, T_q, T_k)

        outputs /= d_k ** 0.5

        # key masking
        outputs = self.mask(outputs, key_masks=key_masks, type=torch.tensor([1]))

        # causality or future blinding masking
        if causality:
            outputs = self.mask(outputs, key_masks=torch.tensor([0]), type=torch.tensor([0]))

        # softmax
        outputs = F.softmax(outputs, dim=-1)
        if self.training:
            outputs = F.dropout(outputs, p=dropout_rate.item())

        # weighted sum (context vectors)
        outputs = torch.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, key_masks, type=torch.tensor([1])):
        padding_num = -2 ** 32 + 1
        if type:
            # Key Masking
            key_masks = key_masks.to(torch.float32)
            repeat_times = inputs.size(0) // key_masks.size(0)

            key_masks = key_masks.repeat_interleave(repeat_times, dim=0)  

            key_masks = key_masks.unsqueeze(1) 

            outputs = inputs + key_masks * padding_num
            
        else:
            diag_vals = torch.ones_like(inputs[0, :, :]).to(inputs.device)  # (T_q, T_k)

            tril = torch.tril(diag_vals)  # (T_q, T_k)

            future_masks = tril.unsqueeze(0).repeat(inputs.size(0), 1, 1)  # (N, T_q, T_k)

            paddings = torch.ones_like(future_masks).to(inputs.device) * padding_num

            outputs = torch.where(future_masks == 0, paddings, inputs)
        return outputs

    def forward(self, q, k, v, mask, causality, edge_fea):

        # Linear layers
        q_ = self.wq(q)  # (batch_size, seq_len, d_model)
        k_ = self.wk(k)  # (batch_size, seq_len, d_model)
        v_ = self.wv(v)  # (batch_size, seq_len, d_model)

        # Split heads
        q_ = torch.cat(torch.chunk(q_, self.num_heads, dim=2), dim=0)  # (batch_size, num_heads, seq_len_q, depth)
        k_ = torch.cat(torch.chunk(k_, self.num_heads, dim=2), dim=0)   # (batch_size, num_heads, seq_len_k, depth)
        v_ = torch.cat(torch.chunk(v_, self.num_heads, dim=2), dim=0)   # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(q_, k_, v_, mask, edge_fea, causality, self.dropout_rate)

        # Concat attention heads
        outputs = torch.cat(torch.chunk(scaled_attention, self.num_heads, dim=0), dim=2)  # (batch_size, seq_len_v, num_heads, depth)

        # Reshape to (batch_size, seq_len_v, d_model)
        # concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        

        # Final linear layer
        # outputs = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)
        outputs = self.dropout(outputs)
        # Residual connection and normalization
        outputs += q
        outputs = self.layer_norm(outputs)  # (batch_size, seq_len_v, d_model)

        return outputs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=torch.tensor([0.1])):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate.item())
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        outputs = self.w_1(inputs)
        outputs = F.relu(outputs)
        outputs = self.w_2(outputs)
        outputs = self.dropout(outputs)
        outputs += inputs
        outputs = self.layer_norm(outputs)
        return outputs