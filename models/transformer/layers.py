import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, pos_enc_type='absolute'):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.pos_enc_type = pos_enc_type
        if pos_enc_type == 'relative':
            # Learnable relative positional bias
            # This is a simplified version (e.g. T5 style buckets or simple lookup)
            # Max relative distance
            self.max_relative_position = 128
            self.relative_attention_bias = nn.Embedding(self.max_relative_position * 2 + 1, n_heads)
            
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Q, K, V: [batch, len, hid_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # Split heads: [batch, n_heads, len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Energy: [batch, n_heads, q_len, k_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        if self.pos_enc_type == 'relative':
            q_len = query.shape[1]
            k_len = key.shape[1]
            # Create relative position matrix
            # Simple version: range matrix
            q_idx = torch.arange(q_len)[:, None]
            k_idx = torch.arange(k_len)[None, :]
            relative_pos = k_idx - q_idx # [q_len, k_len]
            
            # Clip and shift
            relative_pos = torch.clamp(relative_pos, -self.max_relative_position, self.max_relative_position)
            relative_pos += self.max_relative_position
            
            # Lookup bias: [q_len, k_len, n_heads]
            rel_embeddings = self.relative_attention_bias(relative_pos.to(query.device)) 
            # Permute to [1, n_heads, q_len, k_len]
            rel_embeddings = rel_embeddings.permute(2, 0, 1).unsqueeze(0)
            
            energy += rel_embeddings
            
        if mask is not None:
             # Mask is [batch, 1, 1, k_len] or similar
             energy = energy.masked_fill(mask == 0, -1e10)
             
        attention = torch.softmax(energy, dim=-1)
        
        # [batch, n_heads, q_len, head_dim]
        x = torch.matmul(self.dropout(attention), V)
        
        # Concat
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        
        x = self.fc_o(x)
        
        return x, attention

class FeedForward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))
