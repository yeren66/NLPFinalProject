import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000, type='absolute'):
        super().__init__()
        self.type = type
        self.emb_dim = emb_dim
        
        if type == 'absolute':
            pe = torch.zeros(max_len, emb_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0) # [1, max_len, d]
            self.register_buffer('pe', pe)
            
        elif type == 'relative':
            # Simplified Relative Positional Encoding (Shaw et al. or T5 style)
            # For this project, a learnable bias based on relative distance is a common approach.
            # Or just sinusoid based on difference. 
            # We'll stick to a simpler learnable embedding bucket approach or similar.
            # But standard Shaw et al is complex to implement from scratch in limited space.
            # Let's use a simpler variant: T5 uses relative position bias.
            pass
            
    def forward(self, x):
        # x: [batch, seq_len, dim]
        if self.type == 'absolute':
            return x + self.pe[:, :x.size(1)]
        else:
            # For relative, usually handled inside attention.
            # If specified here, we might just pass x through and let attention handle it 
            # OR logic is inside this method if it constructs bias.
            # Given the constraints, I will implement the logic inside Attention layer if 'relative' is chosen,
            # so this module might just return x.
            return x
