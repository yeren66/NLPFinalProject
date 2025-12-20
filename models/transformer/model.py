import torch
import torch.nn as nn
from .layers import MultiHeadAttention, FeedForward, RMSNorm
from .embeddings import PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, norm_type='layer', pos_enc_type='absolute'):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device, pos_enc_type)
        self.ffn = FeedForward(hid_dim, pf_dim, dropout)
        
        if norm_type == 'rms':
            self.attn_layer_norm = RMSNorm(hid_dim)
            self.ffn_layer_norm = RMSNorm(hid_dim)
        else:
            self.attn_layer_norm = nn.LayerNorm(hid_dim)
            self.ffn_layer_norm = nn.LayerNorm(hid_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src: [batch, len, hid_dim]
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.attn_layer_norm(src + self.dropout(_src))
        
        _src = self.ffn(src)
        src = self.ffn_layer_norm(src + self.dropout(_src))
        
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, norm_type='layer', pos_enc_type='absolute'):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device, pos_enc_type)
        self.enc_attn = MultiHeadAttention(hid_dim, n_heads, dropout, device, pos_enc_type='absolute') # Cross attn usually doesn't need relative logic on Query
        self.ffn = FeedForward(hid_dim, pf_dim, dropout)
        
        if norm_type == 'rms':
            self.self_attn_layer_norm = RMSNorm(hid_dim)
            self.enc_attn_layer_norm = RMSNorm(hid_dim)
            self.ffn_layer_norm = RMSNorm(hid_dim)
        else:
            self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
            self.ffn_layer_norm = nn.LayerNorm(hid_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))
        
        _tgt, attention = self.enc_attn(tgt, enc_src, enc_src, src_mask)
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))
        
        _tgt = self.ffn(tgt)
        tgt = self.ffn_layer_norm(tgt + self.dropout(_tgt))
        
        return tgt, attention

class Transformer(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, 
                 encoder_layers, decoder_layers, 
                 input_dim, output_dim, 
                 hid_dim, device, pos_enc_type='absolute'):
        super().__init__()
        
        self.device = device
        
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(encoder_layers)])
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(decoder_layers)])
        
        self.encoder_embedding = nn.Embedding(input_dim, hid_dim)
        self.decoder_embedding = nn.Embedding(output_dim, hid_dim)
        
        self.pos_enc = PositionalEncoding(hid_dim, type=pos_enc_type)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def make_src_mask(self, src):
        # src: [batch, len]
        # mask: [batch, 1, 1, len]
        return (src != 0).unsqueeze(1).unsqueeze(2)
        
    def make_tgt_mask(self, tgt):
        # tgt: [batch, len]
        # mask: [batch, 1, len, len]
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        
        return tgt_pad_mask & trg_sub_mask
        
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        src = self.dropout((self.encoder_embedding(src) * self.scale))
        src = self.pos_enc(src)
        
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
            
        # Decoder
        tgt = self.dropout((self.decoder_embedding(tgt) * self.scale))
        tgt = self.pos_enc(tgt)
        
        for layer in self.decoder_layers:
            tgt, attention = layer(tgt, src, tgt_mask, src_mask)
            
        output = self.fc_out(tgt)
        return output
