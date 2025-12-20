import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method='dot'):
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        if method == 'dot':
            # For dot product, dimensions must match, or we need a projection
            # Use general dot product: decoder_state @ W @ encoder_states
            # But the requirement says "Dot-product", usually implies simple dot if dims match, 
            # or "General" (Luong) if with weight.
            # Let's assume standard dot product attention requires same size, or we project.
            if enc_hid_dim != dec_hid_dim:
                self.project = nn.Linear(enc_hid_dim, dec_hid_dim)
            else:
                self.project = None
                
        elif method == 'additive':
            # Bahdanau: v^T * tanh(W1 * s + W2 * h)
            self.W1 = nn.Linear(enc_hid_dim, dec_hid_dim)
            self.W2 = nn.Linear(dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
            
        elif method == 'multiplicative':
            # Luong General: s^T * W * h
            self.W = nn.Linear(enc_hid_dim, dec_hid_dim)
            
        else:
            raise ValueError(f"Unknown attention method: {method}")

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, dec_hid_dim] (last decoder hidden state)
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]
        # mask: [batch_size, src_len]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        if self.method == 'dot':
            # Score = hidden * encoder_outputs
            # hidden: [b, d], enc: [b, s, e]
            if self.project:
                encoder_outputs = self.project(encoder_outputs) # [b, s, d]
                
            # hidden: [b, 1, d]
            hidden = hidden.unsqueeze(1)
            # [b, 1, d] @ [b, d, s] -> [b, 1, s]
            energy = torch.bmm(hidden, encoder_outputs.permute(0, 2, 1))
            energy = energy.squeeze(1) # [b, s]
            
        elif self.method == 'additive':
            # hidden: [b, d] -> [b, s, d]
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            
            # energy: v(tanh(W1(enc) + W2(hidden)))
            # [b, s, d]
            w1_enc = self.W1(encoder_outputs)
            w2_hid = self.W2(hidden_expanded)
            
            combined = torch.tanh(w1_enc + w2_hid)
            energy = self.v(combined).squeeze(2) # [b, s]
            
        elif self.method == 'multiplicative':
            # score = hidden * W * enc
            # W * enc -> [b, s, d]
            w_enc = self.W(encoder_outputs)
            hidden = hidden.unsqueeze(1) # [b, 1, d]
            energy = torch.bmm(hidden, w_enc.permute(0, 2, 1)).squeeze(1) # [b, s]
            
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        return F.softmax(energy, dim=1)
