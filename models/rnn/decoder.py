import torch
import torch.nn as nn
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, rnn_type='gru', n_layers=2):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # GRU input: embedding + weighted_ctx (concatenated)
        # However, standard attention usually concatenates after RNN or before.
        # Luong: input->rnn->output + context -> concat -> linear -> pred
        # Bahdanau: input + context -> rnn -> output -> pred
        
        # We'll use a standard implementation: Input + Context -> RNN
        # Context comes from previous hidden state attention.
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, 
                          num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
                          
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, 
                          num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None, cell=None):
        # input: [batch_size] (one step)
        # hidden: [n_layers, batch_size, dec_hid_dim] (for RNN input)
        # Actually, in our Encoder we returned [b, dec_hid_dim] as "hidden_last".
        # But proper RNN expects [n_layers, b, d]. 
        # For simplicity in this project, we'll assume 1-layer decoder or handle multi-layer init.
        # Let's assume n_layers match.
        
        input = input.unsqueeze(1) # [b, 1]
        embedded = self.dropout(self.embedding(input)) # [b, 1, emb_dim]
        
        # Calculate attention using the hidden state from the *last* time step
        # Ideally we use the top layer hidden state for attention query.
        # hidden depends on shape. If it's [num_layers, b, d], take [-1]
        
        if hidden.dim() == 3:
            query = hidden[-1]
        else:
            query = hidden
            # If input hidden is just [b, d], we need to unsqueeze for RNN if it expects 3D
            # But here let's standardize: hidden passed in is [num_layers, b, d]
            
        a = self.attention(query, encoder_outputs, mask) # [b, src_len]
        
        # Weighted sum of encoder outputs
        a = a.unsqueeze(1) # [b, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs) # [b, 1, enc_hid_dim * 2]
        
        # RNN input: [embedded; weighted]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        if self.rnn_type == 'gru':
            output, hidden = self.rnn(rnn_input, hidden)
        else:
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Output prediction
        # Concatenate embedded, weighted, and RNN output (Luong style often does this)
        # Or just RNN output.
        # Let's do: output = fc(embedded + output + weighted) or [out; weighted; embed]
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(1))
        
        if self.rnn_type == 'lstm':
            return prediction, hidden, cell
        return prediction, hidden
