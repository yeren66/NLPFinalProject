import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, rnn_type='gru', n_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers
        self.enc_hid_dim = enc_hid_dim
        
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        else:
            raise ValueError("Invalid RNN type")
            
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len=None):
        # src: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence if lengths provided (optional optimization)
        if src_len is not None:
            # Note: requires sorting by length if not enforced, omitting for simplicity/robustness unless needed
            pass
            
        if self.rnn_type == 'gru':
            outputs, hidden = self.rnn(embedded)
            # outputs: [b, s, hid_dim * 2]
            # hidden: [n_layers * 2, b, hid_dim]
        else:
            outputs, (hidden, cell) = self.rnn(embedded)
            
        # Transform final hidden state to decoder magnitude
        # We take the last layer's hidden state. Since it's bidirectional, we concat (fwd, bwd)
        # hidden[-2, :, :] is forward last layer
        # hidden[-1, :, :] is backward last layer
        
        hidden_last = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        if self.rnn_type == 'lstm':
             # For LSTM, we usually initialize cell state too, or just zero it. 
             # Simpler strategy: just project hidden and zero cell, or project both.
             # Here we return hidden only for init, or we return both if decoder is LSTM.
             # Simplification: Assume GRU for main experiments or handle LSTM similarly.
             # If Decoder is LSTM, it needs cell. Let's project cell too.
             cell_last = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
             return outputs, hidden_last, cell_last
             
        return outputs, hidden_last
