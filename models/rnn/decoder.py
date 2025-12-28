import torch
import torch.nn as nn
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, n_layers=2):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # LSTM input: embedding + weighted_ctx (concatenated)
        # Standard attention implementation: Input + Context -> LSTM
        # Context comes from previous hidden state attention.

        # Use LSTM as the RNN architecture
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None, cell=None):
        # input: [batch_size] (one step)
        # hidden: [n_layers, batch_size, dec_hid_dim] (for LSTM input)
        # cell: [n_layers, batch_size, dec_hid_dim] (for LSTM input)
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        # mask: [batch_size, src_len]

        input = input.unsqueeze(1) # [b, 1]
        embedded = self.dropout(self.embedding(input)) # [b, 1, emb_dim]

        # Calculate attention using the hidden state from the top layer
        # hidden: [num_layers, b, d], take [-1] for top layer

        if hidden.dim() == 3:
            query = hidden[-1]
        else:
            query = hidden

        a = self.attention(query, encoder_outputs, mask) # [b, src_len]

        # Weighted sum of encoder outputs
        a = a.unsqueeze(1) # [b, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs) # [b, 1, enc_hid_dim * 2]

        # LSTM input: [embedded; weighted]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # LSTM forward pass
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Output prediction
        # Concatenate embedded, weighted, and LSTM output (Luong style)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(1))

        return prediction, hidden, cell
