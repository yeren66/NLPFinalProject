import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.n_layers = n_layers
        self.enc_hid_dim = enc_hid_dim

        # Use LSTM as the RNN architecture
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len=None):
        # src: [batch_size, src_len]
        # src_len: [batch_size] - actual lengths (optional, for pack_padded_sequence)

        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]

        # Pack padded sequence if lengths provided (important optimization for RNN)
        if src_len is not None:
            # Sort by length (required by pack_padded_sequence)
            src_len_sorted, sort_idx = src_len.sort(descending=True)
            embedded_sorted = embedded[sort_idx]

            # Pack the sequences
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                src_len_sorted.cpu(),  # pack_padded_sequence requires CPU tensor for lengths
                batch_first=True,
                enforce_sorted=True
            )

            # LSTM forward pass on packed sequence
            packed_outputs, (hidden, cell) = self.rnn(packed_embedded)

            # Unpack the sequences
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                padding_value=0
            )

            # Unsort to restore original order
            _, unsort_idx = sort_idx.sort()
            outputs = outputs[unsort_idx]
            hidden = hidden[:, unsort_idx, :]
            cell = cell[:, unsort_idx, :]
        else:
            # LSTM forward pass without packing (fallback)
            outputs, (hidden, cell) = self.rnn(embedded)

        # outputs: [b, s, hid_dim * 2]
        # hidden: [n_layers * 2, b, hid_dim]
        # cell: [n_layers * 2, b, hid_dim]

        # Transform final hidden state to decoder magnitude
        # We take the last layer's hidden state. Since it's bidirectional, we concat (fwd, bwd)
        # hidden[-2, :, :] is forward last layer
        # hidden[-1, :, :] is backward last layer

        hidden_last = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        # Project cell state as well for LSTM decoder
        cell_last = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))

        return outputs, hidden_last, cell_last
