import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5, src_len=None):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        # src_len: [batch] - actual source lengths (optional, for pack_padded_sequence)

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Encoder (LSTM always returns hidden and cell)
        # Pass src_len for pack_padded_sequence optimization
        encoder_outputs, hidden, cell = self.encoder(src, src_len=src_len)

        # Prepare decoder input (first token is <SOS>)
        input = tgt[:, 0]

        # Prepare hidden state
        # Encoder returns [b, dec_hid], Decoder LSTM needs [n_layers, b, dec_hid]
        # We expand it
        hidden = hidden.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)

        # Create mask for attention (pad tokens)
        mask = (src != 0) # Assumes 0 is pad

        for t in range(1, tgt_len):
            # LSTM decoder always returns hidden and cell
            output, hidden, cell = self.decoder(input, hidden, encoder_outputs, mask, cell)

            outputs[:, t] = output

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = tgt[:, t] if teacher_force else top1

        return outputs
