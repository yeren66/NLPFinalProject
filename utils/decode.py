import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


def greedy_decode(model, src, tgt_vocab, max_len=100):
    """
    Simple greedy decoding function for BLEU computation.
    Supports both RNN (Seq2Seq) and Transformer models.

    Args:
        model: Seq2Seq or Transformer model
        src: [1, src_len] tensor
        tgt_vocab: target vocabulary
        max_len: maximum decoding length

    Returns:
        List of token indices
    """
    model.eval()
    device = src.device

    with torch.no_grad():
        # Prepare decoder input
        sos_idx = tgt_vocab.stoi.get("<sos>", tgt_vocab.stoi.get("<SOS>", 1))
        eos_idx = tgt_vocab.stoi.get("<eos>", tgt_vocab.stoi.get("<EOS>", 2))

        # Check if model is Transformer or RNN
        from models.transformer.model import Transformer
        from models.rnn.seq2seq import Seq2Seq

        if isinstance(model, Transformer):
            # Transformer decoding
            tgt_indices = [sos_idx]

            for _ in range(max_len):
                tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)

                # Forward pass
                output = model(src, tgt_tensor)

                # Get prediction for the last token
                pred_token = output[0, -1].argmax().item()
                tgt_indices.append(pred_token)

                # Stop if EOS token is generated
                if pred_token == eos_idx:
                    break

            # Remove SOS token from output
            return tgt_indices[1:]

        elif isinstance(model, Seq2Seq):
            # RNN decoding
            # Encoder returns: outputs, hidden, cell (for LSTM)
            encoder_outputs, hidden, cell = model.encoder(src)

            # Prepare hidden/cell for decoder (expand to match decoder layers)
            # Encoder returns [b, hid], Decoder needs [n_layers, b, hid]
            hidden = hidden.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(model.decoder.rnn.num_layers, 1, 1)

            input_token = torch.tensor([sos_idx]).to(device)

            # Create mask
            pad_idx = tgt_vocab.stoi.get('<PAD>', tgt_vocab.stoi.get('<pad>', 0))
            mask = (src != pad_idx)

            outputs = []

            for _ in range(max_len):
                output, hidden, cell = model.decoder(input_token, hidden, encoder_outputs, mask, cell)

                # Get the token with highest probability
                pred_token = output.argmax(1).item() # output is [b, vocab]
                outputs.append(pred_token)

                # Stop if EOS token is generated
                if pred_token == eos_idx:
                    break

                # Use predicted token as next input
                input_token = torch.tensor([pred_token]).to(device) # [b]

            return outputs

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")


class GreedyDecoder:
    """Greedy decoding for sequence generation."""
    
    def __init__(self, model, device, max_len=100):
        self.model = model
        self.device = device
        self.max_len = max_len
        
    def decode(self, src, src_vocab, tgt_vocab):
        """
        Greedy decoding for a single source sentence.
        
        Args:
            src: [1, src_len] tensor
            src_vocab: source vocabulary
            tgt_vocab: target vocabulary
            
        Returns:
            List of token indices
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode
            if hasattr(self.model.encoder, 'rnn_type') and self.model.encoder.rnn_type == 'lstm':
                encoder_outputs, hidden, cell = self.model.encoder(src)
            else:
                encoder_outputs, hidden = self.model.encoder(src)
                cell = None
            
            # Prepare decoder input
            # Start with <SOS> token
            sos_idx = tgt_vocab.stoi.get("<SOS>", 1)
            eos_idx = tgt_vocab.stoi.get("<EOS>", 2)
            
            input_token = torch.tensor([sos_idx]).to(self.device)
            
            # Prepare hidden state for decoder
            hidden = hidden.unsqueeze(0).repeat(self.model.decoder.rnn.num_layers, 1, 1)
            if cell is not None:
                cell = cell.unsqueeze(0).repeat(self.model.decoder.rnn.num_layers, 1, 1)
            
            # Create mask
            mask = (src != 0)
            
            outputs = []
            
            for _ in range(self.max_len):
                if hasattr(self.model.encoder, 'rnn_type') and self.model.encoder.rnn_type == 'lstm':
                    output, hidden, cell = self.model.decoder(input_token, hidden, encoder_outputs, mask, cell)
                else:
                    output, hidden = self.model.decoder(input_token, hidden, encoder_outputs, mask)
                
                # Get the token with highest probability
                pred_token = output.argmax(1).item()
                outputs.append(pred_token)
                
                # Stop if EOS token is generated
                if pred_token == eos_idx:
                    break
                
                # Use predicted token as next input
                input_token = torch.tensor([pred_token]).to(self.device)
            
            return outputs


class BeamSearchDecoder:
    """Beam search decoding for sequence generation."""
    
    def __init__(self, model, device, beam_width=5, max_len=100):
        self.model = model
        self.device = device
        self.beam_width = beam_width
        self.max_len = max_len
        
    def decode(self, src, src_vocab, tgt_vocab):
        """
        Beam search decoding for a single source sentence.
        
        Args:
            src: [1, src_len] tensor
            src_vocab: source vocabulary
            tgt_vocab: target vocabulary
            
        Returns:
            List of token indices (best sequence)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode
            if hasattr(self.model.encoder, 'rnn_type') and self.model.encoder.rnn_type == 'lstm':
                encoder_outputs, hidden, cell = self.model.encoder(src)
            else:
                encoder_outputs, hidden = self.model.encoder(src)
                cell = None
            
            sos_idx = tgt_vocab.stoi.get("<SOS>", 1)
            eos_idx = tgt_vocab.stoi.get("<EOS>", 2)
            
            # Initialize beams: (sequence, score, hidden, cell)
            # Prepare hidden state
            hidden_init = hidden.unsqueeze(0).repeat(self.model.decoder.rnn.num_layers, 1, 1)
            if cell is not None:
                cell_init = cell.unsqueeze(0).repeat(self.model.decoder.rnn.num_layers, 1, 1)
            else:
                cell_init = None
            
            beams = [([sos_idx], 0.0, hidden_init, cell_init)]
            completed = []
            
            # Create mask
            mask = (src != 0)
            
            for step in range(self.max_len):
                candidates = []
                
                for seq, score, hid, c in beams:
                    # Skip if sequence already ended
                    if seq[-1] == eos_idx:
                        completed.append((seq, score))
                        continue
                    
                    input_token = torch.tensor([seq[-1]]).to(self.device)
                    
                    if cell is not None:
                        output, new_hid, new_c = self.model.decoder(input_token, hid, encoder_outputs, mask, c)
                    else:
                        output, new_hid = self.model.decoder(input_token, hid, encoder_outputs, mask)
                        new_c = None
                    
                    # Get top k predictions
                    log_probs = F.log_softmax(output, dim=1)
                    top_probs, top_indices = log_probs.topk(self.beam_width)

                    for i in range(self.beam_width):
                        token = top_indices[0, i].item()
                        token_score = top_probs[0, i].item()
                        new_seq = seq + [token]
                        new_score = score + token_score
                        candidates.append((new_seq, new_score, new_hid, new_c))

                # Select top beam_width candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:self.beam_width]

                # Check if all beams have ended
                if all(seq[-1] == eos_idx for seq, _, _, _ in beams):
                    completed.extend(beams)
                    break

            # Add remaining beams to completed
            completed.extend(beams)

            # Return the sequence with highest score
            if completed:
                best_seq, _ = max(completed, key=lambda x: x[1])
                # Remove SOS and EOS tokens
                return [token for token in best_seq if token not in [sos_idx, eos_idx]]
            else:
                return []

