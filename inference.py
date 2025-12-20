import torch
from config import Config
from models.rnn.seq2seq import Seq2Seq
# ... import other models ...
# Helper to reload model and predict 

def inference(model, sentence, src_vocab, tgt_vocab, device):
    model.eval()
    # Tokenize
    # Numericalize
    # Tensor
    # Call model
    pass
    # This calls for a proper translation loop using Beam search or Greedy
    # For now, placeholder.
