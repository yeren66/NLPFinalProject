import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import random
from config import Config
from utils.data_loader import NMTDataset, collate_fn
# Import models based on config
from models.rnn.encoder import Encoder
from models.rnn.decoder import Decoder
from models.rnn.seq2seq import Seq2Seq
from models.transformer.model import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
from models.rnn.attention import Attention

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, tgt) in enumerate(iterator):
        src = src.to(Config.DEVICE)
        tgt = tgt.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Transformer output: [batch, tgt_len, vocab]
        # RNN output: [batch, tgt_len, vocab]
        
        if isinstance(model, Seq2Seq):
            output = model(src, tgt)
            # trg = [batch, trg len]
            # output = [batch, trg len, output dim]
            
            output_dim = output.shape[-1]
            
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
        elif isinstance(model, Transformer):
            # Input to transformer: src, tgt_input (exclude last)
            # Target for loss: tgt_output (exclude first)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt = tgt_output.reshape(-1)
            
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(Config.DEVICE)
            tgt = tgt.to(Config.DEVICE)
            
            if isinstance(model, Seq2Seq):
                output = model(src, tgt, 0) # 0 teacher forcing
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
            elif isinstance(model, Transformer):
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                output = model(src, tgt_input)
                output = output.reshape(-1, output.shape[-1])
                tgt = tgt_output.reshape(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def main():
    # Load Data
    train_dataset = NMTDataset(Config.TRAIN_FILE, build_vocab=True)
    valid_dataset = NMTDataset(Config.VALID_FILE, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)
    
    input_dim = len(train_dataset.src_vocab)
    output_dim = len(train_dataset.tgt_vocab)
    
    print(f"Input Dim: {input_dim}, Output Dim: {output_dim}")
    
    # Init Model (Example: RNN)
    attn = Attention(Config.RNN_ENC_LAYERS * Config.RNN_HID_DIM * 2, Config.RNN_DEC_LAYERS * Config.RNN_HID_DIM, method=Config.ATTN_TYPE)
    enc = Encoder(input_dim, Config.RNN_EMB_DIM, Config.RNN_HID_DIM, Config.RNN_HID_DIM, Config.RNN_DROPOUT, n_layers=Config.RNN_ENC_LAYERS)
    dec = Decoder(output_dim, Config.RNN_EMB_DIM, Config.RNN_HID_DIM, Config.RNN_HID_DIM, Config.RNN_DROPOUT, attn, n_layers=Config.RNN_DEC_LAYERS)
    model = Seq2Seq(enc, dec, Config.DEVICE).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # PAD index
    
    best_valid_loss = float('inf')
    
    for epoch in range(Config.N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, Config.CLIP)
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{Config.CHECKPOINT_DIR}/model.pt')
            
        print(f'Epoch: {epoch+1:02} | Time: {end_time - start_time}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {math.exp(train_loss):.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} | PPL: {math.exp(valid_loss):.3f}')

if __name__ == "__main__":
    main()
