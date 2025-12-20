import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import json
import os
from .preprocess import preprocess_sample

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def numericalize(self, text):
        tokenized_text = text if isinstance(text, list) else list(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class NMTDataset(Dataset):
    def __init__(self, file_path, src_vocab=None, tgt_vocab=None, build_vocab=False, max_len=128):
        self.data = []
        self.max_len = max_len
        
        # Load data
        print(f"Loading data from {file_path}...")
        raw_data = [] # List of (src, tgt)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    # Assume keys are 'zh' and 'en' or similar, strict check later
                    # Modify based on actual data key names
                    src = obj.get('zh', obj.get('source', ''))
                    tgt = obj.get('en', obj.get('target', ''))
                    raw_data.append((src, tgt))
        else:
            print(f"File {file_path} not found. Using empty data.")
        
        # Preprocess
        src_texts = []
        tgt_texts = []
        for s, t in tqdm(raw_data, desc="Preprocessing"):
            s_tok, t_tok = preprocess_sample(s, t)
            src_texts.append(s_tok)
            tgt_texts.append(t_tok)
            self.data.append((s_tok, t_tok))
            
        if build_vocab:
            self.src_vocab = Vocabulary()
            self.tgt_vocab = Vocabulary()
            self.src_vocab.build_vocabulary(src_texts)
            self.tgt_vocab.build_vocabulary(tgt_texts)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        src_tokens, tgt_tokens = self.data[index]
        
        src_indices = [self.src_vocab.stoi["<SOS>"]] + self.src_vocab.numericalize(src_tokens) + [self.src_vocab.stoi["<EOS>"]]
        tgt_indices = [self.tgt_vocab.stoi["<SOS>"]] + self.tgt_vocab.numericalize(tgt_tokens) + [self.tgt_vocab.stoi["<EOS>"]]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    # Pad sequences
    src_batch, tgt_batch = [], []
    for s, t in batch:
        src_batch.append(s)
        tgt_batch.append(t)
        
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    
    return src_batch, tgt_batch
