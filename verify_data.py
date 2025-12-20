import json
from utils.data_loader import NMTDataset, collate_fn
from torch.utils.data import DataLoader

# Create dummy data
dummy_data = [
    {"zh": "我爱自然语言处理", "en": "I love natural language processing"},
    {"zh": "机器学习很有趣", "en": "Machine learning is fun"},
    {"zh": "深度学习彻底改变了人工智能", "en": "Deep learning revolutionized AI"}
]

with open('dummy_train.jsonl', 'w', encoding='utf-8') as f:
    for item in dummy_data:
        f.write(json.dumps(item) + '\n')

# Test Dataset and Vocab
dataset = NMTDataset('dummy_train.jsonl', build_vocab=True)
print(f"Dataset Size: {len(dataset)}")
print(f"Source Vocab Size: {len(dataset.src_vocab)}")
print(f"Target Vocab Size: {len(dataset.tgt_vocab)}")

# Test DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

for src, tgt in loader:
    print("Batch Shapes:")
    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")
    break
