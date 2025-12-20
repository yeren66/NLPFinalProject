import torch

class Config:
    # Data
    TRAIN_FILE = 'data/train.jsonl'
    VALID_FILE = 'data/valid.jsonl'
    TEST_FILE = 'data/test.jsonl'
    SRC_LANG = 'zh'
    TGT_LANG = 'en'
    MAX_LEN = 128
    MIN_FREQ = 2
    
    # Model General
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    
    # RNN Hyperparameters
    RNN_ENC_LAYERS = 2
    RNN_DEC_LAYERS = 2
    RNN_HID_DIM = 512
    RNN_EMB_DIM = 256
    RNN_DROPOUT = 0.5
    ATTN_TYPE = 'dot' # dot, multiplicative, additive
    
    # Transformer Hyperparameters
    TRANS_LAYERS = 6
    TRANS_HEADS = 8
    TRANS_HID_DIM = 512
    TRANS_PF_DIM = 2048
    TRANS_DROPOUT = 0.1
    TRANS_POS_ENC = 'absolute' # absolute, relative
    TRANS_NORM = 'layer' # layer, rms
    
    # Training
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10
    CLIP = 1.0
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    VOCAB_DIR = 'data/vocab'
