import torch

class Config:
    # Data
    # Update paths to match the actual directory structure
    DATA_DIR = 'AP0004_Midterm&Final_translation_dataset_zh_en'
    TRAIN_FILE = f'{DATA_DIR}/train_100k.jsonl' # Use train_100k.jsonl for full training
    VALID_FILE = f'{DATA_DIR}/valid.jsonl'
    TEST_FILE = f'{DATA_DIR}/test.jsonl'
    SRC_LANG = 'zh'
    TGT_LANG = 'en'
    MAX_LEN = 128
    MIN_FREQ = 2
    
    # Model General
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    
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

    # T5 Model Configuration
    T5_LOCAL_MODEL_PATH = 'T5_model'  # Local T5 model directory
    T5_MODEL_NAME = 'google-t5/t5-base'  # Fallback HuggingFace model name
    T5_MAX_LENGTH = 128
    
    # Training
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10
    CLIP = 1.0
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    VOCAB_DIR = 'data/vocab'
