import re
import jieba

def clean_text(text, lang):
    """
    Basic text cleaning.
    """
    text = text.lower().strip()
    # Remove some special characters if needed, but keep punctuation
    # For this project, we accept some noise, but removing rare chars is good.
    # Keep chinese chars, english letters, numbers, punctuation.
    return text

def tokenize_zh(text):
    """
    Tokenize Chinese text using jieba.
    """
    return list(jieba.cut(text))

def tokenize_en(text):
    """
    Tokenize English text using simple split or regex.
    For production, spacy is better, but this avoids model dependency issues for now.
    """
    # Simple regex to split by space and punctuation
    return re.findall(r"[\w']+|[.,!?;]", text)

def preprocess_sample(src_text, tgt_text):
    src_text = clean_text(src_text, 'zh')
    tgt_text = clean_text(tgt_text, 'en')
    
    src_tokens = tokenize_zh(src_text)
    tgt_tokens = tokenize_en(tgt_text)
    
    return src_tokens, tgt_tokens
