"""Unified inference script for RNN/Transformer models.

Usage example:
    # RNN
    python inference.py --model-type rnn --checkpoint path/to/model.ckpt --test-file path/to/test.jsonl

    # Transformer
    python inference.py --model-type transformer --checkpoint path/to/model_dir --test-file path/to/test.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from config import Config
from train_rnn import RNNSeq2SeqModule
from train_transformer import TransformerConfig, TransformerForSeq2Seq
from utils.data_loader import NMTDataset
from utils.decode import GreedyDecoder, BeamSearchDecoder, greedy_decode
from utils.metrics import compute_corpus_bleu, format_bleu_score
from utils.preprocess import preprocess_sample


SPECIAL_TOKENS = {"<PAD>", "<pad>", "<SOS>", "<sos>", "<EOS>", "<eos>"}


def build_vocabs(vocab_file: str) -> Tuple[object, object]:
    """Build vocabularies from a data file (usually training data)."""
    print(f"Building vocabulary from {vocab_file}...")
    # We use NMTDataset to build vocab. 
    # Note: In a production setting, you might want to save/load the vocab object directly.
    dataset = NMTDataset(vocab_file, build_vocab=True)
    return dataset.src_vocab, dataset.tgt_vocab


def tokens_from_indices(indices: List[int], vocab) -> List[str]:
    """Convert token indices to tokens while dropping special symbols."""
    return [
        vocab.itos.get(idx, "<UNK>")
        for idx in indices
        if vocab.itos.get(idx, "") not in SPECIAL_TOKENS
    ]


def load_rnn_model(checkpoint_path: Path, src_vocab, tgt_vocab, device: torch.device):
    """Load RNN model from a Lightning checkpoint using stored hyperparameters."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RNN checkpoint not found: {checkpoint_path}")
        
    module = RNNSeq2SeqModule.load_from_checkpoint(
        checkpoint_path,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        map_location=device,
    )
    model = module.model.to(device)
    model.eval()
    return model


def load_transformer_model(checkpoint_dir: Path, device: torch.device):
    """Load Transformer model from a HuggingFace-style checkpoint directory."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {checkpoint_dir}")

    config = TransformerConfig.from_pretrained(checkpoint_dir)
    wrapper = TransformerForSeq2Seq.from_pretrained(checkpoint_dir, config=config)
    wrapper.to(device)
    wrapper.eval()
    core_model = wrapper.model
    core_model.to(device)
    core_model.eval()
    return core_model


def translate_rnn(sentence: str, model, src_vocab, tgt_vocab, device, decoder_type: str, beam_width: int):
    src_tokens, _ = preprocess_sample(sentence, "")
    src_indices = [src_vocab.stoi.get("<SOS>", 1)] + src_vocab.numericalize(src_tokens) + [
        src_vocab.stoi.get("<EOS>", 2)
    ]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)

    decoder = (
        GreedyDecoder(model, device)
        if decoder_type == "greedy"
        else BeamSearchDecoder(model, device, beam_width=beam_width)
    )
    output_indices = decoder.decode(src_tensor, src_vocab, tgt_vocab)
    tokens = tokens_from_indices(output_indices, tgt_vocab)
    return tokens


def translate_transformer(sentence: str, model, src_vocab, tgt_vocab, device, max_len: int):
    src_tokens, _ = preprocess_sample(sentence, "")
    src_indices = [src_vocab.stoi.get("<SOS>", 1)] + src_vocab.numericalize(src_tokens) + [
        src_vocab.stoi.get("<EOS>", 2)
    ]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)

    output_indices = greedy_decode(model, src_tensor, tgt_vocab, max_len=max_len)
    tokens = tokens_from_indices(output_indices, tgt_vocab)
    return tokens


def run_inference(
    model_type: str,
    checkpoint: Path,
    test_file: Path,
    vocab_file: str,
    decoder: str,
    beam_width: int,
    max_len: int,
    max_samples: int,
    output_path: Path,
):
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Build Vocab
    # Ideally, the user should provide the file used to build vocab during training
    src_vocab, tgt_vocab = build_vocabs(vocab_file)

    # 2. Load model
    if model_type == "rnn":
        print(f"Loading RNN checkpoint: {checkpoint}")
        model = load_rnn_model(checkpoint, src_vocab, tgt_vocab, device)
    elif model_type == "transformer":
        print(f"Loading Transformer checkpoint: {checkpoint}")
        model = load_transformer_model(checkpoint, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 3. Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    if max_samples > 0:
        print(f"Limiting to first {max_samples} samples.")
        test_data = test_data[:max_samples]

    references: List[List[str]] = []
    hypotheses: List[List[str]] = []
    translations = []

    print(f"Translating {len(test_data)} sentences with {model_type} ({decoder})...")

    for idx, item in enumerate(tqdm(test_data, desc="Decoding", ncols=90)):
        src_text = item.get("zh", item.get("source", ""))
        tgt_text = item.get("en", item.get("target", ""))

        if model_type == "rnn":
            hypo_tokens = translate_rnn(src_text, model, src_vocab, tgt_vocab, device, decoder, beam_width)
        else:
            hypo_tokens = translate_transformer(src_text, model, src_vocab, tgt_vocab, device, max_len=max_len)

        translation = " ".join(hypo_tokens)
        _, ref_tokens = preprocess_sample("", tgt_text)

        references.append(ref_tokens)
        hypotheses.append(hypo_tokens)
        translations.append({
            "source": src_text,
            "reference": tgt_text,
            "translation": translation,
        })

    # 4. Compute Metrics
    bleu = compute_corpus_bleu(references, hypotheses)
    print(f"\n{'='*30}")
    print(f"BLEU Score: {format_bleu_score(bleu)}")
    print(f"{'='*30}\n")

    # 5. Save Results
    output = {
        "model_type": model_type,
        "checkpoint": str(checkpoint),
        "decoder": decoder,
        "beam_width": beam_width if decoder == "beam" else None,
        "max_len": max_len,
        "test_samples": len(test_data),
        "bleu": bleu,
        "translations": translations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved translations to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for RNN/Transformer models")
    
    # Required arguments for the "downloaded model" scenario
    parser.add_argument("--model-type", choices=["rnn", "transformer"], required=True, help="Model architecture type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (RNN .ckpt) or directory (Transformer)")
    
    # Data arguments
    parser.add_argument("--test-file", type=str, default=Config.TEST_FILE, help="Path to the test dataset (.jsonl)")
    parser.add_argument("--vocab-file", type=str, default=Config.TRAIN_FILE, help="Path to the file used to build vocabulary (usually training data)")
    
    # Decoding arguments
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy", help="Decoding strategy (RNN only)")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for beam search (RNN only)")
    parser.add_argument("--max-len", type=int, default=100, help="Maximum generation length")
    
    # Output arguments
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples to translate (0 = all)")
    parser.add_argument("--output", type=str, default="translations.json", help="Path to save output JSON")
    
    return parser.parse_args()


def main():
    args = parse_args()

    run_inference(
        model_type=args.model_type,
        checkpoint=Path(args.checkpoint),
        test_file=Path(args.test_file),
        vocab_file=args.vocab_file,
        decoder=args.decoder,
        beam_width=args.beam_width,
        max_len=args.max_len,
        max_samples=args.max_samples,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

    # For RNN
    # python inference.py --model-type rnn --checkpoint downloaded_models/rnn_best.ckpt --test-file data/test.jsonl

    # For Transformer
    # python inference.py --model-type transformer --checkpoint downloaded_models/transformer_dir --test-file data/test.jsonl
