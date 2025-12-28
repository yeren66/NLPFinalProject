"""
Transformer Training Script using HuggingFace Transformers
Refactored from train_transformer.py for better modularity and maintainability
"""

import torch
import torch.nn as nn
from transformers import (
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    EvalPrediction,
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.utils.data import Dataset
import argparse
import os
import json
import math
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

# Set Tensor Cores precision for H100 GPU optimization
# This trades off precision for performance on GPUs with Tensor Cores
# Options: 'highest' (default), 'high', 'medium'
# 'high' or 'medium' provides significant speedup with minimal accuracy loss
torch.set_float32_matmul_precision('high')

from config import Config
from utils.data_loader import NMTDataset, collate_fn
from models.transformer.model import Transformer, TransformerEncoderLayer, TransformerDecoderLayer
from utils.metrics import compute_corpus_bleu
from utils.decode import greedy_decode
from tqdm import tqdm


class TransformerConfig(PretrainedConfig):
    """Configuration class for custom Transformer model"""
    
    model_type = "custom_transformer"
    
    def __init__(
        self,
        vocab_size_src=50000,
        vocab_size_tgt=50000,
        hidden_size=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        pos_enc_type='absolute',
        norm_type='layer',
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pos_enc_type = pos_enc_type
        self.norm_type = norm_type


class TransformerForSeq2Seq(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible wrapper for custom Transformer model"""

    config_class = TransformerConfig
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config
        
        # Build encoder and decoder layers
        enc_layer = TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.hidden_dropout_prob,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            norm_type=config.norm_type,
            pos_enc_type=config.pos_enc_type
        )
        
        dec_layer = TransformerDecoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.hidden_dropout_prob,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            norm_type=config.norm_type,
            pos_enc_type=config.pos_enc_type
        )
        
        # Build the transformer model
        self.model = Transformer(
            enc_layer,
            dec_layer,
            config.num_encoder_layers,
            config.num_decoder_layers,
            config.vocab_size_src,
            config.vocab_size_tgt,
            config.hidden_size,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            pos_enc_type=config.pos_enc_type
        )
        
        # Initialize weights
        self.post_init()
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Prepare decoder input
        if decoder_input_ids is None and labels is not None:
            # Shift labels to create decoder input
            decoder_input_ids = labels[:, :-1]

        # Forward pass through model
        logits = self.model(input_ids, decoder_input_ids)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            # Since decoder_input_ids is already shifted (labels[:, :-1]),
            # logits correspond to predictions for labels[:, 1:]
            # So we compare logits with labels[:, 1:]
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class HFNMTDataset(Dataset):
    """HuggingFace-compatible wrapper for NMTDataset"""

    def __init__(self, nmt_dataset):
        self.dataset = nmt_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.dataset[idx]

        return {
            'input_ids': torch.LongTensor(src_tokens),
            'labels': torch.LongTensor(tgt_tokens),
        }


def data_collator(features):
    """Custom data collator for batching"""
    # Get max lengths
    max_src_len = max(len(f['input_ids']) for f in features)
    max_tgt_len = max(len(f['labels']) for f in features)

    batch_input_ids = []
    batch_labels = []

    for f in features:
        # Pad source
        src = f['input_ids']
        src_padded = torch.cat([src, torch.zeros(max_src_len - len(src), dtype=torch.long)])
        batch_input_ids.append(src_padded)

        # Pad target
        tgt = f['labels']
        tgt_padded = torch.cat([tgt, torch.zeros(max_tgt_len - len(tgt), dtype=torch.long)])
        batch_labels.append(tgt_padded)

    return {
        'input_ids': torch.stack(batch_input_ids),
        'labels': torch.stack(batch_labels),
    }


def compute_metrics(eval_preds: EvalPrediction, src_vocab=None, tgt_vocab=None):
    """Compute metrics for evaluation"""
    preds, labels = eval_preds

    # Calculate perplexity from loss
    # Note: HF Trainer computes loss separately
    metrics = {}

    return metrics


class TransformerTrainerWithBLEU(Trainer):
    """Custom Trainer with BLEU computation"""

    def __init__(self, *args, src_vocab=None, tgt_vocab=None, compute_bleu=True, compute_bleu_every_epoch=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.compute_bleu = compute_bleu
        self.compute_bleu_every_epoch = compute_bleu_every_epoch

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Compute BLEU if enabled
        # For validation during training (metric_key_prefix="eval"), only compute if compute_bleu_every_epoch is True
        # For test evaluation (metric_key_prefix="test"), always compute if compute_bleu is True
        
        # Use self.eval_dataset if eval_dataset is None
        actual_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        should_compute_bleu = self.compute_bleu and actual_eval_dataset is not None
        if metric_key_prefix == "eval":
            should_compute_bleu = should_compute_bleu and self.compute_bleu_every_epoch

        if should_compute_bleu:
            print(f"\nComputing BLEU score on {metric_key_prefix} set...")

            # Get the underlying NMT dataset
            if hasattr(actual_eval_dataset, 'dataset'):
                nmt_dataset = actual_eval_dataset.dataset
            else:
                nmt_dataset = actual_eval_dataset

            # For validation, use subset (1000 samples) for speed
            # For test, use all samples
            if metric_key_prefix == "eval":
                max_samples = 1000
            else:
                max_samples = None

            bleu_score = self._compute_bleu(nmt_dataset, max_samples=max_samples)
            output[f'{metric_key_prefix}_bleu'] = bleu_score

            print(f"{metric_key_prefix} BLEU: {bleu_score:.4f}")

        return output

    def _compute_bleu(self, dataset, max_samples=None):
        """Compute BLEU score on dataset"""
        self.model.eval()
        device = self.model.device

        references = []
        hypotheses = []

        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        with torch.no_grad():
            # Add progress bar
            pbar = tqdm(range(num_samples), desc='Computing BLEU', leave=False)
            for idx in pbar:
                src_tokens, tgt_tokens = dataset[idx]
                src = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

                # Greedy decode
                output_indices = greedy_decode(
                    self.model.model, src, self.tgt_vocab, max_len=Config.MAX_LEN
                )

                # Convert to tokens
                pad_idx = self.tgt_vocab.stoi.get('<PAD>', self.tgt_vocab.stoi.get('<pad>', 0))
                sos_idx = self.tgt_vocab.stoi.get('<SOS>', self.tgt_vocab.stoi.get('<sos>', 1))
                eos_idx = self.tgt_vocab.stoi.get('<EOS>', self.tgt_vocab.stoi.get('<eos>', 2))

                ref_tokens = [self.tgt_vocab.itos[idx] for idx in tgt_tokens.tolist()
                             if idx not in [pad_idx, sos_idx, eos_idx]]
                hyp_tokens = [self.tgt_vocab.itos[idx] for idx in output_indices
                             if idx not in [pad_idx, sos_idx, eos_idx]]

                references.append(ref_tokens)
                hypotheses.append(hyp_tokens)

        return compute_corpus_bleu(references, hypotheses)


def train_model(
    pos_enc_type='absolute',
    norm_type='layer',
    n_epochs=10,
    batch_size=256,
    learning_rate=0.0005,
    output_dir='checkpoints',
    experiment_name='transformer_baseline',
    resume_from=None,
    compute_bleu=True,
    compute_bleu_every_epoch=True,
):
    """Train a single Transformer model with specified configuration"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    print("Loading data...")
    train_dataset = NMTDataset(Config.TRAIN_FILE, build_vocab=True)
    valid_dataset = NMTDataset(
        Config.VALID_FILE,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    test_dataset = NMTDataset(
        Config.TEST_FILE,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )

    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    input_dim = len(src_vocab)
    output_dim = len(tgt_vocab)

    print(f"\n{'='*70}")
    print(f"Dataset Information:")
    print(f"  Source Vocab Size: {input_dim}")
    print(f"  Target Vocab Size: {output_dim}")
    print(f"  Train Samples: {len(train_dataset)}")
    print(f"  Valid Samples: {len(valid_dataset)}")
    print(f"  Test Samples: {len(test_dataset)}")
    print(f"  Batch Size: {batch_size}")
    print(f"{'='*70}\n")

    # Wrap datasets for HuggingFace
    hf_train_dataset = HFNMTDataset(train_dataset)
    hf_valid_dataset = HFNMTDataset(valid_dataset)
    hf_test_dataset = HFNMTDataset(test_dataset)

    # Create model config
    config = TransformerConfig(
        vocab_size_src=input_dim,
        vocab_size_tgt=output_dim,
        hidden_size=Config.TRANS_HID_DIM,
        num_encoder_layers=Config.TRANS_LAYERS,
        num_decoder_layers=Config.TRANS_LAYERS,
        num_attention_heads=Config.TRANS_HEADS,
        intermediate_size=Config.TRANS_PF_DIM,
        hidden_dropout_prob=Config.TRANS_DROPOUT,
        attention_probs_dropout_prob=Config.TRANS_DROPOUT,
        max_position_embeddings=Config.MAX_LEN,
        pos_enc_type=pos_enc_type,
        norm_type=norm_type,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    # Initialize model
    model = TransformerForSeq2Seq(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, experiment_name),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'runs/{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,  # Only keep best and last checkpoint
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=['tensorboard'],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Positional Encoding: {pos_enc_type}")
    print(f"  Normalization: {norm_type}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Experiment Name: {experiment_name}")
    print(f"{'='*70}\n")

    # Initialize trainer
    trainer = TransformerTrainerWithBLEU(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_valid_dataset,
        data_collator=data_collator,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        compute_bleu=compute_bleu,
        compute_bleu_every_epoch=compute_bleu_every_epoch,
    )

    # Train
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Evaluate on test set
    print(f"\n{'='*70}")
    print(f"Evaluating on Test Set")
    print(f"{'='*70}\n")

    # This will compute both loss and BLEU (with progress bar)
    test_results = trainer.evaluate(eval_dataset=hf_test_dataset, metric_key_prefix='test')

    # Get BLEU from test_results (already computed in evaluate())
    test_bleu = test_results.get('test_bleu', 0.0)

    print(f"\n{'='*70}")
    print(f"Test Results:")
    print(f"  Test Loss: {test_results.get('test_loss', 'N/A')}")
    if 'test_loss' in test_results:
        print(f"  Test PPL: {math.exp(min(test_results['test_loss'], 100)):.2f}")
    print(f"  Test BLEU: {test_bleu:.4f}")
    print(f"{'='*70}\n")

    # Save test results
    results_dict = {
        'test_bleu': test_bleu,
        'test_loss': test_results.get('test_loss'),
        'pos_enc_type': pos_enc_type,
        'norm_type': norm_type,
    }

    results_path = os.path.join(output_dir, f'{experiment_name}_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Test results saved to: {results_path}")

    return model, results_dict


def main():
    parser = argparse.ArgumentParser(description='Transformer Training Script with HuggingFace')

    # Experiment type
    parser.add_argument('--experiment_type', type=str, default='baseline',
                       choices=['baseline', 'positional_encoding', 'normalization', 't5_finetune'],
                       help='Type of experiment to run')

    # Positional encoding parameters
    parser.add_argument('--pos_enc_types', type=str, nargs='+',
                       default=None,
                       help='Positional encoding types to compare')
    parser.add_argument('--pos_enc_type', type=str, default='absolute',
                       help='Single positional encoding type')

    # Normalization parameters
    parser.add_argument('--norm_types', type=str, nargs='+',
                       default=None,
                       help='Normalization types to compare')
    parser.add_argument('--norm_type', type=str, default='layer',
                       help='Single normalization type')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=None,
                       help='Number of epochs (default: from config)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: from config)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # BLEU computation
    parser.add_argument('--no_bleu', action='store_true',
                       help='Disable BLEU computation during training')

    # T5-specific parameters
    parser.add_argument('--t5_model', type=str, default='google-t5/t5-base',
                       help='T5 model name for fine-tuning (default: google-t5/t5-base)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for T5 (default: 128)')

    args = parser.parse_args()

    # Set defaults from config
    if args.n_epochs is None:
        args.n_epochs = Config.N_EPOCHS
    if args.learning_rate is None:
        args.learning_rate = Config.LEARNING_RATE
    if args.output_dir is None:
        args.output_dir = Config.CHECKPOINT_DIR

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    compute_bleu = not args.no_bleu

    dataset_name = os.path.splitext(os.path.basename(Config.TRAIN_FILE))[0]

    # Run experiment based on type
    if args.experiment_type == 'baseline':
        print("\n" + "="*70)
        print("Running BASELINE experiment with HuggingFace Transformers")
        print("="*70)

        model, test_results = train_model(
            pos_enc_type=args.pos_enc_type,
            norm_type=args.norm_type,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            experiment_name=f'transformer_{dataset_name}_baseline',
            resume_from=args.resume,
            compute_bleu=compute_bleu,
            compute_bleu_every_epoch=compute_bleu,
        )

        print(f'\nTest BLEU: {test_results["test_bleu"]:.4f}')

    elif args.experiment_type == 'positional_encoding':
        print("\n" + "="*70)
        print("Running POSITIONAL ENCODING COMPARISON experiment")
        print("="*70)

        pos_enc_types = args.pos_enc_types or ['absolute', 'relative']
        all_results = {}

        for pos_enc in pos_enc_types:
            print(f"\n{'='*70}")
            print(f"Training with positional encoding: {pos_enc}")
            print(f"{'='*70}")

            model, test_results = train_model(
                pos_enc_type=pos_enc,
                norm_type=args.norm_type,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                experiment_name=f'transformer_{dataset_name}_{pos_enc}',
                resume_from=args.resume,
                compute_bleu=compute_bleu,
                compute_bleu_every_epoch=compute_bleu,
            )

            all_results[pos_enc] = test_results

        # Print comparison
        print("\n" + "="*70)
        print("POSITIONAL ENCODING COMPARISON RESULTS")
        print("="*70)
        for pos_enc, results in all_results.items():
            print(f"{pos_enc:15s}: Test BLEU = {results['test_bleu']:.4f}")

        # Save results
        with open('results/transformer_positional_encoding_comparison_hf.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    elif args.experiment_type == 'normalization':
        print("\n" + "="*70)
        print("Running NORMALIZATION COMPARISON experiment")
        print("="*70)

        norm_types = args.norm_types or ['layer', 'rms']
        all_results = {}

        for norm in norm_types:
            print(f"\n{'='*70}")
            print(f"Training with normalization: {norm}")
            print(f"{'='*70}")

            model, test_results = train_model(
                pos_enc_type=args.pos_enc_type,
                norm_type=norm,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                experiment_name=f'transformer_{dataset_name}_norm_{norm}',
                resume_from=args.resume,
                compute_bleu=compute_bleu,
                compute_bleu_every_epoch=compute_bleu,
            )

            all_results[norm] = test_results

        # Print comparison
        print("\n" + "="*70)
        print("NORMALIZATION COMPARISON RESULTS")
        print("="*70)
        for norm, results in all_results.items():
            print(f"{norm:15s}: Test BLEU = {results['test_bleu']:.4f}")

        # Save results
        with open('results/transformer_normalization_comparison_hf.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    elif args.experiment_type == 't5_finetune':
        print("\n" + "="*70)
        print("Running T5 FINE-TUNING experiment")
        print("="*70)

        # Adjust batch size for T5 (usually needs smaller batch size)
        t5_batch_size = min(args.batch_size // 16, 16)  # T5 is larger, use smaller batch

        model, test_results = train_t5_model(
            model_name=args.t5_model,
            n_epochs=args.n_epochs,
            batch_size=t5_batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            experiment_name=f't5_{dataset_name}_finetuned',
            max_length=args.max_length,
            compute_bleu=compute_bleu,
            compute_bleu_every_epoch=compute_bleu,
        )

        print(f'\nTest BLEU: {test_results.get("test_bleu", 0):.4f}')

        # Save results
        results_dict = {
            't5_finetuned': {
                'model': args.t5_model,
                'test_loss': test_results['test_loss'],
                'test_bleu': test_results.get('test_bleu', 0),
            }
        }

        with open('results/transformer_t5_finetune_hf.json', 'w') as f:
            json.dump(results_dict, f, indent=2)


def train_t5_model(
    model_name='google-t5/t5-base',
    n_epochs=10,
    batch_size=16,
    learning_rate=3e-4,
    output_dir='checkpoints',
    experiment_name='t5_finetuned',
    max_length=128,
    compute_bleu=True,
    compute_bleu_every_epoch=True,
):
    """Fine-tune a pretrained T5 model for Chinese-English translation"""

    print(f"\n{'='*70}")
    print(f"Fine-tuning T5 Model: {model_name}")
    print(f"{'='*70}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if local model exists (prioritize local model)
    local_model_path = Config.T5_LOCAL_MODEL_PATH
    if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
        # Check if it's a valid model directory
        required_files = ['config.json', 'pytorch_model.bin', 'spiece.model']
        has_all_files = all(os.path.exists(os.path.join(local_model_path, f)) for f in required_files)

        if has_all_files:
            print(f"✅ Found local T5 model at: {local_model_path}")
            print("Loading from local directory (no download needed)...")
            model_path = local_model_path
        else:
            print(f"⚠️  Local model directory exists but missing required files")
            print(f"Required files: {required_files}")
            print(f"Downloading from HuggingFace: {model_name}")
            model_path = model_name
    else:
        print(f"⚠️  Local model not found at {local_model_path}")
        print(f"Downloading from HuggingFace: {model_name}")
        model_path = model_name

    # Load T5 tokenizer and model
    print("Loading pretrained T5 model and tokenizer...")

    # Disable problematic CUDA optimizations that may cause compatibility issues
    import os as _os
    _os.environ['DISABLE_FUSED_LAYER_NORM'] = '1'

    try:
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=(model_path == local_model_path)
        )

        # Load model with compatibility settings
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=(model_path == local_model_path),
            # Avoid potential CUDA kernel issues
            attn_implementation="eager"  # Use standard attention instead of flash attention
        )
        print(f"✅ Successfully loaded T5 model from: {model_path}")

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"   Model moved to GPU")

    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        if model_path == local_model_path:
            print(f"Falling back to downloading from HuggingFace: {model_name}")
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="eager"
            )
            if torch.cuda.is_available():
                model = model.cuda()
        else:
            raise

    # Load datasets
    print("Loading data...")
    train_dataset = NMTDataset(Config.TRAIN_FILE, build_vocab=True)
    valid_dataset = NMTDataset(
        Config.VALID_FILE,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    test_dataset = NMTDataset(
        Config.TEST_FILE,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create T5-specific dataset wrapper
    class T5Dataset(Dataset):
        def __init__(self, nmt_dataset, tokenizer, max_length=128):
            self.nmt_dataset = nmt_dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.nmt_dataset)

        def __getitem__(self, idx):
            src, tgt = self.nmt_dataset[idx]

            # Convert indices back to text
            src_text = ' '.join([self.nmt_dataset.src_vocab.itos[i.item()] for i in src if i.item() not in [0, 1, 2]])
            tgt_text = ' '.join([self.nmt_dataset.tgt_vocab.itos[i.item()] for i in tgt if i.item() not in [0, 1, 2]])

            # Add task prefix for T5
            input_text = f"translate Chinese to English: {src_text}"

            # Tokenize
            model_inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )

            # Tokenize targets
            labels = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    # Create T5 datasets
    t5_train_dataset = T5Dataset(train_dataset, tokenizer, max_length)
    t5_valid_dataset = T5Dataset(valid_dataset, tokenizer, max_length)
    t5_test_dataset = T5Dataset(test_dataset, tokenizer, max_length)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, experiment_name),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'runs/{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=['tensorboard'],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        predict_with_generate=True,
        generation_max_length=max_length,
    )

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Length: {max_length}")
    print(f"  Output Directory: {output_dir}")
    print(f"{'='*70}\n")

    # Custom trainer for BLEU computation
    class T5TrainerWithBLEU(Seq2SeqTrainer):
        def __init__(self, *args, nmt_dataset=None, src_vocab=None, tgt_vocab=None,
                     compute_bleu_flag=True, compute_bleu_every_epoch_flag=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.nmt_dataset = nmt_dataset
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.compute_bleu_flag = compute_bleu_flag
            self.compute_bleu_every_epoch_flag = compute_bleu_every_epoch_flag

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

            # Compute BLEU if requested
            should_compute = self.compute_bleu_flag
            if metric_key_prefix == "eval":
                should_compute = should_compute and self.compute_bleu_every_epoch_flag

            if should_compute and self.nmt_dataset is not None:
                print(f"\nComputing BLEU score on {metric_key_prefix} set...")

                # Sample for faster validation during training
                max_samples = 1000 if metric_key_prefix == "eval" else None

                bleu_score = self._compute_bleu_t5(self.nmt_dataset, max_samples)
                output[f'{metric_key_prefix}_bleu'] = bleu_score
                print(f"{metric_key_prefix} BLEU: {bleu_score:.4f}\n")

            return output

        def _compute_bleu_t5(self, nmt_dataset, max_samples=None):
            """Compute BLEU score using T5 model"""
            self.model.eval()

            references = []
            hypotheses = []

            dataset_size = len(nmt_dataset) if max_samples is None else min(max_samples, len(nmt_dataset))

            with torch.no_grad():
                for i in tqdm(range(dataset_size), desc="Computing BLEU"):
                    src, tgt = nmt_dataset[i]

                    # Convert source to text
                    src_text = ' '.join([self.src_vocab.itos[idx] for idx in src if idx not in [0, 1, 2]])
                    input_text = f"translate Chinese to English: {src_text}"

                    # Tokenize and generate
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        max_length=128,
                        truncation=True,
                    ).to(self.model.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                    )

                    # Decode prediction
                    pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Convert target to text
                    tgt_text = ' '.join([self.tgt_vocab.itos[idx] for idx in tgt if idx not in [0, 1, 2]])

                    hypotheses.append(pred_text.split())
                    references.append([tgt_text.split()])

            # Compute BLEU
            bleu_score = compute_corpus_bleu(references, hypotheses)
            return bleu_score

    # Initialize trainer
    trainer = T5TrainerWithBLEU(
        model=model,
        args=training_args,
        train_dataset=t5_train_dataset,
        eval_dataset=t5_valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        nmt_dataset=valid_dataset,
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        compute_bleu_flag=compute_bleu,
        compute_bleu_every_epoch_flag=compute_bleu_every_epoch,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)

    # Update trainer with test dataset for BLEU computation
    trainer.nmt_dataset = test_dataset
    test_results = trainer.evaluate(
        eval_dataset=t5_test_dataset,
        metric_key_prefix="test"
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_results['test_loss']:.4f}")
    print(f"  BLEU: {test_results.get('test_bleu', 0):.4f}")

    return model, test_results


if __name__ == '__main__':
    main()

