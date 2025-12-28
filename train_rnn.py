"""
RNN Training Script using PyTorch Lightning
Refactored from train_rnn.py for better modularity and maintainability
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import argparse
import os
import json
import math
from datetime import datetime
from tqdm import tqdm

# Set Tensor Cores precision for H100 GPU optimization
# This trades off precision for performance on GPUs with Tensor Cores
# Options: 'highest' (default), 'high', 'medium'
# 'high' or 'medium' provides significant speedup with minimal accuracy loss
torch.set_float32_matmul_precision('high')

from config import Config
from utils.data_loader import NMTDataset, collate_fn
from models.rnn.encoder import Encoder
from models.rnn.decoder import Decoder
from models.rnn.seq2seq import Seq2Seq
from models.rnn.attention import Attention
from utils.metrics import compute_corpus_bleu
from utils.decode import greedy_decode


class RNNSeq2SeqModule(pl.LightningModule):
    """PyTorch Lightning Module for RNN Seq2Seq with Attention"""
    
    def __init__(
        self,
        input_dim,
        output_dim,
        src_vocab,
        tgt_vocab,
        attention_type='dot',
        teacher_forcing_ratio=0.5,
        learning_rate=0.0005,
        enc_emb_dim=256,
        dec_emb_dim=256,
        enc_hid_dim=512,
        dec_hid_dim=512,
        enc_layers=2,
        dec_layers=2,
        dropout=0.5,
        clip=1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['src_vocab', 'tgt_vocab'])
        
        # Store vocabularies
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Build model
        attn = Attention(enc_hid_dim * 2, dec_hid_dim, method=attention_type)
        enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers=enc_layers)
        dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn, n_layers=dec_layers)
        # Note: device will be set properly by Lightning when model is moved to GPU
        # We use a placeholder here and update it in setup()
        self.model = Seq2Seq(enc, dec, torch.device('cpu'))

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is PAD
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def setup(self, stage=None):
        """Called when the model is moved to GPU"""
        # Update the device in the Seq2Seq model
        self.model.device = self.device

    def forward(self, src, tgt, teacher_forcing_ratio=None, src_len=None):
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.hparams.teacher_forcing_ratio

        # Ensure inputs are on the correct device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_len is not None:
            src_len = src_len.to(self.device)

        return self.model(src, tgt, teacher_forcing_ratio, src_len=src_len)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch (now includes lengths)
        src, tgt, src_len, tgt_len = batch

        # Forward pass with teacher forcing (device handling is in forward())
        # Pass src_len for pack_padded_sequence optimization
        output = self(src, tgt, self.hparams.teacher_forcing_ratio, src_len=src_len)

        # Calculate loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        # Ensure tgt is on the same device as output
        tgt = tgt.to(output.device)[:, 1:].reshape(-1)

        loss = self.criterion(output, tgt)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/ppl', torch.exp(torch.clamp(loss, max=100)), on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack batch (now includes lengths)
        src, tgt, src_len, tgt_len = batch

        # Forward pass without teacher forcing (device handling is in forward())
        # Pass src_len for pack_padded_sequence optimization
        output = self(src, tgt, teacher_forcing_ratio=0.0, src_len=src_len)

        # Calculate loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        # Ensure tgt is on the same device as output
        tgt = tgt.to(output.device)[:, 1:].reshape(-1)

        loss = self.criterion(output, tgt)

        # Store for epoch end
        self.validation_step_outputs.append(loss)

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return
            
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        ppl = torch.exp(torch.clamp(avg_loss, max=100))
        
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/ppl', ppl, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        # Unpack batch (now includes lengths)
        src, tgt, src_len, tgt_len = batch

        # Forward pass (device handling is in forward())
        # Pass src_len for pack_padded_sequence optimization
        output = self(src, tgt, teacher_forcing_ratio=0.0, src_len=src_len)

        # Calculate loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        # Ensure tgt is on the same device as output
        tgt = tgt.to(output.device)[:, 1:].reshape(-1)

        loss = self.criterion(output, tgt)

        self.test_step_outputs.append(loss)

        return loss

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        avg_loss = torch.stack(self.test_step_outputs).mean()
        ppl = torch.exp(torch.clamp(avg_loss, max=100))

        self.log('test/loss', avg_loss)
        self.log('test/ppl', ppl)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip)


class NMTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for NMT Dataset"""

    def __init__(
        self,
        train_file,
        valid_file,
        test_file,
        batch_size=256,
        num_workers=4,
    ):
        super().__init__()
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.src_vocab = None
        self.tgt_vocab = None

    def setup(self, stage=None):
        """Load data. Lightning calls this before training/validation/testing."""
        if stage == 'fit' or stage is None:
            # Check if data is already loaded (avoid duplicate loading)
            if self.train_dataset is None:
                self.train_dataset = NMTDataset(self.train_file, build_vocab=True)
                self.valid_dataset = NMTDataset(
                    self.valid_file,
                    src_vocab=self.train_dataset.src_vocab,
                    tgt_vocab=self.train_dataset.tgt_vocab
                )
                self.src_vocab = self.train_dataset.src_vocab
                self.tgt_vocab = self.train_dataset.tgt_vocab

        if stage == 'test' or stage is None:
            if self.train_dataset is None:
                self.train_dataset = NMTDataset(self.train_file, build_vocab=True)
                self.src_vocab = self.train_dataset.src_vocab
                self.tgt_vocab = self.train_dataset.tgt_vocab

            # Check if test data is already loaded
            if self.test_dataset is None:
                self.test_dataset = NMTDataset(
                    self.test_file,
                    src_vocab=self.src_vocab,
                    tgt_vocab=self.tgt_vocab
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def compute_bleu_score(model, dataset, src_vocab, tgt_vocab, max_samples=None):
    """Compute BLEU score on a dataset"""
    model.eval()
    device = next(model.parameters()).device

    references = []
    hypotheses = []

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    with torch.no_grad():
        pbar = tqdm(range(num_samples), desc='Computing BLEU', leave=False)
        for idx in pbar:
            src_tokens, tgt_tokens = dataset[idx]
            src = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

            # Greedy decode
            output_indices = greedy_decode(model.model, src, tgt_vocab, max_len=Config.MAX_LEN)

            # Convert to tokens
            pad_idx = tgt_vocab.stoi.get('<PAD>', tgt_vocab.stoi.get('<pad>', 0))
            sos_idx = tgt_vocab.stoi.get('<SOS>', tgt_vocab.stoi.get('<sos>', 1))
            eos_idx = tgt_vocab.stoi.get('<EOS>', tgt_vocab.stoi.get('<eos>', 2))

            ref_tokens = [tgt_vocab.itos[idx] for idx in tgt_tokens.tolist()
                         if idx not in [pad_idx, sos_idx, eos_idx]]
            hyp_tokens = [tgt_vocab.itos[idx] for idx in output_indices
                         if idx not in [pad_idx, sos_idx, eos_idx]]

            references.append(ref_tokens)
            hypotheses.append(hyp_tokens)

    return compute_corpus_bleu(references, hypotheses)


class BLEUCallback(pl.Callback):
    """Callback to compute BLEU score during validation"""

    def __init__(self, datamodule, compute_every_n_epochs=1, max_samples=1000):
        super().__init__()
        self.datamodule = datamodule
        self.compute_every_n_epochs = compute_every_n_epochs
        self.max_samples = max_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.compute_every_n_epochs != 0:
            return

        print(f"\n[Epoch {trainer.current_epoch + 1}] Computing BLEU on validation set...")
        bleu_score = compute_bleu_score(
            pl_module,
            self.datamodule.valid_dataset,
            self.datamodule.src_vocab,
            self.datamodule.tgt_vocab,
            max_samples=self.max_samples
        )

        pl_module.log('val/bleu', bleu_score, prog_bar=True)
        print(f"[Epoch {trainer.current_epoch + 1}] Validation BLEU: {bleu_score:.4f}")


def train_model(
    attention_type='dot',
    teacher_forcing_ratio=0.5,
    n_epochs=10,
    batch_size=256,
    learning_rate=0.0005,
    output_dir='checkpoints',
    experiment_name='rnn_baseline',
    resume_from=None,
    compute_bleu_every_epoch=True,
):
    """Train a single RNN model with specified configuration"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup data module
    datamodule = NMTDataModule(
        train_file=Config.TRAIN_FILE,
        valid_file=Config.VALID_FILE,
        test_file=Config.TEST_FILE,
        batch_size=batch_size,
        num_workers=4,
    )

    # Setup data to get vocab sizes
    datamodule.setup('fit')

    input_dim = len(datamodule.src_vocab)
    output_dim = len(datamodule.tgt_vocab)

    print(f"\n{'='*70}")
    print(f"Dataset Information:")
    print(f"  Source Vocab Size: {input_dim}")
    print(f"  Target Vocab Size: {output_dim}")
    print(f"  Train Samples: {len(datamodule.train_dataset)}")
    print(f"  Valid Samples: {len(datamodule.valid_dataset)}")
    print(f"  Batch Size: {batch_size}")
    print(f"{'='*70}\n")

    # Initialize model
    model = RNNSeq2SeqModule(
        input_dim=input_dim,
        output_dim=output_dim,
        src_vocab=datamodule.src_vocab,
        tgt_vocab=datamodule.tgt_vocab,
        attention_type=attention_type,
        teacher_forcing_ratio=teacher_forcing_ratio,
        learning_rate=learning_rate,
        enc_emb_dim=Config.RNN_EMB_DIM,
        dec_emb_dim=Config.RNN_EMB_DIM,
        enc_hid_dim=Config.RNN_HID_DIM,
        dec_hid_dim=Config.RNN_HID_DIM,
        enc_layers=Config.RNN_ENC_LAYERS,
        dec_layers=Config.RNN_DEC_LAYERS,
        dropout=Config.RNN_DROPOUT,
        clip=Config.CLIP,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best',  # Only save best model with simple name
        monitor='val/loss',
        mode='min',
        save_top_k=1,  # Only keep the best model
        save_last=True,  # Also save the last model
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, lr_monitor]

    if compute_bleu_every_epoch:
        bleu_callback = BLEUCallback(datamodule, compute_every_n_epochs=1, max_samples=1000)
        callbacks.append(bleu_callback)

    # Setup logger
    dataset_name = os.path.splitext(os.path.basename(Config.TRAIN_FILE))[0]
    logger = TensorBoardLogger(
        save_dir='runs',
        name=f'{dataset_name}_{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=Config.CLIP,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Attention Type: {attention_type}")
    print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Experiment Name: {experiment_name}")
    print(f"{'='*70}\n")

    # Train
    if resume_from:
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Test
    print(f"\n{'='*70}")
    print(f"Evaluating on Test Set")
    print(f"{'='*70}\n")

    datamodule.setup('test')
    trainer.test(model, datamodule=datamodule, ckpt_path='best')

    # Compute test BLEU
    print("Computing test BLEU score...")
    test_bleu = compute_bleu_score(
        model,
        datamodule.test_dataset,
        datamodule.src_vocab,
        datamodule.tgt_vocab,
        max_samples=None
    )

    print(f"\n{'='*70}")
    print(f"Test BLEU Score: {test_bleu:.4f}")
    print(f"{'='*70}\n")

    # Save test results
    test_results = {
        'test_bleu': test_bleu,
        'attention_type': attention_type,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'best_model_path': checkpoint_callback.best_model_path,
    }

    results_path = os.path.join(output_dir, f'{experiment_name}_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"Test results saved to: {results_path}")

    return model, test_results


def main():
    parser = argparse.ArgumentParser(description='RNN Training Script with PyTorch Lightning')

    # Experiment type
    parser.add_argument('--experiment_type', type=str, default='baseline',
                       choices=['baseline', 'attention', 'training_strategy'],
                       help='Type of experiment to run')

    # Attention mechanism parameters
    parser.add_argument('--attention_types', type=str, nargs='+',
                       default=None,
                       help='Attention types to compare (for attention experiment)')
    parser.add_argument('--attention_type', type=str, default='dot',
                       help='Single attention type (for baseline/training_strategy)')

    # Training strategy parameters
    parser.add_argument('--teacher_forcing_ratios', type=float, nargs='+',
                       default=None,
                       help='Teacher forcing ratios to compare (for training_strategy experiment)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                       help='Single teacher forcing ratio (for baseline/attention)')

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
        print("Running BASELINE experiment with PyTorch Lightning")
        print("="*70)

        model, test_results = train_model(
            attention_type=args.attention_type,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            experiment_name=f'rnn_{dataset_name}_baseline',
            resume_from=args.resume,
            compute_bleu_every_epoch=compute_bleu,
        )

        print(f'\nTest BLEU: {test_results["test_bleu"]:.4f}')

    elif args.experiment_type == 'attention':
        print("\n" + "="*70)
        print("Running ATTENTION MECHANISM COMPARISON experiment")
        print("="*70)

        attention_types = args.attention_types or ['dot', 'multiplicative', 'additive']
        all_results = {}

        for attn_type in attention_types:
            print(f"\n{'='*70}")
            print(f"Training with attention type: {attn_type}")
            print(f"{'='*70}")

            model, test_results = train_model(
                attention_type=attn_type,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                experiment_name=f'rnn_{dataset_name}_ablation_{attn_type}',
                resume_from=args.resume,
                compute_bleu_every_epoch=compute_bleu,
            )

            all_results[attn_type] = test_results

        # Print comparison
        print("\n" + "="*70)
        print("ATTENTION MECHANISM COMPARISON RESULTS")
        print("="*70)
        for attn_type, results in all_results.items():
            print(f"{attn_type:15s}: Test BLEU = {results['test_bleu']:.4f}")

        # Save results
        with open('results/attention_mechanisms_comparison_lightning.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    elif args.experiment_type == 'training_strategy':
        print("\n" + "="*70)
        print("Running TRAINING STRATEGY COMPARISON experiment")
        print("="*70)

        teacher_forcing_ratios = args.teacher_forcing_ratios or [1.0, 0.5, 0.0]
        all_results = {}

        for tf_ratio in teacher_forcing_ratios:
            print(f"\n{'='*70}")
            print(f"Training with teacher forcing ratio: {tf_ratio}")
            print(f"{'='*70}")

            model, test_results = train_model(
                attention_type=args.attention_type,
                teacher_forcing_ratio=tf_ratio,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                experiment_name=f'rnn_{dataset_name}_tf_{tf_ratio}',
                resume_from=args.resume,
                compute_bleu_every_epoch=compute_bleu,
            )

            all_results[f'tf_{tf_ratio}'] = test_results

        # Print comparison
        print("\n" + "="*70)
        print("TRAINING STRATEGY COMPARISON RESULTS")
        print("="*70)
        for key, results in all_results.items():
            print(f"{key:15s}: Test BLEU = {results['test_bleu']:.4f}")

        # Save results
        with open('results/training_strategies_comparison_lightning.json', 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()

