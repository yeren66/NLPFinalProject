#!/bin/bash

# Transformer Experiments using HuggingFace Transformers
# This script runs various Transformer experiments with different configurations

set -e  # Exit on error

# Use only GPU 0 to avoid device conflicts
export CUDA_VISIBLE_DEVICES=1

echo "=============================================="
echo "Transformer Experiments with HuggingFace"
echo "=============================================="
echo ""
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Configuration
N_EPOCHS=50
BATCH_SIZE=256
LOG_DIR="logs"

# Create timestamp for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ROOT="experiments/100k_transformer_${TIMESTAMP}"

echo "Configuration:"
echo "  Epochs: $N_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Experiment Root: $EXPERIMENT_ROOT"
echo "  Log Dir: $LOG_DIR"
echo ""

# Create experiment directory structure
mkdir -p "$EXPERIMENT_ROOT"
mkdir -p "$LOG_DIR"

# Create experiment config file
cat > "$EXPERIMENT_ROOT/config.json" << EOF
{
  "model_type": "Transformer",
  "timestamp": "$TIMESTAMP",
  "n_epochs": $N_EPOCHS,
  "batch_size": $BATCH_SIZE,
  "experiments": ["baseline", "positional_encoding", "normalization", "t5_finetune"]
}
EOF

# Create main log file
MAIN_LOG="$LOG_DIR/transformer_experiments_${TIMESTAMP}.log"

echo "Experiment directory: $EXPERIMENT_ROOT"
echo "Logging to: $MAIN_LOG"
echo ""

# Experiment 1: Baseline
echo "========================================"
echo "Experiment 1: Baseline Training"
echo "========================================"

EXP_NAME="baseline"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

python train_transformer.py \
    --experiment_type baseline \
    --pos_enc_type absolute \
    --norm_type layer \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    2>&1 | tee "$LOG_DIR/transformer_baseline_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "checkpoints/transformer_baseline_test_results.json" ]; then
    mv "checkpoints/transformer_baseline_test_results.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Baseline training completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Experiment 2: Positional Encoding Comparison
echo "========================================"
echo "Experiment 2: Positional Encoding Types"
echo "========================================"

EXP_NAME="positional_encoding"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

python train_transformer.py \
    --experiment_type positional_encoding \
    --pos_enc_types absolute relative \
    --norm_type layer \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    2>&1 | tee "$LOG_DIR/transformer_positional_encoding_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "results/transformer_positional_encoding_comparison_hf.json" ]; then
    mv "results/transformer_positional_encoding_comparison_hf.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Positional encoding comparison completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Experiment 3: Normalization Comparison
echo "========================================"
echo "Experiment 3: Normalization Types"
echo "========================================"

EXP_NAME="normalization"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

python train_transformer.py \
    --experiment_type normalization \
    --norm_types layer rms \
    --pos_enc_type absolute \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    2>&1 | tee "$LOG_DIR/transformer_normalization_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "results/transformer_normalization_comparison_hf.json" ]; then
    mv "results/transformer_normalization_comparison_hf.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Normalization comparison completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Experiment 4: T5 Fine-tuning
echo "========================================"
echo "Experiment 4: T5 Pretrained Model Fine-tuning"
echo "========================================"

EXP_NAME="t5_finetune"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

echo "Fine-tuning pretrained T5 model..."
echo "Model: google-t5/t5-base (using local model if available at T5_model/)"
echo "Note: Using smaller batch size for T5 due to larger model size"
echo ""

python train_transformer.py \
    --experiment_type t5_finetune \
    --t5_model google-t5/t5-base \
    --max_length 128 \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    2>&1 | tee "$LOG_DIR/transformer_t5_finetune_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "results/transformer_t5_finetune_hf.json" ]; then
    mv "results/transformer_t5_finetune_hf.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ T5 fine-tuning completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Create experiment summary
echo "========================================"
echo "Creating Experiment Summary"
echo "========================================"

SUMMARY_FILE="$EXPERIMENT_ROOT/summary.json"
cat > "$SUMMARY_FILE" << EOF
{
  "experiment_type": "Transformer",
  "timestamp": "$TIMESTAMP",
  "config": {
    "n_epochs": $N_EPOCHS,
    "batch_size": $BATCH_SIZE
  },
  "experiments": {
    "baseline": "$EXPERIMENT_ROOT/baseline/results.json",
    "positional_encoding": "$EXPERIMENT_ROOT/positional_encoding/results.json",
    "normalization": "$EXPERIMENT_ROOT/normalization/results.json",
    "t5_finetune": "$EXPERIMENT_ROOT/t5_finetune/results.json"
  },
  "logs": {
    "main": "$MAIN_LOG",
    "baseline": "$LOG_DIR/transformer_baseline_${TIMESTAMP}.log",
    "positional_encoding": "$LOG_DIR/transformer_positional_encoding_${TIMESTAMP}.log",
    "normalization": "$LOG_DIR/transformer_normalization_${TIMESTAMP}.log",
    "t5_finetune": "$LOG_DIR/transformer_t5_finetune_${TIMESTAMP}.log"
  }
}
EOF

# Create symlink to latest experiment
rm -f experiments/latest_transformer
ln -sf "transformer_${TIMESTAMP}" experiments/latest_transformer

# Summary
echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo ""
echo "Experiment directory: $EXPERIMENT_ROOT"
echo ""
echo "Structure:"
echo "  $EXPERIMENT_ROOT/"
echo "  ├── config.json              # Experiment configuration"
echo "  ├── summary.json             # Experiment summary"
echo "  ├── baseline/"
echo "  │   ├── checkpoints/         # Model checkpoints"
echo "  │   └── results.json         # Test results"
echo "  ├── positional_encoding/"
echo "  │   ├── checkpoints/"
echo "  │   └── results.json"
echo "  ├── normalization/"
echo "  │   ├── checkpoints/"
echo "  │   └── results.json"
echo "  └── t5_finetune/"
echo "      ├── checkpoints/"
echo "      └── results.json"
echo ""
echo "Quick access:"
echo "  Latest experiment: experiments/latest_transformer -> $EXPERIMENT_ROOT"
echo ""
echo "Logs:"
echo "  - Main log: $MAIN_LOG"
echo "  - Individual logs: $LOG_DIR/transformer_*_${TIMESTAMP}.log"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=runs"
echo ""
echo "To view results:"
echo "  cat $EXPERIMENT_ROOT/*/results.json | jq ."
echo ""

