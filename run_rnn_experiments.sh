#!/bin/bash

# RNN Experiments using PyTorch Lightning
# This script runs various RNN experiments with different configurations

set -e  # Exit on error

# Use only GPU 0 to avoid device conflicts
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "RNN Experiments with PyTorch Lightning"
echo "========================================"
echo ""
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Configuration
N_EPOCHS=50
BATCH_SIZE=256
LOG_DIR="logs"

# Create timestamp for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ROOT="experiments/100k_rnn_${TIMESTAMP}"

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
  "model_type": "RNN",
  "timestamp": "$TIMESTAMP",
  "n_epochs": $N_EPOCHS,
  "batch_size": $BATCH_SIZE,
  "experiments": ["baseline", "attention_mechanisms", "teacher_forcing"]
}
EOF

# Create main log file
MAIN_LOG="$LOG_DIR/rnn_experiments_${TIMESTAMP}.log"

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

python train_rnn.py \
    --experiment_type baseline \
    --attention_type dot \
    --teacher_forcing_ratio 0.5 \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    $NO_BLEU 2>&1 | tee "$LOG_DIR/rnn_baseline_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "checkpoints/rnn_baseline_test_results.json" ]; then
    mv "checkpoints/rnn_baseline_test_results.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Baseline training completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Experiment 2: Attention Mechanisms Comparison
echo "========================================"
echo "Experiment 2: Attention Mechanisms"
echo "========================================"

EXP_NAME="attention_mechanisms"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

python train_rnn.py \
    --experiment_type attention \
    --attention_types dot multiplicative additive \
    --teacher_forcing_ratio 0.5 \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    $NO_BLEU 2>&1 | tee "$LOG_DIR/rnn_attention_mechanisms_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "results/attention_mechanisms_comparison_lightning.json" ]; then
    mv "results/attention_mechanisms_comparison_lightning.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Attention mechanisms comparison completed"
echo "  Results: $EXP_DIR/results.json"
echo "  Checkpoints: $EXP_DIR/checkpoints/"
echo ""

# Experiment 3: Teacher Forcing Strategies
echo "========================================"
echo "Experiment 3: Teacher Forcing Strategies"
echo "========================================"

EXP_NAME="teacher_forcing"
EXP_DIR="$EXPERIMENT_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR/checkpoints"

python train_rnn.py \
    --experiment_type training_strategy \
    --attention_type dot \
    --teacher_forcing_ratios 1.0 0.5 0.0 \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$EXP_DIR/checkpoints" \
    $NO_BLEU 2>&1 | tee "$LOG_DIR/rnn_teacher_forcing_${TIMESTAMP}.log" | tee -a "$MAIN_LOG"

# Move results to experiment directory
if [ -f "results/training_strategies_comparison_lightning.json" ]; then
    mv "results/training_strategies_comparison_lightning.json" "$EXP_DIR/results.json"
fi

echo ""
echo "✓ Teacher forcing strategies comparison completed"
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
  "experiment_type": "RNN",
  "timestamp": "$TIMESTAMP",
  "config": {
    "n_epochs": $N_EPOCHS,
    "batch_size": $BATCH_SIZE
  },
  "experiments": {
    "baseline": "$EXPERIMENT_ROOT/baseline/results.json",
    "attention_mechanisms": "$EXPERIMENT_ROOT/attention_mechanisms/results.json",
    "teacher_forcing": "$EXPERIMENT_ROOT/teacher_forcing/results.json"
  },
  "logs": {
    "main": "$MAIN_LOG",
    "baseline": "$LOG_DIR/rnn_baseline_${TIMESTAMP}.log",
    "attention": "$LOG_DIR/rnn_attention_mechanisms_${TIMESTAMP}.log",
    "teacher_forcing": "$LOG_DIR/rnn_teacher_forcing_${TIMESTAMP}.log"
  }
}
EOF

# Create symlink to latest experiment
rm -f experiments/latest_rnn
ln -sf "rnn_${TIMESTAMP}" experiments/latest_rnn

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
echo "  ├── attention_mechanisms/"
echo "  │   ├── checkpoints/"
echo "  │   └── results.json"
echo "  └── teacher_forcing/"
echo "      ├── checkpoints/"
echo "      └── results.json"
echo ""
echo "Quick access:"
echo "  Latest experiment: experiments/latest_rnn -> $EXPERIMENT_ROOT"
echo ""
echo "Logs:"
echo "  - Main log: $MAIN_LOG"
echo "  - Individual logs: $LOG_DIR/rnn_*_${TIMESTAMP}.log"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=runs"
echo ""
echo "To view results:"
echo "  cat $EXPERIMENT_ROOT/*/results.json | jq ."
echo ""

