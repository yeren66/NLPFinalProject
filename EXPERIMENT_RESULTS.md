# Neural Machine Translation Experiments - Results Summary

## Overview
This document summarizes the results of ablation studies on RNN and Transformer models for English-to-German translation on the Multi30k dataset (100k training samples).

---

## 1. RNN Experiments

### 1.1 Attention Mechanism Comparison

**Objective**: Compare different attention mechanisms in RNN-based seq2seq models.

| Attention Type | Test BLEU (%) | Observations |
|----------------|---------------|--------------|
| **Additive** | **2.13** | Best performance among attention mechanisms |
| Dot Product | 2.06 | Slightly lower than additive |
| Multiplicative | 1.92 | Lowest performance |

**Key Findings**:
- Additive attention (Bahdanau) performs best with 2.13% BLEU
- All attention mechanisms show relatively similar performance
- The differences are small (~0.2%), suggesting attention type has modest impact on this dataset

**Visualization**: See `report_figures/rnn_attention_comparison.png`

---

### 1.2 Teacher Forcing Strategy Comparison

**Objective**: Evaluate the impact of teacher forcing ratio on model performance.

| TF Ratio | Test BLEU (%) | Observations |
|----------|---------------|--------------|
| **1.0** (Always) | **2.59** | Best performance - full teacher forcing |
| 0.5 (Mixed) | 2.22 | Moderate performance |
| 0.0 (Never) | 1.44 | Poorest performance |

**Key Findings**:
- Full teacher forcing (TF=1.0) achieves the best BLEU score of 2.59%
- Performance degrades significantly without teacher forcing (1.44%)
- Mixed strategy (TF=0.5) provides intermediate results
- Teacher forcing is crucial for RNN training on this task

**Visualization**: See `report_figures/rnn_teacher_forcing_comparison.png`

---

## 2. Transformer Experiments

### 2.1 Positional Encoding Comparison

**Objective**: Compare absolute vs. relative positional encoding in Transformers.

| Encoding Type | Test BLEU (%) | Test Loss | Observations |
|---------------|---------------|-----------|--------------|
| **Relative** | **7.80** | 4.513 | Slightly better BLEU |
| Absolute | 7.68 | 4.465 | Lower loss but slightly lower BLEU |

**Key Findings**:
- Relative positional encoding achieves marginally better BLEU (7.80% vs 7.68%)
- Absolute encoding has slightly lower test loss (4.465 vs 4.513)
- The differences are minimal (~0.12% BLEU), suggesting both work well
- Transformers significantly outperform RNNs (7.7% vs 2.6% BLEU)

**Visualization**: See `report_figures/transformer_positional_encoding.png`

---

### 2.2 Normalization Comparison

**Objective**: Compare LayerNorm vs. RMSNorm in Transformer architecture.

| Normalization | Test BLEU (%) | Test Loss | Observations |
|---------------|---------------|-----------|--------------|
| **LayerNorm** | **8.37** | 4.435 | Best overall performance |
| RMSNorm | 7.90 | 4.439 | Slightly lower performance |

**Key Findings**:
- LayerNorm achieves the best BLEU score of 8.37%
- RMSNorm performs slightly worse (7.90% BLEU)
- Test losses are very similar (4.435 vs 4.439)
- LayerNorm remains the preferred choice for this task

**Visualization**: See `report_figures/transformer_normalization.png`

---

## 3. Overall Comparison

### Best Configurations

| Model Type | Configuration | Test BLEU (%) |
|------------|---------------|---------------|
| **Transformer** | LayerNorm | **8.37** |
| Transformer | RMSNorm | 7.90 |
| Transformer | Relative PE | 7.80 |
| Transformer | Absolute PE | 7.68 |
| RNN | TF=1.0 | 2.59 |
| RNN | Additive Attention | 2.13 |
| RNN | TF=0.5 | 2.22 |
| RNN | Dot Attention | 2.06 |
| RNN | Multiplicative Attention | 1.92 |
| RNN | TF=0.0 | 1.44 |

### Key Insights

1. **Transformer vs RNN**: Transformers significantly outperform RNNs (8.37% vs 2.59% BLEU, ~3.2x improvement)

2. **RNN Findings**:
   - Teacher forcing is critical (TF=1.0 gives 80% improvement over TF=0.0)
   - Attention mechanism choice has modest impact (~10% variation)
   - Additive attention performs best among attention types

3. **Transformer Findings**:
   - LayerNorm is the best normalization choice
   - Positional encoding type (absolute vs relative) has minimal impact
   - All Transformer variants significantly outperform all RNN variants

4. **Best Overall Model**: Transformer with LayerNorm (8.37% BLEU)

---

## 4. Experimental Setup

- **Dataset**: Multi30k (English → German)
- **Training Samples**: 100,000
- **Validation Samples**: ~1,000
- **Test Samples**: ~1,000
- **Metric**: BLEU Score
- **Training**: 20 epochs for all models
- **Framework**: PyTorch Lightning

---

## 5. Visualizations

All comparison charts are available in the `report_figures/` directory:

1. `rnn_attention_comparison.png` - RNN attention mechanisms
2. `rnn_teacher_forcing_comparison.png` - Teacher forcing strategies
3. `transformer_positional_encoding.png` - Positional encoding types
4. `transformer_normalization.png` - Normalization methods

---

## 6. Conclusion

This ablation study demonstrates that:
- **Transformers are superior** to RNNs for machine translation
- **Teacher forcing is essential** for RNN training
- **LayerNorm** remains the best normalization choice
- **Attention mechanism type** has modest impact on RNN performance
- **Positional encoding type** has minimal impact on Transformer performance

The best model configuration is a **Transformer with LayerNorm**, achieving **8.37% BLEU** on the test set.

