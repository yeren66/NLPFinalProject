## RNN-based NMT: Build and train a RNN-based NMT, including:
- Model: Implement a model using either GRU or LSTM, with both the encoder and decoder consisting of two unidirectional layers.
- Attention mechanism: Implement the attention mechanism and investigate the impact of different alignment functions—such as dot-product, multiplicative, and additive—on model performance.
- Training policy: Compare the effectiveness of Teacher Forcing and Free Running strategies.
- Decoding policy: Compare the effectiveness of  greedy and beam-search decoding strategies.

## Transformer-based NMT: Build and train a Transformer-based NMT, including:
- From scratch: Build a Chinese-to-English translation model using the Transformer architecture with an encoder-decoder structure and train it from scratch.
- Architectural Ablation: Train from scratch and compare the effects of different position embedding schemes (e.g., absolute vs. relative) and normalization methods (e.g., LayerNorm vs. RMSNorm).
- Hyperparameter Sensitivity: Train from scratch with varying batch sizes, learning rates, and model scales to assess their impact on translation performance.
- From pretrained language model: Fine-tune a pretrained language model (e.g., T5) to adapt it for neural machine translation and evaluate its performance in comparison with models trained from scratch.

## Analysis and Comparison: Conduct a comprehensive comparison between the RNN-based and Transformer-based NMT models in terms of: 
- Model architecture (e.g., sequential vs. parallel computation, recurrence vs. self-attention),
- Training efficiency (e.g., training time, convergence speed, hardware requirements),
- Translation performance (e.g., BLEU score, fluency, adequacy),
- Scalability and generalization (e.g., handling long sentences, low-resource scenarios),
- Practical trade-offs (e.g., model size, inference latency, ease of implementation).