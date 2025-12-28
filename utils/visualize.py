"""
Visualization utilities for experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(history, title, save_path):
    """Plot training and validation loss/perplexity curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Perplexity curves
    ax2.plot(epochs, history['train_ppl'], 'b-', label='Training PPL', linewidth=2)
    ax2.plot(epochs, history['valid_ppl'], 'r-', label='Validation PPL', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title(f'{title} - Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_attention_comparison(results_file, save_path):
    """Plot comparison of different attention mechanisms."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    attention_types = list(results.keys())
    best_losses = [results[attn]['best_valid_loss'] for attn in attention_types]
    best_ppls = [results[attn]['best_valid_ppl'] for attn in attention_types]
    avg_times = [results[attn]['avg_epoch_time'] for attn in attention_types]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Best validation loss
    bars1 = ax1.bar(attention_types, best_losses, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Best Validation Loss by Attention Type', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Best validation perplexity
    bars2 = ax2.bar(attention_types, best_ppls, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Best Validation Perplexity by Attention Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Average epoch time
    bars3 = ax3.bar(attention_types, avg_times, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Average Epoch Time by Attention Type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_training_strategy_comparison(results_file, save_path):
    """Plot comparison of different training strategies."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    strategies = list(results.keys())
    best_losses = [results[s]['best_valid_loss'] for s in strategies]
    best_ppls = [results[s]['best_valid_ppl'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Best validation loss
    bars1 = ax1.bar(strategies, best_losses, color=colors[:len(strategies)])
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Best Validation Loss by Training Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Best validation perplexity
    bars2 = ax2.bar(strategies, best_ppls, color=colors[:len(strategies)])
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Best Validation Perplexity by Training Strategy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_decoding_comparison(results_file, save_path):
    """Plot comparison of different decoding strategies."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    strategies = list(results.keys())
    bleu_scores = [results[s]['bleu_score'] * 100 for s in strategies]  # Convert to percentage
    avg_times = [results[s]['avg_time_per_sentence'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # BLEU scores
    bars1 = ax1.bar(strategies, bleu_scores, color=colors[:len(strategies)])
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('BLEU Score by Decoding Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Average decoding time
    bars2 = ax2.bar(strategies, avg_times, color=colors[:len(strategies)])
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Avg Decoding Time per Sentence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def generate_all_plots():
    """Generate all visualization plots."""
    os.makedirs('figures', exist_ok=True)
    
    print("Generating visualization plots...")
    
    # Attention mechanisms comparison
    if os.path.exists('results/attention_mechanisms_comparison.json'):
        plot_attention_comparison(
            'results/attention_mechanisms_comparison.json',
            'figures/attention_comparison.png'
        )
    
    # Training strategies comparison
    if os.path.exists('results/training_strategies_comparison.json'):
        plot_training_strategy_comparison(
            'results/training_strategies_comparison.json',
            'figures/training_strategies_comparison.png'
        )
    
    # Decoding strategies comparison
    if os.path.exists('results/decoding_strategies_comparison.json'):
        plot_decoding_comparison(
            'results/decoding_strategies_comparison.json',
            'figures/decoding_comparison.png'
        )
    
    print("\nAll plots generated successfully!")
    print("Plots saved in the 'figures/' directory")


if __name__ == "__main__":
    generate_all_plots()

