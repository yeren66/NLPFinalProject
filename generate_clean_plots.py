#!/usr/bin/env python3
"""
生成清晰的、分门别类的实验图表
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

def load_results():
    """加载实验结果"""
    with open('parsed_data/experiment_results.json', 'r') as f:
        return json.load(f)

def plot_rnn_attention(results, output_dir):
    """绘制RNN注意力机制对比图"""
    data = results['rnn_attention']
    
    attention_types = list(data.keys())
    bleu_scores = [data[k]['test_bleu_percent'] for k in attention_types]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(attention_types, bleu_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Attention Mechanism', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test BLEU Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('RNN: Attention Mechanism Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(bleu_scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rnn_attention_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  生成: rnn_attention_comparison.png")

def plot_rnn_teacher_forcing(results, output_dir):
    """绘制RNN Teacher Forcing对比图"""
    data = results['rnn_teacher_forcing']
    
    # 按TF比例排序
    tf_ratios = sorted([(k, v['teacher_forcing_ratio'], v['test_bleu_percent']) 
                        for k, v in data.items()], key=lambda x: x[1])
    
    labels = [f"TF={ratio:.1f}" for _, ratio, _ in tf_ratios]
    bleu_scores = [score for _, _, score in tf_ratios]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(labels, bleu_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, score in zip(bars, bleu_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Teacher Forcing Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test BLEU Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('RNN: Teacher Forcing Strategy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(bleu_scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rnn_teacher_forcing_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  生成: rnn_teacher_forcing_comparison.png")

def plot_transformer_positional(results, output_dir):
    """绘制Transformer位置编码对比图"""
    data = results['transformer_positional']
    
    pos_types = list(data.keys())
    bleu_scores = [data[k]['test_bleu_percent'] for k in pos_types]
    losses = [data[k]['test_loss'] for k in pos_types]
    
    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    x = range(len(pos_types))
    width = 0.35
    
    # BLEU分数
    bars1 = ax1.bar([i - width/2 for i in x], bleu_scores, width, 
                     label='BLEU Score', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Positional Encoding Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test BLEU Score (%)', fontsize=14, fontweight='bold', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.capitalize() for p in pos_types])
    
    # Loss
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], losses, width,
                     label='Test Loss', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Test Loss', fontsize=14, fontweight='bold', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # 添加数值标签
    for bar, score in zip(bars1, bleu_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#3498db')
    
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    
    ax1.set_title('Transformer: Positional Encoding Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transformer_positional_encoding.png', bbox_inches='tight')
    plt.close()
    print(f"  生成: transformer_positional_encoding.png")

def plot_transformer_normalization(results, output_dir):
    """绘制Transformer归一化对比图"""
    data = results['transformer_normalization']
    
    norm_types = list(data.keys())
    norm_labels = ['LayerNorm' if n == 'layer' else 'RMSNorm' for n in norm_types]
    bleu_scores = [data[k]['test_bleu_percent'] for k in norm_types]
    losses = [data[k]['test_loss'] for k in norm_types]
    
    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    x = range(len(norm_types))
    width = 0.35
    
    # BLEU分数
    bars1 = ax1.bar([i - width/2 for i in x], bleu_scores, width,
                     label='BLEU Score', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Normalization Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test BLEU Score (%)', fontsize=14, fontweight='bold', color='#2ecc71')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.set_xticks(x)
    ax1.set_xticklabels(norm_labels)
    
    # Loss
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], losses, width,
                     label='Test Loss', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Test Loss', fontsize=14, fontweight='bold', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # 添加数值标签
    for bar, score in zip(bars1, bleu_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2ecc71')
    
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    
    ax1.set_title('Transformer: Normalization Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transformer_normalization.png', bbox_inches='tight')
    plt.close()
    print(f"  生成: transformer_normalization.png")

def main():
    # 加载结果
    results = load_results()
    
    # 创建输出目录
    output_dir = Path('report_figures')
    output_dir.mkdir(exist_ok=True)
    
    print("生成实验对比图表...")
    
    # 生成各类图表
    plot_rnn_attention(results, output_dir)
    plot_rnn_teacher_forcing(results, output_dir)
    plot_transformer_positional(results, output_dir)
    plot_transformer_normalization(results, output_dir)
    
    print(f"\n所有图表已保存到: {output_dir}/")

if __name__ == '__main__':
    main()

