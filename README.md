# NLP Final Project - 中英翻译系统

基于RNN和Transformer的神经机器翻译系统，使用PyTorch Lightning和HuggingFace Transformers框架实现。

## 📁 项目结构

```
NLPFinalProject/
├── train_rnn.py                    # RNN训练脚本 (PyTorch Lightning)
├── train_transformer.py            # Transformer训练脚本 (HuggingFace)
├── inference.py                    # 推理脚本
├── analyze_results.py              # 结果分析和可视化脚本
├── config.py                       # 配置文件
│
├── run_rnn_experiments.sh          # RNN批量实验脚本
├── run_transformer_experiments.sh  # Transformer批量实验脚本
├── view_experiments.sh             # 🆕 查看实验结果脚本
├── start_tensorboard.sh            # 启动TensorBoard
│
├── models/                         # 模型定义
│   ├── rnn/                        # RNN模型
│   └── transformer/                # Transformer模型
│
├── utils/                          # 工具函数
│   ├── data_loader.py              # 数据加载
│   ├── metrics.py                  # 评估指标
│   ├── decode.py                   # 解码策略
│   └── visualize.py                # 可视化工具
│
├── experiments/                    # 🆕 有组织的实验结果目录
│   ├── rnn_YYYYMMDD_HHMMSS/       # RNN实验批次（按时间戳）
│   │   ├── config.json            # 实验配置
│   │   ├── summary.json           # 实验总结
│   │   ├── baseline/              # 基线实验
│   │   │   ├── checkpoints/       # 模型检查点（best.ckpt, last.ckpt）
│   │   │   └── results.json       # 测试结果
│   │   ├── attention_mechanisms/  # 注意力机制对比
│   │   └── teacher_forcing/       # 训练策略对比
│   ├── transformer_YYYYMMDD_HHMMSS/  # Transformer实验批次
│   │   ├── config.json
│   │   ├── summary.json
│   │   ├── baseline/
│   │   ├── positional_encoding/
│   │   └── normalization/
│   ├── latest_rnn -> rnn_YYYYMMDD_HHMMSS/        # 指向最新RNN实验的软链接
│   └── latest_transformer -> transformer_YYYYMMDD_HHMMSS/  # 指向最新Transformer实验的软链接
│
├── logs/                           # 🆕 执行日志（按时间戳分类）
│   ├── rnn_experiments_YYYYMMDD_HHMMSS.log
│   ├── rnn_baseline_YYYYMMDD_HHMMSS.log
│   └── transformer_*_YYYYMMDD_HHMMSS.log
│
└── runs/                           # TensorBoard日志
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

#### RNN实验
```bash
# 单个实验
python train_rnn.py --experiment_type baseline

# 批量实验（注意力机制、训练策略）
./run_rnn_experiments.sh
```

#### Transformer实验
```bash
# 单个实验
python train_transformer.py --experiment_type baseline

# 批量实验（位置编码、归一化）
./run_transformer_experiments.sh
```

### 3. 查看实验结果

#### 方法1: 使用实验查看脚本（🆕 推荐）

```bash
# 列出所有实验
./view_experiments.sh

# 查看特定实验的详细信息
./view_experiments.sh rnn_20231221_143025
./view_experiments.sh transformer_20231221_150130
```

这会显示：
- ✅ 所有可用的实验批次
- ✅ 最新实验的标记
- ✅ 实验配置和结果
- ✅ 检查点文件位置
- ✅ 快速访问命令

#### 方法2: 直接查看实验目录

```bash
# 查看最新RNN实验的所有结果
cat experiments/latest_rnn/*/results.json | jq .

# 查看最新Transformer实验的所有结果
cat experiments/latest_transformer/*/results.json | jq .

# 查看特定实验的摘要
cat experiments/rnn_20231221_143025/summary.json | jq .

# 查看特定子实验的结果
cat experiments/latest_rnn/baseline/results.json | jq .
cat experiments/latest_rnn/attention_mechanisms/results.json | jq .
```

#### 方法3: 使用TensorBoard查看训练曲线

```bash
# 启动TensorBoard
tensorboard --logdir=runs

# 或使用脚本
./start_tensorboard.sh
```

然后在浏览器中打开 `http://localhost:6006`

#### 方法4: 查看执行日志

```bash
# 列出所有日志
ls -lh logs/

# 查看最新的RNN实验日志
tail -f logs/rnn_experiments_*.log

# 查看特定子实验的日志
cat logs/rnn_baseline_20231221_143025.log
```

### 4. 查看可视化图表

```bash
# 列出所有图表
ls -lh figures/

# 查看图表（需要图形界面）
# 或者直接在文件管理器中打开 figures/ 目录
```

生成的图表包括：
- `rnn_attention_comparison.png` - RNN注意力机制对比
- `rnn_training_strategy_comparison.png` - RNN训练策略对比
- `transformer_comparison.png` - Transformer消融实验对比

## 🗂️ 文件组织说明

### 实验结果组织

每次运行实验脚本（`run_rnn_experiments.sh` 或 `run_transformer_experiments.sh`）时，会创建一个带时间戳的实验目录：

```
experiments/
├── rnn_20231221_143025/          # 2023年12月21日 14:30:25 运行的RNN实验
│   ├── config.json               # 实验配置（epochs, batch_size等）
│   ├── summary.json              # 实验总结（包含所有子实验的路径）
│   ├── baseline/                 # 基线实验
│   │   ├── checkpoints/
│   │   │   ├── best.ckpt        # 最佳模型（验证损失最低）
│   │   │   └── last.ckpt        # 最后一个epoch的模型
│   │   └── results.json         # 测试集结果（BLEU分数等）
│   ├── attention_mechanisms/     # 注意力机制对比实验
│   │   ├── checkpoints/
│   │   │   ├── best.ckpt
│   │   │   └── last.ckpt
│   │   └── results.json
│   └── teacher_forcing/          # 训练策略对比实验
│       ├── checkpoints/
│       └── results.json
└── latest_rnn -> rnn_20231221_143025/  # 软链接，始终指向最新实验
```

### 检查点管理

**优化后的策略**：
- ✅ 每个子实验只保存 **2个检查点**：
  - `best.ckpt` - 验证损失最低的模型
  - `last.ckpt` - 最后一个epoch的模型
- ✅ 大幅减少磁盘占用（从每个实验~10个检查点减少到2个）
- ✅ 保留最重要的模型用于后续分析

### 日志管理

**执行日志** (`logs/` 目录)：
- 每次实验运行创建一个主日志文件：`{model}_experiments_{timestamp}.log`
- 每个子实验创建独立日志：`{model}_{experiment}_{timestamp}.log`
- 所有终端输出都会保存到日志文件

**TensorBoard日志** (`runs/` 目录)：
- 训练曲线、损失、BLEU分数等
- 使用 `tensorboard --logdir=runs` 查看

### 快速访问

使用软链接快速访问最新实验：

```bash
# 查看最新RNN实验结果
cat experiments/latest_rnn/*/results.json | jq .

# 查看最新Transformer实验结果
cat experiments/latest_transformer/*/results.json | jq .

# 加载最新RNN基线模型
python inference.py --checkpoint experiments/latest_rnn/baseline/checkpoints/best.ckpt
```

## 📊 实验类型

### RNN实验

1. **Baseline** - 基础训练
   ```bash
   python train_rnn.py --experiment_type baseline
   ```

2. **注意力机制对比** - dot, multiplicative, additive
   ```bash
   python train_rnn.py --experiment_type attention \
       --attention_types dot multiplicative additive
   ```

3. **训练策略对比** - Teacher Forcing比率 (1.0, 0.5, 0.0)
   ```bash
   python train_rnn.py --experiment_type training_strategy \
       --teacher_forcing_ratios 1.0 0.5 0.0
   ```

### Transformer实验

1. **Baseline** - 基础训练
   ```bash
   python train_transformer.py --experiment_type baseline
   ```

2. **位置编码对比** - absolute vs relative
   ```bash
   python train_transformer.py --experiment_type positional_encoding \
       --pos_enc_types absolute relative
   ```

3. **归一化对比** - LayerNorm vs RMSNorm
   ```bash
   python train_transformer.py --experiment_type normalization \
       --norm_types layer rms
   ```

4. **🆕 T5预训练模型微调** - 使用Google T5模型进行迁移学习

   **✅ 本项目已配置本地 T5 模型，无需下载！**

   本地模型位置：`T5_model/` (850MB)

   ```bash
   # 验证本地模型
   ./check_t5_files.sh

   # 运行 T5 微调（自动使用本地模型）
   python train_transformer.py --experiment_type t5_finetune \
       --n_epochs 10 \
       --batch_size 256
   ```

   支持的T5模型：
   - `google/t5-v1_1-small` (60M参数，推荐) - **已下载到本地**
   - `google/t5-v1_1-base` (220M参数)
   - `google/t5-v1_1-large` (770M参数，需要更大显存)

   详细说明：参见 [LOCAL_T5_SETUP.md](LOCAL_T5_SETUP.md)

## 📈 结果查询指南

### 查看实验结果的三种方式

1. **命令行摘要** - 快速查看所有结果
   ```bash
   python analyze_results.py
   ```

2. **JSON详细数据** - 查看完整的实验数据
   ```bash
   cat results/attention_mechanisms_comparison_lightning.json
   ```

3. **TensorBoard可视化** - 查看训练过程
   ```bash
   tensorboard --logdir=runs
   ```

### 结果文件说明

- `results/*.json` - 包含每个实验的BLEU分数、最佳模型路径等
- `figures/*.png` - 对比图表，可直接用于报告
- `runs/` - TensorBoard日志，包含详细的训练曲线
- `checkpoints/` - 保存的模型检查点

## 🔧 常用命令

```bash
# 分析所有结果并生成图表
python analyze_results.py

# 查看特定实验的JSON结果
cat results/attention_mechanisms_comparison_lightning.json | jq .

# 启动TensorBoard
tensorboard --logdir=runs --port=6006

# 列出所有生成的图表
ls -lh figures/

# 查看模型检查点
find checkpoints -name "*.ckpt"
```

## 📝 注意事项

- 所有训练脚本都使用PyTorch Lightning或HuggingFace Transformers框架
- 支持自动混合精度训练（FP16）以加速训练
- 支持多GPU训练（通过`CUDA_VISIBLE_DEVICES`环境变量控制）
- 实验结果自动保存到`results/`目录
- TensorBoard日志自动记录到`runs/`目录

## 📧 联系方式

如有问题，请查看代码注释或联系项目维护者。

