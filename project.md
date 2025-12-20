下面是**重新梳理后的项目说明文档**。我对内容进行了**层级重组与逻辑压缩**，突出「做什么 → 怎么做 → 怎么评 → 怎么交」，同时**完整保留所有资源链接与关键信息**，可直接作为课程项目正式说明文档使用。

---

# 中英机器翻译课程项目说明

**Midterm & Final Project: Chinese–English Machine Translation**

课程：Natural Language Processing and Large Language Models（2025 秋）
项目主题：中英文神经机器翻译（NMT）


---

## 一、项目目标（What to Do）

本项目旨在**系统实现并比较两类主流神经机器翻译模型**：

1. **基于 RNN 的 NMT（Seq2Seq + Attention）**
2. **基于 Transformer 的 NMT**

通过完整的工程实现与实验分析，理解二者在以下方面的差异与取舍：

* 模型结构设计
* 训练与推理效率
* 翻译质量与泛化能力
* 实际工程应用中的优缺点

项目强调：

> **评分不以最终 BLEU 数值为导向，而更关注设计合理性、实验分析深度与理解程度。**

---

## 二、总体任务结构（Overall Structure）

项目由 **四个层次递进的任务模块**构成：

1. **RNN-based NMT 实现与实验**
2. **Transformer-based NMT 实现与实验**
3. **系统性对比分析**
4. **工程提交与展示**

---

## 三、RNN-based NMT 任务要求

### 3.1 模型结构（Model Architecture）

* Encoder–Decoder 框架
* RNN 类型：**GRU 或 LSTM**
* 编码器与解码器：

  * 两层
  * 单向（unidirectional）

---

### 3.2 注意力机制（Attention Mechanism）

需实现并比较不同 alignment function 对性能的影响，包括：

* Dot-product attention
* Multiplicative attention
* Additive attention（Bahdanau）

目标：分析注意力对齐方式对翻译质量的影响。

---

### 3.3 训练策略对比（Training Policy）

比较以下两种训练方式：

* **Teacher Forcing**
* **Free Running（自回归训练）**

分析其对：

* 收敛速度
* 稳定性
* 翻译效果
  的影响。

---

### 3.4 解码策略对比（Decoding Policy）

比较：

* Greedy decoding
* Beam search decoding

重点分析不同解码策略在 BLEU 与译文质量上的差异。

---

## 四、Transformer-based NMT 任务要求

### 4.1 从零实现 Transformer（From Scratch）

* Encoder–Decoder Transformer 架构
* 不使用预训练权重
* 完整训练中英翻译模型

---

### 4.2 架构消融实验（Architectural Ablation）

对以下设计进行对比实验：

**位置编码方式**

* Absolute positional embedding
* Relative positional embedding

**归一化方式**

* LayerNorm
* RMSNorm

---

### 4.3 超参数敏感性分析（Hyperparameter Sensitivity）

通过多组实验分析以下因素对性能的影响：

* Batch size
* Learning rate
* Model scale（层数、hidden size 等）

---

### 4.4 基于预训练模型的微调（From Pretrained Model）

* 使用预训练语言模型（如 **T5**）
* 微调用于中英机器翻译
* 与「从零训练的 Transformer」进行对比分析

预训练模型资源：
[https://huggingface.co/google-t5/t5-base/tree/main](https://huggingface.co/google-t5/t5-base/tree/main)

---

## 五、RNN 与 Transformer 的综合对比分析

需要从**工程与研究双重视角**进行比较，至少覆盖以下维度：

### 5.1 模型结构差异

* RNN：序列计算、递归结构
* Transformer：并行计算、自注意力机制

### 5.2 训练效率

* 训练时间
* 收敛速度
* 硬件资源需求

### 5.3 翻译性能

* BLEU 分数
* 流畅度（fluency）
* 语义充分性（adequacy）

### 5.4 可扩展性与泛化能力

* 长句处理能力
* 低资源数据场景表现

### 5.5 实际工程权衡

* 模型规模
* 推理延迟
* 实现复杂度与可维护性

---

## 六、数据集与预处理要求

### 6.1 数据集说明

数据以 **JSONL 格式**提供，包含四个文件：

| 数据集                | 样本数量    |
| ------------------ | ------- |
| Small Training Set | 10,000  |
| Large Training Set | 100,000 |
| Validation Set     | 500     |
| Test Set           | 200     |

最终评测基于 **Test Set**。

数据下载地址：
[https://piazza.com/class_profile/get_resource/mfzcdlplb7n1n0/mifeas1otj31pl](https://piazza.com/class_profile/get_resource/mfzcdlplb7n1n0/mifeas1otj31pl)

> 若计算资源有限，可仅使用 10k 训练集；但鼓励尝试大规模训练。

---

### 6.2 数据预处理流程

**1）数据清洗**

* 移除非法字符
* 过滤极长句子
* 处理低频词

**2）分词（Tokenization）**

* 英文：

  * 空格分词
  * BPE / WordPiece
  * 工具：NLTK、SentencePiece
* 中文：

  * Jieba（轻量）
  * HanLP（精度更高）

**3）词表构建**

* 基于词频统计
* 控制词表规模

**4）词向量初始化**

* 推荐使用预训练词向量
* 允许训练过程中微调

分词工具资源：

* Jieba：[https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)
* SentencePiece：[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

---

## 七、评测指标（Evaluation Metric）

* **BLEU（Bilingual Evaluation Understudy）**

说明：
BLEU 用于定量评估翻译质量，但**不作为唯一评分依据**。

---

## 八、提交要求（Submission）

### 8.1 代码提交

* 建议使用 GitHub
* 提供一键推理脚本：`inference.py`

---

### 8.2 项目报告

* PDF 格式
* 命名：`ID_name.pdf`（如 `250010001_ZhangSan.pdf`）
* 报告需包含：

  * 模型架构说明
  * 实现过程说明
  * 实验设计与结果分析
  * 可视化分析
  * 个人反思
* 报告首页需注明 **代码仓库链接**

---

### 8.3 提交平台与截止时间

* 提交平台：[https://piazza.com](https://piazza.com)
* 截止时间：**12 月 28 日**

---

## 九、课堂展示（Presentation）

* 每组展示时间：

  * 10 分钟汇报
  * 5 分钟问答
* 全班约 18 组（每组约 7 人）
* **展示为小组进行，但项目报告需个人独立提交**
* 截止时间：**12 月 28 日**

---

## 十、参考资料（References）

* PyTorch Seq2Seq 教程
  [https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

* 经典论文：

  * *Attention is All You Need*（Vaswani et al., 2017）
  * *Neural Machine Translation by Jointly Learning to Align and Translate*（Bahdanau et al., 2015）
  * *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（Raffel et al., 2020）

