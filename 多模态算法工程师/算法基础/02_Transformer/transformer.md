---
title: Transformer
description: Transformer 架构的核心原理与实现详解
category: llm
tags: [transformer, attention, architecture]
status: stable
order: 1
---

# Transformer

## 一句话解释

Transformer 是一种基于自注意力机制（Self-Attention）的神经网络架构，摒弃了传统的循环和卷积结构，完全依赖注意力机制来捕捉序列中的依赖关系，实现了高效的并行计算。

## 1. 它解决什么问题

传统的 RNN 和 LSTM 存在以下问题：
- **无法并行计算**：必须按顺序处理序列，训练速度慢
- **长距离依赖困难**：梯度消失/爆炸导致难以捕捉远距离关系
- **训练效率低**：序列越长，训练越慢，且无法充分利用 GPU 并行能力

Transformer 通过自注意力机制解决了这些问题，允许序列中任意位置直接交互，实现完全并行化计算。

## 2. 架构总览

### 2.1 原始 Transformer 架构图

![Transformer 架构](https://jalammar.github.io/images/t/The_transformer_architecture_1.png)

*图源：The Illustrated Transformer*

Transformer 采用**编码器-解码器（Encoder-Decoder）**结构：

- **编码器（Encoder）**：将输入序列编码为连续表示
- **解码器（Decoder）**：基于编码器输出和已生成的 token 生成输出序列

每个编码器和解码器由 $N$ 个相同的层堆叠而成（原论文 $N=6$）。

### 2.2 完整数据流图

```
输入序列 → [Input Embedding + Positional Encoding]
              ↓
         ┌─────────────────────────────────────┐
         │           Encoder (×N)               │
         │  ┌───────────────────────────────┐  │
         │  │  Multi-Head Self-Attention     │  │
         │  └───────────────┬───────────────┘  │
         │                  ↓                   │
         │         Add & Layer Norm             │
         │                  ↓                   │
         │  ┌───────────────────────────────┐  │
         │  │    Feed-Forward Network        │  │
         │  └───────────────┬───────────────┘  │
         │                  ↓                   │
         │         Add & Layer Norm             │
         └──────────────────┬──────────────────┘
                            ↓
                      Encoder 输出
                            ↓
         ┌─────────────────────────────────────┐
         │           Decoder (×N)               │
         │  ┌───────────────────────────────┐  │
         │  │ Masked Multi-Head Self-Attn    │  │
         │  └───────────────┬───────────────┘  │
         │                  ↓                   │
         │         Add & Layer Norm             │
         │                  ↓                   │
         │  ┌───────────────────────────────┐  │
         │  │ Multi-Head Cross-Attention     │  │
         │  │ (Query from decoder,           │  │
         │  │  Key/Value from encoder)       │  │
         │  └───────────────┬───────────────┘  │
         │                  ↓                   │
         │         Add & Layer Norm             │
         │                  ↓                   │
         │  ┌───────────────────────────────┐  │
         │  │    Feed-Forward Network        │  │
         │  └───────────────┬───────────────┘  │
         │                  ↓                   │
         │         Add & Layer Norm             │
         └──────────────────┬──────────────────┘
                            ↓
                    Linear + Softmax
                            ↓
                        输出概率
```

> **💡 Tip**：理解架构图时，重点把握"编码器输出如何流入解码器的交叉注意力层"——这是 Encoder-Decoder 模型的核心信息传递路径。

## 3. 核心思想

### 3.1 自注意力机制（Self-Attention）

自注意力的核心思想是：**序列中的每个位置都可以直接关注其他所有位置**，无需通过循环传递信息。

![Self-Attention 可视化](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

*图源：The Illustrated Transformer*

### 3.2 多头注意力（Multi-Head Attention）

多头注意力将注意力计算拆分为多个"头"，每个头独立计算注意力后拼接：

![Multi-Head Attention](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

*图源：The Illustrated Transformer*

关键创新：
- **多头注意力**：从不同表示子空间关注不同类型的信息
- **位置编码**：注入序列位置信息，弥补自注意力缺乏位置感知的问题
- **残差连接 + 层归一化**：稳定训练，防止梯度消失

> **💡 Tip**：自注意力的"自"指的是 Q、K、V 都来自同一个序列。如果 Q 来自解码器而 K、V 来自编码器，则称为交叉注意力（Cross-Attention）。

## 4. 数学定义

### 1. 自注意力计算

给定输入序列 $X \in \mathbb{R}^{n \times d_{model}}$，通过三个线性变换得到 Query、Key、Value：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$，$W^V \in \mathbb{R}^{d_{model} \times d_v}$。

**注意力计算公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**分步推导**：

1. **计算注意力分数**：$S = QK^T \in \mathbb{R}^{n \times n}$
   - $S_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的关注程度

2. **缩放**：$S = \frac{S}{\sqrt{d_k}}$
   - 防止 $d_k$ 较大时点积值过大，导致 softmax 梯度消失

3. **Softmax 归一化**：$A = \text{softmax}(S)$
   - 每行归一化为概率分布，$\sum_j A_{ij} = 1$

4. **加权求和**：$\text{Output} = AV$
   - 每个位置的输出是所有位置 Value 的加权和

### 2. 多头注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

参数维度：
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$
- 原论文：$h=8$，$d_k = d_v = d_{model}/h = 64$

**为什么需要多头？**

单头注意力只能学习一种注意力模式。多头注意力允许模型同时关注：
- 不同的语法关系（主语-谓语、动词-宾语）
- 不同的语义关系（指代、修饰）
- 不同的距离模式（局部、全局）

### 3. 位置编码

Transformer 没有循环结构，无法感知序列位置。位置编码通过将位置信息注入输入嵌入来解决这个问题。

**正弦位置编码**：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：位置索引（0, 1, 2, ...）
- $i$：维度索引（0, 1, ..., $d_{model}/2-1$）
- $d_{model}$：模型维度

**为什么用正弦/余弦？**

1. **有界性**：$PE \in [-1, 1]$，不会随位置增大而爆炸
2. **相对位置可学习**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数
3. **泛化能力**：可以泛化到训练时未见过的序列长度

### 4. 前馈网络（FFN）

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$。

原论文：$d_{ff} = 4 \times d_{model} = 2048$。

FFN 相当于两层 MLP，中间维度扩大 4 倍，提供非线性变换能力。

### 5. 残差连接与层归一化

**残差连接**：
$$\text{SubLayer}(x) = x + \text{Dropout}(\text{Sublayer}(x))$$

**层归一化**：
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

其中 $\mu, \sigma$ 是特征维度的均值和标准差，$\gamma, \beta$ 是可学习参数。

残差连接保证梯度直接回传，层归一化稳定每层的输入分布。

> **💡 Tip**：注意力公式中 $\sqrt{d_k}$ 的缩放是面试高频考点。本质是假设 Q、K 元素服从标准正态分布时，点积的方差为 $d_k$，除以 $\sqrt{d_k}$ 将方差归一化为 1。

## 5. 完整实现代码

### 1. 位置编码

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```
### 2. 注意力机制
```python
def attention(query, key, value, mask=None, dropout=None):

# 将query矩阵的最后一个维度值作为d_k
d_k = query.size(-1)

# 将key的最后两个维度互换（转置），才能与query矩阵相乘，乘完了还要除以d_k开根号
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

# 如果存在要进行mask的内容，则将那些为0的部分替换一个很大的负数
if mask is not None:
	scores = scores.masked_fill(mask == 0, -1e9)

# 将mask后的attention矩阵按照最后一个维度进行softmax
p_attn = F.softmax(scores, dim=-1)

# 如果dropout参数设置为非空，则进行dropout操作
if dropout is not None:
	p_attn = dropout(p_attn)

# 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
return torch.matmul(p_attn, value), p_attn
```

### 2. 多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 拼接多头并线性变换
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output
```

### 3. 前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
```

### 4. 编码器层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # 自注意力 + 残差连接
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x
```

### 5. 解码器层

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力（防止看到未来信息）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 交叉注意力（Query from decoder, Key/Value from encoder）
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x
```

### 6. 完整 Transformer 模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 编码器和解码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def generate_mask(self, src, tgt):
        # src_mask: (batch_size, 1, 1, src_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器
        enc_output = self.src_embedding(src) * self.scale
        enc_output = self.positional_encoding(enc_output)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器
        dec_output = self.tgt_embedding(tgt) * self.scale
        dec_output = self.positional_encoding(dec_output)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出
        output = self.fc_out(dec_output)
        return output
```

> **💡 Tip**：实际工程中，`generate_mask` 里的 `src_mask` 用于过滤 padding token，`tgt_mask` 用于保证自回归生成时只看到历史 token。两者缺一不可。

## 6. 输入到输出的完整推演

以**英译中**任务为例，展示数据如何流经整个 Transformer。

### 第 1 步：输入预处理

```
英文句子: "I love AI"
中文句子: "<bos> 我 爱 人工智能 <eos>"

词汇表映射:
"I" → 101, "love" → 205, "AI" → 302
"<bos>" → 1, "我" → 401, "爱" → 402, "人工智能" → 501, "<eos>" → 2

源序列: [101, 205, 302]
目标序列: [1, 401, 402, 501, 2]
```

### 第 2 步：嵌入与位置编码

```python
src = torch.tensor([[101, 205, 302]])  # (1, 3)
tgt = torch.tensor([[1, 401, 402, 501, 2]])  # (1, 5)

# 嵌入: (batch_size, seq_len) → (batch_size, seq_len, d_model)
src_embed = src_embedding(src)  # (1, 3, 512)
tgt_embed = tgt_embedding(tgt)  # (1, 5, 512)

# 加入位置编码
src_embed = positional_encoding(src_embed)  # (1, 3, 512)
tgt_embed = positional_encoding(tgt_embed)  # (1, 5, 512)
```

### 第 3 步：编码器处理

```
编码器输入: (1, 3, 512) — 3个token，每个512维

经过6层编码器:
┌─────────────────────────────────────────────────┐
│ Encoder Layer 1                                  │
│  Self-Attention: 每个token关注其他2个token        │
│  FFN: 非线性变换                                 │
└─────────────────────────────────────────────────┘
                    ↓
              ... (重复6次) ...
                    ↓
编码器输出: (1, 3, 512) — 包含全局上下文信息
```

### 第 4 步：解码器处理

```
解码器输入: (1, 5, 512) — 5个token（含起始符）

经过6层解码器:
┌─────────────────────────────────────────────────┐
│ Decoder Layer 1                                  │
│  Masked Self-Attention:                          │
│    "我" 只能看到 "<bos>"                         │
│    "爱" 能看到 "<bos>", "我"                     │
│    ...                                          │
│  Cross-Attention:                                │
│    Query from decoder, Key/Value from encoder    │
│    "爱" 关注 "love"                              │
│  FFN: 非线性变换                                 │
└─────────────────────────────────────────────────┘
                    ↓
              ... (重复6次) ...
                    ↓
解码器输出: (1, 5, 512)
```

### 第 5 步：输出生成

```python
# 线性层映射到词汇表大小
logits = fc_out(dec_output)  # (1, 5, vocab_size)

# 训练时: 直接用 cross-entropy loss
# 推理时: 逐token生成
predicted_token = torch.argmax(logits[:, -1, :])  # 预测下一个token
```

### 训练时的掩码

```python
# 目标序列的因果掩码 (5×5)
tgt_mask = [
    [1, 0, 0, 0, 0],  # <bos> 只能看到自己
    [1, 1, 0, 0, 0],  # "我" 能看到 <bos> 和自己
    [1, 1, 1, 0, 0],  # "爱" 能看到前面两个
    [1, 1, 1, 1, 0],  # "人工智能" 能看到前面三个
    [1, 1, 1, 1, 1],  # <eos> 能看到所有
]
```

## 7. 面试标准回答

**"请介绍一下 Transformer 架构"**

Transformer 是 2017 年 Google 在论文 "Attention is All You Need" 中提出的序列到序列模型。它摒弃了 RNN 的循环结构，完全基于自注意力机制实现并行计算。

架构上分为编码器和解码器，各由 6 层相同的层堆叠而成。每层包含自注意力子层和前馈网络子层，都使用残差连接和层归一化。

核心创新有三点：(1) 自注意力让任意位置直接交互，解决长距离依赖问题；(2) 多头注意力从不同角度关注信息；(3) 位置编码注入序列顺序信息。

这使得 Transformer 训练效率远超 RNN，且在机器翻译等任务上效果更好。如今 BERT、GPT、T5 等大模型都基于 Transformer 架构。

**"为什么用缩放点积注意力？"**

使用 $\sqrt{d_k}$ 缩放是为了防止 $d_k$ 较大时点积结果过大。假设 $Q$ 和 $K$ 的每个元素独立同分布，均值为 0，方差为 1，那么 $Q \cdot K$ 的方差为 $d_k$。当 $d_k=64$ 时，点积值可能达到 $\pm 16$，导致 softmax 输出趋向 one-hot 分布，梯度接近于 0。除以 $\sqrt{d_k}$ 将方差归一化为 1，保证 softmax 有效工作。

> **💡 Tip**：面试回答时，先说"一句话核心"，再展开细节，最后联系工程（如 BERT/GPT 的应用），形成"总-分-总"结构。

## 常见追问

### Q1: 为什么需要位置编码？

Transformer 没有循环或卷积结构，自注意力计算是置换不变的——打乱输入顺序，输出只是相应打乱，内容不变。位置编码通过将位置信息与词嵌入相加，让模型感知序列顺序。现代模型也常用旋转位置编码（RoPE），它在注意力计算中相对位置信息。

### Q2: 为什么用多头注意力而不是单头？

单头注意力只能学习一种注意力模式。多头注意力让模型同时关注不同类型的关系：语法关系（主谓宾）、语义关系（指代）、距离模式（局部/全局）。原论文使用 8 个头，每个头 $d_k=64$，总计算量与单头 $d_k=512$ 相同。

### Q3: Transformer 的复杂度是多少？

自注意力的时间复杂度是 $O(n^2 \cdot d)$，空间复杂度是 $O(n^2)$（需要存储 $n \times n$ 的注意力矩阵）。其中 $n$ 是序列长度，$d$ 是特征维度。这是 Transformer 处理长序列的主要瓶颈，催生了 Flash Attention、稀疏注意力等优化方案。

### Q4: 掩码多头注意力的作用？

在解码器中，生成第 $t$ 个 token 时只能看到前 $t-1$ 个 token，否则会"偷看"未来信息。掩码通过将未来位置的注意力分数设为 $-\infty$，softmax 后权重为 0，实现因果约束。

### Q5: Encoder-Decoder 交叉注意力怎么工作？

解码器的交叉注意力中，Query 来自解码器上一层输出，Key 和 Value 来自编码器输出。这使得解码器在生成每个 token 时都能"查阅"整个源序列的信息，实现源序列与目标序列的对齐。

## 8. 工程实践

### 训练技巧

- **学习率预热（Warmup）**：前 4000 步线性增大到 $d_{model}^{-0.5}$，之后按步数衰减
- **标签平滑（Label Smoothing）**：$\epsilon_{ls}=0.1$，防止模型过于自信
- **梯度裁剪（Gradient Clipping）**：梯度范数大于 1 时缩放，防止梯度爆炸
- **混合精度训练**：使用 FP16/BF16 加速训练，减少显存占用

### 推理优化

- **KV Cache**：缓存历史 Key-Value，避免重复计算，推理速度提升 2-3 倍
- **模型量化**：将 FP16 量化为 INT8/INT4，减少计算量和显存
- **Flash Attention**：通过分块计算优化注意力，减少 HBM 访问
- **模型剪枝/蒸馏**：压缩模型大小，加速推理

### 常见超参数配置

| 配置 | base | big |
|------|------|-----|
| $d_{model}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $h$ | 8 | 16 |
| $d_k = d_v$ | 64 | 64 |
| 层数 $N$ | 6 | 6 |
| 参数量 | 65M | 213M |

> **💡 Tip**：KV Cache 是推理优化的第一步，它将自回归生成的时间复杂度从 $O(n^2)$ 降到 $O(n)$，但会额外占用显存。实际部署时需要在速度和显存之间权衡。

## 常见误区

- **误区**：Transformer 只能用于 NLP
  **事实**：Transformer 也可用于 CV（ViT）、音频（Wav2Vec）、多模态（CLIP）

- **误区**：注意力权重就是重要性
  **事实**：注意力权重只是模型的一种表示，不一定完全反映重要性，且不同头的权重含义不同

- **误区**：位置编码是唯一的位置信息来源
  **事实**：相对位置编码、旋转位置编码（RoPE）、ALiBi 等也是常用方法

- **误区**：Transformer 训练一定比 RNN 快
  **事实**：虽然可以并行，但自注意力的 $O(n^2)$ 复杂度在长序列上可能更慢

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [Harvard NLP: The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - 注释版实现
- [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) - Google 官方博客

## ✅ 自我检验

- [ ] 能用自己的话解释 Transformer 的核心思想
- [ ] 能说出它解决了什么问题（对比 RNN/LSTM）
- [ ] 能写出注意力公式并解释每个变量（Q、K、V、缩放因子）
- [ ] 能解释多头注意力的作用和实现方式
- [ ] 能说出位置编码的必要性及常见方案
- [ ] 能用 PyTorch 实现一个最简版 Transformer
- [ ] 能说出 2-3 个工程实践中的优化手段（KV Cache、Flash Attention 等）
- [ ] 能回答常见面试追问（位置编码、复杂度、掩码作用等）
