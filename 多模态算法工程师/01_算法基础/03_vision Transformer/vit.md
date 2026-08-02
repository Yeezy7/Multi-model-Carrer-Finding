---
title: ViT 视觉 Transformer
description: Vision Transformer 的原理、Patch Embedding、Self-Attention 机制、模型变体与工程实践
category: multimodal
tags: [ViT, Vision Transformer, Image Classification, Patch Embedding, Self-Attention]
status: draft
order: 2
---

# ViT 视觉 Transformer

## 一句话解释

Vision Transformer（ViT）是一种将 Transformer 架构直接应用于图像分类的模型，通过将图像切分为固定大小的 patch 序列，利用 Self-Attention 建模全局关系，证明了纯 Transformer 也能在视觉任务上达到甚至超越 CNN 的效果。

---

## 1. 它解决什么问题

卷积神经网络（Convolutional Neural Network, CNN）长期以来是视觉领域的主流架构。CNN 通过卷积核在局部区域滑动提取特征，天然具有平移不变性（Translation Invariance）和局部性（Locality），在图像分类、目标检测等任务上取得了巨大成功。然而，CNN 存在几个本质性的局限：

### 1.1 局部感受野的限制

CNN 的核心操作——卷积——本质上是局部的。一个 $3 \times 3$ 的卷积核只能看到 $3 \times 3$ 的局部区域。要获取全局信息，必须堆叠多层卷积。例如，对于一张 $224 \times 224$ 的图像：

- 第 1 层 $3 \times 3$ 卷积的感受野只有 $3 \times 3$ 像素
- 堆叠 3 层后感受野扩展到 $7 \times 7$
- 堆叠 10 层后才能覆盖 $21 \times 21$ 的区域
- 要覆盖整张图像，需要非常深的网络（50+ 层）

这意味着 CNN 的每一层只能看到图像的一小部分，必须通过逐层堆叠来"看到"全局信息。这种设计导致了两个问题：（1）长距离依赖需要很多层才能建立；（2）深层网络存在梯度消失（Gradient Vanishing）的风险。

### 1.2 感受野与计算效率的矛盾

为了增大感受野，CNN 引入了池化（Pooling）和空洞卷积（Dilated Convolution）。池化操作虽然扩大了感受野，但损失了空间分辨率。空洞卷积可以在不增加参数的情况下扩大感受野，但引入了网格效应（Gridding Effect），可能丢失细粒度信息。

以 ResNet-50 为例，为了获取全局特征，网络需要 50 层卷积，参数量约 25.6M，FLOPs 约 4.1G。而 ViT-B/16 只需要 12 层 Transformer Encoder，参数量 86M，但通过 Self-Attention 可以在第一层就建立所有 patch 之间的全局连接。

### 1.3 Transformer 在 NLP 中的成功启示

Transformer 在自然语言处理（Natural Language Processing, NLP）领域已经取得了革命性成功。BERT、GPT 等模型证明了 Transformer 的 Self-Attention 机制能够有效捕获长距离依赖关系。关键问题是：**Transformer 能否直接应用于视觉任务？**

直接将 Self-Attention 应用于图像像素面临计算量爆炸的问题。一张 $224 \times 224$ 的图像有 $224 \times 224 = 50{,}176$ 个像素，Self-Attention 的计算复杂度是 $O(n^2)$，其中 $n$ 是 token 数量。这意味着需要计算 $50{,}176^2 \approx 2.5 \times 10^9$ 次注意力运算，计算量完全不可接受。

### 1.4 ViT 的解决方案：Patch 化

ViT 的核心洞察是：**不处理像素，而是处理图像块（Patch）**。将图像切分为固定大小的 patch，每个 patch 类比为 NLP 中的一个 token。具体来说：

- 一张 $224 \times 224$ 的图像，使用 $16 \times 16$ 的 patch size
- 得到 $N = \frac{224 \times 224}{16 \times 16} = 196$ 个 patch
- 相比 50,176 个像素，token 数量降低了 256 倍
- Self-Attention 的计算量从 $O(50176^2)$ 降低到 $O(196^2)$

这种设计既保留了图像的结构信息，又将计算量控制在可接受的范围内。ViT 论文（An Image is Worth 16x16 Words）在 ImageNet-21K 上预训练后，Top-1 准确率达到 88.55%，超过了当时最好的 CNN 模型。

> **💡 Tip**：ViT 的 patch 化思想不仅用于图像分类，还被 ViTDet、MAE 等工作扩展到目标检测、图像生成等任务中。理解 patch 化是理解所有 Vision Transformer 变体的基础。

---

## 2. 核心思想

### 2.1 Patch Embedding

Patch Embedding 是 ViT 的第一步，也是最关键的一步。它的作用是将二维图像转换为一维 token 序列，使 Transformer 能够处理。

#### 2.1.1 数学定义

给定输入图像 $x \in \mathbb{R}^{H \times W \times C}$，其中 $H$ 是高度，$W$ 是宽度，$C$ 是通道数（RGB 图像为 3）。Patch Embedding 的过程如下：

**第一步：切分为 patch**

将图像 $x$ 切分为 $N$ 个不重叠的 patch，每个 patch 的大小为 $P \times P$：

$$
N = \frac{H \times W}{P \times P}
$$

对于标准配置（$H=W=224, P=16$）：$N = \frac{224 \times 224}{16 \times 16} = 196$

每个 patch $x_p^i$ 是一个 $P \times P \times C$ 的三维张量，展平后为 $P^2 \cdot C$ 维的向量：

$$
x_p^i \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, 2, \ldots, N
$$

**第二步：线性投影**

将每个展平的 patch 通过线性投影矩阵 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 映射到 $D$ 维嵌入空间：

$$
z_p^i = x_p^i \cdot E + b
$$

其中 $D$ 是嵌入维度（ViT-B 为 768），$b$ 是偏置项。

**第三步：等价于卷积实现**

在实际代码中，Patch Embedding 通常用一个卷积层实现：

```python
# Conv2d 的 kernel_size=patch_size, stride=patch_size 实现了不重叠的 patch 切分
self.proj = nn.Conv2d(in_channels=3, out_channels=D, kernel_size=P, stride=P)
```

卷积操作 $y = \text{Conv2d}(x)$ 的输出形状为 $(B, D, H/P, W/P)$，展平后得到 $(B, D, N)$，与手动切分 + 线性投影完全等价。

#### 2.1.2 Patch Size 的影响

| Patch Size | Patch 数量 (224×224) | Token 序列长度 | 参数量 (投影层) | 适用场景 |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 3136 | 3136 | 较小 | 高精度、小图像 |
| 8 | 784 | 784 | 中等 | 平衡精度与速度 |
| 14 | 256 | 256 | 较大 | 高精度任务 |
| 16 | 196 | 196 | 较大 | 标准配置 |
| 32 | 49 | 49 | 较小 | 快速推理 |

Patch size 越小，token 数量越多，Self-Attention 能捕获的细节越丰富，但计算量也成平方增长。实际中 $P=16$ 是最常用的配置。

> **💡 Tip**：在微调阶段，可以使用插值（Interpolation）调整位置编码的大小，支持不同分辨率的输入图像，而不需要重新训练整个模型。

### 2.2 CLS Token 与位置编码

#### 2.2.1 CLS Token

CLS Token（Classification Token）是一个可学习的虚拟 token，拼接在 patch 序列的最前面。它的作用类似于 BERT 中的 [CLS] token——经过多层 Transformer 后，它的表示汇聚了整个图像的信息，用作分类特征。

具体来说，给定 patch 嵌入序列 $z_p = [z_p^1, z_p^2, \ldots, z_p^N]$，拼接 CLS token 后：

$$
z_0 = [z_{cls}; z_p^1; z_p^2; \ldots; z_p^N] \in \mathbb{R}^{(N+1) \times D}
$$

其中 $z_{cls} \in \mathbb{R}^D$ 是一个可学习的参数。最终只取 CLS token 的输出 $z_L^0$ 用于分类：

$$
\hat{y} = \text{MLP}(z_L^0)
$$

CLS Token 的优势：

1. **避免池化操作**：不需要对所有 patch 做平均池化或最大池化，避免信息损失
2. **端到端可训练**：CLS Token 与整个网络一起训练，自动学习最优的全局特征汇聚方式
3. **简洁优雅**：一个 token 即可代表整张图像，输出维度固定

#### 2.2.2 位置编码

Transformer 的 Self-Attention 是置换不变的（Permutation Invariant）——打乱输入 token 的顺序，输出不变。但图像的 patch 有明确的空间关系（左上角的 patch 和右下角的 patch 语义完全不同），所以需要注入位置信息。

ViT 使用可学习的位置编码（Learnable Positional Encoding）：

$$
z_0^i = z_p^i + e_{pos}^i
$$

其中 $e_{pos}^i \in \mathbb{R}^D$ 是第 $i$ 个位置的可学习编码向量。完整的位置编码矩阵为：

$$
E_{pos} = [e_{pos}^0; e_{pos}^1; \ldots; e_{pos}^N] \in \mathbb{R}^{(N+1) \times D}
$$

其中 $e_{pos}^0$ 对应 CLS token 的位置。

可学习位置编码 vs 正弦位置编码（Sinusoidal Positional Encoding）：

| 特性 | 可学习位置编码 | 正弦位置编码 |
|------|:---:|:---:|
| 是否需要训练 | 是 | 否 |
| 泛化到新长度 | 需要插值 | 直接计算 |
| 参数量 | $O(N \times D)$ | 0 |
| 实际效果 | 略优 | 略差 |
| ViT 使用 | ✅ | ❌ |

实验表明，可学习位置编码在 ViT 中略优于正弦编码，但差异不大。重要的是，位置编码训练后能学到 patch 之间的空间关系——相邻 patch 的位置编码更相似，远处 patch 的位置编码差异更大。

> **💡 Tip**：在微调时，如果输入图像分辨率改变（如 $224 \to 384$），patch 数量会变化（$196 \to 576$），位置编码需要插值调整。PyTorch 中可以使用 `torch.nn.functional.interpolate` 实现。

### 2.3 Transformer Encoder 处理

ViT 使用标准的 Transformer Encoder 结构，由多层 Encoder Block 堆叠而成。每个 Encoder Block 包含两个核心子层：多头自注意力（Multi-Head Self-Attention, MHSA）和前馈网络（Feed-Forward Network, FFN）。

#### 2.3.1 Multi-Head Self-Attention

Self-Attention 的核心思想是让每个 token 能够"看到"所有其他 token，并根据相关性加权聚合信息。

给定输入序列 $Z \in \mathbb{R}^{(N+1) \times D}$，首先通过三个线性投影得到 Query、Key、Value：

$$
Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V
$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D_k}$ 是可学习的投影矩阵，$D_k$ 是每个头的维度。

单头自注意力的计算：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

多头自注意力将 $D$ 维空间分成 $h$ 个头，每个头独立计算注意力，再拼接：

$$
\text{MHSA}(Z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

其中每个头：

$$
\text{head}_i = \text{Attn}(Z W_Q^i, Z W_K^i, Z W_V^i)
$$

多头的好处是不同的头可以关注不同的语义信息（如颜色、形状、纹理等），增强模型的表达能力。

#### 2.3.2 Feed-Forward Network (FFN)

每个 Encoder Block 的第二个子层是 FFN，它对每个 token 独立地做非线性变换：

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

其中 $W_1 \in \mathbb{R}^{D \times D_{ff}}$，$W_2 \in \mathbb{R}^{D_{ff} \times D}$，$D_{ff}$ 通常是 $D$ 的 4 倍（如 ViT-B 中 $D=768, D_{ff}=3072$）。

FFN 的作用是在每个 token 上引入非线性变换，增强模型的表达能力。GELU（Gaussian Error Linear Unit）激活函数比 ReLU 更平滑，在 Transformer 中广泛使用。

#### 2.3.3 残差连接与层归一化

每个子层都使用残差连接（Residual Connection）和层归一化（Layer Normalization, LN）：

$$
z' = \text{LN}(z + \text{MHSA}(z))
$$
$$
z'' = \text{LN}(z' + \text{FFN}(z'))
$$

残差连接帮助梯度流动，防止深层网络的梯度消失问题。层归一化稳定训练过程，加速收敛。

完整的 Encoder Block 数据流：

```
输入 z (B, N+1, D)
    │
    ▼
┌─────────────────────────────┐
│  Layer Norm                 │
│      │                      │
│      ▼                      │
│  Multi-Head Self-Attention  │
│      │                      │
│      ▼                      │
│  Add (残差连接)              │
│      │                      │
│      ▼                      │
│  Layer Norm                 │
│      │                      │
│      ▼                      │
│  Feed-Forward Network       │
│      │                      │
│      ▼                      │
│  Add (残差连接)              │
└─────────────────────────────┘
    │
    ▼
输出 z'' (B, N+1, D)
```

> **💡 Tip**：ViT 使用的是 Pre-Norm 结构（先 LN 再 Attention/FFN），而非原始 Transformer 的 Post-Norm。Pre-Norm 训练更稳定，对学习率不那么敏感，是现代 Transformer 的标准做法。

### 2.4 与其他视觉模型对比

ViT 与其他主流视觉模型的架构对比：

| 特性 | ViT | ResNet | Swin Transformer | ConvNeXt |
|------|:---:|:---:|:---:|:---:|
| 核心操作 | Self-Attention | Conv 3×3 | Shifted Window Attention | Conv 3×3 |
| 感受野 | 全局（第一层） | 局部（逐层增大） | 局部窗口 | 局部（逐层增大） |
| 位置信息 | 可学习位置编码 | 卷积隐式编码 | 相对位置偏置 | 卷积隐式编码 |
| 归纳偏置 | 弱 | 强（局部性+平移不变性） | 中等 | 强 |
| 数据需求 | 大 | 中等 | 中等 | 中等 |
| 层级结构 | 无（单尺度） | 有（4 个 stage） | 有（4 个 stage） | 有（4 个 stage） |
| 多尺度特征 | ❌ | ✅ | ✅ | ✅ |
| ImageNet-1K Top-1 | 88.55% (ViT-H) | 78.5% (R-50) | 87.3% (Swin-L) | 87.8% (ConvNeXt-L) |

从对比可以看出：

1. **ViT 的优势**：全局建模能力强，第一层就能建立所有 patch 之间的连接，架构简洁统一
2. **ViT 的劣势**：缺少归纳偏置（局部性、平移不变性），需要更多数据；单尺度特征，不利于检测和分割
3. **Swin Transformer 的改进**：引入层级结构和滑动窗口注意力，兼顾了全局建模和计算效率，成为视觉任务的通用骨干
4. **ConvNeXt 的反击**：用纯 CNN 模拟 Transformer 的设计（大 kernel、LayerNorm、GELU），证明了 CNN 仍有竞争力

> **💡 Tip**：在面试中，能清晰对比 ViT、ResNet、Swin Transformer 三者的异同，是一个很好的加分项。重点关注感受野、归纳偏置、数据需求和多尺度特征四个维度。

---

## 3. 数学定义

### 3.1 Self-Attention 公式

Self-Attention 是 ViT 的核心计算单元。给定输入序列 $Z \in \mathbb{R}^{N \times D}$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q \in \mathbb{R}^{N \times d_k}$：Query 矩阵，由 $Z W_Q$ 得到
- $K \in \mathbb{R}^{N \times d_k}$：Key 矩阵，由 $Z W_K$ 得到
- $V \in \mathbb{R}^{N \times d_v}$：Value 矩阵，由 $Z W_V$ 得到
- $d_k$：每个头的维度，通常 $d_k = D / h$，$h$ 是注意力头数
- $\sqrt{d_k}$：缩放因子，防止点积过大导致 softmax 梯度消失

**缩放的必要性**：假设 $Q$ 和 $K$ 的元素独立同分布，均值为 0，方差为 1。那么 $Q \cdot K$ 的均值为 0，方差为 $d_k$。当 $d_k$ 较大时（如 64），点积的值会很大，softmax 会输出接近 one-hot 的分布，梯度趋近于 0，导致训练困难。除以 $\sqrt{d_k}$ 将方差归一化为 1。

### 3.2 Multi-Head Self-Attention

$$
\text{MHSA}(Z) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W_O
$$

其中每个头：

$$
\text{head}_i = \text{Attention}(Z W_Q^i, Z W_K^i, Z W_V^i)
$$

参数矩阵维度：
- $W_Q^i, W_K^i \in \mathbb{R}^{D \times d_k}$，其中 $d_k = D / h$
- $W_V^i \in \mathbb{R}^{D \times d_v}$，其中 $d_v = D / h$
- $W_O \in \mathbb{R}^{D \times D}$：输出投影矩阵

对于 ViT-B/16：$D = 768$, $h = 12$, $d_k = d_v = 64$

### 3.3 前馈网络 (FFN)

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

其中：
- $W_1 \in \mathbb{R}^{D \times D_{ff}}$：第一层投影，$D_{ff} = 4D$
- $W_2 \in \mathbb{R}^{D_{ff} \times D}$：第二层投影，恢复原始维度
- GELU 激活函数：$\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的累积分布函数

GELU 的近似公式：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
$$

### 3.4 计算复杂度分析

对于一个 Encoder Layer，各操作的计算复杂度：

| 操作 | 计算复杂度 | 空间复杂度 | 说明 |
|------|:---:|:---:|------|
| Self-Attention | $O(N^2 D)$ | $O(N^2)$ | $N$ 个 token 之间的两两注意力 |
| FFN | $O(N D^2)$ | $O(N D_{ff})$ | 每个 token 独立计算 |
| 总计 | $O(N^2 D + N D^2)$ | $O(N^2 + N D)$ | 取决于 $N$ 和 $D$ 的相对大小 |

**关键分析**：

- 当 $N < D$ 时（如 ViT-B 中 $N=196, D=768$），$N^2 D < N D^2$，FFN 是主要瓶颈
- 当 $N > D$ 时（高分辨率输入），$N^2 D > N D^2$，Self-Attention 是主要瓶颈
- 高分辨率图像的 Self-Attention 计算量是主要瓶颈，这也是 Swin Transformer 引入窗口注意力的原因

以 ViT-B/16 处理 $224 \times 224$ 图像为例：
- $N = 196$, $D = 768$, $h = 12$
- Self-Attention: $196^2 \times 768 \approx 29.5M$ 次浮点运算
- FFN: $196 \times 768 \times 3072 \approx 464.5M$ 次浮点运算
- FFN 的计算量约为 Self-Attention 的 15.7 倍

### 3.5 模型变体对比

| 模型 | 层数 $L$ | 隐藏维度 $D$ | 头数 $h$ | FFN 维度 $D_{ff}$ | 参数量 | ImageNet-21K Top-1 | ImageNet-1K Top-1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-Ti/16 | 12 | 192 | 3 | 768 | 5.6M | - | 72.2% |
| ViT-S/16 | 12 | 384 | 6 | 1536 | 22.1M | 79.7% | 81.5% |
| ViT-B/16 | 12 | 768 | 12 | 3072 | 86.6M | 84.0% | 84.2% |
| ViT-L/16 | 24 | 1024 | 16 | 4096 | 304.3M | 85.2% | 85.2% |
| ViT-H/14 | 32 | 1280 | 16 | 5120 | 632.2M | 88.6% | - |
| DeiT-B/16 | 12 | 768 | 12 | 3072 | 86.6M | - | 83.8% |

从表中可以看出：
- 模型越大，性能越好，但参数量和计算量也成倍增长
- ViT-H/14 在 ImageNet-21K 上达到 88.65%，超过当时所有 CNN 模型
- DeiT-B 通过知识蒸馏，仅用 ImageNet-1K 就达到了 83.8%，接近 ViT-B 的性能

> **💡 Tip**：面试时被问到"ViT 有多少参数"，记住 ViT-B/16 约 86M，ViT-L/16 约 304M，ViT-H/14 约 632M。

---

## 4. 算法流程

### 4.1 完整前向传播流程

```
┌─────────────────────────────────────────────────────────────┐
│                    输入图像 (B, 3, 224, 224)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Patch Embedding (Conv2d)                        │
│  kernel_size=16, stride=16, out_channels=768                │
│  输出: (B, 768, 14, 14) → flatten → (B, 768, 196)          │
│  → transpose → (B, 196, 768)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           拼接 CLS Token + 位置编码                           │
│  CLS Token: (B, 1, 768)  随机初始化                          │
│  位置编码: (B, 197, 768)  随机初始化                         │
│  输出: z_0 = [CLS; patch_1; ...; patch_196] + pos_embed      │
│  形状: (B, 197, 768)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Transformer Encoder (× L 层)                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Layer Norm                                          │    │
│  │      │                                               │    │
│  │      ▼                                               │    │
│  │  Multi-Head Self-Attention (h=12 heads)              │    │
│  │      │                                               │    │
│  │      ▼                                               │    │
│  │  Add & Layer Norm (残差连接)                          │    │
│  │      │                                               │    │
│  │      ▼                                               │    │
│  │  Feed-Forward Network (768 → 3072 → 768)            │    │
│  │      │                                               │    │
│  │      ▼                                               │    │
│  │  Add & Layer Norm (残差连接)                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  重复 L 次 (ViT-B: L=12)                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              取 CLS Token 输出                                │
│  z_L^0 = x[:, 0]  取第 0 个 token                            │
│  形状: (B, 768)                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              分类头 (Classification Head)                     │
│  LayerNorm → Linear(768, 1000)                              │
│  输出: (B, 1000)  每个类别的 logits                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 训练流程

1. **数据预处理**：图像 resize 到 $224 \times 224$，归一化到 $[0, 1]$，减去均值除以标准差
2. **前向传播**：按上述流程计算 logits
3. **计算损失**：交叉熵损失（Cross-Entropy Loss）$\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$
4. **反向传播**：计算梯度 $\frac{\partial \mathcal{L}}{\partial \theta}$
5. **参数更新**：AdamW 优化器更新参数

### 4.3 推理流程

推理时与训练基本相同，但有以下区别：
- 使用 `model.eval()` 关闭 Dropout 和 Batch Normalization 的随机性
- 使用 `torch.no_grad()` 关闭梯度计算，节省显存
- 输出 logits 经过 argmax 得到预测类别

> **💡 Tip**：ViT 推理时可以使用 Flash Attention 加速，它将 $O(N^2)$ 的注意力计算优化为 $O(N)$ 的 IO 感知算法，实际加速 2-4 倍。

---

## 5. 代码示例

### 5.1 PatchEmbedding 实现

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并投影到嵌入空间。

    使用 Conv2d 实现不重叠的 patch 切分和线性投影，
    等价于手动切分 + 线性层，但更高效。
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 224//16 = 14, 14*14 = 196

        # Conv2d 实现 patch 切分 + 线性投影
        # kernel_size=patch_size, stride=patch_size → 不重叠的 patch
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入图像, 如 (B, 3, 224, 224)
        Returns:
            (B, num_patches, embed_dim) patch 嵌入序列
        """
        x = self.proj(x)           # (B, embed_dim, H/P, W/P) = (B, 768, 14, 14)
        x = x.flatten(2)           # (B, embed_dim, num_patches) = (B, 768, 196)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim) = (B, 196, 768)
        return x
```

### 5.2 VisionTransformer 完整实现

```python
class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) 完整实现。

    参考论文: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Patch 嵌入层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196

        # 2. CLS Token 和位置编码
        # CLS Token: 可学习的虚拟 token，用于汇聚全局信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码: num_patches + 1 (包含 CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),  # 768 * 4 = 3072
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 5. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化 CLS Token 和位置编码为小随机值。"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            (B, num_classes) 分类 logits
        """
        B = x.shape[0]

        # 1. Patch 嵌入
        x = self.patch_embed(x)  # (B, 196, 768)

        # 2. 拼接 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)

        # 3. 添加位置编码
        x = x + self.pos_embed  # (B, 197, 768)

        # 4. Transformer Encoder
        x = self.transformer(x)  # (B, 197, 768)

        # 5. 取 CLS Token 输出，通过分类头
        x = self.norm(x[:, 0])  # (B, 768) 取 CLS token
        x = self.head(x)  # (B, num_classes)

        return x
```

### 5.3 使用示例

```python
# 创建 ViT-B/16 模型
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
)

# 模拟输入: batch_size=2, 3 通道, 224x224
x = torch.randn(2, 3, 224, 224)

# 前向传播
output = model(x)
print(f"输出形状: {output.shape}")  # 预期: torch.Size([2, 1000])

# 查看模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e6:.1f}M")  # 预期: ~86.6M
```

预期输出：

```
输出形状: torch.Size([2, 1000])
模型参数量: 86.6M
```

### 5.4 带预训练权重加载的使用示例

```python
import timm

# 使用 timm 加载预训练 ViT
model = timm.create_model("vit_base_patch16_224", pretrained=True)

# 创建数据预处理管道
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# 推理示例
from PIL import Image
img = Image.open("example.jpg")
x = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

with torch.no_grad():
    output = model(x)  # (1, 1000)
    pred = output.argmax(dim=-1)
    print(f"预测类别: {pred.item()}")
```

> **💡 Tip**：实际项目中推荐使用 `timm` 库（PyTorch Image Models），它提供了 700+ 种预训练视觉模型，包括各种 ViT 变体，开箱即用。

---

## 6. 面试回答

> **"ViT 是什么？它的核心思想是什么？"**
>
> ViT 就是把 Transformer 直接用在图像上。核心思路很简单：把一张图切成固定大小的 patch，每个 patch 当成一个 token，然后扔给 Transformer 处理。比如一张 224×224 的图，切成 16×16 的 patch，就得到 196 个 token，跟 NLP 里的 196 个词一样处理。
>
> 关键设计有三个：一是 CLS token，一个可学习的虚拟 token，拼在最前面，最后取它的输出做分类；二是位置编码，因为 Transformer 不知道 patch 的位置关系，所以要加上可学习的位置编码；三是标准的 Transformer Encoder，多层堆叠，每层包含多头自注意力和 FFN。
>
> ViT 最大的意义在于证明了纯 Transformer 也能做视觉任务，而且在大数据集上效果比 CNN 还好。但它也有局限——缺少 CNN 的归纳偏置，需要更多数据才能训练好。

> **"ViT 和 CNN 相比，优缺点是什么？"**
>
> 优点方面，ViT 第一层就能建立全局连接，不需要像 CNN 那样堆很多层才能看到全局信息；架构简洁统一，同一个 Transformer 可以处理图像、文本、甚至多模态；在大数据集上（ImageNet-21K、JFT-300M）预训练后，ViT 通常优于同等大小的 CNN。
>
> 缺点方面，ViT 缺少 CNN 的归纳偏置——局部性和平移不变性——所以小数据集上容易过拟合；Self-Attention 的计算复杂度是 $O(N^2)$，高分辨率图像上计算量大；ViT 是单尺度特征，不利于目标检测和分割这类需要多尺度特征的任务。

> **"ViT 的 CLS Token 为什么有效？"**
>
> CLS token 本身是一个可学习的参数，没有初始语义信息。但经过多层 Transformer 后，每一层的 Self-Attention 都会让 CLS token 跟所有 patch 交互，逐渐汇聚全局信息。类比来说，CLS token 就像一个"提问者"，它带着自己的问题去看每个 patch，最后把所有信息汇总成一个向量。这个向量就包含了整张图像的语义信息，直接用于分类。

---

## 7. 工程实践

### 7.1 预训练策略

ViT 的性能高度依赖预训练策略。由于缺少归纳偏置，ViT 需要更多数据和更强的正则化。

| 策略 | 具体方法 | 效果 |
|------|----------|------|
| 大规模数据 | ImageNet-21K (14M), JFT-300M (300M) | Top-1 提升 5-10% |
| 数据增强 | RandAugment, Mixup, CutMix, Random Erasing | Top-1 提升 2-4% |
| 正则化 | Stochastic Depth (随机深度), Label Smoothing | 防止过拟合 |
| 优化器 | AdamW (β1=0.9, β2=0.999), weight decay=0.3 | 训练更稳定 |
| 学习率调度 | Cosine LR with warmup (10 epochs) | 收敛更快更稳 |
| 知识蒸馏 | DeiT 引入蒸馏 token，从 CNN 蒸馏 | 降低数据需求 |

### 7.2 微调技巧

在预训练模型基础上微调到下游任务时，需要注意：

1. **学习率**：微调学习率通常为预训练的 1/10 到 1/100（如 1e-4 到 1e-5）
2. **冻结策略**：可以冻结前几层 Transformer Encoder，只微调后几层和分类头
3. **分辨率调整**：微调时可以增大分辨率（如 224→384），位置编码需要插值
4. **数据增强**：小数据集微调时使用更强的数据增强，如 Mixup、RandAugment
5. **标签平滑**：设置 label_smoothing=0.1 可以防止过拟合

### 7.3 推理优化

| 优化方法 | 加速效果 | 实现难度 | 适用场景 |
|----------|:---:|:---:|------|
| Flash Attention | 2-4x | 低 | 通用，推荐首选 |
| 混合精度 (FP16/BF16) | 1.5-2x | 低 | GPU 推理 |
| ONNX Runtime | 1.2-2x | 中 | 部署推理 |
| TensorRT | 2-5x | 高 | NVIDIA GPU 部署 |
| 模型剪枝 | 1.5-3x | 高 | 边缘设备 |
| 知识蒸馏 | 1.5-2x | 中 | 轻量化部署 |

### 7.4 部署注意事项

1. **模型导出**：使用 `torch.onnx.export` 导出 ONNX 格式，再用 ONNX Runtime 或 TensorRT 部署
2. **动态 batch**：部署时支持动态 batch size，提高吞吐量
3. **输入预处理**：确保推理时的预处理与训练一致（resize、归一化参数）
4. **显存管理**：ViT-B 约占 340MB 显存（FP32），ViT-L 约 1.2GB，ViT-H 约 2.5GB
5. **批量推理**：使用 `torch.cuda.amp` 混合精度 + `DataParallel` 多卡推理

> **💡 Tip**：在生产环境中，推荐使用 `timm` 库加载预训练模型，配合 `torch.onnx.export` 导出 ONNX，再用 ONNX Runtime 部署。这套流程成熟稳定，性能优化做得很好。

---

## 常见追问

**Q1: ViT 需要多少数据才能训练好？**

ViT 缺少 CNN 的归纳偏置，需要大量数据。原始论文在 ImageNet-21K（1400 万张图片）上预训练后，在 ImageNet-1K 上达到 88.55% 的 Top-1 准确率。如果只用 ImageNet-1K（130 万张）从头训练，效果通常不如 ResNet。DeiT（Data-efficient Image Transformer）通过知识蒸馏，仅用 ImageNet-1K 就达到了 83.8%，接近 ViT-B 的性能。实际中，推荐使用 ImageNet-21K 预训练 + ImageNet-1K 微调的两阶段策略。

**Q2: patch_size 怎么选？**

patch_size 越小，token 数量越多，Self-Attention 能捕获的细粒度特征越丰富，但计算量成平方增长。常用配置：$P=16$（ViT-B/16，标准配置）、$P=14$（ViT-B/14，略高精度）、$P=32$（快速推理）。对于小图像（如 CIFAR-10 的 $32 \times 32$），可以用 $P=4$ 或 $P=8$。实际选择需要在精度和速度之间权衡。

**Q3: ViT 可以用于目标检测和分割吗？**

可以，但原始 ViT 不太适合，因为它是单尺度特征。改进方案有：（1）ViTDet：在 ViT 基础上加一个简单的检测头，用于目标检测；（2）Swin Transformer：引入层级结构和滑动窗口注意力，成为检测和分割的通用骨干；（3）MAE（Masked Autoencoder）：自监督预训练后可以用于各种下游任务。

**Q4: 为什么 ViT 在小数据集上不如 CNN？**

CNN 有强归纳偏置：局部连接（假设相邻像素相关）、权值共享（假设平移不变性）。这些先验知识在数据少时很有帮助，相当于给了模型一个"好的起点"。ViT 没有这些偏置，需要从数据中学习空间关系，数据少时容易过拟合或学不到有效的空间结构。类比来说，CNN 像是一个有经验的工程师，ViT 像是一个天赋高但需要大量训练的新手。

**Q5: Flash Attention 是什么？为什么能加速 ViT？**

Flash Attention 是一种 IO 感知（IO-aware）的精确注意力算法。标准 Self-Attention 需要将 $N \times N$ 的注意力矩阵完整写入 GPU 的 HBM（高带宽内存），然后逐行读取计算。Flash Attention 通过分块（tiling）技术，将注意力计算在 SRAM（片上缓存）中完成，避免了 HBM 的读写瓶颈。结果是：计算量不变，但实际速度提升 2-4 倍，同时显存占用从 $O(N^2)$ 降低到 $O(N)$。在 PyTorch 2.0+ 中可以直接使用 `torch.nn.functional.scaled_dot_product_attention` 启用。

---

## 常见误区

❌ **"ViT 完全替代了 CNN"** — 不是。在数据充足时 ViT 效果好，但 CNN 在小数据集、边缘设备上仍有优势。很多实际系统是 CNN + Transformer 混合架构（如 ConvNeXt、Swin Transformer）。

❌ **"ViT 的注意力一定是全局的"** — 原始 ViT 是全局注意力，但 Swin Transformer 引入了局部窗口注意力，计算量更小。实际中，全局注意力在高分辨率图像上计算量过大，需要优化。

❌ **"位置编码是固定的"** — ViT 使用可学习的位置编码，不是固定的正弦编码。训练后模型能学会 patch 之间的空间关系。正弦编码在 NLP 中更常用，但 ViT 实验证明可学习编码效果略好。

❌ **"patch_size 越小效果越好"** — 不一定。太小的 patch 会大幅增加计算量（$O(N^2)$），但效果提升不一定成正比。实际中需要在精度和速度之间权衡，$P=16$ 是最常用的配置。

❌ **"CLS Token 是必须的"** — 不是。MAE（Masked Autoencoder）等方法直接对所有 patch 做池化，没有使用 CLS Token，效果也很好。CLS Token 只是一种方便的设计选择，不是 ViT 成功的关键因素。

❌ **"ViT 不能处理高分辨率图像"** — 可以，但需要优化。原始 ViT 对高分辨率图像计算量过大，但通过 Swin Transformer 的窗口注意力、或 DynamicViT 的动态 token 剪枝等方法，ViT 也能高效处理高分辨率图像。

---

## 参考文献

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020 (ViT 原始论文)
2. [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — Touvron et al., 2021 (DeiT)
3. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Liu et al., 2021 (Swin Transformer)
4. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — He et al., 2021 (MAE)
5. [A ConvNet for the 2020s](https://arxiv.org/abs/2112.04456) — Liu et al., 2022 (ConvNeXt)
6. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022

---

## ✅ 自我检验

- [ ] 能用自己的话解释 ViT 是什么，核心思想是什么
- [ ] 能说出 ViT 解决了 CNN 的哪些局限性
- [ ] 能画出 ViT 的架构流程图
- [ ] 能写出 Self-Attention 的公式并解释每个变量
- [ ] 能实现一个最简版本的 VisionTransformer
- [ ] 能说出 patch_size 对模型性能的影响
- [ ] 能说出 CLS Token 和位置编码的作用
- [ ] 能对比 ViT、ResNet、Swin Transformer 的异同
- [ ] 能说出 2-3 个工程实践中的注意事项（预训练、微调、推理优化）
- [ ] 能回答常见面试追问（数据需求、patch_size 选择、高分辨率处理）
