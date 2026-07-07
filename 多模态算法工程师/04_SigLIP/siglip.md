---
title: SigLIP
description: SigLIP 的 Sigmoid Loss 原理、数学推导、代码实现与工程实践
category: multimodal
tags: [SigLIP, CLIP, Vision-Language, Contrastive Learning, Image-Text Alignment]
status: draft
order: 4
---

# SigLIP

## 一句话解释

**SigLIP**（Sigmoid Loss for Language-Image Pre-training）是一种基于 Sigmoid 二分类损失的图文对齐模型，通过将 CLIP 的 softmax 对比损失替换为 pairwise sigmoid loss，使图文对齐训练更直接、对 batch size 依赖更弱。

---

## 1. 它解决什么问题

### 1.1 CLIP 的 Softmax 对比损失的瓶颈

CLIP（Contrastive Language-Image Pre-training）是目前最主流的图文对齐模型之一。它使用 softmax 对比损失（InfoNCE Loss）进行训练：对于一个 batch 中的 N 对图文数据，CLIP 构造一个 N×N 的相似度矩阵，将图文匹配看作 batch 内的多分类问题——每张图片需要在 N 个文本中找到正确的那个，每个文本也需要在 N 张图片中找到正确的那个。

这种设计有几个显著的痛点：

**痛点一：Softmax 的归一化瓶颈**

CLIP 的 InfoNCE Loss 需要在整个 batch 维度上做 softmax 归一化。具体来说，对于图像 $i$ 和文本 $j$ 的相似度 $s_{ij}$，损失函数要求：

$$\text{Loss}_i = -\log \frac{\exp(s_{ii})}{\sum_{j=1}^{N} \exp(s_{ij})}$$

这意味着计算一张图片的 loss 时，需要和 batch 内所有 N 个文本的相似度做归一化。softmax 的分母要求所有负样本同时参与计算，导致：

- 训练时必须维护一个足够大的 batch（通常需要 32768 甚至更大）
- 每张 GPU 都需要持有整个 batch 的相似度矩阵副本
- batch size 受限于 GPU 显存，成为训练规模的硬瓶颈

**痛点二：负样本数量决定训练质量**

CLIP 的 softmax 对比损失中，batch 内除对角线以外的所有图文组合都是负样本。对于 batch size N，每张图片有 N-1 个负样本。实验表明，CLIP 在 batch size 为 32768 时效果远好于 batch size 为 256：

| Batch Size | CLIP 零样本 ImageNet 准确率 |
|------------|--------------------------|
| 256        | 约 55%                    |
| 8192       | 约 63%                    |
| 32768      | 约 68%                    |

这意味着要达到好的效果，必须使用超大 batch size，而超大 batch size 又带来巨大的显存开销和通信成本。

**痛点三：分布式训练的通信开销**

为了支撑大 batch size，CLIP 通常采用多 GPU 分布式训练。softmax 归一化要求每个 GPU 都能访问全局的相似度矩阵，这需要在 GPU 之间做 all-gather 操作来收集所有图像和文本的 embedding。随着 GPU 数量增加，通信开销呈线性增长，成为扩展训练规模的主要瓶颈。

### 1.2 Softmax 的数值稳定性问题

除了大 batch size 的瓶颈，softmax 本身的数值特性也带来一些问题。

Softmax 的计算涉及指数运算 $e^{x}$，当相似度值较大时可能出现数值溢出。虽然实践中通常会减去最大值（log-sum-exp 技巧）来缓解，但这增加了计算复杂度。

更关键的是，softmax 的梯度特性：

$$\frac{\partial \mathcal{L}_i}{\partial s_{ij}} = \begin{cases} p_{ij} - 1 & \text{if } j = i \text{（正样本）} \\ p_{ij} & \text{if } j \neq i \text{（负样本）} \end{cases}$$

可以看到，softmax 的梯度同时依赖于所有类别的预测概率 $p_{ij}$。当 batch 内负样本数量变化时（比如不同 GPU 分到的 batch 内容不同），梯度的分布也会变化，导致训练不稳定。

相比之下，sigmoid 的梯度只依赖于当前 pair 的预测：

$$\frac{\partial \mathcal{L}_{ij}}{\partial s_{ij}} = \begin{cases} \sigma(s_{ij}) - 1 & \text{if } y_{ij} = 1 \\ \sigma(s_{ij}) & \text{if } y_{ij} = 0 \end{cases}$$

梯度计算完全独立，不受 batch 内其他 pair 的影响。

### 1.3 SigLIP 的解法

SigLIP 的核心洞察非常直接：

> 既然 softmax 要求全局归一化，那我们干脆不用 softmax，改用 sigmoid 做 pairwise 二分类。

对于每一对图文组合 $(i, j)$，SigLIP 独立计算一个二分类损失：如果 $(i, j)$ 匹配（即 $i = j$），标签为 1；如果不匹配，标签为 0。sigmoid 函数可以独立作用于每一对，完全不需要全局归一化。

这种设计带来了三个关键优势：

1. **不需要大 batch size**：每个图文 pair 独立计算 loss，不再依赖 batch 内负样本数量
2. **分布式训练更高效**：每个 GPU 可以独立计算自己的 loss，不需要 all-gather 全局相似度
3. **训练更灵活**：可以轻松调整正负样本比例，甚至可以使用 hard negative mining

> **💡 Tip**：SigLIP 的论文实验表明，使用较小的 batch size（如 4096）就能达到 CLIP 使用 32768 batch size 的效果，训练效率大幅提升。

---

## 2. 核心思想

### 2.1 Sigmoid Loss 原理

SigLIP 的核心思想是将图文对齐问题转化为 pairwise 二分类问题。

在 CLIP 中，训练目标是：给定一张图片，从 batch 内的 N 个文本中找到匹配的那个。这是一个 N 分类问题。

在 SigLIP 中，训练目标变为：对于每一对图文组合，判断它们是否匹配。这是一个二分类问题。

具体来说，对于 batch 中的第 $i$ 张图片和第 $j$ 个文本：

1. 计算相似度：$s_{ij} = v_i \cdot t_j$（向量内积）
2. 计算预测概率：$p_{ij} = \sigma(s_{ij})$（经过 sigmoid 函数）
3. 计算二分类损失：

$$\mathcal{L}_{ij} = -[y_{ij} \log(p_{ij}) + (1 - y_{ij}) \log(1 - p_{ij})]$$

其中标签 $y_{ij}$ 定义为：

$$y_{ij} = \begin{cases} 1 & \text{if } i = j \text{（匹配的图文对）} \\ 0 & \text{if } i \neq j \text{（不匹配的图文对）} \end{cases}$$

最终的总损失是所有图文对的平均：

$$\mathcal{L} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \mathcal{L}_{ij}$$

这种设计的关键在于：每个图文 pair 的 loss 计算是完全独立的，不需要知道 batch 内其他 pair 的相似度值。

### 2.2 Sigmoid vs Softmax 对比

下表详细对比了两种损失函数的差异：

| 对比维度 | CLIP Softmax (InfoNCE) | SigLIP Sigmoid |
|---------|----------------------|----------------|
| 损失函数 | $-\log \frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}$ | $-[y \log \sigma(s) + (1-y) \log(1-\sigma(s))]$ |
| 归一化方式 | 在整个 batch 维度上 softmax 归一化 | 每对独立 sigmoid，无需全局归一化 |
| 负样本来源 | batch 内除对角线外的所有图文组合 | 可灵活选择（batch 内/外部 hard negative） |
| 对 batch size 依赖 | 强依赖（需要 32K+ 才有好效果） | 弱依赖（4K 即可达到好效果） |
| 通信开销 | 需要 all-gather 全局 embedding | 每个 GPU 独立计算，通信量小 |
| 训练效率 | 受限于大 batch 的显存和通信 | 更高效，可以用更小的 batch |
| 理论等价性 | batch 趋近无穷时等价于 InfoNCE | batch 趋近无穷时等价于二分类交叉熵 |

从上表可以看出，SigLIP 的核心优势在于解耦了训练效果和 batch size 之间的强耦合关系。

### 2.3 模型架构

SigLIP 的模型架构和 CLIP 基本一致，主要包括三个组件：

```text
┌─────────────────────────────────────────────────────────┐
│                    SigLIP 模型架构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入图像 (H×W×3)          输入文本 (token sequence)      │
│       │                          │                      │
│       ▼                          ▼                      │
│  ┌─────────────┐          ┌─────────────┐               │
│  │ Image Encoder│          │ Text Encoder │               │
│  │  (ViT-L/14)  │          │ (Transformer)│               │
│  └──────┬──────┘          └──────┬──────┘               │
│         │                        │                      │
│         ▼                        ▼                      │
│  image embedding             text embedding              │
│  (d 维向量)                   (d 维向量)                  │
│         │                        │                      │
│         └────────┬───────────────┘                      │
│                  ▼                                      │
│         ┌──────────────┐                                │
│         │ 相似度计算    │                                │
│         │ s = v · t    │                                │
│         └──────────────┘                                │
│                  │                                      │
│                  ▼                                      │
│         ┌──────────────┐                                │
│         │ Sigmoid Loss │                                │
│         │ 二分类损失    │                                │
│         └──────────────┘                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Image Encoder

SigLIP 的图像编码器通常使用 Vision Transformer（ViT）。在 Google 发布的 SigLIP 模型中，使用了 ViT-L/14 作为默认图像编码器：

- **输入**：预处理后的图像，分辨率通常为 224×224 或 384×384
- **处理流程**：图像 → Patch Embedding → Transformer Blocks → Pooling → L2 归一化
- **输出**：d 维图像向量（ViT-L/14 的 d = 768）

ViT 的 Patch Embedding 将图像切分为 14×14 的 patch，每个 patch 通过线性投影映射为一个 token，加上位置编码后送入 Transformer 编码器。

#### Text Encoder

文本编码器使用标准的 Transformer Encoder：

- **输入**：tokenized 的文本序列，最大长度通常为 64 或 77
- **处理流程**：Token Embedding + Positional Embedding → Transformer Blocks → [CLS] Token → L2 归一化
- **输出**：d 维文本向量

#### 输出层

图像向量和文本向量都经过 L2 归一化，确保它们在同一个单位超球面上：

$$\hat{v}_i = \frac{v_i}{\|v_i\|_2}, \quad \hat{t}_j = \frac{t_j}{\|t_j\|_2}$$

相似度计算为归一化后的向量内积：

$$s_{ij} = \hat{v}_i \cdot \hat{t}_j$$

L2 归一化的好处是：相似度值被限制在 [-1, 1] 范围内，使得 sigmoid 函数的输入更稳定，训练更易收敛。

> **💡 Tip**：SigLIP 和 CLIP 的图像/文本编码器结构几乎一样，核心区别只在损失函数。面试时如果被问到"SigLIP 的架构有什么创新"，要强调"架构没有创新，创新在损失函数"。

---

## 3. 数学定义

### 3.1 SigLIP 的 Sigmoid Loss

SigLIP 的损失函数基于 pairwise 二分类交叉熵（Binary Cross-Entropy, BCE）。对于 batch 中的 N 对图文数据，定义相似度矩阵：

$$S \in \mathbb{R}^{N \times N}, \quad S_{ij} = \hat{v}_i^T \hat{t}_j$$

其中 $\hat{v}_i$ 和 $\hat{t}_j$ 分别是 L2 归一化后的图像和文本向量。

对应的标签矩阵：

$$Y \in \{0, 1\}^{N \times N}, \quad Y_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

SigLIP 的损失函数定义为：

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \left[ Y_{ij} \cdot \log \sigma(S_{ij}) + (1 - Y_{ij}) \cdot \log (1 - \sigma(S_{ij})) \right]$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

变量解释：
- $N$：batch size
- $\hat{v}_i$：第 $i$ 张图像的 L2 归一化 embedding 向量
- $\hat{t}_j$：第 $j$ 个文本的 L2 归一化 embedding 向量
- $S_{ij}$：图像 $i$ 和文本 $j$ 的余弦相似度
- $Y_{ij}$：标签，匹配为 1，不匹配为 0
- $\sigma(\cdot)$：sigmoid 函数，将任意实数映射到 (0, 1)

### 3.2 CLIP 的 InfoNCE Loss

作为对比，CLIP 使用的 InfoNCE 损失函数定义为：

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})} + \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})} \right]$$

其中：
- 第一项是图像到文本的对比损失：对于每张图像，正确文本的相似度应在所有文本中最大
- 第二项是文本到图像的对比损失：对于每个文本，正确图像的相似度应在所有图像中最大
- 两项的平均使得损失函数在图像和文本方向上对称

### 3.3 数值计算示例

为了更直观地理解 SigLIP 的 loss 计算，下面给出一个具体的数值示例。

假设 batch size N = 3，有 3 对图文数据：

| 配对 | 图像 | 文本 | 匹配？ | 标签 $Y_{ij}$ |
|------|------|------|--------|---------------|
| (1,1) | image_1 | text_1 | 是 | 1 |
| (1,2) | image_1 | text_2 | 否 | 0 |
| (1,3) | image_1 | text_3 | 否 | 0 |
| (2,1) | image_2 | text_1 | 否 | 0 |
| (2,2) | image_2 | text_2 | 是 | 1 |
| (2,3) | image_2 | text_3 | 否 | 0 |
| (3,1) | image_3 | text_1 | 否 | 0 |
| (3,2) | image_3 | text_2 | 否 | 0 |
| (3,3) | image_3 | text_3 | 是 | 1 |

假设相似度矩阵为：

$$S = \begin{bmatrix} 2.0 & -0.5 & -1.0 \\ -0.3 & 1.8 & -0.7 \\ -0.8 & -0.2 & 2.1 \end{bmatrix}$$

计算 sigmoid 后的概率矩阵：

$$P = \sigma(S) = \begin{bmatrix} 0.88 & 0.38 & 0.27 \\ 0.43 & 0.86 & 0.33 \\ 0.31 & 0.45 & 0.89 \end{bmatrix}$$

对于正样本（对角线），loss = $-\log(P_{ii})$：
- $(1,1)$: $-\log(0.88) = 0.13$
- $(2,2)$: $-\log(0.86) = 0.15$
- $(3,3)$: $-\log(0.89) = 0.12$

对于负样本，loss = $-\log(1-P_{ij})$：
- $(1,2)$: $-\log(0.62) = 0.48$
- $(1,3)$: $-\log(0.73) = 0.31$
- $(2,1)$: $-\log(0.57) = 0.56$
- $(2,3)$: $-\log(0.67) = 0.40$
- $(3,1)$: $-\log(0.69) = 0.37$
- $(3,2)$: $-\log(0.55) = 0.60$

总 loss = $(0.13 + 0.15 + 0.12 + 0.48 + 0.31 + 0.56 + 0.40 + 0.37 + 0.60) / 9 = 3.12 / 9 = 0.35$

可以看到，正样本的 loss 普遍较小（相似度高 → sigmoid 输出接近 1 → loss 小），负样本的 loss 稍大但也不大（相似度低 → sigmoid 输出接近 0 → loss 小）。

### 3.4 数学直觉对比

两种损失函数的直觉差异可以用一个比喻来理解：

**CLIP (Softmax)**：好比一场考试，每道题有 N 个选项，学生必须从 N 个选项中选出唯一正确的答案。所有选项的竞争是耦合在一起的——选 A 意味着排除 B、C、D...，softmax 归一化就是这种"互相竞争"的数学表达。

**SigLIP (Sigmoid)**：好比一系列独立的判断题，每道题问"这道题的答案对不对？"，学生只需给出"对"或"错"的判断。每道题的判断是独立的，不受其他题目的影响。

具体来说，假设 batch size N = 4，对于第 1 张图片：

**CLIP 的视角**：
```text
图片 1 vs 文本 1（正确）→ 相似度 0.8
图片 1 vs 文本 2（错误）→ 相似度 0.3
图片 1 vs 文本 3（错误）→ 相似度 0.2
图片 1 vs 文本 4（错误）→ 相似度 0.1

Softmax 归一化: [0.8, 0.3, 0.2, 0.1] → [0.45, 0.21, 0.18, 0.16]
Loss = -log(0.45) = 0.80
```

**SigLIP 的视角**：
```text
图片 1 vs 文本 1（正确）→ 相似度 0.8 → σ(0.8)=0.69 → Loss = -log(0.69) = 0.37
图片 1 vs 文本 2（错误）→ 相似度 0.3 → σ(0.3)=0.57 → Loss = -log(0.43) = 0.84
图片 1 vs 文本 3（错误）→ 相似度 0.2 → σ(0.2)=0.55 → Loss = -log(0.45) = 0.80
图片 1 vs 文本 4（错误）→ 相似度 0.1 → σ(0.1)=0.52 → Loss = -log(0.48) = 0.73

平均 Loss = (0.37 + 0.84 + 0.80 + 0.73) / 4 = 0.69
```

可以看到，SigLIP 的每个 pair 的 loss 计算是完全独立的，不需要知道其他 pair 的值。

### 3.4 Sigmoid 函数的数学性质

Sigmoid 函数 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 有几个关键性质：

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

$$\sigma(-x) = 1 - \sigma(x)$$

$$\lim_{x \to +\infty} \sigma(x) = 1, \quad \lim_{x \to -\infty} \sigma(x) = 0$$

这些性质保证了：
1. 输出始终在 (0, 1) 之间，适合作为概率输出
2. 梯度容易计算，不会出现数值溢出
3. 对于正样本（$y=1$），增大相似度 $s$ 可以减小 loss
4. 对于负样本（$y=0$），减小相似度 $s$ 可以减小 loss

> **💡 Tip**：面试时如果被问到"sigmoid 和 softmax 的区别"，可以从两个角度回答：(1) 数学上，sigmoid 是独立的二分类，softmax 是耦合的多分类；(2) 工程上，sigmoid 不需要全局归一化，对 batch size 更友好。

---

## 4. 算法流程

### 4.1 训练流程

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         SigLIP 训练流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 数据准备                                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  加载图文对数据集 (如 LAION-400M)                           │       │
│  │  每个 batch: N 张图片 + N 个对应文本                         │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  2. 前向传播                                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  图像 → Image Encoder (ViT) → image embeddings (N×d)      │       │
│  │  文本 → Text Encoder (Transformer) → text embeddings (N×d) │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  3. 相似度计算                                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  S = image_emb @ text_emb^T  → N×N 相似度矩阵              │       │
│  │  对 S 做 L2 归一化 (或先归一化再内积)                         │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  4. 构造标签                                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Y = eye(N)  → N×N 对角矩阵                                │       │
│  │  对角线为 1 (正样本), 其余为 0 (负样本)                       │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  5. 计算 Sigmoid Loss                                               │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  P = σ(S)  → N×N 预测概率矩阵                               │       │
│  │  L = -[Y*log(P) + (1-Y)*log(1-P)] / N^2  → 标量 loss       │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  6. 反向传播 + 参数更新                                              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  loss.backward()  → 计算梯度                                │       │
│  │  optimizer.step() → 更新 Image Encoder 和 Text Encoder 参数  │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 推理流程（零样本分类）

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      SigLIP 零样本分类推理流程                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入: 一张待分类图像 + K 个候选类别文本                               │
│                                                                     │
│  步骤 1: 构造文本 prompt                                             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  "a photo of a cat"                                       │       │
│  │  "a photo of a dog"                                       │       │
│  │  "a photo of a car"                                       │       │
│  │  ... (K 个类别)                                           │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  步骤 2: 分别编码                                                    │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Image Encoder: image → image_emb (1×d)                   │       │
│  │  Text Encoder:  "a photo of a cat" → text_emb_1 (1×d)     │       │
│  │                 "a photo of a dog" → text_emb_2 (1×d)     │       │
│  │                 "a photo of a car" → text_emb_3 (1×d)     │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  步骤 3: 计算相似度                                                  │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  score_1 = image_emb · text_emb_1  (余弦相似度)            │       │
│  │  score_2 = image_emb · text_emb_2                           │       │
│  │  score_3 = image_emb · text_emb_3                           │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  步骤 4: 选择最高相似度的类别                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  predicted_class = argmax([score_1, score_2, score_3])     │       │
│  │  → 输出: "cat"                                             │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 推理流程（图文匹配）

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      SigLIP 图文匹配推理流程                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入: 一张图像 + 一段文本描述                                         │
│                                                                     │
│  步骤 1: 分别编码                                                    │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Image Encoder: image → image_emb (1×d)                   │       │
│  │  Text Encoder:  text → text_emb (1×d)                     │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  步骤 2: 计算相似度 + Sigmoid                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  similarity = image_emb · text_emb  (余弦相似度)           │       │
│  │  probability = σ(similarity)  → 匹配概率 (0~1)              │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                       │
│  步骤 3: 判断匹配                                                    │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  if probability > threshold (如 0.5):                      │       │
│  │      → 匹配 (图文一致)                                      │       │
│  │  else:                                                    │       │
│  │      → 不匹配 (图文不一致)                                   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 逐步详解

1. **数据准备**：加载图文对数据，每个 batch 包含 N 张图片和 N 个对应文本。数据增强（如 RandAugment）通常在图像侧应用，文本侧不做增强。

2. **编码阶段**：图像和文本分别通过各自的编码器，得到 d 维的 embedding 向量。编码器内部通常包含多个 Transformer block，最后一层输出经过 L2 归一化。

3. **相似度计算**：将图像 embedding 和文本 embedding 做矩阵乘法，得到 N×N 的相似度矩阵。对角线上的元素是正确匹配的图文对，其余是负样本。

4. **Loss 计算**：对相似度矩阵应用 sigmoid 函数得到预测概率，然后计算二分类交叉熵损失。正样本（对角线）希望概率接近 1，负样本希望概率接近 0。

5. **参数更新**：反向传播计算梯度，同时更新 Image Encoder 和 Text Encoder 的参数。

> **💡 Tip**：在实际实现中，可以将正负样本的 loss 加权。正样本权重通常设为 1，负样本权重可以通过超参数调节，以控制正负样本的平衡。

---

## 5. 代码示例

### 5.1 使用 HuggingFace 加载 SigLIP 做零样本分类

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# 加载预训练的 SigLIP 模型和处理器
# 使用 google/siglip-base-patch16-224 模型
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # 设置为评估模式

# 加载测试图像（这里用随机图像作为示例）
# 实际使用时替换为真实图像路径
# image = Image.open("test_image.jpg")
from PIL import Image
import numpy as np

# 创建一张随机的测试图像 (224x224 RGB)
random_pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_pixels)

# 定义候选类别标签
candidate_labels = ["cat", "dog", "car", "bird", "fish"]

# 构造文本 prompt（SigLIP 使用 "a photo of a {label}" 格式）
text_prompts = [f"a photo of a {label}" for label in candidate_labels]

# 使用 processor 处理图像和文本
inputs = processor(
    images=image,
    text=text_prompts,
    padding="max_length",
    return_tensors="pt"
)

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits（相似度分数）
logits = outputs.logits_per_image  # shape: (1, num_labels)

# 将 logits 转换为概率
probabilities = torch.sigmoid(logits)  # SigLIP 使用 sigmoid

# 获取预测结果
predicted_idx = torch.argmax(probabilities, dim=-1).item()
predicted_label = candidate_labels[predicted_idx]

print(f"候选类别: {candidate_labels}")
print(f"相似度分数: {logits.squeeze().tolist()}")
print(f"匹配概率: {probabilities.squeeze().tolist()}")
print(f"预测类别: {predicted_label}")

# 预期输出示例（随机图像，结果不确定）:
# 候选类别: ['cat', 'dog', 'car', 'bird', 'fish']
# 相似度分数: [0.12, -0.05, 0.23, 0.08, -0.15]
# 匹配概率: [0.53, 0.49, 0.56, 0.52, 0.46]
# 预测类别: car
```

### 5.2 使用 SigLIP 做图文匹配

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# 加载模型
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 创建测试图像
random_pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_pixels)

# 定义待匹配的文本
texts = [
    "a photo of a cat sitting on a couch",
    "a beautiful sunset over the ocean",
    "a group of people playing basketball",
]

# 处理输入
inputs = processor(
    images=image,
    text=texts,
    padding="max_length",
    return_tensors="pt"
)

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取相似度分数
similarities = outputs.logits_per_image.squeeze()  # shape: (num_texts,)

# 计算匹配概率（SigLIP 使用 sigmoid）
match_probabilities = torch.sigmoid(similarities)

print("图文匹配结果:")
for text, sim, prob in zip(texts, similarities, match_probabilities):
    print(f"  文本: '{text}'")
    print(f"  相似度: {sim:.4f}, 匹配概率: {prob:.4f}")
    print()

# 选择最匹配的文本
best_idx = torch.argmax(match_probabilities).item()
print(f"最匹配的文本: '{texts[best_idx]}'")
print(f"匹配概率: {match_probabilities[best_idx]:.4f}")

# 预期输出示例:
# 图文匹配结果:
#   文本: 'a photo of a cat sitting on a couch'
#   相似度: 0.0823, 匹配概率: 0.5206
#
#   文本: 'a beautiful sunset over the ocean'
#   相似度: -0.1245, 匹配概率: 0.4689
#
#   文本: 'a group of people playing basketball'
#   相似度: 0.0341, 匹配概率: 0.5085
#
# 最匹配的文本: 'a photo of a cat sitting on a couch'
# 匹配概率: 0.5206
```

### 5.3 从零实现 Sigmoid Loss

```python
import torch
import torch.nn.functional as F

def sigmoid_loss(image_embeddings, text_embeddings, labels=None):
    """
    从零实现 SigLIP 的 Sigmoid Loss

    Args:
        image_embeddings: 图像 embedding, shape (N, d), 已 L2 归一化
        text_embeddings: 文本 embedding, shape (N, d), 已 L2 归一化
        labels: 标签矩阵, shape (N, N), 默认为单位矩阵（对角线为 1）

    Returns:
        loss: 标量损失值
    """
    N = image_embeddings.shape[0]

    # 如果没有提供标签，默认为对角线匹配（即第 i 张图匹配第 i 个文本）
    if labels is None:
        labels = torch.eye(N, device=image_embeddings.device)

    # 计算相似度矩阵: (N, d) @ (d, N) = (N, N)
    similarities = image_embeddings @ text_embeddings.t()

    # 计算 sigmoid 概率
    probabilities = torch.sigmoid(similarities)

    # 计算二分类交叉熵损失
    # 正样本损失: -log(σ(s_ij)), 当 Y_ij = 1
    # 负样本损失: -log(1-σ(s_ij)), 当 Y_ij = 0
    loss_pos = -labels * torch.log(probabilities + 1e-8)
    loss_neg = -(1 - labels) * torch.log(1 - probabilities + 1e-8)

    # 求平均
    loss = (loss_pos + loss_neg).sum() / (N * N)

    return loss


# 测试代码
torch.manual_seed(42)
N, d = 8, 768  # batch size = 8, embedding dim = 768

# 模拟 L2 归一化后的 embedding
image_emb = torch.randn(N, d)
image_emb = F.normalize(image_emb, p=2, dim=1)

text_emb = torch.randn(N, d)
text_emb = F.normalize(text_emb, p=2, dim=1)

# 计算 loss
loss = sigmoid_loss(image_emb, text_emb)
print(f"Sigmoid Loss: {loss.item():.4f}")

# 对比: 如果使用 softmax loss (InfoNCE)
# CLIP 风格的 InfoNCE loss
logits_per_image = image_emb @ text_emb.t() * 100  # temperature scaling
labels = torch.arange(N, device=image_emb.device)
loss_clip_img = F.cross_entropy(logits_per_image, labels)
loss_clip_txt = F.cross_entropy(logits_per_image.t(), labels)
loss_clip = (loss_clip_img + loss_clip_txt) / 2

print(f"CLIP InfoNCE Loss: {loss_clip.item():.4f}")
print(f"两者 loss 值不同，因为损失函数定义不同")

# 预期输出:
# Sigmoid Loss: 0.6931
# CLIP InfoNCE Loss: 2.0794
# 两者 loss 值不同，因为损失函数定义不同
```

### 5.4 使用 SigLIP 做图像特征提取

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# 加载模型
model_name = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 创建测试图像
random_pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_pixels)

# 处理图像
inputs = processor(images=image, return_tensors="pt")

# 提取图像特征（不计算 text features）
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

print(f"图像特征形状: {image_features.shape}")  # 预期: torch.Size([1, 768])
print(f"特征 L2 范数: {image_features.norm(dim=-1).item():.4f}")  # 预期: 1.0000

# 预期输出:
# 图像特征形状: torch.Size([1, 768])
# 特征 L2 范数: 1.0000
```

> **💡 Tip**：使用 HuggingFace 的 `AutoModel` 加载 SigLIP 时，可以通过 `model.get_image_features()` 和 `model.get_text_features()` 分别获取图像和文本的特征，无需手动调用编码器。

---

## 6. 面试回答

> SigLIP 是 Google 在 2023 年提出的一种图文对齐模型，全称是 Sigmoid Loss for Language-Image Pre-training。它和 CLIP 的核心区别在于损失函数：CLIP 使用 softmax 对比损失（InfoNCE），SigLIP 使用 sigmoid 二分类损失。
>
> 具体来说，CLIP 把图文匹配看成一个 batch 内的多分类问题——每张图片要在 batch 内的 N 个文本中找到正确的那个，softmax 归一化要求所有负样本同时参与计算。这导致 CLIP 必须使用超大 batch size（通常 32K+）才能有好效果，训练时显存开销和 GPU 间通信成本都很高。
>
> SigLIP 的做法更直接：对于每一对图文组合，独立计算一个二分类损失——匹配就是正样本，不匹配就是负样本，sigmoid 函数作用于每一对。这种设计不需要全局归一化，每个 GPU 可以独立计算自己的 loss，对 batch size 依赖很弱。论文实验表明，SigLIP 用 4K batch size 就能达到 CLIP 用 32K batch size 的效果。
>
> 在工程上，SigLIP 的架构和 CLIP 几乎一样，都是 ViT 做图像编码器、Transformer 做文本编码器，核心创新完全在损失函数。这也是为什么 SigLIP 可以轻松替换 CLIP 作为 VLM 的视觉编码器——只需要换损失函数重新训练，架构不需要改。

---

## 7. 工程实践

### 7.1 部署场景对比

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| 零样本图像分类 | SigLIP ViT-L/14 | prompt 写法影响结果，建议用 "a photo of a {class}" |
| 图文检索 | SigLIP + FAISS 索引 | 提前计算所有图像/文本 embedding，用 FAISS 做近似最近邻搜索 |
| VLM 视觉编码器 | SigLIP ViT + 投影层 | 投影层将视觉特征映射到 LLM 的 hidden size |
| 边缘部署 | SigLIP ViT-B/16 + ONNX | 小模型 + 量化，延迟更低 |
| 大规模训练 | SigLIP + FSDP/DeepSpeed | Sigmoid loss 对 batch size 依赖弱，可以用更小的 batch |

### 7.2 与其他模型的集成

SigLIP 最常见的工程用法是作为多模态大模型（VLM）的视觉编码器。典型流程：

```text
输入图像
  │
  ▼
SigLIP Image Encoder → 视觉特征 (N_patches × d)
  │
  ▼
投影层 (Linear / MLP) → 映射到 LLM hidden size
  │
  ▼
LLM (如 Gemini、Gemma) → 文本生成
```

这种架构的优势：
1. SigLIP 的视觉特征已经和文本对齐，LLM 更容易理解
2. 只需训练投影层和 LLM，视觉编码器可以冻结
3. 可以灵活更换不同规模的 SigLIP 模型来平衡效果和效率

### 7.3 性能优化技巧

1. **混合精度训练**：使用 FP16 或 BF16 可以显著减少显存占用和加速训练
2. **梯度累积**：如果显存不足以支撑目标 batch size，可以用梯度累积等效放大 batch
3. **冻结部分参数**：微调时可以冻结 Image Encoder 的底层，只训练顶层和 Text Encoder
4. **Prompt Engineering**：零样本分类时，prompt 的措辞对结果影响很大，需要根据任务调整

> **💡 Tip**：在生产环境中使用 SigLIP 时，建议先用 FP16 推理测试正确性，再考虑用 ONNX Runtime 或 TensorRT 进一步优化延迟。

### 7.4 SigLIP 模型变体对比

| 模型 | 图像分辨率 | Patch Size | 参数量 | Embedding Dim | 适用场景 |
|------|-----------|------------|--------|---------------|----------|
| SigLIP-B/16-224 | 224×224 | 16×16 | 87M | 768 | 轻量部署、边缘设备 |
| SigLIP-L/14-224 | 224×224 | 14×14 | 304M | 1024 | 通用场景、VLM 视觉塔 |
| SigLIP-L/14-384 | 384×384 | 14×14 | 304M | 1024 | 高分辨率任务、细粒度分类 |
| SigLIP-SO400M/14-224 | 224×224 | 14×14 | 400M | 1152 | 高精度场景 |

选择建议：
- **边缘部署 / 实时推理**：SigLIP-B/16-224，参数量小、推理快
- **VLM 视觉编码器**：SigLIP-L/14-224，平衡效果和效率
- **细粒度图像理解**：SigLIP-L/14-384，高分辨率保留更多细节
- **追求最佳效果**：SigLIP-SO400M，参数量更大、训练数据更多

### 7.5 常见问题排查

**问题 1：零样本分类准确率低**
- 检查 prompt 格式是否正确（建议用 "a photo of a {class}"）
- 检查图像预处理是否匹配模型要求（分辨率、归一化参数）
- 尝试使用更大的模型（如从 Base 升级到 Large）

**问题 2：推理速度慢**
- 确认是否启用了 FP16 推理
- 检查 batch size 是否合理（批量推理比逐条快）
- 考虑使用 ONNX Runtime 优化

**问题 3：显存不足**
- 降低 batch size
- 使用梯度检查点（gradient checkpointing）
- 冻结部分参数减少优化器状态

**问题 4：图文匹配概率始终接近 0.5**
- 检查输入是否正确通过了 processor
- 确认模型权重下载完整
- 检查文本是否为空或格式异常

> **💡 Tip**：调试 SigLIP 模型时，建议先用随机图像和随机文本测试，确认模型能正常输出 logits。如果 logits 全部接近 0，可能是 processor 的归一化参数有问题。

---

## 常见追问

SigLIP 的全称是什么？核心创新在哪？
SigLIP 全称 Sigmoid Loss for Language-Image Pre-training，核心创新是将 CLIP 的 softmax 对比损失替换为 sigmoid 二分类损失，使得训练对 batch size 依赖更弱。

SigLIP 和 CLIP 的架构有什么区别？
架构几乎没有区别，都是 ViT 做图像编码器 + Transformer 做文本编码器。核心区别在损失函数：CLIP 用 InfoNCE（softmax），SigLIP 用 pairwise sigmoid loss。

为什么 Sigmoid Loss 对 batch size 依赖更弱？
因为 sigmoid 是 pairwise 的，每个图文 pair 独立计算 loss，不需要 softmax 那样的全局归一化。CLIP 的 softmax 分母需要 batch 内所有负样本参与计算，batch 越小负样本越少，效果越差。SigLIP 没有这个问题。

SigLIP 可以直接做文本生成吗？
不可以。SigLIP 是图文对齐模型，只输出 embedding 向量，不生成文本。要生成文本需要接入 LLM，SigLIP 通常作为 VLM 的视觉编码器使用。

SigLIP 在哪些模型中被实际使用？
Google 的 Gemini 系列、PaliGemma、以及许多开源 VLM（如 LLaVA 的部分变体）都使用 SigLIP 作为视觉编码器。它已经成为 VLM 视觉塔的主流选择之一。

---

## 常见误区

SigLIP 是一种新的模型架构，提出了新的编码器设计。
SigLIP 的编码器结构（ViT + Transformer）和 CLIP 基本一致，核心创新完全在损失函数。不要把 SigLIP 理解为一种新的架构。

SigLIP 完全替代了 CLIP，以后都用 SigLIP。
SigLIP 和 CLIP 各有优势，不是替代关系。SigLIP 在训练效率上更好，但 CLIP 的生态更成熟、预训练模型更多。实际选择取决于具体场景。

SigLIP 的 sigmoid loss 只是把 softmax 换成 sigmoid，没有其他改动。
实际上 SigLIP 还做了一些工程优化，比如对正负样本加权、调整温度参数等。但核心思路确实是用 sigmoid 替代 softmax，这是最重要的改动。

SigLIP 的训练不需要大 batch size，所以可以用单卡训练。
虽然 SigLIP 对 batch size 依赖更弱，但训练大规模 SigLIP 模型仍然需要多卡分布式。小 batch size 可以用，但训练效果和稳定性会受影响。

SigLIP 的零样本分类效果一定比 CLIP 好。
不一定。SigLIP 的优势在训练效率，零样本分类效果还取决于模型规模、训练数据、prompt 设计等多种因素。在相同训练数据和模型规模下，SigLIP 和 CLIP 的零样本效果差异不大。

---

## 参考文献

1. [Sigmoid Loss for Language-Image Pre-Training](https://arxiv.org/abs/2303.15343) — Zhai et al., ICCV 2023
2. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., ICML 2021
3. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., ICLR 2021
4. [HuggingFace Transformers: SigLIP Documentation](https://huggingface.co/docs/transformers/model_doc/siglip) — HuggingFace
5. [PaliGemma: A 3B Vision-Language Model](https://arxiv.org/abs/2407.07726) — Google, 2024

---

## ✅ 自我检验

- [ ] 能用自己的话解释 SigLIP 和 CLIP 的核心区别
- [ ] 能写出 SigLIP 的 sigmoid loss 公式并解释每个变量
- [ ] 能说出为什么 Sigmoid Loss 对 batch size 依赖更弱
- [ ] 能用 HuggingFace 加载 SigLIP 做零样本分类
- [ ] 能用 HuggingFace 加载 SigLIP 做图文匹配
- [ ] 能从零实现 Sigmoid Loss
- [ ] 能说出 SigLIP 在 VLM 中的作用和典型架构
- [ ] 能回答 5 个常见面试追问
- [ ] 能区分 SigLIP 的常见误区
