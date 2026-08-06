# SigLIP 图文对齐：Sigmoid Loss 的完整数学与工程剖析

> 本笔记是全系列最详细的一篇。SigLIP（Sigmoid Loss for Language-Image Pre-training）由 Google 于 2023 年提出（ICCV 2023），是 CLIP 之后图文对齐（Vision-Language Alignment）最重要的改进之一。面试被问"图文对齐怎么做"时，SigLIP 是必须能讲透的模型。

## 一、一句话解释

> **SigLIP = 把 CLIP 的 softmax 对比损失（InfoNCE）换成逐对独立的 sigmoid 二分类损失（pairwise sigmoid loss）的图文对齐模型。**

换掉损失函数后，训练不再依赖全局归一化，batch size 从 32K 降到 4K 仍能达到同等效果，分布式训练通信成本大幅下降，训练时间缩短约 3.5 倍，同时可以在训练中灵活控制正负样本比例。

**架构没有创新，创新全在损失函数** —— 这是理解 SigLIP 最重要的一句话。

---

## 二、它解决什么问题：CLIP 的三大瓶颈

### 2.1 痛点一：Softmax 要求全局归一化

CLIP 的 InfoNCE 损失对 batch 内每张图都要做一次 softmax：

$$\mathcal{L}_i = -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ij}/\tau)}$$

softmax 的分母要求**所有负样本的相似度同时参与计算**。这意味着：

1. **计算耦合**：算任何一个 pair 的损失，都要知道整个 batch 所有 pair 的相似度；
2. **显存瓶颈**：N×N 相似度矩阵在每张卡上都要完整存在；
3. **通信瓶颈**：多卡训练时必须 all-gather 全局图像/文本特征，GPU 越多通信量线性增长。

### 2.2 痛点二：batch size 决定训练质量

CLIP 论文中的关键实验数据：

| Batch Size | CLIP 零样本 ImageNet 准确率 |
|------------|--------------------------|
| 256 | 约 55% |
| 8K | 约 63% |
| 32K | 约 68% |
| 64K | 约 68%（收敛） |

原因：InfoNCE 的负样本全部来自 batch 内部，**batch 越小负样本越少，对比越不充分**。要达到好效果必须 32K+ 的超大 batch，而大 batch 又带来显存和通信的双重压力。

### 2.3 痛点三：softmax 的梯度是耦合的

对相似度 $s_{ij}$ 的梯度：

$$\frac{\partial \mathcal{L}_i}{\partial s_{ij}} = \begin{cases} p_{ij} - 1 & j = i \\ p_{ij} & j \neq i \end{cases}, \quad p_{ij} = \text{softmax}_j(s_{ij}/\tau)$$

梯度的分布受 batch 内**所有**其他相似度影响（因为 $p_{ij}$ 的分母包含全部）。负样本的数量变化、batch 内容变化都会扰动梯度。

### 2.4 SigLIP 的解法（一句话版）

> **既然 softmax 要全局归一化，那就干脆不用归一化——把每个图文 pair 当作一个独立的二分类问题，用 sigmoid 逐对计算损失。**

```text
CLIP 视角:  每张图要从 N 个文本里"选 1 个"（N 分类，全 batch 耦合）
SigLIP 视角: 每对 (图, 文) 单独判断"匹不匹配"（二分类，逐对独立）
```

---

## 三、Sigmoid Loss 数学定义

### 3.1 形式化定义

设 batch 内有 N 对图文，图像编码器输出 $v_i \in \mathbb{R}^d$，文本编码器输出 $t_j \in \mathbb{R}^d$（均已 L2 归一化）。

**相似度矩阵**：

$$S_{ij} = v_i^T t_j \in [-1, 1]$$

**标签矩阵**（逐对二分类标签）：

$$Y_{ij} = \begin{cases} +1 & i = j \text{（正样本对，匹配）} \\ -1 & i \neq j \text{（负样本对，不匹配）} \end{cases}$$

> **注意标签用 ±1 而不是 0/1**，这是为了数值技巧（见第五节）。

**SigLIP 损失**：

$$\mathcal{L} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma(Y_{ij} \cdot S_{ij})$$

其中 $\sigma$ 是 sigmoid：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 3.2 展开等价形式

当 $Y_{ij} = +1$（正样本对）：

$$\mathcal{L}_{ij} = -\log \sigma(S_{ij}) = -\log \frac{1}{1+e^{-S_{ij}}} = \log(1 + e^{-S_{ij}})$$

当 $Y_{ij} = -1$（负样本对）：

$$\mathcal{L}_{ij} = -\log \sigma(-S_{ij}) = \log(1 + e^{S_{ij}})$$

**合写**：

$$\mathcal{L}_{ij} = \log(1 + e^{-Y_{ij} S_{ij}})$$

这就是 BCE（Binary Cross-Entropy）在标签为 ±1 时的紧凑写法（softplus 形式）。

### 3.3 与标准 BCE（0/1 标签）的关系

标准二分类交叉熵（标签 0/1）：

$$\mathcal{L} = -[y \log \sigma(s) + (1-y) \log(1-\sigma(s))]$$

令 $y \in \{0,1\}$，替换 $y' = 2y - 1 \in \{-1,+1\}$，利用 $\sigma(-x) = 1 - \sigma(x)$：

$$-[y \log \sigma(s) + (1-y)\log(1-\sigma(s))] = -\log \sigma(y' \cdot s)$$

两者完全等价。**±1 标签写法更简洁、数值更稳定。**

### 3.4 数值示例

假设 batch size N=3，相似度矩阵：

$$S = \begin{bmatrix} 2.0 & -0.5 & -1.0 \\ -0.3 & 1.8 & -0.7 \\ -0.8 & -0.2 & 2.1 \end{bmatrix}$$

标签矩阵 $Y = 2I - 1$（对角线 +1，其余 -1）。

逐对损失 $\mathcal{L}_{ij} = \log(1 + e^{-Y_{ij} S_{ij}})$：

| 位置 | $S_{ij}$ | $Y_{ij}$ | $Y \cdot S$ | 损失 |
|------|---------|----------|-------------|------|
| (1,1) | 2.0 | +1 | 2.0 | log(1+e⁻²) = 0.127 |
| (1,2) | -0.5 | -1 | 0.5 | log(1+e^0.5) = 0.474 |
| (1,3) | -1.0 | -1 | 1.0 | log(1+e¹) = 0.313 |
| (2,1) | -0.3 | -1 | 0.3 | log(1+e^0.3) = 0.554 |
| (2,2) | 1.8 | +1 | 1.8 | log(1+e⁻¹·⁸) = 0.142 |
| (2,3) | -0.7 | -1 | 0.7 | log(1+e^0.7) = 0.403 |
| (3,1) | -0.8 | -1 | 0.8 | log(1+e^0.8) = 0.371 |
| (3,2) | -0.2 | -1 | 0.2 | log(1+e^0.2) = 0.599 |
| (3,3) | 2.1 | +1 | 2.1 | log(1+e⁻²·¹) = 0.120 |

总损失 = 3.103 / 9 = **0.345**

观察：正样本对（相似度高）损失小，负样本对（相似度低、Y·S 为正）损失也不大——模型只需把每个 pair 判对方向即可，不需要和谁比较。

---

## 四、SigLIP vs CLIP：数学本质对比

### 4.1 损失对比表

| 维度 | CLIP (InfoNCE) | SigLIP (Sigmoid) |
|------|----------------|------------------|
| 数学形式 | $-\log \dfrac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ | $\log(1 + e^{-Y_{ij} S_{ij}})$ |
| 归一化 | 行内 softmax 全局归一化 | 无归一化 |
| 每个 pair 的损失是否独立 | 否（分母耦合所有负样本） | **是（完全独立）** |
| 标签 | 每行唯一正样本（soft 分布） | 每个 pair 硬标签 ±1 |
| 负样本来源 | 仅 batch 内 | batch 内 + 可灵活构造 |
| 对 batch 的依赖 | 强 | 弱 |
| 计算复杂度 | O(N²) | O(N²)（但通信少） |
| 温度参数 | 必用 τ（缩放 logits） | 用 b（偏置）+ t（温度） |

### 4.2 梯度对比

**CLIP**（对 $s_{ij}$，固定行 i）：

$$\frac{\partial \mathcal{L}_i}{\partial s_{ij}} = \frac{1}{\tau}(p_{ij} - \mathbb{1}[j=i])$$

梯度同时依赖 batch 内所有 $p_{ij}$（softmax 分母）。

**SigLIP**（对 $S_{ij}$，任意 pair）：

$$\frac{\partial \mathcal{L}_{ij}}{\partial S_{ij}} = -\frac{\partial}{\partial S_{ij}} \log \sigma(Y_{ij} S_{ij}) = \sigma(-Y_{ij} S_{ij}) \cdot (-Y_{ij})$$

对正样本（Y=+1）：梯度 $= \sigma(-S_{ij}) - 1 = -\sigma(S_{ij}) \to $ 增大相似度；
对负样本（Y=-1）：梯度 $= \sigma(S_{ij}) \to $ 减小相似度。

**关键性质：每个 pair 的梯度只取决于该 pair 自己的相似度**——梯度计算完全解耦。

### 4.3 理论等价性（面试加分项）

考虑逐对 sigmoid 损失 + 温度：

$$\mathcal{L}_{ij} = \log(1 + e^{-t \cdot S_{ij} \cdot Y_{ij}})$$

其中 t 是温度（scale）。当 batch size N 趋于无穷大时（或等效地，让每个正样本面对无穷多个均匀分布的负样本），softmax 归一化退化为"与平均负样本比较"，此时：

$$\lim_{N \to \infty} \text{InfoNCE 与 Sigmoid 损失在最优解处等价}$$

直觉：softmax 的行归一化在负样本无穷多时，等于"正样本分数 - log(平均负样本分数)"，而 sigmoid 的决策边界也是 $\sigma(ts) = 0.5 \Leftrightarrow s = 0$——两者在"正样本必须高于负样本"这一核心语义上一致。区别在于有限 batch 下的耦合程度。

> **面试记忆点**：两者的目标都是"让匹配对相似度高于不匹配对"，但 CLIP 用"比赛排名"实现（softmax），SigLIP 用"逐个判断"实现（sigmoid）。

### 4.4 比喻

- **CLIP**：一场考试，每道题 N 个选项，只能选一个正确答案，选 A 就意味着排除了 B、C、D…，选项间完全竞争（耦合）。
- **SigLIP**：N 道独立判断题，每道问"这对图文匹配吗？"答对/答错互不影响（解耦）。

---

## 五、数值稳定性：log-sigmoid 技巧（工程核心）

### 5.1 朴素实现的问题

$$\mathcal{L} = -\log \sigma(Y \cdot S)$$

直接计算会有两个问题：

1. **下溢**：当 $Y \cdot S$ 很大（如 +30），$\sigma(30) = 1 - 4\times10^{-14}$，计算机里 $\sigma(x)$ 可能被舍入为 1，$\log 1 = 0$，梯度消失；
2. **上溢**：当 $Y \cdot S$ 很负（如 -30），$e^{30}$ 在 FP32 中没问题但接近极限（FP16 直接溢出为 inf）。

### 5.2 log-sigmoid 恒等变换

$$\log \sigma(x) = x - \log(1 + e^x) = -\log(1 + e^{-x})$$

推导：$\sigma(x) = \frac{1}{1+e^{-x}}$，两边取对数：

$$\log \sigma(x) = -\log(1 + e^{-x})$$

这个形式叫 **softplus 的负值**（$-\text{softplus}(-x)$），数值上是稳定函数：
- $x \to +\infty$：$\log(1+e^{-x}) \to 0$（不会因为 log(1) 而完全丢失信息，IEEE 浮点给出正确的次正规数）；
- $x \to -\infty$：$\log(1+e^{-x}) \approx -x$（线性，不溢出）。

### 5.3 代码实现对比

```python
import torch
import torch.nn.functional as F

def siglip_loss_naive(logits, labels):
    """朴素实现：可能数值不稳定"""
    probs = torch.sigmoid(logits)          # 可能舍入为 0 或 1
    loss = -labels * torch.log(probs + 1e-8) - (1 - labels) * torch.log(1 - probs + 1e-8)
    return loss.mean()

def siglip_loss_stable(logits, labels):
    """稳定实现：标签 ±1，直接用 logsigmoid"""
    # logits: [N, N]，labels: [N, N] ∈ {-1, +1}
    loglik = F.logsigmoid(labels * logits)     # 恒等变换后的数值稳定形式
    return -loglik.mean()
```

> **💡 项目对应**：项目 04_SigLIP/model/model.py 中的实现正是这个稳定版本：
> ```python
> labels = 2 * torch.eye(b) - torch.ones_like(logits)   # +1/-1 标签
> loglik = F.logsigmoid(labels * logits)
> loss = -loglik.sum(dim=-1).mean()
> ```
> （源码里是 `-torch.sum(loglik, dim=-1)` 后 mean，即对每行求和后平均，与 `-loglik.mean()` 等价——求和后取 mean 只差一个常数因子 N，训练效果一致。）

### 5.4 数值范围分析

| 输入 x = Y·S | 朴素 sigmoid | log(σ(x)) | 稳定式 -log(1+e⁻ˣ) |
|-------------|-------------|-----------|-------------------|
| 10 | 0.99995 | ≈ -4.5e-5 | -4.5e-5（正确） |
| 30 | 1.0000（FP16 舍入） | **0（梯度丢失）** | -3.0e-14（保留） |
| -10 | 4.5e-5 | ≈ -10 | ≈ -10（正确） |
| -30 | 9.4e-14 | ≈ -30 | ≈ -30（线性，不溢出） |

---

## 六、温度与偏置参数（z 参数化）

### 6.1 SigLIP 的独特设计：learnable bias

CLIP 只有温度 $\tau$，SigLIP 除了温度还引入了**可学习的偏置 b**：

$$z_{ij} = t \cdot S_{ij} + b$$

- $t > 0$：温度（可学习，初始约 $e^{2.659}$ ≈ 14.3，即 CLIP 的 1/0.07）；
- $b$：偏置（可学习，初始为 0）。

### 6.2 偏置的语义

$$z_{ij} = t \cdot S_{ij} + b = 0 \ \Leftrightarrow \ S_{ij} = -\frac{b}{t}$$

偏置**移动了决策边界**：默认情况下（b=0）相似度 S>0 判匹配、S<0 判不匹配；b 非零时边界偏移。

论文解释：CLIP 里的温度必须同时完成两件事——"拉大正负样本间距"（scale）和"校准概率"（bias）；SigLIP 把 scale 和 bias 解耦，优化更容易。另一个视角：b 相当于给负样本（占大多数，N²-N 个）一个**先验偏移**，因为负样本比例远大于正样本（1:N-1），模型天然倾向输出低分，b 补偿这种不平衡。

### 6.3 温度的正则化

论文发现直接学习 $t$ 会过早退化为小温度（训练不稳定），因此加了一个**正则项**：

$$\mathcal{L}_{reg} = \lambda \cdot (\log t - \log t_0)^2$$

约束温度保持在初始值附近，用较小的正则强度 $\lambda$ 即可。实践中多数复现直接固定或仅微调温度，bias 单独可学。

### 6.4 代码形式

```python
self.t = nn.Parameter(torch.randn(1))     # 可学习温度
self.b = nn.Parameter(torch.randn(1))     # 可学习偏置

logits = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
```

（注：源码用 `self.t.exp()` 保证温度恒为正，训练中不会出现负温度导致梯度方向翻转。）

---

## 七、正负样本加权（Positive/Negative Balancing）

### 7.1 问题：负样本天然占多数

一个 batch 有 N² 个 pair，其中只有 N 个正样本、N²-N 个负样本。**负样本占比 1 - 1/N**，当 N=4096 时 99.98% 是负样本。如果不加权，损失被负样本主导，模型会偏向"一刀切判负"。

### 7.2 SigLIP 的加权方案

论文对正负样本引入权重：

$$\mathcal{L} = -\frac{1}{N} \sum_{i} \left[ \alpha \log \sigma(t \cdot S_{ii} + b) + \sum_{j \neq i} \beta \log \sigma(-(t \cdot S_{ij} + b)) \right]$$

- $\alpha$：正样本权重；
- $\beta$：负样本权重；
- 论文实验显示 $\alpha = 1$（不放大正样本）时效果最好，即权重并不是越高越好；
- 更一般的做法是给每个 pair 一个权重 $\omega_{ij}$，支持 hard negative mining 等扩展。

### 7.3 为什么 SigLIP 能自然处理不平衡

因为 sigmoid 是逐对的：**每个 pair 独立算梯度，负样本多只意味着负样本对的总梯度贡献多，但每对梯度独立**。而 softmax 中负样本共同争夺归一化分母，不平衡会在耦合中放大。SigLIP 可以通过权重直接精确控制正负样本的相对影响——这是"可控性"的核心卖点。

---

## 八、完整训练流程

### 8.1 数据准备

- **数据集**：WebLI（Google 的 10B+ 图文对，SigLIP 训练用约 13B 子集）；中文场景常用 MUGE、Wukong 等；
- **图像侧增强**：RandAugment 等（分辨率 224/256/384 等）；
- **文本侧**：tokenizer 编码，最大长度 64/77，padding + truncation；
- **配对比率**：论文用 1:1（一个图像一个文本），也支持多文本配比。

### 8.2 训练配置（论文）

| 配置 | SigLIP 论文值 |
|------|--------------|
| Batch size | 4096（vs CLIP 的 32768） |
| 优化器 | AdamW |
| 学习率 | 1e-3（大模型 3e-4） |
| 温度初始 | logit_scale ≈ 14.3 |
| 训练步数 | 通常 10 万步级别 |
| 精度 | bfloat16 / mixed precision |
| 分布式 | FSDP / DDP + 局部 batch 独立算损失 |

### 8.3 为什么小 batch 也能训练好

CLIP 的效果依赖大 batch 是因为负样本来自 batch 内；SigLIP 每个 pair 独立计算，负样本即使少也能给出正确的梯度方向。而且论文表明：**小 batch 下 SigLIP 甚至比同 batch 的 CLIP 更好**（因为 CLIP 小 batch 时负样本严重不足）。

### 8.4 训练流程文字版

```text
1. 采样一个 batch：N 张图 + N 条文本
2. 图像 → ViT → 图像特征 (N×d) → L2 归一化
3. 文本 → Text Transformer → 文本特征 (N×d) → L2 归一化
4. S = V·Tᵀ (N×N)，z = t·S + b
5. 标签 Y = 2I - 1（对角线 +1 其余 -1）
6. loss = -mean(logsigmoid(Y ⊙ z))
7. backward + optimizer.step（AdamW + 温度正则）
8. 同步：DDP 下每卡独立计算 loss，仅梯度 all-reduce
```

> **💡 对比 CLIP 的分布式**：CLIP 需要 all-gather 所有卡的图像/文本特征才能算 softmax；SigLIP 每卡只算自己的 N×N 矩阵，通信量少一个量级。

---

## 九、推理流程

### 9.1 零样本分类

```text
输入图像 + K 个类别文本
  ↓
图像 → ViT → v (D 维, L2 归一化)
每个类别文本 → Text Encoder → t_k (D 维, L2 归一化)
  ↓
score_k = v · t_k（或 z = t·score + b 再 sigmoid）
  ↓
prediction = argmax_k(score_k)
```

SigLIP 的零样本分类与 CLIP 完全一致——**损失函数只影响训练，不影响推理接口**。这也是 SigLIP 能无缝替换 CLIP 的原因。

### 9.2 图文匹配/检索

```text
query 文本 → t_q
候选图库 → {v_1, ..., v_M}（预计算、离线入库）
  ↓
score_i = t_q · v_i → Top-K 排序
```

### 9.3 阈值语义

Sigmoid 概率 $\sigma(t \cdot s + b)$ 天然有"匹配概率"含义（0~1），配合阈值可以做**可校准的图文一致性判断**——比 CLIP 的相似度打分更直觉。实际使用中仍需按业务数据校准阈值。

---

## 十、模型变体

### 10.1 官方 SigLIP 系列

| 模型 | Patch | 图像分辨率 | Params | Embedding Dim | 说明 |
|------|-------|-----------|--------|---------------|------|
| siglip-base-patch16-224 | 16 | 224 | 87M | 768 | 轻量 |
| siglip-large-patch16-224 | 16 | 224 | 307M | 1024 | 通用 |
| siglip-large-patch16-384 | 16 | 384 | 307M | 1024 | 高分辨率 |
| siglip-so400m-patch14-224 | 14 | 224 | 400M | 1152 | 高精度（SigLIP 主力） |
| siglip-so400m-patch14-384 | 14 | 384 | 400M | 1152 | 高分辨率高精度 |
| siglip2-so400m-patch14-384 | 14 | 384 | 400M | 1152 | SigLIP 2（见 10.3） |

### 10.2 双塔与单塔

- 标准 SigLIP 是**双塔**（独立 ViT + 文本 Transformer）；
- 论文还做了**单塔（merged）**实验：把图像 patch 和文本 token 拼成一条序列一起过 Transformer，image-text 语义建模更强（在部分多模态任务上更好），但推理时无法分离塔。

### 10.3 SigLIP 2（2025）

| 改进 | 内容 |
|------|------|
| 训练目标 | 对比 + 自蒸馏（self-distillation）+ 掩码重建多目标联合 |
| 数据 | 加入视频、OCR 数据、德语等 |
| 分辨率 | 最高 384，支持 NaFlex（非方形分辨率） |
| 成果 | 在检索、zero-shot 分类、VLM 视觉塔等全面超越 SigLIP 1 |

---

## 十一、与相关模型的关系

### 11.1 CLIP（基础对比对象）

| | CLIP | SigLIP |
|--|------|--------|
| 损失 | InfoNCE（softmax） | Sigmoid（逐对） |
| 论文 batch | 32768 | 4096 |
| 训练成本 | 高 | 约低 3.5 倍 |
| 通信 | all-gather 全局特征 | 每卡独立 |
| 温度 | 固定 0.07 | 可学习 + 可学习偏置 |
| 权重平衡 | 不可控 | 可精确控制正负权重 |

### 11.2 BLIP 系列

- BLIP：对比 + 匹配（ITM）+ 生成（LM）三目标联合（见 06_BLIP）；
- BLIP-2：冻结视觉塔 + Q-Former 桥接（见 07_BLIP2）；
- **SigLIP 与 BLIP 定位不同**：SigLIP 是纯对齐塔，BLIP 是通用多任务框架。

### 11.3 CoCa（对比 + 生成）

CoCa 把对比损失和 caption 生成损失联合训练：image encoder 输出同时用于对比（CLS）和生成（decoder 输入）。SigLIP 论文对比实验显示在大多数任务上 SigLIP ≥ CoCa（且训练更简单）。

### 11.4 VLM 视觉塔生态

| VLM | 视觉塔 |
|-----|--------|
| LLaVA-1.5 | CLIP ViT-L/14@336 |
| LLaVA-NeXT | CLIP ViT-L/14 多尺度 |
| **PaliGemma** | **SigLIP（+Gemma LLM）** |
| Qwen2-VL | 自研 ViT 675M |
| InternVL | InternViT-6B |
| MiniCPM-V | SigLIP 变体 |

---

## 十二、在 VLM 中的工程应用

### 12.1 作为视觉塔的标准接入

```text
图像 → SigLIP ViT → patch 特征 (N×d) → 投影层 → (N×H_LLM) → LLM
```

注意两种用法：
- **对齐任务**（检索/匹配）：取 [CLS] pooler 输出（单向量）；
- **生成任务**（VLM）：取全部 patch token 输出（逐 token）。

### 12.2 PaliGemma 案例

PaliGemma（Google, 2024）：SigLIP-so400m 视觉塔 + Gemma-2B LLM，经两阶段训练（预训练对齐 → 多任务微调），在 VQAv2、COCO Caption、OCR 等任务上表现突出。证明了 **SigLIP 作为视觉塔的工程可行性**。

### 12.3 微调建议

| 场景 | 做法 |
|------|------|
| 对齐微调（电商图文等） | 冻结或低 lr 微调双塔，sigmoid loss 不变 |
| VLM 视觉塔 | 通常冻结，只训投影层 + LLM |
| 域适应 | 继续用 sigmoid loss 在域内数据训练 |

---

## 十三、完整代码实现

### 13.1 手写 Sigmoid Loss（核心函数）

```python
import torch
import torch.nn.functional as F

def sigmoid_loss(image_embeds, text_embeds, temperature=None, bias=None):
    """
    完整实现 SigLIP 损失
    Args:
        image_embeds: [N, d] 图像特征（未归一化也可）
        text_embeds:  [N, d] 文本特征
        temperature:  [1] 可学习温度（>0），None 则用常数 1
        bias:         [1] 可学习偏置，None 则为 0
    Returns:
        loss 标量
    """
    # 1. L2 归一化 → 余弦相似度
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 2. 相似度矩阵
    logits = image_embeds @ text_embeds.t()          # [N, N]

    # 3. 温度与偏置（z 参数化）
    if temperature is not None:
        logits = logits * temperature.exp()           # 保证温度为正
    if bias is not None:
        logits = logits + bias

    # 4. ±1 标签
    N = logits.size(0)
    labels = 2 * torch.eye(N, device=logits.device) - 1   # 对角 +1，其余 -1

    # 5. 数值稳定的 log-sigmoid
    loglik = F.logsigmoid(labels * logits)
    return -loglik.mean()
```

### 13.2 训练脚本骨架（对齐 HF Trainer 风格）

```python
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch.nn as nn

class SigLIPTrainer(nn.Module):
    def __init__(self, vision_model, text_model, dim=768):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(vision_model)   # 图像塔
        self.text_model = AutoModel.from_pretrained(text_model)       # 文本塔
        self.t = nn.Parameter(torch.randn(1))   # 温度
        self.b = nn.Parameter(torch.randn(1))   # 偏置

    def forward(self, pixel_values, input_ids, attention_mask):
        v = self.vision_model(pixel_values=pixel_values)[1]      # pooler 输出
        t = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]

        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        logits_per_image = v @ t.t() * self.t.exp() + self.b
        N = logits_per_image.size(0)
        labels = 2 * torch.eye(N, device=logits_per_image.device) - 1

        loglik = F.logsigmoid(labels * logits_per_image)
        loss = -loglik.mean()
        return loss, logits_per_image
```

### 13.3 HuggingFace 推理

```python
from transformers import AutoModel, AutoProcessor
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

inputs = processor(images=image, text=["a photo of a cat", "a photo of a dog"],
                   padding="max_length", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image        # [1, 2]
probs = torch.sigmoid(logits)            # SigLIP 用 sigmoid 而非 softmax
```

### 13.4 关键实现细节清单

1. 标签用 ±1（配合 logsigmoid）；
2. 温度用 `exp()` 保证正数；
3. 偏置可学习；
4. 特征必须 L2 归一化；
5. 损失对 N² 个 pair 求平均（或先按行求和再对 N 平均）；
6. 温度可加正则（约束在初值附近）。

---

## 十四、常见误区

**误区 1：SigLIP 是新的模型架构。**
错。图像/文本编码器与 CLIP 几乎一样，创新 100% 在损失函数。面试时强调"架构无创新，创新在损失"。

**误区 2：SigLIP 完全取代 CLIP。**
不完全是。SigLIP 训练效率更高、效果相当或更好，但 CLIP 生态成熟（预训练模型、下游工具丰富）。迁移时 SigLIP 是更优选择，但谈不上"取代"。

**误区 3：Sigmoid loss 就是简单地把 softmax 换 sigmoid。**
换损失之外还有：可学习温度 + 可学习偏置（z 参数化）、log-sigmoid 数值稳定、正负样本加权、温度正则化等工程细节。

**误区 4：SigLIP 不需要大 batch，所以单卡就能训练。**
对 batch 依赖弱 ≠ 单卡可训。4096 batch 仍然需要多卡/梯度累积。只是相比 CLIP 的 32768 门槛低了一个量级。

**误区 5：SigLIP 的零样本分类一定比 CLIP 好。**
不一定。零样本效果还取决于数据规模、模型大小、prompt 设计。论文结论是同等条件下 SigLIP 效率高、效果不差，在 WebLI 上的多个任务优于 CLIP 同规模模型。

**误区 6：sigmoid 输出概率可以直接当"图文匹配概率"用。**
概率是"相对匹配度"，未经过业务数据校准。生产环境必须用真实正负样本对校准阈值（构建 ROC 曲线选阈值）。

---

## 十五、高频面试问答

**Q1：SigLIP 全称和核心创新？**
Sigmoid Loss for Language-Image Pre-training。核心创新：把 CLIP 的 softmax 对比损失换成逐对 sigmoid 二分类损失，训练解耦、对 batch size 依赖弱、通信开销小。

**Q2：为什么 sigmoid loss 对 batch size 依赖更弱？**
softmax 需要 batch 内所有负样本参与归一化，batch 小负样本少 → 区分度差；sigmoid 逐对独立计算，每个 pair 的梯度和损失只依赖自己，负样本少也不影响梯度方向正确性。训练效果不再被 batch size 卡脖子。

**Q3：SigLIP 的标签为什么用 ±1 而不是 0/1？**
为了 log-sigmoid 数值稳定写法：$-\log\sigma(Y \cdot S)$ 比 $-y\log\sigma(s) - (1-y)\log(1-\sigma(s))$ 更紧凑，且避免了 $\log(0)$ 与 $\log(1)$ 的数值问题。两者数学等价。

**Q4：解释 SigLIP 的温度和偏置参数？**
z = t·S + b。温度 t 控制对比力度（同 CLIP 的 τ），偏置 b 移动决策边界，补偿正负样本不平衡。两者可学习，训练时对温度加正则保持稳定。

**Q5：SigLIP 和 CLIP 分布式训练的区别？**
CLIP 需要 all-gather 全局特征算 softmax 归一化；SigLIP 每张卡只用本地 batch 算损失和梯度，通信量大幅减少，这也是训练加速的来源之一。

**Q6：如何用 SigLIP 做 zero-shot 分类？**
候选类别构造 prompt 文本 → 文本塔编码 → 与图像特征算余弦相似度 → argmax。推理接口与 CLIP 完全一致，损失只影响训练。

**Q7：SigLIP 在 VLM 中怎么用？**
做视觉塔：图像 → SigLIP ViT → patch token（或 CLS 向量）→ 投影层 → LLM。PaliGemma 是最典型的案例。

**Q8：SigLIP 的 log-sigmoid 技巧是什么？为什么重要？**
$\log\sigma(x) = -\log(1+e^{-x})$（softplus 负形式），避免 sigmoid 输出被舍入为 0/1 导致梯度消失或溢出，FP16/BF16 训练下尤其重要。

**Q9：SigLIP 和 BLIP/CoCa 的关系？**
BLIP 是多目标框架（对比+匹配+生成），CoCa 是对比+生成联合。SigLIP 专注对齐任务本身，用更优的损失函数把"对齐"这一步做到最好，可以嵌入这些框架作为对齐组件。

**Q10：SigLIP 有什么缺点？**
- 纯对齐模型，不能生成文本（需要接 LLM）；
- 细粒度/计数/空间关系仍有 CLIP 通病；
- 阈值需业务校准；
- 高质量预训练模型主要来自 Google（开源但体积大）。

---

## 十六、自我检验

- [ ] 能写出 SigLIP 损失公式并解释每个符号
- [ ] 能推导 ±1 标签与标准 BCE 的等价性
- [ ] 能解释 log-sigmoid 数值稳定技巧的原理
- [ ] 能说清 CLIP 的三大痛点与 SigLIP 的对应解法
- [ ] 能解释温度与偏置参数的作用
- [ ] 知道正负样本加权的必要性
- [ ] 能画出 SigLIP 训练流程图（8 步）
- [ ] 知道 SigLIP 分布式训练与 CLIP 的差异
- [ ] 能写出手写 sigmoid loss 的稳定实现
- [ ] 能说出 SigLIP 2 的核心改进
- [ ] 能解释 SigLIP 在 VLM 中的两种用法
- [ ] 能区分 6 个常见误区
- [ ] 能完整回答 10 个面试追问

---

## 参考文献

1. [Sigmoid Loss for Language-Image Pre-Training](https://arxiv.org/abs/2303.15343) — Zhai et al., ICCV 2023
2. [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786) — Google, 2025
3. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., ICML 2021
4. [PaliGemma: A 3B VLM with Transferable Generalist Capabilities](https://arxiv.org/abs/2407.07726) — Google, 2024
5. [HuggingFace SigLIP 文档](https://huggingface.co/docs/transformers/model_doc/siglip)
