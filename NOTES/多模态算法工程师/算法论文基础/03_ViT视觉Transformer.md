# ViT 视觉 Transformer：图像如何变成 Token

## 一、为什么需要 ViT

### 1.1 CNN 的三个本质局限

| 局限 | 说明 |
|------|------|
| 感受野受限 | 卷积是局部操作，全局信息需堆叠几十层才能覆盖 |
| 归纳偏置太强 | 平移不变性/局部性假设，限制了表达灵活性 |
| 难以全局建模 | 长距离依赖需要深层，但深层又带来训练困难 |

以 224×224 输入、3×3 卷积为例：单层感受野只有 3×3，堆 10 层约 21×21，要覆盖全图需要 50+ 层。而 Attention 一步就可以让任意两个像素直接交互。

### 1.2 ViT 的核心思想

> **把图像当成"词的序列"**：将图像切成固定大小的 patch，每个 patch 展平后线性投影为一个 token，然后完全按 NLP Transformer 的方式处理。

$$224 \times 224 \div (16 \times 16) = 196 \text{ 个 patch} + 1 \text{ 个 [CLS]} = 197 \text{ 个 token}$$

ViT 证明了：**只要数据量足够大（如 JFT-300M），纯 Transformer 在视觉任务上能超过 CNN**；数据量小时（如 ImageNet-1K），CNN 仍有优势（归纳偏置更适合小数据）。

---

## 二、整体架构

```text
输入图像 (224×224×3)
      │
      ▼
┌─────────────────┐
│ Patch Embedding │  切成 16×16 patch → 线性投影 → 196 个 token
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ [CLS] token     │  在序列头部拼接一个可学习 token
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 位置编码 (可学习) │  197 × d
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Transformer Encoder ×L │  Pre-LN + 多头注意力 + FFN
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  取 [CLS] 输出   │ → 分类头（线性层）→ 类别概率
└─────────────────┘
```

### 关键设计选择

1. **Patch 化**：图像 → 非重叠 patch 网格（16×16 或 14×14）；
2. **[CLS] token**：分类用；没有 [CLS] 时也可以对全部 patch 输出做 mean pooling；
3. **可学习位置编码**：196+1 个位置，每个位置一个可学习向量；
4. **Encoder-only**：ViT 只用 Transformer 的 Encoder 部分（双向注意力）。

---

## 三、Patch Embedding 详解

### 3.1 数学形式

设图像 $x \in \mathbb{R}^{H \times W \times C}$，patch 大小 $P \times P$：

$$N = \frac{HW}{P^2} \text{ 个 patch}, \quad x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

每个 patch 展平后乘可学习矩阵 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$：

$$z_0 = [x_{cls}; \ x_p^1 E; \ x_p^2 E; \ \dots; \ x_p^N E] + E_{pos}$$

- $D$：token 维度（如 768）；
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$：位置编码；
- $x_{cls}$：可学习的 [CLS] token。

### 3.2 数值示例（ViT-Base）

| 参数 | 值 |
|------|-----|
| 输入 | 224×224×3 |
| Patch | 16×16 |
| Patch 数量 | (224/16)² = 196 |
| Token 维度 D | 768 |
| Embedding 层参数 | 16×16×3×768 ≈ 590K |

### 3.3 实现方式（两种等价写法）

```python
# 方式一：卷积实现（官方）
self.proj = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
x = self.proj(x).flatten(2).transpose(1, 2)   # [B, 196, 768]

# 方式二：手动 reshape + Linear
x = x.view(B, 3, 14, 16, 14, 16)   # 切 patch
x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, 196, 16*16*3)
x = self.linear(x)
```

> **面试点**：patch embedding 等价于 stride=P 的卷积（卷积核 = patch 大小）。所以 ViT 第一层也可以用 Conv 实现，工程上更高效（Conv 底层实现优化好）。

---

## 四、位置编码

ViT 使用**可学习位置编码**：初始化一个 $(N+1) \times D$ 的矩阵随训练更新。

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
```

### 4.1 为什么 ViT 不用正弦编码？

论文实验显示：**可学习位置编码与正弦编码效果相当**。但 ViT 的可学习编码有一个特性：**测试时图像分辨率可以比训练时高**（patch 数量变化），此时位置编码数量对不上。

### 4.2 分辨率外推问题（多模态中的大坑）

VLM 常需要高分辨率输入（如 336/384/448/512 甚至更高）。如果模型在 224 训练，测试 448，patch 数变成 4 倍，位置编码不够用。

常见处理方式：
1. **插值（Interpolation）**：把 196 个位置编码插值成更多个（双线性插值），微调时适应；
2. **2D 分解**：把 1D 位置编码分解为 row + col 两个可学习编码（如 14 行 + 14 列），任意分辨率组合；
3. **多尺度位置编码 / 相对位置**：Swin、Qwen-VL 等用的方案。

> 这正是一代代 VLM（Qwen-VL 的动态分辨率等）不断改进位置编码的原因。

---

## 五、Transformer Encoder 与 NLP 的区别

ViT 的 Encoder 与 NLP Transformer 几乎完全相同，只有两个差异：

| 差异 | NLP | ViT |
|------|-----|-----|
| 输入构建 | token embedding | patch embedding |
| 位置编码 | 1D（词序） | 1D（patch 顺序，可视为 2D 拉平） |

### 5.1 标准 Encoder Block（Pre-LN）

```text
z → LayerNorm → MultiHeadAttention → 残差 → LayerNorm → MLP(GELU) → 残差
```

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

### 5.2 注意力在图像上"看"什么

- 底层头：关注局部纹理、边缘、颜色；
- 中层头：关注物体部件（脸、轮子）；
- 高层头：关注物体级语义、全局结构。

这与 CNN 的低-中-高特征层级有异曲同工之处，但 ViT 的注意力是**全局**的（每个 patch 可以直接看所有 patch）。

---

## 六、[CLS] Token 与分类头

### 6.1 为什么用 [CLS] token

- 借鉴 BERT：最后一个 [CLS] 的输出向量经过全序列信息融合，代表"整张图的语义摘要"；
- 避免引入人为的 pooling 操作，让模型自己决定"什么信息进入分类向量"。

### 6.2 两种取特征方式

| 方式 | 实现 | 适用 |
|------|------|------|
| [CLS] token | 取第 0 个位置输出 | 分类、图文对齐（CLIP/SigLIP 的视觉塔） |
| Mean pooling | 对所有 patch 输出取平均 | 稳定、无位置偏差 |

> **注意**：CLIP/SigLIP 的视觉塔用的是 **pooler 输出**（[CLS] 过一层投影）或 `last_hidden_state[:, 0]`。而多模态生成模型（LLaVA 等）不用 [CLS]，而是把**所有 patch 的 token 输出**作为视觉 token 送入 LLM。

---

## 七、预训练与微调

### 7.1 预训练（Pre-training）

- **数据集**：ImageNet-21K、JFT-300M 等大规模数据；
- **任务**：监督分类（带标签）或自监督（MAE、DINO、CLIP 对比学习）；
- **模型配置**：

| 变体 | Layers | Hidden D | Heads | Params | Patch |
|------|--------|----------|-------|--------|-------|
| ViT-Ti | 12 | 192 | 3 | 5.7M | 16 |
| ViT-S | 12 | 384 | 6 | 22M | 16 |
| ViT-B | 12 | 768 | 12 | 86M | 16 |
| ViT-L | 24 | 1024 | 16 | 307M | 14/16 |
| ViT-H | 32 | 1280 | 16 | 632M | 14 |
| ViT-g | 40 | 1408 | 16 | 1.1B | 14 |
| ViT-G | 48 | 1664 | 16 | 1.8B | 14 |

### 7.2 微调（Fine-tuning）

- 分类头换成新的任务头（或冻结）；
- 学习率远小于预训练（1e-5 ~ 1e-4）；
- 分辨率提升（224 → 384）时需插值位置编码并小 lr 适应。

### 7.3 自监督预训练（多模态项目相关的重点）

| 方法 | 思想 | 代表 |
|------|------|------|
| MAE | 随机遮盖 75% patch，重建像素 | MAE (2022) |
| DINO/DINOv2 | 自蒸馏：student 学 teacher 的输出 | DINOv2 特征极强 |
| **CLIP/SigLIP** | **图文对比学习：图像对齐文本** | 多模态视觉塔标配 |
| EVA/EVA-CLIP | 用 CLIP 蒸馏增强 ViT | EVA-02 等 |

> 多模态项目里"视觉塔"几乎都来自 CLIP/SigLIP/InternViT 预训练——因为它的特征是**语义对齐**的，与文本特征在同一个空间。

---

## 八、ViT 主要变体

### 8.1 层级式（Hierarchical）变体

| 变体 | 核心改进 | 说明 |
|------|---------|------|
| **Swin** | 滑动窗口 + 层级下采样 | 线性复杂度，兼容 CNN 范式，检测分割友好 |
| DeiT | 数据蒸馏训练技巧（teacher-student） | 小数据也能训好 |
| PVT | 金字塔结构 + 空间缩减注意力 | 检测/分割 |
| ConvNeXt | 纯 CNN 借鉴 Transformer 设计 | 证明 CNN 也能追赶 |

### 8.2 效率与架构改进

| 方向 | 方法 | 说明 |
|------|------|------|
| 线性注意力 | Performer、Nyströmformer | O(n) 复杂度 |
| 稀疏注意力 | Swin 窗口、PVT | 局部窗口 + 层次 |
| 卷积+注意力 | CoAtNet、MaxViT | 混合架构 |

### 8.3 多模态视觉塔（重点）

| 模型 | 视觉塔 | 说明 |
|------|--------|------|
| CLIP | ViT-B/L/16、14 | 图文对比预训练，零样本分类强 |
| **SigLIP** | ViT（SO400M/L） | sigmoid loss，训练效率更高（见 05） |
| OpenCLIP | 多种 ViT | 开源复现 |
| **InternViT-6B** | 6B ViT | InternVL 的视觉塔 |
| Qwen2-VL ViT | 675M ViT | patch14 + 动态分辨率 |

---

## 九、ViT vs CNN 深度对比（必考）

| 维度 | CNN | ViT |
|------|-----|-----|
| 感受野 | 局部（需堆叠） | 全局（一步到位） |
| 归纳偏置 | 平移不变性/局部性（强） | 几乎无（弱） |
| 数据需求 | 中（偏置补足数据） | 大（需大数据量） |
| 参数量 | 少 | 多 |
| 计算复杂度 | O(n)（线性） | O(n²)（二次） |
| 图像分辨率扩展 | 容易（全卷积） | 需要处理位置编码 |
| 迁移性 | 好 | 好（大数据预训练后更强） |
| 小数据表现 | 更好 | 容易过拟合 |
| 特征可解释性 | 卷积核直观 | attention map 可可视化 |

**结论**：数据量小用 CNN，数据量大用 ViT；现代大模型（VLM 视觉塔）基本全是 ViT。

---

## 十、多模态中的 ViT：视觉塔的输入输出

### 10.1 双塔结构（CLIP/SigLIP 类）

```text
图像 → ViT → [CLS] → pooler → 视觉 embedding (D 维)     ← 与文本 embedding 对齐
文本 → Transformer → [CLS] → pooler → 文本 embedding (D 维)
```

### 10.2 生成结构（LLaVA/Qwen 类）

```text
图像 → ViT → 全部 patch 输出 (196 × D) → 投影层 → (196 × H_LLM) → 送入 LLM
```

视觉 token 数量 = patch 数量。高分辨率 = 更多 token = LLM 计算量暴涨 → 引出 token 压缩（Qwen 的 visual token merger、InternVL 的 pixel shuffle 等，见 09/10）。

### 10.3 视觉塔的显存与计算

| 视觉塔 | 参数 | 224 输入 token 数 |
|--------|------|------------------|
| ViT-B | 86M | 197 |
| ViT-L | 307M | 257（patch14） |
| ViT-g | 1.1B | 257 |
| InternViT-6B | 6B | 257+ |

---

## 十一、高频面试问答

**Q1：ViT 为什么能把图像当序列处理？**
把图像切成 patch 并线性投影为 token，图像就变成"词的序列"，Transformer 的注意力可以任意建模 patch 之间的关系。本质上图像和文本都是"token 序列"，只是构建方式不同。

**Q2：ViT 相比 CNN 的优缺点？**
优点：全局建模（一步长距离依赖）、大数据下更强的可扩展性（scaling law）、注意力可解释。缺点：无归纳偏置（小数据容易过拟合）、O(n²) 计算、位置编码分辨率外推问题、需要更多数据。

**Q3：ViT 的分类 token 和 mean pooling 哪个好？**
CLS token 是 BERT 风格，让模型学习"摘要向量"；mean pooling 对位置无偏好、更稳定。实践中 CLIP 用 CLS，部分模型用 mean pooling，两者效果相近，视结构而定。

**Q4：分辨率变化时位置编码怎么处理？**
可学习 1D 位置编码通过插值扩展；或用 2D 分解（row+col 分开）；或相对位置编码天然支持任意分辨率（如 Swin）。VLM 中常用"训练时多分辨率 + 位置编码插值/2D 化"。

**Q5：patch size 对模型的影响？**
patch 越小 → token 越多 → 计算量越大，但细节保留更好（如 16→14 提升细节）；patch 大 → token 少 → 快但粗。多模态模型里常用 patch14，高分辨率任务用更小 patch 或叠加 CNN。

**Q6：为什么 CLIP 的视觉塔要经过 [CLS] 池化输出单向量？**
双塔对齐需要"一张图 = 一个向量"来做图文对比；多模态生成模型则保留全部 patch token 给 LLM 使用。任务目标决定了输出形态。

**Q7：ViT 在 VLM 里为什么常被冻结？**
CLIP/SigLIP 预训练的视觉特征已经与文本对齐，冻结可以大幅省显存和训练时间，避免微调破坏对齐特征；只训练投影层和 LLM 足以适配任务（如 LLaVA 的早期版本冻结视觉塔）。

**Q8：MAE 和对比学习预训练的区别？**
MAE 是生成式（重建像素），学到的是"像素级结构"；对比学习（CLIP/DINO）是判别式（区分正负样本），学到的是"语义级特征"。CLIP 类特征天然适合图文对齐；MAE 特征更适合分类等任务。

---

## 十二、自我检验

- [ ] 能画出 ViT 的完整架构图
- [ ] 能写出 patch embedding 的两种实现（Conv / reshape+Linear）
- [ ] 能算出 224 输入、patch16 的 token 数量
- [ ] 能解释 [CLS] token 的作用
- [ ] 知道位置编码外推问题的三种解法
- [ ] 能对比 ViT 和 CNN 的优缺点
- [ ] 知道 ViT-B/L/H 的参数量和层数
- [ ] 能说清 CLIP 视觉塔与 LLaVA 视觉塔的输出差异
- [ ] 了解 MAE/DINO/CLIP 三种预训练范式
- [ ] 知道 Swin 的核心思想（窗口注意力 + 层级）
