---
title: ViT 视觉 Transformer
description: Vision Transformer 的原理、架构与图像分类应用
category: multimodal
tags: [multimodal, vit, transformer, vision, image-classification]
status: stable
order: 2
---

# ViT 视觉 Transformer

## 一句话解释

ViT（Vision Transformer）将 Transformer 架构直接应用于图像，通过将图像切分为固定大小的 patch 序列来处理，证明了纯 Transformer 也能在视觉任务上达到甚至超越 CNN 的效果。

## 它解决什么问题

CNN 是视觉领域的主流架构，但其局部感受野限制了全局信息的获取。Transformer 在 NLP 中已经证明了强大的全局建模能力，但直接将 Self-Attention 应用于图像像素计算量是 $O(n^2)$（$n$ 为像素数），对于 224×224 的图像就是 50176 个 token，计算量不可接受。

ViT 的解决方案：不处理像素，而是处理图像块（patch）。将 224×224 的图像切成 16×16 的 patch，得到 196 个 token，计算量降低 256 倍。

## 核心思想

### 图像 Patch 化

将图像 $x \in \mathbb{R}^{H \times W \times C}$ 切分为 $N = \frac{HW}{P^2}$ 个 patch，每个 patch 展平为向量：

$$
x_p^i \in \mathbb{R}^{P^2 \cdot C}
$$

其中 $P$ 是 patch 大小。对于 224×224 图像、$P=16$：$N = \frac{224 \times 224}{16 \times 16} = 196$ 个 patch。

### 位置编码

由于 Transformer 本身不感知位置，需要为每个 patch 添加位置编码。ViT 使用可学习的位置编码：

$$
z_0^i = x_p^i E + e_{pos}^i
$$

其中 $E \in \mathbb{R}^{(P^2 C) \times D}$ 是线性投影矩阵，$e_{pos}^i$ 是位置编码。

### CLS Token

在 patch 序列前添加一个可学习的 [CLS] token，其最终表示用作整个图像的特征：

$$
z_0 = [z_{cls}; z_0^1; z_0^2; ...; z_0^N]
$$

### 架构流程

```
输入图像 (224×224×3)
   │
   ▼
Patch 化 (16×16) → 196 个 patch
   │
   ▼
线性投影 → 196 个 D 维向量
   │
   ▼
拼接 [CLS] token + 位置编码
   │
   ▼
L 层 Transformer Encoder
   │
   ▼
[CLS] token 输出 → 分类头
```

## 数学定义

### Self-Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

对于 ViT，$Q = K = V = z$，即自注意力。

### 计算复杂度

- CNN：$O(k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$，与卷积核大小 $k$ 相关
- ViT：$O(N^2 \cdot D)$，与 patch 数 $N$ 相关

当图像分辨率增大时，$N$ 线性增长，但注意力计算量是 $O(N^2)$，所以 ViT 在高分辨率图像上计算量较大。

### 模型变体

| 模型 | 层数 | 隐藏维度 | 参数量 | ImageNet Top-1 |
|------|------|----------|--------|----------------|
| ViT-B/16 | 12 | 768 | 86M | 77.9% |
| ViT-L/16 | 24 | 1024 | 307M | 85.2% |
| ViT-H/14 | 32 | 1280 | 632M | 88.6% |

## 代码示例

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并投影到嵌入空间"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()

        # Patch 嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch 嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 拼接 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # 取 CLS token 输出
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x

# 使用示例
model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000)
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(output.shape)  # (2, 1000)
```

## 面试标准回答

**"ViT 和 CNN 的核心区别是什么"**

CNN 通过卷积核在局部区域滑动提取特征，天然具有平移不变性和局部性，但感受野有限，需要堆叠多层才能获取全局信息。ViT 通过 Self-Attention 直接建模所有 patch 之间的关系，第一层就能获取全局信息，但缺乏 CNN 的归纳偏置（局部性、平移不变性），需要更多数据才能训练好。在大数据集上 ViT 通常优于 CNN，小数据集上 CNN 更稳定。

**"ViT 为什么需要位置编码"**

Transformer 的 Self-Attention 是置换不变的——打乱输入顺序，输出不变。但图像的 patch 是有空间关系的（左上角和右下角的 patch 语义完全不同），所以需要位置编码来注入位置信息。ViT 使用可学习的位置编码，训练后模型能学会 patch 之间的空间关系。

**"ViT 的 CLS token 有什么用"**

CLS token 是一个可学习的虚拟 token，拼接在 patch 序列前面。经过多层 Transformer 后，它的表示汇聚了整个图像的信息，用作分类特征。这样做的好处是：(1) 避免对所有 patch 做池化；(2) 分类任务只需要一个向量表示。

## 高频追问

**Q1: ViT 需要多少数据才能训练好？**

ViT 缺少 CNN 的归纳偏置，需要大量数据。原始论文在 ImageNet-21K（1400 万张图片）上预训练后，在 ImageNet-1K 上达到 85%+ 的 Top-1 准确率。如果只用 ImageNet-1K（130 万张）从头训练，效果通常不如 ResNet。这也是为什么 DeiT（Data-efficient Image Transformer）引入了知识蒸馏来降低数据需求。

**Q2: patch_size 怎么选？**

patch_size 越小，patch 数量越多，计算量越大，但能捕获更细粒度的特征。常用 patch_size=16（ViT-B/16）或 14（ViT-B/14）。对于小图像（如 CIFAR-10 的 32×32），可以用 patch_size=4 或 8。

**Q3: ViT 可以用于目标检测和分割吗？**

可以。Swin Transformer 引入了层级结构和滑动窗口注意力，成为视觉任务的通用骨干网络。DETR 将 Transformer 用于目标检测，直接预测边界框和类别，不需要 NMS 后处理。

**Q4: 为什么 ViT 在小数据集上不如 CNN？**

CNN 有强归纳偏置：局部连接（假设相邻像素相关）、权值共享（假设平移不变性）。这些先验知识在数据少时很有帮助。ViT 没有这些偏置，需要从数据中学习，所以数据少时容易过拟合或学不到有效的空间关系。

## 工程实践

### 1. 预训练策略

- 在大规模数据集（ImageNet-21K、JFT-300M）上预训练
- 使用强数据增强（RandAugment、Mixup、CutMix）
- 使用正则化（Stochastic Depth、Label Smoothing）

### 2. 微调技巧

- 微调时使用较小的学习率（比训练时小 10 倍）
- 可以只微调最后几层，冻结前面的层
- 对于小数据集，使用更强的数据增强

### 3. 推理优化

- 使用 Flash Attention 加速注意力计算
- 使用混合精度推理（FP16/BF16）
- 对于高分辨率图像，可以使用滑动窗口注意力

## 常见误区

1. **"ViT 完全替代了 CNN"** — 不是。在数据充足时 ViT 效果好，但 CNN 在小数据集、边缘设备上仍有优势。很多实际系统是 CNN + Transformer 混合架构。

2. **"ViT 的注意力一定是全局的"** — 原始 ViT 是全局注意力，但 Swin Transformer 引入了局部窗口注意力，计算量更小。

3. **"位置编码是固定的"** — ViT 使用可学习的位置编码，不是固定的正弦编码。训练后模型能学会空间关系。

4. **"patch_size 越小效果越好"** — 不一定。太小的 patch 会大幅增加计算量，但效果提升不一定成正比。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Training Data-Efficient Image Transformers](https://arxiv.org/abs/2012.12877)
