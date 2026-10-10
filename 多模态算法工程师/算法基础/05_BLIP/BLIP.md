---
title: BLIP — Bootstrapping Language-Image Pre-training
description: BLIP 的统一视觉语言预训练架构、Bootstrapping 数据清洗机制、三大训练目标与工程实践
category: multimodal
tags: [BLIP, Vision-Language, Pre-training, Captioning, VQA]
status: draft
order: 5
---

# BLIP

## 一句话解释

BLIP（Bootstrapping Language-Image Pre-training）是一种统一视觉语言预训练模型，同时支持图文理解（检索、匹配、问答）和文本生成（图像描述）两类任务，并通过 Bootstrapping 机制自动清洗噪声图文数据。

> 和 CLIP 只做图文对齐不同，BLIP 既能判断图文是否匹配，也能根据图像生成文本。

---

## 1. 它解决什么问题

早期视觉语言模型面临两个核心痛点：

### 1.1 痛点一：理解和生成割裂

在 BLIP 之前，视觉语言模型通常只擅长一类任务：

| 类型 | 代表模型/方法 | 擅长 | 不擅长 |
|------|-------------|------|--------|
| 理解型 | CLIP、ALIGN | 图文匹配、图文检索、零样本分类 | 图像描述生成、开放式问答 |
| 生成型 | 早期 Image Captioning 模型 | 图像描述、文本生成 | 图文检索、精确匹配 |

具体来说：

- **CLIP** 用对比学习把图像和文本编码到同一向量空间，可以算相似度做检索，但它没有文本解码器，无法根据图像生成一句话。
- **传统 Captioning 模型**（如 Show-and-Tell）能根据图像生成描述，但无法做图文检索或判断图文是否匹配。

实际业务中，一个电商系统可能同时需要：

```text
1. 用文字搜图（检索，需要图文对齐）
2. 给商品图自动生成描述（生成，需要文本解码）
3. 判断标题和图片是否一致（匹配，需要图文理解）
```

如果理解型和生成型模型分开用，就需要维护两套模型，成本高且能力不统一。

### 1.2 痛点二：网页图文数据噪声大

大模型训练依赖海量图文对数据。最简单的来源是爬取网页，但网页数据通常很脏：

```text
常见噪声：
- 图片旁边的文本是导航栏、广告、页脚，不是真正的图片描述
- 图片和文字在同一个 HTML 页面上，但语义不相关
- alt text 质量低，很多是占位符或机器翻译
```

CLIP 用的 LAION 数据集虽然规模大（4 亿图文对），但噪声比例很高。用噪声数据训练，模型会学到错误的图文关联。

BLIP 的 Bootstrapping 机制就是为了解决这个问题：用模型自身生成更准确的 caption，再用模型自身过滤不匹配的图文对。

> **💡 Tip**：Bootstrapping 这个词在机器学习中指"自举"——用模型自身的输出来改善训练数据，再用改善后的数据训练更好的模型，如此迭代。Self-Play、Self-Instruct 也是类似思路。

---

## 2. 核心思想

### 2.1 架构统一：MED（Multi-modal Mixture of Encoder-Decoder）

BLIP 的核心设计是用一个统一的架构同时处理理解和生成任务。这个架构叫做 MED（Multi-modal Mixture of Encoder-Decoder），由以下组件构成：

```
┌──────────────────────────────────────────────────────────┐
│                    BLIP 架构（MED）                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   图像 ──→ [Image Encoder (ViT)] ──→ 图像特征 I           │
│                                           │              │
│                        ┌──────────────────┼──────────┐   │
│                        │                  │          │   │
│                        ▼                  ▼          ▼   │
│              [Text Encoder]    [Image-Grounded  [Text   │
│              文本编码器           Text Encoder]   Decoder]│
│                  │              融合编码器         文本解码器│
│                  │                  │              │     │
│                  ▼                  ▼              ▼     │
│              文本特征 T        融合特征 [I;T]    生成文本   │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  任务一（理解）：Text Encoder + Image Encoder → 对齐/匹配    │
│  任务二（理解）：Image-Grounded Text Encoder → 匹配判断      │
│  任务三（生成）：Text Decoder → 图像描述生成                   │
└──────────────────────────────────────────────────────────┘
```

BLIP 的文本部分有三种角色，共享部分参数：

| 组件 | 输入 | 输出 | 对应任务 |
|------|------|------|---------|
| **Text Encoder** | 图像特征 + 文本 token | 文本表征 | 图文匹配、图文检索 |
| **Image-Grounded Text Encoder** | 图像特征 + 文本 token | 融合表征 | 图文匹配（更精细） |
| **Text Decoder** | 图像特征 + 前缀 token | 生成的文本 token | 图像描述、视觉问答 |

关键设计：Text Encoder 和 Text Decoder 共享词嵌入（word embedding），但各有独立的 Transformer 层。这样既能复用参数，又能让理解和生成各有专门的表达能力。

> **💡 Tip**：BLIP 的 Image-Grounded Text Encoder 在 Text Encoder 的基础上加了一个 Cross-Attention 层，让文本能直接 attend 到图像特征。这就是它比纯 CLIP 做匹配更准的原因——CLIP 只能对比全局向量，BLIP 可以做 token 级别的图文交互。

### 2.2 Bootstrapping 数据清洗机制

BLIP 的第二个核心思想是用模型自身来改善训练数据质量。流程如下：

```
原始网页图文对（含噪声）
        │
        ▼
┌───────────────────┐
│   Step 1: Captioner │  ← 用 BLIP 的 Text Decoder 为每张图生成候选 caption
│   生成候选描述       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   Step 2: Filter    │  ← 用 BLIP 的 Image-Grounded Text Encoder
│   过滤噪声对         │     判断原始图文对和生成图文对的匹配分数
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│   Step 3: 拼接数据   │  ← 保留高质量的原始对 + 生成对
│   训练更高质量模型   │
└───────────────────┘
```

具体做法：

| 步骤 | 操作 | 目的 |
|------|------|------|
| Captioner 生成 | 用 BLIP 的解码器为每张图生成 10 个候选 caption | 补充高质量文本描述 |
| Filter 过滤 | 对每对 (图, 文本) 计算匹配分数，设定阈值过滤 | 去除噪声图文对 |
| 数据拼接 | 保留过滤后的原始对 + 生成对，合并训练 | 提升数据质量和规模 |

这就是 Bootstrapping 的含义：用当前模型生成更好的数据，再用更好的数据训练更好的模型。BLIP 论文实验表明，经过 Bootstrapping 处理后的数据训练的模型，在多个下游任务上比直接用原始噪声数据训练的模型提升了 5-10%。

> **💡 Tip**：这个思路后来被 BLIP-2、InstructBLIP 等模型继承。现在的 LLM 训练中，用强模型生成合成数据（Synthetic Data）再训练弱模型，也是类似的 Bootstrapping 思想。

---

## 3. 数学定义

BLIP 有三个预训练目标，分别对应三种能力。

### 3.1 Image-Text Contrastive Loss（ITC）

ITC 用于学习图像和文本的全局对齐，和 CLIP 的对比学习类似。

对于一个 batch 中的 $N$ 个图文对 $(I_i, T_i)$，$i = 1, 2, \dots, N$：

$$\mathcal{L}_{ITC} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

其中：

- $I_i$：第 $i$ 张图像经过 Image Encoder 后的全局表征（CLS token 或平均池化）
- $T_i$：第 $i$ 段文本经过 Text Encoder 后的全局表征（CLS token）
- $\text{sim}(I_i, T_j)$：图像 $I_i$ 和文本 $T_j$ 的余弦相似度（cosine similarity）
- $\tau$：温度参数（temperature），控制分布的尖锐程度，通常可学习
- $N$：batch size

直觉理解：对每个图像，让它的正匹配文本得分最高，其他文本得分被压低。文本方向同理。

### 3.2 Image-Text Matching Loss（ITM）

ITM 用于判断图像和文本是否精确匹配，是一个二分类任务。

$$\mathcal{L}_{ITM} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \sigma(h_i) + (1 - y_i) \log (1 - \sigma(h_i)) \right]$$

其中：

- $h_i$：Image-Grounded Text Encoder 输出的融合表征，经过一个二分类头（Linear + Sigmoid）得到的匹配分数
- $y_i$：标签，匹配为 1，不匹配为 0
- $\sigma$：Sigmoid 函数，$\sigma(x) = \frac{1}{1 + e^{-x}}$

和 ITC 的区别：

| 对比项 | ITC | ITM |
|--------|-----|-----|
| 粒度 | 全局向量对比 | token 级图文交互 |
| 模型 | Text Encoder | Image-Grounded Text Encoder |
| 输出 | 相似度分数 | 二分类概率 |
| 能力 | 粗粒度对齐 | 精细粒度匹配 |

### 3.3 Language Modeling Loss（LM）

LM 用于训练模型根据图像生成文本，是标准的自回归语言建模目标。

$$\mathcal{L}_{LM} = -\sum_{t=1}^{L} \log P(w_t \mid w_1, w_2, \dots, w_{t-1}, I)$$

其中：

- $w_t$：第 $t$ 个文本 token
- $L$：文本序列长度
- $I$：输入图像的特征
- $P(w_t \mid w_1, \dots, w_{t-1}, I)$：给定图像和前 $t-1$ 个 token 时，第 $t$ 个 token 的生成概率

Text Decoder 使用交叉注意力（Cross-Attention）关注图像特征，实现图像条件下的文本生成。

### 3.4 总损失

三个目标联合训练：

$$\mathcal{L} = \mathcal{L}_{ITC} + \mathcal{L}_{ITM} + \mathcal{L}_{LM}$$

> **💡 Tip**：ITC 和 ITM 虽然都是做"匹配"，但粒度不同。ITC 用全局向量快速筛选候选，ITM 用 token 级交互做精确判断。两者互补，类似检索系统中"粗排 + 精排"的设计。

---

## 4. 算法流程

### 4.1 预训练流程

```
┌─────────────────────────────────────────────────┐
│              BLIP 预训练流程                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  输入：图文对数据集 {(I₁,T₁), (I₂,T₂), ...}       │
│                                                  │
│  ┌──────────┐                                    │
│  │ 图像 I    │──→ [Image Encoder (ViT)] ──→ I_emb │
│  └──────────┘              │                     │
│                            │                     │
│  ┌──────────┐              │                     │
│  │ 文本 T    │──→ [Text Encoder] ──→ T_emb        │
│  └──────────┘       │                           │
│                     │                           │
│         ┌───────────┼───────────┐               │
│         ▼           ▼           ▼               │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│    │  ITC    │ │  ITM    │ │   LM    │         │
│    │ 对比损失  │ │ 匹配损失 │ │ 生成损失  │         │
│    └─────────┘ └─────────┘ └─────────┘         │
│         │           │           │               │
│         └───────────┼───────────┘               │
│                     ▼                           │
│              总损失 L = L_ITC + L_ITM + L_LM      │
│                     │                           │
│                     ▼                           │
│              反向传播 + 参数更新                    │
└─────────────────────────────────────────────────┘
```

### 4.2 推理流程

不同任务使用不同的组件组合：

```
任务一：图文检索 / 图文匹配
  图像 ──→ [Image Encoder] ──→ I_emb
  文本 ──→ [Text Encoder] ──→ T_emb
  I_emb 和 T_emb 计算余弦相似度 → 匹配分数

任务二：精细图文匹配（需要更准的判断）
  图像 ──→ [Image Encoder] ──→ I_feat
  文本 ──→ [Image-Grounded Text Encoder + I_feat] ──→ 融合表征
  融合表征 ──→ 二分类头 → 匹配/不匹配

任务三：图像描述生成
  图像 ──→ [Image Encoder] ──→ I_feat
  [Text Decoder + I_feat] 自回归生成文本 token → 输出 caption
```

### 4.3 逐步详解

1. **图像编码**：输入图像经 ViT（Vision Transformer）编码为视觉特征序列，CLS token 作为全局表征
2. **文本编码**：输入文本经 Text Encoder 编码为文本表征，CLS token 作为全局表征
3. **ITC 对比**：计算图像全局表征和文本全局表征的余弦相似度矩阵，用 InfoNCE loss 拉近匹配对
4. **ITM 匹配**：将图像特征和文本 token 序列送入 Image-Grounded Text Encoder，用交叉注意力融合，输出匹配概率
5. **LM 生成**：将图像特征送入 Text Decoder，自回归生成文本 token，用交叉注意力关注图像

> **💡 Tip**：实际使用时，CLIP 做检索快但不生成文本，BLIP 检索稍慢但能生成描述。如果业务只需要相似度计算，用 CLIP；如果需要生成能力，用 BLIP。

---

## 5. 代码示例

### 5.1 基础实现：用 BLIP 做图像描述

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 加载预训练的 BLIP 模型和处理器
# BlipForConditionalGeneration 包含 Image Encoder + Text Decoder
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载一张测试图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 图像描述生成（Image Captioning）
# 不提供文本输入时，模型自动为图像生成描述
inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(output[0], skip_special_tokens=True)
print(f"生成的描述: {caption}")
# 预期输出类似: "a cat sitting on a couch looking at the camera"
```

### 5.2 进阶：用 BLIP 做视觉问答（VQA）

```python
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# 加载 VQA 模型（注意和 captioning 模型不同）
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 加载测试图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 视觉问答：输入图片 + 问题，输出答案
question = "How many cats are there?"
inputs = processor(image, question, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=10)
answer = processor.decode(output[0], skip_special_tokens=True)
print(f"问题: {question}")
print(f"回答: {answer}")
# 预期输出: "2"
```

### 5.3 进阶：用 BLIP 做图文匹配

```python
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image
import requests

# 加载图文匹配模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-text-matching-base")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-image-text-matching-base")

# 加载测试图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 判断图文是否匹配
text = "two cats lying on a couch"
inputs = processor(image, text, return_tensors="pt")
outputs = model.get_image_features(**inputs)
# outputs.logits 包含匹配分数
print(f"匹配分数: {outputs.logits.item():.4f}")
# 分数越高表示越匹配
```

> **💡 Tip**：BLIP 在 HuggingFace 上有三个主要模型变体：`blip-image-captioning-base`（描述生成）、`blip-vqa-base`（视觉问答）、`blip-image-text-matching-base`（图文匹配）。选对模型很重要，它们的架构略有不同。

---

## 6. 面试回答

> BLIP 是 Salesforce 提出的一个视觉语言预训练模型，全称 Bootstrapping Language-Image Pre-training。它的核心特点有两个：一是用一个统一架构同时做图文理解和文本生成，二是用 Bootstrapping 机制自动清洗噪声图文数据。
>
> 具体来说，BLIP 的文本部分有三种角色：Text Encoder 做图文对齐和检索，Image-Grounded Text Encoder 做精细图文匹配，Text Decoder 根据图像生成文本。三个角色共享词嵌入参数，各有独立的 Transformer 层。训练时同时优化三个损失：ITC（对比对齐）、ITM（匹配判断）、LM（语言建模生成）。
>
> 工程上，如果只需要图文相似度计算，CLIP 更快更直接；如果需要根据图像生成描述或做 VQA，BLIP 比 CLIP 更合适。但 BLIP 不是现代 MLLM，它的生成能力有限，实际生产中更多用 BLIP-2 或 InstructBLIP。

---

## 7. 工程实践

### 7.1 场景一：图像描述生成

BLIP 的 Captioning 模型可以为电商商品图、社交媒体图片自动生成文字描述。

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| 商品图描述 | `blip-image-captioning-large` | 输入分辨率建议 384×384 |
| 社交媒体图 | `blip-image-captioning-base` | 可加 prefix prompt 控制描述风格 |
| 批量处理 | 用 `batch_size` 批量推理 | 注意 GPU 显存，7B 图片约需 4GB |

### 7.2 场景二：视觉问答

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| 简单 VQA | `blip-vqa-base` | 问题要简洁明确 |
| 文字型 VQA | 需要 OCR 预处理 | BLIP 对复杂 OCR 场景能力有限 |
| 多轮 VQA | 需要拼接上下文 | BLIP 不原生支持多轮对话 |

### 7.3 场景三：图文检索

| 场景 | 推荐方案 | 注意事项 |
|------|----------|----------|
| 文本搜图 | CLIP/SigLIP 更适合 | BLIP 检索不如 CLIP 高效 |
| 精细匹配 | `blip-image-text-matching-base` | 需要更准的判断时用 |
| 大规模检索 | CLIP + FAISS 粗排，BLIP 精排 | 两级架构效率更高 |

> **💡 Tip**：BLIP 的优势在于"一个模型多种能力"，但实际工程中往往不需要同时用所有能力。如果只需要相似度，用 CLIP；如果只需要生成，用 BLIP 的 Captioning 模型；如果需要匹配判断，用 BLIP 的 Matching 模型。按需选择，不要过度使用。

---

## 8. 常见追问

### Q1: BLIP 和 CLIP 最大的区别是什么？

CLIP 是双编码器结构，图像和文本各自编码后算余弦相似度，只输出 embedding，没有文本解码器，不能生成文本。BLIP 在此基础上加了 Image-Grounded Text Encoder 做精细匹配，加了 Text Decoder 做文本生成。简单说：CLIP 是图文对齐工具，BLIP 是统一视觉语言模型。

### Q2: BLIP 的 Bootstrapping 具体怎么做？

分三步：(1) 用 BLIP 的 Text Decoder 为每张图生成多个候选 caption；(2) 用 BLIP 的 Image-Grounded Text Encoder 对原始图文对和生成图文对打分，过滤低质量的；(3) 用过滤后的高质量数据重新训练 BLIP。论文实验显示这个过程能让下游任务提升 5-10%。

### Q3: ITC 和 ITM 都是做匹配，为什么需要两个？

粒度不同。ITC 用全局向量做余弦相似度对比，计算快，适合大规模检索时的粗筛。ITM 用 Image-Grounded Text Encoder 做 token 级图文交互，判断更精细，适合需要高精度的场景。两者类似检索系统中的"粗排 + 精排"。

### Q4: BLIP 的 Image-Grounded Text Encoder 和普通 Text Encoder 有什么区别？

Image-Grounded Text Encoder 在每个 Transformer 层的 Self-Attention 之后加了一个 Cross-Attention 层，让文本 token 可以直接 attend 到图像特征。这样文本在编码时就能看到图像的具体内容，匹配判断更准确。普通 Text Encoder 没有这个 Cross-Attention，只能看到文本自身。

### Q5: BLIP 有什么局限性？

主要有三个局限：(1) 生成能力有限，它是基于小规模 LLM 的，不像现代 VLM 接入了 7B+ 的大语言模型；(2) 视觉编码器是固定分辨率的 ViT，处理高分辨率或多图输入时灵活性不够；(3) 没有指令微调能力，不能像 InstructBLIP 那样遵循复杂指令。所以实际工程中，BLIP 更多作为理解早期多模态发展脉络的模型，新项目通常用 BLIP-2 或更新的 VLM。

---

## 常见误区

### ❌ 误区一：BLIP 和 CLIP 是同一种模型

**正确做法**：两者架构和目标不同。CLIP 是双编码器（Dual Encoder），只输出 embedding，做对比学习；BLIP 是编码器-解码器结构，既有编码能力又有生成能力。如果只需要图文相似度计算，CLIP 更直接高效。

### ❌ 误区二：BLIP-2 只是 BLIP 的大版本升级

**正确做法**：BLIP-2 的架构和训练方式与 BLIP 有本质区别。BLIP 端到端训练整个视觉语言模型，而 BLIP-2 用 Q-Former 桥接冻结的视觉编码器和冻结的大语言模型，目标是低成本地把视觉能力接入 LLM。BLIP-2 的生成能力远强于 BLIP。

### ❌ 误区三：BLIP 可以直接当作现代多模态大模型使用

**正确做法**：BLIP 的文本生成能力基于相对较小的语言模型，不能像 GPT-4V、Qwen-VL 那样处理复杂的多轮对话、长文本推理、指令跟随等任务。BLIP 更适合作为早期多模态模型学习，实际生产推荐 BLIP-2 或更新的 VLM。

### ❌ 误区四：BLIP 的三个损失函数同等重要

**正确做法**：三个损失的侧重不同。ITC 用于大规模检索场景，ITM 用于需要精确判断的匹配场景，LM 用于生成场景。实际使用时根据下游任务选择对应的模型变体，不需要同时激活三个能力。

---

## 参考文献

1. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — Junnan Li et al., 2022
2. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
3. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Junnan Li et al., 2023
4. [HuggingFace BLIP Documentation](https://huggingface.co/docs/transformers/model_doc/blip) — HuggingFace

---

## ✅ 自我检验

- [ ] 能用自己的话解释 BLIP 的核心思想（统一架构 + Bootstrapping）
- [ ] 能说出 BLIP 和 CLIP 的主要区别
- [ ] 能写出三个预训练损失的公式并解释每个变量
- [ ] 能用 HuggingFace 代码实现 BLIP 做图像描述、VQA、图文匹配
- [ ] 能说出 MED 架构中三种文本角色的区别
- [ ] 能回答"BLIP 有什么局限性"这个面试追问
- [ ] 能区分 BLIP、BLIP-2、InstructBLIP 的定位差异
