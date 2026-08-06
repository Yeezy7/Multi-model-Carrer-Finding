# BLIP-2 冻结塔桥接：Q-Former 与两阶段预训练的完整剖析

> BLIP-2（Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models）由 Salesforce 于 2023 年提出（ICML 2023，一作李东）。它是"低成本获得强大 VLM"路线上的里程碑：**视觉塔冻结、LLM 冻结，只训练一个 188M 的轻量桥接器 Q-Former，就能在 zero-shot VQA / 图文检索 / 图像描述上逼近甚至超越 Flamingo-80B**（训练 FLOPs 少了约 54 倍）。面试被问"不花大钱怎么训 VLM""Q-Former 是干什么的""BLIP-2 和 LLaVA 有什么区别"时，这篇笔记必须能完整答出。

---

## 一、一句话解释

> **BLIP-2 = 冻结视觉编码器 + 冻结大语言模型（LLM），只训练一个 32 个可学习 Query 的轻量 Transformer（Q-Former）作为桥接器，用两阶段预训练把图像信息"翻译"成 LLM 能读懂的输入，以极低训练成本获得强大的图文理解与生成能力。**

拆开看三个关键动作：

1. **视觉塔冻结**：CLIP/EVA-CLIP 的 ViT 不参与梯度更新，只做前向推理；
2. **LLM 冻结**：OPT / FlanT5 参数不动，全程只做前向 + 反向传播的终点停在 Q-Former；
3. **Q-Former 全量训练**：唯一可学习的大模块（188M），负责"用 32 个 query 从图像中提取 LLM 需要的视觉信息"。

> **记忆锚点**：BLIP-2 的 "2" 指**两阶段预训练**（表征学习 + 生成预训练）；"Bootstrapping" 指"不重训大模型，而是用轻量模块把已有大模型的能力引导（bootstrap）到多模态上来"。

---

## 二、动机：为什么端到端训练 VLM 太贵，为什么直接拼不行

### 2.1 痛点一：端到端训练 VLM 的成本爆炸

在 BLIP-2 之前，主流的 VLM 预训练范式是"视觉塔 + LLM 一起端到端训练"（如 Flamingo、Frozen、BLIP 等）：

```text
端到端范式:  图像 → ViT(可训练) → 桥接(可训练) → LLM(可训练) → 文本
计算路径:     梯度要同时穿过 LLM 的全部层 + 桥接层 + ViT 的全部层
```

这带来三重代价：

| 代价 | 说明 |
|------|------|
| 显存爆炸 | 梯度要保留 LLM 每一层的激活值，7B 级别 LLM 的优化器状态 + 梯度 + 激活至少几十 GB/样本 |
| 算力巨大 | Flamingo-80B 用了约 4 千卡级别算力、数十亿图文对，普通机构无法复现 |
| 数据需求 | 端到端联合训练需要海量图文对才能让两个大模块"对齐"，数据门槛极高 |
| 灾难性遗忘 | 联合训练中 LLM 的语言能力会被多模态目标干扰，退化明显 |

### 2.2 痛点二：直接"冻结拼接"效果差

既然训练贵，能不能直接冻结两个塔拼起来？**几乎不可用**。原因是存在"模态差距"（modality gap）：

| 问题 | 原因 |
|------|------|
| 特征空间不对齐 | ViT 输出的是"视觉 patch 特征"（每 patch 一个 768~1024 维向量），LLM 输入的是"离散 token 的词嵌入"，两者分布完全不同 |
| token 数不兼容 | 一张 224×224 图像有 196+1 个 patch token，LLM 的位置编码、序列长度、注意力开销都按"文本长度"设计，直接塞 197 个向量会让序列爆炸、位置编码失效 |
| 语义粒度错位 | ViT patch 特征是"局部像素块语义"，LLM 需要的是"可参与推理的高层实体概念"，中间缺少一层"压缩与抽象" |
| LLM 不会"看" | 冻结的 LLM 从未见过视觉输入分布，直接喂 ViT 特征 = 给一个只会英语的模型灌乱码，注意力权重完全无法处理 |

### 2.3 BLIP-2 的解法（核心思路）

> **在"冻结的视觉塔"和"冻结的 LLM"之间，插一个轻量、可训练的"翻译官"：Q-Former。视觉塔负责提取原始视觉特征，Q-Former 负责把它压缩成 LLM 能消费的固定长度表示，LLM 负责语言生成——三者各司其职。**

```text
BLIP-2 范式:
  图像 → ViT(冻结) → 图像特征 [197×768]
                        ↓
                  Q-Former(唯一可训练, 188M)
                  ┌──────────────────────┐
                  │ 32 个可学习 Query     │
                  │ ↓ Self-Attention      │
                  │ ↓ Cross-Attention 到图像 │
                  └──────────────────────┘
                        ↓
                  32 个视觉表示 [32×768]  →  FC 层 →  LLM(冻结) → 文本
```

三个"冻结"带来的直接收益：

1. **训练成本极低**：反向传播只穿过 Q-Former（188M），ViT 和 LLM 只做前向，训练 FLOPs 比 Flamingo-80B 少约 **54 倍**；
2. **算力可复现**：16 张 A100 即可完成全部预训练（阶段一约 1.5 天、阶段二约数天级别）；
3. **能力"白嫖"**：冻结的 CLIP 负责视觉、冻结的 OPT/FlanT5 负责语言与知识，BLIP-2 只是把它们"接起来"，不需要重复训练大模型。

---

## 三、Q-Former 架构详解（核心重点）

### 3.1 总体结构

Q-Former（Querying Transformer）本质上是一个 **BERT-base 尺寸的 Transformer**，关键参数：

| 配置项 | 值 |
|--------|----|
| 层数 | 12 层 |
| 隐藏维度 | 768 |
| 注意力头数 | 12 头 |
| FFN 维度 | 3072 |
| 可学习 Query 数 | **32 个** |
| 每个 Query 维度 | 768 |
| 总参数量 | **188M** |

结构上它由**两个子分支**组成，二者**共享 Self-Attention 层**：

```text
Q-Former 内部结构（第 i 层示意）:

  Query 分支 (图像侧)              文本分支 (文本侧)
  ┌──────────────────┐           ┌──────────────────┐
  │ 32 个 Query token │           │ 文本 token 序列   │
  │ [32 × 768]       │           │ [L × 768]        │
  └────────┬─────────┘           └────────┬─────────┘
           │                              │
           └─────► 共享 Self-Attention ◄──┘      ← 两个分支共用同一套参数
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Cross-Attention           （无 Cross-Attention）
  ← 与冻结 ViT 图像特征交互     ← 纯文本自注意力
        │
        ▼
  FFN → 下一层
```

- **图像分支（Image Transformer）**：输入是 32 个可学习 query token（`[32, 768]`），每层先做 Self-Attention，再做 **Cross-Attention**，注意力 key/value 来自**冻结 ViT 的图像 patch 特征**（如 197 个 patch token），query 来自自身 32 个 query；
- **文本分支（Text Transformer）**：输入是文本 token（如"a photo of a cat"），只做 Self-Attention + FFN，**没有 Cross-Attention**，不直接看图像；
- **关键设计：两个分支共享同一组 Self-Attention 参数**。这样文本分支的语义信息可以通过共享的 Self-Attention 流入 query token，使 query 在提取图像特征时"知道文本问了什么"——这是 Q-Former 能做"文本感知的视觉特征提取"的根本原因。

### 3.2 输入输出流程（逐步拆解）

```text
Step 1  图像预处理
        图像 [224×224] → 冻结 ViT (patch=14) → 197 个 patch token [197×768]
        （只前向，不反传）

Step 2  构造 Q-Former 输入
        32 个可学习 query 向量 q = [q₁, ..., q₃₂]，每维 768
        （这些 query 不是输入数据，而是"可学习参数"，初始化后随训练更新）

Step 3  双分支并行前向
        图像分支: query → 共享 Self-Attention → Cross-Attention(和 197 个 patch 交互) → FFN
        文本分支: 文本 token → 共享 Self-Attention → FFN
        反复 12 层

Step 4  输出
        query 分支输出 32 个视觉特征向量 Z = [z₁, ..., z₃₂]，每个 768 维
        （即"从图像里抽出来的、被文本语义调制过的 32 个信息片段"）

Step 5  接入下游
        阶段一（表征学习）: Z 直接用于 ITC/ITM/ITG 三个损失
        阶段二（生成预训练）: Z 经过 FC 层映射到 LLM 的隐层维度，送入冻结 LLM
```

**注意力掩码是动态的**（这是最容易讲漏的细节）：

| 训练目标 | 图像分支掩码 | 文本分支掩码 | 说明 |
|----------|-------------|-------------|------|
| ITC（对比） | 双向（query 互相可见） | 双向 | query 与文本各自聚合，互不直接交叉 |
| ITM（匹配） | 双向 | 双向 | query 输出与文本 token 拼接，一起判断是否匹配 |
| ITG（生成） | 双向 | **因果（causal）** | 文本逐 token 自回归，只能看前面 token + query 信息 |
| 阶段二 LM | 双向 | 因果（由 LLM 决定） | Q-Former 只负责出视觉特征，生成全交给 LLM |

### 3.3 为什么固定 32 个 query token

| 原因 | 详细说明 |
|------|---------|
| 与分辨率解耦 | 图像分辨率/ patch 数变化（224→336→多尺度），patch token 数随之变化（197→578），而 32 是**恒定**的——query 像"32 个探针"，不管图像多大，最终都输出 32 个表示，下游不用适配 |
| 与 LLM 兼容 | LLM 对序列长度、位置编码、注意力成本敏感，固定 32 个 token 让 LLM 的输入长度稳定可控，容易接入任何 LLM |
| 计算可控 | Cross-Attention 复杂度 O(32 × 197)，远小于 patch 两两交互 O(197²)；Q-Former 只有 188M，前向反传都很轻 |
| 信息瓶颈（双刃剑） | 32 个 token 强制 Q-Former 只能"挑重点"提取信息，不能偷懒把全部 patch 拷贝过去——这既是优点（选择性提取）也是局限（信息压缩损失） |

> 直觉比喻：32 个 query 就像"32 个采访记者"，每个人带着问题（query 向量）去浏览图像 patch（cross-attention），把看到的内容浓缩成一条简报带回。不管图像多大、细节多少，最后只有 32 条简报交到 LLM 手上。

### 3.4 Q-Former vs 简单 MLP Projector（为什么 Q-Former 更好）

LLaVA 早期用线性层/MLP 把 patch token 直接映射到 LLM 隐空间，BLIP-2 用 Q-Former，差异如下：

| 维度 | MLP Projector | Q-Former |
|------|---------------|----------|
| 输出 token 数 | 等于 patch 数（197/576 个） | **固定 32 个** |
| 信息处理 | 逐 patch 独立线性变换，无跨 patch 交互 | query 通过 Cross-Attention **全局选择**信息 |
| 是否压缩 | 不压缩，全量送入 | 强压缩，只保留 32 个"精华" |
| 是否有语义调制 | 无，和文本完全无关 | **共享 Self-Attention 让 query 感知文本**，可做文本条件提取 |
| 训练目标 | 单纯靠 LM 损失对齐 | 阶段一用 ITC/ITM/ITG 三目标充分对齐，阶段二再用 LM 损失适配 LLM |
| 对 LLM 的负担 | 长序列（197+）直接进 LLM，注意力和长度开销大 | 32 token 极短，LLM 注意力开销小 |
| 信息保留 | 无差别保留（大量冗余像素级信息） | **有选择地保留任务相关语义**，自动丢弃无关信息 |

**本质区别一句话**：MLP 是"逐点搬运输送带"，Q-Former 是"先检索再浓缩的信息编辑台"。特别是**文本感知**这一点：LLM 问"这只猫什么颜色"时，query 经过与文本共享的 Self-Attention，会优先去图像里找"颜色"信息——MLP 完全做不到这种条件化提取。

---

## 四、两阶段预训练策略（核心重点）

BLIP-2 把预训练拆成**表征学习**和**生成预训练**两个阶段，每阶段冻结的对象不同、目标不同：

```text
阶段一: 表征学习（Representation Learning）
  冻结: 视觉编码器 (ViT)        可训练: Q-Former（含 query 与文本编码部分）
  目标: ITC + ITG + ITM 三损失联合，让 Q-Former 学会"从图像里提取与文本对齐的信息"
  用到的文本: 训练集中的图文对描述

阶段二: 生成预训练（Generative Pre-training）
  冻结: 视觉编码器 (ViT) + LLM  可训练: Q-Former + 连接 LLM 的 FC 层
  目标: 语言建模损失，让 Q-Former 学会"把视觉信息编码成 LLM 能生成正确文本的输入格式"
  用到的文本: 图文对 + 纯文本数据（防止 LLM 语言能力遗忘）
```

### 4.1 阶段一：表征学习——三目标联合训练

阶段一在**冻结 ViT** 的条件下单独训练 Q-Former，同时优化三个损失：**ITC（Image-Text Contrastive，图文对比）+ ITM（Image-Text Matching，图文匹配）+ ITG（Image-grounded Text Generation，图像条件文本生成）**。总损失直接相加：

$$\mathcal{L}_{\text{Stage 1}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{ITG}}$$

三个目标从"对齐语义、判断配对、生成文本"三个角度**互补**地约束 Q-Former，缺一不可（消融实验显示去掉任何一个都会掉点）。

#### 4.1.1 ITC：图文对比损失（对齐语义）

参考 ALBEF 的做法：对 batch 内每张图 i 和每条文本 j 计算相似度，做 InfoNCE 对比。**特殊之处在于"图像表示"取自 32 个 query 输出**：

$$s_{i,j} = \max_{q \in \{1,\dots,32\}} \left( z_{i,q}^{\top} \tilde{t}_j \right)$$

其中 $z_{i,q}$ 是第 i 张图第 q 个 query 输出（L2 归一化后），$\tilde{t}_j$ 是第 j 条文本的 [CLS] 向量（L2 归一化）。**相似度取 32 个 query 中的最大值**（相当于"只要有一个 query 捕捉到了与文本相关的内容就算匹配"）。损失：

$$\mathcal{L}_{\text{ITC}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s_{i,i} / \tau)}{\sum_{j=1}^{N} \exp(s_{i,j} / \tau)}$$

| 细节 | 值 |
|------|----|
| 文本表示 | 文本分支 [CLS] token 输出（L2 归一化） |
| 图像表示 | 32 个 query 输出的**最大值**相似度 |
| 注意力掩码 | 双分支均双向 |
| 作用 | 拉近"匹配图文对"、推开"不匹配图文对"，建立图像-文本的全局语义对齐 |

#### 4.1.2 ITG：图像条件文本生成（学习生成）

在 query 序列后面追加一个特殊的 **[DEC] token**，文本 token 因果（causal）自回归地生成：

```text
输入序列:  [q₁ ... q₃₂] [DEC] [t₁] [t₂] ... [t_K]
注意力:    query 间双向；文本 token 只能看左侧（含 32 个 query 与 [DEC]）
```

$$\mathcal{L}_{\text{ITG}} = -\sum_{k=1}^{K} \log p_{\theta}(t_k \mid t_{<k},\ I)$$

- $t_{<k}$ 是前 k-1 个文本 token，$I$ 是图像（通过 query 与 Cross-Attention 注入）；
- **query 本身在该目标下不更新**（只作为信息的"载体"），真正被训练的是文本分支的生成能力；
- 文本分支在此目标下用**因果掩码**（其他目标用双向掩码）——同一个网络、动态切换掩码是 Q-Former 的精巧之处；
- 作用：让 Q-Former 学会"根据图像写出对应文本"，赋予其生成式理解能力，同时为阶段二打下"视觉→文本"的转换基础。

#### 4.1.3 ITM：图文匹配（判别配对）

二分类任务：给定图文对，判断是否匹配。query 输出与文本 token **拼接**后过共享 Self-Attention，取 [CLS] 输出过二分类头：

$$\mathcal{L}_{\text{ITM}} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ \log p_{i,i} + \sum_{j \in \text{HN}(i)} \log (1 - p_{i,j}) \Big]$$

其中 $p_{i,j} = \sigma\big(\text{MLP}([\text{CLS}]_{i,j})\big)$ 是匹配概率。

| 细节 | 值 |
|------|----|
| 正样本 | batch 内匹配对（i, i） |
| 负样本 | **hard negative**：利用 ITC 相似度矩阵，在同一 batch 内挑选"相似度高但实际不匹配"的对，而不是随机负样本 |
| 注意力掩码 | 双分支均双向（需要完整上下文才能判断匹配） |
| 作用 | 训练细粒度匹配能力，比 ITC 更严格（ITC 只看整体相似度，ITM 要看逐 token 交互） |

**为什么 ITC + ITM + ITG 三个都要？**

| 目标 | 学什么 | 缺了会怎样 |
|------|--------|-----------|
| ITC | 全局语义对齐（图↔文整体匹配） | 检索能力大幅下降 |
| ITM | 细粒度配对判别（hard negative 区分） | 匹配精度下降、表征不够精细 |
| ITG | 视觉→文本生成（条件生成） | 阶段二无法衔接（LLM 收到的视觉特征没有"生成导向"） |

### 4.2 阶段二：生成预训练——对齐冻结 LLM

阶段二**同时冻结视觉塔和 LLM**，只训练 Q-Former 与连接层（FC），损失是标准的**语言建模损失**：

$$\mathcal{L}_{\text{LM}} = -\sum_{k=1}^{K} \log p_{\phi}(t_k \mid t_{<k},\ Z)$$

其中 $Z = [z_1, \dots, z_{32}]$ 是 Q-Former 输出的 32 个视觉特征（经 FC 映射到 LLM 隐层维度），$t$ 是目标文本（如 VQA 格式的"Question: ... Answer: ..."）。

```text
阶段二前向流程（以 decoder-only 的 OPT 为例）:
  图像 → 冻结 ViT → patch 特征
                          ↓
                     Q-Former(可训练) → 32 个特征 [32×768]
                                              ↓
                                         FC 层 → [32×H_LLM]
                                              ↓
              LLM 输入序列: [z₁ ... z₃₂] [t₁] [t₂] ... [t_K]
                                              ↓
                              因果 LM 损失只算在文本 token 上
```

**关键点：损失只算在文本 token 上**。Q-Former 输出的 32 个向量相当于"视觉 prompt"，LLM 在它们的条件下自回归生成文本；梯度从 LLM 的损失反传到 Q-Former，LLM 本身不动。这让 Q-Former 被迫学习"LLM 能理解的视觉表达方式"——本质是把视觉信息**翻译成 LLM 的"语言分布"**。

### 4.3 两种 LLM 接入方式

阶段二实验了两种主流 LLM，接入方式不同：

| 维度 | Decoder-only（OPT） | Encoder-Decoder（FlanT5） |
|------|--------------------|---------------------------|
| 输入方式 | 32 个 query 特征**拼在文本序列前面**，作为前缀 token | 文本走 FlanT5 的 **encoder**；32 个 query 特征**分组送入 decoder** |
| 分组细节 | 无分组，直接拼接 | 32 个 query 分成 **5 组**（约 6-7 个/组），每组经过独立的 FC 层，从 decoder 的不同层注入 |
| FC 层 | 1 个（768 → OPT 隐层） | 5 个（768 → FlanT5 隐层，每组一个） |
| 注意力 | 因果注意力，只能看左侧 query 与历史文本 | decoder 交叉注意力同时看 encoder 输出文本 + 注入的视觉组 |
| 代表模型 | OPT-2.7B | FlanT5-XXL（11B） |
| 特点 | 通用、生态成熟 | 编码器-解码器结构天然适合"指令 + 条件生成" |

> 5 组分组的动机：把 32 个 query 分散注入 decoder 的不同层，让 decoder 在**不同抽象层次**都能接触到视觉信息，而不是只在序列开头看一次。

### 4.4 为什么必须分两阶段（面试高频追问）

**Q：为什么不能把所有目标（ITC/ITM/ITG/LM）在一个阶段同时训？**

| 理由 | 详细说明 |
|------|---------|
| 训练不稳定 | 对比损失（ITC/ITM）和生成损失（LM）对特征空间的要求互相拉扯：对比要求"全局可比的紧凑向量"，生成要求"逐 token 丰富的细节"，同时优化容易震荡 |
| 梯度路径过长 | LM 损失的梯度要穿透冻结的 LLM 所有层再回到 Q-Former，若还同时更新 ViT，长链路 + 多目标会让优化极度不稳 |
| 各司其职 | 阶段一先让 Q-Former 具备"从图像提取与文本对齐信息"的能力（与 LLM 无关）；阶段二再把这种能力"翻译"成特定 LLM 的输入格式（与视觉塔无关）——每阶段只解决一个问题，收敛快、效果可控 |
| 复用性 | 阶段一产出的 Q-Former 可以接**任意** LLM（OPT、FlanT5、乃至 175B），换 LLM 只需重跑便宜的阶段二；联合训练则每换一个 LLM 全盘重来 |
| 数据效率 | 阶段一靠对比/匹配目标能从海量弱标注图文对里学到对齐，阶段二少量图文对即可完成"适配"；端到端方案对数据质量要求高得多 |

> **记忆点**：阶段一解决"看得懂图像"，阶段二解决"LLM 听得懂"。两阶段让"重训练大模型"变成"轻量适配"，这就是 Bootstrapping 的含义。

---

## 五、各组件参数量与训练成本

### 5.1 参数清单

| 组件 | 参数量 | 阶段一是否可训练 | 阶段二是否可训练 |
|------|--------|:---:|:---:|
| 视觉塔 ViT-L/14（CLIP） | **304M** | 冻结 | 冻结 |
| 视觉塔 ViT-g/14（EVA-CLIP，大模型版） | 约 1.1B | 冻结 | 冻结 |
| Q-Former | **188M** | ✅ 训练 | ✅ 训练 |
| FC 连接层 | 极小（几 M） | 不适用 | ✅ 训练 |
| LLM OPT-2.7B | 2.7B | 不参与 | 冻结 |
| LLM FlanT5-XXL | 11B | 不参与 | 冻结 |

**Q-Former 只占很小比例**：

- 以 ViT-L + OPT-2.7B 计：188M / (304M + 188M + 2700M) ≈ **5.9%**；
- 以 ViT-g + FlanT5-XXL 计：188M / (1100M + 188M + 11000M) ≈ **1.5%**；
- **整个预训练过程只更新约 2~6% 的参数**，其余 95%+ 参数"白嫖"自冻结模型。

### 5.2 训练 FLOPs 对比（论文的招牌数据）

| 模型 | 训练 FLOPs | zero-shot VQAv2 |
|------|-----------|-----------------|
| Flamingo-80B | 约 33 万亿级（预估量级） | 56.3% |
| BLIP-2（ViT-g + FlanT5-XXL） | 约为 Flamingo 的 **1/54** | **65.0%** |
| InstructBLIP（BLIP-2 加指令微调） | 更低（后训练） | 进一步提升（微调后 90+%） |

> **面试金句**："BLIP-2 用 Flamingo 1/54 的训练 FLOPs 拿到更高分数，靠的不是更大的模型，而是'冻结 + 轻量桥接'的范式。"

---

## 六、训练细节

### 6.1 数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| COCO | 113K 图 | 人工标注图文对 |
| Visual Genome | 108K 图 | 密集 caption |
| CC3M（Conceptual Captions 3M） | 约 3.1M | 网络弱标注 |
| CC12M | 10M（训练时采样上限约 8M/epoch） | 网络弱标注 |
| SBU Captions | 860K | 网络弱标注 |
| LAION-400M | 115M（训练时采样上限约 10M/epoch） | 大规模弱标注 |
| **总计** | **约 129M 图文对** | 阶段一用 |

阶段一用全部 129M 图文对；阶段二在图文对上训练，并**混入纯文本数据**缓解 LLM 语言能力的灾难性遗忘。

### 6.2 阶段一训练配置（论文，ViT-L/14 版）

| 配置项 | 值 |
|--------|----|
| 硬件 | 16 × A100 (40G) |
| Batch size | 1920/卡 × 16 = **30720** |
| 步数 | 250K 步（ViT-L）/ 440K 步（ViT-g） |
| 优化器 | AdamW |
| 学习率 | 1e-4，cosine 衰减 |
| Warmup | 5000 步 |
| 权重衰减 | 0.05 |
| 图像分辨率 | 224 |
| 训练时长 | 约 1.5 天（ViT-L 版） |
| 精度 | FP16 混合精度 |

### 6.3 阶段二训练配置（论文）

| 配置项 | 值 |
|--------|----|
| 硬件 | 16 × A100 (40G) |
| Batch size | 3840/卡 × 16 = **61440** |
| 步数 | 80K 步 |
| 学习率 | 1e-4，cosine 衰减 |
| 文本格式 | 任务化 prompt：如 "Question: ... Answer: ..."、"a photo of ..." |
| 数据 | 图文对 + 纯文本数据 |
| 训练时长 | 数天量级（远低于端到端方案） |

### 6.4 后续微调策略

BLIP-2 在下游任务（VQA、captioning、retrieval 等）上可进一步微调：**Q-Former 全量更新，冻结的 ViT 与 LLM 用 LoRA 做参数高效微调**（可选）。这也延续了"绝大部分参数保持冻结"的省钱哲学。

---

## 七、推理应用与生态

### 7.1 三种典型推理

```text
① 图文问答 (zero-shot VQA)
   "Question: What color is the cat? Answer:"
   图像 → ViT(冻结) → Q-Former → FC → LLM(冻结) → 生成 "white"

② 图文检索
   图像 → Q-Former → 32 个 query 特征（或取 max 相似度）
   文本 → 文本塔 → 向量
   相似度打分 → Top-K 排序

③ 图像描述 (captioning)
   "a photo of" → 图像特征前置 → LLM 自回归生成完整描述
```

**BLIP-2 是"零样本开箱即用"的 VLM**：不需要为每个任务重新训练，直接 prompt 即可问答与描述。

### 7.2 生态位置：InstructBLIP 是它的直接后辈

| 模型 | 关系 | 改进 |
|------|------|------|
| BLIP-2（2023, ICML） | 基座 | 冻结塔 + Q-Former 两阶段预训练 |
| InstructBLIP（2023, NeurIPS） | BLIP-2 之上做**指令微调后训练** | 指令感知的 Q-Former（query 由指令文本调制）+ 13 个 held-in 任务 + 13 个 held-out 任务指令微调，zero-shot 泛化大幅提升 |
| MiniGPT-4（2023） | 复刻 Q-Former 思路 | Q-Former + Vicuna 7B，单卡即可微调，验证了该范式的易复现性 |
| Qwen-VL（2023） | 借鉴 Q-Former | 自研"ViT + 类 Q-Former"结构 |

> BLIP-2 → InstructBLIP 的演进路线本身就是一个经典面试题：**预训练解决"能不能理解"，指令微调解决"能不能听懂指令并泛化"**。

---

## 八、BLIP vs BLIP-2 vs LLaVA 对比

| 维度 | BLIP（2022） | **BLIP-2（2023）** | LLaVA（2023） |
|------|-------------|-------------------|---------------|
| 核心架构 | ViT + 文本编码器 + 解码器（三个 BERT 级别模块） | 冻结 ViT + **Q-Former** + 冻结 LLM | 冻结/微调 CLIP ViT + **线性/MLP 投影** + LLM（Vicuna） |
| 桥接方式 | 文本编码器与图像交叉注意力 | 32 query 的 Q-Former（强压缩 + 文本感知） | 逐 patch 线性投影（无压缩、无文本感知） |
| 是否冻结 | **不冻结**，全部联合训练 | **全部冻结**（只训 Q-Former） | 视觉塔冻结 + LLM 微调（LoRA/full） |
| 训练成本 | 中等（全量训练） | **最低**（只训 188M） | 中低（投影 + LLM 微调） |
| 生成模型 | 自研解码器（1.1B 级） | OPT-2.7B / FlanT5-XXL（大且冻结） | Vicuna 7B/13B（微调） |
| 预训练目标 | ITC + ITM + LM 单阶段联合 | 两阶段：ITC+ITM+ITG → LM | 两阶段：特征对齐（caption）→ 指令微调 |
| 零样本 VQA | 不支持（需微调） | ✅ 65.0%（ViT-g + FlanT5-XXL） | 弱（对齐阶段非 VQA 导向） |
| 指令跟随 | 弱 | 弱（靠 LLM 原生能力） | **强**（专为指令微调设计） |
| 参数效率 | 低 | **极高** | 中 |
| 关键局限 | 训练贵、模型小、能力上限低 | 32 token 瓶颈、LLM 冻结无法再增强 | 投影简单、长视觉序列、LLM 微调开销 |

**一句话对比**：

- **BLIP**：三模块全量联合训练，"一步到位"但贵、且生成能力受限于自研小解码器；
- **BLIP-2**：把"最贵的两座塔"全部冻结，用 Q-Former 白嫖它们的强大能力，成本最低；
- **LLaVA**：桥接最简单（MLP），但把成本花在**微调 LLM 上**，因此指令跟随最强——它验证了"LLM 微调换来的是指令能力"。

> 面试加分：**BLIP-2 和 LLaVA 是两条互补路线的代表**——BLIP-2 在"桥接器"上做文章（更好的压缩与文本感知），LLaVA 在"LLM 端"做文章（更强的对齐后微调）。后来的 MiniGPT-4 是两者混合体（Q-Former 桥接 + LLM 微调）。

---

## 九、局限性

| 局限 | 原因 | 后果 / 后续解法 |
|------|------|----------------|
| **信息瓶颈** | 32 个 query 是硬压缩，图像细节（计数、小目标、OCR、空间关系）在压缩中被丢弃 | 小目标/密集场景任务弱；后续模型（InstructBLIP 仍 32 个，Qwen-VL 增大 query 数并引入更高分辨率）加大 query 数 |
| **视觉塔分辨率固定** | 冻结的 ViT 通常 224/336 固定分辨率，无法像 VIT 动态分辨率那样适配 | 高清图、文档 OCR 效果差；后续用多尺度/任意分辨率 ViT（如 LLaVA-NeXT、InternVL 2.x） |
| **生成能力受 LLM 上限约束** | LLM 冻结，BLIP-2 无法通过训练提升语言能力、也无法注入新的多模态知识 | 幻觉、常识错误残留；InstructBLIP 靠指令微调缓解任务泛化问题 |
| **知识引导不足** | 阶段一/二的视觉信息都是"通用提取"，缺少任务导向与检索增强 | 开放域事实问答弱于带知识检索的模型 |
| **文本感知有限** | 共享 Self-Attention 只是"间接"让 query 感知文本，没有显式的文本条件机制（InstructBLIP 才加指令感知） | 复杂指令下的视觉聚焦不够精准 |
| **阶段一成本仍在** | 129M 图文对 + 250K 步预训练对个人开发者仍是门槛 | 社区多直接加载官方权重，本地只做阶段二或微调 |

---

## 十、常见误区

**误区 1：BLIP-2 是"冻结两个塔 + 训练一个 MLP"。**
错。桥接器不是 MLP，而是 Q-Former——一个带 32 个可学习 query、双分支共享 Self-Attention 的 Transformer，它在阶段一受过 ITC/ITM/ITG 三目标训练，具备文本感知的信息选择能力。MLP 做不到压缩与条件化提取。

**误区 2：BLIP-2 完全没有训练 ViT 和 LLM。**
基本对但有细节：预训练阶段二者确实完全冻结（连 LoRA 都没有），但**下游微调**时论文用 LoRA 对两个塔做参数高效微调。所以准确说法是"预训练阶段冻结、微调阶段可低成本注入"。

**误区 3：Q-Former 输出的 32 个向量就是图像的全部信息。**
不对。32 个向量是"被选择的信息摘要"，是有损压缩。图像细节必然丢失——这正是 BLIP-2 计数/OCR 弱的原因。它"挑重点"的能力来自训练目标（ITC 要对齐文本、ITG 要生成文本），而不是无损存储。

**误区 4：BLIP-2 的 zero-shot 能力来自 LLM，Q-Former 只是搬运工。**
不对。Q-Former 是决定性组件：消融显示换成简单投影（LLaVA 式）zero-shot VQA 大幅下降。Q-Former 在阶段一学到的"文本感知压缩"和阶段二学到的"LLM 适配格式"缺一不可。

**误区 5：LLaVA 和 BLIP-2 一样都是"两阶段预训练"。**
名字像，实质不同：BLIP-2 阶段一用三目标（ITC/ITM/ITG）训练 Q-Former，阶段二冻结 LLM 只训 Q-Former；LLaVA 阶段一是"最小化训练量做特征对齐"（MLP 拟合视觉特征到 LLM 词空间），阶段二是**端到端微调 LLM**（指令数据）。关键差异在阶段二是否动 LLM。

---

## 十一、高频面试问答

**Q1：用一句话讲清楚 BLIP-2 做了什么？**
冻结视觉编码器和 LLM，只训练一个 188M 的 Q-Former（32 个可学习 query 的 Transformer），用两阶段预训练（三目标表征学习 + 生成预训练）把图像信息压缩翻译成 LLM 能消费的固定 32 个 token，以 Flamingo 约 1/54 的训练成本达到更强的 zero-shot 图文理解。

**Q2：Q-Former 为什么比 MLP projector 好？**
三点：① 强压缩——MLP 把全部 patch 无差别映射（197+ token），Q-Former 用 32 个 query 做跨 patch 的选择性信息提取，token 数与分辨率解耦；② 文本感知——双分支共享 Self-Attention，query 在提取图像特征时能看到文本，能按"问什么"提取"什么"，MLP 做不到；③ 训练充分——阶段一用 ITC/ITM/ITG 三目标对齐语义，阶段二再适配 LLM，比 MLP 只靠 LM 损失对齐信息量大得多。

**Q3：32 个 query 是怎么工作的？和 ViT 的 [CLS] token 有什么区别？**
query 是可学习参数（32×768），它们通过 Cross-Attention 与图像的 197 个 patch token 交互（query 做 Q，patch 做 K/V），每层交互一次，12 层后输出 32 个特征向量。区别：CLS 只有一个、只做全局聚合；32 个 query 是"多头分诊"式的并行提取，且每个 query 能通过共享 Self-Attention 感知文本语义，输出是任务相关的多视角表示。

**Q4：阶段一为什么同时要 ITC、ITM、ITG 三个损失？只留一个行不行？**
不行，三者互补：ITC 提供全局语义对齐（负样本对比，学"像不像"）；ITM 提供细粒度配对判别（hard negative，学"配不配"）；ITG 提供视觉到文本的生成能力（学"怎么描述"）。消融实验去掉任意一个都会掉点。注意 ITG 中文本分支切到因果掩码、query 不更新，是"同一个网络不同掩码"的复用设计。

**Q5：阶段二为什么冻结 LLM 还能学到对齐？**
因为 LM 损失只在文本 token 上计算，梯度穿过冻结 LLM 的各层后反传回 Q-Former——LLM 相当于一个"固定的语言先验裁判"，Q-Former 被迫调整自己的输出，使得"32 个向量 + 文本前缀"能产生正确文本。本质是让 Q-Former 学习 LLM 的输入分布（把视觉信息翻译成 LLM 的"语言"），而不是让 LLM 学习视觉。

**Q6：为什么必须分两阶段，不能单阶段联合训练？**
① 稳定性：对比目标与生成目标对特征空间要求互相拉扯，联合训练易震荡；② 梯度路径：LM 梯度穿透 LLM 全层再回传，链路长、多目标叠加更不稳定；③ 解耦复用：阶段一产物可接任意 LLM，换 LLM 只需重跑便宜的阶段二；④ 数据效率：两阶段让弱标注图文对与纯文本数据各得其所。

**Q7：BLIP-2 和 LLaVA 的核心区别是什么？**
桥接方式与训练对象不同：BLIP-2 用 Q-Former（强压缩 + 文本感知），预训练全程冻结双塔，只训 188M；LLaVA 用线性/MLP 投影（无压缩），对齐后**端到端微调 LLM**（把成本花在 LLM 上）。结果：BLIP-2 零样本能力强、参数效率高；LLaVA 指令跟随强、适合对话场景。MiniGPT-4 是二者混合。

**Q8：BLIP-2 有什么明显的弱点？为什么？**
① 信息瓶颈：32 token 有损压缩，计数、小目标、OCR、空间关系弱；② 固定分辨率：冻结 ViT 无法处理高清/不规则分辨率；③ 生成能力封顶：LLM 冻结，语言能力与知识无法通过多模态训练增强，有幻觉；④ 开放域知识问答弱于带检索的方案。

**Q9：InstructBLIP 在 BLIP-2 基础上改了什么？**
后训练范式：在 BLIP-2 之上做指令微调（instruction tuning）。核心改进是"指令感知的 Q-Former"——把指令文本也送入 Q-Former 共享 Self-Attention，让 query 按指令提取对应视觉信息；用 13 个 held-in + 13 个 held-out 任务验证，zero-shot 泛化显著优于 BLIP-2，证明"预训练 + 指令微调"是 VLM 的标准后训练路线。

**Q10：如果要你在单卡上做一个自己的 BLIP-2，怎么配置最划算？**
直接加载官方冻结 ViT-L + Q-Former 权重，接一个 7B 级别开源 LLM（如 Vicuna/Qwen），只训练 Q-Former 与 FC 层（约 200M 参数）做阶段二式适配，用 LoRA 微调 LLM。成本集中在数据清洗与 prompt 设计，参数更新量极小——这正是 BLIP-2 范式的工程红利。

---

## 十二、自我检验

- [ ] 能一句话说清 BLIP-2 的三要素（冻结视觉塔、冻结 LLM、Q-Former 桥接）
- [ ] 能讲出端到端训练 VLM 的三重代价与"冻结拼接"失效的三个原因
- [ ] 能画出 Q-Former 结构图（双分支、共享 Self-Attention、Cross-Attention 到图像）
- [ ] 能解释 32 个 query 的完整输入输出流程（query → Self-Att → Cross-Att → 输出 32 特征）
- [ ] 能说出固定 32 token 的三个理由（分辨率解耦、LLM 兼容、计算可控）
- [ ] 能列出 Q-Former 相对 MLP 的三大优势（压缩、文本感知、多目标训练）
- [ ] 能写出 ITC / ITM / ITG 三个损失公式并解释每个符号
- [ ] 能说清三个目标各自的注意力掩码规则与分工
- [ ] 能解释阶段二 LM 损失为什么能训练到 Q-Former（梯度穿过冻结 LLM 回传）
- [ ] 能说出 decoder-only（OPT，前缀拼接）与 encoder-decoder（FlanT5，5 组 32 query 分注入）的接入差异
- [ ] 能讲出分两阶段的五个理由（稳定性、梯度路径、复用、数据效率、各司其职）
- [ ] 能报出各组件参数量：ViT-L 304M、Q-Former 188M、OPT-2.7B、FlanT5-XXL 11B
- [ ] 能说出训练配置关键数字（129M 数据、batch 30720/61440、lr 1e-4、250K/80K 步）
- [ ] 能完成 BLIP vs BLIP-2 vs LLaVA 三方对比并指出各自成本花在哪
- [ ] 能列出 BLIP-2 至少三条局限及其原因
- [ ] 能讲清 InstructBLIP 与 BLIP-2 的关系（后训练 + 指令感知 Q-Former）
- [ ] 能纠正 5 条常见误区
- [ ] 能完整回答 10 个面试追问

---

## 参考文献

1. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al., ICML 2023
2. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — Li et al., ICML 2022
3. [ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651) — Li et al., NeurIPS 2021
4. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — Alayrac et al., NeurIPS 2022
5. [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) — Dai et al., NeurIPS 2023
6. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — Liu et al., NeurIPS 2023
7. [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) — Zhang et al., 2022
8. [Scaling Instruction-Finetuned Language Models (FlanT5)](https://arxiv.org/abs/2210.11416) — Chung et al., 2022
9. [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592) — Zhu et al., 2023
