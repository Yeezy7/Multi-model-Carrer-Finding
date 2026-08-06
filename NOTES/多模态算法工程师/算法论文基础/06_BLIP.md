# BLIP：Bootstrapping Language-Image Pre-training 完整剖析

> 本笔记讲解 BLIP（Bootstrapping Language-Image Pre-training），由 Salesforce 于 2022 年提出（NeurIPS 2022），是"理解 + 生成"统一多模态预训练的开创性工作之一。面试被问"多模态预训练怎么做""如何清洗图文数据""BLIP 和 CLIP 有什么区别"时，本笔记是核心弹药库。

## 一、一句话解释

> **BLIP = 用"对比（ITC）+ 匹配（ITM）+ 生成（LM）"三个目标联合预训练的统一多模态模型，并首创 CapFilt（Captioning + Filtering）自举机制清洗互联网上的噪声图文数据。**

三个要点必须记住：

1. **统一**：一个模型同时覆盖理解（检索、匹配、VQA 推理）与生成（图像描述、答案生成）两类任务，而不是像 CLIP 那样只能理解；
2. **Bootstrapping**：模型先在自己的弱标注（alt-text）数据上训练，再用微调出的"标题生成器（Captioner）+ 过滤器（Filter）"反过来清洗自己的训练数据，形成自举闭环；
3. **效果**：在 COCO 检索、描述、VQA 等多个 benchmark 上刷新 SOTA，成为后续 BLIP-2、InstructBLIP 等一系列工作的基石。

---

## 二、背景与动机：为什么需要 BLIP

### 2.1 痛点一：理解与生成两类能力被割裂

2021-2022 年时的多模态预训练模型大致分两派：

| 派别 | 代表模型 | 训练目标 | 能力 | 短板 |
|------|----------|----------|------|------|
| 理解派（对齐派） | CLIP、ALBEF | 对比学习（+ 匹配） | 检索、零样本分类、图文匹配 | **不能生成文本**，无法做描述/VQA 问答 |
| 生成派 | SimVLM、CoCa | 语言建模（prefixLM） | 图像描述、开放域生成 | 没有显式对齐，**不能做检索/匹配**，细节对齐弱 |

**问题的本质**：检索与生成是互补的能力，但此前没有一个模型能同时做好。理解派模型把图文"压成一个向量"做对比，丢失了生成所需的空间细节；生成派模型只优化文本似然，没有显式把图文表征对齐到同一空间。

> 面试记忆点：CLIP 学的是"图文是否像"，BLIP 学的是"图文是否像 + 图能产出什么文"。检索需要前者，描述需要后者。

### 2.2 痛点二：互联网图文数据噪声严重

CLIP 时代模型的训练数据主要来自网页的 alt-text（图像替代文本），例如 Conceptual Captions（CC3M/CC12M）、SBU Captions。这类数据的问题：

1. **alt-text 质量差**：网页作者写的 alt-text 常常与图像内容关系松散，甚至完全无关；
2. **描述不完整**：alt-text 往往只描述图像的一部分（如只写"纽约"，但图里有完整的街景细节）；
3. **语法噪声**：alt-text 不是规范的句子，充满关键词堆砌；
4. **数据量 vs 质量矛盾**：人工标注数据（如 COCO 12 万对）质量高但太少；网页数据多但脏。

> 一句话：**互联网给了你海量但劣质的图文对，人工标注给了你少量但优质的图文对——BLIP 的思路是用人工数据训练"清洗工具"，去把海量劣质数据变成优质数据。**

### 2.3 从两大痛点看 BLIP 的应对

| 痛点 | BLIP 的应对 |
|------|-------------|
| 理解与生成割裂 | 设计 3 个训练目标（ITC + ITM + LM）联合优化一个统一架构 |
| 数据噪声 | CapFilt 自举机制：Captioner 补全缺失描述，Filter 滤掉错误样本 |

---

## 三、核心架构：一个图像编码器 + 三个文本侧模块

### 3.1 总体结构

```text
                          ┌──────────────────────────────────────────────┐
                          │              预训练目标                       │
┌───────────────┐         │  ITC（图文对比）     —— 用 [CLS] 向量         │
│  图像输入      │         │  ITM（图文匹配）     —— 用 [Encode] 向量       │
│      ↓        │         │  LM（语言建模）      —— 用 [Decode] 生成       │
│ Image Encoder │         └──────────────────────────────────────────────┘
│   (ViT)       │  图像特征
│  (含动量副本)  │ ────────┬───────────────────────────────────────────────┐
└───────────────┘         │         │         │                          │
                          ▼         ▼         ▼                          │
                   ┌───────────┐ ┌───────────┐ ┌──────────────────────┐  │
       文本输入 ──▶ │ Text Enc  │ │ Img-ground│ │ Img-ground Text Dec  │  │
        [CLS]文本   │ (BERT)    │ │ Text Enc  │ │  (因果自注意力+交叉)   │  │
                   │ 纯文本，无交叉│ │ 双向自注意力 │  [Decode] token 开始   │  │
                   │ 输出 [CLS] │ │ + 交叉注意力│  → 生成 caption/答案    │  │
                   │ 供 ITC 用  │ │ 输出[Encode]│  供 LM 用              │  │
                   └───────────┘ │ 供 ITM 用  │                        │  │
                                 └───────────┘                        │  │
                                  参数共享：自注意力 + FFN 与 Text Enc 相同    │
                                  （仅新增 cross-attention）                │
                   Decoder 与上面两个编码器不共享参数（自注意力是因果的）───┘
```

### 3.2 四个模块逐一说明

**模块 1：图像编码器（Image Encoder，ViT）**

- 采用 ViT（Vision Transformer），输入为 $224\times224$（预训练）/ $384\times384$（微调）图像切 patch；
- 与 ALBEF 一致，在 patch 序列前加入 `[CLS]` token，`[CLS]` 的输出向量作为图像整体表征（供 ITC 用）；
- 全部 patch 特征（含 `[CLS]`）都会喂给两个图像增强文本模块做 cross-attention；
- 另有**动量版本**（EMA 副本，不参与梯度更新），用于 ITC 的相似度计算。

**模块 2：文本编码器（Text Encoder，BERT）**

- 就是标准 BERT（12 层、hidden 768、12 头）；
- 输入文本前加 `[CLS]` token，`[CLS]` 输出向量作为文本整体表征（供 ITC 用）；
- 双向自注意力，**看不到图像**，是纯单模态编码器。

**模块 3：图像增强文本编码器（Image-grounded Text Encoder）**

- 用于 ITM 的二分类匹配；
- 构造方式：在文本编码器每一层的"自注意力与 FFN 之间"插入一层 cross-attention，让文本 token 可以 attend 到图像 patch 特征；
- 输入文本末尾追加特殊 token `[Encode]`，`[Encode]` 的输出 embedding 作为"图文融合表征"；
- 该模块是**双向的**（普通自注意力），与解码器形成对比。

**模块 4：图像增强文本解码器（Image-grounded Text Decoder）**

- 用于 LM 的生成；
- 与模块 3 的区别：把双向自注意力换成**因果（causal）自注意力**（只能看过去），保留 cross-attention 到图像特征；
- 输入文本开头加特殊 token `[Decode]`，生成时以 `[Decode]` 的 embedding 为起点逐步解码；
- 文本侧自回归生成：$\log p(T|I)$ 按 token 顺序分解。

### 3.3 三个特殊 token 的分工

| Token | 位置 | 所属模块 | 用途 |
|-------|------|----------|------|
| `[CLS]` | 文本开头（图像侧也有一个） | Text Encoder / ViT | 单模态整体表征，ITC 用 |
| `[Encode]` | 文本末尾 | Img-grounded Text Encoder | 图文融合表征，ITM 用 |
| `[Decode]` | 文本开头 | Img-grounded Text Decoder | 生成起点，LM 用 |

> 面试记忆点：**CLS 管"像不像"，Encode 管"配不配"，Decode 管"说什么"**——三个 token 对应三个目标，一一对应，不容混淆。

### 3.4 为什么设计两个文本侧模块？参数共享关系

面试高频追问："为什么不直接用一个 BERT 同时做匹配和生成？"

**原因一：任务对注意力的要求相反。**

- 匹配（ITM）需要**双向**上下文：判断"图里是否有这只猫"需要同时看到句首句尾；
- 生成（LM）必须**因果**：预测第 t 个词时不能看到后面的词；
- 一个模块无法同时满足，必须拆分。这是 encoder-decoder 架构的经典理由。

**原因二：单模态与多模态信息分离。**

- Text Encoder 提供**纯文本**表征，负责把文本与图像"对齐到同一空间"（ITC），不受图像信息干扰；
- Img-grounded Encoder/Decoder 负责**融合**，在文本语义基础上注入视觉信息；
- 先对齐、后融合（Align before Fuse，继承自 ALBEF），避免噪声图像信息过早污染文本表征。

**参数共享关系（重点）：**

| 关系 | 是否共享 | 说明 |
|------|----------|------|
| Text Encoder 与 Img-grounded Text Encoder | **共享自注意力 + FFN** | 图像增强编码器就是在文本编码器每层中间插入 cross-attention，自注意力与 FFN 权重直接复用，只新增 cross-attention 参数 |
| Encoder 与 Decoder | **不共享** | 自注意力方向不同（双向 vs 因果），共享会产生冲突；但词嵌入表（word embedding）通常共享 |
| 图像编码器与文本侧 | 不共享 | 天然不同模态 |
| 在线模型与动量模型 | 动量是独立的 EMA 副本 | 见 4.2 节 |

> 面试记忆点：共享自注意力/FFN 是为了**省参数、促共训**（同一套语言理解能力同时服务 ITC 与 ITM）；不共享 decoder 是因为**注意力掩码矛盾**。

---

## 四、三大训练目标

### 4.1 总览

| 损失 | 全称 | 使用模块 | 使用 token | 类型 | 核心作用 |
|------|------|----------|-----------|------|----------|
| ITC | Image-Text Contrastive | ViT + Text Encoder | `[CLS]` | 对比 | 全局对齐，粗粒度 |
| ITM | Image-Text Matching | Img-grounded Text Encoder | `[Encode]` | 二分类 | 细粒度匹配，纠错 |
| LM | Language Modeling | Img-grounded Text Decoder | `[Decode]` | 生成 | 学习"看图说话" |

### 4.2 ITC：图像-文本对比损失

**目标**：把匹配的图文对拉近、不匹配的推远（全局表征对齐）。

设 batch 内有 $N$ 对图文，图像编码器输出 $v_i = \text{ImageEnc}(I_i)$，文本编码器输出 $t_j = \text{TextEnc}(T_j)$，相似度为余弦相似度：

$$s_{ij} = s(I_i, T_j) = \frac{v_i^\top t_j}{\|v_i\| \|t_j\|}$$

ITC 是**对称**的 softmax 对比损失（图像→文本方向 + 文本→图像方向）：

$$\mathcal{L}_{ITC} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ij}/\tau)} + \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ji}/\tau)} \right]$$

其中：

- $s_{ii}$：第 $i$ 个 batch 内正样本对的相似度（对角线）；
- 分母中 $j \neq i$ 的项为 batch 内负样本（in-batch negatives）；
- $\tau$：温度参数，可学习（初值类似 CLIP 的 0.07 量级），控制分布的锐利程度。

**Momentum Encoder（动量编码器）**：

BLIP 像 ALBEF 一样，为图像编码器和文本编码器各维护一份**动量副本**，参数不通过反向传播更新，而是按指数滑动平均（EMA）从在线参数拷贝：

$$\theta_m \leftarrow m \cdot \theta_m + (1 - m) \cdot \theta, \quad m = 0.995$$

- 相似度矩阵用**动量编码器**的特征计算（$\hat{v}_i, \hat{t}_j$），提供更稳定的判别信号；
- 作用：动量特征变化缓慢、一致性高，能提供更可靠的负样本区分目标，防止对比学习表征坍塌（representation collapse）；
- 注意：ITC 是唯一使用动量编码器的损失（ITM、LM 都用在线模型，原因见 4.3/4.4）。

### 4.3 ITM：图像-文本匹配损失

**目标**：对"图文是否真正匹配"做细粒度二分类——ITC 只能比较两个全局向量，而 ITM 让每个文本 token 通过 cross-attention 直接看图像，能捕捉"局部细节是否对应"。

**输入构造**：图像 patch 特征 + 文本（末尾加 `[Encode]`），过 Img-grounded Text Encoder，取 `[Encode]` 的输出 $z$，接一个线性分类头：

$$p = \sigma(w^\top z + b)$$

二分类交叉熵：

$$\mathcal{L}_{ITM} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$$

其中 $y_i \in \{0, 1\}$ 是图文对匹配与否的标签。

**Hard Negative Mining（硬负样本挖掘）—— 关键技巧**：

随机负样本太容易区分（"猫"和"一辆汽车"当然不匹配），模型学不到细粒度判别。BLIP 的策略：

1. 用动量编码器的 ITC 相似度矩阵对 batch 内每个正样本对 $(I_i, T_i)$ 排序；
2. 从**相似度最高**的图文对里挑负样本（即"最像但其实不匹配"的样本）；
3. 按相似度成比例地随机采样，形成 1 正 : 3 负 的比例（继承 ALBEF 的设置）；
4. 用这些硬负样本计算 ITM 损失。

```text
对 batch 内每个正样本对 (I_i, T_i):
  1. 用动量编码器算 ITC 相似度 s(I_i, T_j), j = 1..N
  2. 排除 j = i（正样本本身）
  3. 负样本按概率 P(T_j) ∝ exp(s(I_i, T_j)/τ) 采样 3 个
  4. 构造 4 条样本：1 正 + 3 硬负，全部过 ITM 分类器
```

> 为什么不用动量编码器算 ITM？因为 ITM 需要完整的梯度来训练 cross-attention 融合能力（分类目标依赖图像细节），而动量编码器不反传梯度，只适合"提供稳定的相似度目标"这种场景。

### 4.4 LM：语言建模损失

**目标**：给定图像，最大化生成正确文本的概率——让模型学会"看图说话"。

输入 = 图像 patch 特征 + `[Decode]` token + 文本 token 序列（因果掩码），过 Img-grounded Text Decoder。语言建模损失即下一个 token 预测的交叉熵：

$$\mathcal{L}_{LM} = -\sum_{t=1}^{L} \log p_\theta(y_t \mid y_{<t}, I_i)$$

其中：

- $y_{<t}$：前 $t-1$ 个已生成的 token；
- $L$：文本长度；
- 损失只加在文本 token 上（`[Decode]` 不计算损失）；
- 因为自注意力是因果的，训练时可以像 GPT 一样并行计算整句所有位置的预测。

> **MLM vs LM 的区别**（面试常考）：ALBEF 用掩码语言建模（MLM，BERT 式，双向、预测被掩掉的 token），只能理解；BLIP 换成自回归语言建模（LM，GPT 式，因果、预测下一个 token），**才能生成**。这是 BLIP 相对 ALBEF 的关键升级。

### 4.5 三个目标的互补性

```text
ITC（对比）    —— 全局对齐：图文表征落到同一空间，但"向量像"不代表"内容真配"
ITM（匹配）    —— 细粒度验证：token 级 cross-attention 检查细节，但只判是非、不产出内容
LM（生成）     —— 内容产出：学会从图像解码出语言，但不直接学表征对齐
```

三者缺一不可：**ITC 打地基、ITM 做质检、LM 长出表达能力**。

---

## 五、Bootstrapping：CapFilt 数据清洗机制（重点章节）

### 5.1 要解决的问题

预训练数据 = 129M 张网页图像 + 对应的 alt-text。alt-text 噪声的量化表现：

- 大量 alt-text 与图像内容不匹配（如营销文案、标签堆砌）；
- 大量图像根本没有描述或描述严重不足（缺失信息 → 模型学不到完整语义）。

### 5.2 总体流程

```text
阶段 0：用 129M 原始（噪声）数据预训练一个初始 BLIP
阶段 1：在高质量人工标注数据 COCO 上微调出两个专用模块
          ├── Captioner（标题生成器）：Img-grounded Text Decoder，LM 目标
          └── Filter（过滤器）：Img-grounded Text Encoder，ITM 目标
阶段 2：对每张网页图像 I：
          ├── 原始 alt-text：T_w（网页自带的弱标注）
          ├── 合成 caption：T_s = Captioner(I)（beam search 生成）
          └── Filter 给两条都打分 p = P(匹配 | I, T)
阶段 3：按阈值保留
          ├── 保留 T_w，若 p(I, T_w) > τ_w
          └── 保留 T_s，若 p(I, T_s) > τ_s
阶段 4：用清洗后的数据（规模与 129M 相当）重新预训练最终 BLIP
```

### 5.3 Captioner（补全者）

- **来源**：初始 BLIP 的图像增强文本解码器在 COCO 人工标注上微调（LM 损失）；
- **作用**：给没有好描述的图像**生成合成描述**，弥补 alt-text 的信息缺失；
- **生成方式**：beam search（beam size = 3），每张图像生成 1 条合成 caption；
- **本质**：用"少量人工标注"教会模型什么叫"好描述"，再让模型自己给海量图像写描述——这就是"Bootstrapping（自举）"的第一层含义。

### 5.4 Filter（过滤器）

- **来源**：初始 BLIP 的图像增强文本编码器在 COCO 上微调（ITM 二分类）；
- **作用**：对 (图像, caption) 对打分，过滤掉不匹配的样本；
- **打分**：输出匹配概率 $p = P(\text{match} \mid I, T)$；
- **本质**：模型用"少量人工正负样本"学会判别图文匹配，再回来质检海量数据——"用模型清洗模型自己的训练数据"，这是第二层自举。

### 5.5 阈值设计

| 样本类型 | 阈值 | 为什么 |
|----------|------|--------|
| 原始 alt-text（T_w） | $\tau_w = 0.5$ | 门槛低：原始文本是"弱但真实"的信号，宁滥勿缺 |
| 合成 caption（T_s） | $\tau_s = 0.7$ | 门槛高：模型自产的数据有"自我确认偏差"，必须严格筛选 |

> 直觉：别人写的再差也是"ground truth 的弱版本"，自己写的再好也可能"自嗨"——所以对自己的产物更严格。这是论文中一个非常精巧的设计点，面试常被追问。

### 5.6 效果数字（论文报告）

| 指标 | 数值 | 说明 |
|------|------|------|
| 数据集规模 | 129M → 过滤后规模基本不变（约 129M） | 不是砍量，而是**换质** |
| 零样本 COCO Caption（CIDEr） | **113.2**（此前最优约 89.3） | 大幅领先，证明数据质量提升 |
| COCO 检索零样本 TR@1 / IR@1 | **82.4 / 64.1**（ViT-B） | 相对同类模型显著提升 |
| COCO 检索微调 TR@1 / IR@1 | **88.0 / 73.4**（ViT-B, Karpathy split） | |
| VQA test-dev | **78.25**（ViT-B） | |

论文整体结论：BLIP 相对此前最优（ALBEF）在检索平均 R@1 提升 +2.7%、COCO 描述 CIDEr 提升 +2.8%、VQA 提升 +1.6%，**CapFilt 是其中最大的增益来源之一**。

### 5.7 为什么有效？局限在哪里？

**有效性解释**：

1. **补全信息**：Captioner 把"没描述"变成"有描述"，模型从"图里有东西"学到"图里是什么"；
2. **去噪**：Filter 把"描述错"的样本剔除，模型不再被错误监督信号带偏；
3. **自举的可扩展性**：清洗工具由 12 万条人工标注训出，却可以免费处理 129M 条数据，质量提升的成本极低。

**风险与局限**（面试加分点）：

- **自我确认偏差（confirmation bias）**：Captioner 生成的 caption 延续了预训练模型的错误认知，Filter 又信任它，错误可能被放大；
- **只自举一次**：论文强调 CapFilt 只在预训练中应用一轮，多轮迭代自训练有收敛到偏差的风险；
- **依赖初始模型质量**：阶段 0 的模型在噪声数据上训练，如果初始对齐就崩了，后面的清洗全是空转。

---

## 六、预训练与微调细节

### 6.1 预训练数据

| 组成 | 规模 | 说明 |
|------|------|------|
| 自采网页图文对 | 115M | 来自网络爬取，含 alt-text 噪声 |
| 公开数据集 | 约 14M | CC3M、CC12M、SBU、COCO、Visual Genome 等 |
| **合计** | **129M** | 全部用于预训练 |

### 6.2 超参数配置

| 配置 | 数值 |
|------|------|
| 图像分辨率 | 预训练 224×224；微调 384×384 |
| 图像编码器 | ViT-B/16（Base）/ ViT-L/16（Large） |
| 文本侧 | BERT-base（12 层，hidden 768，12 头） |
| 预训练 epoch | ViT-B 约 20 epoch；ViT-L 约 10 epoch |
| Batch size | 256 |
| 优化器 | AdamW |
| 学习率 | 3e-4（ViT-B）/ 1e-4（ViT-L） |
| 学习率调度 | 线性 warmup + 余弦退火 |
| 动量系数 m | 0.995 |
| 温度 τ | 可学习 |
| 数据增强 | 图像 RandAugment；文本 BERT tokenizer |
| CapFilt 阈值 | τ_w = 0.5（原始），τ_s = 0.7（合成） |

### 6.3 完整训练流水线（阶段）

```text
Step 1  原始 129M 数据 + 三损失（ITC+ITM+LM）预训练 → 初始 BLIP
Step 2  在 COCO 上分别微调出 Captioner 与 Filter
Step 3  CapFilt 清洗 129M 数据 → 高质量数据集（原规模）
Step 4  用清洗后数据重新预训练（从零开始）→ 最终预训练权重
Step 5  下游任务微调（检索 / 描述 / VQA / NLVR2）
```

### 6.4 各下游任务微调方式

| 任务 | 微调损失 | 输入构造 | 备注 |
|------|----------|----------|------|
| 图文检索 | ITC + ITM | 图像 + 文本 | 384 分辨率 |
| 图像描述 | LM | 图像 + `[Decode]` + caption | beam search size 3 |
| VQA | LM | 图像 + 问题 + `[Decode]` + 答案 | 把问答建模为答案生成 |
| NLVR2 | ITM（二分类） | 图像 + 陈述 | 用 `[Encode]` 表征分类 |

---

## 七、推理应用

### 7.1 图像描述（Captioning）

```text
图像 I → ViT → 图像特征
文本输入: [Decode]
解码器按自回归逐步生成: y_t = argmax p(y_t | y_<t, I)
推理策略: greedy 或 beam search（微调默认 beam=3）
```

### 7.2 VQA（视觉问答）

BLIP 把 VQA 当作**答案生成**任务（而非分类任务），这是开放域问答的关键：

```text
输入: 图像 I + 问题 Q + [Decode]
文本序列: [Decode] Q ? [Decode] A        （问题后追加 Decode token 开始生成答案）
输出: 自回归生成答案 A = "cat"
```

- 好处：不受固定答案集限制，可以回答训练集中没见过的答案；
- 数据：VQA2.0 + Visual Genome QA 联合微调。

### 7.3 图文检索（Retrieval）

```text
图文检索: 用 ITC 余弦相似度初筛（快），再用 ITM 细排（准）
最终分数: s = s_ITC(I, T) + s_ITM(I, T)
```

ITC 打分做全局粗筛（向量点积，百万级候选秒级完成），ITM 打分对 Top-K 候选做细粒度验证，两者互补。

### 7.4 模型变体

| 变体 | 图像编码器 | 文本侧 | 用途 |
|------|-----------|--------|------|
| BLIPBase | ViT-B/16 | BERT-base | 通用，效果/成本均衡 |
| BLIPLarge | ViT-L/16 | BERT-base | 追求上限，VQA/检索/描述全面更强 |

---

## 八、BLIP vs 相关模型对比

### 8.1 大对比表

| 维度 | CLIP | ALBEF | SimVLM | **BLIP** |
|------|------|-------|--------|----------|
| 提出者 / 年份 | OpenAI 2021 | Microsoft 2021 | Google 2021 | Salesforce 2022 |
| 架构 | 双塔（ViT + Text Trans.） | ViT + BERT（+交叉） | 编码器-解码器（prefixLM） | ViT + BERT（编码器×2 + 解码器×1） |
| 训练目标 | ITC（softmax 对比） | ITC + ITM + MLM | LM（prefixLM） | **ITC + ITM + LM** |
| 理解（检索/匹配） | ✅ | ✅ | ❌ | ✅ |
| 生成（描述/VQA） | ❌ | ❌ | ✅ | ✅ |
| Momentum Encoder | ❌ | ✅（ITC） | ❌ | ✅（ITC） |
| Hard Negative Mining | ❌（batch 内随机负样本） | ✅ | ❌ | ✅（ITC 相似度加权） |
| 数据清洗机制 | ❌ | ❌ | ❌ | ✅ CapFilt |
| 预训练数据规模 | 400M | 14M | 约 1.8B | 129M |
| 定位 | 零样本对齐 | 对齐 + 融合理解 | 大规模生成 | **理解 + 生成统一框架** |

### 8.2 与 ALBEF 的关系（血缘最近）

BLIP 几乎是在 ALBEF 上"加生成能力 + 加数据清洗"：

| 改动 | ALBEF | BLIP | 动机 |
|------|-------|------|------|
| MLM → LM | 掩码语言建模 | 自回归语言建模 | 获得生成能力 |
| 文本编码器 | 独立 BERT | 与图文编码器共享自注意力/FFN | 省参数、促共训 |
| 解码器 | 无 | 新增因果解码器 | 生成 |
| 数据 | 直接用 14M | CapFilt 清洗 129M | 对抗数据噪声 |

### 8.3 与 SimVLM 的对比要点

- SimVLM 是"纯生成"路线：prefixLM 目标、无显式对比对齐、靠 1.8B 超大规模数据碾压；BLIP 是"对齐 + 生成"路线：小一个量级的数据（129M）也能取得可比的生成性能，同时免费获得检索能力；
- 面试结论句：**SimVLM 证明"大力出奇迹"，BLIP 证明"结构设计 + 数据清洗也能出奇迹"，且更便宜、能力更全。**

### 8.4 生态位置

```text
CLIP/SigLIP —— 纯对齐塔（理解）
CoCa       —— 对比 + 生成联合（无 ITM、无 CapFilt）
ALBEF      —— 对齐 + 匹配 + 掩码理解
BLIP       —— 对齐 + 匹配 + 生成 + 数据清洗  ← 本笔记主角
BLIP-2     —— BLIP 的下一代（Q-Former，见 07_BLIP2）
```

---

## 九、效果概览（关键 benchmark）

| Benchmark | 任务 | BLIP（ViT-B） | 当时的 SOTA 基线（ALBEF） |
|-----------|------|---------------|---------------------------|
| COCO 检索（零样本） | TR@1 / IR@1 | 82.4 / 64.1 | 约 77.5 / 59.7 |
| COCO 检索（微调） | TR@1 / IR@1 | 88.0 / 73.4 | 85.6 / 71.8 |
| COCO Caption（零样本） | CIDEr | 113.2 | 约 89（不可直接生成） |
| COCO Caption（微调） | CIDEr | 133.3（ViT-L） | 126.1 |
| VQA（test-dev） | VQA score | 78.25（ViT-B） | 76.79 |

> 使用提示：以上为论文报告值（ViT-B 为主），面试引用时说明"基于 129M 数据、ViT-B/16"即可，具体数字记忆 2-3 个关键的即可（零样本 CIDEr 113.2、VQA 78.25、COCO 检索 88.0/73.4）。

---

## 十、局限与 BLIP-2 的动机

BLIP-1 很成功，但暴露了几个结构性瓶颈：

| 局限 | 具体表现 | BLIP-2 的解法（预告） |
|------|----------|----------------------|
| 训练成本高 | 三目标 + 全部参数端到端更新，每加一点规模都要全量重训 | 冻结视觉塔 + 冻结 LLM，只训练轻量 Q-Former |
| 灾难性遗忘 | 联合训练会破坏视觉塔/文本侧已有的单模态能力 | 冻结使预训练知识完全保留 |
| 浪费单模态红利 | 大规模纯视觉（如 ViT-g）与纯语言（如 OPT）预训练成果无法复用 | 直接用现成最强单模态模型桥接 |
| 自举偏差 | CapFilt 依赖初始模型质量，错误认知可能被放大 | 用更好的冻结骨干规避部分偏差 |

> 一句话：**BLIP-2 把"训一个大模型"变成"训一个小桥"——用 Q-Former 连接冻结的视觉编码器和冻结的 LLM，成本低一个量级、能力却更强。**（详见 07_BLIP2 笔记）

---

## 十一、高频面试问答

**Q1：BLIP 全称是什么？一句话说清核心贡献。**
BLIP = Bootstrapping Language-Image Pre-training。两个核心贡献：一是用 ITC + ITM + LM 三个目标统一"理解 + 生成"两类任务；二是提出 CapFilt 自举机制，用模型自己生成的 caption 和过滤分数清洗海量噪声图文数据。

**Q2：为什么既要有 ITC 又要有 ITM？不重复吗？**
不重复。ITC 只比较两个全局向量（`[CLS]`），是粗粒度对齐，便宜但判别力弱；ITM 让文本 token 通过 cross-attention 逐字观察图像（`[Encode]`），是细粒度验证，能发现"整体像但细节不对"的样本。ITC 负责把表征拉到同一空间，ITM 负责质检细节，并且 ITM 的硬负样本恰好由 ITC 的相似度挑选——两者是协作关系。

**Q3：CapFilt 具体怎么运作？为什么叫"自举"？**
流程：初始 BLIP 在 COCO 上分别微调出 Captioner（LM 目标，生成 caption）和 Filter（ITM 目标，打分）。对每张网页图像，原始 alt-text 与 Captioner 合成 caption 一起交给 Filter 打分，按阈值（原始 0.5、合成 0.7）决定去留，再用清洗后数据重新预训练。叫"自举"是因为：模型先在有噪声的数据上训练，再用人工数据教出的能力反过来清洗自己的训练数据——不用额外人工标注，靠模型自己提升数据质量，形成闭环。

**Q4：为什么 ITM 要做 hard negative mining？怎么选负样本？**
随机负样本太容易区分，模型学不到细粒度判别。做法：用动量编码器的 ITC 相似度对 batch 内样本排序，按相似度成比例采样"最像但不匹配"的样本作负样本（1 正 : 3 负），逼模型只能靠局部细节区分。

**Q5：为什么只有 ITC 用 momentum encoder，ITM 和 LM 不用？**
ITC 的目标是"特征本身的一致性"，动量编码器特征变化缓慢，能提供稳定目标、防止表征坍塌，且不需要为特征区分反向传播；ITM 和 LM 需要完整梯度训练融合/生成能力（依赖图像细节），动量副本不反传梯度，无法承担这两个任务。

**Q6：BLIP 怎么把 VQA 变成生成任务？**
把"图文匹配/分类"改成"答案生成"：输入图像 + 问题 + `[Decode]` token，解码器自回归生成答案文本。优点是开放域，不受训练答案集限制，问题与答案的推理由 cross-attention 完成。

**Q7：BLIP 与 ALBEF 的核心区别？**
三处：一是 ALBEF 用 MLM（只能理解），BLIP 换成 LM（能生成）；二是 BLIP 的文本编码器与图像增强编码器共享自注意力/FFN 参数；三是 BLIP 新增 CapFilt 数据清洗，数据规模从 14M 提升到 129M。BLIP 可视为"ALBEF + 生成能力 + 数据自举"。

**Q8：为什么 BLIP 能同时做好理解与生成，而 CLIP/SimVLM 不行？**
CLIP 只有对比目标，表征是"判别式"的，丢掉了生成所需的 token 级细节；SimVLM 只有语言建模目标，没有显式对齐，检索/匹配能力弱。BLIP 把三种目标联合训练在共享架构上：ITC 负责对齐、ITM 负责验证、LM 负责产出，三者在同一套骨干上互相增强。

**Q9：CapFilt 的阈值为什么原始 caption 低、合成 caption 高？**
原始 alt-text 是"弱但真实"的人类信号，宁可多留；合成 caption 是模型自产，存在自我确认偏差，必须更高标准过滤，所以 τ_w = 0.5 < τ_s = 0.7。

**Q10：BLIP 的局限？为什么还要做 BLIP-2？**
BLIP-1 端到端全量训练成本高、易灾难性遗忘、无法复用大规模单模态预训练成果。BLIP-2 用 Q-Former 连接冻结的视觉塔和冻结的 LLM，只训练轻量桥接模块，训练成本降一个量级、知识保留更完整、性能更强。

---

## 十二、常见误区

**误区 1：BLIP 只是"CLIP 加了生成能力"。**
错。BLIP 有三目标联合 + 两个文本侧融合模块 + CapFilt 数据自举，是一个完整的"预训练框架 + 数据治理方案"，而不是简单的损失叠加。

**误区 2：CapFilt 是"用别的模型清洗数据"。**
错。Captioner 和 Filter 都是 BLIP 自己在 COCO 上微调的产物，是"用模型自己清洗自己的训练数据"——这正是"自举"的含义。

**误区 3：ITM 的负样本是随机采的。**
错。ITM 使用基于 ITC 相似度的硬负样本挖掘（相似度加权采样），随机负样本会导致 ITM 学不到细粒度判别，这是 BLIP（继承 ALBEF）的关键工程细节。

**误区 4：Momentum encoder 服务全部三个损失。**
错。只有 ITC 用动量编码器（提供稳定目标）；ITM、LM 必须用在线模型保证梯度完整。

**误区 5：BLIP 是对话/生成大模型，能直接做图文对话。**
错。BLIP 是预训练框架，产出的是"检索、匹配、描述、VQA 答案生成"能力；真正的对话式多模态模型（如 InstructBLIP、LLaVA）需要额外的指令微调。BLIP 是它们的基座，不是最终产品。

**误区 6：把三个 token 的职责搞混。**
`[CLS]` 管全局对齐（ITC）、`[Encode]` 管融合匹配（ITM）、`[Decode]` 管生成起点（LM）。回答架构题时能准确对应，面试观感完全不同。

---

## 十三、自我检验

- [ ] 能一句话讲清 BLIP 的核心贡献（统一理解+生成 + CapFilt 自举）
- [ ] 能说出 2021-2022 年多模态预训练"理解派 vs 生成派"的割裂现状
- [ ] 能画出四模块架构图并说明各自输入输出
- [ ] 能解释为什么文本侧要拆成"编码器 + 图像增强编码器 + 解码器"三个模块
- [ ] 能说清 Text Encoder 与 Img-grounded Text Encoder 的参数共享关系
- [ ] 能默写 ITC 对称对比损失公式并解释每个符号
- [ ] 能解释 momentum encoder 的 EMA 公式与作用（$m=0.995$）
- [ ] 能写出 ITM 二分类损失并描述 hard negative mining 流程
- [ ] 能写出 LM 交叉熵损失并解释因果掩码
- [ ] 能完整复述 CapFilt 四阶段流程（初始预训练 → 微调 Captioner/Filter → 打分过滤 → 重训）
- [ ] 能说出 τ_w=0.5、τ_s=0.7 的设计理由
- [ ] 能报出 2-3 个关键效果数字（零样本 CIDEr 113.2、VQA 78.25、COCO 检索 88.0/73.4）
- [ ] 能说出 BLIP 与 CLIP、ALBEF、SimVLM 的差异（架构/目标/能力）
- [ ] 能说出 BLIP-1 的局限与 BLIP-2 的动机（Q-Former 桥接冻结模型）
- [ ] 能完整回答 10 个高频面试追问
- [ ] 能区分 6 个常见误区

---

## 参考文献

1. [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — Li et al., NeurIPS 2022
2. [ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651) — Li et al., NeurIPS 2021
3. [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904) — Wang et al., ICLR 2022
4. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., ICML 2021
5. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al., ICML 2023
6. [Salesforce BLIP 官方代码](https://github.com/salesforce/BLIP)
