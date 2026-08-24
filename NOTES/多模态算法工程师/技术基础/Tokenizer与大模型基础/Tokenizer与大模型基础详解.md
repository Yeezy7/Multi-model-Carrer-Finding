---
learned: false
---

# Tokenizer与大模型基础详解：模型的第一道门

> 本模块索引见 [Tokenizer与大模型基础详解](Tokenizer与大模型基础详解.md)

## 子篇索引

| 文件名                                                      | 一句话简介                                                                                                           |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [BPE.md](BPE.md)                                         | 最经典的分词算法：从字符出发反复合并"最高频相邻对"，GPT 系列（byte-level 变体）与 LLaMA 都在用，附完整可运行实现与逐步合并演示                                     |
| [WordPiece与SentencePiece.md](WordPiece与SentencePiece.md) | 另一条技术路线：BERT 的 WordPiece（LM 打分 + ## 前缀）、Unigram LM 迭代剪枝、以及"不预分词、▁ 标记空格"的 SentencePiece 框架（日语/中文场景主力）            |
| [预训练目标与ScalingLaw.md](预训练目标与ScalingLaw.md)               | 模型学什么、长大多少：next-token / masked LM / prefix LM 三种目标、Chinchilla 数据-参数幂律（70B 为什么需要 1.4T tokens），附小 GPT 训练循环与幂律拟合演示 |


> 阅读顺序：先读本文建立"tokenizer 到底在整个大模型链路里干什么"的整体认知 → [BPE.md](BPE.md)（最主流算法，必读核心）→ [WordPiece与SentencePiece.md](WordPiece与SentencePiece.md)（理解替代方案与多语言场景）→ [预训练目标与ScalingLaw.md](预训练目标与ScalingLaw.md)（从分词走向模型训练本身）。

## 一、一句话解释

**Tokenizer（分词器）= 文本与整数 id 序列之间的桥梁，是大模型的"第一道门"——所有文本进入模型之前都必须先经过它。**

- 编码方向：文本 → 一串整数（token ids），整数指向词表（vocabulary）中的某一项；
- 解码方向：模型输出的 id 序列 → 文本（生成回答时最后一步）；
- 词表（vocab）：一个"id ↔ 词/子词/字符"的映射表，例如 $id=4632 \leftrightarrow$ `"机器"`、$id=1300 \leftrightarrow$ `"学"`。

大模型自己"看不见"文本，只能处理浮点向量。tokenizer 把离散文本变成离散整数，embedding 层再把整数变成连续向量，模型的世界才正式开始。所以面试里最基础的一问是："**一个句子从输入到输出，经过哪些步骤？**"——答案的第一环就是 tokenizer。

## 二、tokenizer 在 LLM/VLM 全链路中的位置

```
文本输入 "今天天气不错"
    │
    ▼
[Tokenizer 编码]          ← 本模块：文本 → token ids
"今天天气不错" → [104, 2563, 512, 882]
    │
    ▼
[Embedding 层]            权重矩阵 E ∈ R^(V×d)，V=词表大小
每个 id 查表 → 向量序列 x ∈ R^(T×d)
    │
    ▼
[Transformer 主干]        ← 注意力/FFN（见注意力机制模块）
若干层编码上下文，输出每个 token 的隐状态
    │
    ▼
[LM Head]                线性层 W ∈ R^(d×V) + softmax
预测"下一个 token"的概率分布 p ∈ R^V
    │
    ▼
[Tokenizer 解码]          ← 本模块：id 序列 → 文本
[104, 2563, 512] → "今天天气不"
```

几个关键观察：

1. **词表大小 $V$ 出现在链路的两个端点**：Embedding 矩阵 $E \in \mathbb{R}^{V \times d}$ 和 LM Head $W \in \mathbb{R}^{d \times V}$。$V$ 越大，模型首尾两层的参数越多；
2. **模型中间层完全不知道"词"的存在**，它看到的只是浮点数。语义的分界（哪里是词、哪里是子词）由 tokenizer 在入口处决定；
3. **训练和推理的每一步都在"分词后的世界里"**：语料先分词存成 id 再训练；推理时先分词、最后把输出 id 拼回文本。tokenizer 与模型绑定发布，换 tokenizer 等于换模型。

**VLM 里的第二路**：图像不经过 tokenizer，而是经过视觉编码器（ViT）切成 patch 后投影成视觉 token（见第六节）。

### 2.1 一个具体例子（GPT-2 风格 byte-level BPE）

以 GPT-2 tokenizer 为例，"hello world" 实际编码为两个 token（真实 id 来自 GPT-2 词表）：

```python
# gpt2 分词示意（id 为 GPT-2 真实值）
# "hello world" -> ["hello", " world"] -> ids [31373, 995]
# "I love you"  -> ["I", " love", " you"] -> ids [40, 1842, 345]
```

注意 **英文的空格通常挂在下一个 token 前面**（" world"），这是 GPT 系列 byte-level BPE 的惯例——解码时直接拼接即可恢复原文本。中文模型（如 LLaMA 的 SentencePiece 版）则用 `▁` 标记词首：`"我爱北京" -> ["▁我", "爱", "北京"]`。

### 2.2 为什么"换 tokenizer = 换模型"

- 训练语料、预处理、tokenizer、模型权重四者是一套整体，任何一个环节变了，id 的语义映射就失效；
- 开源模型的 tokenizer 文件（如 `tokenizer.json` / `tokenizer.model` / `vocab.json + merges.txt`）必须随权重一起发布；
- 面试陷阱题：**"能不能给 LLaMA 换个更大的词表继续微调？"**——不能直接换（embedding 维度变了、既有 id 语义全乱），只能做词表扩展（把新 token 追加到词表尾并随机初始化对应 embedding 行），且需要谨慎重训。

## 三、主流分词算法总览

| 算法 | 基本思想 | 代表模型 | 词表规模 | 备注 |
| --- | --- | --- | --- | --- |
| 词级（Word） | 按空格/标点切整词 | 早期统计模型 | 数十万 | 词表爆炸、OOV 严重，已淘汰 |
| 字符级（Char） | 每个字符一个 token | 早期 RNN | 几十~几百 | 序列过长，信息密度低 |
| **BPE** | 从字符开始，反复合并**最高频**相邻对 | GPT-2/GPT-4、LLaMA、DeepSeek | 32K~128K | 语言无关，词表可控 |
| **Byte-level BPE** | 先用 UTF-8 编码成字节，再在字节上做 BPE | GPT-2 及其后几乎所有 LLM | 基础 256 → 合成数万 | 完美覆盖任何 Unicode 文本 |
| **WordPiece** | 合并**使语言模型似然提升最大**的对，## 前缀标记词内片段 | BERT、ELECTRA | ~30K | 英文最佳实践 |
| **Unigram LM** | 先造大词表再迭代**剪枝**，保留使语料损失最小的子词 | SentencePiece 默认、T5、Gemma | ~32K | 概率化、可采样 |
| **SentencePiece** | 不是算法而是**框架**：不预分词 + ▁ 标记空格 | LLaMA/T5/Gemma 等多语言模型 | 32K~128K | 日语/中文友好，见子篇三 |

> 记忆要点：**BPE 看频率，WordPiece 看似然，Unigram 看剪枝，SentencePiece 是载体**。完整对比见 [WordPiece与SentencePiece.md](WordPiece与SentencePiece.md) 第五节。

### 3.1 特殊 token（每个词表都预留的元符号）

| 符号 | 含义 | 典型 id 约定 |
| --- | --- | --- |
| `<bos>` / `<s>` | 序列开始 | 1（或 2） |
| `<eos>` / `</s>` | 序列结束 | 2（或 1） |
| `<pad>` | 批内对齐填充 | 0 |
| `<unk>` | 词表外字符 | 1（BERT） |

多模态与指令模型还会追加 `<image>`、`<|im_start|>`、`<|im_end|>` 等。**这些 token 不能出现在正常语料里**，训练与推理有严格约定，是很多数据 bug 的来源。

## 四、为什么 tokenizer 重要

### 4.1 词表大小 ↔ Embedding 显存

Embedding 层（不含 Adam 状态）的参数量 = $V \times d$。BF16 下每个参数占 2 字节：

$$\text{Embedding 显存} = V \times d \times 2 \text{ 字节}$$

| 词表 $V$ | 维度 $d$ | 参数量 | BF16 显存 |
| --- | --- | --- | --- |
| 32K | 4096 | 1.34 亿 | 268 MB |
| 50K | 4096 | 2.05 亿 | 410 MB |
| 100K | 8192 | 8.19 亿 | 1.64 GB |
| 250K | 8192 | 20.5 亿 | 4.1 GB |

Embedding 和 LM Head 通常**权重共享**（tied），参数翻倍影响更大。词表从 32K 涨到 250K，光首尾两层就多出近 40 亿参数——这就是为什么多语言模型要精打细算词表。

### 4.2 分词粒度 ↔ 训练效率与效果

- **粒度太细（字符级）**：序列变长。一个 1000 词的句子字符级约 5000~8000 token，注意力计算是 $O(T^2)$，训练与推理成本随序列长度平方上涨；上下文窗口（如 4K）能装的"内容"也大幅缩水；
- **粒度太粗（整词级）**：词表爆炸（数十万），每个词的样本稀少学不好，且无法表达新词/变形词（run、runs、running 是三个 id）；
- **子词级（当前主流）**：常见词一个 id，罕见词拆成多个子词，词表固定在 3 万~10 万级别。经验上**中文一个汉字约 0.6~1.5 token，英文一个单词约 1~2 token**。

### 4.3 多语言 ↔ 词表膨胀

- 英文：26 个字母，BPE 合并后 32K 词表覆盖极好；
- 中文：常用汉字 3500~10000，Unicode 全集 9 万+，字符级即可工作，子词提升有限；
- 多语言混合：每种语言都要"分一杯羹"，词表被迫膨胀（如 250K），embedding 显存与训练数据稀疏性同时恶化——**多语言是词表设计最难的问题**，常见对策是 byte-level 分词（[BPE.md](BPE.md) 第四节）与共享子词。

### 4.4 分词是不可逆的：数据侧陷阱

1. **分词结果决定训练数据的样子**：同一条语料，用 32K 和 128K 词表分词，token 序列完全不同，模型学到的"单元"也不同；
2. **平均 token/字 是成本系数**（直接影响训练数据量、上下文容量、推理时延）：

| 文本类型 | 经验值（token / 字符） |
| --- | --- |
| 英文 | 约 0.70~0.80 |
| 中文 | 约 0.90~1.10 |
| 代码 | 约 0.60~0.75 |
| 多语言混排 | 约 0.80~1.20 |

同样的 1 亿字语料，中文分出的 token 比英文多约 30%，训练成本按比例上涨——多语言模型的"语言配比"设计（[预训练目标与ScalingLaw.md](预训练目标与ScalingLaw.md) 第六节）必须带上分词粒度这个系数；
3. **数字与标点**：是否把 `123456` 拆成单数字 token（`split_digits`）影响数学能力；引号是否归一化影响鲁棒性。这些细节藏在 SentencePiece 的训练参数里（见 [WordPiece与SentencePiece.md](WordPiece与SentencePiece.md) 第三节）。

## 五、大模型基础概念速览

这一节把"tokenizer 之后"的模型基础知识先铺一遍，细节在后续模块展开。

### 5.1 Prefill 与 Decode：推理的两个阶段

- **Prefill（预填充）**：把用户输入的 $T$ 个 token 一次并行送入模型，算出所有位置的隐状态与 KV Cache，得到第一个输出 token 的概率分布。耗时短、计算密集；
- **Decode（解码/自回归生成）**：每步只输入 1 个新 token，模型读取 KV Cache + 新 token，预测下一个 token，循环直到生成 EOS。**逐 token 串行**，是推理时延的主要来源；
- 总时延 ≈ prefill 一次 + decode $N$ 次（$N$ 为生成长度）。KV Cache 显存 ≈ $2 \times \text{层数} \times \text{头数} \times \text{头维} \times \text{序列长}$，长上下文下是关键瓶颈。

一个 KV Cache 的数量级示例：LLaMA-7B（32 层、32 头、头维 128），8K 上下文：

$$\text{KV Cache} = 2 \times 32 \times 32 \times 128 \times 8192 \text{ 参数} = 21.5 \text{ 亿} \approx 4.3 \text{ GB (BF16)}$$

这解释了为什么 8K 窗口在 7B 模型上就吃掉一张 A100 的一成多显存，也解释了为何视觉 token（一张图 576 个）会显著放大 KV 成本（详见 [分布式训练详解](../分布式训练/分布式训练详解.md) 与 [GQA与MQA](../注意力机制专题/GQA与MQA.md)）。

### 5.2 Next-Token 预测：唯一的训练目标

LLM 的训练目标只有一个：**给定前 $t$ 个 token，预测第 $t+1$ 个 token**。损失是交叉熵：

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log p(x_{t+1} \mid x_1, x_2, \dots, x_t)$$

"上下文 → 下一个词"这一件事反复做万亿次，就涌现出语言能力。目标本身的细节（含 masked LM、prefix LM 的对比）见 [预训练目标与ScalingLaw.md](预训练目标与ScalingLaw.md) 第一节。

### 5.3 上下文窗口（Context Window）

- 定义：模型一次能"看到"的 token 上限（如 4096、32768、128K），由位置编码与训练长度共同决定；
- 训练窗口 vs 外推：在 4K 上训练的模型直接处理 32K 文本会崩（注意力 logits 尺度失配、位置编码未见），需要 RoPE/ALiBi 等外推技术，见 [长度外推技术](../位置编码专题/长度外推技术.md)；
- 上下文窗口的单位是 **token 而不是字**——所以分词粒度直接决定"4K 窗口能装多长的对话"。

### 5.4 温度与 Top-P：解码策略

- **温度 $T$**：采样前把 logits 除以 $T$，$T<1$ 更确定、$T>1$ 更随机；$T \to 0$ 退化为贪心。数学上是对概率分布做"锐化/平滑"：

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- **Top-P（核采样）**：只从累计概率达到 $P$ 的最小 token 集合里采样，动态截断低概率尾巴；
- 公式与代码见 [解码策略详解](../解码策略/解码策略详解.md)（[温度采样.md](../解码策略/温度采样.md)、[TopK与TopP.md](../解码策略/TopK与TopP.md)）。

### 5.5 Scaling Law：一句话版

**模型越大、数据越多，loss 按幂律下降，但收益递减；算力固定时存在"最优的模型-数据配比"**（约 1 参数配 20 token，Chinchilla）。70B 模型要 1.4T tokens 数据、$6ND \approx 5.9 \times 10^{23}$ FLOPs——详见 [预训练目标与ScalingLaw.md](预训练目标与ScalingLaw.md) 第二节。

## 六、多模态中的 tokenizer

VLM（视觉语言模型）里存在**两套"token"**，最后拼接进同一条序列：

1. **文本 token**：走常规 tokenizer，与纯 LLM 完全一致，id 来自文本词表；
2. **视觉 token**：图像被切成 patch（如 $14\times14$ 像素），ViT 编码成特征后经投影层变成"伪 token"，每个伪 token 是一个 $d$ 维向量，**没有对应的字符串，不经过文本 tokenizer**。

```python
# 文本: "一张猫的照片"  → ids [104, 882, ...]（文本词表）
# 图像: 1 张 336x336 图 → 24x24 = 576 个 patch → 576 个视觉 token
# 拼接后的模型输入序列（示意）:
#   [<image>] [<image>] ... [<image>] | 一 张 猫 的 照 片
#     576 个视觉 token                      7 个文本 token
```

几个常见设计：

- **占位符方案（LLaVA 系）**：文本流里放一个特殊 `<image>` token，模型内部把视觉特征替换进去——文本 tokenizer 只需认识 `<image>` 这一个"新词"；
- **图像 token 不参与文本词表**：视觉 token 的"词表大小"由 patch 分辨率和特征维度决定，与文本词表 $V$ 无关，这是"两套 token"的本质；
- **数量失衡**：一张图动辄 256~1024 个视觉 token，远超一句话的文本 token 数。视觉 token 的具体数量由分辨率与 patch 大小决定：

| 输入分辨率 | patch 大小 | 视觉 token 数（$H/14 \times W/14$） |
| --- | --- | --- |
| 224×224 | 14×14 | 256 |
| 336×336 | 14×14 | 576 |
| 1024×1024（高分辨率文档） | 14×14 | 5476 |
| 1280×720（视频单帧） | 14×14 | 4704 |

- MoE 模型的稀疏激活、视觉 token 压缩（Pixel Shuffle）都与这个失衡直接相关，见 [MoE详解](../MoE/MoE详解.md) 第八节；
- **特殊 token**：`<bos>` `<eos>` `<pad>` `<unk>` 是 tokenizer 里预留的元符号，多模态还常加 `<image>`、`<|im_start|>` 等指令 token。

## 七、高频面试问答

**Q1：为什么模型不直接吃字符，非要分词？**
字符粒度信息密度低、序列太长，$O(T^2)$ 注意力成本不可接受；整词粒度词表爆炸、OOV 严重。子词粒度在"词表大小"与"序列长度"之间取平衡。

**Q2：Tokenizer 的词表和模型的 embedding 是什么关系？**
embedding 层就是一个 $V \times d$ 的查表矩阵，行下标就是 tokenizer 输出的 id；解码时 LM Head 是 $d \times V$，softmax 后 argmax 出的 id 再由 tokenizer 反查成文本。三者（tokenizer、embedding、LM Head）绑定发布。

**Q3：为什么 tokenizer 要和模型一起保存？**
训练时语料按该词表分词，模型学到的是"这个分词方式下的模式"；换了词表，id 与语义的对应全变，模型直接不可用。

**Q4：中文为什么不太需要复杂分词？**
中文以字为基本单位，字本身语义密度高，字符级/子词级差异小；而英文单词由字母组合而来，子词合并收益巨大。真正的难题是中日韩+多语言混合时的词表平衡。

**Q5：一个 4K 上下文窗口的模型，能装下多长的英文对话？**
英文约 0.75 token/字，4K 窗口约 5000~6000 字符（约 1000 词）；中文约 1 token/字，4K 窗口约 4000 字。所以"上下文多长"先问分词粒度。

**Q6：视觉 token 算 tokenizer 吗？**
不算文本 tokenizer，但概念同构：把连续信号离散成模型序列里的"token"。多模态里的"两套 token"（文本 id + 视觉向量）在同一条序列里共存，是理解 VLM 架构的起点。

**Q7：为什么推理要分 prefill 和 decode 两个阶段？**
输入阶段所有 token 并行、可一次算完（prefill），输出阶段每步依赖上一步的输出、必须串行（decode）。两个阶段的计算特征不同：prefill 是 compute-bound，decode 是 memory-bound，优化手段也因此不同（连续批处理、KV Cache、投机解码等）。

**Q8：Token 数怎么影响推理成本？**
API 按 token 计费；每多一个 token，decode 多一步、KV Cache 多一行。用户输入越长，prefill 与 KV 开销越大；生成长度越长，decode 步数越多。因此"prompt 压缩"和"控制输出长度"是降本两大杠杆。

## 八、自我检验

- [ ] 能用一句话说出 tokenizer 是什么、在链路中处于什么位置
- [ ] 能画出"文本→tokenizer→embedding→主干→LM Head→tokenizer→文本"全链路
- [ ] 能说出 BPE / WordPiece / Unigram / SentencePiece 各自的"一秒钟总结"
- [ ] 会算词表 $V$ 对 embedding 显存的影响（$V \times d \times 2$ 字节）
- [ ] 能说出 prefill / decode、next-token 目标、上下文窗口、温度 / Top-P 的定义
- [ ] 能讲清多模态里文本 token 与视觉 token 如何在一条序列中共存
- [ ] 已通读 [BPE.md](BPE.md) 并完成其自我检验
