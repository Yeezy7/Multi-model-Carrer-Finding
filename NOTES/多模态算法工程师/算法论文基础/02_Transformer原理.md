# Transformer 原理：从 Attention 到 LLM 的基石

## 一、为什么需要 Transformer

### 1.1 序列建模的演进

在 Transformer 出现之前，序列建模主要靠 RNN 家族：

| 模型 | 核心思想 | 致命缺陷 |
|------|---------|---------|
| RNN | 逐时间步更新隐状态 h_t = f(h_{t-1}, x_t) | 无法并行、长依赖消失 |
| LSTM | 门控机制（遗忘/输入/输出门）+ 细胞状态 | 仍无法并行、长依赖仍困难 |
| GRU | 简化版 LSTM（两个门） | 同上 |
| **Transformer** | **全局自注意力 + 并行计算** | 计算复杂度 O(n²)（可接受） |

**RNN 的本质问题**：
1. **无法并行**：t 时刻的隐状态依赖 t-1 时刻，必须串行计算，训练慢；
2. **长距离依赖**：信息逐时间步传递，距离越远梯度越小（梯度消失），难以建模远距离关系；
3. **LSTM 只能缓解**：门控可以"记住"信息，但路径仍然过长。

### 1.2 Transformer 的三个核心创新

1. **Self-Attention**：任意两个位置可以直接交互，距离 O(1)，全局建模；
2. **完全并行**：attention 计算是矩阵运算，整条序列一次算出，GPU 并行友好；
3. **位置编码**：因为注意力对位置不敏感（置换等变），必须显式注入位置信息。

> **一句话**：Transformer = 全连接（自注意力） + 并行计算 + 显式位置编码 + 残差与归一化工程细节。

---

## 二、整体架构

以经典的 Encoder-Decoder 结构（如机器翻译）为例：

```text
输入序列 ──→ Embedding ──→ 位置编码 ──→ ┌── Encoder Block × N ──┐
                                        │  ① Self-Attention     │
                                        │  ② Residual + LayerNorm│
                                        │  ③ Feed-Forward       │
                                        │  ④ Residual + LayerNorm│
                                        └───────────────────────┘
                                                  │
                                            （K、V 来自 Encoder）
                                                  ▼
                                   ┌── Decoder Block × N ───────────┐
                                   │  ① Masked Self-Attention        │
                                   │  ② Cross-Attention（Q=解码端，  │
                                   │     K、V=编码端输出）            │
                                   │  ③ Residual + LayerNorm         │
                                   │  ④ Feed-Forward                 │
                                   └─────────────────────────────────┘
                                                  │
                                                  ▼
                                            Linear + Softmax
                                                  │
                                                  ▼
                                              输出序列
```

每一层的组成（现代标准结构）：

```text
x → LayerNorm → Self-Attention → +（残差）→ LayerNorm → FFN → +（残差）
```

> **注意**：原版 Transformer 是 Post-LN（先残差后 LN）；现代大模型（GPT、LLaMA、Qwen）普遍采用 **Pre-LN**（先 LN 再残差），训练更稳定。

---

## 三、Self-Attention 数学原理

### 3.1 从输入到 Q、K、V

输入序列 $X \in \mathbb{R}^{n \times d}$（n 个 token，每个 d 维），通过三个可学习矩阵映射：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$，$d_k$ 通常等于 $d$。

### 3.2 注意力分数

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

逐步拆解（以单个 query 为例）：

1. **点积打分**：$s_i = q \cdot k_i$——query 与每个 key 的相似度；
2. **缩放**：除以 $\sqrt{d_k}$——防止点积过大导致 softmax 梯度消失；
3. **Softmax**：归一化为权重，$\sum_i \alpha_i = 1$；
4. **加权求和**：$o = \sum_i \alpha_i v_i$。

### 3.3 为什么除以 $\sqrt{d_k}$（必考）

设 $q, k_i \in \mathbb{R}^{d_k}$ 各分量独立同分布，均值 0 方差 1，则：

$$q \cdot k_i = \sum_{j=1}^{d_k} q_j k_{i,j}$$

方差为 $d_k$。当 $d_k$ 大时，点积的方差也大，部分点积值会非常大。softmax 的梯度特性：输入过大 → softmax 输出接近 one-hot → 梯度接近 0 → **梯度消失**。除以 $\sqrt{d_k}$ 把方差重新归一化到 1，使 softmax 输入保持在梯度良好的区域。

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

### 3.4 计算复杂度

$$O(n^2 \cdot d)$$

- 注意力矩阵 $QK^T$ 是 $n \times n$，序列越长开销越大（二次增长）；
- 这是长上下文的核心瓶颈 → 引出 KV Cache、Flash Attention、稀疏注意力等优化（见下文及 15_推理优化）。

---

## 四、Multi-Head Attention（多头注意力）

### 4.1 动机

单个 attention 只能学习一种"关系模式"。多头让模型在**不同的表示子空间**并行捕捉不同关系：

> 例：一个头关注"主语-宾语"句法关系，另一个头关注"指代关系"，再一个关注"局部共现"。

### 4.2 计算过程

设 h 个头，每个头维度 $d_h = d / h$：

1. 将 Q、K、V 拆成 h 份：$Q_i = Q[:, i\cdot d_h : (i+1)\cdot d_h]$；
2. 每个头独立计算：$\text{head}_i = \text{Attn}(Q_i, K_i, V_i)$；
3. 拼接所有头输出：$[head_1; head_2; \dots; head_h]$；
4. 线性投影回 $d$ 维：$O = \text{Concat} \cdot W_O$。

### 4.3 关键数字

| 配置 | 值 |
|------|----|
| d_model | 512 |
| num_heads | 8 |
| d_k = d_v | 64 |
| 多头参数量 | 与单头相同（只是切分 + 重组） |

**多头不增加参数量**：总维度不变，只是把 $d$ 维空间切成 h 个子空间各自做注意力。

---

## 五、位置编码

注意力是**置换等变**的：交换两个 token 的位置，注意力输出也相应交换（因为只有内容交互，没有位置概念）。所以必须注入位置信息。

### 5.1 绝对位置编码（原版 Transformer）

正余弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- pos 是 token 位置，i 是维度下标；
- 不同维度频率不同（低频/高频混合），可表示任意长度；
- 性质：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，模型可学到相对关系。

### 5.2 可学习位置编码（BERT 风格）

直接初始化一个位置 embedding 矩阵 $P \in \mathbb{R}^{max\_len \times d}$，随训练更新。缺点：**外推性差**——超过 max_len 无法表示。

### 5.3 RoPE（旋转位置编码，现代大模型标配）

RoPE（Rotary Position Embedding）被 LLaMA、Qwen、InternVL 等广泛使用：

**核心思想**：把位置信息编码为旋转角度，施加在 Q 和 K 上，使**注意力分数只依赖相对位置**。

对位置 m 的向量 $x$，将其按 2 维一组旋转角度 $m \theta$：

$$f(x, m) = R(m) \cdot x, \quad R(m) = \begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix}$$

旋转后内积满足：

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

即注意力分数只和相对距离 $m-n$ 有关。优点：

| 特性 | 说明 |
|------|------|
| 相对位置 | 天然编码相对距离 |
| 外推性 | 可外推更长序列（配合 NTK/插值） |
| 无额外参数 | 只改变 Q、K 表示 |

### 5.4 位置编码对比总结

| 类型 | 代表 | 相对位置 | 外推能力 | 额外参数 |
|------|------|---------|---------|---------|
| 正弦/余弦 | 原版 Transformer | 隐式 | 理论上无限 | 无 |
| 可学习 | BERT | 无 | 差（截断） | max_len×d |
| RoPE | LLaMA/Qwen/Gemma | 显式 | 好（+插值） | 无 |
| ALiBi | BLOOM | 显式（线性偏置） | 好 | 无 |

---

## 六、残差连接与 LayerNorm

### 6.1 残差连接（Residual）

$$x_{out} = x_{in} + \text{SubLayer}(x_{in})$$

作用：
1. **缓解梯度消失**：梯度可以沿着残差捷径直接回传（恒等映射梯度为 1）；
2. **让深层网络可训练**：堆叠 100+ 层 Transformer 的基础；
3. 可以理解为每层在"学习增量"。

### 6.2 Pre-LN vs Post-LN（面试高频）

| 结构 | 顺序 | 特点 |
|------|------|------|
| Post-LN（原版） | Attention → 残差 → LN | 原始论文，深层训练不稳定，需要 careful warmup |
| **Pre-LN（现代）** | LN → Attention → 残差 | 训练稳定、免 warmup 也能收敛，被 GPT/LLaMA/Qwen 采用 |

Pre-LN 稳定性来源：残差路径上无 LN 阻挡，梯度畅通，但表达能力略有损失。

### 6.3 LayerNorm 细节

对单个样本的所有特征维度归一化（与 batch 无关）：

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

- $\mu, \sigma$ 是**该样本**特征维的均值/方差；
- $\gamma, \beta$ 是可学习参数（缩放/平移）；
- $\epsilon$（如 1e-5）防止除零；
- 相比 BatchNorm：与 batch size 无关、对变长输入友好、Transformer 标配。

---

## 七、Feed-Forward Network（FFN）

每层 attention 后接一个两层的 MLP：

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

| 细节 | 说明 |
|------|------|
| 中间维度 | 4× d_model（如 512 → 2048 → 512） |
| 激活函数 | 原版 ReLU；现代 GPT 系列用 GELU/SwiGLU |
| 参数量占比 | **约 2/3 的参数在 FFN** |
| 作用 | 逐 token 的非线性变换与信息混合 |

**为什么 attention 之外还要 FFN？** Attention 是做"token 之间"的信息交换（混合序列维度），FFN 做"每个 token 内部"的非线性变换（混合特征维度）。两者互补，缺一不可。没有 FFN 的堆叠 attention 表达能力有限（近似线性变换组合）。

**SwiGLU**（LLaMA 等现代模型）：$\text{SwiGLU}(x) = \text{SiLU}(xW_1) \otimes (xW_2)$，参数量为 2/3 d → 8/3 d，效果更好。

---

## 八、Mask（掩码）机制

Transformer 有三种 mask，**面试必考**：

### 8.1 Padding Mask

处理变长序列时，短序列补 `<pad>` token。padding 位置不应参与注意力：

```text
text:  [hello, world, <pad>, <pad>]
mask:  [  1,     1,     0,     0   ]   # 0 表示忽略
```

实现：mask 位置的值设为 $-\infty$，softmax 后权重为 0（而不是加 0，因为 softmax 会把 0 参与归一化）。

### 8.2 Causal Mask（因果掩码 / 下三角掩码）

Decoder 生成时，位置 i 只能看到 0~i 的 token，不能偷看未来：

```text
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

实现方式：`torch.tril(torch.ones(n, n))`，或 `attn_mask` 为下三角布尔矩阵。GPT 系列全程用 causal mask；Decoder-only 结构不需要 Cross-Attention。

### 8.3 Cross-Attention 中的 mask

Decoder 的 Cross-Attention：Q 来自 decoder，K、V 来自 encoder 输出，K、V 侧可加 padding mask（只对 encoder 输入有效位置做注意力）。

---

## 九、KV Cache：推理加速的核心机制

### 9.1 现象

GPT 类模型生成时是**逐 token 自回归**的：预测第 t 个 token 后，把它拼到序列尾部，再预测 t+1。

**朴素实现的问题**：每步都重新计算整个序列的 attention——前面的 Q、K、V 重复计算了 t 次，浪费算力。

### 9.2 解决方案

注意到：**已生成 token 的 K、V 不会变**（它们不依赖后续 token）。所以把每步算出的 K、V 缓存下来，下一步只计算新 token 的 K、V，与缓存的 K、V 拼接：

```text
第 1 步: 计算 K1, V1                     → 预测 token1
第 2 步: 只需计算 K2, V2，与缓存 [K1, V1] 拼接 → 预测 token2
第 3 步: 只需计算 K3, V3，与缓存 [K1,V1,K2,V2] 拼接 → 预测 token3
```

### 9.3 KV Cache 的显存代价

KV Cache 大小 = $2 \times \text{layers} \times \text{seq\_len} \times \text{hidden\_dim} \times 2\ \text{bytes (FP16)}$

以 7B 模型（32 层、4096 维）生成 2048 token：
$$2 \times 32 \times 2048 \times 4096 \times 2 \approx 1 \text{GB}$$

**这就是为什么长上下文 + 长输出显存爆炸**——KV Cache 随序列长度线性增长，是服务端推理显存的头号开销。优化手段：GQA（分组查询注意力）、MQA、PagedAttention（vLLM）、KV 量化等。

### 9.4 MHA vs MQA vs GQA

| 类型 | 结构 | 参数量 | KV Cache | 代表 |
|------|------|--------|----------|------|
| MHA | 每头独立 K、V | 大 | 大 | 原版 Transformer |
| MQA | 所有头共享 1 组 K、V | 小 | 最小 | PaLM |
| **GQA** | 每几组头共享 1 组 K、V | 中 | 中 | **LLaMA2/3、Qwen、InternVL** |

GQA 是质量与显存的折中：把 h 个头的 K、V 分成 g 组共享（g 通常 8），KV Cache 缩减 h/g 倍，质量损失很小。

---

## 十、Flash Attention：显存与速度优化

标准 attention 需要把 $QK^T$ 的 $n \times n$ 矩阵写入显存再读回 softmax，显存和 IO 开销大。Flash Attention 的核心理念：

1. **分块计算（tiling）**：把 Q、K、V 切成块，在 SRAM（片上高速缓存）内完成小块计算，避免大矩阵进出显存；
2. **在线 softmax（重计算）**：不存完整 softmax 统计量，用 running max/sum 技巧，多次扫描；
3. **效果**：显存从 O(n²) 降到 O(n)，速度提升 2~5 倍。

$$O(n^2) \to O(n) \text{ 显存占用}$$

> 面试要点：Flash Attention 不是改变 attention 数学，而是**改变计算方式（IO 优化 + 分块重算）**，结果数值上等价。

---

## 十一、训练技巧（Transformer 专属）

### 11.1 为什么用 AdamW + warmup + 梯度裁剪

| 技巧 | 原因 |
|------|------|
| AdamW | 权重衰减解耦，Transformer 训练标准（见 01_PyTorch） |
| 学习率 warmup | 训练初期 attention 对 lr 极敏感，大 lr 会震荡发散 |
| 梯度裁剪 | 深网络梯度爆炸的保险 |
| Label Smoothing | 防止过度自信，提升泛化 |
| Pre-LN | 训练稳定，深层可训 |

### 11.2 典型超参

| 超参 | 典型值 |
|------|--------|
| batch size | 越大越好（attention 训练对 batch 敏感） |
| lr | 3e-4 ~ 1e-3（预训练）；1e-5 ~ 3e-5（微调/指令微调） |
| weight_decay | 0.01 ~ 0.1 |
| dropout | 0.1 |
| warmup ratio | 前 1%~10% steps |
| 梯度裁剪 max_norm | 0.5 ~ 1.0 |

---

## 十二、Transformer 生态速览

### 12.1 三大流派

| 流派 | 结构 | 代表 | 任务 |
|------|------|------|------|
| Encoder-only | 双向注意力 | BERT、RoBERTa、DeBERTa | 理解、分类、抽取、embedding |
| Decoder-only | 因果注意力 | GPT、LLaMA、Qwen、DeepSeek | 生成、对话、通用 LLM |
| Encoder-Decoder | 双向 + 因果 | T5、BART、原版 Transformer | 翻译、摘要 |

**为什么 LLM 主流选 Decoder-only？** 训练目标统一（下一个 token 预测）、能处理任意任务（生成）、Scaling Law 友好（同样的数据量效果更好）、KV Cache 简化。多模态 VLM 也几乎全部是 Decoder-only LLM + 视觉编码器。

### 12.2 输入输出的张量形态

```text
输入:  [batch, seq_len]                    # token ids
Embedding: → [batch, seq_len, d_model]
Attention: → [batch, heads, seq_len, seq_len]  # 注意力分数矩阵
输出:  [batch, seq_len, vocab_size]        # logits
```

### 12.3 多模态中的 Transformer 角色

| 组件 | 用的 Transformer 变体 |
|------|---------------------|
| 视觉编码器 | ViT（Encoder 结构，见 03） |
| 文本编码器 | BERT 类（Encoder）或 LLM（Decoder） |
| 投影层 | 简单线性层 / Q-Former（见 07_BLIP2） |
| LLM 主体 | Decoder-only（LLaMA/Qwen 等） |
| 视频/多帧 | 3D 位置编码、时间 attention 扩展 |

---

## 十三、高频面试问答

**Q1：self-attention 和 cross-attention 的区别？**
Self-Attention：Q、K、V 都来自同一个序列（如编码器自身），建模序列内部关系。Cross-Attention：Q 来自一个序列（如 decoder），K、V 来自另一个序列（如 encoder），建模两个序列之间关系。

**Q2：为什么 attention 需要缩放？**
点积随维度增大方差增大，softmax 输入过大会导致梯度消失。除以 $\sqrt{d_k}$ 将方差归一化到 1。

**Q3：Transformer 相比 RNN 的优缺点？**
优点：并行、长依赖、可扩展（Scaling）。缺点：O(n²) 复杂度、无归纳偏置（需要大量数据）、位置信息需显式编码。

**Q4：为什么 Transformer 需要位置编码？**
注意力本身是置换等变的——对 token 顺序不敏感，交换顺序输出按同样方式交换，模型无法知道"谁在谁前面"。必须显式注入位置信息。

**Q5：解释 KV Cache 的原理和代价？**
见 9 节：缓存已算 token 的 K、V 避免重复计算；代价是显存随序列长度线性增长，长序列生成显存大。

**Q6：Pre-LN 和 Post-LN 区别？为什么现代模型用 Pre-LN？**
Pre-LN 训练更稳定（残差路径无 LN 阻碍），允许更大 lr、免 warmup 也能收敛；Post-LN 原版结构，深层需精细调参。现代 LLM 用 Pre-LN。

**Q7：MHA、MQA、GQA 的区别？**
MHA 每头独立 KV；MQA 全部头共享一组 KV（最省）；GQA 分组共享（折中，质量损失小、KV Cache 少）。现代大模型用 GQA。

**Q8：Decoder-only 模型如何理解上下文？**
通过 causal mask 使每个位置只能看到之前的位置，训练时并行计算所有位置的预测（teacher forcing），推理时逐 token 生成（自回归）。

**Q9：temperature 参数在生成中的作用？**
softmax 温度 $\tau$：$\text{softmax}(logits/\tau)$。$\tau$ 小 → 分布更尖锐（更确定），$\tau$ 大 → 分布更平（更随机）。注意这与训练中的 temperature（对比学习缩放）是两回事。

**Q10：什么是 teacher forcing？**
训练 decoder 时，用**真实的前缀 token** 作为输入（而非模型自己生成的 token），并行计算所有位置 loss；推理时用生成结果。训练推理分布不一致的问题叫 exposure bias。

---

## 十四、自我检验

- [ ] 能徒手写出 attention 公式并解释每个符号
- [ ] 能解释为什么除以 √d_k
- [ ] 能说清多头注意力的动机和计算流程
- [ ] 知道三种位置编码的区别和 RoPE 的原理
- [ ] 能解释 Pre-LN/Post-LN、残差、LayerNorm 的作用
- [ ] 知道三种 mask（padding/causal/cross）的用途
- [ ] 能讲清 KV Cache 的原理、显存代价和 GQA 优化
- [ ] 了解 Flash Attention 的核心思想（分块 + 在线 softmax）
- [ ] 知道为什么 LLM 用 Decoder-only
- [ ] 能对比 self-attention / cross-attention / masked self-attention
- [ ] 掌握 Transformer 训练技巧（AdamW/warmup/梯度裁剪）及其原因
