# 参数高效微调 PEFT 详解：LoRA、QLoRA、Adapter、Prompt Tuning 全谱系

## 子篇索引

| 文件名 | 一句话简介 |
|--------|-----------|
| [LoRA.md](LoRA.md) | 低秩分解 W+BA：B=0 初始化、α/r 缩放、target_modules、推理合并，附手写 LoRALayer 与 peft 全流程代码 |
| [QLoRA.md](QLoRA.md) | NF4 4-bit 量化基座 + bf16 LoRA：双重量化、分页优化器，7B 微调显存从 ~140GB 压到 ~16GB，附手写量化代码 |
| [Adapter.md](Adapter.md) | Bottleneck 适配器 d→r→d：串行 vs 并行、残差与零初始化、手写 AdapterLayer 注入，与 LoRA 的对比 |
| [Prompt与PrefixTuning.md](Prompt与PrefixTuning.md) | 软提示家族：Prompt Tuning（输入侧）、Prefix Tuning（每层 K/V）、P-Tuning v1/v2，手写 soft prompt 实现 |

## 一、为什么需要 PEFT

### 1.1 全参微调（Full Fine-tuning）的三座大山

大模型时代，对 7B/70B 参数模型做全参微调（更新全部权重）面临三个现实问题：

| 问题 | 具体表现 | 量级估计（以 7B 模型为例） |
|------|---------|--------------------------|
| **显存爆炸** | 需同时存参数 + 梯度 + 优化器状态（Adam 需要 fp32 的 m、v 两份动量） | 参数 14GB(bf16) + 梯度 14GB + Adam 状态 56GB(fp32×2) ≈ **84GB+**，加激活值轻松破 100GB |
| **存储负担** | 每个任务一份完整模型副本，多任务就是多份 14GB | 10 个任务 = 140GB+，部署/分发成本高 |
| **灾难性遗忘** | 全量更新会破坏预训练学到的通用能力（知识遗忘、指令跟随退化） | 下游任务过拟合，通用能力下降 |

**核心矛盾**：预训练成本极高（数千 GPU 卡时），我们希望在**只移动少量参数**的前提下让模型适配新任务，这就是 PEFT 的出发点。

### 1.2 直觉：微调任务是"少量参数即可适配"的

- 预训练已经让模型具备了强大的通用表示；
- 下游任务（分类、指令跟随、多模态对话）与预训练任务是**同分布附近的迁移**，不需要大范围改写权重；
- 实验观察：**微调时权重更新量 $\Delta W$ 非常小**（相对 $W$ 而言），且**集中在一个低维子空间**中——即 $\Delta W$ 是低秩的（LoRA 论文的核心观察）。

$$W_{\text{new}} = W_{\text{pretrained}} + \Delta W, \qquad \text{rank}(\Delta W) \ll \min(d_{\text{in}}, d_{\text{out}})$$

### 1.3 PEFT 的定义与分类

PEFT（Parameter-Efficient Fine-Tuning）：**冻结绝大部分预训练参数，只训练少量新增/选中的参数**，以极低成本达到接近全参微调的效果。

| 家族 | 代表方法 | 训练什么 |
|------|---------|---------|
| 增量式（Additive） | LoRA、Adapter、Prompt Tuning、Prefix Tuning | 新增参数 |
| 选择式（Selective） | BitFit | 选中一部分原参数 |
| 重参数化（Reparameterization） | LoRA 本质属于此类（低秩分解） | 用低秩矩阵近似更新量 |

> **面试记忆点**：PEFT 的性价比公式 = 极小可训练参数（0.01%~1%） × 接近全参的效果（90%~99%） × 大幅降低的显存/存储。

---

## 二、LoRA（Low-Rank Adaptation）—— 最主流的 PEFT

### 2.1 动机：低秩假设

论文（Hu et al., ICLR 2022）观察到：
1. **预训练模型的权重本身是满秩/高秩的**，表达了通用知识；
2. **微调时的更新量 $\Delta W$ 是低秩的**——可以压缩到一个低秩子空间中（论文通过 $\Delta W$ 的奇异值分布验证，绝大多数奇异值接近 0）。

结论：与其直接优化高维的 $\Delta W$，不如**把它参数化为两个低秩矩阵的乘积**，大幅减少可训练参数量。

### 2.2 数学形式

对任意线性层 $W_0 \in \mathbb{R}^{d \times d}$（或多模态中 $W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$），前向变为：

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

其中：
- $A \in \mathbb{R}^{r \times d}$：**高斯初始化**（$\mathcal{N}(0, \sigma^2)$，$\sigma$ 通常 0.02）；
- $B \in \mathbb{R}^{d \times r}$：**零初始化**；
- $r \ll d$：秩（rank），通常 4~64；
- 训练时只更新 $A, B$，$W_0$ 冻结。

训练完成后的等效权重：$W' = W_0 + B A$。

### 2.3 为什么 B 初始化为 0（必考）

**核心原因：微调要从"预训练权重出发"，而不是从随机状态出发。**

- 若 $B$ 也高斯初始化，则初始时 $BA \neq 0$，模型一开始的输出就被注入了随机扰动，等于偏离了预训练权重，破坏了"在预训练能力上做增量适配"的前提；
- $B = 0$ 保证微调**初始时 $W' = W_0$**，前向/反向完全等价于原模型，训练从预训练起点平滑出发；
- 对称地，$A$ 必须非零初始化（高斯），否则 $BA \equiv 0$，梯度也为 0，**参数永远不更新**。

> 同理，$A$ 若也零初始化会导致梯度消失（$\frac{\partial L}{\partial A} \propto B^T$，$B=0$ 时梯度为 0）。

### 2.4 为什么 LoRA 有效（效果解释，必考）

1. **低秩假设成立**：微调更新量本质上落在低维子空间，$B A$（秩 ≤ r）足以表达 $\Delta W$ 的主要成分；
2. **论文实验佐证**：在 GPT-3 175B 上，$r = 4$（可训练参数仅 0.01%）就已接近全参微调的效果，$r$ 增大到 8/16 并无明显增益——说明有效更新量秩很低；
3. **约束即正则**：把更新量限制在低秩空间，天然防止过度拟合下游数据（与全参微调相比更抗遗忘）；
4. **正交于原权重**：$W_0$ 负责通用能力，$BA$ 负责任务增量，两条信息通路互不干扰。

### 2.5 超参数详解

| 超参 | 含义 | 常见取值 | 说明 |
|------|------|---------|------|
| $r$ | 低秩秩 | 4~64 | 越小参数越少；任务越难/数据越多，可取更大 r |
| $\alpha$ | 缩放常数 | 8~32 | 与 r 配合，实际缩放为 $\alpha / r$ |
| $\alpha / r$ | 有效缩放系数 | 一般 1~4 | 控制 LoRA 分支的学习强度 |
| dropout | LoRA 分支上的 Dropout | 0.05~0.1 | 加在 $A x$ 之后或 $BAx$ 之前，防过拟合 |
| target_modules | 加 LoRA 的模块 | q/k/v/o/gate/up/down | 决定哪些线性层被注入 |

**为什么是 $\frac{\alpha}{r} \cdot BAx$ 而不是 $BAx$？**

- 直接修改 $\alpha$ 相当于改变 LoRA 分支的整体学习率，而 $r$ 变化会同时改变参数数量；
- 用 $\alpha/r$ 归一化后，**改 r 时不用重调学习率**：先定 r 再调 α/r 即可；
- 实际操作中常设 $\alpha = 2r$（有效缩放为 2）或直接用固定比例。

> 数学形式（带缩放）：$$h = W_0 x + \frac{\alpha}{r} B A x$$

### 2.6 target_modules 怎么选

Transformer 中的候选线性层：

| 模块 | 作用 | 是否推荐加 LoRA |
|------|------|----------------|
| q/k/v | Attention 投影 | **必加**（LoRA 论文主实验对象） |
| o | Attention 输出 | 常加 |
| gate | FFN 门控（LLaMA 类） | 常加 |
| up/down | FFN 上下投影 | 常加 |
| embedding / lm_head | 词表 | 一般不 LoRA（参数量大、易崩），有时加 embedding |
| 视觉投影层（多模态） | 视觉-语言桥 | 见第八节 |

经验：训练 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` 全加（LLaMA 系 7B 时约 4.5M 可训练参数，约 0.13%），效果稳定。

### 2.7 推理：可合并（Merge）也可不合并

**合并模式**：训练完把两个分支相加，得到新权重 $W' = W_0 + \frac{\alpha}{r} B A$，导出为新模型：

- 推理时**零额外延迟**（跟原模型结构完全一致）；
- 部署简单，一个权重文件搞定；
- 代价：每个任务一份模型副本，失去 LoRA 的多任务优势。

**分支模式（不合并）**：保留 $W_0$ + 多份 $BA$：

- 同一基座加载多组 LoRA（如 LoRA 路由/切换器），**服务多任务/多用户只需一份基座 + 多个小分支**；
- 每个 LoRA 分支只有几 MB~几十 MB，可动态热插拔；
- 代价：推理时多一次 $BAx$ 的矩阵乘法（延迟增加，可用 LoRA-X 等技巧优化）。

> **面试记忆点**：合并是"效果等价、零延迟"；分支是"省显存、多任务灵活"。多模态推理服务常用分支模式。

### 2.8 参数量与显存账本

以 7B LLaMA、$r = 8$、全 attention+FFN 注入为例：

| 项目 | 全参微调 | LoRA |
|------|---------|------|
| 可训练参数 | 7B（100%） | ≈ 8.4M（约 0.12%） |
| 可训练参数权重（bf16） | 14GB | 17MB |
| 需要保存的优化器状态 | 56GB（fp32 Adam） | ≈ 67MB |
| 梯度 | 14GB | ≈ 17MB |
| 基座权重 | 需梯度/状态（占显存大头） | **冻结**（仍加载在显存，但无梯度与优化器状态） |
| 训练峰值显存（估算） | 70~100GB+ | **16~30GB**（主要被冻结基座 + 激活值占据） |
| 单卡训练 | 需多卡/流水并行 | **单卡（24G）即可** |

> 为什么省这么多：**Adam 优化器状态（m、v 两份 fp32）是显存大头**（约 2×参数量×4 字节）。LoRA 只对极少参数维护优化器状态，冻结的基座权重只需以 bf16 加载。

---

## 三、QLoRA（Quantized LoRA）—— 单卡微调利器

### 3.1 思路：先量化基座，再 LoRA

QLoRA（Dettmers et al., 2023）的核心理念：**基座权重的精度可以大幅压缩而不损失能力**——因为它完全冻结、不参与梯度更新，只需在反传时即时反量化参与计算。

- 基座权重：4-bit 量化存储（冻结）；
- LoRA 分支：bf16 训练（可训练部分保持高精度）；
- 前向/反向：把 4-bit 权重反量化回 bf16 计算；
- 结果：7B 模型微调显存从 ~70GB 降至 **~16GB** 量级，一块 16~24GB 消费级显卡即可微调 7B。

### 3.2 归一化 4-bit 量化（NF4）简述

**NF4（NormalFloat4）**是 QLoRA 提出的量化格式，专门针对**神经网络权重近似正态分布**这一先验：

- 标准均匀量化把 16 个档位均匀分布在 $[-1, 1]$，但权重集中在中部，两端档位浪费；
- NF4 用**标准正态分布的分位数**来定 16 个档位（分位函数 $\Phi^{-1}(k/16)$），让每个档位覆盖的概率质量相等，**量化误差最小化**（信息论意义上的最优）；
- 量化公式：$x_q = \text{round}(\frac{x}{s}) $，反量化 $x \approx s \cdot x_q$，其中 $s$ 是放缩因子。

### 3.3 双重量化（Double Quantization）

- 每个权重块（如 64 个权重一组）需要一个 fp32 放缩因子 $s$；
- 放缩因子本身也占显存（每 64 个权重 +4 字节 ≈ 每权重 +0.5 bit）；
- 双重量化：**把放缩因子再做一次 8-bit 量化**，使平均显存再降约 0.37 bit/权重；
- 例：7B 模型仅权重项从 fp16 的 14GB 降到 4-bit 的 ~3.5GB。

### 3.4 为什么 4-bit 损失这么小

1. **权重是高度冗余的**：LLM 权重分布接近正态且存在大量冗余，4-bit 分位量化已能捕捉主体信息；
2. **NF4 针对正态分布优化**，比普通均匀 4-bit 量化误差更小；
3. **可训练部分仍是 bf16**：LoRA 分支不量化，微调信号质量有保证；
4. **bf16 反量化计算**：量化只影响存储，计算时还原为 bf16，数值范围大、防溢出，误差可控。

> 实验结论（论文）：QLoRA 在 4-bit 下微调的效果与 16-bit LoRA **几乎持平**，甚至在部分任务上更好（量化本身带来轻微正则化）。

### 3.5 QLoRA 显存数字示例（7B）

| 显存构成 | 全参微调 | LoRA(bf16) | QLoRA(4-bit) |
|----------|---------|-----------|--------------|
| 基座权重 | 14GB(bf16) | 14GB(bf16) | **3.5GB(4-bit)** |
| LoRA/梯度/优化器 | 84GB+ | ~0.2GB | ~0.2GB |
| 激活值（部分 offload） | 大 | 大 | 大 |
| **合计（量级）** | **~100GB** | **~20-30GB** | **~16GB（16G 卡可跑）** |

> 工程注意：QLoRA 训练时**不要把 LoRA 分支也量化**；推理时可用 GPTQ/AWQ 等更极端的量化。

---

## 四、Adapter（适配器）—— 最早的 PEFT 之一

### 4.1 串行 Adapter（Houlsby et al., 2019）

在每个 Transformer 子层后插入一个小型前馈网络（bottleneck 结构）：

```text
x → Attn → Adapter → 残差+ → x'     （或 FFN 后同样插入）
```

**Bottleneck 设计**：$d \to r \to d$（$r \ll d$，如 64）：

$$\text{Adapter}(x) = W_{\text{out}} \, \sigma(W_{\text{in}} x + b_{\text{in}}) + b_{\text{out}}, \qquad W_{\text{in}} \in \mathbb{R}^{r \times d},\; W_{\text{out}} \in \mathbb{R}^{d \times r}$$

- 先降维（压缩信息）、非线性、再升维（恢复维度）；
- 参数量 ≈ $2 \times d \times r$（两个投影矩阵），通常只有全模型的 0.5%~5%；
- **残差连接**（$x + \text{Adapter}(x)$）：保证初始时输出与原模型一致（Adapter 随机初始化时残差让扰动可控），训练稳定。

### 4.2 并行 Adapter（Parallel Adapter）

```text
x → [Attn 分支 + Adapter 分支] → 求和 → 残差
```

- Adapter 与原子层并行、输出相加；
- 优点：不增加网络深度（串行会加深度，推理延迟更高）；梯度路径更短；
- 与 LoRA 的数学形式高度相似：**LoRA 可以看作并行 Adapter 的特例**（对线性层做 $W_0 x + BA x$）。

### 4.3 Adapter 与 LoRA 的效率对比

| 维度 | Adapter | LoRA |
|------|---------|------|
| 可训练参数 | 0.5%~5%（通常高于 LoRA） | 0.01%~1% |
| 推理延迟 | 串行 Adapter **增加延迟**（多了两层小网络）；并行 Adapter 几乎无感 | 合并后**零延迟** |
| 需要改动网络结构 | 是（插入新模块） | 是（但只改前向公式） |
| 与量化/编译工具的兼容性 | 较差（新增结构需定制） | 好（可合并回原权重） |
| 效果 | 与 LoRA 相当 | 与全参微调差距小 |

> **面试记忆点**：LoRA 可以合并权重 → 推理零开销；Adapter 一般不能"折叠"回原权重（非线性），有推理开销。这是 LoRA 胜出的关键工程优势之一。

---

## 五、Prompt Tuning / Prefix Tuning / P-Tuning —— 软提示家族

### 5.1 连续（Soft）Prompt vs 离散（Discrete）Prompt

- **离散 Prompt（硬提示）**：手工设计文本模板（如 "Translate to French: ..."），在 token 空间中搜索，不可微、效果不稳；
- **连续 Prompt（软提示）**：把 prompt 当作**可学习的连续向量**（soft prompt），直接在嵌入空间中优化，完全可微。

$$x' = [P; E(x)], \qquad P \in \mathbb{R}^{l \times d}$$

其中 $P$ 是长度为 $l$、维度 $d$ 的可学习矩阵，$E(x)$ 是输入 $x$ 的嵌入。

### 5.2 Prompt Tuning（Lester et al., 2021）

- **只在输入侧加 soft prompt**：在输入 token 序列前拼接 $l$ 个可学习向量（仅参数量 $l \times d$，如 100×4096 ≈ 0.4M，约全参的 0.005%）；
- 模型其余部分全部冻结，只训练 $P$；
- 特点：参数最少、最简单；
- **局限**：表达力有限——只在输入层注入信息，模型深层感知不到"任务信号"；
- 效果：模型规模越大效果越好（论文观察），但小模型上弱于 LoRA；纯文本指令任务上一般 **prompt tuning 弱于 LoRA**。

### 5.3 Prefix Tuning（Li & Liang, 2021）

- **每层都加 prefix**：在每个 Transformer 层的 attention 中拼接可学习的 key-value 前缀（以及可选的一个 MLP 把 prefix 参数映射为 K/V 向量）：

$$[\text{key}_i; P_i^K], \qquad [\text{value}_i; P_i^V], \qquad i = 1 \dots L$$

- $P_i \in \mathbb{R}^{l \times d}$（每层 $l$ 个 prefix token）；
- 初始用**一层 MLP（重参数化）**生成 prefix 参数，训练稳定后再去掉（类似 P-Tuning 的思路）；
- 因为每层都注入，表达力强于 Prompt Tuning，效果接近 Adapter/LoRA，但**推理时 prefix 增加了序列长度**（KV cache 变大，延迟增加）；
- 注意：prefix 不经过输入 embedding，直接作用在 attention 的 K/V 上。

### 5.4 P-Tuning v1 / v2

| 方法 | 结构 | 特点 |
|------|------|------|
| P-Tuning v1（2021） | 输入侧 soft prompt，但用 **LSTM/MLP 生成 prompt embedding** | 解决离散 prompt 不可微问题；LSTM 捕捉 prompt 内部 token 依赖；用于 NLU 任务（如 SuperGLUE） |
| P-Tuning v2（2022） | **逐层 prefix** + 各任务 head，类似 Prefix Tuning | 修复 v1 在通用任务上的不足；对标 Prefix Tuning，参数量更大但效果更稳 |

P-Tuning v1 的生成过程：

$$p_i = \text{LSTM}(h_{i-1}, \text{embed}_i), \qquad h_0 = \text{可学习向量}$$

### 5.5 软提示家族对比与选择

| 方法 | 加在哪 | 参数量（7B 示例） | 表达力 | 推理开销 | 适用场景 |
|------|--------|------------------|--------|----------|---------|
| Prompt Tuning | 仅输入嵌入前 | ~0.4M | 低 | 无 | 大模型、多任务共享基座、轻量适配 |
| Prefix Tuning | 每层 attention 的 K/V | ~10M | 中高 | **序列变长（延迟↑）** | 生成任务、NLG |
| P-Tuning v2 | 每层 prefix + 任务头 | ~10-20M | 中高 | 序列变长 | 通用 NLU/NLG |
| LoRA | 每层线性层 | ~4.5M | 高 | 可合并为 0 | **通用首选** |

> **面试记忆点**：三类都是"往输入/中间注入可学习向量"，区别在于**注入的位置**（输入 only vs 每层 K/V）与**生成方式**（直接学习 vs LSTM/MLP 重参数化）。LLM 纯文本任务上软提示通常弱于 LoRA；但在**多模态场景可与 LoRA 搭配**（如视觉侧 soft prompt 做模态对齐、语言侧 LoRA 做任务适配）。

---

## 六、其他 PEFT 方法一览

### 6.1 BitFit（只训 Bias）

- 只训练所有 **bias 项**，冻结其余权重；
- 参数量：约全参的 **0.1%** 以下；
- 效果：简单任务（分类）上意外地好；复杂生成任务上明显弱于 LoRA；
- 意义：证明了"**参数更新集中在少数位置**"，与低秩假设互为印证。

### 6.2 IA3（激活缩放，T-Few 论文）

- 为每个线性层引入**三个可学习缩放向量** $\ell_k, \ell_v, \ell_{ff}$，缩放 attention 的 K、V 与 FFN 的激活：

$$K \leftarrow \ell_k \otimes K, \qquad V \leftarrow \ell_v \otimes V, \qquad FFN \leftarrow \ell_{ff} \otimes FFN$$

- 参数量极小（每层 3×d，7B 约 0.01%），效果可与 LoRA 媲美；
- 缩放**整个激活通道**（向量级），比 LoRA 的矩阵级更新更轻量。

### 6.3 VeRA（向量化低秩）

- LoRA 的 $A, B$ 矩阵**全局共享**（一组广播后的随机初始化向量 $a, b$），每个任务只学**两个对角缩放向量** $\Lambda_a, \Lambda_b$：

$$\Delta W = \Lambda_a A \cdot B \Lambda_b$$

- 参数量比 LoRA 再降一个数量级（约 LoRA 的 1/10）；
- 代价：共享基座约束了表达空间，效果略弱于 LoRA，适合极小参数预算场景。

### 6.4 DoRA（权重分解 LoRA）

- 把权重分解为**幅度（magnitude）与方向（direction）**两部分：

$$W = m \cdot \frac{V}{\|V\|_c}, \qquad \text{只微调 } V \text{ 的方向用 LoRA，幅度 } m \text{ 单独学}$$

- 动机：研究发现**微调中幅度与方向的学习是不对称的**（LLaMA 上幅度变化很小、方向变化大，而全参微调两者都在动）；
- 效果：比 LoRA 更强（论文称约 +3.7 分基准提升），代价是参数量增加（幅度向量）与推理合并复杂度略增。

### 6.5 其他快速盘点

| 方法 | 一句话 | 一句话缺点 |
|------|--------|-----------|
| AdaLoRA | 按奇异值重要性自适应分配秩 | 实现复杂 |
| LoRA+ | 给 A、B 不同学习率（B 的学习率大 16 倍） | 只提效不降参 |
| rsLoRA | 用 $\alpha/\sqrt{r}$ 缓解大 r 时秩塌缩 | 需调超参 |
| LoRA 系列进阶 | rank 分配（MoRA）、训练与推理解耦（LoRA-FA） | 工程复杂 |

---

## 七、PEFT 对比总表

| 方法 | 可训练参数量（7B） | 训练显存 | 效果（对齐全参） | 推理开销 | 适用场景 |
|------|-------------------|---------|-----------------|---------|---------|
| 全参微调 | 100%（7B） | ~100GB | 100%（基准） | 0 | 数据充足、算力充足 |
| **LoRA** | 0.01%~1%（~4-9M） | 16~30GB(bf16)/~10GB(QLoRA) | ~95-99% | **0（可合并）** | **通用首选** |
| **QLoRA** | 同 LoRA | **~10-16GB** | ~93-97% | 0（合并后）+ 推理可量化 | 消费级单卡、低成本 |
| Adapter | 0.5%~5% | 略高于 LoRA | ~95-99% | 串行有延迟 | 老库兼容、研究场景 |
| Prompt Tuning | 0.005%（~0.4M） | 最低 | 60-90%（任务相关） | 0 | 超大模型、极端参数预算 |
| Prefix/P-Tuning | 0.1%~0.5% | 低 | 80-95% | **序列变长↑** | NLG、少样本 |
| IA3 | ~0.01% | 低 | ~90-95% | 0（可合并） | 极轻量适配 |
| BitFit | <0.1% | 低 | 60-85%（复杂任务差） | 0 | 简单分类、消融实验 |

> 结论速记：**效果 ≈ LoRA/Adapter > Prefix > Prompt；参数量 Prompt < IA3 < LoRA < Adapter；推理开销 Adapter(串行) > Prefix > LoRA(合并后为 0)。综合性价比 LoRA 最强，显存极限选 QLoRA，参数极限选 Prompt/IA3。**

---

## 八、多模态 PEFT 特有问题（重点）

### 8.1 微调面选择：只 LLM vs 全塔 vs 桥接层

多模态模型（如 LLaVA）通常由三部分构成：**视觉塔（ViT） + 投影层（Projector） + LLM**。PEFT 时"微调哪些面"是核心决策：

| 方案 | 训练内容 | 效果 | 显存 | 适用 |
|------|---------|------|------|------|
| 只 LLM | 仅 LLM 上的 LoRA | 中（视觉侧不动，视觉-语言对齐依赖投影层） | 低 | 快速适配 |
| LLM + 投影层 | LLM LoRA + 投影层全量/LoRA | 较高（更新对齐桥） | 中 | 数据量中等 |
| 全塔 | 视觉塔 LoRA + LLM LoRA + 投影层 | **最高**（视觉特征本身也适配任务） | 较高 | 数据充足、领域视觉差异大（如医学影像） |

经验法则：
- 视觉域与预训练 CLIP 域差异小（自然图像）→ **只 LLM LoRA 即可**；
- 视觉域差异大（医学/遥感/文档）→ **视觉塔必须也加 LoRA**（可只加后半层 block 或 attention 部分）；
- 投影层参数小（几百万），**建议直接全量训练**或加 LoRA，它决定视觉 token 能否"翻译"给 LLM。

### 8.2 多模态 LoRA 实践：分开还是统一

- **分开 LoRA**：ViT 一套 LoRA（通常 r 小，如 8-16）、LLM 一套 LoRA（r 32-64）分别注入，便于独立控制两个塔的适配强度；
- **投影层 LoRA**：投影层是视觉到文本的"翻译器"，若冻结投影层只训两端 LoRA，可能出现"视觉信号变了但翻译规则没变"的错位；
- 模块选择注意：ViT 的 patch embedding、position embedding 一般不 LoRA（参数量大且收益低）；LLM 的 embedding/lm_head 一般也不 LoRA。

### 8.3 LoRA 与视觉 token 的关系（常考）

- **LoRA 不改变 token 数量、不改变序列长度**：它只替换线性层的前向计算（$W_0x + BAx$），输入输出的形状完全不变；
- 视觉 token 数量由 ViT 的 patch 划分（如 576 个 token）与分辨率决定，与是否加 LoRA 无关；
- 因此 LoRA 微调后，**位置编码、attention 掩码、token 流结构均无需调整**——这是与 Prompt/Prefix（会改变 token 序列）的重要区别；
- 想要改变视觉 token 数量/粒度需要的是 patch 层面改动（高分辨率切分、多尺度融合），属于架构设计而非 PEFT 范畴。

### 8.4 多模态 LoRA 的常见失败模式

| 失败现象 | 根因 | 对策 |
|---------|------|------|
| 只训 LLM 后模型"答非所问/不看图" | 视觉特征与 LLM 指令空间未对齐（视觉侧没动） | 加视觉塔 LoRA / 训练投影层 |
| 训了视觉塔但灾难性遗忘（CLIP 能力退化） | 视觉 LoRA 秩过大、lr 过大、数据分布过偏 | 减小视觉 LoRA 的 r/lr，或只训后半层 |
| 视觉与语言 LoRA 冲突（一个任务好另一个崩） | 共享基座两个分支互相干扰 | 分开加载、任务专属 LoRA 或 LoRA 融合（算术组合 $W_A + W_B$） |
| 多轮对话中视觉信息丢失 | 只有首轮注入图像 token，后续轮次无图像引用 | 数据构造多轮图文交错样本 |

> **LoRA 融合技巧**：多任务 LoRA 可直接对权重做**线性算术**（$W_{\text{merge}} = W_0 + \alpha_A B_A A_A + \alpha_B B_B A_B$），这是参数高效的多任务合并方案，面试加分点。

### 8.5 多模态场景的 PEFT 组合拳

- 图像侧：**视觉塔 LoRA（小 r） + 投影层全量/ LoRA**；
- 文本侧：**LLM LoRA（大 r）**；
- 可选：视觉侧 **soft prompt（P-Tuning 式）** 注入可学习的图像描述前缀，帮助 LLM 理解图像 token；
- 训练时：**分阶段**（先投影层+视觉 LoRA 对齐、再 LLM LoRA 适配指令）或**联合**（全部同时训，lr 分层：投影层大、视觉塔小）。

---

## 九、PEFT 训练工程（HuggingFace PEFT 实践）

### 9.1 LoraConfig + get_peft_model

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "Qwen/Qwen2-VL-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=32,                    # 秩
    lora_alpha=64,           # 缩放（实际 α/r = 2）
    target_modules=[          # 注入目标（含视觉塔与 LLM 的线性层）
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",             # 不训练 bias
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["mlp2"],  # 额外全量训练投影层的最后一层
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: ~11,700,000 || all params: 7,600,000,000 || trainable%: 0.1538
```

- `get_peft_model` 会自动冻结原模型、包装注入层，可训练参数只剩 LoRA 分支（与 modules_to_save 指定的模块）；
- QLoRA 只需把基座用 `BitsAndBytesConfig` 以 4-bit 加载：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度（反量化后）
    bnb_4bit_use_double_quant=True,     # 双重量化
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

### 9.2 训练与保存

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps", save_steps=200,
)
trainer = Trainer(model=peft_model, args=args, train_dataset=dataset)
trainer.train()

# 只保存 LoRA 分支（几十 MB），不导出基座
peft_model.save_pretrained("./lora_weights")
```

### 9.3 合并权重与加载

```python
from peft import PeftModel

# 方式一：不合并，直接加载基座 + LoRA 分支
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "./lora_weights")

# 方式二：合并回基座权重（推理零开销，导出单一模型文件）
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

> 工程要点：
> 1. **合并后不可逆**：想恢复多任务分支要保留原始 LoRA 权重文件；
> 2. QLoRA 训练后的 LoRA 可直接用在 **bf16 基座**上（LoRA 分支本身是 bf16）；
> 3. 服务端多任务：加载一份基座 + 按请求切换 `PeftModel` 权重或做 LoRA 路由。

---

## 十、高频面试问答

**Q1：LoRA 为什么有效？**
预训练权重本身是高秩的、承载通用能力，而微调的更新量 $\Delta W$ 是低秩的（论文通过奇异值分解验证），低秩矩阵 $BA$ 足以表达其主成分。$r=4$ 时 GPT-3 175B 效果已接近全参微调就是证据。低秩约束还天然带来正则化，减少过拟合。

**Q2：为什么 B 初始化为 0、A 高斯初始化？**
$B=0$ 保证初始时 $W'=W_0$，微调严格从预训练权重出发，不偏离原分布；且训练初始输出与预训练一致，训练更稳。$A$ 若也为 0，则梯度 $\frac{\partial L}{\partial A} \propto B^T=0$，参数永远不更新。两者一个保证"起点正确"，一个保证"能够学习"。

**Q3：LoRA 和全参微调差距有多大？**
一般任务上 LoRA 可达到全参微调 95% 以上（论文及大量复现结论）；数据量大、任务与预训练域差异大时差距拉大；"遗忘"方面 LoRA 更优（约束在低秩子空间，对通用能力破坏小）。极端情况（大规模指令数据）下全参仍更稳。

**Q4：QLoRA 为什么能 4-bit 微调而不崩？**
基座权重完全冻结，只做存储上的量化（NF4 分位量化 + 双重量化），计算时反量化为 bf16，精度损失小；真正被训练更新的 LoRA 分支保持 bf16 高精度。量化误差只影响前向的冻结部分，不影响可学习参数的梯度质量。

**Q5：Adapter 和 LoRA 的区别？**
都是增量式 PEFT。区别：① Adapter 是插入新的小网络（非线性），LoRA 是低秩重参数化；② Adapter 串行会增加深度与推理延迟，LoRA 可合并进原权重、推理零开销；③ 参数量 LoRA 通常更小。数学上 LoRA ≈ 并行 Adapter 对线性层的特例。

**Q6：LoRA 推理时可以合并吗？**
可以。$W' = W_0 + \frac{\alpha}{r}BA$ 与两分支前向完全等价（矩阵乘法线性），合并后无任何额外延迟。不合并则保留多任务灵活性（一份基座 + 多分支热插拔），代价是每次推理多一次低秩矩阵乘。

**Q7：Prompt Tuning 和 Prefix Tuning 的区别？**
Prompt Tuning 只在输入序列前拼可学习向量（只作用于输入嵌入层），参数最少、表达力最弱；Prefix Tuning 在每一层的 attention 的 K/V 前拼接可学习前缀，逐层注入任务信号，表达力强，但会增长序列长度（KV cache 变大、推理变慢）。

**Q8：P-Tuning v1 和 v2 的区别？**
v1 用 LSTM/MLP 生成输入侧 prompt embedding（解决 prompt 初始化与依赖建模），效果局限在 NLU；v2 改为逐层 prefix + 任务头（对标 Prefix Tuning），修复了 v1 在序列标注、生成等任务上的不足。

**Q9：多模态微调时视觉塔要不要 LoRA？**
取决于视觉域差异。自然图像（与 CLIP 预训练域一致）只训 LLM LoRA 即可；医学/遥感等域差异大时必须视觉塔 LoRA（可只加后半层），并建议投影层也训练（它是视觉到语言的"翻译器"）。LoRA 不改变 token 数量，所以不会破坏视觉 token 流。

**Q10：LoRA 的 r 和 α 怎么调？**
r 决定表达能力（4-64，任务难/数据多取大），α/r 决定分支学习强度（一般取 2-4）。**改 r 时用 α 跟随（如 α = 2r）可保持有效学习率不变**。先定 r，再小范围搜 α/r；lr 一般比全参微调大一个量级（1e-4~3e-4）。

**Q11：LoRA 能解决灾难性遗忘吗？**
能缓解。LoRA 把更新约束在低秩子空间、且 $W_0$ 冻结，对预训练权重的破坏远小于全参微调；但仍非绝对免疫——数据过度偏移时 LoRA 分支本身会漂移。更强方案：LoRA 加回放数据、LoRA 融合、或 Adapter 隔离。

**Q12：PEFT 之后模型还能量化推理吗？**
可以。LoRA 合并后就是一个普通模型，可再走 GPTQ/AWQ 等后训练量化；QLoRA 训练（4-bit 基座）与推理量化是两回事，训练时的 4-bit 是存储格式，推理量化追求更小显存与吞吐。

---

## 十一、常见误区

1. **误区：LoRA 会降低推理速度。** 合并后零额外计算；只有"不合并、保留分支"的部署模式才有极小延迟增加（多一次低秩乘法），且可换取多任务灵活性。
2. **误区：r 越大越好。** 论文与大量实践显示 r 超过 32~64 后效果饱和甚至下降（过拟合、秩塌缩），r=4~16 往往已足够；关键是 α/r 与 lr 的配合。
3. **误区：QLoRA 的 4-bit 权重会被训练更新。** 不会。基座完全冻结，量化只影响存储与计算精度；更新的是 bf16 的 LoRA 分支。
4. **误区：软提示（Prompt/Prefix）会修改模型权重。** 不会，它们只是往输入/KV 拼接可学习向量，模型权重保持冻结；这也是它们参数最少的原因。
5. **误区：多模态微调"只训 LLM 就够"。** 在视觉域差异大的任务（医学、文档、遥感）上必须同时适配视觉塔与投影层，否则模型"看不懂图"，这是多模态 LoRA 最常见的失败原因。

---

## 十二、自我检验

- [ ] 能说清全参微调的三大问题（显存、存储、遗忘）与 PEFT 的解决思路
- [ ] 能写出 LoRA 的数学形式 $h = W_0x + \frac{\alpha}{r}BAx$，并解释每个符号
- [ ] 能解释 B=0、A 高斯初始化的原因（起点正确 + 梯度不为零）
- [ ] 能用"低秩假设 + 奇异值证据 + r=4 接近全量"讲清 LoRA 为什么有效
- [ ] 能背出 LoRA 超参（r、α/r、dropout、target_modules）及其调节直觉
- [ ] 能说明推理合并（零延迟）与不合并（多任务）两种模式及取舍
- [ ] 能讲清 QLoRA 的 NF4 量化、双重量化、bf16 反量化计算与显存账本（~16GB 微调 7B）
- [ ] 能区分 Adapter（串行/并行、bottleneck、残差）与 LoRA 的推理开销差异
- [ ] 能对比 Prompt Tuning / Prefix Tuning / P-Tuning v1/v2（注入位置、参数量、生成方式）
- [ ] 能完成 PEFT 总表的口头复述（参数量/显存/效果/推理开销/适用场景）
- [ ] 能回答多模态微调面的选择（LLM only vs 全塔）与视觉域差异判断
- [ ] 知道 LoRA 不改变 token 数量，与 Prompt/Prefix 的本质区别
- [ ] 能列举多模态 LoRA 常见失败模式与对策（对齐、遗忘、冲突、多轮）
- [ ] 能写出一段 LoraConfig + get_peft_model + merge_and_unload 的完整流程代码
- [ ] 能回答 12 个高频面试追问并指出 5 条常见误区
