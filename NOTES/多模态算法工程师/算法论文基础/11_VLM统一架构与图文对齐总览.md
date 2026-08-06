# VLM 统一架构与图文对齐总览（多模态系列总纲）

> 本笔记是整个"多模态算法工程师"系列的**总纲**：前面 01-10 篇分别拆解了 Transformer、ViT、CLIP/SigLIP、BLIP/BLIP-2、LLaVA 等单点技术，本篇把它们统一收拢到一张图里——**一切现代 VLM（视觉语言大模型）都是"视觉编码器 + 桥接模块 + LLM"三段式结构的实例化**。面试开场讲架构、后续被追问任何细节，都从这里展开。

## 一、VLM 一句话定义

> **VLM = 把图像"翻译"成 LLM 能读的 token 序列，让原本只会读文字的 LLM 获得视觉感知与多模态推理能力的模型。**

形式化地，VLM 的生成过程可以写成：

$$\text{输出} = \text{LLM}\Big( \text{文本token} \ \|\ \text{桥接模块}\big( \text{视觉编码器}(\text{图像}) \big) \Big)$$

其中 $\|$ 表示序列拼接（concatenation）。这个式子就是全系列的"母式"：
| 模块 | 职责 | 比喻 |
|------|------|------|
| 视觉编码器 | 把图像变成特征（patch token 序列） | 翻译官（把像素翻译成机器特征） |
| 桥接模块 | 把视觉特征投影/压缩到 LLM 的语义与维度空间 | 口译员（让两种语言互相对得齐） |
| LLM | 在混合序列上做自回归生成 | 决策者（综合两种信息回答问题） |

**全系列的模型都是这个框架的实例化**，区别只在"视觉塔是谁、桥怎么搭、LLM 是谁、怎么训"：
| 模型 | 视觉编码器 | 桥接模块 | LLM | 关键创新 |
|------|-----------|----------|-----|---------|
| BLIP-2 (2023) | CLIP ViT-L/14（冻结） | Q-Former | OPT / Flan-T5 | 可学习的 32 个 query 跨模态采样 |
| LLaVA (2023) | CLIP ViT-L/14（冻结） | MLP projector | Vicuna-7B/13B | 极简桥接 + GPT-4 生成指令数据 |
| Qwen-VL (2023) | CLIP ViT（冻结） | Resampler | Qwen-7B | 256→1024 token 多分辨率 |
| InternVL (2024) | InternViT-6B（最大的公开视觉塔） | pixel shuffle + MLP | InternLM | 视觉塔巨型化 + 动态切块 |
| Qwen2-VL (2024) | 自研 ViT 675M | MLP（单层） | Qwen2 | 原生任意分辨率 + M-RoPE |

> **面试记忆点**：先说出上面的母式，再指出"任何 VLM 论文你都可以用这张表做模板复述"，这是总纲笔记的价值。

---

## 二、三段式统一架构（重点）

### 2.1 架构总览
```text
图像 224×224×3 ──────► 视觉编码器(ViT) ──► N_v×d_v patch特征
                                                  │
文本 "<image> 图里有猫吗" ─► Tokenizer/Embedding ─┤ 桥接模块(MLP/Q-Former/压缩)
                                                  ▼
                             N_v'×d_llm 视觉token + N_t×d_llm 文本embedding
                                                  │
                                        ▼ 序列拼接 (concatenate)
                                     ┌─────────────────────┐
                                     │ LLM (Decoder-only)   │
                                     │ 因果注意力 + 前馈网络  │
                                     └──────────┬──────────┘
                                                ▼
                                        自回归生成回答
```

三个模块的参数量典型比例：视觉塔 0.3B~6B，桥接模块 0（MLP）~ 数百 M（Q-Former），LLM 2B~72B。**LLM 永远是最大的那块，也永远是"主脑"。**

### 2.2 视觉编码器：ViT 家族的选型逻辑
#### 2.2.1 两种来源
| 来源 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| 借用图文预训练塔（CLIP/SigLIP） | LLaVA、BLIP-2、MiniCPM-V | 特征已与文本语义对齐过，开箱即用；预训练数据 10B 级，质量有保证 | 分辨率被限制在预训练分辨率附近（224/336/384）；结构不可定制 |
| 自研视觉塔 | InternViT-6B、Qwen2-VL ViT 675M | 分辨率/位置编码可定制（M-RoPE、任意分辨率）；可与 LLM 联合设计 | 需要额外图文预训练数据与算力，成本高 |

**为什么大多数模型"借用"**：视觉塔预训练需要 10B 级图文对 + 数千卡日，对绝大多数团队不可负担；而 CLIP/SigLIP 的特征已天然落在"语义空间"，后续桥接和 SFT 的成本低得多。**只有头部厂商（Qwen、InternVL）才有能力自研视觉塔。**

#### 2.2.2 输出形态：CLS vs 全部 patch token

ViT 最后一层输出有两类取法，用途完全不同：
| 取法 | 输出形状 | 用途 | 代表 |
|------|---------|------|------|
| [CLS] token（1 个） | $1 \times d_v$ | 全局单向量，做检索/匹配/分类 | CLIP、SigLIP 的图文对齐任务 |
| 全部 patch token | $N_v \times d_v$ | 逐位置细粒度信息，供 LLM 细读 | 一切生成式 VLM |

**VLM 必须用全部 patch token**："图里左上角是什么""树上有几只鸟"这类问题需要空间位置信息，一个全局向量把所有位置的信息平均掉了（类比：一句话总结 ≠ 逐句可查）。代价是 token 数从 1 暴增到几百上千，直接决定 KV cache 与算力。

#### 2.2.3 主流 VLM 视觉塔选型一览
| VLM | 视觉塔 | 输入分辨率 | patch token 数 |
|-----|--------|-----------|----------------|
| LLaVA-1.5 | CLIP ViT-L/14 | 336 | 576（24×24，丢 CLS） |
| BLIP-2 | CLIP ViT-L/14 | 224 | 257（含 CLS） |
| Qwen-VL | CLIP ViT | 224/448 | 256/1024 |
| InternVL2 | InternViT-6B | 动态 tile（448²/个） | 每个 tile 256 |

> **面试记忆点**：token 数计算是基本功——$N_{patch} = \frac{H}{P} \times \frac{W}{P}$。336/14=24，24²=576；224/16=14，14²=196。

### 2.3 桥接模块：三选一（重点中的重点）

桥接模块解决两个核心问题：
1. **维度不匹配**：ViT 输出维度 $d_v$（768~1152）≠ LLM 隐藏维度 $d_{llm}$（2048~8192）；
2. **语义不匹配**：视觉特征分布与 LLM 词嵌入分布完全不同，直接拼接训练不收敛。

#### 2.3.1 方案一：MLP Projector（简单直接）

$$\mathbf{z} = W_2 \cdot \sigma(W_1 \cdot \mathbf{x} + b_1) + b_2$$

- 对每个 patch token **逐点**作用（per-token，不跨 token 融合）；
- LLaVA 早期用 Linear，1.5 之后用两层 MLP（GELU），效果显著提升；
- **token 数不变**：$N_v \times d_v \to N_v \times d_{llm}$；参数量 ≈ $d_v \times d_{llm} \times 2$（约 8M~16M，可忽略）；
- 缺点：完全没有跨位置信息交互，LLM 承担了所有融合压力。

#### 2.3.2 方案二：Q-Former / Resampler（固定 token 数）

初始化 M 个**可学习 query**（通常 32 个），通过交叉注意力"查询"图像特征，输出固定 M 个 token：

$$\mathbf{Q} \in \mathbb{R}^{M \times d}, \quad \mathbf{z} = \text{CrossAttn}(\mathbf{Q}, \mathbf{V}_{img}), \quad \mathbf{z} \in \mathbb{R}^{M \times d_{llm}}$$

- **token 数由 M 决定，与图像分辨率解耦**——输入 224 和 896 都是 32 个 token，KV cache 恒定，这是最大卖点；
- 内部有自注意力 + 交叉注意力，可在桥接阶段就做图文融合；
- 缺点：参数量数百 M 级；token 太少信息有损（32 个 token 装不下一张复杂图）。

#### 2.3.3 方案三：Token 压缩/合并（自适应）

在保留全部 patch 特征的前提下**自适应减少 token 数**：

- **pixel shuffle 下采样**（InternVL）：patch 特征 reshape，每 2×2 合并成 1 个 token；
- **平均池化/卷积下采样**（Honeybee、TokenPacker）；
- **注意力合并**（Token Merging）：相邻 token 按相似度合并；
- **动态合并**（Qwen2-VL）：相邻 2×2 区域合并，保持位置编码对齐。

#### 2.3.4 三方案对比表（面试必考）
| 维度 | MLP Projector | Q-Former / Resampler | Token 压缩/合并 |
|------|--------------|----------------------|-----------------|
| 输出 token 数 | 不变（N_v） | 固定（M=32/64） | 按比例减少（1/4 等） |
| 参数量 | 最小（~10M） | 中等（数百 M） | 小（几乎无新增） |
| 信息保留 | 100%（逐 token） | 有损（压进 M 个向量） | 部分保留（压缩冗余） |
| 跨 token 交互 | 无 | 有（交叉注意力） | 有（合并时相似度计算） |
| 训练复杂度 | 最低 | 最高（需精心初始化） | 低 |
| 对分辨率变化的适应 | 好（token 随分辨率增长） | 最好（token 恒定） | 好 |
| 典型代表 | LLaVA-1.5/1.6 | BLIP-2、Qwen-VL | InternVL2、Qwen2-VL |

> **面试记忆点**：三者本质是"**信息量与 token 数的权衡**"——MLP 保信息但 token 多（贵），Q-Former 省 token 但丢信息，压缩是折中。现代趋势（Qwen2-VL、InternVL2）是**压缩派**：用更少的 token 保住更多信息。

### 2.4 LLM 部分：Decoder-only 的统治地位

#### 2.4.1 为什么都是 Decoder-only
| 原因 | 解释 |
|------|------|
| 自回归天然适配生成 | 图文问答/描述本质是"给定上下文生成文本"，与 LM 目标一致 |
| 生态成熟 | LLaMA、Qwen、Gemma 等强开源底座；训练工具链完备 |
| 训练与推理一致 | 预训练/微调/推理都是 next-token prediction，无 Encoder-Decoder 的初始化鸿沟 |
| KV cache 高效 | 单向注意力可缓存历史，多轮对话便宜 |

#### 2.4.2 冻结 / 部分微调 / 全参微调
| 策略 | 冻结范围 | 优点 | 缺点 |
|------|---------|------|------|
| 全部冻结（只训桥） | 视觉塔 + LLM | 极快、数据需求小、防遗忘（用于两阶段训练的第一阶段） | 对齐能力上限低 |
| 冻结视觉塔，微调 LLM | 只冻视觉塔 | 兼顾效率与效果（SFT 主流选择） | 视觉塔表达力是天花板 |
| 全参微调 | 全部 | 效果上限最高（数据充足时） | 训练贵、易灾难性遗忘 |
| LoRA 微调 | 注入低秩适配器 | 显存省 70%+，可插拔 | 效果略低于全参 |

#### 2.4.3 视觉 token 如何与文本 token 拼接

拼接发生在 **embedding 层之后、进入 Transformer 之前**：文本 token 查词表得 $\mathbb{R}^{N_t \times d_{llm}}$，视觉 token（已投影）是 $\mathbb{R}^{N_v \times d_{llm}}$，直接在序列维度 cat。关键机制是**特殊占位符（special token）**：
| 机制 | 说明 |
|------|------|
| `<image>` 占位符 | 文本中该位置被"替换"为视觉 token 序列，最终序列位置严格对齐 |
| `<|image_pad|>` 填充符 | Qwen-VL 中占位符后填充 padding token，保证位置编码正确 |
| `<|vision_start|>/<|vision_end|>` 边界符 | Qwen2-VL 标记视觉序列起止，让 LLM 学会"这里开始/结束看图像" |
| 新行分隔符 | LLaVA-NeXT 在多个 tile 间插入 `\n` token 作为"图像边界" |

**Prompt 模板示例（面试能默写一个就够）**：
```text
LLaVA-1.5 模板:
USER: <image>
What are the things I should be cautious about when I visit here?
ASSISTANT:

Qwen2-VL 模板（ChatML 风格）:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
这张图里有什么动物？<|im_end|>
<|im_start|>assistant

InternVL2 模板:
<image>
Human: 图中有几只猫？
Assistant:
```

模板的本质作用：**让"图像何时出现、视觉 token 从哪里开始"成为模型可学习的位置信号**。推理时系统把 `<image>` 替换成实际的视觉 token embedding。

### 2.5 四大代表模型的实例化对照（总纲图）
| | BLIP-2 | LLaVA-1.5 | Qwen-VL | InternVL2 |
|--|--------|-----------|---------|-----------|
| 视觉塔 | ViT-L/14 (224) | ViT-L/14 (336) | ViT (448) | InternViT-6B (448 tile) |
| 桥 | Q-Former | MLP-2 层 | Resampler | pixel shuffle + MLP |
| 桥后 token | 32 | 576 | 1024 | 256/tile |
| LLM | Flan-T5-XL | Vicuna-7B | Qwen-7B | InternLM2-7B |
| 训练策略 | 两阶段（Q-Former→解冻微调） | 两阶段（投影器→+LLM） | 三阶段（预训练→冻结微调→全量） | 两阶段 + 动态 tile |

> 同一套骨架，四个不同侧重点：BLIP-2 重在桥、LLaVA 重在数据、Qwen-VL 重在分辨率、InternVL 重在视觉塔规模。

---

## 三、图像如何变成"视觉 Token"的完整流程（重点）

### 3.0 总流程图
```text
像素输入 ─► ①预处理 ─► ②Patch化 ─► ③ViT编码 ─► ④位置编码
                                                      │
        ⑧与文本拼接 ◄─ ⑦维度投影 ◄─ ⑥池化/压缩 ◄──────┘
```

### 3.1 逐环节拆解：可选方案 + 影响

#### ① 输入与预处理
| 环节 | 可选方案 | 影响 |
|------|---------|------|
| 分辨率 | 224 / 336 / 384 / 448 / 动态 | 直接决定 patch 数（$N \propto (H/P)(W/P)$），即视觉 token 总量 |
| 归一化 | CLIP 风格 mean/std 固定归一化（如 μ=[0.481,0.458,0.408]） | 与预训练视觉塔输入分布一致，否则特征漂移 |
| 像素格式 | RGB，归一化到 [-1,1] 或 [0,1] | 影响数值稳定性 |
| 长宽比 | 正方形 resize / 保比例 pad / 动态分块 | 保比例+pad 避免物体变形；动态分块零信息损失（Qwen2-VL） |

#### ② Patch 化（tokenize 图像）

图像切分为 $P \times P$ 的 patch，展平后过一层线性投影：

$$N_{patch} = \frac{H}{P} \times \frac{W}{P}, \qquad \mathbf{x}_i = \text{Linear}(\text{flatten}(patch_i)) \in \mathbb{R}^{d_v}$$
| patch size P | 224² 图像 | 336² 图像 | 感受野 | 信息粒度 |
|--------------|----------|----------|--------|---------|
| 32 | 49 | 110 | 大（只见局部结构） | 粗 |
| 16 | 196 | 441 | 中 | 中 |
| 14 | 256 | 576 | 小 | 细（主流选择） |

#### ③ ViT 编码

patch embedding 过 L 层 Transformer block（Pre-LN 结构），每层都是标准注意力：

$$\text{Attn}(Q,K,V) = \text{softmax}\Big(\frac{QK^T}{\sqrt{d_k}}\Big)V$$
| 配置 | 说明 | 影响 |
|------|------|------|
| 层数 | ViT-L 24 层 / ViT-g 40 层 | 越深语义越强，越贵 |
| 是否含 CLS | 有（分类用）/ 无（VLM 可丢弃） | 决定最终取法 |

#### ④ 位置编码（视觉的空间信息就靠它）
| 方案 | 机制 | 代表 |
|------|------|------|
| 可学习绝对位置编码 | 每位置一个可学向量，相加 | CLIP ViT |
| 2D-RoPE | 行、列分别做旋转位置编码 | InternVL |
| M-RoPE（多模态 RoPE） | 文本 1D、图像 2D、视频 3D RoPE，多模态位置解耦 | Qwen2-VL |
| 动态分辨率适配 | 位置编码按实际分辨率插值/重建 | Qwen2-VL |

> 位置编码决定"模型能不能分清左/右、上/下、先/后"。**对比对齐的双塔中位置信息从文本隐式学习；生成式 VLM 中位置编码质量直接决定空间推理能力。**

#### ⑤ 池化/压缩（可选）

见 2.3.3。结论一句话：**要么不压缩保全部信息（LLaVA），要么压缩换 token 数（InternVL2 每 tile 1024→256）**。

#### ⑥ 维度投影

把 $d_v$ 维映射到 $d_{llm}$ 维——桥接模块的核心计算（MLP 逐 token / Q-Former 交叉注意力），输出记作 $\mathbf{z}_i \in \mathbb{R}^{d_{llm}}$。

#### ⑦ 拼接

视觉 token 序列与文本 embedding 拼接：

$$\mathbf{X} = \text{Concat}([\mathbf{z}_1, \dots, \mathbf{z}_{N_v}], [\mathbf{e}_1, \dots, \mathbf{e}_{N_t}]) \in \mathbb{R}^{(N_v + N_t) \times d_{llm}}$$

之后 $\mathbf{X}$ 进入 LLM，后续所有计算与纯文本完全一致——**"图像进入 LLM 后没有特殊待遇"是理解 VLM 的关键**。

### 3.2 全流程数值示例（一张图走完，以 LLaVA-1.5 为例）
```text
输入: 336×336×3 RGB 像素
  ↓ ① 归一化 + 转 tensor
[1, 3, 336, 336]
  ↓ ② patch 化 (P=14) → 24×24 = 576 个 patch
[1, 576, 1024]        (CLIP ViT-L 维度 d_v = 1024)
  ↓ ③+④ ViT 24 层编码 + 位置编码（丢弃 CLS）
[1, 576, 1024]
  ↓ ⑥ MLP 投影 1024 → 4096 (Vicuna 隐藏维)
[1, 576, 4096]        ← 这就是"视觉 token"
  ↓ ⑦ 与文本拼接（"USER: <image>\n..." 模板）
[1, 576+12, 4096]     (12 个文本 token)
  ↓ LLM 自回归生成
"assistant: 图片中有一只橘色的猫……"
```

> 面试加分细节：**576 个视觉 token 占整个序列的 98%**，这直接决定了 VLM 推理比纯文本 LLM 慢/贵的原因（见第九节）。

---

## 四、图文对齐的四种范式（重点）

"对齐（Alignment）"是多模态最核心的概念，但**存在四种不同含义的"对齐"**，面试最容易混为一谈。

### 4.1 范式一：对比对齐（Contrastive Alignment）

**思想**：匹配的图文对在共享空间中靠近，不匹配的远离。双塔结构（image encoder + text encoder）共享一个特征空间。

**CLIP（InfoNCE）**：

$$\mathcal{L}_{CLIP} = -\frac{1}{2N}\sum_{i=1}^{N}\Big[\log \frac{e^{s_{ii}/\tau}}{\sum_{j} e^{s_{ij}/\tau}} + \log \frac{e^{s_{ii}/\tau}}{\sum_{j} e^{s_{ji}/\tau}}\Big], \quad s_{ij} = \frac{v_i^T t_j}{\|v_i\|\|t_j\|}$$

**SigLIP（逐对 sigmoid）**：

$$\mathcal{L}_{SigLIP} = -\frac{1}{N^2}\sum_{i,j} \log \sigma(Y_{ij} \cdot (t \cdot s_{ij} + b))$$

- 产出：**共享语义空间**，支持零样本检索/分类；不需要 LLM，是"嵌入级对齐"，**不做生成**；
- 代表：CLIP、SigLIP、ALIGN、EVA-CLIP。

### 4.2 范式二：匹配对齐（Matching Alignment）

**思想**：给定图像和文本，判断"是否匹配"——二分类问题。

$$\mathcal{L}_{ITM} = -\mathbb{E}_{(I,T)} \big[ y \log \sigma(h_{[ITM]}) + (1-y) \log(1-\sigma(h_{[ITM]})) \big]$$

- 做法（BLIP）：图文融合后取 `[ITM]` 特殊 token 的输出过分类头，得到**匹配判别器**（图文是否一致的可校准概率）；
- 特点：比对比更精细（能判断局部不匹配），但不能检索也不能生成；代表：BLIP ITM、FLAVA、ALBEF。

### 4.3 范式三：生成对齐（Generative Alignment）

**思想**：让模型**生成**与图像匹配的文本（caption），以文本生成质量作为对齐信号。

$$\mathcal{L}_{LM} = -\sum_{t} \log p(y_t \mid y_{<t}, \mathbf{z}_{img}, \text{prompt})$$

- 从"嵌入距离"变成"生成概率"——信号更粗犷但更全面（逼着模型真的把图"读懂"才能生成对）；
- 代表：CoCa（对比+生成联合）、BLIP-2/LLaVA 的预训练阶段；**它是支撑一切生成式 VLM 的地基**，数据可用海量弱标注图文对。

### 4.4 范式四：指令对齐（Instruction Alignment / SFT）

**思想**：在人工标注的指令数据上微调，让模型学会"按用户意图回答问题"，对齐发生在**任务语义层**，而非特征层。

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{answer}} \log p_\theta(y_t \mid y_{<t}, \mathbf{z}_{img}, \text{instruction})$$

- 数据形态："图像 + 指令 + 参考答案"（VQA、对话、推理、OCR 等）；**对齐的是行为**（会按指令看图、格式正确、拒绝不该答的）；
- 代表：LLaVA-Instruct-158K、ShareGPT4V + 全系列 VLM。

### 4.5 四范式对比总表
| 维度 | 对比对齐 | 匹配对齐 | 生成对齐 | 指令对齐 |
|------|---------|---------|---------|---------|
| 对齐什么 | 嵌入空间 | 判别边界 | 生成分布 | 任务行为 |
| 损失 | InfoNCE / Sigmoid | 二分类 BCE | 交叉熵（LM） | 交叉熵（answer 掩码） |
| 输出 | 特征向量 | 概率 | 文本 | 文本 |
| 是否可检索 | 是（核心能力） | 否 | 可（弱） | 否 |
| 是否可生成 | 否 | 否 | 是 | 是 |
| 数据要求 | 图文对（10B 级） | 图文对（可少） | 图文对 + 文本 | 指令数据（10⁵~10⁶ 级） |
| 训练成本 | 高 | 中 | 高 | 中 |
| 代表 | CLIP/SigLIP | BLIP-ITM | CoCa/VLM 预训练 | LLaVA-SFT/Qwen-VL |

### 4.6 四种范式的关系与演进（为什么现代 VLM 只用后两种）
```text
对比对齐 (2021) ──► 匹配对齐 (2021-22) ──► 生成对齐 (2022-23) ──► 指令对齐 (2023-)
   检索/零样本分类      细粒度判别          可生成（VLM 地基）       会听话（任务级）
        │                    │                    │                     │
        └────────────────────┴────────────────────┴─────────────────────┘
       前两者被吸收为"视觉塔预训练"（CLIP/SigLIP 塔如今只是食材）
       后两者成为 VLM 本体（生成是骨架、指令是灵魂）
```

**演进逻辑（面试必讲）**：

1. **对比对齐先出现**：双塔+超大 batch 在 10B 图文对上训练，得到通用语义空间——"图像能被语言描述"的第一步证明；
2. **匹配对齐补细节**：对比对齐对局部矛盾不敏感（配无关文本相似度也可能高），ITM 用判别式信号补上细粒度；
3. **生成对齐取代前两者成为 VLM 地基**：产品最终形态是"会说话的模型"，**对齐目标必须与产品目标一致**（生成文本）才能端到端优化；且生成信号不依赖特征空间设计，天然可扩展；
4. **指令对齐是最后一公里**：生成对齐让模型"会说图像内容"，指令对齐让模型"按用户要的方式说"（问答/总结/推理/拒绝）。

> **面试记忆点**：CLIP 是"给视觉塔用的"（食材），VLM 本体吃的是生成对齐 + 指令对齐两道菜。所以做 VLM 的团队仍在拼命训 CLIP/SigLIP 塔——不是要用它对齐，是要它当视觉塔。

---

## 五、对齐的目标与度量

### 5.1 对齐的目标

一切对齐的最终目标都是让**跨模态语义一致**：

$$\text{sim}(I, T) \ \text{大} \iff I \text{ 与 } T \text{ 语义一致}$$

但在不同范式下，"一致"被具象化为不同目标：
| 范式 | 对齐目标的具体形式 | 可度量指标 |
|------|------------------|-----------|
| 对比 | 匹配对余弦相似度 > 不匹配对 | Recall@K、MRR、zero-shot 分类 acc |
| 匹配 | 匹配概率接近 1 / 0 | 分类 acc、AUC |
| 生成 | 生成文本与参考答案接近 | BLEU、CIDEr、ROUGE、perplexity |
| 指令 | 回答被评判为"正确且有用" | VQA 准确率、人工评分、GPT-4 评分 |

### 5.2 核心度量：余弦相似度

对比对齐的基本运算（也是唯一的"距离"）：

$$\cos(v, t) = \frac{v \cdot t}{\|v\|_2 \|t\|_2} = \frac{\sum_{k} v_k t_k}{\sqrt{\sum_k v_k^2}\sqrt{\sum_k t_k^2}} \in [-1, 1]$$

工程上通常先 L2 归一化再点积（余弦 ≈ 点积），并乘温度缩放：

$$z_{ij} = \tau \cdot \cos(v_i, t_j), \quad p_{ij} = \text{softmax}_j(z_{ij})$$

### 5.3 检索类指标（对比对齐考核）
| 指标 | 定义 | 公式 |
|------|------|------|
| Recall@K | 查询的 GT 是否出现在相似度 Top-K 中 | $R@K = \frac{1}{Q}\sum_{q=1}^{Q} \mathbb{1}[gt_q \in \text{top}_K(q)]$ |
| 中位排名 (MedR) | GT 排名的中位数 | $\text{MedR} = \text{median}\big(\text{rank}(gt_q)\big)$ |
| MRR | 倒数排名的均值 | $MRR = \frac{1}{Q}\sum_q \frac{1}{\text{rank}(gt_q)}$ |
| 零样本 Top-1 准确率 | 图像与 K 个类别文本的最高相似度类别 | $\hat{y} = \arg\max_k \cos(v, t_k)$ |

### 5.4 生成类指标（生成/指令对齐考核）
| 指标 | 原理 | 适用 |
|------|------|------|
| Perplexity | $PPL = \exp\big(-\frac{1}{T}\sum_t \log p(y_t \mid y_{<t})\big)$ | 训练收敛监控 |
| BLEU | n-gram 精确率 + 长度惩罚 | 翻译、caption（偏低） |
| CIDEr | TF-IDF 加权 n-gram 相似度 | **caption 主力指标** |
| VQA Acc | 答案与 GT 集合精确/模糊匹配 | VQA 任务 |

> **对齐任务全景**：检索/分类（对比）、图文一致性判断（匹配）、caption/VQA/引用定位（生成）、多模态对话（指令）——任务形态决定选哪种对齐，通常组合使用（生成+指令）。

---

## 六、多模态输入处理：多图、视频、高分辨率

### 6.1 多图输入
| 方案 | 做法 | 优缺点 | 代表 |
|------|------|--------|------|
| 直接拼接 | 多图 token 序列依次 concat，多个 `<image>` 占位符 | 简单；但模型可能分不清"第几张图" | 早期 LLaVA 多图版 |
| 独立编码 + 共享 LLM | 每图独立过视觉塔+桥，token 平铺 | 与拼接本质相同，靠占位符位置区分 | LLaVA-NeXT |
| 引用机制 | 图间插入特殊分隔 token，prompt 中引用图编号 | 支持"对比图 1 和图 2"类任务 | Qwen2-VL、InternVL2 |
| 帧级时间戳 | 视频/多图带时间或序号标记 | 支持时序推理 | Qwen2-VL |

> 多图的关键难点是**指代消解**（"这张图""第二张图"）——模型必须把指代词与具体视觉段绑定，靠训练数据中的引用标注学会。

### 6.2 视频输入

视频的本质是"多帧 + 时间轴"，工程上：
| 环节 | 常见做法 |
|------|---------|
| 抽帧 | 均匀抽 8~16 帧（或按内容抽）；16 帧 × 256 token = 4096 token，直接考验上下文长度 |
| 帧内编码 | 每帧独立走视觉塔（与单图一致） |
| 时间信息 | 帧间插入时间戳 token（如 `<|time_00:00|>`）或 3D 位置编码（M-RoPE） |
| 跨帧压缩 | Qwen2-VL 将相邻 2 帧按时间维 merge，16 帧压成 8 帧的 token 量 |
| 时间池化 | 相邻帧 token 平均/拼接压缩（视频专用桥接） |

**视频 VLM 的算力瓶颈**：token 数 = 帧数 × 每帧 token 数，8 帧 1080p 切块后轻松上万 token，KV cache 与注意力平方级爆炸——所以"视频专用压缩"是独立研究线（TimeChat、Video-LLaVA 等）。

### 6.3 高分辨率：切块（AnyRes / Any Resolution）

高分辨率 + ViT 的矛盾：**ViT 有位置编码，直接放大分辨率会破坏位置分布**（模型没见过那么大的位置编码）。解决方案是"切块"：

**LLaVA-NeXT（AnyRes）**：

1. 整图缩略图（resize 到 336×336）→ 576 token（保留全局上下文）；
2. 按长宽比把图切成若干 336×336 的 tile（最多 6 个）→ 每 tile 576 token；
3. 全部拼接 + 行间插入换行分隔 token。

$$\text{总token} = 576 + 576 \times N_{tile} + 1$$

例：1024×1024 图像，3×3 共 9 tile（超限取 6）→ 576 + 6×576 ≈ 4032 token。
| 方案 | token 数（1024×1024 示例） | 信息保留 | 代表 |
|------|--------------------------|---------|------|
| 直接 resize 到 336 | 576 | 丢失大量细节 | LLaVA-1.5 |
| AnyRes 切块 | ~4032 | 高（细节全保，全局靠缩略图） | LLaVA-NeXT |
| 原生动态分辨率（位置编码可扩） | 1024²/14² = 5376 | 最高（无任何信息损失） | Qwen2-VL |

**为什么切块有效**：tile 内位置编码仍是预训练见过的 336 尺度，未破坏 ViT 分布；全局信息由缩略图兜底。代价是 token 暴涨。

---

## 七、训练范式总览（全生命周期）

VLM 训练从不是一步到位，而是**阶梯式的四阶段**：

### 7.1 阶段零：视觉-语言预训练（搭地基）
| 内容 | 说明 |
|------|------|
| 训什么 | 视觉塔（+文本塔），双塔对比（CLIP/SigLIP）或对比+生成（CoCa） |
| 冻什么 | 无（全量训练，10B 级图文对，数千卡日） |
| 数据 | 互联网抓取图文对（WebLI、LAION、Wukong） |
| 产出 | **预训练视觉塔**——后续一切 VLM 的食材 |

> 大多数 VLM 团队不执行此阶段（太贵），直接加载开源塔。

### 7.2 阶段一：特征对齐（两阶段训练的第一阶段，以 LLaVA 为模板）
| 子阶段 | 训什么 | 冻什么 | 数据 | 目标 |
|--------|--------|--------|------|------|
| 1a. 投影器预训练 | 仅桥接模块 | 视觉塔 + LLM 全冻 | 图文对（CC3M 级，500K 规模） | 让视觉特征"挤进"LLM 的词嵌入语义空间 |
| 1b. 视觉指令预训练 | 桥 + LLM | 视觉塔冻结 | 图文对 + 简单问答 | LLM 学会"看着图像续写" |

**为什么先冻住 LLM 只训投影器**：投影器参数随机、输出分布离谱，若同时更新 LLM，LLM 会被垃圾输入带偏（灾难性遗忘）；先固定两端、只调中间，相当于"先打通管道，再通水"。

### 7.3 阶段二：指令微调（SFT）
| 内容 | 说明 |
|------|------|
| 训什么 | 桥 + LLM（视觉塔仍冻结，或低 lr 微调） |
| 冻什么 | 视觉塔（主流） |
| 数据 | 指令数据：VQA、对话、推理、OCR、引用等，10⁴~10⁶ 级 |
| 损失 | answer token 上的交叉熵（user 部分 mask 掉，只算回答部分） |
| 效果 | 从"会描述"升级为"会按指令服务"，这是产品级能力的分水岭 |

### 7.4 阶段三：偏好对齐（RLHF / DPO）

让回答更符合人类偏好（更有帮助、更少幻觉、更安全）：

**RLHF（PPO）**：先训奖励模型 $\mathcal{R}(x, y)$，再用 PPO 优化：

$$\max_{\theta} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [\mathcal{R}(x,y)] - \beta \cdot \mathbb{D}_{KL}\big(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)\big)$$

**DPO（无需 RL）**：直接以偏好对 $(y_w, y_l)$ 为监督：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$
| 方案 | 需要奖励模型 | 训练稳定性 | 数据形态 |
|------|-------------|-----------|---------|
| PPO | 是（actor/ref/critic/reward 四模型） | 难调 | 离线奖励 + 在线采样 |
| DPO | 否 | 简单 | 偏好对（好/坏回答） |
| 其他 | 否 | 简单 | RLAIF（用强模型评分） |

### 7.5 全周期一览表（面试画这张表）
| 阶段 | 训什么 | 冻什么 | 数据（量级） | 损失 | 类比 |
|------|--------|--------|-------------|------|------|
| 0. 图文预训练 | 双塔（视觉塔） | 无 | 图文对 10⁹ | InfoNCE/Sigmoid | 学语言（外语环境浸泡） |
| 1. 特征对齐 | 桥接模块 | 双塔+LLM | 图文对 10⁵~10⁶ | LM CE | 学翻译（先把词对应上） |
| 2. 指令微调 | 桥+LLM | 视觉塔 | 指令数据 10⁴~10⁶ | answer CE | 学礼仪（知道怎么回话） |
| 3. 偏好对齐 | LLM（+桥） | 视觉塔 | 偏好对 10³~10⁵ | DPO/PPO | 学做人（知道什么话不该说） |

---

## 八、完整问答链路示例：一张图的旅程

### 8.1 推理链路（文字步骤）
```text
请求: 用户上传图片 + 文本 "图片里有什么动物？"

步骤 1  图像预处理: 解码 → resize(336×336) → 归一化 → tensor [1,3,336,336]
步骤 2  视觉编码:  ViT 切 patch → 24 层 Transformer → 576 个 patch 特征
步骤 3  桥接投影:  MLP: [1,576,1024] → [1,576,4096] （= 576 个视觉 token）
步骤 4  文本编码:  tokenizer → id → embedding: [1,9,4096]
步骤 5  模板组装:  "<image>\nUSER: 图片里有什么动物？\nASSISTANT:"
                   → 视觉 token 替换 <image> → 拼接 [1,585,4096]
步骤 6  LLM 前向:  585 个 token 依次过注意力；每步只预测下一个 token
步骤 7  自回归:    "这是一" → "这是..." → ... 直到 <eos>
步骤 8  后处理:    去掉特殊 token → "这是一只橘猫和一只白狗。"
```

### 8.2 张量形状变化表（面试手写版）
| 阶段 | 张量形状 | 说明 |
|------|---------|------|
| 原始图像 | [1, 3, 336, 336] | 输入 |
| 输入文本 | [1, 9] | token id |
| patch embedding | [1, 576, 1024] | 576=24×24 patch |
| ViT 输出 | [1, 576, 1024] | 丢弃 CLS |
| 投影后（视觉 token） | [1, 576, 4096] | d_v → d_llm |
| 文本 embedding | [1, 9, 4096] | 查词表 |
| 拼接序列 | [1, 585, 4096] | 576 + 9 |
| 注意力掩码 | [1, 585, 585] | 因果三角掩码 |
| 每步 logits | [1, 585, V] | V = 词表大小 |
| 采样 | [1] | 下一个 token id |

### 8.3 训练链路与推理的差异
| 差异点 | 推理 | 训练 |
|--------|------|------|
| 文本部分 | 增量生成，逐步拼接 | 一次性给出完整回答（teacher forcing） |
| 掩码 | 因果掩码 | 因果掩码 + answer mask（user 部分不贡献损失） |
| 目标 | 采样 token | 对 answer 每位置算 CE 损失回传梯度 |
| KV cache | 复用（逐步追加） | 不需要（每步全量前向） |

---

## 九、性能与显存考量

### 9.1 VLM 推理显存构成

以"7B LLM + 400M 视觉塔 + 576 视觉 token + 1024 文本上下文"为例（BF16）：
| 组成 | 计算 | 显存 |
|------|------|------|
| 模型权重 | 7B × 2B | 14 GB |
| 视觉塔权重 | 400M × 2B | 0.8 GB |
| KV cache | $2 \times L \times d_{kv} \times N \times \text{bytes}$ | 约 1.3 GB（见下） |
| 激活（推理时少量） | — | ~1 GB |
| 合计 | | **约 17 GB（单卡 24GB 勉强，40GB 舒适）** |

### 9.2 KV Cache 公式（视觉 token 的隐藏成本）

$$\text{KV显存} = 2 \times L \times n_{kv\_heads} \times d_{head} \times N_{tokens} \times \text{dtype\_bytes}$$

以 LLaMA-2-7B 配置（L=32, 32 头, d_head=128, BF16）计算：

$$\text{每token} = 2 \times 32 \times 32 \times 128 \times 2 = 512 \ \text{KB/token}$$
| 场景 | token 数 | KV 显存 |
|------|---------|---------|
| 纯文本 1024 token | 1024 | 512 MB |
| 加 576 视觉 token | 1600 | 800 MB（**多占 288 MB**） |
| 高分辨率 4032 token | 5056 | 2.5 GB |

> **核心结论：视觉 token 每个都和其他文本 token 一样占 KV 显存，且注意力成本按平方增长。这就是 VLM 比 LLM 慢/贵的根本原因。**

### 9.3 视觉 token 数量对计算量的影响（公式）

单层 Transformer 的注意力 FLOPs（近似）：

$$\text{FLOPs}_{\text{attn}} \approx 2 N^2 d + 4 N d^2, \quad N = N_v + N_t$$

**视觉 token 增加 Δ 带来的额外开销**：

$$\Delta \text{FLOPs} \approx 2d \cdot L \cdot \big[(N_t + N_v)^2 - N_t^2\big] = 2dL(2 N_t N_v + N_v^2)$$
| 视觉 token 数 N_v | 相对纯文本的注意力开销（N_t=512, d=4096, L=32） |
|------------------|------------------------------------------------|
| 32（Q-Former） | +12%（几乎无感） |
| 256 | +80% |
| 576（LLaVA 默认） | +200%（3 倍注意力开销） |
| 4032（AnyRes） | +3000%（数量级爆炸） |

- **线性项 $2N_t N_v$**：视觉 token 与每个文本 token 都要交互，逃不掉；
- **平方项 $N_v^2$**：视觉 token 内部互相注意，**是切块/高分辨率爆炸的元凶**；
- 这也是 Q-Former（32 token）"虽丢信息但快"、以及 token 压缩的动机来源。

### 9.4 训练显存（粗估）

AdamW + 混合精度下每可训练参数约 **16 B**（FP16 权重 2B + 梯度 2B + FP32 主权重 4B + Adam m/v 8B）。7B 全参微调 ≈ 112 GB 权重态 + 激活，所以业界要么 FSDP/ZeRO-3 分片，要么 LoRA（只训 1%~2% 参数，显存直降）。

---

## 十、高频面试问答

**Q1：VLM 为什么能"看图"？**
图像被编码成与文本 token 同构的向量序列（视觉 token），与文本拼接后进入同一个 Transformer。注意力机制天然支持任意顺序的混合序列——模型看的是"序列"，不区分来源。而"看得懂"靠对齐训练：生成对齐让视觉 token 的语义与文本一致，指令对齐让模型学会使用这些语义作答。

**Q2：视觉 token 和文本 token 有什么区别？**
| | 文本 token | 视觉 token |
|--|-----------|-----------|
| 来源 | 词表查表（离散） | 连续特征投影（无词表） |
| 语义单位 | 子词（wordpiece） | patch（16×16 像素块） |
| 长度 | 由句子决定（通常短） | 由分辨率决定（通常很长） |
| 语言先验 | 有（模型预训练过） | 无（需要对齐训练建立） |
| 位置含义 | 文本顺序 | 二维空间位置 |

**Q3：projector（桥接模块）的作用是什么？**
两个：一是**维度对齐**（d_v → d_llm），二是**语义映射**（把视觉特征从 CLIP 空间"翻译"到 LLM 词嵌入空间）。LLaVA 实验证明 2 层 MLP 显著优于 1 层 Linear——说明非线性语义映射是必要的。训练后它成为"两种模态的字典"。

**Q4：为什么大多数 VLM 冻结视觉塔？**
① 视觉塔预训练成本极高，已有特征质量足够好；② 微调视觉塔需要成比例的图文数据，否则**灾难性遗忘**（丢失通用视觉能力）；③ 视觉塔参数量大（0.4B~6B），训练/显存成本高；④ 实践表明冻结时 SFT 效果与微调相当（LLaVA 实验结论）——"对齐的关键在桥和 LLM，不在塔"。少数例外（InternVL）用联合微调换更强视觉推理。

**Q5：VLM 的幻觉为什么发生？**
① **数据偏差**：训练 caption 中高频对象（"蓝天""草地"）被过度关联，模型按统计惯性补全而非看图；② **语言先验过强**：自回归倾向"顺口"的答案，视觉证据权重不足；③ **视觉信息丢失**：token 压缩/CLS 池化丢了细节（计数、小物体）；④ **评估歧义**。缓解：细粒度数据、偏好对齐（RLHF/DPO）、对比解码、幻觉检测模型。

**Q6：Q-Former 和 MLP projector 怎么选？**
看应用：① token 数敏感（超长上下文、视频、高分辨率）→ Q-Former/压缩，固定 token 控成本；② 追求细粒度（OCR、计数、小目标）→ MLP 保全部 token；③ 资源紧张 → MLP（参数少）。注意 Q-Former 训练更复杂（预训练初始化、查询初始化），近年趋势是压缩式桥接（pixel shuffle）兼顾两者。

**Q7：视觉 token 太多/太少各有什么问题？**
太多：KV cache 线性涨、注意力平方涨、长上下文被图像挤占，推理变慢变贵（9.3 公式）；太少：信息有损——32 个 token 无法表达空间细节，计数/定位/OCR 能力崩。工程上 256~1024 token/图是常见甜点区，Qwen2-VL 按分辨率动态分配。

**Q8：为什么现代 VLM 用生成+指令对齐，而不是对比对齐？**
对比对齐的产物是"嵌入空间"，只支持检索/分类，不支持生成——而产品最终形态是对话；且对比损失只在特征层监督，与"生成正确文本"之间存在 gap。生成对齐的损失就是生成本身（交叉熵），目标与产品一致、可端到端优化；指令对齐补齐"按用户意图"的行为层。所以对比对齐退居为"视觉塔预训练"，VLM 本体用生成+指令。

**Q9：两阶段训练为什么第一阶段只训 projector？**
投影器初始输出分布与 LLM 词嵌入空间差距巨大（随机投影的"噪声"）。若第一阶段就放开 LLM，LLM 会在噪声视觉输入上产生错误学习（灾难性遗忘基础能力）。先固定 LLM、只训投影器，是"把输入端调好再让 LLM 接手"；第二阶段再放开 LLM 联合训练，此时输入已稳定。这是 BLIP-2/LLaVA 验证过的工程经验。

**Q10：KV cache 里为什么会有视觉 token？显存怎么算？**
视觉 token 和文本 token 一起作为 LLM 输入，每个 token 的 K、V 都会被缓存以复用。公式：`2 × L × n_kv_heads × d_head × (N_v + N_t) × bytes`。576 个视觉 token 在 7B 模型上多占约 300MB 且使注意力计算翻倍——"视觉 token 是要钱的"。

---

## 十一、常见误区

**误区 1：VLM 能看图 = LLM 自己有视觉能力。**
错。LLM 自始至终只处理 token 序列，"看"是视觉编码器 + 桥接模块完成的，LLM 只是"读"了视觉 token。这也是换掉视觉塔模型能力大变、而换 LLM 影响更大的原因——能力分布在不同模块。

**误区 2：视觉 token 越多越好。**
错。token 多信息多，但计算量平方级增长、KV 显存线性增长、长上下文被挤占，注意力还可能被海量视觉 token 稀释。工程上是在"信息保留"与"算力成本"之间权衡——压缩与切块都是这种权衡的产品。

**误区 3：冻结视觉塔 = 视觉塔完全不参与训练。**
基本对但要看细节：冻结指不更新梯度，但特征仍前向传播并影响桥和 LLM 的梯度；且"冻结"程度可调（全冻 / 低 lr / 后几层解冻）。说"完全无关"是错的——它输出的特征质量仍是全系统的天花板。

**误区 4：CLIP 对比对齐是 VLM 的核心对齐方式。**
错。对比对齐是视觉塔预训练的方式，VLM 本体的对齐是生成对齐（预训练）+ 指令对齐（SFT）。面试时把"食材"和"菜"分清。

**误区 5：幻觉只能靠训练数据解决。**
错。数据与训练（SFT/DPO）是主因缓解，但推理期手段同样有效：解码温度、beam search、对比解码（CD）、视觉证据约束解码、以及"先检索后回答"。幻觉是"统计先验与证据冲突"的系统性问题，单一手段不解决。

---

## 十二、自我检验

- [ ] 能用一句话+一个公式定义 VLM 三段式结构，并列出 4 个实例化模型
- [ ] 能画出三段式架构图（视觉塔→桥→拼接→LLM）
- [ ] 能说清视觉塔选型（CLIP 借用 vs 自研）的利弊
- [ ] 能解释为什么 VLM 必须用全部 patch token 而不是 CLS
- [ ] 能默写 patch 数公式 $N=(H/P)(W/P)$，算 224/16、336/14、224/14
- [ ] 能讲清桥接三方案（MLP/Q-Former/压缩）的对比表并各举一个代表模型
- [ ] 能默写一个 prompt 模板并解释 `<image>` 占位符机制
- [ ] 能画出"图像→视觉 token"的 7 个环节，每个环节说出可选方案与影响
- [ ] 能写出四种对齐范式的损失公式与核心思想
- [ ] 能解释"为什么现代 VLM 用生成+指令对齐"（演进逻辑）
- [ ] 能写出余弦相似度、Recall@K 的公式
- [ ] 能说清多图/视频/AnyRes 三种输入处理的方案与 token 成本
- [ ] 能画出四阶段训练范式表（训什么/冻什么/数据/损失）
- [ ] 能写出 DPO 损失公式并说明与 PPO 的区别
- [ ] 能完整走一遍问答链路（文字步骤 + 张量形状）
- [ ] 能写出 KV cache 公式并算 7B 模型 576 视觉 token 的开销
- [ ] 能直接回答 10 个高频面试问答
- [ ] 能识别并反驳 5 个常见误区

---

## 参考文献

1. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al., ICML 2023
2. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — Liu et al., NeurIPS 2023
3. [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) — Liu et al., 2023
4. [LLaVA-NeXT: Scaling to Visual Reasoning](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — Liu et al., 2024
5. [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) — Bai et al., 2023
6. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) — Wang et al., 2024
7. [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) — Chen et al., CVPR 2024
8. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., ICML 2021
9. [Sigmoid Loss for Language-Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) — Zhai et al., ICCV 2023
10. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., NeurIPS 2023
