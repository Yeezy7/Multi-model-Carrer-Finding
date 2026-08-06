# InternVL 与主流 VLM 架构横向对比

> 本笔记回答一个面试核心问题：**开源 VLM 那么多，它们到底差在哪？** 答案是：差在视觉塔（多大）、桥接器（怎么把视觉变成 token）、token 策略（留多少、怎么压）、训练范式（训到哪一步）。InternVL 系列是这四件事做得最全的开放模型，以它为轴心，横向对比 LLaVA、Qwen-VL、MiniCPM-V、PaliGemma、Llama 3.2-Vision，并延伸到 GPT-4o / Gemini 的原生多模态路线。

## 一、一句话解释

> **InternVL = 以 6B 参数超大视觉塔 + 可压缩桥接（QLLaMA → pixel shuffle + MLP）+ 各种开源 LLM（Vicuna / InternLM / Qwen）组合的开放 VLM 家族，其演进浓缩了 VLM 架构的全部核心命题：视觉信息如何最高效地"翻译"成语言模型能消费的 token。**

各家的定位一句话：

| 模型 | 一句话定位 |
|------|-----------|
| LLaVA-1.5 | "最大众"的 MLP 直通方案，全民基线 |
| BLIP-2 | "最省 token"的 Q-Former 方案，32 个 query 打天下 |
| Qwen-VL 系列 | "最灵活"的动态分辨率 + 原生长上下文，产品化最彻底 |
| InternVL 系列 | "最重"的视觉塔（6B）+ "最省"的像素重组，开源推理标杆 |
| MiniCPM-V | 端侧部署王者，重采样器 + 极致压缩 |
| GPT-4o / Gemini | 原生多模态（传闻），不接投影层直接训 |

---

## 二、先建立坐标系：VLM 的标准三段式

### 2.1 通用骨架

几乎所有开放 VLM 都是三段式（这也是"连接方式分类"的前提）：

```text
图像
 ↓
视觉塔 Vision Tower（ViT 或 SigLIP 类，通常冻结或低学习率）
 ↓ 输出 N 个 patch token（每个 token 是 C 维向量）
桥接器 Connector（MLP / Q-Former / 重采样器 / 线性层 / pixel shuffle）
 ↓ 输出 M 个 LLM 可直接消费的 token（M ≤ N，或等于 N）
语言模型 LLM（7B~78B，next-token prediction 目标）
 ↓
文本输出
```

**关键变量只有四个：视觉塔多大（N 与 C）、桥接器怎么算（映射方式）、token 留多少（压缩率）、训练到哪个阶段（对齐/预训练/SFT/RL）。** 后续所有对比都围绕这四个变量展开。

### 2.2 三大连接方式总表（先给分类，第四节展开）

| 类别 | 机制 | 代表模型 | token 行为 | 优点 | 缺点 |
|------|------|---------|-----------|------|------|
| MLP / 线性投影类 | 每个 patch token 过一层（或多层）MLP 直接进 LLM | LLaVA、InternVL-1.5/2、PaliGemma、Llama 3.2、Qwen-VL（前有 resampler） | 不压缩或仅轻度压缩，语义逐 patch 保留 | 简单、通用、训练数据需求低、对齐快 | token 多 → 计算贵；细节靠 LLM 自己消化 |
| Q-Former / 重采样类 | 可学习 query 通过 cross-attention 从视觉特征中"提炼"固定数量 token | BLIP-2、InstructBLIP、Qwen-VL、MiniCPM-V、InternVL-1（QLLaMA） | 强压缩（如 257→32），token 数固定 | 大幅降计算、可控性好 | 细节丢失风险、两阶段训练耦合、query 语义不可解释 |
| 原生多模态类 | 没有独立视觉塔 + 桥接器，全部模态直接进统一 Transformer | Gemini、GPT-4o（均为传闻/推断） | 视觉即"token"或连续特征，与文本统一建模 | 信息无瓶颈、端到端最优 | 训练成本极高、几乎无法复现、技术细节不公开 |

---

## 三、InternVL 系列演进（重点）

### 3.1 InternVL-1（2023）：为什么视觉塔要 6B？

#### 3.1.1 架构

InternVL-1 是第一个"LLM 级视觉塔"的开源 VLM：

```text
图像 (448×448, patch 14)
 ↓
InternViT-6B（48 层，hidden ≈ 3200，输出 32×32=1024 个 patch token，每 token 3200 维）
 ↓
QLLaMA（Q-Former + LLaMA 嵌入融合：64 个可学习 query，输出 64×256 维）
 ↓
LLM（Vicuna-7B/13B、InternLM-20B 等，可换）
```

**QLLaMA 的融合技巧**：普通的 Q-Former 只做 cross-attention 提特征；QLLaMA 在每个 block 里把 64 个 query 与"投影后的 LLM 输入嵌入"一起送入 self-attention，让 query 直接"看"到 LLM 的嵌入空间再决定自己长什么样。这比纯 Q-Former 的隐式对齐更直接，是 InternVL-1 对齐效果好的关键。

#### 3.1.2 为什么视觉塔要 6B（高频考点）

1. **规模收益真实存在**：InternVL-1 论文把视觉塔从 ViT-B（87M）→ ViT-L（304M）→ 6B 逐级放大，多模态基准持续涨点；在 16 项 benchmark 上，InternViT-6B + Vicuna-13B 全面碾压当时用 CLIP ViT-L 的 LLaVA-1.5 和 InstructBLIP。视觉理解的规模定律（vision scaling law）与 LLM 一样存在。
2. **容量匹配原则**：LLM 是 7B~20B，而标准 CLIP ViT-L 只有 304M——**语言侧容量比视觉侧高一个数量级，弱视觉塔是整个系统的信息瓶颈**。视觉信息在塔里就被压缩失真，后面任何桥接器都救不回来（"garbage in, garbage out"）。
3. **对齐时不丢信息**：与 LLM 对齐/微调时，6B 塔有足够容量把 OCR、计数、空间关系、细粒度属性编码进表征；小塔为了塞进 300M 参数必然丢弃这些细节。InternVL-1 实验表明大塔带来的收益在文档、图表、细粒度分类任务上尤其显著。
4. **数据门槛**：6B 塔需要大规模预训练（LAION-400M 级图文对 + 内部数据），这不是普通团队能复制的——所以后来 InternVL2 对小模型折中为 300M 塔，验证了"**视觉塔规模与 LLM 规模匹配**"的设计原则。
5. **代价与配套**：6B 塔推理贵、显存大，因此 InternVL-1.5/2 必须配套动态分辨率 + pixel shuffle 压缩 token，否则训练和推理都不可行。**大塔是前提，压缩是配套，两者缺一不可。**

> 记忆点：**视觉塔小 → 信息瓶颈在视觉侧；视觉塔大 → 瓶颈转移到计算量；所以大塔必须配强压缩。**

### 3.2 InternVL-1.5（2024）：动态分辨率 + pixel shuffle + 4K 训练

InternVL-1.5 用"三件套"解决了 InternVL-1 的痛点（固定 448 分辨率、token 太多、不支持高分辨率细节）：

1. **动态分辨率（dynamic resolution）**：图像不再强行 resize 到固定尺寸，而是按纵横比切成若干 448×448 的 tile，另加一张全局缩略图（thumbnail）；
2. **Pixel Shuffle（像素重组）压缩**：每个 tile 的 patch 特征经 pixel shuffle 4× 压缩，token 大幅减少；
3. **4K 级训练**：训练数据引入 4K 级高分辨率图像（通过多 tile 组合覆盖），让模型真正见过高分辨细节而不是只在推理时硬塞。

#### 3.2.1 动态分辨率设计

```text
输入图像（任意尺寸、任意纵横比）
 ↓
策略 A：缩略图 thumbnail —— 整图缩放到 448×448（全局上下文，1 张）
策略 B：tile 切块 —— 按 448×448 网格切割，保留原纵横比（细节，最多 11 块）
 ↓
合计最多 12 个子图（1 缩略图 + 11 tile），每个子图独立过视觉塔
```

- 每个子图输出 32×32 = 1024 个 patch token，经 pixel shuffle 4× 压到 **256 个 token**；
- 整图最多约 12 × 256 = **3072 个视觉 token**（对比 LLaVA-NeXT 的约 5300 个，同样分辨率覆盖下省一半计算）；
- 缩略图保证"全局语义"，tile 保证"局部细节"，两者互补——这是 2024 年主流动态分辨率方案的共识结构。

#### 3.2.2 Pixel Shuffle 数学原理（必背）

设视觉塔输出特征图为 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$，其中 $H \times W = N$ 是 patch token 数量。Pixel shuffle（实现上等价于 space-to-depth，论文沿用 super-resolution 领域的名字）把相邻 $s \times s$ 的 token 块**合并为一个 token，通道数变为 $s^2 C$**：

$$\mathbf{X} \in \mathbb{R}^{H \times W \times C} \;\xrightarrow{\text{pixel shuffle }(s)}\; \tilde{\mathbf{X}} \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times s^2 C}$$

token 数量与通道数变化：

$$N' = \frac{N}{s^2}, \qquad C' = s^2 C$$

以 $s = 2$ 为例，每个 $2 \times 2$ 空间块的 4 个 token 的通道向量首尾拼接成新 token：

$$\tilde{\mathbf{x}}_{i,j} = \big[\, \mathbf{x}_{2i,2j}^{\top},\; \mathbf{x}_{2i,2j+1}^{\top},\; \mathbf{x}_{2i+1,2j}^{\top},\; \mathbf{x}_{2i+1,2j+1}^{\top} \,\big]^{\top} \in \mathbb{R}^{4C}$$

然后接一个可学习线性层，把 $s^2 C$ 维压缩到 LLM 的 hidden size $d$：

$$\mathbf{Z} = \tilde{\mathbf{X}} \mathbf{W}, \qquad \mathbf{W} \in \mathbb{R}^{s^2 C \times d}$$

**数值例子**（448×448、patch 14、InternVL-1.5 默认 $s=2$）：

| 阶段 | token 数 | 每 token 维度 |
|------|---------|--------------|
| 视觉塔输出 | 32 × 32 = 1024 | C |
| pixel shuffle 后 | 16 × 16 = 256（4× 压缩） | 4C |
| 线性投影后 | 256 | d（LLM hidden） |

**关键性质（面试加分点）**：

1. **pixel shuffle 本身是无损的信息重排**——只是把空间上的 4 个 token 换成了通道上的 4 个分组，可以逆操作还原；信息丢失发生在后续线性压缩 $4C \to d$（因为 $4C > d$，这一步才是有损的）；
2. 等价数学视角：token 数按 $s^2$ 缩减，等价于把 patch 网格下采样 $s$ 倍，但每个新 token 保留了 $2 \times 2$ 邻域的**全部原始通道信息**，比直接空间下采样（丢弃 3/4 token）信息保留完整得多；
3. 相比 Q-Former 的"学习式提炼"，pixel shuffle 是**结构式压缩**：无额外可学习参数做信息选择，训练更稳定、压缩比可预期、不引入额外的两阶段训练复杂度。

#### 3.2.3 与 LLaVA-NeXT anyres 的异同

| 维度 | LLaVA-NeXT anyres | InternVL-1.5 动态分辨率 |
|------|-------------------|------------------------|
| 基本思路 | 缩略图 + tile 切块 | 缩略图 + tile 切块（相同） |
| tile 尺寸 | 336×336 | 448×448 |
| tile 上限 | 最多 9 个 tile + 1 缩略图 | 最多 11 个 tile + 1 缩略图（共 12 子图） |
| 每 tile token | 24×24 = 576（不压缩） | 1024 → pixel shuffle 4× → 256 |
| 缩略图处理 | 缩放后特征 1/4 下采样（576→144） | 与 tile 相同，pixel shuffle 4×（256） |
| 全图 token 上限 | 9×576 + 144 ≈ 5328 | 12 × 256 ≈ 3072 |
| 位置信息 | 每 tile 独立编码后拼接（无跨 tile 位置建模） | 同左（靠 LLM 隐式对齐） |
| 训练 | 主要在 336 分辨率 + anyres 推理增强 | 训练阶段即用动态分辨率（含 4K 数据） |
| 计算成本 | 高（token 全保留） | 低（4× 压缩） |
| 细节保留 | 极好（几乎无压缩） | 好（有损但保留了通道信息） |

> 面试结论：**anyres 是"用计算换细节"，InternVL 是"用通道换空间"**——两者都解决"高分辨率下 token 爆炸"问题，但 InternVL 的 pixel shuffle 让高分辨率训练/推理的成本可控到可落地。

### 3.3 InternVL2（2024）：全尺寸家族 + MoE

InternVL2 把 InternVL-1.5 的配方产品化：**把视觉塔、桥接器、LLM 做成可自由组合的积木**，一口气发布 1B~76B（后增 78B）全系列：

| 尺寸 | LLM | 视觉塔 | 备注 |
|------|-----|--------|------|
| 1B / 2B | InternLM2-1.8B / InternLM2-2B | InternViT-300M | 端侧 |
| 4B | Phi-3-mini | InternViT-300M | 小容量 |
| 8B | internlm2.5-7B | InternViT-300M | 单卡部署主力 |
| 26B | internlm2.5-20B | InternViT-6B | 中档 |
| 40B | Qwen2-7B MoE 上采样（≈13B 激活） | InternViT-6B | MoE |
| 76B | 70B 级 MoE（激活约 1/4） | InternViT-6B | 旗舰 |

要点：

1. **视觉塔选择原则**：小模型（≤8B）配 300M 塔，大模型（≥26B）配 6B 塔——再次印证"塔规模与 LLM 规模匹配"（一个 2B LLM 挂 6B 塔，视觉侧算力浪费且推理失衡）；
2. **MoE upcycling**：40B/76B 不是从头训练的 MoE，而是把稠密 checkpoint（如 Qwen2-7B）的 FFN 层复制成多个 expert + 可学习 router，即"MoE 上采样"：
   - 总参数放大 $k$ 倍（$k$ ≈ expert 数），激活参数基本不变（每 token 只走少数 expert）；
   - 效果：**容量接近大模型、推理成本接近小模型**——对视觉 token 多、长上下文场景尤其划算；
   - 代价：router 需要额外训练，expert 间负载不均衡需要正则；
3. **统一桥接**：InternVL2 把 QLLaMA 换成了更简单的 **pixel shuffle + MLP**（Q-Former 路线被淘汰，因为生成式预训练数据变多后，MLP 直通 + 结构压缩更高效、更稳），证明"桥接器随训练数据规模增加而简化"的趋势；
4. **协议**：InternVL2 全系 MIT 开源（权重 + 训练代码），商业友好，这是它成为开源事实标准之一的重要原因。

### 3.4 InternVL2.5 / InternVL3：OCR 强化 → 推理与规划

| 版本 | 时间 | 核心变化 |
|------|------|---------|
| InternVL2.5 | 2024.12 | 1B~78B 全系；**78B 采用 Qwen2.5-72B 作 LLM**；OCR / 表格 / 图表 / 文档理解大幅强化；视觉塔微调策略更激进 |
| InternVL3 | 2025.4 | 1B~78B；**两阶段对齐预训练**（单图对齐 → 交错图文对齐）；引入 **RL（偏好优化 + GRPO 类推理强化）**；强调推理（reasoning）、规划（planning）、Agent 工具调用、视觉定位（grounding）能力 |

InternVL3 与 2.5 的本质区别不在架构而在**训练配方**：

- 视觉塔与 LLM 不再是"冻结塔 + 训桥接"，而是**分层解锁、联合训练**（塔只解锁最后一层 / 低学习率）；
- 对齐预训练分两步：先单图对齐（把视觉 token 学进 LLM 嵌入空间），再做交错图文对齐（interleaved image-text，多图/图文交替序列），解决了"单图 SFT 好但多图/交错差"的经典问题；
- 强化学习从"人类偏好"（RLHF-V / DPO 类）扩展到"推理信号"（GRPO 类，用可验证奖励如数学答案、代码执行结果），让模型学会长链推理与自我反思——这条路线与 DeepSeek-R1 的 LLM 侧经验完全同构。

### 3.5 多阶段训练管线（对比对齐 → 生成预训练 → 指令微调）

InternVL 确立并固化了 VLM 训练的标准三阶段（+ 可选的第四阶段 RL）：

| 阶段 | 目标 | 数据 | 冻结策略 | 损失 | 说明 |
|------|------|------|---------|------|------|
| 0. 视觉塔预训练 | 通用视觉表征 | 图文对（CLIP 类） | 单独训练 | 对比损失 | 复用现有塔（CLIP/SigLIP/InternViT），一般不再做 |
| 1. 对比对齐（alignment） | 把视觉特征拉进 LLM 嵌入空间 | 百万级图文对 | 塔冻结或微调最后一层 | 对比 + 生成 | InternVL-1 用 QLLaMA + 对比损失；1.5 起以生成为主 |
| 2. 生成预训练（generative pretraining） | 学会"看着图说话" | 千万~亿级图文交错数据 | 塔冻结（低 lr），桥 + LLM 全训 | NTP（next-token prediction） | 与 LLM 预训练目标完全一致，数据异构是关键 |
| 3. 指令微调（SFT） | 学会回答问题/跟随指令 | 百万级指令数据（VQA、对话、OCR…） | 塔低 lr 或冻结，其余全训 | NTP | 各任务混比（caps:VQA:对话 ≈ 2:1:1 类经验值） |
| 4. 偏好对齐 / RL | 对齐人类偏好、提升推理 | 偏好对 / 可验证奖励 | 同 SFT | DPO / RLHF-V / GRPO | InternVL3 引入；GPT-4o / Gemini 高度依赖 |

面试要能讲清两点：

1. **为什么阶段必须分离**：对齐阶段数据少但质量高，用于"校准"（calibrate）桥接器；直接让 LLM 带着未对齐的桥接器看亿级数据，梯度信号里视觉信息占比太小，会被文本侧淹没（loss 由文本主导），视觉表征学不进去。分离后每阶段目标单一、可控。
2. **阶段之间的竞争关系**：数据够多时（如 Qwen2-VL），阶段 1+2 可以合并甚至退化为"单阶段端到端"（LLaVA-OneVision 等），因为大模型 + 大数据自会收敛；数据少时（如 BLIP-2、InternVL-1），阶段分离是保效果的必须。

### 3.6 演进总结表

| 版本 | 年份 | 视觉塔 | 桥接 | token 策略 | 训练亮点 |
|------|------|--------|------|-----------|---------|
| InternVL-1 | 2023 | InternViT-6B | QLLaMA（64 query） | 固定 448，64 token | 首个 6B 塔开源 VLM，三阶段雏形 |
| InternVL-1.5 | 2024 | InternViT-6B | pixel shuffle + MLP | 动态分辨率 + 4× 压缩，≈3K token | 4K 训练、动态分辨率、视频支持 |
| InternVL2 | 2024 | 300M / 6B 分级 | pixel shuffle + MLP | 同上 | 全尺寸家族 + MoE、MIT 开源 |
| InternVL2.5 | 2024 | 300M / 6B 分级 | pixel shuffle + MLP | 同上 | 78B、OCR / 文档强化 |
| InternVL3 | 2025 | 300M / 6B 分级 | 同左 + 更激进联合训练 | 同上 | 两阶段对齐 + RL、推理 / Agent / 定位 |

---

## 四、主流 VLM 架构分类总表

### 4.1 分类总表

| 大类 | 桥接机制 | 代表模型 | 视觉 token 数（典型） | 训练难度 | 部署成本 |
|------|---------|---------|---------------------|---------|---------|
| MLP 直通类 | 两层 MLP / 线性层，token 逐一映射 | LLaVA-1.5 / LLaVA-NeXT、InternVL-1.5/2（+pixel shuffle）、PaliGemma（线性）、Llama 3.2-Vision、MiniCPM-V*、Qwen-VL* | 576（336²）；可到 5K+ | 低 | 高（token 多） |
| Q-Former / 重采样类 | 可学习 query + cross-attention | BLIP-2、InstructBLIP、Qwen-VL（256 query）、MiniCPM-V（Perceiver resampler）、InternVL-1（QLLaMA） | 32~256 | 中（两阶段耦合） | 低 |
| 原生多模态类 | 无独立塔/桥，统一 Transformer | Gemini 系列、GPT-4o（均为传闻） | 未知（原生 token / 连续特征） | 极高 | 未知（极高） |

> \* MiniCPM-V 与 Qwen-VL 属于混合：视觉特征先过 resampler（固定压缩）再过 MLP/线性层，两套机制叠加。

### 4.2 各类机制点评

**MLP 直通类**：信息路径最短、最透明，"每个 patch 都保留"，LLM 决定用多少。训练最简单（只需要图文数据 + SFT），是研究社区默认基线。瓶颈：token 数与分辨率线性增长，高分辨率下 attention 计算 $O(N^2)$ 爆炸。**这是 LLaVA 生态（LLaVA / LLaVA-NeXT / InternVL / CogVLM）的共性。**

**Q-Former / 重采样类**：用 $M$ 个固定 query 通过 cross-attention 从 $N$ 个 patch 中"提炼"信息，$M \ll N$（BLIP-2 的 32 个 query 对应 257 个 patch，约 8× 压缩）。query 本质是**可学习的"信息筛选器"**：LLM 只看被 query 选中的信息。优点：token 恒定、计算可控、能对接任意视觉塔；缺点：压缩有损（细粒度细节、空间关系容易丢）、query 语义黑盒、两阶段训练（先对齐塔和 query，再训 LLM）不稳定。**InstructBLIP、MiniCPM-V 的 Perceiver resampler 都是同类思想。**

**原生多模态类**：Gemini（2023）首次提出统一多模态 tokenizer（图像被 tokenizer 切成原生视觉 token，与文本 token 一起建模）；GPT-4o（2024）据传是"any-to-any"统一 Transformer（音频也进同一序列）。这类架构没有"桥接器"概念——视觉信息从第一层就参与统一的注意力计算，理论上无信息瓶颈。**代价是训练数据、算力、系统复杂度都到极致，目前只能由大厂承担，技术细节不公开。**

---

## 五、视觉 token 处理路线对比（重点，最详细）

### 5.1 为什么 token 数量是命门

Transformer 每层的自注意力复杂度：

$$O(N^2 \cdot d), \quad N = \text{token 数}, \; d = \text{hidden size}$$

视觉 token 从 576（336²）翻倍到 5760（anyres 9 tile），**注意力计算量 ×100**。同时长上下文（LLM 侧 32K~128K）本来就吃显存，视觉 token 一多，推理延迟、KV cache、显存全部失控。因此 2024 年之后所有 VLM 的架构创新，一半以上都在回答同一个问题：**token 太多怎么办？**

### 5.2 路线一：保持全部 token（LLaVA / PaliGemma / Llama 3.2）

- **做法**：视觉塔输出 $N$ 个 patch token，投影后全部拼进 LLM 序列，一个不丢；
- **代表**：LLaVA-1.5（576 个）、PaliGemma（224²→256 个）、Llama 3.2-Vision；
- **优点**：信息无损、实现最简单、训练要求最低（这也是 LLaVA 全民基线的原因）；
- **缺点**：分辨率被锁死（LLaVA-1.5 只有 336²，PaliGemma 靠整图放大到 448/896 支持高分辨率，token 随之膨胀到 1024/4096）；无法承受高分辨率 + 长文本同时在线；
- **适用**：研究基线、中低分辨率通用任务、对实现复杂度敏感的团队。

### 5.3 路线二：固定压缩（Q-Former 32 个 / InternVL pixel shuffle）

| 方案 | 压缩机制 | 典型压缩比 | token 数 | 信息损失点 |
|------|---------|-----------|---------|-----------|
| Q-Former（BLIP-2） | 32 个 query 从 257 个 patch 中提炼 | 约 8× | 32 | query 选择本身有损（学习式） |
| Perceiver resampler（Qwen-VL / MiniCPM-V） | 256/96 个 query | 约 2~8× | 96~256 | 同上 |
| Pixel shuffle（InternVL） | 结构重排 + 线性压缩 | 4×（s=2）或 16×（s=4） | 256/tile | 线性压缩 $4C \to d$ 有损（结构式） |
| QLLaMA（InternVL-1） | 64 query + LLM 嵌入融合 | 约 16× | 64 | 学习式 |

两种压缩的本质区别：

- **Q-Former 是"学习式压缩"**：压缩器自己决定哪些信息重要，可解释性差，训练需要两阶段（先对齐）；好处是 token 数绝对可控（始终 32/256），LLM 侧开销极低；
- **pixel shuffle 是"结构式压缩"**：先无损重排再线性降维，压缩比由 $s$ 决定、可预期，不需要额外训练技巧；缺点是压缩比固定（不能按图像内容调节）。

### 5.4 路线三：自适应压缩（Qwen2.5-VL token merger）

固定压缩的痛点：**简单图（纯色背景）和复杂图（满屏文字）用同样的 token 数，要么浪费、要么不够。** 自适应压缩按内容动态决定留多少 token：

**Qwen2.5-VL 的 token merger（受 ToMe / token merging 启发）**：

- 在视觉塔的 Transformer 层里，每层用**基于 patch 相似度**的双向匹配（bipartite matching）把最相似的两个相邻 token 合并成一个（特征取均值，必要时加权）；
- 合并比率可配置（如每层合并 $\alpha$ 比例），整体实现"高信息区域保留、冗余区域压缩"；
- 效果：1M 像素级大图生成数千 token，经 merger 后压缩数倍，**图像信息密度低时压缩更多、信息密度高时压缩更少**；
- 相比 pixel shuffle：压缩率随内容动态变化，但引入匹配开销，且压缩不可逆（确实丢信息）。

其他自适应思路（面试可补充）：token pruning（按注意力分数剪枝，如 FastV）、LLaVA-PruMerge（聚类后保留代表性 token）等。

### 5.5 路线四：动态分辨率（Qwen2-VL / InternVL-1.5 / LLaVA-NeXT）

| 方案 | 机制 | 分辨率上限 | token 上限 | 位置建模 |
|------|------|-----------|-----------|---------|
| LLaVA-NeXT anyres | 缩略图 + 336 tile 切块 | 3×（约 1008² 级） | 约 5328 | 隐式（无跨 tile 位置编码） |
| InternVL-1.5/2 | 缩略图 + 448 tile + pixel shuffle | 4K 级（12 子图） | 约 3072 | 隐式 |
| Qwen2-VL 原生动态 | 28×28 像素网格切分 + M-RoPE | 约 4M 像素（单边最长约 3584） | 约 2 万（4M 像素时） | **显式二维 RoPE（M-RoPE）** |

Qwen2-VL 与前两者的本质区别：

1. **没有"缩略图 + tile"的双流结构**：整图按 28×28 像素网格缩放（每格 4 个 patch token），长宽比完全保留，一条序列搞定；
2. **M-RoPE（Multimodal RoPE）显式建模二维位置**：把一维 RoPE 推广到二维平面（图像）与三维时空（视频）。每维位置角度：

$$\theta^{(i)} = 10000^{-\frac{2(i-1)}{d}}, \qquad \mathrm{Rot}(x) = \begin{bmatrix} \cos(x\theta) & -\sin(x\theta) \\ \sin(x\theta) & \cos(x\theta) \end{bmatrix}$$

   hidden 维度按轴向分组：一部分维度用 x 轴旋转、另一部分用 y 轴旋转（视频再加时间轴），各轴独立计算后再拼接，模型天然知道每个 patch 在图像里的精确坐标，**跨 tile / 跨分辨率的相对位置不再依赖 LLM 隐式猜测**——这是 Qwen2-VL 高分辨率 + 细粒度定位（grounding）能力的重要来源；
3. **训练与推理统一**：动态分辨率在训练阶段就用（不是推理时才切图），模型真正见过各种长宽比。

> 结论：**"动态分辨率"解决的是"任意尺寸图像怎么吃进去"；"高分辨率"解决的是"细节看不看得清"；"压缩"解决的是"看清了之后算不算法得起"。三件事可以自由组合，但最容易的落地顺序是：先动态分辨率，再压缩，最后再上高分辨率。**

### 5.6 综合对比大表

| 方案 | 代表模型 | token 数（典型） | 压缩率 | 细节保留 | 计算量 | 适用场景 |
|------|---------|----------------|--------|---------|--------|---------|
| 全部保留 | LLaVA-1.5、PaliGemma、Llama 3.2 | 576（336²）；256（224²） | 1× | ★★★★★ | 高 | 通用基线、低分辨率、快速开发 |
| 固定压缩-Q | BLIP-2、InstructBLIP | 32 | 约 8× | ★★ | 极低 | 检索、单 token 级语义、低算力 |
| 固定压缩-pixel shuffle | InternVL-1.5/2 | 256/tile | 4×~16× | ★★★★ | 低 | 高分辨率 + 大模型组合 |
| 自适应压缩 | Qwen2.5-VL | 数百~千级（内容自适应） | 数倍（动态） | ★★★★ | 中 | 超高分辨率、图片密度差异大的场景 |
| 动态分辨率（切块） | LLaVA-NeXT、InternVL-1.5 | 5.3K / 3K | 1×~4× | ★★★★★ | 高/中 | 文档、图表、多物体场景 |
| 动态分辨率（原生网格） | Qwen2-VL | 可达约 2 万 | 1× | ★★★★★ | 很高 | 定位、Agent、长视频帧、任意长宽比 |
| 原生多模态 | Gemini、GPT-4o | 未知 | 未知 | 未知 | 未知 | 大厂全栈 |

---

## 六、训练范式对比

### 6.1 演进路线

```text
双塔对比预训练        生成式预训练           SFT               RLHF / DPO / 推理RL
(CLIP/SigLIP/BLIP)    (BLIP-2/Qwen/LLaVA)  (所有 VLM)         (LLaVA-RLHF/InternVL3/GPT-4o)
  │                     │                   │                 │
  └─ 学会"匹配"          └─ 学会"看着图写"   └─ 学会"答指令"   └─ 学会"讨好人/会推理"
     视觉塔+文本塔         冻结塔+训桥接        全模型微调          偏好优化 + 推理奖励
```

### 6.2 各模型所处阶段

| 模型 | 对比预训练 | 生成预训练 | SFT | 偏好 / RL | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| CLIP / SigLIP | ✅ 终点 | ❌ | ❌ | ❌ | 只做对齐，是别人的视觉塔 |
| BLIP-2 | 复用 | ✅ | ✅ | ❌ | 冻结塔 + Q-Former 两阶段 |
| LLaVA-1.5 | 复用 | ✅（仅投影层） | ✅ | ❌ | 经典两阶段（投影对齐 + SFT） |
| Qwen-VL 系列 | 复用 | ✅（ViT 参与） | ✅ | ❌ | 数据驱动，阶段边界模糊 |
| InternVL-1 | 自研 6B 塔 + 对比对齐 | ✅ | ✅ | ❌ | 三阶段雏形 |
| InternVL-1.5/2 | 复用 | ✅（大规模） | ✅ | ❌ | 三阶段成熟版 |
| InternVL3 | 复用 | ✅（两阶段对齐） | ✅ | ✅ GRPO / 偏好 | 开源最早把推理 RL 做进 VLM |
| MiniCPM-V | 复用 | ✅ | ✅ | ✅ DPO（部分版本） | 端侧 + 偏好对齐 |
| GPT-4o / Gemini | 未知 | 端到端 | ✅ | ✅ 重度 | 全程闭源，RL 程度极高 |

### 6.3 三个趋势（面试高频观点）

1. **阶段在合并**：数据与算力增长后，"对齐 + 预训练"合并、甚至"SFT 前两步"合并（Qwen2-VL 直接端到端），两阶段框架只在数据不足时必要；
2. **冻结策略在松动**：早期铁律"视觉塔必须冻结"（防止灾难性遗忘 + 省算力）被打破——Qwen2-VL / InternVL3 都开始低学习率微调视觉塔，因为任务数据（OCR、图表）与预训练分布差异大，塔需要再适应；
3. **RL 从"偏好"走向"推理"**：RLHF-V（视觉偏好）→ DPO（稳定替代）→ GRPO（可验证奖励 + 推理链），与 LLM 侧 R1 路线完全同构，是 2025 年开源 VLM 的护城河。

---

## 七、六大开源 VLM 横向对比表（必背大表）

| 维度 | LLaVA-1.5 | Qwen2-VL | InternVL2 | MiniCPM-V | PaliGemma | Llama 3.2-Vision |
|------|-----------|----------|-----------|-----------|-----------|------------------|
| 视觉塔 | CLIP ViT-L/14@336（304M） | 自研 ViT（675M，patch 14） | InternViT-300M / 6B | SigLIP SoViT-400M/14 | SigLIP-So400m/14 | MetaCLIP ViT（11B 版约 100M 级） |
| 桥接 | 两层 MLP | 线性投影（无 resampler） | pixel shuffle + MLP | Perceiver resampler + 线性 | 线性投影 | MLP 投影 |
| 动态分辨率 | 无（固定 336²） | 原生动态（28 网格 + M-RoPE，约 4M 像素） | 动态（448 tile + 缩略图，12 子图） | 动态（约 1.8M 像素，2.6 版） | 无（224/448/896 整图放大） | 无（固定分辨率） |
| token 策略 | 全部保留（576） | 全部保留（最高约 2 万） | pixel shuffle 4×（≈3K） | resampler 压缩（2.6 上限约 768） | 全部保留（256/1024/4096） | 全部保留 |
| 上下文 | 2K~4K | 128K（M-RoPE 支持 2D+时间） | 8K~32K（按尺寸） | 4K 级 | ≤512（任务级短序列） | 128K（继承 Llama 3.1） |
| 参数量 | 7B / 13B | 2B / 7B / 72B | 1B~76B/78B（40B+ 为 MoE） | 2.4B~8B | 3B / 10B | 11B / 90B |
| 多图 / 交错 | 弱 | 强（原生多图） | 强 | 中 | 单图为主 | 中 |
| 开源协议 | Apache-2.0 | 2B/7B Apache-2.0，72B 自定义 | MIT | Apache-2.0（以官方为准） | Gemma 许可（需同意条款） | Llama Community License |
| 定位 | 研究基线 | 产品 + Agent | 研究 + 推理 SOTA | 端侧部署 | 多任务迁移 | 生态整合（Agent 工具） |

逐行点评（面试可引用的要点）：

1. **LLaVA-1.5**：历史地位极高（定义了两阶段范式），但架构已落后——固定分辨率、无压缩、上下文短；其生态（LLaVA-NeXT、LLaVA-OneVision）继承贡献更大；
2. **Qwen2-VL**：原生动态分辨率 + M-RoPE + 128K 上下文是架构三件套，**没有 resampler、没有切块**，是最接近"原生多模态"的开放实现；72B 的协议限制需注意；
3. **InternVL2**：唯一"大塔 + 强压缩 + 全尺寸 + MIT"全占的家族，MoE 版本把推理成本打下来了；视觉塔与 LLM 的组合自由度高；
4. **MiniCPM-V**：SigLIP 塔 + resampler 的组合是"小参数出大效果"的典范，2.6 支持多语言（100+ 语种），量化后 2~4GB 可跑端侧；
5. **PaliGemma**：极简架构（线性投影 + 全 token + 全任务微调），验证了"数据和方法 > 架构花活"；短上下文限制其通用性；
6. **Llama 3.2-Vision**：视觉塔小但工程化最好（与 Llama 生态、工具调用无缝），适合已有 Llama 栈的团队；视觉能力本身不算最强。

---

## 八、中文多模态生态

| 模型 | 出品 | 核心差异 | 强项 | 弱点 |
|------|------|---------|------|------|
| Qwen-VL 系列 | 阿里通义 | 产品化最彻底、迭代最快（Qwen-VL → 2 → 2.5） | 中文 + 英文、OCR、表格、Grounding、Agent 工具调用、长上下文 | 大版本协议非全 Apache；架构跟随性（非原创塔） |
| InternVL 系列 | 上海 AI Lab | 学术开源最强、自研 6B 塔 | 推理基准常驻榜首、全尺寸家族、MIT 协议、训练配方公开 | 产品化配套（API/生态）弱于 Qwen |
| MiniCPM-V | 面壁智能 | 端侧 / 移动端定位 | 量化友好、100+ 语种、8B 出 72B 级效果 | 上限受参数规模限制 |
| DeepSeek-VL | 深度求索 | 研究探索型（SigLIP+SamViT 混合塔 → DeepSeek-VL2 MoE） | 中文数据、数学/OCR 均衡、MoE 探索 | 团队重心转向 DeepSeek-R1 系，VLM 线已停更 |

定位差异一句话：

- **想直接落地产品 / Agent：选 Qwen-VL 系**（API、工具链、中文指令数据最全）；
- **想跟进研究 / 做推理 benchmark / 自己训练：选 InternVL 系**（MIT + 全配方公开）；
- **想端侧 / 低资源部署：选 MiniCPM-V**（int4 量化 2~4GB）；
- **想读论文学 MoE 多模态设计：读 DeepSeek-VL2**（但不要依赖它做生产）。

---

## 九、工程师 / 面试者如何选型

### 9.1 决策维度

| 维度 | 关键问题 | 对应选择 |
|------|---------|---------|
| 部署成本 | GPU 显存？单卡还是多卡？端侧？ | ≤3B：MiniCPM-V、InternVL2-2B、Qwen2.5-VL-3B；单卡 24G：7B~8B 级；多卡：26B+ |
| 分辨率需求 | 文档 / 细密表格 / 大图？ | 需要 → Qwen2.5-VL（1M 像素）、InternVL（4K）、MiniCPM-V 2.6（1.8M 像素）；不需要 → 任何模型 |
| 语言 | 中英混合？多语种？ | 中文 → Qwen / InternVL / MiniCPM；多语种 → MiniCPM-V 2.6 / Qwen2.5-VL |
| Agent 能力 | 工具调用、截图定位、GUI 操作？ | Qwen2.5-VL（Grounding 强）、InternVL3（Agent 配方）、Llama 3.2（工具生态） |
| 微调意愿 | 要自己训吗？协议允许吗？ | MIT（InternVL）＞ Apache（LLaVA / Qwen 小版本）＞ Llama / Gemma 协议 |
| 推理性能 | 延迟敏感？ | 压缩强的（InternVL / MiniCPM / Qwen2.5-VL merger）优先 |

### 9.2 场景速查

| 场景 | 首选 | 理由 |
|------|------|------|
| 研究基线 / 复现实验 | LLaVA-1.5 / InternVL2 | 生态资料最多、协议最松 |
| 中文文档理解 + Agent 产品 | Qwen2.5-VL-7B/32B | 动态分辨率 + 定位 + 工具调用 + 中文数据 |
| 高精度长文档离线分析 | InternVL2.5/3-78B | 大塔 + 4K 分辨率 + 推理能力 |
| 手机 / 边缘设备 | MiniCPM-V 2.6（int4） | 端侧实测最强 |
| 已有 Llama 技术栈 | Llama 3.2-11B-Vision | 生态一致，工程成本最低 |
| 纯视觉检索 / embedding | SigLIP（非 VLM） | 双塔对齐比生成式 VLM 更合适 |

---

## 十、常见误区

**误区 1："VLM 就是 LLM 前面加个视觉编码器。"**
错。桥接方式（MLP vs Q-Former vs 原生）、token 策略（全保留 vs 压缩）、训练范式（对齐 → 预训练 → SFT → RL）共同决定性能，架构只占一半。同样挂 7B LLM，BLIP-2（32 token）和 LLaVA（576 token）是两种完全不同的系统。

**误区 2："视觉塔必须冻结。"**
早期 VLM（LLaVA-1.5、BLIP-2）冻结塔是省算力 + 防遗忘的工程决策，不是理论必然。Qwen2-VL、InternVL3 已用低学习率微调视觉塔（尤其 OCR / 图表等与预训练分布差异大的任务），效果更好。

**误区 3："pixel shuffle 会丢失信息，所以不如全保留。"**
pixel shuffle 的 reshape 步骤本身**信息无损**（通道重排，可逆），有损的是后续线性降维。相比直接丢弃 3/4 的 token（空间下采样），它保留全部通道信息，压缩 4× 而细节保留远好于同压缩比的朴素下采样。

**误区 4："动态分辨率 = 高分辨率。"**
两个维度。动态分辨率解决"任意长宽比、任意尺寸都能吃进去"；高分辨率解决"细节看得清"。LLaVA-NeXT 动态切块但训练以 336 为主，InternVL-1.5 动态 + 4K 训练才真正把两者结合。

**误区 5："InternVL2 全系列都是 MoE。"**
只有 40B / 76B 等大版本是 MoE（基于 Qwen2-7B 的 MoE 上采样），小版本（1B~8B）是稠密架构。同理，"MoE 的激活参数少"不等于"总参数少"——部署时仍要按总参数准备显存。

**误区 6："Qwen2-VL 的动态分辨率会丢失全局信息。"**
它没有缩略图，但 M-RoPE 显式建模 patch 的二维坐标，全局信息由"所有 patch + 完整位置关系"共同承载；且训练阶段就是动态分辨率，模型已适应。切块方案（anyres / InternVL）反而靠"额外一张缩略图"弥补跨 tile 信息割裂。

---

## 十一、高频面试问答

**Q1：视觉 token 太多怎么办？**
四条路线：① 学习式固定压缩（Q-Former / resampler，32~256 token，有损且需两阶段训练）；② 结构式固定压缩（pixel shuffle，4×/16×，无损重排 + 线性降维，InternVL）；③ 自适应压缩（token merging / pruning，Qwen2.5-VL，按内容密度动态留 token）；④ 分辨率侧控制（动态分辨率 + 切块，控制单图 token 总量）。选择取决于：细节要求（OCR 要高保真 → 慎用强压缩）、计算预算、是否需要 LLM 细看每个 patch。

**Q2：讲一下 pixel shuffle 的原理？**
把视觉塔输出的 $H \times W \times C$ 特征中相邻 $s \times s$ 的 token 块在通道维拼接，token 数变为 $N/s^2$、通道变为 $s^2 C$，再线性投影到 LLM 隐藏维度。$s=2$ 时 1024 token → 256 token。重排本身无损，损失发生在线性压缩。等价于 space-to-depth，与超分领域的 pixel shuffle 互为逆操作。

**Q3：为什么 InternVL 要用 6B 的视觉塔？**
容量匹配 + 规模收益：LLM 是 7B~20B，300M 的 CLIP ViT-L 是信息瓶颈，视觉细节在塔内就被压缩失真；6B 塔能把 OCR / 计数 / 空间关系编码进表征，与 LLM 对齐时信息不丢。实验上 6B 塔全面超过 ViT-L 基线。代价是推理贵，所以配套动态分辨率 + pixel shuffle 压缩。小模型（≤8B）用 300M 塔即可，因为 LLM 容量本身小。

**Q4：动态分辨率 vs 固定分辨率？**
固定分辨率简单、batch 稳定、实现容易，但长宽比被破坏（形变）且高分辨率不可行。动态分辨率按内容决定实际输入尺寸：切块式（LLaVA-NeXT / InternVL：缩略图 + tile）和原生网格式（Qwen2-VL：28×28 网格 + M-RoPE 二维位置）。动态方案在文档、图表、任意长宽比图像上显著更好，代价是 batch 内 token 数不齐、工程复杂。

**Q5：InternVL 和 LLaVA 的区别？**
架构上：LLaVA 是"小塔（304M）+ MLP + 全 token 保留"；InternVL 是"大塔（300M/6B）+ pixel shuffle 压缩 + 动态分辨率"，token 策略完全不同。训练上：LLaVA 定义了投影对齐 → SFT 两阶段；InternVL-1 用 QLLaMA 三阶段，1.5 起改为 pixel shuffle + MLP + 大规模生成预训练。数据上：InternVL 的数据规模与配方（4K 训练、视频、交错数据）更接近工业级。可以理解为"同一个范式的两个不同代际的实现"。

**Q6：InternVL 的训练三阶段各自做什么？**
① 对比/生成对齐：把视觉特征拉进 LLM 嵌入空间（InternVL-1 用 QLLaMA + 对比损失，数据百万级）；② 生成式预训练：千万~亿级图文交错数据上做 next-token prediction（塔冻结，桥 + LLM 全训）；③ 指令微调：百万级 VQA / 对话 / OCR 指令数据对齐人类任务。可选第四阶段 RL（InternVL3 用 GRPO / 偏好优化提升推理与规划）。

**Q7：Q-Former 和 MLP projector 怎么选？**
Q-Former：固定 token 数（如 32），计算极省，适合桥接冻结塔 + 检索类任务；但学习式压缩有损、两阶段训练耦合、细节易丢。MLP projector：逐 patch 直通，信息保留最好、实现简单，但 token 多、计算贵，高分辨率场景必须配套压缩（pixel shuffle / merger）。当前主流趋势是 MLP 直通 + 结构/自适应压缩（InternVL、Qwen2.5-VL），Q-Former 路线逐渐退居检索 / 端侧。

**Q8：M-RoPE 是什么？为什么 Qwen2-VL 能不要 resampler？**
M-RoPE 是把一维 RoPE 推广到二维（图像 x-y 坐标）与三维（视频 + 时间）的位置编码：hidden 维度按轴分组，各轴独立旋转，patch 位置精确可算。因此任意长宽比 / 分辨率的 patch 都有明确相对位置，模型可以直接吃"原生分辨率特征"，不再需要 resampler 把不定长特征压成固定 token（Qwen-VL 1.x 时代 resampler 的首要动机就是"不定长 → 定长"）。resampler 解决"token 数不整齐"，M-RoPE 解决了"位置乱"，两者目的不同。

**Q9：InternVL-1 的 QLLaMA 和 BLIP-2 的 Q-Former 什么关系？**
同源：都是可学习 query + cross-attention 从视觉特征提炼信息。区别：QLLaMA 在每层把 query 与"投影后的 LLM 嵌入"一起过 self-attention（LLM 融合），让 query 直接对齐 LLM 嵌入空间，对齐质量更高；输出 64 个 256 维 query（BLIP-2 是 32 个 768 维）。InternVL-1.5 起 QLLaMA 被弃用，换 pixel shuffle + MLP——因为生成式预训练数据够多后，结构式压缩更简单高效。

**Q10：如何给 VLM 选型（作为工程师）？**
先定约束：显存（端侧 → MiniCPM-V / ≤3B；单卡 → 7B~8B；多卡 → 26B+）→ 再定分辨率需求（文档 / 图表 → 动态高分辨率：Qwen2.5-VL / InternVL / MiniCPM-V 2.6）→ 语言（中文优先 Qwen / InternVL）→ 功能（Agent / Grounding → Qwen2.5-VL、InternVL3）→ 协议（商用 → InternVL MIT 最省心）→ 最后用目标任务的公开 benchmark + 自建测试集实测对比，不要只看榜单。

---

## 十二、自我检验

- [ ] 能画出 VLM 三段式骨架并说清四个关键变量（塔 / 桥 / token / 训练阶段）
- [ ] 能写出 pixel shuffle 的 reshape 公式、token 数变化公式（$N' = N/s^2$），并说明"重排无损、线性降维有损"
- [ ] 能用数值例子讲清 448² tile 的 token 变化（1024 → 256）
- [ ] 能说清"为什么 6B 视觉塔"的三层理由（规模收益 / 容量匹配 / 配套压缩）
- [ ] 能对比 InternVL-1.5 动态分辨率与 LLaVA-NeXT anyres 的 4 点以上差异
- [ ] 能说出 InternVL 三阶段训练的每阶段目标、数据规模、冻结策略
- [ ] 能讲清 InternVL-1 → 1.5 → 2 → 2.5 → 3 的架构与训练演进主线
- [ ] 能说出 MoE upcycling 的原理（复制 expert + router，总参放大、激活不变）及 40B/76B 的例子
- [ ] 能默写出三大连接方式分类表（MLP / Q-Former / 原生）
- [ ] 能比较 4 条视觉 token 处理路线（全保留 / 固定压缩 / 自适应 / 动态分辨率）的优缺点
- [ ] 能解释 M-RoPE 并说清 Qwen2-VL 为什么可以去掉 resampler
- [ ] 能默写六大开源 VLM 对比大表（塔 / 桥 / 分辨率 / 上下文 / 参数 / 协议）
- [ ] 能说出 Qwen-VL、InternVL、MiniCPM-V、DeepSeek-VL 四家的定位差异
- [ ] 能按场景给出选型建议（端侧 / 中文 / OCR / Agent / 商用协议）
- [ ] 能区分 6 个常见误区
- [ ] 能完整回答 10 个高频面试问答

---

## 参考文献

1. [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) — 上海 AI Lab 等，2023
2. [InternVL1.5: Towards Professional Multimodal LLM across Levels of Vision-Language Understanding](https://arxiv.org/abs/2404.16821) — 2024
3. [InternVL2: Better than the Best, Experiencing the Boundaries of Vision-Language Model](https://arxiv.org/abs/2412.06671) — 2024
4. [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://arxiv.org/abs/2504.10479) — 2025
5. [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) — Liu et al., 2023
6. [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — 2024
7. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) — 2024
8. [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) — 2025
9. [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) — 2024
10. [PaliGemma: A 3B VLM with Transferable Generalist Capabilities](https://arxiv.org/abs/2407.07726) — Google, 2024
11. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta, 2024（含 Llama 3.2-Vision 架构）
12. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al., 2023
13. [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](https://arxiv.org/abs/2412.10302) — 2024
14. [ToMe: Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) — token merging 思想来源
