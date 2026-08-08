# 主流 MoE 模型对比：从 GShard 到 Qwen3

> 本模块索引见 [MoE详解](MoE详解.md)

## 一、定义与公式

### 1.1 一个公式看懂所有 MoE

现代 MoE 层可以统一写成（以 DeepSeek/Qwen3 的"共享 + 路由"结构为例）：

$$y = \sum_{j=1}^{k} \hat{g}_{i_j}(x)\, f_{i_j}(x) \;+\; \sum_{s \in \mathcal{S}_{\text{shared}}} f_s(x)$$

- $f_{i_1}, \dots, f_{i_k}$：每 token 由路由器从 $N$ 个路由专家中选出的 top-k 专家；
- $\mathcal{S}_{\text{shared}}$：共享专家（Mixtral 没有；DeepSeek-V2/V3 有 1 个；Qwen3 有 4 个），每个 token 必走；
- 模型之间的差异集中在三处：**专家怎么切（fine-grained？）**、**路由怎么打分（softmax/sigmoid/top-k 多少）**、**负载怎么均衡（aux loss / capacity / bias）**。

### 1.2 演进谱系总览

| 模型 | 年份 | 总参 / 激活 | 每层专家 | top-k | 标志性贡献 |
| --- | --- | --- | --- | --- | --- |
| GShard | 2020 | 600B / ≈9.4B | 128 | 2 | **首个工业级 MoE**：top-2 路由、aux loss 负载均衡、专家切分到 2048 卡 |
| Switch Transformer | 2021 | 1.57T / ≈12B | 2048 | 1 | top-1 简单路由，**抛弃 aux loss**，改 capacity factor + token 丢弃；预训练提速 4~7× |
| GLaM | 2021 | 1.2T / 96.3B | 64 | 2 | GPT-3 规模 MoE：少样本全面超 GPT-3，训练能耗仅 1/3 |
| Mixtral 8x7B | 2023 | 46.7B / 12.9B | 8 | 2 | **开源里程碑**：分组 top-2、无 aux loss，质量 ≈ Llama2-70B |
| DeepSeek-MoE | 2024 | 16.4B / 2.8B | 64+2 共享 | 2 | 提出 **fine-grained experts + shared experts**，解决专家重复化 |
| DeepSeek-V2 | 2024 | 236B / 21B | 160+1 共享 | 6 | fine-grained 规模化 + MLA 注意力，训练成本砍到 1/3 |
| DeepSeek-V3 | 2024 | 671B / 37B | 256+1 共享 | 8 | **aux-loss-free**（sigmoid 门控 + bias）+ FP8 + MTP，逼近 Llama-3.1-405B |
| Qwen3-MoE | 2025 | 30.5B / 3.3B 与 235B / 22B | 128+4 共享 | 8 | "总参-A激活"命名，30B-A3B 成端侧新标杆 |

> 注：GShard 论文发表于 2020 年（arXiv:2006.16668，部分资料记为 2017 系指其框架早期形态）；GLaM 论文 2021 年 12 月发布（arXiv:2112.06931）。

### 1.3 谱系脉络的三条主线

1. **容量线**：600B（GShard）→ 1.6T（Switch）→ 671B（V3）→ 235B-A22B（Qwen3），总参一路狂涨，激活参数却稳定在 3B~96B——"容量与算力解耦"越走越远；
2. **均衡线**：aux loss（GShard/Switch 早期）→ 抛弃 aux loss（Switch 用 capacity、Mixtral 用分组 top-2、DeepSeek-V3 用 bias）——**aux loss 从"标配"变成"可选"**；
3. **开源线**：Mixtral（2023 开源王者）→ DeepSeek 系列（2024~2025 开源 SOTA）→ Qwen3（2025 双版本矩阵），开源 MoE 从"可玩"走向"可部署"。

### 1.4 读懂模型命名：8x7B、671B、30B-A3B 各是什么意思

| 命名 | 读法 | 含义 | 易错点 |
| --- | --- | --- | --- |
| Mixtral 8x7B | "8 乘 7B" | 每层 8 个专家 FFN，单专家规模约 7B 级 | **不是 56B**：总参只有 46.7B（专家层分摊进 32 层 + 共享 attention），每 token 激活 12.9B |
| DeepSeek-V3 (671B) | 总参 671B | 只写总参，激活参数（37B）要看技术报告 | 671B 不是激活参数；激活仅 5.5% |
| Qwen3-30B-A3B | "30B-Active 3B" | 总参 30.5B、每 token 激活 3.3B（A = active） | A 不是 attention；"A3B" 是激活量不是专家数 |
| Qwen3-235B-A22B | 同上 | 总参 235B、激活 22B，128 专家 top-8 | 同总参下激活参数是 30B-A3B 的约 7 倍 |

命名法演进的本质：Mixtral 时代用"专家数 × 单专家规模"（8x7B）描述结构，DeepSeek/Qwen3 时代直接用"总参-A激活"（671B/37B、30B-A3B）描述**部署成本与单 token 成本**——因为后者才是工程决策真正需要的两个数。

## 二、核心原理：四大关键技术

### 2.1 各代模型详细设计表

| 模型 | 总参数 | 激活参数 | 路由专家 | top-k | shared | 负载均衡方法 | 代表论文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GShard | 600B | ≈9.4B | 128 | 2 | 无 | **aux loss（首次提出）** + capacity 1.25~2 + 余量路由 | GShard (2020) |
| Switch-T-XXL | 1.57T | ≈12B | 2048 | 1 | 无 | 无 aux loss，capacity factor + **超量 token 直接丢弃** | Switch Transformer (2021) |
| GLaM | 1.2T | 96.3B | 64 | 2 | 无 | aux loss + 随机路由兜底 | GLaM (2021) |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | 无 | **分组 top-2**（每 2 个 token 一组选组内 top-2），无 aux loss | Mixtral of Experts (2023) |
| DeepSeek-MoE | 16.4B | 2.8B | 64 | 2 | **2** | aux loss + 互补路由 | DeepSeekMoE (2024.1) |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | **1** | aux loss + bias + MLA | DeepSeek-V2 (2024.5) |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | **1** | **aux-loss-free**：sigmoid 门控 + 动态 bias | DeepSeek-V3 (2024.12) |
| Qwen3-30B-A3B | 30.5B | 3.3B | 128 | 8 | **4** | sigmoid 门控 + bias（同 V3 路线） | Qwen3 Tech Report (2025) |
| Qwen3-235B-A22B | 235B | 22B | 128 | 8 | **4** | sigmoid 门控 + bias | Qwen3 Tech Report (2025) |

（aux loss、capacity、bias 的数学细节见 [路由机制与负载均衡](路由机制与负载均衡.md) 第四节。）

### 2.2 fine-grained experts：把专家切细

**定义**：在"每 token 激活参数总量"不变的前提下，把专家数从 $N$ 个"大专家"切成 $n \cdot N$ 个"小专家"，同时把 top-k 从 $k$ 提高到 $n \cdot k$。

设单个专家参数量 $p_e$、激活专家数 $k$，激活参数 $k \cdot p_e$ 保持不变：

- Mixtral：$N=8$，每专家 FFN 参数量大，$k=2$ → 组合数 $C(8,2) = 28$；
- DeepSeek-V3：$N=256$，每专家只是"中等 FFN"，$k=8$ → 组合数 $C(256,8) \approx 4.6 \times 10^{13}$。

**为什么更灵活**：
1. **组合空间爆炸**：token 的表示 = 被激活专家的"组合"，组合数随 $N$ 指数级增长——细粒度专家可以用"更多专家的拼图"更精细地组合知识；
2. **专家更专一**：小专家学到的模式更单一（"一个专家=一种风格/一种语法结构"），DeepSeek 论文发现大专家内部混杂多种知识，切细后分化更干净；
3. **均衡粒度更细**：token 分散到更多专家，单专家的负载波动更小，负载均衡调节更平滑；
4. **代价**：专家间通信模式更复杂（每 token 的 top-k 更分散），且"专家太少 token"导致训练不充分——需要配合共享专家（见 2.3）。

### 2.3 shared experts：每个 token 必走的专家

**定义**：$M$ 个不参与路由的专家（DeepSeek-MoE 2 个、V2/V3 1 个、Qwen3 4 个），每个 token 无条件经过，输出直接与路由专家的加权输出相加（见 1.1 公式第二项）。

**为什么需要**（DeepSeek-MoE 论文的核心发现）：
1. **专家重复化是 MoE 的通病**：论文分析了路由专家的激活模式，发现大量专家在学高度相似的知识（公共语法、通用语义），重复率远高于预期——白白浪费容量；
2. **公共知识交给 shared**：所有 token 共享的知识（语言共性、任务通用能力）沉淀到 shared 专家，路由专家只负责个性化/领域化知识 → 专家分化度显著提升；
3. **与 fine-grained 互补**：专家切得越细，公共知识被重复存放的浪费越严重，shared 恰好把"公共部分"剥离出来——两者是 DeepSeek 架构的一对组合拳；
4. **推理时零路由成本**：shared 不经过路由决策，计算是确定性的，反而稳定了负载。

**类比**：shared 专家是"常任委员"（每次都参会），路由专家是"按议题抽调的专家"（谁合适谁来）。

### 2.4 DeepSeek-V3 的 aux-loss-free 负载均衡

传统 aux loss 的缺陷：均衡信号与任务梯度纠缠，路由器"被迫均匀"而牺牲路由质量（公式与推导见 [路由机制与负载均衡](路由机制与负载均衡.md) 4.2）。V3 的替代方案分三步：

1. **sigmoid 门控（专家间无竞争）**：

$$s_{i,t} = \mathrm{Sigmoid}(x_t^{\top} w_i)$$

与 softmax 不同，每个专家的分数只由自身决定、不必和为 1——加 bias 只平移"选中边界"，不会像 softmax 那样被归一化抵消；

2. **路由分数 = 门控 + 专家 bias**，选 top-k：

$$r_{i,t} = s_{i,t} + b_i, \qquad \mathcal{S}_t = \mathrm{TopK}\big(\{r_{i,t}\}_{i=1}^{N},\; k\big)$$

bias 只参与"选谁"，不参与"加权"（加权仍用 sigmoid 概率归一化）——内容打分与均衡信号**完全解耦**；

3. **bias 动态调节（负反馈回路）**：

$$b_i \leftarrow b_i - \gamma \cdot \mathrm{sign}\Big(\mathrm{Load}_i - \frac{T \cdot k}{N}\Big), \qquad \mathrm{Load}_i = \big|\{t : i \in \mathcal{S}_t\}\big|$$

超载专家（Load 高于均值）bias 被压低 → 被选中概率下降；欠载专家反之。bias 的梯度**不流入主损失**（解耦更新），所以不会扭曲任务训练，也无需调 aux loss 的 $\alpha$。

### 2.5 MoE 层放深层还是浅层

| 模型 | 放置方式 |
| --- | --- |
| GShard | 除**首尾各 2 层**外全部为 MoE |
| Mixtral | **全部 32 层** MoE |
| DeepSeek-V3 | **前 3 层稠密**，其余 58 层 MoE |
| Qwen3-MoE | 前 3 层稠密，其余层 MoE |

**经验规律：浅层放稠密、深层放专家**：
1. **浅层职责通用**：前几层主要负责语法、位置、局部结构等所有 token 共用的低阶特征，共享一个 FFN 就够，放专家收益小；
2. **深层知识专门化**：深层 token 表示语义分化大（不同领域、不同任务），专家在这里更容易分化出"领域分工"，收益最大；
3. **研究佐证**：Switch 论文对比过不同放置方式，倾向把专家放在更深的位置；"稠密 → MoE 转换"研究（LLaMA-MoE 等）同样发现**浅层转专家收益低、深层转专家收益高**（注：Meta 未发布官方 LLaMA 3.3 MoE，该结论来自转换研究与实践经验）；
4. **兜底解释**：浅层若放 MoE，所有 token 都要经受路由噪声，而浅层表示还很粗糙、路由分数噪声大——先让稠密层"打好底"，再让专家"分好工"。

## 三、源码实现：aux-loss-free 均衡模拟（可运行）

模拟 DeepSeek-V3 的"负载 → bias 负反馈"机制：先用 sigmoid 门控随机路由制造天然不均衡，再观察动态 bias 如何把负载拉回均匀：

```python
import torch

torch.manual_seed(0)

# ---------- 配置 ----------
T, N, K = 512, 8, 2            # 512 个 token、8 个专家、top-2
x = torch.randn(T, 32)         # token 表示（模拟）
Wg = torch.randn(32, N) * 0.3  # 路由器权重
bias = torch.zeros(N)          # 专家 bias，从 0 开始

def route():
    g = torch.sigmoid(x @ Wg)              # 逐专家 sigmoid 门控（无 softmax 竞争）
    return torch.topk(g + bias, K, dim=-1).indices   # 路由分数 = 门控 + bias

def load_stat(tag):
    cnt = torch.bincount(route().flatten(), minlength=N).float()
    print(f"{tag}: 负载 {cnt.int().tolist()}  max/avg = {cnt.max().item()/cnt.mean().item():.2f}")
    return cnt

cnt0 = load_stat("初始（bias=0）")
# 输出示例: 初始（bias=0）: 负载 [115, 126, 116, 154, 117, 131, 114, 151]  max/avg = 1.20

# ---------- bias 负反馈：超载专家 bias 下调 ----------
alpha = 0.05
for step in range(300):
    cnt = torch.bincount(route().flatten(), minlength=N).float()
    bias = bias - alpha * (cnt - cnt.mean()) / T   # 负载高于均值 → bias 变小
    # 真实 V3 用 sign 式更新（见 2.4），这里是比例式变体，收敛更平滑

cnt1 = load_stat("300 步后（bias 负反馈）")
# 输出示例: 300 步后（bias 负反馈）: 负载 [128, 128, 128, 128, 128, 128, 128, 128]  max/avg = 1.00
#          负反馈把 8 个专家完美拉平（512 token × top-2 ÷ 8 = 128）

print("bias 终值:", [round(b, 3) for b in bias.tolist()])
# 输出示例: bias 终值: [0.013, 0.007, 0.022, -0.034, 0.008, -0.0, 0.021, -0.037]
#          初始负载最高的专家 3（154）与 7（151）的 bias 被压到负值，其余专家为正
```

**输出解读**：初始路由因 $x$ 与 $W_g$ 的随机偏好天然不均衡（max/avg ≈ 1.20）；300 步负反馈后负载收敛到完美均匀（max/avg = 1.00），且 bias 的正负与初始偏好完全对应——这就是 DeepSeek-V3"无 aux loss 也能均衡"的可运行证据：**均衡不靠损失函数，靠的是 bias 的显式控制回路**。

## 四、深入分析

### 4.1 多模态 MoE：Qwen3-VL 与 DeepSeek-VL2

**Qwen3-VL-30B-A3B（2025）**：
- 结构：视觉塔（ViT，约 0.68B 参数）把图像/视频转成视觉 token + MoE 语言模型（30.5B 总参、3.3B 激活、128 专家 top-8 + 4 shared）；
- 视觉 token 与文本 token 一起走 MoE 路由；视觉 token 量大，稀疏激活的算力优势被放大；
- 与 Qwen3-30B-A3B 共享同一个 MoE 底座——多模态能力主要由视觉塔 + 投影层提供，训练成本可控。

**DeepSeek-VL2（2024）**：
- 总参 27.5B、激活约 4.5B，底座继承 DeepSeek-MoE 的"细粒度专家 + 共享专家"架构（sigmoid 门控 + bias 均衡、无 aux loss）；
- 视觉侧：SigLIP + SAM 混合视觉编码器 + Pixel Shuffle（视觉 token 压缩 4 倍）；
- **创新点：视觉专家（vision experts）**——为视觉 token 单独配备的专家子集。多模态 token 差异大，若视觉/文本 token 共享同一批路由专家，会互相挤占；视觉专家让两类 token 各走各路，又通过 shared 专家保留公共能力。

**共性挑战**（详见 [MoE详解](MoE详解.md) 8.3）：模态不平衡（视觉 token 远多于文本）、局部过载（同图 patch 偏好同一专家）、SFT 阶段路由漂移。

**两者横向对比**：

| 对比项 | Qwen3-VL-30B-A3B | DeepSeek-VL2 |
| --- | --- | --- |
| 总参 / 激活 | 30.5B / 3.3B | 27.5B / ≈4.5B |
| 视觉塔 | 单 ViT（约 0.68B） | SigLIP + SAM 混合编码器 |
| MoE 结构 | 128 路由专家 + 4 shared，top-8 | 细粒度专家 + shared，sigmoid 门控无 aux loss |
| 视觉专用设计 | 视觉 token 走统一路由 + 超长上下文 | **视觉专家**（vision experts）+ Pixel Shuffle 压缩 4 倍 |
| 典型定位 | 端侧可用的小激活多模态旗舰 | 高分辨率文档、图表、OCR |

### 4.2 效果对比：相对定位

| 模型 | 规模 | 关键 benchmark | 相对定位 |
| --- | --- | --- | --- |
| Switch-T-XXL | 1.57T / ≈12B | 预训练同算力提速 4~7× | vs T5-XXL 质量持平、算力大省 |
| GLaM | 1.2T / 96.3B | 少样本 29/42 任务超 GPT-3 | vs GPT-3（175B 稠密），训练能耗 1/3 |
| Mixtral 8x7B | 46.7B / 12.9B | MMLU 70.6% | ≈ Llama2-70B，开源里程碑 |
| DeepSeek-V2 | 236B / 21B | MMLU 78.5% | 报告称综合接近 GPT-4-Turbo 水平 |
| DeepSeek-V3 | 671B / 37B | MMLU 88.5%，AIME 2024 大幅领先 | ≈ Llama-3.1-405B，训练成本约其 1/10 |
| Qwen3-30B-A3B | 30.5B / 3.3B | MMLU 82.5% | 3B 级激活打平 32B 级稠密 |
| Qwen3-235B-A22B | 235B / 22B | MMLU 89.5% | 22B 激活逼近 235B 稠密水平 |

（数值均取各论文/技术报告报告值；"≈ 位置"为相对说法。）

**规律**：激活参数相同时 MoE 显著优于稠密；总参越大优势越明显（容量红利），但收益递减且对均衡、通信的要求更高。

### 4.3 训练成本账本：MoE 省钱的量化证据

| 模型 | 训练成本 | 对比对象 |
| --- | --- | --- |
| Switch-T-XXL | 同 T5-XXL 的算力、4~7× 预训练提速 | vs T5-XXL（11B 稠密） |
| GLaM | ≈ GPT-3 的 1/3 训练能耗 | vs GPT-3（175B 稠密） |
| Mixtral 8x7B | 每 token 算力 ≈ 13B 稠密 | 47B 的容量、13B 的成本 |
| DeepSeek-V3 | 2.788M H800 GPU-hours | ≈ Llama-3.1-405B（30.8M H100 GPU-hours）的 1/10~1/11，综合效果持平 |
| Qwen3-30B-A3B | 每 token 算力 ≈ 3.3B 稠密 | 对标 32B 级稠密效果 |

**两点解读**：
1. **训练省的是 FLOPs 不是显存**：V3 仍要 2048 卡 × 数月的显存基础设施，省的是"等量的算力买到更多模型能力"；
2. **激活参数越小的模型性价比越高**：30B-A3B 用 3.3B 的每 token 算力达到 32B 稠密水平——这正是 2025 年"小激活大容量"成为端侧主流的原因。

## 五、优缺点总结

### 5.1 MoE 家族的统一优缺点

| 优点 | 缺点 |
| --- | --- |
| 容量-算力解耦：47B/13B、671B/37B 这类"大容量小算力" | 全部专家权重常驻显存，部署门槛高 |
| 同算力下质量显著优于稠密（Switch 4~7× 效率） | 负载均衡机制复杂，均衡与质量互相牵制 |
| 专家可独立扩展（modular 式增量） | 通信开销随 top-k 线性涨 |
| 开源生态成熟（Mixtral → DeepSeek → Qwen3） | 微调易死专家、路由漂移，调优成本高 |

### 5.2 各代模型解决的核心矛盾

| 模型 | 解决什么 | 引入什么代价 |
| --- | --- | --- |
| GShard | 把 MoE 推到工业规模 | aux loss 与任务竞争、容量调参 |
| Switch | 简化路由（top-1） | 弃 token 丢信息、质量天花板 |
| GLaM | 验证 MoE 能对标 GPT-3 | 激活参数过大（96.3B）性价比一般 |
| Mixtral | 开源、不丢 token、无需 aux loss | 8 专家组合空间小，专家易重复 |
| DeepSeek-MoE | 专家重复化问题（fine-grained + shared） | 专家多、均衡调节更精细但更复杂 |
| DeepSeek-V3 | aux loss 三宗罪（sigmoid + bias + FP8 + MTP） | 训练基建复杂度极高 |
| Qwen3-MoE | 端侧可行：30B-A3B 双版本矩阵 | 小专家训练不充分风险 |

## 六、与同类对比

### 6.1 显存与部署对比

| 模型 | 总参 | 激活参 | FP16 权重 | FP8/INT4 | 部署方案 |
| --- | --- | --- | --- | --- | --- |
| Mixtral 8x7B | 46.7B | 12.9B | ≈ 94GB | INT4 ≈ 23GB | FP16 需 2×80GB；INT4 单卡 80GB 可跑 |
| DeepSeek-V2 | 236B | 21B | ≈ 472GB | FP8 ≈ 236GB | 8×80GB（FP8）轻松装下 |
| DeepSeek-V3 | 671B | 37B | ≈ 1.34TB | FP8 ≈ 671GB | 官方称 8×H800 单节点极限部署；稳妥 8×96GB 或 16 卡 |
| Qwen3-30B-A3B | 30.5B | 3.3B | ≈ 61GB | INT4 ≈ 16GB | FP16 单卡 80GB；INT4 单卡 24GB 也能跑 |
| Qwen3-235B-A22B | 235B | 22B | ≈ 470GB | FP8 ≈ 235GB | 8×80GB（FP8） |

**部署规律**：显存只看总参 × 精度；每 token 推理成本只看激活参。所以"总参-A激活"命名（30B-A3B、235B-A22B）同时给出了**部署成本（总参）和单 token 成本（激活）**两个关键数字。

### 6.2 常见误区

| # | 误区 | 事实 |
| --- | --- | --- |
| 1 | "MoE 推理省显存" | **错**。全部专家权重必须常驻（94GB 就是 94GB），省的是算力（FLOPs），不是显存 |
| 2 | "MoE 参数少" | **错**。MoE 参数更多（Mixtral 47B vs 稠密 13B），只是激活参数少（13B） |
| 3 | "MoE 一定比稠密快" | **错**。吞吐（大 batch）快，单请求延迟（小 batch）因两轮 all-to-all 反而可能更慢 |
| 4 | "专家越多越好 / top-k 越大越好" | **错**。通信量 ∝ top-k 线性涨；专家太多则单专家训练不充分、均衡更难 |
| 5 | "各专家都在学不同的东西" | **错**。专家天然高度重复（DeepSeek-MoE 实测），才需要 fine-grained + shared 对症下药 |

## 七、高频面试问答

**Q1：Switch Transformer 和 Mixtral 的区别？**
Switch 是 top-1 + capacity 机制（超量 token 直接丢弃）且不使用 aux loss；Mixtral 是 top-2（分组 top-2，每 2 个 token 一组选组内 top-2）保证均衡、不丢 token，8 个专家规模小。Switch 证明了"简单路由也能训超大 MoE"，Mixtral 证明了"不丢 token 质量更好且开源可行"。

**Q2：shared expert 有什么用？**
捕捉所有 token 共享的公共知识（语法、通用语义），把公共能力从路由专家中剥离，避免专家重复学习同一模式；同时让路由专家更专注个性化知识。与 fine-grained 是一对组合拳。

**Q3：fine-grained 专家的好处？**
激活参数不变的情况下，专家数变多、每个专家更"专一"：① 组合空间指数级增长（C(8,2)=28 vs C(256,8)≈4.6×10¹³）；② 专家分化更干净、重复度更低；③ 负载均衡粒度更细。代价是需要 shared 专家兜底公共知识、通信模式更复杂。

**Q4：DeepSeek-V3 怎么解决负载不均衡？**
aux-loss-free 三件套：sigmoid 门控（专家间无竞争，bias 可独立平移）→ 路由分数 = 门控 + 专家 bias → bias 按负载做负反馈更新（超载降、欠载升），且 bias 梯度不流入主损失。均衡从"损失函数"变成"显式控制回路"，无需调 α。

**Q5：sigmoid 门控 vs softmax 门控？**
softmax 归一化后专家间竞争（一个升全体降），bias 会被归一化抵消；sigmoid 逐专家独立、和为 1 不要求，加 bias 只平移选择边界，语义不被扭曲——这是 DeepSeek 系列能免 aux loss 的数学前提。

**Q6：MoE 层放深层还是浅层？为什么？**
实践上浅层放稠密（DeepSeek-V3 前 3 层、GShard 首尾各 2 层稠密），深层放专家。浅层负责通用低阶特征、路由噪声大收益小；深层知识专门化，专家分化收益最大。

**Q7：多模态 MoE 与纯文本 MoE 的差别？**
① 视觉 token 量大，稀疏收益更明显；② 模态不平衡：视觉 token 挤占专家空间，均衡要考虑模态维度；③ 同图 patch 语义相关，易局部过载；④ DeepSeek-VL2 为此设计了视觉专家（vision experts），视觉/文本 token 各走各路。

**Q8：为什么 2023 年之前 MoE 没像现在这么火？**
基建三座大山：① 大规模 all-to-all 通信对互联带宽要求高（NVLink/IB 普及前跨卡通信是灾难）；② 负载均衡机制不成熟（aux loss 调参难、容量管理复杂）；③ 显存：所有专家常驻，大 MoE 必须多卡。Mixtral（开源 + 无 aux loss）与 DeepSeek-V3（免调参均衡 + FP8）把门槛降下来后，MoE 才进入主流。

## 八、自我检验

- [ ] 能按时间线说出 8 个模型的演进：GShard → Switch → GLaM → Mixtral → DeepSeek-MoE → V2 → V3 → Qwen3
- [ ] 能默写三个关键数字：Mixtral 47B/13B、DeepSeek-V3 671B/37B、Qwen3 30.5B/3.3B
- [ ] 能说清 fine-grained 与 shared experts 为什么是一对组合拳
- [ ] 能写出 aux-loss-free 的公式：sigmoid 门控 + bias + 负反馈更新
- [ ] 能解释 sigmoid 门控为什么可以加 bias、softmax 为什么不行
- [ ] 能说出 MoE 层放置经验：浅层稠密、深层专家（DeepSeek-V3 前 3 层）
- [ ] 能复述 5 条常见误区（省显存、参数少、一定快、越多越好、各学各的）
- [ ] 能回答 8 个面试追问
