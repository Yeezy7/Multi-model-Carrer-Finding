# Adapter：瓶颈适配器（Bottleneck Adapter）

> 本模块索引见 [参数高效微调PEFT详解](参数高效微调PEFT详解.md)

## 一、定义与公式

### 1.1 一句话定义

Adapter（Houlsby et al., 2019，最早的 PEFT 之一）：在 Transformer 子层（Attention / FFN）之间**插入一个小型瓶颈网络**，冻结主干、只训练插入的小网络。其数学结构与自编码器同构：降维压缩 → 非线性 → 升维恢复。

### 1.2 Bottleneck 结构与公式

$$\text{Adapter}(x) = W_{\text{out}}\,\sigma\big(W_{\text{in}} x + b_{\text{in}}\big) + b_{\text{out}}, \qquad W_{\text{in}} \in \mathbb{R}^{r \times d},\; W_{\text{out}} \in \mathbb{R}^{d \times r}$$

- 先降维（$d \to r$，信息压缩），非线性激活 $\sigma$（ReLU / GELU），再升维（$r \to d$）；
- $r \ll d$（典型 $r = 32 \sim 128$，而 $d = 768 \sim 4096$）——中间的窄瓶颈就是"参数少"的来源。

**参数量推导**：两个投影矩阵加 bias：

$$\text{param} = \underbrace{r \times d + r}_{W_{\text{in}}} + \underbrace{d \times r + d}_{W_{\text{out}}} \approx 2dr$$

对比"直接插一层 $d \to d$ 全连接"（$d^2$ 参数）：瓶颈化后参数比约 $\dfrac{2dr}{d^2} = \dfrac{2r}{d}$。例：$d = 4096$、$r = 64$，一个 Adapter 约 53 万参数，占该层（1678 万）的 3.2%；作为对比，LoRA 同维度 $r=8$ 仅 0.39%。

### 1.3 串行 vs 并行公式

**串行 Adapter（Houlsby）**——插在子层之后：

$$y = x + \text{Adapter}\big(\text{SubLayer}(x)\big)$$

```text
x → Attn → Adapter → 残差+ → x'      （FFN 后同理）
```

**并行 Adapter（Parallel Adapter）**——与子层并行、输出相加：

$$y = \text{SubLayer}(x) + \text{Adapter}(x)$$

```text
x → [Attn 分支 + Adapter 分支] → 求和 → 残差
```

数学上注意：**LoRA 就是并行 Adapter 对线性层的特例**（$W_0 x + BA x$，其中 Adapter 退化为线性、无激活）。

### 1.4 残差连接与初始化

$\text{Adapter}(x) + x$ 残差 + **输出层零初始化**（`up.weight = 0`）保证：初始时 Adapter 输出为 0，模型前向与预训练完全一致——这与 LoRA 的 $B=0$ 是同一个设计哲学：**起点必须等于预训练模型**。

## 二、核心原理

### 2.1 为什么 Adapter 有效

与低秩假设同源：下游任务需要的"能力增量"维度很低，一个小瓶颈网络足以表达。三要素缺一不可：

1. **瓶颈 $d \to r \to d$**：把可学信息压缩到 $r$ 维，参数量 $O(dr)$ 而非 $O(d^2)$；
2. **非线性 $\sigma$**：保证 Adapter 不是线性子层（否则与 LoRA 等价），表达力更强；
3. **残差 + 零初始化**：初始恒等、训练稳定，主干冻结不被破坏。

### 2.2 串行 vs 并行取舍

| 维度 | 串行 Adapter | 并行 Adapter |
|------|-------------|-------------|
| 网络深度 | 每层 +2 层小网络，**深度增加** | 不增加深度 |
| 推理延迟 | 增加（多两次小矩阵乘，串行依赖） | 几乎无感 |
| 梯度路径 | 更长（穿过子层 + Adapter） | 更短（并行分支） |
| 与 LoRA 关系 | 独立插入 | 线性化后就是 LoRA |
| 训练稳定性 | 经典方案，稳定 | 与残差叠加更稳 |

### 2.3 放置位置与经典配置

| 配置 | 位置 | 特点 |
|------|------|------|
| Houlsby（原始） | Attention 后 + FFN 后各一个 | 效果最好，参数稍多 |
| Pfeiffer | 只在 FFN 后放一个 | 参数更少，跨任务迁移友好 |
| 只放 Attention 后 | 单 Adapter | 最少参数，效果略降 |

### 2.4 变体一览

| 变体 | 思路 |
|------|------|
| AdapterFusion | 多个任务 Adapter 并行，学一个融合权重 $\sum_i \lambda_i \text{Adapter}_i(x)$ |
| MAD-X | 跨语言迁移：语言 Adapter + 任务 Adapter 解耦 |
| Compacter | 用 Kronecker 积压缩 Adapter 权重，参数再降一个数量级 |
| Layer-Adaptive | 不同层放不同大小 Adapter，自动选择 |

## 三、源码实现

### 3.1 手写 AdapterLayer（含残差与零初始化）

```python
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """bottleneck 适配器：d → r → d，可选残差连接与零初始化"""

    def __init__(self, d_model, r=64, dropout=0.1, residual=True, zero_init=True):
        super().__init__()
        self.down = nn.Linear(d_model, r)          # 降维压缩
        self.up = nn.Linear(r, d_model)            # 升维恢复
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.residual = residual
        if zero_init:
            nn.init.zeros_(self.up.weight)         # 与 LoRA 的 B=0 同理：起点 = 恒等
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.act(self.down(x))                 # (..., d) -> (..., r)
        out = self.up(self.dropout(h))             # (..., r) -> (..., d)
        return x + out if self.residual else out

# 自检：零初始化 + 残差时，初始输出 == 输入（等于"没插入"）
torch.manual_seed(0)
ad = AdapterLayer(64, r=16, zero_init=True)
x = torch.randn(2, 10, 64)
print(torch.allclose(ad(x), x))                    # True
```

### 3.2 串行注入：包一层"子层 + Adapter"

```python
class SerialBlock(nn.Module):
    """串行 Adapter：子层输出 → Adapter → 残差（子层自身残差也在）"""

    def __init__(self, d, r=16):
        super().__init__()
        self.linear = nn.Linear(d, d)              # 代替原 q/k/v/o 线性层
        self.adapter = AdapterLayer(d, r=r, zero_init=True)

    def forward(self, x):
        h = self.linear(x)                         # 原子层
        h = x + h                                  # 子层残差
        return self.adapter(h)                     # Adapter 自带残差：h + Adapter(h)

# 等价的前向公式：y = x + Attn(x) + Adapter(x + Attn(x))
```

### 3.3 并行注入：与原子层并行相加

```python
class ParallelBlock(nn.Module):
    """并行 Adapter：y = Linear(x) + Adapter(x)"""

    def __init__(self, d, r=16):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.adapter = AdapterLayer(d, r=r, zero_init=True)

    def forward(self, x):
        return self.linear(x) + self.adapter(x)

# 对比：把 Adapter 换成无激活的 LoRA 分支，就是 LoRALinear —— LoRA ⊂ 并行 Adapter
```

### 3.4 端到端训练（注入 + 冻结 + 只训 Adapter）

```python
class TinyLM(nn.Module):
    """迷你语言模型：嵌入 + 单层自注意力"""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d); self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d); self.o_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x=None, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed(x if x is not None else input_ids)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        h = h + self.o_proj(attn @ v)
        return self.head(h)

def inject_serial(model, r=16):
    """把 q/k/v/o 四个线性层替换为"线性层 + 串行 Adapter"，并冻结全部基座"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(model, name, SerialBlock(child.in_features, r=r))
        else:
            inject_serial(child, r)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, AdapterLayer):            # 只解冻 Adapter 参数
            for p in m.parameters():
                p.requires_grad = True
    return model

torch.manual_seed(0)
model = TinyLM()
inject_serial(model, r=16)                         # 4 层各一个 Adapter
total = sum(p.numel() for p in model.parameters())
n_adapter = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"注入后全模型 {total} 参数，Adapter 可训练 {n_adapter}（{n_adapter/total:.1%}）")
# 注入后全模型 12736 参数，Adapter 可训练 4288（33.7%）（玩具模型；7B 实际约 0.5%）

x = torch.randint(0, 64, (16, 10))
y = torch.randint(0, 64, (16, 10))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
crit = nn.CrossEntropyLoss()
model.train()
for step in range(100):
    opt.zero_grad()
    loss = crit(model(x).reshape(-1, 64), y.reshape(-1))
    loss.backward(); opt.step()
print(f"训练后 loss = {loss.item():.3f}")          # 从 ~4.43 降到 ~2.57
```

### 3.5 peft 库现状说明

> peft 库在 **0.14 版本移除了旧版 Houlsby Adapter 的 `AdapterConfig`**，现在推荐：① 用 LoRA（`LoraConfig`，效果等价且可合并）；② 按上面 3.1-3.4 的方式手写注入（结构完全可控）；③ 或使用专门库 `llm-adapter`。手写注入与 peft 的 LoRA 流程（`get_peft_model` + 冻结）思路完全一致，可以对照阅读 [LoRA](LoRA.md) 3.4 节。

## 四、参数与显存账本

### 4.1 参数量计算（LLaMA-7B 场景，$d = 4096$，$r = 64$）

每个 Adapter：$2 \times 4096 \times 64 \approx 52.4$ 万参数（含 bias 约 52.8 万）。

| 配置 | 每层 Adapter 数 | 32 层总量 | 占 7B 比例 |
|------|----------------|-----------|-----------|
| Pfeiffer（仅 FFN 后） | 1 | 1690 万 | 0.25% |
| Houlsby（Attn 后 + FFN 后） | 2 | 3380 万 | **0.50%** |
| 对照：LoRA r=8 全注入 | — | 2000 万 | 0.30% |

### 4.2 显存账本（7B，Houlsby 配置）

| 项目 | 数值 |
|------|------|
| 基座权重（bf16，冻结） | 14GB |
| Adapter 参数（bf16） | 3380 万 × 2B ≈ 68MB |
| 梯度 + Adam 状态（fp32） | 3380 万 × 4B × 3 ≈ 406MB |
| 训练峰值（估算） | **~20-32GB**（略高于同配置 LoRA） |
| 推理开销 | 串行每层多 2 次小矩阵乘，延迟明显增加 |

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 结构简单直观，训练稳定（残差 + 零初始化） | 串行 Adapter **增加网络深度与推理延迟**，且不能"折叠"回原权重（非线性） |
| 参数量 0.5%~5%，比 LoRA 表达空间大（带非线性） | 通常参数多于 LoRA（约 2~10 倍） |
| 天然模块化：多个任务 Adapter 可并存/组合（Fusion） | 与量化/编译工具兼容性差（新增结构需定制） |
| 有大量成熟变体（MAD-X、Compacter 等） | peft 主库已不维护旧版 Adapter |

## 六、与同类方法对比（重点 vs LoRA）

| 维度 | Adapter | LoRA |
|------|---------|------|
| 数学形式 | 非线性瓶颈：$W_{out}\,\sigma(W_{in}x)$ | 线性低秩：$W_0x + sBAx$ |
| 可训练参数（7B） | 0.5%~5% | 0.01%~1% |
| 推理开销 | 串行增加延迟；并行几乎无感 | **可合并，0 开销** |
| 能否合并回原权重 | **不能**（非线性激活不可折叠） | **能**（线性可精确合并） |
| 结构改动 | 插入新模块（深度 +2/层） | 只改前向公式 |
| 效果 | 与 LoRA 相当 | 与全参差距更小 |
| 适合场景 | 老框架兼容、研究变体 | 生产部署、多任务 |

> **面试记忆点**：LoRA 与 Adapter 的核心分野在**线性 vs 非线性**——线性可合并（零推理开销），非线性不可合并（有推理开销）。这是 LoRA 成为主流的关键工程优势。

## 七、高频面试问答

**Q1：Adapter 为什么参数量少？**
瓶颈结构 $d \to r \to d$ 把两个投影矩阵都压在 $r$ 维上：参数 $2dr$ 而非 $d^2$。$r \ll d$ 时参数比 $\approx 2r/d$。

**Q2：串行和并行 Adapter 的区别？**
串行插在子层之后，增加网络深度、梯度路径长、推理延迟高；并行与子层并列相加，不增深度、梯度短、延迟低。并行 Adapter 线性化后就是 LoRA。

**Q3：为什么需要残差连接和零初始化？**
保证初始时 Adapter 输出为 0（`up.weight=0`），模型起点严格等于预训练模型，训练稳定。与 LoRA 的 $B=0$ 同一哲学。

**Q4：Adapter 为什么不能像 LoRA 那样合并权重？**
LoRA 是线性低秩，$(W_0 + BA)x = W_0x + BAx$ 可精确折叠；Adapter 含非线性激活 $\sigma$，$W_{out}\,\sigma(W_{in}x)$ 无法分解回原线性层，只能保留分支 → 串行有额外推理延迟。

**Q5：Adapter 和 LoRA 谁更好？**
工程上 LoRA：参数更少、可合并、兼容量化。效果上两者相当；Adapter 的优点是模块化与变体生态（Fusion/MAD-X），且非线性提供更大表达空间，多任务解耦场景仍有价值。

**Q6：AdapterFusion 是什么？**
训练阶段各任务一个独立 Adapter，推理时用可学习权重 $\sum_i \lambda_i \text{Adapter}_i(x)$ 融合，多任务共享主干 + 参数级组合，不需要重新训练。

**Q7：Adapter 会不会破坏预训练能力？**
不会。主干完全冻结，Adapter 只在特定位置加增量，且零初始化保证起点不变；比全参微调抗遗忘，但比 LoRA 的"低秩约束"更自由——任务数据偏移大时 Adapter 表现更稳（有非线性兜底）。

## 八、自我检验

- [ ] 能写出 bottleneck 公式与参数量推导（$2dr \ll d^2$）
- [ ] 能画出串行 vs 并行的结构图并讲清各自延迟/深度/梯度差异
- [ ] 能解释残差 + 零初始化与 LoRA 的 B=0 是同一设计哲学
- [ ] 能说出 LoRA ⊂ 并行 Adapter 的数学关系（线性化）
- [ ] 能讲清"Adapter 不能合并、LoRA 能合并"的根本原因是线性 vs 非线性
- [ ] 能写出 AdapterLayer + 串行/并行注入 + 冻结基座的完整代码
- [ ] 能背出 7B 模型的 Adapter 参数量账本（Houlsby ~0.5%）与显存账本
- [ ] 知道 peft 已移除旧版 AdapterConfig，并会手写注入代替
- [ ] 能列举 AdapterFusion / MAD-X / Compacter 等变体的思路
- [ ] 能回答 7 个面试追问
