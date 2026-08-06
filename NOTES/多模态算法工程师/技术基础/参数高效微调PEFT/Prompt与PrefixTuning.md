# Prompt Tuning 与 Prefix Tuning：软提示家族

> 本模块索引见 [参数高效微调PEFT详解](参数高效微调PEFT详解.md)

## 一、定义与公式

### 1.1 离散 Prompt vs 连续 Prompt

- **离散 Prompt（硬提示）**：手工设计文本模板（如 "Translate to French: ..."），在 token 空间搜索。不可微（token 是离散的）、效果不稳定、每任务要重新设计；
- **连续 Prompt（软提示，soft prompt）**：把 prompt 当作**可学习的连续向量**，直接在嵌入空间中优化，完全可微、不占用词表。

### 1.2 Prompt Tuning（Lester et al., 2021）：输入侧拼接

在输入 token 的嵌入序列前拼接 $l$ 个可学习向量：

$$x' = [\,P;\ E(x)\,], \qquad P \in \mathbb{R}^{l \times d}$$

- $P$：可学习的 soft prompt 矩阵（$l$ 个虚拟 token，每个 $d$ 维）；
- $E(x)$：输入 $x$ 的嵌入（原 token 部分）；
- 模型其余部分**全部冻结**，只训练 $P$（可选训练下游分类头）；
- 参数量：$l \times d$。例：$l=100$、$d=4096$ → 0.41M，约 7B 模型的 **0.005%**。

特点：参数最少、实现最简单；但**只在输入层注入信息**，深层感知不到任务信号，表达力有限。

### 1.3 Prefix Tuning（Li & Liang, 2021）：每层 K/V 拼接

在每个 Transformer 层的 attention 里，为 K 和 V 各拼接一组可学习前缀：

$$[\text{K}_i;\ P_i^K], \qquad [\text{V}_i;\ P_i^V], \qquad i = 1, \dots, L$$

- $P_i^K, P_i^V \in \mathbb{R}^{l \times d}$：第 $i$ 层的前缀（$l$ 个虚拟 token 的 K/V）；
- 前缀**不经过输入嵌入**，直接注入每层 attention 的注意力计算；
- 训练早期用一层 MLP 重参数化生成前缀参数（训练稳定后去掉，见 2.3）；
- 参数量：$2 \times l \times d \times L$（$L$ 为层数）。例：LLaMA-7B 32 层、$l=20$、$d=4096$ → 5.24M，约 **0.08%**；
- **代价：序列变长**——每层 KV cache 多 $l$ 个 token，推理延迟与显存上升。

### 1.4 P-Tuning v1 / v2

**P-Tuning v1（Liu et al., 2021）**——输入侧 soft prompt，但用 **LSTM/MLP 生成** prompt 向量，而不是直接学：

$$p_i = \text{LSTM}(h_{i-1}, \text{embed}_i), \qquad h_0 = \text{可学习向量}$$

动机：直接学 $P$ 的初始化很敏感、且 prompt 内部 token 之间有关联，LSTM 能把依赖关系编码进生成过程。早期用于 NLU（SuperGLUE 等）。

**P-Tuning v2（Liu et al., 2022）**——改为**逐层 prefix**（deep prompt）+ 各任务 head：

$$\text{每层注入：} [\text{K}_i; P_i^K], [\text{V}_i; P_i^V], \quad \text{叠加任务输出头}$$

修复了 v1 在通用任务（分类/序列标注/生成）上效果弱的问题，对标 Prefix Tuning，参数量更大但更稳。

## 二、核心原理

### 2.1 为什么 soft prompt 有效

1. **可微**：连续向量可以直接对 loss 求梯度，比离散 token 搜索（不可微、要采样/束搜索）高效得多；
2. **少即是多**：模型越大，soft prompt 越有效（论文观察：T5-XXL 上 Prompt Tuning 接近全参微调；小模型上弱）；
3. **注意力机制放大**：prompt 向量参与每层注意力，其信息会通过 attention 权重传播到全部 token；
4. **参数最少**：7B 模型只动几十万参数，显存增量可忽略。

局限：纯输入侧注入（Prompt Tuning v1）信息只能从第一层开始传播，层数深时任务信号被稀释——这是 Prefix 逐层注入的动机。

### 2.2 为什么 Prefix 加在 K/V 而不是输入

- **调制注意力**：直接往每层 attention 的 K/V 拼接前缀，等价于给每个 query 一个"额外的上下文池"，逐层都有任务信号；
- **不破坏输入流**：输入嵌入序列保持不变，位置编码、因果掩码的结构不受影响（只对 prefix 部分放宽掩码）；
- **代价**：序列长度 $T \to T + l$，注意力矩阵变宽、KV cache 变大 → 推理延迟上升。

### 2.3 重参数化（MLP/LSTM 生成）的意义

直接优化高维 prefix 参数在训练初期不稳定（初始化敏感）。先用一个小 MLP/LSTM 把低维参数映射成 prefix：

$$\theta = \text{MLP}(\theta') , \qquad \theta' \text{ 可训练，} \theta \text{ 为实际前缀}$$

训练收敛后可以**丢弃生成器、直接存 $\theta$**（P-Tuning 与 Prefix Tuning 都这么做）。收益：加速收敛、稳定训练；代价：初期参数更多。

### 2.4 参数量对比与选择

| 方法 | 注入位置 | 参数量（7B 例） | 表达力 | 推理开销 | 适用 |
|------|---------|----------------|--------|----------|------|
| Prompt Tuning | 仅输入嵌入前 | ~0.4M（0.005%） | 低 | 可预计算，几乎无 | 超大模型、极省参数 |
| Prefix Tuning | 每层 K/V | ~5.2M（0.08%） | 中高 | **KV 变长** | 生成任务、NLG |
| P-Tuning v2 | 每层 K/V + 任务头 | ~5-20M | 中高 | KV 变长 | 通用 NLU/NLG |
| 对照：LoRA | 每层线性层 | ~4.5-20M | 高 | 0（可合并） | 通用首选 |

## 三、源码实现

### 3.1 手写 SoftPrompt（Prompt Tuning 输入侧）

```python
import torch
import torch.nn as nn

class SoftPrompt(nn.Module):
    """Prompt Tuning：可学习的 l×d soft prompt，拼接到输入嵌入之前"""

    def __init__(self, l=10, d=32):
        super().__init__()
        self.l, self.d = l, d
        self.prompt = nn.Parameter(torch.randn(l, d) * 0.02)   # 唯一可训练参数

    def forward(self, input_ids, embed_fn):
        h = embed_fn(input_ids)                                      # (B, T, d)
        P = self.prompt.unsqueeze(0).expand(h.shape[0], -1, -1)      # (B, l, d)
        return torch.cat([P, h], dim=1)                              # (B, l+T, d)

# 自检
torch.manual_seed(0)
embed = nn.Embedding(64, 32)
sp = SoftPrompt(l=10, d=32)
ids = torch.randint(0, 64, (4, 16))
out = sp(ids, embed)
print(out.shape)                                    # torch.Size([4, 26, 32])
print(sum(p.numel() for p in sp.parameters()))      # 320（10×32，仅此而已）
```

### 3.2 SoftPrompt 训练 demo（模型冻结，只训 prompt + 任务头）

```python
class TinyLM(nn.Module):
    """迷你语言模型：嵌入 + 单层自注意力"""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.config = Cfg(hidden_size=32, num_layers=1, num_attention_heads=2,
                          vocab_size=64, model_type="custom")   # peft 包装器需要
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

    def get_input_embeddings(self):          # peft 包装器需要
        return self.embed

    @property
    def device(self):                        # peft 包装器需要
        return next(self.parameters()).device

class Cfg(dict):
    """peft 读取模型配置时既用 .get 又用属性访问，做成 dict+attr 双兼容"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def to_dict(self): return dict(self)

torch.manual_seed(0)
model = TinyLM()
for p in model.parameters():
    p.requires_grad = False                         # 整个模型冻结

sp = SoftPrompt(l=10, d=32)                         # 只学 prompt 向量
head_ft = nn.Linear(32, 64)                         # 任务输出头（可选）

x = torch.randint(0, 64, (16, 10))
y = torch.randint(0, 64, (16, 10))
opt = torch.optim.Adam([*sp.parameters(), *head_ft.parameters()], lr=3e-3)
crit = nn.CrossEntropyLoss()
model.eval()
for step in range(100):
    opt.zero_grad()
    h = sp(x, model.embed)                          # (16, 20, 32)：前 10 个是 prompt
    logits = head_ft(model.o_proj(model.ln(h)))     # 冻结主干只走一层投影
    loss = crit(logits[:, 10:].reshape(-1, 64), y.reshape(-1))  # 只看真实 token 部分
    loss.backward(); opt.step()
print(f"Prompt Tuning 训练后 loss = {loss.item():.3f}")   # ~4.43 -> ~2.45
# 注意：只动了 320 个 prompt 参数 + 一个输出头，主干全程冻结
```

### 3.3 手写 Prefix KV（每层注入 attention）

```python
def attention_with_prefix(q, k, v, prefix_k, prefix_v, mask=None):
    """在 attention 的 K/V 序列前拼接 prefix（Prefix Tuning 的核心操作）
    q/k/v: (B, H, T, D)；prefix_k/v: (l, H, D) —— 每层一组可学习参数"""
    B, H, T, D = q.shape
    pk = prefix_k.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)  # (B, H, l, D)
    pv = prefix_v.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)
    k = torch.cat([pk, k], dim=2)          # 序列维拼接：T -> l+T
    v = torch.cat([pv, v], dim=2)
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)   # (B, H, T, l+T)
    if mask is not None:
        scores = scores + mask
    return torch.softmax(scores, dim=-1) @ v        # (B, H, T, D)

torch.manual_seed(0)
q = torch.randn(4, 2, 8, 16); k = torch.randn(4, 2, 8, 16); v = torch.randn(4, 2, 8, 16)
prefix_k = torch.randn(10, 2, 16); prefix_v = torch.randn(10, 2, 16)   # 每层 l=10 前缀
out = attention_with_prefix(q, k, v, prefix_k, prefix_v)
print(out.shape)                       # torch.Size([4, 2, 8, 16])
# 关键：scores 是 (4, 2, 8, 18)——注意力宽了 10 个 prefix 位置，这就是"序列变长"的来源
```

### 3.4 P-Tuning v1：LSTM 生成 prompt（手写生成器）

```python
class PTuningPrompt(nn.Module):
    """P-Tuning v1：用 LSTM 生成 l 个 prompt 向量（解决初始化敏感 + 内部依赖）"""

    def __init__(self, l=10, d=32):
        super().__init__()
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, d))          # 可学习初始隐状态
        self.c0 = nn.Parameter(torch.zeros(1, d))
        self.seeds = nn.Parameter(torch.randn(l, d) * 0.02)   # 每个 prompt token 的种子

    def forward(self, batch_size):
        seeds = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)  # (B, l, d)
        h0 = self.h0.unsqueeze(1).expand(-1, batch_size, -1)        # (1, B, d)
        c0 = self.c0.unsqueeze(1).expand(-1, batch_size, -1)
        out, _ = self.lstm(seeds, (h0.contiguous(), c0.contiguous()))
        return out                                     # batch_first → (B, l, d)

torch.manual_seed(0)
pt = PTuningPrompt(l=10, d=32)
print(pt(4).shape)                          # torch.Size([4, 10, 32])
# 训练后可以丢弃 LSTM，只保存生成的 (l, d) 向量 —— 与 Prefix Tuning 的 MLP 同思路
```

### 3.5 peft 库用法（三种软提示配置）

```python
from peft import (PromptTuningConfig, PrefixTuningConfig,
                  PromptEncoderConfig, get_peft_model, TaskType)

x = torch.randint(0, 64, (2, 8))

# Prompt Tuning：只学输入侧 4 个虚拟 token
pm = get_peft_model(TinyLM(), PromptTuningConfig(
    num_virtual_tokens=4, token_dim=32,
    task_type=TaskType.FEATURE_EXTRACTION,
    num_layers=1, num_attention_heads=2))
pm.print_trainable_parameters()      # trainable params: 128 || all params: 8,576 || trainable%: 1.4925

# Prefix Tuning：每层 K/V 前缀（自动注入含 attn 的模块）
pm = get_peft_model(TinyLM(), PrefixTuningConfig(
    num_virtual_tokens=4, token_dim=32,
    task_type=TaskType.FEATURE_EXTRACTION,
    num_layers=1, num_attention_heads=2))
pm.print_trainable_parameters()      # trainable params: 256 || all params: 8,704 || trainable%: 2.9412

# P-Tuning（PromptEncoder）：MLP/LSTM 生成 prompt
pm = get_peft_model(TinyLM(), PromptEncoderConfig(
    num_virtual_tokens=4, token_dim=32, encoder_hidden_size=16,
    task_type=TaskType.FEATURE_EXTRACTION,
    num_layers=1, num_attention_heads=2))
pm.print_trainable_parameters()      # trainable params: 1,472 || all params: 9,920 || trainable%: 14.8387

print(pm(x).shape)                   # torch.Size([2, 12, 64])：输入序列多了 4 个 prompt token
```

## 四、参数与显存账本

### 4.1 参数量数值（LLaMA-7B 场景：$d=4096$，$L=32$ 层）

| 方法 | 公式 | 数值 | 占比 |
|------|------|------|------|
| Prompt Tuning | $l \times d$（$l=100$） | 0.41M | 0.006% |
| Prefix Tuning | $2 \times l \times d \times L$（$l=20$） | 5.24M | 0.08% |
| P-Tuning v2 | 同 Prefix + 任务头 | 5~20M | 0.1%~0.3% |
| 对照：LoRA r=8 | $r(d_{in}+d_{out})$ 全注入 | ~20M | 0.30% |

### 4.2 KV cache 增量（Prefix 的推理代价）

Prefix 每层把 K 和 V 各加 $l$ 个 token，bf16 下每层 KV 增量：

$$\text{增量} = \underbrace{l}_{\text{K}} \times d \times 2\text{B} + \underbrace{l}_{\text{V}} \times d \times 2\text{B} = 4\,l\,d\ \text{字节}$$

LLaMA-7B（$d=4096$，$l=20$，32 层）：$4 \times 20 \times 4096 \times 32 \approx 10.5$MB——加上注意力矩阵变宽带来的计算，生成延迟明显上升。这是 Prefix 家族相对 LoRA 的核心劣势。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 参数最少（Prompt Tuning 可到 0.005%） | 纯输入侧注入表达力有限，小模型上效果弱 |
| 模型越大越有效（T5-XXL 接近全参） | Prefix 使序列变长 → 推理延迟与 KV 显存上升 |
| 不修改任何模型权重，天然不遗忘 | 对生成类任务的初始 token 分布敏感 |
| 多任务只需多存几个向量，切换成本极低 | 需要重参数化技巧（MLP/LSTM）才稳 |
| 与 LoRA 正交，可叠加使用 | 通用 LLM 任务上通常弱于 LoRA |

## 六、与同类方法对比

| 维度 | Prompt Tuning | Prefix / P-Tuning v2 | LoRA | Adapter |
|------|--------------|----------------------|------|---------|
| 注入位置 | 输入嵌入前 | 每层 attention K/V | 每层线性层 | 子层之间 |
| 参数量（7B） | ~0.4M | ~5-20M | ~4.5-20M | ~17-34M |
| 效果 | 60-90% | 80-95% | 95-99% | 95-99% |
| 推理开销 | 可预计算，几乎无 | **序列变长** | 0（可合并） | 串行有延迟 |
| 改权重 | 否 | 否 | 是（可合并） | 是 |
| 典型场景 | 超大模型、极端预算 | NLG、少样本 | 通用首选 | 多任务解耦 |

> **面试记忆点**：三类软提示的区别就在**注入位置**（输入 only vs 每层 K/V）与**生成方式**（直接学 vs LSTM/MLP 重参数化）。多模态场景常把视觉侧 soft prompt 与语言侧 LoRA 组合使用（见总览第八节）。

## 七、高频面试问答

**Q1：离散 prompt 和连续 prompt 的区别？**
离散 prompt 是文本 token，不可微、需手工设计、效果不稳；连续 prompt 是可学习向量，直接在嵌入空间梯度优化，可微、可端到端训练。

**Q2：Prompt Tuning 和 Prefix Tuning 的区别？**
Prompt Tuning 只在输入嵌入前拼 $P$（参数 $l \times d$，表达力低）；Prefix Tuning 在每层 attention 的 K/V 前拼前缀（参数 $2ldL$，逐层注入任务信号，表达力强），但序列变长、推理变慢。

**Q3：为什么 Prefix 加在 K/V 而不是输入？**
直接调制每层注意力分布，任务信号逐层生效；不破坏输入嵌入流与位置编码结构。代价是 KV cache 与注意力矩阵变宽。

**Q4：P-Tuning v1 为什么用 LSTM 生成 prompt？**
直接学 prompt 初始化敏感、且 prompt token 间有依赖；LSTM 能建模内部依赖、提供稳定初始化。训练后可丢弃生成器只存向量。

**Q5：P-Tuning v1 和 v2 的区别？**
v1 是输入侧 LSTM 生成 prompt，只在 NLU 上有效；v2 改为逐层 prefix（deep prompt）+ 任务头，对标 Prefix Tuning，通用 NLU/NLG 上更稳、效果更好，但参数更多。

**Q6：软提示会增加推理开销吗？**
Prompt Tuning 的向量可预先拼进嵌入并缓存，几乎无感；Prefix/P-Tuning v2 每层 KV 各多 $l$ 个 token，KV cache 与注意力计算增大（7B 场景约 +10MB/20 前缀），生成延迟上升。

**Q7：为什么大模型上软提示效果更好？**
模型越大，预训练知识越完整，任务适配所需的"提示"越轻量；论文在 T5-XXL（11B）上 Prompt Tuning 接近全参微调。小模型则依赖权重级适配（LoRA 等）。

**Q8：soft prompt 会修改模型权重吗？**
不会。权重全程冻结，只是往输入/KV 拼接可学习向量——因此天然不产生灾难性遗忘，也是它参数最少的原因。这也是它与 LoRA/Adapter（权重级改动）的本质区别。

## 八、自我检验

- [ ] 能写出 $x' = [P; E(x)]$ 并解释 $P$ 的维度与参数量（$l \times d$）
- [ ] 能写出 Prefix Tuning 的 $[\text{K}_i; P_i^K], [\text{V}_i; P_i^V]$ 并解释为何序列变长
- [ ] 能说清 P-Tuning v1（LSTM 输入侧）与 v2（逐层 prefix）的差异
- [ ] 能解释重参数化（MLP/LSTM → 前缀）的作用与"训练后丢弃生成器"
- [ ] 能手写 SoftPrompt 拼接、Prefix K/V 注入 attention 的完整代码
- [ ] 能背出 7B 场景参数量：PT 0.4M / Prefix 5.2M / P-Tuning v2 5-20M / LoRA 4.5-20M
- [ ] 能计算 Prefix 的 KV cache 增量（$4ldL$ 字节）并解释推理代价
- [ ] 能说出 soft prompt 不修改权重、不产生遗忘的特点
- [ ] 能写出 peft 三种软提示配置（PromptTuning/PrefixTuning/PromptEncoder）代码
- [ ] 能回答 8 个面试追问
