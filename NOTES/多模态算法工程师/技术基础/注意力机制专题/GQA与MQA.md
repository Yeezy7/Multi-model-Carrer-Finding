# GQA 与 MQA：共享 KV 的多头注意力

> 本模块索引见 [注意力机制专题详解](注意力机制专题详解.md)

## 一、定义与公式

### 1.1 三种注意力结构

标准的 MHA（Multi-Head Attention）中每个 Q 头都有**独立的一组 K、V 投影**；MQA（Multi-Query Attention）与 GQA（Grouped-Query Attention）让多个 Q 头**共享同一组 K、V**：

| 结构 | Q 头数 | KV 头数 | 共享方式 | 提出时间 |
|------|--------|---------|---------|---------|
| MHA | $h$ | $h$（每头独立） | 不共享 | Transformer 原版（2017） |
| GQA | $h$ | $g$（$g \mid h$） | $h/g$ 个 Q 头共享一组 KV | 2022（Ainslie et al.） |
| MQA | $h$ | $1$ | 所有 Q 头共享一组 KV | 2019（Shazeer） |

```text
MHA:              GQA (h=8, g=2):       MQA (h=8):
 Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
 |  |  |  |  |  |  |  |      |__|  |__|  |__|  |__|      \  \  \  \  \  \  \  \
 K1 K2 K3 K4 K5 K6 K7 K8     KV1     KV2     KV3     KV4   \        KV          \
```

- $g = 1$ 时 GQA 退化为 MQA，$g = h$ 时退化为 MHA——**GQA 是 MHA 与 MQA 的插值**；
- 代表模型：MHA（LLaMA1、GPT-2）、GQA（**LLaMA2/3、Mistral、Qwen2**）、MQA（Falcon、PaLM 部分层）。

### 1.2 KV cache：为什么要在意 KV 头数

自回归推理时，每生成一个 token 都要把历史 token 的 K、V 保存下来供后续 attention 使用（K/V 只依赖已生成的历史，可复用），这套缓存叫 **KV cache**。推理阶段它的大小只与 KV 头数成正比：

$$\text{KV cache 每 token 显存} = 2 \times L \times h_{KV} \times d_{KV} \times \text{bytes}$$

（因子 2 = K 和 V 各一份；$h_{KV}$ 是 KV 头数，MHA 为 $h$，GQA 为 $g$，MQA 为 1。）

长上下文推理时 KV cache 常常**比模型权重本身还大**，是吞吐量的第一瓶颈——GQA/MQA 就是为压缩它而生。

### 1.3 GQA 的前向公式

第 $i$ 个 Q 头属于组 $\lfloor i / (h/g) \rfloor$，与同组 Q 头共享 KV：

$$\text{head}_i = \text{Attn}\left(Q_i,\; K_{\lfloor i / (h/g) \rfloor},\; V_{\lfloor i / (h/g) \rfloor}\right)$$

## 二、核心原理

### 2.1 共享 KV 为什么可行

1. **注意力分布由 Q 主导**：$QK^\top$ 中 Q 每头不同，共享 K 只是让多个头"看同一批信息源"，每头仍用自己的查询选择不同的关注点；
2. **K/V 的信息容量可以共用**：相邻 Q 头的语义高度相关，各头独立学出的 K/V 投影往往冗余——共享相当于对 KV 施加了"低秩约束"，把冗余去掉；
3. **计算量几乎不变**：训练时 $QK^\top$ 仍是 $h \times n^2$ 次内积（Q 头数没变），省下的只是 KV 投影参数和推理缓存。

### 2.2 为什么是 GQA 而不是 MQA

MQA 把全部 Q 头压到**一组** KV（cache 省到 $1/h$），但单组 KV 信息容量太窄，大模型上质量损失明显。GQA 用 $g$ 组 KV 做折中：cache 只压到 $g/h$，容量足够，质量接近 MHA。经验配置：32 头 8 组、64 头 8 组（LLaMA2-70B）、8 头 8 组（LLaMA3-8B）。

### 2.3 训练与推理的不对称

- **训练**：GQA 省的主要是参数和少量显存（KV 投影变小），$QK^\top$ 计算量不变；
- **推理**：KV cache 和每步 KV 投影的计算量降为 $g/h$，且 KV 投影只需算 $g$ 份——**大批次 × 长上下文的吞吐提升**是 GQA 的最大价值。

## 三、源码实现

### 3.1 通用的缩放点积注意力

```python
import math
import torch
import torch.nn as nn

def sdpa(q, k, v, mask=None):
    """缩放点积注意力：q/k/v 形状均为 [B, H, n, d]"""
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)   # [B, H, n_q, n_k]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v
```

### 3.2 MHA / MQA / GQA 三种 nn.Module

```python
class MHA(nn.Module):
    """每个 Q 头配独立 K/V 投影"""
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)          # h 份 KV
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)   # [B, h, n, d_k]
        k = self.W_k(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        out = sdpa(q, k, v, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, n, -1))

class MQA(nn.Module):
    """所有 Q 头共享同一组 K/V"""
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)         # 只投影 1 份 KV
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)   # [B, h, n, d_k]
        k = self.W_k(x).unsqueeze(1)                                   # [B, 1, n, d_k]
        v = self.W_v(x).unsqueeze(1)
        out = sdpa(q, k, v, mask)                                      # 广播到 h 头
        return self.W_o(out.transpose(1, 2).contiguous().view(B, n, -1))

class GQA(nn.Module):
    """h 个 Q 头分成 g 组，每组共享一组 K/V"""
    def __init__(self, d_model, h, g):
        super().__init__()
        assert h % g == 0, "组数必须整除头数"
        self.h, self.g, self.d_k = h, g, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, g * self.d_k)     # 只投影 g 份 KV
        self.W_v = nn.Linear(d_model, g * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)   # [B, h, n, d_k]
        k = self.W_k(x).view(B, n, self.g, self.d_k).transpose(1, 2)   # [B, g, n, d_k]
        v = self.W_v(x).view(B, n, self.g, self.d_k).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.g, dim=1)  # 同组 Q 头复制同一份 KV
        v = v.repeat_interleave(self.h // self.g, dim=1)  # → [B, h, n, d_k]
        out = sdpa(q, k, v, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, n, -1))
```

### 3.3 三种版本的一致性验证

```python
torch.manual_seed(0)
d_model, h, g = 128, 8, 2
x = torch.randn(2, 16, d_model)
mha, mqa, gqa = MHA(d_model, h), MQA(d_model, h), GQA(d_model, h, g)
gqa_same = GQA(d_model, h, g=h)   # g=h 时应退化为 MHA

with torch.no_grad():
    # 1) GQA(g=h) 与 MHA 输出完全一致（同一套参数）
    for m in (gqa_same,):
        m.W_q.weight.copy_(mha.W_q.weight); m.W_q.bias.copy_(mha.W_q.bias)
        m.W_o.weight.copy_(mha.W_o.weight); m.W_o.bias.copy_(mha.W_o.bias)
        m.W_k.weight.copy_(mha.W_k.weight); m.W_k.bias.copy_(mha.W_k.bias)
        m.W_v.weight.copy_(mha.W_v.weight); m.W_v.bias.copy_(mha.W_v.bias)
    y_mha, y_gqa_same = mha(x), gqa_same(x)
    print("GQA(g=h) == MHA:", (y_mha - y_gqa_same).abs().max().item())   # ~0

    # 2) 构造与 GQA(g=2) 等价的 MHA：同组内头共享同一条 K/V 投影
    with torch.no_grad():
        for i in range(h):
            gi = i // (h // g)                       # 头 i 属于组 gi
            for prj in ("k", "v"):
                w_src = getattr(gqa, f"W_{prj}").weight[gi*d_model//h:(gi+1)*d_model//h]
                b_src = getattr(gqa, f"W_{prj}").bias[gi*d_model//h:(gi+1)*d_model//h]
                getattr(mha, f"W_{prj}").weight[i*d_model//h:(i+1)*d_model//h].copy_(w_src)
                getattr(mha, f"W_{prj}").bias[i*d_model//h:(i+1)*d_model//h].copy_(b_src)
        mha.W_q.weight.copy_(gqa.W_q.weight); mha.W_q.bias.copy_(gqa.W_q.bias)
        mha.W_o.weight.copy_(gqa.W_o.weight); mha.W_o.bias.copy_(gqa.W_o.bias)
    y_mha2, y_gqa = mha(x), gqa(x)
    print("GQA(g=2) == 共享式MHA:", (y_mha2 - y_gqa).abs().max().item())  # ~1e-6

    # 3) MQA 与 GQA(g=1) 一致
    mqa_ = GQA(d_model, h, 1)
    mqa_.W_k.weight.copy_(mqa.W_k.weight); mqa_.W_k.bias.copy_(mqa.W_k.bias)
    mqa_.W_v.weight.copy_(mqa.W_v.weight); mqa_.W_v.bias.copy_(mqa.W_v.bias)
    mqa_.W_q.weight.copy_(mqa.W_q.weight); mqa_.W_q.bias.copy_(mqa.W_q.bias)
    mqa_.W_o.weight.copy_(mqa.W_o.weight); mqa_.W_o.bias.copy_(mqa.W_o.bias)
    print("MQA == GQA(g=1):", (mqa(x) - mqa_(x)).abs().max().item())     # ~0
```

### 3.4 KV cache 的显存计算（数值示例）

```python
def kv_cache_bytes(L, kv_heads, d_kv, seq_len, bits=2):
    """每层每 token 存 K 和 V 两份；返回单条序列的 KV cache 总字节数"""
    return 2 * L * kv_heads * d_kv * seq_len * bits

# LLaMA2-70B：L=80, h=64, g=8, d_k=128, FP16(2B)
mha_70b = kv_cache_bytes(80, 64, 128, 8192)
gqa_70b = kv_cache_bytes(80, 8, 128, 8192)
print(f"MHA : {mha_70b/2**30:.2f} GB")    # 20.00 GB
print(f"GQA : {gqa_70b/2**30:.2f} GB")    # 2.50 GB（÷8）
print(f"压缩比: {mha_70b / gqa_70b:.0f}x")

# LLaMA3-8B：L=32, h=32, g=8, d_k=128
L, h, g, d, n = 32, 32, 8, 128, 8192
print(f"LLaMA3-8B MHA: {kv_cache_bytes(L, h, d, n)/2**30:.2f} GB, "
      f"GQA: {kv_cache_bytes(L, g, d, n)/2**30:.2f} GB")
# MHA: 8.59 GB, GQA: 1.07 GB
```

## 四、复杂度与显存分析

### 4.1 三个维度对比（h 个 Q 头、g 组 KV、序列 n、层数 L）

| 维度 | MHA | GQA | MQA |
|------|-----|-----|-----|
| KV 投影参数 | $h \cdot d_k \cdot d_{model}$ | $g \cdot d_k \cdot d_{model}$ | $1 \cdot d_k \cdot d_{model}$ |
| 训练 FLOPs（$QK^\top$） | $O(h n^2 d_k)$ | 不变 | 不变 |
| 推理每步 KV 投影 | $h n d_k$ | $g n d_k$ | $n d_k$ |
| KV cache 每 token 每层 | $2 h d_k$ 字节 | $2 g d_k$ 字节 | $2 d_k$ 字节 |
| 推理 KV 访存（带宽） | 基准 | $g/h$ | $1/h$ |

### 4.2 数值示例：为什么 70B 推理必须 GQA

LLaMA2-70B（$L=80, h=64, d_k=128$，FP16），上下文 8192：

$$\text{MHA KV cache} = 2 \times 80 \times 64 \times 128 \times 8192 \times 2 \text{ B} \approx 20.0 \text{ GiB}$$

- 20.0 GiB 已经超过很多显卡的整卡显存，且要存 K 和 V 两份；
- GQA（$g=8$）：2.50 GiB，**省掉 17.5 GiB**；
- 推理时每个新 token 还要重读全部历史 KV 做 $QK^\top$——cache 越小，访存越少，吞吐越高；
- 训练侧：KV 投影参数从 $64 \times 128 \times d_{model}$ 降到 $8 \times 128 \times d_{model}$，参数量和优化器状态也省。

## 五、优缺点

| 结构 | 优点 | 缺点 |
|------|------|------|
| MHA | 每头表达完全独立，质量上限最高 | KV cache 最大，推理显存/带宽压力最大 |
| GQA | cache 压到 $g/h$，质量接近 MHA；$g$ 可调，灵活 | 比 MHA 略降容量（大模型上几乎无损）；$g$ 是超参数要调 |
| MQA | cache 压到 $1/h$，最省 | 单组 KV 容量不足，质量损失明显（小模型更甚） |

**为什么大模型（LLaMA2/3）选 GQA 而不是 MQA**：MQA 省过头，质量保不住；GQA 在"省显存"与"保质量"之间取平衡，且 $h:g$（如 32:8）可人为控制压缩率。

## 六、与同类对比

| 维度 | GQA/MQA（共享 KV） | MLA（DeepSeek，低秩压缩 KV） | 稀疏注意力（省计算） |
|------|-------------------|------------------------------|---------------------|
| 解决的问题 | 推理 KV cache 显存与带宽 | 推理 KV cache 显存与带宽 | attention 计算量 $O(n^2)$ |
| 手段 | 砍 KV 头数（共享） | KV 压缩到低维潜变量 $c$，只存 $c$ | 剪掉部分 Q-K 边 |
| 训练计算量 | 不变 | 降低（KV 投影变小） | 降低 |
| 质量影响 | 轻微/轻微 | 轻微（低秩近似） | 中（长程依赖丢失） |
| 与 FlashAttention | 正交，可叠加 | 正交，可叠加 | 与分块优化冲突 |

> 三者正交：FlashAttention 解决"attention 跑得快"，GQA/MLA 解决"推理存得下"，稀疏解决"算得少"。

## 七、高频面试问答

**Q1：KV cache 是什么？为什么是推理瓶颈？**
自回归生成时，每生成一个新 token 都需要与全部历史 token 做 attention，历史 token 的 K、V 只依赖历史本身，可缓存复用（不必重算）。它的大小 $= 2 \times L \times h_{KV} \times d_{KV} \times n \times$ bytes，随序列线性增长，长上下文时超过模型权重，且每次解码都要读全量——显存和带宽双重瓶颈。

**Q2：GQA 和 MQA 的区别？**
MQA 所有 Q 头共享 1 组 KV（cache 降为 $1/h$）；GQA 把 $h$ 个 Q 头分成 $g$ 组、每组共享 1 组 KV（cache 降为 $g/h$）。$g=1$ 退化为 MQA，$g=h$ 退化为 MHA。GQA 是二者的插值。

**Q3：为什么 LLaMA2/3 用 GQA 而不是 MQA？**
MQA 单组 KV 信息容量不足，大模型质量损失明显；GQA 保留 $g$ 组 KV，容量够、质量接近 MHA，同时把 cache 压缩到 $g/h$。用可控的 $h:g$ 平衡"省多少"与"掉不掉点"。

**Q4：GQA 省的是训练还是推理？**
主要是推理。训练时 $QK^\top$ 仍是全量 $h$ 头计算（Q 没变），只省了 KV 投影的参数/计算；推理时 KV cache、KV 投影、KV 访存全部按 $g/h$ 缩减。

**Q5：GQA 为什么质量损失小？**
相邻 Q 头的语义高度相关，各头独立的 K/V 投影冗余严重；共享 KV 相当于对 KV 加低秩约束去掉冗余，而注意力选择性主要靠 Q 表达（每头仍用自己的查询挑关注点）。大模型容量大，这点容量损失可忽略。

**Q6：GQA 的注意力矩阵还是"每头独立"的吗？**
是。$Q_i K_{group}^\top$ 仍是逐头独立计算，同组头只是用相同的 K/V；头间信息混合仍发生在 $W^O$。这保证了与 MHA 的实现可以无缝替换（repeat_interleave 展开即可）。

**Q7：MLA 和 GQA 是什么关系？**
不同路线的两兄弟。GQA 砍 KV 头数（共享），MLA 把 KV 低秩压缩成一个潜向量 $c$（每 token 每层只存 $d_c$ 维，DeepSeek-V2 为 512），压缩比更高；二者都直击推理 KV cache 瓶颈。

## 八、自我检验

- [ ] 能画出 MHA / GQA / MQA 的 Q-KV 对应结构示意图
- [ ] 能写出 KV cache 显存公式并算出 70B 模型的数值示例
- [ ] 能手写三种 nn.Module 并验证 GQA(g=h)==MHA、GQA(g=1)==MQA
- [ ] 能说清 repeat_interleave 展开共享 KV 的原理
- [ ] 能解释"GQA 省推理、不省训练计算量"的原因
- [ ] 能回答"为什么 LLaMA2/3 用 GQA 而不是 MQA"
- [ ] 知道 MQA（Falcon）、GQA（LLaMA2/3、Mistral、Qwen2）、MLA（DeepSeek）的代表模型
- [ ] 能说清 GQA 与 FlashAttention、稀疏注意力的正交关系
- [ ] 能回答 7 个面试追问
