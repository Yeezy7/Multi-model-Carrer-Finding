---
learned: false
---

# KV Cache 原理与实现：从流程推导到可运行源码

> 本模块索引见 [KVCache详解](KVCache详解.md)

本文是 KVCache 模块的核心：从自回归解码的第一步推导出 KV Cache，用 4 个 token 的小例子逐步看矩阵形状变化；严格证明 K/V 为什么可以复用；量化分析计算量与显存公式（7B 完整账本）；最后给出可直接运行的手写 KV Cache 注意力源码（不缓存 vs 缓存对比，实测乘法节约 96.4%）。优化工程（GQA/量化/PagedAttention/裁剪）见 [KV Cache优化技术.md](KV Cache优化技术.md)。

## 一、自回归解码流程：KV Cache 是怎么"长"出来的

### 1.1 没有缓存时的朴素做法（为什么慢）

假设模型已经接受了 prompt（前 $P$ 个 token），要逐个生成接下来 $n$ 个 token。朴素做法：第 $t$ 步把**前 $t+P$ 个 token 全部重新送进模型**跑一遍完整前向，取最后一个位置的输出做预测。这意味着：

- 第 $t$ 步要对长度为 $P+t$ 的完整前缀**重新做 K/V 投影**（位置 0..P+t-1 全部重算一遍）；
- 第 $t$ 步的注意力矩阵也要对这个前缀**全部重算**。

每步都在重复做已做过的事——这就是 O(n²) 乃至更糟的开销来源。

### 1.2 有缓存的做法（标准方案）

标准方案把生成拆成两个阶段：

1. **prefill（预填充）**：一次性把整段 prompt 塞进去，算出全部 $P$ 个位置的 K/V 并缓存；
2. **decode（解码）**：每步只送**当前 1 个新 token**，算出它的 Q/K/V，K/V 拼到缓存后面，注意力只让这个新 Q 去查"历史 KV + 它自己"，得到该位置的输出后采样出下一个 token。

### 1.3 四个 token 的逐步演示（矩阵形状变化）

以小模型为例：$d_{model}=64$，4 个头（展示用 1 头的形状，全头只是行数 ×4），$d_{head}=16$，**一层**。prompt = [w1, w2]，再生成 [w3, w4]，共 4 个 token。记每层 KV 维度为 $D_{kv}=n_{kv\_heads}\times d_{head}=16$（单头示例）。

| 步骤 | 输入 | 新计算 Q/K/V 形状 | 拼接后 K/V 缓存形状 | 注意力形状 $Q K^\top$ | 输出（新 token） | 缓存变化 |
| --- | --- | --- | --- | --- | --- | --- |
| **Step 1 prefill** | w1, w2 | $X\in\mathbb{R}^{2\times64}$; $Q,K,V\in\mathbb{R}^{2\times16}$ | 空 → k,v 各 2 行：$\mathbb{R}^{2\times16}$ | $\mathbb{R}^{2\times2}$（因果掩码 $w_2$ 只看 $w_1,w_2$） | 两个位置都算（取末位采样得 **w3**） | 2 行 K + 2 行 V |
| **Step 2 decode** | w3 | $X\in\mathbb{R}^{1\times64}$; $Q,K_{new},V_{new}\in\mathbb{R}^{1\times16}$ | $\mathbb{R}^{2\times16}$ → $\mathbb{R}^{3\times16}$ | $Q_{w3}\,K_{cat}^\top\in\mathbb{R}^{1\times3}$（w3 查 w1,w2,w3） | $\mathbb{R}^{1\times64}$ → 采样得 **w4** | 3 行 K + 3 行 V |
| **Step 3 decode** | w4 | $X\in\mathbb{R}^{1\times64}$; $Q,K_{new},V_{new}\in\mathbb{R}^{1\times16}$ | $\mathbb{R}^{3\times16}$ → $\mathbb{R}^{4\times16}$ | $Q_{w4}\,K_{cat}^\top\in\mathbb{R}^{1\times4}$ | $\mathbb{R}^{1\times64}$ → 采样得 **w5** | 4 行 K + 4 行 V |

几个规律（把单头换成 4 头只是把所有行数 ×4，规律不变）：

1. **prefill 一次算 $P$ 行**（并行、GPU 友好），decode 每步只算 **1 行**；
2. **K/V 缓存每步只增长 1 行**（prefill 一次 +$P$ 行），Q 从不进缓存；
3. **新 query 的注意力总是对"历史全部 + 自己"**：打分宽度从 $P$ 开始每步 +1。这就是"有缓存时注意力每步仍是 O(序列长度)"的来源（见第三节）；
4. 缓存是**逐层**一份：32 层的模型 = 上面这套每层存一遍。

### 1.4 生成循环的伪代码

```
kv = {}
X_prompt = embed(prompt)
for layer:
    kv[layer] = (W_k(X_prompt), W_v(X_prompt))      # prefill：一次算全部 prompt 的 K/V
y = model_with_cache(X_prompt, kv)                   # 取最后一个位置输出
next = sample(y[-1])
while not stop:
    x_new = embed(next)[None, :]                     # 只有 1 个 token
    for layer:                                        # 每层：只算新 token 的 K/V 并拼入缓存
        k_new, v_new = W_k(x_new), W_v(x_new)
        kv[layer] = (cat(kv[layer][0], k_new), cat(kv[layer][1], v_new))
    y = attention_with_cache(x_new, kv)              # 新 Q 查全缓存
    next = sample(y[0])
```

## 二、为什么 K/V 可以复用：严格推导

### 2.1 两个前提

**前提 A（因果性）**：生成式 Transformer 用因果注意力，位置 $i$ 的隐藏状态只依赖前 $i+1$ 个输入：

$$x_i = f_{\theta}(w_0, w_1, \dots, w_i)$$

**前提 B（权重冻结）**：推理时模型参数 $\theta$ 固定不变。

### 2.2 推导

位置 $i$ 的 K/V 是它隐藏状态的线性投影：

$$K_i = x_i W_K, \qquad V_i = x_i W_V$$

要在第 $t$ 步（$t>i$）做注意力：

$$\text{attn}(Q_t, [K_0, \dots, K_t], [V_0, \dots, V_t]) = \text{softmax}\!\Big(\frac{Q_t\,[K_0,\dots,K_t]^\top}{\sqrt{d_{head}}}\Big)\,[V_0, \dots, V_t]^\top$$

要算这个式子需要 $K_0..K_t$。但请注意：$K_i = x_i W_K$，而 $x_i$ 由前提 A **只依赖 $w_0..w_i$**（不依赖任何未来 token），$W_K$ 由前提 B **不变**。因此在"生成 w_{i+1} 的那一步"第一次算出 $K_i$ 之后，无论后面再生成多少 token，$K_i$ 的值都**恒等不变**——它不依赖任何之后才出现的输入。$V_i$ 同理。

整个缓存序列的注意力就可以等价写成**拼接形式**：

$$\text{attn}(Q_{new},\, [K_{cache}, K_{new}],\, [V_{cache}, V_{new}])$$

其中 $K_{cache}$ 是之前所有步逐次拼出来的历史，$K_{new}$ 是当前 token 刚投影出的 1 行。**关键点：$K_{cache}$ 的每一行在被缓存时就已经被完整确定，与"何时查询"无关**——这是缓存能成立的充要条件。

> 注意：**Q 的前半段不成立**。$Q_t$ 只在第 $t$ 步存在（它属于新进 token），第 $t+1$ 步的 $Q_{t+1}$ 是另一个 token 的查询，必须现场算——这与 K/V"算过一次永不变"的位置性质根本不同（呼应总览第三节）。

### 2.3 是精确复用，不是近似

因为推导用的是恒等式（无任何舍入外的近似），**带 KV Cache 与不带 Cache 的网络输出在数学上完全等价**。实践中最大误差只有浮点噪声量级（见第五节源码验证，逐位置误差 ≤ $2.3\times10^{-7}$）。这一点常常被误解（见第九节误区）。

### 2.4 代码验证"输入不变 → K_old 不变"

```python
import torch, torch.nn as nn
x_a = torch.randn(1, 2, 16); x_b = torch.randn(1, 2, 16)
lin = nn.Linear(16, 4, bias=False)
ka1 = lin(x_a)                                      # 单独算 x_a 的 K
ka2 = lin(torch.cat([x_a, x_b], dim=1)[:, :2])      # 后面多拼了 x_b，x_a 部分的 K 不变
print("K_old 恒等（同输入必同输出）:", torch.allclose(ka1, ka2))
# K_old 恒等（同输入必同输出）: True
```

## 三、计算量分析（重点）：prefill vs decode

### 3.1 无缓存时每步重算的额外开销是 O(n²)

朴素方案生成第 $t$ 个新 token（此时序列长 $P+t$）时，要把位置 0..P+t-1 的 **K/V 全部重新投影**一遍。

- 每步额外投影量：O(P+t) 个位置 × 2（K 和 V）；
- 累计 $n$ 步：$\sum_{t=1}^{n}(P+t) = O(n^2)$ 的**重复投影**；
- 若连注意力打分也全量重算（最朴素实现），每步 O((P+t)²)，总 O(n³) 更不可行。

**KV Cache 正是把这一项 O(n²) 的重复开销整个删掉**（只留下每步 O(1) 的新 token 投影），这就是总览第二节那个"157 万 → 2048 次、约 768 倍"数字的来源。

### 3.2 有缓存后的每步成本构成

decode 第 $t$ 步（已有 $S=P+t$ 个缓存位置）的成本：

- **投影**：只算 1 个新 token 的 Q/K/V + 输出投影 → O(d²)，**与序列长度无关**；
- **注意力**：1 个 query 对全部 $S$ 个 key 打分 + 加权 → **O(S)**，随序列长度线性增长。

所以有缓存后单步成本 ≈ O(d²) + O(S·d)，**随 S 线性上涨**——KV Cache 消掉了平方项，但没消掉线性打分项（那就是窗口化/PagedAttention/稀疏注意力的舞台，见 [KV Cache优化技术.md](KV Cache优化技术.md)）。

### 3.3 prefill 与 decode：两个阶段两种瓶颈

| 维度     | Prefill（预填充）                                           | Decode（解码）                                          |
| ------ | ------------------------------------------------------ | --------------------------------------------------- |
| 任务     | 一次性并行计算整段 prompt 的全部 K/V 与注意力                          | 逐 token 生成，每步只算 1 个新 token                          |
| 每步规模   | 一批长序列（~几千~几万 token）                                    | 1 个 token，串行 $n$ 步                                  |
| 计算特征   | **计算密集（compute-bound）**，GPU 利用率高，可跨序列批量                | **访存受限（memory-bound）**：每 token 计算量小，却要读全部权重 + 全部 KV |
| 关键延迟指标 | **TTFT**（Time To First Token）：从请求到吐出第一个 token          | **TPOT**（Time Per Output Token）：平均每生成一个 token 的时间   |
| 总延迟公式  | $\text{延迟} \approx \text{TTFT} + n \times \text{TPOT}$ | ——                                                  |


**面试要点**：prefill 是长矩阵乘法（算得多、读得少），decode 是"读得远比算得多"——单 token 前向只有约 $2N$ 次乘加，但要搬动 $N$ 个权重 + $O(S)$ 个 KV 出显存，**带宽成为 decode 的瓶颈**。这也是 decode 阶段用 Batch 提高吞吐、以及 KV 量化能同时省显存和带宽的原因。

## 四、显存公式（重点）：7B 完整账本

### 4.1 公式

一个序列的 KV Cache 显存：

$$M = 2 \times L \times n_{kv\_heads} \times d_{head} \times S \times \text{bytes}$$

| 符号              | 含义                            | 7B 示例取值                    |
| --------------- | ----------------------------- | -------------------------- |
| 2               | K 和 V 两份                      | 2                          |
| $L$             | 层数                            | 32                         |
| $n_{kv\_heads}$ | 每层 KV 头数（MHA = 全部头；GQA = 分组数） | 8（GQA-8）/ 32（全 MHA）        |
| $d_{head}$      | 头维度                           | 128                        |
| $S$             | 序列长度（prompt + 已生成）            | 2048                       |
| bytes           | 每元素字节数                        | 2（FP16）/ 1（FP8）/ 0.5（INT4） |

### 4.2 7B（32 层、4096 维、GQA 8 组、2048 token、FP16）完整计算

**单层**先算（KV 维度 = $8 \times 128 = 1024$）：

$$M_{layer} = 2 \times 8 \times 128 \times 2048 \times 2 = 8{,}388{,}608 \ \text{B} = 8 \ \text{MiB}$$

**全部 32 层**：

$$M_{total} = 32 \times 8{,}388{,}608 = 268{,}435{,}456 \ \text{B} = 256 \ \text{MiB} \approx \mathbf{0.25 \ GB}$$

对应的一行账本（每 token 显存）：

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \ \text{B} = \mathbf{128 \ KB / token}$$

### 4.3 为什么网文常说"7B 2048 token 约 1GB"

上面的 GQA-8 只有 0.25 GB。"约 1 GB"对应另外两种情形：

1. **全 MHA 的 7B（如 LLaMA-2-7B：32 个 KV 头，KV 维度 = 4096）**：每层 KV = $2\times32\times128\times2048\times2 = 33{,}554{,}432$ B = 32 MiB，32 层 = 1,073,741,824 B = **1.0 GB**（= GQA-8 的 4 倍）；
2. **GQA-8 模型做 batch=4 并发**：$0.25\times4 = 1.0$ GB。

两条数值都由程序实测（见第五节源码末尾输出）：

```
[GQA-8 组] 7B 32层 2048token FP16 单序列 KV = 0.250 GB
[MHA 32 头] 7B 32层 2048token FP16 单序列 KV = 1.000 GB
```

**记法**：背公式时记住 7B 的 "128 KB/token"（GQA-8）与 "512 KB/token"（MHA），序列长度 × batch 直接乘。

### 4.4 batch（并发）维度

多并发下**每序列一份** KV Cache，总量线性相乘：

$$M_{batch} = \text{batch} \times M(\text{单序列}) = 2 \times L \times n_{kv} \times d_{head} \times S \times \text{batch} \times \text{bytes}$$

- 并发 8 × 32K 上下文 × MHA-7B：$8 \times 16\ \text{GB} = 128\ \text{GB}$——这就是 **KV Cache 成为在线推理显存最大头**、必须用 PagedAttention 按 token 动态分配（而非按整序列预留）的直接原因；
- GQA/量化把 $\text{batch}\times S$ 这个"总 token 面积"按倍率压缩，是优化篇的核心思路。

## 五、源码实现：手写最小 KV Cache 注意力（可运行）

下面用 **PyTorch 实测**一个最小 MHA 注意力层，同时实现"无缓存逐 token 重算"与"有缓存 decode"，验证两者数学等价，并对比乘法次数与耗时。（作者机器输出已附在注释；torch 2.8，CPU。）

```python
import torch
import torch.nn as nn
import math
import time

torch.manual_seed(42)

# ---------------- 最小 MHA 自注意力层 ----------------
class HeadedAttention(nn.Module):
    def __init__(self, d=64, n_heads=4, d_head=16):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d_head
        hdim = n_heads * d_head
        self.wq = nn.Linear(d, hdim, bias=False)
        self.wk = nn.Linear(d, hdim, bias=False)
        self.wv = nn.Linear(d, hdim, bias=False)
        self.wo = nn.Linear(hdim, d, bias=False)

    def project(self, x, w):               # (B,T,D) -> (B,H,T,d_head)
        B, T, _ = x.shape
        return w(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    # ---------- 方式 A：无缓存，每步对完整前缀重算 K/V ----------
    def forward_full(self, x):
        B, T, _ = x.shape
        q = self.project(x, self.wq)
        k = self.project(x, self.wk)
        v = self.project(x, self.wv)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        att = att + torch.triu(torch.full_like(att, float('-inf')), diagonal=1)
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.wo(y)

    # ---------- 方式 B：有缓存，decode 一步只算 1 个新 token 的 K/V ----------
    def decode_step(self, x_new, kv):      # x_new: (B,1,D)
        k_hist, v_hist = kv
        q = self.project(x_new, self.wq)   # (B,H,1,d_head)
        k = self.project(x_new, self.wk)
        v = self.project(x_new, self.wv)
        k = torch.cat([k_hist, k], dim=2)  # 新 K 拼进缓存
        v = torch.cat([v_hist, v], dim=2)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)   # 1 个 query 查全部历史
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(1, 1, self.n_heads * self.d_head)
        return self.wo(y), (k, v)          # 返回输出 & 更新后的缓存

    def make_cache(self, x):               # prefill：把一批 token 的 K/V 一次性算好
        return (self.project(x, self.wk), self.project(x, self.wv))

# ---------------- 验证：无缓存 与 有缓存 数学等价 ----------------
model = HeadedAttention()
d, nh, dh = 64, 4, 16
P, N = 4, 4                                 # prompt 4 token，再生成 4 token
tok = torch.randn(1, P + N, d)              # 8 个位置的隐状态序列
prompt = tok[:, :P]

with torch.no_grad():
    y_full = model.forward_full(tok)                     # 直接一次算 8 个位置
    y_prefill = model.forward_full(prompt)               # A) prefill：先算 prompt 4 个位置
    kv = model.make_cache(prompt)                        #    并缓存 K/V
    outs = []
    for t in range(P, P + N):                            # B) decode：逐 token 只算新 K/V
        y_step, kv = model.decode_step(tok[:, t:t + 1], kv)
        outs.append(y_step)
    y_cached = torch.cat([y_prefill, *outs], dim=1)

print("无缓存输出形状:", tuple(y_full.shape), " 有缓存输出形状:", tuple(y_cached.shape))
# 无缓存输出形状: (1, 8, 64)  有缓存输出形状: (1, 8, 64)
print("逐位置最大误差:",
      [float((y_full[:, t] - y_cached[:, t]).abs().max()) for t in range(P + N)])
# 逐位置最大误差: [2.24e-07, 1.60e-07, 1.04e-07, 9.69e-08, 1.19e-07, 7.64e-08, 5.22e-08, 6.71e-08]
# （全部 ≤ 2.3e-07，纯浮点噪声 → 数学等价，零近似）

# ---------------- 乘法次数（FLOPs 语义）对比 ----------------
def proj_mults(T):   # 每 token: Q/K/V 3 个投影 + 1 个输出投影
    hdim = nh * dh
    return T * (4 * d * hdim)

def attn_mults(T):   # 因果注意力：第 t 个位置平均关注 (t+1)/2 个位置，每对 2*nh*dh 次乘
    return sum(range(1, T + 1)) * 2 * nh * dh

P2, N2 = 128, 32
def no_cache():
    m = 0
    for t in range(P2, P2 + N2):             # 每生成一步重算 0..t 全部 K/V
        m += proj_mults(t + 1) + attn_mults(t + 1)
    return m

def with_cache():
    m = proj_mults(P2) + attn_mults(P2)      # prefill 只算一次
    for i in range(N2):
        T = P2 + i + 1
        m += proj_mults(1)                   # 每步只投影 1 个新 token
        m += T * 2 * nh * dh                 # 注意力：1 个 query 对 T 个 key
    return m

m1, m2 = no_cache(), with_cache()
print(f"prompt={P2} token 生成 {N2} token | 无缓存乘法≈{m1:,} | 有缓存乘法≈{m2:,} "
      f"| 减少 {(1 - m2 / m1) * 100:.1f}%")
# prompt=128 token 生成 32 token | 无缓存乘法≈118,992,896 | 有缓存乘法≈4,270,080 | 减少 96.4%

# ---------------- 时间对比：整体重算 vs 缓存(cat 拼接 vs 预分配) ----------------
d2, nh2, dh2 = 256, 8, 32
big = HeadedAttention(d2, nh2, dh2)
seq = torch.arange(1, 513, dtype=torch.float32).view(1, 512, 1).repeat(1, 1, d2)
n_iter, n_dec = 20, 300

with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = big.forward_full(seq)            # 全量重算前向（无缓存）
    t_full = (time.perf_counter() - t0) / n_iter

with torch.no_grad():
    kv = big.make_cache(seq)
    t0 = time.perf_counter()
    for _ in range(n_dec):
        x1 = torch.randn(1, 1, d2)
        y, kv = big.decode_step(x1, kv)      # 每步 torch.cat 拼接
    t_cat = (time.perf_counter() - t0) / n_dec

with torch.no_grad():
    maxlen = 512 + n_dec + 64
    kk = torch.zeros(1, nh2, maxlen, dh2)    # 方式 B2：预分配整块内存，指针写入
    vv = torch.zeros(1, nh2, maxlen, dh2)
    kk[:, :, :512] = big.project(seq, big.wk)   # prefill 直接落位
    vv[:, :, :512] = big.project(seq, big.wv)
    ptr = 512
    t0 = time.perf_counter()
    for _ in range(n_dec):
        x1 = torch.randn(1, 1, d2)
        q = big.project(x1, big.wq)
        kk[:, :, ptr:ptr + 1] = big.project(x1, big.wk)   # 写入新位置，不整块拷贝
        vv[:, :, ptr:ptr + 1] = big.project(x1, big.wv)
        att = q @ kk[:, :, :ptr + 1].transpose(-1, -2) / math.sqrt(dh2)
        att = torch.softmax(att, dim=-1)
        _ = att @ vv[:, :, :ptr + 1]
        ptr += 1
    t_prealloc = (time.perf_counter() - t0) / n_dec

print(f"全量重算前向(seq=512): {t_full * 1000:.2f} ms/步")
# 全量重算前向(seq=512): 2.9X ms/步（本机 CPU 实测，数值随负载小幅波动）
print(f"有缓存 decode 单新token: cat拼接版 {t_cat * 1000:.3f} ms | 预分配版 {t_prealloc * 1000:.3f} ms")
# 有缓存 decode 单新token: cat拼接版 ~0.17 ms | 预分配版 ~0.12 ms（本机 CPU 实测）

# ---------------- KV 显存账本 ----------------
gb = 1024 ** 3
for label, nkv in [("GQA-8 组", 8), ("MHA 32 头", 32)]:
    M = 2 * 32 * nkv * 128 * 2048 * 2
    print(f"[{label}] 7B 32层 2048token FP16 单序列 KV = {M / gb:.3f} GB")
# [GQA-8 组] 7B 32层 2048token FP16 单序列 KV = 0.250 GB
# [MHA 32 头] 7B 32层 2048token FP16 单序列 KV = 1.000 GB

# ---------------- K_old 与后续输入无关（可复用性的来源） ----------------
x_a = torch.randn(1, 2, 16); x_b = torch.randn(1, 2, 16)
lin = nn.Linear(16, 4, bias=False)
ka1 = lin(x_a)
ka2 = lin(torch.cat([x_a, x_b], dim=1)[:, :2])
print("K_old 恒等（同输入必同输出）:", torch.allclose(ka1, ka2))
# K_old 恒等（同输入必同输出）: True
```

**例子速读**：

1. **等价验证**：带缓存与不带缓存逐位置输出最大误差 ≤ $2.3\times10^{-7}$——浮点噪声级别，证明 KV Cache 是**精确加速**而非近似；
2. **乘法计数**：生成 32 个新 token，无缓存 ≈ 1.19 亿次乘、有缓存 ≈ 427 万次，**减少 96.4%**（与 prefill 的 128 长度下的 O(n²)→O(n) 理论吻合）；
3. **时间对比**：全量重算单步（seq=512）约 2.9 ms；缓存后**单个新 token 约 0.17 ms（cat 拼接版）/ 0.12 ms（预分配版）**——预分配省掉了每次 `cat` 的整块拷贝与内存分配，工程上选预分配（见第六节）。

## 六、缓存初始化与两种存储实现

### 6.1 torch.cat 拼接版（教学、正确性优先）

第五节 `decode_step` 用 `torch.cat` 每次把新 K/V 拼到缓存后：

```python
k = torch.cat([k_hist, k], dim=2)   # 每步分配一块新内存并整块拷贝历史
```

- 优点：实现简单、形状直观、训练/调试友好；
- 缺点：**每步 O(S) 的整块拷贝 + 内存分配开销**，生成长序列/高并发时是额外成本（实测比预分配慢约 40%）。

### 6.2 预分配 tensor（生产标准做法）

推理框架（HF `generate`、vLLM、TensorRT-LLM）的做法：**按最大序列长度一次性分配**，用指针/位置索引写入，不整块拷贝。伪代码：

```python
maxlen = max_prompt_len + max_new_tokens          # 预分配整段
k_cache = torch.zeros(B, n_heads, maxlen, d_head)
v_cache = torch.zeros(B, n_heads, maxlen, d_head)
ptr = prompt_len                                    # 位置指针
# ... prefill: k_cache[:, :, :ptr] = project(prompt, W_k) ...
# decode 每步: k_cache[:, :, ptr] = project(new_token, W_k)  ; ptr += 1
```

- 优点：写入即落位、无拷贝、无重复 malloc；可以与 CUDA Graph、连续 batch 缓存配合；
- 缺点：为每个序列预留 maxlen 会**按"最长可能"占内存**——短序列也吃满额度，这就是内存碎片/浪费的来源，也是 PagedAttention 出现的动机（物理块按需分配 + 页表映射，见 [KV Cache优化技术.md](KV Cache优化技术.md)）；
- **初始化时机**：prefill 拿到 prompt 后一次性写入全部 K/V 并记录长度；decode 每步只写 1 行；生成结束即释放；跨请求各自独立，仅 prefix caching 共享公共前缀（详见 8.1）。

## 七、与 FlashAttention 的关系

**FlashAttention（FA）不改变 KV Cache 的语义——它只是把"带缓存后的注意力计算"本身算得更快。**

| 维度     | KV Cache          | FlashAttention                      |
| ------ | ----------------- | ----------------------------------- |
| 解决什么   | 避免历史 K/V 的重算（算法级） | 注意力 kernel 加速 + 省中间矩阵显存（kernel 级）   |
| 是否影响输出 | 无（精确等价）           | 无（数学等价，只是分块计算 + on-the-fly softmax） |
| 与序列长度  | 缓存大小 O(S)         | 显存从 O(S²) 降到 O(S)，但语义不变             |
| 是否互斥   | 否，工程上**同时使用**     | 否，同时使用                              |

两个常见混淆点：

1. **"FA 省显存"≠"KV Cache 省显存"**：FA 省掉的是**注意力打分矩阵 $QK^\top$（S×S）本身**不落显存；KV Cache 则是**给要存的 K/V 增加显存**（换来免重算）。FA 不减少 KV 的大小；
2. decode 阶段 FA 依然要读全部 KV 来算注意力——**FA 优化的是算术强度，读 KV 的带宽压力依旧**，所以 KV 量化和 PagedAttention 的优化与 FA 正交、可叠加。

## 八、常见工程问题

### 8.1 缓存何时初始化、分配多大

- decode 开始前按 `max_prompt_length + max_new_tokens`（或直接 `max_seq_length`）**一次性预分配**，避免运行中的反复 malloc；
- 无法确知最终长度时用分块/指针增长策略（如 PagedAttention 按固定 token 数块分配、不够再挂新块）；
- 预计算还是不预分配？——每块显存都有 cost：预分配省时但可能浪费额度；动态分配省显存但每步有分配开销。主流推理框架统一走"预分配 + 长度指针"，配合 CUDA Graph 进一步省 launch 开销。

### 8.2 多 batch 的缓存管理

- **每个序列一份** K/V，batch 维按并发数分配：`k_cache[B, n_heads, maxlen, d_head]`；
- 变长序列会有 padding 空洞（同一 batch 内短序列的尾部额度闲置）→ 浪费；vLLM 的 PagedAttention 用**逻辑序列 → 物理块**的页表解耦，让不同请求按需共用/复用物理块，把浪费压到最低；
- **冲突写入**：同一地址不能同时被两个请求占用 → 由 scheduler 统一仲裁；beam search 分支分裂前共享同一段 K/V，分裂后独立（优化篇）。

### 8.3 cache 显存碎片

- 变长 + 预分配 maxlen → 短序列预留段闲置（外部碎片）；反复分配/释放不同长度块 → 碎片累积，可能"总量够却分不出连续块"；
- 解法：固定大小块（PagedAttention）、KV 换出到 CPU（swap）、前缀共享、低精度存储（压缩碎片的"粒度"）。

## 九、高频面试问答

**Q1：为什么只缓存 K/V 而不缓存 Q？**
Q 属于"当前新 token"，它只在本步被用来生成本步输出，用完即弃；每个新 token 的位置不同，Q 必须用自己当前的隐藏状态现场算。而 K/V 是"历史档案"，未来的每一步 query 都要来查它们，加上因果注意力与权重冻结使得已算出的 K/V 恒定，所以缓存 K/V 才能复用得起来（详见第二节推导）。

**Q2：KV Cache 显存怎么算？给个 7B 的数字？**
$M = 2 \times L \times n_{kv} \times d_{head} \times S \times \text{bytes}$。7B（32 层、d=4096、GQA-8、d_head=128、FP16、2048 token）= $2\times32\times8\times128\times2048\times2 \approx 0.25$ GB（约 128 KB/token）；全 MHA 的头数是 4 倍 → 1 GB；再乘 batch 得到并发总用量。

**Q3：prefill 和 decode 的区别？TTFT/TPOT？**
prefill 一次性并行处理整个 prompt、算全部 K/V，计算密集（compute-bound），其耗时决定 TTFT（首 token 延迟）；decode 逐 token 生成、每步只算 1 个新 token，访存受限（memory-bound），单步耗时即 TPOT。总延迟 ≈ TTFT + 序列长度 × TPOT；视觉/长 prompt 拉高 TTFT，长序列拉高 TPOT。

**Q4：为什么不缓存中间层输出（隐藏状态）？**
根本原因：**会被"未来"重算的只有 K/V 一份信息**。中间层隐藏状态 $x_i$ 虽然在因果下也固定，但它真正的用途是投影出 K/V 供后面被查询——直接缓存 K/V 已经把这个信息完整保留，若只缓存 $x$ 而不缓存 K/V，每步还是要把它再投影一次（等于没省）；而把每层每位置的 $x$ 全存下来（S × d × L）显存远大于投影后的 K/V，纯属浪费。所以缓存**投影后的 K/V**是在"信息保留 + 免重算"上的最优选择。

**Q5：KV Cache 会造成精度损失、改变模型输出吗？**
不会。缓存是精确复用：位置 i 的 K/V 由"前 i 个输入 + 固定权重"唯一确定，与何时查询无关，数学上严格等价（第五节实测最大误差 ≤ $2.3\times10^{-7}$，仅浮点噪声）。真正引入损失的是一派"压缩缓存"的变体（KV 量化、MQA 强行共享、MLA 低秩），那是显存放不下的主动取舍，不是 KV Cache 本身。

**Q6：长上下文/高并发 KV Cache 爆显存怎么办？**
链条式响应：① 架构层 GQA/MQA 减 KV 头（4 倍收益）；② 存储层 KV 量化（INT8/INT4，1/2~1/4）；③ 布局层 PagedAttention 消除碎片、按 token 动态分配；④ 范围层窗口化只留最近 W 个；⑤ 源头层 token 裁剪（尤其视觉 token）。详见 [KV Cache优化技术.md](KV Cache优化技术.md)。

**Q7：为什么说 decode 阶段 GPU 利用率低 / 是带宽瓶颈？**
每步只有 1 个 token 的计算量（≈2N 乘加），但要**把全部权重 + 全部 KV 从显存搬到寄存器**来完成这极少量的算术——内存带宽远慢于算术能力，GPU 空转等数据。所以 decode 吞吐通常以带宽计（token/s 被 bytes/step 除），高并发靠 batch 摊薄每 token 的权重读取才能拉满。

## 十、常见误区

1. **"KV Cache 有损耗，输出和无缓存不一样"** —— 错。缓存是精确复用、数学等价（见 Q5）；有损的是主动压缩的变体，不是 KV Cache 本身。
2. **"KV Cache 能省显存"** —— 说反了。它省的是**算力/时延**（把 O(n²) 重算降为 O(n)），显存开销反而**增加**（要把 K/V 存下来）。真正省显存的是 FlashAttention（省注意力矩阵）、KV 量化、PagedAttention（省碎片），它们和 KV Cache 是不同层面的东西。
3. **"有 KV Cache 后，每步注意力就不用扫描全序列了"** —— 错。decode 每步的 query 仍要对**全部历史 KV** 打分，成本 O(S) 随序列线性上涨；KV Cache 只消除了 K/V 的重复投影。要砍掉这个 O(S) 得靠窗口化/稀疏注意力。
4. **"KV Cache 是一次性的，所有请求都能共用"** —— 每请求独立缓存；只有**公共前缀**（prefill 的头 N 个 token 相同）在 prefix caching 下才能复用，其余部分不共享。
5. **"预分配 maxlen 缓存最省显存"** —— 预分配省的是运行开销，但按"最长可能"预留会为短序列多占额度；要省显存靠动态分块（PagedAttention）或压缩（GQA/量化），两者不矛盾但针对的问题不同。

## 十一、自我检验

- [ ] 能画出 prefill（P 个位置一次算 K/V）→ decode（每步 1 个新 K/V 拼接）的完整流程
- [ ] 能用手写 4-token 例子说明 Q/K/V 形状与缓存每次 +1 行的规律
- [ ] 能严格推导"K_i 只依赖前 i 个输入 + 权重冻结 ⟹ K/V 可复用"，并说清 Q 为什么不成立
- [ ] 会写 $M = 2\,L\,n_{kv}\,d_{head}\,S\,\text{bytes}$ 并现场算 7B GQA-8（0.25 GB）与 MHA（1 GB）
- [ ] 能分清 prefill（compute-bound，TTFT）与 decode（memory-bound，TPOT）
- [ ] 能解释缓存后的单步成本 = O(d²) 投影 + O(S) 注意力，O(n²) 重算被删掉
- [ ] 能跑通第五节源码并复现：输出等价（误差 ≤2.3e-07）、乘法减少 96.4%、cat ≈0.17ms vs 预分配 ≈0.12ms
- [ ] 能说明 KV Cache 与 FlashAttention 正交、与 PagedAttention 的边界
- [ ] 能回答 7 道面试题并指出 5 条常见误区的错处在哪
- [ ] 已通读总览 [KVCache详解](KVCache详解.md)，准备进入 [KV Cache优化技术.md](KV Cache优化技术.md)
