---
learned: false
---

# KV Cache 优化技术

> 本模块索引见 [KVCache详解](KVCache详解.md)

## ① 优化动机总览：KV cache 是长上下文与高并发的第一显存瓶颈

自回归推理时，每生成一个 token，模型都要"回头"看历史所有 token 的 K、V 来算注意力。K/V 只依赖已生成的历史、与当前生成位置无关，所以把它们缓存复用（不用重算），这套缓存就是 **KV cache**。

### 1.1 大小公式：随 batch × seq 线性增长

$$\text{KV cache 显存} = \underbrace{2}_{\text{K 和 V}} \times L \times n_{\text{kv}} \times d_{\text{head}} \times b \times B \times S$$

- $L$：层数；$n_{\text{kv}}$：KV 头数（MHA 为 $h$）；$d_{\text{head}}$：每头维度；$b$：每元素字节数（FP16 为 2）；$B$：batch；$S$：序列长度。

**关键洞察**：权重显存是**常数**（加载一次），KV cache 却随 $B\times S$ **线性增长**——上下文越长、并发越高，KV 越大，最终反超权重本身。

### 1.2 显存账本：7B / 70B 在不同序列长度下的 KV 占用（FP16）

| 模型         | 参数量(FP16) | 层数 L | $d_{\text{head}}$ | KV 头数     | 每 token 每请求 |
| ---------- | --------- | ---- | ----------------- | --------- | ----------- |
| LLaMA2-7B  | ~14 GB    | 32   | 128               | 32 (MHA)  | 0.5 MiB     |
| LLaMA2-70B | ~140 GB   | 80   | 128               | 8 (GQA-8) | 0.31 MiB    |

| 序列长度 S  | 7B KV 占用 | 70B KV 占用 | 7B KV / 权重      |
| ------- | -------- | --------- | --------------- |
| 512     | 0.25 GiB | 0.16 GiB  | 1.9%            |
| 2,048   | 1 GiB    | 0.63 GiB  | 7.3%            |
| 8,192   | 4 GiB    | 2.5 GiB   | 29%             |
| 32,768  | 16 GiB   | 10 GiB    | 1.14×（**超过权重**） |
| 131,072 | 64 GiB   | 40 GiB    | 4.6×            |

> 7B 每 token 计算：$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ 字节 = **0.5 MiB/token**。131K 上下文单请求 KV 就要 64 GiB——**一张 A100-80G 卡除去 14 GB 权重后放不下**，必须在 KV 上下功夫。

### 1.3 为什么必须缓存（而不是重算）

自回归解码每个新 token 前，若重算历史所有 K/V：第 $n$ 步要重新前向 $n$ 遍、显存小但算力爆炸（复杂度 $O(n^2)$），而**缓存后每一步只增量计算 1 个 token 的 K/V**，复杂度 $O(n)$。这是"用显存换算力"的根本交易——KV 缓存的显存成本，就是换取"每步只算一遍"的速度。

推理分两个阶段，KV 压力来源不同：

| 阶段               | 行为             | 显存/算力特征                                            |
| ---------------- | -------------- | -------------------------------------------------- |
| **Prefill（预填充）** | 一次性处理整段 prompt | 高算力（$O(S^2)$ 注意力）、单次把 $S$ 个 token 的 KV 全部落盘        |
| **Decode（解码）**   | 逐 token 生成     | KV 缓存只增不减（第 $n$ 步已缓存 $n$ 个 token），每步只算新 token 的 KV |

高并发服务中 Decode 阶段大家同时累积 KV，显存按 $B\times S$ 飞速膨胀——**优化的核心目标是 Decode 阶段的 KV 峰值**。

### 1.4 四种主流优化路线

| 路线   | 针对问题     | 代表技术                        | 省多少               |
| ---- | -------- | --------------------------- | ----------------- |
| 结构压缩 | KV 头数太多  | GQA / MQA                   | 省 $h/g$ ~ $h$ 倍   |
| 数值压缩 | 每元素 2 字节 | INT8 / FP8 量化               | 省 1/2（INT4 省 3/4） |
| 显存管理 | 分配碎片与浪费  | PagedAttention (vLLM)       | 吞吐提升 2~7×         |
| 内容裁剪 | 长历史冗余    | Sliding Window / 视觉裁剪 / H2O | 限制峰值              |

这四条路线**正交可叠加**（GQA + INT8 + PagedAttention + 窗口裁剪可以同时用），下面逐条展开。

---

## ② GQA / MQA：从源头减少 KV 头数（重点）

### 2.1 三种结构

| 结构  | Q 头数 | KV 头数 $n_{\text{kv}}$               | KV 显存相对值          | 代表模型                            |
| --- | ---- | ----------------------------------- | ----------------- | ------------------------------- |
| MHA | $h$  | $h$（每头独立）                           | 1                 | LLaMA1、GPT-2                    |
| GQA | $h$  | $g$（$g\mid h$，$h/g$ 个 Q 头共享 1 组 KV） | $g/h$ = $1/(h/g)$ | LLaMA2-70B、LLaMA3、Mistral、Qwen2 |
| MQA | $h$  | $1$（所有 Q 头共享 1 组 KV）                | $1/h$             | Falcon                          |

$$\text{KV 显存} \propto n_{\text{kv}} \quad\Rightarrow\quad \text{GQA 压缩率} = \frac{g}{h},\quad \text{MQA 压缩率} = \frac{1}{h}$$

**数值示例**：32 个 Q 头的模型，MHA 用 32 组 KV；换成 8 组 GQA 后 KV cache 从 32 份降到 8 份——**KV 显存下降 4 倍**（$h/g = 32/8 = 4$）；换成 MQA 则降 32 倍。对应到 1.2 账本：7B 模型 131K 上下文 KV 从 64 GiB（MHA）→ 16 GiB（GQA-8）→ 2 GiB（MQA）。

### 2.2 为什么共享 KV 质量损失小？

1. **注意力分布由 Q 主导**：$\text{softmax}(Q_i K^\top / \sqrt{d})$ 里每个头用的是**自己的** $Q_i$，共享 $K$ 只是让多个头"看同一批信息源"，每个头仍各取所需——压缩的是"信息源副本"，不是"查询能力"；
2. **KV 信息天然冗余**：相邻 Q 头的 KV 投影学出的内容高度相关，共享等价于对 KV 投影加了一个低秩约束，把冗余去掉；
3. **计算量几乎不变**：$QK^\top$ 仍是对每个 Q 头做 $n^2$ 次内积（训练时主要成本不变），省下的是**参数**（KV 投影）与**推理期的缓存和每步 KV 投影计算**。

经验结论：单组 KV（MQA）信息容量太窄，大模型掉点明显；8 组（GQA-8）容量足够，质量接近 MHA——所以 LLaMA2-70B / LLaMA3 / Mistral 全用 GQA 而非 MQA。

### 2.3 源码实现：手写 MHA / GQA / MQA 的 KV 构建与显存对比

```python
import math
import torch

torch.manual_seed(0)
B, T, C = 2, 8, 512          # batch=2, seq=8, hidden=512
H, G, D = 8, 2, 64           # 8 个 Q 头，GQA 分 2 组，每头 64 维

q = torch.randn(B, T, C, dtype=torch.float16)          # Q 投影宽 C = H*D
kv_gqa = torch.randn(B, T, G * D, dtype=torch.float16) # GQA 投影宽 G*D
kv_mqa = torch.randn(B, T, D, dtype=torch.float16)     # MQA 投影宽 D

def reshape_to_heads(x, n_heads):
    """[B,T,C] -> [B,n_heads,T,d]，head 维度拆开"""
    B, T, _ = x.shape
    return x.view(B, T, n_heads, x.shape[-1] // n_heads).transpose(1, 2)

q_h = reshape_to_heads(q, H)                 # [B,8,T,64]  Q 头
k_g, v_g = reshape_to_heads(kv_gqa, G), reshape_to_heads(kv_gqa, G)  # [B,2,T,64]
k_m, v_m = reshape_to_heads(kv_mqa, 1), reshape_to_heads(kv_mqa, 1)  # [B,1,T,64]

def sdpa(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=-1) @ v

def kv_mib(seq, kv_heads, d_head=64, nbytes=2):
    """1 层、seq 个 token 的 KV 显存(MiB)：2 × n_kv × d_head × bytes × seq"""
    return 2 * kv_heads * d_head * nbytes * seq / 1024 ** 2

for name, kvh in [("MHA (h=8)", H), ("GQA (g=2)", G), ("MQA (g=1)", 1)]:
    print(f"{name}: KV 显存 = {kv_mib(4096, kvh):.1f} MiB  (seq=4096, 1 层)")
# MHA (h=8): KV 显存 = 8.0 MiB  (seq=4096, 1 层)
# GQA (g=2): KV 显存 = 2.0 MiB  (seq=4096, 1 层)
# MQA (g=1): KV 显存 = 1.0 MiB  (seq=4096, 1 层)

# 正确性：GQA 把 2 组 KV repeat 到 8 头，MQA 把 1 组 expand 到 8 头，输出形状一致
out_gqa = sdpa(q_h, k_g.repeat(1, H // G, 1, 1), v_g.repeat(1, H // G, 1, 1))
out_mqa = sdpa(q_h, k_m.expand(B, H, T, D), v_m.expand(B, H, T, D))
print("shapes:", out_gqa.shape, out_mqa.shape)
# shapes: torch.Size([2, 8, 8, 64]) torch.Size([2, 8, 8, 64])
```

> 注意真实的 KV 投影宽度：GQA 的 K/V 投影只输出 $g \times d_{\text{head}}$ 维（不是 $h \times d_{\text{head}}$），用 `repeat`/`expand` 展开回 $h$ 头再参与注意力——**显存省在"没存那份投影输出"，不是省计算**。上式里 $g=h$ 时重复次数为 1，GQA 与 MHA 逐位恒等；$g=1$ 时即 MQA——三种结构是同一方程的不同取值。

### 2.4 GQA 的完整前向公式

设 Q 有 $h$ 个头、KV 有 $g$ 组，第 $i$ 个 Q 头归属组 $\lfloor i / (h/g) \rfloor$，该组共享第 $\lfloor i/(h/g) \rfloor$ 份 K、V：

$$\text{head}_i = \underbrace{\text{softmax}\left(\frac{Q_i\, K^\top_{\lfloor i/(h/g) \rfloor}}{\sqrt{d}}\right)}_{\text{注意力权重全由各自的 }Q_i\text{ 决定}} \; V_{\lfloor i/(h/g) \rfloor}$$

$$\text{Attention}(Q,K,V) = \left[\text{head}_1; \dots; \text{head}_h\right] \cdot W_O$$

注意 $K$ 的索引是**组号**而非头号——这正是"组间独立、组内共享"的数学表达。

### 2.5 从 MHA 到 GQA 的工程演进（为什么最终停在 GQA）

1. **2017 Transformer（MHA）**——每头独立 KV，质量最好但 KV 缓存最贵；
2. **2019 MQA（Falcon、PaLM）**——KV 省到 $1/h$，但单组信息容量不足，大模型掉点明显；
3. **2022 GQA（LLaMA2 起）**——取中间：$g$ 组（工业界取 8），KV 省到 $g/h$，质量逼近 MHA；
4. 工程结论：**g=8 是质量和显存的甜点**，LLaMA2-70B / LLaMA3-8B-70B / Mistral-7B / Qwen2 全系采用。

---

## ③ KV 量化：FP8 / INT8 存储，省一半显存（重点）

### 3.1 为什么 KV 量化"不怎么掉点"

- **数值分布集中**：K、V 经过 LayerNorm + RoPE 或残差后，数值大多落在 $[-1, 1]$ 小范围、分布集中对称，几乎没有极端离群值（不像激活那样有少数大值），所以均匀对称量化误差小；
- **注意力对 KV 误差不敏感**：$\text{softmax}(QK^\top/\sqrt d)$ 是"选最大"的归一化过程，KV 上的微小扰动只会轻微移动注意力权重，最终输出归一化后误差被进一步平滑；
- **动态量化**：按每个 token 单独算 scale 反量化回 FP16 再算注意力，误差不累积。

**量化收益**：INT8/FP8 每元素 1 字节（vs FP16 的 2 字节）→ **省 50%**；INT4/FP4 每元素 0.5 字节 → **省 75%**。对 1.2 账本：7B 模型 32K 上下文 KV 从 16 GiB（FP16）→ 8 GiB（INT8）。

### 3.2 主流框架支持

- **vLLM**：`--kv-cache-dtype fp8`（及 `e5m2`/`e4m3` 两种 FP8 格式），配合 PagedAttention 在插入/读取时实时反量化；
- **TensorRT-LLM / SGLang**：同样支持 FP8/INT8 KV cache；
- **Llama.cpp**：`-ctk/-ctv q8_0` 等。

实现要点：① 量化粒度（per-token 或 per-channel）；② 存储 INT8，**读**的时候动态反量化回 FP16 与 FP16 的 Q 做 $QK^\top$；③ 用 `scaled_dot_product_attention` 或自定义 kernel 让反量化在内存带宽层面省（不把量化 KV 整体展开到显存）。

### 3.3 源码实现：手写 KV INT8 量化 + 注意力误差对比

```python
import math
import torch

torch.manual_seed(0)
B, H, T, D = 2, 4, 1024, 128

q = torch.randn(B, H, T, D, dtype=torch.float16) * 0.3
k = torch.randn(B, H, T, D, dtype=torch.float16) * 0.3   # 分布集中，贴近真实 K
v = torch.randn(B, H, T, D, dtype=torch.float16) * 0.3

def quant_int8(x):
    """对称 INT8 量化：per-token（每个 [B,H,T] 用各自的 scale）"""
    amax = x.abs().amax(dim=-1, keepdim=True)     # [B,H,T,1] 每个 token 一个 scale
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    qq = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return qq, scale                               # INT8 + scale（scale 可再存 FP32 或 FP16）

def dequant_int8(qq, scale):
    return qq.to(torch.float16) * scale            # 反量化回 FP16 参与计算

k_q, k_scale = quant_int8(k)
k_hat = dequant_int8(k_q, k_scale)
print("K 反量化 MSE:", (k - k_hat).float().pow(2).mean().item())
# K 反量化 MSE: 3.7e-06

def sdpa(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=-1) @ v

out_fp16 = sdpa(q, k, v)                       # 基准：FP16 KV
out_int8 = sdpa(q, k_hat, v)                   # 量化 KV 反量化后同样计算
err = (out_fp16 - out_int8).abs().mean().item()
rel = err / out_fp16.abs().mean().item()
print(f"注意力输出：绝对误差 {err:.2e}，相对误差 {rel:.2%}")
# 注意力输出：绝对误差 1.53e-05，相对误差 0.19%

# 显存收益验证（7B：32 层 32 头 d=128）
def kv_mib(seq, nbytes=2):
    return 2 * 32 * 32 * 128 * nbytes * seq / 1024 ** 2
seq = 32768
for name, b in [("FP16", 2), ("INT8/FP8", 1), ("INT4/FP4", 0.5)]:
    print(f"{name}: 7B@32K 上下文 = {kv_mib(seq, b):.0f} MiB")
# FP16: 7B@32K 上下文 = 16384 MiB
# INT8/FP8: 7B@32K 上下文 = 8192 MiB
# INT4/FP4: 7B@32K 上下文 = 4096 MiB
```

> 实测该示例 INT8 KV 注意力相对误差 0.19%——远低于模型本身对输出 logits 的容差，这就是"为什么不怎么掉点"的实验证据。生产上为保精度常用 FP8-E4M3（权重侧）与 FP8-E5M2（KV 侧），或 per-channel 量化进一步降误差。

### 3.4 两种量化粒度对比：per-token vs per-channel

per-token 给每个 token 一个 scale（沿 $d$ 维共享），per-channel 给每个 $d$ 通道一个 scale（跨 token 共享）。两者对分布集中的 KV 误差都极小，实测差距在一个量级内：

```python
import torch
k = torch.randn(2, 4, 1024, 128, dtype=torch.float16) * 0.3

def quant_int8(x):
    """per-token：每个 [B,H,T] token 一个 scale（沿 d 共享）"""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    qq = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return qq, scale

def quant_int8_channel(x):
    """per-channel：每个 d 通道一个 scale（跨 token 共享）"""
    amax = x.abs().amax(dim=-2, keepdim=True)          # [B,H,1,D]
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    qq = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return qq, scale

def err_of(fn):
    q, s = fn(k)
    return (k - q.to(torch.float16) * s).float().pow(2).mean().item()
print(f"per-token 量化 MSE: {err_of(quant_int8):.2e}")
# per-token 量化 MSE: 3.72e-06
print(f"per-channel 量化 MSE: {err_of(quant_int8_channel):.2e}")
# per-channel 量化 MSE: 5.34e-06
```

两种粒度都远低于可感知阈值；实际工程取舍看 kernel 与硬件：per-token 对已知 token 量级更贴合，per-channel 便于向量化但 scale 数量多 $d$ 倍。生产中（vLLM 等）默认 per-token 即可。

### 3.5 FP8 的两个子格式

| 格式 | 指数位/尾数位 | 数值范围 | 用途 |
| --- | --- | --- | --- |
| FP8-E4M3 | 4/3 | ~±448，精度相对好 | 权重、激活（范围可控） |
| FP8-E5M2 | 5/2 | ~±57344，范围大 | KV 缓存（跨 token 数值范围波动大，需更宽指数） |

KV 的量级会随上下文增长、不同 token 波动，所以 KV 侧倾向大指数的 E5M2；权重侧用 E4M3 保精度。INT8（对称、范围 ±127）是更简单的替代，无需硬件 FP8 支持。

---

## ④ PagedAttention / vLLM：解决显存碎片与浪费（重点）

### 4.1 传统推理的 KV 显存痛点

1. **预分配浪费**：服务端往往按请求的**最大长度**（如 4096 token）预分配连续 KV 显存。请求实际只生成了 256 token，则 93.75% 的槽位空着——一个 GPU 能服务的并发上限被"最坏长度"锁死；
2. **动态增长的复制**：若按需增长，扩容时要整体 `realloc` 搬到更大连续内存，产生显存碎片 + 拷贝开销；
3. **碎片化**：不同长度请求反复分配/释放，留下大量不可合并的小空洞。

### 4.2 分页思想：把 KV cache 切成固定大小的块

沿袭操作系统**虚拟内存**：物理内存按固定大小的页（page）分配，逻辑上连续的内容可以散落在物理上不连续的页里，由页表（block table）映射。vLLM 把 KV cache 切成 **block（默认 16 个 token 一块）**：

```text
请求 A 的 KV (逻辑连续):  [token0..15][16..31][32..47] ...
物理显存(按 block 分配):   [空][A块2][空][A块0][A块1] ...
                           ↑ 每次生成只追加一个物理块，需要时才分配空闲块
```

- **按需分配**：生成了多少 token 就占多少 block，不再预留最大长度；
- **共享**：同一 prompt 的多条不同生成（prefix caching / beam search）可共享同一批物理块（写时复制）；
- **连续批处理**：一个 GPU 上随时动态插入新请求、腾出完成请求的块，没有"一次调度一批、等整批结束"的空窗。

### 4.3 数值示例：固定预分配 vs 分页

```python
token_mib = 0.5          # 7B FP16：每个 token 的 KV = 0.5 MiB（见①账本）
gpu_free_mib = 66 * 1024 # A100-80G 减 7B 权重(~14GB)后剩 ~66 GiB 做 KV
block_size, max_len, avg_len, n_req = 16, 4096, 256, 32

fixed_mib  = n_req * max_len * token_mib                                # 固定预分配
blocks     = (avg_len + block_size - 1) // block_size                   # ceil(256/16)=16 块
paged_mib  = n_req * blocks * block_size * token_mib                    # 按需分页

print(f"固定预分配: {fixed_mib/1024:.0f} GiB，利用率 {avg_len/max_len:.1%}")
# 固定预分配: 64 GiB，利用率 6.2%
print(f"分页按需分配: {paged_mib/1024:.0f} GiB，省 {(1 - paged_mib/fixed_mib):.0%}")
# 分页按需分配: 4 GiB，省 94%
concurrent = gpu_free_mib / (n_req * blocks * block_size * token_mib)
print(f"同样 32 并发下已用 KV 归零为 4 GiB，内存空余折算可再塞 {concurrent:.0f} 倍请求")
```

### 4.4 为什么 vLLM 吞吐提升大

吞吐 = 同时可跑请求数 × 每请求生成速度。KV 显存利用率上去了，GPU 同一时刻能驻留的请求数**成倍增加**（上例 94% 的浪费被回收），每个请求仍以相同速度 decode，总体 token/秒自然大幅上升。论文实测：vLLM 相比 FasterTransformer/Orca 可达 **2~4×（长序列甚至 7×）吞吐**；配合 prefix caching 多轮对话/Agent 场景收益更大。

### 4.5 分页底层要维护什么（Block Table）

每个请求有一张 **block table**（逻辑块号 → 物理块号），注意力 kernel 拿到逻辑索引后查表定位物理块；新增 token 时只需追加一个物理块并更新表项：

```text
请求 R 的逻辑 KV:      #0        #1        #2
                    [t0..15] [t16..31] [t32..47]
                       ↓映射      ↓映射      ↓映射
物理显存块:           [P7]      [P2]      [P0]   <- 物理上不连续，查表即可
Block table R = [7, 2, 0]
```

- **共享**：两条请求共用同一 prompt 前缀时，block table 的若干项指向同一批物理块（引用计数 + 写时复制），共享前缀只需在显存里存一份；
- **回收**：请求结束立即归还物理块回空闲池，供后续请求复用——配合**连续批处理**（迭代级调度，每个 Decode 步都重新决定批成员），GPU 几乎没有空闲显存。

### 4.6 连续批处理 vs 朴素批处理

| 维度 | 朴素批处理（静态 batch） | 连续批处理 / 迭代级调度 |
| --- | --- | --- |
| 批成员 | 整批同进同出，最慢请求拖累全批 | 每步动态插入/移除请求 |
| 显存 | 按最大长度预分配整批 | 按实际长度分页按需 |
| 空闲 | 批内其余请求等最慢的 | 无空闲等待，始终抢占资源 |
| 适用 | 实验、离线推理 | 在线高并发服务（vLLM/SGLang/TensorRT-LLM） |

---

## ⑤ 窗口化与裁剪：限制峰值 KV（重点）

### 5.1 Sliding Window：只保留最近 W 个 token 的 KV

Mistral 的做法：注意力只看**过去 W 个** token（含自己），更早的 KV **直接丢弃**，不参与计算。KV cache 峰值被控制在 $W \times B \times L$，**不随总序列长度增长**——无限上下文也能跑，只是"记不住太早的事"。

实现上无需真删：生成新 token 时新 KV 覆盖最旧的槽位（环形缓冲），注意力只算最近 W 行。

**Mistral-7B 的混合做法**（SWA + 全局层）：共 32 层，其中前 4 层全局注意力（看全部历史）、后 28 层滑动窗口（$W=4096$）。全局层负责"记住早期全局信息"，窗口层负责细粒度局部——用约 $W/L$ 的 KV 预算拿到接近全注意力的质量。推理时 KV 总缓存由窗口层封顶控制，显存峰值可预测。

```python
import math, torch

B, H, T, D = 1, 2, 12, 16
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))

def sw_mask(n, W):
    """下三角 causal + 距离 < W → 每行最多 W 个可见位置"""
    pos = torch.arange(n)
    causal = pos[None, :] <= pos[:, None]
    window = pos[None, :] >= pos[:, None] - W + 1
    return causal & window

W = 4
mask = sw_mask(T, W)
print(mask.int().tolist())   # 每行最多 4 个 1（前 4 行因 padding 少于 W 个）
# [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#  [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], ... 第 5 行起只看最近 4 个]

def masked_sdpa(q, k, v, mask):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask.bool(), float("-inf"))
    return torch.softmax(s, dim=-1) @ v

out_sw = masked_sdpa(q, k, v, mask)
out_full = masked_sdpa(q, k, v, torch.tril(torch.ones(T, T, dtype=torch.bool)))
print("窗口内前 2 行(无截断)与全注意力一致:",
      torch.allclose(out_sw[:2], out_full[:2]))
# 窗口内前 2 行(无截断)与全注意力一致: True
print("第 6 行起(截断远历史)结果不同(符合预期):",
      not torch.allclose(out_sw[5:], out_full[5:]))
# 第 6 行起(截断远历史)结果不同(符合预期): True

T_total, W_keep = 10000, 4096
print(f"总序列 {T_total} token，窗口只保留 {W_keep} 个 → 峰值 KV 降到 {W_keep/T_total:.0%}")
# 总序列 10000 token，窗口只保留 4096 个 → 峰值 KV 降到 41%
```

### 5.2 视觉 token 裁剪：多模态里视觉 KV 才是大头

多模态模型（Qwen2-VL / Qwen3-VL、LLaVA 等）把图像切成一堆 patch 变成几万甚至几十万视觉 token，**视觉 KV 占 KV cache 的绝对大头**。Qwen3-VL 这类部署场景常用：

- **重要度裁剪**：用某层注意力分数/CLS 相关性给视觉 token 打分，丢弃低重要度的视觉 token（只保留图像中"被关注"的区域），可砍掉 50%~90% 视觉 KV；
- **合并 / 压缩**：相邻相似视觉 token 平均合并（类似 token merge）；
- 对文本 token 则**不做**裁剪（文本信息密度高、token 少），裁剪主要集中在图像区域——因此能显著拉低首字延迟与 KV 峰值。

### 5.3 Token Eviction：H2O（Heavy-Hitter Oracle）简述

H2O 观察到注意力分数服从"稀疏 + 重头（heavy hitter）"结构：少部分历史 token 拿到绝大多数注意力分数。做法是每个 KV 行维护一个**累计注意力分数**，分数高（重头）的 token 长期保留，新 token 进、最不重要的旧 token 出（evict）。相比 Sliding Window 的"只看最近"，H2O 允许**关键旧 token 幸存**，相同 cache 预算下 PPL 更低——代价是需要在线维护分数（较难完全 kernel 化）。

### 5.4 三种裁剪策略的关键差异

| 策略 | 保留什么 | 遗忘什么 | 实现 | 典型收益 |
| --- | --- | --- | --- | --- |
| Sliding Window | 最近 W 个 | 所有早期 token | 环形缓冲，kernel 友好 | KV 封顶，不随 S 增长 |
| H2O | 注意力分数最高的 | 低分 token | 在线维护分数 | 关键旧 token 幸存、低 PPL |
| 视觉裁剪 | 高重要度视觉 patch | 低重要度图像区 | 依赖视觉 saliency | 视觉 KV 省 50~90% |

工程上常**组合**：窗口层 + 少量全局层（Mistral）、视觉裁剪 + 文本窗口、H2O 只用于最难的极长上下文。

---

## ⑥ 无 KV 的路线：Mamba / 线性注意力，与 KV 的极致压缩

### 6.1 Mamba / 状态空间模型（SSM）

SSM 没有"每个历史 token 一份 K/V"的缓存，只有一个**固定大小的隐状态** $h_t$（如 16×d_model），新 token 只跟这个状态交互：

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t,\qquad y_t = C h_t$$

- **省显存**：缓存就是 $h_t$，大小与序列长度**无关**，理论上支持无限长上下文、推理显存平坦；
- **代价**：① 固定维度隐状态是"有损压缩"，长程精确回溯能力弱（对"拷回原文第 5 段的数字"这类任务弱）；② 生态不成熟：与 FlashAttention/ROPE/量化/投机解码等 CUDA 生态兼容度低，训练技巧（稳定并行扫描）门槛高。

### 6.2 线性注意力（Linear Attention）

用核技巧把注意力改写为**右结合**形式，把 $\sum K V$ 先做矩阵结合律一次，使**计算复杂度 O(n²)→O(n)** 且记忆状态固定：

$$\text{Attn}(Q,K,V) \approx \varphi(Q)\,(\varphi(K)^\top V)\,/\ \text{归一化}$$

状态矩阵 $\phi(K)^\top V$（$d\times d$）固定大小，等同"恒定显存的 KV"。代价：去掉 softmax 会损失部分表达能力，需要额外归一化、门控补偿。

### 6.3 混合架构（Mamba + Attention）：兼顾性价比

纯 SSM 长程精确回溯弱，纯 Transformer KV 贵——于是出现**混合模型**（Jamba / Zamba / Hymba）：大多数层用 Mamba（无 KV），穿插 1/8~1/4 比例的自注意力层。效果：

- 绝大部分层的 KV 成本为 0，KV cache 显著小于纯 Transformer、质量接近；
- 但注意力层的 KV 仍按 ④⑤ 的窗口/分页方式压缩，两者叠加后 CoT/RAG 场景更实用；
- 生态仍然是硬伤：Mamba 层的 CUDA kernel、并行扫描、投机解码支持度低于 FlashAttention 生态。

### 6.4 KV 的极致：量化 + 稀疏 + 分层

- **FP4 / 稀疏 2:4**：KV 用 INT4 + 50% 结构剪枝（DropKV / Dynamic Sparse），配合 FP8 权重可把 7B KV 在 131K 上下文压到 ~2 GiB；
- **分层策略**：浅层 KV 精度要求低（全局信息粗粒度），深层更敏感——可浅层 INT4、深层 INT8 混部。

一句话总结：**能在 Transformer 内解决的（GQA/量化/分页/窗口）优先做；Mamba/线性注意力是"换结构"的根治路线，但以精度与生态为代价**（完整的逐项收益/代价对比见 ⑦）。

---

## ⑦ 各技术对比总表

| 技术 | 压缩/收益 | 质量损失 | 实现成本 | 适用场景 |
| --- | --- | --- | --- | --- |
| GQA / MQA | KV 省 $h/g$ ~ $h$ 倍 | 很低（GQA）；MQA 偏高 | 低（改模型结构） | 新模型训练时就内置；推理侧静态 |
| KV INT8/FP8 | 省 50% | 很低（~0.1% 相对误差） | 中（量化 kernel + 反量化） | 短上下文也有收益；成熟框架已内置 |
| KV INT4 | 省 75% | 低~中（长上下文累积） | 中高 | 超长上下文 / 显存极紧 |
| PagedAttention | 吞吐 2~4×（显存利用率） | 无 | 中（框架级改造） | 高并发服务、动态长度场景 |
| Sliding Window | 峰值 KV 封顶不随 S 增长 | 中（遗忘远历史） | 低 | 长对话、流式；配合全局 token |
| 视觉 token 裁剪 | 视觉 KV 砍 50~90% | 低（保留重要区域） | 中 | 多模态长图/视频、RAG 图像 |
| H2O eviction | 同预算下更省/更低 PPL | 低~中 | 高（在线分数维护） | 极致长上下文，kernel 化难的场景 |
| Mamba / 线性注意力 | KV 恒定，S 任意 | 中高（表达力/生态） | 高（重构模型） | 研究前瞻；对显存有硬约束的端侧 |

> 实践组合拳：**GQA-8（结构）+ INT8（数值）+ PagedAttention（管理）+ 视觉裁剪（多模态内容）** 可叠加，互不冲突，这是主流 LLM 服务（vLLM）与多模态模型（Qwen3-VL 部署版）的标配。

---

## ⑧ 高频面试问答

**Q1：vLLM 为什么快？**
吞吐 = 并发度 × 单请求速度。vLLM 通过 PagedAttention 按需分配 KV 块、消除了预分配浪费（利用率从个位数%→接近 100%），并用连续批处理随时插单/释放，让 GPU 同时驻留的请求数成倍增加；同速度 decode 下总 token/秒自然大幅上升（论文报告 2~4× 吞吐，长序列更高）。

**Q2：PagedAttention 原理一句话？**
把逻辑上连续的 KV cache 切成固定 16-token 的物理块（block），按需分配、物理可散落、由 block table 映射，像操作系统虚拟内存；同一 prompt 的多个请求可共享物理块（写时复制），解决碎片与浪费。

**Q3：KV 量化为什么不怎么掉点？**
KV 数值经归一化后分布集中对称、无大离群值；且 softmax 的归一化性质对 KV 上的微小扰动不敏感；按 token 动态量化 + 反量化回 FP16 计算，误差不发散（实测注意力输出相对误差 ~0.1%）。

**Q4：GQA 和 KV 量化能叠加吗？**
能，且正交。GQA 减**数量**（头数），量化减**每元素字节**，两者相乘：GQA-8（÷4）× INT8（÷2）= KV 显存 ÷8。生产上 LLaMA-3 + vLLM FP8 KV cache 正是这么叠的。

**Q5：多模态的视觉 KV 怎么处理？**
裁剪为主：用注意力分数给视觉 patch 打分，丢弃/合并低重要度 token（Qwen3-VL、LLaVA 系可省 50~90% 视觉 KV）；其次视觉 KV 也走量化，颜色/纹理区域数值分布更集中、量化更安全。

**Q6：Sliding Window 和 H2O 有什么区别？**
SW 只看"最近 W 个"，实现简单（环形缓冲）但会遗忘关键早期 token；H2O 保留"注意力分数累计最高的 token"（heavy hitter），允许关键旧 token 幸存，同预算下更准但需在线维护分数、难 kernel 化。

**Q7：为什么 MQA 质量下降而 GQA 不会？**
共享 KV 把信息源容量从 h 份压到 1 份，单份信息容量太窄，学不下全部头需要的信息；压到 g 份（如 8）时容量仍充裕，且注意力由各自 Q 主导、共享只去冗余，故接近 MHA。

**Q8：7B 模型 128K 上下文 KV cache 大概多大？怎么压？**
FP16 下约 64 GiB（2×32层×32头×128维×2字节×131072）。GQA-8 → 16 GiB，再 INT8 → 8 GiB；配 PagedAttention 按需分配，一张 A100-80G 可服务，还不算权重余量。

---

## ⑨ 常见误区

1. **"KV cache 只在长上下文才有问题"**——错。短上下文高并发一样爆显存：并发 32、每请求 2K 上下文，7B 的 KV 就是 32 GiB。瓶颈是 $B\times S$，两者任一变大都会撑爆。
2. **"GQA 省的是计算量"**——只对推理的 KV 投影和缓存成立；训练时 $QK^\top$ 计算量不变，GQA 主要省参数与推理显存。
3. **"量化就是把 KV 存成 INT8 就完了"**——注意要 per-token/per-channel 动态量化并在**读取时反量化回 FP16** 参与计算；scale 本身也要管理，越界/零 scale 要处理，否则误差放大。
4. **"Sliding Window 能无损无限扩展上下文"**——会遗忘窗口外的信息，对需要精确长程引用（法律合同、代码文件）的任务明显掉点，常需配全局 token 或全局层。
5. **"vLLM 快是因为 kernel 优化"**——提速主因是**显存管理与调度**（PagedAttention + 连续批处理）带来的更高并发，而非单纯的算子快；算子层面 FlashAttention 是前置增益。

---

## ⑩ 自我检验

- [ ] 能写出 KV cache 显存公式，并口算出 7B 在 8K/32K/131K 上下文下的 KV 大小
- [ ] 能解释 GQA 共享 KV 为什么质量损失小（Q 主导 + 冗余 + 低秩约束）
- [ ] 能手写 MHA/GQA/MQA 的 KV 构建（repeat/expand）并说清 KV 投影宽度差异
- [ ] 能手写 KV INT8 量化/反量化，并说出为什么注意力输出误差小
- [ ] 能用一句话讲清 PagedAttention 的分页思想，及为什么 vLLM 吞吐高
- [ ] 能对比 Sliding Window 与 H2O 的取舍；知道视觉 token 裁剪的动机与做法
- [ ] 能说出 GQA、KV 量化、PagedAttention 三者如何正交叠加
- [ ] 能回答 8 个面试追问，并纠正 5 条常见误区
