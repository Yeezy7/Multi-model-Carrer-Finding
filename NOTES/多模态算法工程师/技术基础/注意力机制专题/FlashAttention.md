# FlashAttention：IO 感知的分块注意力

> 本模块索引见 [注意力机制专题详解](注意力机制专题详解.md)

## 一、定义与公式

### 1.1 标准注意力为什么"贵"

标准 scaled dot-product attention：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

前向必须依次完成四步，其中 $S = QK^\top/\sqrt{d_k}$ 和 $P = \text{softmax}(S)$ 都是 $n \times n$ 矩阵，**必须完整写出到 HBM（GPU 显存）再读回**：

$$S \xrightarrow{\text{写 HBM}} \text{softmax} \xrightarrow{\text{写 HBM}} P \xrightarrow{\text{读 HBM}} P V$$

当 $n$ 大时，这 $O(n^2)$ 次的 HBM 读写成为绝对瓶颈——**现代 GPU 上 attention 是访存受限（memory-bound）问题**，计算本身（FLOPs）很快，慢在搬数据。

### 1.2 Tiling：分块

把 $Q, K, V$ 按行切成块：

$$Q = \begin{bmatrix} Q_1 \\ Q_2 \\ \vdots \end{bmatrix}, \quad Q_i \in \mathbb{R}^{B_r \times d}, \quad K_j, V_j \in \mathbb{R}^{B_c \times d}$$

每次把一块 $Q_i$、一块 $K_j$、一块 $V_j$ 载入片上 **SRAM**（A100 约 20MB，快但小），在 SRAM 内完成 $Q_i K_j^\top$、softmax、加权求和，结果累加到输出块 $O_i$ 中。**$n \times n$ 的 $S$ 和 $P$ 从不落盘 HBM**，只在 SRAM 中按块出现。

### 1.3 Online softmax（流式 softmax）推导

标准 softmax 需要"两趟"：先扫一遍整行找最大值 $m_i$ 和分母 $l_i$，再算 $\exp(s_{ij} - m_i)$。分块后看不到整行，必须**单趟完成**，因此维护每行的 running max $m$ 与 running sum $l$。

处理第 $t$ 个列块 $S^{(t)}$（行下标省略）时：

**① 更新 running max：**

$$m^{(t)} = \max\left(m^{(t-1)},\; \max_j S^{(t)}_j\right)$$

**② 更新 running sum：**新块以新最大值 $m^{(t)}$ 为准，旧块的贡献需要乘以缩放因子 $\alpha = e^{m^{(t-1)} - m^{(t)}} \le 1$：

$$l^{(t)} = l^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}} + \sum_j e^{S^{(t)}_j - m^{(t)}}$$

**③ 更新输出累加器**（同样要乘 $\alpha$ 重缩放旧值）：

$$O^{(t)} = O^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}} + e^{S^{(t)} - m^{(t)}} V^{(t)}$$

**④ 全部块处理完后统一归一化：**

$$O = \frac{O^{(T)}}{l^{(T)}}$$

**正确性证明（为什么严格等价）**：softmax 对整行减同一常数结果不变：

$$\text{softmax}(s_i) = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}, \quad \forall m \in \mathbb{R}$$

分块过程中每一步只是"把当前看到的最大值作为减去的常数"，旧块先按旧常数 $m^{(t-1)}$ 算，之后用 $\alpha$ 修正到新常数 $m^{(t)}$；分母则是**加法结合律**（分块累加）。全程没有做任何近似，只改变了浮点舍入顺序——所以 FlashAttention 与标准实现**逐位等价（up to rounding）**。

> FlashAttention v1 与 v2 的差异：v1 每处理一块就除一次当前 $l^{(t)}$（引入更多非矩阵运算），v2 像上面第④步那样**把归一化推迟到最后统一做一次**，且精简了 max 更新，Tensor Core 利用率更高——两者数学等价。

## 二、核心原理

### 2.1 计算层次与"访存受限"

| 层次 | 容量 | 带宽 | 作用 |
|------|------|------|------|
| 寄存器 / SRAM | 20MB（A100） | 19 TB/s 量级 | 片内计算，极快 |
| HBM（显存） | 40~80GB | 2 TB/s 量级 | 存储权重、激活、中间量 |

把数据在 HBM ↔ SRAM 之间搬 1 次的时间 ≈ 在 SRAM 里算几百次浮点运算的时间。标准 attention 中每个元素 $S_{ij}$ 至少要搬 4 次（写 S、读 S、写 P、读 P），**搬数据的耗时远超计算耗时**，这就是 memory-bound 的本质。

### 2.2 IO 感知的核心思路

1. **一个 kernel 做完整个 attention**：不调用 `bmm → softmax → bmm` 三次 kernel（每次都要经过 HBM），而是把分块循环放进单一 CUDA kernel；
2. **分块驻留 SRAM**：每个块内的 $Q_i K_j^\top$、softmax、$P V$ 全部在 SRAM 完成；
3. **每个 Q/K/V 元素只从 HBM 读 $O(1)$ 次**，$O$ 只写一次；
4. **不需要存 $S$ 和 $P$**：中间显存从 $O(n^2)$ 降到 $O(n)$（只存 $O$ 和每行的 $m, l$）。

### 2.3 Backward：不存 S，重计算

标准实现 forward 存 $S$ 或 $P$ 供 backward 用（$O(n^2)$ 显存）。FlashAttention 的 backward **不存注意力矩阵**，只存 $O(n)$ 的 $m, l$，反向时重新算出 $S = QK^\top$ 再求梯度：

$$\frac{\partial L}{\partial Q} = \left(\frac{\partial L}{\partial P}\right) P^\top K, \quad \frac{\partial L}{\partial K} = Q^\top \left(\frac{\partial L}{\partial P}\right) P, \quad \dots$$

用**额外 FLOPs 换省掉的 HBM 带宽**——在访存受限场景下这个交换是赚的（计算是廉价的，搬数据是昂贵的）。

## 三、源码实现

### 3.1 标准实现（对比基准）

```python
import torch
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """标准实现：完整写出 S、P 两个 n×n 矩阵（都要落盘 HBM）"""
    d = K.shape[-1]
    S = Q @ K.transpose(-2, -1) / (d ** 0.5)   # [B, n, n]
    P = torch.softmax(S, dim=-1)               # [B, n, n]
    return P @ V
```

### 3.2 纯 torch 模拟 FlashAttention（分块 + online softmax）

```python
def flash_attention_sim(Q, K, V, B_c=64):
    """纯 torch 模拟 FlashAttention 的分块流程（v2 风格：最后统一归一化）。
    真实实现中每个块在 SRAM 内完成且不写回，这里用 torch 运算模拟同样的数学过程。
    """
    B, n, d = Q.shape
    O = torch.zeros_like(Q)                                  # 输出累加器 [B, n, d]
    l = torch.zeros(B, n, device=Q.device)                   # running sum（分母）
    m = torch.full((B, n), float('-inf'), device=Q.device)   # running max
    for j in range(0, n, B_c):                               # 沿列方向分块
        K_b = K[:, j:j+B_c]                                  # [B, B_c, d]
        V_b = V[:, j:j+B_c]                                  # [B, B_c, d]
        S = Q @ K_b.transpose(-2, -1) / (d ** 0.5)           # [B, n, B_c] 本块分数
        m_new = torch.maximum(m, S.max(dim=-1).values)       # ① 更新 running max
        alpha = torch.exp(m - m_new)                         # ② 旧块缩放因子 ≤ 1
        P = torch.exp(S - m_new.unsqueeze(-1))               # 本块 exp(score - m_new)
        O = O * alpha.unsqueeze(-1) + P @ V_b                # ③ 累加输出（暂不归一化）
        l = l * alpha + P.sum(dim=-1)                        # ② 累加分母
        m = m_new
    return O / l.unsqueeze(-1)                               # ④ 最后统一归一化
```

### 3.3 与标准实现对比验证

```python
torch.manual_seed(0)
B, n, d = 2, 256, 64
Q = torch.randn(B, n, d); K = torch.randn(B, n, d); V = torch.randn(B, n, d)

O_ref = standard_attention(Q, K, V)
O_flash = flash_attention_sim(Q, K, V, B_c=32)

print(torch.allclose(O_flash, O_ref, atol=1e-5))   # True：逐位等价
print((O_flash - O_ref).abs().max().item())        # ~1e-6 量级（仅浮点舍入差异）

# 大序列 + 不同块大小都不影响结果
n2 = 1024
Q2 = torch.randn(1, n2, 64); K2 = torch.randn(1, n2, 64); V2 = torch.randn(1, n2, 64)
for Bc in (16, 64, 512):
    ok = torch.allclose(flash_attention_sim(Q2, K2, V2, B_c=Bc),
                        standard_attention(Q2, K2, V2), atol=1e-5)
    print(f"B_c={Bc}: {ok}")   # 三行都是 True
```

### 3.4 小例子逐步观察 running max / sum

```python
torch.manual_seed(1)
Q = torch.randn(1, 4, 2); K = torch.randn(1, 4, 2); V = torch.randn(1, 4, 2)

B, n, d = Q.shape
O = torch.zeros_like(Q); l = torch.zeros(B, n); m = torch.full((B, n), float('-inf'))
for j in range(0, n, 2):                       # 每块 2 列，共 2 块
    S = Q @ K[:, j:j+2].transpose(-2, -1) / (d ** 0.5)
    m_new = torch.maximum(m, S.max(dim=-1).values)
    alpha = torch.exp(m - m_new)
    P = torch.exp(S - m_new.unsqueeze(-1))
    O = O * alpha.unsqueeze(-1) + P @ V[:, j:j+2]
    l = l * alpha + P.sum(dim=-1)
    print(f"块 {j//2}: m={m_new[0].tolist()}, l={l[0].tolist()}, alpha={alpha[0].tolist()}")
    m = m_new

O = O / l.unsqueeze(-1)
S_full = Q @ K.transpose(-2, -1) / (d ** 0.5)
print("最终 l 与整行 softmax 分母一致:", torch.allclose(l, torch.exp(S_full - S_full.max(-1, keepdim=True).values).sum(-1)))  # True
```

```python
# 输出示例（数值随随机种子不同）：
# 块 0: m=[0.28, 0.64, ...], l=[1.9, 2.3, ...], alpha=[0.0, 0.0, ...]
# 块 1: m=[1.02, 0.98, ...], l=[3.1, 3.4, ...], alpha=[0.42, 0.71, ...]
# 第一块时 alpha=0（m 从 -inf 起步）；第二块若出现更大 max，旧块贡献被重缩放
```

## 四、复杂度与显存分析

### 4.1 三个层面的账

| 指标 | 标准 attention | FlashAttention |
|------|---------------|----------------|
| FLOPs | 基准 | **略增**（backward 重算 $S$） |
| 算法复杂度 | $O(n^2 d)$ | $O(n^2 d)$（**不变**） |
| 中间显存 | $O(n^2)$（$S, P$ 落盘） | $O(n)$（只存 $O$ 与 $m, l$） |
| HBM 访问量 | $S, P$ 各读写一次：约 $4 \times Bhn^2$ 元素 | 每个 $Q/K/V$ 元素读 $O(1)$ 次，约 $O(Bhnd)$ |

> 记忆点：**FlashAttention 不省 FLOPs、不改变 $O(n^2)$ 复杂度，省的是 HBM 访问次数和中间显存**。因为 attention 是访存受限任务，省访存 ≈ 直接换算成 wall-clock 提速。

### 4.2 数值示例（B=2, h=32, n=8192, d=128, FP16）

- 单个 $S$ 矩阵：$2 \times 32 \times 8192^2 \times 2$ B $= 8.59$ GB；标准实现写读 S（17.2 GB）+ 写读 P（17.2 GB）≈ **35 GB HBM 流量**；
- $Q/K/V/O$ 各：$2 \times 32 \times 8192 \times 128 \times 2$ B $= 134$ MB，Flash 总流量 ≈ $4 \times 134 \approx 0.54$ GB（每元素约读一次）；
- 比值 ≈ **65 倍**。A100 HBM 带宽 2 TB/s：标准 ≈ 17.5 ms 纯搬数据，Flash ≈ 0.3 ms；
- FLOPs 侧：单次 attention 约 $2 \times 2 \times 32 \times 8192^2 \times 128 \times 2 \approx 2.2$ TFLOPs，A100 Tensor Core（312 TFLOPS）只需 ~7 ms——**计算远快于搬运**，这就是访存受限的量化证据。

### 4.3 什么时候收益最大

收益 ∝ attention 在总耗时中的占比：序列越长（$n$ 越大）占比越高，收益越大；短序列（$n \le 512$）FLOPs 占比高，Flash 收益有限。causal 掩码下还能进一步只算下三角块（省一半 FLOPs）。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 中间显存 $O(n^2) \to O(n)$，支持超长序列（64K+） | 仍是 $O(n^2)$ 计算量，序列无限长时天花板仍在 |
| HBM 访问减少 ~10~60 倍，端到端快 2~4 倍（v1）/ 6~9 倍（v2） | 需专用 CUDA kernel（flash-attn 库），普通 CPU 环境无法直接跑 |
| 数学严格等价，可无风险替换任意下游模型 | 实现复杂：分块、重计算、并行策略相互耦合 |
| backward 不存 $S$，省显存且不受 batch/head 数限制 | 对 FP16 精度敏感的极端场景仍需验证（累加顺序改变） |

## 六、与同类对比

| 维度 | FlashAttention | 稀疏注意力（SWA 等） | 线性注意力（Performer） |
|------|---------------|---------------------|------------------------|
| 计算复杂度 | $O(n^2 d)$ | $O(nw)$ | $O(nmd)$ |
| 中间显存 | $O(n)$ | $O(nw)$ | $O(nm)$ |
| 数值等价 | **精确**（重排求值顺序） | 精确（但剪掉了边） | **近似**（随机特征） |
| 质量损失 | 无 | 中（长程依赖丢失） | 较高（检索/拷贝任务） |
| 工程成熟度 | 极高（事实标准） | 低（硬件不友好） | 低（已被 SSM 取代） |
| 角色 | 把"精确 attention 跑得快" | 砍掉不需要的边 | 换掉 softmax 的指数核 |

**实践结论**：能跑精确 $O(n^2)$ 的地方优先 FlashAttention；稀疏与线性近似在大模型时代基本退场。

## 七、高频面试问答

**Q1：FlashAttention 快在哪？**
快在**少搬数据**，不是少算数。它把 attention 放进一个 kernel，Q/K/V 分块驻留 SRAM（tiling），$n \times n$ 的 S 和 P 从不落盘 HBM，HBM 流量从 $O(n^2)$ 降到 $O(n)$ 量级；而 attention 是访存受限任务，带宽省了速度就上来了。

**Q2：online softmax 为什么数学上等价？**
softmax 对整行减同一常数结果不变。分块时 running max 就是"当前看到的最大值"，旧块贡献用 $e^{m_{old}-m_{new}}$ 修正；分母是分块累加（加法结合律）；归一化推迟到最后一刻。全部是求值顺序重排，只改浮点舍入，不改数学结果。

**Q3：FlashAttention 省显存省在哪？**
省在**不存 $S$ 和 $P$**（$O(n^2)$），只存输出 $O$ 和每行 $m, l$（$O(n)$）；backward 也不存注意力矩阵，靠重计算。序列 64K 时 S 就有 16GB+（FP16），不省根本放不下。

**Q4：FlashAttention 的 backward 为什么敢不存 S？**
用 $m, l$ 重算 $S = QK^\top$。代价是额外 FLOPs，省的是 $O(n^2)$ 的 HBM 读写——访存受限场景下"多算少搬"是净赚。

**Q5：v1 和 v2 的区别？**
① v2 在序列维度也并行（v1 只在 batch/head 并行）；② v2 不再每块除一次 $l$，归一化推迟到最后统一做（减少非矩阵运算，提升 Tensor Core 利用率）；③ 更优的 warp 分配与 backward 优化。v2 比 v1 再快约 2 倍。

**Q6：FlashAttention 是近似方法吗？**
不是。它与 Performer（随机特征近似）、稀疏注意力（剪边）本质不同——只是工程重排求值顺序，数学上逐位等价（仅舍入差异），可以安全替换任何标准 attention。

**Q7：为什么大模型训练都开 FlashAttention？**
训练序列越来越长，attention 在总耗时中占比越来越高；它同时解决显存（放得下）和速度（跑得快）两个问题，且无精度代价。长序列场景（上下文 32K/128K）没有它基本不可行。

## 八、自我检验

- [ ] 能解释"attention 是访存受限任务"以及 SRAM/HBM 带宽差异
- [ ] 能徒手写出 online softmax 的四步更新公式（m、l、O、α）
- [ ] 能证明"减同一常数 softmax 不变"并说明它如何保证等价性
- [ ] 能用手写分块代码跑通并与标准 attention 对比（allclose）
- [ ] 能说出 v1 与 v2 的三个差异点
- [ ] 能算 HBM 流量：标准 vs Flash（n=8192, d=128 数值示例）
- [ ] 能说清 FlashAttention"不省 FLOPs、不减复杂度"这个易错点
- [ ] 能解释 backward 重计算的动机（多算少搬）
- [ ] 能回答 7 个面试追问
