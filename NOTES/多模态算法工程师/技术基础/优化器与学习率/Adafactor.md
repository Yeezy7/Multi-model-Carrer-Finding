# Adafactor 优化器

> 本模块索引见 [优化器与学习率详解](优化器与学习率详解.md)

## 一、定义与更新公式（含推导）

### 1.1 动机：Adam 的显存账单

Adam 每参数维护 $m, v$ 两份 FP32 状态 = **每个参数多 8 字节**。LLaMA-7B 用 AdamW 光优化器状态就要 $7\times10^9 \times 8\,\text{B} \approx 56\,\text{GB}$——一张 A100 都不够装。大模型训练必须压缩优化器状态。

Adafactor（Shazeer & Stern, 2018）的核心思路：**不存完整的二阶矩矩阵，用低秩近似代替**。它在 T5 时代是 Google 大模型的训练主力。

### 1.2 二阶矩的低秩分解

对参数张量 $W \in \mathbb{R}^{n \times m}$，完整二阶矩 $V$ 是 $n\times m$ 矩阵。Adafactor 只维护两个向量：

$$\begin{aligned}
R_t &= \beta_{2t}\, R_{t-1} + (1-\beta_{2t})\, \text{mean}_j(G_t^2) & \text{（行统计，} n\text{ 维）} \\
C_t &= \beta_{2t}\, C_{t-1} + (1-\beta_{2t})\, \text{mean}_i(G_t^2) & \text{（列统计，} m\text{ 维）} \\
\end{aligned}$$

其中衰减率随步数变化（论文设计）：$\beta_{2t} = 1 - t^{-0.8}$（早期窗口短、响应快，后期窗口长、更平滑）。

用二者的**外积（rank-1 近似）**重建二阶矩：

$$\hat{V}_{ij} = \frac{R_i \cdot C_j}{\text{mean}(R)}$$

分母 $\text{mean}(R)$ 是归一化因子。为什么用外积？最小二乘角度看：在"只允许 rank-1 表示"的约束下，$\hat V_{ij} = R_i C_j / \text{mean}(R)$ 是使 $\sum_{ij}(\hat V_{ij} - E[G^2_{ij}])^2$ 最小的可分离近似（行均值 × 列均值的外积形式）。

**内存从 $O(nm)$ 降到 $O(n+m)$**。LLM 的权重矩阵绝大多数是 2D 的，二阶矩状态几乎省完；对 1D 参数（bias、embedding 的行向量）则退化为普通逐元素 EMA。

### 1.3 两个配套简化

1. **去掉一阶矩 $m$**（可选，默认不用）：实验显示对 Transformer 训练没有明显帮助——又省一半；
2. **相对学习率（relative step size）**替代固定 lr + $\epsilon$：

$$\alpha_t = \min\left(t^{-0.5},\; t \cdot \text{warmup}^{-1.5}\right), \qquad \theta \leftarrow \theta - \alpha_t \cdot \text{scale} \cdot u_t$$

- warmup 阶段 $\alpha_t \propto t$ 线性上升，之后 $\propto 1/\sqrt{t}$ 缓慢衰减（与 Transformer 论文的 inv-sqrt 调度同构）；
- 对 2D 参数还乘一个与矩阵宽度相关的缩放 $\text{scale}=\max(m, 1)$，使更新量不依赖参数的绝对尺度（单位无关性）；
- $u_t$ 是裁剪后的"单位化梯度"（见 1.4）——由于 $\alpha_t$ 本身随梯度尺度缩放，**无需手调 lr**。

### 1.4 更新与裁剪

$$\begin{aligned}
u_t &= \frac{G_t}{\sqrt{\hat V_t} + \epsilon_1} & \text{（}\epsilon_1 = 10^{-30}\text{，仅防除零）} \\
u_t &= \text{clamp}(u_t,\; -\rho,\; +\rho), \quad \rho = 1.0 & \text{（裁剪，防极端步长）} \\
\theta_t &= \theta_{t-1} - \alpha_t \cdot \text{scale} \cdot u_t
\end{aligned}$$

另设 $\epsilon_2 = 10^{-3}$ 作为 $\sqrt{\hat V_t}$ 的下限（`clamp_min`），防止二阶矩接近 0 时除出天文数字——它是"相对 lr"机制下替代 Adam 固定 $\epsilon$ 的数值护栏。

## 二、数学性质与直觉（几何解释）

### 2.1 低秩近似的几何意义

二阶矩 $E[G^2]$ 的完整信息是 $n\times m$ 个值，Adafactor 只保留"行方向平均水平"和"列方向平均水平"两个向量，外积恢复的矩阵秩为 1。直觉类比：

- 完整 V 像一张**灰度照片**（每个像素独立亮度）；
- Adafactor 只存**每行平均亮度**和**每列平均亮度**，再外积出照片——丢失了行内/列内的起伏细节，但保留了整体明暗结构。

对梯度结构近似可分离（$G^2_{ij} \approx r_i \cdot c_j$，比如 attention 权重矩阵 $QK^T$ 的梯度）的问题，这个近似几乎无损；对强相关结构（单元素梯度独大）则有偏差——表现为收敛略慢于 AdamW。

### 2.2 为什么外积要除以 mean(R)

外积 $R \otimes C$ 的尺度是 $R$ 尺度和 $C$ 尺度之积，会随维度放大（$n \cdot m$ 个元素的和 ≈ n·mean(R)·mean(C)）。除以 mean(R) 使重建矩阵的元素水平与真实 $E[G^2]$ 同量级，否则 $\hat V$ 整体偏大、步长被系统性压缩。

### 2.3 相对学习率：省掉 lr 的超参

Adam 的 lr 是"绝对步长"，需要针对任务调；Adafactor 的 $\alpha_t$ 是**无量纲的**（只依赖步数），乘以单位化梯度 $u_t$（|u|≤ρ）后，更新量自动与梯度尺度解耦。梯度整体放大 100 倍时，$\hat V$ 同步放大 $10^4$ 倍，$u$ 不变——**对梯度尺度变化免疫**，这是"免调 lr"的数学保证。

### 2.4 显存账本（面试常考的数字）

以 $W \in \mathbb{R}^{1024\times1024}$（100 万参数）为例：

| 项目 | 计算 | 显存 |
|------|------|------|
| 参数（FP32） | $10^6 \times 4\,\text{B}$ | 4 MB |
| Adam 状态 m+v | $2 \times 10^6 \times 4\,\text{B}$ | **8 MB** |
| Adafactor 状态 R+C | $(1024+1024) \times 4\,\text{B}$ | **8 KB** |

7B 模型（全部按 1024×1024 矩阵折算）：Adam 状态 52.2 GB vs Adafactor 状态约 0.05 GB——**省约 1000 倍**（严格说是从"参数量的 2 倍"降到"接近 0"）。

## 三、源码实现（手写 vs torch 官方，可直接运行）

> torch 官方目前没有 Adafactor（社区实现主要在 HuggingFace transformers 中）。下面实现严格对照论文（Shazeer & Stern, 2018）算法，并与官方 `torch.optim.AdamW` 在同一个回归问题上对比收敛与显存。

### 3.1 手写 Adafactor（矩阵参数 + 低秩分解分支）

多输出线性回归 $Y = XW$，参数是 $8\times4$ 矩阵，正好走"factored"（低秩）分支：

```python
import torch

torch.manual_seed(0)

# 数据: 多输出线性回归 Y = X·W  (参数是矩阵 → 适合低秩分解演示)
N, D, M = 64, 8, 4
X = torch.randn(N, D)
W_true = torch.randn(D, M)
Y = X @ W_true

def mse_loss(W):
    return ((X @ W - Y) ** 2).mean()

# ---- 手写 Adafactor (论文算法) ----
def adafactor_handmade(steps=10000, warmup=100, rho=1.0, eps1=1e-30, eps2=1e-3):
    torch.manual_seed(0)
    W = torch.zeros(D, M)               # 二维参数 → 走低秩分解分支
    R = torch.zeros(D)                  # 行统计 (D,)  ← 只存这两个向量!
    C = torch.zeros(M)                  # 列统计 (M,)
    for t in range(1, steps + 1):
        G = 2 * (X.T @ (X @ W - Y)) / N          # 解析梯度 dL/dW
        beta2 = 1 - t ** -0.8                    # 论文: β2_t = 1 - t^{-0.8}
        R = beta2 * R + (1 - beta2) * G.square().mean(dim=1)   # 行 EMA
        C = beta2 * C + (1 - beta2) * G.square().mean(dim=0)   # 列 EMA
        V_hat = torch.outer(R, C) / R.mean()     # 外积重建二阶矩 (rank-1)
        denom = torch.sqrt(V_hat + eps1).clamp_min(eps2)       # ε2 下限护栏
        u = (G / denom).clamp(-rho, rho)         # 单位化梯度 + 裁剪
        alpha = min(t ** -0.5, t * warmup ** -1.5)   # 相对学习率
        W -= alpha * max(M, 1) * u               # 按矩阵宽度缩放
        if t in (1, 10, 100, 1000, 5000, 10000):
            print(f"Adafactor t={t:5d}  loss={mse_loss(W).item():.4e}")
    return mse_loss(W).item()

l_ada = adafactor_handmade()
print(f"Adafactor 10000 步后 loss: {l_ada:.4e}")

# ---- 官方 AdamW 基线 ----
torch.manual_seed(0)
W = torch.zeros(D, M, requires_grad=True)
opt = torch.optim.AdamW([W], lr=0.1)
for _ in range(2000):
    opt.zero_grad()
    mse_loss(W).backward()
    opt.step()
print(f"AdamW 2000 步后 loss: {mse_loss(W).item():.4e}")
```

```text
Adafactor t=    1  loss=1.0960e+01
Adafactor t=   10  loss=8.2174e+00
Adafactor t=  100  loss=2.6939e-01
Adafactor t= 1000  loss=4.0514e-02
Adafactor t= 5000  loss=8.6911e-03
Adafactor t=10000  loss=4.4774e-03
Adafactor 10000 步后 loss: 4.4774e-03
AdamW 2000 步后 loss: 1.0267e-05
```

读表要点：Adafactor 一直在降（前 100 步是 warmup，之后靠 $1/\sqrt{t}$ 调度持续精修），但最终精度比 AdamW 低约 2.5 个数量级——**rank-1 近似丢掉了梯度方差的行内结构**，收敛略慢、终值略差。这正是论文里"Adafactor 与 Adam 精度相当、偶有差距"的复现，也是它作为"省显存方案"而非"精度方案"的定位。

### 3.2 显存账本（对比演示）

```python
D2, M2 = 1024, 1024
n = D2 * M2
print(f"参数 W({D2}x{M2}) 共 {n/1e6:.0f}M 个")
print(f"Adam 状态(m+v): {2 * n * 4 / 1024 / 1024:.1f} MB")
print(f"Adafactor 状态(r+c): {(D2 + M2) * 4 / 1024:.1f} KB")

n_mat = 7e9 / (D2 * M2)                      # 7B 参数按 1024x1024 矩阵折算的个数
ada_bytes = n_mat * (D2 + M2) * 4
print(f"7B 模型(全按 1024x1024 矩阵计): Adam 状态 {7e9*2*4/1024**3:.1f} GB, Adafactor 状态 {ada_bytes/1024**3:.3f} GB")
```

```text
参数 W(1024x1024) 共 1M 个
Adam 状态(m+v): 8.0 MB
Adafactor 状态(r+c): 8.0 KB
7B 模型(全按 1024x1024 矩阵计): Adam 状态 52.2 GB, Adafactor 状态 0.051 GB
```

**账本结论**：单矩阵省 1000 倍；7B 规模下 52.2 GB → 0.05 GB，优化器状态从"卡死一张 A100"变成"可以忽略"。这笔账就是 Adafactor 存在的全部理由。

### 3.3 1D 参数（bias 等）退化分支

`len(shape) < 2` 的参数（bias、标量）不适用低秩分解，直接退化为 RMSProp 式逐元素 EMA：

```python
import torch

torch.manual_seed(0)
p = torch.randn(8)                            # 1D 参数(如 bias)
v = torch.zeros(8)                            # 逐元素二阶矩(不低秩)
for t in range(1, 101):
    g = 2 * p                                 # 模拟梯度(平方损失的解析梯度)
    beta2 = 1 - t ** -0.8
    v = beta2 * v + (1 - beta2) * g.square()  # 逐元素 EMA
    alpha = min(t ** -0.5, t * 100 ** -1.5)   # 相对学习率(1D 分支 scale=1)
    p -= alpha * g / torch.sqrt(v).clamp_min(1e-3)
print(f"1D 分支 100 步后 |p| 均值: {p.abs().mean().item():.4f}")
```

```text
1D 分支 100 步后 |p| 均值: 0.0038
```

LLM 中这类参数占比极小，不影响整体显存账本。

## 四、超参与调参经验

| 超参 | 默认 | 说明 |
|------|------|------|
| relative_step | True | 用相对学习率（免调 lr）；想手动控 lr 时设 False |
| warmup_steps | 1~2% 总步数 | 相对 lr 的上升段长度 |
| scale_parameter | True | 按矩阵宽度 $\max(m,1)$ 缩放步长 |
| clip_threshold ρ | 1.0 | 单位化梯度的裁剪阈值，一般不动 |
| decay_rate | -0.8（即 $1-t^{-0.8}$） | 二阶矩衰减随步数变化 |
| beta1 | None | 默认无动量（省一半显存）；需要可开启 0.9 |
| eps | (1e-30, 1e-3) | ε1 防除零、ε2 为二阶矩下限 |

经验要点：

1. **何时用**：显存是瓶颈且模型以矩阵参数为主（LLM 预训练、长序列）→ Adafactor；显存宽裕追求精度 → AdamW；
2. **相对 lr 的行为**：它自带 inv-sqrt 式调度，**不要**再叠一层 cosine 调度（会双重衰减）；
3. 与梯度裁剪（grad clip）兼容：ρ 裁剪作用于单位化梯度，全局 grad clip 照常加；
4. 从 AdamW 切到 Adafactor 时，loss 曲线初期下降略慢属正常（warmup + 低秩近似），给足步数即可。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 优化器状态从"参数量的 2 倍"降到"≈0"（矩阵参数省约 1000 倍） | rank-1 近似丢失二阶矩的行内结构 → 收敛略慢、终值略差 |
| 相对学习率：免调 lr、对梯度尺度变化免疫 | 只对 2D 参数省显存；1D 参数退化（但占比小） |
| 默认无动量，再省一半 | 矩阵内梯度结构强相关（非可分离）时近似偏差大 |
| 数值稳定：裁剪 ρ + ε2 下限，混合精度友好 | 生态上 torch 无官方实现，需引 transformers |
| T5 等早期大模型验证充分 | 收敛曲线不如 AdamW 平滑（β2 随时间变化） |

## 六、与同类对比

| 维度 | Adam/AdamW | Adafactor | 8-bit Adam |
|------|-----------|-----------|-----------|
| 省显存方式 | 不省 | **算法近似**（低秩） | **数值近似**（量化） |
| 状态显存/参数 | 2× | ≈0（矩阵） | ≈0.5× |
| 与全精度 Adam 的数值差异 | — | 有（近似误差） | 几乎无 |
| 收敛速度 | 快 | 略慢 | 与 Adam 相同 |
| 免调 lr | 否 | 是（相对 lr） | 否 |
| 精度取向 | 精度优先 | 显存优先 | 显存优先且无损 |
| 典型场景 | 通用 | T5 等早期大模型 | LLM 微调、大 batch 训练 |

> 记忆：**Adafactor 省"结构"（低秩近似），8-bit Adam 省"精度"（压缩存储）**——前者改了算法，后者只改了存储。

## 七、高频面试问答

**Q1：Adafactor 怎么省显存？**
完整二阶矩 $V\in\mathbb{R}^{n\times m}$ 用行统计 $R$（$n$ 维）与列统计 $C$（$m$ 维）的外积 $\hat V = R\otimes C/\text{mean}(R)$ 近似，内存从 $O(nm)$ 降到 $O(n+m)$；再默认去掉一阶矩 $m$，状态几乎归零。

**Q2：为什么外积近似有效？**
LLM 权重多为 2D 矩阵，其梯度平方的统计结构近似可分离（$E[G^2_{ij}] \approx r_i c_j$）；外积是"可分离表示下的最小二乘最优"（行均值 × 列均值）。

**Q3：相对学习率是什么？**
$\alpha_t = \min(t^{-0.5}, t\cdot\text{warmup}^{-1.5})$：warmup 内线性上升、之后 $1/\sqrt{t}$ 衰减；乘以单位化梯度后更新量与梯度绝对尺度解耦，因此**免调 lr**。

**Q4：Adafactor 的显存账本？**
1024×1024 矩阵：Adam 8 MB → Adafactor 8 KB；7B 模型从 52.2 GB 降到约 0.05 GB（≈1000 倍）。

**Q5：Adafactor 和 8-bit Adam 的区别？**
Adafactor 是算法近似（低秩分解，数值与 Adam 有差异）；8-bit Adam 是存储压缩（逐 block 量化，反量化后计算与全精度 Adam 几乎一致）。前者省结构，后者省精度。

**Q6：为什么 Adafactor 收敛比 AdamW 慢？**
rank-1 近似丢失了二阶矩的行内/列内起伏，个别参数的缩放不准；β2 随时间变化也使统计更"毛糙"。实践上给足步数即可。

**Q7：什么时候不该用 Adafactor？**
显存宽裕、追求最优精度；参数以 1D 张量为主；矩阵梯度结构强相关（低秩近似失真大）。这些场景用 AdamW/8-bit Adam 更好。

## 八、自我检验 checklist

- [ ] 能写出 R/C 的递推公式与外积重建 $\hat V = R\otimes C/\text{mean}(R)$
- [ ] 能推导内存从 $O(nm)$ 到 $O(n+m)$
- [ ] 能写出相对学习率 $\alpha_t$ 公式并解释"免调 lr"
- [ ] 能手写 Adafactor 循环并在矩阵回归上验证收敛
- [ ] 能背出显存账本（8 MB vs 8 KB；7B：52.2 GB vs 0.05 GB）
- [ ] 能说清 Adafactor vs 8-bit Adam 的本质区别（结构 vs 精度）
- [ ] 知道 ρ 裁剪与 ε2 下限的作用
- [ ] 知道何时用 Adafactor、何时用 AdamW
- [ ] 能回答 7 个面试追问
