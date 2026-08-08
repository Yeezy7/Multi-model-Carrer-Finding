# RoPE：旋转位置编码（Rotary Position Embedding）

> 本模块索引见 [位置编码专题详解](位置编码专题详解.md)

## 一、定义与公式

RoPE（Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding"）把位置编码设计为对 **Q、K 向量的旋转**，使注意力分数天然只依赖相对位置，零参数。LLaMA、Qwen、GLM、Mistral、DeepSeek 等几乎全部开源大模型都在使用。

### 1.1 旋转矩阵的几何定义

把 $d$ 维向量按相邻两维配对：$(x_0, x_1), (x_2, x_3), \dots$，每一对看作二维平面上的一个点。位置 $m$ 对向量 $x$ 的作用是**把每一对按对应角度旋转**：

$$f_q(x, m) = R_m x, \qquad R_m = \text{diag}\big(R(\theta_0 m), R(\theta_1 m), \dots, R(\theta_{d/2-1} m)\big)$$

其中二维旋转矩阵：

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

对第 $i$ 对维度显式写出：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

**频率设计**（与正弦编码同源）：

$$\theta_i = 10000^{-2i/d}, \qquad i = 0, 1, \dots, d/2-1$$

- $i=0$：$\theta=1$，波长 $2\pi$（最高频，"秒针"）；
- $i = d/2-1$：$\theta \approx 10000^{-(d-2)/d}$，波长 $\approx 62832$（最低频，"时针"）。

### 1.2 作用位置（关键）

| 对象           | 是否旋转             | 原因                   |
| ------------ | ---------------- | -------------------- |
| Q            | 是（$f_q(x_q, m)$） | 参与相对位置计算             |
| K            | 是（$f_k(x_k, n)$） | 参与相对位置计算             |
| V            | **否**            | 内容聚合不需要位置            |
| 输入 embedding | **否**            | 位置信息只在 attention 内生效 |

## 二、核心原理与直觉

### 2.1 几何直觉：位置 = 旋转角度

每个配对维度 $i$ 是一根"指针"，位置 $m$ 让指针转过 $m\theta_i$ 弧度。**位置越靠后，转过的角度越大**；$d/2$ 根指针转速不同（秒针到时针），形成多分辨率相位指纹。

与加性编码（$e = x + p$）的本质区别：

| 方案         | 变换                | 作用对象         | 范数                          |
| ---------- | ----------------- | ------------ | --------------------------- |
| 加性（正弦/可学习） | $x \to x + p$     | 输入 embedding | 改变                          |
| RoPE       | $x \to R_m x$（乘性） | 仅 q/k        | **保持**（$\|R_m x\| = \|x\|$） |

### 2.2 复数表示：旋转 = 复数乘法（推导）

二维旋转与复数乘法同构：把 $(x_{2i}, x_{2i+1})$ 写成复数 $z_i = x_{2i} + \mathrm{i}\,x_{2i+1}$，则"旋转 $\theta_i m$ 弧度"等价于乘以单位复数 $e^{\mathrm{i}m\theta_i}$：

$$f(z_i, m) = z_i \cdot e^{\mathrm{i}m\theta_i} = z_i \big(\cos(m\theta_i) + \mathrm{i}\sin(m\theta_i)\big)$$

验证：$(a + \mathrm{i}b)(\cos\theta + \mathrm{i}\sin\theta) = (a\cos\theta - b\sin\theta) + \mathrm{i}(a\sin\theta + b\cos\theta)$，实部虚部恰好是旋转矩阵的作用——**复数乘法和旋转矩阵是同一运算的两种写法**。沿所有维度拼接，整个向量的旋转就是逐对复数乘法。实现上 `torch.view_as_complex` + 复数相乘正是利用了这一点。

### 2.3 核心推导：旋转后内积只依赖相对位置（面试必考）

**复数形式推导**。设 $q$ 在位置 $m$、$k$ 在位置 $n$，第 $i$ 对维度写为复数 $q = q_{2i} + \mathrm{i}q_{2i+1}$、$k = k_{2i} + \mathrm{i}k_{2i+1}$：

$$f_q(q, m) = q\,e^{\mathrm{i}m\theta_i}, \qquad f_k(k, n) = k\,e^{\mathrm{i}n\theta_i}$$

二维实数内积 = 复乘积的实部（$a \cdot b = \mathrm{Re}(a\,\bar{b})$）：

$$\begin{aligned}
\langle f_q(q,m), f_k(k,n) \rangle
&= \mathrm{Re}\Big( q\,e^{\mathrm{i}m\theta_i} \cdot \overline{k\,e^{\mathrm{i}n\theta_i}} \Big) \\
&= \mathrm{Re}\Big( q\,\overline{k} \, e^{\mathrm{i}(m-n)\theta_i} \Big)
\end{aligned}$$

**旋转后的内积只通过 $e^{\mathrm{i}(m-n)\theta_i}$ 依赖 $m-n$**——相对位置严格成立。

**矩阵形式推导**（等价视角）。$R_m$ 是正交矩阵：$R_m^\top = R_m^{-1} = R_{-m}$，且同频旋转可合并：$R_a R_b = R_{a+b}$。于是：

$$\langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{-m} R_n k = q^\top R_{n-m} k$$

同样只依赖 $n-m$。**含义**：

1. attention 分数 = 内容匹配 $q^\top k$ 经相对距离 $n-m$ 的旋转调制，位置信息 100% 是**距离函数**；
2. 距离越远，$q$、$k$ 相位错位越大，匹配天然衰减——与 ALiBi 的"近因偏置"异曲同工，但机制是**乘性余弦调制**而非减性线性惩罚；
3. 范数保持 → 数值稳定，与 RMSNorm 天然兼容，不改变注意力 softmax 的 scale 结构。

### 2.4 与绝对编码的关系（对比推导）

绝对加性编码下内积展开含四项：$x_i^\top x_j + x_i^\top PE_j + PE_i^\top x_j + PE_i^\top PE_j$，其中 $PE_i^\top PE_j$ 项是"位置-位置"耦合，模型必须**自己学会忽略**它（位置向量与内容向量的干扰）。

RoPE 的展开只有两项（复数推导可拆为内容项 $+$ 距离调制项），且**显式只依赖 $n-m$**——不需要模型学习，位置与内容的干扰项在数学上不存在。这是 RoPE 相比加性编码最本质的优势。

## 三、源码实现

### 3.1 手写完整实现（频率表 + 旋转）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freqs_cis(head_dim, max_len, base=10000.0):
    """预计算旋转角度表：返回 e^{i*m*theta}（复数）: [max_len, head_dim/2]"""
    # theta: 几何级数频率，[head_dim/2]
    theta = base ** (-torch.arange(0, head_dim, 2).float() / head_dim) # 频率
    m = torch.arange(max_len, dtype=torch.float32)           # 每个位置
    freqs = torch.outer(m, theta)                     # 位置 x 频率 ：角度 [L, D/2]
    # e^{i*angle} = cos + i*sin
    return torch.polar(torch.ones_like(freqs), freqs)        # [L, D/2] 复数

def rotate_half(x):
    """把每对相邻维度的两个分量"错位交换"：旋转矩阵的快速实现"""
    x1, x2 = x.chunk(2, dim=-1)        # 前一半 / 后一半
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs_cis):
    """标准旋转（逐对复数相乘），q/k: [B, L, H, D]，freqs_cis: [L, D/2]"""
    q = q.reshape(*q.shape[:-1], -1, 2)                     # [..., D/2, 2]
    k = k.reshape(*k.shape[:-1], -1, 2)
    q_c = torch.view_as_complex(q)                          # [..., D/2] 复数
    k_c = torch.view_as_complex(k)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)         # [1, L, 1, D/2]
    q_rot = torch.view_as_real(q_c * freqs_cis).flatten(-2) # 复数乘法=旋转
    k_rot = torch.view_as_real(k_c * freqs_cis).flatten(-2)
    return q_rot, k_rot

class RotaryAttention(nn.Module):
    """带 RoPE 的单头注意力：验证内积只依赖相对位置"""

    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # 位置频率表（零参数）
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(self.head_dim, max_len))

    def forward(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, self.freqs_cis[:L])
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
        return out

# 测试：验证"旋转保持范数"（正交性）与"相同内容内积只依赖距离"
torch.manual_seed(0)
d, heads = 16, 4
head_dim = d // heads
wq = torch.randn(head_dim, head_dim) / head_dim
freqs = precompute_freqs_cis(head_dim, 8)
x = torch.randn(head_dim)                      # 相同内容向量
q = wq @ x                                     # 一个 token 的 q 向量 [D]
q = q.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # [1,1,1,D]
q = apply_rotary_pos_emb(q, q, freqs)[0]       # 旋转到所有位置 → [1,8,1,D]
q0, q3 = q[0, 0, 0], q[0, 3, 0]
print(q0.shape)                                            # torch.Size([4])
print((q3.norm() - q0.norm()).abs().item() < 1e-5)         # True（旋转保持范数）
# 相对性数值验证：<q_i, q_j> 只依赖 |i-j|（平移不变）
dists = torch.stack([q[0, i, 0] @ q[0, j, 0]
                     for i in range(8) for j in range(8)]).view(8, 8)
print(torch.allclose(dists[3, 0], dists[7, 4], atol=1e-5))  # True（都是距离 3）
print(torch.allclose(dists[2, 5], dists[6, 3], atol=1e-5))  # True（都是距离 3）
```

### 3.2 与官方实现对比验证（HF LlamaRotaryEmbedding）

参考实现：transformers `LlamaRotaryEmbedding` + `rotate_half`。差别只在工程细节：① 官方用 `torch.outer(m, theta)` 后**先存角度表、前向时再生成复数**（支持动态长度）；② 旋转时官方对"对偶维度"应用 `rotate_half`（等价于相邻两维配对旋转的数学）。内嵌官方写法验证等价性：

```python
def rotate_half_official(x):
    """transformers/models/llama/modeling_llama.py rotate_half 官方实现"""
    x1 = x[..., : x.shape[-1] // 2]     # 前半
    x2 = x[..., x.shape[-1] // 2 :]     # 后半
    return torch.cat((-x2, x1), dim=-1)

# 等价性验证：官方"前半/后半对偶 + cos/sin 复制"与 3.1 的"相邻配对复数乘法"
# 是对同一旋转的两种内存布局。注意：rotate_half 布局要求输入先把
# 配对分量重排为"前半 = 各对第一分量、后半 = 各对第二分量"
torch.manual_seed(0)
D = 8
x = torch.randn(1, 1, 1, D)          # [B, L, H, D]，交错布局（相邻两维成对）
m = 3                                # 位置 3
theta = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
angle = torch.outer(torch.tensor([float(m)]), theta)  # [1, D/2]
freqs_c = torch.polar(torch.ones_like(angle), angle)  # e^{imθ} 复数

# 写法 A：相邻配对复数乘法（3.1 实现），输入/输出都交错
a = torch.view_as_real(
    torch.view_as_complex(x.reshape(1, 1, 1, -1, 2)) * freqs_c.unsqueeze(1)
).flatten(-2)

# 写法 B：官方 rotate_half（cos/sin 复制成整维），输入/输出半区对偶
x_b = x.reshape(1, 1, 1, -1, 2).transpose(-1, -2).reshape(1, 1, 1, D)  # 重排为对偶
cos = torch.cat((freqs_c.real, freqs_c.real), dim=-1).unsqueeze(1)      # [1,1,D]
sin = torch.cat((freqs_c.imag, freqs_c.imag), dim=-1).unsqueeze(1)
b = x_b * cos + rotate_half_official(x_b) * sin

# 把 B 的输出从"先所有第一分量、再所有第二分量"重排回交错布局再比较
b_perm = b.reshape(1, 1, 1, 2, -1).transpose(-1, -2).reshape(1, 1, 1, D)
print(torch.allclose(a, b_perm, atol=1e-6))   # True（数学等价，仅布局不同）
```

> **实现细节表（面试细节）**：
> 
> | 要点 | 说明 |
> | ------ | ------ |
> | 旋转对象 | 只有 q/k，v 与输入 embedding 不动 |
> | 配对方式 | 相邻两维 $(2i, 2i+1)$；代码常用"前/后半错位对偶"（rotate_half）等价实现 |
> | 频率维度 | 每对独立 $\theta_i$，几何级数 $10000^{-2i/d}$ |
> | 位置起点 | 因果模型位置从 0 计；跨片段拼接可平移/重置 |
> | 与加性编码 | 可叠加（如 GPT-NeoX），但不必要 |

## 四、性质分析

### 4.1 相对位置：严格成立

由 2.3 推导，旋转后内积 $\langle R_m q, R_n k\rangle = q^\top R_{n-m} k$ **只依赖 $n-m$**——相对位置不是"隐式可学"而是"数学强制"，这是 RoPE 与正弦编码（隐式相对）的根本区别。

### 4.2 外推性：差（需要插值技术）

旋转角 $\phi = m\theta_i$ 与位置线性增长。训练长度 $L$ 内模型见过的相位组合集是 $[0, L\theta_i)$：

1. $m > L$ 时，$d/2$ 维相位向量 $(m\theta_0, \dots)$ 是**从未出现的组合**——严格 OOD 输入；
2. 高频维（$i$ 小）已转过很多圈，$q,k$ 的相位错位模式与训练分布完全不同 → PPL 在 $L$ 处突然尖峰（典型：4K 训练的 LLaMA 直接跑 8K，PPL 暴涨、输出退化）；
3. 与 ALiBi 不同，RoPE 没有"距离值域全覆盖"保护——位置信息是乘性相位，训练中永远无法覆盖所有相位组合。

> 解决方案：PI / NTK / YaRN / LongRoPE 等长度扩展技术，见 [长度外推技术](长度外推技术.md)。

### 4.3 参数

- 位置参数 **0**：频率表是固定函数，不参与训练；
- 额外成本：无（旋转是逐元素乘加，GPU 上几乎免费）；
- 与因果掩码、GQA（分组多头）完全兼容——位置只作用于 q/k 的最后一维。

### 4.4 近因偏置与长程衰减

距离 $n-m$ 越大，各配对维度相位差 $|n-m|\theta_i$ 越大，内积中 $\cos((n-m)\theta_i)$ 项被调制得越剧烈，高频维迅速随机化 → 长距离匹配**统计上衰减**（近因偏置）。但注意这是"调制"而非硬截断：低频率段仍保留长程信息（波长 62832 的维度在 4K 长度内几乎不绕圈）。

## 五、优缺点

| 优点                             | 缺点                                  |
| ------------------------------ | ----------------------------------- |
| **严格相对位置**：内积只依赖 $m-n$（数学强制）   | 超训练长度外推差（相位 OOD，PPL 尖峰）             |
| 零参数、零额外存储                      | 需配合 PI/NTK/YaRN 才能扩展上下文             |
| 范数保持（$\|R_m x\| = \|x\|$），数值稳定 | 与 FlashAttention 融合需 kernel 支持（已解决） |
| 位置与内容无干扰项（对比加性编码的四项展开）         | 相对信息是"乘性调制"，长程衰减机制不如 ALiBi 直接       |
| 天然支持 2D/3D 坐标（M-RoPE）与任意分辨率    | 无法表达"绝对位置"（对需要绝对语义的任务有损）            |
| 与 RMSNorm、GQA、线性注意力兼容          |                                     |


## 六、与同类对比

### 6.1 RoPE vs 绝对编码（正弦/可学习）

| 维度   | 绝对编码           | RoPE              |
| ---- | -------------- | ----------------- |
| 注入方式 | embedding 相加   | q/k 乘性旋转          |
| 相对性  | 隐式（需模型学 $T_k$） | **严格**（数学强制）      |
| 展开项数 | 4 项（含位置-位置耦合）  | 内容项 + 距离调制项       |
| 范数   | 改变             | 保持                |
| 外推   | 弱/崩溃           | 弱（相位 OOD），但可用插值恢复 |

### 6.2 RoPE vs ALiBi

| 维度 | RoPE | ALiBi |
|------|------|-------|
| 机制 | 乘性旋转（余弦调制） | 减性线性惩罚 $-m|i-j|$ |
| 外推 | 差（需插值） | **好**（距离值域全覆盖） |
| 长程信息 | 低频段保留 | 惩罚线性增长，长程读取受限 |
| 代表模型 | LLaMA、Qwen、Mistral | BLOOM、MPT、GPT-NeoX |
| 兼容性 | 需 kernel 支持 | 实现最简单（一行代码） |

**面试观点**：RoPE 是"位置越远、匹配概率被调制得越随机"；ALiBi 是"位置越远、分数被罚得越多"。ALiBi 外推更省事，但长文本"远距信息读取"弱于 RoPE+插值；RoPE 与 FlashAttention 生态绑定更深，故成为主流。

## 七、从 1D 到 2D/3D：多模态扩展（2D-RoPE / 3D-RoPE）

> 前面讲的都是**文本 1D RoPE**（只有序列位置）。多模态模型要处理图像（2D 网格）和视频（3D 时空），RoPE 如何扩展？答案：**把隐藏维度分组，每组用不同轴的 position id 计算旋转角**——2D-RoPE 分 2 组（高、宽），3D-RoPE 分 3 组（时间、高、宽）。这就是 M-RoPE 家族（Qwen2-VL 等）的底层机制，详见 [M-RoPE.md](M-RoPE.md)。

### 7.1 为什么需要 2D/3D 位置编码

| 模态 | 几何维度 | 需要的位置信息 | 1D RoPE 的缺陷 |
|------|---------|--------------|---------------|
| 文本 | 1D 序列 | 第几个 token | 无（1D 天然够用） |
| 图像 | 2D 网格 | 第几行、第几列 | 只能给"展平后的第几个 patch"，丢失行列结构 |
| 视频 | 3D 时空 | 第几帧、第几行、第几列 | 完全无法表达时间先后 |

**为什么"展平后的序号"不够**：一张 16×16 patch 的图展平后，patch ①（左上角）和 patch ⑯（左上角旁边）的 1D 距离是 15，但空间距离只有 1——1D 编码无法区分"上下相邻"与"左右相邻"，空间关系被扭曲。

### 7.2 2D-RoPE：把维度分成两组

**核心思想**：把嵌入维度 $d$ 分成两段（各 $d/2$），第一段用**高度坐标 h** 的旋转角，第二段用**宽度坐标 w** 的旋转角：

$$\text{RoPE}_{2D}(x, (h, w)) = \begin{bmatrix} R_{h}(x_{[:d/2]}) \\ R_{w}(x_{[d/2:]}) \end{bmatrix}$$

- 图像中第 $r$ 行第 $c$ 列的 token → position id $(h, w) = (r, c)$；
- 文本第 $i$ 个 token → position id $(h, w) = (i, i)$（两组同频旋转，**严格退化为 1D RoPE**）；
- 两个轴各自满足"相对位置"性质：同轴内积只依赖该轴距离差。

**性质**：
1. 行轴与列轴解耦：两个 patch 的行差和列差分别由各自的维度组承载；
2. 旋转正交性在每个分组内保持，范数依然不变；
3. **外推友好**：坐标是"网格内局部坐标"（最大不过 16/32 量级），远比"全序列绝对位置"（可达 100K+）小——这就是动态分辨率下 2D-RoPE 比绝对 PE 更稳的原因。

### 7.3 2D-RoPE 源码实现（可运行）

```python
import torch
import torch.nn.functional as F

def apply_rotary_1d(x, ids, base=10000.0):
    """标准 1D RoPE（HF half-split 布局）：pair j = (x[j], x[j+d/2])"""
    d = x.shape[-1]
    freqs = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))  # d/2 对
    theta = ids[:, None].float() * freqs[None, :].to(x.dtype)                 # [T, d/2]
    cos, sin = theta.cos(), theta.sin()
    cos = cos[None, :, None, :]; sin = sin[None, :, None, :]                  # [1,T,1,d/2]
    x1, x2 = x.chunk(2, dim=-1)                                               # 实部/虚部
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

def apply_rotary_2d(q, h_ids, w_ids, base=10000.0):
    """
    2D-RoPE（官方同款结构）：共享一张完整频率表（与 1D 完全相同），
    高轴贡献前 d/4 对、宽轴贡献后 d/4 对，拼接后与 1D 全局布局一致。
    Args:
        q:     [B, T, H, d]，d 必须为 4 的倍数
        h_ids: [T] 高度坐标（行）
        w_ids: [T] 宽度坐标（列）
    """
    d = q.shape[-1]
    pairs = d // 2                                  # 完整频率表对数（与 1D 同一张表）
    freqs = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    p_per_axis = d // 4                             # 每轴 d/4 对

    def _theta(ids, start):
        return ids[:, None].float() * freqs[start:start + p_per_axis][None, :]

    # 高轴用前 d/4 对频率，宽轴用后 d/4 对频率，拼回完整 d/2 对
    theta = torch.cat([_theta(h_ids, 0), _theta(w_ids, p_per_axis)], dim=-1)  # [T, d/2]
    cos, sin = theta.cos(), theta.sin()
    cos = cos[None, :, None, :]; sin = sin[None, :, None, :]
    x1, x2 = q.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

# 测试 1：4×4 图像的 16 个 patch token
Hp = Wp = 4
rows, cols = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
h_ids = rows.flatten()   # [0,0,0,0,1,1,1,1,...] 行坐标
w_ids = cols.flatten()   # [0,1,2,3,0,1,...]     列坐标

q = torch.randn(1, 16, 8, 16)          # [B, T, heads, d]
q_rot = apply_rotary_2d(q, h_ids, w_ids)
print(q_rot.shape)                     # torch.Size([1, 16, 8, 16])

# 测试 2：相对位置性质——所有 token 用同一个内容向量，
# 则相似度只由"位置位移"决定：相同位移 → 严格相等，不同位移 → 不同
# （base 用 10 而非 10000，让旋转角更大，便于观察差异）
x = torch.randn(1, 8, 16)                          # 一份内容 [B,H,d]
q = x.unsqueeze(1).expand(-1, 16, -1, -1)          # 复制到 16 个 token
q_rot = apply_rotary_2d(q, h_ids, w_ids, base=10.0)

def sim(a_idx, b_idx):
    va, vb = q_rot[:, a_idx], q_rot[:, b_idx]
    return F.cosine_similarity(va.flatten(1), vb.flatten(1)).item()

d1 = sim(0, 5)    # (0,0)→(1,1)：位移 (+1,+1)
d2 = sim(10, 15)  # (2,2)→(3,3)：位移 (+1,+1)，应与 d1 严格相等
d3 = sim(0, 4)    # (0,0)→(1,0)：位移 (+1,0)
d4 = sim(0, 1)    # (0,0)→(0,1)：位移 (0,+1)
print(f"位移(+1,+1): {d1:.6f} vs {d2:.6f}（应严格相等，只依赖相对位移）")
print(f"位移(+1,0):  {d3:.6f} | 位移(0,+1): {d4:.6f}（应互不相同）")

# 测试 3：文本退化等价——(i,i) 的 2D-RoPE == 1D-RoPE（核心性质）
ids = torch.arange(8)
q2 = torch.randn(1, 8, 4, 16)
r1 = apply_rotary_1d(q2, ids)
r2 = apply_rotary_2d(q2, ids, ids)
print(f"文本退化等价最大误差: {(r1 - r2).abs().max().item():.2e}")  # ≈0
```

### 7.4 3D-RoPE（M-RoPE）：把维度分成三组

**核心思想**：在 2D 基础上再加时间轴，三个轴**共享同一张完整频率表**（与 1D 完全相同），各轴只贡献一段频率，按 `mrope_section` 顺序拼接（例如 Qwen2-VL-7B 的 `[16, 24, 24]`，表示 d/2=64 对频率中时间轴贡献前 16 对、高轴 24 对、宽轴 24 对）：

$$\text{MRoPE}(x, (t, h, w)): \quad \theta = \begin{bmatrix} t \cdot \omega_{0{:s_t}} \\ h \cdot \omega_{s_t{:s_t+s_h}} \\ w \cdot \omega_{s_t+s_h{:}} \end{bmatrix}, \quad x' = \text{Rot}(\theta, x)$$

其中 $\omega$ 是完整频率表。**这个"各轴切一段、拼回完整表"的设计保证了文本 (i,i,i) 严格退化为 1D RoPE**。

**三种模态的 position id 分配**：

| 模态 | 时间 id t | 高度 id h | 宽度 id w |
|------|----------|----------|----------|
| 文本第 i 个 token | i | i | i |
| 图像第 r 行第 c 列 | 常数（单帧） | r | c |
| 视频第 f 帧第 r 行第 c 列 | f | r | c |

**三个关键性质**：
1. **文本严格退化**：$(i,i,i)$ 三段的 $\theta$ 拼起来恰好等于 $i \cdot \omega$ 全表 → 等价 1D RoPE（文本能力零损失）；
2. **图像 = 单帧视频**：时间 id 固定，只剩 2D 行为；
3. **位置 id 数值小 → 外推收益**：图像/视频 token 按局部坐标编号而非全局序列号，Qwen2-VL 16K 训练长度可外推到 80K 推理。

**源码**（与官方同结构：每轴取完整频率表的一段切片，拼回全局 theta 后统一旋转；完整实现见 [M-RoPE.md](M-RoPE.md)）：

```python
def apply_rotary_3d(q, t_ids, h_ids, w_ids, section, base=10000.0):
    """
    3D-RoPE（M-RoPE）：三轴共享完整频率表，各取一段拼接（官方同款结构）
    Args:
        q:       [B, T, H, d]
        t/h/w_ids: [T] 各轴 position id
        section: 三元组（频率"对"数），如 [16, 24, 24]，满足 2*sum(section)=d
    """
    d = q.shape[-1]
    s_t, s_h, s_w = section
    assert 2 * (s_t + s_h + s_w) == d, "section 必须满足 2*sum(section)=d"

    freqs = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))  # 完整表 d/2 对

    def _theta(ids, start, n):
        return ids[:, None].float() * freqs[start:start + n][None, :]

    # 时间轴贡献前 s_t 对、高轴 s_h 对、宽轴 s_w 对，拼回完整 d/2 对
    theta = torch.cat([
        _theta(t_ids, 0, s_t),
        _theta(h_ids, s_t, s_h),
        _theta(w_ids, s_t + s_h, s_w),
    ], dim=-1)                                    # [T, d/2]，与 1D 全局布局一致
    cos, sin = theta.cos(), theta.sin()
    cos = cos[None, :, None, :]; sin = sin[None, :, None, :]
    x1, x2 = q.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

# 测试：4 帧视频，每帧 2×2 patch，共 16 个 token（section=[2,3,3]，d=16）
section = [2, 3, 3]
d = 2 * sum(section)
q = torch.randn(1, 16, 8, d)
f_ids = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])  # 帧 id
h_ids = torch.tensor([0,0,1,1]*4)                             # 行
w_ids = torch.tensor([0,1,0,1]*4)                             # 列
q_rot = apply_rotary_3d(q, f_ids, h_ids, w_ids, section)
print(q_rot.shape)   # torch.Size([1, 16, 8, 16])

# 文本退化等价：(i,i,i) 的 3D-RoPE == 1D-RoPE
ids = torch.arange(16)
q3 = torch.randn(1, 16, 4, 16)
r3 = apply_rotary_3d(q3, ids, ids, ids, section)
r1 = apply_rotary_1d(q3, ids)
print(f"3D 文本退化等价最大误差: {(r1 - r3).abs().max().item():.2e}")  # ≈0
```

### 7.5 1D / 2D / 3D RoPE 对比总表

| 维度 | 1D-RoPE | 2D-RoPE | 3D-RoPE（M-RoPE） |
|------|---------|---------|------------------|
| 轴数 | 1（序列） | 2（高、宽） | 3（时间、高、宽） |
| 维度分组 | 不分组 | 2 组（各 d/2） | 3 组（mrope_section 配置） |
| 文本 position id | i | (i, i) | (i, i, i) |
| 图像 position id | 展平序号 | (行, 列) | (常数, 行, 列) |
| 视频 position id | 不支持 | 不支持 | (帧, 行, 列) |
| 文本退化等价 | — | 严格等价 1D | 严格等价 1D |
| 相对位置 | 序列距离 | 行差/列差 | 帧差/行差/列差 |
| 代表模型 | LLaMA、Qwen3 文本 | Qwen2-VL/InternVL 的 ViT | Qwen2-VL/2.5-VL/3-VL LLM 侧 |
| 外推特点 | 相位 OOD | 坐标局部、较稳 | 坐标局部、16K 训 80K 用 |

**面试记忆点**：2D/3D 不是新的位置编码，而是"**把同一个 RoPE 机制按轴分组重复使用**"——维度切给几个轴，就用几组旋转角。文本永远退化为 1D，所以纯文本能力不受影响。

## 八、高频面试问答

**Q1：RoPE 一句话原理？**
把 q/k 每对相邻维度看成二维平面上的点，位置 $m$ 把它们各自旋转 $m\theta_i$（复数视角即乘以 $e^{im\theta_i}$）；利用旋转矩阵正交性与可合性，内积 $\langle R_m q, R_n k\rangle = q^\top R_{n-m}k$ 只依赖相对距离——零参数、严格相对位置。

**Q2：为什么旋转后内积只依赖相对位置？**
$R_m^\top R_n = R_{-m}R_n = R_{n-m}$（正交 + 同频可合）；复数视角 $q e^{im\theta} \cdot \overline{k e^{in\theta}} = q\bar{k}e^{i(m-n)\theta}$，实部只含 $m-n$。

**Q3：为什么只旋转 q/k 而不旋转 v？**
位置信息只在"匹配"（q·k 内积）时有用；v 只做内容聚合，旋转 v 会被 attention 权重线性组合掉，无意义且增加计算。

**Q4：rotate_half 为什么是"后半取负拼接前半"？**
$(a+ib)e^{i\theta} = (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)$。若向量前半存 $a$、后半存 $b$（对偶布局），旋转 = $x \cdot \cos + \text{rotate\_half}(x) \cdot \sin$，其中 $\text{rotate\_half} = [-b, a]$。与"相邻配对复数乘法"数学等价，只是维度布局不同。

**Q5：RoPE 为什么不能直接外推？**
旋转角 $m\theta_i$ 超出训练区间后，所有维度的相位组合是未见过的 $d/2$ 维向量（严格 OOD），高频维早已多圈，注意力分布漂移 → PPL 尖峰。需 PI/NTK/YaRN 插值或微调。

**Q6：RoPE 与绝对编码的本质区别？**
绝对编码是加性注入、位置与内容在 embedding 层混合（内积展开 4 项，位置-位置耦合项污染注意力）；RoPE 是乘性、只在 q/k 上、内积展开后位置项自动退化为 $e^{i(m-n)\theta}$——位置信息"数学强制"为距离函数，模型无需学习。

**Q7：RoPE 的旋转会改变向量范数吗？为什么不改？**
不改变。$R_m$ 是正交矩阵（$R^\top R = I$），$\|R_m x\|^2 = x^\top R_m^\top R_m x = \|x\|^2$。因此与 RMSNorm 兼容、数值稳定，注意力 scale 结构不受位置影响。

**Q8：实际部署中 RoPE 如何与 FlashAttention 结合？**
位置频率表预计算为 cos/sin，前向把 q/k 旋转完再进 FA kernel（旋转是内存受限的轻量算子，FlashAttention 官方已把 RoPE 融入 kernel）。LLaMA 系推理引擎（vLLM、TensorRT-LLM）均有融合实现。

**Q9：2D-RoPE 和 1D-RoPE 的关系？**
2D-RoPE 把隐藏维度分成两组，分别用 (行, 列) 坐标旋转；文本 token 用 (i, i) 时两组同频，严格退化为 1D-RoPE。图像场景下它能区分"上下相邻"与"左右相邻"，而展平的 1D 序号无法做到。

**Q10：为什么 Qwen2-VL 的 ViT 要用 2D-RoPE 而不是绝对位置编码？**
动态分辨率输入下 token 网格数量可变（从 4 到 16384），绝对 PE 的固定网格假设失效；2D-RoPE 用"网格内局部坐标"旋转，坐标值域小（行/列不超过几十），对任意分辨率输入都稳定，且零参数。这是动态分辨率能成立的位置编码前提。

**Q11：M-RoPE 的 (t,h,w) 三个轴是怎么分配的？**
隐藏维度按 mrope_section（如 Qwen2-VL 的 [16,24,24]，共 128 维）切成三段：前 16 维对（32 维）用时间 id 旋转、中间 24 对用高度、后 24 对用宽度。文本 (i,i,i) 退化 1D；图像时间轴固定为常数；视频用帧号。详细推导见 M-RoPE.md。

## 九、自我检验

- [ ] 能写出 $R(\theta)$ 旋转矩阵与 $f_q(x,m) = R_m x$ 的定义
- [ ] 能写出复数表示 $z \to z e^{im\theta_i}$ 并说明与旋转矩阵同构
- [ ] 能独立完成复数与矩阵两种"内积只依赖 $m-n$"的推导
- [ ] 能解释范数保持（正交性）与 RMSNorm 的兼容性
- [ ] 能手写 precompute_freqs_cis + apply_rotary_pos_emb（含 rotate_half）
- [ ] 能说明 HF 官方实现（cos/sin 表 + rotate_half）与复数乘法的等价性
- [ ] 能解释 RoPE 外推崩溃的机制（相位 OOD）
- [ ] 能对比 RoPE vs 绝对编码 vs ALiBi 的机制与适用场景
- [ ] 能讲清 2D-RoPE 的维度分组思想与文本退化等价性
- [ ] 能写出 2D-RoPE 的手写实现（双轴 position ids + 分组旋转）
- [ ] 能讲清 3D-RoPE（M-RoPE）的三轴分配规则与 mrope_section 含义
- [ ] 能解释"坐标局部 → 位置 id 小 → 利于外推"的链路
- [ ] 能回答 11 个面试追问
