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
    theta = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    m = torch.arange(max_len, dtype=torch.float32)           # 位置
    freqs = torch.outer(m, theta)                            # 角度 [L, D/2]
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

## 七、高频面试问答

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

## 八、自我检验

- [ ] 能写出 $R(\theta)$ 旋转矩阵与 $f_q(x,m) = R_m x$ 的定义
- [ ] 能写出复数表示 $z \to z e^{im\theta_i}$ 并说明与旋转矩阵同构
- [ ] 能独立完成复数与矩阵两种"内积只依赖 $m-n$"的推导
- [ ] 能解释范数保持（正交性）与 RMSNorm 的兼容性
- [ ] 能手写 precompute_freqs_cis + apply_rotary_pos_emb（含 rotate_half）
- [ ] 能说明 HF 官方实现（cos/sin 表 + rotate_half）与复数乘法的等价性
- [ ] 能解释 RoPE 外推崩溃的机制（相位 OOD）
- [ ] 能对比 RoPE vs 绝对编码 vs ALiBi 的机制与适用场景
- [ ] 能回答 8 个面试追问
