# SwiGLU 激活函数（LLM FFN 标配）

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

SwiGLU（Shazeer 2020，GLU Variants Improve Transformer）是现代大模型（LLaMA、Mistral、Qwen、Gemma）前馈网络的标准激活结构。它把"激活函数"从**单输入单输出**升级为**双输入门控**：

$$\text{SwiGLU}(x, W_1, W_2, W_3) = \underbrace{\text{SiLU}(x W_1)}_{\text{门分支 gate}} \otimes \underbrace{(x W_2)}_{\text{数据分支 up}} \cdot W_3^{\top}$$

| 符号        | 含义              | 形状                  |
| --------- | --------------- | ------------------- |
| $x$       | 输入（上一层输出）       | $(B, S, d)$         |
| $W_1$     | 门投影矩阵           | $(d, d_{ff})$       |
| $W_2$     | 数据（up）投影矩阵      | $(d, d_{ff})$       |
| $W_3$     | 输出（down）投影矩阵    | $(d, d_{ff})$ 转置后投影 |
| $\otimes$ | 逐元素相乘（Hadamard） | —                   |

**GLU 通用定义**（Dauphin et al. 2017）：

$$\text{GLU}(x, W, V, b, c) = \sigma(xW + b) \otimes (xV + c)$$

SwiGLU 就是"门函数换成 SiLU"的 GLU 变体。

> **记忆点**：SwiGLU 不是单点的函数 $f(x)$，而是一个**门控结构**：分支 A 算出 0~1 左右的"阀门开度"（SiLU 门），分支 B 是"水管里的水"（线性变换），两者逐元素相乘控制流量。**门与数据来自两组独立的投影**，这就是它比"单路激活"强的根本原因。

## 二、门控原理

### 2.1 为什么门控优于单路激活

单路激活（如 GELU FFN）对**同一个**线性输出 $xW$ 施加非线性，信息与门绑死；门控结构让**两组独立变换**互相调节：

```text
单路激活:  y = f(xW)                     # 门和数据是同一个量
门控结构:  y = gate(xW₁) ⊗ data(xW₂)     # 门和数据解耦，各自优化
```

直观类比：单路激活是"一个人决定自己的音量"（自己算门又算内容）；门控是"调音师控制乐队的音量"（门看整体，内容看本身）——门的输入 $xW_1$ 可以聚合**多个维度**的信息来决定每个通道开多大。

### 2.2 SwiGLU 与标准 FFN 的对比（重点）

```text
标准 FFN（BERT/ViT）:     Linear(d→4d) → GELU → Linear(4d→d)
SwiGLU FFN（LLaMA）:      Linear(d→8d/3) ──────────────┐
                          Linear(d→8d/3) → SiLU → ×  → Linear(8d/3→d)
```

| 维度    | 标准 FFN               | SwiGLU FFN                       |
| ----- | -------------------- | -------------------------------- |
| 中间维度  | $4d$                 | $8d/3 \approx 2.67d$             |
| 投影矩阵数 | 2 个（W_in, W_out）     | 3 个（W_gate, W_up, W_out）         |
| 参数量   | $4d^2 + 4d^2 = 8d^2$ | $3 \times \frac{8}{3}d^2 = 8d^2$ |
| 非线性   | 1 处（GELU）            | 1 处（SiLU 门）                      |
| 效果    | 基线                   | **各规模均优于基线**（论文实证）               |

**为什么参数量相同**：SwiGLU 比标准 FFN 多一个投影，但 LLaMA 把中间维度从 $4d$ 压到 $8d/3$，总参数恰好持平：

$$\underbrace{2 \times 4d \times d}_{\text{标准 FFN 两个矩阵}} = \underbrace{3 \times \frac{8}{3}d \times d}_{\text{SwiGLU 三个矩阵}} = 8d^2$$

### 2.3 GLU 家族

| 变体         | 门函数          | 公式                               | 代表模型                         |
| ---------- | ------------ | -------------------------------- | ---------------------------- |
| GEGLU      | GELU         | $\text{GELU}(xW_1) \otimes xW_2$ | T5、PaLM                      |
| **SwiGLU** | **SiLU**     | $\text{SiLU}(xW_1) \otimes xW_2$ | **LLaMA、Mistral、Qwen、Gemma** |
| ReGLU      | ReLU         | $\text{ReLU}(xW_1) \otimes xW_2$ | —                            |
| GeGLU 变体   | tanh 近似 GELU | —                                | GPT-2（部分实现）                  |

论文在 8 亿参数下测了 4 种 GLU 变体，SwiGLU 与 GEGLU 表现最好且接近，SwiGLU 因实现简单（sigmoid 比 erf 便宜）成为工业界事实标准。

## 三、源码实现

### 3.1 纯张量实现（含手动反向的 autograd.Function）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFunction(torch.autograd.Function):
    """自定义 SwiGLU：接收已投影的张量 a（门）与 b（数据），逐元素相乘"""

    @staticmethod
    def forward(ctx, a, b):
        s = torch.sigmoid(a)                 # 门开度
        ctx.save_for_backward(a, b, s)
        return a * s * b                     # SiLU(a) ⊗ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b, s = ctx.saved_tensors
        # d(SiLU(a))/da = s + a·s·(1-s)
        grad_a = grad_output * b * (s + a * s * (1.0 - s))
        grad_b = grad_output * (a * s)
        return grad_a, grad_b

a = torch.randn(4, 64, requires_grad=True)   # 门分支输出 xW_gate
b = torch.randn(4, 64, requires_grad=True)   # 数据分支输出 xW_up
y = SwiGLUFunction.apply(a, b)
y.sum().backward()
print(a.grad.shape, b.grad.shape)  # torch.Size([4, 64]) torch.Size([4, 64])

# 梯度校验
a0 = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
b0 = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
print("gradcheck:", torch.autograd.gradcheck(SwiGLUFunction.apply, (a0, b0)))
# gradcheck: True
```

### 3.2 完整 SwiGLU FFN 模块（LLaMA 风格，3 个线性层）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """LLaMA-2 风格的 SwiGLU FFN（对应 transformers 库 LlamaMLP）"""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        self.hidden_dim = d_ff if d_ff else int(2 * d_model * 4 / 3)  # 8d/3
        self.gate = nn.Linear(d_model, self.hidden_dim, bias=False)   # W_gate
        self.up = nn.Linear(d_model, self.hidden_dim, bias=False)     # W_up
        self.down = nn.Linear(self.hidden_dim, d_model, bias=False)   # W_down

    def forward(self, x):
        # x: (B, S, d) → (B, S, 8d/3)
        return self.down(F.silu(self.gate(x)) * self.up(x))

ffn = SwiGLUFFN(512)
out = ffn(torch.randn(2, 10, 512))
print(out.shape)  # torch.Size([2, 10, 512])

# 参数检查：总参数 ≈ 8d²（与标准 FFN 持平）
n_params = sum(p.numel() for p in ffn.parameters())
print(n_params)  # 2096640 = 3 × 512 × 1365 ≈ 8 × 512² = 2097152 ✓
```

### 3.3 与标准 FFN 的等价参数量验证

```python
import torch

d = 512
# 标准 FFN: 两个 4d 宽矩阵
std_params = 2 * (4 * d) * d
# SwiGLU: 三个 8d/3 宽矩阵
swiglu_params = 3 * (int(8 * d / 3)) * d
print(f"标准 FFN 参数: {std_params}")     # 2097152
print(f"SwiGLU 参数:   {swiglu_params}")  # 2096640（int 截断后的微小差值，实际取整对齐）
```

### 3.4 在完整 LLM 解码块中的用法（LlamaDecoderLayer 片段）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """LLaMA-2 风格的 SwiGLU FFN（与 3.2 相同，为独立运行重复定义）"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        self.hidden_dim = d_ff if d_ff else int(2 * d_model * 4 / 3)  # 8d/3
        self.gate = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.up = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.down = nn.Linear(self.hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class LlamaAttention(nn.Module):
    """简化版因果自注意力（仅示意 SwiGLU 所在完整前向路径）"""
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) + mask
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.wo(out)

class LlamaDecoderLayer(nn.Module):
    """LLaMA 解码层：Attention + SwiGLU FFN + 预归一化 + 残差"""
    def __init__(self, d, n_heads):
        super().__init__()
        self.attention = LlamaAttention(d, n_heads)
        self.ffn = SwiGLUFFN(d)                 # ← SwiGLU 在这里
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        h = x + self.attention(self.norm1(x), mask)   # 预归一化 + 残差
        return h + self.ffn(self.norm2(h))

layer = LlamaDecoderLayer(512, 8)
mask = torch.full((1, 1, 10, 10), float("-inf")).triu(1)
out = layer(torch.randn(2, 10, 512), mask)
print(out.shape)  # torch.Size([2, 10, 512])
```

## 四、深入分析

### 4.1 梯度分析

SwiGLU 的反向梯度同时流向三条路径：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \text{out}} \otimes b \cdot \text{SiLU}'(xW_1) \cdot x^{\top}, \quad \frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \text{out}} \otimes \text{SiLU}(xW_1) \cdot x^{\top}$$

- **门分支梯度**：受 $b$（数据分支）调制——数据分支输出小时，门几乎不更新（抑制无用维度的学习）；
- **数据分支梯度**：受 $\text{SiLU}(xW_1) \in (0, \infty)$ 调制——门关小时数据权重也几乎不更新；
- 这种"**互锁**"让两组矩阵自动分工：门决定"学不学"，数据决定"学什么"。

### 4.2 数值稳定性

1. SiLU 门无饱和死区：$xW_1 \to -\infty$ 时 SiLU→0（门关），输出→0 但梯度不恒 0；
2. 无 exp 上溢风险：sigmoid 的指数行为自然（与 SiLU 章节相同）；
3. **乘积量级**：gate ∈ (0, ∞)、up 无界，相乘后量级约等于单路 FFN，无需特殊缩放；
4. FP16 下稳定，LLaMA 全系列在 bf16 下直接训练。

### 4.3 计算与内存复杂度

| 项目        | 标准 FFN                                              | SwiGLU FFN                                                    |
| --------- | --------------------------------------------------- | ------------------------------------------------------------- |
| 矩阵乘 FLOPs | $2 \times 2 \cdot B \cdot S \cdot 4d^2 = 16B S d^2$ | $2 \times 3 \cdot B \cdot S \cdot \frac{8}{3}d^2 = 16B S d^2$ |
| 逐元素激活     | GELU（约 4 次 op）                                      | SiLU（1 exp + 1 mul）+ 1 次 hadamard                             |
| 激活内存      | $4d$ 宽激活                                            | $2 \times \frac{8}{3}d$ 宽激活（略少）                               |
|           |                                                     |                                                               |

**结论**：矩阵乘总量完全相同（这是参数量持平的另一面），SwiGLU 只多一个便宜的门分支线性层 + 逐元素乘，计算代价可忽略，收益来自门控表达。

### 4.4 为什么 LLM 全员选 SwiGLU（面试核心）

1. **效果**：Shazeer 2020 论文在 8B 规模内测所有 GLU 变体，SwiGLU/GEGLU 系统性优于标准 FFN（GELU、ReLU）；
2. **参数量可控**：$8d/3$ 中间维度让 3 矩阵与标准 FFN 2 矩阵参数持平（见 2.2 表）；
3. **实现简单**：sigmoid + 逐元素乘，kernel 简洁，FP16/bf16 稳定；
4. **社区正反馈**：LLaMA 开源后成为新基线，Mistral/Qwen/Gemma 全线跟进，生态（KV 缓存、kernel 优化）围绕 SwiGLU 演进。

## 五、优缺点总结

| 优点                      | 缺点                        |
| ----------------------- | ------------------------- |
| 门控解耦：门与数据独立投影，表达更强      | 比标准 FFN 多一个线性层（参数变多，需压维度） |
| 论文实证各规模优于标准 FFN         | 无法用"单个函数"描述，理论分析较繁琐       |
| 参数量与标准 FFN 持平（8d/3 技巧）  | 训练时要同时存 gate 与 up 两组激活    |
| 与 SiLU 门组合 FP16/bf16 稳定 | 激活内存比单路略多（两个分支）           |

## 六、与同类激活函数对比

| 激活 | 结构 | 门控 | 参数量 | 代表模型 | 场景 |
|------|------|------|--------|---------|------|
| ReLU | 单路 | 硬门控 | — | ResNet | CNN 隐层 |
| GELU | 单路 | 软门控 xΦ(x) | — | BERT/ViT/GPT-2 | Transformer 单层 |
| SiLU | 单路 | 软门控 xσ(x) | — | ConvNeXt | 单层激活 |
| **SwiGLU** | **双路门控** | **SiLU(xW₁) ⊗ xW₂** | **= 标准 FFN** | **LLaMA/Qwen/Mistral** | **LLM FFN** |
| GEGLU | 双路门控 | GELU(xW₁) ⊗ xW₂ | = 标准 FFN | T5、PaLM | LLM FFN（早期） |

**架构选型速查**：

| 架构 | 隐层/FFN 激活 |
|------|--------------|
| ResNet | ReLU |
| BERT / ViT / GPT-2 | GELU |
| **LLaMA / Qwen / Mistral / Gemma** | **SwiGLU** |
| T5 / PaLM | GEGLU |

演进路线：`Sigmoid → Tanh → ReLU → LeakyReLU → GELU → SiLU → SwiGLU`——每步都是"上一代的平滑化/门控化"。

## 七、高频面试问答

**Q1：SwiGLU 是什么？**
GLU 家族变体：$\text{SiLU}(xW_1) \otimes xW_2$，用 SiLU 作门控的双线性结构。LLaMA/Qwen/Mistral 的 FFN 标准结构。

**Q2：为什么 LLM 都用 SwiGLU？**
论文实证各规模优于标准 FFN（GELU/ReLU）；参数量用 8d/3 中间维度压到与标准 FFN 持平；实现简单、数值稳定；LLaMA 开源后成事实标准。

**Q3：SwiGLU 与标准 FFN 的参数对比？**
标准 FFN 两个 4d 矩阵共 8d²；SwiGLU 三个 8d/3 矩阵共 8d²——用"更多更窄"的投影换取门控表达，总参数不变，FLOPs 也不变。

**Q4：为什么门控比单路激活强？**
单路激活的门与数据是同一个量，表达受限；门控让两组独立投影互相调制（gate 决定学不学，up 决定学什么），梯度互锁分工，表达能力更强。

**Q5：SwiGLU 与 GEGLU 的区别？为什么选了 SiLU？**
门函数不同：GEGLU 用 GELU（erf/tanh 近似），SwiGLU 用 SiLU。论文两者指标接近，但 SiLU 只需一次 sigmoid，无 erf/立方，实现与 FP16 更稳。

**Q6：手写 SwiGLU 的反向？**
对 gate 分支：grad ⊗ b ⊗ SiLU'(a)；对 up 分支：grad ⊗ SiLU(a)。两条梯度各自流回对应的投影矩阵。

**Q7：SwiGLU 的中间维度为什么是 8d/3？**
为了参数与标准 FFN 持平：3 矩阵 × 中间维 = 2 矩阵 × 4d → 中间维 = 8d/3。若用 4d 则参数多 50%（某些模型为提性能会这么做）。

**Q8：SwiGLU 在推理加速上有讲究吗？**
门分支与数据分支是两个独立 GEMM，可以并行；gate 的 SiLU 是逐元素 op，可用融合 kernel（如 Flash Attention 系列里的 fused SwiGLU）；bf16 下数值稳定，是 LLM 推理优化（vLLM/TensorRT-LLM）的标准组件。

## 八、自我检验

- [ ] 能写出 SwiGLU 公式与 GLU 通用定义
- [ ] 能画标准 FFN vs SwiGLU FFN 结构对比图并说清 8d/3 的由来
- [ ] 能推导"参数与 FLOPs 双持平"的数学关系
- [ ] 能写出手写反向的 autograd.Function 版本（两个梯度）
- [ ] 能写出 LLaMA 风格 3 线性层 SwiGLU FFN 并验证参数 = 8d²
- [ ] 知道 GLU 家族（GEGLU/ReGLU）与各代表模型
- [ ] 能回答 8 个面试追问
