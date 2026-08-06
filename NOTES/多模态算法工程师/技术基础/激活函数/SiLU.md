# SiLU / Swish 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

SiLU（Sigmoid Linear Unit，Ramachandran et al. 2017）即 Swish 在 β=1 时的特例：输入先过 Sigmoid 得到 0~1 的软门控，再与自身相乘：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

广义 Swish 带可调参数 β：

$$\text{Swish}(x) = x \cdot \sigma(\beta x), \qquad \beta = 1 \text{ 时} = \text{SiLU}$$

| 性质   | 值                                                   |
| ---- | --------------------------------------------------- |
| 定义域  | $(-\infty, +\infty)$                                |
| 值域   | $(-0.2785, +\infty)$                                |
| 单调性  | 非单调（负半轴有极小谷值）                                       |
| 谷值   | $x \approx -1.278$ 处取最小值 $\approx -0.2785$          |
| 零点   | $\text{SiLU}(0) = 0$                                |
| 渐近行为 | $x \to -\infty$ 时 $\to 0$；$x \to +\infty$ 时 $\to x$ |

> **记忆点**：SiLU = "输入自己当门控"（self-gating）：$\sigma(x)$ 是门，$x$ 是数据。Sigmoid 输出 0~1，所以 SiLU 的输出几乎就是 x 打折后的版本——但它不是单调的，负半轴先探底再归零，提供了"负向抑制"能力。

## 二、数学性质

### 2.1 导数推导

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma'(x) = \sigma(x)\left(1 + x(1 - \sigma(x))\right)$$

推导（乘法法则 + $\sigma'(x) = \sigma(x)(1-\sigma(x))$）：

$$\frac{d}{dx}\left(x\sigma(x)\right) = \sigma(x) + x\sigma'(x) = \sigma(x) + x\sigma(x)(1-\sigma(x)) = \sigma(x)\left(1 + x(1-\sigma(x))\right)$$

**关键数值**（实测）：

| 位置                | SiLU(x)          | SiLU'(x)           |
| ----------------- | ---------------- | ------------------ |
| $x = 0$           | 0                | 0.5                |
| $x = \pm 1$       | 0.7311 / -0.2689 | 0.9277 / 0.0723    |
| $x = -1.278$      | **-0.2785（谷值）**  | 0                  |
| $x = \pm 3$       | 2.8577 / -0.1423 | 1.0881 / -0.0881   |
| $x \approx -2.40$ | —                | **-0.0998（梯度极小值）** |
| $x \to +\infty$   | $\to x$          | $\to 1$            |

**关键结论**：
- 0 处梯度 0.5（与 GELU 相同），处处可导、无饱和、无死区；
- 负半轴存在**负谷值**：输出可以略负，表达"抑制"（这是与 GELU 最大的行为差异）；
- **非单调**：在 $(-\infty, -1.278)$ 上递减、$(-1.278, +\infty)$ 上递增；梯度在 $x \approx -2.4$ 处有小负值（-0.10），随后渐近回 0；
- 正半轴梯度最大 1.09 后趋近 1，无封顶，深层梯度畅通。

### 2.2 与 Swish 的关系（β 的意义）

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

- $\beta = 0$：退化为 $\text{Swish}(x) = x/2$（线性），无非线性；
- $\beta \to +\infty$：退化为 ReLU（硬门控）；
- $\beta = 1$：就是 SiLU；β 可学习时收敛值常略大于 1；
- 实验表明 β=1 的固定版本已足够好，**工程上直接叫 SiLU**，省一个超参数。

### 2.3 谷值的数学来源

谷值位置：解 $\text{SiLU}'(x) = 0$，即 $1 + x(1-\sigma(x)) = 0 \Rightarrow x = \sigma(x) - 1$ 的负解，约 $x = -1.278$，此时函数值 $\approx -0.2785$。这个"先抑后扬"的形状让网络能在负区微调信号而非直接丢弃。

## 三、源码实现

### 3.1 autograd.Function 手写版（含手动反向）

```python
import torch
import torch.nn as nn

class SiLUFunction(torch.autograd.Function):
    """自定义 SiLU：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x):
        # 前向：把 x 和 sigmoid 输出都保存，反向复用
        s = torch.sigmoid(x)
        ctx.save_for_backward(x, s)
        return x * s

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        # dSiLU/dx = s + x·s·(1-s)，s 是前向算好的 sigmoid
        return grad_output * (s + x * s * (1.0 - s))

x = torch.randn(4, 8, requires_grad=True)
y = SiLUFunction.apply(x)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])

# 梯度校验（float64 下与数值微分对比）
x0 = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
print("gradcheck:", torch.autograd.gradcheck(SiLUFunction.apply, (x0,)))
# gradcheck: True
```

> **注意**：反向需要同时用到 $x$ 和 $s = \sigma(x)$（公式是 $s + x\cdot s(1-s)$），所以前向必须**同时保存两个张量**——这与 Sigmoid（只存 out）不同。

### 3.2 nn.Module 包装

```python
import torch
import torch.nn as nn

class SiLU(nn.Module):
    """自定义 SiLU 模块（等价 nn.SiLU / nn.Swish）"""
    def forward(self, x):
        return x * torch.sigmoid(x)

m = SiLU()
x = torch.linspace(-5, 5, 7)
print(m(x))  # tensor([-3.3464e-02, -1.1482e-01, -2.6478e-01,  5.9605e-08,  1.4019e+00, 3.2185e+00,  4.9665e+00])
```

### 3.3 在模型中的典型用法（ConvNeXt 风格 + SwiGLU 的激活基础）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    """ConvNeXt 卷积块：Conv → LN → SiLU → 逐深度卷积 → 残差（简化版）"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = F.silu(self.pwconv1(h))     # SiLU 在中间
        h = self.pwconv2(h)
        h = h.permute(0, 3, 1, 2)
        return x + h

block = ConvNeXtBlock(64)
out = block(torch.randn(2, 64, 8, 8))
print(out.shape)  # torch.Size([2, 64, 8, 8])
```

### 3.4 作为 SwiGLU 的门控分支

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈（LLaMA 风格）：门控分支用 SiLU——这是 SiLU 在现代 LLM 的核心位置"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)     # 门分支
        self.up = nn.Linear(d_model, d_ff)       # 数据分支
        self.down = nn.Linear(d_ff, d_model)     # 输出投影

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

ffn = SwiGLUFFN(512, 1365)
print(ffn(torch.randn(2, 16, 512)).shape)  # torch.Size([2, 16, 512])
```

## 四、深入分析

### 4.1 梯度分析

- **无死亡、无饱和**：正半轴梯度趋 1，两端都不会锁死；
- **梯度范围约 [-0.10, 1.09]**：负半轴在 $x \approx -2.4$ 处有轻微负梯度（-0.10，对应函数快速下滑段），$x \to -\infty$ 时从下方渐近 0；$x = -1.28$（谷值）处梯度精确为 0，但仅此一点，不影响训练；
- **与 GELU 同构**：$0$ 处梯度同为 0.5；负半轴梯度形状相近（GELU 在 $x\approx-1.4$ 处也有一小段负梯度 -0.13）。

```python
import torch

x = torch.linspace(-8, 8, 1000, requires_grad=True)
y = (x * torch.sigmoid(x)).sum()
grad = torch.autograd.grad(y, x)[0]
print(f"x=0 处梯度: {grad[grad.shape[0] // 2].item():.4f}")       # 0.5040（网格上离 0 最近的采样点）
print(f"谷值处(x≈-1.28)梯度: {grad[420].item():.6f}")            # 0.0011（接近 0，穿过谷值）
print(f"梯度范围: [{grad.min().item():.3f}, {grad.max().item():.3f}]")  # [-0.100, 1.100]
```

### 4.2 数值稳定性

1. **无上溢**：$\sigma(x)$ 对 $x\to-\infty$ 下溢为 0，SiLU 输出正确趋 0；
2. **无下溢**：$x$ 很大时 $x\sigma(x) \to x$，自然过渡，不需要数值技巧；
3. **FP16 友好**：唯一指数在 sigmoid 内，$x > 15$（FP16）时 sigmoid→1，输出→x，依然正确——**FP16 下比 GELU 的 tanh 近似（含 x³）更稳**。

```python
import torch

x = torch.tensor([-100.0, -5.0, 0.0, 5.0, 100.0])
print(x * torch.sigmoid(x))
# tensor([-0.0000e+00, -3.3464e-02,  0.0000e+00,  4.9665e+00,  1.0000e+02])
```

### 4.3 计算复杂度

$$O(1) \text{ 逐元素运算：1 次 exp（sigmoid 内）+ 1 次乘法}$$

与 GELU-sigmoid 近似同级，比 GELU-tanh 少一次立方运算。在 LLM 中，FFN 的逐元素激活开销相对矩阵乘可忽略，但 SiLU 仍是门控家族里最便宜的之一。

### 4.4 为什么 LLM 门控用 SiLU 而不是 GELU

1. **SwiGLU 论文（Shazeer 2020）实证**：SiLU 门控的 SwiGLU 各项指标优于 GELU 门控的 GEGLU（T5/PaLM 用的 GEGLU 其实常以近似 GELU 实现）；
2. **负谷值表达**：SiLU 输出可微负（-0.28），提供"抑制"通道，信息不白白清零；
3. **实现简单 + FP16 稳定**：sigmoid 比 erf/tanh 近似少开方少立方，kernel 更简单。

## 五、优缺点总结

| 优点 | 缺点 |
|------|------|
| 处处可导、无死区无饱和 | 计算含 exp，比 ReLU 贵 |
| 非单调（负谷值），表达"抑制" | 有谷值 → 输出可略负，直觉性差一点 |
| 0 处梯度 0.5，与 GELU 同级的平滑 | 公式是"两个函数相乘"，推导略绕 |
| FP16 稳定，实现简单 | 单独用时与 GELU 差距不大（优势在门控） |

## 六、与同类激活函数对比

| 激活 | 公式 | 0 处梯度 | 负半轴行为 | 输出范围 | 现代用法 |
|------|------|---------|-----------|---------|---------|
| GELU | xΦ(x) | 0.5 | 渐变趋 0（恒 ≥ -0.17） | (-0.17, ∞) | Transformer 单层激活 |
| **SiLU** | **xσ(x)** | **0.5** | **谷值 -0.28 后归 0** | **(-0.28, ∞)** | **ConvNeXt、SwiGLU 门** |
| Swish | xσ(βx) | β/4 | 同 SiLU（β=1） | (-0.28, ∞) | EfficientNet |
| ReLU | max(0,x) | 约定 1 | 硬置 0 | [0, ∞) | CNN 隐层 |
| Sigmoid | σ(x) | 0.25 | 饱和趋 0 | (0, 1) | 输出层 |

- **vs GELU**：同源于"self-gating"思想（GELU 论文与 Swish 论文同年独立提出），0 处梯度相同；差别只在负半轴——GELU 单调、SiLU 有谷值。在 FFN 门控场景 SiLU 实证胜出；
- **vs Swish**：β=1 的特例，β 可学时收敛值通常接近 1，工程上直接用 SiLU；
- **vs ReLU**：SiLU 是"软门控 + 连续"版本，无死神经元，但贵一个 exp；
- **vs Sigmoid**：SiLU 是"数据 × 自己门的 sigmoid"，比纯 sigmoid 多携带输入幅度信息，梯度上限也从 0.25 提到 1。

**当前残存用途**：ConvNeXt（CNN 时代最后的"新激活"）、EfficientNet（Swish）、以及 **LLM FFN 门控（SwiGLU 的门分支）——现代大模型最活跃的位置**。

## 七、高频面试问答

**Q1：SiLU 和 Swish 的关系？**
Swish(x) = xσ(βx)，β=1 时就是 SiLU。β=0 退化为线性 x/2，β→∞ 退化为 ReLU。

**Q2：SiLU 与 GELU 的区别？**
公式不同但思想同源（self-gating）：$x\Phi(x)$ vs $x\sigma(x)$，0 处梯度都是 0.5。最大差异在负半轴：GELU 单调，SiLU 有谷值 -0.28@x≈-1.28，输出可负，能表达"抑制"。

**Q3：为什么 LLM 的 FFN 用 SiLU 而不是 GELU？**
SwiGLU 论文实证 SiLU 门控优于 GELU 门控；且 sigmoid 比 erf/tanh 实现简单、FP16 稳定。注意：GELU 在**单层**激活（BERT/ViT），SiLU 的舞台是**门控组合**（SwiGLU）。

**Q4：SiLU 的导数？**
$\sigma(x)(1 + x(1-\sigma(x)))$，0 处 0.5，正半轴趋 1，负半轴从 0.07（x=-1）渐近 0。处处连续无饱和。

**Q5：SiLU 的谷值在哪里？为什么存在？**
解 SiLU'(x)=0，得 x≈-1.278，f≈-0.2785。源于 $1+x(1-\sigma(x))=0$。负值输出使网络能"反相抑制"小信号。

**Q6：SiLU 在 FP16 下安全吗？**
安全。唯一的 exp 在 sigmoid 里，FP16 下 x>15 时 sigmoid→1、SiLU→x，行为自然；不像 GELU-tanh 近似含 x³ 有溢出风险。

**Q7：EfficientNet 里的 Swish 和 LLaMA 里的 SiLU 是同一个东西吗？**
数学上是：EfficientNet 用 β≈1 的 Swish，LLaMA 用 SiLU=Swish(β=1)。区别只在语境——EfficientNet 当单层激活用，LLaMA 当门控分支用。

**Q8：SiLU 单独用和 GELU 相比谁强？**
实验中差距不大（Swish 论文中略优或持平）；真正的差距在门控场景——SiLU 作为 GLU 门控分支时系统性优于 GELU 门控，这是 LLM 选它的直接原因。

## 八、自我检验

- [ ] 能写出 SiLU 公式、导数公式并手推（乘法法则 + sigmoid 导数）
- [ ] 知道谷值位置 x≈-1.278、值 ≈-0.2785
- [ ] 能说清 SiLU 与 GELU 的同源与差异
- [ ] 能写出 autograd.Function 手写版（保存 x 和 s 两个张量）并通过 gradcheck
- [ ] 知道 Swish 的 β 参数如何退化到线性/ReLU
- [ ] 知道 SiLU 在 ConvNeXt 与 SwiGLU 门控中的位置
- [ ] 能回答 8 个面试追问
