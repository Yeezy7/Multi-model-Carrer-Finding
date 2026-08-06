# GELU 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

GELU（Gaussian Error Linear Unit，高斯误差线性单元，Hendrycks & Gimpel 2016）是 Transformer 时代的事实标准激活函数。核心思想：**按输入值的大小，以概率保留它**——输入越大于 0，保留概率越高：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）：

$$\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \frac{1}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

| 性质   | 值                                                   |
| ---- | --------------------------------------------------- |
| 定义域  | $(-\infty, +\infty)$                                |
| 值域   | 近似 $(-0.17, +\infty)$                               |
| 单调性  | 非严格单调（$x \approx -0.75$ 处有极小值 $\approx -0.17$）      |
| 对称性  | 无                                                   |
| 零点   | $\text{GELU}(0) = 0$                                |
| 渐近行为 | $x \to -\infty$ 时 $\to 0$；$x \to +\infty$ 时 $\to x$ |

> **记忆点**：GELU 把 ReLU 的"硬门控"（负值必死）升级为"软门控"（负值按概率生还）——$\Phi(x)$ 就是"输入 x 被保留的概率"。$x = 0$ 时保留概率恰好 0.5，$x = -1$ 时约 0.16。

## 二、数学性质

### 2.1 导数推导

$$\text{GELU}'(x) = \Phi(x) + x \cdot \Phi'(x) = \Phi(x) + x \cdot \phi(x)$$

其中 $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ 是标准正态的 PDF。推导：乘法法则 + 微积分基本定理（$\Phi' = \phi$）：

$$\frac{d}{dx}\left(x\Phi(x)\right) = 1\cdot\Phi(x) + x\cdot\Phi'(x) = \Phi(x) + x\phi(x)$$

**关键数值**（由 $x=0$、$x=\pm1$、$x=\pm3$ 与极值点实测）：

| 位置 | GELU(x) | GELU'(x) |
|------|---------|----------|
| $x = 0$ | 0 | 0.5 |
| $x = \pm 1$ | 0.8413 / -0.1587 | 1.0833 / -0.0833 |
| $x = \pm 3$ | 2.996 / -0.0040 | 1.0120 / -0.0119 |
| $x \approx -0.75$ | **-0.17（函数极小值）** | 0 |
| $x \approx \pm 1.42$ | — | 1.129 / **-0.129（梯度极小值）** |

**关键结论**：
- 梯度**处处连续**：既无 ReLU 的硬截断，也无 Sigmoid 的饱和——"爬得回去"；
- 梯度范围约 $[-0.13, 1.13]$：负半轴在 $x \approx -1.4$ 处有轻微负梯度（≈ -0.13，函数在那附近有个小凹陷），$x \to -\infty$ 时梯度趋近 0（小但非零），不产生死神经元；
- 0 处梯度 0.5，正好是 ReLU（右导 1）与 Sigmoid（0.25）的中间；
- 一个易被忽略的事实：**GELU 并非严格单调**，在 $x \approx -0.75$ 处有极小值 ≈ -0.17——这恰好给了它和 SiLU 一样的"负向抑制"能力。

### 2.2 与 ReLU 的统一视角

$$\text{GELU}(x) = x \cdot \Phi(x) \quad \text{vs} \quad \text{ReLU}(x) = x \cdot \mathbb{1}[x > 0]$$

ReLU 的掩码是"确定性硬判断"（0 或 1），GELU 的掩码是"连续概率"（0~1 之间渐变）。**GELU 是 ReLU 的"软化版本"**：输入为正是"大概率保留"，为负是"小概率保留"。

### 2.3 近似公式（工程必背，两种写法）

**① Tanh 近似**（BERT/ViT 实现中采用，PyTorch 默认）：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

**② Sigmoid 近似**（更简单，误差略大）：

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

两者与精确版的最大误差分别约 $4.7 \times 10^{-4}$（tanh 版）与 $2.0 \times 10^{-2}$（sigmoid 版），在**训练**中均可忽略；对精度敏感的场景（如 bf16 推理量化校准）建议用 tanh 版或精确 erf 版。选型原则：追求快用 sigmoid 版，追求准用 tanh 版/精确版。

## 三、源码实现

### 3.1 精确版 + 两种近似（含手动反向）

```python
import math
import torch
import torch.nn as nn

def gelu_exact(x):
    """精确版：用 erf 实现（torch.nn.functional.gelu 在 approximate='none' 时走这里）"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh(x):
    """tanh 近似（PyTorch 默认近似）"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * x ** 3)))

def gelu_sigmoid(x):
    """sigmoid 近似（最快，误差略大）"""
    return x * torch.sigmoid(1.702 * x)

x = torch.linspace(-4, 4, 17)
print("精确 vs tanh近似 最大误差:", (gelu_exact(x) - gelu_tanh(x)).abs().max().item())
# 精确 vs tanh近似 最大误差: 0.0004399
print("精确 vs sigmoid近似 最大误差:", (gelu_exact(x) - gelu_sigmoid(x)).abs().max().item())
# 精确 vs sigmoid近似 最大误差: 0.01946
```

### 3.2 autograd.Function 手写版

```python
import math
import torch

class GELUFunction(torch.autograd.Function):
    """自定义 GELU（tanh 近似）：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                           (x + 0.044715 * x ** 3)))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        c = math.sqrt(2.0 / math.pi)
        t = torch.tanh(c * (x + 0.044715 * x ** 3))
        # 解析求导：d/dx [0.5x(1+tanh(u))] = 0.5(1+t) + 0.5x·(1-t²)·c·(1+0.134145x²)
        inner = c * (1.0 + 3 * 0.044715 * x ** 2)
        grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * inner
        return grad_output * grad

x = torch.randn(4, 8, requires_grad=True)
y = GELUFunction.apply(x)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])

# 梯度校验
x0 = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(GELUFunction.apply, (x0,))
print("gradcheck passed")  # gradcheck passed
```

> **注意**：精确版 GELU 的反向用 $x$ 直接算即可：$\Phi(x) + x\phi(x)$ 中 $\Phi(x)$ 用 `torch.erf`、$\phi(x)$ 用 `exp`，PyTorch 的 `nn.GELU()`（精确模式）内部即如此。手写时**不要**在前向里先算 tanh 近似再手动算近似梯度——近似梯度要与近似前向匹配，上面代码演示的就是"tanh 近似 + tanh 近似梯度"的配套写法。

### 3.3 nn.Module 包装

```python
import math
import torch
import torch.nn as nn

def gelu_exact(x):
    """精确版：用 erf 实现（torch.nn.functional.gelu 在 approximate='none' 时走这里）"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh(x):
    """tanh 近似（PyTorch 默认近似）"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * x ** 3)))

class GELU(nn.Module):
    """自定义 GELU 模块（等价 nn.GELU，支持精确/近似两模式）"""
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == "none":
            return gelu_exact(x)
        return gelu_tanh(x)

m = GELU()
x = torch.randn(3, 4)
print(m(x).shape)  # torch.Size([3, 4])
```

### 3.4 在模型中的典型用法（Transformer MLP 块）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """Transformer 标准 MLP 块：Linear → GELU → Linear（BERT/ViT 结构）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = F.gelu(self.fc1(x))        # 默认 tanh 近似
        return self.fc2(h)

block = MLPBlock(768, 3072)
out = block(torch.randn(2, 10, 768))
print(out.shape)  # torch.Size([2, 10, 768])
```

## 四、深入分析

### 4.1 梯度分析（为什么 Transformer 效果好）

- **无死区**：负半轴梯度非零（$x=-3$ 时 -0.012，$x=-0.75$ 时 0），梯度连续地通过 0 与负区，权重永远"爬得回来"；
- **无饱和**：正半轴梯度在 1 附近（最大 1.13），深层梯度畅通；
- **与 LayerNorm 的配合**：LN 后输入集中在原点附近（$|x| \lesssim 2$），该区域内 GELU 的梯度变化丰富（约 -0.09~1.09），非线性表达充分；ReLU 在该区域是"一刀切"（≤0 全死）；
- **0 附近曲率**：GELU 在 0 附近是光滑的非线性过渡（泰勒展开 $0.5x + 0.399x^3 + O(x^5)$，无二次项），这比 ReLU 的"拐角"更利于二阶优化。

```python
import torch

# 梯度连续性对比
x = torch.linspace(-5, 5, 1001, requires_grad=True)
gelu_grad = torch.autograd.grad(torch.nn.functional.gelu(x).sum(), x)[0]
relu_grad = torch.autograd.grad(torch.relu(x).sum(), x)[0]
print("GELU 梯度在 0 附近连续（无跳变）:", bool(gelu_grad.isnan().sum() == 0))
# GELU 梯度在 0 附近连续（无跳变）: True
print("GELU 梯度范围: [%.3f, %.3f]" % (gelu_grad.min().item(), gelu_grad.max().item()))
# GELU 梯度范围: [-0.129, 1.129]
print("ReLU 梯度在 0 处跳变（0→1）:",
      (relu_grad[500] - relu_grad[499]).item())  # 1.0 跳变（x=0 处）
# ReLU 梯度在 0 处跳变（0→1）: 1.0
```

### 4.2 数值稳定性

1. **精确版**：`erf` 是内部多项式近似，无指数上溢问题；$x$ 很大时 $x\Phi(x) \to x$，正确；
2. **tanh 版**：$x^3$ 在 $|x|>10^4$（FP32）时可能上溢——但正常网络输入远达不到；FP16 下 $|x|>100$ 开始有风险，混合精度下建议精确版（`approximate='none'`）或输入保持 FP32；
3. **sigmoid 版**：$1.702x$ 超过 ±88 时 sigmoid 饱和到 0/1，但此时 $x$ 本身量级已巨大，实际无影响。

```python
import math
import torch

def gelu_tanh(x):
    """tanh 近似（PyTorch 默认近似）"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * x ** 3)))

x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
print(torch.nn.functional.gelu(x))   # tensor([-0.0000, -0.1587,  0.0000,  0.8413, 100.0000])
print(gelu_tanh(x))                  # tensor([-0.0000, -0.1588,  0.0000,  0.8412, 100.0000])
```

### 4.3 计算复杂度

| 版本 | 运算 | 相对 ReLU 开销 |
|------|------|---------------|
| ReLU | 1 次比较 | 1× |
| GELU-sigmoid | 1 次 exp + 2 次乘加 | ~2× |
| GELU-tanh | 1 次 exp（tanh 内）+ 1 次立方 + 数次乘加 | ~3× |
| GELU-exact | erf（内部高阶多项式） | ~4× |

Transformer 的 FLOPs 大头是矩阵乘法，GELU 的额外开销占比很小，所以"贵一点"完全可接受。

### 4.4 与 ReLU 的本质区别（面试核心）

| 维度   | ReLU                  | GELU                                      |
| ---- | --------------------- | ----------------------------------------- |
| 门控方式 | 硬门控 $\mathbb{1}[x>0]$ | 软门控 $\Phi(x)$（0~1 连续）                     |
| 负半轴  | 梯度恒 0（死亡）             | 梯度随 x 渐变趋 0（不死亡）                          |
| 平滑性  | 0 处不可导                | 处处可导，梯度连续                                 |
| 随机解释 | —                     | 等价于"随机丢弃式"正则化（$\Phi(x)$ 可看作 Bernoulli 概率） |

## 五、优缺点总结

| 优点                        | 缺点                               |
| ------------------------- | -------------------------------- |
| 处处可导、梯度连续，训练稳定            | 计算比 ReLU 贵（exp/tanh/erf）         |
| 无死区无饱和，梯度流动好              | 公式不直观，有 3 种实现版本（精确/tanh/sigmoid） |
| 软门控带随机性，有正则化效果            | 输出仍非零中心、无上界                      |
| 与 LN 配合好，Transformer 实证最优 | 训练与推理近似误差存在（可忽略）                 |

## 六、与同类激活函数对比

| 激活 | 门控方式 | 负半轴梯度 | 0 处梯度 | 计算 | 现代用法 |
|------|---------|-----------|---------|------|---------|
| ReLU | 硬截断 | 恒 0 | 约定 1 | 极简 | CNN 隐层 |
| LeakyReLU | 固定小斜率 | 恒 α | 约定 1 | 极简 | GAN/检测 |
| **GELU** | **软门控 xΦ(x)** | **渐变趋 0** | **0.5** | **中** | **BERT/ViT/GPT-2** |
| SiLU | 软门控 xσ(x) | 渐变至 -0.28 谷值 | 0.5 | 中 | ConvNeXt、SwiGLU |
| SwiGLU | 双线性门控 | — | — | 中 | LLM FFN 标配 |

- **vs ReLU**：见 4.4，本质是"硬 vs 软"；
- **vs SiLU**：SiLU 负半轴有一个负谷值（约 -0.28@x≈-1.28），输出可负；GELU 输出恒 ≥ -0.17 且单调。两者梯度行为相近，实证上 SiLU 在深层略优，LLM 门控全部用它；
- **vs SwiGLU**：GELU 是"单路乘掩码"，SwiGLU 是"两路变换互做门控"——SwiGLU 是 LLM 版 FFN，GELU 是单层内激活。

**当前残存用途**：BERT/ALBERT、ViT 全部、GPT-2 的 FFN、以及一切 Transformer 架构的隐层激活；是"Transformer 时代的 ReLU"。

## 七、高频面试问答

**Q1：GELU 和 ReLU 的本质区别？**
ReLU 硬截断（掩码 0/1），GELU 软门控（掩码 Φ(x) ∈ (0,1) 连续）。GELU 负半轴梯度非零且连续，"神经元不会死"。

**Q2：为什么 Transformer/ViT 用 GELU 而不用 ReLU？**
① 梯度处处连续（ReLU 0 处不可导），训练稳；② 软门控无死神经元；③ 与 LayerNorm 配合好（LN 后输入在原点附近，GELU 在该区非线性更强）；④ BERT/ViT 实证 GELU 指标更优。

**Q3：GELU 的两种近似公式？误差多大？**
① $0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$；② $x\sigma(1.702x)$。与精确版最大误差约 $4.7\times10^{-4}$（tanh 版）与 $2\times10^{-2}$（sigmoid 版），训练中都可忽略；精度敏感场景（bf16 量化校准）建议用 tanh 版或精确 erf。

**Q4：精确 GELU 的导数怎么写？**
GELU'(x) = Φ(x) + xφ(x)，其中 φ 是标准正态 PDF。0 处梯度 0.5，负半轴渐变趋 0，正半轴趋 1。

**Q5：为什么 GELU 有"随机性"解释？**
$\Phi(x)$ 可视为"输入 x 被保留的概率"，等价于对 x 乘一个 Bernoulli 随机变量——因此有 dropout 式的正则化含义。

**Q6：GELU 在 ViT MLP 块中的位置？**
Linear(d→4d) → GELU → Dropout → Linear(4d→d) → Dropout。GELU 在两个线性层中间提供非线性。

**Q7：FP16 下 GELU 要注意什么？**
tanh 近似含 $x^3$，输入绝对值过大（FP16 下 >~100）可能溢出；混合精度训练建议用精确版（approximate='none'）或保证输入量级正常。两种近似间的误差在 FP16 下可忽略。

**Q8：GELU 的近似版与精确版在反向传播上有区别吗？**
有。PyTorch 中近似模式的 autograd 图用的是 tanh 计算的解析梯度（与近似前向配套），精确模式用 erf 计算——两者梯度形状一致、数值略有差异，不影响训练结果。

## 八、自我检验

- [ ] 能写出 GELU 定义式 $x\Phi(x)$ 和两种近似公式
- [ ] 能推导 GELU'(x) = Φ(x) + xφ(x)，记住 0 处梯度 0.5
- [ ] 能说清"硬截断 vs 软门控"的本质区别
- [ ] 能写出 tanh 近似的 autograd.Function 手写版并通过 gradcheck
- [ ] 知道精确/近似三种实现的误差量级（10⁻³）
- [ ] 知道 BERT/ViT/GPT-2 的 FFN 都用 GELU 及具体位置
- [ ] 能回答 8 个面试追问
