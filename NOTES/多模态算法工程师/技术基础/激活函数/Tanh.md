# Tanh 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

Tanh（双曲正切，hyperbolic tangent）把任意实数映射到 $(-1, 1)$，是 Sigmoid 平移缩放后的零中心版本：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

| 性质           | 值                                                                |
| ------------ | ---------------------------------------------------------------- |
| 定义域          | $(-\infty, +\infty)$                                             |
| 值域           | $(-1, 1)$                                                        |
| 单调性          | 严格单调递增                                                           |
| 对称性          | $\tanh(-x) = -\tanh(x)$（奇函数，关于原点对称）                              |
| 零点           | $\tanh(0) = 0$                                                   |
| 渐近线          | $x \to +\infty$ 时 $\tanh \to 1$；$x \to -\infty$ 时 $\tanh \to -1$ |
| 与 Sigmoid 关系 | $\tanh(x) = 2\sigma(2x) - 1$                                     |

> **记忆点**：Tanh 是 Sigmoid 的"零中心化"版本。Sigmoid 输出恒为正导致梯度单向偏置，Tanh 输出以 0 为中心，收敛更快，所以它是 Sigmoid 时代隐层的默认选择。

## 二、数学性质

### 2.1 导数（最重要的性质）

$$\tanh'(x) = 1 - \tanh^2(x)$$

推导（基于定义 $f = e^x$）：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

用商法则，令 $u = e^x - e^{-x}$，$v = e^x + e^{-x}$，则 $u' = v$，$v' = u$：

$$\tanh'(x) = \frac{u'v - uv'}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \left(\frac{u}{v}\right)^2 = 1 - \tanh^2(x)$$

**关键结论**：
- 梯度最大值在 $x = 0$ 处：$\tanh'(0) = 1$（比 Sigmoid 的 0.25 大 4 倍！）；
- $x$ 远离 0 时 $\tanh' \to 0$：**两端依然饱和**，深层网络照样梯度消失；
- 梯度只依赖输出 $\tanh(x)$，"前向结果存着，反向直接用"（与 Sigmoid 同理）。

### 2.2 与 Sigmoid 的换算

$$\tanh(x) = 2\sigma(2x) - 1, \qquad \sigma(x) = \frac{1}{2}\left(\tanh\left(\frac{x}{2}\right) + 1\right)$$

Sigmoid 和 Tanh 是同一个函数的两种参数化，任何一方的性质都可以换算到另一方。

### 2.3 零中心的意义

Sigmoid 输出恒正（$\sigma(x) > 0$），导致下一层输入恒正 → 参数梯度方向被一致偏置，更新呈 zig-zag。Tanh 输出零中心，梯度方向随输入正负自然变化，**经验上收敛更快**。这也是历史上隐层从 Sigmoid 换到 Tanh 的核心动机。

## 三、源码实现

### 3.1 纯 PyTorch 手写（含手动反向）

```python
import torch
import torch.nn as nn

class TanhFunction(torch.autograd.Function):
    """自定义 Tanh：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x):
        # 前向：保存输出供反向复用（省一次 exp 计算）
        out = torch.tanh(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        # d(tanh)/dx = 1 - tanh²(x)，直接复用前向输出
        return grad_output * (1.0 - out * out)

x = torch.randn(4, 8, requires_grad=True)
y = TanhFunction.apply(x)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])

# 梯度校验：与 torch.autograd.gradcheck 对比
x0 = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(TanhFunction.apply, (x0,))
print("gradcheck passed")  # gradcheck passed
```

### 3.2 数值稳定版本

```python
import torch

def tanh_fraction(x):
    """直接按定义写分式——注意：大 |x| 时 (inf - inf) 会得到 nan，仅作错误示范"""
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# 量级正常时，分式写法与内建一致
x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
print(torch.tanh(x))       # tensor([-1.0000, -0.7616,  0.0000,  0.7616,  1.0000])
print(tanh_fraction(x))    # tensor([-1.0000, -0.7616,  0.0000,  0.7616,  1.0000])

# 但 x=±100 时 exp 溢出为 inf：inf - inf = nan，分式写法崩坏
x_big = torch.tensor([-100.0, 100.0])
print(tanh_fraction(x_big))  # tensor([nan, nan])  ← 朴素分式不安全
print(torch.tanh(x_big))     # tensor([-1., 1.])    ← 内建实现内部已做稳定处理
```

> **注意**：朴素分式写法（$(e^x-e^{-x})/(e^x+e^{-x})$）在 $|x|$ 较大时会因 inf 运算产生 nan。工程中直接使用 `torch.tanh` / `nn.Tanh`（内部有稳定实现，等价于对 $2x$ 用 sigmoid 的稳定算法或直接 IEEE 精度保证），不要自己写分式。

### 3.3 nn.Module 包装

```python
import torch
import torch.nn as nn

class Tanh(nn.Module):
    """自定义 Tanh 模块（等价 nn.Tanh）"""
    def forward(self, x):
        return torch.tanh(x)

# 用法
m = Tanh()
x = torch.randn(2, 3)
y = m(x)          # y ∈ (-1, 1)
print(y.min().item() > -1 and y.max().item() < 1)  # True
```

### 3.4 在模型中的典型用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """经典 MLP：Tanh 时代的标准写法（现在隐层多用 ReLU/GELU）"""
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))   # 隐层 Tanh
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)            # 回归输出不加激活

model = MLP(128)
out = model(torch.randn(5, 128))
print(out.shape)  # torch.Size([5, 1])
```

## 四、深入分析

### 4.1 梯度分析

反向传播中梯度逐层连乘。Tanh 每过一层最多乘 1（但只有 $x=0$ 附近接近 1）：

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \prod_{i=1}^{n-1} \tanh'(x_i) \le 1^{n-1} \cdot \frac{\partial L}{\partial x_n}$$

| 输入位置         | tanh'(x)             | 说明    |
| ------------ | -------------------- | ----- |
| $x = 0$      | 1                    | 最大梯度  |
| $x = \pm 1$  | 0.42                 | 尚可    |
| $x = \pm 2$  | 0.07                 | 已明显衰减 |
| $x = \pm 5$  | $1.8 \times 10^{-4}$ | 基本消失  |
| $x = \pm 10$ | $8.2 \times 10^{-9}$ | 完全饱和  |

对比 Sigmoid：Tanh 在原点附近梯度是 Sigmoid 的 4 倍，且零中心，所以**前几层还好，深了照样死**。

```python
import torch

# 梯度可视化（取 3 个代表点验证两端饱和，用 float64 避免 tanh(10) 被舍入为 1.0）
for x in [0.0, 5.0, 10.0]:
    g = 1 - torch.tanh(torch.tensor(x, dtype=torch.float64)) ** 2
    print(f"x={x:>4} 时梯度: {g.item():.3e}")  # x=0 时 1.000e+00；x=5 时 1.816e-04；x=10 时 8.245e-09
```

### 4.2 数值稳定性

1. **上溢**：$|x| > 88$（FP32）时 $\exp(\pm x)$ 溢出为 inf——如果按定义分式手写，`inf - inf = nan` 直接崩坏（见 3.2 演示）；FP16 下 $|x| > 11$ 就开始出问题；
2. **内建实现**：`torch.tanh` 内部做了稳定处理（大 |x| 时直接返回 ±1），工程上无条件用它；
3. **混合精度**：输入超过 FP16 范围时先在 FP32 下计算 tanh 再回传，或对输入做缩放。

### 4.3 计算复杂度

$$O(1) \text{ 次逐元素运算：} 1 \text{ 次 exp、} 1 \text{ 次除、} 2 \text{ 次加减}$$

与 Sigmoid 同级（exp 是昂贵操作），比 ReLU（一次比较）贵约一个数量级，这也是它被 ReLU 取代的原因之一。

## 五、优缺点总结

| 优点                       | 缺点                  |
| ------------------------ | ------------------- |
| 输出零中心（(-1,1)），梯度方向不偏置    | 两端仍饱和 → 深层网络梯度消失    |
| 原点处梯度 = 1（Sigmoid 的 4 倍） | 指数计算昂贵              |
| 奇函数，适合以 0 为中心的数据         | 单调有界，表达能力受限于线性区很窄   |
| 输出负值可表示"抑制"，信息不丢弃        | 深网络中效果远不如 ReLU/GELU |

## 六、与同类激活函数对比

| 激活       | 值域          | 零中心   | 原点梯度    | 饱和问题     | 现代用法              |
| -------- | ----------- | ----- | ------- | -------- | ----------------- |
| Sigmoid  | (0, 1)      | 否     | 0.25    | 严重       | 输出层二分类、门控         |
| **Tanh** | **(-1, 1)** | **是** | **1.0** | **严重**   | **输出层映射到 (-1,1)** |
| ReLU     | [0, +∞)     | 否     | 1（右侧）   | 无（负侧为 0） | CNN 隐层            |
| GELU     | (-∞, +∞)    | 近似    | 0.5     | 无        | Transformer/ViT   |

- **vs Sigmoid**：Tanh 是零中心改进版，收敛更快，但饱和问题同源；数学上 $\tanh(x) = 2\sigma(2x) - 1$；
- **vs ReLU**：Tanh 无死区问题（梯度处处非零），但指数贵 + 饱和；ReLU 便宜且正半轴无饱和，深网胜出；
- **vs GELU**：GELU 无上界不饱和，梯度流动比 Tanh 好得多，Transformer 时代完全取代。

**当前残存用途**：
1. **LSTM 细胞状态**：更新门输出后经 Tanh 压缩到 (-1,1)（现代 LLM 用门控前馈，但 LSTM 系架构仍有）；
2. **输出层归一化**：需要把网络输出映射到 (-1,1) 的回归/生成任务（如 GAN 生成器的图像像素归一）；
3. **数值稳定的 softmax 近似**：某些注意力变体（如线性注意力）用它代替指数运算。

**使用场景速查**：

| 场景 | 为什么用 | 示例 |
|------|---------|------|
| 输出映射到 (-1,1) | 值域天然匹配 | GAN 生成器图像输出、音频波形生成 |
| LSTM 细胞状态 | 有界压缩 + 残差友好 | LSTM/GRU（门控时代的标配） |
| 有界可导输出 | 平滑 + 不爆炸 | 坐标/置信度归一化 |
| 激活研究基线 | 与 ReLU 系列对比 | 论文 ablation 实验 |

## 七、高频面试问答

**Q1：Tanh 和 Sigmoid 的区别？**
Tanh = 2σ(2x) - 1，是 Sigmoid 的零中心版本：值域 (-1,1) vs (0,1)，原点梯度 1 vs 0.25，输出对称。收敛更快，但饱和问题一样存在。

**Q2：为什么 Tanh 收敛比 Sigmoid 快？**
两个原因：① 零中心，梯度方向不会单向偏置（消除 zig-zag）；② 原点附近梯度是 Sigmoid 的 4 倍，初期更新幅度大。

**Q3：Tanh 的梯度范围？**
[0, 1]，最大 1 在 x=0，x 远离 0 时指数级衰减到 0。所以它只能缓解、不能根治梯度消失。

**Q4：为什么现代网络不用 Tanh 了？**
两端饱和 + 指数计算贵。ReLU 正半轴梯度恒 1 不饱和且便宜，深网络训练效果更好。Tanh 仅残留在 LSTM 等场景。

**Q5：Tanh 还有哪些应用场景？**
① 输出映射到 (-1,1)：GAN 生成器输出层、归一化图像像素；② LSTM 细胞状态压缩；③ 一些激活的近似计算（如 Swish 系列的推导起点）。

**Q6：手写 Tanh 反向传播？**
前向存 out = tanh(x)，反向返回 grad_output * (1 - out²)。复杂度 O(1)，与 Sigmoid 同样"梯度复用前向输出"。

**Q7：Tanh 在 FP16 下要注意什么？**
输入超过约 11 时 e^x 上溢。虽然比值仍趋向 ±1，但中间结果 inf 可能触发 nan 传播，实际中把计算保持在 FP32 或用框架自带 tanh。

## 八、自我检验

- [ ] 能写出 Tanh 公式、导数公式并手推一遍（商法则推导）
- [ ] 能说出 Tanh 与 Sigmoid 的换算关系 2σ(2x) - 1
- [ ] 能说清"零中心 + 原点梯度 4 倍"为何收敛更快
- [ ] 能写出手写反向传播的 autograd.Function 版本并通过 gradcheck
- [ ] 知道 Tanh 为什么没解决梯度消失（两端仍饱和）
- [ ] 知道 Tanh 在现代架构中的残存位置（LSTM、输出层归一化）
- [ ] 能回答 7 个面试追问
