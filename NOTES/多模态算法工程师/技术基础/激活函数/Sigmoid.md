# Sigmoid 激活函数

> 本模块索引见 [激活函数详解](../激活函数/激活函数详解.md)

## 一、定义与公式

Sigmoid（逻辑斯蒂函数）是深度学习最早的激活函数之一，把任意实数映射到 (0, 1)：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| 性质  | 值                                                                 |
| --- | ----------------------------------------------------------------- |
| 定义域 | $(-\infty, +\infty)$                                              |
| 值域  | $(0, 1)$                                                          |
| 单调性 | 严格单调递增                                                            |
| 对称性 | $\sigma(-x) = 1 - \sigma(x)$（关于 (0, 0.5) 中心对称）                    |
| 零点  | $\sigma(0) = 0.5$                                                 |
| 渐近线 | $x \to +\infty$ 时 $\sigma \to 1$；$x \to -\infty$ 时 $\sigma \to 0$ |

## 二、数学性质

### 2.1 导数（最重要的性质）

$$\sigma'(x) = \sigma(x)\left(1 - \sigma(x)\right)$$

推导：

$$\sigma'(x) = \frac{d}{dx}(1 + e^{-x})^{-1} = -(1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

**关键结论**：
- 梯度最大值在 x=0 处：$\sigma'(0) = 0.25$；
- $x$ 远离 0 时 $\sigma' \to 0$：**两端饱和**，这是梯度消失的根源；
- 梯度只依赖输出 $\sigma(x)$，实现时"前向结果存着，反向直接用"。

### 2.2 与 Tanh 的关系

$$\tanh(x) = 2\sigma(2x) - 1$$

Sigmoid 平移缩放后就是 Tanh——所以两者性质同源，只是 Tanh 零中心。

### 2.3 作为概率的意义

Sigmoid 可看作"对数几率（log-odds）为 x 时的概率"：

$$x = \log\frac{p}{1-p} \iff p = \sigma(x)$$

这一性质让 Sigmoid 成为二分类输出层的默认选择（配合 BCE 损失）。

## 三、源码实现

### 3.1 纯 PyTorch 手写（含手动反向）

```python
import torch
import torch.nn as nn

class SigmoidFunction(torch.autograd.Function):
    """自定义 Sigmoid：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x):
        # 前向：保存输出供反向复用（省一次计算）
        out = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        # dσ/dx = σ(1-σ)，直接复用前向结果
        return grad_output * out * (1.0 - out)

x = torch.randn(4, 8, requires_grad=True)
y = SigmoidFunction.apply(x)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])
```

### 3.2 数值稳定版本（防溢出）

```python
def sigmoid_stable(x):
    """x 很大时 exp(-x) 溢出为 0 没问题；x 很小时 exp(-x)=inf 会出错 → 分情况处理"""
    return torch.where(x >= 0,
                       1.0 / (1.0 + torch.exp(-x)),   # x ≥ 0 用原式
                       torch.exp(x) / (1.0 + torch.exp(x)))  # x < 0 用等价式 e^x/(1+e^x)

# 对比验证
x = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
print(torch.sigmoid(x))       # tensor([0., 4.5e-05, 0.5, 1.0, 1.0])
print(sigmoid_stable(x))      # tensor([0., 4.5e-05, 0.5, 1.0, 1.0])
```

> **注意**：PyTorch 的 `torch.sigmoid` 和 `nn.Sigmoid` 内部已做数值稳定处理，FP32 下不需要自己写稳定版。上面的稳定写法主要用于 FP16 或自己手写 kernel 的场景。

### 3.3 nn.Module 包装

```python
class Sigmoid(nn.Module):
    """自定义 Sigmoid 模块（等价 nn.Sigmoid）"""
    def forward(self, x):
        return torch.sigmoid(x)

# 用法
m = Sigmoid()
x = torch.randn(2, 3)
y = m(x)          # y ∈ (0, 1)
```

### 3.4 在模型中的典型用法

```python
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    """二分类模型：sigmoid 只出现在输出层"""
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 1)      # 输出 1 个 logit

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))  # 概率 p ∈ (0,1)

model = BinaryClassifier(128)
probs = model(torch.randn(5, 128))
print(probs)  # 每行是正类概率
```

## 四、梯度分析（为什么深度学习时代被淘汰）

### 4.1 梯度消失

反向传播中梯度是逐层连乘。Sigmoid 每过一层最多乘 0.25：

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \prod_{i=1}^{n-1} \sigma'(x_i) \le 0.25^{n-1} \cdot \frac{\partial L}{\partial x_n}$$

| 层数   | 梯度衰减上限          |
| ---- | --------------- |
| 1 层  | ×0.25           |
| 5 层  | ×0.25⁴ ≈ 0.004  |
| 10 层 | ×0.25⁹ ≈ 4×10⁻⁶ |
| 20 层 | 4×10⁻¹³（完全消失）   |

### 4.2 非零中心问题

Sigmoid 输出恒为正（>0），导致：
- 下一层输入全为正；
- 权重梯度方向被一致偏置（同一方向 zig-zag 更新），收敛慢。

### 4.3 前向时机的对比

```python
# 梯度消失可视化
x = torch.linspace(-10, 10, 100)
y = torch.sigmoid(x)
dy = y * (1 - y)
print(f"x=0 时梯度最大: {dy.max().item():.4f}")          # 0.25
print(f"x=±5 时梯度: {dy[abs(x) >= 5].max().item():.6f}")  # ~0.0067
```

## 五、数值稳定性

1. **上溢**：$x$ 非常大（>88，FP32）时 $e^{-x} \to 0$，结果正确趋于 1；FP16 下 >15 就溢出，需要稳定版；
2. **下溢**：$x$ 非常小（<-88）时 $e^{-x} \to \infty$，`exp(-x)` 溢出为 inf，`1/inf = 0`，结果仍正确但浪费；稳定写法见 3.2；
3. **训练中**：Logits 过大时 Sigmoid 输出饱和（≈0 或 1），梯度为 0 → 建议配合 `BCEWithLogitsLoss`（内部用 log-sigmoid 恒等式，数值稳定）而不是"先 sigmoid 再 log"。

```python
# 错误写法（数值不稳定）与正确写法
import torch.nn.functional as F
logits = torch.tensor([100.0, -100.0])

bad = -torch.log(torch.sigmoid(logits))          # 可能得到 [0, 100]，100 处精度丢失
good = F.binary_cross_entropy_with_logits(
    torch.tensor([1.0, 0.0]), logits)            # 数值稳定
print(bad, good)
```

## 六、使用场景

| 场景        | 为什么用       | 示例                                     |
| --------- | ---------- | -------------------------------------- |
| 二分类输出层    | 输出可解释为概率   | 图文匹配二分类（ITM）                           |
| 门控机制      | 0~1 的软门控   | LSTM 门、GLU 家族的前身                       |
| 注意力权重（早期） | 归一化到 (0,1) | 已基本被 Softmax 取代                        |
| 对比学习概率化   | 把相似度变概率    | SigLIP 的 pairwise loss（配合 log-sigmoid） |
| 输出归一化     | 控制在 (0,1)  | 置信度、掩码预测                               |

**多模态中的两个高频位置**：
1. **图文匹配头**：BLIP 的 ITM 用线性层输出 logit → Sigmoid → 匹配概率（或直接用 BCEWithLogits）；
2. **SigLIP 损失**：$-\log\sigma(Y\cdot z)$ 的逐对二分类（标签 ±1），数值上用 `F.logsigmoid` 实现。

## 七、优缺点总结

| 优点              | 缺点                      |
| --------------- | ----------------------- |
| 输出 (0,1)，可解释为概率 | 两端饱和 → 梯度消失             |
| 平滑、处处可导         | 输出非零中心 → 收敛慢            |
| 导数可复用前向输出，实现简单  | 指数计算相对昂贵                |
| 与 log-odds 直接对应 | 深网络中效果差（被 ReLU/GELU 取代） |

## 八、高频面试问答

**Q1：Sigmoid 的梯度范围？**
[0, 0.25]，最大在 x=0 处。这是它"梯度消失"问题的量化来源：深网络中每层梯度最多乘 0.25。

**Q2：Sigmoid 和 Softmax 的区别？**
Sigmoid 独立处理每个输出（多标签，和为 1 不要求）；Softmax 在类别维度归一化（多分类，和为 1）。数学上 Softmax 是 Sigmoid 在多类的推广，单类时两者等价。

**Q3：为什么隐层不用 Sigmoid？**
梯度消失 + 非零中心 + 计算贵。隐层用 ReLU/GELU，Sigmoid 只保留在输出层（概率输出）与门控位置。

**Q4：Sigmoid 输出为什么非零中心？怎么影响训练？**
输出恒 >0，下一层梯度同向偏置，参数更新 zig-zag，收敛慢。Tanh 零中心是改进，但饱和问题依旧。

**Q5：二分类为什么用 BCEWithLogits 而不是 sigmoid+BCE？**
BCEWithLogits 内部用 log-sigmoid 恒等变换（log-sum-exp 技巧），避免 exp/log 的数值误差与梯度消失；且梯度形式更干净。

**Q6：Sigmoid 在对比学习里的用途？**
SigLIP 把图文相似度经 Sigmoid 变成匹配概率做逐对二分类，配合 log-sigmoid 数值稳定写法。这是 Sigmoid 在 Transformer 时代最重要的新用途。

## 九、自我检验

- [ ] 能写出 Sigmoid 公式、导数公式并手推一遍
- [ ] 能说出梯度最大值 0.25 与梯度消失的关系
- [ ] 能写出手写反向传播的 autograd.Function 版本
- [ ] 知道数值稳定写法与 torch.sigmoid 的差异
- [ ] 能说清 Sigmoid vs Softmax 的区别
- [ ] 知道 Sigmoid 在 BLIP/SigLIP 中的两个高频位置
- [ ] 能回答 6 个面试追问
