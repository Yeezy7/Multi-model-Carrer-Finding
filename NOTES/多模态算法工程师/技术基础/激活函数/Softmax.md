# Softmax 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

Softmax 把一组 logits（任意实数）映射为概率分布（非负且和为 1），是多分类输出层与注意力机制的核心：

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

| 性质 | 值 |
|------|-----|
| 输入 | logits 向量 $z \in \mathbb{R}^K$ |
| 输出 | 概率向量 $p \in (0, 1)^K$ |
| 归一化 | $\sum_i p_i = 1$（输出是概率分布） |
| 单调性 | 对每个 $z_i$ 单调递增（保持顺序） |
| 平移不变性 | $\text{softmax}(z + c) = \text{softmax}(z)$ |
| 温度版 | $\text{softmax}(z_i / \tau)$，温度 $\tau > 0$ |

**本质**：Softmax 是 Sigmoid 的多类推广（两类时两者等价，见下），也是"对数几率归一化"的产物——logits 被解释为类别对数几率（log-odds），Softmax 是其软最大（soft argmax）的概率化。

## 二、数学性质

### 2.1 导数推导（Jacobian 矩阵）

Softmax 是**向量→向量**函数，导数是一个 $K \times K$ 的 Jacobian。令 $p_i = e^{z_i}/S$，$S = \sum_k e^{z_k}$：

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & i = j \\ -p_i p_j & i \ne j \end{cases}$$

推导（$i = j$ 时用商法则，$i \ne j$ 时分子不含 $z_j$，只有分母受 $z_j$ 影响）：

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = p_i - p_i^2 = p_i(1-p_i)$$

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} e^{z_j}}{S^2} = -p_i p_j$$

紧凑写法：

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

**关键结论**：
- 对角项 $p_i(1-p_i) \le 0.25$（和 Sigmoid 同构）——softmax 也会"饱和"（当某个 $p_i \to 1$ 时）；
- 非对角项为负：提高 $z_j$ 会**降低**所有其他类别的概率（"抢概率"）；
- 整行和为 0：$\sum_j \partial p_i / \partial z_j = p_i(1 - \sum_j p_j) = 0$（概率守恒）。

### 2.2 与 Sigmoid 的关系

两类情形（$K=2$）：

$$\text{softmax}([z_1, z_2])_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

多类是二类的逐对推广。同时有"对数域"关系：

$$\log p_i = z_i - \underbrace{\log\sum_k e^{z_k}}_{\text{log-sum-exp，归一化常数}}$$

### 2.3 温度的作用

$$\text{softmax}(z_i / \tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

| 温度 | 行为 | 用途 |
|------|------|------|
| $\tau \to 0^+$ | 退化为 argmax（one-hot，硬决策） | 贪婪解码、蒸馏中的"硬标签" |
| $\tau = 1$ | 标准 softmax | 训练 |
| $\tau \to \infty$ | 退化为均匀分布 | 完全随机采样 |
| $\tau < 1$ | 分布更尖锐（置信度更高） | 推理时收紧分布 |
| $\tau > 1$ | 分布更平坦（更随机） | 解码多样性、知识蒸馏软化标签 |

**对比学习**里温度 $\tau$ 还是可学习/可调超参（CLIP 学习 $\tau$ 约 0.07），控制"正样本得分的尖锐程度"——温度越低，正样本被拉得越紧。

## 三、源码实现

### 3.1 朴素实现（会溢出的错误写法）

```python
import torch

def softmax_naive(z):
    """错误写法：exp(z) 在 z 很大时上溢为 inf，演示数值问题"""
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)

z = torch.tensor([1000.0, 0.0, -1000.0])
print(softmax_naive(z))
# tensor([nan, 0., 0.])  ← exp(1000)=inf，inf/inf=nan；其余项被 inf 稀释为 0
```

### 3.2 数值稳定版：减最大值（log-sum-exp 技巧）

$$\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \quad m = \max_j z_j$$

由于平移不变性（分子分母同乘 $e^{-m}$），减去最大值后所有指数项都在 $[0, 1]$ 内：

```python
import torch

def softmax_stable(z):
    """稳定写法：先减最大值再求 exp，利用平移不变性"""
    m = z.max(dim=-1, keepdim=True).values   # 减最大值：e^z / Σe^z = e^(z-m) / Σe^(z-m)
    e = torch.exp(z - m)
    return e / e.sum(dim=-1, keepdim=True)

z = torch.tensor([1000.0, 0.0, -1000.0])
print(softmax_stable(z))
# tensor([1.0000e+00, 0.0000e+00, 0.0000e+00])
print(torch.softmax(z, dim=-1))  # 与 PyTorch 内建一致
# tensor([1.0000e+00, 0.0000e+00, 0.0000e+00])
```

### 3.3 温度版 Softmax（手写 + 内建）

```python
import torch

def softmax_temperature(z, tau=1.0):
    """温度版：tau→0 趋近 one-hot，tau→∞ 趋近均匀分布"""
    z = z / tau
    m = z.max(dim=-1, keepdim=True).values
    e = torch.exp(z - m)
    return e / e.sum(dim=-1, keepdim=True)

z = torch.tensor([[2.0, 1.0, 0.0]])
print(softmax_temperature(z, tau=1.0))
# tensor([[0.6652, 0.2447, 0.0900]])
print(softmax_temperature(z, tau=0.5))
# tensor([[0.8668, 0.1173, 0.0159]])  ← 更尖锐
print(softmax_temperature(z, tau=10.0))
# tensor([[0.3672, 0.3322, 0.3006]])  ← 更平坦
```

### 3.4 手写反向传播（autograd.Function）

```python
import torch

class SoftmaxFunction(torch.autograd.Function):
    """自定义 softmax（按最后一维）"""

    @staticmethod
    def forward(ctx, z):
        m = z.max(dim=-1, keepdim=True).values
        e = torch.exp(z - m)
        p = e / e.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        # dL/dz = p ⊙ grad - p ⊙ (p · grad 沿类别维求和) = (grad - p·Σ(grad·p)) ⊙ p
        grad_p = grad_output - (grad_output * p).sum(dim=-1, keepdim=True)
        return p * grad_p

z = torch.randn(4, 10, requires_grad=True)
p = SoftmaxFunction.apply(z)
print(p.sum(dim=-1))  # tensor([1., 1., 1., 1.], grad_fn=<SumBackward1>)（输出和为 1）

# 梯度校验
z0 = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
print("gradcheck:", torch.autograd.gradcheck(SoftmaxFunction.apply, (z0,)))
# gradcheck: True
```

### 3.5 与交叉熵联合（重点：不要显式 softmax）

数学上交叉熵损失为 $L = -\sum_i y_i \log p_i$，$p_i$ 是 softmax 输出。若直接计算 $\log p_i$ 会先求概率再取对数，数值与梯度都不干净。正确做法用 **log-softmax** 恒等式：

$$\log p_i = z_i - \text{logsumexp}(z)$$

```python
import torch
import torch.nn.functional as F

def softmax_stable(z):
    """稳定版 softmax（减最大值）"""
    m = z.max(dim=-1, keepdim=True).values
    e = torch.exp(z - m)
    return e / e.sum(dim=-1, keepdim=True)

def log_softmax(z):
    """log-softmax：logsumexp 写法，直接得 log 概率"""
    m = z.max(dim=-1, keepdim=True).values
    return z - m - torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True))

# 对比三种"softmax + CE"写法
z = torch.tensor([[2.0, 1.0, 0.1]])
y = torch.tensor([0])  # 真实类别

# 写法 A：先 softmax 再 log（数值差，梯度散）
p = softmax_stable(z)
loss_a = -torch.log(p[0, 0])
print("A. softmax→log:", loss_a.item())  # 0.417

# 写法 B：log-softmax 一步到位（数值好）
loss_b = -log_softmax(z)[0, 0]
print("B. log-softmax:", loss_b.item())  # 0.417（结果相同）

# 写法 C：PyTorch 内建（等价于 B，等价于 NLLLoss）
loss_c = F.cross_entropy(z, y)
print("C. F.cross_entropy:", loss_c.item())  # 0.417

# 数值极端场景对比：写法 A 会损失精度
z_extreme = torch.tensor([[1000.0, 0.0]])
p = softmax_stable(z_extreme)
print("A 极端:", -torch.log(p[0, 0]).item())   # -0.0（log(1) = 0，且无下溢）
print("C 极端:", F.cross_entropy(z_extreme, torch.tensor([0])).item())  # 0.0（同样为 0，但中间量更干净）
```

**为什么内建 loss 更优**：`F.cross_entropy = log_softmax + NLLLoss`，logsumexp 一次性完成"归一化 + 取对数"，消除了中间概率的数值误差，梯度也直接从 logits 传播（相当于 softmax 的 Jacobian 与 log 链式合成的干净版本，见 4.3）。

## 四、深入分析

### 4.1 数值稳定性

1. **上溢**：$z_i > 88$（FP32）时 $\exp(z_i) = \inf$ → 减最大值 $m$ 后指数项 ∈ $[0, 1]$，根除；
2. **下溢**：$z_i \ll m$ 时概率下溢为 0，**但** log-softmax 场景必须用 logsumexp（$z_i - m - \log\sum e^{z_j - m}$）避免 $\log 0 = -\inf$；
3. **logsumexp 是统一框架**：softmax（= exp(logsumexp 的分量差）、log-softmax、交叉熵）全部建立在其上：
$$\text{logsumexp}(z) = m + \log\sum_j e^{z_j - m}$$

### 4.2 梯度分析（为什么"softmax + 交叉熵"梯度干净）

把 softmax 输出代入 CE（one-hot 标签 $y_i$），复合函数的梯度**不经过"中间 log(p)"的饱和**：

$$\frac{\partial L}{\partial z_j} = p_j - y_j$$

| 情形 | 梯度 |
|------|------|
| 正确类别 $j = y$ | $p_j - 1 < 0$（把 logits 推高） |
| 错误类别 $j \ne y$ | $p_j > 0$（把 logits 压低） |
| 模型已自信（$p_j \to 1$） | 梯度 → 0（自动减速，不振荡） |

这个"$p - y$"的简洁形式只出现在联合优化中——单独 softmax（MSE 等损失）的梯度要过 Jacobian 矩阵，数值和效率都差。

### 4.3 注意力中的 Softmax

$$\text{attn}_{ij} = \frac{e^{q_i^{\top} k_j / \sqrt{d}}}{\sum_{j'} e^{q_i^{\top} k_j' / \sqrt{d}}}$$

- 沿**每行**（序列维）做 softmax，每个位置输出对全序列的权重分布；
- $\sqrt{d}$ 缩放防止点积过大进入 softmax 饱和区（$p \to 0/1$ 时梯度消失）；
- 训练用 $\tau = 1$ 标准版；**解码采样**用温度版 softmax(z/τ) 控制生成多样性；
- FlashAttention 用在线 logsumexp 技巧避免显存存全矩阵。

### 4.4 复杂度

$$O(K) \text{ 次 exp + } O(K) \text{ 次求和（减最大值版多一次 max 遍历）} + \text{归一化除法}$$

- 数值稳定版比朴素版多一次 max 的遍历（约 2 倍常数开销），工程上可接受；
- 注意力场景是 $B \times H \times S^2$ 个独立 softmax，FlashAttention 通过 blockwise 在线计算把它压到 $O(1)$ 额外显存。

## 五、优缺点总结

| 优点 | 缺点 |
|------|------|
| 输出规范的概率分布（非负、和为 1） | 指数运算昂贵（K 个 exp） |
| 与 CE 联合梯度 = p - y，极其干净 | 类别多时计算随 K 线性增长 |
| 平移不变性 → 数值稳定技巧的数学基础 | 无法输出 0/1 硬决策（需 argmax） |
| 温度控制分布锐度，统一训练/推理 | 过度自信时（p→1）梯度消失 |

## 六、与同类激活函数对比

| 激活 | 输出 | 归一化 | 单输入/向量 | 用途 |
|------|------|--------|-----------|------|
| Sigmoid | (0,1) 逐元素 | 无（和为 1 不保证） | 标量 | 二分类/门控 |
| **Softmax** | **(0,1) 向量** | **和为 1（概率分布）** | **向量** | **多分类/注意力** |
| GELU/SiLU | 逐元素 | 无 | 标量 | 隐层非线性 |
| argmax（硬） | one-hot | 和为 1 | 向量 | 推理解码 |

- **vs Sigmoid**：多标签场景（每个类别独立判断）用 Sigmoid；单标签多分类用 Softmax（互斥）。数学上 K=2 时等价（见 2.2）；
- **vs GELU/SiLU**：那些是"逐元素非线性"（坐标级变换），Softmax 是"跨维度竞争"（类别级变换）——它是唯一改变坐标间关系的激活；
- **vs argmax**：argmax 不可导（用于推理），softmax 是其可导的"软"版本。

**当前残存用途**：分类输出层（配合 CE）、**注意力权重（Transformer 核心）**、对比学习的相似度归一化（CLIP/SigLIP 的行 softmax + 温度）、解码采样分布。

## 七、高频面试问答

**Q1：softmax 为什么要减最大值？**
防止 exp 上溢（FP32 阈值约 88）。减最大值利用平移不变性（分子分母同乘 $e^{-m}$），不改变结果。

**Q2：softmax 与交叉熵联合时要注意什么？**
不要"先 softmax 再 log"（中间概率会下溢/损失精度），直接用 log-softmax 或 F.cross_entropy（内部 logsumexp + NLLLoss）。

**Q3：softmax 的梯度公式？**
$\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$；与交叉熵联合后梯度退化为干净的 $p - y$。

**Q4：温度的作用？训练和推理分别怎么用？**
$\tau \to 0$ 趋近 one-hot，$\tau \to \infty$ 趋近均匀。训练：对比学习设小 τ（如 CLIP 的 0.07）拉紧正样本；推理：解码采样 τ>1 增加多样性，τ<1 更确定。

**Q5：注意力里的 softmax 和分类的 softmax 有区别吗？**
数学相同，但沿行做（每行=每个位置对序列的分布），且输入先除以 $\sqrt{d}$ 防饱和；生成解码时额外除以温度。

**Q6：为什么 softmax 会梯度消失？**
当某个 $p_i \to 1$（模型过度自信），对角导数 $p_i(1-p_i) \to 0$。这也是注意力 logits 要缩放、logits 过大要处理的原因。

**Q7：logsumexp 和 softmax 的关系？**
logsumexp 是 log 域归一化常数：$\log\sum e^{z_j} = m + \log\sum e^{z_j-m}$。softmax = exp(z - logsumexp(z))；log-softmax = z - logsumexp(z)。一个函数统一三种写法。

**Q8：手写 softmax 反向传播？**
前向存 p；反向 $\text{grad} = p \odot (\text{grad}_p - \sum_j p_j \text{grad}_p)$，即减掉梯度在概率分布上的均值投影。

## 八、自我检验

- [ ] 能写出 softmax 公式、Jacobian 公式并手推
- [ ] 能解释减最大值的数学依据（平移不变性）
- [ ] 知道 logsumexp 与 softmax/log-softmax/CE 的统一关系
- [ ] 能写出温度版 softmax 并说明 τ 的极限行为
- [ ] 能写出手写反向的 autograd.Function 版本并通过 gradcheck
- [ ] 能推导"softmax+CE"联合梯度 = p - y
- [ ] 知道注意力 softmax 的行方向与 √d 缩放原因
- [ ] 能回答 8 个面试追问
