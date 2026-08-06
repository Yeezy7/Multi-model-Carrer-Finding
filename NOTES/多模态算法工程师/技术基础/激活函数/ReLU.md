# ReLU 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

ReLU（Rectified Linear Unit，修正线性单元）是深度学习使用最广泛的激活函数，2011 年前后确立标准地位。它把负输入直接置零，正输入原样保留：

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}$$

| 性质 | 值 |
|------|-----|
| 定义域 | $(-\infty, +\infty)$ |
| 值域 | $[0, +\infty)$ |
| 单调性 | 非严格单调递增（负半轴恒 0） |
| 对称性 | 无 |
| 零点 | 所有 $x \le 0$ 输出 0 |
| 计算量 | 一次比较，**零指数运算** |

> **记忆点**：ReLU 的哲学是"模拟神经元的放电特性"——信号强就通过，信号弱或为负就沉默。它用最简单的计算换来了正半轴恒为 1 的梯度，直接终结了 Sigmoid/Tanh 的梯度消失问题。

## 二、数学性质

### 2.1 导数（分段常数）

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

在 $x = 0$ 处**不可导**（左导 0、右导 1），工程上约定取 0 或 1（PyTorch 取 0），这在实际中完全不影响训练。

**关键结论**：
- 正半轴梯度恒为 1：梯度不会因激活函数连乘而消失（$\prod 1 = 1$）；
- 负半轴梯度恒为 0：**死区**——输入为负的神经元收不到任何梯度；
- 分段线性：局部线性、全局非线性，既能走梯度又能拟合复杂函数。

### 2.2 稀疏激活

一半以上的输入会输出 0（当权重初始化和数据分布使输入有正有负时），产生**稀疏激活**：
- 神经元彼此解耦，类似稀疏编码，有正则化效果；
- 计算上跳过 0 值可能加速（稀疏矩阵运算）。

### 2.3 无上界问题

输出无上界 → 激活值在深层累积可能越来越大。这引出了"输出非零中心"和"无上界"两个结构性缺点。

## 三、源码实现

### 3.1 纯 PyTorch 手写（含手动反向）

```python
import torch
import torch.nn as nn

class ReLUFunction(torch.autograd.Function):
    """自定义 ReLU：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x):
        # 前向：保存输入掩码，反向时复用（注意是保存 x 而不是 out）
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # dReLU/dx = 1[x > 0]，注意这里 x==0 处梯度取 0
        mask = (x > 0).to(grad_output.dtype)
        return grad_output * mask

x = torch.randn(4, 8, requires_grad=True)
y = ReLUFunction.apply(x)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])
print(x.grad.min().item())  # 0.0（所有负输入位置的梯度为 0）
```

> **对比 Sigmoid**：Sigmoid 反向用输出算梯度（存 out），ReLU 反向用**输入**算梯度（存 x）——因为 $1[x>0]$ 只依赖输入的正负号。

### 3.2 数值稳定性

ReLU 没有 exp/log，**不存在溢出问题**，这是它计算上最大的优势之一：

```python
import torch

x = torch.tensor([-1e8, -1.0, 0.0, 1.0, 1e8])
print(torch.relu(x))  # tensor([0., 0., 0., 1., 1e+08])，任意量级都安全
```

唯一隐患是**无上界**：正输入很大时输出同样很大，深层网络可能出现激活值爆炸，配合 BatchNorm/LayerNorm 或较小的初始化即可。

### 3.3 nn.Module 包装

```python
import torch
import torch.nn as nn

class ReLU(nn.Module):
    """自定义 ReLU 模块（等价 nn.ReLU，inplace 参数可配置）"""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.relu(x) if not self.inplace else x.relu_()

# 用法
m = ReLU(inplace=True)
x = torch.randn(2, 3)
y = m(x)          # 负值被置 0
print((y >= 0).all().item())  # True
```

### 3.4 在模型中的典型用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """CNN 标准范式：Conv → BN → ReLU"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))   # Conv→BN→ReLU 顺序
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.adaptive_avg_pool2d(h, 1)       # 全局池化到 (B, 64, 1, 1)
        return self.fc(h.flatten(1))

model = ConvNet()
out = model(torch.randn(2, 3, 32, 32))
print(out.shape)  # torch.Size([2, 10])
```

## 四、深入分析

### 4.1 梯度分析（vs Sigmoid 连乘）

ReLU 正半轴梯度恒 1，多层连乘 $\prod 1 = 1$，**梯度消失被根除**：

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \prod_{i=1}^{n} \text{ReLU}'(x_i) = \begin{cases} \frac{\partial L}{\partial x_n} & \text{路径全在正半轴} \\ 0 & \text{路径经过死区} \end{cases}$$

**代价转移到 Dead ReLU**：某神经元输入长期为负 → 梯度恒 0 → 权重永不更新。

### 4.2 Dead ReLU（神经元死亡）

触发原因：
1. **学习率过大**：参数一步跨过最优区域，把输入推到负区；
2. **初始化不当**：权重把输入映射到负区（如激活值偏移）；
3. **非零中心**：上一层的输出全为正时，梯度对权重偏置方向一致，积累偏移。

```python
import torch

# Dead ReLU 演示：给一个长期负输入的神经元
x = torch.full((3, 4), -5.0, requires_grad=True)   # 输入恒为负
y = torch.relu(x)
y.sum().backward()
print(x.grad)  # tensor([[0., 0., 0., 0.]...]) 梯度全 0，神经元死亡
```

缓解手段：① 小学习率 + 合理初始化（Kaiming）；② LeakyReLU/PReLU 给小斜率；③ 换 GELU 等软门控；④ 调整网络结构（BN 在前）。

**实践排查：统计"死神经元"占比**（面试常问怎么定位）：

```python
import torch
import torch.nn as nn

torch.manual_seed(0)                         # 固定随机种子保证输出可复现
layer = nn.Linear(128, 64)
relu = nn.ReLU()
x = torch.randn(512, 128)                    # 一个 batch 的输入
act = relu(layer(x))
dead_ratio = (act == 0).float().mean().item()
print(f"该层激活中 0 的占比: {dead_ratio:.3f}")  # 0.508（随机初始化下约一半，ReLU 稀疏性使然）
# 若训练中该比例异常升高（明显 >0.5 且持续），说明该层在"批量死亡"
```

### 4.3 初始化配合：Kaiming 初始化

Kaiming（He）初始化是**专门为 ReLU 推导**的：ReLU 丢弃一半激活，方差减半，为保持前向/反向方差守恒，权重方差应取 $2/n$（Sigmoid 时代用 $1/n$ 的 Xavier）：

$$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$

### 4.4 复杂度

$$O(1) \text{ 逐元素运算：1 次比较，0 次指数}$$

比 Sigmoid/Tanh（各 1 次 exp）便宜约一个数量级，是 CNN 时代推理速度的关键因素之一。

## 五、优缺点总结

| 优点 | 缺点 |
|------|------|
| 正半轴梯度恒 1，根除梯度消失 | 负半轴梯度恒 0 → **Dead ReLU** |
| 计算极简（一次比较，无指数） | 输出非零中心（恒 ≥0），梯度方向偏置 |
| 稀疏激活，有正则化效果 | 输出无上界，深层激活值可能膨胀 |
| x=0 处分段线性，收敛快 | 不可导点（实际影响可忽略） |

## 六、与同类激活函数对比

| 激活 | 值域 | 正半轴梯度 | 负半轴梯度 | 死亡神经元 | 计算 | 现代用法 |
|------|------|-----------|-----------|-----------|------|---------|
| Sigmoid | (0,1) | ≤0.25 | ≤0.25 | — | 贵 | 输出层 |
| Tanh | (-1,1) | ≤1 | ≤1 | — | 贵 | LSTM 等 |
| **ReLU** | **[0,∞)** | **恒 1** | **恒 0** | **有** | **极简** | **CNN 隐层** |
| LeakyReLU | (-∞,∞) | 恒 1 | 恒 α | 缓解 | 极简 | 检测/GAN |
| GELU | (-∞,∞) | ~1 | ~0.1~0.5 | 无 | 中 | Transformer |

- **vs Sigmoid/Tanh**：正半轴梯度恒 1 无饱和，负半轴直接置 0，计算便宜 → 深网络胜出；
- **vs LeakyReLU**：负半轴给小斜率 α（默认 0.01），"死亡"变成"缓慢呼吸"，但多一个超参且收益有限；
- **vs GELU**：ReLU 是硬截断（负值丢光），GELU 是软门控（负值按概率保留）——Transformer 时代 GELU 全面取代 ReLU。

**当前残存用途**：CNN 隐层标准（ResNet 等）、部分 MLP、以及作为"对照基线"出现（SwiGLU 论文对比的标准 FFN）。

## 七、高频面试问答

**Q1：为什么 ReLU 比 Sigmoid 好？**
① 正半轴梯度恒 1，多层连乘不消失；② 计算常数时间（一次比较）；③ 稀疏激活有正则化效果。缺点：死亡神经元、非零中心、无上界。

**Q2：什么是 Dead ReLU？怎么解决？**
输入长期为负 → 梯度恒 0 → 权重永不更新。解决：LeakyReLU、小学习率、Kaiming 初始化、GELU 软门控。

**Q3：ReLU 在 0 处不可导，为什么还能用？**
工程上约定次梯度取 0 或 1（PyTorch 取 0）。测度为零的单点不影响 SGD 的实际行为，理论上用次梯度（subgradient）即可。

**Q4：为什么 ReLU 输出非零中心？有什么影响？**
恒 ≥0 → 下一层输入全为正 → 对权重的梯度单向偏置，更新呈 zig-zag 收敛慢。这是它相对 Tanh 的退化点，但实际影响远小于梯度消失。

**Q5：ReLU 会梯度爆炸吗？**
正半轴梯度恒 1 本身不放大梯度；但输出无上界，若权重矩阵谱范数 >1，激活值仍可能随层数指数增长 → 梯度爆炸。解决：BN/LN、合理初始化、梯度裁剪。

**Q6：ReLU 适合什么初始化？**
Kaiming（He）初始化，权重方差 $2/n_{in}$。它正是考虑 ReLU 丢一半激活的方差衰减推导出来的；Sigmoid/Tanh 用 Xavier（$1/n_{in}$）。

**Q7：训练中大量 ReLU 死亡，你怎么排查？**
① 统计各层激活的 0 占比（dead 比例 > 30% 要警惕）；② 检查学习率是否过大；③ 看 BN 是否在 Conv 之后正确摆放；④ 考虑换 LeakyReLU/GELU 对比实验。

**Q8：ReLU 的稀疏性为什么好？**
一半输出为 0 → 神经元去相关、表征解耦，类似稀疏编码的正则化；同时减少无效计算量。

## 八、自我检验

- [ ] 能写出 ReLU 公式与分段导数，知道 x=0 处约定取 0
- [ ] 能解释正半轴梯度恒 1 为何根除梯度消失
- [ ] 能写出 Dead ReLU 的产生原因与 4 种缓解手段
- [ ] 能写出手写反向传播的 autograd.Function 版本（存输入而非输出）
- [ ] 知道 ReLU 与 Kaiming 初始化的关系（方差 $2/n_{in}$）
- [ ] 能说清 ReLU vs GELU 的本质区别（硬截断 vs 软门控）
- [ ] 能回答 8 个面试追问
