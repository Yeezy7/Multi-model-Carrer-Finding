# LeakyReLU 激活函数

> 本模块索引见 [激活函数详解](激活函数详解.md)

## 一、定义与公式

LeakyReLU（带泄漏的修正线性单元）是 ReLU 的直接修补：负半轴不再硬置 0，而是给一个很小的斜率 $\alpha$（默认 0.01），让梯度"泄漏"过去：

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}, \quad \alpha = 0.01$$

| 性质 | 值 |
|------|-----|
| 定义域 | $(-\infty, +\infty)$ |
| 值域 | $(-\infty, +\infty)$（$\alpha>0$ 时） |
| 单调性 | 严格单调递增（$\alpha>0$ 时） |
| 对称性 | 无（负半轴斜率不同） |
| 零点 | $\text{LeakyReLU}(0) = 0$ |
| 超参数 | 负斜率 $\alpha$（默认 0.01，PyTorch 中 0.01） |

> **记忆点**：ReLU 的问题是负输入直接把神经元"杀掉"，LeakyReLU 给死者"吊一口气"——梯度从 0 变成 0.01，神经元永远不会完全死亡。

## 二、数学性质

### 2.1 导数（分段常数）

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x < 0 \end{cases}$$

在 $x = 0$ 处不可导（左导 $\alpha$、右导 1），工程上约定取 1（PyTorch 默认取 1）。

**关键结论**：
- 正半轴梯度恒 1：与 ReLU 相同，根除正半轴梯度消失；
- 负半轴梯度恒 $\alpha = 0.01$：**梯度不再为 0**，死亡神经元问题被根治；
- 代价：负半轴输出不再是稀疏的 0，稀疏性略降。

### 2.2 与 ReLU 的统一视角

$$\text{LeakyReLU}(x) = \max(x, \alpha x) = \text{ReLU}(x) - \alpha \cdot \text{ReLU}(-x)$$

ReLU 是 $\alpha = 0$ 的 LeakyReLU 特例。负半轴从"全杀"变成"打折放行"。

### 2.3 PReLU 与 RReLU（家族扩展）

- **PReLU（参数化）**：$\alpha$ 变成**可学习参数**（每通道一个），训练中自动学最佳负斜率；
- **RReLU（随机化）**：训练时 $\alpha$ 从均匀分布 $U(\text{lower}, \text{upper})$ 中随机采样，测试时取均值，相当于一种正则化；
- **ELU**：负半轴用指数平滑 $a(e^x - 1)$，梯度在 0 处连续但计算更贵。

## 三、源码实现

### 3.1 纯 PyTorch 手写（含手动反向）

```python
import torch
import torch.nn as nn

class LeakyReLUFunction(torch.autograd.Function):
    """自定义 LeakyReLU：展示前向/反向的实现细节"""

    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.save_for_backward(x)
        ctx.negative_slope = negative_slope
        return torch.where(x > 0, x, negative_slope * x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        slope = ctx.negative_slope
        # d/dx = 1[x>0] + α·1[x≤0]，注意 x==0 处约定取 1
        grad_x = torch.where(x > 0,
                             torch.ones_like(x),
                             torch.full_like(x, slope))
        return grad_output * grad_x, None

x = torch.randn(4, 8, requires_grad=True)
y = LeakyReLUFunction.apply(x, 0.01)
y.sum().backward()
print(x.grad.shape)  # torch.Size([4, 8])
print((x.grad == 0).any().item())  # False（没有零梯度，神经元不会死）
```

### 3.2 nn.Module 包装

```python
import torch
import torch.nn as nn

class LeakyReLU(nn.Module):
    """自定义 LeakyReLU 模块（等价 nn.LeakyReLU）"""
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x > 0, x, self.negative_slope * x)

# 用法
m = LeakyReLU(0.1)
x = torch.tensor([[-2.0, 0.0, 3.0]])
print(m(x))  # tensor([[-0.2,  0.,  3.]])，负值乘 0.1 而非置 0
```

### 3.3 在模型中的典型用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """GAN 判别器标准范式：LeakyReLU 是 DCGAN 的默认选择"""
    def __init__(self, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(img_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)   # DCGAN 常用 slope=0.2
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc3(h)

model = Discriminator(784)
out = model(torch.randn(4, 784))
print(out.shape)  # torch.Size([4, 1])
```

### 3.4 PReLU（可学习斜率）

```python
import torch
import torch.nn as nn

class PReLU(nn.Module):
    """简化版 PReLU：α 作为可学习参数（每通道共享一个）"""
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        return torch.where(x > 0, x, self.weight * x)

m = PReLU(1, init=0.25)
x = torch.randn(2, 3)
y = m(x)
y.sum().backward()
print(m.weight.grad is not None)  # True（α 能学到梯度，随训练更新）
```

## 四、深入分析

### 4.1 梯度分析（死亡神经元 vs 泄漏梯度）

与 ReLU 的唯一差别在负半轴：

| 场景 | ReLU 梯度 | LeakyReLU 梯度 |
|------|-----------|---------------|
| 输入全正 | 1 | 1 |
| 输入全负（长期负区） | **0（死亡）** | **0.01（缓慢恢复）** |
| 0.01 梯度下 1000 步 | 永不更新 | 权重累计更新 10×lr 的量 |

关键差异不是"梯度大小"而是"**能否恢复**"：负半轴恒 0 意味着权重**永远**拿不到信号；α=0.01 虽小，但持续累积，一旦权重被推回正区即可恢复。

```python
import torch

# 对比：负区输入下两者的梯度累计
x = torch.full((1000, 4), -0.5, requires_grad=True)
torch.relu(x).sum().backward()
print("ReLU 梯度累计:", x.grad.sum().item())        # 0.0（完全死亡）

x2 = torch.full((1000, 4), -0.5, requires_grad=True)
torch.nn.functional.leaky_relu(x2, 0.01).sum().backward()
print("LeakyReLU 梯度累计:", x2.grad.sum().item())  # 39.999996（≈40，即 4000 个元素 × α=0.01）
```

### 4.2 数值稳定性

与 ReLU 相同，**无指数运算、无溢出风险**，任意量级输入都安全：

```python
import torch

x = torch.tensor([-1e8, -0.5, 0.0, 5.0, 1e8])
print(torch.nn.functional.leaky_relu(x, 0.01))
# tensor([-1.0000e+06, -5.0000e-03,  0.0000e+00,  5.0000e+00,  1.0000e+08])
```

### 4.3 超参数 α 的选择

- α 过小（→0）：退化回 ReLU，死亡问题复现；
- α 过大（→1）：退化为线性函数 $y = x$，丧失非线性；
- 经验值：PyTorch 默认 0.01，DCGAN 用 0.2，检测头常用 0.1~0.2；
- 调 α 不如直接换 GELU/SiLU 来得平滑——**GELU 家族让 α 隐式地随 x 连续变化**（负半轴梯度从 0 到 ~0.5 渐变），比固定斜率更优雅。

### 4.4 复杂度

$$O(1) \text{ 逐元素运算：1 次比较 + 1 次乘法，0 次指数}$$

与 ReLU 同量级（多一次乘法），是计算最便宜的激活家族之一。

## 五、优缺点总结

| 优点 | 缺点 |
|------|------|
| 负半轴梯度恒 α > 0，**根治神经元死亡** | 多一个超参数 α，需要调 |
| 正半轴梯度恒 1，无梯度消失 | 非线性弱于 GELU/SiLU（负半轴是直线） |
| 计算极简（比较 + 乘法） | 输出仍非零中心、无上界（同 ReLU） |
| 数值稳定，无溢出 | 实践收益与 ReLU 差异常不显著 |

## 六、与同类激活函数对比

| 激活 | 负半轴行为 | 死亡神经元 | 梯度消失 | 计算 | 现代用法 |
|------|-----------|-----------|---------|------|---------|
| ReLU | 硬置 0 | 有 | 无 | 极简 | CNN 隐层 |
| **LeakyReLU** | **αx（α=0.01）** | **缓解** | **无** | **极简** | **GAN/检测** |
| PReLU | αx（可学习） | 缓解 | 无 | 极简 | 人脸识别等 |
| ELU | a(eˣ-1)（指数平滑） | 无 | 无 | 中 | 少用 |
| GELU | xΦ(x)（软门控） | 无 | 无 | 中 | Transformer |

- **vs ReLU**：只改负半轴斜率，death 缓解但收益在 ImageNet 上并不稳定——说明 ReLU 的死亡问题实际不严重（有 BN 后更难触发）；
- **vs GELU**：LeakyReLU 是"分段线性 + 固定斜率"，GELU 是"连续平滑 + 渐变斜率"，后者梯度更稳、表达更自然，Transformer 时代胜出；
- **vs ELU**：ELU 负半轴指数平滑、梯度连续（0 处可导），但多一次 exp 且收敛速度收益不明确，工程上反而不如 LeakyReLU 常用。

**当前残存用途**：GAN 判别器/生成器（DCGAN 范式 slope=0.2）、目标检测头部（SSD/YOLO 部分变体）、以及所有"怕死神经元"的轻量场景。

## 七、高频面试问答

**Q1：LeakyReLU 和 ReLU 的区别？**
负半轴从硬置 0 改为 αx（默认 0.01）。正半轴完全相同；区别仅在于负半轴梯度恒 α>0，神经元不会完全死亡。

**Q2：LeakyReLU 为什么能解决 Dead ReLU？**
梯度不再是 0：只要输入为负，权重就能收到 0.01 的梯度并持续更新，一旦被推回正区即恢复活性。0.01 小但**不等于 0**——"能恢复"比"梯度大"更关键。

**Q3：为什么 LeakyReLU 没有完全取代 ReLU？**
实践上收益不稳定：有 BN 和良好初始化后 ReLU 死亡问题并不严重；多一个 α 超参增加调参成本。而 GELU 等更平滑的激活在 Transformer 上收益明显，抢占了演进方向。

**Q4：α 取多大合适？取 1 会怎样？**
常用 0.01（PyTorch 默认）、0.1~0.2（GAN/检测）。α→0 退化为 ReLU；α→1 退化为线性函数 $y=x$，非线性消失，无法拟合复杂函数。

**Q5：PReLU 和 LeakyReLU 的区别？**
PReLU 的 α 是**可学习参数**（每通道一个，反向传播更新），LeakyReLU 的 α 是固定的。PReLU 表达更强但有过拟合风险；RReLU 则是训练时随机采样 α，做正则化。

**Q6：LeakyReLU 在 0 处的梯度取多少？**
不可导点（左导 α、右导 1），PyTorch 约定取 1。与 ReLU 同理，单点对训练无影响。

**Q7：为什么 GAN 里爱用 LeakyReLU？**
判别器要输出连续置信度（0~1），若用 ReLU 死神经元会导致判别信号丢失；LeakyReLU 保证梯度始终流动，且计算便宜，稳定训练。DCGAN 的标准配置 slope=0.2。

## 八、自我检验

- [ ] 能写出 LeakyReLU 公式、导数公式（分段），知道默认 α=0.01
- [ ] 能说清"梯度 0.01 ≠ 梯度 0"是解决死亡神经元的关键
- [ ] 知道 α→0 退化为 ReLU、α→1 退化为线性函数
- [ ] 能写出手写反向传播的 autograd.Function 版本（含 None 返回）
- [ ] 知道 PReLU/RReLU/ELU 与 LeakyReLU 的关系
- [ ] 知道 LeakyReLU 在 GAN/检测中的典型用法（slope=0.2）
- [ ] 能回答 7 个面试追问
