# LayerNorm（层归一化）

> 本模块索引见 [归一化技术详解](归一化技术详解.md)

## 一、定义与公式

LayerNorm（LN，层归一化）对**单个样本的全部特征维度**做归一化，完全不依赖 batch 内其他样本。这是它与 BatchNorm 最本质的区别，也是 Transformer 选择它的根本原因。

对输入 $x \in \mathbb{R}^{N \times L \times D}$（batch、序列长度、特征维度），对每个 (n, l) 位置独立计算：

$$\mu_{n,l} = \frac{1}{D}\sum_{d=1}^{D} x_{n,l,d}, \qquad \sigma_{n,l}^2 = \frac{1}{D}\sum_{d=1}^{D} (x_{n,l,d} - \mu_{n,l})^2$$

$$\hat{x}_{n,l,d} = \frac{x_{n,l,d} - \mu_{n,l}}{\sqrt{\sigma_{n,l}^2 + \epsilon}}, \qquad y_{n,l,d} = \gamma_d \hat{x}_{n,l,d} + \beta_d$$

- $D$：特征维数（Transformer 中即 d_model）；
- $\gamma, \beta$：形状为 $[D]$ 的可学习参数（初始 $\gamma=1, \beta=0$）；
- LN 在 CNN 上则对整个 $(C,H,W)$ 统计（`nn.LayerNorm([C,H,W])`）。

| 属性 | 值 |
|------|-----|
| 统计维度 | 单样本全特征（NLP 中每个 token 内部） |
| 统计量个数 | $N \times L$ 个（每个位置一组） |
| 依赖 batch | 无（batch=1 也能跑） |
| 训练/推理行为 | **完全一致**（无 running stats） |
| 可学习参数 | γ、β 每特征维度 |
| 典型位置 | Transformer 的每层前、输出前（Pre-LN） |

## 二、数学性质

### 2.1 输出约束

忽略 γ、β，LN 输出满足：

$$\mathbb{E}_{d}[\hat{x}_{n,l,\cdot}] = 0, \qquad \mathrm{Var}_{d}[\hat{x}_{n,l,\cdot}] = 1$$

即每个位置的特征向量被"拉"到单位方差超球附近（方向保留、模长固定）。

### 2.2 各向同性缩放性质（LN 梯度稳定的关键，面试加分）

考虑一个位置的特征向量 $x \in \mathbb{R}^D$，LN 输出 $\hat{x} = (x - \mu\mathbf{1}) / \sqrt{\sigma^2+\epsilon}$。其雅可比矩阵为：

$$\frac{\partial \hat{x}}{\partial x} = \frac{1}{\sqrt{\sigma^2+\epsilon}} \left( I - \frac{\mathbf{1}\mathbf{1}^\top}{D} - \frac{\hat{x}\hat{x}^\top}{D} \right)$$

**推导**（对 $\hat{x}_i$ 求偏导，注意 $\mu$、$\sigma^2$ 都依赖 $x$，分三路求导）：

$$\frac{\partial \hat{x}_i}{\partial x_j} = \underbrace{\frac{\delta_{ij}}{\sqrt{\sigma^2+\epsilon}}}_{\text{直接项}} - \underbrace{\frac{1}{D\sqrt{\sigma^2+\epsilon}}}_{\text{经 } \mu} - \underbrace{\frac{\hat{x}_i \hat{x}_j}{D\sqrt{\sigma^2+\epsilon}}}_{\text{经 } \sigma^2}$$

三项分别来自 $x_j$ 影响 $\hat{x}_i$ 本身、影响均值 $\mu$、影响方差 $\sigma^2$。

**性质**：括号内矩阵作用在任意向量上等价于"去掉该向量均值分量 + 去掉沿 $\hat{x}$ 方向的分量"，**各方向被均匀缩放**（特征值只有两种取值，且只差常数倍）。这意味着反传梯度不会在方向上有选择性地放大/缩小——梯度在深网络中传递时方向不被扭曲，这是 LN 稳定深层 Transformer 训练的重要机理。

### 2.3 平移不变性

对输入整体加常数，LN 输出不变（均值被减掉）：

$$f(x + c\mathbf{1}) = f(x)$$

这使得 LN 对输入的绝对尺度不敏感。

## 三、源码实现

### 3.1 手写前向 + 与 nn.LayerNorm 逐位对齐

```python
import torch
import torch.nn as nn

def layer_norm_manual(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdim=True)                                   # [..., 1]
    var = x.var(-1, unbiased=False, keepdim=True)                     # 有偏方差
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

torch.manual_seed(0)
x = torch.randn(4, 6, 8)                        # [N, L, D]
y1 = layer_norm_manual(x, torch.ones(8), torch.zeros(8))
y2 = nn.LayerNorm(8)(x)
print((y1 - y2).abs().max().item())             # 2.38e-07：完全对齐（默认 eps 同为 1e-5）
```

### 3.2 手写反向（autograd.Function，三项分解的代码形态）

```python
class LayerNormFunction(torch.autograd.Function):
    """手写 LN：前向保存 x_hat、gamma、std，反向复用三项分解公式"""

    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        x_hat = (x - mean) / std
        ctx.save_for_backward(x_hat, gamma, std)
        return gamma * x_hat + beta

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, gamma, std = ctx.saved_tensors
        g = grad_output * gamma                        # ∂L/∂x_hat
        g_mean = g.mean(-1, keepdim=True)              # 均值路径
        g_hat = (g * x_hat).mean(-1, keepdim=True) * x_hat   # 方差路径
        grad_x = (g - g_mean - g_hat) / std            # 三项分解
        grad_gamma = (grad_output * x_hat).sum(dim=tuple(range(grad_output.dim() - 1)))
        grad_beta = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
        return grad_x, grad_gamma, grad_beta, None

torch.manual_seed(0)
x1 = torch.randn(4, 6, 8, requires_grad=True)
ln = nn.LayerNorm(8)
y1 = ln(x1)
g = torch.randn_like(y1)
y1.backward(g)                                        # 标准反传

x2 = x1.detach().clone().requires_grad_(True)
y2 = LayerNormFunction.apply(x2, ln.weight, ln.bias, 1e-5)
y2.backward(g)
print((x1.grad - x2.grad).abs().max().item())         # 4.17e-07：手写反向与官方一致
```

公式级验证（不依赖官方实现，直接对公式与 autograd 对比）：

```python
torch.manual_seed(0)
xa = torch.randn(3, 5, requires_grad=True)
ya = (xa - xa.mean(-1, keepdim=True)) / torch.sqrt(xa.var(-1, unbiased=False, keepdim=True) + 1e-5)
ga = torch.randn_like(ya)
ya.backward(ga)

x_hat = (xa.detach() - xa.mean(-1, keepdim=True)) / torch.sqrt(xa.var(-1, unbiased=False, keepdim=True) + 1e-5)
std = torch.sqrt(xa.var(-1, unbiased=False, keepdim=True) + 1e-5)
formula = (ga - ga.mean(-1, keepdim=True) - x_hat * (ga * x_hat).mean(-1, keepdim=True)) / std
print((xa.grad - formula).abs().max().item())         # 1.19e-07：公式正确
```

### 3.3 train/eval 一致性演示（与 BN 的关键差异）

```python
torch.manual_seed(0)
ln2 = nn.LayerNorm(8)
x = torch.randn(4, 6, 8)
ln2.train(); a = ln2(x)
ln2.eval();  b = ln2(x)
print((a - b).abs().max().item())                     # 0.0：LN 两阶段输出完全一致
```

> 对照 BatchNorm 篇：同一输入 train/eval 差异达 2.0 量级。**LN 无需 running stats、无需 eval()**，这是它适合推理友好的根本原因。

### 3.4 变长序列友好演示

```python
torch.manual_seed(0)
x6 = torch.randn(2, 8, 16)    # 两个样本序列长度不同（如 5 和 8）也能拼一个 batch
y = nn.LayerNorm(16)(x6)
print(y.mean(-1).abs().max().item())                  # 5.96e-08：每个位置输出均值 ≈ 0
print(y.var(-1, unbiased=False).max().item())         # 0.99999：每个位置输出方差 ≈ 1
# 统计量只在每个位置内部（16 维特征）计算，与 batch 大小、序列长度无关
```

### 3.5 在 Transformer Block 中的使用（Pre-LN 结构）

```python
class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer Block：LN 在子层之前，残差路径保持恒等"""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

torch.manual_seed(0)
block = TransformerBlock(64, 4, 128)
y = block(torch.randn(2, 10, 64))
print(y.shape)                                        # torch.Size([2, 10, 64])
```

## 四、深入分析

### 4.1 梯度性质与数值行为

- **梯度均值被移除**：由 2.2 的公式，$\partial L/\partial x$ 在特征维度上求和为 0——归一化使梯度不带"直流分量"，避免梯度在反传中累积偏置；
- **无梯度消失风险**：$\hat{x}$ 方向分量被移除，残差路径 + LN 的组合下梯度乘积稳定（对比 Sigmoid 的 ≤0.25 衰减，见激活函数模块）；
- **数值上**：`mean/var` 在 FP16 下精度差，官方实现与 HF 实现都会在 FP32 下计算统计量，再转回输入精度。

### 4.2 为什么 Transformer 用 LN 而不是 BN（必考，至少答 4 条）

1. **与 batch 无关**：batch=1（单条生成）、任意 batch 大小都能正常归一化；
2. **变长序列友好**：BN 跨样本统计会被 padding 位置污染；LN 每个 token 独立统计，天然免疫；
3. **训练/推理一致**：无 running stats，两阶段行为相同（见 3.3 演示）；
4. **统计量数量充足**：一个 token 的特征维 $D$ 通常很大（512~8192），统计量稳健；而 BN 的通道统计依赖 batch 大小；
5. **与逐 token 预测任务匹配**：每个位置的表示被独立规范化，attention 的相似度计算在统一尺度上进行。

### 4.3 LN 的位置选择：Pre-LN vs Post-LN

```text
Post-LN（原版 Transformer）: x → Attn → 残差+ → LN → FFN → 残差+ → LN
Pre-LN（现代主流，含 ViT）:  x → LN → Attn → 残差+ → LN → FFN → 残差+ → x'
```

Pre-LN 的残差捷径上没有任何变换，梯度恒为 1，深层模型可训；Post-LN 需要 warmup 才能训练深模型（详见总览篇）。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 与 batch 无关，任意 batch/序列长度可用 | 统计量在特征维上计算，特征维小时统计不稳（如 D<16 的浅层） |
| 训练/推理行为完全一致 | 归一化"样本内"尺度，对样本间相对信息不敏感 |
| 梯度方向不被扭曲（各向同性缩放） | 比 BN 少正则化作用（无跨样本统计噪声） |
| 无需 running stats，实现简单、部署友好 | 计算量比 RMSNorm 高（多一次均值归约与减法） |
| Transformer/NLP/ViT/多模态全场景通吃 | 对绝对尺度的敏感性因样本而异（与输入分布耦合） |

## 六、与同类归一化对比

| 维度 | LayerNorm | BatchNorm | RMSNorm |
|------|-----------|-----------|---------|
| 统计维度 | 单样本全特征 | 跨样本按通道 | 单样本全特征（无均值） |
| 依赖 batch | 无 | 强 | 无 |
| 训练/推理一致 | 一致 | 不一致 | 一致 |
| 均值减除 | 有 | 有 | 无 |
| 可学习参数 | γ、β | γ、β | γ（一般） |
| 计算量 | 中 | 中 | 低（约省 30%） |
| 代表模型 | BERT、ViT、原版 Transformer | ResNet 等 CNN | LLaMA、Qwen、Mistral |
| 小 batch / 变长 | 好 | 差 | 好 |

**一句话**：BN 靠 batch 归一化、LN 靠样本自身归一化；RMSNorm 是 LN 去掉均值中心化的简化版（详见 RMSNorm 篇）。

## 七、高频面试问答

**Q1：LayerNorm 和 BatchNorm 的本质区别？**
归一化维度不同：BN 跨样本按通道统计（依赖 batch、训练推理不一致），LN 单样本全特征统计（与 batch 无关、两阶段一致）。这决定了 BN 适合 CNN、LN 适合 Transformer。

**Q2：LN 的梯度有什么特点？**
反传梯度在特征维上求和为 0，且雅可比矩阵各向同性缩放（梯度方向不被扭曲）；配合残差结构，梯度乘积稳定，深层可训。

**Q3：为什么 LN 不需要 running stats？**
LN 的统计量（μ、σ²）每步都由当前输入实时计算，且只依赖单样本自身，因此训练和推理的公式完全一样；BN 的统计量依赖整个 batch，推理时 batch 不存在（或太小），必须换成训练期累计的 running stats。

**Q4：LN 和 GN 什么关系？**
GN 在 CNN 上等价于"通道分组的 LN"：G=1 时 GN 与 LN（对整个 [C,H,W]）数学上完全等价（可写代码验证，见 GroupNorm 篇）；GN 是 LN 思想在 CNN 上的推广。

**Q5：LN 的 eps 为什么需要存在？能设很大吗？**
防除零 + 防止方差极小时数值爆炸。通常 1e-5；BF16/FP16 训练下建议 ≥1e-6；设到 1e-2 会显著扭曲归一化结果（归一化不彻底），一般不用。

**Q6：Pre-LN 和 Post-LN 各自优缺点？**
Pre-LN 残差捷径无变换、梯度恒为 1、可大 lr、深模型稳定，现代主流；Post-LN 是原版结构，表示能力理论更强但需要 warmup、深模型训练困难。

**Q7：为什么说 LN 对变长序列友好？**
LN 每个 token 独立统计，padding 位置只是自己参与自己的归一化，不会像 BN 那样把 padding 的分布混进统计量；因此变长 batch 可以直接 padding 后一起前向。

**Q8：多模态里 CLIP 双塔为什么 LN 后面还要 L2 归一化特征？**
两回事：LN 稳定训练、L2 归一化统一模长用于余弦相似度。LN 减均值会改变方向，所以相似度计算必须在 LN 之后再单独做 L2 归一化（详见总览篇第七节）。

## 八、自我检验

- [ ] 能写出 LN 公式并指出统计维度（单样本全特征）
- [ ] 能手推 LN 雅可比矩阵的三项分解
- [ ] 能解释"各向同性缩放"为何让梯度稳定
- [ ] 能写手写前向/反向并与 nn.LayerNorm 对齐验证
- [ ] 能说出 Transformer 用 LN 不用 BN 的 4 条理由
- [ ] 能演示 LN train/eval 输出一致（对比 BN 篇的不一致）
- [ ] 能区分 Pre-LN / Post-LN 并说明现代模型的选择
- [ ] 能回答 8 个面试追问
