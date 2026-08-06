# InstanceNorm（实例归一化）

> 本模块索引见 [归一化技术详解](归一化技术详解.md)

## 一、定义与公式

InstanceNorm（IN，实例归一化，Ulyanov et al., 2016）对**单个样本、单个通道**的 [H, W] 空间维度做归一化。它是归一化家族中最"个体化"的一员：不跨样本、也不跨通道。

对四维特征 $x \in \mathbb{R}^{N \times C \times H \times W}$：

$$\mu_{n,c} = \frac{1}{HW}\sum_{h,w} x_{n,c,h,w}, \qquad \sigma_{n,c}^2 = \frac{1}{HW}\sum_{h,w} (x_{n,c,h,w} - \mu_{n,c})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}, \qquad y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

- $\gamma_c, \beta_c$：每通道可学习参数（`affine=True` 时，默认无仿射参数）；
- 统计量个数：$N \times C$ 个（每个样本每个通道一组）。

| 属性 | 值 |
|------|-----|
| 统计维度 | H、W（单样本单通道） |
| 统计量个数 | N × C 个 |
| 依赖 batch | 无 |
| 训练/推理行为 | 默认一致（track_running_stats=False） |
| 可学习参数 | γ、β 每通道（可选 affine） |
| 典型场景 | 风格迁移（AdaIN）、图像生成（StyleGAN 变体） |

## 二、数学性质

### 2.1 去除"亮度/对比度"，保留"纹理/结构"

IN 把每个样本每个通道的均值和方差全部移除。图像里，均值对应**亮度**（全局直流分量）、方差对应**对比度**（动态范围）。IN 把这些"风格性"信息剥掉，剩下的就是与内容相关的纹理结构——这正是风格迁移想要的：

$$\text{IN}(x) = \frac{x - \mu_{n,c}}{\sigma_{n,c}} \quad \Rightarrow \quad \text{每个通道输出} \sim \mathcal{N}(0, 1) \text{（不区分样本）}$$

### 2.2 与 BN 的关系（从 BN 角度看）

BN 统计的是 $\mu_c$（跨 N、H、W），IN 统计的是 $\mu_{n,c}$（只跨 H、W）。若 batch 内所有样本分布相同，则 $\mathbb{E}_N[\mu_{n,c}] = \mu_c$——**IN 是"BN 在每个样本上的分解"**，把 batch 级归一化细化到实例级。batch=1 时，IN ≡ BN（统计量相同）。

### 2.3 风格统计量（AdaIN 的核心）

IN 归一化后，每个通道的均值/方差携带了"这个样本在这个通道上的风格"信息。风格迁移把内容特征用风格特征的 μ/σ 重新调制（见 3.3 AdaIN），实现任意风格注入。

### 2.4 梯度（三项分解，与 LN 同构）

IN 的反传与 LN 完全同构，只是"特征维"换成"H、W 维"。设 $g_i = \partial L / \partial \hat{x}_i$，块内共 $M = H \cdot W$ 个元素：

$$\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( g_i - \frac{1}{M}\sum_j g_j - \frac{\hat{x}_i}{M} \sum_j g_j \hat{x}_j \right)$$

同样具有"梯度块内均值为 0、与 $\hat{x}$ 正交"的性质，且每个 (n, c) 块完全独立，反传天然按块解耦。

### 2.5 对比度不变性

IN 同时移除每通道均值和方差 → 对"整体亮度平移"和"对比度缩放"都不变（亮度/对比度属于风格属性）。保留下来的是**像素间的相对结构**（纹理、边缘、形状）——内容信息。

## 三、源码实现

### 3.1 手写 IN 前向 + 与 nn.InstanceNorm2d 对齐

```python
import torch
import torch.nn as nn

def instance_norm_manual(x, gamma, beta, eps=1e-5):
    # x: [N, C, H, W]，对每个 (n, c) 的 H、W 统计
    mean = x.mean(dim=(2, 3), keepdim=True)                       # [N, C, 1, 1]
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

torch.manual_seed(0)
x = torch.randn(4, 3, 8, 8)
y1 = instance_norm_manual(x, torch.ones(3), torch.zeros(3))
y2 = nn.InstanceNorm2d(3, affine=True)(x)
print((y1 - y2).abs().max().item())     # 4.77e-07：完全对齐
```

### 3.2 每样本每通道独立（显式张量演示）

```python
# x: [2, 2, 1, 3]，四个 (样本, 通道) 组合的输入各不相同
x2 = torch.tensor([[[[1.0, 2.0, 6.0]], [[1.0, 1.0, 4.0]]],
                   [[[3.0, 4.0, 11.0]], [[2.0, 5.0, 8.0]]]])
in2 = nn.InstanceNorm2d(2, affine=True)
out = in2(x2)
print(out[0, 0].flatten().tolist())     # [-0.9258, -0.4629, 1.3887]  ← 只统计 [1,2,6]
print(out[0, 1].flatten().tolist())     # [-0.7071, -0.7071, 1.4142]  ← 只统计 [1,1,4]
print(out[1, 0].flatten().tolist())     # [-0.8429, -0.5620, 1.4049]  ← 只统计 [3,4,11]
print(out[1, 1].flatten().tolist())     # [-1.2247, 0.0, 1.2247]      ← 只统计 [2,5,8]
```

四个位置的归一化因子互不影响——每个 (样本, 通道) 独立计算。

### 3.3 train/eval 一致性（与 BN 的鲜明对比）

```python
torch.manual_seed(0)
inn = nn.InstanceNorm2d(3)              # 默认 track_running_stats=False
x = torch.randn(4, 3, 8, 8)
inn.train(); a = inn(x)
inn.eval();  b = inn(x)
print((a - b).abs().max().item())       # 0.0：IN 两阶段行为一致（无 running stats）
```

> 若设置 `track_running_stats=True`，IN 会像 BN 一样累计 running stats 并用于推理——但实践极少用。

### 3.4 batch=1 时 IN ≡ BN（关系验证）

```python
torch.manual_seed(0)
x = torch.randn(1, 4, 8, 8)                 # batch=1
in_y = nn.InstanceNorm2d(4, affine=True)(x)
bn_y = nn.BatchNorm2d(4)(x)                 # batch=1 的 BN：统计量只剩 H、W
print((in_y - bn_y).abs().max().item())     # 0.0：batch=1 时数学上等价（统计量只剩 H、W）
```

### 3.5 手写反向验证（与 autograd 对齐）

```python
class InstanceNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        std = torch.sqrt(var + eps)
        x_hat = (x - mean) / std
        ctx.save_for_backward(x_hat, gamma, std)
        return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, gamma, std = ctx.saved_tensors
        M = x_hat.shape[2] * x_hat.shape[3]
        g = grad_output * gamma.view(1, -1, 1, 1)
        g_mean = g.mean(dim=(2, 3), keepdim=True)
        g_hat = (g * x_hat).mean(dim=(2, 3), keepdim=True) * x_hat
        grad_x = (g - g_mean - g_hat) / std
        grad_gamma = (grad_output * x_hat).sum(dim=(0, 2, 3))
        grad_beta = grad_output.sum(dim=(0, 2, 3))
        return grad_x, grad_gamma, grad_beta

torch.manual_seed(0)
x1 = torch.randn(2, 3, 8, 8, requires_grad=True)
in_ref = nn.InstanceNorm2d(3, affine=True)
y1 = in_ref(x1)
g = torch.randn_like(y1)
y1.backward(g)

x2 = x1.detach().clone().requires_grad_(True)
y2 = InstanceNormFunction.apply(x2, in_ref.weight, in_ref.bias)
y2.backward(g)
print((x1.grad - x2.grad).abs().max().item())   # 4.77e-7：手写反向与官方一致
```

### 3.6 风格迁移核心：AdaIN（Adaptive Instance Normalization）

```python
def adain(content, style, eps=1e-5):
    """把内容特征的分布对齐到风格特征的分布（Huang et al., 2017）"""
    mc = content.mean(dim=(2, 3), keepdim=True)
    sc = content.std(dim=(2, 3), keepdim=True)
    ms = style.mean(dim=(2, 3), keepdim=True)
    ss = style.std(dim=(2, 3), keepdim=True)
    return ss * (content - mc) / (sc + eps) + ms

torch.manual_seed(0)
content = torch.randn(1, 3, 8, 8)                       # 内容特征
style = torch.randn(1, 3, 8, 8) * 2.0 + 5.0             # 风格特征（均值 5、标准差 2）
out = adain(content, style)
print(out.mean().item(), out.std().item())              # 5.1433 1.9080
print(style.mean().item(), style.std().item())          # 5.1433 1.9081
# 输出的均值/方差 = 风格的均值/方差：内容结构 + 风格统计，一次前向完成风格迁移
```

## 四、深入分析

### 4.1 梯度与训练行为

- 反传公式与 LN 同构（只是把"特征维"换成"H、W 维"）：三项分解中均值/方差路径都存在，梯度在每个 (n,c) 块内均值为 0；
- IN 的统计量是**纯确定性变换**（无 batch 噪声），因此没有 BN 的正则化副作用，也没有 BN 的 train/eval 不一致；
- 每个 (n,c) 块的统计独立 → 反传也天然按块解耦，容易并行/算子融合。

### 4.2 为什么不适合判别式任务（分类/检测）

IN 会**移除样本间的判别性信息**：图片亮度、对比度差异（可能正是分类的线索）被归一化抹平；而分类任务恰恰需要跨样本的对比。因此 IN 在判别任务上不如 BN/GN，主要服务于**生成/风格任务**——那里"风格信息被剥离"正是目标。

### 4.3 数值行为

- 与 LN 类似，统计量只依赖单样本，batch=1 完全可用；
- **特征图很小时不稳定**：如 1×1 的特征图（全局池化后的瓶颈层），H·W=1 时方差为 0，IN 输出恒为 0——这与 BN batch=1 崩溃同理；
- eps 默认 1e-5，混合精度下建议不小于 1e-6。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 与 batch 无关，训练/推理行为一致 | 移除亮度/对比度等判别性信息，不擅判别任务 |
| 天然契合风格迁移（AdaIN 的理论基础） | 每 (n,c) 独立统计，统计量个数多（N×C），样本/通道多时开销大 |
| 实现简单、无 running stats | 特征图空间尺寸小时统计不稳（如 1×1 输出为 0） |
| 反传按块解耦，计算高效 | 没有 BN 的正则化效应 |
| 生成任务（图像/视频生成）效果好 | 论文领域集中在图像，NLP/多模态场景验证少 |

## 六、与同类归一化对比

| 维度 | InstanceNorm | BatchNorm | LayerNorm | GroupNorm |
|------|-------------|-----------|-----------|-----------|
| 统计维度 | H、W（单样本单通道） | N、H、W（跨样本按通道） | C、H、W（单样本全通道） | 组内通道的 H、W |
| 统计量个数 | N×C | C | N | N×G |
| 依赖 batch | 无 | 强 | 无 | 无 |
| 训练/推理一致 | 一致（默认） | 不一致 | 一致 | 一致 |
| 典型场景 | 风格迁移、图像生成 | CNN 分类检测 | Transformer | 小 batch 检测分割 |
| 代表工作 | AdaIN、StyleGAN 系 | ResNet | BERT、ViT | Mask R-CNN |

**一句话**：IN 是"每个样本每个通道自己归一化"；GN 是它的折中——把通道分组共享统计量，降低了统计量噪声（见 GroupNorm 篇）。

## 七、高频面试问答

**Q1：InstanceNorm 统计哪一维？**
单个样本、单个通道内的 H、W 维：μ、σ² 各有 N×C 个。每个 (样本, 通道) 完全独立。

**Q2：IN 和 BN 的关系？**
BN 统计跨样本的通道统计（μ_c），IN 细化到每个样本（μ_nc）。batch=1 时两者数学上等价；IN 是 BN 在"实例级"的分解，去掉了 batch 依赖。

**Q3：为什么 IN 适合风格迁移？**
IN 移除每通道的均值/方差（亮度/对比度），保留纹理结构；AdaIN 再把风格特征的 μ/σ 乘加回去，实现"内容结构 + 风格统计"的合成——这一进一出正是风格迁移的本质操作。

**Q4：IN 为什么不适合分类任务？**
分类需要样本间的判别差异（亮度、对比度等整体统计特征可能是类别线索），IN 把这些信息全部抹平；且无 batch 统计噪声，缺少 BN 的隐式正则。

**Q5：IN 的 train/eval 行为？**
默认 track_running_stats=False，两阶段完全一致（无 running stats），这点与 LN/GN 相同、与 BN 不同。

**Q6：特征图是 1×1 时 IN 会怎样？**
H·W=1 时方差为 0，输出恒等于 β（或 0）——与 BN batch=1 崩溃同理，特征图过小的层不要用 IN。

**Q7：IN 和 GN 的关系？**
GN 把通道分组后统计（G=C 时退化为 IN）。IN 是 GN 的极端情况（每组 1 个通道），统计量噪声最大；GN 用组内共享降低噪声，是小 batch 判别任务的推荐。

**Q8：StyleGAN 里 IN 怎么用的？**
StyleGAN 的调制解调（modulation/demodulation）本质是"AdaIN 的权重化版本"：用 style 向量调制卷积权重而非直接调制激活，配合解调保持输出尺度——同一思想（风格统计注入）的两种实现。

## 八、自我检验

- [ ] 能写出 IN 公式并指出统计维度（H、W 单样本单通道）
- [ ] 能说出统计量个数 N×C 及与 BN 的关系（batch=1 时等价）
- [ ] 能手写 IN 前向并与 nn.InstanceNorm2d 对齐
- [ ] 能写 AdaIN 并验证"输出统计量 = 风格统计量"
- [ ] 能解释 IN 为什么适合风格迁移、不适合分类
- [ ] 知道 IN 默认无 running stats（train/eval 一致）
- [ ] 知道 1×1 特征图时 IN 崩溃的边界情况
- [ ] 能回答 8 个面试追问
