# BatchNorm（批归一化）

> 本模块索引见 [归一化技术详解](归一化技术详解.md)

## 一、定义与公式

BatchNorm（BN，批归一化）是 CNN 体系的标准配置（ResNet 每个 block 都有）。它对**每个通道、跨 batch 内所有样本**求均值方差，把激活归一化到均值为 0、方差为 1 的分布。

对四维特征 $x \in \mathbb{R}^{N \times C \times H \times W}$，第 $c$ 个通道的统计量：

$$\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n,h,w} x_{n,c,h,w}, \qquad \sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n,h,w} (x_{n,c,h,w} - \mu_c)^2$$

归一化与仿射变换：

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

- $\gamma_c, \beta_c$：每个通道一组可学习参数（初始 $\gamma=1, \beta=0$）；
- $\epsilon$：防除零小常数（PyTorch 默认 1e-5）；
- **训练用 batch 统计量，推理用累计的 running_mean / running_var**（见第三节）。

| 属性 | 值 |
|------|-----|
| 统计维度 | N、H、W（跨样本，按通道） |
| 统计量个数 | C 个（每个通道一组 μ、σ²） |
| 依赖 batch | 强依赖（batch 大小决定统计质量） |
| 训练/推理行为 | **不一致**（batch 统计 vs 移动平均统计） |
| 典型位置 | Conv → BN → ReLU，全连接层也可用 |

## 二、数学性质

### 2.1 归一化保证激活分布稳定

在训练模式下，BN 层的输出（忽略 γ、β）满足：

$$\mathbb{E}[y] \approx \beta, \qquad \mathrm{Var}[y] \approx \gamma^2$$

即输出均值只由 $\beta$ 决定、方差只由 $\gamma$ 决定，与输入尺度无关。这抑制了**内部协变量偏移（Internal Covariate Shift）**：前层参数更新导致的输入分布漂移被逐层吸收。

### 2.2 running stats 的递推公式（必考）

训练中每个 step 用指数移动平均（EMA）更新累计统计量：

$$\text{running\_mean} \leftarrow (1 - m) \cdot \text{running\_mean} + m \cdot \mu_{batch}$$

$$\text{running\_var} \leftarrow (1 - m) \cdot \text{running\_var} + m \cdot \sigma^2_{batch}$$

$m$ 即 momentum（默认 0.1）。**推导要点**：$m$ 越小，历史贡献保留越多，统计量越平滑但收敛越慢；$m=0.1$ 意味着每个新 batch 只贡献 10% 的信息。

> **PyTorch 细节（易踩坑）**：更新 running_var 用的是**无偏方差**（除以 $N-1$），而前向归一化用的是**有偏方差**（除以 $N$）。两者的差异在 batch 小时不可忽略，手写复现时要用 `torch.var(x, dim=..., unbiased=False)` 归一化、`unbiased=True` 更新 running_var（PyTorch 源码即如此）。

### 2.3 一个关键性质：归一化的"撤销性"

BN 是"有界的平移缩放变换"，网络可以通过学出 $\gamma_c = \sigma_c$、$\beta_c = \mu_c$ 完全撤销归一化——因此 BN **不会损失表达能力**，只会改变优化曲面。

## 三、源码实现

### 3.1 手写 BN 前向（训练模式，batch 统计量）

```python
import torch
import torch.nn as nn

def batch_norm_forward_train(x, gamma, beta, eps=1e-5):
    # x: [N, C, H, W]，沿 N,H,W 求每个通道的统计量
    mean = x.mean(dim=(0, 2, 3), keepdim=True)          # [1, C, 1, 1]
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # 有偏方差（除 N·H·W）
    x_hat = (x - mean) / torch.sqrt(var + eps)
    y = gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)
    return y, mean, var, x_hat

x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]])  # [2,1,2,2]
gamma = torch.tensor([1.0]); beta = torch.tensor([0.0])
y, mean, var, x_hat = batch_norm_forward_train(x, gamma, beta)
print(mean.flatten())     # tensor([4.5])
print(var.flatten())      # tensor([5.25])
print(x_hat.flatten())    # tensor([-1.5275, -1.0911, -0.6547, -0.2182, 0.2182, 0.6547, 1.0911, 1.5275])
```

### 3.2 完整手写 Module：running stats 更新 + eval 行为（本篇重点）

```python
class BatchNorm1dManual(nn.Module):
    """手写 BN：完整复现 PyTorch 的 train/eval 两阶段行为"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))  # 初始 0
        self.register_buffer("running_var", torch.ones(num_features))    # 初始 1

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            with torch.no_grad():   # running stats 不走梯度
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                # 注意：PyTorch 用无偏方差更新 running_var
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * x.var(dim=0, unbiased=True))
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:   # eval：用 running stats，不再计算 batch 统计量
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

bn = BatchNorm1dManual(2)
x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # N=3, C=2
out1 = bn(x1)                                             # 训练模式
print(out1)               # tensor([[-1.2247, -1.2247], [0., 0.], [1.2247, 1.2247]])
print(bn.running_mean)    # tensor([0.3000, 0.4000])  ← 0.1×batch_mean
print(bn.running_var)     # tensor([1.3000, 1.3000])  ← 0.9×1 + 0.1×4（无偏方差）

x2 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
bn(x2)                    # 第二个 batch
print(bn.running_mean)    # tensor([1.1700, 1.3600])
print(bn.running_var)     # tensor([1.5700, 1.5700])

# ===== 关键演示：同样的输入，eval 模式输出完全不同 =====
bn.eval()
out1_eval = bn(x1)
print(out1)               # tensor([[-1.2247, -1.2247], [0., 0.], [1.2247, 1.2247]])
print(out1_eval)          # tensor([[-0.1357,  0.5108], [1.4605, 2.1069], [3.0567, 3.7031]])
```

**输出解读**：训练模式输出是"标准正态化"（均值 0、方差 1）；eval 模式用 running stats（尚未收敛到数据分布），输出跟训练时差异巨大——这就是"训练/推理行为不一致"的直接证据。模型训练完成后 `model.eval()` 是 BN 推理的**强制要求**。

### 3.3 与 nn.BatchNorm1d 对齐验证

```python
torch.manual_seed(1)
xr = torch.randn(16, 8)
manual = BatchNorm1dManual(8)
nn_bn = nn.BatchNorm1d(8)

y1 = manual(xr); y2 = nn_bn(xr)                 # 训练模式
print((y1 - y2).abs().max().item())             # 2.38e-07（浮点误差，完全对齐）

manual.eval(); nn_bn.eval()
y1e = manual(xr); y2e = nn_bn(xr)               # 推理模式
print((y1e - y2e).abs().max().item())           # 2.38e-07（含 running stats 也一致）
```

### 3.4 nn.BatchNorm2d 标准接口 + 模型内 train/eval 行为

```python
torch.manual_seed(0)
bn2 = nn.BatchNorm2d(num_features=4)            # 参数：gamma/beta 各 4 个 + running 缓冲
x3 = torch.randn(8, 4, 16, 16)
y3 = bn2(x3)                                    # 训练模式：用 batch 统计量
print(round(y3.mean().item(), 6), round(y3.var().item(), 6))   # 0.0 1.000112（已归一化）
bn2.eval()
y3e = bn2(x3)                                   # 推理模式：用 running stats
print(round(y3e.mean().item(), 6), round(y3e.var().item(), 6)) # -0.007416 1.003226
print(bn2.running_mean[:3])                     # tensor([-0.0029, 0.0042, 0.0013])
```

模型级演示（Conv+BN+ReLU），直观看到 train/eval 输出差异：

```python
torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
)
x4 = torch.randn(4, 3, 32, 32)
model.train(); y_t = model(x4)
model.eval();  y_e = model(x4)
print(round(y_t.mean().item(), 4), round(y_t.var().item(), 4))   # 0.3983 0.345（ReLU 后非零中心）
print(round(y_e.mean().item(), 4), round(y_e.var().item(), 4))   # 0.224 0.1161
print((y_t - y_e).abs().max().item())           # 2.0476（同一输入，两阶段输出天差地别）
```

> **推理部署优化**：因为 eval 时 BN 只是线性变换，可以**折叠进 Conv**（$w' = \gamma w / \sqrt{\sigma^2+\epsilon}$，$b' = \gamma(b-\mu)/\sqrt{\sigma^2+\epsilon} + \beta$），推理省掉整层 BN，见 4.4 代码验证。

## 四、深入分析

### 4.1 梯度（BN 反传的性质）

设 $g_i = \partial L / \partial \hat{x}_i$，BN 对输入 $x_i$ 的反传梯度为（$x_i$ 是某个通道内展平的第 $i$ 个元素，共 $N' = N \cdot H \cdot W$ 个）：

$$\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( g_i - \frac{1}{N'}\sum_j g_j - \frac{\hat{x}_i}{N'} \sum_j g_j \hat{x}_j \right)$$

**三条结论（面试加分）**：
1. 梯度有"去均值"项 $\frac{1}{N'}\sum g_j$ 和"去相关"项——BN 让梯度在 batch 内**均值为 0**、与 $\hat{x}$ 正交；
2. 反传梯度**依赖于整个 batch**，因此 BN 反传需要跨样本通信（分布式时这就是 SyncBN 存在的原因）；
3. 前向把 batch 统计量算好存下来，反向复用即可，与 Sigmoid 复用输出的技巧类似。

```python
torch.manual_seed(0)
x5 = torch.randn(4, 8, requires_grad=True)
bn5 = nn.BatchNorm1d(8)
bn5(x5).sum().backward()
print(x5.grad.shape)              # torch.Size([4, 8])
print(x5.grad.sum(dim=0))         # tensor([0., -5.68e-14, ..., 2.22e-16])：按通道求和 ≈ 0
```

### 4.2 小 batch 问题（BN 的最大软肋）

统计量是随机样本估计，其噪声反比于 batch 大小：

$$\mathrm{Var}(\hat{\mu}) = \frac{\sigma^2}{N}, \qquad \mathrm{Var}(\hat{\sigma}^2) \propto \frac{1}{N}$$

- **batch=1**：方差为 0，$\hat{x} = 0$（除以 $\epsilon$），输出直接塌缩成 $\beta$，完全无信息；
- **batch 小（2~8）**：统计噪声大，归一化引入随机扰动，训练不稳、BN 的收益消失；
- **实践上**：batch < 16 建议换 GN/LN；跨卡时用 SyncBN 把统计量收集到全局。

```python
torch.manual_seed(0)
for bs in [1, 2, 8, 64]:
    b = BatchNorm1dManual(16)           # 手写版（nn.BatchNorm1d 在 bs=1 时直接报错）
    yy = b(torch.randn(bs, 16))
    print(f"batch={bs}: out mean={yy.mean().item():.4f} var={yy.var().item():.4f}")
# batch=1 : out mean=0.0000 var=0.0000   ← 输出塌缩为 0
# batch=2 : out mean=-0.0000 var=1.0318  ← 统计噪声大，方差偏离 1
# batch=8 : out mean=0.0000 var=1.0079
# batch=64: out mean=-0.0000 var=1.0010
```

> **注意**：PyTorch ≥ 2.0 中 `nn.BatchNorm1d` 在 batch=1 训练时会抛 `ValueError: Expected more than 1 value per channel`，因为此时方差估计无意义。

### 4.3 momentum 与 running stats 的坑

1. **momentum 默认 0.1**：对长训练来说收敛得慢但稳；快速验证/小数据集可调大到 0.3；
2. **预训练 BN 参数的模型在没训过的域上直接推理**：running stats 与真实分布不符，输出被扭曲（OOD 退化问题）；
3. **fine-tune 时 BN 的 running stats 也会被更新**（只要在 train 模式），换数据集后旧统计会被逐渐覆盖；
4. **混合精度**：FP16 下统计量精度损失大，大模型预训练基本不用 BN。

### 4.4 BN 折叠（推理部署优化）验证

```python
torch.manual_seed(0)
conv = nn.Conv2d(3, 4, 3, padding=1)
bn6 = nn.BatchNorm2d(4)
bn6.eval()                                  # 折叠用的就是 eval 模式的 running stats
x6 = torch.randn(1, 3, 8, 8)
y6 = bn6(conv(x6))                          # 原模型输出

scale = (bn6.weight / torch.sqrt(bn6.running_var + bn6.eps)).view(4, 1, 1, 1)
w = conv.weight * scale
b = (conv.bias - bn6.running_mean) * scale.view(4) + bn6.bias
y6b = torch.nn.functional.conv2d(x6, w, b, padding=1)   # 折叠后的单 Conv 输出
print((y6 - y6b).abs().max().item())        # 1.79e-07：完全等价，推理省掉 BN 层
```

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| CNN 上收敛快，允许更大的学习率 | 强依赖 batch size，小 batch 统计不准、效果崩 |
| 对激活尺度不敏感，容忍大 lr | 训练/推理行为不一致（统计量来源不同） |
| 有轻微正则化效果（batch 统计噪声） | 不适合变长序列/Transformer（NLP 场景基本不可用） |
| 推理可折叠进 Conv，部署零成本 | 分布式训练需 SyncBN 才能保证统计一致性 |
| 与卷积的通道结构天然契合 | OOD 数据推理退化（running stats 失真） |

## 六、与同类归一化对比

| 维度 | BatchNorm | LayerNorm | GroupNorm |
|------|-----------|-----------|-----------|
| 统计维度 | 跨样本、按通道 (N,H,W) | 单样本、全特征 (C,H,W) | 单样本、组内通道 (H,W,组内) |
| 依赖 batch | 强 | 无 | 无 |
| 训练/推理一致性 | 不一致 | 一致 | 一致 |
| 可学习参数 | γ、β 每通道 | γ、β 每特征 | γ、β 每通道 |
| running stats | 有 | 无 | 无 |
| 适用场景 | CNN、大 batch | Transformer/NLP/ViT | 检测/分割小 batch |
| 小 batch 表现 | 差 | 好 | 好（G=32 时略逊 BN 大 batch） |

**一句话**：BN 用"别人的样本"帮你归一化，代价是必须依赖别人；LN/GN 只靠自己，因此自适应任何 batch 与序列长度。

## 七、高频面试问答

**Q1：BN 训练和推理的区别？**
训练用当前 batch 统计量（每个 step 实时计算），同时用 EMA 更新 running_mean/running_var；推理用 running stats 做固定线性变换，不再算 batch 统计量。所以 `model.eval()` 对 BN 是必须的。

**Q2：momentum 是梯度更新的动量吗？**
不是。BN 的 momentum（默认 0.1）控制 running stats 的 EMA 速度：$\text{running} \leftarrow (1-m)\text{running} + m \cdot \text{batch}$，与优化器的动量（累积梯度方向）完全是两回事。

**Q3：为什么 batch=1 时 BN 会崩？**
batch=1 时通道内方差为 0，归一化变成除以 $\epsilon$，输出恒等于 $\beta$，梯度无法传递有效信息（且 PyTorch 直接报错）。

**Q4：BN 为什么有正则化效果？**
batch 统计量本身是随机采样估计，含噪声；这给激活注入随机扰动，类似 dropout 的隐式正则（论文观察，非设计目标）。

**Q5：SyncBN 是什么？为什么需要？**
把跨卡的所有 batch 统计量先通信聚合再归一化，等效于扩大 batch。分布式训练 batch 小时（每卡只有几个样本）必须用，否则每卡统计独立、效果差。

**Q6：BN 对 1D 输入（如文本）能用吗？**
能（nn.BatchNorm1d），但 NLP 中序列变长 + 位置统计污染，效果差；Transformer 场景统一用 LN/RMSNorm。

**Q7：为什么 BN 能把 Conv 和 BN 合并？**
推理时 BN 是确定的线性仿射变换 y = γ(x-μ)/√(σ²+ε)+β，可以解析地并入上一层的 Conv 权重和偏置，推理图里少一层且省内存（见 4.4）。

**Q8：BN 的 running stats 初始值？**
running_mean=0、running_var=1（buffers 而非参数，不参与梯度更新，但会随模型保存/加载）。若从 checkpoint 加载预训练模型直接推理，跑出来的就是预训练域的统计。

## 八、自我检验

- [ ] 能写出 BN 公式与统计维度（N,H,W 按通道）
- [ ] 能手推 running stats 的 EMA 递推公式
- [ ] 知道 PyTorch 更新 running_var 用无偏方差、归一化用有偏方差
- [ ] 能解释"同一输入 train/eval 输出不同"并写代码演示
- [ ] 能推导 BN 反传的三项分解公式并说出"梯度 batch 内均值为 0"
- [ ] 知道 batch=1 崩溃的原因与解决手段（GN/LN/SyncBN）
- [ ] 能写出 Conv+BN 折叠公式并验证
- [ ] 能回答 8 个面试追问
