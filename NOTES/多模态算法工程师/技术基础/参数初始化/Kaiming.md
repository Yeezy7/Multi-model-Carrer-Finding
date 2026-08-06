# Kaiming 初始化（He 初始化）

> 本模块索引见 [参数初始化详解](参数初始化详解.md)

## 一、定义与公式

Kaiming 初始化（He et al., 2015，论文 *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*）是 ReLU 系网络的默认初始化。它修正了 Xavier 的一个致命问题：**ReLU 把信号方差减半**，导致 Xavier 尺度下每过一层方差衰减一半，深层网络梯度消失。

### 1.1 核心公式（高斯基）目标方差

$$Var(w) = \frac{2}{n_{in}}$$

**高斯分布版本**（论文原版）：

$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**均匀分布版本**（PyTorch 默认）：

$$w \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \; \sqrt{\frac{6}{n_{in}}}\right]$$

推导：$U[-a,a]$ 方差为 $a^2/3$，令 $a^2/3 = 2/n_{in}$ 得 $a = \sqrt{6/n_{in}}$。

### 1.2 与 Xavier 的直观对比

| 维度 | Xavier | Kaiming |
|------|--------|---------|
| 目标方差 | $\frac{2}{n_{in}+n_{out}}$ | $\frac{2}{n_{in}}$ |
| 差别 | 基准 | **额外 ×2**（补偿 ReLU 减半） |
| 适用激活 | Sigmoid/Tanh | ReLU/LeakyReLU 族 |

### 1.3 参数含义

- **fan_in**（输入扇入）：一层内一个神经元接收的输入个数（对 Linear 是输入维度，对 Conv 是 $C_{in} \times K \times K$）；
- **fan_out**（输出扇出）：一层内一个神经元影响的输出个数（Linear 是输出维度，Conv 是 $C_{out} \times K \times K$）；
- **a**：负半轴斜率，LeakyReLU/PReLU 的参数。$a=0$ 时是标准 ReLU，公式退化为 $2/n$；$a>0$ 时负半轴也保留梯度，方差损失变小，目标方差变为 $\frac{2}{(1+a^2) n}$。

## 二、数学原理

### 2.1 ReLU 方差减半推导（核心）

设前一层输出 $x_{l-1} \sim \mathcal{N}(0, \sigma^2)$（均值 0、方差 $\sigma^2$），过 ReLU 后 $z = \max(0, x_{l-1})$。

ReLU 把负半轴压成 0，概率质量折半，**正半轴密度翻倍**（保持积分等于 1）：

$$p(z) = \frac{2}{\sigma\sqrt{2\pi}} e^{-z^2/2\sigma^2}, \quad z \ge 0$$

于是：

$$E[z^2] = \int_0^{\infty} z^2 \cdot \frac{2}{\sigma\sqrt{2\pi}} e^{-z^2/2\sigma^2} dz = \frac{1}{2} \int_{-\infty}^{\infty} x^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-x^2/2\sigma^2} dx = \frac{\sigma^2}{2}$$

（正负半轴被 ReLU 对称性地去掉了一半，另一半保持不变，故 $E[z^2] = Var(x)/2$。）

$$Var(\text{ReLU}(x)) = E[\text{ReLU}(x)^2] = \frac{Var(x)}{2}$$

**这就是"方差减半"的严格来源**。

### 2.2 前向方差守恒推导

第 $l$ 层 $y = Wz$ 的输出方差（沿用 $w$、$z$ 独立、零均值假设）：

$$Var(y) = n_{in} \cdot Var(w) \cdot E[z^2] = n_{in} \cdot Var(w) \cdot \frac{\sigma^2}{2}$$

令 $Var(y) = \sigma^2$（方差守恒），解得：

$$Var(w) = \frac{2}{n_{in}}$$

**关键点**：与 Xavier 的 $1/n_{in}$ 相比多出的因子 **2**，正是对 ReLU 方差减半的补偿——Xavier 在 ReLU 网络上每层方差 ×1/2，Kaiming 通过权重方差 ×2 恰好抵消。

### 2.3 反向传播对称性

ReLU 的导数在负半轴为 0：$z > 0$ 时 $\frac{dz}{dx} = 1$（概率 1/2），$z \le 0$ 时 $\frac{dz}{dx} = 0$（概率 1/2）。因此梯度反传时**以 1/2 概率被置零**，梯度方差同样减半，反向守恒条件为：

$$Var(w) = \frac{2}{n_{out}}$$

He 论文的结论：**前向与反向取一个即可（两者相近），论文建议保证前向**——因为权重是共享的，一个方向的守恒会自动带动另一个方向。PyTorch `nn.Linear` 默认 `mode='fan_in'`，`nn.Conv2d` 也默认 fan_in。

### 2.4 a 参数：LeakyReLU 的推广

LeakyReLU 为 $f(x) = \max(x, ax)$（$a$ 为小正数，如 0.01），负半轴不再完全置零。推导（PReLU 论文附录）：

$$Var(w) = \frac{2}{(1 + a^2) n_{in}}$$

- $a = 0$：退化为标准 ReLU，$\frac{2}{n}$；
- $a = 1$：退化为线性恒等，$\frac{2}{2n} = \frac{1}{n}$，恰好是 Xavier 的 $1/n$——**当激活趋向线性时 Kaiming 平滑过渡到 Xavier**，两种理论自洽；
- PyTorch `nn.init.kaiming_uniform_(w, a=math.sqrt(5))` 的 `a=√5` 是为了让**均匀分布的方差与负半轴斜率匹配**的数学约定（使得 $E[w] \ne 0$ 下的方差公式成立），不是 LeakyReLU 的真实斜率——注意区分。

### 2.5 守恒的到底是谁：二阶矩，不是方差

ReLU 输出**均值不为零**：$z = \text{ReLU}(x)$，$x \sim \mathcal{N}(0, \sigma^2)$ 时 $E[z] = \sigma/\sqrt{2\pi} \approx 0.4\sigma$。因此 Kaiming 推导中"守恒"的其实是**二阶矩** $E[z^2]$（推导中通篇用的是 $E[z^2]$ 而非 $Var(z)$）：

$$E[z^2] \equiv 1 \quad \text{（守恒量）}, \qquad Var(z) = E[z^2] - E[z]^2 = 1 - \frac{2}{\pi} \approx 0.36$$

即激活 std 稳定在 $\sqrt{1 - 2/\pi} \approx 0.83$，而不是 1。这是 Kaiming 与 Xavier（方差 $\equiv 1$）的微妙区别，面试时区分"二阶矩守恒"与"方差守恒"能加分。

## 三、源码实现

### 3.1 手写两种分布（对照公式逐行实现）

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)                          # 固定种子，保证注释里的输出可复现

def kaiming_normal_manual(w, a=0.0):
    """高斯版：N(0, sqrt(2/((1+a^2)·fan_in)))，a 为 LeakyReLU 斜率"""
    fan_in = w.size(1)                        # 只考虑 fan_in（He 建议优先保证前向）
    std = math.sqrt(2.0 / ((1 + a * a) * fan_in))
    with torch.no_grad():
        w.normal_(0.0, std)
    return w

def kaiming_uniform_manual(w, a=0.0):
    """均匀版：U(-b, b)，b = sqrt(6/((1+a^2)·fan_in))"""
    fan_in = w.size(1)
    b = math.sqrt(6.0 / ((1 + a * a) * fan_in))
    with torch.no_grad():
        w.uniform_(-b, b)
    return w

w = torch.empty(64, 128)
kaiming_normal_manual(w)
print(f"手写高斯版: std={w.std().item():.4f}")
# 手写高斯版: std=0.1240  (期望 sqrt(2/128)=0.1250 ✓)

w2 = torch.empty(64, 128)
kaiming_uniform_manual(w2)
print(f"手写均匀版: std={w2.std().item():.4f}")
# 手写均匀版: std=0.1241  (期望 b/sqrt(3)=sqrt(6/128)/sqrt(3)=0.1250 ✓)

w3 = torch.empty(64, 128)
kaiming_uniform_manual(w3, a=1.0)
print(f"a=1 时: std={w3.std().item():.4f}（退化为 1/sqrt(128)=0.0884，接近 Xavier）")
# a=1 时: std=0.0888
```

### 3.2 nn.init 官方接口

```python
# PyTorch 官方接口：fan_in/fan_out 与维度自动探测（Linear/Conv 都能用）
w4 = torch.empty(64, 128)
nn.init.kaiming_normal_(w4, a=0.0, mode='fan_in')
print(f"官方高斯版 fan_in: std={w4.std().item():.4f}")    # 官方高斯版 fan_in: std=0.1244

# 裸调用默认 a=0：与手写版 a=0 一致
w5 = torch.empty(64, 128)
nn.init.kaiming_uniform_(w5)
print(f"裸调用(默认a=0): std={w5.std().item():.4f}")      # 裸调用(默认a=0): std=0.1253

# 注意：nn.Linear 内部默认显式传 a=math.sqrt(5)，与裸调用不同！
w6 = torch.empty(64, 128)
nn.init.kaiming_uniform_(w6, a=math.sqrt(5))
print(f"nn.Linear 风格(a=sqrt(5)): std={w6.std().item():.4f}")  # nn.Linear 风格(a=sqrt(5)): std=0.0510
# 验证: bound=sqrt(3)·gain/sqrt(fan_in), gain=sqrt(2/6)=0.577 → bound=0.0884, U(-b,b) 的 std=b/sqrt(3)=0.0510 ✓
```

### 3.3 用随机网络验证激活方差传播（可运行）

```python
torch.manual_seed(0)

def build_mlp(init_fn, depth=8, width=128, act=torch.relu):
    layers = []
    for i in range(depth):
        lin = nn.Linear(width, width)
        init_fn(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    net = nn.Sequential(*layers)
    x = torch.randn(256, width)               # 输入方差 = 1
    stats = []
    with torch.no_grad():
        for i, layer in enumerate(net):
            x = act(layer(x))
            stats.append(x.std().item())
    return stats

print("Kaiming + ReLU 各层激活 std（期望稳定在 ≈0.83）:")
for layer, std in enumerate(build_mlp(nn.init.kaiming_normal_)):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=0.818   <- 线性层后 E[y²]=2，ReLU 后 E[z²]=1 守恒
# layer 3: std=0.758   <- std = sqrt(E[z²] - E[z]²) = sqrt(1-2/π) ≈ 0.83
# layer 7: std=0.804   <- 8 层深度稳定在 0.75~0.83：二阶矩守恒 ✓

print("对照组: Xavier + ReLU（每层 ×0.7 持续衰减）:")
for layer, std in enumerate(build_mlp(nn.init.xavier_normal_, act=torch.relu)):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=0.579
# layer 3: std=0.208
# layer 7: std=0.044   <- 8 层只剩 4%：信号消失（Xavier 在 ReLU 网络失效的铁证）
```

**梯度方差验证（反向对称性，注意 net 里必须包含 ReLU）：**

```python
def grad_std_scan(init_fn, depth=8, width=128):
    net = nn.Sequential()
    for i in range(depth):
        lin = nn.Linear(width, width)
        init_fn(lin.weight)
        nn.init.zeros_(lin.bias)
        net.add_module(f"lin{i}", lin)
        net.add_module(f"relu{i}", nn.ReLU())     # ReLU 必须在网络里，否则测的不是 ReLU 网络
    x = torch.randn(8, width)
    y = net(x).pow(2).mean()
    y.backward()
    return [m.weight.grad.std().item() for m in net if isinstance(m, nn.Linear)]

print("Kaiming+ReLU 各层权重梯度 std（期望同量级）:")
for layer, g in enumerate(grad_std_scan(nn.init.kaiming_normal_)):
    print(f"  layer {layer}: grad_std={g:.2e}")
# layer 0: grad_std=1.51e-02   <- 首层
# layer 3: grad_std=2.19e-02
# layer 7: grad_std=1.75e-02   <- 末层，与首层同量级（最值比仅约 1.5 倍）

print("对照组 Xavier+ReLU 梯度（信号已坍缩，整体小两个量级）:")
for layer, g in enumerate(grad_std_scan(nn.init.xavier_normal_)):
    print(f"  layer {layer}: grad_std={g:.2e}")
# layer 0: grad_std=1.24e-04   <- 比 Kaiming 小约 120 倍：
# layer 7: grad_std=1.26e-04   <-   前向信号逐层衰减 → loss 与梯度整体坍缩，训练等于在做"零梯度更新"
```

> **结论**：Kaiming 严格守恒的是**前向二阶矩** $E[z^2]=1$（激活 std ≈ 0.83）；反向以每层 ×0.5（方差）温和衰减（He 论文证明：前向守恒成立时反向也被保证有界）。对照组 Xavier 则是前向 ×0.667/层衰减 → 8 层后信号只剩 4%，梯度整体坍缩两个量级。

## 四、深入分析

### 4.1 为什么 He 论文建议 fan_in 优先

- 前向和反向分别要求 $2/n_{in}$ 与 $2/n_{out}$，不相等时只能保一个；
- 论文实验表明：**前向守恒更重要**——信号必须先正常流到输出层，loss 才存在；反向梯度是"从 loss 出发往回传播"，首层偏小尚可被后续更新补偿；
- PyTorch 默认 `mode='fan_in'`（Linear/Conv 均如此），与论文一致；
- 如果层结构很不均衡（如 $n_{out} \gg n_{in}$），可显式传 `mode='fan_out'` 以保反向。

### 4.2 为什么 nn.Linear 默认 a=√5 而不是 0（重要坑）

PyTorch 的 `nn.Linear.reset_parameters` 默认 `kaiming_uniform_(w, a=math.sqrt(5))`。代入边界公式可发现：

$$b = \sqrt{\frac{6}{(1+5) \cdot n}} = \sqrt{\frac{1}{n}}$$

即 a=√5 恰好让均匀分布变成经典的 **$U[-1/\sqrt{n}, 1/\sqrt{n}]$**（Caffe 风格），其方差 $= \frac{1}{3n}$，只有标准 Kaiming（$\frac{2}{n}$）的 **1/6**。实测对深 ReLU 网络的影响：

```python
# nn.Linear 默认初始化（a=√5）下，8 层 ReLU MLP 的激活 std
torch.manual_seed(0)
net = nn.Sequential(*[nn.Linear(128, 128) for _ in range(8)])
x = torch.randn(256, 128)
with torch.no_grad():
    for i, l in enumerate(net):
        x = torch.relu(l(x))
        print(f"Linear默认 layer {i}: std={x.std().item():.3f}")
# layer 0: std=0.339
# layer 1: std=0.143   <- 每层约 ×0.4，前几层快速衰减
# layer 7: std=0.031   <- 稳定在 3% 量级（比标准 Kaiming 的 0.83 小 27 倍）
```

> **结论**：PyTorch 的 `nn.Linear` 默认初始化是**保守的 Caffe 风格**，浅层网络或带 BN/LN 的网络无碍；但**深 ReLU 网络务必显式换回 a=0（或 `kaiming_normal_`）**，否则信号衰减严重。这也是"框架默认 ≠ 论文最优"的典型例子。实测：a=0 时 std=0.1250、a=√5 时 std=0.0510，两者相差 2.5 倍。

### 4.3 Kaiming 在 Transformer / 多模态中的位置

- **CNN + ReLU**：ResNet 系、ViT 的卷积 stem、CLIP 的视觉塔（ViT 用 `kaiming_uniform_`/`trunc_normal_`，视框架而定）；
- **MLP projector**：LLaVA 等模型的视觉投影层常用 `kaiming_uniform_`；
- **Transformer 主体**：不用 Kaiming（尺度偏大），用 N(0, 0.02)（见 [Transformer初始化](Transformer初始化.md)）——**因为 Transformer 有 LayerNorm 兜底 + 残差结构，保守小尺度更稳**；
- **LeakyReLU/PReLU 网络**：必须显式传 a（否则默认 a=0 会算错尺度）。

### 4.4 理论边界

1. **假设输入零均值**：ReLU 输出均值 > 0，Kaiming 推导同样依赖 $E[x]=0$（对"方差部分"成立，均值漂移被忽略）；
2. **只针对单层矩**：二阶矩守恒 ≠ 分布完全高斯，深层尾部偏差仍可能累积；
3. **无残差结构假设**：带残差的网络（ResNet/Transformer）方差随深度线性累积，需要残差缩放/零初始化（见 [Transformer初始化](Transformer初始化.md) 与 [初始化总结](初始化总结.md)）。

## 五、优缺点与适用

| 优点 | 缺点 |
|------|------|
| ReLU 网络方差严格守恒（×2 补偿） | 只对 ReLU 族有效，换 GELU/线性需重调 |
| 有完整的积分推导，理论自洽 | 默认 a=√5 的语义易误解 |
| PyTorch 原生默认支持（nn.Linear/nn.Conv2d） | 对带 LN 的 Transformer 尺度仍偏大 |
| fan_in/fan_out 可切换，适配前后向 | 不处理残差结构的多层累积 |

**适用**：CNN+ReLU（ResNet/VGG）、MLP+ReLU、LeakyReLU/PReLU 网络、LLaVA 类视觉投影层、ViT 的 patch embedding（部分框架）。

**不适用**：Sigmoid/Tanh 网络（用 Xavier）、Transformer 全量权重（用 N(0,0.02)）、残差分支末端（需零初始化或 1/√N 缩放）。

## 六、与同类对比

| 维度 | Xavier | Kaiming | N(0, 0.02) |
|------|--------|---------|-----------|
| 目标方差 | $\frac{2}{n_{in}+n_{out}}$ | $\frac{2}{(1+a^2)n_{in}}$ | $0.02^2$（与维度无关） |
| 核心补偿 | 前后向双守恒 | ReLU 方差减半 ×2 | 残差恒等起点 + 保守尺度 |
| 激活假设 | 线性区（Sigmoid/Tanh） | 半线性（ReLU 族） | 有 LN 兜底，任意激活 |
| 维度依赖 | $O(1/n)$ | $O(1/n)$ | $O(1)$（常数） |
| 典型场景 | 浅 MLP、Tanh 门控 | CNN、MLP projector | Transformer 全家 |

> 本质区别一句话：Xavier 假设激活"在线性区"（方差无损），Kaiming 假设激活"减半损失方差"（×2 补偿），N(0,0.02) 则放弃方差守恒假设、靠归一化兜底。

## 七、高频面试问答

**Q1：Kaiming 初始化的公式怎么推导？**
ReLU 把零均值高斯输入负半轴置零，输出方差减半（$E[\text{ReLU}(x)^2] = Var(x)/2$）。套用 $Var(y) = n_{in}Var(w)E[z^2]$，令 $Var(y) = Var(x)$ 得 $Var(w) = 2/n_{in}$。相比 Xavier 的 $1/n_{in}$ 多出因子 2，就是补偿 ReLU 减半。

**Q2：为什么 ReLU 方差减半？**
ReLU 输出 $z = \max(0,x)$，负半轴概率质量被压到 0，正半轴密度翻倍保持积分 1。$E[z^2]$ 恰为 $E[x^2]/2 = Var(x)/2$（对称分布下正负半轴各贡献一半，ReLU 保留正半轴全部）。

**Q3：fan_in 和 fan_out 的区别？Kaiming 为什么默认 fan_in？**
fan_in 是神经元接收的输入数（前向方差的项数），fan_out 是它影响的输出数（反向梯度方差的项数）。He 论文推导前向 $2/n_{in}$、反向 $2/n_{out}$，建议优先保证前向——信号必须先正常到达输出层，且权重共享下前向守恒会自动带动反向稳定。

**Q4：Kaiming 的 a 参数是什么？a=0 和 a=√5 的区别？**
a 是 LeakyReLU 的负半轴斜率：$a=0$ 是标准 ReLU（$Var(w)=2/n$），$a>0$ 时目标方差为 $\frac{2}{(1+a^2)n}$，$a=1$ 时退化为 Xavier 的 $1/n$。PyTorch `nn.Linear` 默认 a=√5 让均匀边界恰好等于 $1/\sqrt{n}$（Caffe 风格，方差只有标准 Kaiming 的 1/6）——深 ReLU 网络建议显式传 a=0。

**Q5：Kaiming 和 Xavier 的本质区别？**
两者都是"令前向/反向方差守恒解出 $Var(w)$"，区别在激活假设：Xavier 假设激活在线性区（方差无损），Kaiming 假设 ReLU 减半（二阶矩 ×1/2）。因此 Kaiming 目标方差是 Xavier 的约 2 倍（$2/n$ vs $2/(n_{in}+n_{out})$）。

**Q6：Kaiming 能用于 Transformer 吗？**
一般不直接用于主体：Transformer 有 LayerNorm + 残差，尺度由 LN 兜底，业界默认 N(0,0.02) 的保守小尺度更稳。Kaiming 适用于 CNN+ReLU 与 MLP projector（如 LLaVA）。

**Q7：PReLU 网络怎么初始化？**
按 $Var(w) = \frac{2}{(1+a^2)n}$ 传对应 a 的 Kaiming；若 a 可学习，通常取初始值 0.25 并配 Kaiming。标准做法：`kaiming_uniform_(w, a=0.25)`。

**Q8：如何快速验证 Kaiming 初始化正确？**
前向一次打印各层激活 std：ReLU 网络的二阶矩 $E[z^2]$ 应守恒（实测 std 稳定 ≈0.83 = $\sqrt{1-2/\pi}$）；backward 后各层梯度 std 同量级。若逐层 ×0.4 持续衰减，说明用错了尺度（如 nn.Linear 默认的 a=√5 或 Xavier）。

## 八、自我检验

- [ ] 能积分推导 $E[\text{ReLU}(x)^2] = Var(x)/2$（含"正半轴密度翻倍"的关键一步）
- [ ] 能从 $Var(y) = n \cdot Var(w) \cdot E[z^2]$ 解出 $Var(w) = 2/n$，并说出 ×2 的补偿含义
- [ ] 能解释反向梯度以 1/2 概率置零 → 反向守恒 $2/n_{out}$
- [ ] 能写出 fan_in/fan_out 的定义、He 建议 fan_in 优先的原因
- [ ] 能解释 a 参数：$a=0$、$a=1$（退化 Xavier）、PyTorch `nn.Linear` 默认 a=√5 的边界 $1/\sqrt{n}$ 与保守性
- [ ] 能写出 PyTorch 手写版与官方 `kaiming_uniform_`/`kaiming_normal_`（含 a、mode）
- [ ] 能跑通 3.3 实验：Kaiming+ReLU 稳定 ≈0.83（二阶矩守恒），Xavier+ReLU 衰减到 0.044，梯度对照差两个量级
- [ ] 能说出"守恒的是二阶矩而非方差"（$Var(z) = 1 - 2/\pi \approx 0.36$）
- [ ] 能说出 Kaiming 与 Xavier、N(0,0.02) 的本质区别与适用场景
- [ ] 能回答 8 个面试追问
