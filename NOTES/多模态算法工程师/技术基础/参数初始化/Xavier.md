# Xavier 初始化（Glorot 初始化）

> 本模块索引见 [参数初始化详解](参数初始化详解.md)

## 一、定义与公式

Xavier 初始化（Glorot & Bengio, 2010，论文 *Understanding the difficulty of training deep feedforward neural networks*）是第一个从**信号传播视角**系统推导的初始化方法。核心思想：好的初始化应该让信号在前向传播和反向传播中**方差都守恒**。

### 1.1 核心公式

$$Var(w) = \frac{2}{n_{in} + n_{out}}$$

**均匀分布版本**（最常用，PyTorch `xavier_uniform_`）：

$$w \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \; \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**高斯分布版本**（PyTorch `xavier_normal_`）：

$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

其中 $n_{in}$（fan_in）为输入维度，$n_{out}$（fan_out）为输出维度。

### 1.2 一个直觉例子

以 $n_{in} = n_{out} = 100$ 的线性层为例：$Var(w) = 2/200 = 0.01$，即 $std \approx 0.1$，比"简单用 std=1"小一个量级。**直觉**：权重数量越多，每个权重要越小，这样 $n$ 个独立贡献叠加后总方差才不至于爆炸。

## 二、数学原理

### 2.1 单层方差传播：Var(y) = n · Var(w) · Var(x)

考虑一个神经元 $y = \sum_{i=1}^{n} w_i x_i$（$n$ 为 fan_in）。假设：

1. $w_i$ 与 $x_i$ 相互独立；
2. $E[w_i] = 0$ 且 $E[x_i] = 0$（均值归零）；
3. 各 $(w_i, x_i)$ 对之间独立同分布。

则：

$$Var(y) = Var\left(\sum_{i=1}^{n} w_i x_i\right) = \sum_{i=1}^{n} Var(w_i x_i) = n \cdot Var(w) \cdot Var(x)$$

其中乘积方差的化简依赖均值为零：

$$Var(wx) = E[(wx)^2] - (E[wx])^2 = E[w^2]E[x^2] - (E[w]E[x])^2 = Var(w) \cdot Var(x)$$

> **如果 $E[w] \ne 0$**：$Var(wx) = Var(w)Var(x) + Var(w)E[x]^2 + Var(x)E[w]^2$，结果更大、推导变复杂。**零均值假设是方差传播分析的地基**——这也是为什么要配零中心的激活（Tanh）或对输入做归一化。

### 2.2 均匀分布 [-a, a] 的方差推导

设 $w \sim U[-a, a]$，密度函数 $f(w) = \frac{1}{2a}$，均值为 0，则：

$$Var(w) = E[w^2] = \int_{-a}^{a} w^2 \cdot \frac{1}{2a} \, dw = \frac{1}{2a} \cdot \frac{w^3}{3}\Big|_{-a}^{a} = \frac{1}{2a} \cdot \frac{2a^3}{3} = \frac{a^2}{3}$$

**通用记忆**：均匀分布 $U[b_1, b_2]$ 的方差为 $\frac{(b_2-b_1)^2}{12}$，代入 $[-a, a]$ 得 $\frac{(2a)^2}{12} = \frac{a^2}{3}$。

**反向换算（钥匙公式）**：给定目标方差 $\sigma^2$，均匀分布边界为：

$$a = \sqrt{3\sigma^2}$$

由 Xavier 目标方差 $\sigma^2 = \frac{2}{n_{in}+n_{out}}$ 代入：

$$a = \sqrt{3 \cdot \frac{2}{n_{in}+n_{out}}} = \sqrt{\frac{6}{n_{in}+n_{out}}}$$

这就是均匀版边界 $\sqrt{6/(n_{in}+n_{out})}$ 的完整来源。

### 2.3 前向与反向双守恒

| 方向 | 方差守恒条件 | 解得 |
|------|------------|------|
| 前向（$y = Wx$，每行 n_in 个元素） | $n_{in} \cdot Var(w) = 1$ | $Var(w) = 1/n_{in}$ |
| 反向（$\partial L/\partial x = W^T \partial L/\partial y$，每行 n_out 个元素） | $n_{out} \cdot Var(w) = 1$ | $Var(w) = 1/n_{out}$ |

同时满足两者不可能（除非 $n_{in} = n_{out}$），Xavier 取**调和折中**：

$$Var(w) = \frac{2}{n_{in}+n_{out}}$$

> 注意这是**均值**式的折中，不是调和平均 $\frac{2n_{in}n_{out}}{n_{in}+n_{out}}$——Xavier 论文对 n_in≠n_out 的情形按上式处理即可，两个方向的方差偏离都控制在可接受范围。

### 2.4 前提假设（为什么会失效的伏笔）

Xavier 推导依赖三条假设，后续方法正是逐条打破它们：

1. **激活函数在线性区**：$y = Wx$ 未计非线性。Sigmoid/Tanh 在 0 附近近似线性，成立；
2. **输入均值归零**：$E[x] = 0$。Tanh 输出零中心，成立；
3. **激活不损失方差**：饱和区会截断方差，但初始尺度小、远离饱和区，近似成立。

## 三、源码实现

### 3.1 手写两种分布（对照公式逐行实现）

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)                          # 固定种子，保证注释里的输出可复现

def xavier_uniform_manual(w):
    """均匀版：U(-a, a)，a = sqrt(6 / (fan_in + fan_out))"""
    fan_in, fan_out = w.size(1), w.size(0)
    a = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        w.uniform_(-a, a)   # inplace 均匀采样
    return w

def xavier_normal_manual(w):
    """高斯版：N(0, 2/(fan_in+fan_out))"""
    fan_in, fan_out = w.size(1), w.size(0)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        w.normal_(0.0, std)
    return w

w = torch.empty(64, 128)                      # [out=64, in=128]，fan_in=128, fan_out=64
xavier_uniform_manual(w)
print(f"手动均匀版: mean={w.mean().item():.2e}, std={w.std().item():.4f}")
# 手动均匀版: mean=-3.52e-04, std=0.1017
# 验证: a=sqrt(6/192)=0.1768, U(-a,a) 的 std = a/sqrt(3) = 0.1021 ✓

w2 = torch.empty(64, 128)
xavier_normal_manual(w2)
print(f"手动高斯版: mean={w2.mean().item():.2e}, std={w2.std().item():.4f}")
# 手动高斯版: mean=-7.07e-04, std=0.1027
# 验证: sqrt(2/192) = 0.1021 ✓（两版本目标方差相同，效果等价）
```

### 3.2 nn.init 官方接口

```python
# 官方接口：与手写版完全一致，且自动处理 fan_in/fan_out 的维度探测
w3 = torch.empty(64, 128)
nn.init.xavier_uniform_(w3)                  # U(-a, a)
print(f"官方均匀版: std={w3.std().item():.4f}")   # 官方均匀版: std=0.1021

w4 = torch.empty(64, 128)
nn.init.xavier_normal_(w4)                   # N(0, 2/(fan_in+fan_out))
print(f"官方高斯版: std={w4.std().item():.4f}")   # 官方高斯版: std=0.1021
```

> **gain 参数**：`nn.init.xavier_normal_(w, gain=1.0)` 默认 gain=1。如果前向代码显式乘了 0.5（如 $\tanh(0.5x)$），应传 gain=2 来匹配——公式变成 $Var(w) = \frac{gain^2 \cdot 2}{n_{in}+n_{out}}$。

### 3.3 用随机网络验证激活方差传播（可运行）

```python
torch.manual_seed(0)

def build_mlp(init_fn, depth=8, width=128, act=torch.tanh):
    layers = []
    for i in range(depth):
        lin = nn.Linear(width, width)
        init_fn(lin.weight)                   # 用指定初始化
        nn.init.zeros_(lin.bias)
        layers.append(lin)
    net = nn.Sequential(*layers)
    x = torch.randn(256, width)               # 输入方差 = 1
    stats = []
    with torch.no_grad():
        for i, layer in enumerate(net):
            x = act(layer(x))
            stats.append(x.std().item())      # 逐层记录输出 std
    return stats

print("A. Xavier(gain=1) + tanh（注意 tanh 的压缩效应）:")
for layer, std in enumerate(build_mlp(nn.init.xavier_normal_)):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=0.625   <- 线性层后 std=1，tanh 压缩到 0.625（tanh(±1)≈0.76）
# layer 1: std=0.485   <- 每层约 ×0.75 轻微衰减（tanh 是收缩映射）

print("B. Xavier(gain=5/3) + tanh（gain 补偿后方差守恒）:")
for layer, std in enumerate(build_mlp(lambda w: nn.init.xavier_normal_(w, gain=5.0/3.0))):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=0.757
# layer 7: std=0.643   <- 逐层稳定不再衰减：gain 补偿了 tanh 的压缩
```

**对照组 1——ReLU 网络用 Xavier（每层方差减半 → 指数消失）：**

```python
print("C. Xavier + ReLU（每层约 ×0.7，深层信号消失）:")
for layer, std in enumerate(build_mlp(nn.init.xavier_normal_, act=torch.relu)):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=0.591   <- ReLU 把方差砍一半，std 直接掉到 0.59
# layer 3: std=0.265
# layer 7: std=0.057   <- 8 层后只剩 6%，深层梯度必然消失
# 这就是 Kaiming 初始化要解决的动机
```

**对照组 2——过大初始化（N(0,1) + ReLU，指数爆炸）：**

```python
print("D. N(0,1) + ReLU（指数爆炸）:")
def big_init(w):
    nn.init.normal_(w, std=1.0)
for layer, std in enumerate(build_mlp(big_init, act=torch.relu)):
    print(f"  layer {layer}: std={std:.3f}")
# layer 0: std=6.642    <- 每层放大 sqrt(128)≈11.3 倍（n·Var(w)=128）
# layer 3: std=3041.932
# layer 7: std=15150731.000  <- 再深两层直接 inf
```

**三个实验的结论**：Xavier 的目标是让 $n \cdot Var(w) = 1$；A/B 验证"方差守恒 + 激活补偿"的正确姿势，C 展示激活不匹配（ReLU）导致消失，D 展示尺度失配（n·Var(w)=128）导致爆炸。

## 四、深入分析

### 4.1 为什么配合 Sigmoid / Tanh

- Sigmoid/Tanh 在 0 附近近似线性：$\sigma(x) \approx 0.5 + 0.25x$，$\tanh(x) \approx x$；
- Xavier 把激活前信号的方差控制在 1 附近，恰好落在线性区，激活函数不扭曲信号；
- Tanh 零中心，满足 $E[x]=0$ 假设；
- 若信号进入饱和区（|x| 大），导数趋 0 → 梯度消失，Xavier 的尺度设计恰好避免这一点。

### 4.2 为什么 ReLU 网络不能直接用

1. **方差减半**：ReLU 负半轴输出恒 0，$E[\text{ReLU}(x)^2] = Var(x)/2$，每层方差减半，深层按 $2^{-L}$ 指数衰减（见 3.3 实验）；
2. **均值漂移**：ReLU 输出均值 > 0，破坏 $E[x] = 0$ 假设，且逐层累积正均值；
3. **反向同样减半**：梯度过 ReLU 以 1/2 概率被置零，反向方差也减半。

### 4.3 实践中仍在使用的场景

1. **Tanh 网络**（如 LSTM 的 gate 前后层、某些 RNN）；
2. **Transformer 线性投影**：DeepNorm 用 gain=β 的 `xavier_normal_` 初始化 FFN/v_proj/out_proj；
3. **归一化层配套的场景**：有 LN 兜底时，用 Xavier 的合理尺度即可，不必苛求 Kaiming。

### 4.4 与"零均值假设"的互相印证

`xavier_uniform_`/`xavier_normal_` 都是零均值分布，因此**任何把均值不为零的初始化的做法**（如直接用 `torch.rand` 的 U(0,1)）都必然产生偏差——不仅破坏方差守恒，还会产生固定偏置叠加。初始化分布的均值必须为 0。

## 五、优缺点与适用

| 优点 | 缺点 |
|------|------|
| 有严格数学推导，前后向方差双守恒 | 假设激活线性、输入零均值，ReLU 网络失效 |
| 均匀/高斯双版本，覆盖全场景 | 尺度对 d_model 大的 Transformer 仍偏大（N(0,0.02) 更小） |
| PyTorch 原生支持，接口简单 | 不处理饱和激活的方差截断 |
| 天然打破对称性（随机采样） | 需要 fan_in/fan_out，无法用于标量/不规则形状 |

**适用场景**：Tanh/Sigmoid 网络、MLP、LSTM 类门控、Transformer 的 DeepNorm 基线、按 gain 缩放的自定义变体。

**不适用**：ReLU 系 CNN/MLP（用 Kaiming）、大 d_model 的 Transformer 全量权重（用 N(0,0.02)）、残差分支末端（需额外缩小或零初始化）。

## 六、与同类对比

| 维度 | Xavier | Kaiming | N(0, 0.02) | 零初始化 |
|------|--------|---------|-----------|---------|
| 论文 | Glorot 2010 | He 2015 | BERT/GPT 实践 | Fixup/ReZero/LoRA |
| 目标方差 | $\frac{2}{n_{in}+n_{out}}$ | $\frac{2}{n_{in}}$（额外 ×2） | 常数 $0.02^2$ | 0 |
| 适用激活 | Sigmoid/Tanh 线性区 | ReLU/LeakyReLU | Transformer（LN 兜底） | 残差分支/微调新分支 |
| 均匀版边界 | $\sqrt{6/(n_{in}+n_{out})}$ | $\sqrt{6/n_{in}}$ | 无（高斯） | — |
| 核心思想 | 前后向双守恒 | 补偿 ReLU 减半 | 恒等起点 + 保守尺度 | 初始零影响 |

## 七、高频面试问答

**Q1：Xavier 初始化的方差公式怎么来的？**
单层神经元 $y = \sum w_i x_i$，在 $w$、$x$ 独立且零均值时 $Var(y) = n \cdot Var(w) \cdot Var(x)$。令前向守恒 $n_{in}Var(w)=1$、反向守恒 $n_{out}Var(w)=1$，两者折中取 $Var(w) = 2/(n_{in}+n_{out})$。

**Q2：均匀分布版本的边界 $\sqrt{6/(n_{in}+n_{out})}$ 怎么推导？**
$U[-a,a]$ 的方差是 $a^2/3$（由 $\int_{-a}^{a}\frac{w^2}{2a}dw$ 积分，或通式 $(b_2-b_1)^2/12$ 代入）。令 $a^2/3 = 2/(n_{in}+n_{out})$ 解出 $a = \sqrt{6/(n_{in}+n_{out})}$。

**Q3：为什么 Xavier 对 ReLU 网络不合适？**
ReLU 负半轴输出 0，输入方差减半、均值变正，Xavier 的"线性 + 零均值"两条假设都被破坏。ReLU 网络每层方差衰减一半，深层指数消失，需用 Kaiming 的 ×2 补偿。

**Q4：Xavier 为什么同时用 fan_in 和 fan_out？**
前向传播的方差与输入个数（fan_in）相关，反向传播梯度的方差与输出个数（fan_out）相关。只用 fan_in 会导致反向梯度不守恒（深层梯度指数消失），只取其一都只能保证一个方向。

**Q5：xavier_normal_ 的 gain 参数是干什么的？**
gain 是激活函数的尺度补偿：若激活写成了 $\tanh(cx)$，有效线性区斜率变成 $c$ 倍，需要 gain=c 来匹配。公式变为 $Var(w) = \frac{2 \cdot gain^2}{n_{in}+n_{out}}$。

**Q6：均匀版和正态版有什么区别？怎么选？**
数学期望上两者方差相同、效果几乎等价；均匀版有硬边界（不会有离群大权重），正态版无边界但实现简单。工程上"默认均匀版"（PyTorch 惯例），追求论文复现用正态版。

**Q7：手写 Xavier 时为什么注意不到方向（weight.size 顺序）？**
PyTorch Linear 的 weight 形状是 [out, in]，fan_in = weight.size(1)、fan_out = weight.size(0)。写自定义初始化时搞反 fan 会直接导致尺度错误——这是手写初始化最常见的 bug。

**Q8：Xavier 在 Transformer 里还有位置吗？**
有。DeepNorm 对 FFN/v_proj/out_proj 用 gain=β 的 `xavier_normal_`；部分小模型（MLP projector、门控层）也用 Xavier 作为合理尺度。但它不是 Transformer 全量权重的主流选择（主流是 N(0,0.02)）。

## 八、自我检验

- [ ] 能手推 $Var(y) = n \cdot Var(w) \cdot Var(x)$ 及零均值前提的作用
- [ ] 能积分推导 $U[-a,a]$ 的方差 $a^2/3$，并反向解出 $a = \sqrt{3\sigma^2}$
- [ ] 能从目标方差 $\frac{2}{n_{in}+n_{out}}$ 推出均匀版边界 $\sqrt{6/(n_{in}+n_{out})}$
- [ ] 能说清前向守恒用 fan_in、反向守恒用 fan_out 的原因
- [ ] 能写出 PyTorch 手写版与官方 `xavier_uniform_`/`xavier_normal_`（含 gain 语义）
- [ ] 能跑通 3.3 的四个实验，并解释 A/B（tanh 压缩与 gain 补偿）、C（ReLU 衰减）、D（爆炸）现象
- [ ] 能说出 Xavier 的三条假设及其在 ReLU 下的失效方式
- [ ] 能回答 8 个面试追问
