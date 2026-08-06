# Dropout

> 本模块索引见 [正则化与训练技巧详解](正则化与训练技巧详解.md)

## 一、定义与公式

Dropout 由 Srivastava et al. (2014) 提出：**训练时以概率 $p$（保留概率）随机保留神经元，以 $1-p$ 丢弃**，防止神经元之间的"共适应"。

设第 $j$ 个神经元的输出为 $y_j$，采样掩码 $r_j \sim \text{Bernoulli}(p)$：

$$\tilde{y}_j = r_j \cdot y_j, \qquad r_j = \begin{cases} 1, & \text{w.p. } p \\ 0, & \text{w.p. } 1-p \end{cases}$$

### 1.1 期望偏移问题（为什么需要 Inverted Dropout）

直接丢弃时，训练输出的期望为：

$$E[\tilde{y}_j] = p \cdot y_j + (1-p) \cdot 0 = p \cdot y_j \neq y_j$$

而推理时不丢弃、输出 $y_j$。**训练/推理的期望输出相差 $p$ 倍**，网络在训练时看到的是"被缩小"的激活，语义漂移。

### 1.2 Inverted Dropout 推导（必考）

**Inverted Dropout**（PyTorch 的实现方式）在训练时把保留的神经元放大 $1/p$：

$$\tilde{y}_j = \frac{r_j}{p} \cdot y_j$$

此时期望：

$$E\left[\frac{r_j}{p} y_j\right] = \frac{p \cdot y_j}{p} = y_j$$

**结论**：训练时放大 $1/p$ 后，训练期望与推理输出 $y_j$ 完全一致 → **推理时无需任何缩放**，直接原样前向。

> 注意表述习惯：有的教程用"丢弃概率 $p_{\text{drop}}$"，此时缩放系数为 $1/(1-p_{\text{drop}})$。**本质不变：训练除以保留概率，保证期望不变。**

> 换个写法（与原 dropout 论文一致）：原论文是"推理时乘 $p$"（Compensated Dropout），Inverted 把它移到训练侧，让推理更干净、也省一次推理时的缩放运算。

## 二、核心原理

### 2.1 为什么有效（三条主线，面试必须展开）

1. **打破共适应（Co-adaptation）**：神经元无法依赖"某个特定神经元总是存在"，被迫各自学到更独立、更鲁棒的特征。类比：多模态模型中注意力头不能互相"抄答案"；
2. **隐式集成（Ensemble）**：每次 forward 都采样一棵不同的"瘦子网络"，$2^n$ 个子网络共享权重；推理时用全部神经元，等价于对这些子网络做集成（期望意义上）；
3. **数据增强视角**：随机丢弃等价于对隐藏特征空间做随机扰动，让特征对局部信息缺失鲁棒。

### 2.2 训练与推理行为差异（工程铁律）

| 阶段 | 行为 | 原因 |
|------|------|------|
| 训练（train 模式） | 随机丢弃神经元，并放大 $1/p$ | 引入随机性、防共适应、保持期望一致 |
| 推理（eval 模式） | **不丢弃，全部保留，不缩放** | 确定性输出；等价于子网络集成 |

> **工程铁律**：`model.train()` 开 dropout，`model.eval()` 关 dropout。**`no_grad` 不管 dropout**！推理时忘了 `eval()` 会导致结果随机抖动——多模态模型部署/评测的经典事故。

### 2.3 Dropout 概率的选择

| 位置 | 常见值 | 说明 |
|------|--------|------|
| 输入层 / Embedding | 0.1~0.2 | 输入噪声不宜过大，否则丢失信息 |
| 隐藏层（MLP/FFN） | 0.1~0.5 | 越大正则越强，也越阻碍拟合 |
| Attention 矩阵 | 0.0~0.1 | 见 2.4 |
| 输出层前 | 通常不加 | 防止破坏输出校准（calibration） |

### 2.4 Attention Dropout（注意力矩阵上的 Dropout）

对**注意力权重矩阵本身**（softmax 之后、乘 V 之前）做 dropout：

$$\text{Attn}(Q, K, V) = \text{Dropout}\left(\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\right) V$$

作用于 `[batch, num_heads, seq_len, seq_len]` 的概率上，随机屏蔽某些 token 之间的注意力连接。每行的注意力概率和不再为 1（被随机置零并缩放），但**期望上仍为 1**。作用：防止注意力分数过度集中于少数 token（注意力坍缩），鼓励注意力分布多样。

Transformer 中 attention dropout 与 FFN dropout 必须分开设置：

| 类型 | 作用对象 | 常用值 | 过大的后果 |
|------|---------|--------|-----------|
| attention dropout | softmax 后注意力矩阵（信息路由） | 0~0.1 | 路由断裂、注意力分布被破坏 |
| FFN dropout | FFN 隐藏层激活（特征变换） | 0.1~0.3 | 拟合不足 |
| embedding dropout | 输入 embedding | 0.1~0.2 | 输入信息丢失 |

### 2.5 结构化 Dropout：DropBlock（简述）

标准 Dropout 逐元素独立丢弃，对**卷积网络基本无效**：感受野重叠，相邻像素信息可以"补位"，随机丢像素等于没丢。

**DropBlock**（Ghiasi et al., 2018）改为丢弃**连续矩形块**（block size $b \times b$），破坏感受野内的成片连续区域，防止信息绕过。两个超参：块大小、丢弃率 $\gamma$（与 keep prob 换算），keep prob 做线性 warmup。

| 方法 | 丢弃单位 | 适用 |
|------|---------|------|
| 标准 Dropout | 单个激活值 | MLP、Transformer FFN |
| DropBlock | 连续矩形块 | CNN、ViT 特征图 |
| 通道 Dropout | 整个 channel（`[N, C, 1, 1]` 广播） | CNN、ViT（CLIP 视觉塔微调常用） |
| DropPath | 整个子层（残差分支） | ViT、ConvNeXt、Swin（隐式正则） |

## 三、源码实现

### 3.1 纯 PyTorch 手写 Inverted Dropout（含反向）

```python
import torch
import torch.nn as nn

class DropoutFunction(torch.autograd.Function):
    """自定义 Dropout：mask 在前向采样并保存，反向按 mask 传播"""

    @staticmethod
    def forward(ctx, x, p=0.5):
        # p 为保留概率；训练才采样 mask
        mask = (torch.rand_like(x) < p) / p   # 保留的乘 1/p（inverted）
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None       # 反向同样按 mask 走

x = torch.randn(2000, 4, requires_grad=True)
y = DropoutFunction.apply(x, 0.5)
print(y.shape)                                # torch.Size([2000, 4])
print((y != 0).float().mean().item())         # 约 0.5（保留比例 ≈ p）
y.sum().backward()
print(x.grad is not None)                     # True
```

### 3.2 nn.Module 版（支持 train/eval 切换）

```python
import torch
import torch.nn as nn

class InvertedDropout(nn.Module):
    """与 nn.Dropout 等价的手写实现：train 丢弃+缩放，eval 直通"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x                          # 推理：全保留、不缩放
        mask = (torch.rand_like(x) < self.p) / self.p
        return x * mask

m = InvertedDropout(0.5)
x = torch.randn(10000)
y_train = m.train()(x)                        # 训练：部分为 0
y_eval = m.eval()(x)                          # 推理：原样输出
print(y_eval.equal(x))                        # True
print(y_train.mean().item() - x.mean().item())  # 期望一致，差值约 0
```

### 3.3 对比官方接口 nn.Dropout 与 F.dropout

```python
import torch
import torch.nn as nn

m1 = nn.Dropout(p=0.5)                        # 官方：p 是丢弃概率！
m2 = InvertedDropout(p=0.5)                   # 我们：p 是保留概率
x = torch.randn(20000)

o1 = m1.train()(x)
o2 = m2.train()(x)
# 官方丢弃 p=0.5 等价于我们保留 p=0.5
print((o1 != 0).float().mean().item())        # ~0.5
print((o2 != 0).float().mean().item())        # ~0.5
print(o1.mean().item(), o2.mean().item(), x.mean().item())  # 三者期望一致
```

> **参数方向坑（必记）**：`nn.Dropout(p)` 的 $p$ 是**丢弃概率**（PyTorch 约定），手写时容易混；上面手写版用"保留概率"只是教学习惯，商用代码直接 `nn.Dropout(p=0.1)`。

### 3.4 在 Transformer 中的完整用法（attention + FFN 双 dropout）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerBlock(nn.Module):
    """示意：attention dropout 与 FFN dropout 分别设置"""

    def __init__(self, d_model=64, n_heads=4,
                 attn_dropout=0.1, ffn_dropout=0.1, hidden_ratio=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)   # 作用在注意力矩阵上
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_ratio * d_model),
            nn.GELU(),
            nn.Dropout(ffn_dropout),                 # FFN 隐藏层 dropout
            nn.Linear(hidden_ratio * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力 + attention dropout
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)                  # ★ 对概率矩阵 dropout
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.ln1(x + self.wo(out))
        return self.ln2(x + self.ffn(x))

model = SimpleTransformerBlock()
x = torch.randn(2, 8, 64)
model.train()
y = model(x)                                        # 训练：随机路径
model.eval()
y2 = model(x)                                       # 推理：确定路径
print(y.shape, y2.shape)                            # torch.Size([2, 8, 64]) 两次
print(torch.equal(y, y2))                           # False（训练是随机的）
```

### 3.5 训练循环中的标准用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 3.4 的 SimpleTransformerBlock（若单独运行本段，先执行 3.4 定义它）
model = SimpleTransformerBlock()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

def train_step(x, y):
    model.train()                       # 关键：打开 dropout
    opt.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()

def evaluate(x, y):
    model.eval()                        # 关键：关闭 dropout，否则指标带噪
    with torch.no_grad():
        return F.mse_loss(model(x), y).item()

x = torch.randn(2, 8, 64)
y = torch.randn(2, 8, 64)
print(f"train loss: {train_step(x, y):.4f}")   # 如 0.56
print(f"eval loss:  {evaluate(x, y):.4f}")     # 如 0.62（eval 模式，确定）
```

## 四、深入分析

### 4.1 为什么对 CNN 无效而 DropBlock 有效

普通 dropout 逐元素独立：CNN 特征图中相邻位置感受野高度重叠（如 stride=1 时 3×3 卷积相邻输出共享 6/9 的输入），某像素被丢，相邻像素仍携带同样信息 → **信息冗余让"丢像素"形同虚设**。DropBlock 丢的是成片 block，使被丢区域的信息无法从周围补位，等价于"丢了一张小图"。ViT 虽无感受野重叠，但 patch 级语义冗余类似，DropBlock/通道 dropout 同样适用。

### 4.2 Dropout 与 BN 的冲突（著名争议）

Dropout 改变单样本激活的方差，BN 又按 batch 统计归一化，两者叠加时正则强度不稳定（Dropout + BN 在 CNN 上常出现"训练慢、测试掉点"）。实践中 CNN 用 BN 通常**弃用 dropout**；Transformer 全用 LN（LayerNorm，按特征维度归一化、不跨样本统计），与 dropout 无冲突，两者共存无碍。

### 4.3 Dropout 与 LoRA 微调的组合经验

- LoRA 只更新低秩增量 $\Delta W$，主干的 dropout 仍生效（作用于激活而非权重）；
- 微调大模型时 dropout 通常**调小**（0.0~0.1）：预训练权重已高度泛化，过强 dropout 反而阻碍拟合新任务；
- 数据量极小时可把 FFN dropout 提到 0.2~0.3 配合冻结视觉塔，是 VLM 微调常见防过拟合组合。

### 4.4 数值视角：缩放为什么必须与丢弃同步

Inverted dropout 中 mask 的期望是 $E[r_j/p] = 1$，所以 `x * mask` 的期望恒等于 `x`——这保证：① 训练/推理期望一致；② **LayerNorm 之后加 dropout 不会系统性改变归一化统计**。如果只丢弃不缩放（non-inverted），LN 后的分布均值偏移 $p$ 倍，深层误差逐层累积。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 极简：一行 `nn.Dropout`，零推理开销 | 对 CNN 基本无效（需 DropBlock 等结构化变体） |
| 防共适应、隐式集成，泛化收益显著 | 训练变慢（需更多 epoch 拟合） |
| 与 LN 系架构（Transformer）天然兼容 | 超参 $p$ 敏感：过大欠拟合、过小无效 |
| 可作用在激活、注意力矩阵、embedding 多处 | 与 BN 叠加效果差 |
| 结构简单，极易与 LoRA 等组合 | 推理忘关 `eval()` 会造成随机抖动事故 |

## 六、与同类对比

| 方法 | 丢弃单位 | 随机性来源 | 适用场景 | 与 Dropout 关系 |
|------|---------|-----------|---------|-----------------|
| Dropout | 单个激活 | Bernoulli | MLP、Transformer FFN | 基准方法 |
| DropBlock | 连续 block | 伯努利 + 块采样 | CNN、ViT 特征图 | Dropout 的结构化扩展 |
| 通道 Dropout | 整个 channel | 伯努利 | CNN/ViT（CLIP 微调） | 维度级扩展 |
| DropPath / Stochastic Depth | 整个残差分支 | 伯努利 | ViT/Swin 深层 | 层级扩展（隐式集成思想同源） |
| L2 正则 | 权重（确定性） | 无 | 通用 | 与 Dropout 互补：一个动参数、一个动激活 |

**与数据增强的关系**：Dropout 可看作"隐藏特征空间的增强"（对特征做遮挡扰动），与输入空间增强（裁剪/翻转/MixUp）互补；多模态训练中二者同时使用，各管一个层次。

## 七、高频面试问答

**Q1：Dropout 为什么有效？**
三点：① 随机丢弃打破神经元共适应，迫使特征独立冗余；② 每个 batch 训练不同子网络，推理全量等价于子网络集成；③ 对隐藏特征加噪，类数据增强。

**Q2：inverted dropout 为什么缩放 1/(1-p)（或 1/p）？**
直接丢弃时训练期望是 $p \cdot y$、推理是 $y$，两阶段期望不一致。训练时放大 $1/p$ 后期望恢复为 $y$，推理无需任何补偿。缩放必须与丢弃同步，保证激活分布不被系统性偏移。

**Q3：为什么训练和推理时 Dropout 行为不同？**
训练需要随机性引入正则；推理要求确定性输出且想用全部特征（等价于子网络集成）。PyTorch 靠 `model.train()/model.eval()` 切换，`no_grad` 不管 dropout。

**Q4：attention dropout 和 FFN dropout 为什么要分开设置？**
作用对象不同（注意力矩阵是信息路由、FFN 隐藏层是特征变换）；敏感度不同——注意力矩阵是概率分布，dropout 过大会直接破坏路由语义，必须小（0~0.1），FFN 高维表达冗余空间大（0~0.5）。

**Q5：为什么 Dropout 对 CNN 无效？DropBlock 怎么解决？**
CNN 感受野重叠，相邻像素可补位，逐元素丢弃无效。DropBlock 丢弃连续矩形块，破坏成片区域的信息冗余，让被丢区域无法从周围补位。

**Q6：Dropout 和 BN 能一起用吗？**
能但不推荐：BN 按 batch 归一化、Dropout 改变激活方差，两者统计假设冲突，常导致训练慢、测试掉点。Transformer 用 LN 无此问题。

**Q7：微调大模型时 dropout 怎么设？**
调小（0~0.1）：预训练权重已泛化，过大 dropout 阻碍拟合；attention dropout 设 0，FFN dropout 0.1 是 LLaMA 系常见配置；数据极小时可配合冻结塔适度加大。

## 八、自我检验

- [ ] 能写出直接丢弃的期望公式并推出 inverted dropout 的 $1/p$ 缩放
- [ ] 能解释"训练期望 = 推理输出"为什么成立
- [ ] 能说清 `nn.Dropout(p)` 的 p 是丢弃概率、手写版要用保留概率的坑
- [ ] 能写出手写 inverted dropout 的 autograd.Function 版本
- [ ] 能说出 train/eval 下 dropout 行为差异及 `no_grad` 不管 dropout
- [ ] 能说出 attention dropout 与 FFN dropout 的分开设置原因与常用值
- [ ] 能简述 DropBlock 的原理与两个超参
- [ ] 能回答 7 个面试追问
