# GroupNorm（分组归一化）

> 本模块索引见 [归一化技术详解](归一化技术详解.md)

## 一、定义与公式

GroupNorm（GN，分组归一化，Wu & He, 2018）把通道分成 $G$ 组，对**单个样本、每组内通道**的 [H, W] 空间统计。它是 BN 在"小 batch 场景"的替代方案，也是 LN 与 IN 之间的插值。

对四维特征 $x \in \mathbb{R}^{N \times C \times H \times W}$，把 C 个通道分成 G 组（每组 $C/G$ 个通道，要求 $C$ 能被 $G$ 整除）：

$$\mu_{n,g} = \frac{1}{C_g \cdot H \cdot W} \sum_{c \in \text{group}_g} \sum_{h,w} x_{n,c,h,w}, \qquad \sigma_{n,g}^2 = \frac{1}{C_g \cdot H \cdot W} \sum_{c \in \text{group}_g} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,g})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma_{n,g(c)}^2 + \epsilon}}, \qquad y = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

- $g(c) = \lfloor c / (C/G) \rfloor$：通道 $c$ 所属的分组；
- $\gamma_c, \beta_c$：每通道可学习参数；
- 统计量个数：$N \times G$ 个。

| 属性 | 值 |
|------|-----|
| 统计维度 | 单样本、组内通道的 H、W |
| 统计量个数 | N × G 个 |
| 依赖 batch | 无 |
| 训练/推理行为 | 完全一致 |
| 可学习参数 | γ、β 每通道 |
| 典型场景 | 检测/分割/视频（batch 小）、3D 医疗影像 |
| 代表模型 | Mask R-CNN、YOLO 系、SlowFast |

## 二、数学性质

### 2.1 归一化家族中的"插值点"

GN 是 LN 与 IN 的连续统一体：

$$G = 1 \Rightarrow \text{GN} \equiv \text{LN（整个样本全部通道一个组）}$$

$$G = C \Rightarrow \text{GN} \equiv \text{IN（每个通道一组）}$$

$G$ 取中间值时，GN 在"组内共享统计量（降噪声）"与"组间保持独立（保精度）"之间权衡。

### 2.2 为什么 GN 在检测/分割任务制胜（动机）

Mask R-CNN 等检测模型的 batch 通常只有 2~16（输入大图、显存受限），BN 统计量噪声大甚至不可用（batch=2 时方差估计的噪声超过 30%）。GN 与 BN 一样按通道分组统计（保留卷积的通道语义），但不跨样本——**channel 维的统计质量与 batch 大小解耦**。实验表明：小 batch 下 GN 显著优于 BN，大 batch 下接近 BN。

### 2.3 统计量个数与噪声

GN 每个分组内共享一个 μ/σ，分组内元素个数 = $C/G \cdot H \cdot W$。统计量越多（G 越大），每个统计量越"局部"、噪声越大；G 越小则越"全局"、越平滑。原论文实验：G=32 与 G=8 表现接近，G 过小（趋近 LN）或过大（趋近 IN）在小 batch 任务上略降。

### 2.4 梯度（组内三项分解）

反传与 LN/IN 同构，只是统计块变成"组"。设 $g_i$ 为组内第 $i$ 个元素的梯度、组内共 $M = (C/G) \cdot H \cdot W$ 个元素：

$$\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma_g^2 + \epsilon}} \left( g_i - \frac{1}{M}\sum_j g_j - \frac{\hat{x}_i}{M} \sum_j g_j \hat{x}_j \right)$$

性质：梯度在**每个 (n, g) 组内**均值为 0、与组内 $\hat{x}$ 正交；组与组之间梯度解耦，无跨组通信需求（对比 BN 的跨样本梯度耦合）。

### 2.5 小 batch 下 GN vs BN 的噪声对比（量化理解）

BN 每个统计量只覆盖 N×H×W 个元素（batch 小则样本数少）；GN 每个统计量覆盖 C/G×H×W 个元素（与 batch 无关）。batch=4 时 BN 的统计量有效样本只有 4 个，而 GN(32) 的每个统计量覆盖 C/32×H×W 个激活值——这就是小 batch 下 GN 稳、BN 崩的本质。

## 三、源码实现

### 3.1 手写 GN 前向 + 与 nn.GroupNorm 对齐

```python
import torch
import torch.nn as nn

def group_norm_manual(x, gamma, beta, num_groups, eps=1e-5):
    # x: [N, C, H, W]，先 reshape 成 [N, G, C/G·H·W] 再按最后一维统计
    N, C, H, W = x.shape
    x_g = x.view(N, num_groups, C // num_groups * H * W)
    mean = x_g.mean(-1, keepdim=True)
    var = x_g.var(-1, unbiased=False, keepdim=True)
    x_hat = (x_g - mean) / torch.sqrt(var + eps)
    x_hat = x_hat.view(N, C, H, W)
    return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

torch.manual_seed(0)
x = torch.randn(2, 8, 4, 4)
y1 = group_norm_manual(x, torch.ones(8), torch.zeros(8), 4)
y2 = nn.GroupNorm(4, 8)(x)
print((y1 - y2).abs().max().item())     # 2.38e-07：完全对齐
```

### 3.2 两个极端：G=1 ≡ LN，G=C ≡ IN（代码验证）

```python
torch.manual_seed(0)
x = torch.randn(2, 6, 4, 4)

ln_y = nn.LayerNorm([6, 4, 4])(x)       # LN：归一化除 batch 外全部维度
gn1_y = nn.GroupNorm(1, 6)(x)           # GN：全部通道一个组
print((ln_y - gn1_y).abs().max().item())    # 2.38e-07：G=1 时 GN ≡ LN

in_y = nn.InstanceNorm2d(6)(x)          # IN：每通道独立
gnC_y = nn.GroupNorm(6, 6)(x)           # GN：每通道一组
print((in_y - gnC_y).abs().max().item())    # 2.38e-07：G=C 时 GN ≡ IN
```

### 3.3 组数 G 的选择

```python
torch.manual_seed(0)
x5 = torch.randn(4, 128, 8, 8)
for G in [1, 2, 8, 32, 128]:
    gn = nn.GroupNorm(G, 128)
    y = gn(x5)
    print(f"G={G}: 每组 {128 // G} 通道, 统计量 {4 * G} 个, 输出 var={y.var().item():.4f}")
# G=1  : 每组 128 通道, 统计量 4  个, 输出 var=1.0000
# G=2  : 每组 64 通道,  统计量 8  个, 输出 var=1.0000
# G=8  : 每组 16 通道,  统计量 32 个, 输出 var=1.0000
# G=32 : 每组 4 通道,   统计量 128 个, 输出 var=1.0000
# G=128: 每组 1 通道,   统计量 512 个, 输出 var=1.0000
```

所有组数都能归一化到方差 1，区别在于统计量个数（噪声）与组间独立性。实践推荐 G=32（通道数 < 32 时退化为每通道一组，即 IN）或 G=8。

### 3.4 检测/分割头中的典型用法

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNHead(nn.Module):
    """Mask R-CNN 风格 FPN 头：小 batch 训练用 GN(32) 替代 BN"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.gn = nn.GroupNorm(32, in_ch)      # 检测头标配：G=32
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.gn(self.conv1(x))))

torch.manual_seed(0)
head = FPNHead(64, 80)
print(head(torch.randn(2, 64, 16, 16)).shape)   # torch.Size([2, 80, 16, 16])
```

### 3.5 小 batch 稳定性演示：统计量估计噪声的蒙特卡洛对比

归一化输出的宏观 var 都是 1（构造保证），真正区别在于**统计量估计的噪声**：batch 越小、每个统计量覆盖的元素越少，估计的 μ/σ² 偏离真实分布越远。用随机采样重复试验量化：

```python
torch.manual_seed(0)
def estimate_error(n_elements, trials=3000):
    """重复随机采样，测量均值/方差估计的均方根误差（真实分布为 N(0,1)）"""
    mean_err = var_err = 0.0
    for _ in range(trials):
        x = torch.randn(n_elements)
        mean_err += x.mean().item() ** 2
        var_err += (x.var(unbiased=False).item() - 1.0) ** 2
    return (mean_err / trials) ** 0.5, (var_err / trials) ** 0.5

cases = [
    ("BN batch=2（每通道 2×64=128 元素）", 2 * 8 * 8),
    ("GN(8) C=32（每组 4×64=256 元素）",   32 // 8 * 8 * 8),
    ("BN batch=32（每通道 32×64=2048 元素）", 32 * 8 * 8),
]
for name, n in cases:
    me, ve = estimate_error(n)
    print(f"{name}: 均值估计误差≈{me:.4f}, 方差估计误差≈{ve:.4f}")
# BN batch=2（每通道 2×64=128 元素）: 均值估计误差≈0.0864, 方差估计误差≈0.1250
# GN(8) C=32（每组 4×64=256 元素）: 均值估计误差≈0.0623, 方差估计误差≈0.0890
# BN batch=32（每通道 32×64=2048 元素）: 均值估计误差≈0.0219, 方差估计误差≈0.0308
```

> 理论值：均值估计的标准误 = 1/√n，方差估计的波动 ≈ √(2/(n-1))。batch=2 的 BN 方差估计误差高达 12.6%，相当于给激活注入 ±12% 的随机缩放扰动——训练极易不稳；GN 的统计量与 batch 解耦，误差只由组内元素数决定。

## 四、深入分析

### 4.1 梯度与训练行为

- 反传公式与 LN/IN 同构，只是统计块变成"组"：梯度在每个 (n, g) 块内均值为 0、与块内 $\hat{x}$ 正交；
- 无 running stats，train/eval 完全一致，推理无需特殊处理；
- 统计量只依赖单样本 → **batch 大小与 GN 行为无关**，这是它对检测/分割/视频任务的决定性优势。

### 4.2 与 BN 的性能对比规律（论文结论）

| batch size | BN | GN |
|-----------|----|----|
| 大（32+） | 好（统计准 + 隐式正则） | 接近 BN（略差 0.1~0.5%） |
| 中（8~16） | 明显退化 | 稳定 |
| 小（2~4） | 崩（统计噪声大） | 稳定，是唯一可靠选择 |

结论：**GN 是 BN 的"无 batch 依赖版本"**——牺牲大 batch 下的一点点精度，换取任意 batch 下的稳定性。

### 4.3 数值与实现细节

- 要求 $C$ 能被 $G$ 整除；分组按通道顺序连续切分（不是交错）；
- 与 LN 相同，FP16 下建议在 FP32 计算统计量；
- GN 的 kernel 融合（cuDNN/fused GroupNorm）支持好，推理开销低；
- 大 G 时每个组元素少，若特征图也小（如 7×7）则统计噪声变大，G 不宜过大。

### 4.4 变体：Weight Standardization + GN

原论文发现 GN + Weight Standardization（对卷积权重做归一化）在小 batch 下效果进一步提升，被用于训练超深网络——GN 常与参数归一化配合使用。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 与 batch 大小完全无关，小 batch 稳定 | 大 batch 下精度略逊 BN（少隐式正则） |
| train/eval 一致，无 running stats | 分组数 G 是超参，需要调（32/8 经验值） |
| 保留通道语义（组内通道共享统计） | 要求 C 能被 G 整除，通道数小时分组受限 |
| 检测/分割/视频任务标准配置 | 对图像任务为主，Transformer 场景不如 LN 常用 |
| 与 BN 同为通道级归一化，替换成本低 | 无 BN 的正则化效应 |

## 六、与同类归一化对比

| 维度 | GroupNorm | BatchNorm | LayerNorm | InstanceNorm |
|------|-----------|-----------|-----------|--------------|
| 统计维度 | 组内通道 H、W | 跨样本按通道 | 单样本全特征 | 单样本单通道 |
| 统计量个数 | N×G | C | N | N×C |
| 依赖 batch | 无 | 强 | 无 | 无 |
| 训练/推理一致 | 一致 | 不一致 | 一致 | 一致 |
| 大 batch | 接近 BN | 最优 | 可用 | 一般 |
| 小 batch | **最优** | 崩 | 可用（特征语义不分组） | 噪声大（统计量最多） |
| 典型场景 | 检测/分割/视频 | 分类（大 batch） | Transformer | 风格迁移 |
| 代表模型 | Mask R-CNN、YOLO | ResNet | BERT、ViT | AdaIN |

**一句话**：GN = "LN 的分组版、IN 的合并版"，用组内通道共享统计量来换取小 batch 稳定性，是 CNN 上替代 BN 的首选。

## 七、高频面试问答

**Q1：GN 的统计维度？与 BN 什么关系？**
单样本、组内通道的 H、W：统计量 N×G 个。与 BN 同属"通道级"归一化（通道语义保留），但统计不跨样本，因此不依赖 batch——相当于把 BN 的"跨样本通道统计"换成"样本内分组统计"。

**Q2：为什么检测/分割模型用 GN？**
这些任务 batch 极小（2~16，大图占显存），BN 统计量噪声大甚至不可用；GN 统计量与 batch 无关，小 batch 稳定，效果显著优于 BN。

**Q3：G 怎么选？两个极端是什么？**
G=1 ≡ LN、G=C ≡ IN。实践推荐 G=32（通道不足 32 时退化为 IN）或 G=8；过大/过小都略降。

**Q4：GN 能替换 BN 吗？有什么代价？**
能，替换成本低（同样的通道级 γ/β、无 batch 依赖）。代价：大 batch 下精度略逊 BN（约 0.1~0.5%），且没有 BN 的正则化效应。

**Q5：GN 有 running stats 吗？训练/推理需要区分吗？**
没有，也不需要。GN 的统计量每步实时计算且只依赖单样本，train/eval 输出完全一致。

**Q6：GN 和 LN 的本质区别？**
LN 把样本所有通道作为一个组（等价 GN G=1），归一化在"特征/通道级全局"；GN 按通道分组，每组独立归一化。NLP 中特征维天然是一个整体（LN），CNN 中通道分组保留局部通道语义（GN 更优）。

**Q7：GN 的统计量噪声和什么有关？**
组内元素个数 $C/G \cdot H \cdot W$：G 越大每统计量覆盖元素越少、噪声越大；batch 大小与噪声无关。

**Q8：GN 在 Transformer（ViT）里能用吗？**
可以（ViT 早期有 GN 变体实验），但 Transformer 的主流仍是 LN/RMSNorm；GN 的主要战场是 CNN 类的小 batch 任务（检测、分割、视频）。

## 八、自我检验

- [ ] 能写出 GN 公式并指出统计维度（组内通道的 H、W）
- [ ] 能代码验证 G=1 ≡ LN、G=C ≡ IN
- [ ] 能手写 GN 前向并与 nn.GroupNorm 对齐
- [ ] 能解释检测/分割任务为什么用 GN 不用 BN
- [ ] 知道统计量个数 N×G 与噪声的关系
- [ ] 知道 G 的经验取值（32/8）与两个极端
- [ ] 能说出 GN 相对 BN 的代价（大 batch 精度略降、无正则化）
- [ ] 能回答 8 个面试追问
