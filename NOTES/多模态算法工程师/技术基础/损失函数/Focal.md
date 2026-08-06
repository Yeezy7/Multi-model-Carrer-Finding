# Focal Loss（焦点损失）

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 从 BCE 出发：类别不平衡的痛点

二分类中正样本比例极低（如目标检测：一图 99% 的框是背景）时，普通 BCE 的梯度被**海量简单负样本**主导：每个简单样本贡献很小的梯度，但数量巨大，合起来淹没了少数困难样本和正样本的信号。

Focal Loss（Lin et al., 2017，RetinaNet）在 BCE 上乘一个**难易调制因子**：

$$\mathcal{L}_{focal} = -\alpha_t (1 - p_t)^{\gamma} \log p_t$$

其中 $p_t$ 是**该样本的"正确类概率"**：

$$p_t = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}, \qquad \alpha_t = \begin{cases} \alpha & y = 1 \\ 1 - \alpha & y = 0 \end{cases}$$

- $\gamma$（默认 2）：调制指数，控制"简单样本被压多狠"；
- $\alpha$（默认 0.25）：类别平衡权重，控制正负样本整体比例。

### 1.2 推导：从"容易样本的损失占比"出发

BCE 的批量损失可以写成"每个样本的贡献之和"：

$$\mathcal{L}_{BCE} = \sum_i -\log p_{t,i}$$

在极度不平衡时，绝大多数项来自简单负样本（$p_t \to 1$，每项损失 ≈ 0），而它们数量巨大，累计贡献仍然压过少数难样本。Focal 的想法：**按"容易程度"降权**：

$$\mathcal{L}_{focal} = \sum_i -\alpha_{t,i}(1 - p_{t,i})^{\gamma} \log p_{t,i}$$

权重因子 $(1-p_t)^{\gamma}$ 的取值：

| $p_t$ | $(1-p_t)^2$（γ=2） | 含义 |
|-------|-------------------|------|
| 0.9（简单） | 0.01 | 损失降为 1% |
| 0.7 | 0.09 | 损失降为 9% |
| 0.5 | 0.25 | 半保留 |
| 0.3（中等难） | 0.49 | 基本保留 |
| 0.1（很难） | 0.81 | 几乎全额 |

**直觉**：简单样本已经学好了，把它们的梯度压到接近 0，让总梯度预算留给困难样本。$\gamma$ 越大压制越狠（$\gamma=0$ 退化为加权 BCE）。

### 1.3 多分类版本

$$p_t = p_y \quad（\text{softmax 后正确类概率}）, \qquad \mathcal{L}_{focal} = -\alpha (1 - p_y)^{\gamma} \log p_y$$

与多分类 CE 的关系：$-\log p_y$ 就是 CE，前面乘调制因子。实现上不能直接对 `CrossEntropyLoss` 加参数，要自己拼 `log_softmax + 调制`（见 3.4）。

## 二、数学性质与直觉

### 2.1 γ 与 α 的职责分离

- **γ 管"难易"**：压简单样本、保困难样本（难易是动态的，训练后期所有样本都变"容易"，整体损失变小、梯度变细）；
- **α 管"类别比例"**：固定放大/缩小某类，不随训练动态变化（正负比 1:99 时取 $\alpha \approx 0.01 \sim 0.25$ 平衡累计贡献）；
- 两者独立：可以只调 γ 或只调 α，正交控制。

### 2.2 与 (重加权) BCE 的本质区别

| | 加权 BCE（α 重加权） | Focal（γ 调制） |
|---|---|---|
| 权重依据 | 类别（静态） | 难度（动态） |
| 简单负样本 | 仍被 α 权重压住 | 被 $(1-p_t)^\gamma$ 自动压制 |
| 困难正样本 | 权重不变 | 自动获得接近 1 的权重 |
| 超参 | 每类一个权重 | α + γ 两个标量 |

Focal 的巧妙处：**不需要给每个难度层次设计权重**，$(1-p_t)^\gamma$ 自动完成"越容易越不重要"的软加权。

### 2.3 数值区间直觉

- 完全随机预测（$p_t = 0.5$）时 $\gamma=2$ 的权重 = 0.25：早期训练损失约为加权 BCE 的 1/4；
- 收敛后期简单样本权重 → 0，有效训练样本只剩"仍然分错/勉强分对"的样本——**等价于自动 hard mining**。

## 三、源码实现（手写版本 + 官方接口对比）

### 3.1 手写二分类版

```python
import torch
import torch.nn.functional as F

def focal_binary(logits, targets, gamma=2.0, alpha=0.25):
    """二分类 Focal：在 BCEWithLogits 基础上加调制（数值稳定版）"""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)        # 正确类概率
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()

logits = torch.tensor([1.5, -0.5])
targets = torch.tensor([1.0, 0.0])
print(focal_binary(logits, targets))          # tensor(0.0262)
print(focal_binary(logits, targets, gamma=0, alpha=1.0))  # tensor(0.3377) = 纯 BCE
```

### 3.2 简单样本 vs 困难样本损失对比

```python
# 固定 logits 下，γ 如何压简单样本
probs = torch.tensor([0.9, 0.6, 0.3])         # 正确类概率：易 → 难
ce = -torch.log(probs)                        # 对应 BCE/CE 损失
print(ce)                                     # tensor([0.1054, 0.5108, 1.2040])
for gamma in [0, 1, 2]:
    focal = (1 - probs).pow(gamma) * ce
    print(f"gamma={gamma}: {focal.tolist()}")
# gamma=0: [0.1054, 0.5108, 1.2040]  ← 就是 BCE
# gamma=1: [0.0105, 0.2043, 0.8428]
# gamma=2: [0.0011, 0.0817, 0.5899]  ← 简单样本被压到 1%
```

### 3.3 官方接口：无原生 Focal，用 BCE/CE 组合

```python
import torch.nn as nn

# PyTorch 没有 nn.FocalLoss。两种标准拼法：
# ① 二分类：BCEWithLogitsLoss 的 reduction='none' + 调制（同 3.1）；
# ② 多分类：CrossEntropyLoss 的 reduction='none' + 调制：

def focal_multiclass(logits, targets, gamma=2.0, alpha=0.25):
    log_probs = F.log_softmax(logits, dim=-1)
    pt = log_probs.gather(-1, targets.unsqueeze(-1)).exp()      # 正确类概率 p_y
    ce = F.nll_loss(log_probs, targets, reduction='none')       # -log p_y
    return (alpha * (1 - pt).pow(gamma) * ce).mean()

logits_mc = torch.tensor([[1.0, 2.0, 3.0]])
print(focal_multiclass(logits_mc, torch.tensor([2])))   # tensor(0.0457)
# 对照：CE = 0.4076，调制系数 (1-0.6652)² = 0.1121
print(F.cross_entropy(logits_mc, torch.tensor([2])))    # tensor(0.4076)
```

### 3.4 输出对比验证

```python
# 手写 vs 官方组合：随机张量上一致（reduction 统一为 mean）
torch.manual_seed(0)
z = torch.randn(64, 1)
y = torch.randint(0, 2, (64, 1)).float()
print(focal_binary(z, y).item())                          # 输出示例：~0.13
# 与"官方等价拼法"逐元素对比
p = torch.sigmoid(z)
pt = torch.where(y == 1, p, 1 - p)
bce_e = F.binary_cross_entropy_with_logits(z, y, reduction='none')
print((0.25 * (1 - pt).pow(2) * bce_e).mean().item())     # 与上行完全相同
```

### 3.5 目标检测中的典型用法（RetinaNet）

```python
class FocalLoss(nn.Module):
    """分类头输出 logits [B, A, C]，前景/背景二分类焦点损失"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, cls_logits, targets):
        # cls_logits: [B, A, C]，targets: [B, A]（-1 忽略，0 背景，1~C 类别）
        mask = targets != -1
        logits = cls_logits[mask]                       # 有效框
        labels = targets[mask]
        num_pos = (labels > 0).sum().clamp(min=1)
        # 背景 logits 与 前景 logits 分开做二分类（交叉熵等价形式）
        p = torch.sigmoid(logits)
        pt = torch.where(labels > 0, p, 1 - p)
        alpha_t = torch.where(labels > 0, self.alpha, 1 - self.alpha)
        ce = -torch.where(labels > 0, torch.log(p), torch.log(1 - p))
        return (alpha_t * (1 - pt).pow(self.gamma) * ce).sum() / num_pos
```

## 四、梯度分析

### 4.1 对 logit 的梯度（二分类）

令 $p_t = \sigma(y' z)$（$y'=\pm1$），$\mathcal{L} = -\alpha_t (1-p_t)^{\gamma}\log p_t$。利用 $\dfrac{dp_t}{dz} = p_t(1-p_t)y'$：

$$\frac{\partial \mathcal{L}}{\partial z} = \alpha_t (1-p_t)^{\gamma}\Big[\, \underbrace{(p_t - y)}_{\text{BCE 主项}} + \underbrace{\gamma\, y'\, p_t \log p_t}_{\text{调制项导数}} \,\Big]$$

- **主项**与加权 BCE 的梯度 $(p_t - y)$ 同方向，幅度被 $(1-p_t)^{\gamma}$ 缩放——Focal 不改变收敛方向，只改变"谁分到的梯度多"；
- **调制项**来自 $(1-p_t)^\gamma$ 自身的导数，相对主项通常较小（$|p_t \log p_t| < 0.37$），符号与之相同（正样本同负、负样本同正）。

| 样本类型 | BCE 梯度幅度 | Focal(γ=2) 主项幅度 |
|---------|------------|--------------------|
| 简单正样本（p=0.9） | 0.10 | ≈ 0.003 |
| 中等（p=0.6） | 0.40 | ≈ 0.16 |
| 困难（p=0.3） | 0.70 | ≈ 0.70 |
| 极难（p=0.1） | 0.90 | ≈ 1.10（被调制项放大） |

### 4.2 多分类版本同理

$$\frac{\partial \mathcal{L}}{\partial z_k} = \alpha (1-p_y)^\gamma (p_k - \mathbb{1}[k=y]) + \text{调制项}$$

主体与 CE 梯度同构，仅缩放 $p_t$ 依赖的调制系数。

## 五、数值稳定性

1. **与 BCE 同源的问题**：必须先算 logit 再调制，不能先 sigmoid 再 log（饱和/溢出）——手写版用 `binary_cross_entropy_with_logits` 或 `log_softmax` 的 reduction='none' 版本，天然稳定；
2. **$(1-p_t)^\gamma$ 的精度**：$p_t$ 很接近 1 时（简单样本）$(1-p_t)^\gamma$ 下溢到 0 是**期望行为**（权重该为 0），不产生 NaN；
3. **$p_t$ 下溢**：$z$ 极负时 $\sigma(z)=0$ → $\log p_t = -\infty$ → 贡献 NaN。解决：给 logit 加 clamp（如 ±20）或用 logsigmoid 形式重写调制项；
4. FP16 训练时上述饱和更常见，建议把损失计算留在 FP32。

## 六、使用场景（含多模态场景）

| 场景 | 为什么用 Focal | 示例 |
|------|---------------|------|
| 目标检测 | 背景框 99%+，前景极少 | RetinaNet 分类头 |
| 长尾多标签分类 | 少数类被多数类淹没 | 属性预测、标签预测 |
| 医学图像/异常检测 | 阳性样本极稀少 | 病灶检测、缺陷检测 |
| 文本分类长尾 | 高频类主导 | 意图识别 |
| 多模态负样本采样 | 难负样本加权 | 对比学习辅助损失 |

**多模态中的两个高频位置**：
1. **长尾图文检索/生成**：稀有概念（罕见物体、小众风格）在预训练数据中样本极少，用 Focal 把损失预算让给这些困难样本；
2. **检测类多模态任务（视觉 grounding / 指代表达理解）**：分类头仍然极度不平衡，Focal 是标准配置（如接地检测模型的目标分类头）。

> 注意：对比学习（InfoNCE/SigLIP）不需要 Focal——softmax 概率机制已经自动给难负样本加权；Focal 用于"损失 = 每样本独立分类"的场合。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 自动 hard mining，无需采样 | 多一个 γ 超参（需要调） |
| 梯度方向不变，仅缩放幅度 | 简单样本权重为 0 会减少有效样本数（小数据集敏感） |
| α 与 γ 正交控制类别/难度 | 理论解释是启发式的（无最优性保证） |
| 数值实现简单（BCE/CE + 调制） | 噪声标签的简单样本被压制 → 模型更容易记住噪声 |
| 与 BCE/CE 平滑过渡（γ=0 退化） | 训练后期几乎无梯度（所有样本都"容易"） |

## 八、高频面试问答

**Q1：Focal Loss 怎么解决类别不平衡？**
两类机制：① α 静态平衡正负类累计贡献；② $(1-p_t)^\gamma$ 动态压低简单样本、保留困难样本的梯度，让总损失预算从"海量简单负样本"转向"少数难样本"。

**Q2：γ 取 2 的依据？α 为什么常取 0.25？**
γ=2 是论文在 RetinaNet 上的网格搜索结果（对 α 不敏感）；α=0.25 配合 γ=2 是经验最优——正样本只占极小比例，但 γ 已经大幅压制简单负样本，α 不需要完全按 1:99 比例取。

**Q3：Focal 和加权 BCE 的区别？**
加权 BCE 的权重是静态类别权重；Focal 的权重随 $p_t$ 动态变化（难度自适应），简单样本自动降权。Focal = 加权 BCE + 动态难度调制。

**Q4：Focal 的梯度有什么性质？**
与 BCE 同方向、幅度被 $(1-p_t)^\gamma$ 缩放：简单样本梯度≈0，困难样本梯度≈全额。收敛方向不变，只是"谁更重要"变了。

**Q5：为什么对比学习不用 Focal？**
InfoNCE 的 softmax 概率机制已自动按难度加权（难负样本梯度大）；SigLIP 可用 α/β 显式加权。Focal 适合"每样本独立二分类"的极度不平衡场景。

**Q6：γ=0 时 Focal 变成什么？**
$\alpha_t$ 加权的 BCE（二分类）或加权 CE（多分类）——Focal 是 BCE/CE 的推广，调试时可先用 γ=0 验证基线再逐步加大 γ。

**Q7：Focal 有什么坑？**
训练后期所有样本变简单 → 总梯度变小，收敛变慢（可配合后期调低 γ）；噪声标签的简单样本会被"保护"导致记忆噪声；小数据集上梯度稀疏。

## 九、自我检验

- [ ] 能写出 Focal 公式并解释 $p_t$、$\alpha$、$\gamma$ 各自的角色
- [ ] 能手推 $(1-p_t)^\gamma$ 对简单/困难样本权重的取值（γ=2）
- [ ] 会写二分类与多分类两个手写版本
- [ ] 知道 Focal 的梯度与 BCE 同方向、幅度缩放
- [ ] 能说清 Focal vs 加权 BCE vs hard mining 的关系
- [ ] 知道哪些场景该用、哪些场景不该用（对比学习）
- [ ] 能回答 7 个面试追问
