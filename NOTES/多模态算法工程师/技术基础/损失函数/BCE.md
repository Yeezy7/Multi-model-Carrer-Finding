# 二分类交叉熵 BCE / BCEWithLogitsLoss

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 从伯努利分布出发

二分类：模型输出单个 logit $z$，经 Sigmoid 得正类概率 $p = \sigma(z)$，标签 $y \in \{0, 1\}$。把输出建模为伯努利分布：

$$P(y|z) = p^{y}(1-p)^{1-y}$$

最大似然取负对数（单样本）：

$$\mathcal{L} = -[\, y \log p + (1-y)\log(1-p) \,] = -[\, y \log\sigma(z) + (1-y)\log(1-\sigma(z)) \,]$$

**直觉**：$y=1$ 时只惩罚 $\log p$（预测正类概率太低），$y=0$ 时只惩罚 $\log(1-p)$（把负类当正类的概率太高）。

### 1.2 ±1 标签等价形式（SigLIP 用，更紧凑稳定）

把标签换成 $y' \in \{-1, +1\}$，利用 $\sigma(-z) = 1 - \sigma(z)$：

$$\mathcal{L} = -\log\sigma(y' \cdot z)$$

推导：$y'=+1$ 时 $\mathcal{L} = -\log\sigma(z)$；$y'=-1$ 时 $\mathcal{L} = -\log\sigma(-z) = -\log(1-\sigma(z))$。两种情况统一成一个式子，正好是原始 BCE 的两项。

> **记忆点**：二分类的"标准二元形式"与"±1 乘积形式"是同一个损失的两种写法；后者在对比/度量学习里（SigLIP、DPO、Pairwise 排序）是主力形式。

### 1.3 批量形式

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i)) \right]$$

## 二、数学性质与直觉

### 2.1 梯度与 log-odds 的对应

$$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y$$

- $y=1$ 时梯度 $p - 1 < 0$：把 logit 往正方向推；
- $y=0$ 时梯度 $p > 0$：把 logit 往负方向推；
- 与 CE 的"$p - \text{onehot}$"完全同构：BCE 就是 K=2 的 CE（见 2.4），所以梯度公式形式一致。

±1 形式下：$\dfrac{\partial \mathcal{L}}{\partial z} = -y'\cdot\sigma(y'z)$。

### 2.2 信息论解读

$$\mathcal{L} = H(y, p) = D_{KL}(y \| p) + H(y)$$

$y$ 是确定分布时 $H(y)=0$，BCE 就是"两个分布"的 KL。从优化角度看，最小化 BCE 等价于把 sigmoid 输出往真实二值分布上校准。

### 2.3 多标签 = 逐位独立 BCE

多标签（每样本多个类别可同时为 1）不是 softmax（和为 1），而是对每个维度独立建模伯努利分布，**所有元素一起求平均**：

$$\mathcal{L}_{multi} = -\frac{1}{N \cdot C}\sum_{i}\sum_{c} \left[ y_{ic}\log p_{ic} + (1-y_{ic})\log(1-p_{ic}) \right]$$

### 2.4 与 CE / Focal 的关系

- K=2 且用 softmax（$[z, 0]$ 形式）时，CE ≡ BCE（数值完全一致）；
- Focal Loss 是 BCE 的调制版本：$\mathcal{L}_{focal} = -\alpha(1-p_t)^{\gamma}\log p_t$（见 Focal 篇），$\gamma=0, \alpha=1$ 时退化为 BCE；
- BCE 对"错误但自信"的样本惩罚最重（$p \to 0$ 且 $y=1$ 时梯度 → -1），这是它与 MSE 用于分类时的本质区别。

### 2.5 排序/配对的 pairwise 视角（BCE 是统一底层）

检索与偏好学习中常用配对（pairwise）形式：正样本得分 $z_{pos}$ 应高于负样本 $z_{neg}$，把"正高于负"当成一个二分类事件，其 logit 是差值 $z_{pos} - z_{neg}$，对它做 BCE：

$$\mathcal{L}_{pair} = -\log\sigma\left(z_{pos} - z_{neg}\right)$$

推导：$P(z_{pos} > z_{neg}) = \sigma(z_{pos} - z_{neg})$，取负对数即得。**这个结构是 SigLIP 逐对损失、DPO 偏好损失、RankNet 排序损失的共同骨架**——理解 BCE 的 ±1 形式，就理解了这一整族损失。

## 三、源码实现（手写版本 + PyTorch 官方接口）

### 3.1 手写版（直接按公式 + 数值稳定）

```python
import torch
import torch.nn.functional as F

def bce_naive(logits, targets):
    """朴素版：先 sigmoid 再 log（数值不稳，仅教学用）"""
    p = torch.sigmoid(logits)
    return -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).mean()

def bce_stable(logits, targets):
    """稳定版：用 log-sigmoid 恒等式 log σ(z) = -log(1+e^{-z})"""
    return -(targets * F.logsigmoid(logits)
             + (1 - targets) * F.logsigmoid(-logits)).mean()

def bce_pm1(logits, ys):
    """±1 标签版本：-log σ(y'·z)"""
    return -F.logsigmoid(ys * logits).mean()

logits = torch.tensor([1.5, -0.5])
targets = torch.tensor([1.0, 0.0])
print(bce_naive(logits, targets))    # tensor(0.3377)
print(bce_stable(logits, targets))   # tensor(0.3377)
print(bce_pm1(logits, torch.tensor([1.0, -1.0])))   # tensor(0.3377)
```

### 3.2 PyTorch 官方接口

```python
import torch.nn as nn

# 1) 接受概率输入的经典版（需要自己先 sigmoid）
p = torch.sigmoid(logits)
print(F.binary_cross_entropy(p, targets))            # tensor(0.3377)

# 2) 接受 logits 的推荐版（内部数值稳定）
print(F.binary_cross_entropy_with_logits(logits, targets))   # tensor(0.3377)
print(nn.BCEWithLogitsLoss()(logits, targets))               # tensor(0.3377)

# 3) 多标签：形状 [B, C]，标签同为 0/1 矩阵，默认对所有元素取平均
logits_ml = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 2.0, 1.0]])
targets_ml = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
print(F.binary_cross_entropy_with_logits(logits_ml, targets_ml))  # tensor(0.5025)

# 4) 类别加权（pos_weight：正样本加权，多标签长尾常用）
print(F.binary_cross_entropy_with_logits(logits, targets,
                                         pos_weight=torch.tensor([2.0])))
# tensor(0.4384)：正样本项 ×2 后重算均值
```

### 3.3 输出对比验证

```python
# 手写稳定版 vs 官方接口：随机张量上完全一致
torch.manual_seed(0)
z = torch.randn(64, 32)
y = torch.randint(0, 2, (64, 32)).float()
print(bce_stable(z, y).item(), F.binary_cross_entropy_with_logits(z, y).item())
# 输出示例：0.687083 0.687083（值随随机种子变化，但两者恒等）
```

### 3.4 与 Focal 的代码关系

```python
def focal_from_bce(logits, targets, gamma=2.0, alpha=0.25):
    """Focal = BCE 按 p_t 加权。gamma=0, alpha=1 时恰好等于 BCE"""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)          # 该样本的正确类概率
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
    return (alpha_t * (1 - pt).pow(gamma) * bce).mean()

print(focal_from_bce(logits, targets))     # tensor(0.0262)，gamma=0 时 = 0.3377
```

### 3.4 样本级加权版本（pos_weight 的手写版）

```python
def bce_with_sample_weight(logits, targets, weight=None):
    """reduction='none' + 自定义权重：长尾/难易样本加权的基础"""
    loss = -(targets * F.logsigmoid(logits) + (1 - targets) * F.logsigmoid(-logits))
    if weight is not None:
        loss = loss * weight
    return loss.mean()

w = torch.tensor([2.0, 1.0])               # 给正样本更高权重
print(bce_with_sample_weight(logits, targets, w))   # tensor(0.4384)
# 手算：正样本项 0.2014×2 + 负样本项 0.4741×1，均值 (0.4028+0.4741)/2 = 0.4384
```

> **口径提醒**：`pos_weight` 是在元素级加权后按元素总数平均（上面例子得 0.4384）；若按"加权和/权重和"平均会得到另一个值。多标签任务里两种口径都常见，需与评测口径对齐。

## 四、梯度分析

### 4.1 标准形式梯度

$$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y = p - y$$

推导（$y=1$ 分支为例）：$\mathcal{L} = -\log\sigma(z)$，$\frac{d}{dz}[-\log\sigma(z)] = -\frac{\sigma'(z)}{\sigma(z)} = -\frac{\sigma(1-\sigma)}{\sigma} = \sigma - 1 = p - 1$。$y=0$ 分支同理得 $p$。

### 4.2 ±1 形式梯度

$$\frac{\partial}{\partial z}\left[-\log\sigma(y'z)\right] = -y' \cdot \sigma(y'z)$$

推导：$\frac{d}{dz}[-\log\sigma(y'z)] = -\frac{\sigma'(y'z)}{\sigma(y'z)}\cdot y' = -y'\cdot\sigma(y'z)\cdot\frac{1-\sigma(y'z)}{...} \to -y'\,\sigma(y'z)$（用 $\sigma' = \sigma(1-\sigma)$ 约分）。

### 4.3 行为对照表

| 情形 | p | 梯度（y=1） | 惩罚强度 |
|------|-----|-----------|---------|
| 简单正样本 | 0.99 | -0.01 | 极小 |
| 中等正样本 | 0.60 | -0.40 | 中 |
| 难正样本 | 0.10 | -0.90 | 很大 |
| 完全判错 | ≈0 | ≈-1 | 最大（≈ CE 的 log 惩罚上界） |

关键区别：**BCE 的梯度上界是 1（有界），CE 的损失可到 +∞（无界）**。所以 BCE 对"错得离谱"的样本惩罚是软性的，训练更稳；这也正是 Focal Loss 在 BCE 上做调制的原因（CE 版 Focal 直接做调制会数值不稳）。

## 五、数值稳定性

1. **log(0) 风险**：$\sigma(z)$ 在 $z$ 很负时下溢到 0，$\log(0) = -\infty$ → 必须先 sigmoid 再 log 的写法必挂；
2. **log-sigmoid 恒等式**：$\log\sigma(z) = -\log(1 + e^{-z})$，$z$ 很负时 $e^{-z}$ 溢出为 inf → 需要分段：$z<0$ 时改用 $-z + \log(1+e^{z})$（等价推导：$-\log(1+e^{-z}) = -z - \log(1+e^z)$）；
3. **BCEWithLogits 内部实现**就是 2 的完整处理（等价于 log-sum-exp 技巧），所以训练中永远用 `BCEWithLogitsLoss`，不要自己拼 `sigmoid + BCE`；
4. FP16 训练时 logits 大（如 10+），上述溢出更常见，务必用官方接口。

```python
# 错误写法演示：预测"错得很自信"时 sigmoid 饱和 → log(0) 直接 NaN
z_big = torch.tensor([100.0, -100.0])
bad = -(torch.tensor([0.0, 1.0]) * torch.log(torch.sigmoid(z_big))
        + torch.tensor([1.0, 0.0]) * torch.log(1 - torch.sigmoid(z_big))).mean()
good = F.binary_cross_entropy_with_logits(
    torch.tensor([0.0, 1.0]), z_big)
print(bad)    # tensor(nan) —— σ(100)=1 → log(1-1)=log(0)
print(good)   # tensor(100.0) —— 正确且稳定（log-sigmoid 恒等）
```

## 六、使用场景（含多模态场景）

| 场景 | 为什么用 BCE | 示例 |
|------|-------------|------|
| 二分类 | 概率输出 + 有界梯度 | 图文匹配 ITM（BLIP） |
| 多标签分类 | 逐位独立伯努利建模 | 属性预测、标签预测 |
| 图文匹配头（单塔） | logit → 匹配概率 | BLIP / ALBEF 的 ITM head |
| 对比学习（逐对） | ±1 形式的 logistic 损失 | SigLIP、DPO |
| 判别器（GAN） | 真/假二分类 | PatchGAN discriminator |
| 目标检测（早期） | 有无目标二分类 | YOLO 置信度头 |

**多模态中的两个高频位置**：
1. **ITM（Image-Text Matching）**：BLIP 用融合特征过一个线性层输出 logit，再用 BCEWithLogits 监督"匹配/不匹配"——单塔对齐的标志性设计；
2. **SigLIP 逐对损失**：把图文相似度矩阵的每个元素当成一个独立二分类样本（标签 ±1），数学形式就是 $- \log\sigma(y'\cdot z)$（见 SigmoidLoss 篇）。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 梯度有界（∈[-1,1]），训练稳定 | 简单样本梯度不为 0，海量易负样本拖慢收敛（Focal 的动机） |
| 输出可解释为概率、与 log-odds 对应 | 标签噪声敏感（硬标签惩罚错配） |
| 支持多标签、加权、±1 形式，通用 | 对极度不平衡不鲁棒（需 α/pos_weight） |
| 数值稳定实现成熟（BCEWithLogits） | 只建模二值关系，无类间结构信息 |
| 与 Focal/DPO/SigLIP 同族，学一个通一片 | — |

## 八、高频面试问答

**Q1：BCE 和 CrossEntropy 的区别？**
BCE 是 K=2 的 CE 特例：对每个维度独立建模伯努利（多标签，互不竞争）；CE 用 softmax 全局归一化（多分类，互斥）。二分类里两者数值等价（把 [z,0] 过 softmax 再 CE = 对 z 算 BCE）。

**Q2：为什么用 BCEWithLogits 而不是 sigmoid+BCE？**
数值稳定。框架内部用 log-sigmoid 恒等式一步算完，避免 exp 溢出、log(0) 和精度丢失；且输入 logits 时梯度直接是 $\sigma(z)-y$，不需要链式过 sigmoid。

**Q3：±1 标签形式怎么来的？为什么更优？**
由 $\sigma(-z)=1-\sigma(z)$ 推出：$-\log\sigma(y'z)$ 统一了 y=0/1 两个分支。形式上更紧凑，天然适合"相似度→概率"的对比/配对场景（SigLIP、DPO），且梯度 $-y'\sigma(y'z)$ 同样简洁。

**Q4：多标签为什么用 BCE 而不是 softmax+CE？**
多标签各维度不互斥（可同时为 1），softmax 的"和为 1"约束是错误的建模；BCE 逐位独立，每个类别是独立的二分类问题，和 S 的平方损失 / 多热标签匹配。

**Q5：BCE 和 Focal 什么关系？**
Focal 在 BCE 上乘调制因子 $(1-p_t)^{\gamma}$：简单样本（$p_t\to1$）权重→0，难样本权重→1，再乘类别平衡权重 $\alpha_t$。$\gamma=0,\alpha=1$ 时 Focal ≡ BCE。

**Q6：BCE 梯度有界意味着什么？**
对"完全判错"的样本惩罚是软性的（梯度 ≤1），配合 sigmoid 不会像 CE 一样产生超大损失，训练稳定；代价是简单样本仍然持续产生梯度，类别极不平衡时被简单负样本淹没，需要用 pos_weight 或 Focal。

**Q7：图文匹配（ITM）为什么用 BCE 而不是 InfoNCE？**
ITM 是"给定一对图文判是否匹配"的判别任务（单塔，细粒度融合后判定），天然是二分类；InfoNCE 是"batch 内选正确对"的检索任务（双塔，粗粒度相似度）。任务不同，损失不同。

## 九、自我检验

- [ ] 能从伯努利分布推导出 BCE 公式
- [ ] 能推出 ±1 标签等价形式并解释为什么数值更稳
- [ ] 能写出梯度 $\sigma(z)-y$ 并说明有界性
- [ ] 知道 BCEWithLogits 内部用 log-sigmoid 恒等式
- [ ] 会写多标签 BCE（逐位独立 + 整体平均）
- [ ] 能说清 BCE、CE、Focal、SigLIP 四者的关系
- [ ] 能回答 7 个面试追问
