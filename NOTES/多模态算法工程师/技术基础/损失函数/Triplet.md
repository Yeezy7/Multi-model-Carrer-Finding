# Triplet Loss（三元组损失）

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 三元组设定

度量学习（metric learning）：学一个嵌入函数 $f(\cdot)$，让**同类样本距离近、异类样本距离远**。Triplet Loss（FaceNet, Schroff et al., 2015）每次取三个样本：

- **Anchor $a$**：锚点；
- **Positive $p$**：与 $a$ 同类的样本（如同一人的另一张脸）；
- **Negative $n$**：与 $a$ 异类的样本。

### 1.2 公式（平方距离版，FaceNet 原始定义）

$$d_{ap} = \|f(a) - f(p)\|_2^2, \qquad d_{an} = \|f(a) - f(n)\|_2^2$$

$$\mathcal{L} = \max\left(0,\ d_{ap} - d_{an} + m\right)$$

目标：$d_{ap} + m \le d_{an}$——正样本距离至少比负样本距离小一个间隔 $m$。违反约束时产生损失（被"推开"的量正好是欠了多少间隔）；满足约束则损失为 0。

### 1.3 欧氏距离版（PyTorch 默认）

$$\mathcal{L} = \max\left(0,\ \|f(a) - f(p)\|_2 - \|f(a) - f(n)\|_2 + m\right)$$

两种版本的差别只在距离是否平方，训练行为略有不同（平方版对"大距离"更敏感）。

### 1.4 批量形式

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \max\left(0,\ d_{ap}^{(i)} - d_{an}^{(i)} + m\right)$$

$B$ 个三元组（batch 内常通过采样构造，见 3.4 hard mining）。

### 1.5 常见变体（面试加分项）

| 变体 | 公式 | 特点 |
|------|------|------|
| Soft-margin Triplet | $\log(1 + e^{d_{ap} - d_{an} + m})$ | 处处有梯度，无硬阈值 |
| Lifted Structured | 聚合所有负样本 | 结构化批量损失 |
| N-pair Loss | 一个 anchor 配多个负样本 | 向 InfoNCE 过渡 |
| Angular Loss | 用角度选负样本 | 对嵌入尺度不敏感 |

这些变体的共同目标：**让"负样本的选择与加权"更聪明**——沿 Triplet → N-pair → InfoNCE 的路线，本质都是把 1 个负样本变成"多个且加权"。

## 二、数学性质与直觉

### 2.1 间隔 m 的含义

- $m$ 是"正负距离之间至少要差多少"的最小间隔：$m$ 越大，嵌入被推得越开，但训练越难（更多三元组违反约束）；
- $m=0$ 只要求 $d_{ap} \le d_{an}$，退化成"弱排序"约束，容易欠拟合；
- 经验值：平方距离版 $m \in [0.1, 0.5]$（L2 归一化后），欧氏距离版 $m \approx 1.0$；
- $m$ 的作用与 SVM 的 margin、对比损失的 interval 同源——都是"最小安全距离"。

### 2.2 稀疏梯度（最重要的性质）

$$\mathcal{L} = \max(0, \cdot) \implies \text{违反约束的三元组才有梯度}$$

- **简单三元组**（$d_{an}$ 远大于 $d_{ap}+m$）：损失 0，无梯度——模型对它们"无感"；
- **半难/难三元组**：产生梯度，且**梯度大小与"违反程度"成正比**（欠得越多推得越狠）；
- 后果：**采样质量决定训练质量**——随机采样的三元组绝大多数是简单样本，模型几乎学不到东西。这是 Triplet 最大的工程难点（见 2.3）。

### 2.3 三元组的三种难度

| 类型 | 定义 | 学习价值 |
|------|------|---------|
| 简单 | $d_{ap} + m < d_{an}$ | 无梯度（浪费） |
| 半难 | $d_{ap} < d_{an} < d_{ap} + m$ | 有梯度，价值中等 |
| 难 | $d_{an} < d_{ap}$ | 梯度最大，但易带来塌缩 |

**困难样本悖论**：全用最难的负样本可能让模型崩溃（嵌入塌缩到一点，因为"最难"不断变难）；工业界常用**半难采样**（semi-hard mining）折中——只在 batch 内选比正样本稍远一点的负样本。

### 2.4 距离度量的选择

- **L2 距离**：FaceNet 原版，适合归一化嵌入；
- **平方 L2**：梯度线性化（$\partial d^2/\partial f = 2(f_a - f_p)$）；
- **cosine 距离**：多模态场景常用（与检索评测口径一致），配合 L2 归一化后等价于"1 - cosine"；
- 距离函数需与评测指标一致——检索用 cosine 排序，训练就用 cosine 距离。

## 三、源码实现（手写版本 + PyTorch 官方接口）

### 3.1 手写版（平方距离）

```python
import torch
import torch.nn.functional as F

def triplet_loss_squared(anchor, positive, negative, margin=1.0):
    """FaceNet 原版：平方 L2 距离"""
    d_ap = (anchor - positive).pow(2).sum(dim=-1)     # [B]
    d_an = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(d_ap - d_an + margin).mean()

a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
p = torch.tensor([[0.9, 0.1], [0.8, 0.9]])
n = torch.tensor([[0.6, 0.5], [0.3, 0.2]])
print(triplet_loss_squared(a, p, n))
# tensor(0.3050)：L = [max(0, 0.02-0.41+1), max(0, 0.05-1.13+1)] 的均值 = [0.61, 0]
```

### 3.2 PyTorch 官方接口

```python
import torch.nn as nn

# 官方默认：欧氏距离（p=2）+ margin=1.0 + mean
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
print(criterion(a, p, n))        # tensor(0.3309)
# L = [max(0, ‖a-p‖₂ - ‖a-n‖₂ + 1)] = [0.5011, 0.1606] → mean = 0.3309

# 距离换成平方版：p 取 2 且用 squared L2？——官方没有"平方"选项，
# 但可以用 p=2 配合 margin 自行实现（见 3.1），或把嵌入先缩放再传
print(nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: (x - y).pow(2).sum(-1),
    margin=1.0)(a, p, n))        # tensor(0.3050)，与手写平方版一致

# 常用配置：L2 归一化 + cosine 距离
a_n, p_n, n_n = F.normalize(a), F.normalize(p), F.normalize(n)
cos_criterion = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1 - (x * y).sum(-1), margin=0.3)
print(cos_criterion(a_n, p_n, n_n))   # tensor(0.1783)，margin 0.3 与归一化距离匹配
```

### 3.3 输出对比验证

```python
# 手写欧氏版 vs 官方 TripletMarginLoss：随机张量上完全一致
torch.manual_seed(0)
a_r = torch.randn(8, 16); p_r = torch.randn(8, 16); n_r = torch.randn(8, 16)

def triplet_euclid(a, p, n, margin=1.0):
    d_ap = (a - p).pow(2).sum(-1).sqrt()
    d_an = (a - n).pow(2).sum(-1).sqrt()
    return F.relu(d_ap - d_an + margin).mean()

print(triplet_euclid(a_r, p_r, n_r).item())     # 输出示例：0.6445
print(nn.TripletMarginLoss(margin=1.0, p=2)(a_r, p_r, n_r).item())  # 相同值
```

### 3.4 Hard Negative Mining（batch 内构造难负样本）

```python
def hard_triplet_batch(embeddings, labels, margin=1.0):
    """batch 内挖难负样本：对每个 anchor，用同类中最近的正样本 + 异类中最近的负样本"""
    emb = F.normalize(embeddings, dim=-1)          # [B, D]
    sim = emb @ emb.t()                            # 相似度矩阵
    d = 1 - sim                                    # 余弦距离 [B, B]
    same = labels[:, None] == labels[None, :]      # 同类掩码
    same.fill_diagonal_(False)                     # 去掉自身
    d_ap = d.masked_fill(~same, float('inf')).min(dim=-1).values   # 最近正样本
    d_an = d.masked_fill(same, float('inf')).min(dim=-1).values    # 最近负样本
    return F.relu(d_ap - d_an + margin).mean()

labels = torch.tensor([0, 0, 1, 1, 2, 2])
print(hard_triplet_batch(a_r[:6], labels).item())  # 输出示例：0.4152（全难样本，loss 通常更大）
```

### 3.5 全局（离线）难负样本挖掘

```python
def offline_mine(embeddings, labels, num_hard=1):
    """全局范围挖难负样本：对每个 anchor 返回最近的 num_hard 个异类索引"""
    emb = F.normalize(embeddings, dim=-1)
    d = 1 - emb @ emb.t()
    same = labels[:, None] == labels[None, :]
    d = d.masked_fill(same, float('inf'))              # 同类（含自身）设为无穷
    return d.topk(num_hard, dim=-1, largest=False).indices   # 最近的 k 个负样本

emb_fix = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
                        [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
labels_fix = torch.tensor([0, 0, 1, 1, 2, 2])
print(offline_mine(emb_fix, labels_fix, num_hard=1))
# tensor([[4], [4], [5], [5], [1], [3]]) —— 每行是该 anchor 最近的非同类样本
# 例：anchor 0（[1,0]）最近异类是 index 4（[0.6,0.4]，余弦距离 0.168）
```

## 四、梯度分析

### 4.1 三种情形

对 anchor 的梯度（平方版，$\partial d^2/\partial a = 2(a-p)$ 等）：

$$\frac{\partial \mathcal{L}}{\partial a} = \begin{cases} 0 & d_{ap} + m \le d_{an} \quad\text{（简单，无梯度）} \\ 2(n - p) & \text{违反约束（推 anchor 离开负、靠近正）} \end{cases}$$

**重要结论**：
- 违反约束时，梯度只与 $(n - p)$ 有关——把 anchor 推向 $p$、拉离 $n$ 的合力；
- 梯度幅度与"违反量"无关（relu 之后是线性！）——这与其他损失不同：$\mathcal{L} = d_{ap} - d_{an} + m$ 对 $d$ 是线性的，**不会越违反推得越狠**；
- 平方版与欧氏版梯度差在 $\partial d/\partial a = (a-p)/\|a-p\|$（单位化），欧氏版梯度有界、对大距离更宽容。

### 4.2 数值验证

```python
# 违反约束的三元组：d_ap=0.41, d_an=0.45, L = max(0, 0.41-0.45+1) = 0.96
a_g = torch.tensor([[1.0, 0.0]], requires_grad=True)
p_g = torch.tensor([[0.6, 0.5]])
n_g = torch.tensor([[0.7, 0.6]])
loss = triplet_loss_squared(a_g, p_g, n_g)
loss.backward()
print(loss.item())        # tensor(0.9600)
print(a_g.grad)           # tensor([[0.2000, 0.2000]]) = 2(n - p)，推 a 远离 n、靠近 p
```

### 4.3 与对比损失梯度的关系

Contrastive Loss（成对）：$\mathcal{L} = y\,d^2 + (1-y)\max(0, m-d)^2$——Triplet 是它的"锚点化"变体（用三元组代替成对），梯度行为相似：只对违反约束的样本对产生梯度。

## 五、数值稳定性

1. **距离下溢/溢出**：大嵌入范数下 $d^2$ 可能很大，但 relu 线性增长无 NaN 风险；FP16 下建议先做 L2 归一化再算距离（范围受控）；
2. **NaN 来源**：`masked_fill(inf)` 后 `.min()` 在整行都是 inf（如 batch 内某类只有一个样本）→ 得到 inf → relu(inf) = inf → NaN。**加 eps 或过滤**（见 3.4 中 batch 内同类至少 2 个的要求）；
3. **margin 与归一化匹配**：L2 归一化后距离 ∈ [0, 2]，margin 不能取 1.0 以上（否则全部违反、退化为纯排序优化）；
4. 嵌入维度高时距离平方的数值范围大，建议始终使用归一化嵌入 + 余弦距离。

## 六、使用场景（含多模态场景）

| 场景 | 用法 | 说明 |
|------|------|------|
| 人脸识别 | FaceNet 标准配置 | 身份嵌入 |
| 图像检索 | 商品/行人/车辆 re-ID | 三元组 + hard mining |
| 图文 embedding 对齐（早期） | 双塔预训练 | 已被 InfoNCE 取代 |
| 视频-文本检索 | 跨模态三元组 | 同类(视频,文本)为正，其余为负 |
| 细粒度相似学习 | 品类嵌入 | 分类标签不足时的替代监督 |
| 多模态融合校验 | 锚点=图像，正=正确描述 | 检索前对齐 |

**多模态中的两个高频位置**：
1. **跨模态对齐早期方案（VSE++ 等）**：图文检索在 CLIP 之前普遍用"文本锚点 + 匹配/不匹配图像"的 Triplet 变体，配合难负样本挖掘；
2. **细粒度/实例级检索**：同一物品不同视角的图像做三元组，或"图-文-图"链式三元组，用于电商检索、视频去重等场景。

> 现状：跨模态**预训练**对齐已被 InfoNCE / SigLIP 取代（负样本更多、梯度更稠密）；Triplet 仍活跃在**标签稀疏、负样本难以从 batch 内获得**的领域（如小批量细粒度检索、re-ID）。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 概念简单、无需 softmax/温度 | 采样困难：随机采样几乎无梯度 |
| 间隔 m 语义清晰（最小安全距离） | 梯度稀疏且幅度恒定（不随违反量加大） |
| 灵活的距离度量（L2/平方/cosine） | 需要 triplet 构造，训练集要求高 |
| batch 内构造硬负样本可行 | 难负样本过头会导致嵌入塌缩 |
| 对标签稀疏场景友好 | 无理论保证（无互信息/概率视角） |

## 八、高频面试问答

**Q1：Triplet Loss 的目标函数是什么？**
$d_{ap} + m \le d_{an}$：正样本距离必须比负样本距离小至少 m。违反则损失 = 欠了多少；满足则 0。

**Q2：为什么说"采样比损失本身更重要"？**
$\max(0,\cdot)$ 使简单三元组无梯度；随机采样的样本 99% 是简单样本 → 模型学不到任何东西。必须做 hard mining（batch 内挖最近负样本）或离线挖掘。

**Q3：难负样本 vs 半难负样本怎么选？**
全难样本梯度大但易让嵌入塌缩（最难的不断变难、不断重复）；半难（$d_{ap} < d_{an} < d_{ap}+m$）折中，训练稳定。FaceNet 用半难，OpenFace 类实现用批量内挖掘。

**Q4：Triplet 和 InfoNCE 的区别？**
Triplet 每样本只看 1 个负样本、梯度稀疏且幅度恒定；InfoNCE 用 N-1 个负样本 + softmax 概率加权，难负样本自动获得更大梯度，梯度稠密。所以预训练对齐普遍用 InfoNCE。

**Q5：间隔 m 怎么选？**
与距离度量匹配：平方 L2 取 0.1~0.5，欧氏 L2 取 ~1.0，归一化 cosine 距离取 0.2~0.4。m 太小欠拟合（只要相对顺序）、太大全部违反（退化为纯排序）。

**Q6：为什么 Triplet 会有嵌入塌缩？怎么防？**
所有样本被推近同一个点（d_ap 与 d_an 同时变小）→ 可分性消失。防法：hard mining 上限、margin 衰减、与其他损失（CE/InfoNCE）联合、约束嵌入范数。

**Q7：多模态检索为什么现在不用 Triplet 做主损失？**
预训练阶段 batch 内天然有海量负样本（配对样本），InfoNCE/SigLIP 直接利用且梯度更稠密、自动难负加权；Triplet 仍用于负样本稀疏、需手动构造的小规模/细粒度场景。

## 九、自我检验

- [ ] 能写出平方版与欧氏版公式并说明 m 的作用
- [ ] 能说清简单/半难/难三类三元组及梯度有无
- [ ] 会写手写版并用 nn.TripletMarginLoss 验证一致
- [ ] 会写 batch 内 hard negative mining（掩码 + min）
- [ ] 能解释"采样决定质量"与嵌入塌缩
- [ ] 能对比 Triplet vs InfoNCE vs Contrastive 三种负样本机制
- [ ] 能回答 7 个面试追问
