# InfoNCE 对比损失

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 问题设定

双塔/对比学习：一个 batch 里有 $N$ 个（图像, 文本）配对样本，编码器分别得到 $N$ 个图像表征 $u_i$ 和文本表征 $v_i$。相似度矩阵 $S_{ij} = \langle u_i, v_j \rangle$（内积或 cosine）。目标：让"配对的"（$i=j$）相似度远大于"非配对的"。

### 1.2 InfoNCE 公式

对第 $i$ 个样本（以图像→文本方向为例）：

$$\mathcal{L}_i = -\log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ij}/\tau)}$$

其中 $\tau$ 是温度超参。批量损失对所有 $i$ 取平均：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i$$

**一句话**：对相似度矩阵的每一行做带温度的 softmax，取对角线位置的交叉熵——"N 选 1"分类任务。

### 1.3 从互信息下界推导（了解直觉即可）

InfoNCE 的动机是最大化互信息 $I(x; y)$（表征与内容之间）。可证明 InfoNCE 是互信息的估计下界：

$$I(x;y) \ge \log N - \mathcal{L}_{InfoNCE}$$

推导思路（Oord et al., 2018）：采样 $N-1$ 个负样本 $\tilde{y}_1, \dots, \tilde{y}_{N-1}$，用噪声对比估计（NCE）的技巧把"配对检测"写成二分类：

$$P(\text{pair}\ |\ x, \{y_j\}_{j=1}^{N}) = \frac{f(x, y_{pos})}{f(x, y_{pos}) + \sum_{j \ne pos} f(x, y_j)}$$

令 $f(x,y) = \exp(S/\tau)$，对分子分母同除以 $f(x,y_{pos})$，得到的就是每个样本的对数几率，取负对数即式 (1.2)。配对数越多（$N$ 大），下界越紧——这就是"负样本越多越好"的理论依据。

### 1.4 对称版本（CLIP）

图文两个方向各算一次，取平均：

$$\mathcal{L}_{CLIP} = \frac{1}{2}\left( \mathcal{L}_{i2t} + \mathcal{L}_{t2i} \right), \qquad \mathcal{L}_{t2i} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{jj}/\tau)}{\sum_{i=1}^{N} \exp(S_{ij}/\tau)}$$

$\mathcal{L}_{t2i}$ 就是"对相似度矩阵的每一列做 softmax、取对角线"——代码里对 $S^T$ 重复同一段逻辑即可。

## 二、数学性质与直觉

### 2.1 与交叉熵完全等价

$$\mathcal{L}_i = \text{CrossEntropy}(S_i / \tau,\ \text{label}=i)$$

- 相似度矩阵行 = logits，对角线索引 = 标签；
- 因此 InfoNCE 的全部梯度性质继承自 CE：$\partial \mathcal{L}_i / \partial S_{ij} = (p_{ij} - \mathbb{1}[i=j]) / \tau$，其中 $p_{ij}$ 是行 softmax 概率；
- **这是理解 CLIP 训练行为（loss 数值、收敛曲线、梯度分布）的关键**。

### 2.2 温度 τ 的双重作用

- 概率软化程度：$\tau$ 小 → 分布尖 → 更关注最难负样本（与正样本相似度接近的负样本）；
- 梯度幅度：整体乘以 $1/\tau$，$\tau=0.07$ 比 $\tau=1$ 梯度大 ~14 倍，需配合更小的学习率；
- $\tau$ 太大会"摆烂"（所有样本均匀分布，loss 稳定在 $\log N$ 附近无学习信号）；太小会训崩（梯度爆炸）。

### 2.3 负样本来源与 batch 依赖

| 特性 | 说明 |
|------|------|
| 负样本数 | 每个正样本对用 batch 内其余 $N-1$ 个 |
| batch 依赖 | 极强：loss 数值与负样本质量强相关 |
| 损失值范围 | 下限 0（全部配对自己），随机时 ≈ $\log N$ |
| 硬负样本 | 与正样本相似的负样本主导梯度（贡献最大） |

**"负样本是 batch 内其他样本"的隐含假设**：batch 采样要保证配对唯一（一个图像只配一个文本），且不重复采样，否则对角线标签会破坏训练。

### 2.4 对称性带来的"双重优化"

两个方向同时优化让图像与文本表征互相对齐（既学"图找文"又学"文找图"），且 $S$ 的对称性（$S_{ij}$ 与 $S_{ji}$）让两个方向的梯度互为强化——CLIP 训练的核心机制。

### 2.5 提升负样本质量的三种工程手段

1. **增大 batch**：CLIP 用 32k 以上 batch（跨卡拼接，见 3.5），负样本越多下界越紧；
2. **负样本队列（MoCo）**：维护滚动特征队列，负样本量与 batch 解耦，可到数万；
3. **难负样本挖掘**：用文本检索预先召回"最像但不对"的图文对作为难负样本（DeCLIP、LAION 清洗常用），缓解 batch 内全是简单负样本的问题。

这三种手段的本质都是**让"假阴性"更少、让负样本更硬**——InfoNCE 的下界与学习信号强弱都取决于此。

## 三、源码实现（手写版本 + 官方等价接口）

### 3.1 手写版（逐行实现公式）

```python
import torch
import torch.nn.functional as F

def info_nce_manual(sim, tau=1.0):
    """sim: [N, N] 相似度矩阵，对角线为正样本对；返回 i2t 方向均值"""
    sim = sim / tau
    log_probs = sim - torch.logsumexp(sim, dim=-1, keepdim=True)   # 行 log-softmax
    labels = torch.arange(sim.size(0), device=sim.device)
    return -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).mean()

sim = torch.tensor([[0.5, 0.1, -0.2],
                    [0.3, 0.7, -0.1],
                    [-0.3, 0.2, 0.6]])
print(info_nce_manual(sim))        # tensor(0.7518)，τ=1
print(info_nce_manual(sim, 0.5))   # tensor(0.5029)，τ=0.5：对角概率更集中 → loss 更小
```

> **注意**：τ 变小 loss 变小（概率更集中在对角），但梯度幅度 ×1/τ 变大——loss 数值下降不代表训练变简单，看梯度。
```

### 3.2 官方等价接口（CE 一行实现）

```python
def info_nce_via_ce(sim, tau=1.0):
    """与手写版完全等价：温度缩放 + 对角线标签的 CE"""
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim / tau, labels)

# 手写版 vs 官方等价：完全一致
sim = torch.tensor([[0.5, 0.1, -0.2],
                    [0.3, 0.7, -0.1],
                    [-0.3, 0.2, 0.6]])
print(info_nce_manual(sim))        # tensor(0.7518)
print(info_nce_via_ce(sim))        # tensor(0.7518)
print(torch.allclose(info_nce_manual(sim), info_nce_via_ce(sim)))   # True
```

### 3.5 分布式跨卡负样本（CLIP 工程核心）

```python
import torch.distributed as dist

def all_gather_concat(t):
    """把各卡特征拼成 [总N, D]（CLIP 式跨卡负样本的关键）"""
    ts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(ts, t)
    ts[dist.get_rank()] = t              # 本卡用本地张量，保留自动微分
    return torch.cat(ts, dim=0)

# 使用：每卡有本地特征 [N_local, D]，收集后得到 [N_total, D]，
# 相似度矩阵是 [N_total, N_total]；loss 时每卡只取自己 N_local 行的对角线——
# 标签仍为本地索引，负样本却包含全卡（32k 负样本就是这样来的）。
```

> **注意**：跨卡负样本让"标签唯一性"更难保证（别的卡可能采样了重复图文对），大规模训练常用聚类去重或哈希过滤数据管道保证。

### 3.3 CLIP 对称损失完整实现（含梯度回流）

```python
def clip_loss(image_emb, text_emb, tau=0.07):
    """image_emb/text_emb: [N, D]，L2 归一化后的表征"""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = image_emb @ text_emb.t() / tau            # [N, N]
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_i2t = F.cross_entropy(sim, labels)         # 行方向（图找文）
    loss_t2i = F.cross_entropy(sim.t(), labels)     # 列方向（文找图）
    return (loss_i2t + loss_t2i) / 2

torch.manual_seed(0)
img = torch.randn(4, 8)
txt = torch.randn(4, 8)
loss = clip_loss(img, txt)
print(loss.item())
# 输出示例：tensor(5.4) 附近（τ=0.07 把随机相似度放大 ~14 倍，loss 远大于 log(4)≈1.39）
print(loss.item() > torch.log(torch.tensor(4.0)))
# True：随机初始化下 loss 的期望 ≥ log N（Jensen 不等式，τ 越小平移越大）
```

### 3.4 手写 vs 官方输出对比

```python
# 随机相似度矩阵上逐元素验证（值随随机种子变化，但两者恒等）
torch.manual_seed(1)
sim = torch.randn(16, 16) / 2
for tau in [0.07, 0.5, 2.0]:
    a = info_nce_manual(sim, tau)
    b = info_nce_via_ce(sim, tau)
    print(f"tau={tau}: manual={a.item():.6f} ce={b.item():.6f} 一致={torch.allclose(a, b)}")
# 输出示例：tau=0.07: manual=13.163242 ce=13.163242 一致=True（大温度数值接近 log16）
```

## 四、梯度分析

### 4.1 逐元素梯度（同 CE）

$$\frac{\partial \mathcal{L}_i}{\partial S_{ij}} = \frac{1}{\tau}\left(p_{ij} - \mathbb{1}[i=j]\right), \qquad p_{ij} = \frac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}}$$

- 对角线（正样本）：梯度为负，把自身相似度抬高；
- 非对角线（负样本）：梯度为正，且正比于 $p_{ij}$——**与正样本越像的负样本，被推开的力量越大**（自动 hard negative 加权）；
- 全部梯度之和为 0（softmax 归一化），这保持了嵌入空间的稳定。

### 4.2 温度对梯度分布的影响

```python
# 演示：τ 越小，梯度越集中到"最像正样本"的负样本上
sim_row = torch.tensor([0.9, 0.8, 0.1, -0.5, -1.0])   # 0.9 是正样本，0.8 是最难负样本
for tau in [1.0, 0.07]:
    p = F.softmax(sim_row / tau, dim=-1)
    print(f"tau={tau}: 最难负样本梯度占比 {p[1].item():.3f}，其余负样本合计 {1 - p[0].item() - p[1].item():.4f}")
# tau=1.0:   最难负样本 0.329，其余负样本合计 0.307（被平摊）
# tau=0.07:  最难负样本 0.193，其余负样本合计 ~0.000 —— 小温度把所有"火力"集中在最难负样本
```

### 4.3 梯度流动路径

```python
# 完整链式：loss → sim → image_emb / text_emb → 编码器
img.requires_grad_(); txt.requires_grad_()
loss = clip_loss(img, txt)
loss.backward()
print(img.grad.shape, txt.grad.shape)   # torch.Size([4, 8]) torch.Size([4, 8])
```

## 五、数值稳定性

1. **减最大值**：相似度通常无界（内积可到 ±10+），$S_{ij}/\tau$ 在 $\tau=0.07$ 时可到 ±150 → 必须用 logsumexp（自动减 max）而不是手写 exp/sum/log；
2. **绝对不要拆开写** `log(exp(a)/Σexp(b))`：分子可能下溢为 0 → log(0) = -inf；
3. **表征先做 L2 归一化**：cosine 相似度 ∈ [-1, 1]，天然控制数值范围，同时让温度语义更清晰（τ=0.07 对应"角度阈值" ~7°）；
4. **FP16 训练**：相似度矩阵 $S/\tau$ 数值很大，half 精度下 logsumexp 的减 max 步骤必须保留，必要时把 loss 计算放 FP32。

## 六、使用场景（含多模态场景）

| 场景 | 用法 | 说明 |
|------|------|------|
| 图文对齐（CLIP） | 对称 InfoNCE + cosine 相似度 | 双塔预训练标杆 |
| 视频-文本 | 视频帧与句子对齐 | CLIP 系列扩展 |
| 音频-文本 | 声学编码器对齐 | AudioCLIP |
| 图像自监督 | 不同增强视图互为正样本 | SimCLR、MoCo |
| 多模态检索 | 训练后直接取 Top-K 相似度 | Recall@K 评测 |
| 表征学习 | 任意模态对 | 多模态预训练通用框架 |

**多模态中的三个高频位置**：
1. **CLIP / 类 CLIP 预训练**：对称 InfoNCE 是图文对齐的第一选择，温度 $\tau=0.07$ 是经验最佳值；
2. **BLIP 的 ITC（Image-Text Contrastive）**：双塔信息用 InfoNCE 对齐，与单塔 ITM 的 BCE 互补；
3. **Q-Former / 轻量对齐模块**：BLIP-2 用 InfoNCE 把视觉特征对齐到文本空间。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 与 CE 同构，实现简单、梯度干净 | 强依赖 batch 大小（负样本少则退化） |
| 自动 hard negative 加权（概率调制） | 需要精心构造配对（重复样本破坏训练） |
| 对称版本双向对齐，表征互惠 | 温度 τ 敏感，需要调参 |
| 理论上有互信息下界解释 | batch 内负样本质量不可控（全简单样本则学习信号弱） |
| 直接优化检索目标（相似度排序） | 大规模训练需处理分布式跨卡负样本（通信开销） |

## 八、高频面试问答

**Q1：InfoNCE 和交叉熵什么关系？**
完全等价：InfoNCE = 对相似度矩阵行做带温度 softmax、取对角线标签的 CE。代码里通常就是一行 `F.cross_entropy(sim / tau, labels)`。

**Q2：温度 τ 的作用是什么？怎么调？**
控制分布的锐化程度：τ 小 → 更关注难负样本、梯度放大 1/τ；τ 大 → 均匀分布、loss 趋近 log N、无学习信号。CLIP 经验值 0.07，一般 0.05~0.1 有效。

**Q3：为什么负样本越多越好？**
两方面：① 互信息下界 $\log N - \mathcal{L}$ 随 N 增大更紧；② 更多负样本让分类任务更难，迫使表征学到更细的判别结构。实际做法是跨卡拼接负样本（CLIP 的 32k+ 负样本）。

**Q4：为什么 batch 内配对不能重复？**
对角线是唯一正样本假设。若两个样本共享相同正样本对，对角线标签与其它行也包含该正样本，softmax 概率被稀释、梯度方向被破坏。数据加载时必须保证配对唯一性。

**Q5：InfoNCE 与 Triplet Loss 的区别？**
InfoNCE 把"1 个负样本"变成"N-1 个，且按概率加权"；Triplet 只有一个硬负样本、梯度稀疏。InfoNCE 是 Triplet 在"多负样本 + 软加权"上的推广，梯度更稠密、训练更稳。

**Q6：为什么 CLIP 用对称损失而不是单向？**
双向对齐让两个模态的表征互相成为对方的监督信号，训练信号翻倍；且对称性让相似度矩阵更一致，检索在两个方向（图找文/文找图）都可用。

**Q7：相似度为什么先归一化？**
① 数值稳定（cosine ∈ [-1,1]）；② 温度语义清晰（0.07 ≈ 7° 的夹角阈值）；③ 梯度被约束在单位球上，避免表征尺度漂移（尺度漂移是对比学习常见的退化模式）。

## 九、自我检验

- [ ] 能写出 InfoNCE 公式并说明与 CE 的等价性
- [ ] 能解释温度 τ 对概率分布与梯度的双重作用
- [ ] 会手写 info_nce_manual 并用 CE 一行验证等价
- [ ] 能写出 CLIP 对称损失的两行核心代码
- [ ] 能解释梯度 $p_{ij} - 1[i=j]$ 与自动 hard negative 加权
- [ ] 知道 batch 大小、负样本质量对训练的影响
- [ ] 能回答 7 个面试追问
