# Sigmoid Loss（SigLIP 逐对损失）

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 从"逐对二分类"出发

SigLIP（Zhai et al., 2023）放弃了 InfoNCE 的"全局归一化 + batch 竞争"，把图文对齐建模为**每一个（图像, 文本）对上的独立二分类**：配对的对为正样本，不配对的对为负样本。

设 batch 大小为 $N$，相似度矩阵 $S_{ij} = \langle u_i, v_j \rangle$（cosine 或内积），配对标签：

$$Y_{ij} = \begin{cases} +1 & i = j \\ -1 & i \ne j \end{cases}$$

SigLIP 损失（temperature $t$，可学习 bias $b$）：

$$\mathcal{L} = -\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma\left(Y_{ij} \cdot (t \cdot S_{ij} + b)\right)$$

### 1.2 推导：从 BCE 的 ±1 形式

第 $(i,j)$ 对就是二分类，直接用 BCE 的 ±1 紧凑形式（见 BCE 篇）：

$$\mathcal{L}_{ij} = -\log\sigma\left(Y_{ij} \cdot z_{ij}\right), \qquad z_{ij} = t \cdot S_{ij} + b$$

- $Y_{ij}=+1$：$-\log\sigma(z_{ij})$，要求配对相似度尽量大；
- $Y_{ij}=-1$：$-\log\sigma(-z_{ij}) = -\log(1-\sigma(z_{ij}))$，要求不配对相似度尽量小。

把 $N^2$ 对全部求平均即得 SigLIP。**一个 batch 产生 $N^2$ 个二分类样本**（$N$ 个正对 + $N^2-N$ 个负对），正负比天然是 $1 : (N-1)$。

### 1.3 log-sigmoid 恒等（实现必须用这个形式）

$$\log\sigma(z) = -\log(1 + e^{-z}) = -\text{softplus}(-z)$$

所以损失可写成：

$$\mathcal{L} = \frac{1}{N^2}\sum_{i,j} \text{softplus}\left(-Y_{ij} \cdot (t S_{ij} + b)\right)$$

- $z$ 很大（预测正确）：softplus → 0；
- $z$ 很负（预测错误）：softplus → $-z$（线性，无饱和），梯度恒定，**不会梯度消失**。

## 二、数学性质与直觉

### 2.1 与 InfoNCE 的本质区别：归一化 vs 独立

| | InfoNCE | SigLIP Sigmoid Loss |
|---|---|---|
| 建模 | 行内 softmax 竞争（N 选 1） | 每对独立二分类 |
| 负样本 | batch 内 N-1 个，概率加权 | batch 内 N²-N 个，等权（可加权） |
| 全局归一化 | 需要 | 不需要 |
| batch 依赖 | 强（负样本质量决定 loss） | 弱（每对独立贡献） |
| 分布式通信 | 需跨卡收集相似度矩阵 | 每卡独立算，只需最后同步 loss |

**直觉**：InfoNCE 问"这行里谁是对的"；SigLIP 问"这一对匹配吗"。后者更像判别式分类器，训练信号更稠密（$N^2$ vs $N$ 个样本），但对"全局排序"的直接优化弱于 softmax。

### 2.2 温度 t 与偏置 b 的角色

- $t$：把相似度缩放到 logit 尺度（默认约 2.6，对数空间可学习），$t$ 大 → 二分类更"硬"（σ 更陡）；
- $b$：可学习的偏置，**自动吸收正负样本的先验不平衡**（负对数量是正对的 N-1 倍，$b$ 会学成负值把整体 logit 拉低）；
- 相比 InfoNCE 的温度 τ，$t$ 与 $b$ 是**可学习参数**，随训练自适应，少一个敏感超参。

### 2.3 加权版本（类不平衡处理）

正对只有 $N$ 个、负对有 $N^2-N$ 个，可按 $\alpha / \beta$ 加权：

$$\mathcal{L} = -\frac{1}{N^2}\sum_{i,j}\Big[\alpha \cdot \mathbb{1}[Y_{ij}=+1]\cdot \log\sigma(z_{ij}) + \beta \cdot \mathbb{1}[Y_{ij}=-1]\cdot \log\sigma(-z_{ij})\Big]$$

- 默认 $\alpha = \beta = 1$；
- 负样本权重 $\beta$ 可调（越大越强调"不匹配"惩罚）；
- 与 Focal 的 $\alpha_t$ 思路同源，只是按标签而不是按难度加权。

### 2.4 与 DPO / 排序损失同构（值得记住的结构）

DPO 的偏好损失：

$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

把"log-ratio"看成 logit，它就是对 logit 的 sigmoid 二分类——与 SigLIP 的 $-\log\sigma(Y\cdot z)$ 数学结构完全一致。**"对 logit（或 logit 差值）做 logistic 二分类"是 sigmoid 损失族的统一骨架**：BCE → SigLIP → DPO → RankNet，学一个就通了全部。

## 三、源码实现（手写版本 + 数值稳定版本对比）

### 3.1 手写版（直接按公式）

```python
import torch
import torch.nn.functional as F

def siglip_loss_naive(sim, tau=10.0, bias=0.2):
    """朴素版：sigmoid 后再 log（仅教学，数值不稳）"""
    n = sim.size(0)
    y = 2 * torch.eye(n) - 1                      # 对角线 +1，其余 -1
    z = y * (tau * sim + bias)
    return -torch.log(torch.sigmoid(z)).mean()

def siglip_loss_stable(sim, tau=10.0, bias=0.2):
    """稳定版：log-sigmoid 恒等 + 温度偏置"""
    n = sim.size(0)
    y = 2 * torch.eye(n) - 1
    z = y * (tau * sim + bias)        # z 已含 ±1 符号：正对 -logσ(z)，负对 -logσ(z)=log(1+e^z) 统一由 logsigmoid 处理
    return -F.logsigmoid(z).mean()

sim = torch.tensor([[0.5, 0.2], [0.1, 0.8]])
print(siglip_loss_stable(sim).item())     # tensor(0.9435)
```

### 3.2 简洁版（softplus 一行）

```python
def siglip_loss_softplus(sim, tau=10.0, bias=0.2):
    """最简洁且数值稳定的写法：softplus(-y'z)"""
    n = sim.size(0)
    y = 2 * torch.eye(n) - 1
    return F.softplus(-y * (tau * sim + bias)).mean()

print(siglip_loss_softplus(sim).item())   # tensor(0.9435)，与稳定版一致
```

### 3.3 输出对比验证

```python
# 三个版本（naive/稳定/softplus）在小数值下完全一致；极端 logits 下只有稳定版正确
z_test = torch.tensor([5.2, -2.2, -1.2, 8.2])
print(-torch.log(torch.sigmoid(z_test)))     # 朴素版：tensor([0.0055, 2.3051, 1.4633, 0.0003])
print(-F.logsigmoid(z_test))                 # 稳定版：同上
print(F.softplus(-z_test))                   # 恒等变换：同上
# 极端值演示
z_big = torch.tensor([100.0, -100.0])
print(-torch.log(torch.sigmoid(z_big)))      # tensor([0.0000, 100.0000]) 精度丢失
print(-F.logsigmoid(z_big))                  # tensor([0.0000, 100.0000]) —— 也 OK（内部已稳定）
print(F.softplus(-z_big))                    # tensor([0.0000, 100.0000])
```

> **实现要点**：PyTorch 的 `F.logsigmoid` 与 `F.softplus` 内部都做了分段稳定处理，直接用即可；不要自己拼 `log(1+e^x)`，FP16 下必炸。

### 3.4 加权版本

```python
def siglip_loss_weighted(sim, tau=10.0, bias=0.2, alpha=1.0, beta=1.0):
    """α/β 加权：正对乘 α，负对乘 β（N=2 时负对占 2/4，可调权重）"""
    n = sim.size(0)
    y = 2 * torch.eye(n) - 1
    z = y * (tau * sim + bias)
    w = torch.where(y == 1, alpha, beta)
    return -(w * F.logsigmoid(z)).mean()

print(siglip_loss_weighted(sim).item())            # tensor(0.9435)（α=β=1 退化为原版）
print(siglip_loss_weighted(sim, beta=3.0).item())  # tensor(2.8277)：负对权重 ×3 后均值
```

### 3.5 大 batch 分块计算（避免 N×N 显存爆炸）

```python
def siglip_loss_chunked(img_emb, txt_emb, chunk=64, tau=10.0, bias=0.0):
    """分块计算：相似度矩阵按行切块，显存从 N×N 降到 chunk×N"""
    n = img_emb.size(0)
    total = 0.0
    for i in range(0, n, chunk):
        s = img_emb[i:i + chunk] @ txt_emb.t()            # [chunk, N]
        row_idx = torch.arange(i, i + s.size(0), device=s.device)
        col_idx = torch.arange(n, device=s.device)
        y = torch.where(row_idx[:, None] == col_idx[None, :], 1.0, -1.0)
        total += -F.logsigmoid(y * (tau * s + bias)).sum()
    return total / n ** 2

img_e = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
txt_e = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
print(siglip_loss_chunked(img_e, txt_e, chunk=1, tau=10.0, bias=0.2).item())
# tensor(0.3991) —— 与不切块版本逐位一致（sum 后统一除以 N²）；对角两项≈0，负对两项各 0.7982
```

### 3.5 端到端使用（双塔）

```python
class SigLIPHead(nn.Module):
    """可学习温度与偏置的 SigLIP 头"""
    def __init__(self, logit_scale_init=2.6):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init)))
        self.logit_bias = nn.Parameter(torch.zeros(()))

    def forward(self, img_emb, txt_emb):
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        sim = img_emb @ txt_emb.t()
        tau = torch.exp(self.logit_scale.clamp(-4.0, 6.0))   # 限制温度范围
        n = sim.size(0)
        y = 2 * torch.eye(n, device=sim.device) - 1
        return -F.logsigmoid(y * (tau * sim + self.logit_bias)).mean()

torch.manual_seed(0)
model = SigLIPHead()
loss = model(torch.randn(8, 16), torch.randn(8, 16))
print(loss.item())   # 输出示例：~0.74（随机初始化：cosine 相似度 ~0，每对损失 ≈ log2 ≈ 0.69）
```

## 四、梯度分析

### 4.1 单对梯度

$$\frac{\partial \mathcal{L}_{ij}}{\partial S_{ij}} = -Y_{ij} \cdot \sigma\left(Y_{ij} z_{ij}\right) \cdot t$$

推导：$\frac{d}{dz}[-\log\sigma(Yz)] = -Y\sigma(Yz)$（见 BCE 篇 4.2），再乘链式 $\frac{\partial z}{\partial S} = t$。

- 正对（Y=+1）：梯度 $-t\cdot\sigma(z) < 0$，相似度拉高；匹配越差梯度越强（$\sigma(z)\to 0$ → 梯度 $-t$）；
- 负对（Y=-1）：梯度 $+t\cdot\sigma(-z) > 0$，相似度压低；**$z$ 很正（错误地把负对判成匹配）时 $\sigma(-z)\to 0$，梯度趋近 0——这是 sigmoid 损失的固有弱点**：严重判错的负对贡献变小（饱和），与 Focal 想解决的相反；
- 梯度对 $t$：温度越大梯度越强，同时越容易饱和。

### 4.2 与 InfoNCE 的梯度对比

| | InfoNCE | SigLIP |
|---|---|---|
| 负样本梯度 | 按 softmax 概率分配（难负样本大） | 每个负对独立、恒定强度（可加权） |
| 饱和现象 | 无（log 惩罚无上界） | 有（σ 饱和，严重错判样本梯度→0） |
| 梯度密度 | 每行 N 个位置 | 每对 1 个位置，共 N² 个 |
| 权重控制 | 隐式（概率） | 显式（α/β） |

## 五、数值稳定性

1. **必须用 log-sigmoid / softplus**：直接 $\log\sigma(z)$ 在 $z$ 很大时 $\sigma\to1$、$\log\to0$（精度丢失），$z$ 很小时 $\sigma\to0$、$\log\to-\infty$；
2. **恒等式的两种稳定写法**：$-\log\sigma(z) = \text{softplus}(-z)$，PyTorch 内部对 $x<0$ 分支用 $-x + \log(1+e^x)$，全程无溢出；
3. **温度限制**：$t$ 是可学习参数，训练初期可能爆掉 → 用 `clamp`（或 log-space 参数化，见 3.5）限制在合理区间（论文 clamp 到 $[e^{-4}, e^{6}]$）；
4. **bias 的作用**：$b$ 学习为负值相当于给所有 logit 加平移，本质上就是让 $\sigma$ 的工作点避开 0 附近（正负对 logit 分布分离开），对数值稳定也有帮助。

## 六、使用场景（含多模态场景）

| 场景 | 为什么用 | 说明 |
|------|---------|------|
| 图文对齐（SigLIP） | 弱 batch 依赖、可加权、通信少 | SigLIP 论文主场景 |
| 大规模分布式预训练 | 不需要跨卡交换相似度矩阵 | 每卡独立算 loss，只同步数值 |
| 图文匹配 + 对齐融合 | 与 ITM 的 BCE 形式统一 | 两者同为 logistic 二分类 |
| 召回排序 | 逐对判匹配 | 冷启动到对比检索 |
| 偏好对齐（DPO） | 对 log-ratio 做二分类 | 结构同 SigLIP（见损失函数详解 7.3） |

**多模态中的两个高频位置**：
1. **SigLIP 视觉塔**：WebLI 数据上 34 亿参数训练，用本损失替代 InfoNCE——训练吞吐更高（无全 gather）、正负样本可控；
2. **SigLIP-2 / 衍生模型**：把 sigmoid 损失与蒸馏、token 级损失组合，仍以逐对二分类为主损失。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 无需全局归一化，batch 依赖弱 | 负对严重错判时梯度饱和（σ→0） |
| 每个 batch 产生 N² 个训练样本，信号稠密 | 对"全局排序"（谁是第一）不敏感 |
| 温度与偏置可学习，少调参 | 负样本等权处理（难易不区分） |
| 支持 α/β 显式加权（类不平衡） | 理论解释弱于 InfoNCE（无互信息视角） |
| 分布式训练通信开销极小 | 相似度矩阵仍要 N² 显存（大 batch 贵） |

## 八、高频面试问答

**Q1：SigLIP 为什么用 sigmoid 而不是 softmax？**
逐对独立二分类不需要全局归一化：① 不依赖 batch 内负样本竞争，小 batch 也稳；② 训练时不需要跨卡收集相似度矩阵（通信少）；③ 支持显式加权与可学习温度偏置；④ 数值上 log-sigmoid 极稳定。

**Q2：SigLIP 和 CLIP 的损失区别？**
CLIP 用对称 InfoNCE（行 softmax + 对角线标签，batch 强依赖）；SigLIP 对全部 N² 对做独立 BCE（±1 标签）。CLIP 优化"排序"，SigLIP 优化"匹配判定"；实证上 SigLIP 在相近算力下效果更好。

**Q3：损失里的 t 和 b 是什么？怎么学？**
t 是温度（把 cosine 相似度缩放成 logit），b 是偏置（吸收正负样本先验不平衡）。两者都是 nn.Parameter，随反向传播学习；t 通常 clamp 在合理范围防爆。

**Q4：为什么说 SigLIP 对难负样本不敏感（相比 InfoNCE）？**
负对 logit 很正（严重误判）时 σ 饱和、梯度→0，惩罚反而变小；InfoNCE 的 log 惩罚无上界、梯度按概率分配，天然关注难负样本。这是 sigmoid 族的固有权衡（DPO 同样有）。

**Q5：N² 个负对会不会太重？怎么权衡？**
负对数量是正对的 N-1 倍，bias 会自动把决策面拉低来平衡；也可用 α/β 加权控制负对贡献。显存上需要存 N×N 相似度矩阵，大 batch 需分块计算。

**Q6：SigLIP 损失和 BCE 什么关系？**
完全同族：SigLIP 就是"对相似度矩阵的每个元素做带 ±1 标签的 BCE（BCEWithLogits）"，温度 t 和偏置 b 是它的 logit 变换。理解 BCE 就理解了 SigLIP。

## 九、自我检验

- [ ] 能写出 SigLIP 损失公式并说明 Y 的定义
- [ ] 能推导 log-sigmoid 恒等并写出 softplus 等价式
- [ ] 会说清 SigLIP vs InfoNCE 的 4 个关键区别
- [ ] 知道 t、b 是可学习参数及其作用
- [ ] 能写出 α/β 加权版本
- [ ] 知道 sigmoid 族"严重错判样本梯度饱和"的弱点
- [ ] 能回答 6 个面试追问
