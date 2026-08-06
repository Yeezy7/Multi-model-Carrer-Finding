# 交叉熵损失 CrossEntropyLoss

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 从最大似然出发

K 类分类问题：模型输出 logits $z \in \mathbb{R}^K$，真实类别 $y \in \{1, \dots, K\}$。分类模型把 logits 经 Softmax 变成类别分布：

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

训练目标是最大化"预测出正确类别"的概率（最大似然）。对单样本取负对数即得交叉熵：

$$\mathcal{L} = -\log p_y = -\log \frac{e^{z_y}}{\sum_{k=1}^{K} e^{z_k}} = -z_y + \log\sum_{k=1}^{K} e^{z_k}$$

**这是整个深度学习中最重要的一个公式**：前半项 $-z_y$ 把正确类别的 logit 拉大，后半项 $\log\sum e^{z_k}$（log-sum-exp）把其他类别 logit 压小。

### 1.2 log-softmax 恒等（数值上必须用这个形式）

$$\log p_k = z_k - \log\sum_{j=1}^{K} e^{z_j} = z_k - \text{logsumexp}(z)$$

交叉熵与 NLL（负对数似然）的关系：

$$\mathcal{L} = \text{NLLLoss}(\text{LogSoftmax}(z), y) = \text{CrossEntropyLoss}(z, y)$$

即：**CE = NLL(log-softmax(z))**，框架内部从不先算 $e^{z_k}$ 再算 $\log$，而是直接用恒等式合并。

### 1.3 一般形式（one-hot / 软标签）

对任意标签分布 $q$（one-hot 或软标签）：

$$\mathcal{L} = -\sum_{k=1}^{K} q_k \log p_k = \underbrace{-\sum_k q_k \log p_k + \sum_k q_k \log q_k}_{} - H(q) = H(p, q) - H(q)$$

即交叉熵 = 交叉项 + 常数项（$q$ 固定时 $H(q)$ 是常数），最小化 CE 等价于最小化交叉熵 $H(p,q)$。

## 二、数学性质与直觉

### 2.1 梯度是"预测减标签"（最重要直觉）

$$\frac{\partial \mathcal{L}}{\partial z_k} = p_k - \mathbb{1}[k = y]$$

- 正确类别：$\partial \mathcal{L}/\partial z_y = p_y - 1 < 0$ → 反向传播时增大 $z_y$；
- 错误类别：$\partial \mathcal{L}/\partial z_k = p_k > 0$ → 降低 $z_k$；
- 预测越自信（$p_y \to 1$），梯度越小；**预测错误但很自信时（$p_y \to 0$），梯度接近 -1，惩罚巨大**。这是交叉熵"杀一儆百"的直觉：分类错得越离谱，推得越狠。

### 2.2 下界与最优值

- $\mathcal{L} \ge 0$，当且仅当 $p = \text{onehot}(y)$ 时取 0（理论上）；
- 随机初始化下，K 类均匀预测的期望损失 $\approx \log K$（10 类 ≈ 2.30，1000 类 ≈ 6.91）——**训练早期 sanity check 用这个**；
- 对 logits 是凸函数（对 $z_y$ 的导数单调），配合 softmax 无局部最优陷阱。

### 2.3 温度版（蒸馏 / 对比学习共用）

把 logits 除以温度 $T$ 再 softmax：

$$p_k^{(T)} = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}, \qquad \mathcal{L}_T = -\log p_y^{(T)}$$

- $T > 1$：分布变平（软化），保留类间相似关系；
- $T < 1$：分布变尖（锐化），近似 one-hot，梯度更大（对比学习常用 $T \approx 0.07$）；
- 梯度缩放：$\partial \mathcal{L}_T / \partial z_k = (p_k^{(T)} - \mathbb{1}[k=y]) / T$，温度越小梯度越大。

### 2.4 Label Smoothing（标签平滑）

one-hot 换成软标签 $q = (1-\varepsilon)\,\text{onehot}(y) + \varepsilon / K$：

$$\mathcal{L} = -(1-\varepsilon)\log p_y - \frac{\varepsilon}{K}\sum_{k=1}^{K}\log p_k$$

- 防止 softmax 概率被推向 1（模型过度自信），缓解过拟合、提升校准；
- Transformer 系训练标配（$\varepsilon = 0.1$）；
- 代价：训练 loss 不再能收敛到 0，指标判断以准确率为准。

## 三、源码实现（手写版本 + PyTorch 官方接口）

### 3.1 手写版（naive + 数值稳定）

```python
import torch
import torch.nn.functional as F

def softmax_naive(z):
    """朴素版：e^z 直接算（x 大时会溢出）"""
    return torch.exp(z) / torch.exp(z).sum(dim=-1, keepdim=True)

def log_softmax_stable(z):
    """稳定版：先减最大值，再 log-sum-exp"""
    m = z.max(dim=-1, keepdim=True).values
    return z - m - torch.log((z - m).exp().sum(dim=-1, keepdim=True))

def cross_entropy_manual(logits, target):
    """手写 CE = NLL(log_softmax)，reduction='mean'"""
    log_probs = log_softmax_stable(logits)
    return -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).mean()

logits = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([2])
print(cross_entropy_manual(logits, target))   # tensor(0.4076)
```

### 3.2 PyTorch 官方接口

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()          # 默认 reduction='mean'
print(criterion(logits, target))           # tensor(0.4076)

# 等价写法：NLL + LogSoftmax
print(F.nll_loss(F.log_softmax(logits, dim=-1), target))   # tensor(0.4076)

# label smoothing（PyTorch 1.10+ 原生支持）
criterion_s = nn.CrossEntropyLoss(label_smoothing=0.1)
print(criterion_s(logits, target))         # tensor(0.5076)

# 温度版：手动除以温度即可（对比学习/蒸馏常用）
tau = 2.0
print(F.cross_entropy(logits / tau, target))               # tensor(0.6802)
```

### 3.3 输出对比验证

```python
# 手写 vs 官方：随机张量上应完全一致（数值稳定后误差 < 1e-6）
torch.manual_seed(0)
z = torch.randn(4, 10)
y = torch.randint(0, 10, (4,))
manual = cross_entropy_manual(z, y)
official = F.cross_entropy(z, y)
print(f"manual={manual.item():.6f}  official={official.item():.6f}")
# 输出示例：manual=2.599109  official=2.599109（多次运行值不同但两者恒等）
```

### 3.4 语言建模（Causal LM）中的用法

```python
class LMHead(nn.Module):
    """生成任务：逐 token 交叉熵，只对回答部分计算"""
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, hidden_states, labels, loss_mask):
        logits = self.fc(hidden_states)                 # [B, T, V]
        logits = logits.view(-1, logits.size(-1))       # [B*T, V]
        labels = labels.view(-1)
        mask = loss_mask.view(-1)                       # 1=算 loss，0=忽略(prompt)
        loss = F.cross_entropy(logits, labels, reduction='none')
        return (loss * mask).sum() / mask.sum().clamp(min=1)
```

## 四、梯度分析

### 4.1 联合梯度推导（面试必考）

设 $p_k = e^{z_k} / \sum_j e^{z_j}$，$\mathcal{L} = -\log p_y$。链式展开：

$$\frac{\partial \mathcal{L}}{\partial z_k} = \frac{\partial}{\partial z_k}\left(-z_y + \log\sum_j e^{z_j}\right) = -\mathbb{1}[k=y] + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - \mathbb{1}[k=y]$$

| 情形 | 梯度 | 效果 |
|------|------|------|
| 正确类别（k=y） | $p_y - 1 \in (-1, 0)$ | 增大该 logit |
| 错误类别（k≠y） | $p_k \in (0, 1)$ | 按概率比例减小各 logit |

```python
# 数值验证：梯度 = softmax 概率 - one-hot
z = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
F.cross_entropy(z, torch.tensor([2])).backward()
print(z.grad)  # tensor([[ 0.0900,  0.2447, -0.3348]])  = p - onehot
```

### 4.2 预测错误的惩罚强度

| 预测概率 p_y | CE 损失 | 梯度幅度 |
|------------|---------|---------|
| 0.99 | 0.010 | 0.01（几乎不推） |
| 0.90 | 0.105 | 0.10 |
| 0.60 | 0.511 | 0.40 |
| 0.10 | 2.303 | 0.90（错得狠推得狠） |
| 0.01 | 4.605 | 0.99 |

### 4.3 温度对梯度的缩放

温度版梯度 $\partial \mathcal{L}_T / \partial z_k = (p_k^{(T)} - \mathbb{1}[k=y]) / T$：小温度同时放大"梯度强度"和"类间对比"，所以对比学习用 $\tau=0.07$ 能把难负样本的梯度压出来；代价是梯度方差变大、训练不稳定。

## 五、数值稳定性

1. **上溢**：$z_k$ 很大时 $e^{z_k} \to \infty$（FP32 下 $e^{88} \approx 1.6 \times 10^{38}$ 溢出）→ 减最大值技巧：$\log\sum e^{z_k} = m + \log\sum e^{z_k - m}$；
2. **下溢**：$e^{z_k} \to 0$ 时 $\log p_y \to -\infty$ → log-softmax 恒等式直接返回 $z_k - \text{lse}(z)$，规避了 log(0)；
3. **不要先算 softmax 再取 log**：log(softmax) = 对 0~1 之间的小数取 log，精度损失巨大；正确写法是 log-softmax 一步到位；
4. **语言模型中**：logits 可能高达数百，LM head 的 log-softmax 若未稳定实现会 NaN——这也是 FlashAttention 时代把 softmax 融入 kernel 的原因。

```python
# 演示不稳定的 naive 写法在极端 logits 下出错
z_extreme = torch.tensor([[1000.0, 1000.0, 1000.0]])
try:
    print(softmax_naive(z_extreme))   # nan（exp(1000) 溢出）
except Exception as e:
    print("naive 溢出:", type(e).__name__)
print(log_softmax_stable(z_extreme))  # tensor([[-1.0986, -1.0986, -1.0986]])，正确
```

## 六、使用场景（含多模态场景）

| 场景 | 用法 | 备注 |
|------|------|------|
| 多分类 | `CrossEntropyLoss(logits, labels)` | 默认选择 |
| 语言建模（caption/VQA/对话） | 逐 token CE，mask 掉 prompt | 所有生成式多模态模型（LLaVA、InstructBLIP）的主损失 |
| 多模态预训练分类头 | CLIP/ALIGN 前的分类式预训练 | 已基本被 InfoNCE 取代 |
| 知识蒸馏 | 温度化 CE / 软标签 CE | 见 KL 散度篇 |
| 对比学习 | 温度化 CE + 对角线标签 | 等价于 InfoNCE（见 InfoNCE 篇） |
| VQA 选择题 | CE over 候选答案 logits | VQA v2 早期做法 |
| 视觉 tokenizer | VQ-VAE 的 codebook 分类 CE | 离散表征学习 |

**多模态中的两个高频位置**：
1. **生成式模型（LLaVA/Qwen-VL/InstructBLIP）**：图像编码后作为 prompt，文本侧逐 token CE 是唯一训练信号；
2. **对比学习的底层实现**：CLIP 的 InfoNCE 在代码层面就是"温度缩放 + CE"，吃透 CE 的梯度才能理解 CLIP 的训练行为。

## 七、优缺点总结

| 优点 | 缺点 |
|------|------|
| 梯度形式优美（p - onehot），优化稳定 | 对噪声标签敏感（噪声样本被大幅惩罚） |
| 配合 softmax 有概率解释、凸性保证 | 硬标签可能过拟合（需 label smoothing） |
| 数值稳定写法成熟（log-sum-exp） | 只关心正确类，不利用类间结构 |
| 与 NLL/InfoNCE/蒸馏损失统一 | 不擅长极度不平衡场景（用 Focal） |
| 支持软标签/温度/掩码，通用性强 | 生成任务中的"逐个 token 独立"假设略粗糙 |

## 八、高频面试问答

**Q1：CrossEntropyLoss 为什么输入 logits 而不是概率？**
数值稳定 + 梯度干净。框架内部用 log-softmax 恒等式 $z_y - \text{logsumexp}(z)$ 一步算完，避免 exp 溢出与 log(0)；梯度直接是 $p - \text{onehot}$，不用经过 softmax 的 Jacobian 回传。

**Q2：推导 CrossEntropy 对 logits 的梯度。**
$\frac{\partial \mathcal{L}}{\partial z_k} = p_k - \mathbb{1}[k=y]$：正确类梯度为负（抬高），错误类梯度为正且正比于概率（压低），整体概率和保持为 1。

**Q3：Label Smoothing 为什么有效？**
one-hot 要求 softmax 冲到 1，参数被推向极端 → 过拟合、校准差。平滑后目标概率上限 $(1-\varepsilon)+\varepsilon/K < 1$，模型更保守；ε=0.1 是 Transformer 惯例。

**Q4：温度 T 对交叉熵的影响？**
logits 除以 T 后分布变平（T>1）或变尖（T<1），梯度额外乘 1/T。蒸馏用 T=2~8 软化教师分布；对比学习用 T≈0.07 锐化分布放大难负样本梯度。

**Q5：CE 和 KL 散度什么关系？**
$\mathcal{L}_{CE} = D_{KL}(q \| p) + H(q)$，标签分布 q 固定时 $H(q)$ 为常数，所以最小化 CE ≡ 最小化 KL。这是蒸馏"最小化 KL 就是最小化 CE"的理论基础。

**Q6：为什么生成任务也用 CE 而不是直接优化 CIDEr/BLEU？**
BLEU/CIDEr 不可微（离散 n-gram 匹配），CE 提供逐 token 平滑梯度。训练与推理的分布偏差（exposure bias）是另一个问题，靠 scheduled sampling / RL 优化解决。

**Q7：CE 在类别极度不平衡时有什么问题？**
海量简单负样本的平均梯度淹没少数正类 → 损失被"简单类"主导，模型退化成常数预测。解法：Focal Loss、类别加权 CE、或换对比损失。

## 九、自我检验

- [ ] 能写出 CE 公式并推导梯度（softmax - onehot）
- [ ] 能写出 log-softmax 恒等式并解释为什么数值稳定
- [ ] 能解释 label smoothing 的公式与动机
- [ ] 知道温度版 CE 的梯度缩放因子 1/T
- [ ] 能说明 CE、NLL、LogSoftmax、KL 四者的关系
- [ ] 会写手写版并验证与官方输出一致
- [ ] 能说出 CE 在生成模型与对比学习中的两个高频位置
- [ ] 能回答 7 个面试追问
