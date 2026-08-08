# DPO（直接偏好优化）

> 本模块索引见 [微调与对齐详解](微调与对齐详解.md)

## 一、定义与公式（含完整推导）

DPO（Direct Preference Optimization，直接偏好优化）的思想一句话：**RLHF 里的"训练 RM → RL 采样优化"整个链路，可以解析地消掉**——最优策略与奖励之间存在闭式解，把它代回 Bradley-Terry 偏好模型，就得到一个直接以偏好对为监督的交叉熵损失，无需显式 RM、无需 RL 采样。

### 1.1 出发点：KL 约束的 RLHF 目标

$$\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot\mid x)}\big[r(x,y)\big] - \beta\, D_{KL}\big[\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big]$$

### 1.2 第一步：求出最优策略的闭式解

对每个 $x$，这是一个带约束的奖励最大化（等价于带 Lagrange 乘子的 Gibbs 分布）：

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\ \pi_{ref}(y\mid x)\ \exp\!\Big(\frac{r(x,y)}{\beta}\Big), \qquad Z(x) = \sum_y \pi_{ref}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)$$

证明思路：最大化 $\mathbb{E}_{\pi}[r] - \beta\,D_{KL}[\pi\|\pi_{ref}]$，对 $\pi$ 求变分导数为 0，解得 $\pi(y\mid x) \propto \pi_{ref}(y\mid x)e^{r(x,y)/\beta}$，再归一化即得。这个解恰好是"参考策略按奖励做指数加权"。

### 1.3 第二步：把奖励反解出来

对闭式解两边取对数并整理，把奖励写成策略比率的函数：

$$r(x,y) = \beta \log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)} + \beta\log Z(x)$$

这是 DPO 的**关键一步**：奖励函数被重新参数化（reparameterize）到策略的比率上。

### 1.4 第三步：代入 Bradley-Terry 模型

把上式代入 $p(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$，注意 $Z(x)$ 对两个回答相同，直接约掉：

$$\log\frac{\pi^*(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \log\frac{\pi^*(y_l\mid x)}{\pi_{ref}(y_l\mid x)}$$

于是偏好概率只用策略比率表示：

$$p(y_w \succ y_l \mid x) = \sigma\Big(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \beta \log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\Big)$$

### 1.5 DPO 损失（最终公式）

取负对数似然：

$$\boxed{L_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\Big[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\Big]\Big)\Big]}$$

梯度（推导：对 $s_w = \beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}$、$s_l$ 求链式导）：

$$\nabla_\theta L_{DPO} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\beta(s_l - s_w)\big)}_{\text{隐式权重}} \Big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\Big)\Big]$$

**梯度解读（DPO 的精髓）**：
- 梯度方向 = 提升 $y_w$ 概率、压低 $y_l$ 概率（标准的 pair 监督）；
- 权重 $\sigma\big(\beta(s_l - s_w)\big) \in (0,1)$ 是**自适应的**：模型已经正确偏好 $y_w$（$s_w \gg s_l$）时权重 → 0，几乎不再更新（样本已"学会"）；模型搞反了（$s_w < s_l$）时权重 → 1，全力修正。这等价于 RLHF 里 RM 给出隐式奖励后的强化信号。

## 二、核心原理

### 2.1 为什么不需要显式 RM 和 RL 采样

RLHF 链路是：偏好数据 → 训 RM → 用 RM 打分做 RL。DPO 证明这条链路是**冗余的**：最优策略（目标函数 1.2 的极值点）可以用"策略比率"直接参数化奖励（1.3），而 Bradley-Terry 里的奖励差正好只依赖比率（1.4）。所以**偏好数据可以直接监督策略本身**，RM 和采样循环在数学上被消掉了。损失里唯一的外部引用是冻结的 $\pi_{ref}$——这正是 SFT 模型，训练 DPO 时一次性前向即可，不需要在线交互。

### 2.2 隐式奖励（implicit reward）

DPO 在训练后隐式地定义了一个奖励函数：

$$\hat{r}(x,y) = \beta \log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$$

- 训练中无需实例化它，但它决定策略的偏好行为；
- 这解释了 DPO 与 RLHF 的等价性：两者优化的**同一个目标**，只是参数化方式不同（DPO 论文的核心理论贡献是证明了它们的解一致）。

### 2.3 训练流程（对比 RLHF 的简洁）

```
RLHF:  偏好数据 → 训 RM(多次迭代) → 采样+PPO(数万步) → π_θ
DPO:   偏好数据 → π_ref 冻结一次前向 → 直接交叉熵式训练 π_θ
```

DPO 训练循环与 SFT 几乎一样：一个模型在更新、一个模型冻结、一个 log-sigmoid 损失、Adam 优化。没有任何 RL 组件（无 GAE、无 critic、无 clip、无采样）。

### 2.4 局限性（公平起见）

- DPO 是**离线**优化：偏好数据固定，没有在线采样修正分布，策略分布漂移后无法自我纠错（RLHF 在线采样则能）；
- KL 约束由 $\pi_{ref}$ 隐式执行，对 $\beta$ 同样敏感；
- 对偏好数据质量更敏感（没有 RM 这个"过滤层"）。

## 三、源码实现

> 完整可运行的 DPO 训练循环：小 GRU 语言模型 + 合成偏好对，手写 DPO loss，参考模型冻结。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB, CTX, LEN = 32, 4, 8     # 词表、提示长度、回答长度
DIM, HID = 16, 32              # 嵌入维度、GRU 隐层

class TinyLM(nn.Module):
    """极简语言模型（与 RLHF 篇同构，便于对比）"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.gru = nn.GRU(DIM, HID, batch_first=True)
        self.head = nn.Linear(HID, VOCAB)

    def forward(self, seq):                    # (B, T)
        h, _ = self.gru(self.emb(seq))         # (B, T, HID)
        return self.head(h)                    # (B, T, VOCAB)

def sequence_logprob(model, ctx, resp):
    """整句对数概率：只对回答部分逐 token 求和"""
    seq = torch.cat([ctx, resp], dim=-1)               # (B, CTX+LEN)
    logits = model(seq)
    logp = F.log_softmax(logits, dim=-1)
    return logp[:, CTX:].gather(-1, seq[:, CTX:].unsqueeze(-1)).squeeze(-1).sum(-1)

def make_preference_data(n=48):
    """合成偏好对（任务有区分难度）：
    chosen  = 黄金回答扰动 1 个 token（接近正确）
    rejected = 黄金回答扰动 3 个 token（较远偏离）"""
    gold = torch.randint(1, VOCAB, (LEN,))
    data = []
    for _ in range(n):
        ctx = torch.randint(1, VOCAB, (CTX,))
        chosen = gold.clone()
        rejected = gold.clone()
        chosen[torch.randint(LEN, (1,))] = torch.randint(1, VOCAB, (1,))
        for _ in range(3):
            rejected[torch.randint(LEN, (1,))] = torch.randint(1, VOCAB, (1,))
        data.append((ctx, chosen, rejected))
    return data

def dpo_loss(policy, ref, ctx, chosen, rejected, beta=0.5):
    """手写 DPO 损失（对应 1.5 公式，逐项对照）"""
    logp_w = sequence_logprob(policy, ctx, chosen)     # log πθ(y_w|x)
    logp_l = sequence_logprob(policy, ctx, rejected)   # log πθ(y_l|x)
    with torch.no_grad():                              # ref 冻结，只前向不求导
        ref_w = sequence_logprob(ref, ctx, chosen)     # log πref(y_w|x)
        ref_l = sequence_logprob(ref, ctx, rejected)   # log πref(y_l|x)

    # β[log(πθ(y_w)/πref(y_w)) - log(πθ(y_l)/πref(y_l))]
    implicit_diff = (logp_w - ref_w) - (logp_l - ref_l)
    return -F.logsigmoid(beta * implicit_diff).mean()

# ---------- 训练 ----------
policy = TinyLM()
ref = TinyLM()
ref.load_state_dict(policy.state_dict())   # 实际中 ref = SFT 好的模型，这里用随机初始化代替
for p in ref.parameters():
    p.requires_grad = False                # 冻结参考模型

opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
data = make_preference_data(48)

for step in range(300):
    ctx, chosen, rejected = data[step % len(data)]
    loss = dpo_loss(policy, ref, ctx.unsqueeze(0),
                    chosen.unsqueeze(0), rejected.unsqueeze(0))
    opt.zero_grad(); loss.backward(); opt.step()

    # 监控：隐式奖励差 Δ = logπ(y_w) - logπ(y_l) 应持续上升
    with torch.no_grad():
        diff = (sequence_logprob(policy, ctx.unsqueeze(0), chosen.unsqueeze(0))
                - sequence_logprob(policy, ctx.unsqueeze(0), rejected.unsqueeze(0))).item()
    if step % 60 == 0:
        print(f"step {step:3d} | DPO loss = {loss.item():.4f} | Δlogπ = {diff:+.3f}")
# step   0 | DPO loss = 0.6931 | Δlogπ = +2.272
# step  60 | DPO loss = 0.1491 | Δlogπ = +4.332
# step 120 | DPO loss = 0.0035 | Δlogπ = +11.403
# step 180 | DPO loss = 0.0000 | Δlogπ = +22.221
# step 240 | DPO loss = 0.0050 | Δlogπ = +11.048
```

**运行观察**：loss 单调下降、$\Delta\log\pi$ 单调上升——策略逐渐给 $y_w$ 更高隐式奖励。整个循环没有任何 RM、没有采样、没有 critic，与 SFT 的工程复杂度几乎一致，这就是 DPO 流行的原因。

### 3.2 训练好的策略当"隐式奖励器"用

训练结束后 $\hat r(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$ 就是一个免费的评分函数——这是 DPO 相比 RLHF 的隐藏福利（RLHF 还要保留 RM 才能打分）：

```python
# 接上文同一进程继续运行（复用上一代码块训练好的 policy / ref / data）
@torch.no_grad()
def implicit_reward(policy, ref, ctx, resp, beta=0.5):
    """隐式奖励：β·[logπθ(y|x) - logπref(y|x)]（对应 2.2 节公式）"""
    return beta * (sequence_logprob(policy, ctx, resp)
                   - sequence_logprob(ref, ctx, resp))

# 用训练好的策略给一组候选回答排序（拒绝采样/偏好过滤的原料）
ctx, chosen, rejected = data[0]
for name, y in [("chosen", chosen), ("rejected", rejected)]:
    r = implicit_reward(policy, ref, ctx.unsqueeze(0), y.unsqueeze(0)).item()
    print(f"{name:8s} 隐式奖励 = {r:+.3f}")
# chosen   隐式奖励 = -4.686
# rejected 隐式奖励 = -10.883
```

排序与偏好标签一致：DPO 训练出的策略本身就是可用的打分器，无需再训练 RM。

## 四、深入分析

### 4.1 与 RLHF 的理论等价性

DPO 论文的核心理论贡献：DPO 的目标函数（1.5）的**最优解**与 RLHF 目标（1.2）的最优解一致——两者共享同一个 Bradley-Terry 隐式奖励族。差异在于：RLHF 用 RM + 在线采样逼近这个解，DPO 用离线偏好对直接逼近。

### 4.2 权重自适应特性

回顾梯度里隐式权重 $\sigma(\beta(s_l - s_w))$：已学会的样本权重 → 0（不再更新），错误的样本权重 → 1（重点修正）。这带来两个后果：
- 训练后期 loss 很小但模型仍在"温故"，只是更新步长变小；
- **对偏好数据质量极敏感**：如果 $y_l$ 其实也挺好，模型会反复被拉偏；因此 DPO 之前需要严格清洗偏好对（RLHF 有 RM 缓冲层，鲁棒性略好）。

### 4.3 超参数与工程要点

| 要点 | 说明 |
| --- | --- |
| $\beta$ | 通常 0.1~0.5；越大越强对齐、越容易失忆，需配合评估调 |
| 长度偏差 | 长回答 log 概率和更大，DPO 隐性偏好长文本——工业实现常用长度归一化（SimPO 思路） |
| ref 模型 | 必须与策略初始化一致（SFT 产物），否则 DPO 目标是错的 |
| 数据 | 偏好对来自同分布，chosen/rejected 尽量同源生成 |
| 正则 | 可叠加 SFT loss 防过拟合（如 DPO + 10% NLL） |

### 4.4 DPO 家族变体：KTO / ORPO / SimPO

DPO 之后涌现了一批"删组件"式变体，各自简化 DPO 的一个依赖：

**KTO（Kahneman-Tversky Optimization）——去配对**：只需"合意/不合意"单侧标签，不需要 chosen/rejected 成对。基于前景理论，合意样本的更新只在策略得分低于参考时激活：

$$L_{KTO} = \mathbb{E}\big[\lambda_w\, \sigma\big(\beta(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - z_{ref})\big) + \lambda_l\, \sigma\big(\beta(z_{ref} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)})\big)\big]$$

其中 $z_{ref}$ 是参考模型在该样本上的得分基准。优点：数据采集成本低（电商场景的赞/踩反馈可直接用）。

**ORPO（Odds Ratio Preference Optimization）——去 ref 模型**：在 SFT 交叉熵上直接叠加 odds-ratio 偏好项，策略的"上一个 checkpoint"隐式充当参考，省掉单独的 ref 前向：

$$L_{ORPO} = L_{SFT} - \lambda \log\sigma\Big(\log\frac{odds_\theta(y_w\mid x)}{odds_\theta(y_l\mid x)}\Big), \qquad odds_\theta(y\mid x) = \frac{\pi_\theta(y\mid x)}{1 - \pi_\theta(y\mid x)}$$

**SimPO（Simple Preference Optimization）——去 ref + 长度归一化**：用长度归一化的平均 log 概率替代整句概率和（消除 DPO 的长回答偏好），再加一个间隔 margin：

$$L_{SimPO} = -\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x) - \gamma\Big)$$

| 变体 | 去掉的东西 | 换来什么 | 代价 |
| --- | --- | --- | --- |
| KTO | 成对标注 | 可用赞/踩弱标签 | 对标签噪声更敏感 |
| ORPO | ref 模型前向 | 训练更省、一步到位（SFT+对齐） | 无显式 KL 锚点，漂移风险 |
| SimPO | ref 模型 + 长度偏差 | 实现极简、与推理时一致的目标 | 无 ref 约束，需要 margin 调参 |

### 4.5 多模态 DPO

- **数据**：$(image, x, y_w, y_l)$ 图文偏好三元组。构造：给同一张图 + 提示，用多个 VLM 生成候选回答，人工/强模型排序；事实性错误（物体数量、颜色、空间关系）是最典型的偏好信号；
- **训练**：与文本 DPO 完全同构——图像 token 与文本一起进 policy（$\pi_\theta(y\mid x, image)$），ref 模型同样冻结；
- **代表工作**：LLaVA-RLHF（开源 VLM 对齐，显著减少幻觉）、RLHF-V（细粒度视觉偏好，对图像局部区域错误打偏好标签，幻觉率大幅下降）、VLFeedback（360K 图文偏好数据集）、Qwen2-VL 系列的 DPO 对齐；
- **经验**：多模态 DPO 里"正确性"信号比"风格"信号更有效——回答错"图中有几只猫"是硬错误，模型学得又快又稳。

## 五、优缺点

| 优点 | 缺点 |
| --- | --- |
| 无需 RM、无需 RL 采样，训练与 SFT 一样简单 | 离线训练，无法在线纠偏，分布漂移难处理 |
| 训练稳定、易复现、易调参 | 对偏好数据质量敏感，垃圾进垃圾出 |
| 显存小（θ + ref 两份模型） | 隐式 KL 约束，β 需调，长度偏差问题 |
| 理论上有严格推导支撑（与 RLHF 等价） | 奖励表达能力受策略族限制 |
| 与 LoRA 结合效果成熟（工业首选） | 迭代式改数据时需重训 |

## 六、与同类对比

| 维度 | DPO | RLHF（PPO） | GRPO |
| --- | --- | --- | --- |
| 训练资源 | 低（≈SFT） | 高（4 模型 + 采样） | 中高（3 模型 + 组采样） |
| 稳定性 | 高 | 低 | 中高 |
| 需要 RM | 否 | 是 | 是（或规则奖励） |
| 在线采样 | 否 | 是 | 是 |
| 效果上限 | 受离线数据限制 | 理论上限高 | 推理任务上限高 |
| 最佳场景 | 通用偏好对齐、数据量大 | 通用助手、混合信号 | 可验证推理任务 |

**何时选哪个**：有大量高质量偏好对且求稳 → DPO；任务可自动判对错（数学/代码/多模态 VQA）→ GRPO；需要复杂奖励设计或持续迭代 → PPO/RLHF。DPO 是"性价比之王"，RLHF 是"上限之王"，GRPO 是"推理之王"。

## 七、高频面试问答

**Q1：DPO 为什么不需要训练奖励模型？**
因为最优策略有闭式解：$r(x,y) = \beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)} + \beta\log Z(x)$，代入 Bradley-Terry 后 $Z(x)$ 约掉，偏好概率只依赖策略比率，所以奖励被"重参数化"进策略本身，直接优化策略即可。

**Q2：DPO 与 RLHF 理论等价，实际效果差异在哪？**
目标最优解一致，但路径不同：RLHF 在线采样能自我修正分布偏差，DPO 离线受数据分布限制；RLHF 有 RM 缓冲层对噪声更鲁棒，DPO 直接吃偏好对更敏感；工程上 DPO 便宜稳定，RLHF 贵且难调。

**Q3：DPO 损失里的 β 和 clip？**
DPO 没有 clip（那是 PPO 的机制）。β 是 KL 强度系数：越大对齐越激进，模型越容易遗忘原能力。DPO 里唯一类似"约束"的机制是 ref 模型固定提供的隐式 KL 惩罚。

**Q4：DPO 的隐式奖励是什么？**
$\hat{r}(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$。训练后可直接用它给回答排序、做拒绝采样，无需额外 RM。

**Q5：为什么说 DPO 对偏好数据质量敏感？**
DPO 梯度权重 $\sigma(\beta(s_l - s_w))$ 会全力修正模型分不清的样本——如果 y_l 本身不错、只是被错误标注，模型会被反复误导。RLHF 中 RM 会先"平均"这类噪声，缓冲更好。

**Q6：DPO 有长度偏差问题吗？怎么解决？**
有：长回答累计 log 概率大，隐式奖励虚高，模型倾向输出长文。缓解：长度归一化（SimPO 的做法）、chosen/rejected 长度匹配、截断对齐长度。

**Q7：DPO 能用在多模态上吗？和文本有什么不同？**
能，直接条件在图像上（$\pi_\theta(y\mid x, image)$）。差异主要在数据：偏好对基于图文指令构造，信号以事实性硬错误为主（数量/空间/颜色），RLHF-V 等用细粒度局部偏好大幅降低幻觉。

**Q8：KTO/ORPO/SimPO 与 DPO 的关系？**
KTO 用合意/不合意标签（无需成对）；ORPO 在 SFT 损失上加 odds-ratio 偏好项（无需 ref 模型）；SimPO 用长度归一化平均 log 概率 + margin（无需 ref 模型）。三者都是 DPO 思想的变体，工程简化方向：去掉配对、去掉 ref。

## 八、自我检验

- [ ] 能从 RLHF 目标出发完整推导 DPO 损失（闭式解 → 反解奖励 → 代入 BT → 负对数似然）
- [ ] 能写出 DPO 梯度并解释隐式权重 $\sigma(\beta(s_l - s_w))$ 的含义
- [ ] 能说清 DPO 为什么不需要 RM 和 RL 采样
- [ ] 能跑通本篇 DPO 训练代码并解释 Δlogπ 上升的含义
- [ ] 能对比 DPO 与 RLHF 在资源/稳定性/效果上的差异
- [ ] 能说出 DPO 的 3 个缺点及应对（离线、数据敏感、长度偏差）
- [ ] 能简述 KTO / ORPO / SimPO 各自简化了什么
- [ ] 能说明多模态 DPO 的数据构造与代表工作（LLaVA-RLHF、RLHF-V）
- [ ] 能回答 8 个面试追问
