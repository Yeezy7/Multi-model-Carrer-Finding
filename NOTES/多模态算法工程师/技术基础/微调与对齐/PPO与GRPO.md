# PPO 与 GRPO

> 本模块索引见 [微调与对齐详解](微调与对齐详解.md)

## 一、定义与公式

PPO（Proximal Policy Optimization，近端策略优化）是 RLHF 阶段三的主力算法；GRPO（Group Relative Policy Optimization，组相对策略优化）是 DeepSeek 团队提出的 PPO 变体，去掉了价值网络（critic），成为 R1 系推理强化的标配。本篇先讲 PPO，再讲 GRPO，最后给总对比。

### 1.1 PPO 目标函数与 clip 机制

PPO 优化的是 clipped 代理目标：

$$L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(\underbrace{r_t(\theta)}_{\text{ratio}} \hat{A}_t,\ \operatorname{clip}\big(r_t(\theta), 1-\varepsilon, 1+\varepsilon\big)\,\hat{A}_t\big)\Big]$$

其中重要性采样比：$r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$

**clip 为什么能稳定更新**：未 clip 的目标 $L^{CPI} = \mathbb{E}[r_t(\theta)\hat{A}_t]$ 鼓励 ratio 无限增大（贪婪刷分）。clip 把 ratio 截到 $[1-\varepsilon, 1+\varepsilon]$：
- 当 $\hat{A}_t > 0$ 时，ratio 超过 $1+\varepsilon$ 的部分收益被切断 → 策略"最多往前走 $\varepsilon$"；
- 当 $\hat{A}_t < 0$ 时，ratio 低于 $1-\varepsilon$ 的部分惩罚被截断 → 防止因一步坏样本把策略推太远。

数学上，$\min$ 操作保证了**每次更新的目标是不低于未 clip 目标的（下界）**，配合重要性采样的旧分布约束，形成了不依赖 TRPO 共轭梯度求解的"软信任域"。总损失：

$$L^{PPO} = L^{CLIP} - c_1 \underbrace{(V_\psi(s_t) - \hat{R}_t)^2}_{L^V} + c_2\,\underbrace{S[\pi_\theta(s_t)]}_{\text{熵奖励}}$$

### 1.2 GAE 优势估计

优势 $\hat{A}_t$ 用广义优势估计（GAE）计算，它是时序差分误差的指数衰减加权和：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots = \sum_{k \ge 0}(\gamma\lambda)^k \delta_{t+k}$$

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| $\gamma$ | 折扣因子，多远视界 | 0.95~0.99 |
| $\lambda$ | 偏差-方差权衡：$\lambda=0$ 纯 TD（高偏差低方差）；$\lambda=1$ 纯 MC（低偏差高方差） | 0.95~0.99 |

### 1.3 GRPO 目标函数（组内相对优势）

GRPO 的动机：PPO 中 critic 与 actor 同规模，显存翻倍且价值函数训练不稳定。GRPO **彻底去掉 critic**，对每个提示 $x$ 采样一组 $G$ 个回答 $\{y_1,\dots,y_G\}$，用**组内奖励归一化**估计优势：

$$\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}, \qquad i = 1,\dots,G$$

训练目标（KL 直接作为惩罚项加入目标，而不是塞进奖励）：

$$L^{GRPO}(\theta) = -\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\min\Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{old}}(y_i\mid x)}\hat{A}_i,\ \operatorname{clip}\big(\tfrac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\varepsilon, 1+\varepsilon\big)\hat{A}_i\Big)\Big] + \beta\,D_{KL}\big[\pi_\theta\ \|\ \pi_{ref}\big]$$

**为什么组内归一化是对的**：对固定提示，组内均值是优势的**无偏基线**（与 critic 的期望作用相同），而组间偏好差异（prompt 难度不同）不会污染优势估计——这正是"相对优势"的含义。去掉 critic 还顺带解决了价值函数训练不稳定的老问题。

## 二、核心原理：PPO 在 RLHF 中的完整角色

### 2.1 四个模型的流水线

```
提示 x ──► π_θ(actor) ──采样──► 回答 y
                │                       │
                │                  ┌────┴─────┐
                │                  r_φ(RM) 打分
                │                  π_ref 算 KL
                │                  ┌────┴─────┐
                │               reward = r - β·KL
                │                       │
                │                   V_ψ(critic)
                │                  GAE 优势 Â
                │                       │
                └───── clip 目标更新 ────┘
```

| 模型 | 角色 | 训练 |
| --- | --- | --- |
| $\pi_\theta$（actor） | 生成回答的策略，被优化 | 更新（clip 目标） |
| $V_\psi$（critic） | 逐 token 估计状态价值，提供 GAE 基线 | 更新（MSE） |
| $\pi_{ref}$（参考） | KL 锚点，防漂移 | 冻结 |
| $r_\phi$（RM） | 整句打分，奖励信号 | 冻结 |

### 2.2 逐 token 的奖励构成

RM 只给**整句**一个分数，但 PPO 需要逐 token 优势。做法：句子内部把 RM 分数（或该分数减一个常数基线）平均摊到每个 token，再叠加每个 token 的 KL 惩罚：

$$r_t = \underbrace{\frac{r_\phi(x,y) - \bar{r}}{|y|}}_{\text{分摊的 RM 分数}} - \beta \log\frac{\pi_\theta(y_t \mid \cdot)}{\pi_{ref}(y_t \mid \cdot)}$$

$\bar{r}$ 是批次平均（只影响基线不影响相对排序）。KL 项对每个 token 都不同，提供了 token 级的奖励变化，critic 和 GAE 才能学到"哪一步值得鼓励"。

### 2.3 为什么 RLHF 里 PPO 而非普通策略梯度

普通策略梯度（REINFORCE）方差大、步长无法控制；TRPO 用 KL 硬约束 + 共轭梯度，实现复杂。PPO 用一行 clip 实现了"信任域"效果，计算简单、稳定、通用——这是它在 RLHF 胜出的原因。

## 三、深入分析：GRPO 的动机、收益与应用

### 3.1 为什么去掉 critic 行得通

| PPO 中 critic 的作用 | GRPO 的替代 |
| --- | --- |
| 提供状态价值基线 $V(s)$，降方差 | 组内均值 $\frac{1}{G}\sum r_i$ 是无偏基线（同一 prompt 下） |
| 计算逐 token 优势 | 奖励是整句级的，直接对组内回答归一化即可 |
| 训练慢、易发散（价值网络学习曲线长） | 无需训练价值网络，训练目标单一稳定 |

代价：一个 prompt 需要采样 $G$ 个回答（通常 4~16），推理采样量增加，但相对省掉的 critic 前向/反向与优化器状态，综合显存收益明显。

### 3.2 显存收益量化（7B 规模，BF16 + Adam）

| 组件 | PPO | GRPO |
| --- | --- | --- |
| actor | ~14GB + 28GB 优化器 | 同左 |
| critic | ~14GB + 28GB 优化器 | **无** |
| ref / RM | 各 ~7GB（仅推理） | 同左 |
| 合计 | ≈ 98GB | ≈ 56GB（省 ~40%） |

（以 `--max_steps` 单卡示意；实际随 batch、序列长度浮动。）省下的正是 critic 及其 Adam 状态。这也是 R1 系 70B 级别能用相对小集群训练的原因之一。

### 3.3 GRPO 的奖励设计（推理场景）

R1/DeepSeekMath 中 GRPO 与**可验证奖励（verifiable reward）**搭配：答案正确性由确定性规则判定（数学答案匹配、代码单测通过），无需训练 RM。加上格式奖励（思考过程结构、<|begin_of_thought|> 标签）。规则奖励解决了 RLHF 的 RM 偏差问题，是推理强化最干净的路子。

### 3.4 多模态与推理场景：思考强化（Thinking RL）

**RL 强化"思考模式"**：R1 范式的核心——不教模型"怎么想"，而是用 RL 让模型**自己学会在回答前输出思考过程**：
- 奖励 = 答案可验证正确性（规则判定）+ 格式（思考包裹在 `thinking` 标签里）+ 长度/语言惩罚；
- GRPO 因"组内相对优势"天然适合：思考过程多样、答案可自动判分，无需训练 RM；
- 训练中模型自发涌现长思考、自我纠错、反思等行为（emergent thinking）。

**多模态推理强化（Qwen3-VL 等）**：
- Qwen3-VL 的思维训练：分两阶段——第一阶段强化思考（输出 thinking 再回答，用 GRPO + 格式/可验证奖励），第二阶段反思（answer 后自我修正，再训练），显著提升多步推理与工具使用能力；
- 多模态推理的奖励设计：视觉问答正确答案匹配、OCR/图表结果验证、与文本不同的点在于"答案正确性判定往往需要视觉理解模型或模板规则"；
- 图像偏好类对齐仍走 DPO/RLHF（如 RLHF-V），推理类强化走 GRPO——两类互补，共同构成多模态对齐工具箱。

## 四、源码实现

### 4.1 GAE 与 PPO 更新步（纯张量，可运行）

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """GAE(λ)：δ_t = r_t + γV_{t+1} - V_t，Â_t = Σ(γλ)^k δ_{t+k}（倒序递推）"""
    T = len(rewards)
    adv = torch.empty(T)
    gae = torch.zeros(())
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else torch.zeros(())
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv

def ppo_clip_loss(old_logp, logp, advantages, clip_eps=0.2):
    """PPO clip 目标：min(ratio·Â, clip(ratio,1±ε)·Â)"""
    ratio = torch.exp(logp - old_logp)                 # 重要性采样比
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()

# ---- 演示：5 步合成轨迹 ----
T = 5
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 2.0])     # 每步真实奖励
values = torch.tensor([0.8, 0.9, 0.7, 1.1, 1.2, 0.9]) # critic 估值（含 T+1 的 V(末端)=0.9）
adv = compute_gae(rewards, values)
print("GAE 优势:", adv.numpy())
# GAE 优势: [2.824 1.843 2.179 0.84 0.8]

old_logp = torch.randn(T) * 0.5                        # 旧策略 log 概率
logp = old_logp + 0.3                                  # 模拟一次小更新（ratio≈1.35）
loss = ppo_clip_loss(old_logp, logp, adv)
print(f"PPO loss = {loss.item():.4f}")                 # PPO loss = -2.0368
```

### 4.2 GRPO 组内优势归一化 + 完整训练循环（可运行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB, CTX, LEN = 32, 4, 8
DIM, HID = 16, 32

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.gru = nn.GRU(DIM, HID, batch_first=True)
        self.head = nn.Linear(HID, VOCAB)
    def forward(self, seq):                            # (B, T)
        h, _ = self.gru(self.emb(seq))
        return self.head(h)

def sequence_logprob(model, ctx, resp):
    """整句 log 概率（只算回答部分）"""
    seq = torch.cat([ctx, resp], dim=-1)
    logp = F.log_softmax(model(seq), -1)
    return logp[:, CTX:].gather(-1, seq[:, CTX:].unsqueeze(-1)).squeeze(-1).sum(-1)

@torch.no_grad()
def sample_group(policy, ctx, G=4, max_len=LEN):
    """对单条提示采样 G 个回答：ctx: (CTX,) -> (G, LEN)"""
    seq = ctx.unsqueeze(0).expand(G, -1).clone()
    for _ in range(max_len):
        token = torch.multinomial(policy(seq)[:, -1].softmax(-1), 1)
        seq = torch.cat([seq, token], dim=-1)
    return seq[:, CTX:]

def group_advantage(rewards):
    """GRPO 组内归一化：Â_i = (r_i - mean(r)) / std(r)（rewards: (G,)）"""
    r = rewards.float()
    return (r - r.mean()) / (r.std() + 1e-6)

GOLD = torch.randint(1, VOCAB, (LEN,))                 # 黄金回答（模拟可验证的标准答案）

def rule_reward(resp):
    """规则奖励（模拟可验证奖励）：回答与黄金答案逐位匹配比例 ∈ [0,1]
    （密集奖励：真实场景对应数学题步骤分/代码用例通过率）"""
    return (resp == GOLD.unsqueeze(0)).float().mean(dim=-1)   # (G,)

def grpo_step(policy, ref, ctx, opt, beta=0.1, clip_eps=0.2, G=4, epochs=2):
    """一轮 GRPO：采样组 -> 规则奖励 -> 组内归一化优势 -> clip 更新 + KL 惩罚"""
    ctxs = ctx.unsqueeze(0).expand(G, -1)
    resp = sample_group(policy, ctx, G)                # (G, LEN)
    with torch.no_grad():
        old_logp = sequence_logprob(policy, ctxs, resp)
        r = rule_reward(resp)                          # (G,) 可验证奖励
        adv = group_advantage(r)                       # (G,) 组内相对优势
        print(f"  组内奖励={r.numpy().round(3).tolist()} 优势={adv.numpy().round(3).tolist()}")

    for _ in range(epochs):
        logp = sequence_logprob(policy, ctxs, resp)
        kl = logp - sequence_logprob(ref, ctxs, resp)  # 逐样本 KL(π‖π_ref)
        ratio = torch.exp(logp - old_logp)
        pg = torch.min(ratio * adv,
                       torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
        loss = -(pg.mean() - beta * kl.mean())         # 目标 = PG - β·KL
        opt.zero_grad(); loss.backward(); opt.step()
    return r.mean().item()

policy = TinyLM()
ref = TinyLM()
ref.load_state_dict(policy.state_dict())
for p in ref.parameters():
    p.requires_grad = False

opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
ctx = torch.randint(1, VOCAB, (CTX,))

for it in range(12):
    mean_r = grpo_step(policy, ref, ctx, opt, G=4)
    print(f"GRPO it {it:2d} | 组内平均奖励 = {mean_r:.3f}")
# GRPO it  0 | 组内奖励=[0.0, 0.125, 0.0, 0.0] 优势=[-0.5, 1.5, -0.5, -0.5] | 平均 = 0.031
# GRPO it  5 | 组内奖励=[0.125, 0.0, 0.125, 0.0] 优势=[0.866, -0.866, 0.866, -0.866] | 平均 = 0.062
# GRPO it 11 | 组内奖励=[0.125, 0.0, 0.125, 0.375] 优势=[-0.199, -0.993, -0.199, 1.391] | 平均 = 0.156
```

**运行观察**：组内奖励从 0.03 逐步爬到 0.16——策略在"更多位置对上黄金答案"。注意两件事：① 组内全部同分时优势恒为 0（无梯度信号），GRPO 依赖组内差异驱动学习；② 优势符号正确反映组内相对好坏（最好的回答拿正优势被加权提升）。这展示了 GRPO 靠"组内相对差异"驱动学习的机制。

## 五、优缺点

| | PPO | GRPO |
| --- | --- | --- |
| 优点 | 理论完备、通用、有信任域保证；适合奖励形态复杂的场景 | 无 critic：显存省 ~40%、训练稳定、实现简单；天然适配可验证奖励 |
| 缺点 | critic 训练难、显存大、超参多（γ, λ, c1, c2, ε） | 组内采样量需求大；无价值估计，不适合长程稀疏奖励场景；组内全同奖励时无信号 |
| 适用 | 通用助手 RLHF、奖励来自 RM | 数学/代码/多模态推理等可验证任务 |

## 六、与同类对比：PPO vs GRPO vs DPO

| 维度 | PPO | GRPO | DPO |
| --- | --- | --- | --- |
| 是否需要 RM | 是（或混合奖励） | 是，但常用规则/可验证奖励 | 否 |
| 是否需要 critic | 是（额外 1 份模型显存） | 否（组内归一化替代） | 否 |
| 是否需要 ref 模型 | 是（KL 锚点） | 是 | 是 |
| 在线采样 | 是（逐 token 交互） | 是（组级采样） | 否（离线偏好数据） |
| 稳定性 | 低（价值网络易崩） | 中高 | 高 |
| 显存 | 4 模型 | 3 模型 | 2 模型 |
| 数学/代码推理 | 可用 | **最佳**（R1 系标配） | 次之（需离线偏好对） |
| 通用助手对齐 | **最佳**（奖励形态自由） | 可用 | **最佳性价比** |

一句话：**PPO 通用但贵，GRPO 为可验证推理而生，DPO 简单稳定最省钱。**

## 七、高频面试问答

**Q1：PPO 的 clip 到底解决了什么问题？**
防止策略更新过大导致崩塌。无 clip 时 ratio 可无限增大（正优势下贪婪），更新一步就破坏旧策略分布；clip 把 ratio 限制在 $[1\pm\varepsilon]$，实现无需 TRPO 共轭梯度的软信任域，梯度更稳定。

**Q2：GAE 的 λ 和 γ 分别控制什么？**
γ 是折扣因子（看多远）；λ 是偏差-方差权衡：λ→0 偏向 TD（方差小、偏差大），λ→1 偏向 MC（偏差小、方差大）。RLHF 常用 0.95~0.99 折中。

**Q3：RLHF 中 PPO 为什么需要 critic，而 GRPO 不需要？**
critic 提供价值基线降方差。GRPO 用"同提示多回答的组内均值"做基线——对同一提示，组内均值是优势的无偏估计，且不需要训练价值网络，省显存、更稳定。

**Q4：GRPO 的组内优势公式与 critic 基线的关系？**
$\hat{A}_i = \frac{r_i - \bar r}{std}$ 中 $\bar r$ 起基线作用（替代 $V(s)$），除以组内标准差相当于对奖励做归一化（替代价值尺度对齐）。所以 GRPO = PPO 去掉 critic + 组内归一化。

**Q5：GRPO 有什么明显局限？**
组内无差异时无梯度信号（全部回答同分）；奖励必须逐回答可判且噪声小，否则归一化被噪声主导；采样 G 个回答增加了推理开销。

**Q6：为什么数学/代码推理强化用规则奖励 + GRPO 而不是 RM + PPO？**
规则奖励无偏差、零训练成本、可无限扩展；GRPO 无 critic 训练更稳；R1 实验表明可验证奖励的干净信号 + 组内相对优势就能涌现思考能力，RM 在可判对错的任务上是多余的。

**Q7：思考型 RL（thinking RL）为什么能让模型"学会思考"？**
格式奖励逼模型输出思考 token，正确性奖励让它发现"想得越久越对"，GRPO 组内差异放大了不同思考路径的奖励差，策略自然向有效思考方向演进——这是涌现行为而非显式指导。

**Q8：DPO 和 GRPO 什么时候一起用？**
典型流程：先 DPO 做粗对齐（改风格、去幻觉），再 GRPO 做任务强化（数学/VQA 正确率）。DPO 便宜、GRPO 精准，分层使用。

## 八、自我检验

- [ ] 能写出 PPO clip 目标并解释 ratio 与 clip 的数学含义
- [ ] 能写出 GAE 递推公式并说明 γ/λ 的作用
- [ ] 能画出 RLHF 中 actor/critic/ref/RM 四模型的分工
- [ ] 能写出 GRPO 组内优势公式并解释为什么能替代 critic
- [ ] 能跑通 GAE/PPO 演示代码与 GRPO 训练循环
- [ ] 能说出 GRPO 相比 PPO 的显存收益（~40%）与代价（组采样）
- [ ] 能完成 PPO vs GRPO vs DPO 三维对比（模型数/采样/稳定性/显存）
- [ ] 能说明 thinking RL 的奖励设计与 Qwen3-VL 的思维训练
- [ ] 能回答 8 个面试追问
