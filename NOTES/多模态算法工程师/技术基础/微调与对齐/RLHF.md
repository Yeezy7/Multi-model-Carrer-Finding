# RLHF（基于人类反馈的强化学习）

> 本模块索引见 [微调与对齐详解](微调与对齐详解.md)

## 一、定义与公式

RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）是让大模型对齐人类偏好的经典三阶段流水线：先用人类偏好数据训练一个**奖励模型（Reward Model, RM）**，再用强化学习（通常是 PPO）以 RM 打分为奖励信号优化策略模型。

### 1.1 三阶段流水线总览

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐
│ 阶段一：SFT  │   │ 阶段二：RM   │   │ 阶段三：RL（PPO 为主）     │
├─────────────┤   ├──────────────┤   ├──────────────────────────┤
│ 指令数据     │   │ 偏好数据对    │   │ 提示词 ──→ 策略采样回答     │
│ (x, y)      │   │ (x, y_w, y_l)│   │     ↓                    │
│ 交叉熵训练    │   │ Bradley-    │   │ RM 打分 − β·KL(π‖π_ref)   │
│ 得到 π_SFT   │   │ Terry 损失   │   │     ↓                    │
│             │   │ 得到 r_φ     │   │ PPO(clip) 更新 π_θ       │
└─────────────┘   └──────────────┘   └──────────────────────────┘
```

### 1.2 Bradley-Terry 偏好模型（RM 的数学基础）

对同一提示 $x$，人类在回答 $y_w$（胜者）与 $y_l$（败者）间更偏好 $y_w$。Bradley-Terry 模型假设：偏好概率由两个回答的奖励分数差决定，且服从 logistic 分布：

$$p(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

其中 $r_\phi$ 是奖励模型（只有分数差值有意义，整体平移不影响偏好）。

### 1.3 RM 损失函数（极大似然推导）

给定偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$，用极大似然估计奖励模型参数 $\phi$：

$$L_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

梯度推导：记 $d = r_w - r_l$，对 $-\log\sigma(d)$ 求导：

$$\frac{\partial}{\partial r_w}\big[-\log\sigma(d)\big] = -\sigma(-d), \qquad \frac{\partial}{\partial r_l}\big[-\log\sigma(d)\big] = +\sigma(-d)$$

**解读**：当模型已正确给出 $r_w \gg r_l$ 时 $\sigma(-d) \to 0$，梯度消失不再更新；当模型给反了（$r_w < r_l$）时 $\sigma(-d) \to 1$，梯度最大。这与二分类交叉熵完全同构——RM 本质上就是在学"哪个回答更好"这个二分类问题。

### 1.4 奖励 = RM 分数 − KL 惩罚

RL 阶段的目标是最大化带 KL 约束的期望奖励：

$$\max_{\theta}\ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)}\Big[\underbrace{r_\phi(x, y)}_{\text{RM 分数}} - \underbrace{\beta \log\frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)}}_{\text{KL 惩罚项}}\Big]$$

- $\pi_{ref}$：SFT 阶段产出的参考模型（**冻结不动**），是策略漂移的"锚点"；
- $\beta > 0$：KL 惩罚系数，控制"对齐程度 vs 保持能力"的平衡；
- KL 惩罚阻止策略为了刷分而忘掉语言能力（见 4.4 过优化）。

### 1.5 PPO 目标函数

RLHF 使用 PPO（Proximal Policy Optimization）优化上式，核心是 clipped 代理目标：

$$L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(\underbrace{r_t(\theta)}_{\text{ratio}} \hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat{A}_t\big)\Big] - c_1 L^V(\theta)$$

其中重要性采样比（importance ratio）：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

- 当 $r_t(\theta)$ 偏离 $[1-\varepsilon, 1+\varepsilon]$（通常 $\varepsilon = 0.2$）时梯度被截断 → **每次更新不会偏离旧策略太远**（软信任域）；
- $\hat{A}_t$ 是优势估计（GAE，见 PPO 与 GRPO 篇）：$\hat{A}_t = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$；
- $L^V(\theta)$ 是价值网络（critic）的 MSE 损失：$(V(s_t) - \hat{R}_t)^2$。

## 二、核心原理

### 2.1 阶段一：SFT（有监督微调）

用人工撰写的高质量指令-回答对 $(x, y)$ 做交叉熵微调，得到 $\pi_{SFT}$。它是后面所有阶段的"底座"：
- 没有 SFT，策略连"像人话"都做不到，RL 无法从噪声中学习；
- $\pi_{SFT}$ 同时充当 RL 阶段的**参考模型 $\pi_{ref}$** 和**初始化**。

### 2.2 阶段二：训练奖励模型 RM

**为什么 RM 需要偏好数据对，而不是单个绝对分数？**

| 原因 | 说明 |
| --- | --- |
| 相对判断更可靠 | 人类擅长说"A 比 B 好"，不擅长打 7.2 分；标注噪声大幅降低 |
| 分数不可比 | 不同标注者尺度不同（有人只给 1-4，有人给 6-10），绝对分数无法跨标注者比较；偏好对天然消除个人尺度偏差 |
| 信息等价 | 排序蕴含的信息量足够恢复奖励：$r$ 与 $r + c$ 给出相同偏好，所以只需学到排序一致的分数 |
| 数据易得 | 同一 prompt 用多个模型各生成一份回答，人工/规则排序即可批量构造 |
| 与任务同构 | "偏好即二分类"，损失简洁，训练稳定 |

训练完成后 RM 冻结。**注意 RM 是纯打分器，不参与生成**——它只在 RL 阶段当"裁判"。

### 2.3 阶段三：RL 优化（PPO 的角色）

四个模型分工：

| 模型 | 作用 | 训练状态 |
| --- | --- | --- |
| 策略 $\pi_\theta$（actor） | 负责生成回答，是被优化的对象 | 更新 |
| 价值 $V_\psi$（critic） | 估计状态价值，提供优势基线（降方差） | 更新 |
| 参考 $\pi_{ref}$ | KL 锚点，防止策略漂移 | 冻结 |
| 奖励 $r_\phi$（RM） | 给整句打分，提供训练信号 | 冻结 |

循环过程：采样一批提示 → 策略生成回答 → 计算奖励 $r_\phi(x,y) - \beta\log(\pi_\theta/\pi_{ref})$ → 用 critic 算 GAE 优势 → 按 clip 目标更新 actor、按 MSE 更新 critic。每一步的奖励里都带着 KL 惩罚，所以策略每走一步都在"提升奖励"和"不偏离参考"之间权衡。

### 2.4 KL 惩罚为什么必需

1. **防崩坏（reward hacking）**：没有 KL，策略很快发现刷 RM 分数的捷径（重复、语无伦次），输出质量断崖下跌；
2. **保持能力**：SFT 学到的语法、事实、多样性与对齐能力不可兼得时，KL 保住前者；
3. **数值上等价于正则化**：带 KL 的 RL 目标有闭式最优解 $\pi^*(y|x) \propto \pi_{ref}(y|x)\exp(r(x,y)/\beta)$——最优策略是参考策略按奖励做的吉布斯重加权，这是 DPO 推导的出发点（见 DPO 篇）。

## 三、源码实现

> 以下实现一个**可运行**的微型 RLHF：合成偏好对训练 RM → 用 RM 打分 + KL 惩罚做 PPO 更新。模型用 GRU 小语言模型（无 context 交叉编码，纯教学演示）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB, CTX, LEN = 32, 4, 8   # 词表、提示长度、回答长度
DIM, HID = 16, 32            # 嵌入维度、GRU 隐层

class TinyLM(nn.Module):
    """极简语言模型：Embedding + GRU + 输出头"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.gru = nn.GRU(DIM, HID, batch_first=True)
        self.head = nn.Linear(HID, VOCAB)

    def forward(self, seq):                    # seq: (B, T)
        h, _ = self.gru(self.emb(seq))         # (B, T, HID)
        return self.head(h)                    # (B, T, VOCAB)

def sequence_logprob(model, ctx, resp):
    """整句对数概率：拼接(提示+回答)过模型，只对回答部分逐 token 求和"""
    seq = torch.cat([ctx, resp], dim=-1)               # (B, CTX+LEN)
    logits = model(seq)                                # (B, CTX+LEN, V)
    logp = F.log_softmax(logits, dim=-1)
    resp_tokens = seq[:, CTX:].unsqueeze(-1)           # (B, LEN, 1)
    return logp[:, CTX:].gather(-1, resp_tokens).squeeze(-1).sum(-1)  # (B,)

@torch.no_grad()
def sample_responses(policy, ctx, n=4, max_len=LEN):
    """自回归采样 n 个回答：ctx 是 (CTX,) 单条提示"""
    seq = ctx.unsqueeze(0).expand(n, -1).clone()       # (n, CTX)
    for _ in range(max_len):
        logits = policy(seq)[:, -1]                    # (n, V) 最后位置
        token = torch.multinomial(logits.softmax(-1), 1)   # 多项式采样
        seq = torch.cat([seq, token], dim=-1)
    return seq[:, CTX:]                                # (n, LEN)

class RewardModel(nn.Module):
    """奖励模型：读完整序列，输出一个标量分数"""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.gru = nn.GRU(DIM, HID, batch_first=True)
        self.head = nn.Linear(HID, 1)

    def forward(self, seq):                            # (B, T) -> (B,)
        h, _ = self.gru(self.emb(seq))
        return self.head(h[:, -1]).squeeze(-1)

# ---------- 阶段二：训练奖励模型 ----------
def make_preference_data(n=48):
    """合成偏好数据（带噪声，模拟真实标注不完美）：
    chosen 有 70% 概率是'黄金回答'，rejected 有 70% 概率是随机串"""
    gold = torch.randint(1, VOCAB, (LEN,))
    data = []
    for _ in range(n):
        ctx = torch.randint(1, VOCAB, (CTX,))
        chosen = gold.clone() if torch.rand(()) < 0.7 else torch.randint(1, VOCAB, (LEN,))
        rejected = torch.randint(1, VOCAB, (LEN,)) if torch.rand(()) < 0.7 else gold.clone()
        data.append((ctx, chosen, rejected))
    return data

def rm_loss(rm, ctx, chosen, rejected):
    """Bradley-Terry 损失：-log σ(r_w - r_l)"""
    rw = rm(torch.cat([ctx, chosen], dim=-1))
    rl = rm(torch.cat([ctx, rejected], dim=-1))
    return -F.logsigmoid(rw - rl).mean()

data = make_preference_data(48)
rm = RewardModel()
rm_opt = torch.optim.Adam(rm.parameters(), lr=1e-2)

for step in range(120):
    ctx, chosen, rejected = data[step % len(data)]
    loss = rm_loss(rm, ctx.unsqueeze(0), chosen.unsqueeze(0), rejected.unsqueeze(0))
    rm_opt.zero_grad(); loss.backward(); rm_opt.step()
    if step % 20 == 0:
        print(f"RM step {step:3d} | loss = {loss.item():.4f}")
# RM step   0 | loss = 0.7344
# RM step  20 | loss = 0.2507
# RM step  40 | loss = 0.1761
# RM step  60 | loss = 0.0324
# RM step  80 | loss = 0.0265
# RM step 100 | loss = 0.0203

# ---------- 阶段一简版：SFT 热身（让策略见过"黄金回答"，RL 才有东西学） ----------
policy = TinyLM()
pol_opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
for step in range(12):
    ctx, chosen, _ = data[step % len(data)]
    loss = -sequence_logprob(policy, ctx.unsqueeze(0), chosen.unsqueeze(0)).mean()
    pol_opt.zero_grad(); loss.backward(); pol_opt.step()

# 参考模型 = SFT 产物（冻结），策略继续从 SFT 权重出发
ref = TinyLM()
ref.load_state_dict(policy.state_dict())
for p in ref.parameters():
    p.requires_grad = False

# ---------- 阶段三：RL 优化（PPO clip） ----------
def rl_update(policy, ref, rm, ctx, opt, beta=0.1, clip_eps=0.2, n=4, epochs=3):
    """一轮 PPO：采样 -> 奖励 = RM - β·KL -> clip 目标多轮小步更新"""
    resp = sample_responses(policy, ctx, n)                # (n, LEN)
    ctxs = ctx.unsqueeze(0).expand(n, -1)
    seq = torch.cat([ctxs, resp], dim=-1)                  # (n, CTX+LEN)

    with torch.no_grad():
        old_logp = sequence_logprob(policy, ctxs, resp)
        r = rm(seq) * 2.0                                  # RM 分数（放大便于区分）
        kl = sequence_logprob(ref, ctxs, resp) - old_logp  # KL(π‖π_ref) 逐样本
        reward = r - beta * kl                             # 总奖励 = RM - β·KL
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)  # 简单归一化优势

    for _ in range(epochs):
        logp = sequence_logprob(policy, ctxs, resp)
        ratio = torch.exp(logp - old_logp)                 # 重要性采样比
        pg1 = ratio * adv
        pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss = -torch.min(pg1, pg2).mean()                 # PPO clip 目标
        opt.zero_grad(); loss.backward(); opt.step()

for it in range(30):
    rl_update(policy, ref, rm, data[0][0], pol_opt)
    if it % 10 == 0:
        resp = sample_responses(policy, data[0][0], 1)[0]
        with torch.no_grad():
            kl = (sequence_logprob(policy, data[0][0].unsqueeze(0), resp.unsqueeze(0))
                  - sequence_logprob(ref, data[0][0].unsqueeze(0), resp.unsqueeze(0))).item()
        print(f"RL it {it:3d} | KL(policy‖ref) = {kl:7.3f}")
# RL it   0 | KL(policy‖ref) =   2.004
# RL it  10 | KL(policy‖ref) =   8.937
# RL it  20 | KL(policy‖ref) =   9.172
```

**运行观察**：RM 损失单调下降（学会区分好坏）；RL 阶段策略逐渐偏离参考模型（KL 上升），同时往 RM 高分区域移动——这就是 RLHF 在玩具尺度上的完整闭环。真实场景把 `TinyLM` 换成 7B/72B 模型、数据换成人工偏好对即可。

## 四、深入分析

### 4.1 训练不稳定的四大来源

| 来源 | 机理 | 缓解手段 |
| --- | --- | --- |
| RM 误差 | RM 是近似人类偏好的**代理**，打分有噪声；策略会钻 RM 的漏洞 | KL 惩罚、RM 集成、偏好数据质量 |
| 采样方差 | 策略自回归采样，梯度跨整句传播，方差大 | GAE、组内基线（GRPO 思路）、大批次 |
| 奖励尺度漂移 | 模型升级后 RM 分布变化，旧 RM 失效 | 定期重新采样偏好数据重训 RM |
| 梯度爆炸/消失 | 长序列 + 连续多轮 PPO 内更新 | 梯度裁剪、clip 超参、预热 |

### 4.2 显存占用（为什么 RLHF 贵）

同时驻留 4 份模型：actor + critic + ref + RM（每份含参数/梯度/优化器状态）。7B 规模下每份模型裸显存约 14GB（FP16）+ 优化器状态（Adam 约 8× 参数字节数），**总显存通常是 SFT 的 3-4 倍**。缓解：
- **LoRA/QLoRA**：只训低秩适配器，其余冻结（生产标配）；
- 共享底层权重：actor/ref 参数共享（仅需额外存 ref 的推理开销）；
- critic 单独上小模型（如 7B 配 1B critic）。

### 4.3 Reward Hacking（奖励黑客）

策略发现"刷分捷径"：重复刷关键词、结构模板化、谄媚、回避问题。本质是**优化器比奖励设计者聪明**。缓解：KL 惩罚、奖励归一化/校准、人工抽查、规则惩罚项、训练中用真实偏好对 RM 做持续校验。

### 4.4 过优化（Overoptimization）

KL 惩罚系数 $\beta$ 与最终效果的经典 tradeoff：$\beta$ 太小 → KL 大、策略走偏、真实人类评估分数先升后降（**过优化曲线**：RM 分数持续上升但人类满意度在某一轮后下滑）；$\beta$ 太大 → 策略几乎不动，对齐效果差。实践中用一个**独立的留出偏好集**（不被 RM 训练看到）监控人类/真实指标，找"早停点"。

### 4.5 多模态 RLHF

图文场景下 RLHF 的差异：

- **偏好对象**：对同一张图 + 提示，比较多个候选回答（描述准确性、物体数量、空间关系、颜色等）；
- **偏好数据构造流程**：① 收集图文指令（如"描述图中第三个人穿什么颜色的衣服"）；② 用多个模型各生成回答；③ 人工（或强多模态模型当裁判，即 RLAIF）排序得到 $(x, image, y_w, y_l)$；
- **RM 输入**：奖励模型须能编码图像，通常用冻结视觉塔 + 可训的跨模态融合层打分；
- **代表性工作**：LLaVA-RLHF（LLaVA 上的 RLHF，减少幻觉）、RLHF-V（引入细粒度图像偏好——对"图中有几只猫"这类事实性错误按**局部区域**给偏好，降低幻觉率）、VLFeedback（开源图文偏好数据集）；
- **已知难点**：图像维度奖励信号稀疏（大部分 token 与图无关），纯文本 RM 无法覆盖；常需把事实正确性（如 VQA 精确匹配）做成规则奖励与 RM 混合使用。

## 五、优缺点

| 优点 | 缺点 |
| --- | --- |
| 显式建模人类偏好，奖励信号可解释、可扩展 | 流水线复杂：SFT → RM → RL 三段，工程成本高 |
| RM 可复用：一次训练服务所有策略 | 三阶段误差累积，RM 偏差被策略放大 |
| PPO 理论上限高，适合任务导向优化 | 训练不稳定，超参敏感（β、ε、GAE λ） |
| 可混合规则奖励（正确性）+ 偏好奖励 | 显存开销大（4 个模型驻留） |
| 与 RL 研究生态打通（GAE、trust region） | 样本效率低，采样开销大 |

## 六、与同类对比

| 维度 | RLHF（PPO） | DPO | GRPO |
| --- | --- | --- | --- |
| 需要 RM | 是 | **否**（偏好对直接监督策略） | 是（常用规则/可验证奖励） |
| 需要 critic | 是 | 否 | 否（组内归一化替代） |
| 在线采样 | 是 | 否（离线偏好数据） | 是（组内采样） |
| 稳定性 | 较差，需大量调参 | 好，训练类 SFT | 较好（无 critic 方差） |
| 显存 | 4 模型，最贵 | 2 模型（θ + ref），最省 | 3 模型（θ + ref + rm） |
| 适用场景 | 通用对齐、奖励可扩展 | 偏好数据丰富、追求简单稳定 | 推理任务（答案可验证）、DeepSeek 系 |

**何时选谁**：偏好数据足且想最简稳定 → DPO；任务是数学/代码等可验证答案的推理强化 → GRPO（+规则奖励）；通用助手对齐、想灵活注入多信号 → RLHF（PPO）。三者的详细对比见 [DPO](DPO.md) 与 [PPO与GRPO](PPO与GRPO.md) 两篇。

## 七、高频面试问答

**Q1：为什么 RLHF 需要三个阶段？SFT 不也用了人工数据吗？**
SFT 学的是"模仿答案"（教师强制，与生成分布不一致），而 RLHF 的目标是优化偏好。RM 把模糊的"人类偏好"变成可微的标量信号，PPO 让策略在**自己生成的数据**上优化——这是 SFT 做不到的（暴露偏差）。

**Q2：RM 为什么用偏好对而不是绝对分数？**
相对判断标注噪声小、跨标注者可比；绝对分数受个人尺度影响且与排序信息等价（奖励差决定偏好，整体平移无意义）。用偏好对还能直接套 Bradley-Terry 二分类损失，训练稳定。

**Q3：KL 惩罚项具体防什么？β 大了会怎样？**
防 reward hacking 与能力遗忘。β 太大策略几乎不更新，对齐效果差；β 太小策略漂移、真实人类满意度先升后降（过优化）。

**Q4：RLHF 里 PPO 的 clip 机制作用是什么？**
限制每步更新不偏离旧策略太远（软信任域）：ratio 超出 $[1-\varepsilon,1+\varepsilon]$ 时梯度被截断，避免一次更新破坏已学到的语言能力，是 RLHF 稳定的关键之一。

**Q5：RLHF 为什么显存开销巨大？怎么缓解？**
actor/critic/ref/RM 四份模型常驻 + 优化器状态，是 SFT 的 3-4 倍。缓解：LoRA/QLoRA 冻结主干、共享 actor 与 ref 权重、critic 用更小模型。

**Q6：reward hacking 是什么？举例。**
策略发现 RM 打分漏洞并以牺牲真实质量的方式刷分，如重复输出高分关键词、模板化废话。缓解：KL 惩罚、奖励校准、规则惩罚、独立留出集监控。

**Q7：多模态 RLHF 和纯文本 RLHF 的差别？**
RM 需编码图像（视觉塔 + 融合层）；偏好对基于图文指令构造，重点对齐细节（数量、空间、颜色）；信号稀疏，常叠加规则奖励（VQA 正确性），代表工作 LLaVA-RLHF、RLHF-V。

**Q8：为什么最终评估要独立于 RM 训练数据？**
RM 有系统偏差（偏好数据标注噪声 + 模型近似误差），策略会钻 RM 漏洞。独立留出偏好集上的真实指标能发现过优化拐点，决定早停。

## 八、自我检验

- [ ] 能画出 RLHF 三阶段流程图并说出每个阶段的输入输出
- [ ] 能手写 Bradley-Terry 公式并推导 RM 的 log-sigmoid 损失
- [ ] 能说清"为什么 RM 需要偏好对"（至少 3 个理由）
- [ ] 能写出总奖励 = RM − β·KL 的公式并解释每一项
- [ ] 能写出 PPO clip 目标并解释 ratio 与 clip 的作用
- [ ] 能跑通本篇 RLHF toy 代码并解释 KL 上升的含义
- [ ] 能说出 RLHF 四大难点（不稳定/显存/reward hacking/过优化）及缓解手段
- [ ] 能说明多模态偏好数据如何构造、RM 如何编码图像
- [ ] 能回答 8 个面试追问
