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
        self.emb = nn.Embedding(VOCAB, DIM)  # 嵌入层: 将词表索引映射到嵌入向量
        self.gru = nn.GRU(DIM, HID, batch_first=True)  # gru: RNN 神经网络
        self.head = nn.Linear(HID, VOCAB)  # 输出头: 将 GRU 的输出映射到词表大小
        
    def forward(self, seq):                    # seq: (B, T)
        h, _ = self.gru(self.emb(seq))         # (B, T, HID)
        return self.head(h)

def sequence_logprob(model, ctx, resp):
    """SFT损失 log(sigmoid)，整句对数概率：拼接(提示+回答)过模型，只对回答部分逐 token 求和"""
    seq = torch.cat([ctx, resp], dim=-1)               # (B, CTX+LEN)
    logits = model(seq)                                # (B, CTX+LEN, V)
    logp = F.log_softmax(logits, dim=-1)
    pred = logp[:, CTX-1:-1]                     # (B, LEN, 1)
    tgt  = resp.unsqueeze(- 1 )
    return -pred.gather(- 1 , tgt).squeeze(- 1 ). sum (- 1 )   # (B,) 负对数似然

@torch.no_grad()
def sample_responses(policy, ctx, n=4, max_len=LEN):
    """推理时输出结果：自回归采样 n 个回答：ctx 是 (CTX,) 单条提示"""
    seq = ctx.unsqueeze(0).expand(n, -1).clone()       # (n, CTX)
    for _ in range(max_len):
        logits = policy(seq)[:, -1]                    # (n, V) 最后位置
        token = torch.multinomial(logits.softmax(-1), 1)  # 多项式采样
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
    rm_opt.zero_grad()
    loss.backward()
    rm_opt.step()
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
    pol_opt.zero_grad()
    loss.backward()
    pol_opt.step()
    

# 参考模型 = SFT 产物（冻结），策略继续从 SFT 权重出发
ref = TinyLM()
ref.load_state_dict(policy.state_dict())
for p in ref.parameters():
    p.requires_grad = False


# ---------- 阶段三：RL 优化（PPO clip） ----------
def rl_update(policy, ref, rm, ctx, opt, beta=0.1, clip_eps=0.2, n=4, epochs=3):
    """一轮 PPO：采样 -> 奖励 = RM - β·KL -> clip 目标多轮小步更新"""
    resp = sample_responses(policy, ctx, n)
    ctxs = ctx.unsqueeze(0).expand(n, -1)
    seq = torch.cat([ctxs, resp], dim=-1)
    
    with torch.no_grad():
        old_logp = sequence_logprob(policy, ctxs, resp)  # (n,)
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
        opt.zero_grad()
        loss.backward()
        opt.step()
        
for it in range(30):
    rl_update(policy, ref, rm, data[0][0], pol_opt)
    if it % 10 == 0:
        resp = sample_responses(policy, data[0][0], 1)[0]
        with torch.no_grad():
            kl = (sequence_logprob(policy, data[0][0].unsqueeze(0), resp.unsqueeze(0)) -
                  sequence_logprob(ref, data[0][0].unsqueeze(0), resp.unsqueeze(0))).item()
            r = rm(torch.cat([data[0][0].unsqueeze(0), resp.unsqueeze(0)], dim=-1))
        print(f"RL it {it:3d} | KL(policy‖ref) = {kl:7.3f}")
   
# RL it   0 | KL(policy‖ref) =   2.004
# RL it  10 | KL(policy‖ref) =   8.937
# RL it  20 | KL(policy‖ref) =   9.172

    