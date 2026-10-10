import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB, CTX, LEN = 32, 4, 8     # 词表、提示长度、回答长度
DIM, HID = 16, 32              # 嵌入维度、GRU 隐层

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
    """整句对数概率：只对回答部分逐 token 求和"""
    seq = torch.cat([ctx, resp], dim=-1)               # (B, CTX+LEN)
    logits = model(seq)
    logp = F.log_softmax(logits, dim=-1)
    return logp[:, CTX:].gather(-1, seq[:, CTX:].unsqueeze(-1)).squeeze(-1).sum(-1)

def make_preference_data(n=48):
    """合成偏好数据（带噪声，模拟真实标注不完美）：
    chosen 有 70% 概率是'黄金回答'，rejected 有 70% 概率是随机串"""
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
    with torch.no_grad():
        ref_w = sequence_logprob(ref, ctx, chosen)     # log π_ref(y_w|x)
        ref_l = sequence_logprob(ref, ctx, rejected)   # log π_ref(y_l|x)
    
    # β[log(πθ(y_w)/πref(y_w)) - log(πθ(y_l)/πref(y_l))]
    implicit_diff = (logp_w - ref_w) - (logp_l - ref_l)
    return -F.logsigmoid(beta * implicit_diff).mean()

# ---------- 训练 ----------
policy = TinyLM()
ref = TinyLM()
ref.load_state_dict(policy.state_dict())   # 实际中 ref = SFT 好的模型，这里用随机初始化代替

for p in ref.parameters():
    p.requires_grad = False

opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
data = make_preference_data(48)

for step in range(300):
    ctx, chosen, rejected = data[step % len(data)]
    loss = dpo_loss(policy, ref, ctx.unsqueeze(0), chosen.unsqueeze(0), rejected.unsqueeze(0))
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # 监控：隐式奖励差 Δ = logπ(y_w) - logπ(y_l) 应持续上升
    with torch.no_grad():
        diff = (sequence_logprob(policy, ctx.unsqueeze(0), chosen.unsqueeze(0))
                - sequence_logprob(policy, ctx.unsqueeze(0), rejected.unsqueeze(0))).item()
    
    if step % 50 == 0:
        print(f"step {step:3d} | DPO loss = {loss.item():.4f} | Δlogπ = {diff:+.3f}")