import torch
import torch.nn as nn
from torch.distributions import Categorical

class TinyPolicy(nn.Module):
    """极简策略语言模型：根据前文预测下一个 token。"""
    
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        # input_ids: [batch, sequence_length]
        x = self.embedding(input_ids)
        hidden_states, _ = self.gru(x)
        logits = self.lm_head(hidden_states)
        return logits  # [batch, sequence_length, vocab_size]

@torch.no_grad()
def sample_group(policy, prompts, group_size, response_length):
    """
    每个 prompt 采样 group_size 个回答。

    返回：
      responses:   [batch * group_size, response_length]
      old_log_probs: [batch * group_size, response_length]
    """
    batch_size = prompts.size(0)
    
    # 每个 prompt 复制 G 份，组内回答共享同一个 prompt
    repeated_prompts = prompts.repeat_interleave(group_size, dim=0)
    prefix = repeated_prompts
    
    sampled_tokens = []
    sampled_log_probs = []
    
    for _ in range(response_length):
        # 取最后一个位置的 logits 来预测下一个 token
        logits = policy(prefix)[:, -1, :]
        dist = Categorical(logits=logits)
        
        # 按当前旧策略采样 token，并记录采样概率
        token = dist.sample()
        log_prob = dist.log_prob(token)
        
        sampled_tokens.append(token)
        sampled_log_probs.append(log_prob)
        
        # 把刚生成的 token 接到上下文末尾，继续生成
        prefix = torch.cat([prefix, token[:, None]], dim=1)
    
    # 拼接所有采样的 token 和 log_prob
    responses = torch.stack(sampled_tokens, dim=1)  # [batch * group_size, response_length]
    old_log_probs = torch.stack(sampled_log_probs, dim=1)  # [batch * group_size, response_length]
    
    return responses, old_log_probs

def grpo_update(policy, ref_policy, optimizer, prompts, responses, old_log_probs, rewards, group_size,
                clip_epsilon=0.2, beta=0.05):
    """
    对采样到的一组回答执行一次 GRPO 策略更新（DeepSeek 原版，含 KL 项）。

    prompts:       [batch, prompt_length]
    responses:     [batch * group_size, response_length]
    old_log_probs: [batch * group_size, response_length]
    rewards:       [batch * group_size]
    ref_policy:    冻结的参考模型（SFT 产物），KL 锚定目标
    beta:          KL 惩罚系数，越大策略越难偏离参考模型
    """
    batch_size, prompt_length = prompts.size()
    response_length = responses.size(1)
    
    # 组内标准化奖励，作为每个回答的优势
    # 正优势：回答高于本组平均表现
    # 负优势：回答低于本组平均表现
    rewards_by_group = rewards.view(batch_size, group_size)
    
    group_mean = rewards_by_group.mean(dim=1, keepdim=True)                  # [batch, 1]
    group_std = rewards_by_group.std(dim=1, keepdim=True, unbiased=False)    # [batch, 1]
    advantages = (rewards_by_group - group_mean) / (group_std + 1e-4)        # [batch, group_size]
    advantages = advantages.reshape(-1, 1)
    
    # 简化处理：同一回答的优势广播到它生成的每个 token
    advantages = advantages.expand(-1, response_length)  # [batch * group_size, response_length]
    
    # 准备语言模型输入。
    # 输入为 prompt + response[:-1]，让模型预测 response 中的 token。
    repeated_prompts = prompts.repeat_interleave(group_size, dim=0)  # [batch * group_size, prompt_length]
    model_inputs = torch.cat([repeated_prompts, responses], dim=1)[:, :-1]
    
    all_logits = policy(model_inputs)

    # 第 prompt_length - 1 个位置开始，正好对应回答 token 的预测位置
    response_logits = all_logits[:, prompt_length - 1:, :]
    dist = Categorical(logits=response_logits)
    
    new_log_probs = dist.log_prob(responses)  # [batch * group_size, response_length]

    # 参考模型（冻结，不回传梯度）对同一批 token 的对数概率
    with torch.no_grad():
        ref_logits = ref_policy(model_inputs)[:, prompt_length - 1:, :]
        ref_log_probs = Categorical(logits=ref_logits).log_prob(responses)

    # k3 估计器：逐 token 估计 KL(π_θ ‖ π_ref)，恒非负且比直接相减（k1）方差更低
    log_ratio = ref_log_probs - new_log_probs
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1

    # 新旧策略对已采样 token 的概率比
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # PPO 风格的 clipping 目标
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    clipped = clipped_ratio * advantages
    
    # 目标是最大化 surrogate objective 并最小化 KL 惩罚；
    # PyTorch optimizer 执行最小化，因此 clip 项加负号，KL 项为正
    policy_loss = (-torch.minimum(unclipped, clipped) + beta * per_token_kl).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    return policy_loss.item(), advantages.detach(), per_token_kl.mean().item()


# ---------------- 教学用的小例子 ----------------

torch.manual_seed(0)

vocab_size = 12
batch_size = 2
group_size = 4
prompt_length = 3
response_length = 5

policy = TinyPolicy(vocab_size)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 参考模型：RL 开始前策略的快照（真实场景中就是 SFT 产物），全程冻结
ref_policy = TinyPolicy(vocab_size)
ref_policy.load_state_dict(policy.state_dict())
for p in ref_policy.parameters():
    p.requires_grad_(False)

# 两个 prompt，每个 prompt 会分别采样 4 个回答
prompts = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_length))

# 第一步：旧策略对每个 prompt 采样多个回答，并记录旧 log-prob
responses, old_log_probs = sample_group(
    policy=policy,
    prompts=prompts,
    group_size=group_size,
    response_length=response_length,
)

# 第二步：用一个简单的玩具规则给回答打分
# 这里把 token 3 当作“目标 token”，回答里出现得越多，奖励越高。
# 实际任务中，奖励通常来自规则评分器或奖励模型。
target_token_id = 3
rewards = (responses == target_token_id).float().mean(dim=1)

# 第三步：计算组内相对优势，并更新策略（KL 项锚定在参考模型上）
loss, advantages, kl = grpo_update(
    policy=policy,
    ref_policy=ref_policy,
    optimizer=optimizer,
    prompts=prompts,
    responses=responses,
    old_log_probs=old_log_probs,
    rewards=rewards,
    group_size=group_size,
)


print("采样回答：")
print(responses)
print("每条回答的奖励：")
print(rewards.view(batch_size, group_size))
print("组内标准化优势：")
print(advantages[:, 0].view(batch_size, group_size))
print("GRPO policy loss:", loss, "| 平均 KL:", kl)