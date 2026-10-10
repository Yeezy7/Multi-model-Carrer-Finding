import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Actor 输出动作分布的 logits；Critic 输出状态价值 V(s)。"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)  # 每个动作一个 logit
        self.critic = nn.Linear(64, 1)          # 状态价值 V(s)
        
    def forward(self, states):
        features = self.shared(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

def ppo_update(
    model, optimizer, states, actions, old_log_probs, advantages,
    returns, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01
):
    """
    用一批已经采集好的数据，执行一次 PPO 更新。

    states:       [batch, state_dim]
    actions:      [batch]，采样时实际执行的动作
    old_log_probs:[batch]，采样时旧策略给这些动作的 log 概率
    advantages:   [batch]，优势 A_t
    returns:      [batch]，Critic 要拟合的回报目标
    """
    
    # 当前策略重新计算同一批动作的概率
    logits, values = model(states)
    dist = Categorical(logits=logits)
    
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # r_t = 新策略概率 / 旧策略概率
    # log(a / b) = log(a) - log(b)，所以用 exp(log差)
    ratio = torch.exp(new_log_probs - old_log_probs)  # [batch]
    
    # PPO clipped surrogate objective
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages
    
    # 目标原本要最大化；PyTorch 优化器做最小化，所以取负号
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    
    # Critic 学习预测回报：V(s_t) ≈ return_t
    value_loss = ((values - returns) ** 2).mean()
    
    # 熵奖励鼓励策略保留一定探索性，因此 loss 中减去它
    loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }
    
    
# -------------------- 构造一小批示例数据 --------------------

torch.manual_seed(0)

state_dim = 4
action_dim = 2
batch_size = 8

model = ActorCritic(state_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)    

# 假设这是从环境中采集到的 8 个状态
states = torch.randn(batch_size, state_dim)
# 用更新前的策略采样动作，并记录旧策略概率
with torch.no_grad():
    old_logits, old_values = model(states)
    old_dist = Categorical(logits=old_logits)
    
    actions = old_dist.sample()  # [batch]
    old_log_probs = old_dist.log_prob(actions)  # [batch]

# 假设这些是采样后得到的回报目标
returns = torch.randn(batch_size) + 1.0

# 教学简化版优势：A_t ≈ 回报 - 旧 Critic 对状态价值的预测
advantages = returns - old_values

# 常见做法：标准化优势，让更新更稳定
advantages = (advantages - advantages.mean()) / (
    advantages.std() + 1e-8
)

# 用这批数据更新一次策略和价值网络
stats = ppo_update(
    model=model,
    optimizer=optimizer,
    states=states,
    actions=actions,
    old_log_probs=old_log_probs,
    advantages=advantages,
    returns=returns,
)

print(stats)
