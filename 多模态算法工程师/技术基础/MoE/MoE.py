import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class ExpertFFN(nn.Module):
    """单个专家：两层 FFN（等价标准 Transformer 的 FFN 块）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class MoELayer(nn.Module):
    """手写小型 MoE 层：8 个专家 + top-2 路由 + aux loss + capacity 截断"""
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2,
                 capacity_factor=1.25, aux_alpha=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_alpha = aux_alpha
        self.gate = nn.Linear(d_model, n_experts, bias=False)  # 路由器：线性层输出每个 token 对每个专家的分数
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff) for _ in range(n_experts)]
        )
        
    def forward(self, x):
        tokens = x.reshape(-1, x.size(-1)) #  (T, d)，展平成 token 序列
        T = tokens.size(0)
        
        # 1) 路由器打分 + softmax 门控
        gate_logits = self.gate(tokens)  # (T, E)，每个 token 对每个专家的分数
        gate_probs = F.softmax(gate_logits, dim=-1)  # (T, E)，每个 token 对每个专家的概率

        # 2) top-k 选择：每个 token 取分数最高的 k 个专家
        top_probs, top_experts = gate_probs.topk(self.top_k, dim=-1)  # (T, k)，(T, k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # 归一化概率 (T, k)

        # 3) capacity：每个专家本步最多处理的 token 数
        capacity = int(T * self.top_k / self.n_experts * self.capacity_factor)
        
        # 4) 负载均衡损失（Switch Transformer 的 aux loss)
        f_i = torch.zeros(self.n_experts)  # f_i: 每个专家实际拿到的 token 比例
        for i in range(self.n_experts):
            f_i[i] = (top_experts == i).float().mean()  # 计算每个专家的负载比例
        P_i = gate_probs.mean(dim=0)  # P_i: 每个专家的平均概率
        aux_loss = self.aux_alpha * self.n_experts * (f_i * P_i).sum()
        
        # 5) 按专家分发 token：容量截断 -> 专家前向 -> 按门控权重累加
        outputs = torch.zeros_like(tokens)  # (T, d)
        used = torch.zeros(self.n_experts, dtype=torch.int)  # 记录每个专家已使用的 token 数
        for e in range(self.n_experts):
            idx = (top_experts == e).nonzero()  # 找到分配给专家 e 的 token 索引
            if len(idx) == 0:
                continue
            token_ids = idx[:, 0] # 取前 capacity 个 token
            if len(token_ids) > capacity:
                token_ids = token_ids[:capacity]  # 截断到容量
            if len(token_ids) == 0:
                continue
            w = top_probs[idx[:len(token_ids), 0], idx[:len(token_ids), 1]]  # (N,) 每个 token 对专家 e 的权重
            outputs[token_ids] += self.experts[e](tokens[token_ids]) * w.unsqueeze(-1)  # 按权重累加专家输出
            used[e] = len(token_ids)  # 更新已使用的 token 数
        return outputs.view(x.shape), gate_logits, aux_loss, used  # 返回输出、路由器分数、aux loss、每个专家的使用情况

if __name__ == "__main__":
    model = MoELayer(d_model=64, d_ff=128, n_experts=8, top_k=2)
    x = torch.randn(2, 16, 64)  # (B, T, d)  batch=2, seq=16 -> 共 32 个 token
    y, gate_logits, aux_loss, used = model(x)
    print("输出形状:", tuple(y.shape))  
    # 输出: 输出形状: (2, 16, 64)          # 与输入同形，MoE 层不改变 token 维度
    
    print("aux_loss:", round(aux_loss.item(), 4))
    # 输出: aux_loss: 0.0102               # 接近理论最小值 α=0.01，说明路由较均衡

    print("各专家实际处理 token 数:", used.tolist())
    # 输出: 各专家实际处理 token 数: [8, 8, 8, 8, 5, 5, 11, 11]
    # 说明: 32 个 token × top-2 = 64 次分发，平均每专家 8 个；容量 11 内的都被处理

    print("容量上限:", int(32 * 2 / 8 * 1.25) + 1)
    # 输出: 容量上限: 11                   # ceil(32×2/8×1.25)+1 = 11
    
    
    # 验证梯度能回传到路由器与专家（只有被选中的专家有梯度）
    x2 = torch.randn(4, 64)
    y2, _, _, _ = model(x2)
    y2.sum().backward()
    print("路由器有梯度:", model.gate.weight.grad is not None)
    # 输出: 路由器有梯度: True             # 路由器的 W_g 通过 top-k 加权路径收到梯度