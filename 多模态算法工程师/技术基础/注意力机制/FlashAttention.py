import torch


def flash_attention_sim(Q, K, V, B_c=64):
    """纯 torch 模拟 FlashAttention 的分块流程（v2 风格：最后统一归一化）。
    真实实现中每个块在 SRAM 内完成且不写回，这里用 torch 运算模拟同样的数学过程。
    """
    B, n, d = Q.shape
    O = torch.zeros_like(Q)     # 输出累加器 [B, n, d]

    
    l = torch.zeros(B, n, device=Q.device)  # running sum（分母）
    m = torch.full((B, n), float('-inf'), device=Q.device)  # running max（分子）
    
    for j in range(0, n, B_c):      # 沿列方向分块
        K_b = K[:, j:j+B_c]     # [B, B_c, d]
        V_b = V[:, j:j+B_c]     # [B, B_c, d]
        S = Q @ K_b.transpose(-2, -1) / (d ** 0.5)  # [B, n, B_c]
        
        m_new = torch.maximum(m, S.max(dim=-1).values)  # ① 更新 running max
        alpha = torch.exp(m - m_new)                    # ② 旧块缩放因子 ≤ 1
        P = torch.exp(S - m_new.unsqueeze(-1))          # 本块 exp(score - m_new)
        O = O * alpha.unsqueeze(-1) + P @ V_b           # ③ 累加输出（暂不归一化）
        l = l * alpha + P.sum(dim=-1)                   # ④ 累加分母 
        m = m_new
    return O / l.unsqueeze(-1)  # 最终归一化
        