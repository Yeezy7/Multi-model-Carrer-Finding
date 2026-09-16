import math
import torch


def sdpa(q, k, v, attn_mask=None):
    """attn_mask: 与分数同形状的 bool 掩码，True 的位置保留，False 的位置置 -inf"""
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)   # [..., n_q, n_k]
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v


def sliding_mask(n, W):
    """causal + 窗口：query i 只保留 key j ∈ [i-W, i]"""
    i = torch.arange(n)
    diff = i[None, :] - i[:, None]     # j - i
    return (diff <= 0) & (diff >= -W)  # [n, n]，True 表示保留

def sliding_window_attention(x, W):
    """掩码实现：正确且直观，但掩码矩阵仍是 n×n（只用于验证/教学）"""
    B, n, d = x.shape
    return sdpa(x, x, x, attn_mask=sliding_mask(n, W).to(x.device))

def sliding_window_loop(x, W):
    """真正 O(nW)：每个 query 只取窗口内的 key，不构造 n×n 掩码"""
    B, n, d = x.shape
    out = torch.empty_like(x)
    for i in range(n):                                   # 串行演示；GPU 上可并行
        lo = max(0, i - W)
        k_i = x[:, lo:i+1]                               # [B, W+1, d]（首行更短）
        scores = (x[:, i:i+1] @ k_i.transpose(-2, -1)) / math.sqrt(d)
        out[:, i:i+1] = torch.softmax(scores, dim=-1) @ k_i
    return out


torch.manual_seed(0)
x = torch.randn(2, 64, 32)
y1 = sliding_window_attention(x, W=8)
y2 = sliding_window_loop(x, W=8)
print(torch.allclose(y1, y2, atol=1e-6))                 # True：两种实现一致


# Strided 与 Longformer（sliding + global）
def strided_attention(x, stride=16):
    """query i 只看 j ≡ i (mod stride) 且 j ≤ i 的 key"""
    n = x.shape[1]
    i = torch.arange(n)
    mask = ((i[:, None] - i[None, :]) % stride == 0) & (i[None, :] <= i[:, None])
    return sdpa(x, x, x, attn_mask=mask)

def longformer_attention(x, W, global_idx):
    """sliding + global：普通 query 看窗口+全部 global key；global query 看全部"""
    n = x.shape[1]
    i = torch.arange(n)
    window = ((i[:, None] - i[None, :]) >= -W) & (i[None, :] <= i[:, None])  # causal 窗口
    is_global = torch.zeros(n, dtype=torch.bool); is_global[global_idx] = True
    mask = window | is_global[None, :] | (is_global[:, None] & (i[None, :] <= i[:, None]))
    # ↑ 窗口边 + global key 所有人都能看 + global query 看全部（含 causal）
    return sdpa(x, x, x, attn_mask=mask)

x = torch.randn(1, 32, 16)
print(longformer_attention(x, W=4, global_idx=[0]).shape)          # [1, 32, 16]
print(sliding_window_attention(x, W=4).shape)                      # [1, 32, 16]
