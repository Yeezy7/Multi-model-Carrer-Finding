import torch
import torch.nn as nn

def layer_norm_manual(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)  # [N, C] -> [N, 1]
    var = x.var(dim=-1, unbiased=False, keepdim=True)  # [N, C] -> [N, 1]   有偏方差
    x_hat = (x - mean) / torch.sqrt(var + eps)  # 标准化
    return gamma * x_hat + beta

torch.manual_seed(0)
x = torch.randn(4, 6, 8)
y1 = layer_norm_manual(x, torch.ones(8), torch.zeros(8))     # [N, L, D]
y2 = nn.LayerNorm(8)(x)
print((y1 - y2).abs().max().item())     # 2.38e-07：完全对齐（默认 eps 同为 1e-5）

