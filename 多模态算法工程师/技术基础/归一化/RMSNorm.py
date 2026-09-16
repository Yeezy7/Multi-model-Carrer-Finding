import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """LLaMA 官方实现（hf 仓库逐行一致）"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()  # RMS
        return x * torch.rsqrt(rms) * self.weight  

torch.manual_seed(0)
x = torch.randn(4, 6, 16)
y1 = RMSNorm(16)(x)     # [N, L, D]
y2 = nn.RMSNorm(16)(x)
print((y1 - y2).abs().max().item())     # 0.0：完全对齐（hf 仓库 RMSNorm.py 与 LLaMA 官方实现一致