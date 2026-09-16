import math
import torch
import torch.nn as nn


def sdpa(q, k, v, mask=None):
    """标准点积注意力（Scaled Dot-Product Attention）"""
    d = q.shape[-1]
    score = q @ k.transpose(-2, -1) / math.sqrt(d)  # [B, n, n]
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    p = torch.softmax(score, dim=-1)
    return p @ v


class MHA(nn.Module):
    """多头注意力（Multi-Head Attention）
       每个 Q 头配独立 K/V 投影
    """
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)  # [B, h, n, d_k]
        k = self.W_k(x).view(B, n, self.h, self.d_k).transpose(1, 2)  # [B, h, n, d_k]
        v = self.W_v(x).view(B, n, self.h, self.d_k).transpose(1, 2)  # [B, h, n, d_k]
        out = sdpa(q, k, v, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, n, -1))  # [B, n, d_model]

class MQA(nn.Module):
    """Multi-Query Attention（多查询注意力）
       所有 Q 头共享 K/V 投影
    """
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)     # 只投影 1 份 KV
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)  # [B, h, n, d_k]
        k = self.W_k(x).unsqueeze(1)  # [B, 1, n, d_k]
        v = self.W_v(x).unsqueeze(1)  # [B, 1, n, d_k]
        out = sdpa(q, k, v, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, n, -1))  # [B, n, d_model]
    
class GQA(nn.Module):
    """Grouped-Query Attention（分组查询注意力）
        将 Q 头分组，每组共享 K/V 投影
    """
    def __init__(self, d_model, h, g):
        super().__init__()
        assert h % g == 0, "组数必须整除头数"
        self.h, self.g, self.d_k = h, g, d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k * g)  # 每组 KV 投影 1 份
        self.W_v = nn.Linear(d_model, self.d_k * g)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, n, _ = x.shape
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)  # [B, h, n, d_k]
        k = self.W_k(x).view(B, n, self.g, self.d_k).transpose(1, 2)  # [B, g, n, d_k]
        v = self.W_v(x).view(B, n, self.g, self.d_k).transpose(1, 2)  # [B, g, n, d_k]
        k = k.repeat_interleave(self.h // self.g, dim=1)  # 同组 Q 头复制同一份 KV
        v = v.repeat_interleave(self.h // self.g, dim=1)  # → [B, h, n, d_k]
        out = sdpa(q, k, v, mask)
        return self.W_o(out.view(B, self.h, n, self.d_k).transpose(1, 2).contiguous().view(B, n, -1))  # [B, n, d_model]