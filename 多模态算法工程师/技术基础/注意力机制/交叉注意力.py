import torch
import math
import torch.nn as nn


class CrossAttention(nn.Module):
    """交叉注意力：Q 来自 query 序列，K/V 来自 key 序列（维度可以不同）"""
    def __init__(self, d_query, d_key, h):
        super().__init__()
        self.h = h
        self.d_k = d_query // h
        self.W_q = nn.Linear(d_query, d_query)        # Q 投影：query 侧维度
        self.W_k = nn.Linear(d_key, d_query)          # K 投影：key 侧维度
        self.W_v = nn.Linear(d_key, d_query)          # V 投影：key 侧维度
        self.W_o = nn.Linear(d_query, d_query)
        
    def forward(self, query, key, value, mask=None):
        B, n_q, _ = query.shape
        n_k = key.shape[1]
        q = self.W_q(query).view(B, n_q, self.h, self.d_k).transpose(1, 2)  # [B, h, n_q, d_k]
        k = self.W_k(key).view(B, n_k, self.h, self.d_k).transpose(1, 2)    # [B, h, n_k, d_k]
        v = self.W_v(value).view(B, n_k, self.h, self.d_k).transpose(1, 2)  # [B, h, n_k, d_k]
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, h, n_q, n_k]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        out = torch.softmax(scores, dim=-1) @ v  # [B, h, n_q, d_k]
        
        out = out.transpose(1, 2).contiguous().view(B, n_q, -1)  # [B, n_q, d_query]
        return self.W_o(out)  
        