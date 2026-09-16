import torch
import torch.nn as nn
import math
import time


torch.manual_seed(0)

class HeadedAttention(nn.Module):
    """最小 MHA 自注意力层"""
    def __init__(self, d=64, n_heads=4, d_head=16):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d_head
        hdim = n_heads * d_head
        self.wq = nn.Linear(d, hdim, bias=False)
        self.wk = nn.Linear(d, hdim, bias=False)
        self.wv = nn.Linear(d, hdim, bias=False)
        self.wo = nn.Linear(hdim, d, bias=False)
    
    def project(self, x, w):
        """将输入 x 通过线性变换 w 投影到多头注意力的格式"""
        B, T, _ = x.shape
        return w(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d_head]
    
    def forward_full(self, x):
        """方式 A：无缓存，每步对完整前缀重算 K/V"""
        B, T, _ = x.shape
        q = self.project(x, self.wq)  # [B, h, T, d_head]
        k = self.project(x, self.wk)  # [B, h, T, d_head]
        v = self.project(x, self.wv)  # [B, h, T, d_head]
        att = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        att = att + torch.triu(torch.full_like(att, float('-inf')), diagonal=1)  # causal mask
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.wo(y)
    
    def decode_step(self, x_new, kv):
        """有缓存，decode 一步只算 1 个新 token 的 K/V"""
        k_hist, v_hist = kv
        q = self.project(x_new, self.wq)  # [B, h, 1, d_head]
        k = self.project(x_new, self.wk)  # [B, h, 1, d_head]
        v = self.project(x_new, self.wv)  # [B, h, 1, d_head]
        k = torch.cat([k_hist, k], dim=2)   #  新 K 拼进缓存
        v = torch.cat([v_hist, v], dim=2)
        attn = q @ k.transpose(-1, -2) / math.sqrt(self.d_head) # 1 个 query 查全部历史
        attn = torch.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(x_new.shape[0], 1, self.n_heads * self.d_head)
        return self.wo(y), (k, v)  # 输出 + 新缓存
    
    def make_cache(self, x):
        """初始化缓存 K/V"""
        return (self.project(x, self.wk), self.project(x, self.wv))  # [B, h, T, d_head]
    
if __name__ == "__main__":
    
    # 验证：无缓存 与 有缓存 数学等价
    model = HeadedAttention()
    d, nh, dh = 64, 4, 16
    P, N = 4, 4                                 # prompt 4 token，再生成 4 token
    tok = torch.randn(1, P + N, d)              # 8 个位置的隐状态序列
    prompt = tok[:, :P]

    with torch.no_grad():
        y_full = model.forward_full(tok)                     # 直接一次算 8 个位置
        y_prefill = model.forward_full(prompt)               # A) prefill：先算 prompt 4 个位置
        kv = model.make_cache(prompt)                        #    并缓存 K/V
        outs = []
        for t in range(P, P + N):                            # B) decode：逐 token 只算新 K/V
            y_step, kv = model.decode_step(tok[:, t:t + 1], kv)
            outs.append(y_step)
        y_cached = torch.cat([y_prefill, *outs], dim=1)

    print("无缓存输出形状:", tuple(y_full.shape), " 有缓存输出形状:", tuple(y_cached.shape))
    # 无缓存输出形状: (1, 8, 64)  有缓存输出形状: (1, 8, 64)
    print("逐位置最大误差:",
        [float((y_full[:, t] - y_cached[:, t]).abs().max()) for t in range(P + N)])
    # 逐位置最大误差: [2.24e-07, 1.60e-07, 1.04e-07, 9.69e-08, 1.19e-07, 7.64e-08, 5.22e-08, 6.71e-08]
    # （全部 ≤ 2.3e-07，纯浮点噪声 → 数学等价，零近似）