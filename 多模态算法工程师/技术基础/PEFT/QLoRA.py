import torch

"""手写 NF4 量化 / 反量化"""
# bitsandbytes 的 NF4 档位表（16 个值，[-1, 1]）
NF4_LEVELS = torch.tensor([
    -1.0000, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.18477343,
    -0.09105003, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0000,
])

def quantize_nf4(x, block_size=64):
    """分块 NF4 量化：每块一个 absmax 放缩因子，返回 4-bit 码 + 放缩因子"""
    n = x.numel()                   # 总元素数
    pad = (-n) % block_size         # 补齐到 block_size 的倍数
    xp = torch.cat([x.flatten(), torch.zeros(pad, dtype=x.dtype, device=x.device)])
    blocks = xp.view(-1, block_size)  # 分块
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)       # 每块放缩因子 s
    xn = (blocks / absmax[:, None]).clamp(-1.0, 1.0)         # 归一化到 [-1, 1]
    q = torch.abs(xn[:, :, None] - NF4_LEVELS).argmin(dim=-1).to(torch.uint8)  # 每块 64 个元素，4-bit 压缩成 32 个 uint8
    return q, absmax, n

def dequantize_nf4(q, absmax, n, block_size=64):
    """反量化：查档位表 × 放缩因子（每次前向计算前执行）"""
    levels = NF4_LEVELS.to(q.device)[q.long()].float()     # 4-bit 码查表
    out = (levels * absmax[:, None]).reshape(-1)
    return out[:n]          # 去掉补齐的部分


# 自检：量化-反量化闭环 
torch.manual_seed(0)
w = torch.randn(4096)
q, absmax, n = quantize_nf4(w)                            # 模拟一层权重（近似正态）
w_hat = dequantize_nf4(q, absmax, n)
rmse = ((w - w_hat) ** 2).mean().sqrt().item()
print(f"NF4 量化 RMSE = {rmse:.5f}")                       # ~0.091（标准差 1 的权重）




"""双重量化 + 比特账本"""
def quantize_8bit(x, block_size=256):
    """8-bit 分块量化：用于对放缩因子 s 做第二次量化"""
    n = x.numel()
    pad = (-n) % block_size
    xp = torch.cat([x.flatten(), torch.zeros(pad, dtype=x.dtype)])
    blocks = xp.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    xn = (blocks / absmax[:, None]).clamp(-1.0, 1.0)
    q = (xn * 127).round().to(torch.int8) 
    return q, absmax, n

def dequantize_8bit(q, absmax, n, block_size=256):
    return (q.float() / 127 * absmax[:, None]).reshape(-1)[:n]


torch.manual_seed(0)
w = torch.randn(1 << 20)                        # 100 万权重，避免 padding 干扰账本
q, absmax, n = quantize_nf4(w)                  # NF4 量化
q2, s2, n2 = quantize_8bit(absmax)              # 8-bit 量化放缩因子
absmax_hat = dequantize_8bit(q2, s2, n2)        # 反量化放缩因子
wq2 = dequantize_nf4(q, absmax_hat, n)          # 反量化权重
rmse = lambda a, b: ((a - b) ** 2).mean().sqrt().item()
print(f"双重量化 RMSE = {rmse(w, wq2):.5f}")    # ~0.092，与单层量化几乎无差
bits = (q.numel() * 4 + q2.numel() * 8 + s2.numel() * 32) / n
print(f"每权重比特 = {bits:.3f}")               # 4.127
print(f"7B 权重存储 = {bits * 7e9 / 8 / 1e9:.2f} GB")   # 3.61 GB



"""手写 QLoRA 训练：NF4 基座 + LoRA vs bf16 基座 + LoRA
"""
import torch.nn as nn
import torch

class NF4Linear(nn.Module):
    """以 NF4 存储的冻结线性层：前向时反量化回 bf16 计算"""
    
    def __init__(self, in_f, out_f, block_size=64):
        super().__init__()
        torch.manual_seed(0)
        w = torch.randn(out_f, in_f) * (in_f ** -0.5)  # 高斯初始化
        q, absmax, n = quantize_nf4(w.flatten(), block_size)  # NF4 量化
        self.register_buffer("q", q)              # 4-bit 码（存储大头）
        self.register_buffer("absmax", absmax)    # 放缩因子
        self.n = n
        self.block_size = block_size
        self.bias = nn.Parameter(torch.zeros(out_f))    # 偏置项仍然是 bf16 可训练
    
    def forward(self, x):
        w = dequantize_nf4(self.q, self.absmax, self.n, self.block_size).view(-1, x.shape[-1])
        return x @ w.t() + self.bias                # 反量化后照常做 bf16 矩阵乘
    
class QLoraToy(nn.Module):
    """玩具 QLoRA：NF4/bf16 基座 + LoRA 分支（d=32, r=8）"""
    
    def __init__(self, d, vocab, nf4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.base = NF4Linear(d, d) if nf4 else nn.Linear(d, d)  # 基座
        self.lora_A = nn.Parameter(torch.randn(8, d) * 0.02)   # LoRA：高斯  [8, d]
        self.lora_B = nn.Parameter(torch.zeros(d, 8))          # LoRA：零初始化 [d, 8]
        self.head = nn.Linear(d, vocab)
        
    def forward(self, x):
        h = self.embed(x)
        h = self.base(h) + (h @ self.lora_A.t()) @ self.lora_B.t()  # LoRA 分支
        return self.head(h)
    
def build(d, vocab, nf4):
    """构建 QLoRA 玩具模型"""
    m = QLoraToy(d, vocab, nf4)
    for p in m.parameters():
        p.requires_grad = False  # 冻结基座
    m.lora_A.requires_grad = m.lora_B.requires_grad = True  # LoRA 分支可训练
    m.head.weight.requires_grad = m.head.bias.requires_grad = True  # 输出层可训练
    return m

torch.manual_seed(0)
x = torch.randint(0, 64, (16, 10))
y = torch.randint(0, 64, (16, 10))
criterion = nn.CrossEntropyLoss()
for nf4, name in [(False, "bf16 基座 + LoRA"), (True, "NF4 基座 + LoRA")]:
    m = build(32, 64, nf4)
    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(m(x).reshape(-1, 64), y.reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"{name}: 100 步后 loss = {loss.item():.3f}")
# bf16 基座 + LoRA: 100 步后 loss = 1.532
# NF4 基座 + LoRA: 100 步后 loss = 1.380  （两者基本持平，量化没有拖累训练）
    
