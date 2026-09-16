import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.02)   # 高斯初始化：保证能学
        self.B = nn.Parameter(torch.zeros(out_features, r))         # 零初始化：保证起点对
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        return self.dropout(x) @ self.A.t() @ self.B.t() * self.scaling
    
    def delta_weight(self):
        """返回 (d_out, d_in) 的更新量 (α/r)·BA"""
        return self.B @ self.A * self.scaling


# ---- 自检：形状与数学等价 ----
torch.manual_seed(0)
lora = LoRALayer(64, 128, r=8, alpha=16)
x = torch.randn(4, 16, 64)
y = lora(x)
print(y.shape)                  # torch.Size([4, 16, 128])
delta = lora.delta_weight()
print(delta.shape)              # torch.Size([128, 64])
print(torch.allclose(y, x @ delta.T))   # True：前向 == x @ ΔWᵀ



""" LoRALinear：把 LoRA 并接到 nn.Linear 上（含合并）"""
class LoRALinear(nn.Module):
    """原线性层（冻结）+ 可训练 LoRA 分支，支持合并回原权重"""
    
    def __init__(self, in_features, out_features, bias=True, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
        self._merged = False
        for p in self.base.parameters():
            p.requires_grad = False  # 冻结原线性层
        
    @classmethod
    def from_linear(cls, linear, **kwargs):
        """从已有 nn.Linear 构建并复制原权重（用于注入替换）"""
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, **kwargs)
        layer.base.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.base.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x):
        if self._merged:
            return self.base(x)
        return self.base(x) + self.lora(x)
    
    def merge(self):
        """把 LoRA 写回 base 权重（W' = W0 + (α/r)BA），返回纯 nn.Linear"""
        if not self._merged:
            self.base.weight.data.add_(self.lora.delta_weight())
            self._merged = True
        return self.base


def inject_lora(model, target=("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"), r=8, alpha=16, dropout=0.1):
    """递归地把名字命中 target 的 nn.Linear 替换为 LoRALinear，并冻结全部基座"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in target:
            setattr(model, name, LoRALinear.from_linear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora(child, target=target, r=r, alpha=alpha, dropout=dropout)


    # 冻结整个基座，只留 LoRA 分支可训练
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora.parameters():
                p.requires_grad = True
    return model


"""端到端训练 + 合并等价性验证"""
class TinyLM(nn.Module):
    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, 2*d)
        self.up_proj = nn.Linear(d, 2*d)
        self.down_proj = nn.Linear(2*d, d)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
    
    def forward(self, x=None, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed(x if x is not None else input_ids)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        h = h + self.o_proj(attn @ v)                              # attention 残差
        h = h + self.down_proj(torch.relu(self.up_proj(self.ln(h))))  # FFN 残差
        return self.head(h)
    
torch.manual_seed(0)
model = TinyLM()
inject_lora(model, r=8, alpha=16)                   # 注入 7 个线性层并冻结基座
total = sum(p.numel() for p in model.parameters())
lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"全模型 {total} 参数，LoRA 可训练 {lora_params}（{lora_params/total:.1%}）")
# 全模型 19104 参数，LoRA 可训练 4352（22.8%）
# 注：玩具模型占比高；真实 7B 模型全注入 r=8 时约 0.30%

# 训练 100 步（固定随机数据，仅验证"只动 LoRA 也能学"）
x = torch.randint(0, 64, (8, 16))
y = torch.randint(0, 64, (8, 16))
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-2)
criterion = nn.CrossEntropyLoss()
model.train()
for step in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x).reshape(-1, 64), y.reshape(-1))
    loss.backward()
    optimizer.step()
print(f"训练后 loss = {loss.item():.3f}")   # 从 ~4.43 降到 ~1.68（基座完全冻结）


# —— 合并等价性验证（必考）——
model.eval()
with torch.no_grad():
    out_before = model(x)                        # 双分支前向
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()                             # 合并回基座
    out_after = model(x)                         # 单分支前向
print("合并前后最大误差:", (out_before - out_after).abs().max().item())
# 8.6e-06：数学上精确等价，仅剩浮点加法顺序的舍入误差