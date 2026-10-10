"""Nesterov 加速梯度（NAG）"""
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, gamma, steps = 0.05, 0.9, 100

# ---- 手写 Nesterov (PyTorch 形式: 更新量 = 当前梯度 + γ·速度) ----
w = torch.tensor(1.5); b = torch.tensor(-0.8)
vw = vb = 0.0
for t in range(steps):
    pred = w * x - b
    dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
    vw = gamma * vw + dw             # 先照常累积速度
    vb = gamma * vb + db
    w -= lr * (gamma * vw + dw)      # 更新 = g + γ·v (前瞻修正)
    b -= lr * (gamma * vb + db)
print(f"手写 Nesterov: w={w.item():.6f}  b={b.item():.6f}")


# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=lr, momentum=0.9, nesterov=True)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch  Nesterov: w={w.item():.6f}  b={b.item():.6f}")
    
