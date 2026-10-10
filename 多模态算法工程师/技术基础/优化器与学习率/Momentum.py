import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, gamma, steps = 0.05, 0.9, 100

# ---- 手写 Momentum ----
w = torch.tensor(1.5); b = torch.tensor(-0.8)
vw = vb = 0.0                  # 速度状态(对应 torch 的 momentum_buffer)
for t in range(steps):
    pred = w * x - b
    dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
    vw = gamma * vw + dw       # 速度 = 衰减的历史 + 当前梯度
    vb = gamma * vb + db
    w -= lr * vw                # 参数沿速度方向更新
    b -= lr * vb
print(f"手写 Momentum: w={w.item():.6f}  b={b.item():.6f}")


# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=lr, momentum=gamma)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch  Momentum: w={w.item():.6f}  b={b.item():.6f}")