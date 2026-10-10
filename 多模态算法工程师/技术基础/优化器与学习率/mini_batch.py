"""用 torch.optim.SGD + autograd 实现 mini-batch 训练"""

import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64).reshape(-1, 1)   # (64, 1)

w = torch.tensor([[1.5]], requires_grad=True)  # (1, 1)
b = torch.tensor([-0.8], requires_grad=True)  # (1,)

opt = torch.optim.SGD([w, b], lr=0.05)
for epoch in range(5):
    perm = torch.randperm(64)  # 打乱索引
    for i in range(0, 64, 8):
        batch = x[perm[i:i+8]]   # 取 mini-batch
        opt.zero_grad()          # 梯度清零
        ((w * batch - b) ** 2).mean().backward()  # autograd 计算梯度
        opt.step()               # 更新参数
    full = ((w * x - b) ** 2).mean().item()    # 每轮算全量 loss 看真实进度
    print(f"epoch {epoch+1}  full-batch loss={full:.2e}  w={w.item():.4f}  b={b.item():.4f}")