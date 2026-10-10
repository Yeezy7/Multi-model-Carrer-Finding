import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)

def run(mode, lr=0.2, steps=100):
    """mode: 'bgd'(全量) / 'mini'(8 样本) / 'sgd'(单样本)"""
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    traj = []
    for t in range(steps):
        if mode == "bgd":
            xb = x
        elif mode == "mini":
            xb = x[torch.randperm(64)[:8]]  # 随机取 8 个样本
        else:
            xb = x[torch.randint(0, 64, (1,))]  # 随机取 1 个样本
        pred = w * xb - b
        w -= lr * (2 * pred * xb).mean()
        b -= lr * (-2 * pred).mean()
        if (t + 1) % 20 == 0:
            traj.append(((w * x - b) ** 2).mean().item())
    return traj

for m in ["bgd", "mini", "sgd"]:
    print(f"{m}: " + "  ".join(f"{v:.2e}" for v in run(m)))
            