"""BGD（Batch Gradient Descent）批量梯度下降 算法实现"""

import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 32)            # 训练数据: 32 个点

def loss_fn(w, b):
    return ((w * x - b) ** 2).mean()     # 损失函数 y=(w·x-b)²

w = torch.tensor(1.5)                  # 非零初始化, 让梯度下降"真正走下坡"
b = torch.tensor(-0.8)
lr = 0.05                              # 学习率
for t in range(80):
    pred = w * x - b
    dw = (2 * pred * x).mean()         # w 的梯度，对所有样本求均值
    db = (-2 * pred).mean()            # b 的梯度
    w -= lr * dw                       # 参数更新: θ ← θ - η·∇L
    b -= lr * db                       # 参数更新: θ ← θ - η·∇L
    if  (t + 1) % 10 == 0:
        print(f"step {t+1:3d}  loss={loss_fn(w,b).item():.2e}  w={w.item():.4f}  b={b.item():.4f}")
    
    