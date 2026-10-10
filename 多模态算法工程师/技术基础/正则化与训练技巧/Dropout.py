import torch
import torch.nn as nn


class DropoutFunction(torch.autograd.Function):
    """自定义 Dropout：mask 在前向采样并保存，反向按 mask 传播"""
    
    @staticmethod
    def forward(ctx, x, p=0.5):  # p 为保留概率
        # p 为保留概率；训练才采样 mask
        mask = (torch.rand_like(x) < p) / p   # 保留的乘 1/p（inverted）
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None       # 反向同样按 mask 走


x = torch.randn(2000, 4, reqquires_grad=True)
y = DropoutFunction.apply(x, 0.5)
print(y.shape)                                # torch.Size([2000, 4])
print((y != 0).float().mean().item())         # 约 0.5（保留比例 ≈ p）
y.sum().backward()
print(x.grad is not None)                     # True