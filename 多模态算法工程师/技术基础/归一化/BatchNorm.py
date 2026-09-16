import torch
import torch.nn as nn


def batch_norm_forward_train(x, gamma, beta, eps=1e-5):
    # x: [N, C, H, W], 沿 N,H,W 求每个通道的统计量
    mean = x.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # [1, C, 1, 1]
    # 有偏方差（除 N·H·W）
    x_hat = (x - mean) / torch.sqrt(var + eps)  # 标准化
    y = gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)  # 缩放和平移
    return y, mean, var, x_hat


x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]])  # [2, 1, 2, 2]
gamma = torch.tensor([1.0]); beta = torch.tensor([0.0])
y, mean, var, x_hat = batch_norm_forward_train(x, gamma, beta)
print(mean.flatten())     # tensor([4.5])
print(var.flatten())      # tensor([5.25])
print(x_hat.flatten())    # tensor([-1.5275, -1.0911, -0.6547, -0.2182, 0.2182, 0.6547, 1.0911, 1.5275])
print(y.flatten())        # tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])



class BatchNorm1dManual(nn.Module):
    """手写 BN：完整复现 PyTorch 的 train/eval 两阶段行为"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(num_features))  # 平移参数
        self.register_buffer("running_mean", torch.zeros(num_features))  # 初始 0
        self.register_buffer("running_var", torch.ones(num_features))    # 初始 1
    
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)  # unbiased=False 有偏方差（除 N）
            with torch.no_grad():   # running stats 不走梯度
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                # 注意：PyTorch 用无偏方差更新 running_var
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * x.var(dim=0, unbiased=True))
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)  # unbiased=True 无偏方差（除 N-1）
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta
                

bn = BatchNorm1dManual(2)  # 
x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # N=3, C=2，[3, 2]
out1 = bn(x1)    # 训练模式
print(out1)      # tensor([[-1.2247, -1.2247], [0., 0.], [1.2247, 1.2247]])
print(bn.running_mean)    # tensor([0.3000, 0.4000])  ← 0.1×batch_mean
print(bn.running_var)     # tensor([1.3000, 1.3000])  ← 0.9×1 + 0.1×4（无偏方差）

x2 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
bn(x2)                    # 第二个 batch
print(bn.running_mean)    # tensor([1.1700, 1.3600])
print(bn.running_var)     # tensor([1.5700, 1.5700])


# ===== 关键演示：同样的输入，eval 模式输出完全不同 =====       
bn.eval()
out1_eval = bn(x1)
print(out1)               # tensor([[-1.2247, -1.2247], [0., 0.], [1.2247, 1.2247]])
print(out1_eval)          # tensor([[-0.1357,  0.5108], [1.4605, 2.1069], [3.0567, 3.7031]])