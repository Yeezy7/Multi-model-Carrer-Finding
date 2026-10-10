import torch
from torch import nn   
import math
import torch.nn.functional as F

"""
层归一化（Layer Normalization）是一种归一化方法，它在每个样本的特征维度上进行归一化，而不是在批次维度上进行归一化。它的主要作用是加速训练过程，提高模型的稳定性。
在Transformer模型中，层归一化通常用于对每个子层的输出进行归一化，以便在后续的计算中保持数值的稳定性。
"""

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__(LayerNorm, self).__init__()
        # 初始化a为全1，而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps
        
    def forward(self, x):
        # 按最后一个维度计算均值和方差
        # keepdim=True确保输出的维度与输入相同
        mean = x.mean(-1, keepdim=True) # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True)   # 计算最后一个维度的标准差
        
        # 返回Layer Norm的结果
        # Layer Norm公式：y = a * (x - mean) / sqrt(std^2 + eps) + b  
        # 其中 a 和 b 是可学习的参数，eps是平滑项，防止除以0
        return self.a_2 * (x - mean) / torch.sqrt(std**2 + self.eps) + self.b_2