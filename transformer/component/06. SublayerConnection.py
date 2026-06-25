import torch
from torch import nn   
import math
import torch.nn.functional as F

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

class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention 和 Feed Forward层的输出进行残差连接和层归一化
    每一层输出之后都要先做残差连接，然后再做层归一化
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # 初始化LayerNorm层
        self.dropout = nn.Dropout(dropout)  # 初始化Dropout层
        
    def forward(self, x, sublayer):
        # 返回 残差连接 + 层归一化的结果
        # 其中sublayer是一个函数，表示当前子层的计算过程
        return x + self.dropout(sublayer(self.norm(x))) # 残差连接 + 层归一化