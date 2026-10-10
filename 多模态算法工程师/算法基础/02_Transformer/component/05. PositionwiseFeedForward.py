import torch
from torch import nn   
import math
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数：
            d_model: 模型的输入维度
            d_ff: 前馈神经网络的隐藏层维度
            dropout: dropout的概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将输入维度映射到隐藏层维度
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将隐藏层维度映射回输入维度
        self.dropout = nn.Dropout(dropout)  # dropout层，用于防止过拟合
    
    def forward(self, x):
        # 前向传播，先经过第一个线性层和ReLU激活函数，再经过dropout，最后经过第二个线性层
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  