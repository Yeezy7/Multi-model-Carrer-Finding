from tkinter import Variable
import torch
from torch import nn
import math

device = torch.device("mps")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化一个size为 max_len x d_model 的位置编码矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=device)
        # 生成一个位置下标的tensor矩阵（每一行都是位置下标)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        
        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term) # 0::2 表示从第0列开始，每隔两列取一次
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 加一个维度，使得pe维度变为：1 * max_len * embedding维度
        # （方便后续与一个batch的句子所有词的embedding批量相加）
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下（不会作为要训练的参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embedding矩阵相加
        # （这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值）
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
        