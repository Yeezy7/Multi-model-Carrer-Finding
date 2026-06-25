import torch
from torch import nn   
import math
import torch.nn.functional as F

"""
注意力机制 
"""
def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换（转置），才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 如果存在要进行mask的内容，则将那些为0的部分替换一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)
    
    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # 得到head的数量
        self.h = h
        # 定义4个线性层，供后续作为WQ、WK、WV矩阵 和 最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # 对mask进行扩展，增加一个维度，以便后续广播
        # query的第一个维度值为batch_size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵，得到query、key、value矩阵
        # 并将结果拆成h个块，然后将第二个和第三个维度值互换
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个注意力矩阵concat起来，并将第二个和第三个维度值互换回来，然后再乘以最后一个线性层，得到最终的输出
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # contiguous()的作用是将tensor在内存中变为连续的，方便后续的view操作
        x = self.linears[-1](x)
        return x