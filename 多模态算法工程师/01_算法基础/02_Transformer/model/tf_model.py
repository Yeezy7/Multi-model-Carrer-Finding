import torch
from torch import nn
import math
from torch.nn import functional as F
import copy
from tkinter import Variable

device = torch.device("mps")

"""
将输入的离散词索引转换为连续的向量表示
例如，将词汇表中的第5个词映射为一个512维的向量
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        # Embedding层，将词汇表的大小映射为d_model维的向量
        self.lut = nn.Embedding(vocab_size, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model
        
    def forward(self, x):
        # 返回x对应的embedding矩阵
        # 为了保证在后续的计算中，embedding的尺度与模型的维度相匹配，我们将embedding的输出乘以sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)


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
    
class  EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 多头自注意力机制
        self.feed_forward = feed_forward  # 前馈神经网络
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout)] for _ in range(2))  # 两个子层连接
        self.size = size  # 编码器层的大小
        
    def forward(self, x, mask):
        # 将embedding层进行Multi Head Attention计算，并进行残差连接和层归一化
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # 自注意力机制
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)  # 前馈神经网络

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个EncoderLayer层，组成一个完整的编码器
        self.layers = nn.ModuleList([layer for _ in range(N)])
        # Layer Norm
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """
        使用循环连续encode N次
        这里的Encoderlayer 会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 最后再进行一次Layer Norm

class DecodeLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecodeLayer, self).__init__()
        self.size = size
        # Self Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout)] for _ in range(3))  # 三个子层连接
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终 hidden表示结果
        m = memory
        
        # Self-Attention：注意self-attention的q、k和v都是来自于decoder的输入x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Encoder-Decoder Attention：注意这里的q来自于decoder的输入x，而k和v来自于encoder的输出m
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward) # 前馈神经网络
    
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个DecoderLayer层，组成一个完整的解码器
        self.layers = nn.ModuleList([layer for _ in range(N)])
        # Layer Norm
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)  # 最后再进行一次Layer Norm

class Generator(nn.Module):
    """
    Generator的作用是将Decoder的输出映射到词汇表大小的维度上，并进行softmax归一化，得到每个词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 线性层，将d_model维度映射到词汇表大小的维度
        
    def forward(self, x):
        # 将Decoder的输出x通过线性层映射到词汇表大小的维度，并进行softmax归一化
        return F.log_softmax(self.proj(x), dim=-1)  # 返回log softmax结果

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源语言的embedding层
        self.tgt_embed = tgt_embed  # 目标语言的embedding层
        self.generator = generator  # 输出生成器
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)  # 编码器处理源语言输入
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 解码器处理目标语言输入
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: 源语言输入序列
        tgt: 目标语言输入序列
        src_mask: 源语言输入序列的mask
        tgt_mask: 目标语言输入序列的mask
        """
        # 编码器将源语言输入序列编码为上下文表示
        memory = self.encode(src, src_mask)
        # 解码器将目标语言输入序列和编码器的上下文表示解码为输出序列
        return self.decode(memory, src_mask, tgt, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout)

    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecodeLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 使用Xavier均匀分布初始化参数
            
    return model.to(device)  # 将模型移动到指定设备
    
    