import torch
from torch import nn
import math

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

if __name__ == "__main__":
    # 测试Embedding类
    vocab_size = 10000  # 假设词汇表大小为10000
    d_model = 512       # 假设模型的维度为512
    embeddings = Embeddings(d_model, vocab_size)
    
    # 创建一个示例输入，假设输入是一个包含词索引的张量
    input_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 示例输入，形状为(2, 3)
    
    # 获取对应的embedding输出
    output_embeddings = embeddings(input_indices)
    
    print("Input indices:\n", input_indices)
    print("Output embeddings shape:", output_embeddings.shape)  # 输出的形状应为(2, 3, 512)
    print("Output embeddings:\n", output_embeddings)
        