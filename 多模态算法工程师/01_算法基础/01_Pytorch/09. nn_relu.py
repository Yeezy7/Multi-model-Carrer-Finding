import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class MyNN(nn.Module):
    def __init__(self):
        super( MyNN, self).__init__()
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(x)
        return x

mynn = MyNN()
output = mynn(input)
print(output)