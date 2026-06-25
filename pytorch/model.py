from torch import nn
import torch

# 搭建神经网络
class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    mynn = MyNN()
    input = torch.ones((64, 3, 32, 32))
    output = mynn(input)
    print(output.shape)
