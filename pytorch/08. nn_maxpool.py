import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
print(input.shape)

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=False)
        
    def forward(self, x):
        x = self.maxpool1(x)
        return x

myNN = MyNN()
output = myNN(input)
print(output)


for data in dataloader:
    imgs, targets = data
    print(myNN(imgs))
    break
    

