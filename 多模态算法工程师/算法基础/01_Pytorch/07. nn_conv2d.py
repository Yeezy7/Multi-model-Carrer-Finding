import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    
    
    def forward(self, x):
        x = self.conv1(x)
        return x

myNN = MyNN()  


for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = myNN(imgs)
    # print(output)
    print(output.shape)
    break
        
