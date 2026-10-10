import torch
import torchvision
from torch import nn

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

mynn = MyNN()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = mynn(output)
    print(output.shape)
    break
