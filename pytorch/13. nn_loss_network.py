import torch
from torch import nn
import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

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
   
loss = nn.CrossEntropyLoss()
    
mynn = MyNN()
for data in dataloader:
    imgs, targets = data
    outputs = mynn(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
    result_loss.backward()
    break