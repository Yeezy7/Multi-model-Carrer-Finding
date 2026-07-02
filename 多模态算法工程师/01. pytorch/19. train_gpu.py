import torchvision
import torch
from torch import nn
from model import *


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


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print(f"测试数据集的长度为: {test_data_size}")

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# 创建网络模型
mynn = MyNN()
if torch.cuda.is_available():
    mynn = mynn.cuda()  

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2 # 0.01
optimizer = torch.optim.SGD(mynn.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print(f"----------------第{i+1}轮训练开始------------")

    # 训练步骤开始
    mynn.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = mynn(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}")    
    
    # 测试步骤开始
    mynn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = mynn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            
    print(f"整体测试集上的Loss: {total_test_loss}, 准确率total_accuracy: {total_accuracy / test_data_size}")
            
    
        