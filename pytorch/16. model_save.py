import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights=False)

# 保存方式
# torch.save(vgg16, "vgg16_method1.pth")

torch.save(vgg16.state_dict(), "vgg16_method2.pth")
