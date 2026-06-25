import torch
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(weights=False)
# vgg16_true = torchvision.models.vgg16(pretrained=True)


trian_data = torchvision.datasets.CIFAR10("./dataset",  train=True, transform=torchvision.transforms.ToTensor())

vgg16_false.add_module("add_linear", nn.Linear(1000, 10))
vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))

print(vgg16_false)