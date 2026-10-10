import torch
import torchvision

model = torch.load("vgg16_method1.pth", weights_only=False)
# print(model)

# 方式2 加载模型
vgg16 = torchvision.models.vgg16(weighted=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth", weights_only=False))


print(model)