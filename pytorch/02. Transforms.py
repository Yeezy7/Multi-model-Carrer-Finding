from torchvision import transforms
from PIL import Image


img = Image.open('pytorch/丁欣颖.png')
print(img)

# ToTensor() 会将PIL图像转换为Tensor，并且会将像素值从[0, 255]缩放到[0.0, 1.0]
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img) # 将PIL图像转换为Tensor
print(img_tensor)

# Normalize() 会对Tensor图像进行归一化处理，使用给定的均值和标准差对每个通道进行标准化
trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Resize() 将图像调整为指定的大小
print(img.size)
trans_resize = transforms.Resize((256, 256)) # 将图像调整为256x256
img_resized = trans_resize(img) # 调整图像大小
img_resized = trans_totensor(img_resized) # 将调整大小后的图像转换为Tensor
print(img_resized)

# Compose() 可以将多个变换组合在一起，按顺序执行
trans_compose = transforms.Compose([
    transforms.Resize((512, 512)), # 调整图像大小
    transforms.ToTensor(), # 将图像转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化处理
])

# RandomCrop() 会随机裁剪图像，指定裁剪的大小
trans_compose_2 = transforms.Compose([
    transforms.RandomCrop((224, 224)), # 随机裁剪图像为224x224
    transforms.ToTensor(), # 将图像转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化处理
])