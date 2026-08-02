import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=False, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=False, transform=dataset_transform)

print(train_set[0])

# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()