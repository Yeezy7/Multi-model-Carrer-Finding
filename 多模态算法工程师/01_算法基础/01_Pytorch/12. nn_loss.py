import torch

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = torch.nn.L1Loss(reduction="sum")
results = loss(inputs, targets)

loss_mse = torch.nn.MSELoss()
result_mse = loss_mse(inputs, targets)


print(results)
print(result_mse)


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = torch.nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)