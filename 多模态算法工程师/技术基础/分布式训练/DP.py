# 单机多卡 DDP 训练（线性回归玩具模型）
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

class ToyDataset(Dataset):
    """y = 2*x 各维之和 + 0.5 + 噪声"""

    def __init__(self, n=10000):
        torch.manual_seed(0)
        self.x = torch.randn(n, 16)
        self.y = (self.x * 2.0).sum(dim=1, keepdim=True) + 0.5
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class ToyModel(nn.Module):
    """线性回归模型"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def main():
    # ① 初始化进程组：nccl 是英伟达 GPU 集合通信库，所有分布式操作都建立在它之上
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程的 GPU id
    torch.cuda.set_device(local_rank)  # 设置当前进程的 GPU
    
    rank = dist.get_rank()  # # 全局进程编号 0 .. world_size-1
    world_size = dist.get_world_size()  # 总进程数
    
    # ② 数据分片：DistributedSampler 把数据集按 rank 无重叠切分
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=2)
    
    # ③ 模型：每卡一份完整副本，再交给 DDP 包装
    model = ToyModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        # ④ 每个 epoch 开头必须 sampler.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()  # 反向结束自动触发梯度 all-reduce
            optimizer.step() # 每卡用平均梯度各自更新，参数保持一致
        
        # ⑤ 只在 rank 0 打印 / 保存（否则每张卡各打一份、文件被反复覆盖）
        if rank == 0:
            torch.save(model.module.state_dict(), f"ddp_model.pt")

    dist.destroy_process_group()  # 训练结束销毁进程组

if __name__ == "__main__":
    main()

# 运行命令 ： torchrun --nproc_per_node=2 ddp_train.py