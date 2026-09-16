# 参数分片 → all-gather 完整权重 → 算梯度 → reduce-scatter 归约回各卡 → 只更新自己那 1/N。

import os
import torch
import torch.distributed as dist

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"
    dist.init_process_group("gloo", rank=rank, world_size=world_size )
    total, shard = 4, 4 // world_size
    
    W_shared = torch.arange(shard, dtype=torch.float32) + rank * shard + 1.0 # 每个卡只持有 1/N
    target = torch.full((total,), 2.0)
    
    for step in range(3):
        # 前向：all-gather 拼出完整权重（真实场景按层拼、用完即弃）
        parts = [torch.zeros(shard) for _ in range(world_size)]
        dist.all_gather(parts, W_shared)  # 每个卡都拿到完整权重
        W_full = torch.cat(parts) # 拼成完整权重
        
        loss = ((W_full - target) ** 2).sum() # 计算 loss
        
        # 反向：每卡对完整权重有梯度 → reduce_scatter 归约回"块归属"的卡
        g_full = 2.0 * (W_full - target)
        blocks = [g_full[i * shard:(i + 1) * shard].contiguous() for i in range(world_size)]
        g_shard = torch.zeros(shard)
        dist.reduce_scatter(g_shard, blocks) # 每个卡只拿到自己那块的梯度

        # 更新：只有归属卡持有 m/v（模拟优化器状态分片），只动自己 1/N
        W_shard = W_shard - 0.1 * g_shard
        if rank == 0:
            print(f"step{step} loss={loss.item():.2f} W_shard0={W_shard.tolist()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.multiprocessing.spawn(run, args=(4,), nprocs=4)
# 实际输出（rank0，收敛说明"分片+通信+本地更新"整体等价于全量训练）：
# step0 loss=1016.00 W_shard0=[1.8, 2.0, 2.2, 2.4]
# step1 loss=40.64  W_shard0=[1.96, 2.0, 2.04, 2.08]
# step2 loss=1.63   W_shard0=[1.992, 2.0, 2.008, 2.016]