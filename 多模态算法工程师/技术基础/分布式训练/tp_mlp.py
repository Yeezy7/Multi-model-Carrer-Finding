import torch
import torch.nn.functional as F
import torch.distributed as dist


class AllReduce(torch.autograd.Function):
    """可微 all-reduce：前向求和，反向时梯度再 all-reduce（通信算子是线性映射，梯度不变式）"""

    @staticmethod
    def forward(ctx, x):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        return g

def main(rank=0, world=2):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    bs, in_f, hidden, out_f, shard = 4, 8, 16, 8, 16 // world
    
    W1_full = torch.randn(hidden, in_f) * 0.1  # 全局权重 (仅 rank0 用于验证)
    W2_full = torch.randn(out_f, hidden) * 0.1
    # 各 rank 持有的分片：fc1 切输出维（列并行），fc2 切输入维（行并行）
    W1 = W1_full[rank * shard:(rank + 1) * shard].clone().requires_grad_()  # [shard, in_f]
    W2 = W2_full[:, rank * shard:(rank + 1) * shard].clone().requires_grad_()  # [out_f, shard]
    X = torch.randn(bs, in_f)  # 输入数据
    optimizer = torch.optim.SGD([W1, W2], lr=1e-1)
    
    for step in range(20):
        H = F.gelu(X @ W1.t())            # ① fc1 列并行：本地算，输出分片
        Y = AllReduce.apply(H @ W2.t())   # ② fc2 行并行：本地算 + all-reduce 得到全局输出

        loss = (Y ** 2).mean()
        loss.backward()         # all-reduce 梯度也走一遍 all-reduce
        optimizer.step()        
        optimizer.zero_grad()
        
        if step == 0 and rank == 0:
            W1_g, W2_g = W1_full.clone().requires_grad_(), W2_full.clone().requires_grad_()
            Y_ref = F.gelu(X @ W1_g.t()) @ W2_g.t()
            (Y_ref ** 2).mean().backward()
            assert torch.allclose(Y, Y_ref, atol=1e-6), "TP 输出与全量计算不一致"
            print("验证通过：列并行+行并行输出 == 全量计算")
    if rank == 0:
        print(f"训练完成 loss={loss.item():.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()