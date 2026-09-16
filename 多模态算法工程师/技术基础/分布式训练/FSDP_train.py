import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import os


class TransformerBlock(nn.Module):
    """一个分片单元 = 自注意力 + FFN"""
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm(x + h)
        return self.norm(x + self.ffn(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(4)])
        self.head = nn.Linear(128, 1000)
    
    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)

def main():
    dist.init_process_group("nccl")  # GPU 环境用 nccl；CPU 演示可换 "gloo"
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )

    model = TinyGPT().cuda()
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(10):
        x = torch.randint(0, 1000, (4, 32)).cuda()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, 1000), x.view(-1))
        optimizer.zero_grad()
        loss.backward()   # FSDP 在反向内部自动完成 all-gather + reduce-scatter
        optimizer.step()  # 优化器只更新本 rank 持有的分片
        if rank == 0:
            print(f"step {step}: loss = {loss.item():.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # torchrun --nproc_per_node=2 --nnodes=1 FSDP_train.py