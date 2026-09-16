# 片段示例：核心配置参数逐个说明

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, CPUOffload, MixedPrecision, BackwardPrefetch
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={TransformerBlock},   # 每个 TransformerBlock 一个分片单元
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 参数/梯度/优化器状态全切 = ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),    # 参数冷落到 CPU 内存（训练变慢）
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,   # 参数以 BF16 存储/通信
        reduce_dtype=torch.float16,  # 梯度规约用 BF16
        buffer_dtype=torch.float16,  # buffer 用 BF16
    ),
    auto_wrap_policy=wrap_policy,  # 自动按 TransformerBlock 包装
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 提前 gather 下一层参数，通信与计算重叠
    device_id=rank,   # 显示绑定计算卡
)