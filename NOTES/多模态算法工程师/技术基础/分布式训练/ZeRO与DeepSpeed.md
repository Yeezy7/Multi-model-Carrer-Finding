# ZeRO 与 DeepSpeed

> 本模块索引见 [分布式训练详解](分布式训练详解.md)

## 一、定义与公式

### 1.1 动机：DDP 的"每卡三份冗余"

数据并行（DDP）的规则是"每张卡都放一份完整模型"。训练 7B 模型（BF16）时，每张卡要同时为三类数据腾出显存：

- **权重（weights）**：7B × 2B ≈ 14GB（取整口径记 16GB）；
- **梯度（gradients）**：与权重同尺寸，≈ 16GB；
- **优化器状态（optimizer states）**：Adam 保存一阶矩 $m$、二阶矩 $v$（FP32），混合精度还常保留 FP32 主权重——量级约为权重的 **3 倍**，≈ 48GB。

$$\text{每卡显存(全量)} = W + G + O = 16 + 16 + 48 = 80\text{GB（不含激活）}$$

8 卡 DDP = **8 份权重、8 份梯度、8 份优化器状态**，全部冗余——而训练结束后每卡权重是一样的，梯度归约后也一样。**冗余是显存的死敌，ZeRO 的存在意义就是消灭冗余。**

### 1.2 ZeRO 定义

**ZeRO（Zero Redundancy Optimizer）**，微软 DeepSpeed 提出的数据并行增强方案。核心思想：**三份状态从"每卡全量"改为"全体均分、按需通信"**，N 卡时每卡只放 $1/N$：

$$\text{ZeRO 后每卡显存} = \frac{W + G + O}{N} + \text{激活（未分片部分）} + \text{通信 buffer}$$

它仍是数据并行（每卡算不同的 batch、跑同一模型），但**显存随卡数线性下降**——这是 DDP 永远做不到的。

## 二、核心原理

### 2.1 ZeRO 三阶段切分表

| 阶段 | 切分对象 | 每卡保留 | 通信成本 | 典型场景 |
| --- | --- | --- | --- | --- |
| ZeRO-1 | 仅优化器状态 | 权重+梯度全量，优化器状态 $O/N$ | ≈ DDP（几乎无额外通信） | 显存瓶颈在优化器 |
| ZeRO-2 | 优化器状态 + 梯度 | 权重全量，梯度 $G/N$、优化器 $O/N$ | 梯度多一次 reduce-scatter（与 all-reduce 融合，成本≈DDP） | 大 batch、激活占比高时 |
| ZeRO-3 | 优化器 + 梯度 + 权重 | 全部 $1/N$ | 权重前向/反向各一次 all-gather + 梯度 reduce-scatter ≈ 1.5×DDP | 单卡装不下模型本身（>80GB） |

从 ZeRO-1 到 ZeRO-3 是"递进式"：每多切一份状态，显存再省一块，但通信多一次。工程上 stage 从低到高平滑迁移。

### 2.2 7B 显存数学账本（N=8）

取整口径：$W=16$GB，$G=16$GB，$O=48$GB，合计 80GB。

| 方案 | 公式 | N=8 每卡（不含激活） |
| --- | --- | --- |
| DDP | $W + G + O$ | 80 GB |
| ZeRO-1 | $W + G + O/N$ | 16+16+6 = **38 GB** |
| ZeRO-2 | $W + (G + O)/N$ | 16+8 = **24 GB** |
| ZeRO-3 | $(W + G + O)/N$ | **10 GB** |

精确版（7B×2B=14GB，Adam FP32 的 m+v=56GB，含 FP32 主权重共 84GB 优化器状态）：同样按 $1/N$ 缩放，结论不变——**省下来的量级是"十几个 GB × 卡数"**。训练 13B/70B 模型时，ZeRO-3 是唯一让单卡能装下的方案（70B：DDP 约 800GB/卡 vs ZeRO-3 N=64 时约 12.5GB/卡）。

> 激活（activation）不在切分之列（ZeRO-3 另有 partition-activation 选项）。7B 长序列训练激活可达 30~60GB，配合激活重计算（activation checkpointing）进一步压低。

### 2.3 ZeRO-3 的前向/反向数据流

以 Transformer 一层的参数 $W$ 为例（权重被切成 N 份，分属 N 卡）：

```text
前向：  每卡 all_gather(自己那 1/N 份) → 拼出完整 W → 该层前向 → 立刻释放 W
         （其他层的参数仍处于分片状态，不占显存）

反向：  backprop 到该层时再次 all_gather(W) 计算梯度
         → 每卡算出完整梯度 g
         → reduce_scatter(g)：把属于卡 k 的那块梯度归约到卡 k
         → 释放完整 W 与 g

更新：  只有持有 W 第 k 块的卡，在其上做 Adam 更新
         （优化器状态 m_k、v_k 也只存在于卡 k —— 这就是优化器状态被"分片"的本质）
```

一轮训练里，每个参数经历 **前向 all-gather + 反向 all-gather + 反向 reduce-scatter 共 3 次全局通信**，这就是 ZeRO-3 比 DDP 慢 30%~50% 的根源（详见 4.1）。

## 三、源码实现

### 3.1 显存账本计算器（纯 Python，可直接运行）

```python
def memory_ledger(w_gb, n, activation_gb=0.0):
    """显存账本：权重 W(GB, BF16)；优化器状态按 3x 权重；梯度按 1x 权重"""
    o_gb = 3 * w_gb
    g_gb = w_gb
    rows = [
        ("DDP(全量)",        w_gb + o_gb + g_gb),
        (f"ZeRO-1 (N={n})",  o_gb / n + w_gb + g_gb),
        (f"ZeRO-2 (N={n})",  (o_gb + g_gb) / n + w_gb),
        (f"ZeRO-3 (N={n})",  (o_gb + g_gb + w_gb) / n),
    ]
    for name, v in rows:
        print(f"{name:16s} 小计 = {v:6.1f} GB, 加激活后 = {v + activation_gb:6.1f} GB")
    return rows


if __name__ == "__main__":
    print("=== 7B 模型 (BF16, 权重取整 16GB), N=8 ===")
    memory_ledger(w_gb=16, n=8)
    print("\n=== 7B 模型, N=16, 激活 10GB ===")
    memory_ledger(w_gb=16, n=16, activation_gb=10)
# 实际输出：
# === 7B 模型 (BF16, 权重取整 16GB), N=8 ===
# DDP(全量)        小计 =   80.0 GB, 加激活后 =   80.0 GB
# ZeRO-1 (N=8)    小计 =   38.0 GB, 加激活后 =   38.0 GB
# ZeRO-2 (N=8)    小计 =   24.0 GB, 加激活后 =   24.0 GB
# ZeRO-3 (N=8)    小计 =   10.0 GB, 加激活后 =   10.0 GB
# === 7B 模型, N=16, 激活 10GB ===
# DDP(全量)        小计 =   80.0 GB, 加激活后 =   90.0 GB
# ZeRO-1 (N=16)   小计 =   35.0 GB, 加激活后 =   45.0 GB
# ZeRO-2 (N=16)   小计 =   20.0 GB, 加激活后 =   30.0 GB
# ZeRO-3 (N=16)   小计 =    5.0 GB, 加激活后 =   15.0 GB
```

### 3.2 手工 ZeRO-3 微缩模拟（torch.distributed + gloo，可直接运行）

用集合通信原语复现 2.3 的数据流：参数分片 → all-gather 完整权重 → 算梯度 → reduce-scatter 归约回各卡 → 只更新自己那 1/N。

```python
import os
import torch
import torch.distributed as dist


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29520"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    total, shard = 16, 16 // world_size

    W_shard = torch.arange(shard, dtype=torch.float32) + rank * shard + 1.0  # 每卡只持有 1/N
    target = torch.full((total,), 2.0)

    for step in range(3):
        # 前向：all-gather 拼出完整权重（真实场景按层拼、用完即弃）
        parts = [torch.zeros(shard) for _ in range(world_size)]
        dist.all_gather(parts, W_shard)
        W_full = torch.cat(parts)
        loss = ((W_full - target) ** 2).sum()

        # 反向：每卡对完整权重有梯度 → reduce_scatter 归约回"块归属"的卡
        g_full = 2.0 * (W_full - target)
        blocks = [g_full[i * shard:(i + 1) * shard].contiguous() for i in range(world_size)]
        g_shard = torch.zeros(shard)
        dist.reduce_scatter(g_shard, blocks, op=dist.ReduceOp.SUM)

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
```

> 真实 ZeRO-3 的前向/反向在每层内按需进行（`stage3_param_persistence_threshold` 控制大参数全量保留），本模拟只展示通信骨架，loss 在每卡冗余计算（真实实现每卡只算 1/N 的前向）。

### 3.3 DeepSpeed 工程：initialize + JSON 配置（需 `pip install deepspeed`）

```python
import deepspeed
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim=64, hidden=256, out=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


model = TinyMLP()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",          # 配置即代码：ZeRO 方案全在 JSON 里
)

for step in range(100):
    x, y = torch.randn(8, 64), torch.randint(0, 10, (8,))
    loss = torch.nn.functional.cross_entropy(model_engine(x), y)
    model_engine.backward(loss)       # DeepSpeed 接管梯度归约（reduce-scatter 等）
    model_engine.step()               # 优化器状态已按 stage 切分/卸载
```

配套 `ds_config.json`（ZeRO-3 + CPU offload 示例）：

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000
  },
  "optimizer": { "type": "AdamW", "params": { "lr": 3e-5 } },
  "bf16": { "enabled": true }
}
```

启动（两种等价方式）：

```bash
deepspeed --num_gpus 8 train.py            # 方式一：DeepSpeed 启动器
torchrun --nproc_per_node=8 train.py       # 方式二：torchrun 启动器（DeepSpeed 兼容）
```

### 3.4 与 HuggingFace Trainer 集成（改 JSON + 一行参数）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

args = TrainingArguments(
    output_dir="./qwen2-7b-zero3",
    deepspeed="ds_config.json",             # 传入 3.3 的同一个 JSON 即可
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
    logging_steps=10,
)
trainer = Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset)
trainer.train()
# 预期输出（示意）：启动日志中出现 "deepspeed info: stage=3  offload_optimizer=cpu ..."
# 训练循环中每 10 步打印 loss 逐步下降
```

> Trainer 集成要点：`TrainingArguments` 里设 `deepspeed` 路径即可，ZeRO 方案的梯度分桶、混合精度、offload 全部由 JSON 驱动；零代码改动。

### 3.5 FSDP 对照（PyTorch 原生，可直接运行）

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29521"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                      device_id=torch.device("cpu"))   # GPU 环境去掉 device_id 即可
    opt = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)

    for step in range(3):
        loss = fsdp_model(torch.randn(8, 8)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if rank == 0:
            print(f"step{step} loss={loss.item():.2f}")
    print(f"[rank{rank}] 分片后可见参数量: {sum(p.numel() for p in fsdp_model.parameters())}")
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(run, args=(4,), nprocs=4)
# 实际输出（4 进程）：step0 loss=-4.03  step1 loss=-74.91  step2 loss=-434.49
# [rank0..3] 分片后可见参数量: 53   （全量 8*16+16 + 16*4+4 = 212 参数，212/4 = 53 → 权重确实被切成 4 份）
```

## 四、深入分析

### 4.1 通信开销：ZeRO-3 vs DDP

设模型大小 $M$ 字节（7B BF16：$M$ = 14GB），每步训练：

| 方案 | 每步通信内容 | 通信量 |
| --- | --- | --- |
| DDP | 1 次梯度 all-reduce（ring 算法 ≈ reduce-scatter + all-gather 融合） | $2M$ ≈ 28GB |
| ZeRO-1/2 | 梯度 all-reduce 同上（ZeRO-2 的 reduce-scatter 融合在其中） | ≈ $2M$ |
| ZeRO-3 | 前向参数 all-gather $M$ + 反向参数 all-gather $M$ + 梯度 reduce-scatter $M$ | $3M$ ≈ 42GB |

$$\text{ZeRO-3 通信量} = \frac{3M}{2M} = 1.5 \times \text{DDP}$$

两点关键结论：

1. **通信量不随卡数 N 爆炸**：all-gather/reduce-scatter 都采用带宽最优的 ring 算法，每卡只收发自己缺的 $(N-1)/N$ 部分，总时间与 N 近似无关（与集合通信篇的推导一致）；
2. **多出的 1 倍 $M$ 来自"参数权重两次 all-gather"**：前向一次、反向一次。DeepSpeed 用 `overlap_comm`（通信与计算重叠）、`stage3_prefetch_bucket_size`（预取下一层参数）把可见延迟压到最低，但仍比 DDP 慢 30%~50%。ZeRO-1/2 无此代价，因此**能不用 stage-3 就不用**。

### 4.2 梯度分桶与流水线重叠

ZeRO-2/3 的梯度归约同样做 bucket 化（`reduce_bucket_size` 控制桶大小）：反向传播算出一部分梯度就立刻 reduce-scatter，不等整个模型反完，让"计算下一层梯度"与"通信归约已就绪桶"并行执行。这正是 3.3 里 `contiguous_gradients: true` 的意义（把梯度搬到连续 buffer，减少碎片化拷贝）。

### 4.3 Offload：显存与速度的权衡

**ZeRO-Offload**：把优化器状态（甚至权重）移到 CPU 内存或 NVMe SSD。

- 通信路径：GPU HBM ↔ CPU 内存（PCIe，~32GB/s 单向）或 NVMe（~3~7GB/s）；
- 对比 GPU HBM 带宽（~3TB/s）：**差 100 倍量级**，所以 offload 的每一步更新都要搬数据，训练显著变慢（典型 2~10 倍）；
- **什么时候值得用**：
  1. 单卡（或 2 卡）跑大模型，没有更多 GPU 可加——offload 是"能跑 vs 跑不动"的区别；
  2. CPU 内存大（几百 GB）而显存只有 24~80GB；
  3. 以吞吐量换取可行性：先 offload 跑通，再谈优化；
  4. **不值得**：GPU 充足且能用 ZeRO-1/2 解决时（offload 的通信开销远大于省显存收益）；追求训练速度的场景。
- 工程事实：`offload_optimizer` 成本低（每步一次状态搬运），`offload_param` 成本高（每次前向/反向都要搬权重）；NVMe offload 通常只用于"存储 checkpoint/极低频访问"。

### 4.4 与其他省显存手段的协同

| 手段 | 省的是什么 | 与 ZeRO 关系 |
| --- | --- | --- |
| ZeRO-1/2/3 | 权重/梯度/优化器状态 | 本文主角 |
| 激活重计算（activation checkpointing） | 前向激活 | 正交，可叠加；ZeRO-3+重计算是 70B 单卡训练标配 |
| 混合精度（BF16/FP16） | 权重/梯度各减半 | 与 ZeRO 完全正交，DeepSpeed 默认开启 |
| 梯度累积 | 不省显存（省的是吞吐优化） | 常与 ZeRO 配合使用 |
| 模型并行（Megatron-TP） | 每层参数切分到多卡 | 与 ZeRO 可组合（DeepSpeed 3D 并行），但 ZeRO 只解决数据并行侧 |

## 五、优缺点

| 优点 | 缺点 |
| --- | --- |
| 显存随 N 线性下降，能训练单卡装不下的模型 | ZeRO-3 通信量 1.5×DDP，吞吐下降 30%~50% |
| 仍是数据并行：代码侵入小（JSON 配置即可） | 参数分片导致每步多次 all-gather，延迟更高 |
| 与 DDP/重计算/混合精度正交可叠加 | 调试难（`to('cuda')`/原地修改参数等操作需注意分片语义） |
| 从 stage-1 到 stage-3 平滑升级 | CPU/NVMe offload 速度损失大，只适合显存极端受限 |
| DeepSpeed + HF Trainer 集成成熟 | 生态绑定 DeepSpeed 框架，PyTorch 原生替代是 FSDP |

## 六、与同类对比

### 6.1 ZeRO vs FSDP：同一思想的两个实现

FSDP（Fully Sharded Data Parallel，PyTorch 2.x 原生）本质就是"PyTorch 版的 ZeRO-3"（可降级为 ZeRO-2 语义，`SHARD_GRAD_OP`）。

| 维度 | DeepSpeed ZeRO | PyTorch FSDP |
| --- | --- | --- |
| 分片粒度 | 参数按**层/整段**切分 | 每个参数独立切分 + `FlatParameter` 展平（优化通信），可自定义 wrapping 粒度 |
| 配置方式 | JSON 配置（`zero_optimization.stage`） | Python API（`ShardingStrategy`、`auto_wrap_policy`） |
| 附加能力 | offload（CPU/NVMe）、3D 并行、英伟达生态深度调优 | 原生集成 autograd/检查点、`device_mesh` 数据并行 |
| 生态 | DeepSpeed 社区、HF Trainer 一等公民 | PyTorch 官方、HuggingFace 也支持 |
| 适用 | 生产大集群、多框架混用 | 想少一个框架依赖的团队 |

### 6.2 DDP / ZeRO / Megatron-TP 三者关系

| | DDP | ZeRO-1/2/3 | Megatron 张量并行 |
| --- | --- | --- | --- |
| 并行维度 | 数据 | 数据（显存分片） | 模型（每层切块到多卡） |
| 每卡是否持有完整权重 | 是 | ZeRO-3 否 | 否（每层只持一块） |
| 通信 | 每步 1 次 all-reduce | 1.5×all-reduce 量（Z3） | 每层 2 次 all-reduce（前向/反向各 1） |
| 显存节省 | 无 | 线性 | 线性 |
| 组合 | 基线 | 与 TP 组合 = 3D 并行（DeepSpeed） | 与 ZeRO 组合 |

经验法则：**能 DDP 就 DDP；显存差一点上 ZeRO-1/2；装不下模型上 ZeRO-3；超大模型（100B+）再叠加 Megatron-TP/流水线并行**。

## 七、高频面试问答

**Q1：ZeRO 和 DDP 的区别？**
DDP 每卡持有完整的权重+梯度+优化器状态，一次 all-reduce 同步梯度，显存不随卡数下降；ZeRO 把三份状态切分到 N 卡（stage-3 全切），显存随 N 线性下降，代价是权重前向/反向各多一次 all-gather，通信量约为 1.5×DDP。

**Q2：ZeRO-1/2/3 各切什么？各能省多少？**
ZeRO-1 只切优化器状态（$O/N$）；ZeRO-2 切优化器状态+梯度（$(O+G)/N$）；ZeRO-3 三者全切（$(O+G+W)/N$）。以 7B 取整账本（16/16/48GB）N=8 计：80 → 38 → 24 → 10GB。

**Q3：为什么 ZeRO 能省显存？**
因为 DDP 的权重/梯度/优化器状态在每卡冗余：训练过程中这些数据本质相同，切分只损失冗余、不损失正确性；每卡只存 1/N 并按需通信（all-gather/reduce-scatter）即可恢复完整语义。

**Q4：ZeRO 的通信代价是多少？**
DDP 每步约 $2M$（一次梯度 all-reduce）；ZeRO-1/2 近似相同；ZeRO-3 每步 $3M$（前向 all-gather $M$ + 反向 all-gather $M$ + 梯度 reduce-scatter $M$），约 1.5×DDP。通信量不随 N 爆炸（ring 算法带宽最优），但步数与延迟随 N 增长。

**Q5：offload 什么时候值得用？**
GPU 数固定且显存不足、CPU 内存充足时（单卡/双卡跑大模型），offload 是"能不能跑"的差别；GPU 足够或可用 ZeRO-1/2 时用 offload 得不偿失——PCIe 带宽比 HBM 慢约 100 倍，训练速度损失 2~10 倍。

**Q6：ZeRO 和 FSDP 的区别？**
同一思想的两种实现：FSDP 是 PyTorch 原生的 ZeRO-3（参数独立切分+FlatParameter 展平），DeepSpeed 靠 JSON 配置、附带 offload/3D 并行等工程能力；性能上 FSDP 在 PyTorch 生态内整合度更高，DeepSpeed 在超大规模训练上积累更久。

**Q7：ZeRO-3 训练为什么比 DDP 慢？怎么缓解？**
慢在参数权重每步两次 all-gather + 梯度 reduce-scatter（1.5×通信量）和分片带来的更高延迟。缓解：`overlap_comm` 通信计算重叠、`stage3_prefetch_bucket_size` 参数预取、梯度 bucket 化、ZeRO-1/2 够用时不用 stage-3。

**Q8：为什么 70B 模型只能用 ZeRO-3/FSDP 这类方案？**
DDP 每卡需约 800GB（含优化器），单卡 H100 80GB 装不下；ZeRO-3 N=64 时每卡约 12.5GB + 激活，配合激活重计算即可训练。省显存与带宽最优的通信算法使它成为"单卡装不下"场景的唯一解。

## 八、自我检验

- [ ] 能说出 DDP 的 3 份冗余与 7B 模型的 16/48/16GB 账本
- [ ] 能画出 ZeRO-1/2/3 切分表并背出 N=8 时的 38/24/10GB 结果
- [ ] 能手推 ZeRO-3 与 DDP 的通信量比值（3M/2M）
- [ ] 能讲清 ZeRO-3 前向/反向的 all-gather 与 reduce-scatter 时机
- [ ] 能说出 offload 的适用条件与速度代价
- [ ] 能写出 DeepSpeed JSON 的 stage/offload/混合精度关键字段
- [ ] 能说明 ZeRO 与 FSDP、Megatron-TP 的异同
- [ ] 能回答 8 个面试追问
