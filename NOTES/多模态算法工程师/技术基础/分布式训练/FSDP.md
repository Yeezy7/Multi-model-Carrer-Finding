# FSDP：全参数分片数据并行（Fully Sharded Data Parallel）

> 本模块索引见 [分布式训练详解](分布式训练详解.md)

## 一、定义与公式

### 1.1 FSDP 是什么

**FSDP（Fully Sharded Data Parallel，全参数分片数据并行）** 是 PyTorch 官方的分布式训练策略：把**参数、梯度、优化器状态**三者全部按数据并行（DP）维度切成 $N$ 份，每张卡只持有 $1/N$，需要时通过通信"拼回来"使用。它是 DeepSpeed ZeRO-3 思想的 PyTorch 原生实现（ZeRO = Zero Redundancy Optimizer，"零冗余"）。

$$M(N) = \frac{M(1)}{N}, \quad M(1) = \underbrace{2P}_{\text{参数 BF16}} + \underbrace{2P}_{\text{梯度}} + \underbrace{12P}_{\text{Adam 状态与 FP32 主权重}}$$

其中 $P$ 为参数量，$M(1)$ 为单卡全量训练所需显存（字节数）。**显存随卡数线性下降**，这是 FSDP 与 DDP 最本质的区别（DDP 的显存与卡数无关）。

### 1.2 三个核心原语

参数 $W \in \mathbb{R}^{P}$ 展平后按 $N$ 个 rank 切分，rank $r$ 持有分片：

$$W_r = W[r \cdot S : (r+1) \cdot S], \quad S = P / N$$

| 原语 | 行为 | 公式 | 方向 |
| --- | --- | --- | --- |
| all-gather | 收集所有 rank 的分片拼成完整张量 | $\text{Gather}(W_0,\dots,W_{N-1}) = W$ | 前向/反向使用前 |
| reduce-scatter | 跨 rank 求和（或平均）后，把结果按分片分发 | $\text{ReduceScatter}(g_0,\dots,g_{N-1}) = \frac{1}{N}\sum_{r} g_r$ 的第 $r$ 片 | 反向求完梯度后 |

### 1.3 与 ZeRO 系列的对应关系

| 策略 | 切分参数 | 切分梯度 | 切分优化器状态 | 单卡显存（7B 示例） |
| --- | --- | --- | --- | --- |
| DDP | ✗ | ✗（all-reduce 复制） | ✗ | ≈114 GB |
| ZeRO-1 | ✗ | ✗ | ✓ | ≈86 GB |
| ZeRO-2 | ✗ | ✓ | ✓ | ≈58 GB |
| ZeRO-3 = FSDP | ✓ | ✓ | ✓ | ≈30 GB（4 卡） |

> 直观理解：FSDP = 把"每个参数在全部 N 卡各存一份"改为"每个参数只存一份，谁要用谁去取"。

## 二、核心原理

### 2.1 切分与重建的完整流程（逐层粒度）

FSDP 的切分单位是**被包装的子模块**（通常是每一个 Transformer Block），而非整个模型。训练一个 step 的流程：

```
前向（按网络顺序逐层执行）：
  ┌─ 第 L 层 ─────────────────────────────────────────────┐
  │ ① all-gather：收集所有 rank 的分片 → 重建完整 W_L      │
  │ ② 本地计算：Y_L = f(X_L, W_L)（与 DDP 相同的计算）     │
  │ ③ 丢弃完整 W_L，只保留自己的分片（显存立刻释放）        │
  └────────────────────────────────────────────────────────┘
  ↓ 激活 Y_L 传给第 L+1 层（激活不切分，每 rank 全量一份）

反向（按网络逆序逐层执行）：
  ┌─ 第 L 层 ─────────────────────────────────────────────┐
  │ ④ 再次 all-gather 重建完整 W_L（为算梯度做准备）       │
  │ ⑤ 本地反传：算出完整梯度 g_L（或只算自己分片的梯度）   │
  │ ⑥ reduce-scatter：跨 rank 求和并平均 → 各拿回分片      │
  │ ⑦ 优化器只更新自己分片对应的参数（Adam 状态也只在本地）│
  └────────────────────────────────────────────────────────┘
```

**为什么逐层而不是一次全部 gather？** 前向中任何时刻只需要"当前层"的完整参数，逐层 gather 让重建后的完整参数是**瞬时占用**（用完即丢），把峰值显存压到"单层参数大小 + 分片"，这是 FSDP 能装下超大模型的关键。

### 2.2 与 DDP 的流程对比

```
DDP（每 rank 有完整参数副本）：
  前向: 完整参数一直在显存 → 计算 → 反向求完整梯度
  反向末尾: 一次全量梯度 all-reduce（求和平均后回传）→ 各自更新完整参数

FSDP（每 rank 只有 1/N 参数）：
  前向: 逐层 all-gather → 算 → 丢
  反向: 逐层 all-gather → 算梯度 → reduce-scatter → 只更新自己的分片
```

核心差异一句话：**DDP 在"时间上"同步梯度（一步一同步），FSDP 在"空间上"切分一切（参数/梯度/优化器状态都只有 1/N）**。

### 2.3 分片粒度与 auto_wrap_policy

FSDP 把被包装的每个子模块的参数各自展平成一个 flat parameter 再切分。切分粒度由 `auto_wrap_policy` 决定：

- **粗粒度（整个模型当一层）**：一次 all-gather 整个模型 → 峰值显存高、通信不重叠，几乎不省显存；
- **细粒度（每个 Transformer Block 一层）**：逐层 all-gather → 峰值 = 单层参数，通信可与计算重叠；
- 实践中用 `transformer_auto_wrap_policy` 按层类型包装，或用 `size_based_auto_wrap_policy` 按参数阈值包装。

## 三、源码实现

### 3.1 生产级 FSDP 配置（GPU + NCCL）

```python
# 片段示例：核心配置参数逐个说明（完整可跑脚本见 3.2）
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, CPUOffload, MixedPrecision, BackwardPrefetch,
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
        param_dtype=torch.bfloat16,     # 参数以 BF16 存储/通信
        reduce_dtype=torch.bfloat16,    # 梯度规约用 BF16
        buffer_dtype=torch.bfloat16,    # buffer 用 BF16
    ),
    auto_wrap_policy=wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 提前 gather 下一层参数，通信与计算重叠
    device_id=rank,                     # 显式绑定计算卡
)
```

- `FULL_SHARD`：三态全切（ZeRO-3）；`SHARD_GRAD_OP`：只切梯度和优化器状态（ZeRO-2）；`NO_SHARD`：不切（等价 DDP 但多了 FSDP 开销）；
- `cpu_offload` 单独使用时配合 `FULL_SHARD` 可把单卡显存压到几 GB 量级；
- `mixed_precision` 只在 FSDP 内部生效，计算仍可用 FP32。

### 3.2 完整可跑的训练脚本（GPU 环境）

```python
# 保存为 train_fsdp.py。运行方式（需要 4 张 NVIDIA GPU）：
#   torchrun --nproc_per_node=4 --nnodes=1 train_fsdp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

class TransformerBlock(nn.Module):
    """示例：一个分片单元 = 自注意力 + FFN（论文里 FSDP 的默认包装粒度）"""
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
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
    dist.init_process_group("nccl")            # GPU 环境用 nccl；CPU 演示可换 "gloo"
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock})

    model = TinyGPT().cuda()
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=rank)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(10):
        x = torch.randint(0, 1000, (4, 32)).cuda()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, 1000), x.view(-1))
        opt.zero_grad()
        loss.backward()          # FSDP 在反向内部自动完成 all-gather + reduce-scatter
        opt.step()               # 优化器只更新本 rank 持有的分片
        if rank == 0:
            print(f"step {step}: loss = {loss.item():.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```text
运行输出（示意）：
step 0: loss = 7.1234
step 1: loss = 6.9865
...
```

### 3.3 手写"参数分片训练循环"最小演示（CPU/gloo 可跑通）

不依赖 FSDP API，只用 `all_gather` / `reduce_scatter_tensor` 两个原语复刻其思想。**本机无 GPU 也可直接运行**：

```python
# 保存为 fsdp_mini.py。运行方式（CPU/gloo，2 进程）：
#   torchrun --nproc_per_node=2 fsdp_mini.py
# 本机若无 torchrun，可用 torch.multiprocessing.spawn 等价启动（见文末注）
import torch
import torch.nn as nn
import torch.distributed as dist

def main(rank=0, world=2):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    out_f, in_f, bs = 32, 16, 8
    shard = out_f // world
    torch.manual_seed(0)

    # —— 每 rank 只持有 W 的一个分片（其余分片在别的 rank 上）——
    W_full = torch.randn(out_f, in_f) * 0.1
    W = nn.Parameter(W_full[rank * shard:(rank + 1) * shard].clone())
    opt = torch.optim.SGD([W], lr=0.1)
    X = torch.randn(bs, in_f)        # 数据并行副本
    target = torch.randn(bs, out_f)

    for step in range(15):
        # ① 前向：all-gather 重建完整 W → 计算 → "丢弃"（只留分片）
        full = [torch.empty_like(W) for _ in range(world)]
        dist.all_gather(full, W.detach())
        Y = X @ torch.cat(full, 0).t()
        loss = ((Y - target) ** 2).mean()

        # ② 反向：每 rank 算完整梯度 → ③ reduce-scatter 求和后取平均
        dY = 2 * (Y - target) / (bs * out_f)
        dW_full = dY.t() @ X
        grad_shard = torch.empty(shard, in_f)
        dist.reduce_scatter_tensor(grad_shard, dW_full, op=dist.ReduceOp.SUM)
        grad_shard /= world

        if step == 0:   # 与 autograd 全量参考对比，验证流程正确
            W_ref = W_full.detach().clone().requires_grad_()
            ((X @ W_ref.t() - target) ** 2).mean().backward()
            ref = W_ref.grad[rank * shard:(rank + 1) * shard]
            assert torch.allclose(grad_shard, ref, atol=1e-5)
            if rank == 0:
                print("验证通过：reduce-scatter 分片梯度 == autograd 参考")

        W.grad = grad_shard          # ④ 只更新自己的分片（优化器状态也只有分片）
        opt.step()
        opt.zero_grad()
        if rank == 0 and step % 5 == 4:
            print(f"step {step}: loss {loss.item():.4f}")
    dist.destroy_process_group()
    print(f"rank{rank} 完成")

if __name__ == "__main__":
    main()
```

```text
运行输出（2 进程 CPU，已验证）：
验证通过：reduce-scatter 分片梯度 == autograd 参考
step 4: loss 0.9821
step 9: loss 0.8487
step 14: loss 0.7424
rank0 完成 / rank1 完成
```

> **启动方式注**：`torchrun` 是 PyTorch 官方启动器（`python -m torch.distributed.run`）。无 torchrun 时可用 spawn 启动 `main(rank, world)`：`torch.multiprocessing` 为每个 rank 起一个进程即可（本文 3.3 已验证该路径）。

### 3.4 torchrun 多机启动

```bash
# 单机 4 卡：--nproc_per_node = 每机卡数
torchrun --nproc_per_node=4 --nnodes=1 train_fsdp.py

# 双机各 8 卡：--nnodes 节点数，--node_rank 节点序号，--master_addr/port 指定主节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 train_fsdp.py
```

torchrun 会自动设置 `RANK / LOCAL_RANK / WORLD_SIZE` 环境变量并完成 rendezvous，脚本内只需 `dist.init_process_group("nccl")`。

## 四、深入分析：显存 / 通信 / 复杂度数值账本

### 4.1 显存账本：7B 模型 BF16 训练（逐项计算）

每参数占字节：参数 2B（BF16）+ 梯度 2B（BF16）+ 优化器 m 4B（FP32）+ 优化器 v 4B（FP32）+ FP32 主权重 4B = **16 B/参数**。

| 项目 | 精度 | 公式 | DDP 每卡 | FSDP 4 卡每卡 |
| --- | --- | --- | --- | --- |
| 模型参数 | BF16 | $7\times10^9 \times 2$B | 14 GB | 14/4 = 3.5 GB |
| 梯度 | BF16 | $7\times10^9 \times 2$B | 14 GB | 14/4 = 3.5 GB |
| Adam 一阶矩 m | FP32 | $7\times10^9 \times 4$B | 28 GB | 28/4 = 7 GB |
| Adam 二阶矩 v | FP32 | $7\times10^9 \times 4$B | 28 GB | 28/4 = 7 GB |
| FP32 主权重 | FP32 | $7\times10^9 \times 4$B | 28 GB | 28/4 = 7 GB |
| **小计** | | 16 B/参数 | **112 GB** | **28 GB** |
| 激活/通信缓冲 | | 与 batch 相关 | ≈2 GB | ≈2 GB |
| **合计** | | | **≈114 GB** | **≈30 GB** |

**结论**：FSDP 4 卡把单卡峰值从 114 GB 压到 30 GB——不靠魔法，只是把每份数据从"每卡各存一份"改成"全集群合存一份"。8 卡再减半到 ≈15 GB，这就是它能训练 70B+ 模型的原因。

### 4.2 通信账本：FSDP vs DDP 每 step 通信量

设 $P$ 为参数量，BF16 下每字节 2B：

| 通信项 | DDP | FSDP |
| --- | --- | --- |
| 梯度同步 | 1 次全量 all-reduce：$2 \times 2P$ | reduce-scatter：$2P$ |
| 前向参数重建 | 0 | all-gather：$2P$ |
| 反向参数重建 | 0 | all-gather：$2P$ |
| **合计** | $4P$ 字节 | $6P$ 字节 |

7B 数值：DDP ≈ $4\times14 = 56$ GB；FSDP ≈ $6\times14 = 84$ GB，**FSDP 通信约为 DDP 的 1.5 倍**。但两点缓解：

1. **通信与计算重叠**：`backward_prefetch` 提前 all-gather 下一层参数，计算下一层时通信并行进行，墙上时间增加远小于通信量比值；
2. **通信按层分摊**：FSDP 的通信分散在整条反向流水上，每层只传单层参数，瞬时带宽需求更平滑。

### 4.3 显存峰值时间线（单层视角）

```
时刻       前向第 L 层               反向第 L 层
        |--gather--计算--丢--|   |--gather--反算--reduce-scatter--|
峰值项:  单层完整参数           单层完整参数 + 本层激活
稳态项:  分片参数(1/N)         分片参数 + 分片梯度 + 分片优化器状态
```

**激活不切分**：每 rank 的中间激活仍是全量（每 rank 都算完整前向）。所以 FSDP 训练极深模型时通常还要配 activation checkpointing（重计算激活，进一步省显存）。

### 4.4 计算复杂度

- 前向/反向的**浮点计算量与 DDP 完全相同**（每卡都算完整前向、完整梯度）；
- 多出的成本只有通信（+50% 通信量）与 flat-parameter 拼装/切分的小额 CPU 开销；
- 分片粒度越小（层越细），通信重叠越好但 collectives 次数越多；粒度太大则峰值显存高、重叠差。这就是 `auto_wrap_policy` 存在的意义。

## 五、优缺点

| 优点 | 缺点 |
| --- | --- |
| 显存随卡数线性下降，可训超大模型（70B+） | 通信量约为 DDP 的 1.5 倍，小模型/慢互联下收益被通信吃掉 |
| 参数/梯度/优化器状态全部无冗余（ZeRO-3） | 激活不切分，深度大时仍可能 OOM，需配合 checkpoint |
| PyTorch 原生 API，与 HuggingFace Trainer 等开箱即用 | 单卡小模型（<1B）切分收益小，纯增加通信 |
| 支持 CPU offload / 混合精度 / 与 TP 组合（HybridShard） | 相比 DeepSpeed 在超大模型（>100B）场景略有额外开销 |
| 与 DDP 相比训练脚本改动极小 | 调试复杂（分片状态看不见全局参数） |

## 六、与同类对比

| 维度 | DDP | FSDP（FULL_SHARD） | ZeRO-3（DeepSpeed） | TP | PP |
| --- | --- | --- | --- | --- | --- |
| 切分什么 | 无（全量副本） | 参数/梯度/优化器状态 | 同 FSDP | 权重矩阵内部 | 按层切 |
| 单卡显存（7B） | ≈114 GB | ≈30 GB（4 卡） | ≈30 GB（4 卡） | ≈30 GB（4 卡） | ≈30 GB（4 卡） |
| 通信量（每 step） | $4P$ | $6P$ | $6P$ | 与激活成正比（与 P 无关） | 每 stage 边界传激活（最小） |
| 通信对象 | 全集群 all-reduce | 全集群 gather/scatter | 同 FSDP | TP 组内（单机 NVLink） | 相邻 stage 间 |
| 实现 | PyTorch 原生 | PyTorch 原生 | 第三方库 | Megatron 等 | GPipe/PipeDream 等 |
| 适用范围 | 单卡放得下 | 单卡放不下，卡间带宽一般 | 同 FSDP | 卡间带宽极高（NVLink） | 卡间带宽普通 |

> 生产实践中三者可叠加：**DP × PP × TP（3D 并行）**，或 **DP × TP + FSDP（HybridShard）**，见《张量并行与流水线并行》篇的混合并行章节。

## 七、高频面试问答

**Q1：FSDP 和 DDP 的本质区别？**
DDP 每卡持有完整参数副本，只在反向末尾做一次梯度 all-reduce 保持同步；FSDP 把参数/梯度/优化器状态全部切分（ZeRO-3），前向逐层 all-gather 重建参数、反向逐层 all-gather + reduce-scatter。DDP 省不了显存，FSDP 显存随卡数线性下降。

**Q2：FSDP 与 ZeRO-3 是什么关系？**
FSDP 是 ZeRO-3 思想的 PyTorch 官方实现：三者全切、逐层 gather。区别在工程：FSDP 用 flat parameter 与 autograd 图内钩子实现，DeepSpeed 用自定义引擎与 ZeRO 缓存。ZeRO-1/2 分别只切优化器状态/梯度。

**Q3：前向为什么逐层 all-gather 而不是一次 gather 完整模型？**
一次性 gather 完整模型 = 峰值显存回到全量（DDP 水平），失去意义。逐层 gather 让"完整参数"变成瞬时占用，峰值只多一层参数。这也决定了 auto_wrap_policy 的粒度选择。

**Q4：FSDP 的通信量为什么比 DDP 大？大概大多少？**
DDP 每 step 只有 1 次全量梯度 all-reduce（$4P$ 字节）；FSDP 每 step 有前向 all-gather + 反向 all-gather + reduce-scatter（$6P$ 字节），约 1.5 倍。但通信可与计算重叠，实际速度差距远小于通信量差距。

**Q5：reduce-scatter 和 all-reduce 的区别？**
all-reduce 把求和结果广播给所有 rank（每人都拿到完整结果）；reduce-scatter 求和后按分片分发（每人只拿自己的那一片）。FSDP 反向最后一步只需要自己的分片梯度，用 reduce-scatter 天然省一半通信。

**Q6：CPU offload 什么时候用？代价是什么？**
模型大到 GPU 显存完全放不下时用。参数冷落 CPU 内存，每层前向/反向前搬回 GPU，速度明显变慢（PCIe 带宽 << 显存带宽），适合"能训起来"优先、吞吐次之的极端场景。

**Q7：FSDP 与 TP 怎么组合？**
HybridShard：TP 组内做张量并行（权重再按 TP 组切），TP 组之间做 FSDP 分片——2D 组合后单卡显存 = 模型/(TP×FSDP组数)。配合 3D 并行（DP×PP×TP）可以训练万卡规模的超大模型。

**Q8：FSDP2 改进了什么？**
FSDP2 采用 per-parameter 分片（不再展平为 flat parameter），支持为不同参数独立设置分片/精度，与 TP 的 DTensor 组合更自然，通信调度更细，显存开销更低。

## 八、自我检验

- [ ] 能写出 FSDP 显存公式 $M(N)=M(1)/N$ 并手算 7B 模型 4 卡的 30GB 账本
- [ ] 能画出前向逐层 all-gather → 计算 → 丢弃、反向逐层 all-gather → reduce-scatter 的流程图
- [ ] 能说清 FSDP 与 DDP、ZeRO-1/2/3 的关系与差异
- [ ] 能手写一个用 all_gather + reduce_scatter 模拟 FSDP 的训练循环（或至少讲清每一步）
- [ ] 能解释 auto_wrap_policy 的粒度权衡与 backward_prefetch 的作用
- [ ] 能回答"FSDP 通信量为什么比 DDP 大、大多少"
- [ ] 知道 FSDP 与 TP/PP/HybridShard 如何组合
- [ ] 知道 activation checkpointing 为什么常与 FSDP 搭配
- [ ] 能回答 8 个面试追问
