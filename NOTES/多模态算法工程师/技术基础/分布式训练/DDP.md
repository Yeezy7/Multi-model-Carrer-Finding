# DDP：分布式数据并行

> 本模块索引见 [分布式训练详解](分布式训练详解.md)

## 一、定义与原理

DDP（Distributed Data Parallel，分布式数据并行）是 PyTorch 在"单卡放得下模型"时最主流的训练加速方案。核心思想一句话：**每张卡一份完整的模型副本 + 各自读不同的 batch + 梯度 all-reduce 求平均 + 各自用平均梯度更新，所有副本时刻保持一致。**

### 1.1 训练流程

```text
                  数据按 rank 分片（互不重叠）
                         ↓
      ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
      │  GPU0   │  │  GPU1   │  │  GPU2   │  │  GPU3   │
      │ 模型副本  │  │ 模型副本  │  │ 模型副本  │  │ 模型副本  │
      │ batch 0 │  │ batch 1 │  │ batch 2 │  │ batch 3 │
      │ 前向     │  │ 前向     │  │ 前向     │  │ 前向     │
      │ 反向 g0  │  │ 反向 g1  │  │ 反向 g2  │  │ 反向 g3  │
      └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
           └────────────┴─────┬─────┴────────────┘
                        all-reduce 梯度求平均
                              g_avg = (g0+g1+g2+g3)/4
                               ↓ 广播给所有卡
               每张卡各自执行：w ← w − η·g_avg（参数保持一致）
```

流程拆解：

1. **数据分片**：用 `DistributedSampler` 把数据集按进程号（rank）无重叠切分，每卡只看自己的那部分；
2. **独立前向反向**：每卡的 forward/backward 完全独立，互不等待，各自算出本卡 batch 的梯度 $g_i$；
3. **梯度 all-reduce**：backward 结束后，所有卡用集合通信把梯度求和并除以卡数 $N$，得到平均梯度（每个参数都同步一次）；
4. **各自更新**：每卡用同一个平均梯度执行 `optimizer.step()`，数学上各卡参数永远一致。

### 1.2 数学等价性

设 $N$ 张卡，第 $i$ 张卡的梯度为 $g_i = \frac{\partial \mathcal{L}_i}{\partial w}$（$\mathcal{L}_i$ 是本卡小 batch 的平均损失）。all-reduce 求平均：

$$g_{avg} = \frac{1}{N}\sum_{i=1}^{N} g_i = \frac{1}{N}\sum_{i=1}^{N}\frac{\partial \mathcal{L}_i}{\partial w} = \frac{\partial}{\partial w}\left(\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_i\right)$$

右边恰好是"全局 batch（$N$ 卡 batch 拼接）平均损失"的梯度。因此：

> **DDP 训练结果 ≈ 用 $N$ 倍 batch size 在单卡上训练的结果**（与梯度累积语义相同），唯一区别是每步多一次通信开销。

## 二、与 DP 的对比（为什么 DDP 取代了 DP）

PyTorch 早期的 `torch.nn.DataParallel`（DP）是单进程多线程实现，已基本被 DDP 取代。对比总表：

| 维度 | DP（DataParallel） | DDP（DistributedDataParallel） |
| --- | --- | --- |
| 进程模型 | 单进程 + 多线程（线程绑定卡） | 多进程（每卡一个独立 Python 进程） |
| 梯度通信 | 主卡（GPU0）逐个 gather 梯度 → 更新 → broadcast | Ring all-reduce，所有卡对称参与 |
| 通信量 | 主卡收发约 $O(NG)$，随卡数线性增长 | 每卡约 $2\times\frac{N-1}{N}G \approx 2G$，与卡数基本无关 |
| GIL | 单进程内线程受 GIL 约束，CPU 调度有瓶颈 | 多进程无 GIL 问题 |
| 负载均衡 | 主卡额外承担通信与参数更新，成为瓶颈 | 所有卡对称，天然均衡 |
| 多机 | 不支持 | 支持（跨节点） |

### 2.1 通信量对比（为什么 DP 扩展性差）

设梯度总量为 $G$（≈ 参数量 × 2 字节），卡数 $N$：

- **DP**：主卡要收 (N-1) 份梯度、再广播 (N-1) 份更新后的参数，主卡通信量 $O(NG)$——卡越多，主卡越慢，8 卡以上收益骤降；
- **DDP**：Ring all-reduce 中每卡只需收发 $2\times\frac{N-1}{N}G \approx 2G$ 字节，$N$ 增大时每卡通信量基本不变（推导见"集合通信原语"子篇）——这是 DDP 能扩展到上百卡的根本原因。

### 2.2 为什么 DP 受 GIL 拖累

DP 是一个 Python 进程 + 多线程。CPython 的 GIL 保证同一时刻只有一个线程执行 Python 字节码；虽然 CUDA 计算大多在 C 库内部（会释放 GIL），但数据搬运、梯度合并、Python 层调度等仍互相竞争，线程多了之后吞吐上不去。DDP 每个进程独立解释器，彻底绕开 GIL。

## 三、源码实现（重点）

### 3.1 单机多卡完整可运行代码

```python
# ddp_train.py —— 单机多卡 DDP 训练（线性回归玩具模型）
# 运行方式：torchrun --nproc_per_node=4 ddp_train.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class ToyDataset(Dataset):
    """y = 2·x 各维之和 + 0.5 + 噪声"""

    def __init__(self, n=10000):
        torch.manual_seed(0)
        self.x = torch.randn(n, 16)
        self.y = (self.x * 2.0).sum(dim=1, keepdim=True) + 0.5

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # ① 初始化进程组：nccl 是英伟达 GPU 集合通信库，所有分布式操作都建立在它之上
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])    # 本机第几块卡（torchrun 注入）
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()              # 全局进程编号 0 .. world_size-1
    world_size = dist.get_world_size()  # 进程总数（这里 = 4）

    # ② 数据分片：DistributedSampler 把数据集按 rank 无重叠切分
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)

    # ③ 模型：每卡一份完整副本，再交给 DDP 包装
    model = ToyModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10):
        # ④ 每个 epoch 开头必须 sampler.set_epoch(epoch)（原因见 3.2-②）
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()   # 反向结束自动触发梯度 all-reduce（详见第四节）
            optimizer.step()  # 每卡用平均梯度各自更新，参数保持一致

        # ⑤ 只在 rank 0 打印 / 保存（否则每张卡各打一份、文件被反复覆盖）
        if rank == 0:
            print(f"epoch {epoch} loss = {loss.item():.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "ddp_model.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

**运行命令（单机 4 卡）：**

```bash
torchrun --nproc_per_node=4 ddp_train.py
```

- `torchrun` 是官方启动器：自动注入 `RANK / LOCAL_RANK / WORLD_SIZE` 环境变量、均匀分配 GPU、进程崩溃自动重启；
- 等价写法：`python -m torch.distributed.run --nproc_per_node=4 ddp_train.py`；
- **多机**（2 台 × 8 卡，master 节点执行）：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --master_addr=10.0.0.1 --master_port=29500 --node_rank=0 ddp_train.py
# worker 节点把 --node_rank 换成 1，其余参数不变
```

- 无法实跑时的验证：单卡机器用 `torchrun --nproc_per_node=1 ddp_train.py` 逻辑同样成立；只验证分布式 API 时可用 CPU 后端 `backend="gloo"` + `--nproc_per_node=2` 做单元测试（仅验证逻辑，与性能无关）。

### 3.2 代码要点逐条拆解

**① `dist.init_process_group(backend="nccl")`**：进程组初始化，是分布式训练的第一步。nccl 为英伟达 GPU 专用后端（NVLink/PCIe 上性能最好）；CPU 场景用 gloo。所有进程必须同时调用，否则互相等待死锁。

**② `DistributedSampler` + `set_epoch`（高频考点）**：sampler 保证"每卡看到不重叠的样本"，但它的 shuffle 用的是**进程号 + epoch 号**做随机种子：

- 如果不调 `set_epoch(epoch)`，每个 epoch 的 shuffle 结果都一样——同一个 epoch 内各卡样本不重叠，但**跨 epoch 样本顺序永远不变**，相当于每卡只在一个固定子集上训练；
- 更隐蔽的问题：多进程同时取随机数，默认种子相同会导致**所有卡 shuffle 出同一份顺序**，数据并行退化为"每卡重复全量数据"；
- 正确姿势：`for epoch in range(...): sampler.set_epoch(epoch)`。

**③ `DDP(model, device_ids=[local_rank])`**：包装后模型行为不变（forward 自动解包），但 backward 的梯度会被自动 all-reduce。取原始模块用 `model.module`。

**④ 只在 rank 0 打印/保存**：所有进程执行同一份代码，不判断 rank 的话：打印会输出 N 份、`torch.save` 会 N 个进程同时写同一个文件（损坏或互相覆盖）。保存模型时 `model.module.state_dict()` 与 `model.state_dict()` 等价（DDP 转发 state_dict）。

### 3.3 梯度累积与 DDP 的配合（重点）

大 batch 模拟：显存放不下大 batch 时，把小 batch 的梯度累 N 次再更新一次。与 DDP 配合的完整写法：

```python
accum_steps = 4   # 4 个 micro-batch 累计一次参数更新

for epoch in range(10):
    sampler.set_epoch(epoch)
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        # ① 除以 accum_steps 归一化：保证累加后的平均梯度与
        #    "直接使用 accum_steps×batch 的大 batch"数学等价
        loss = criterion(model(x), y) / accum_steps

        if (i + 1) % accum_steps == 0:
            loss.backward()              # 第 N 次：正常反向，触发 all-reduce
            optimizer.step()             # 更新后清梯度
            optimizer.zero_grad()
        else:
            with model.no_sync():        # ② 前 N-1 次：跳过梯度通信
                loss.backward()
```

三个关键点：

- **① 除以 accum_steps**：loss 是均值形式时，$N$ 次 backward 累加的梯度 ≈ 大 batch 梯度的 $N$ 倍，必须先归一化；
- **② DDP 默认每个 backward 都触发一次 all-reduce**：梯度累积 N 次就有 N 次通信，纯属浪费 → 用 `model.no_sync()` 上下文管理器关掉中间 N-1 次；
- **③ 数学等价性**：no_sync 期间本卡梯度是"局部小 batch 的梯度"，最后一次统一 all-reduce 求平均。all-reduce 是线性运算，多次同步再累加 ≡ 最后一次同步，结果完全一致（但显存/通信省了）。

### 3.4 find_unused_parameters

DDP 默认假设**所有参数**都在 backward 中产生梯度。若某些参数没被用到，DDP 会一直等它的梯度，造成**死锁/卡死**或报错。常见场景（多模态尤其常见）：batch 里部分图缺失、多任务模型的某个头没有参与损失、前向有条件分支、冻结层仍参与前向。

解法与取舍：

- `DDP(model, find_unused_parameters=True)`：DDP 主动检查哪些参数未产生梯度并跳过等待。代价：每步多一次参数遍历，性能下降几个百分点；
- **更推荐**：不参与训练的参数直接 `requires_grad=False`（如冻结的视觉塔），DDP 不会等待它们的梯度，无需开开关；
- 多模态经验：视觉塔冻结 + 投影层训练时，若视觉塔输出参与了 forward 但不参与 loss（如 CLIP 对比损失只接投影层输出），就属于"used in forward, unused in loss"——要么 find_unused_parameters=True，要么把视觉塔移出 DDP 包装（先 forward 取特征，再喂给 DDP 内的模型）。

### 3.5 常见坑位清单

1. **batch size 语义**：`DataLoader(batch_size=64)` 是**每卡** 64，全局 batch = 64 × world_size = 256；学习率要按全局 batch 缩放（见第五节）；
2. **随机种子**：每进程种子用 `seed + rank`，否则模型初始化等随机源在各卡一致，会破坏梯度平均的多样性；
3. **混合精度**：`torch.cuda.amp` / `torch.amp` 与 DDP 完全兼容，直接叠加；
4. **`model.module` vs `model`**：DDP 包装后 `model.state_dict()` 可用；取原始模块、或调自定义方法时必须用 `model.module`；
5. **`device_ids` 与 `LOCAL_RANK`**：`LOCAL_RANK` 是本机卡号，`RANK` 是全局进程号（多机时不同）；单机时两者相等。

## 四、通信机制：梯度何时同步、如何同步

### 4.1 时机：每个 backward 结束时（默认）

DDP 通过注册在 autograd 图上的 hook 自动完成梯度同步，用户代码不需要（也不应该）手动调 all-reduce。默认行为：**每次 `loss.backward()` 结束后**，本卡所有参数梯度完成 all-reduce 求平均，之后 `optimizer.step()` 用的是平均梯度。

更精确地说，不是"全部算完再通信"，而是**边反向边通信**（见 4.3 重叠）。

### 4.2 梯度桶（Bucket）

逐参数通信太慢（几万个小 all-reduce），DDP 把参数按注册顺序（`model.parameters()` 顺序）分组装桶，每个桶默认容量约 25MB（`bucket_cap_mb` 可配）：

```text
参数列表: [p0, p1, p2, ..., pN]
             ↓ 按顺序装桶（默认每桶 ~25MB）
  桶1: [p0 ... pk]  桶2: [p_{k+1} ... p_m]  ...  桶N: [... pN]
```

- backward 中，一个桶内所有参数的梯度一凑齐，就立刻对**这个桶**发起一次 all-reduce，无需等全部反向算完；
- 桶是"通信与计算重叠"的实现基础；同桶参数顺序还影响梯度计算完成的先后，故不要随意改变 `parameters()` 顺序。

### 4.3 通信与计算重叠

反向传播是逐层从输出向输入推进的。DDP 的策略是"算完一层、通信一层"：

```text
backward 时间轴（单卡视角）：
层 L 梯度计算完成 → 入桶 → 桶满 → 异步发起 all-reduce
                                    ↓（NCCL 在独立 CUDA stream 上跑，不阻塞）
层 L-1 反向继续计算 → 入桶 → 桶满 → 异步 all-reduce → ...
```

- 梯度一产生就入桶，桶满立即异步通信；
- all-reduce 由 NCCL 在专用 CUDA stream 上执行，与后续层的反向计算（另一 stream）并行 → **大部分通信时间被计算时间隐藏**，宏观上表现为"backward 时间 ≈ 单卡反向时间 + 少量残余通信"。

### 4.4 Ring all-reduce 原理（一句话版）

设梯度总量 $G$、卡数 $N$。把每卡梯度切成 $N$ 块、各卡排成环：

1. **scatter-reduce**：沿环转 $N-1$ 圈，每卡累加出完整的一块和；
2. **all-gather**：再转 $N-1$ 圈，把每块结果广播给所有人。

每卡总通信量：

$$2 \times \frac{N-1}{N}G \approx 2G \quad (N \text{ 较大时})$$

与卡数 $N$ 无关——这正是 DDP 能扩展到几十上百卡的根本原因（完整推导见"集合通信原语"子篇）。

## 五、性能分析：通信占比与缩放规则

### 5.1 步时间模型与通信占比

单步训练时间 ≈ 计算时间与通信时间（理想重叠后取 max，实际有残余）：

$$T_{step} \approx \max(T_{compute},\ T_{comm}) + \text{overhead}$$

- 通信量 ∝ 模型参数规模；计算量 ∝ 参数量 × 每卡 batch 的 token 数；
- **每卡 batch 越大，通信占比越低**：通信量不变、计算量变大；
- 小模型 + 小 batch 时通信占比可达 50%+；大模型 + 大 batch 可降到 10% 以下；
- 经验数据：7B 模型、每卡 batch 1（4096 token）时通信占比约 30%+；每卡 batch 8 时降到个位数；
- 结论：DDP 适合"单卡放得下、尽量加大每卡 batch"的场景；每卡梯度过小（模型太小、batch 太小）时 all-reduce 固定开销占比高，效率差。

### 5.2 batch size 与学习率：线性缩放规则

DDP 使全局 batch 变为 $B_{global} = B_{per\_card} \times N$。为保持收敛行为等价，学习率应同步缩放（Goyal et al., 2017 线性缩放规则）：

$$\eta_{new} = \eta_{base} \times \frac{B_{new}}{B_{base}}$$

- 例：单卡 batch 64、lr 1e-3；4 卡全局 batch 256 → lr 建议 4e-3；
- 注意：lr 放大后 **warmup 步数也要等比延长**（避免大 lr 直接冲坏早期训练）；
- 边界：batch 超过某个阈值后线性缩放失效（大 batch 收敛变慢），此时改用平方根缩放 $\sqrt{N}$ 或 warmup 曲线调整。

### 5.3 吞吐评估与调优

- 关注指标：每秒样本数、每卡 GPU 利用率（`nvidia-smi`、`torch.profiler`）；
- 若 backward 时间随卡数增长明显，说明重叠没做好（通信没藏住）；
- 调优手段：加大每卡 batch；调大 `bucket_cap_mb` 减少通信次数；NCCL 环境变量（P2P、GDRCopy）；`torch.compile` 与 DDP 叠加。

## 六、与 ZeRO/FSDP 的关系：DDP 的边界在哪

### 6.1 DDP 的局限：每卡都有一份完整状态

DDP 每张卡都要保存**完整**的四件套（BF16 混合精度）：

| 显存项 | 7B 模型 |
| --- | --- |
| 权重（BF16） | 14 GB |
| 梯度（BF16） | 14 GB |
| FP32 master weight | 28 GB |
| Adam 状态（m + v） | 56 GB |
| **合计（不含激活）** | **112 GB / 卡** |

关键：这 112 GB 是**每卡**都有的，4 卡总显存 448 GB，但模型只能训一份——显存随卡数"线性叠加"而不是"摊薄"。模型本身放不进单卡（如 70B 需 1.1 TB）时，DDP 无能为力：

> **DDP 只解决"跑得快"，不解决"放得下"。**

### 6.2 解决"放不下"：ZeRO / FSDP

ZeRO（DeepSpeed）与 FSDP 的核心思想：**把四件套分片到各卡**：

- ZeRO-1：只分片优化器状态（省 56/112 ≈ 一半）；
- ZeRO-2：分片优化器状态 + 梯度；
- ZeRO-3 / FSDP：分片优化器状态 + 梯度 + 权重（模型本身也摊到各卡，前向时 all-gather 权重、反向后 reduce-scatter 梯度）；
- 通信从 all-reduce 变为 **reduce-scatter + all-gather**（通信量从 $2G$ 增到约 $3G$ 量级），用多一点通信换显存大幅节省。

### 6.3 演进路线

```text
DP（单进程多线程）→ DDP（多进程 + all-reduce）→ ZeRO/FSDP（分片）
                                                   ↓
                   单卡放不下 → 张量并行 TP / 流水线并行 PP（见对应子篇）
```

- DDP 是理解一切后续方案的地基：FSDP 与 DDP 共用进程组、sampler、rank 概念，只是把"每卡完整模型"换成"每卡分片模型"；
- 一句话总结：**DDP 解决"多卡一起跑"，ZeRO/FSDP 解决"单卡放不下"，TP/PP 解决"单层放不下"。**

## 七、高频面试问答

**Q1：DDP 和 DP 的区别？**
进程模型（多进程 vs 单进程多线程）、通信方式（ring all-reduce vs 主卡 gather+broadcast）、通信量（$2G$ vs $O(NG)$）、GIL、负载均衡（对称 vs 主卡瓶颈）、是否支持多机。DDP 在各方面均优于 DP，是事实标准。

**Q2：all-reduce 是什么？**
一种集合通信原语：所有进程传入自己的张量，返回时**每个进程都得到所有进程张量的和**（DDP 里再除以 N 即平均）。Ring all-reduce 把数据切成 N 块在环上流转，每卡通信量约 $2G$，与卡数无关。

**Q3：DDP 之后为什么还是放不下大模型？**
DDP 每卡保存完整权重+梯度+优化器状态（7B 约 112 GB/卡），显存随卡数线性叠加而非摊薄。模型本身超过单卡显存时 DDP 无法启动，需要 ZeRO/FSDP 分片或 TP/PP 切模型。

**Q4：DistributedSampler 为什么要 set_epoch？**
shuffle 的随机种子由"进程号 + epoch 号"决定。不调用 set_epoch，则：① 每个 epoch 的划分与顺序完全不变（epoch 间无新 shuffle）；② 多进程默认随机种子相同，可能 shuffle 出同一样本顺序，数据并行退化为重复训练。每个 epoch 开头调用一次即可。

**Q5：梯度累积和 DDP 怎么配合？**
loss 除以累积步数归一化；用 `model.no_sync()` 包住前 N-1 次 backward 跳过通信，只在第 N 次正常 backward 触发 all-reduce。否则累积 N 次就有 N 次通信，浪费带宽。数学上结果等价（all-reduce 是线性运算）。

**Q6：什么时候需要 find_unused_parameters=True？**
有参数参与了 forward 但没产生梯度（多任务分支未激活、冻结层仍参与前向、batch 内缺失样本）时，DDP 等待其梯度会死锁。开 find_unused_parameters 有性能代价；更优做法是把冻结参数置 requires_grad=False，或将不参与 loss 的子模块移出 DDP 包装。

**Q7：为什么只在 rank 0 保存模型？**
所有进程执行同一份代码。不判断 rank，N 个进程会同时写同一个文件（覆盖/损坏），打印输出 N 份。训练流程也必须以 rank 0 为唯一入口（如学习率打印、日志、early stop 判断）。

**Q8：DDP 和 TP/PP 是什么关系？**
DDP 是数据并行（切数据、通信梯度）；TP 切单层权重、PP 按层切模型（通信激活）。三者正交可叠加（Megatron 3D 并行：DP × TP × PP）。DDP 在单卡放得下时用；TP/PP 在模型放不下时用，组合时 DDP 的梯度 all-reduce 在"模型副本"之间进行。

## 八、自我检验

- [ ] 能画出一张卡视角的 DDP 训练流程图（分片→前向→反向→all-reduce→更新）
- [ ] 能推出 $g_{avg} = \frac{1}{N}\sum g_i$ 与全局 batch 梯度的等价关系
- [ ] 能说出 DDP 取代 DP 的四个理由（进程模型/通信量/GIL/多机）
- [ ] 能独立写出单机多卡 DDP 训练代码（init_process_group + DDP 包装 + DistributedSampler + rank 0 保存 + torchrun 启动）
- [ ] 能解释 set_epoch 的两个作用与梯度累积中 no_sync 的正确用法
- [ ] 能说出梯度桶的作用与"边反向边通信"的重叠原理
- [ ] 能写出 Ring all-reduce 每卡约 $2G$ 通信量的结论并简述两阶段过程
- [ ] 能解释线性缩放规则 $\eta \propto B_{global}$ 及 warmup 联动
- [ ] 能说清 DDP 的显存边界（112 GB/卡）与 ZeRO/FSDP 的分片思路
- [ ] 能回答第八节 8 个面试追问
