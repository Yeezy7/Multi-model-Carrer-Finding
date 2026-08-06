# PyTorch 基础：从张量到完整训练

## 一、PyTorch 总体认知

PyTorch 是目前深度学习研究与工程落地最主流的框架，多模态算法岗几乎必考。它围绕两个核心设计：

1. **动态计算图**：每次前向传播即时构建计算图，反向传播时按图求导。调试直观、代码即模型。
2. **张量 + 自动求导**：一切数据都是 Tensor（张量），一切梯度由 autograd 引擎自动计算。

面试中"PyTorch 和 TensorFlow 的区别"是高概率题：

| 维度 | PyTorch | TensorFlow |
|------|---------|------------|
| 计算图 | 动态（define-by-run） | 早期静态图，TF2 后默认动态 |
| 调试 | 断点调试直观，中间结果可打印 | 静态图时代难以调试 |
| 生态 | 学术界研究主流 | 工业界部署、移动端生态更强 |
| 部署 | TorchScript / ONNX / TensorRT | TFLite / TF Serving |
| 分布式 | DDP 成熟简洁 | TF 分布式方案较繁琐 |

> **一句话**：研究用 PyTorch（代码灵活、调试方便），部署不一定用 PyTorch（常用 ONNX/TensorRT 等优化引擎）。

---

## 二、张量（Tensor）详解

### 2.1 张量的本质

张量是 PyTorch 中最基本的数据结构，本质是一个**多维数组 + 数据类型（dtype）+ 设备（device）**。

| 维度 | 数学名称 | 例子 | shape |
|------|---------|------|-------|
| 0 维 | 标量 | 一个 loss 值 | `[]` |
| 1 维 | 向量 | 一个文本 embedding | `[768]` |
| 2 维 | 矩阵 | batch 的图像 | `[32, 3]` |
| 3 维 | 三维张量 | 单张图像 | `[3, 224, 224]` |
| 4 维 | 四维张量 | batch 图像 | `[32, 3, 224, 224]` |
| 5 维 | 五维张量 | batch 视频 | `[8, 3, 16, 224, 224]` |

### 2.2 创建张量

```python
import torch

torch.tensor([1, 2, 3])              # 从数据创建
torch.zeros(2, 3)                    # 全 0
torch.ones(2, 3)                     # 全 1
torch.full((2, 3), 7)                # 全 7
torch.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
torch.randn(2, 3)                    # 标准正态分布
torch.rand(2, 3)                     # [0,1) 均匀分布
torch.eye(4)                         # 单位矩阵（对角线为 1）
torch.randperm(10)                   # 0~9 的随机排列（打乱索引）
torch.from_numpy(np_array)           # numpy → tensor（共享内存）
torch.tensor(np_array)               # numpy → tensor（拷贝）
```

关键细节：
- `torch.tensor()` 默认**拷贝数据**；`torch.from_numpy()` 与 numpy **共享底层内存**，改 numpy 会影响 tensor。
- `torch.Tensor`（类）与 `torch.tensor`（函数）不同：`torch.Tensor(3, 4)` 创建未初始化张量（值是垃圾内存），几乎不用。

### 2.3 dtype、device 与 shape 操作

```python
x = torch.randn(2, 3)
x.shape          # torch.Size([2, 3])
x.size()         # 同上
x.dtype          # torch.float32
x.device         # cpu

# dtype 转换（训练/推理精度管理的基础）
x.half()         # → float16（FP16，显存减半）
x.float()        # → float32
x.bfloat16()     # → bfloat16（大模型训练常用，动态范围同 FP32）
x.double()       # → float64

# 设备移动
x.cuda()         # 搬到 GPU
x.cpu()          # 搬回 CPU
x.to('mps')      # Apple Silicon 的 MPS 加速（mac 上训练）
```

> **💡 显存计算**：一个 float32 张量占 4 字节/元素，`[32, 3, 224, 224]` 占 `32×3×224×224×4 ≈ 19.3 MB`。换 float16 减半。大模型显存估算一般用"参数量 × 字节数"：7B 模型 FP16 权重 ≈ 14 GB。

### 2.4 张量形状操作（高频面试点）

```python
x = torch.randn(4, 768)   # batch=4, dim=768

x.view(4, 768, 1)         # 视图变换，不复制内存（要求内存连续）
x.reshape(4, 768, 1)      # 尽量复用 view，不连续时复制
x.permute(0, 2, 1)        # 维度交换（如 HWC→CHW），会改变内存布局
x.transpose(0, 1)         # 交换两个维度
x.squeeze()               # 去掉所有长度为 1 的维度
x.unsqueeze(0)            # 在 0 位置加一个维度 → [1, 4, 768]
x.flatten()               # 展平为 1 维
x.flatten(start_dim=1)    # 从第 1 维开始展平 → [4, 768]
```

**`view` vs `reshape` vs `permute` vs `contiguous`** 是必考细节：

| 方法 | 是否复制内存 | 要求 | 适用场景 |
|------|------------|------|---------|
| `view` | 否 | 原张量内存连续 | 最快的形状变换 |
| `reshape` | 视情况 | 不连续时自动复制 | 通用的形状变换 |
| `permute` | 否（只改 strides） | 无 | 维度重排（转置） |
| `transpose` | 否 | 无 | 两个维度交换 |
| `.contiguous()` | 可能复制 | 无 | 把不连续张量变成连续 |

为什么 `view` 要求连续？因为 view 只是重新解释底层一维内存的划分方式。`permute` 之后内存顺序变了（strides 变化），底层数据还是按旧顺序排列的，直接 view 会得到错误结果，必须 `.contiguous()` 先复制成连续内存。

```python
x = torch.randn(4, 768)
y = x.transpose(0, 1)     # [768, 4]，非连续
y.view(768, 4, 1)         # RuntimeError: view size is not compatible with input tensor's size
y.contiguous().view(768, 4, 1)  # 先复制为连续，再 view
```

### 2.5 广播机制（Broadcasting）

不同 shape 的张量做运算时，PyTorch 自动按以下规则广播：

1. 从最右边维度开始对齐；
2. 维度相等或其中一个为 1，则可广播；为 1 的维度会"复制扩展"到对方大小；
3. 无法对齐（不相等且都不为 1）则报错。

```python
a = torch.randn(4, 3, 2)
b = torch.randn(3, 1)      # 自动扩展为 (1, 3, 1) → (4, 3, 2)
a + b                      # 合法

c = torch.randn(4, 3, 5)
# a + c 不合法：最后一维 2 vs 5 且都不为 1，报错
```

> **💡 工程经验**：广播机制是"隐式复制"，不占额外显存，很高效。但写代码时容易踩坑——两个形状"看起来能加"实则错位广播。建议在关键运算前后打印 shape 验证。

---

## 三、自动求导（Autograd）原理

### 3.1 核心机制

`autograd` 是 PyTorch 的灵魂。核心概念：

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 标记该张量需要计算梯度 |
| 动态计算图 | 每次前向传播自动记录运算历史 |
| `loss.backward()` | 从 loss 反向传播求每个叶节点的梯度 |
| `.grad` | 梯度存于张量属性，shape 与张量相同 |
| `.grad_fn` | 记录该张量是怎么产生的（运算节点） |

```python
x = torch.tensor([2.0], requires_grad=True)   # 叶子节点
y = x ** 2                                     # y.grad_fn = PowBackward0
z = y + 3

z.backward()          # 从 z 反传
print(x.grad)         # tensor([4.0])，因为 dz/dx = 2x = 4
```

### 3.2 链式法则示例

```python
x = torch.tensor([2.0], requires_grad=True)
a = x * 3             # 3x
b = torch.sin(a)      # sin(3x)
c = b * b             # sin²(3x)

c.backward()
print(x.grad)         # 6·sin(6)·cos(6) ≈ 6 × 0.279 × 0.960 ≈ 1.61
```

### 3.3 梯度累积问题

PyTorch 的 `.grad` 是**累加**的，不是覆盖的。每调用一次 `backward()`，梯度会加到已有梯度上。所以每个 step 必须 `optimizer.zero_grad()`（或 `model.zero_grad()`）清零。

```python
# 标准三步
loss.backward()        # 1. 反向传播
optimizer.step()       # 2. 更新参数
optimizer.zero_grad()  # 3. 清空梯度（必须在 backward 之前或之后均可，习惯在 step 后）
```

梯度累加本身也是一种**技巧**（gradient accumulation），用于显存不足时等效放大 batch size：

```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accum_steps     # 平均各步 loss
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

> 等效 batch size = `per_device_batch × grad_accum_steps × 设备数`。

### 3.4 推理时为什么必须 no_grad

```python
model.eval()
with torch.no_grad():
    output = model(input)
```

`torch.no_grad()` 的作用：**不构建计算图、不保存中间激活**。带来的收益：

1. **省显存**：不保存每层激活值用于反传，显存占用大幅下降；
2. **加速**：省去记录运算历史的开销；
3. **结果一致**：推理不需要梯度。

> **面试必答**：推理时 `no_grad` 主要省显存和提速，因为不需要保存计算图和中间激活。注意 `model.eval()` 和 `no_grad` 是两回事：`eval()` 只切换 BN/Dropout 行为（用统计量、关 Dropout），`no_grad` 只关梯度记录。**训练和推理都要 `model.eval()` 前记得 `model.train()` 切换回去。**

### 3.5 冻结参数与 requires_grad

多模态微调中大量使用"冻结视觉编码器、只训练投影层"：

```python
for param in model.vision_encoder.parameters():
    param.requires_grad = False

# 只把 requires_grad=True 的参数传给优化器
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
```

注意：`requires_grad=False` 的参数**不会出现在 optimizer 中时完全不更新**；如果传入 optimizer 但 requires_grad=False，也会被跳过。冻结的收益：省显存（不保存其激活梯度）、省算力、防过拟合、保持预训练特征。

---

## 四、Dataset 与 DataLoader

### 4.1 三个组件的分工

| 组件 | 职责 |
|------|------|
| `Dataset` | 定义"一条样本长什么样"（索引 → 样本） |
| `DataLoader` | 负责取 batch、打乱、多进程预取 |
| `Sampler` | 控制取样本的顺序（随机、顺序、按权重） |

### 4.2 自定义 Dataset 完整模板

```python
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        path = self.file_list[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': 0}

    def __len__(self):
        return len(self.file_list)
```

必须实现两个方法：
- `__len__`：数据集大小，DataLoader 据此计算 batch 数量；
- `__getitem__`：按索引返回一条样本。

### 4.3 DataLoader 核心参数

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # 每个 epoch 打乱（train 用 True，val 用 False）
    num_workers=4,         # 数据加载子进程数（IO 密集任务提速明显）
    pin_memory=True,       # 锁页内存，GPU 拷贝更快
    drop_last=False,       # 最后一个不足 batch_size 的 batch 是否丢弃
    collate_fn=None,       # 自定义如何把样本拼成 batch
    prefetch_factor=2,     # 预取多少批（配合 num_workers）
)
```

**collate_fn 什么时候必须自定义？**
当 `__getitem__` 返回的样本长度不一（如变长文本、不同尺寸图像）时，默认的拼接会失败，需要自己实现"填充到统一长度再拼接"。

```python
def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    # 文本长度不一 → padding 到 batch 内最长
    max_len = max(b['input_ids'].size(0) for b in batch)
    input_ids = torch.stack([F.pad(b['input_ids'], (0, max_len - b['input_ids'].size(0)), value=0) for b in batch])
    return {'images': images, 'input_ids': input_ids, 'labels': torch.tensor([b['label'] for b in batch])}
```

### 4.4 DataLoader 的并发机制（面试高频）

- `num_workers > 0` 时，主进程外的 worker 子进程并行执行 `__getitem__`，主进程做 batch 组装；
- **苹果 MPS 训练注意**：`pin_memory` 对 CPU→GPU 拷贝有效，MPS 下部分场景建议 `pin_memory=False`；
- worker 数不是越大越好，一般 = CPU 核数；太多会因 GIL 争抢、IPC 开销反而变慢；
- worker 内不要做 GPU 操作（会崩）。

---

## 五、nn.Module：模型构建

### 5.1 基本用法

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = MLP(768, 1024, 10)
```

关键机制：
- 子模块注册到 `self.xxx` 属性后，自动进入 `model.parameters()` 和 `model.state_dict()`；
- 只用局部变量、不赋给 self 的模块**不会被注册**（无法自动管理）；
- `nn.Sequential` 按顺序堆叠模块，适合流水线式结构。

### 5.2 常用层速查

| 层 | 作用 | 参数 |
|----|------|------|
| `nn.Linear(in, out)` | 全连接：y = xWᵀ + b | 权重 [out, in] |
| `nn.Conv2d(Cin, Cout, k, s, p)` | 卷积 | 权重 [Cout, Cin, k, k] |
| `nn.Embedding(vocab, dim)` | 词表→向量 | 权重 [vocab, dim] |
| `nn.LayerNorm(dim)` | 层归一化（NLP/ViT 常用） | gamma/beta |
| `nn.BatchNorm2d(C)` | 批归一化（CNN 常用） | gamma/beta + 统计量 |
| `nn.Dropout(p)` | 随机置零 | p 是丢弃率 |
| `nn.MultiheadAttention` | 多头注意力封装 | num_heads |
| `nn.TransformerEncoderLayer` | 标准 Transformer 块 | d_model |

### 5.3 train / eval / 梯度检查

```python
model.train()      # 训练模式：BN 用 batch 统计量、Dropout 生效
model.eval()       # 评估模式：BN 用累计统计量、Dropout 关闭
model(x)           # 调用 forward

# 梯度检查（调试必备）
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.abs().mean())
```

### 5.4 参数统计与显存估算

```python
total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total / 1e6:.2f} M")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable / 1e6:.2f} M")
```

> **面试点**：参数量 1 亿 ≈ 100M ≈ 0.1B。FP32 每参数 4B，FP16/BF16 每参数 2B。训练时显存构成：权重 + 优化器状态（Adam 每个参数多存 2 份 float32 状态）+ 梯度 + 激活值。**推理只占"权重 + 激活"两部分。**

---

## 六、损失函数与优化器

### 6.1 常见损失函数

| 损失 | 用途 | 公式要点 |
|------|------|---------|
| `CrossEntropyLoss` | 多分类 | `-log(softmax(logits)[label])`，内部已含 softmax |
| `BCEWithLogitsLoss` | 二分类/多标签 | `-[(y·log σ(z)) + (1-y)·log(1-σ(z))]`，内部已含 sigmoid |
| `MSELoss` | 回归 | 均方误差 |
| `L1Loss` | 回归 | 平均绝对误差 |
| `KLDivLoss` | 分布匹配 | KL 散度 |
| `TripletMarginLoss` | 对比/度量学习 | 锚点-正样本-负样本 |
| `InfoNCE / ContrastiveLoss` | 对比学习 | 见 04_对比学习与CLIP |

**重要细节**：`CrossEntropyLoss` 与 `BCEWithLogitsLoss` 的参数是**原始 logits**，不要在传入前先 `softmax/sigmoid`——框架内部已经做了（数值上更稳定）。

```python
# 正确
loss = nn.CrossEntropyLoss()(logits, labels)      # logits: [N, C]
loss = nn.BCEWithLogitsLoss()(logits, labels)     # logits: [N]
# 错误（数值不稳定、梯度错误）
loss = F.cross_entropy(F.softmax(logits, dim=-1), labels)
```

### 6.2 优化器对比

| 优化器 | 特点 | 适用 |
|--------|------|------|
| SGD + momentum | 泛化好、调参讲究 | 经典 CNN 训练 |
| Adam | 自适应学习率、收敛快 | 通用默认 |
| AdamW | Adam + **权重衰减解耦**（decoupled weight decay） | **Transformer/大模型事实标准** |
| Adafactor | 省显存（8-bit 状态） | 超大模型 |
| Lion / Sophia | 近年大模型训练新选择 | 大模型预训练 |

**为什么 Transformer 用 AdamW 而不是 Adam？**
- Adam 把权重衰减实现为 L2 正则，会与 Adam 的自适应梯度（二阶矩归一化）耦合，导致大权重衰减过小、小权重衰减过大；
- AdamW 把 weight decay 单独作用在参数上（`w = w - lr·λ·w`），不经过一阶/二阶矩，更干净，已被证明对 Transformer 类模型更优。

### 6.3 学习率调度器

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# 或 warmup + 线性衰减（Transformer 标准做法）
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

# 每个 epoch 后 step 一次
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()
```

warmup 为什么必要？训练初期参数远离最优点，梯度噪声大，用大学习率容易震荡甚至发散；小学习率起步，模型先稳定下来再提速。

---

## 七、完整训练循环模板

### 7.1 单卡完整流程

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP(768, 1024, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # 必须切回训练模式
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in dataloader:
        images, labels = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), total_correct / total

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        logits = model(images)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_acc = evaluate(model, val_loader, device)
    print(f"Epoch {epoch}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
```

### 7.2 混合精度训练（AMP）

多模态训练中几乎必开 FP16/BF16 混合精度。核心思想：**前向和反向用 FP16 算，权重和优化器状态用 FP32 存，梯度从 FP16 缩放后再回 FP32**。

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():                       # FP16 前向
        logits = model(images)
        loss = criterion(logits, labels)

    scaler.scale(loss).backward()          # 梯度先放大，防 FP16 下溢
    scaler.step(optimizer)                 # 若梯度溢出则跳过本步
    scaler.update()
```

为什么需要 `GradScaler`？FP16 的最小正数约 6e-5，小梯度会**下溢为 0**。先把 loss 放大 2^16 再反传，梯度也相应放大，回 FP32 前再缩小。

FP16 vs BF16 对比：

| 属性 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 位（范围小） | 8 位（范围同 FP32） |
| 尾数位 | 10 位（精度较高） | 7 位（精度低） |
| 下溢风险 | 有，需要 GradScaler | 几乎没有 |
| 精度要求高的训练 | 需小心 | 更稳 |
| 硬件支持 | CUDA 全部 | Ampere 及以后、MPS |

> **面试点**：大模型（LLaMA/Qwen 等）预训练和微调事实标准是 **BF16**，因为范围大不需要 scaler，且梯度不会下溢。

### 7.3 MPS（Apple Silicon）训练注意

```python
device = torch.device('mps')
x = x.to(device)
```

- MPS 支持大多数算子，但部分操作（如某些 index 操作、`torch.linalg` 部分函数）不支持或慢；
- `num_workers>0` 的 DataLoader + MPS 有已知问题，可设 `num_workers=0` 或 `pin_memory=False`；
- 大 batch 训练 MPS 明显慢于 N 卡，学习用小 batch 即可。

---

## 八、模型保存与加载

### 8.1 两种保存方式

```python
# 方式一：只保存权重（推荐）
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# 方式二：保存整个模型（不推荐：耦合类定义、文件大）
torch.save(model, 'model.pt')
model = torch.load('model.pt')
```

推荐 `state_dict` 的原因：
- 只存参数，文件小；
- 不依赖类定义位置，跨环境加载安全；
- 兼容性最好（新版本框架改动不影响）。

### 8.2 多模态项目中的保存细节

```python
# 同时保存优化器状态（用于恢复训练）
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'best_acc': best_acc,
}, 'checkpoint.pth')

# 恢复训练
ckpt = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

### 8.3 权重文件格式

| 格式 | 说明 |
|------|------|
| `.pth` / `.pt` | PyTorch 原生 |
| `.safetensors` | HuggingFace 新标准，无 pickle 反序列化安全风险、加载快 |
| `.onnx` | ONNX 通用格式 |
| `.engine` | TensorRT 专属格式 |

> 部署方向：PyTorch 权重 → `torch.onnx.export` → ONNX → TensorRT/ONNXRuntime 推理。详见 15_推理优化与部署。

---

## 九、训练技巧与问题排查

### 9.1 过拟合与正则化

| 手段 | 原理 | 使用注意 |
|------|------|---------|
| Dropout | 随机丢弃神经元，防共适应 | 训练 0.1~0.5，eval 自动关闭 |
| Weight Decay | 参数范数惩罚 | AdamW 用 `weight_decay=0.01~0.1` |
| 数据增强 | 增加样本多样性 | 图像：翻转/裁剪/颜色抖动；文本：回译 |
| Early Stopping | val loss 不降即停 | 配合 best model 保存 |
| Label Smoothing | 标签软化，防过度自信 | `CrossEntropyLoss(label_smoothing=0.1)` |
| 冻结底层/小模型 | 降低模型容量 | 微调时冻结视觉塔 |

### 9.2 loss 为 NaN/Inf 的排查清单

1. **学习率过大** → 调小 lr（最常见）；
2. **FP16 溢出** → 换 BF16 或减小 batch；
3. **log 了 0 或负数** → `logits` 里加 epsilon（如 `log(p + 1e-8)`）；
4. **数据问题** → 检查标签是否有 NaN、图像是否有全黑/损坏文件；
5. **梯度爆炸** → `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`；
6. **初始化问题** → 换更小的初始化范围或改初始化策略。

```python
# 梯度裁剪（Transformer 训练必备）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 9.3 显存不足（OOM）的解法优先级

| 手段 | 效果 |
|------|------|
| 减小 batch size | 最直接 |
| 梯度累积 | 等效 batch 不变 |
| `torch.cuda.empty_cache()` | 释放缓存碎片（治标） |
| FP16/BF16 | 激活减半 |
| 梯度检查点 `checkpoint_activations` | 用算力换显存（重算激活） |
| 冻结部分参数 | 省梯度与优化器状态 |
| `find_unused_parameters=True` | DDP 时允许未使用参数 |

### 9.4 代码调试三板斧

```python
torch.manual_seed(42)          # 固定随机种子，复现结果
np.random.seed(42)
torch.backends.cudnn.deterministic = True   # 确定性卷积

# 单 step 调试
model.train()
loss = model(batch)
loss.backward()
print({n: p.grad.abs().mean().item() for n, p in model.named_parameters() if p.grad is not None})
# 如果某层梯度为 0 → 该层可能没参与前向，或 requires_grad=False
```

---

## 十、分布式训练 DDP

### 10.1 DDP 原理（一句话）

DDP（Distributed Data Parallel）= 每个 GPU 一份完整模型副本 + 各自处理一部分 batch + **梯度 AllReduce 求平均** + 同步更新。

关键点：
- 数据并行：每卡独立前向/反向；
- 梯度通信：每步反向后 all-reduce 所有卡梯度求和取平均；
- 参数同步：因为梯度一致，各卡参数天然一致；
- 相比 DP（单进程多线程）：DDP 每卡一个进程，无 GIL 争抢，通信只在梯度层面（稀疏通信），性能更好。

### 10.2 启动方式

```bash
# torchrun 启动（推荐）
torchrun --nproc_per_node=4 train.py
```

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup():
    dist.init_process_group('nccl')          # 初始化进程组
    torch.cuda.set_device(local_rank)

def main():
    setup()
    model = model.to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # 数据并行采样
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)       # 每个 epoch 换打乱顺序
        train_one_epoch()
        if dist.get_rank() == 0:             # 只在主卡保存/打印
            torch.save(model.module.state_dict(), 'model.pth')
```

### 10.3 多模态训练的分布式坑

- **视觉塔/文本塔参数同步**：DDP 包裹整个模型即可；
- **`find_unused_parameters`**：如果模型有部分参数不参与某些输入的 forward（如文本塔在纯图像任务中不更新），DDP 会报错，需设置 `find_unused_parameters=True`；
- **save/load**：只在 rank 0 保存；加载时 `load_state_dict` 用 `model.module.state_dict()`；
- **数据平衡**：不同卡数据分布差异大时，用 `DistributedSampler` 保证每卡独立随机，配合 `set_epoch`。

---

## 十一、HuggingFace Trainer（工程主流）

多模态项目中常用 HF `Trainer` 替代手写循环（见项目 04_SigLIP/train.py）：

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir='./outputs',
    per_device_train_batch_size=32,     # 每卡 batch
    learning_rate=1e-4,
    num_train_epochs=40,
    fp16=True,                          # 混合精度
    gradient_accumulation_steps=8,      # 梯度累积
    save_steps=2000,
    save_total_limit=5,
    logging_steps=100,
    dataloader_num_workers=1,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=MyDataCollator(tokenizer),
)
trainer.train()
```

Trainer 自动处理：batch 组装、梯度累积、AMP、断点续训、日志、保存 best model 等，是工程落地的标准选择。

---

## 十二、高频面试问答

**Q1：nn.Module 的 forward 和 __call__ 有什么关系？**
`model(x)` 触发 `__call__`，内部先执行 hooks，再调用 `forward(x)`。所以 `model(x)` 能自动管理 train/eval 模式与 hooks；直接调 `model.forward(x)` 会绕过部分机制（不推荐）。

**Q2：为什么推理要 no_grad？eval() 和 no_grad 的区别？**
`no_grad` 不构建计算图、不存中间激活 → 省显存提速。`eval()` 切换 BN/Dropout 行为。两者独立：训练时为了统计验证集 loss 仍需 no_grad；推理时通常两个都开。

**Q3：梯度消失和梯度爆炸怎么解决？**
梯度消失：换 ReLU/GELU 激活、残差连接、LayerNorm/BatchNorm、初始化（Xavier/Kaiming）、残差预归一化。梯度爆炸：梯度裁剪、降低 lr、BatchNorm。Transformer 系列正是靠"残差 + LayerNorm + GELU"解决了深层训练问题。

**Q4：学习率太大/太小分别什么现象？**
太大：loss 震荡不降、直接发散 NaN。太小：loss 下降极慢、收敛到次优。判断标准：打印每个 step 的 loss 变化曲线，前几十 step 应稳步下降。

**Q5：BatchNorm 和 LayerNorm 的区别？**
BN 对**样本维度**归一化（一个 batch 内、同一通道的所有样本），需要统计 batch 统计量，与 batch size 相关、受 batch 分布影响；LN 对**特征维度**归一化（每个样本内部的所有特征），与 batch size 无关。NLP/ViT/Transformer 用 LN（变长序列、小 batch 更稳），CNN 常用 BN。

**Q6：param.grad 为 None 可能的原因？**
`requires_grad=False`、该参数未参与前向、参数由 `no_grad` 创建、DDP 下未设置 find_unused_parameters。

**Q7：如何估算模型推理显存？**
权重字节数 + 单条样本前向激活峰值。7B 模型 BF16 权重约 14GB；单张 224 图像 ViT 激活几 MB~几百 MB。实际用 `torch.cuda.max_memory_allocated()` 实测最准。

**Q8：训练时 loss 一直不降怎么办？**
先跑 1 个 batch 过拟合测试（loss 应能降到接近 0）→ 确认模型前向/反向正确；再检查数据（打乱、标注错误、归一化错误）；然后调 lr、换优化器；最后检查是不是 loss 定义问题（如 log 为 0、softmax 位置错误）。

---

## 十三、自我检验

- [ ] 能说清 view / reshape / permute / contiguous 的区别
- [ ] 能讲明白 autograd 动态计算图和链式法则
- [ ] 知道为什么每步要 zero_grad、梯度累积的原理
- [ ] 能写出自定义 Dataset + collate_fn
- [ ] 能解释 train/eval/no_grad 三者区别
- [ ] 知道为什么 Transformer 用 AdamW + warmup + 梯度裁剪
- [ ] 理解 AMP 的原理和 GradScaler 的作用
- [ ] 知道 FP16 和 BF16 的区别及适用场景
- [ ] 能估算模型参数/显存
- [ ] 能说出 DDP 原理和常见坑
- [ ] 掌握 loss=NaN、OOM、loss 不降的排查流程
