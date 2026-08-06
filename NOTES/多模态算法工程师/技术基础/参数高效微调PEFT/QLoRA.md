# QLoRA：4-bit 量化基座 + LoRA（单卡微调利器）

> 本模块索引见 [参数高效微调PEFT详解](参数高效微调PEFT详解.md)

## 一、定义与公式

### 1.1 一句话定义

QLoRA（Dettmers et al., 2023）= **NF4 4-bit 量化基座（完全冻结） + bf16 训练的 LoRA 分支 + 双重量化 + 分页优化器**。目标：把 7B 模型的全参微调显存从 ~140GB 压到 **~16GB**，消费级单卡即可微调。

### 1.2 分块对称量化公式（量化与反量化）

对权重张量按 $B$ 个元素一组（block，通常 $B=64$）分块，每块独立做对称量化：

$$x_q = \left\lfloor \frac{x}{s} \right\rceil, \qquad \hat{x} = s \cdot x_q, \qquad s = \frac{\max_{i \in \text{block}} |x_i|}{q_{\max}}$$

- $x_q$：整数码（4-bit，0~15）；
- $s$：每块的放缩因子（fp32），量化时"向下取整/就近取整"（$\lfloor \cdot \rceil$ 表示 round-to-nearest）；
- $\hat{x}$：反量化后的近似权重，**计算时用它替换原权重**。

反量化发生在**每次前向/反向之前**，量化只影响存储，计算仍是 bf16 精度。

### 1.3 NF4：面向正态分布的 4-bit 量化

NF4（NormalFloat4）不再均匀铺满 $[-1, 1]$，而是把 16 个档位放在**标准正态分布的分位点**上：

$$b_i = \Phi^{-1}\!\left(\frac{i}{16}\right),\quad i = 0,\dots,16, \qquad
q_i = \mathbb{E}\big[\, z \mid b_i < z \le b_{i+1} \,\big],\quad z \sim \mathcal{N}(0,1)$$

其中 $\Phi^{-1}$ 是标准正态的逆累积分布函数（分位函数）。每个档位 $q_i$ 是"落在该概率区间内的正态变量的期望"，即该区间的代表性数值。这样做的效果：

- 每 1/16 概率质量对应一个档位，**中央密集、两端稀疏**——和权重分布一致；
- 在"权重来自正态分布"的假设下，这是信息论意义上**均方量化误差最小**的定点格式；
- 16 个档位具体取值（bitsandbytes 实现）：$-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0, 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0$。

> 对比：均匀 4-bit 把 16 个档位等距分布在 $[-1,1]$，两端大量档位几乎用不到、中间精度不足；NF4 在正态假设下误差更小（数值对比见 3.2）。

### 1.4 双重量化（Double Quantization）公式

第一层量化产生的每块放缩因子 $s$ 本身是 fp32（每 64 个权重 +4 字节 ≈ +0.5 bit/权重）。把 $s$ 集合再按块做一次 8-bit 量化：

$$s_q = \left\lfloor \frac{s}{s^{(2)}} \right\rceil \in [-127, 127], \qquad s^{(2)} = \text{第二层的块放缩因子（fp32，每 256 个 } s \text{ 一个）}$$

比特账本（每权重）：

$$\text{bits} = \underbrace{4}_{\text{主权重}} + \underbrace{\frac{8}{64}}_{\text{第一层 } s} + \underbrace{\frac{32}{64 \times 256}}_{\text{第二层 } s^{(2)}} = 4 + 0.125 + 0.002 \approx 4.127\ \text{bit}$$

7B 模型权重存储：$4.127 \times 7 \times 10^9 / 8 \approx 3.6$GB（fp16 是 14GB，省 75%）。

## 二、核心原理

### 2.1 为什么 4-bit 训练不会崩（必考）

1. **基座完全冻结**：量化权重不参与梯度更新，量化误差只会出现在前向的冻结部分，不影响可学习参数（LoRA）的梯度质量；
2. **计算时反量化回 bf16**：4-bit 只是存储格式，每次前向先还原为 bf16 再算矩阵乘，数值范围大、误差可控；
3. **NF4 针对正态权重最优**：量化误差期望最小，实测误差远小于均匀 4-bit；
4. **LoRA 分支保持 bf16 高精度**：真正"学新东西"的部分不量化，微调信号质量有保证。

论文结论：QLoRA 4-bit 微调效果与 16-bit LoRA **几乎持平**，部分任务甚至更好（量化带来轻微正则化）。

### 2.2 NF4 vs 均匀 4-bit：误差来源

均匀量化在权重最集中的 $[-0.3, 0.3]$ 区间只有约 4~5 个档位可用，分辨率粗糙；NF4 在这个区间有约 8 个档位。量化误差近似为：

$$\mathbb{E}\!\left[(x - \hat{x})^2\right] \approx \sum_{i} p_i \cdot \text{Var}(x \mid b_i < x \le b_{i+1})$$

分位量化使每个区间概率质量相等（$p_i = 1/16$），对正态分布最小化上式。3.2 的数值实验给出：同样 4-bit，NF4 的 RMSE 比均匀量化低约 12%。

### 2.3 双重量化为什么值得做

| 方案 | 每权重额外比特 | 7B 额外显存 |
|------|---------------|------------|
| 不量化放缩因子（fp32/64 权重） | +0.5 bit | +0.44GB |
| 双重量化（8-bit/64 + fp32/16384） | +0.127 bit | +0.11GB |

省约 0.37 bit/权重 ≈ 7B 模型再省 330MB，且**不引入任何额外误差来源**（误差只发生在两层放缩因子上，对精度影响可忽略）。

### 2.4 分页优化器（Paged Optimizers）

- 问题：训练中 Adam 状态（m、v）在显存与 CPU 内存之间搬移时，偶尔出现瞬时显存尖峰导致 OOM；
- 方案：用 **CUDA Unified Memory（统一内存）** 管理优化器状态，GPU 显存不足时自动分页换页到 CPU RAM，避免 OOM 崩溃；
- 收益：同样的 batch size 下更稳；显存压力大时自动降级而不是直接失败。

## 三、源码实现

### 3.1 手写 NF4 量化 / 反量化（CPU 可运行）

```python
import torch

# bitsandbytes 的 NF4 档位表（16 个值，[-1, 1]）
NF4_LEVELS = torch.tensor([
    -1.0000, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.18477343,
    -0.09105003, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0000,
])

def quantize_nf4(x, block_size=64):
    """分块 NF4 量化：每块一个 absmax 放缩因子，返回 4-bit 码 + 放缩因子"""
    n = x.numel()
    pad = (-n) % block_size
    xp = torch.cat([x.flatten(), torch.zeros(pad, dtype=x.dtype)])
    blocks = xp.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)     # 每块放缩因子 s
    xn = (blocks / absmax[:, None]).clamp(-1.0, 1.0)       # 归一化到 [-1, 1]
    q = torch.abs(xn[:, :, None] - NF4_LEVELS).argmin(dim=-1).to(torch.uint8)
    return q, absmax, n                                    # 4-bit 码、s、原始长度

def dequantize_nf4(q, absmax, n, block_size=64):
    """反量化：查档位表 × 放缩因子（每次前向计算前执行）"""
    levels = NF4_LEVELS.to(q.device)[q.long()].float()     # 4-bit 码查表
    out = (levels * absmax[:, None]).reshape(-1)
    return out[:n]

# 自检：量化-反量化闭环
torch.manual_seed(0)
w = torch.randn(4096)                                      # 模拟一层权重（近似正态）
q, absmax, n = quantize_nf4(w)
w_hat = dequantize_nf4(q, absmax, n)
rmse = ((w - w_hat) ** 2).mean().sqrt().item()
print(f"NF4 量化 RMSE = {rmse:.5f}")                       # ~0.091（标准差 1 的权重）
```

### 3.2 与均匀 4-bit 的误差对比（数值实验）

```python
def quantize_4bit_uniform(x, block_size=64):
    """均匀 4-bit：把 [-1,1] 均匀切成 16 档（对照用）"""
    n = x.numel()
    pad = (-n) % block_size
    xp = torch.cat([x.flatten(), torch.zeros(pad, dtype=x.dtype)])
    blocks = xp.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    xn = (blocks / absmax[:, None]).clamp(-1.0, 1.0)
    q = ((xn + 1) / 2 * 15).round().to(torch.uint8)
    return q, absmax, n

def dequantize_4bit_uniform(q, absmax, n, block_size=64):
    levels = (q.float() / 15 * 2 - 1)
    return (levels * absmax[:, None]).reshape(-1)[:n]

torch.manual_seed(0)
w = torch.randn(4096)
q, absmax, n = quantize_nf4(w)
wq = dequantize_nf4(q, absmax, n)
q2, absmax2, n2 = quantize_4bit_uniform(w)
wu = dequantize_4bit_uniform(q2, absmax2, n2)
rmse = lambda a, b: ((a - b) ** 2).mean().sqrt().item()
print(f"NF4     RMSE = {rmse(w, wq):.5f}")    # ~0.091
print(f"uniform RMSE = {rmse(w, wu):.5f}")    # ~0.104（NF4 低约 12%）
```

### 3.3 双重量化 + 比特账本

```python
def quantize_8bit(x, block_size=256):
    """8-bit 分块量化：用于对放缩因子 s 做第二次量化"""
    n = x.numel()
    pad = (-n) % block_size
    xp = torch.cat([x.flatten(), torch.zeros(pad, dtype=x.dtype)])
    blocks = xp.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    xn = (blocks / absmax[:, None]).clamp(-1.0, 1.0)
    q = (xn * 127).round().to(torch.int8)
    return q, absmax, n

def dequantize_8bit(q, absmax, n, block_size=256):
    return (q.float() / 127 * absmax[:, None]).reshape(-1)[:n]

torch.manual_seed(0)
w = torch.randn(1 << 20)                        # 100 万权重，避免 padding 干扰账本
q, absmax, n = quantize_nf4(w)
q2, s2, n2 = quantize_8bit(absmax)              # 第一层的 s 再做 8-bit 量化
absmax_hat = dequantize_8bit(q2, s2, n2)        # 反量化回近似的 s
wq2 = dequantize_nf4(q, absmax_hat, n)
rmse = lambda a, b: ((a - b) ** 2).mean().sqrt().item()
print(f"双重量化 RMSE = {rmse(w, wq2):.5f}")    # ~0.092，与单层量化几乎无差
bits = (q.numel() * 4 + q2.numel() * 8 + s2.numel() * 32) / n
print(f"每权重比特 = {bits:.3f}")               # 4.127
print(f"7B 权重存储 = {bits * 7e9 / 8 / 1e9:.2f} GB")   # 3.61 GB
```

### 3.4 手写 QLoRA 训练：NF4 基座 + LoRA vs bf16 基座 + LoRA

```python
import torch.nn as nn

class NF4Linear(nn.Module):
    """以 NF4 存储的冻结线性层：前向时反量化回 bf16 计算"""

    def __init__(self, in_f, out_f, block_size=64):
        super().__init__()
        torch.manual_seed(1)
        w = torch.randn(out_f, in_f) * (in_f ** -0.5)
        q, absmax, n = quantize_nf4(w.flatten(), block_size)
        self.register_buffer("q", q)              # 4-bit 码（存储大头）
        self.register_buffer("absmax", absmax)    # 放缩因子
        self.n = n
        self.block_size = block_size
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        w = dequantize_nf4(self.q, self.absmax, self.n,
                           self.block_size).view(-1, x.shape[-1])
        return x @ w.T + self.bias                # 反量化后照常做 bf16 矩阵乘

class QLoraToy(nn.Module):
    """玩具 QLoRA：NF4/bf16 基座 + LoRA 分支（d=32, r=8）"""

    def __init__(self, d, vocab, nf4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.base = NF4Linear(d, d) if nf4 else nn.Linear(d, d)
        self.lora_A = nn.Parameter(torch.randn(8, d) * 0.02)   # LoRA：高斯
        self.lora_B = nn.Parameter(torch.zeros(d, 8))          # LoRA：零
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.embed(x)
        h = self.base(h) + (h @ self.lora_A.t()) @ self.lora_B.t()  # 基座 + LoRA
        return self.head(h)

def build(d, vocab, nf4):
    m = QLoraToy(d, vocab, nf4)
    for p in m.parameters():
        p.requires_grad = False                  # 冻结基座
    m.lora_A.requires_grad = m.lora_B.requires_grad = True
    m.head.weight.requires_grad = m.head.bias.requires_grad = True
    return m

torch.manual_seed(0)
x = torch.randint(0, 64, (16, 10)); y = torch.randint(0, 64, (16, 10))
crit = nn.CrossEntropyLoss()
for nf4, name in [(False, "bf16 基座 + LoRA"), (True, "NF4 基座 + LoRA")]:
    m = build(32, 64, nf4)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    for step in range(100):
        opt.zero_grad()
        loss = crit(m(x).reshape(-1, 64), y.reshape(-1))
        loss.backward(); opt.step()
    print(f"{name}: 100 步后 loss = {loss.item():.3f}")
# bf16 基座 + LoRA: 100 步后 loss = 1.532
# NF4 基座 + LoRA: 100 步后 loss = 1.380  （两者基本持平，量化没有拖累训练）
```

### 3.5 bitsandbytes 实战（需要 CUDA GPU）

```python
# 需 GPU 与 pip install bitsandbytes
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 加载基座
    bnb_4bit_quant_type="nf4",              # NF4 量化（可选 fp4）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度：反量化后 bf16
    bnb_4bit_use_double_quant=True,         # 双重量化（省 ~0.37bit/权重）
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",                      # 自动切分到 GPU
)

# 基座已 4-bit 冻结，再用 peft 加 LoRA 即可训练
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()     # trainable%: ~0.10
```

## 四、参数与显存账本（7B：140GB → 16GB）

### 4.1 比特账本

| 项目 | 计算 | 显存 |
|------|------|------|
| 基座权重（NF4 + 双重量化） | 4.127 bit × 7e9 / 8 | **3.6GB** |
| LoRA 分支（bf16，r=8 全注入 20M） | 20M × 2B | 0.04GB |
| LoRA 梯度 + Adam 状态（fp32×2） | 20M × 4B × 3 | 0.24GB |
| 激活值（batch 小 + 梯度检查点） | 与 batch/seq 相关 | ~10GB |
| **训练峰值（合计）** | — | **~16GB（16G 卡可跑）** |

### 4.2 三种方案对比

| 显存构成 | 全参微调 | LoRA（bf16） | QLoRA（NF4） |
|----------|---------|--------------|--------------|
| 基座权重 | 14GB（bf16+梯度） | 14GB（bf16） | **3.6GB（4-bit）** |
| 可训练参数 | 14GB | 40MB | 40MB |
| 梯度 | 14GB | 40MB | 40MB |
| Adam 状态 | 56GB | 160MB | 160MB |
| 激活值 | 大 | 大 | 大 |
| **峰值（量级）** | **~140GB** | **~20-30GB** | **~16GB** |
| 单卡 | 需多卡/并行 | 24G 卡 | **16G 卡** |

> 关键认知：QLoRA 省的是**存储与显存**，不是计算量——矩阵乘仍是 bf16 全尺寸运算，训练速度与 LoRA 相近（反量化有少量开销）。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 7B 微调显存降至 ~16GB，消费级单卡可行 | 量化-反量化带来少量计算开销与数值噪声 |
| 效果与 16-bit LoRA 几乎持平（论文实验） | 基座冻结，无法适配需要动基座的场景 |
| NF4 + 双重量化在正态权重上误差最小 | 依赖 bitsandbytes/CUDA，CPU 不可用 |
| 与 LoRA 生态完全兼容（权重格式一致） | 反量化按块进行，与某些编译/算子不兼容 |
| 分页优化器缓解 OOM 尖峰 | 4-bit 训练 ≠ 推理量化，推理仍需另做 GPTQ/AWQ |

## 六、与同类方法对比

| 方法 | 基座精度 | 训练显存（7B） | 效果 | 适用 |
|------|---------|---------------|------|------|
| 全参微调 | bf16 | ~140GB | 100% | 算力充足 |
| LoRA | bf16 | ~20-30GB | ~95-99% | 通用首选 |
| **QLoRA** | **NF4 4-bit** | **~16GB** | ~93-97%（≈LoRA） | 消费级单卡、低成本 |
| GPTQ/AWQ 等 | 4-bit | —（推理量化） | 训练无关 | 推理部署，不是训练 |

> **区别要点**：QLoRA 的 4-bit 是**训练期存储格式**（前向反量化计算、LoRA 保持 bf16）；GPTQ/AWQ 是**推理期权重压缩**（离线校准、不可训练）。两者可串联：QLoRA 训练完合并权重后，再用 GPTQ 做推理量化。

## 七、高频面试问答

**Q1：QLoRA 为什么 4-bit 训练不崩？**
基座完全冻结，量化误差只影响前向冻结部分，不污染可学习参数的梯度；计算时反量化为 bf16；真正更新的 LoRA 分支保持 bf16 高精度。误差不进入学习信号，所以不崩。

**Q2：NF4 和普通 4-bit 量化的区别？**
普通量化档位均匀分布，浪费两端；NF4 用标准正态分位数定档位，每个档位概率质量相等，在"权重近似正态"前提下均方量化误差最小。数值上 RMSE 低约 12%。

**Q3：双重量化是什么？**
把量化时每个 block 的 fp32 放缩因子 s 再做一次 8-bit 分块量化。每权重额外比特从 +0.5 降到 +0.127，误差可忽略。7B 模型再省约 330MB。

**Q4：分页优化器解决了什么问题？**
Adam 状态（m、v）在显存不足时需要换页到 CPU 内存，传统实现瞬时尖峰会 OOM。分页优化器用 CUDA Unified Memory 自动换页，显存压力大时平滑降级而不是崩溃。

**Q5：QLoRA 和 GPTQ/AWQ 的区别？**
QLoRA 是训练方法：基座 4-bit 存储 + bf16 反量化计算 + LoRA 高精度训练；GPTQ/AWQ 是推理量化：离线校准压缩权重、不可再训练。两者可以先后串联使用。

**Q6：7B 全参微调 ~140GB，QLoRA 为什么只要 ~16GB？**
全参：权重 14 + 梯度 14 + Adam 56 + 激活 ≈ 100GB+（含冗余后约 140GB）。QLoRA：权重 3.6GB(4-bit) + LoRA 参数/梯度/状态 <0.3GB + 激活 ~10GB ≈ 16GB。省掉的本质是"梯度 + 优化器状态 + 权重精度"。

**Q7：QLoRA 的 LoRA 分支能不能也量化？**
不能，也不应该。LoRA 分支是唯一被更新的部分，必须保持 bf16 精度以保证梯度质量；量化只针对冻结的基座。推理阶段则相反，可用 GPTQ/AWQ 压缩整个模型。

**Q8：QLoRA 训练出来的 LoRA 权重能直接用在 bf16 基座上吗？**
可以。LoRA 分支本身就是 bf16，与基座精度无关；部署时可直接加载到 bf16 基座或合并，不必保留 4-bit 基座。

## 八、自我检验

- [ ] 能写出分块量化公式 $x_q = \lfloor x/s \rceil$、$\hat{x} = s \cdot x_q$ 并解释每块的 s
- [ ] 能推导 NF4 分位档位 $q_i = \mathbb{E}[z \mid b_i < z \le b_{i+1}]$ 并解释为何误差最小
- [ ] 能写出双重量化比特账本：4 + 8/64 + 32/16384 ≈ 4.127 bit/权重 → 7B ≈ 3.6GB
- [ ] 能说出"4-bit 训练不崩"的四点原因（冻结、反量化、NF4、LoRA bf16）
- [ ] 能讲清分页优化器的作用（unified memory 换页防 OOM）
- [ ] 能背出全参 140GB / LoRA 20-30GB / QLoRA 16GB 的显存构成表
- [ ] 能区分 QLoRA（训练期 4-bit 存储）与 GPTQ/AWQ（推理期量化）
- [ ] 能手写 NF4 量化/反量化与"量化基座 + LoRA"的训练闭环代码
- [ ] 能写出 BitsAndBytesConfig + LoraConfig 的完整 QLoRA 训练流程
- [ ] 能回答 8 个面试追问
