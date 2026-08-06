# LoRA 初始化：A 随机、B 全零与 α/r 缩放

> 本模块索引见 [参数初始化详解](参数初始化详解.md)

## 一、定义与公式

LoRA（Low-Rank Adaptation, Hu et al. 2021）不是"从头训练"的初始化问题，而是**"微调起点"问题**：在冻结的预训练权重 $W_0 \in \mathbb{R}^{d \times d}$ 旁挂一条低秩可训练分支，让微调对 base 模型的初始行为**零破坏**。

### 1.1 LoRA 的前向公式

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

| 符号 | 形状 | 角色 |
|------|------|------|
| $W_0$ | $d \times d$ | base 权重（冻结） |
| $A$ | $r \times d$ | 把输入压到低秩空间（可训练） |
| $B$ | $d \times r$ | 把低秩表示投影回去（可训练） |
| $\frac{\alpha}{r}$ | 标量 | 整体缩放（$\alpha$ 常数，$r$ 为秩） |

### 1.2 初始化规则（LoRA 论文 §4.1 原文）

> "We use a random Gaussian initialization for A and **zero for B**, so ΔW = BA is zero at the beginning of training."

$$A \sim \mathcal{N}(0, \sigma_A^2) \quad \text{（随机高斯，} \sigma_A \text{ 取 } 0.02 \text{ 或 kaiming 风格）}, \qquad B = 0$$

**初始时 $\Delta W = BA = 0$**，LoRA 分支输出为 0，前向结果与冻结的 base 模型**逐位一致**。

### 1.3 为什么这套规则"一石三鸟"

1. **起点正确**：微调从"原模型预测"出发，不破坏预训练行为；
2. **梯度激活**：$\partial L/\partial B \ne 0$（A 随机），B 第一步就获得梯度，训练立即开始；
3. **对称性破坏**：A 的随机性保证同层不同 rank 方向有差异化的梯度信号。

## 二、数学原理

### 2.1 为什么 B 必须为零（初始 ΔW = 0）

若 A、B 都随机初始化，初始 $\Delta W = BA \ne 0$，等价于给 base 模型**叠加了一个随机噪声矩阵**。后果：

1. 训练起点偏离预训练分布——每一 batch 的梯度都要先"纠正噪声"再"学习任务"；
2. 噪声量级若不可控（与 $\alpha/r$ 相乘），微调效果对初始化敏感；
3. 对比：$B=0$ 时起点严格等于 base 模型预测，模型完全处于"已学会"的状态，只需学习增量。

**梯度不对称是设计好的**（设 $g = \partial L/\partial h$）：

$$\frac{\partial L}{\partial B} = g \, (Ax)^T \ne 0 \quad\text{（A 随机，Ax ≠ 0）}$$

$$\frac{\partial L}{\partial A} = B^T g \, x^T = 0 \quad\text{（B = 0）}$$

**第一步只有 B 更新**；B 一旦非零，A 立刻获得梯度，两者随后同步训练。这一"先动 B、后动 A"的动态完全正常，不影响收敛。

> 对称性讨论：反过来"B 随机、A 为零"同样满足 $BA = 0$，理论上等价可行，但论文约定 + 主流实现统一为 **A 随机、B 为零**（HF PEFT、Unsloth 等全部如此），面试按此回答。

### 2.2 为什么梯度第一步不为零（B=0 但训练不冻结）

虽然 $\partial L/\partial A = 0$，但 $\partial L/\partial B \ne 0$，所以**整个 LoRA 分支在第一步就参与更新**。B=0 只影响"起点"，不影响"训练性"——这与"全零初始化的网络无法训练"完全不同：

| 场景 | 初始前向 | 初始梯度 | 能训练吗 |
|------|---------|---------|---------|
| 全零初始化的网络 | 输出全 0 | 梯度全 0（W=0 → y=0 → ∂L/∂y 逐层为 0） | 不能 |
| LoRA（B=0） | 输出 = base 输出（非零） | ∂L/∂B ≠ 0（损失来自 base 预测的错误） | 能 |

关键区别：LoRA 的**旁路**为零，但**主路**（base 模型）非零——梯度从主路的 loss 反传而来，天然非零。

### 2.3 scale = α/r 的由来（更新幅度与秩解耦）

$BA$ 的每个元素是 $r$ 个 rank-1 外积项的求和：

$$(BA)_{ij} = \sum_{k=1}^{r} B_{ik} A_{kj}$$

其量级随 $r$ 增长（约 $\sqrt{r}$ 量级，独立随机项平方和）。若不缩放，**改变秩 $r$ 会直接改变 LoRA 分支的更新幅度**，等效于偷偷改了学习率。因此：

$$\text{scale} = \frac{\alpha}{r}$$

1. **解耦秩与更新幅度**：除以 $r$ 后，不同 $r$ 下初始更新量级一致，$r$ 只负责"表达空间大小"；
2. **α 是 LoRA 的学习率**：调 α 即整体缩放分支更新，与 base 学习率解耦；惯例取 $\alpha = r$（scale=1）为基线；
3. 若不用 $\alpha/r$，把 $r$ 翻倍会隐式放大更新，产生难排查的不稳定。

### 2.4 A 的尺度选择：大 or 小？

- 论文：随机高斯（未指定 σ，常用 $\sigma = 0.02$ 或与模型同尺度）；
- HF PEFT 默认：**Kaiming-uniform 风格**（`nn.Linear` 同款 a=√5，见 [Kaiming](Kaiming.md) §4.2）；
- 关键约束：A 的尺度通过"B 的梯度"影响学习率——$\partial L/\partial B = g(Ax)^T$，A 越大 B 的梯度越大。**A 过大会让 B 的更新过冲**，实践中常把 σ_A 调小（如 0.01）；
- 理论上有无影响？有，但与 $\alpha/r$ 相互耦合：工程上统一约定"A 用标准随机、B 全零、α 调学习率"即可，避免多因素纠缠。

## 三、源码实现

### 3.1 手写 A/B 初始化（对照规则逐行实现）

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)                          # 固定种子，保证注释里的输出可复现

def init_lora_ab(a, b, r, d, a_std=0.02):
    """LoRA 初始化：A 随机高斯，B 全零（论文 §4.1）"""
    nn.init.normal_(a, mean=0.0, std=a_std)   # A: N(0, 0.02²)，保证 B 第一步有梯度
    nn.init.zeros_(b)                         # B: 全零，保证初始 ΔW = BA = 0
    return a, b

d, r = 768, 8
a = torch.empty(r, d)                          # A: [r, d]
b = torch.empty(d, r)                          # B: [d, r]
init_lora_ab(a, b, r, d)
print(f"A: mean={a.mean().item():.2e}, std={a.std().item():.4f}")   # A: mean≈0.00e+00, std≈0.02（随机高斯）
print(f"B: mean={b.mean().item():.2e}, std={b.std().item():.2e}")   # B: mean=0.00e+00, std=0.00e+00（全零）
dw = b @ a                                    # ΔW = BA
print(f"初始 ΔW 全零? {(dw == 0).all().item()}")                  # 初始 ΔW 全零? True
```

### 3.2 LoRA 层完整实现（可直接运行）

```python
class LoRALayer(nn.Module):
    """标准 LoRA 层：冻结 base 权重 + A/B 可训练分支 + α/r 缩放"""

    def __init__(self, in_dim, out_dim, rank=8, alpha=8, a_std=0.02):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        # base 权重（冻结，不参与训练）
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight.requires_grad_(False)
        # LoRA 分支：A 随机高斯、B 全零
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_b = nn.Parameter(torch.empty(out_dim, rank))
        init_lora_ab(self.lora_a, self.lora_b, rank, in_dim, a_std)
        self.scale = alpha / rank             # scale = α/r

    def forward(self, x):
        base_out = x @ self.weight.T          # W0 x（冻结路径）
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T   # x·Aᵀ·Bᵀ = BAx
        return base_out + self.scale * lora_out

torch.manual_seed(0)
layer = LoRALayer(in_dim=64, out_dim=64, rank=8, alpha=8)
x = torch.randn(4, 64)
out = layer(x)
# 初始（B=0）时输出必须等于纯 base 输出
base_out = x @ layer.weight.T
print(f"初始输出 == base 输出? {torch.allclose(out, base_out, atol=1e-6)}")
# 初始输出 == base 输出? True
```

### 3.3 验证梯度不对称性（先动 B、后动 A）

```python
torch.manual_seed(0)
layer = LoRALayer(64, 64, rank=8, alpha=8)
y = layer(x).pow(2).mean()                    # 随便一个标量损失
y.backward()

ga = layer.lora_a.grad
gb = layer.lora_b.grad
print(f"第一步: A 梯度范数={ga.norm().item():.2e}, B 梯度范数={gb.norm().item():.2e}")
# 第一步: A 梯度范数=0.00e+00, B 梯度范数=1.13e-02   <- 只有 B 有梯度（∂L/∂A = Bᵀg·xᵀ = 0）

# 更新一步后（模拟 B 非零），A 立即获得梯度
with torch.no_grad():
    layer.lora_b -= 0.1 * gb
    layer.lora_a.grad = None
y2 = layer(x).pow(2).mean()
y2.backward()
ga2 = layer.lora_a.grad
print(f"第二步: A 梯度范数={ga2.norm().item():.2e}（B 非零后 A 开始训练）")
# 第二步: A 梯度范数=1.16e-04（B 非零后 A 开始训练）
```

### 3.4 完整训练演示：LoRA 微调确实在学

```python
torch.manual_seed(1)

# 目标函数: 学 y = 2x（base 模型初始输出是 y = x·W0，W0 随机）
target_w = torch.tensor([[2.0] * 8])          # 要逼近的权重
base = nn.Linear(8, 1, bias=False)
nn.init.constant_(base.weight, 0.5)           # base 固定为 0.5（模拟预训练后的冻结权重）
base.weight.requires_grad_(False)

lora = LoRALayer(8, 1, rank=4, alpha=8)
opt = torch.optim.AdamW(lora.parameters(), lr=1e-2)
xs = torch.randn(512, 8)

for step in range(300):
    opt.zero_grad()
    pred = base(xs) + lora(xs)                # 组合输出 = base + LoRA
    loss = ((pred - xs @ target_w.T) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    merged = base.weight + (lora.lora_b @ lora.lora_a) * lora.scale
print(f"loss={loss.item():.2e}, 合并权重≈{merged.flatten()[0].item():.2f}（目标 2.0）")
# loss=2.86e-06, 合并权重≈2.01（目标 2.0）   <- LoRA 分支成功把 0.5 学到接近 2.0
```

## 四、深入分析

### 4.1 B=0 与 Fixup 的关系（都是"零初始化残差"思想）

LoRA 的 B=0 与 Fixup（残差分支最后一层置零）、ReZero（可学习标量初始 0）是**同一个思想的三个实例**：让"新增的可训练路径"初始输出为零，从恒等/原模型起点出发。区别：

| 方法 | 零的位置 | 目的 |
|------|---------|------|
| Fixup | 残差分支最后一层权重 | 训练初期网络 = 恒等映射 |
| ReZero | 残差分支可学习标量 | 同时学习每层残差贡献 |
| LoRA | 低秩分支的 B | 微调起点 = 原模型预测 |

### 4.2 微调场景的"初始化三原则"

多模态微调（如给 CLIP 加 LoRA）通用：

1. **冻结的主路不动**：base 权重零更新（requires_grad=False 或 LoRA 化替换）；
2. **新增分支零影响**：新增参数初始输出为 0（LoRA B=0、新 head 配残差零初始化）；
3. **尺度由 α/学习率控制**：分支更新幅度由 α/r 与 lr 显式控制，不与秩耦合。

### 4.3 工程细节（HF PEFT 的差异）

- PEFT 默认 A 用 `nn.Linear` 同款 kaiming-uniform（a=√5），B 用 `zeros_`——与论文的"随机高斯 A"略不同，但同样满足"初始 ΔW=0"的核心约束；
- **合并与解合并**：推理时可把 $W_{merged} = W_0 + \frac{\alpha}{r}BA$ 预合并，省去分支计算；训练时解合并保持冻结语义；
- **量化组合**：QLoRA 里 base 是 4bit 量化，只有 A/B 是 BF16——初始化规则不变，B 依然全零；
- **秩的选择**：$r$ 只影响表达能力与计算量（$2 \cdot d \cdot r$ 个参数），不影响更新幅度（α/r 已解耦）。

### 4.4 常见坑

1. **忘记乘 scale**：$h = W_0x + BAx$ 而漏掉 $\alpha/r$，切换 r 时隐式改学习率；
2. **B 没置零**：初始 ΔW ≠ 0，微调起点偏离预训练分布（效果略差且难复现）；
3. **把 α 当普通超参忘了与 r 联动**：同时调 r、α 时更新幅度不变（scale=α/r），只有 α 单独调才有"调学习率"效果；
4. **对 embedding 用 LoRA 的矩阵约定混乱**：`nn.Embedding` 的 LoRA 需转成 `[vocab, dim]` 的 Linear 视角，A/B 形状别搞反。

## 五、优缺点与适用

| 优点 | 缺点 |
|------|------|
| 初始零破坏：起点 = 原模型预测 | 表达能力受秩限制（低秩假设） |
| 只训 2·d·r 参数，显存/通信开销小 | α 需要调（scale 与 lr 耦合） |
| 与 Adam 解耦：α/r 缩放稳定训练 | 秩太小时欠拟合任务 |
| 可合并、可量化、生态成熟 | 对"需要大量新知识"的任务不如全量微调 |
| 初始化规则简单（A 随机、B 零） | A 尺度选择有自由度，需保持一致 |

**适用**：LLM 指令微调、多模态对齐（CLIP/LLaVA 适配）、任何"从预训练权重出发"的低资源微调。
**不适用**：从头训练（无预训练权重可冻结，LoRA 无意义）、需要大幅扩展知识面/语言的场景（秩上限不足）。

## 六、与同类对比

| 维度 | LoRA | Adapter（Houlsby 2019） | Prefix-Tuning | Fixup/ReZero |
|------|------|------------------------|---------------|--------------|
| 新增参数 | 低秩旁路 A/B | 串接瓶颈 MLP | 前缀 key/value | 残差缩放 |
| 初始行为 | ΔW=0（B=0） | 旁路非零（需残差连接） | 零初始化前缀 | 恒等映射 |
| 参数占比 | 2dr / d² | 约 d² 级（按层） | 序列长度级 | 0（仅初始化方案） |
| 推理开销 | 可合并为 0 | 恒有额外计算 | 序列更长 | 0 |
| 共同点 | 都在"冻结主体、最小改动"范式下，都依赖"初始零影响"原则 |

## 七、高频面试问答

**Q1：LoRA 为什么 B 初始化全零？**
保证初始 $\Delta W = BA = 0$，LoRA 分支输出为零，微调严格从原模型预测出发、不破坏 base 行为。若 A、B 都随机，初始 ΔW≠0 相当于给 base 叠加随机噪声，训练起点偏离预训练分布。

**Q2：B=0 时 A 的梯度也是 0，网络第一步岂不是没更新？**
$\partial L/\partial A = B^Tgx^T = 0$，但 $\partial L/\partial B = g(Ax)^T \ne 0$（A 随机）——**B 第一步就更新**，B 非零后 A 立即获得梯度。这与"全零初始化的网络"本质不同：全零网络主路输出也是 0 导致梯度全零，而 LoRA 的主路（base）非零，loss 梯度天然存在。

**Q3：scale=α/r 为什么存在？**
$(BA)_{ij} = \sum_k B_{ik}A_{kj}$ 的量级随 r 增长（~√r），不除以 r 则切换秩会隐式改变更新幅度（等效改学习率）。除以 r 后更新量级与秩解耦，α 承担"LoRA 学习率"角色。

**Q4：A 初始化用高斯还是 kaiming？**
论文用随机高斯（σ≈0.02）；HF PEFT 用 kaiming-uniform（a=√5）。两者都满足核心约束"B=0 → 初始 ΔW=0"，效果无实质差异。真正重要的是 A 的**尺度一致性**——A 越大，B 的梯度 $\partial L/\partial B = g(Ax)^T$ 越大。

**Q5：LoRA 的 r 变大，更新幅度会变大吗？**
不会。scale=α/r 已把更新幅度归一化；r 变大只增加表达能力与参数量（2dr），更新量级不变。若不用 α/r，r 翻倍会使更新约放大 √2 倍。

**Q6：B 随机、A 为零可行吗？**
数学上对称可行（同样满足 BA=0），但论文约定与主流实现（PEFT）统一为 A 随机、B 零。面试说"理论可行、约定如此"即可。

**Q7：LoRA 能合并回 base 权重吗？推理开销？**
能：$W_{merged} = W_0 + (\alpha/r)BA$，训练后直接替换，推理零额外开销。这也是 LoRA 相比 Adapter/Prefix 的优势。

**Q8：LoRA 与 Fixup/ReZero 有什么关系？**
同一思想的不同实例：让新增可训练路径初始输出为零（LoRA B=0 / Fixup 残差末层置零 / ReZero 标量初始 0），从"原模型（恒等）"起点出发，随后梯度逐步激活分支。

## 八、自我检验

- [ ] 能写出 LoRA 公式 $h = W_0x + (\alpha/r)BAx$ 及 A、B 的形状
- [ ] 能完整回答"为什么 B=0"（初始 ΔW=0 + 梯度不对称 + 对比全零初始化）
- [ ] 能推导 $(BA)_{ij} = \sum_k B_{ik}A_{kj}$ 的量级随 r 增长 → α/r 解耦的动机
- [ ] 能说清"第一步只有 B 更新、随后 A 获得梯度"的动态
- [ ] 能写出 LoRA 层完整实现（冻结 base + A/B 初始化 + scale）
- [ ] 能跑通 3.3 梯度不对称实验与 3.4 训练演示
- [ ] 能说出 LoRA 与 Fixup/ReZero、Adapter 的对比
- [ ] 能回答 8 个面试追问
