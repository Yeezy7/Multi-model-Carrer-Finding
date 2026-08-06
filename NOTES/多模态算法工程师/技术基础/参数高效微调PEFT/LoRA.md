# LoRA：低秩适配（Low-Rank Adaptation）

> 本模块索引见 [参数高效微调PEFT详解](参数高效微调PEFT详解.md)

## 一、定义与公式

### 1.1 一句话定义

LoRA（Hu et al., ICLR 2022）是目前最主流的 PEFT 方法：**冻结预训练权重 $W_0$，把微调的更新量 $\Delta W$ 参数化为两个低秩矩阵 $B, A$ 的乘积**，训练时只更新 $A, B$，可训练参数通常不到全模型的 0.1%~1%。

### 1.2 数学形式

对任意线性层 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$（Transformer 里的 q/k/v/o/gate/up/down 投影都是这种形式），前向变为：

$$h = W_0 x + \frac{\alpha}{r} B A x$$

其中：
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$：**高斯初始化**（$\mathcal{N}(0, \sigma^2)$，$\sigma \approx 0.02$）；
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$：**零初始化**；
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$：秩，通常 4~64；
- $\alpha$：缩放常数，有效缩放系数为 $\alpha / r$。

训练完成后的等效权重：$W' = W_0 + \frac{\alpha}{r} B A$。

### 1.3 参数量推导

原线性层参数为 $d_{\text{out}} \times d_{\text{in}}$（忽略 bias）。LoRA 分支参数为两个低秩矩阵之和：

$$\text{param}(\text{LoRA}) = \underbrace{r \times d_{\text{in}}}_{A} + \underbrace{d_{\text{out}} \times r}_{B} = r\,(d_{\text{in}} + d_{\text{out}})$$

对方阵 $d \times d$，占比约 $\dfrac{2r}{d}$。数值例（$d = 4096$，$r = 8$）：

| 项目 | 原层 | LoRA 分支 |
|------|------|-----------|
| 参数量 | 4096×4096 ≈ 1678 万 | 8×(4096+4096) = 6.55 万 |
| 占比 | 100% | **0.39%** |

### 1.4 梯度推导（必考）

记 $s = \alpha / r$，分支输出 $y = s\,B\,A\,x$，令 $z = A x$，$g = \partial L / \partial y$：

$$\frac{\partial L}{\partial B} = s\, g\, z^\top, \qquad
\frac{\partial L}{\partial A} = s\, B^\top g\, x^\top, \qquad
\frac{\partial L}{\partial x} = W_0^\top g + s\, A^\top B^\top g$$

由第二条公式立刻得到关键事实：**初始化时 $B = 0$，则 $\partial L/\partial A = 0$**——若 $A$ 也零初始化，LoRA 分支永远得不到梯度。这就是"$B$ 必须零初始化、$A$ 必须非零初始化"的数学根源。

## 二、核心原理

### 2.1 低秩假设（LoRA 为什么有效，必考）

论文的核心观察有两条：

1. **预训练权重本身是高秩/满秩的**，承载通用能力；
2. **微调的更新量 $\Delta W$ 是低秩的**：把全参微调得到的 $\Delta W$ 做 SVD，绝大多数奇异值接近 0，能量集中在少数大奇异值方向——更新量落在一个低维子空间里。

$$\Delta W \approx \sum_{i=1}^{r} \sigma_i u_i v_i^\top = B A$$

因此 $r$ 很小的 $BA$ 就足以表达 $\Delta W$ 的主体。论文在 GPT-3 175B 上的证据：$r = 4$（可训练参数仅 0.01%）效果已接近全参微调，$r$ 增大到 8/16 无明显增益。

低秩约束还天然带来**正则化**：更新被限制在低秩子空间，比全参微调更不易过拟合下游数据、更抗灾难性遗忘。

### 2.2 为什么 B 初始化为 0、A 高斯初始化（必考）

三个层次：

1. **起点正确**：$B = 0$ 时初始 $BA = 0$，前向严格等于 $W_0 x$，微调从预训练权重出发。若 $B$ 也随机初始化，模型一开始就被注入随机扰动，偏离"在预训练能力上做增量"的前提，初始损失大、训练不稳；
2. **梯度可得**：由 1.4 的公式，$B = 0$ 时 $\partial L/\partial A = 0$，参数无法更新——所以 $A$ 必须用高斯 $\mathcal{N}(0, \sigma^2)$（$\sigma \approx 0.02$）初始化，保证初始梯度非零；
3. **对称性**：$A, B$ 都随机 → 初始扰动破坏起点；都为零 → 永远不学习。两者分工：一个保证"起点对"，一个保证"能学习"。

### 2.3 α/r 缩放系数的推导

分支 $BAx$ 的量级与 $r$ 有什么关系？设 $A, B$ 元素方差分别为 $\sigma_A^2, \sigma_B^2$，$z = Ax$ 的分量方差为 $\sigma_A^2 \|x\|^2$，则分支输出第 $i$ 个分量：

$$y_i = \sum_{k=1}^{r} B_{ik} z_k, \qquad \text{Var}(y_i) = r\, \sigma_B^2 \sigma_A^2 \|x\|^2 \propto r$$

**分支量级随 $\sqrt{r}$ 增长**（当 $B$ 学到固定规模时）。若不缩放，改 $r$ 会连带改变分支量级 → 等效学习率变化 → 必须重调超参。除以 $r$ 后：

- 分支量级 $\propto \sqrt{r}/r = 1/\sqrt{r}$，$r$ 变化带来的量级漂移被抑制；
- $\alpha$ 直接乘在分支上，等价于单独调节分支的学习率；
- **$r$ 与 $\alpha$ 解耦**：想加表达力改 $r$，想调强度改 $\alpha$，互不干扰。工程上常设 $\alpha = 2r$（有效缩放为 2）。

### 2.4 target_modules 怎么选

| 模块 | 作用 | 推荐 |
|------|------|------|
| q_proj / k_proj / v_proj | Attention 的 Q/K/V 投影 | **必加** |
| o_proj | Attention 输出投影 | 常加 |
| gate_proj / up_proj / down_proj | FFN 门控与上下投影（LLaMA 系） | 常加 |
| embedding / lm_head | 词表嵌入与输出头 | 一般不 LoRA（参数大、易崩、收益低） |

经验：7 个线性层全加，LLaMA-7B、$r=8$ 时约 20M 可训练参数（0.30%）；只加 q/k/v/o 约 8.4M（0.12%）。多模态中视觉塔与投影层的选择见总览第八节。

### 2.5 推理合并（Merge）——零延迟的关键

因为矩阵乘法满足分配律：

$$(W_0 + s\,B A)\,x = W_0 x + s\,B A\,x$$

所以训练完把两个分支**直接加进原权重**是精确等价（不是近似）：

$$W' = W_0 + \frac{\alpha}{r} B A$$

- **合并模式**：模型结构与预训练完全一致 → 推理零额外延迟、可直接量化/编译、单文件部署；
- **分支模式（不合并）**：保留 $W_0$ + 多份几 MB 的 $BA$，一份基座服务多任务/多用户、可热插拔，代价是每次推理多一次低秩乘法。

## 三、源码实现

### 3.1 手写 LoRALayer（完整可训练）

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """可训练的 LoRA 分支：ΔW = (α/r)·B·A"""

    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.02)   # 高斯初始化：保证能学
        self.B = nn.Parameter(torch.zeros(out_features, r))         # 零初始化：保证起点对
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = self.dropout(x) @ self.A.t()          # (..., d_in) -> (..., r)
        return z @ self.B.t() * self.scaling      # (..., r) -> (..., d_out)

    def delta_weight(self):
        """合并用：返回 (d_out, d_in) 的更新量 (α/r)·BA"""
        return self.B @ self.A * self.scaling

# ---- 自检：形状与数学等价 ----
torch.manual_seed(0)
lora = LoRALayer(64, 128, r=8, alpha=16)
x = torch.randn(4, 16, 64)
y = lora(x)
print(y.shape)                  # torch.Size([4, 16, 128])
delta = lora.delta_weight()
print(delta.shape)              # torch.Size([128, 64])
print(torch.allclose(y, x @ delta.T))   # True：前向 == x @ ΔWᵀ
```

### 3.2 LoRALinear：把 LoRA 并接到 nn.Linear 上（含合并）

```python
class LoRALinear(nn.Module):
    """原线性层（冻结）+ 可训练 LoRA 分支，支持合并回原权重"""

    def __init__(self, in_features, out_features, bias=True, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, r, alpha, dropout)
        self._merged = False
        for p in self.base.parameters():
            p.requires_grad = False                    # 冻结基座

    @classmethod
    def from_linear(cls, linear, **kwargs):
        """从已有 nn.Linear 构建并复制原权重（用于注入替换）"""
        layer = cls(linear.in_features, linear.out_features,
                    bias=linear.bias is not None, **kwargs)
        layer.base.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.base.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x):
        if self._merged:
            return self.base(x)                        # 合并后走纯 Linear
        return self.base(x) + self.lora(x)             # 双分支：W₀x + (α/r)BAx

    def merge(self):
        """把 LoRA 写回 base 权重（W' = W0 + (α/r)BA），返回纯 nn.Linear"""
        if not self._merged:
            self.base.weight.data.add_(self.lora.delta_weight())
            self._merged = True
        return self.base


def inject_lora(model, target=("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"),
                r=8, alpha=16, dropout=0.1):
    """递归地把名字命中 target 的 nn.Linear 替换为 LoRALinear，并冻结全部基座"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in target:
            setattr(model, name,
                    LoRALinear.from_linear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora(child, target, r, alpha, dropout)
    # 冻结整个基座，只留 LoRA 分支可训练
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora.parameters():
                p.requires_grad = True
    return model
```

### 3.3 端到端训练 + 合并等价性验证

```python
class TinyLM(nn.Module):
    """迷你语言模型：嵌入 + 单层自注意力 + FFN（LLaMA 单层缩略版）"""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d); self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d); self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, 2*d); self.up_proj = nn.Linear(d, 2*d)
        self.down_proj = nn.Linear(2*d, d)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x=None, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed(x if x is not None else input_ids)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        h = h + self.o_proj(attn @ v)                              # attention 残差
        h = h + self.down_proj(torch.relu(self.up_proj(self.ln(h))))  # FFN 残差
        return self.head(h)

torch.manual_seed(0)
model = TinyLM()
inject_lora(model, r=8, alpha=16)                   # 注入 7 个线性层并冻结基座
total = sum(p.numel() for p in model.parameters())
lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"全模型 {total} 参数，LoRA 可训练 {lora_params}（{lora_params/total:.1%}）")
# 全模型 19104 参数，LoRA 可训练 4352（22.8%）
# 注：玩具模型占比高；真实 7B 模型全注入 r=8 时约 0.30%

# 训练 100 步（固定随机数据，仅验证"只动 LoRA 也能学"）
x = torch.randint(0, 64, (8, 16))
y = torch.randint(0, 64, (8, 16))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-3)
crit = nn.CrossEntropyLoss()
model.train()
for step in range(100):
    opt.zero_grad()
    loss = crit(model(x).reshape(-1, 64), y.reshape(-1))
    loss.backward()
    opt.step()
print(f"训练后 loss = {loss.item():.3f}")   # 从 ~4.43 降到 ~1.68（基座完全冻结）

# —— 合并等价性验证（必考）——
model.eval()
with torch.no_grad():
    out_before = model(x)                        # 双分支前向
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()                            # 全部合并回 base
    out_after = model(x)                         # 纯 Linear 前向
print("合并前后最大误差:", (out_before - out_after).abs().max().item())
# 8.6e-06：数学上精确等价，仅剩浮点加法顺序的舍入误差
```

### 3.4 peft 库标准用法（实战必会）

```python
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

config = LoraConfig(
    r=8,
    lora_alpha=16,                    # 有效缩放 α/r = 2
    lora_dropout=0.1,
    bias="none",                      # 不训练 bias
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.FEATURE_EXTRACTION,
)
peft_model = get_peft_model(TinyLM(), config)    # 自动冻结基座 + 注入
peft_model.print_trainable_parameters()
# trainable params: 4,352 || all params: 19,104 || trainable%: 22.7806

# 训练完保存（只保存 LoRA 分支，几十 KB 的权重文件）
peft_model.save_pretrained("./tiny_lora")

# 推理方式一：加载到新基座上（不合并，多任务热插拔模式）
base = TinyLM()
loaded = PeftModel.from_pretrained(base, "./tiny_lora")

# 推理方式二：合并回基座（推理零开销，单文件部署）
merged = loaded.merge_and_unload()
```

## 四、参数与显存账本

### 4.1 LLaMA-7B 参数量逐步计算（r=8）

LLaMA-7B：$d_{\text{model}} = 4096$，FFN 维度 11008，共 32 层。每层注入 7 个线性层：

| 层 | 维度 | 每层 LoRA 参数（r=8） |
|----|------|---------------------|
| q/k/v/o（4 个） | 4096×4096 | 4 × 8×(4096+4096) = 262,144 |
| gate/up（2 个） | 4096×11008 | 2 × 8×(4096+11008) = 241,664 |
| down（1 个） | 11008×4096 | 1 × 8×(11008+4096) = 120,832 |
| 每层合计 | — | 624,640 |

$$\text{全注入（7 层×32）} = 624{,}640 \times 32 \approx 20.0\text{M} \ (0.30\%), \qquad \text{仅 q/k/v/o} = 262{,}144 \times 32 \approx 8.4\text{M} \ (0.12\%)$$

### 4.2 训练显存账本（7B 模型）

| 显存构成 | 全参微调 | LoRA（r=8 全注入） |
|----------|---------|-------------------|
| 基座权重 | 14GB（bf16，需梯度） | 14GB（bf16，**冻结**，只读） |
| 可训练参数 | 14GB | 20M × 2B ≈ 40MB |
| 梯度 | 14GB | ≈ 40MB |
| Adam 状态（fp32 m、v） | 2×14GB×4B = 112GB… 实为 56GB（m+v 各 28GB） | 20M × 2 × 4B ≈ 160MB |
| 激活值 | 大（数 GB~数十 GB） | 同左（与参数量无关） |
| **训练峰值（估算）** | **100GB+** | **16~30GB** |

> 核心记忆点：**Adam 优化器状态（m、v 两份 fp32）是显存大头**（每参数 8 字节）。LoRA 只对 0.1%~0.3% 的参数维护优化器状态，冻结基座只有只读的 bf16 权重——省掉的是"梯度 + 优化器状态"，不是"权重本身"。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 可训练参数仅 0.01%~1%，显存/存储成本极低 | 低秩约束限制了表达空间，复杂任务上略弱于全参 |
| 可合并进原权重，**推理零延迟** | 超参多（r、α、target_modules、dropout），需调 |
| 一份基座 + 多分支，天然支持多任务热插拔 | 秩 r 过大有过拟合/秩塌缩风险 |
| 更新被限制在低秩子空间，更抗灾难性遗忘 | 与量化感知训练等新方法的协同需要专门设计 |
| 与 bf16/4-bit 基座兼容（QLoRA 的基础） | 对 kernel 算子（如 fused attention）仍需逐层注入 |

## 六、与同类方法对比

| 维度 | LoRA | Adapter | Prompt/Prefix | 全参微调 |
|------|------|---------|---------------|---------|
| 可训练参数（7B） | 0.01%~1%（~4-20M） | 0.5%~5% | 0.005%~0.5% | 100% |
| 训练显存 | 16~30GB | 略高于 LoRA | 最低 | 100GB+ |
| 效果（对齐全参） | ~95-99% | ~95-99% | 60-95% | 100% |
| 推理开销 | **0（可合并）** | 串行有延迟 | Prefix 使序列变长 | 0 |
| 改动面 | 线性层前向公式 | 插入新子网络 | 输入/KV 拼接 | 全部 |
| 与量化/编译工具兼容性 | 好（可合并） | 较差 | 一般 | — |

> 一句话选型：**通用首选 LoRA**；显存极限选 QLoRA（见 [QLoRA](QLoRA.md)）；参数预算极小选软提示（见 [Prompt与PrefixTuning](Prompt与PrefixTuning.md)）；研究兼容旧框架的场景考虑 Adapter（见 [Adapter](Adapter.md)）。

## 七、高频面试问答

**Q1：LoRA 为什么有效？**
微调更新量 $\Delta W$ 是低秩的（SVD 后绝大多数奇异值接近 0），$BA$ 足以表达其主成分；论文在 GPT-3 175B 上 $r=4$（0.01% 参数）即接近全参效果。低秩约束还带来正则化，更抗过拟合与遗忘。

**Q2：为什么 B=0、A 高斯初始化？**
$B=0$ 保证初始 $W' = W_0$，微调从预训练起点平滑出发（起点正确）；同时 $\partial L/\partial A \propto B^\top$，若 $A$ 也为 0 则梯度恒为零（能学习）。二者一个保证"起点对"，一个保证"学得动"。

**Q3：α/r 为什么不直接用 1？**
分支量级随 $\sqrt{r}$ 增长，不缩放则改 $r$ 会改变等效学习率、必须重调超参；$\alpha/r$ 把"秩"与"学习强度"解耦，先定 $r$ 再调 $\alpha$（常设 $\alpha = 2r$）。

**Q4：target_modules 怎么选？**
q/k/v 必加，o/gate/up/down 常加；embedding/lm_head 一般不加。LLaMA-7B r=8 全加约 20M（0.30%），只加 q/k/v/o 约 8.4M（0.12%）。

**Q5：合并权重和不合并有什么区别？**
数学上 $(W_0 + sBA)x = W_0x + sBAx$ 精确等价。合并：推理零开销、单文件部署；不合并：一份基座多分支热插拔、多任务灵活，代价是每次推理多一次低秩乘法。

**Q6：r 越大越好吗？**
不是。论文与大量实践显示 r 超过 32~64 后效果饱和甚至下降（过拟合/秩塌缩），r=4~16 往往已足够；关键是 $\alpha/r$ 与 lr 的配合。

**Q7：LoRA 和全参微调差距多大？什么时候必须全参？**
一般任务可达全参 95% 以上；数据量大、任务与预训练域差异大时差距拉大。LoRA 在"遗忘"上更优。大规模高质量指令数据场景下全参仍更稳。

**Q8：LoRA 能解决灾难性遗忘吗？**
能缓解：$W_0$ 冻结 + 更新约束在低秩子空间，对预训练能力破坏远小于全参；但数据过度偏移时分支本身会漂移，更强方案是 LoRA 融合、回放数据或与 Adapter 组合。

## 八、自我检验

- [ ] 能写出 $h = W_0x + \frac{\alpha}{r}BAx$ 并解释每个符号的维度与初始化
- [ ] 能手推 $\partial L/\partial A = sB^\top g x^\top$，并由此解释 B=0 / A 高斯的原因
- [ ] 能讲清"微调更新量低秩"的直觉与论文证据（SVD、r=4 接近全量）
- [ ] 能推导分支量级 ∝ √r，解释 α/r 缩放的必要性
- [ ] 能说出 target_modules 的经验选法与 LLaMA-7B 的参数量级（20M/8.4M）
- [ ] 能写出 LoRALayer + 注入 + merge 的手写实现，并验证合并前后输出一致
- [ ] 能背出 7B 模型 LoRA 与全参微调的显存账本及"Adam 状态是大头"的结论
- [ ] 能说清合并（零延迟）与不合并（多任务）两种部署模式
- [ ] 能独立写出 LoraConfig + get_peft_model + save_pretrained + merge_and_unload 流程
- [ ] 能回答 8 个面试追问
