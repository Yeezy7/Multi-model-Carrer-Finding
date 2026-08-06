# EMA（指数移动平均）

> 本模块索引见 [正则化与训练技巧详解](正则化与训练技巧详解.md)

## 一、定义与公式

EMA（Exponential Moving Average，指数移动平均）维护一组**影子权重**（shadow weights）$\theta_{ema}$，每个训练 step 用当前模型参数 $\theta$ 滑动更新：

$$\theta_{ema} \leftarrow \lambda \cdot \theta_{ema} + (1 - \lambda) \cdot \theta$$

- $\lambda$ 为衰减系数（decay），典型 0.99（短期平均）~ 0.999（长期平均），大模型微调常用 0.9999；
- $\theta_{ema}$ 初值取 $\theta_0$（或直接复制初始权重）；
- **EMA 权重不参与梯度计算**，只被"平均"，因此没有额外的显存占用问题之外的开销（需要一份影子副本的显存）。

### 1.1 递推展开（EMA 到底平均了什么）

把递推式展开到 $t$ 步：

$$\theta_{ema}^{(t)} = \lambda^t \theta_0 + (1-\lambda)\left[\theta_t + \lambda\theta_{t-1} + \lambda^2\theta_{t-2} + \cdots + \lambda^{t-1}\theta_1\right]$$

**最近一步权重 $\theta_t$ 的系数最大（$1-\lambda$），越老的权重系数按 $\lambda^k$ 指数衰减**——这就是"指数移动平均"名字的由来，也说明它只关心**最近约 $1/(1-\lambda)$ 步**的权重。

### 1.2 等效平均窗口

$$\text{有效平均窗口} \approx \frac{1}{1-\lambda}$$

| $\lambda$ | 窗口（step） | 场景 |
|-----------|-------------|------|
| 0.99 | ~100 | 短任务（分类微调） |
| 0.999 | ~1000 | 常规预训练/微调 |
| 0.9999 | ~10000 | LLM/VLM 长训练（Qwen/LLaVA 常用） |

> **记忆点**：$\lambda=0.999$ 意味着每个新 step 只把影子权重向当前权重移动 0.1%，等效平均窗口约 1000 步。

## 二、核心原理

### 2.1 为什么 EMA 权重泛化更好（三条主线）

1. **参数空间平均 = 隐式集成**：EMA 相当于对训练轨迹上的多个模型做指数加权平均，是集成的一种廉价实现。集成的方差更低（$\text{Var}(\text{平均}) = \text{Var}/N$），泛化更稳；
2. **收敛到平坦极小值（flat minima）**：训练后期梯度含噪，参数在损失景观的"凹坑"里震荡。EMA 加权平均让权重落在**更平坦的区域**——平坦极小值对参数扰动不敏感（loss 的 Hessian 特征值小），泛化更好。这是随机权重平均理论的核心直觉；
3. **缓解灾难性遗忘（微调场景）**：EMA 权重保留了"旧知识"的信息，微调 VLM/LLM 时 EMA 权重能显著减少通用能力下降（如 LLaVA 微调后 CV 任务掉点变少）。

> 直观理解：训练中权重像掷骰子一样跳来跳去，EMA 是这些骰子的加权平均；**单点可能踩坑，平均点更接近"真实解"。**

### 2.2 与 SGD 动量、Adam 的关系（防混淆）

EMA 与优化器内部的动量**不是一回事**：

- Adam 的动量 $\hat{m}$ 平均的是**梯度**（用于计算更新方向）；
- EMA 平均的是**参数**（用于得到最终模型权重）。

两者作用对象不同、目的不同，可以共存：训练用 Adam（动量在梯度上），评估用 EMA（平均在参数上）。

### 2.3 训练与评估的使用流程（核心工程点）

1. 每个 step：optimizer 更新 $\theta$ 后，同步更新 $\theta_{ema}$（**用更新后的参数**）；
2. **评估时切换成 $\theta_{ema}$，训练时用 $\theta$**（评估完再切回）；
3. 通常**只保存 EMA checkpoint**（或两者都存）作为最终发布/推理模型。

## 三、源码实现

### 3.1 手写 EMA 类（含 warmup 衰减）

```python
import torch
import torch.nn as nn

class EMA:
    """指数移动平均：维护 shadow weights，评估时切换到 EMA 权重"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        # 每个 step 调一次：θ_ema ← λ·θ_ema + (1-λ)·θ
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        # 评估前：备份当前权重，载入 EMA 权重
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model):
        # 评估后：恢复训练权重
        model.load_state_dict(self.backup)

    def apply_decay(self, step, warmup_steps=1000):
        # 训练早期用更小的 decay（warmup），防止早期噪声参数被过分平均
        self.decay = min(self.decay, (1 + step) / (10 + step))
```

### 3.2 完整训练循环：EMA 与评估切换

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

ema = EMA(model, decay=0.999)              # 复用 3.1 的 EMA 类

for step in range(200):
    ema.apply_decay(step)                    # warmup 期 decay 从小往大涨
    x = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    ema.update(model)                        # ★ 关键：优化器 step 后更新 EMA

# 评估：切到 EMA 权重 → 推理 → 切回
ema.apply_shadow(model)
with torch.no_grad():
    x_test = torch.randn(16, 16)
    pred_ema = model(x_test)                 # 用 EMA 权重推理
ema.restore(model)
pred_online = model(x_test)                  # 用在线权重推理
diff = (pred_ema - pred_online).abs().max().item()
print(pred_ema.shape, diff)                  # torch.Size([16, 4])，diff 如 0.023（两者不同）
```

### 3.3 对比 torch.optim.swa_utils 的 AVERAGER

PyTorch 官方提供了 SWA 工具类（torch.optim.swa_utils.AveragedModel），可以配置 EMA 或均匀平均：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

def ema_avg(avg_param, new_param, num_averaged):
    """官方 EMA 平均函数（avg_fn 约定签名）"""
    return 0.999 * avg_param + 0.001 * new_param

model2 = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
averaged = AveragedModel(model2, avg_fn=ema_avg)   # 官方影子模型

for step in range(50):
    opt2.zero_grad()
    loss = F.cross_entropy(model2(torch.randn(32, 16)), torch.randint(0, 4, (32,)))
    loss.backward()
    opt2.step()
    averaged.update_parameters(model2)             # 等价于手写 ema.update

print(type(averaged.module).__name__)              # Sequential（averaged.module 是影子模型）
```

### 3.4 训练中常犯的坑（代码层面）

```python
import torch
import torch.nn as nn

# 坑 1：在 optimizer.step 之前 update —— 平均的是"旧参数"
opt.step()
ema.update(model)        # ✅ 正确顺序：先 step 再 update

# 坑 2：忘记只克隆不共享 —— shadow 必须与在线参数断开梯度与内存
model = nn.Linear(4, 2)
shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # ✅ clone
# bad = {k: v for k, v in model.state_dict().items()}                    # ❌ 同一张量，更新即变
print(len(shadow))       # 2（weight 与 bias 各一份影子副本）

# 坑 3：评估后忘记 restore —— 下一轮训练的梯度会基于 EMA 权重
# ema.apply_shadow(model)
# ... eval ...
# ema.restore(model)     # ✅ 必须恢复，否则在线权重被 EMA 覆盖
```

### 3.5 与梯度累积 / AMP 的配合

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 梯度累积时：EMA 只在"真正更新参数"的 step 后更新一次（与 optimizer 同步）
model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
ema = EMA(model, decay=0.999)               # 复用 3.1 的 EMA 类

accum_steps = 4
opt.zero_grad()
for i in range(accum_steps):
    x = torch.randn(32, 16)
    loss = F.cross_entropy(model(x), torch.randint(0, 4, (32,))) / accum_steps
    loss.backward()
opt.step()
ema.update(model)        # ✅ 每个累积周期一次，而不是每个 micro-batch 一次
print("accum 完成，EMA 已同步更新")
```

> EMA 只关心参数值，与 AMP/梯度累积天然兼容；要点是**与 optimizer.step 同步**（参考模块总览 11.7 的 step 时机表）。

## 四、深入分析

### 4.1 EMA 与 SWA 的区别

| 维度 | EMA | SWA |
|------|-----|-----|
| 平均范围 | 全程指数加权（每个 step 参与） | 只对训练尾部（如最后 10%）均匀平均 |
| 权重分配 | 指数衰减（最近最重要） | 尾部等权 |
| 实现 | 在线增量更新，O(1) 内存 | 需记录尾部权重，可循环队列 |
| 场景 | 预训练/微调全程 | 训练末期、超参不佳时抢救 |

实践结论：两者都收敛到平坦极小值；EMA 是全流程的默认选择，SWA 常作为"训练快结束时最后冲刺"的手段（微调 CLIP 后 SWA 可再提 0.5~1 个点检索指标）。

### 4.2 warmup 期为什么 decay 要小

训练早期参数从初始化出发剧烈变动、噪声大，此时用 $\lambda=0.999$ 会把**初始噪声**以 0.999 的高权重"焊死"在影子权重里，之后几百步都洗不掉。对策：`decay = min(decay, (1+step)/(10+step))`——第 0 步 decay≈0.1，让 EMA 快速跟上前几步，之后再涨到目标值。

### 4.3 Buffer 要不要平均

EMA 一般不平均 BN 的 running_mean/running_var：这些 buffer 是**统计量**而非可训练参数，平均后反而偏离真实分布。但对 LN/RMSNorm（LLM/VLM 主体架构）没有 buffer，直接整份 state_dict 平均也没问题。PyTorch 中可用 `state_dict()` 的 `_metadata` 区分，或手动只取 `requires_grad=True` 的参数。

### 4.4 EMA 与早停、checkpoint 保存

- 发布模型：**保存 EMA checkpoint**（推理部署用 EMA 权重）；
- EMA 权重免疫震荡，可放宽 early stopping 的 patience；
- 微调场景下 EMA 权重与在线权重混用（在线训、EMA 评）是最常见配置；半监督（Mean Teacher）则把 EMA 当 teacher 生成伪标签，$\lambda$ 用 0.999~0.9999。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单（几十行），效果稳定 | 需一份影子权重显存（大模型下约 +100% 参数显存） |
| 隐式集成，方差更低、泛化更好 | decay 需按任务步数调（0.99~0.9999），调错收益下降 |
| 收敛到平坦极小值，对损失景观鲁棒 | 平均不了 buffer（BN 场景需注意） |
| 缓解灾难性遗忘（VLM 微调核心收益） | 响应慢：短任务里 EMA 还没追上训练就结束 |
| 与 AMP/梯度累积/LoRA 天然兼容 | 与调度器/累积耦合，step 时机出错会静默劣化 |

## 六、与同类对比

| 方法 | 平均对象 | 加权方式 | 主要收益 | 与 EMA 关系 |
|------|---------|---------|---------|-------------|
| EMA | 参数 | 指数加权（全程） | 泛化 + 防遗忘 | 基准方法 |
| SWA | 参数 | 尾部等权 | 平坦极小值 | 尾部特化版 EMA |
| 模型集成（Ensemble） | 多个独立模型 | 等权 | 最强收益 | EMA 是其廉价近似 |
| Polyak Averaging | 参数 | 等权（旧式） | 平滑 | 现代等价于 EMA |
| Adam 动量 | 梯度 | 指数加权 | 加速收敛 | 作用在梯度，与 EMA 互补 |

**与"权重快照"对比**：若不做任何平均、直接保存中途 checkpoint，参数处于震荡高点，泛化不稳；EMA 相当于"把每个 step 的快照按指数加权合并"，是零推理开销的免费增益。

## 七、高频面试问答

**Q1：EMA 的公式？衰减系数怎么选？**
$\theta_{ema} \leftarrow \lambda\theta_{ema} + (1-\lambda)\theta$。短任务 0.99~0.995，常规预训练/微调 0.999，LLM/VLM 长训练 0.999~0.9999；warmup 期要小（`min(decay, (1+step)/(10+step))`）。

**Q2：为什么 EMA 权重比在线权重好？**
三点：① 参数空间加权平均是隐式集成，方差更低；② 平均后落在平坦极小值，对扰动鲁棒、泛化好；③ 微调场景保留旧知识、缓解灾难性遗忘。

**Q3：训练和评估时 EMA 怎么用？**
训练每 step 在 optimizer.step 后 update 影子权重；评估时 apply_shadow 切换到 EMA 权重、评估完 restore 切回；发布/部署只保存 EMA checkpoint。

**Q4：EMA 的动量与 Adam 的动量有什么区别？**
Adam 平均梯度（决定更新方向），EMA 平均参数（决定最终权重）。作用对象不同、目的不同、可共存。

**Q5：EMA 和 SWA 的区别？**
EMA 全程指数加权、O(1) 增量更新；SWA 只对尾部均匀平均。两者都收敛到平坦极小值；EMA 是全流程默认，SWA 常用作训练末期冲刺。

**Q6：EMA 平均 buffer（BN 统计量）吗？**
一般不平均：BN 的 running stats 是统计量不是参数，平均后偏离真实分布。LN/RMSNorm 无 buffer，整份 state_dict 平均无问题。

**Q7：EMA 的 step 时机？配合梯度累积？**
EMA 与 optimizer.step 同步：梯度累积每 accum 完成一次 step 才 update 一次 EMA，而不是每个 micro-batch 一次；与 AMP 兼容（EMA 只关心参数值）。

## 八、自我检验

- [ ] 能写出 $\theta_{ema} \leftarrow \lambda\theta_{ema} + (1-\lambda)\theta$ 并展开推导等效窗口 $1/(1-\lambda)$
- [ ] 能说出 EMA 泛化好的三个原因（集成/平坦极小/防遗忘）
- [ ] 能写出手写 EMA 类（update / apply_shadow / restore 三方法）
- [ ] 知道 warmup 期 decay 要小及实现写法
- [ ] 能说出训练/评估的使用流程与"只存 EMA checkpoint"惯例
- [ ] 能区分 EMA（平均参数）与 Adam 动量（平均梯度）
- [ ] 能说清 EMA 与 SWA 的区别
- [ ] 能回答 7 个面试追问
