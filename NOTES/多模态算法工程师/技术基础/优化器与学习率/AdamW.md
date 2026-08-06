# AdamW 优化器

> 本模块索引见 [优化器与学习率详解](优化器与学习率详解.md)

## 一、定义与更新公式（含推导）

### 1.1 先分清两个概念（必考）

- **L2 正则化**：损失中加 $\frac{\lambda}{2}\|\theta\|^2$，梯度里出现 $\lambda\theta$ 项，**被优化器当作"梯度的一部分"处理**；
- **权重衰减（weight decay）**：每次更新时把参数独立地往 0 缩一点：$\theta \leftarrow \theta - \eta\lambda\theta$，**独立于梯度的乘性收缩**。

对 **SGD 而言二者数学上等价**（L2 的梯度项 $\lambda\theta$ 与显式衰减 $-\eta\lambda\theta$ 逐项一致），对 **Adam 则完全不等价**——这就是 AdamW（Loshchilov & Hutter, 2019）的动机。

### 1.2 AdamW 更新公式：解耦权重衰减

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, & \hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_t \right)
\end{aligned}$$

拆开看就是两步独立操作：

$$\theta_{t+1} = \underbrace{\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}}_{\text{自适应更新}} - \underbrace{\eta \lambda\, \theta_t}_{\text{解耦的权重衰减}}$$

**关键点**：衰减项 $\eta\lambda\theta_t$ 中**没有**除以 $\sqrt{\hat v}$、也不进入 $m_t/v_t$ 的统计——它对每个参数是均匀、恒定比例的收缩，与梯度大小无关。PyTorch 常用 `weight_decay=0.01`。

### 1.3 Adam + L2 为什么不对：完整推导（面试必考）

在 Adam 中加 L2 正则，损失变为 $L' = L + \frac{\lambda}{2}\|\theta\|^2$，梯度变为 $g_t' = g_t + \lambda\theta_t$。逐项追踪正则项 $\lambda\theta_t$ 的命运：

**第一步：混入一阶矩。** $\lambda\theta_t$ 作为梯度的一部分进入 EMA：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t)$$

**第二步：被二阶矩缩放。** 更新公式里整个 $m_t$ 都要除以 $\sqrt{\hat v_t}$，于是正则项的贡献变成：

$$\Delta\theta_{\text{reg}} = \eta\lambda\theta_t \cdot \frac{1}{\sqrt{\hat v_t}+\epsilon}$$

对比 AdamW 的 $\eta\lambda\theta_t$——**Adam+L2 的每步收缩量被 $\sqrt{\hat v_t}$ 缩放**，而 $\hat v_t$ 取决于该参数的梯度活跃度：

| 参数情况 | $\sqrt{\hat v_t}$ | Adam+L2 实际收缩 | AdamW 收缩 |
|---------|-------------------|------------------|-----------|
| 梯度小（稀疏/收敛后） | 小 | $\eta\lambda\theta/\sqrt{\hat v}$ **偏大**（过度惩罚） | $\eta\lambda\theta$ 恒定 |
| 梯度大（活跃参数） | 大 | $\eta\lambda\theta/\sqrt{\hat v}$ **被稀释**（几乎不衰减） | $\eta\lambda\theta$ 恒定 |

**第三步：被动量污染。** $\lambda\theta_t$ 进入 $m_t$ 后残留在滑动平均里，其作用方向与时机都不受控——历史的"衰减梯度"还在影响当前更新方向。

> 一句话：**L2 的收缩应作用于参数本身，而不该被二阶矩缩放——AdamW 把"收缩"从"梯度流"中剥离开，各干各的。** 这就是"解耦（decoupled）"的含义。

### 1.4 与 SGD 的对比：为什么 SGD 不需要解耦

SGD 更新 $\theta \leftarrow \theta - \eta g - \eta\lambda\theta$，L2 与衰减逐项一致；Adam 多了 $\sqrt{\hat v}$ 归一化这一"非线性环节"，把两项混在一起后无法还原。所以**解耦只对自适应优化器有意义**。

## 二、数学性质与直觉（几何解释）

### 2.1 均匀收缩的几何图像

把参数空间想象成三维地形，$\|\theta\|$ 是到原点的距离。权重衰减每步把参数向原点**匀速**拉近 $\eta\lambda$ 的比例——不管参数在陡坡还是平地，收缩力度相同。而 Adam+L2 的收缩力度取决于该点"当地坡的陡峭程度"（$\sqrt{\hat v}$），收缩强度与正则意图脱钩：

- 在平坦区（梯度小）的参数被**过度收缩**——稀疏 embedding 的向量被莫名压扁；
- 在陡峭区（梯度大）的参数**几乎不收缩**——正则失效，模型照样过拟合。

### 2.2 与 BN/归一化的互补

权重衰减让参数保持"小范数" → 梯度 $\nabla L$ 对参数的敏感度相对均衡 → 配合 LayerNorm/BN 的尺度不敏感性，训练更稳。Transformer 深堆 attention 层时，AdamW 的均匀收缩是长程稳定的重要保证。

### 2.3 为什么 Transformer/LLM 标配（必考）

1. **自适应步长适配异构梯度尺度**：attention、FFN、token embedding 的梯度尺度差异巨大，Adam 类缩放必需；
2. **正则强度精确可控**：预训练要强正则（$\lambda=0.01\sim0.1$），只有 AdamW 能"想要多少衰减就有多少衰减"；
3. **数值稳定**：$\epsilon=1e-8$ 在 BF16/FP16 混合精度下稳定（配合 $\epsilon=1e-6$ 更稳）；
4. **经验事实**：GPT-2/GPT-3、LLaMA、Qwen、BERT、ViT、CLIP 等几乎所有主流模型的训练配置都是 AdamW + warmup + cosine，被大量复现实验验证为"SOTA 组合"。

```text
AdamW 训练配置（LLaMA 风格）:
    lr = 3e-4（小模型）/ 1.5e-4（7B+ 大模型）
    weight_decay = 0.01（解耦衰减）
    betas = (0.9, 0.95)（部分实现 β2 用 0.95 而非 0.999）
    epsilon = 1e-8（或 1e-6 于 BF16）
    warmup_steps = 2000（或总步数的 1%~3%）
    lr_schedule = cosine decay 到最大 lr 的 10%（或 0）
```

## 三、源码实现（手写 vs torch 官方，可直接运行）

### 3.1 手写 AdamW 对比 torch.optim.AdamW

损失函数 $y=(w\cdot x-b)^2$。注意衰减项写在括号外（解耦），与 Adam 的写法只有这一处差异：

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, beta1, beta2, eps, wd, steps = 1e-2, 0.9, 0.999, 1e-8, 0.01, 500

# ---- 手写 AdamW ----
def adamw_handmade():
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    mw = mb = 0.0; vw = vb = 0.0
    for t in range(1, steps + 1):
        pred = w * x - b
        dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
        mw = beta1 * mw + (1 - beta1) * dw
        mb = beta1 * mb + (1 - beta1) * db
        vw = beta2 * vw + (1 - beta2) * dw * dw
        vb = beta2 * vb + (1 - beta2) * db * db
        mw_h, mb_h = mw / (1 - beta1 ** t), mb / (1 - beta1 ** t)
        vw_h, vb_h = vw / (1 - beta2 ** t), vb / (1 - beta2 ** t)
        # 解耦: 自适应更新 + 独立的 η·λ·θ 收缩(不进 √v̂, 不进动量)
        w -= lr * (mw_h / (torch.sqrt(vw_h) + eps) + wd * w)
        b -= lr * (mb_h / (torch.sqrt(vb_h) + eps) + wd * b)
    return w.item(), b.item()

wh, bh = adamw_handmade()
print(f"手写 AdamW: w={wh:.6f}  b={bh:.6f}")

# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.AdamW([w, b], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch AdamW: w={w.item():.6f}  b={b.item():.6f}")
```

```text
手写 AdamW: w=0.000225  b=0.000000
torch AdamW: w=0.000225  b=0.000000
```

逐位一致。对比 Adam 子篇 3.1：同一 λ 下 AdamW 收敛终值与 Adam 不同（w=2.57e-4 vs 2.25e-4）——衰减路径不同。

### 3.2 Adam+L2 的扭曲：稳态收缩系数验证

用 §1.3 的结论做数值验证：稳态下 $\hat v \approx g^2$（梯度恒定），对比两种优化器的"每步收缩量"：

```python
lr2, lam = 0.1, 0.05
for g in [0.01, 1.0, 100.0]:
    shrink_adam = lr2 * lam / (abs(g) + 1e-8)   # Adam+L2: 收缩被 1/√v̂ ≈ 1/|g| 缩放
    shrink_adamw = lr2 * lam                    # AdamW: 恒定
    print(f"g={g:8.2f}: Adam+L2 收缩≈{shrink_adam:9.4f} | AdamW 收缩={shrink_adamw:.4f}")
```

```text
g=    0.01: Adam+L2 收缩≈   0.5000 | AdamW 收缩=0.0050
g=    1.00: Adam+L2 收缩≈   0.0050 | AdamW 收缩=0.0050
g=  100.00: Adam+L2 收缩≈   0.0000 | AdamW 收缩=0.0050
```

读表要点：梯度从 0.01 到 100（4 个数量级），**Adam+L2 的收缩量跨了 100 倍**（0.5 → 0.005 → 0），而 AdamW 恒定 0.005。梯度小的参数被 Adam+L2 惩罚 100 倍于 AdamW——稀疏参数被"冤杀"，活跃参数则"逃税"。

### 3.3 零梯度下的衰减实验（看谁在"真衰减"）

把梯度人为置零（训练收敛后的近似场景），只观察权重衰减的作用：

```python
import torch

def zero_grad_agent(opt_cls, steps=200):
    torch.manual_seed(0)
    theta = torch.tensor(1.0, requires_grad=True)
    opt = opt_cls([theta], lr=0.1, weight_decay=0.05)
    for _ in range(steps):
        theta.grad = torch.tensor(0.0)          # 人为固定零梯度
        opt.step()
    return theta.item()

print(f"零梯度 200 步: Adam(wd) = {zero_grad_agent(torch.optim.Adam):.4f}")
print(f"零梯度 200 步: AdamW(wd)= {zero_grad_agent(torch.optim.AdamW):.4f}")
print(f"理论 AdamW: (1-0.1*0.05)^200 = {(1 - 0.1 * 0.05) ** 200:.4f}")
```

```text
零梯度 200 步: Adam(wd) = -0.0000
零梯度 200 步: AdamW(wd)= 0.3670
理论 AdamW: (1-0.1*0.05)^200 = 0.3670
```

读表要点：AdamW 精确按 $(1-\eta\lambda)^t$ 几何收缩（0.367 与理论逐位吻合）；**Adam 的"衰减"其实是把 $\lambda\theta$ 当梯度喂进 $\sqrt{\hat v}$ 归一化，变成每步约 $\eta$ 的线性推挤，直接把参数打到 0**——衰减力度与 $\lambda$ 完全脱钩。这就是"扭曲"的可运行证据。

## 四、超参与调参经验

| 超参 | 常用值 | 说明 |
|------|--------|------|
| weight_decay | 0.01（LLM）/ 0.1（预训练强正则） | 与 SGD 的 1e-4 不在一个量级（解耦后语义不同） |
| $\beta_1, \beta_2$ | (0.9, 0.999) 或 (0.9, 0.95) | LLaMA 风格用 0.95：二阶矩响应更快 |
| $\epsilon$ | 1e-8；BF16 下 1e-6 | 防下溢 |
| lr | 3e-4（<1B）/ 1.5e-4（7B+） | 大模型 lr 随规模下降 |

经验要点：

1. **从 Adam 迁移**：把 lr 保持不变或减半，wd 直接用 0.01 起步，通常直接提升；
2. **wd 与 lr 联动**：解耦后衰减量是 $\eta\lambda$，lr 变化时 wd 的相对强度也变——调 lr 后复查 wd；
3. **不衰减的参数**：bias、LayerNorm 的 γ/β 通常放 `no_decay` 组（PyTorch 按参数名分组传参）：
4. 多模态训练（LLaVA 等）：视觉塔 lr 常设为 LLM 的 1/10，可用两个 param_group 分别设置。

no_decay 分组的标准写法（先建分组列表，再传给优化器）：

```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)        # 含 bias
        self.norm = nn.LayerNorm(8)      # 含 weight / bias

model = TinyNet()
no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
grouped = [
    {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
     "weight_decay": 0.0},
]
opt = torch.optim.AdamW(grouped, lr=3e-4)
n_decay = sum(p.numel() for p in grouped[0]["params"])
n_nodecay = sum(p.numel() for p in grouped[1]["params"])
print(f"衰减组 {n_decay} 个参数, 不衰减组 {n_nodecay} 个参数")
```

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 权重衰减精确可控、均匀作用于所有参数 | 与 Adam 同样需要 m、v 两份状态（显存 ×3） |
| 正则与梯度解耦，理论上更干净 | 在**非 Transformer** 任务上不一定优于 SGD+wd（如小 CNN） |
| 泛化优于 Adam+L2（同 λ 下正则更强更均匀） | 超参多（λ、β₁、β₂、ε、lr），不过默认值已很鲁棒 |
| Transformer/LLM 的事实标准，生态验证充分 | 低精度训练需调大 ε（1e-6）否则下溢 |
| 与 warmup + cosine 组合稳定（大模型标配） | — |

## 六、与同类对比

| 维度 | SGD + L2/decay | Adam + L2 | AdamW |
|------|---------------|-----------|-------|
| 每步收缩量 | $\eta\lambda\theta$ | $\eta\lambda\theta/(\sqrt{\hat v}+\epsilon)$（扭曲） | $\eta\lambda\theta$ |
| 收缩是否均匀 | 均匀 | **不均匀**（依赖梯度大小） | 均匀 |
| 收缩是否被动量污染 | 否 | 是 | 否 |
| 自适应步长 | 无 | 有 | 有 |
| 泛化 | 强 | 弱于 SGD | **强于 Adam** |
| 典型场景 | CNN、经典 CV | 历史遗留 | **Transformer/LLM 标准** |

> 一句话选型：**需要自适应步长 + 可控正则 → 直接 AdamW；追求极致泛化 → SGD+Momentum。**

## 七、高频面试问答

**Q1：L2 正则和权重衰减的区别？**
L2 在损失里加 $\frac{\lambda}{2}\|\theta\|^2$，通过梯度生效；权重衰减是独立的乘性收缩 $\theta\leftarrow\theta-\eta\lambda\theta$。SGD 下两者等价，Adam 下不等价。

**Q2：为什么 Adam+L2 不对？**
正则项 $\lambda\theta$ 混入梯度后被两件事破坏：① 除以 $\sqrt{\hat v}$，收缩量依赖该参数梯度大小——梯度小则过度惩罚、梯度大则形同虚设；② 进入 $m_t$ 的 EMA，历史正则梯度残留在动量里。AdamW 把 $\eta\lambda\theta$ 独立加在更新外，均匀收缩。

**Q3：AdamW 的更新公式？**
$\theta_{t+1}=\theta_t-\eta(\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta_t)$，即自适应更新与解耦衰减两步。

**Q4：为什么 Transformer/LLM 标配 AdamW？**
四点：异构梯度尺度需要自适应步长；预训练需要强且可控的正则；混合精度数值稳定；BERT/GPT/LLaMA/Qwen/ViT/CLIP 全部验证过该组合。

**Q5：PyTorch 里 Adam 的 weight_decay 是解耦的吗？**
不是。`torch.optim.Adam` 的 wd 实现的是 L2（混入梯度、被二阶矩扭曲）；`torch.optim.AdamW` 才是解耦权重衰减。想要均匀收缩必须用 AdamW。

**Q6：wd 应该怎么选？**
LLM 预训练 0.01 起步；想要更强正则试 0.1；bias 与 norm 参数通常放 no_decay 组。注意解耦后 wd 语义与 SGD 时代不同，别沿用 1e-4。

**Q7：AdamW 和 Lion/Sophia 的关系？**
Lion 保留了解耦衰减思想（$\theta_t=\theta_{t-1}-\eta(\text{sign}(c_t)+\lambda\theta_t)$ 形式）；Sophia 用 Hessian 估计替代 $\sqrt{\hat v}$ 做预条件。它们都以 AdamW 为基准做对比。

## 八、自我检验 checklist

- [ ] 能说清 L2 与权重衰减的区别，以及"为何 SGD 等价而 Adam 不等价"
- [ ] 能默写 AdamW 公式并指出解耦项的位置
- [ ] 能完整推导"$\lambda\theta$ 被 $\sqrt{\hat v}$ 扭曲"的三步（混入 m → 被缩放 → 污染动量）
- [ ] 能手写 AdamW 循环并验证与 torch.optim.AdamW 逐位一致
- [ ] 能用稳态公式解释"梯度小的参数被过度惩罚 100 倍"
- [ ] 能运行零梯度实验并解释 $(1-\eta\lambda)^t$ 收缩
- [ ] 能背出 LLaMA 风格训练配置（lr/wd/betas/ε/warmup/cosine）
- [ ] 知道 no_decay 分组与多塔不同 lr 的工程做法
- [ ] 能回答 7 个面试追问
