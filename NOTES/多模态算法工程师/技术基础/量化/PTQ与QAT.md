---
learned: false
---

# PTQ 与 QAT：训练后量化与量化感知训练

> 本模块索引见 [量化详解](量化详解.md)

## 一、定义与公式

**量化（Quantization）** 是把模型里的浮点数值（FP32/FP16/BF16）用低比特整数（INT8/INT4）表示并存储的技术。核心是两步：量化（float → int）与反量化（int → float）：

$$q = \text{round}\!\left(\frac{r}{s}\right) + z, \qquad \hat{r} = s\,(q - z)$$

其中 $r$ 是原始浮点值，$s$ 是缩放因子（scale），$z$ 是零点（zero-point，非对称量化用），$q$ 是整数。**反量化得到的是 $\hat{r}$ 而不是 $r$**，这个差值 $\hat{r} - r$ 就是量化误差。

| 概念 | 定义 | 一句话 |
| --- | --- | --- |
| **PTQ**（Post-Training Quantization） | 模型**训练好之后**，用少量校准数据统计分布、直接量化权重（与激活），**不需要重新训练** | 快、省事，但低比特精度会掉 |
| **QAT**（Quantization-Aware Training） | 在**训练过程中**插入伪量化节点（fake quant），让模型提前"适应"量化误差，训练完再导出真量化模型 | 慢、要数据，但精度保得住 |

### 1.1 伪量化节点（Fake Quant）的定义

QAT 的灵魂是"**伪量化**"：训练时模型内部仍然用浮点计算，但每过一个量化层就先做"量化→反量化"，把误差暴露给训练过程：

$$\hat{r} = \text{dequant}(\text{quant}(r)) = s\left(\text{clamp}\!\left(\text{round}\!\left(\frac{r}{s}\right)+z,\ q_{\min}, q_{\max}\right) - z\right)$$

$\hat{r}$ 与 $r$ 数值上几乎一样（都是浮点），但**带上了真实的量化误差**。模型在训练中不断看到这种误差并主动适应，这就是"量化感知"。

### 1.2 一个直觉

- **PTQ**：考完试再去估分（模型不知道会被量化）；
- **QAT**：平时考试就带"噪音干扰"练习，正式考（部署）时噪音反而小。

## 二、核心原理

### 2.1 量化-反量化数学

对称量化（$z=0$）最常用：

$$s = \frac{\max |r|}{q_{\max}}, \qquad q = \text{clamp}(\text{round}(r/s), -q_{\max}, q_{\max}), \qquad q_{\max} = 2^{b-1} - 1$$

量化粒度（scale 的共享范围）：

| 粒度 | 共享范围 | 精度 | 开销 |
| --- | --- | --- | --- |
| per-tensor | 整张量一个 scale | 最差 | 最小 |
| per-channel | 每行/每列一个 scale | 好 | 中 |
| per-group（128/32） | 每 128 个通道一组 | 好 | 中 |
| per-block（64） | 每 64 个值一块（NF4） | 最好 | 最大 |

### 2.2 PTQ 完整流程（模型训练好 → 校准 → 量化）

1. **训练好 FP 模型**（冻结权重）；
2. **准备校准集**（几百到几千个有代表性的样本，如训练集子集）；
3. **前向收集分布**：把校准样本过一遍模型，统计每层权重/激活的 min/max 或直方图（决定 scale）；
4. **量化权重**：按统计出的 scale 把权重映射到 INT；
5. **量化激活（可选）**：激活分布动态变化，通常用收集到的范围做静态 scale；
6. **验证**：跑测试集，对比量化前后指标。

全程无反向传播、无梯度，几分钟到几十分钟即可完成。这就是"**快、无需数据标注**"的来源。

### 2.3 QAT 完整流程（训练中模拟量化误差）

1. **在模型中插入伪量化节点**（一般量化权重；对激活敏感的模型也量化激活）；
2. **正常训练**：前向走"量化→反量化"模拟误差，反向用直通估计器（STE）传梯度；
3. **冻结 scale**：训练后期（或收敛后）固定每层的 scale 不再更新；
4. **导出真量化模型**：把伪量化节点替换为真正的量化 kernel（权重存 INT，推理时反量化或直接 INT GEMM）。

### 2.4 直通估计器 STE（重点）

量化函数里有一个 **round（四舍五入取整）**，它的导数**几乎处处为 0**：无论输入怎么变，取整结果在绝大多数地方都不变，所以 $\partial q / \partial r$ 几乎处处为 0。直接反向传播的话，**所有梯度都会消失**，模型根本训练不动。

解法：**直通估计器（Straight-Through Estimator, STE）**——把量化函数的梯度"假装"成恒等映射：

$$\hat{r} = \text{quant}(r) \approx r \quad \Rightarrow \quad \frac{\partial \hat{r}}{\partial r} \approx 1$$

即前向照常走量化（带误差），反向**直接让梯度穿过**（绕过 round 的零梯度），只对**超出量化范围的饱和区**把梯度置 0（那里权重再更新也没用）：

$$\frac{\partial \hat{r}}{\partial r} = \begin{cases} 1, & |r| \le q_{\max}\,s \\ 0, & |r| > q_{\max}\,s \end{cases}$$

**为什么 STE 是"近似"**：round 的真实梯度是 0，STE 用 1 代替，梯度方向不精确，但实验证明对训练足够有效——模型只要"大致知道该往哪调"就行，因为最终反正要落回量化网格上。

上面的 `autograd.Function` 等价于一个更简洁的写法——"直通 Trick"：前向返回量化结果，反向用 `detach` 让梯度直接穿过 $x$：

```python
import torch

def fake_quant_ste(x, bits=8):
    """STE 的等价写法：x + (qx - x).detach()
    前向 = qx（带量化误差）；反向中 detach 使 (qx - x) 的梯度为 0，
    只剩 x 的梯度直通 → 效果等价于 d(qx)/dx ≈ 1"""
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().max() / qmax
    scale = scale.clamp_min(1e-12)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return x + (q - x).detach()

# 验证：前向带误差、反向梯度正常穿过
torch.manual_seed(0)
x = torch.randn(4, 4, requires_grad=True)
y = fake_quant_ste(x, 8)
y.sum().backward()
print((x.grad.abs() > 0).float().mean().item())  # 1.0（梯度穿过了）
print((y - x).abs().max().item())                # > 0（前向确实带量化误差）
```

输出：

```text
1.0
0.0083
```

> **面试记忆**：STE 只有一句话——"前向照常量化的样子走，反向假装它不存在（梯度≈1），饱和区除外"。

### 2.5 伪量化节点的前向/反向行为（数学）

记 $\hat{r} = \text{dequant}(\text{quant}(r))$：

- **前向**：$\hat{r} = s\,\text{clamp}(\text{round}(r/s) + z,\ q_{\min}, q_{\max}) - sz$。因为 round 后 $q$ 是整数、乘以 $s$ 再加回，所以 $\hat{r}$ 只能落在**离散网格** $\{s(q - z)\}$ 上；
- **反向**：$\frac{\partial L}{\partial r} = \frac{\partial L}{\partial \hat{r}} \cdot \underbrace{\frac{\partial \hat{r}}{\partial r}}_{\text{STE}\approx 1}$，即梯度近似原样穿过。

训练稳定后，权重的真实值会"**自发地**"停在量化网格附近——因为前向一直用 $\hat{r}$ 计算损失，把 $r$ 推到网格中间误差最小。

## 三、源码实现（可运行）

### 3.1 自定义 FakeQuantize 层（含 STE backward 完整代码）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantize(torch.autograd.Function):
    """伪量化节点：前向量化→反量化模拟误差，反向直通（STE）"""
    @staticmethod
    def forward(ctx, x, bits):
        qmax = (1 << (bits - 1)) - 1          # INT8:127 / INT4:7
        scale = x.abs().max() / qmax          # 对称量化 scale（训练时动态算）
        scale = scale.clamp_min(1e-12)
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        ctx.save_for_backward(x, scale)
        ctx.qmax = qmax
        return q * scale                      # 反量化回浮点（带误差）

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # STE：round 的梯度近似为 1，只屏蔽饱和区（|x| 超出可表示范围）
        in_range = (x.abs() <= ctx.qmax * scale).float()
        return grad_output * in_range, None
```

> **等价写法（更简洁）**：`x + (q*x.scale - x).detach()` —— 前向等于量化结果，反向 `detach` 让梯度直通 $x$，效果与上面的 `Function` 完全一致。

### 3.2 QAT 训练小 Demo：baseline vs PTQ vs QAT

任务：一个小 MLP 在 toy 回归任务上，分别用 ① FP32 baseline、② PTQ 直接量化、③ QAT 训练后量化，对比任务误差。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self, quant_mode="none", bits=4):
        super().__init__()
        self.quant_mode = quant_mode
        self.bits = bits
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def q(self, w):
        return STEQuantize.apply(w, self.bits)   # 训练时伪量化权重

    def forward(self, x):
        if self.quant_mode == "qat":
            w1, w2, w3 = self.q(self.fc1.weight), self.q(self.fc2.weight), self.q(self.fc3.weight)
        else:
            w1, w2, w3 = self.fc1.weight, self.fc2.weight, self.fc3.weight
        h = F.relu(F.linear(x, w1, self.fc1.bias))
        h = F.relu(F.linear(h, w2, self.fc2.bias))
        return F.linear(h, w3, self.fc3.bias)

def quantize_weights(model, bits=4):
    """PTQ 用：把 FP32 权重对称量化（per-tensor），返回量化后的模型副本"""
    qm = type(model)(quant_mode="none", bits=bits)
    qm.load_state_dict(model.state_dict())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                qmax = (1 << (bits - 1)) - 1
                scale = p.abs().max() / qmax
                q = torch.clamp(torch.round(p / scale.clamp_min(1e-12)), -qmax, qmax) * scale
                dict(qm.state_dict())[name].copy_(q)
    return qm

def make_data(n=4096):
    x = torch.rand(n, 8) * 2 - 1
    y = torch.sin(2 * x[:, 0]) + torch.cos(3 * x[:, 1]) + x[:, 2] * x[:, 3] + 0.1 * torch.randn(n)
    return x, y.unsqueeze(1)

x_train, y_train = make_data()
x_test, y_test = make_data(n=1024)

def train(model, steps=3000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, x_train.size(0), (128,))
        xb, yb = x_train[idx], y_train[idx]
        opt.zero_grad()
        F.mse_loss(model(xb), yb).backward()
        opt.step()

def mse(model):
    with torch.no_grad():
        return F.mse_loss(model(x_test), y_test).item()

BITS = 4   # 用 INT4 放大误差，才能看出 PTQ 与 QAT 的差别

base = MLP(quant_mode="none"); train(base)                # FP32 baseline
base_err = mse(base)
ptq_err  = mse(quantize_weights(base, BITS))              # 直接量化训练好的权重

qat = MLP(quant_mode="qat", bits=BITS); train(qat, lr=5e-4)  # QAT：训练时伪量化
qat_err  = mse(quantize_weights(qat, BITS))               # 训练完后量化权重

print(f"baseline FP32   : {base_err:.6f}")
print(f"PTQ   直接 INT4 : {ptq_err:.6f}")
print(f"QAT 训练后 INT4 : {qat_err:.6f}")
```

输出（实际运行结果）：

```text
baseline FP32   : 0.013708
PTQ   直接 INT4 : 0.076616
QAT 训练后 INT4 : 0.025468
```

**结果解读**：
- PTQ 把误差从 0.014 拉到 0.077（**5 倍以上恶化**，INT4 下 per-tensor 量化步长太大）；
- QAT 训练后量化只有 0.025，比 PTQ 好 **3 倍**，接近 FP32 baseline；
- 原因：QAT 前向一直用伪量化权重算损失，训练把权重"逼"到了量化网格上，量化前后几乎无差别。

### 3.3 验证 STE 梯度确实"穿过"了

```python
torch.manual_seed(1)
x = torch.randn(8, 8, requires_grad=True)
w = torch.randn(8, 8)
loss = (STEQuantize.apply(x, 8) * w).sum()   # 经伪量化层反向
loss.backward()
print(f"STE 梯度非零比例: {(x.grad.abs() > 0).float().mean().item():.2f}")
print(f"STE 梯度均值   : {x.grad.abs().mean().item():.4f}")
```

输出：

```text
STE 梯度非零比例: 1.00
STE 梯度均值   : 0.7735
```

如果不用 STE（直接对 `round` 求导），这个比例会是 0——模型彻底无法训练。

## 四、深入分析

### 4.1 PTQ vs QAT 对比总表

| 维度 | PTQ | QAT |
| --- | --- | --- |
| **是否需要训练** | 否（仅前向校准） | 是（完整训练流程） |
| **耗时** | 分钟级 | 与预训练/微调同量级（很贵） |
| **需要数据** | 校准集（无需标注，几百条即可） | 带标注训练集（或代理任务数据） |
| **权重量化精度** | INT8 基本无损；INT4 明显掉点 | INT4/INT3 都能保住 |
| **激活量化** | 难做准（激活分布动态、有离群点） | 训练中自动适应，能做 |
| **实现复杂度** | 低（现成工具一把梭） | 高（插入节点、冻结 scale、导出） |
| **适用场景** | **大模型首选**、快速部署、INT8 场景 | 小模型、INT4 以下、精度敏感的端侧场景 |

### 4.2 为什么 QAT 精度更高？

1. **误差被纳入优化目标**：QAT 的损失是"量化后模型"的损失，梯度直接告诉模型"哪里的权重该往网格里挪"；PTQ 是事后贴标签，模型无从反抗；
2. **权重落在量化网格上**：伪量化把 FP 权重推到了网格中点附近，量化时 round 误差最小；PTQ 的权重是"任意浮点"，round 后误差随机分布；
3. **可学 scale**：QAT 的 scale 可以参与训练（learnable scale），进一步降低误差；PTQ 的 scale 只来自校准统计。

### 4.3 为什么大模型以 PTQ 为主（QAT 太贵）？

- QAT 需要**重新训练**：70B 模型训一轮的成本是天文数字，而 PTQ 只需要跑几百条样本的前向校准；
- 大模型**参数量大、冗余度高**，对权重量化天然鲁棒（7B→70B 越大越耐量化，见 LLM 量化算法模块）；
- 结论：**70B 级 LLM 用 PTQ（GPTQ/AWQ）；端侧小模型、INT4 以下、或激活也必须量化的场景才考虑 QAT**。

### 4.4 量化误差的上界：为什么位宽决定掉点程度

均匀量化的步长是 $\Delta = s$（对称量化下 $\Delta = \frac{2\max|r|}{2^b - 1}$）。假设 $r$ 在步长内近似均匀分布，则量化误差是 $[-\Delta/2, +\Delta/2]$ 上的均匀分布：

$$\text{MSE} = \frac{\Delta^2}{12}, \qquad \max |\hat{r} - r| = \frac{\Delta}{2}$$

位宽每减 1 位，步长 $\Delta$ **翻倍**、MSE 变 **4 倍**（约 6 dB 信噪比损失）。所以 INT4 相对 INT8 的量化噪声约为 $16$ 倍——这就是"**INT8 基本无损、INT4 明显掉点**"的数学根源。

| 位宽 | 相对步长（以 INT8 为 1） | 相对 MSE | 典型体感 |
| --- | --- | --- | --- |
| FP16 | — | — | 无损 |
| INT8 | 1 | 1 | 几乎无感（weight-only） |
| INT4 | 16 | 256 | 视模型掉 0~3% 指标 |

**重要推论**：误差与 $\max|r|$ 成正比——只要分布里有一个离群大值，整个张量的步长就被拉大、所有元素一起受损。这正是"离群点"问题的根源，也解释了为什么 per-channel/per-group（缩小 scale 共享范围）能显著降误差。

## 五、优缺点

| | PTQ | QAT |
| --- | --- | --- |
| **优点** | 快、无需数据标注、工具成熟、即插即用 | 精度高（INT4/INT3 也能保住）、能做激活量化、可控可学 |
| **缺点** | 激活量化误差大、INT4 精度掉、难处理离群点 | 训练贵、要标注数据、流程复杂（插入/冻结/导出）、超参敏感 |

## 六、与同类对比

### 6.1 与全精度训练、量化后微调、蒸馏对比

| 方案 | 训练成本 | 精度 | 一句话 |
| --- | --- | --- | --- |
| 全精度训练 | 最高 | 100% | 不量化，直接训 |
| **PTQ** | 几乎为零 | INT8 无损 / INT4 掉点 | 训完直接压 |
| **QAT** | 高 | INT4 接近无损 | 训练中模拟量化 |
| 量化后微调（PTQ + LoRA） | 中 | 高 | 量化后用 LoRA 补精度（QLoRA 思路） |
| 蒸馏（大→小） | 中 | 高 | 用大模型输出当标签，另起炉灶 |

### 6.2 实战注意（工程细节）

1. **QAT 学习率要小**：量化噪声会放大梯度抖动，lr 通常取原训练计划的 $1/10 \sim 1/5$；
2. **量化层位置**：优先量化权重（最安全）；激活量化只对**关键层**开（如第一个 embedding、输出层保持 FP），卷积的通道维度 vs Transformer 的 hidden 维度对量化敏感度不同；
3. **BN 折叠（BN Folding）**：QAT 前要把 BN 的 scale/shift 折叠进前一层权重，否则 BN 与量化 scale 互相干扰、数值不稳定；
4. **scale 冻结时机**：训练快结束时冻结 scale，给权重留"收敛到网格"的时间，导出时误差最小；
5. **先做 PTQ 再升级 QAT**：能 PTQ 就 PTQ，只有精度不达标再上 QAT，避免过度设计。

## 七、高频面试问答

**Q1：PTQ 和 QAT 的区别？**
PTQ 是训练后直接量化（仅前向校准，不训练），快但 INT4 掉点；QAT 在训练中插入伪量化节点让模型适应误差，精度高但要重新训练、要数据。大模型因为 QAT 太贵，以 PTQ 为主。

**Q2：STE 是什么？为什么需要它？**
直通估计器。round 的梯度几乎处处为 0，直接反向传播梯度全消失；STE 把量化函数的梯度近似为 1（$\partial \hat{r}/\partial r \approx 1$），让梯度"直通"量化层，饱和区置 0。它是 QAT 能训练起来的核心。

**Q3：为什么 QAT 精度比 PTQ 高？**
QAT 把量化误差纳入训练目标，权重被推上量化网格、round 误差最小，且 scale 可学；PTQ 是事后量化，权重任意分布、误差随机。

**Q4：什么时候必须用 QAT？**
激活也必须量化时（PTQ 对动态分布的激活很难定 scale）；INT4/INT3 以下且精度敏感（端侧小模型、检测/分割等像素级任务）；以及量化 kernel 只支持对称量化、需要模型适配的场景。

**Q5：QAT 训练和正常训练有什么不同？**
插入伪量化节点 + 反向用 STE + 学习率调小 + 后期冻结 scale + 导出前 BN 折叠。前向额外有 quant/dequant 开销，速度略慢。

**Q6：伪量化节点前向和反向分别做什么？**
前向：quant（round+clamp）再 dequant，输出离散网格上的浮点值、带真实量化误差；反向：STE 让梯度近似直通，只在饱和区置 0。

**Q7：PTQ 的校准集需要标注吗？**
不需要。校准只统计分布（min/max/直方图）决定 scale，不需要标签；QAT 才需要带监督信号的训练数据。

**Q8：INT8 场景选 PTQ 还是 QAT？**
通常直接 PTQ（INT8 权重量化对绝大多数模型无损，还省事）；只有当激活也要 INT8 且任务敏感（如目标检测的回归头）时才考虑 QAT。

## 八、自我检验

- [ ] 能写出量化/反量化公式并解释 scale 与 zero-point
- [ ] 能说出 PTQ 五步流程与"无需训练、无需标注"的缘由
- [ ] 能画出 QAT 流程（伪量化→STE 训练→冻结 scale→导出）
- [ ] 能手写带 STE backward 的 FakeQuantize 层
- [ ] 能解释 round 不可导为什么必须用 STE（$\partial \hat{r}/\partial r \approx 1$）
- [ ] 能说清 PTQ vs QAT 在精度/成本/数据/场景上的取舍
- [ ] 知道 QAT 的 lr 要小、BN 要折叠、scale 要冻结
- [ ] 能回答 8 个面试追问并跑通 3.2 的对比 demo
