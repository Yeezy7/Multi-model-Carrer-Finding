# Momentum 与 Nesterov

> 本模块索引见 [优化器与学习率详解](优化器与学习率详解.md)

## 一、定义与更新公式（含推导）

### 1.1 物理直觉（面试叙述模板）

想象一个小球从山坡滚下：

- **没有动量的 SGD**：每步只看当前位置的坡（梯度），像"一碰就停、瞬时改向"的粒子——在谷底两侧来回震荡，在平坦区寸步难行；
- **有动量的 SGD**：小球有**速度（velocity）**，速度是过去所有梯度的加权累积，改向需要时间——它能"冲过"小坑，在平坦区靠惯性持续前进。

$$\text{速度 } v = \text{衰减的历史速度} + \text{当前梯度}$$

### 1.2 Momentum 公式（PyTorch 形式）

$$\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - v_t
\end{aligned}$$

- $\gamma$（动量系数，常取 0.9）：历史速度的衰减率；$\gamma \approx 1$ 时几乎完全保留历史；
- 教科书写法把 $\eta$ 放在梯度前，PyTorch 把 $\eta$ 乘在速度上，两者等价；
- 推导：展开 $v_t$ 的递推，得到**指数加权**的历史梯度之和：

$$v_t = \eta \sum_{k=0}^{t-1} \gamma^{k}\, \nabla L(\theta_{t-k})$$

历史上越早的梯度，权重 $\gamma^k$ 指数衰减。以 $\gamma=0.9$ 为例：当前梯度权重 1，10 步前的梯度权重 $0.9^{10} \approx 0.35$，50 步前 $0.9^{50} \approx 0.005$——**动量"记住"约 $\frac{1}{1-\gamma}=10$ 步的历史**。

### 1.3 等效步长推导（为什么能加速）

若梯度方向长期一致（如长斜坡上 $g_t = g$ 恒定），速度会累积：

$$v_\infty = \eta g \sum_{k=0}^{\infty}\gamma^k = \frac{\eta g}{1-\gamma}$$

即等效步长为 SGD 的 $\dfrac{1}{1-\gamma}$ 倍：$\gamma=0.9$ 时 **10 倍**，$\gamma=0.99$ 时 100 倍。这就是"沿斜坡加速冲"的数学来源。

### 1.4 Nesterov 加速梯度（NAG）—— 前瞻一步

**核心思想（look-ahead）**：既然已知速度 $v$，就先按速度方向"跳出去"再看那里的坡，比站在原地看梯度更准：

$$\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla L(\theta_t - \gamma v_{t-1}) \\
\theta_{t+1} &= \theta_t - v_t
\end{aligned}$$

梯度在**前瞻点** $\theta_t - \gamma v_{t-1}$ 处计算，而不是当前位置。

> **直观类比**：盲人下坡会先用拐杖往前探一步（前瞻），感受那里是不是悬崖，再决定用多大力——当速度太大即将冲过头时，前瞻点上的梯度会给出"刹车"信号。

**PyTorch 的实现形式**（`SGD(momentum=0.9, nesterov=True)` 内部）：

$$\begin{aligned}
v_t &= \gamma v_{t-1} + g_t \\
\theta_{t+1} &= \theta_t - \eta\,(g_t + \gamma v_t)
\end{aligned}$$

即先照常累积速度，但更新用"当前梯度 + 动量贡献"之和。它与教科书 look-ahead 形式在变量代换下**数学等价**（Sutskever et al., 2013），只是表述不同；代码实现用后一种。

## 二、数学性质与直觉（几何解释）

### 2.1 两大作用（必考）：同向叠加、反向抵消

1. **加速收敛（同向叠加）**：梯度方向一致的平坦区（长斜坡），速度不断累加，等效步长放大（§1.3），快速冲过平原；
2. **抑制震荡（反向抵消）**：在等高线为扁椭圆的窄谷中，沿陡峭轴梯度来回翻转——当前梯度与历史速度方向相反，两者相消，垂直方向的摆动被抑制，路径更直。

> 记忆：**"同向叠加、反向抵消"**——八个字概括动量本质。

### 2.2 为什么动量能逃离局部极小与平坦区

| 场景 | SGD 的表现 | 动量表现 |
|------|-----------|---------|
| 浅的局部极小 | 梯度为 0 即停 | 速度不为 0，凭惯性**冲出去** |
| 平坦区/鞍点 | 梯度接近 0，几乎不动 | 累积速度持续前进，**穿过**高原 |
| 陡峭谷底 | 来回震荡 | 速度对冲震荡，稳步下降 |
| 长斜坡 | 步长固定，慢 | 沿斜坡加速，等效大步长 |

**本质**：动量把优化从"一阶状态"升级为"二阶状态"（多记住一个速度），决策不再只看当前位置的瞬时梯度，而考虑运动趋势——正好弥补了"梯度为零但势能未释放"的盲区（局部极小、鞍点、平坦区）。

### 2.3 Nesterov 为什么更好：提前刹车

普通动量在接近极小值时"冲过头"才发现要减速（震荡变缓但存在）；NAG 在前瞻点计算梯度，**提前感知**"前方要反弹"，当场给出相反方向的梯度分量——刹车更早、震荡更小。理论上 NAG 在强凸光滑问题达到最优 $O(1/t^2)$ 收敛率，而普通动量是 $O(1/t)$。

### 2.4 什么时候动量没用/有害

- 损失曲面各向同性（条件数≈1）时，动量与 SGD 差距不大；
- 目标函数剧烈变化（损失曲面频繁改向）时，动量记忆反而拖后腿；
- $\gamma$ 过大（>0.99）且 lr 较大时，速度累积过猛会**发散**——动量放大了有效步长，稳定性边界更紧（§1.3 的 $1/(1-\gamma)$ 倍）。

## 三、源码实现（手写 vs torch 官方，可直接运行）

### 3.1 手写 Momentum 对比 torch.optim.SGD(momentum=0.9)

损失函数仍用 $y=(w\cdot x-b)^2$。手写版严格对照 PyTorch 形式：$v = \gamma v + g$，$\theta \leftarrow \theta - \eta v$。

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, gamma, steps = 0.05, 0.9, 100

# ---- 手写 Momentum ----
w = torch.tensor(1.5); b = torch.tensor(-0.8)
vw = vb = 0.0                                  # 速度状态(对应 torch 的 momentum_buffer)
for t in range(steps):
    pred = w * x - b
    dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
    vw = gamma * vw + dw                       # 速度 = 衰减的历史 + 当前梯度
    vb = gamma * vb + db
    w -= lr * vw                               # 参数沿速度方向更新
    b -= lr * vb
print(f"手写 Momentum: w={w.item():.6f}  b={b.item():.6f}")

# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=lr, momentum=0.9)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch  Momentum: w={w.item():.6f}  b={b.item():.6f}")
```

```text
手写 Momentum: w=0.007407  b=-0.002991
torch  Momentum: w=0.007407  b=-0.002991
```

结果逐位一致——手写版就是 `torch.optim.SGD(momentum=0.9)` 的完整语义。

### 3.2 手写 Nesterov 对比 torch nesterov=True

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, gamma, steps = 0.05, 0.9, 100

# ---- 手写 Nesterov (PyTorch 形式: 更新量 = 当前梯度 + γ·速度) ----
w = torch.tensor(1.5); b = torch.tensor(-0.8)
vw = vb = 0.0
for t in range(steps):
    pred = w * x - b
    dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
    vw = gamma * vw + dw                       # 先照常累积速度
    vb = gamma * vb + db
    w -= lr * (dw + gamma * vw)                # 更新 = g + γ·v (前瞻修正)
    b -= lr * (db + gamma * vb)
print(f"手写 Nesterov: w={w.item():.6f}  b={b.item():.6f}")

# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=lr, momentum=0.9, nesterov=True)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch  Nesterov: w={w.item():.6f}  b={b.item():.6f}")
```

```text
手写 Nesterov: w=0.000005  b=-0.000019
torch  Nesterov: w=0.000005  b=-0.000019
```

注意对比 3.1：**同样 100 步，Momentum 停在 $w=0.0074$，Nesterov 收敛到 $w=0.000005$**——前瞻刹车让它在接近最优时少震荡 3 个数量级。

### 3.3 三路对比：病态椭圆上动量优势的量化

$y=(w\cdot x-b)^2$ 的条件数接近 1（各向同性），动量优势不明显；换成**条件数 100 的椭圆** $L=100w^2+b^2$（陡峭轴 $w$、平缓轴 $b$），SGD 在平缓轴上龟速爬，动量靠惯性穿越：

```python
import torch

# L(w, b) = 100·w² + b²   (条件数 100, 等高线是扁椭圆)
def final_loss(mode, lr=0.004, gamma=0.9, steps=500):
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    vw = vb = 0.0
    for t in range(steps):
        dw = 200 * w            # ∂L/∂w (陡峭轴, 梯度大)
        db = 2 * b              # ∂L/∂b (平缓轴, 梯度小)
        if mode == "momentum":
            vw = gamma * vw + dw; vb = gamma * vb + db
            dw, db = vw, vb
        elif mode == "nesterov":
            vw = gamma * vw + dw; vb = gamma * vb + db
            dw, db = dw + gamma * vw, db + gamma * vb
        w -= lr * dw; b -= lr * db
    return (100 * w * w + b * b).item()

for m in ["sgd", "momentum", "nesterov"]:
    print(f"{m:9s} 500 步后 loss: {final_loss(m):.3e}")
```

```text
sgd       500 步后 loss: 2.079e-04
momentum  500 步后 loss: 1.357e-21
nesterov  500 步后 loss: 3.863e-26
```

读表要点：SGD 被平缓轴拖住（500 步只到 $10^{-4}$）；动量在平缓轴上等效步长大 10 倍，直接碾压到 $10^{-21}$；Nesterov 再快 5 个数量级。**条件数越大，动量/NAG 的收益越明显**。

## 四、超参与调参经验

| 超参 | 默认/常用 | 经验 |
|------|----------|------|
| $\gamma$（momentum） | 0.9 | 0.9 是安全默认；0.95~0.99 用于大 batch/长训练（记忆更长）；过大会发散 |
| lr | 与 SGD 同量级 | 动量使有效步长 ×$1/(1-\gamma)$，加动量后 lr 常需减半再微调 |
| nesterov | PyTorch 建议 True | 与 momentum 同参数，效果更好，几乎零成本 |
| dampening | 0 | 不为 0 时速度会"漏"，一般不用 |

经验要点：

1. **CV 训分类模型的标准配方**：`SGD(momentum=0.9, nesterov=True, weight_decay=5e-4)` + StepLR/cosine；
2. 从 SGD 切到 Momentum 时，如果训练曲线震荡变强，把 lr 除以 2~10；
3. 动量在**小 batch + 高噪声**场景收益最大（噪声被 EMA 平均掉）；
4. 微调大模型不建议用动量过大的配置：历史梯度来自预训练分布，γ 大容易"刹不住"。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 平坦区/鞍点/局部极小都能穿越（惯性） | 引入了"状态"，多一份速度内存（参数量的 1 倍） |
| 抑制窄谷震荡，收敛路径更直 | $\gamma$ 过大时速度累积过猛，可能发散 |
| 实现只多一行代码，代价极小 | 在方向频繁改向的损失曲面上，动量记忆可能拖慢 |
| 理论上强凸问题可达到 $O(1/t^2)$（NAG） | 对 lr 依然敏感（有效步长被放大） |
| 泛化能力与纯 SGD 相当，比 Adam 好 | 无法自适应不同参数的梯度尺度 |

## 六、与同类对比

| 维度 | SGD | Momentum | NAG | Adam 系 |
|------|-----|----------|-----|---------|
| 状态量 | 无 | 速度 v | 速度 v | m + v |
| 平坦区穿越 | 停滞 | 惯性穿越 | 惯性穿越 | 穿越 + 自适应 |
| 震荡抑制 | 差 | 好 | 更好（提前刹车） | 好 |
| 收敛率（强凸） | $O(1/t)$ | $O(1/t)$ | $O(1/t^2)$（最优） | 经验上快 |
| 泛化 | 强 | 强 | 强 | 弱于前者 |
| 显存（相对参数） | 1× | 2× | 2× | 3× |
| 与 L2 的关系 | 等价 | 等价 | 等价 | 扭曲（见 AdamW 子篇） |

> 关系图：SGD $\xrightarrow{+速度}$ Momentum $\xrightarrow{前瞻}$ NAG；Adam 的 $m_t$ 就是"动量"，NAG 的 look-ahead 思想也被 Adam 论文吸纳（`adam_nesterov` 选项）。

## 七、高频面试问答

**Q1：Momentum 为什么能加速收敛？**
梯度方向一致时速度持续累积，等效步长放大到 SGD 的 $1/(1-\gamma)$ 倍（γ=0.9 时 10 倍），平坦区/长斜坡上快速通过；反向梯度到来时历史速度与之相消，抑制震荡——"同向叠加、反向抵消"。

**Q2：Momentum 和 NAG 的区别？**
Momentum 在当前位置算梯度再叠加速度；NAG 先按速度"前瞻"到 $\theta-\gamma v$ 再算梯度——先在动量方向迈一步、再看新的坡。前瞻让 NAG 在接近极小值时提前感知"要冲过头"而刹车，震荡更小、收敛更快。

**Q3：γ 取太大有什么问题？**
等效步长 $\eta/(1-\gamma)$ 爆炸，超过稳定性边界会发散；且记忆过长，损失曲面改向时反应迟钝。默认 0.9，大 batch 长训练可试 0.95。

**Q4：动量相当于记住多少步历史？**
指数衰减的窗口约 $\frac{1}{1-\gamma}$ 步：γ=0.9 → 约 10 步，γ=0.99 → 约 100 步。

**Q5：动量在鞍点为什么有用？**
鞍点处梯度接近 0，SGD 原地不动；动量里的历史速度不为 0，凭惯性穿过鞍点区域（与 Adam 的 $\sqrt{\hat v}$ 放大机理不同，两者可叠加）。

**Q6：PyTorch 的 nesterov=True 和论文里的 look-ahead 形式为什么看起来不一样？**
两者在变量代换下数学等价（Sutskever et al. 的经典结论）：look-ahead 形式 $v=\gamma v+\eta g(\theta-\gamma v)$ 变换后就是 PyTorch 的 $v=\gamma v+g$、更新 $g+\gamma v$。实现用后者，理解用前者。

**Q7：动量加在 Adam 里是什么？**
Adam 的 $m_t$ 就是一阶矩（动量），$\beta_1=0.9$ 即动量系数；所以"Adam 内置动量，SGD 需要显式加 momentum"。

## 八、自我检验 checklist

- [ ] 能写出 Momentum 的完整递推公式并展开推导 $v_t = \eta\sum\gamma^k g_{t-k}$
- [ ] 能推导等效步长 $\frac{1}{1-\gamma}$ 倍
- [ ] 能解释"同向叠加、反向抵消"并画出椭圆等高线上的路径对比
- [ ] 能手写 Momentum/Nesterov 循环并验证与 torch 逐位一致
- [ ] 知道 PyTorch 与论文 NAG 公式等价的结论
- [ ] 能用病态椭圆问题解释动量收益的来源（条件数越大收益越大）
- [ ] 知道 γ 过大导致发散、动量加 lr 需减小的经验
- [ ] 能回答 7 个面试追问
