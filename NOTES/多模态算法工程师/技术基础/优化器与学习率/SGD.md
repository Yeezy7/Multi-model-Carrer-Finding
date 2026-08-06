# SGD 梯度下降

> 本模块索引见 [优化器与学习率详解](优化器与学习率详解.md)

## 一、定义与更新公式（含推导）

### 1.1 优化问题设定

训练神经网络就是最小化损失函数 $L(\theta)$（$\theta$ 为全部参数）：

$$\theta^* = \arg\min_{\theta} L(\theta) = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i; \theta), y_i)$$

**梯度下降思想**：不知道损失曲面的全局形状，就在当前位置朝"下降最快的方向"（负梯度）走一小步，反复迭代。为什么负梯度是下降最快的方向？对 $L$ 在 $\theta$ 处做一阶 Taylor 展开：

$$L(\theta - \eta g) \approx L(\theta) - \eta g^T g = L(\theta) - \eta \|g\|^2, \qquad g = \nabla L(\theta)$$

在步长 $\eta$ 固定的约束下，$L(\theta - \eta g) - L(\theta) \approx -\eta\|g\|^2$ 的下降量只与 $\|g\|$ 有关，**方向任意**；真正的最速下降是"单位步长"约束下 $\min_{d: \|d\|=1} g^T d = -\|g\|$，取 $d = -g/\|g\|$——即**负梯度方向**。

### 1.2 批量梯度下降（BGD / GD）

用**全部**训练样本计算平均梯度后更新一次：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N}\nabla \ell(f(x_i;\theta_t), y_i)$$

| 特点 | 说明 |
|------|------|
| 优点 | 梯度方向最准确，收敛曲线平滑 |
| 缺点 1 | 每步遍历全量数据，计算开销大 |
| 缺点 2 | 显存放不下全量梯度（现代数据集上亿样本） |
| 缺点 3 | 梯度为零即停：陷入局部极小/鞍点时无法逃逸 |

### 1.3 随机梯度下降（SGD，单样本）

每次只用一个样本 $x_i$ 估计梯度：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla \ell(f(x_i;\theta_t), y_i)$$

- 单个样本梯度是真实梯度（全量梯度）的**无偏估计**：$E_i[\nabla\ell_i] = \nabla L$；
- 方差大 → 更新轨迹剧烈震荡；
- 噪声本身就是一种"抖动"，能帮助跳出浅的局部极小；
- 但由于逐样本计算无法利用矩阵并行，实际硬件上并不快。

### 1.4 小批量梯度下降（Mini-batch SGD）—— 实际训练的标准

每步用一个小批 $B = \{x_1,...,x_m\}$（$m$ 通常取 32/64/128/256）的**平均梯度**更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m}\sum_{i=1}^{m}\nabla \ell(f(x_i;\theta_t), y_i)$$

Mini-batch 梯度的方差是单样本的 $1/m$（对独立样本）：$\text{Var}(g_{\text{mini}}) = \frac{1}{m}\text{Var}(g_{\text{single}})$——用一次矩阵运算的代价换来了 $\sqrt{m}$ 倍的噪声衰减。

### 1.5 三种模式对比表（面试高频）

| 维度 | BGD | SGD（单样本） | Mini-batch SGD |
|------|-----|--------------|----------------|
| 每次用样本数 | 全部 N | 1 | m（32~1024） |
| 梯度噪声 | 无 | 很大 | 适中 |
| 收敛曲线 | 平滑单调 | 剧烈震荡 | 较平滑、有小噪声 |
| 迭代成本 | 极高 | 最低（但向量化差） | 适中，GPU 友好 |
| 跳出局部极小能力 | 差 | 强 | 中 |
| 能否利用矩阵计算 | 能 | 差 | **能（关键优势）** |
| 实际使用 | 几乎不用 | 几乎不用 | **主流** |

## 二、数学性质与直觉（几何解释）

### 2.1 几何图像

把损失曲面想成一幅"地形图"，等高线是椭圆：

- **BGD**：像精确定位的地质测量队，每步都算全图平均坡向，路径最直，但移动一次要"走完全图"；
- **单样本 SGD**：像闭着眼扔飞镖探路，每步方向都是带噪声的估计——方向忽左忽右，但长期来看均值方向还是下坡；噪声大时轨迹像布朗运动，围着最优解打转；
- **Mini-batch**：每步只问 8 个"路人"坡向，噪声可控且每条腿都能并行走。

### 2.2 为什么需要 mini-batch（面试必答点）

1. **梯度估计与计算成本的权衡**：batch 越大梯度越准，但收益递减——经验上 batch 增到 256/512 后，精度提升已远小于算力开销（batch 大小与训练收益大致呈对数关系）；
2. **GPU 向量化/并行**：mini-batch 一次矩阵乘法算完，单样本逐个算反而浪费硬件；
3. **内在噪声 = 正则化**：小批抽样引入的梯度噪声有轻微正则化效果（与 BN 的 batch 噪声类似），泛化通常比大 batch 好；
4. **显存可控**：batch 大小直接决定激活值显存占用，可以按显存调节；
5. **在线/流式学习**：数据无需全部装入内存。

> 反直觉结论：**大 batch 并不总是更好**——batch 过大时收敛变慢、泛化变差（大 batch 难以逃出窄的极小值），需要配合更大的 lr 与更长的 warmup 补偿（见《学习率调度》子篇）。

### 2.3 学习率与收敛行为

对二次损失（如 $\frac{1}{2}a\theta^2$），GD 的迭代为 $\theta_{t+1} = (1-\eta a)\theta_t$，收敛要求 $|\eta a| < 2$：

- $\eta$ 过小：指数式慢爬，收敛慢；
- $\eta$ 过大（$\eta a > 2$）：震荡发散；
- 最优收敛速率 $\eta = 1/a$，一步到位。

**这解释了为什么 lr 是最重要的超参**：它决定步长是否落在稳定区间内。

## 三、源码实现（手写 vs torch 官方，可直接运行）

### 3.1 手写全量梯度下降（不用 autograd，纯手推梯度）

以损失函数 $y=(w\cdot x-b)^2$（即 $L(w,b)=\text{mean}((wx-b)^2)$）为例，解析求导：$\partial L/\partial w = \text{mean}(2(wx-b)x)$，$\partial L/\partial b = \text{mean}(-2(wx-b))$。

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 32)          # 训练数据: 32 个点

def loss_fn(w, b):
    return ((w * x - b) ** 2).mean()   # 损失函数 y=(w·x-b)²

w = torch.tensor(1.5)                  # 非零初始化, 让梯度下降"真正走下坡"
b = torch.tensor(-0.8)
lr = 0.05
for t in range(80):
    pred = w * x - b
    dw = (2 * pred * x).mean()         # 手推梯度 ∂L/∂w
    db = (-2 * pred).mean()            # 手推梯度 ∂L/∂b
    w -= lr * dw                       # 参数更新: θ ← θ - η·∇L
    b -= lr * db
    if (t + 1) % 10 == 0:
        print(f"step {t+1:3d}  loss={loss_fn(w,b).item():.2e}  w={w.item():.4f}  b={b.item():.4f}")
```

```text
step  10  loss=2.27e-01  w=0.3246  b=-0.2789
step  20  loss=1.65e-02  w=0.0702  b=-0.0973
step  30  loss=1.48e-03  w=0.0152  b=-0.0339
step  40  loss=1.55e-04  w=0.0033  b=-0.0118
step  50  loss=1.77e-05  w=0.0007  b=-0.0041
step  60  loss=2.10e-06  w=0.0002  b=-0.0014
step  70  loss=2.53e-07  w=0.0000  b=-0.0005
step  80  loss=3.06e-08  w=0.0000  b=-0.0002
```

loss 单调下降约 7 个数量级——这就是 BGD 在凸问题上的收敛形态（指数收敛）。

### 3.2 用 torch.optim.SGD + autograd 实现 mini-batch 训练

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64).reshape(-1, 1)   # (64, 1)

w = torch.tensor([[1.5]], requires_grad=True)
b = torch.tensor([-0.8], requires_grad=True)

opt = torch.optim.SGD([w, b], lr=0.05)         # 官方 SGD 优化器
for epoch in range(5):
    perm = torch.randperm(64)                  # 每轮打乱数据
    for i in range(0, 64, 8):                  # 切成 8 个 mini-batch
        batch = x[perm[i:i + 8]]
        opt.zero_grad()
        ((w * batch - b) ** 2).mean().backward()
        opt.step()
    full = ((w * x - b) ** 2).mean().item()    # 每轮算全量 loss 看真实进度
    print(f"epoch {epoch+1}  full-batch loss={full:.2e}  w={w.item():.4f}  b={b.item():.4f}")
```

```text
epoch 1  full-batch loss=3.98e-01  w=0.4439  b=-0.3561
epoch 2  full-batch loss=4.80e-02  w=0.1306  b=-0.1567
epoch 3  full-batch loss=6.62e-03  w=0.0405  b=-0.0660
epoch 4  full-batch loss=1.01e-03  w=0.0127  b=-0.0280
epoch 5  full-batch loss=1.59e-04  w=0.0035  b=-0.0119
```

（对比 3.1：同样 40 步参数更新，BGD 已到 $10^{-4}$，mini-batch 在 $10^{-4}$ 附近——mini-batch 稍慢一点点，但每步便宜得多。）

### 3.3 三种模式的轨迹对比（看噪声）

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)

def run(mode, lr=0.2, steps=100):
    """mode: 'bgd'(全量) / 'mini'(8 样本) / 'sgd'(单样本)"""
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    traj = []
    for t in range(steps):
        if mode == "bgd":
            xb = x                                      # 全部样本
        elif mode == "mini":
            xb = x[torch.randperm(64)[:8]]              # 随机抽 8 个
        else:
            xb = x[torch.randint(0, 64, (1,))]          # 只抽 1 个
        pred = w * xb - b
        w -= lr * (2 * pred * xb).mean()
        b -= lr * (-2 * pred).mean()
        if (t + 1) % 20 == 0:
            traj.append(((w * x - b) ** 2).mean().item())
    return traj

for m in ["bgd", "mini", "sgd"]:
    print(f"{m}: " + "  ".join(f"{v:.2e}" for v in run(m)))
```

```text
bgd: 8.56e-10  1.14e-18  1.53e-27  2.04e-36  2.80e-45
mini: 1.63e-09  5.00e-19  3.35e-28  9.72e-38  0.00e+00
sgd: 1.92e-06  9.00e-14  1.49e-20  1.39e-26  1.78e-32
```

读表要点：前 20 步三者都在降；20 步之后 BGD 每 20 步掉 9 个数量级（纯几何收敛），单样本 SGD 始终比 BGD 慢 3~4 个数量级且后期被"噪声地板"拖住——**噪声越大，最终精度越低**。这就是"SGD 用噪声换单步代价"的量化代价。

## 四、超参与调参经验

| 超参 | 建议 | 说明 |
|------|------|------|
| 学习率 $\eta$ | 0.01~0.1（CV 常用 0.1 起） | 最重要超参；震荡就除以 10，太慢就乘以 2~10 |
| batch size | 32~512 | 显存允许下尽量大；**batch ×k，lr 可 ×√k~k**（线性缩放规则） |
| momentum | 0.9（配合 SGD 使用） | 见《Momentum 与 Nesterov》子篇 |
| weight_decay | 1e-4~5e-4 | 配合 momentum 使用；纯 SGD 与 L2 等价 |
| nesterov | 现代 CV 标配 `nesterov=True` | 免费加速，几乎无副作用 |

经验流程：

1. 先固定 batch=128，用 LR finder 扫出峰值 lr（见《学习率调度》子篇）；
2. 若 loss 震荡 → lr 减半或加 warmup；
3. 训练后期若验证集不涨 → 切 StepLR/cosine 降 lr；
4. 追求泛化 → SGD+Momentum 收官（大模型后期切回 SGD 微调是常见 trick）。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 实现极简、计算开销小、内存占用低（无额外状态） | 收敛慢：平坦区/鞍点处梯度近 0，几乎停滞 |
| 泛化能力强：收敛到"平坦极小值"，测试精度常优于 Adam | 对 lr 极度敏感：lr 大一点震荡，小一点龟速 |
| 每参数行为可解释（无自适应缩放黑盒） | 无法自适应不同参数的梯度尺度（embedding 等稀疏参数吃亏） |
| 理论成熟：凸问题收敛率 $O(1/t)$，最优可达 $O(1/\sqrt{t})$ 噪声鲁棒 | 手动调参成本高（lr、momentum、wd、schedule 都要配） |
| mini-batch 噪声自带正则化 | 病态条件数（如扁椭圆）下锯齿形振荡，收敛慢 |

## 六、与同类对比

| 维度 | SGD | SGD+Momentum | Adam 系 |
|------|-----|--------------|---------|
| 自适应步长 | 无 | 无 | 有（二阶矩缩放） |
| 对抗震荡 | 差 | 好（速度对冲） | 好 |
| 平坦区/鞍点 | 停滞 | 惯性穿越 | 惯性穿越 + 自适应放大 |
| 泛化 | 强 | 强 | 弱于 SGD |
| 对 lr 敏感度 | 高 | 高 | 低 |
| 显存状态 | 0 | 1 份速度 | 2 份（m, v） |
| 调参成本 | 高 | 高 | 低 |
| 典型场景 | — | **CNN/经典 CV/收官微调** | Transformer/LLM |

## 七、高频面试问答

**Q1：BGD、SGD、Mini-batch 的区别？**
每次更新的样本量不同：全量 / 1 个 / m 个。全量梯度最准但贵；单样本噪声大、向量化差；mini-batch 是精度与成本的折中，能利用 GPU 矩阵并行，是实际训练的标准。

**Q2：为什么实际不用 BGD？**
数据集太大：一是每步计算量无法接受；二是显存放不下全量梯度；三是全量梯度在局部极小/鞍点梯度为零即停，mini-batch 的噪声反而能帮助逃逸。

**Q3：为什么需要 mini-batch？**
五点：梯度估计与算力权衡（收益对数递减）；GPU 向量化；噪声正则化；显存可控；支持流式数据。

**Q4：单样本 SGD 的梯度为什么无偏？**
$E_i[\nabla \ell_i] = \frac{1}{N}\sum_i \nabla\ell_i = \nabla L$，样本的期望等于全量梯度；方差 $\text{Var} = \frac{1}{m}\text{Var}_\text{single}$，所以 batch 越大噪声越小。

**Q5：大 batch 有什么问题？怎么补偿？**
大 batch 噪声小 → 更容易收敛到尖锐极小值、泛化差，且收敛初期更新步数少。补偿：lr 按 batch 线性放大 + 延长 warmup + 更长的训练。

**Q6：SGD 的收敛率是多少？**
凸情形（光滑强凸）GD 线性收敛；带噪声的 SGD 期望收敛率 $O(1/t)$ 量级；非凸深度网络理论上无保证，靠实践经验调参。

**Q7：为什么说 SGD 泛化比 Adam 好？**
两种主流解释：① SGD 等步长更新更易落入"平坦极小值"（泛化好），Adam 自适应缩放倾向收敛到"尖锐极小值"；② Adam 后期 $\sqrt{\hat v}$ 按历史梯度缩放，方向偏离真实曲率。实践中可"AdamW 预训练 + SGD 收尾"。

## 八、自我检验 checklist

- [ ] 能写出 BGD / SGD / Mini-batch 三种更新公式
- [ ] 能用 Taylor 展开推导"负梯度是最速下降方向"
- [ ] 能说出 mini-batch 的 5 个理由（面试能答 4 个即可）
- [ ] 能写出 $\text{Var}(g_{mini}) = \text{Var}(g_{single})/m$
- [ ] 能手写梯度下降循环（不依赖 autograd）验证收敛
- [ ] 知道大 batch 的坑与补偿手段（lr 放大 + warmup）
- [ ] 能说清"噪声 = 正则化"的含义
- [ ] 能回答 7 个面试追问
