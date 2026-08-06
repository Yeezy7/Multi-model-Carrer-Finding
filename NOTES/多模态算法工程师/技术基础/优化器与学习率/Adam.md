# Adam 优化器

> 本模块索引见 [优化器与学习率详解](优化器与学习率详解.md)

## 一、定义与更新公式（含推导）

### 1.1 动机：两个问题的合体

- **Momentum 解决了"方向"问题**：同向叠加、反向抵消（见 Momentum 子篇）；
- **RMSProp/AdaGrad 解决了"步长"问题**：梯度大的参数走小步，梯度小的参数走大步（逐参数自适应 lr）。

Adam（Kingma & Ba, 2015）= **一阶矩（动量）+ 二阶矩（自适应缩放）+ 偏差校正**：

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t & \text{（梯度的一阶矩，带动量）} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 & \text{（梯度的二阶矩，方差估计）} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} & \text{（偏差校正）} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t
\end{aligned}$$

各量含义：

| 量 | 含义 |
|----|------|
| $m_t$ | 梯度方向的动量（加速 + 抑制震荡） |
| $v_t$ | 梯度平方的 EMA，逐参数缩放步长（自适应 lr） |
| $\hat m_t, \hat v_t$ | 校正后的无偏估计（见 1.2） |
| $\eta$ | 全局学习率（默认 1e-3） |
| $\beta_1, \beta_2$ | 一阶/二阶矩衰减率（默认 0.9 / 0.999） |
| $\epsilon$ | 防除零小常数（默认 1e-8） |

### 1.2 偏差校正（bias correction）推导（面试必考）

**问题**：初始化 $m_0 = 0$。展开 $m_t$ 的递推：

$$m_t = (1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}\, g_k$$

$m_t$ 是历史梯度的加权和，但**权重之和不为 1**：

$$\sum_{k=1}^{t}(1-\beta_1)\beta_1^{t-k} = (1-\beta_1)\cdot\frac{1-\beta_1^t}{1-\beta_1} = 1 - \beta_1^t$$

所以 $m_t$ 的期望被"往 0 拉"——**估计有偏**。假设梯度平稳（$E[g_k] = E[g]$）：

$$E[m_t] = \left(1 - \beta_1^t\right)\, E[g]$$

要得到无偏估计，只需除以权重和：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad E[\hat{m}_t] = E[g]$$

对二阶矩同理：$E[v_t] = (1-\beta_2^t)\,E[g^2]$，所以 $\hat v_t = v_t/(1-\beta_2^t)$。

**为什么 t 很小时影响大**（这是本题核心）：

| t | $1-\beta_1^t$ | 校正量 |
|---|---------------|--------|
| 1 | $1-0.9 = 0.1$ | $m_1$ 只有真实值的 10%，校正放大约 10 倍 |
| 5 | $1-0.9^5 \approx 0.41$ | 放大约 2.4 倍 |
| 50 | $1-0.9^{50} \approx 0.995$ | 几乎不校正 |
| 100 | $1-0.9^{100} \approx 1$ | 可忽略 |

> 记忆点：**"初始化 0 → 加权和权重不足 1 → 除以 $1-\beta^t$ 补权"**——一句话复述整个推导。校正只在训练初期起作用，这也从数学上解释了"初期要配 warmup"（见《学习率调度》子篇）。

### 1.3 更新步长上界推导（为什么 Adam 对 lr 不敏感）

忽略 $\epsilon$，单步更新量近似 $\|\Delta\theta_t\| \approx \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t}}$。由于 $\hat v_t \ge \hat m_t^2$（$E[g^2] \ge E[g]^2$，Cauchy 不等式），有：

$$\frac{\hat m_t}{\sqrt{\hat v_t}} \le 1$$

即**每参数每步的等效更新量被 $\eta$ 量级封顶**——lr 从 1e-4 调到 1e-2（100 倍），每一步的方向不变、幅度线性缩放，训练依然稳定。这正是"SGD 对 lr 敏感、Adam 鲁棒"的数学根源。

## 二、数学性质与直觉（几何解释）

### 2.1 逐参数自适应：对损失曲面做"粗糙曲率归一化"

除以 $\sqrt{\hat v_t}$ 等效于按每个参数的历史梯度幅度把坐标轴伸缩到"单位方差"（类似白化/whitening）：

- 梯度大的方向（陡峭轴）自动缩小步长 → 抑制震荡；
- 梯度小的方向（平缓轴、稀疏参数）自动放大步长 → 加快学习；
- **不同尺度的参数一视同仁**——对 embedding、词表等梯度稀疏的高维参数尤其关键（AdaGrad 最早解决该问题，Adam 继承了它）。

### 2.2 一阶矩 = 动量

$m_t$ 的展开与 Momentum 的 $v_t$ 完全同构（§1.1 的 $\gamma$ 即 $\beta_1$），因此 Adam 同时获得：同向加速、反向抵消、惯性穿越平坦区/鞍点。

### 2.3 更新量有界 → 天然"乐观"

$\hat m_t/\sqrt{\hat v_t}$ 是"单位化梯度"，逐元素落在 $[-1,1]$。这意味着 Adam 不会像 SGD 那样被某个巨大梯度顶飞——**大梯度出现时自动降速**，这是 Adam 收敛快的另一面。

### 2.4 二阶矩窗口的直觉

$\beta_2=0.999$ 对应约 1000 步的指数窗口——$v_t$ 反映"近期梯度尺度"而非瞬时值，因此对梯度变化响应慢（这也是 Adam 缺陷之一，见第五节）。

## 三、源码实现（手写 vs torch 官方，可直接运行）

### 3.1 手写 Adam 对比 torch.optim.Adam

损失函数 $y=(w\cdot x-b)^2$，手写版完整包含一阶矩、二阶矩与偏差校正：

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, beta1, beta2, eps, steps = 1e-2, 0.9, 0.999, 1e-8, 500

def adam_handmade(steps=500, bias_correction=True):
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    mw = mb = 0.0                       # 一阶矩 m
    vw = vb = 0.0                       # 二阶矩 v
    for t in range(1, steps + 1):
        pred = w * x - b
        dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
        mw = beta1 * mw + (1 - beta1) * dw          # m_t = β1·m + (1-β1)·g
        mb = beta1 * mb + (1 - beta1) * db
        vw = beta2 * vw + (1 - beta2) * dw * dw     # v_t = β2·v + (1-β2)·g²
        vb = beta2 * vb + (1 - beta2) * db * db
        if bias_correction:
            mw_h, mb_h = mw / (1 - beta1 ** t), mb / (1 - beta1 ** t)
            vw_h, vb_h = vw / (1 - beta2 ** t), vb / (1 - beta2 ** t)
        else:
            mw_h, mb_h, vw_h, vb_h = mw, mb, vw, vb
        w -= lr * mw_h / (torch.sqrt(vw_h) + eps)   # θ ← θ - η·m̂/(√v̂+ε)
        b -= lr * mb_h / (torch.sqrt(vb_h) + eps)
    return w.item(), b.item(), ((w * x - b) ** 2).mean().item()

w_h, b_h, loss_h = adam_handmade()
print(f"手写 Adam(带校正): w={w_h:.6f}  b={b_h:.6f}  loss={loss_h:.3e}")

# ---- torch 官方 ----
torch.manual_seed(0)
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(-0.8, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=lr, betas=(beta1, beta2), eps=eps)
for t in range(steps):
    opt.zero_grad()
    ((w * x - b) ** 2).mean().backward()
    opt.step()
print(f"torch Adam:      w={w.item():.6f}  b={b.item():.6f}  loss={((w*x-b)**2).mean().item():.3e}")
```

```text
手写 Adam(带校正): w=0.000257  b=0.000000  loss=9.103e-08
torch Adam:      w=0.000257  b=0.000000  loss=9.103e-08
```

逐位一致。**注意 torch.optim.Adam 的 `weight_decay` 参数实现的是"L2 扭曲版"**（混入梯度被二阶矩缩放），不是解耦权重衰减——要解耦请用 AdamW（见 AdamW 子篇）。

### 3.2 偏差校正到底改了啥：前 10 步对比

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)
lr, beta1, beta2, eps = 1e-2, 0.9, 0.999, 1e-8

def adam_run(correct, steps=10):
    torch.manual_seed(0)
    w = torch.tensor(1.5); b = torch.tensor(-0.8)
    mw = mb = 0.0; vw = vb = 0.0
    out = []
    for t in range(1, steps + 1):
        pred = w * x - b
        dw = (2 * pred * x).mean(); db = (-2 * pred).mean()
        mw = beta1 * mw + (1 - beta1) * dw
        mb = beta1 * mb + (1 - beta1) * db
        vw = beta2 * vw + (1 - beta2) * dw * dw
        vb = beta2 * vb + (1 - beta2) * db * db
        mw_h = mw / (1 - beta1 ** t) if correct else mw
        mb_h = mb / (1 - beta1 ** t) if correct else mb
        vw_h = vw / (1 - beta2 ** t) if correct else vw
        vb_h = vb / (1 - beta2 ** t) if correct else vb
        w -= lr * mw_h / (torch.sqrt(vw_h) + eps)
        b -= lr * mb_h / (torch.sqrt(vb_h) + eps)
        out.append((w.item(), b.item()))
    return out

with_corr = adam_run(True)
no_corr = adam_run(False)
for t in [1, 2, 5, 10]:
    print(f"step {t:2d}: 带校正 w={with_corr[t-1][0]:+.6f}   无校正 w={no_corr[t-1][0]:+.6f}")
```

```text
step  1: 带校正 w=+1.490000   无校正 w=+1.468377
step  2: 带校正 w=+1.480002   无校正 w=+1.425907
step  5: 带校正 w=+1.450030   无校正 w=+1.264786
step 10: 带校正 w=+1.400227   无校正 w=+0.956471
```

读表要点：第一步带校正只走 0.01（$\eta=10^{-2}$ 量级），**无校正直接走了 0.03**（$m_1$ 只有真实 10% 但 $v_1$ 只有真实 0.1%，$\hat m/\sqrt{\hat v}$ 被错误放大约 3 倍）；10 步后偏差累计让无校正版本多走了 5 倍路程——初期步长失真，配合大 lr 就会发散。

### 3.3 Adam 对 lr 的鲁棒性验证

```python
import torch

torch.manual_seed(0)
x = torch.linspace(-2, 2, 64)

for lr_ in [1e-3, 1e-1]:
    torch.manual_seed(0)
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(-0.8, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr_)
    for _ in range(5000):
        opt.zero_grad()
        ((w * x - b) ** 2).mean().backward()
        opt.step()
    print(f"lr={lr_:7.4f}: 5000 步后 loss={((w*x-b)**2).mean().item():.3e}")
```

```text
lr= 0.0010: 5000 步后 loss=8.809e-13
lr= 0.1000: 5000 步后 loss=0.000e+00
```

lr 差 100 倍都能收敛（快慢不同而已）——换 SGD 早就发散了。这就是"Adam 默认 1e-3 一般都能训出不错结果"的底气。

## 四、超参与调参经验（默认值必须背）

| 超参 | 默认值 | 含义与经验 |
|------|--------|-----------|
| $\beta_1$ | 0.9 | 一阶矩衰减率（动量强度，窗口约 10 步），一般不动 |
| $\beta_2$ | 0.999 | 二阶矩衰减率（窗口约 1000 步）；**部分 LLM 用 0.95**，响应更快更稳 |
| $\epsilon$ | $10^{-8}$ | 防除零；FP16/BF16 混合精度下建议 1e-6（防下溢） |
| $\eta$ | 0.001（原始论文）/ 3e-4（Transformer） | 唯一主要调节的超参 |
| weight_decay | 0 | **建议用 AdamW 而非 Adam 的 wd**（见 AdamW 子篇） |

调参流程：

1. lr 从 3e-4（Transformer）或 1e-3（通用）起步，看训练曲线；
2. loss 震荡/发散 → lr 减半，或加 warmup（初期 $\hat v$ 小，步长被错误放大，见 1.2）；
3. 后期收敛慢 → 加 cosine 退火；
4. 需要正则 → 换 AdamW，别用 Adam+wd。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 收敛快：动量 + 自适应步长双加速 | **泛化弱于 SGD**：倾向收敛到尖锐极小值（§六解释），测试精度常不如 SGD |
| 对 lr 鲁棒：默认 1e-3 即可训练，调参成本低 | 每参数多存 m、v 两份 FP32 状态（显存 ×3，7B 模型状态 56GB） |
| 稀疏参数友好：embedding 等梯度小的参数自动放大步长 | **权重衰减与 L2 纠缠**：wd 被 $\sqrt{\hat v}$ 扭曲，正则不可控（AdamW 的动机） |
| 初期偏差校正后估计无偏 | 二阶矩窗口过长（$\beta_2=0.999$），对梯度尺度变化响应慢 |
| 更新量有界，天然抗梯度爆炸 | 极端情况（lr 大 + $\hat v$ 小）步长 $\approx \eta/\epsilon$ 可震荡发散 |

**缺陷的工程对策**：预训练/新任务用 Adam 快速到位 → 后期切 SGD 或加长 cosine 精修 → 显存紧张换 Adafactor / 8-bit Adam。

## 六、与同类对比

| 维度 | AdaGrad | RMSProp | Adam |
|------|---------|---------|------|
| 历史梯度统计 | 全量累加 $\sum g^2$ | EMA $E[g^2]$ | EMA $E[g^2]$ + EMA $E[g]$ |
| 学习率单调递减 | 是（致命：后期冻死） | 否 | 否 |
| 动量 | 无 | 无 | **有** |
| 偏差校正 | 无 | 无 | **有** |
| 收敛速度 | 慢 | 中 | 快 |
| 地位 | 早期稀疏特征 | 被 Adam 取代 | **通用默认** |

关系链：**AdaGrad（全量累加）→ RMSProp（改 EMA，修单调递减）→ Adam（+动量+校正）**。

## 七、高频面试问答

**Q1：Adam 的更新公式？**
四行：$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$，$v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$，$\hat m_t=m_t/(1-\beta_1^t)$，$\hat v_t=v_t/(1-\beta_2^t)$，$\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。

**Q2：偏差校正推导？为什么早期影响大？**
$m_0=0$ → 展开 $m_t$ 是加权和，权重和 $=1-\beta_1^t<1$ → $E[m_t]=(1-\beta_1^t)E[g]$ → 除以 $1-\beta_1^t$ 得无偏。$t=1$ 时 $m_1$ 只有真实值 10%（$1-0.9=0.1$），$t=100$ 时 $0.9^{100}\approx 0$ 可忽略——校正只作用于初期，这也解释了 warmup 的必要性。

**Q3：Adam 为什么对 lr 不敏感？**
更新量 $\eta\hat m/\sqrt{\hat v}$ 中 $\hat m/\sqrt{\hat v}\le 1$（Cauchy 不等式），单步更新被 $\eta$ 封顶；lr 线性缩放更新幅度而非改变方向，稳定性远好于 SGD。

**Q4：Adam 和 SGD 泛化差异？**
主流解释：Adam 的自适应缩放使各方向步长均一，倾向落入尖锐极小值（泛化差）；SGD 等步长更易到达平坦极小值（泛化好）。对策：AdamW 预训练 + SGD 收尾，或加大权重衰减。

**Q5：Adam 的默认超参？**
$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$，$\eta=0.001$（Transformer 常用 3e-4）。

**Q6：Adam 的缺陷？**
四点：泛化弱于 SGD；权重衰减被二阶矩扭曲（→AdamW）；$\beta_2$ 窗口过长响应慢；极端 lr 下 $\eta/\epsilon$ 量级步长可发散。

**Q7：什么时候不用 Adam？**
追求极致泛化的小模型（CNN 用 SGD+Momentum 更好）；显存受限的大模型（换 Adafactor/8-bit Adam）；训练后期精修（切 SGD）。

**Q8：Adam 和 AdamW 的关系？**
AdamW = Adam 把权重衰减从"混入梯度（L2，被 $\sqrt{\hat v}$ 扭曲）"改成"独立于梯度的解耦收缩"。PyTorch 中两个独立类；Transformer/LLM 用 AdamW（详见 AdamW 子篇）。

## 八、自我检验 checklist

- [ ] 能默写 Adam 全部更新公式（含偏差校正）
- [ ] 能独立完成偏差校正推导：展开 $m_t$ → 权重和 $1-\beta^t$ → 除以 $1-\beta^t$
- [ ] 能说出默认超参（$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \eta=10^{-3}$）
- [ ] 能手写 Adam 循环并验证与 torch.optim.Adam 逐位一致
- [ ] 能解释"无校正时初期步长被放大 3 倍"的机理
- [ ] 能推导更新量上界 $\eta$ 并解释 lr 鲁棒性
- [ ] 能说出 Adam 四大缺陷与对应对策
- [ ] 能画出 AdaGrad→RMSProp→Adam 的进化链
- [ ] 能回答 8 个面试追问
