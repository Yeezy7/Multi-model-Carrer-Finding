# MSE 与 MAE（回归损失）与 Smooth L1

> 本模块索引见 [损失函数详解](损失函数详解.md)

## 一、定义与公式（含完整推导）

### 1.1 MSE（均方误差）

预测 $\hat{y}$，真实 $y$，$N$ 个样本：

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**从高斯噪声的最大似然推导**：假设观测 $y = f(x) + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，则

$$P(y|x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \hat{y})^2}{2\sigma^2}\right)$$

取负对数似然，丢掉与参数无关的常数项：

$$-\log P(y|x) = \frac{(y - \hat{y})^2}{2\sigma^2} + \text{const} \propto (y - \hat{y})^2$$

**MSE 是"高斯噪声假设"下的最大似然估计**——这解释了它为什么是回归默认损失。

### 1.2 MAE（平均绝对误差）

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**从拉普拉斯噪声推导**：$\epsilon \sim \text{Laplace}(0, b)$ 时，负对数似然 ∝ $|y - \hat{y}|$。所以 **MSE/MAE 的选择本质上是对噪声分布（高斯 vs 拉普拉斯）的假设**。

### 1.3 Smooth L1（Huber Loss 的特例）

$$\text{SmoothL1}(x) = \begin{cases} 0.5\,x^2 & |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}, \qquad x = y - \hat{y}$$

- $|x| < 1$ 时用 MSE（二次，最优点附近平滑收敛）；
- $|x| \ge 1$ 时用 MAE（线性，对离群点鲁棒）；
- 在 $|x|=1$ 处连续且导数连续（$1 = 1$），比 Huber 少一个 $\delta$ 超参。

### 1.4 批量形式对比表

| 损失 | 公式 | 单样本梯度 |
|------|------|-----------|
| MSE | $(y-\hat{y})^2$ | $2(\hat{y}-y)$（∝ 误差大小） |
| MAE | $\|y-\hat{y}\|$ | $\pm 1$（恒定） |
| SmoothL1 | 见 1.3 | $x$（\|x\|<1）或 $\pm1$ |

## 二、数学性质与直觉

### 2.1 异常值敏感性（最重要的直觉）

| 误差 | 平方（MSE） | 绝对值（MAE） | 相对放大 |
|------|-----------|--------------|---------|
| 0.1 | 0.01 | 0.1 | MSE 更小 |
| 1.0 | 1.0 | 1.0 | 相等 |
| 5.0 | 25.0 | 5.0 | MSE 5 倍 |
| 10.0 | 100.0 | 10.0 | MSE 10 倍 |

**MSE 对离群点进行平方放大**：一个误差 10 的样本，对损失的贡献等于 100 个误差 1 的样本。数据有噪声标签/离群点时，MSE 的梯度会被单个离群点主导——这是选 MAE/SmoothL1 的核心动机。

### 2.2 凸性与最优解

- 两者都是凸函数，MSE 最优解是**均值** $\mathbb{E}[y]$，MAE 最优解是**中位数** $\text{median}(y)$；
- 中位数对离群点鲁棒（不受尾部影响），均值会被离群点拉偏——再次印证"数据有离群点选 MAE 系"；
- SmoothL1 的最优解介于均值与中位数之间，是两者的稳健折中。

### 2.3 梯度行为：收敛动态的差异

- **MSE**：误差越大梯度越大 → 初期收敛快，但离群点会造成大梯度抖动，且接近最优点时梯度线性趋小（收敛慢但不振荡）；
- **MAE**：梯度恒为 ±1 → 对离群点温和，但**接近最优点时梯度仍为 1**，会在最优值附近持续振荡，收敛精度差；
- **SmoothL1**：小误差区间二次（梯度→0，精细收敛），大误差区间线性（鲁棒）——**两头的好处都占**。

### 2.4 与分类损失的区别

回归损失度量"数值差距"，分类损失度量"置信度错误"。分类任务用 MSE 时（错配）梯度会被"置信度饱和"抵消（sigmoid 饱和区梯度→0），且没有概率解释——所以分类不用回归损失。

## 三、源码实现（手写版本 + PyTorch 官方接口）

### 3.1 手写版

```python
import torch

def mse_manual(y_pred, y):
    return ((y_pred - y) ** 2).mean()

def mae_manual(y_pred, y):
    return (y_pred - y).abs().mean()

def smooth_l1_manual(y_pred, y, beta=1.0):
    """beta=1 即标准 SmoothL1（beta 可调，PyTorch 的 smooth_l1 支持 beta）"""
    x = (y_pred - y).abs()
    return torch.where(x < beta,
                       0.5 * x ** 2 / beta,
                       x - 0.5 * beta).mean()

y = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.2, 1.8, 2.5])
print(mse_manual(y_pred, y))        # tensor(0.1100)
print(mae_manual(y_pred, y))        # tensor(0.3000)
print(smooth_l1_manual(y_pred, y))  # tensor(0.0550)
```

### 3.2 PyTorch 官方接口

```python
import torch.nn as nn

criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_sl1 = nn.SmoothL1Loss(beta=1.0)

print(criterion_mse(y_pred, y).item())     # 0.1100
print(criterion_mae(y_pred, y).item())     # 0.3000
print(criterion_sl1(y_pred, y).item())     # 0.0550

# 注意输入顺序：第一个参数是预测，第二个是目标
print(criterion_mse(y, y_pred).item())     # 0.1100 —— MSE 对调顺序结果相同
```

### 3.3 输出对比验证（手写 vs 官方）

```python
torch.manual_seed(0)
yp = torch.randn(8, 4)
y_true = torch.randn(8, 4) * 2 + 1
print(mse_manual(yp, y_true).item(), nn.MSELoss()(yp, y_true).item())
# 输出示例：5.245638 5.245638（三者恒等）
print(mae_manual(yp, y_true).item(), nn.L1Loss()(yp, y_true).item())
# 输出示例：1.840854 1.840854
print(smooth_l1_manual(yp, y_true).item(), nn.SmoothL1Loss()(yp, y_true).item())
# 输出示例：1.263695 1.263695
```

### 3.4 离群点敏感性演示

```python
# 同一批数据，第 3 个样本变成离群点
y_clean = torch.tensor([1.0, 2.0, 3.0])
yp_clean = torch.tensor([1.2, 1.8, 2.5])
y_out = torch.tensor([1.0, 2.0, 3.0])
yp_out = torch.tensor([1.2, 1.8, 10.5])      # 预测离群

print("干净数据:", mse_manual(yp_clean, y_clean).item(),   # 0.1100
      mae_manual(yp_clean, y_clean).item(),                # 0.3000
      smooth_l1_manual(yp_clean, y_clean).item())          # 0.0550
print("带离群点:", mse_manual(yp_out, y_out).item(),       # 18.7767 ← 爆炸
      mae_manual(yp_out, y_out).item(),                    # 2.6333 ← 温和
      smooth_l1_manual(yp_out, y_out).item())              # 2.3467 ← 温和
```

### 3.5 目标检测框回归（SmoothL1 的经典用法）

```python
def smooth_l1_box_loss(pred_boxes, target_boxes, weights=None):
    """Faster R-CNN 式：delta 编码的框回归 + SmoothL1"""
    loss = nn.SmoothL1Loss(reduction='none')(pred_boxes, target_boxes)
    loss = loss.mean(dim=-1)                       # 4 个坐标平均
    if weights is not None:                        # 只对正样本框计损
        loss = (loss * weights).sum() / weights.sum().clamp(min=1)
        return loss
    return loss.mean()

pred_boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
target_boxes = torch.tensor([[0.12, 0.22, 0.25, 0.45]])
print(smooth_l1_box_loss(pred_boxes, target_boxes).item())   # tensor(0.0007)
```

## 四、梯度分析

### 4.1 三者的梯度对比（$x = \hat{y} - y$）

$$\frac{\partial \text{MSE}}{\partial \hat{y}} = 2(\hat{y} - y) = 2x, \qquad \frac{\partial \text{MAE}}{\partial \hat{y}} = \text{sign}(x), \qquad \frac{\partial \text{SmoothL1}}{\partial \hat{y}} = \begin{cases} x & |x| < 1 \\ \text{sign}(x) & \text{otherwise} \end{cases}$$

| 误差 x | MSE 梯度 | MAE 梯度 | SmoothL1 梯度 |
|--------|---------|---------|--------------|
| 0.1 | 0.2 | 1 | 0.1 |
| 0.5 | 1.0 | 1 | 0.5 |
| 1.0 | 2.0 | 1 | 1.0 |
| 5.0 | 10.0 | 1 | 1.0 |
| 10.0 | 20.0 | 1 | 1.0 |

### 4.2 梯度行为解读

- **MSE 梯度随误差线性增长**：离群点会产生超大梯度 → 需要小学习率，或梯度被其他任务淹没（多任务不均衡）；
- **MAE 梯度恒定 ±1**：离群点无害，但最优点附近梯度不衰减 → 收敛到最优点附近后持续振荡，精度受限；
- **SmoothL1 梯度分段**：$|x|<1$ 时随误差线性趋 0（精细收敛），$|x|\ge1$ 时恒定（鲁棒）——**梯度视角下它是 MSE 与 MAE 的完美折中**；
- 推导 MSE 二阶导恒为 2（凸二次），MAE 在 0 处不可导（次梯度），SmoothL1 处处可导。

### 4.3 数值验证

```python
x_g = torch.tensor([2.0], requires_grad=True)      # 误差 2 的单样本
mse_manual(x_g, torch.zeros(1)).backward()
print(x_g.grad)    # tensor([4.0]) —— 2*2 = 4
x_g2 = torch.tensor([2.0], requires_grad=True)
smooth_l1_manual(x_g2, torch.zeros(1)).backward()
print(x_g2.grad)   # tensor([1.0]) —— 线性区梯度恒 1
```

## 五、数值稳定性

1. **平方上溢**：误差大时 $x^2$ 可溢出（FP16 下 $x>256$ 就爆）→ 大数值回归任务先归一化目标（除以尺度），或用 SmoothL1/MAE；
2. **MSE 在 FP16 下的累积**：loss 大但梯度正常，主要是"损失数值"展示问题；梯度 $2x$ 在 $x$ 大时也可能溢出 → clamp 预测值；
3. **检测框回归的关键**：框坐标绝对值范围大（0~1000px）→ 必须用 delta 编码（dx, dy, dw, dh 的对数/归一化形式），否则 MSE 的平方放大直接爆炸——Faster R-CNN 的工程细节；
4. **梯度消失的反面**：SmoothL1 在 $|x|<1$ 时梯度 $x\to0$，若所有样本都到小误差区，梯度会整体变细（收敛变慢属正常）；
5. 深度估计/光度误差等任务常用相对误差（$\frac{\|y-\hat{y}\|}{y}$）替代绝对误差，防止深度值大的区域主导损失。

## 六、使用场景（含多模态场景）

| 场景 | 损失 | 说明 |
|------|------|------|
| 一般回归（数值稳定数据） | MSE | 高斯噪声假设，收敛快 |
| 含离群点回归 | MAE / SmoothL1 | 稳健 |
| 目标检测框回归 | SmoothL1（delta 编码） | Faster R-CNN 标配 |
| 图像重建（去噪/超分） | MSE（L2）或 MAE（L1） | L2 偏平滑、L1 保边缘 |
| 深度估计 | SmoothL1 / BerHu | 深度值范围大 |
| 姿态估计 | L1 / L2（heatmap 版用 CE） | 多峰值回归问题 |

**多模态中的四个高频位置**：
1. **检测类多模态模型**（如指代表达 grounding）：框回归头用 SmoothL1 + delta 编码；
2. **图像/视频重建与压缩**：自监督重建项常用 MSE（VQ-VAE 的 codebook 重建、MAE 视觉重建的像素 L2）；
3. **CLIP 式模型的度量对齐（微调）**：有时在 InfoNCE 之外加 embedding 的 L2 对齐项（蒸馏场景）；
4. **扩散模型的噪声预测**：损失本质是 MSE（预测噪声 $\epsilon$ 与真实噪声的 L2），重加权版本（SNR 加权）是对 MSE 的改良。

> **重点**：扩散模型的核心损失（$\mathbb{E}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$）就是 MSE——理解了 MSE 的"高斯噪声最大似然"视角，就能理解扩散为什么用 L2 预测噪声。

## 七、优缺点总结

| 损失 | 优点 | 缺点 |
|------|------|------|
| MSE | 凸、光滑、大误差快速修正；高斯 MLE 理论 | 离群点主导；大误差平方爆炸 |
| MAE | 离群点鲁棒；梯度恒定稳定 | 最优点附近振荡；0 处不可导 |
| SmoothL1 | 两端优势兼备；处处可导 | 需 beta/阈值；小于 1 时梯度变细 |

## 八、高频面试问答

**Q1：MSE 和 MAE 怎么选？**
看数据噪声与离群点：干净数据用 MSE（收敛快、理论好）；有离群点用 MAE/SmoothL1。本质是噪声分布的假设差异（高斯 vs 拉普拉斯）。

**Q2：为什么离群点对 MSE 影响这么大？**
平方放大：误差 10 的样本损失 = 100 个误差 1 的样本。梯度也线性放大（$2x$），单个离群点就能主导一次参数更新。

**Q3：MAE 的缺点？**
最优点附近梯度恒为 ±1 不衰减 → 收敛到小误差区后持续振荡，无法精细收敛。另外 0 处不可导（用次梯度）。

**Q4：SmoothL1 为什么比两者好？**
分段设计：小误差用二次（梯度→0，精细收敛），大误差用线性（梯度恒 1，鲁棒），在 |x|=1 处连续可导。目标检测框回归的标准选择。

**Q5：检测框回归为什么必须 SmoothL1 + delta 编码？**
框坐标绝对值范围大，MSE 的平方放大在尺度大的坐标上爆炸；delta 编码（对数宽高、归一化平移）把目标归一化到小范围，配合 SmoothL1 的线性区对离群框鲁棒。

**Q6：为什么扩散模型预测噪声用 MSE？**
扩散的每步目标是估计高斯噪声（重参数化的高斯最大似然），其负对数似然正是 L2 距离；加 SNR 重加权后仍是 MSE 家族。

**Q7：MSE 的缺点在分类任务上表现为什么？**
与 sigmoid 组合时梯度被饱和区抵消（梯度消失）；且 MSE 假设高斯噪声，对离散标签的建模错误——分类必须用 CE/BCE。

## 九、自我检验

- [ ] 能从高斯/拉普拉斯噪声 MLE 推出 MSE/MAE
- [ ] 能画出三个损失的曲线与梯度曲线
- [ ] 会手写三个损失并用官方接口验证一致
- [ ] 能解释离群点敏感性（误差 10 vs 100 个误差 1）
- [ ] 知道 SmoothL1 分段点处连续可导
- [ ] 知道检测框回归的 delta 编码与 SmoothL1 搭配
- [ ] 知道扩散模型损失就是 MSE
- [ ] 能回答 7 个面试追问
