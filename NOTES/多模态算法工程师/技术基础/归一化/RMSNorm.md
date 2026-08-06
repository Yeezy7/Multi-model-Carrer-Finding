# RMSNorm（均方根归一化）

> 本模块索引见 [归一化技术详解](归一化技术详解.md)

## 一、定义与公式

RMSNorm（Root Mean Square Normalization，Zhang & Sennrich, 2019）是 LayerNorm 的简化版：**去掉均值中心化，只按均方根（RMS）缩放**。它是现代大模型（LLaMA、Qwen、Mistral、Gemma）的标配。

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{d=1}^{D} x_d^2 + \epsilon}$$

$$\hat{x}_d = \frac{x_d}{\text{RMS}(x)}, \qquad y_d = \gamma_d \hat{x}_d$$

- 没有 $\mu$ 项、一般也没有 $\beta$（主流实现只有 $\gamma$，PyTorch 的 `nn.RMSNorm` 支持 `elementwise_affine` 但默认无 bias）；
- $\epsilon$：防除零，LLaMA 用 1e-6，一般取 1e-5~1e-6；
- 对输入 $x \in \mathbb{R}^{N \times L \times D}$，对最后特征维逐位置计算。

| 属性 | 值 |
|------|-----|
| 统计维度 | 单样本全特征（最后维） |
| 依赖 batch | 无 |
| 训练/推理行为 | 完全一致 |
| 可学习参数 | γ 每特征维度（无 β） |
| 计算量 | 比 LN 少一次均值归约与一次减法（约省 30%） |
| 代表模型 | LLaMA 1/2/3、Qwen、Mistral、Gemma |

## 二、数学性质

### 2.1 为什么可以去掉均值（核心动机）

原论文观察：**LN 中均值减除（centering）对最终效果的贡献很小**，归一化的主要收益来自缩放（scaling）让激活保持在稳定的量级。在残差网络 + 现代初始化（如 Llama 风格）下，各层激活均值天然接近 0，减均值变得可有可无；而省掉它直接减少一次全局归约（reduce）和一次减法，对训练/推理吞吐都有实打实的收益。

### 2.2 输出范数恒定性质（RMSNorm 的"甜点"）

当 $\gamma = 1$ 时：

$$\lVert y \rVert_2 = \frac{\lVert x \rVert_2}{\text{RMS}(x)} = \sqrt{D}$$

即 RMSNorm 把每个位置的输出**钉在 L2 范数恰好为 $\sqrt{D}$ 的球面上**（方向不变、只缩放模长）。相比 LN 的"减均值再缩放"，RMSNorm 保留了均值信息（见 2.3），这在因果语言模型逐 token 预测中往往是优势。

### 2.3 均值去哪了？

RMSNorm 输出均值 = 输入均值 / RMS(x)，**不等于 0**——它不像 LN 那样移除均值，而是保留输入的"整体偏移"信息。当输入均值本就 ≈ 0（标准正态输入）时，两者几乎一致：

| 输入情况 | LN 输出均值 | RMSNorm 输出均值 | 两者差异 |
|---------|-----------|-----------------|---------|
| 标准正态（均值≈0） | 0 | ≈0.14（来自有限样本的随机偏差） | 小 |
| 均值=5（输入整体偏移） | 0（被减掉） | ≈0.98（保留偏移） | 大 |

实验验证（同一随机输入）：

```python
import torch
torch.manual_seed(0)
x = torch.randn(4, 16)

for name, xx in [("均值≈0", x), ("均值=5", x + 5.0)]:
    ln_out = (xx - xx.mean(-1, keepdim=True)) / xx.std(-1, unbiased=False, keepdim=True)
    rms_out = xx / torch.sqrt(xx.pow(2).mean(-1, keepdim=True) + 1e-5)
    ln_mean = ln_out.mean(-1).abs().mean().item()
    rms_mean = rms_out.mean(-1).abs().mean().item()
    diff = (ln_out - rms_out).abs().mean().item()
    print(f"{name}: LN 输出均值={ln_mean:.5f}, RMS 输出均值={rms_mean:.5f}, 平均绝对差={diff:.4f}")
# 均值≈0: LN 输出均值=0.00000, RMS 输出均值=0.13964, 平均绝对差=0.1396
# 均值=5: LN 输出均值=0.00000, RMS 输出均值=0.98033, 平均绝对差=1.0700
```

> **结论**：均值中心化的唯一效果是"移除均值"，而现代大模型实践中这既非必需也无明显收益——这是 RMSNorm 的核心论点。

### 2.4 缩放不变性（与 L2 归一化的关系）

RMSNorm（γ=1）对输入的整体缩放完全不变：

$$f(c \cdot x) = \frac{c x}{\text{RMS}(c x)} = \frac{c x}{c \cdot \text{RMS}(x)} = f(x)$$

进一步展开：$\text{RMS}(x) = \lVert x \rVert_2 / \sqrt{D}$，所以

$$\hat{x} = \frac{x}{\lVert x \rVert_2} \cdot \sqrt{D}$$

**RMSNorm（γ=1）≡ 每个位置做 L2 归一化再乘 √D**——它把方向归一化（L2 normalize，与 CLIP 双塔输出层同款操作）和尺度设定（√D）合二为一。这也是为什么说 RMSNorm 的输出"天然适合做相似度/注意力计算"：方向已归一化，内积即余弦相似度。

```python
torch.manual_seed(0)
x = torch.randn(3, 5, 16) * 7.3                     # 任意尺度的输入
rms_out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
l2_out = x / x.norm(dim=-1, keepdim=True) * (16 ** 0.5)
print((rms_out - l2_out).abs().max().item())        # 2.38e-7：除 eps 外完全等价
```

## 三、源码实现

### 3.1 LLaMA 风格实现（HuggingFace/官方一致）

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """LLaMA 官方实现（hf 仓库逐行一致）"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.rsqrt(rms) * self.weight

torch.manual_seed(0)
x = torch.randn(4, 6, 16)
y1 = RMSNorm(16)(x)
y2 = nn.RMSNorm(16)(x)                # PyTorch ≥2.4 官方接口
print((y1 - y2).abs().max().item())   # 2.86e-06：与官方一致（仅浮点顺序差异）
```

### 3.2 输出范数 = √D 性质验证

```python
torch.manual_seed(0)
x = torch.randn(2, 8, 64)
y = RMSNorm(64)(x)
print(y.norm(dim=-1).mean().item())   # 8.0000 = sqrt(64)：每个位置范数钉在 √D
```

### 3.3 与 LayerNorm 的等价性实验（均值≈0 时）

当输入已零中心时，$x^2$ 的均值等于方差，RMSNorm 与 LN（γ=1、β=0）的归一化因子相同，输出只差一个符号无关的缩放方向：

```python
torch.manual_seed(0)
x = torch.randn(2, 8, 32)                       # 标准正态：均值≈0
ln_y = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-6)
rms_y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
print((ln_y - rms_y).abs().max().item())        # 0.4634（D=32，差异来自有限样本均值不为 0）
print((ln_y - rms_y).abs().mean().item())       # 0.1410
```

> 差异随 D 增大快速趋近 0（D=256 时均值差 0.055，D=4096 时 0.014）——大模型 d_model 达数千，两者几乎重合，这是"去掉均值效果不大"的直接证据。

### 3.4 在 LLaMA Decoder Block 中的用法（唯一归一化层）

```python
class LlamaDecoderBlock(nn.Module):
    """LLaMA 风格 block：RMSNorm 出现在注意力和 FFN 之前（Pre-Norm），残差不经过任何归一化"""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)                    # 注意力前
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.post_attention_layernorm = RMSNorm(dim)           # FFN 前
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = x + self.attn(self.input_layernorm(x), self.input_layernorm(x), self.input_layernorm(x))[0]
        return h + self.mlp(self.post_attention_layernorm(h))

torch.manual_seed(0)
block = LlamaDecoderBlock(64, 4)
y = block(torch.randn(2, 10, 64))
print(y.shape)                                        # torch.Size([2, 10, 64])
```

> LLaMA 系列模型中：token embedding 之后 → 每个 block 前 → 最终输出前（`model.norm`），三处全部使用 RMSNorm。

## 四、深入分析

### 4.1 梯度行为

RMSNorm 的梯度比 LN 更简单（没有均值路径）：

$$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}(x)} \left( g_i - \hat{x}_i \cdot \frac{1}{D}\sum_j g_j \hat{x}_j \right)$$

只保留"直接项 + 方差路径"两项，少一项均值路径。**梯度投影到与 $\hat{x}$ 正交的方向**，输出方向上的分量被移除——这防止了激活方向的无界增长，是它稳定训练的原因之一。

### 4.2 训练稳定性：为什么 LLaMA 系列效果还更好

论文及后续实践（LLaMA 论文附录、Qwen 技术报告）报告：去掉均值中心化后，大模型训练的 loss 曲线与 LN 相当甚至更好。可能的原因：

1. 残差结构下均值中心化会**移除残差流的直流分量**，而该分量可能携带有用信号（如 token 频率、句长信息）；
2. 更少的归约操作 → 更少的数值误差传播；
3. 参数更少（无 β）→ 优化更简单。

### 4.3 数值细节

- **eps 选择**：LLaMA 官方 1e-6；BF16 训练下建议 1e-6~1e-5，防止 `rsqrt` 除零；
- **精度**：`pow(2).mean` 的归约在 FP16 下可能溢出（值域大的中间结果），HF 实现统一在 FP32 计算 RMS、输出转回输入精度（新版 `F.rms_norm` 内部已处理）；
- `torch.rsqrt` 比先 `1/sqrt` 更稳、更快（单指令）。

### 4.4 推理部署

RMSNorm 无 running stats，ONNX/TensorRT 直接导出即可；在长序列推理（KV cache 场景）下，每 token 的 RMS 计算独立，无跨位置依赖，便于算子融合。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 比 LN 少一次均值归约与减法，约快 7%~30% | 不移除均值，对输入的绝对偏移敏感（理论劣势，实践影响小） |
| 参数更少（只有 γ） | 均值非零中心 → 输出分布有直流分量 |
| 训练稳定性与 LN 相当甚至更好（LLaMA 验证） | 归一化不彻底（只有缩放没有平移），统计理论上的"标准正态"保证弱于 LN |
| 实现极简（3 行 forward） | 论文为 Transformer/NLP 场景设计，CNN 任务验证少 |
| 输出范数钉在 √D，尺度可控 | 特征维 D 极小时（如 D<32）均方根估计噪声大 |

## 六、与同类归一化对比

| 维度 | RMSNorm | LayerNorm | BatchNorm |
|------|---------|-----------|-----------|
| 均值减除 | 无 | 有 | 有 |
| 缩放因子 | 均方根 | 标准差 | 标准差 |
| 可学习参数 | γ（无 β） | γ、β | γ、β |
| 计算量 | 最低（一趟归约） | 中（两趟归约） | 中 |
| 依赖 batch | 无 | 无 | 强 |
| 训练/推理一致 | 一致 | 一致 | 不一致 |
| 代表模型 | **LLaMA/Qwen/Mistral/Gemma** | BERT/ViT/原版 Transformer | ResNet 等 CNN |
| 典型任务 | LLM 预训练/微调 | 多模态双塔、预训练模型 | 图像分类检测 |

**选型规律（面试必答）**：BERT 时代用 LayerNorm；LLM 时代（LLaMA 系）全部换成 RMSNorm——省算力 + 更稳定 + 实现简单，三位一体。

## 七、高频面试问答

**Q1：RMSNorm 和 LayerNorm 的区别？**
RMSNorm 去掉均值中心化，只按均方根缩放：无 μ、无 β，只保留 γ。少一次归约与减法（约快 30%），参数更少，训练稳定性相当或更好。

**Q2：为什么 LLaMA 用 RMSNorm 不用 LayerNorm？**
① 训练/推理吞吐省算力（大模型每 token 都要过归一化，省一次归约是实打实的）；② 论文与 LLaMA 实践表明去掉 centering 不伤效果甚至更稳；③ 实现简单，只有 γ 一个参数。

**Q3：RMSNorm 输出范数有什么性质？**
γ=1 时输出 L2 范数恒等于 √D（方向保留、模长固定），每个位置的表示被钉在超球面上，尺度完全可控。

**Q4：去掉均值会损失什么吗？**
理论上有——输入的绝对偏移（直流分量）不再被移除，保留在输出中。但残差网络 + 现代初始化的实践中，均值信息不构成干扰，反而可能携带有用信号（LLaMA 效果更好）。

**Q5：RMSNorm 的 eps 一般取多少？为什么？**
LLaMA 用 1e-6；混合精度（BF16）下建议 1e-6~1e-5，过小会导致 rsqrt 除零或精度爆炸。

**Q6：RMSNorm 可以用在 CNN 吗？**
可以（对 [C,H,W] 或按位置对特征维），但实践少；CNN 的主流仍是 BN（大 batch）或 GN（小 batch）。

**Q7：为什么说 RMSNorm 适合长序列推理？**
每个 token 的归一化计算相互独立、无跨位置归约，方便算子融合与 KV cache 流式解码；且无 running stats，导出/部署零额外处理。

**Q8：HF 的 LlamaRMSNorm 与手写版本有什么区别？**
HF 用 `torch.nn.functional.rms_norm`（底层有 fused kernel），且内部按输入精度处理统计量计算（FP32 归约）；数学上完全等价，手写版便于阅读。

**Q9：RMSNorm 和 L2 归一化（特征归一化）有什么关系？**
γ=1 时两者数学等价：RMSNorm = 对每个位置做 L2 归一化 × √D（方向归一 + 模长设定）。区别在用法：L2 归一化通常用在模型输出（相似度计算），RMSNorm 用在网络内部每层（训练稳定）——同一操作的两个舞台。

**Q10：RMSNorm 和 Weight Standardization 一样吗？**
不一样。RMSNorm 归一化激活值（每个位置的特征向量）；Weight Standardization 归一化卷积/线性层的权重向量（对 weight 求 μ/σ）。两者可叠加（如一些稳定训练方案），但作用对象完全不同。

## 八、自我检验

- [ ] 能写出 RMSNorm 公式并指出与 LN 的唯一差别（无均值）
- [ ] 能解释"为什么可以去掉均值"的论文动机
- [ ] 能手写 LLaMA 风格 RMSNorm 并与 nn.RMSNorm 对齐
- [ ] 能说出输出范数 = √D 的性质并写代码验证
- [ ] 能演示"均值≈0 时 RMSNorm≈LN"的实验
- [ ] 能说出 LLaMA/Qwen 选 RMSNorm 的三条理由
- [ ] 知道 RMSNorm 的梯度只有两项（无均值路径）
- [ ] 知道 RMSNorm（γ=1）≡ L2 归一化 × √D
- [ ] 能回答 10 个面试追问
