# Transformer 初始化：N(0, 0.02)、Scaled Init 与 DeepNorm

> 本模块索引见 [参数初始化详解](参数初始化详解.md)

## 一、定义与公式

Transformer 的初始化自成体系：既不用 Xavier（尺度偏大），也不用 Kaiming（ReLU 专用），而是围绕**残差结构 + LayerNorm 兜底**设计的一套保守小尺度方案。

### 1.1 三个核心公式

**① 全量权重（BERT/GPT/HF 默认）**：

$$W \sim \mathcal{N}(0, 0.02^2)$$

（HF 中为 `initializer_range = 0.02`；Google BERT 官方用**截断正态**：从 $\mathcal{N}(0, 0.02^2)$ 采样，裁掉 $\pm 2\sigma$ 外样本重新归一化，PyTorch 里 `trunc_normal_(w, std=0.02)`。）

**② GPT-2 scaled init（残差分支缩放）**：

$$W_{out\_proj} \sim \mathcal{N}\left(0, \left(\frac{0.02}{\sqrt{N}}\right)^2\right), \quad N = 2 \times \text{层数}$$

（每个 block 有 attention、FFN 两个残差分支，HF 实现里 `scale = 1/sqrt(2·n_layer)`。）

**③ DeepNorm（Post-LN 深度模型）**：

$$x_{l+1} = LN\left(\alpha x_l + G_l(x_l)\right), \quad W_{FFN/v/out} \sim \text{Xavier}(gain = \beta)$$

其中 $\alpha = (2N)^{1/4}$、$\beta = (8N)^{-1/4}$（encoder-only，N 为层数）。

### 1.2 直觉

| 尺度 | 量级（d_model=768） | 用途 |
|------|--------------------|------|
| N(0, 0.02) | std=0.02 | embedding、q/k/v/o、FFN 全量 |
| N(0, 0.01) | std=0.01 | GPT-2 残差分支（12 层） |
| N(0, 0.007) | std=0.007 | GPT-2 残差分支（48 层） |
| Xavier | ≈0.051 | 仅 DeepNorm 的 q/k（gain=1） |

> 对比 Xavier 的 $\sqrt{2/768} \approx 0.051$：N(0,0.02) 小了 2.5 倍。**Transformer 初始化哲学 = "初始行为接近恒等映射 + 依赖 LayerNorm 保尺度"**。

## 二、数学原理

### 2.1 为什么是 0.02（而不是推导出来的）

N(0, 0.02) 是**经验选择**，不是推导结果——但能解释为什么它合理：

1. **残差恒等起点**：$x_{l+1} = x_l + F(x_l)$，初始化应让 $F(x_l)$ 远小于 $x_l$，这样 $L$ 层堆叠后输出 ≈ 输入，深层信号不衰减不爆炸；
2. **embedding 无 LN 保护**：embedding 直接进第一层，且直接参与 loss，std 太大则初始 loss 不稳定；
3. **数值稳定**：d_model 大时逐点累加值偏大，FP16 下 0.02 量级安全；
4. **softmax 不饱和**：QKᵀ/√d_k 的 logits 量级与 std² 相关，0.02 保证初始 attention 分布不太尖锐。

### 2.2 残差方差累积：为什么需要 1/√N 缩放

残差结构下第 $l$ 层输出 $x_l = x_0 + \sum_{i<l} F_i(x_i)$。若每个残差分支 $F_i$ 输出方差为 $\sigma^2$ 且相互独立，则：

$$Var(x_L) = Var(x_0) + L \cdot \sigma^2 \approx L \sigma^2$$

**方差随深度线性增长**（而非指数）。解法：把每个残差分支输出缩小 $\frac{1}{\sqrt{L}}$，则：

$$Var(x_L) \approx Var(x_0) + L \cdot \frac{\sigma^2}{L} = Var(x_0) + \sigma^2$$

深层与浅层方差只差一个常数，不再随深度增长。**注意是 1/√L 而不是 1/L**：因为方差要缩 L 倍，对应 std 缩小 √L 倍。

> **实现细节**：若残差分支的输入**没有**被 LN/RMSNorm 固定尺度（分支方差会随当前 x 的方差一起膨胀），实际增长会快于线性。因此实验 3.3 在分支前做了 RMSNorm 式归一化，让"线性累积"理论可被精确验证——这也解释了为什么"分支输入必须过 LN"与"分支输出必须小"是同一件事的两个侧面。

### 2.3 T-Fixup：从解析推导看 Transformer 训练不稳定的根源

Huang et al. (ICML 2020) 定量分析了 Adam + LN 下的方差传播，给出**无 warmup、甚至无 LN** 也能训练的初始化：

1. 全部 bias 置 0；
2. embedding 用高斯（$\sigma \sim d^{-1/2}$ 量级）；
3. attention 的 value 投影、输出投影与 FFN 权重乘 **$0.67 \cdot N^{-1/4}$**（N 为层数）；
4. 解释：深层 Transformer 的更新量随层数增长，根源在"每个残差分支都做全量更新"——按 $N^{-1/4}$ 缩放后更新量级与层数无关。

> 推导直觉：Adam 的更新量 $\sim \text{lr} \times \text{符号方向}$（与梯度量级解耦），残差叠加 L 次后权重偏移 $\sim \sqrt{L}$ 量级，故需按 $N^{-1/4}$ 缩放使更新有界。T-Fixup 证明了"缩放残差分支"是普适原理，DeepNorm、GPT-2 scaled init 都是它的特例/变体。

### 2.4 DeepNorm 为什么能训 1000 层

DeepNet (Wang et al., 2022) 把残差改为 $x_{l+1} = LN(\alpha x_l + G_l(x_l))$，α 放大直连路径、β 缩小子层权重，并证明模型更新量被常数界定：

$$\|\Delta F\| \le \sum \frac{\sqrt{v^2 + w^2}}{\alpha}\|\theta^* - \theta\|$$

- **α > 1**：放大恒等路径，让"模型更新"相对"残差噪声"变小；
- **β < 1**：子层初始输出缩小，初始更像恒等映射；
- 结果：Post-LN 的表达能力（不因旁路 LN 削弱表征）+ Pre-LN 的训练稳定性，可训到 1000 层。

### 2.5 为什么只缩 FFN / v_proj / out_proj，不缩 q/k

| 参数 | 作用 | 是否缩放 | 原因 |
|------|------|---------|------|
| q_proj / k_proj | 只决定 attention **权重** | 不缩（gain=1） | QKᵀ 有 1/√d_k 缩放 + softmax 归一化，不改变输出幅度上界（DeepNet Lemma 4.1） |
| v_proj | 决定 attention 输出**幅度** | ×β | V 直接进入输出求和 |
| out_proj | 残差分支末端 | ×β | 输出要加回残差，必须小 |
| FFN 两层 | 同样进入残差 | ×β | 与 attention 同理 |

**通用原则**：只影响"加权方式"的参数（q/k、bias）可保持标准尺度；影响"信号幅度"且输出进入残差的参数（v、out_proj、FFN）必须额外缩小。

## 三、源码实现

### 3.1 手写三件套（对照公式）

```python
import math
import torch
import torch.nn as nn

def normal_002(w, std=0.02):
    """N(0, 0.02²)：BERT/GPT/HF 全量权重"""
    with torch.no_grad():
        w.normal_(0.0, std)
    return w

def scaled_init_normal(w, n_layer):
    """GPT-2 风格：残差分支末端投影 × 1/sqrt(2·n_layer)"""
    scale = 1.0 / math.sqrt(2 * n_layer)      # 每个 block 有 attention、FFN 两个残差分支
    with torch.no_grad():
        w.normal_(0.0, 0.02 * scale)
    return w

def deepnorm_init(w, kind, n_layer):
    """DeepNorm：FFN/v/out 用 gain=β 的 Xavier；q/k 用 gain=1"""
    beta = (8 * n_layer) ** -0.25             # β = (8N)^(-1/4)
    gain = beta if kind in ("ffn", "v_proj", "out_proj") else 1.0
    nn.init.xavier_normal_(w, gain=gain)
    return w

w = torch.empty(768, 768)
normal_002(w)
print(f"N(0,0.02): std={w.std().item():.4f}")                    # N(0,0.02): std=0.0200
w2 = torch.empty(768, 768)
scaled_init_normal(w2, n_layer=12)
print(f"GPT-2 12层残差分支: std={w2.std().item():.4f}")          # GPT-2 12层残差分支: std=0.0041
w3 = torch.empty(768, 768)
deepnorm_init(w3, "ffn", n_layer=24)
print(f"DeepNorm FFN (N=24): std={w3.std().item():.4f}")        # DeepNorm FFN (N=24): std=0.0097
# 验证: beta=(8·24)^(-1/4)=0.2686, xavier gain=beta → std=0.2686·sqrt(2/1536)=0.0097 ✓
```

### 3.2 nn.init 官方接口与 HF 实际代码

```python
# 官方接口：截断正态（BERT TF 版风格，PyTorch 2.3+ 内置）
w4 = torch.empty(768, 768)
nn.init.trunc_normal_(w4, std=0.02)          # 裁掉 ±2σ 外样本并重新归一化
print(f"trunc_normal_(0.02): std={w4.std().item():.4f}")        # trunc_normal_(0.02): std=0.0200

# HuggingFace GPT-2 的真实实现（源代码摘录）
# self.c_proj = nn.Linear(config.n_embd, config.n_embd)
# scale = 1.0 / math.sqrt(2 * config.n_layer)
# self.c_proj.weight.data.normal_(mean=0.0, std=config.initializer_range * scale)

# HuggingFace BERT 的真实实现（源代码摘录）
# for module in self.modules():
#     if isinstance(module, nn.Linear):
#         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         if module.bias is not None:
#             module.bias.data.zero_()
#     elif isinstance(module, nn.Embedding):
#         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
```

### 3.3 验证：残差累积 vs Scaled Init（可运行）

```python
torch.manual_seed(0)
d_model, n_block = 768, 12

def residual_stack(scale_out_proj, n_block=12, d_model=768, std=0.02):
    """L 个 block 的简化残差网络: x += out_proj(attn(x))，统计末层方差
       分支输入先 RMSNorm 到 std=1（对应真实 block 内 LN 的角色），
       否则分支方差随 x 膨胀，增长快于线性（见 2.2 注记）"""
    out_projs = [torch.randn(d_model, d_model) * std for _ in range(n_block)]
    if scale_out_proj:
        out_projs = [p / math.sqrt(2 * n_block) for p in out_projs]   # 1/√(2N)
    x = torch.randn(256, d_model)
    var0 = x.var()
    with torch.no_grad():
        for p in out_projs:
            xn = x / x.std(dim=1, keepdim=True)     # RMSNorm 式：分支输入归一化
            h = xn @ p
            x = x + h                                # 残差：x += F(x)
    return var0.item(), x.var().item()

v0, v1 = residual_stack(scale_out_proj=False)
print(f"不缩放: 首层 var={v0:.2f}, 12block 后 var={v1:.2f}, 比值={v1/v0:.1f}")
# 不缩放: 首层 var=1.00, 12block 后 var=4.69, 比值=4.7
# 理论: 分支方差=768·0.02²=0.307, 1+12·0.307=4.69 ✓ 线性增长被精确复现

v0, v1 = residual_stack(scale_out_proj=True)
print(f"1/√N 缩放: 首层 var={v0:.2f}, 12block 后 var={v1:.2f}, 比值={v1/v0:.2f}")
# 1/√N 缩放: 首层 var=1.00, 12block 后 var=1.15, 比值=1.15
# 理论: 分支方差=768·0.0041²=0.0128, 1+12·0.0128=1.15 ✓ 方差不再随深度增长
```

**结论**：不缩放时 12 层残差把方差放大 4.7 倍（与 $1 + L\sigma^2/\sigma^2 = 1+L$ 的线性累积理论吻合）；缩放 1/√(2L) 后仅 1.15 倍——这就是 GPT-2 scaled init 的全部作用。

## 四、深入分析

### 4.1 N(0, 0.02) 与 LayerNorm 的分工

- **LayerNorm 负责尺度**：LN 把每层激活归一化到固定 mean/std，因此权重大小**不直接影响激活尺度**——这让"经验小尺度"成为可能；
- **初始化负责"初始方向"与数值稳定**：虽然 LN 保尺度，但过大的初始权重仍会造成：① LN 反向梯度 $\partial LN(x)/\partial x = O(\sqrt{d}/\|x\|)$ 变小；② 残差分支方差大 → 累积方差大 → 深层 LN 输入极端；③ FP16 溢出。
- **分工结论**：有 LN 的网络"初始化错了也能训"（容错高），但"对的初始化训得更稳更快"。

### 4.2 为什么 BERT 用截断正态而 HF 用普通正态

两者分布几乎一样（0.02 下 ±2σ 外概率 < 5%，截断只是保证无极端样本）。截断正态是 TF 版 BERT 的遗产，普通正态是 PyTorch 生态的简化。**工程上不区分**，面试提到"BERT 用截断正态 N(0,0.02)"即可。

### 4.3 各代 Transformer 的初始化演进

| 模型 | 初始化 | 说明 |
|------|--------|------|
| Transformer (2017) | Xavier + sinusoid 位置编码 | 无 LN 兜底的原始方案 |
| BERT (2018) | 截断正态 N(0, 0.02) | 确立"小尺度经验标准" |
| GPT-2 (2019) | N(0, 0.02) + 残差分支 1/√N | 首次系统化残差缩放 |
| GPT-3 (2020) | N(0, 0.02) + 残差分支 1/√N | 沿用 GPT-2 方案 |
| T-Fixup (2020) | N^{-1/4} 缩放 + bias=0 | 解析推导，去 LN 去 warmup |
| DeepNet (2022) | DeepNorm α/β | Post-LN 训到 1000 层 |
| LLaMA (2023) | N(0, 0.02) + Pre-LN + 1/√(2L) 缩放（RMSNorm） | 现代标配 |

### 4.4 Embedding 初始化的特殊性

- **无 LN 保护**：embedding 是模型"第一站"，直接参与 loss（tied embedding 时同时是输出层）；
- **与位置编码相加**：$x = e + p$，两者尺度应一致，否则弱的一方被淹没；
- **tied embedding**：若输入/输出共享 embedding，尺度必须同时兼容"输入侧"与"softmax 前"的量级；
- 主流做法：`N(0, 0.02²)`（HF 默认），或小一点如 LLaMA 的 `0.02`/`0.03` 高斯。**可学习位置编码与 token embedding 同尺度**。

## 五、优缺点与适用

| 优点 | 缺点 |
|------|------|
| 有残差恒等起点，深层天然稳定 | 0.02 是经验值，非推导结果 |
| 与 LayerNorm 分工清晰，容错高 | 浅层/小模型略保守（收敛慢一点） |
| FP16/BF16 安全 | 直接套用可能过小（d 很小或网络很浅时） |
| 工程生态统一（HF 全系列默认） | 残差分支不缩放时方差随深度线性增长 |

**适用**：任何带 LayerNorm/RMSNorm 的 Transformer（BERT/GPT/LLaMA/CLIP text tower）、embedding、多模态模型的分塔主干。
**不适用**：无 LN 的深层网络（需 T-Fixup/Fixup 思路）、CNN+ReLU（用 Kaiming）、需要精确方差守恒的科研场景（用 Xavier/DeepNorm）。

## 六、与同类对比

| 维度 | Xavier | Kaiming | N(0,0.02)+Scaled | DeepNorm |
|------|--------|---------|------------------|----------|
| 尺度来源 | 推导（方差守恒） | 推导（ReLU 减半补偿） | 经验（恒等起点） | 推导（更新量有界） |
| 是否随维度缩放 | 是（1/n） | 是（1/n） | 否（常数 0.02） | 是（Xavier×β） |
| 残差处理 | 无 | 无 | 1/√N 缩放 | α/β 双重控制 |
| 依赖归一化 | 否 | 否 | 是（LN 兜底） | 是（LN 在残差外） |
| 极深网络 | 50 层极限 | 100 层极限 | ~100 层稳定 | 1000 层 |

## 七、高频面试问答

**Q1：为什么 Transformer 用 N(0, 0.02) 而不是 Xavier？**
① 残差结构要求初始输出≈恒等映射，残差分支必须小；② 避免 QKᵀ/√d 进入 softmax 饱和区；③ 避免 d_model 大时逐点数值偏大与 FP16 溢出；④ embedding 无 LN 保护且直接参与 loss；⑤ 深层残差方差按层数累积，初始小才能撑住深模型。Xavier 的 $\sqrt{2/d}$ 尺度在这些场景下偏大。

**Q2：GPT-2 的 scaled initialization 是什么？为什么是 1/√N？**
残差分支末端投影（out_proj）的初始化 std 乘以 1/√N（HF 实现 N=2·n_layer，每个 block 两个残差分支）。因为 L 个残差分支的方差线性累积为 Lσ²，把每个分支输出缩小 1/√L，总方差变为 σ²——**方差要缩 L 倍，所以 std 只缩 √L 倍**。

**Q3：DeepNorm 的 α 和 β 怎么取？**
encoder-only：α=(2N)^{1/4}、β=(8N)^{-1/4}；decoder-only：α=(2M)^{1/4}、β=(8M)^{-1/4}；encoder-decoder 按 N、M 组合查表。α 放大残差直连、β 缩小 FFN/v_proj/out_proj（q/k 不缩）。

**Q4：为什么 DeepNorm 只缩 v_proj/out_proj/FFN，不缩 q/k？**
q/k 只影响 attention 的加权方式（QKᵀ 有 1/√d_k 缩放且 softmax 归一化，输出幅度有上界，DeepNet Lemma 4.1）；v/out/FFN 直接决定信号的幅度且输出进入残差求和，必须缩小。

**Q5：T-Fixup 的核心思想？**
在 Adam（更新量级与梯度解耦）+ LN 的分析下，残差叠加使权重偏移量随深度增长，把残差分支权重乘 0.67·N^{-1/4} 后更新量有界，从而**无需 warmup、甚至无需 LayerNorm**。它是"残差分支缩放"思想的解析版，DeepNorm 是其后续发展。

**Q6：位置编码 / embedding 初始化要注意什么？**
与 token embedding 逐元素相加、尺度需一致；不能太大（淹没 token 语义、softmax 饱和、深层累积爆炸）；主流做法与全量权重同尺度（HF 默认 N(0,0.02)）。固定 sinusoid 或 RoPE 不涉及初始化。

**Q7：N(0,0.02) 有理论推导吗？**
没有，是 BERT 时期的经验选择，靠"LN 兜底 + 恒等起点 + 数值安全"三条理由支撑合理性。需要推导支持的场景用 DeepNorm（α/β 有解析公式）或 T-Fixup（N^{-1/4}）。

**Q8：训练 GPT 时 loss 在第一步就 NaN，首先查什么？**
按概率排序：① FP16 下 embedding/logits 溢出（调小 initializer_range 或加 grad clip）；② 残差分支未做 1/√N 缩放（深层方差爆炸）；③ 学习率过大（配合 warmup）；④ 数据问题（NaN label）。初始化自查方法见 [初始化总结](初始化总结.md)。

## 八、自我检验

- [ ] 能说出 N(0,0.02) 的来源（BERT/GPT/HF 经验标准）与至少 4 条合理性理由
- [ ] 能推导残差方差线性累积 $Var(x_L) \approx L\sigma^2$ 与 1/√N 缩放的数学
- [ ] 能写出 GPT-2 scaled init 的 HF 代码（scale = 1/sqrt(2·n_layer)）并解释 N=2L
- [ ] 能写出 DeepNorm 公式 $x_{l+1}=LN(\alpha x_l + G_l(x_l))$ 与 encoder/decoder 的 α、β
- [ ] 能解释 Q/K/V 与输出投影的初始化差异（DeepNet Lemma 4.1 直觉）
- [ ] 能说出 T-Fixup 的 N^{-1/4} 缩放与"去 warmup 去 LN"的结论
- [ ] 能解释 embedding/位置编码为什么与全量权重同尺度
- [ ] 能跑通 3.3 残差累积实验（不缩放 10 倍 vs 缩放 1.2 倍）
- [ ] 能回答 8 个面试追问
