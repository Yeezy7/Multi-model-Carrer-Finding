# ALiBi：注意力线性偏置（Attention with Linear Biases）

> 本模块索引见 [位置编码专题详解](位置编码专题详解.md)

## 一、定义与公式

ALiBi（Press et al., 2021）**完全不做位置 embedding**，只在 attention 分数上加一个线性的距离惩罚。零参数、实现极简、外推性好，被 BLOOM、GPT-NeoX、MPT 等采用。

### 1.1 基本公式

$$\text{score}(i, j) = q_i^\top k_j \; - \; m \cdot |i - j|$$

- $i$：query 位置；$j$：key 位置；
- $m$：**每头不同的固定斜率**（不训练，零参数）；
- 因果注意力中 $i \ge j$，等价于 $\text{logits} = qk^\top/\sqrt{d} - m \cdot (i - j)$。

### 1.2 斜率设定（几何序列）

常用两种设定（论文用 $2^{-8h/H}$，MPT/BLOOM 用 $2^{-h}$）：

$$m_h = 2^{-8h/H}, \qquad h = 1, 2, \dots, H$$

- 第 1 头：$m_1 = 2^{-8/H}$（最陡，近因偏置最强）；
- 第 $H$ 头：$m_H = 2^{-8}$（最平缓，几乎不惩罚）。

**几何序列的意义**：头与头之间斜率成倍递减，覆盖"强近因偏置"到"几乎无偏置"的整个谱系——不同头负责不同距离尺度（与 RoPE/正弦的"多分辨率"思想一致）。

### 1.3 与 T5 相对偏置的联系

两者都把偏置加在 logits 上（$\text{score} = qk^\top + b_{i-j}$）。区别：

| | ALiBi | T5 relative bias |
|---|-------|-----------------|
| 偏置函数 | $b = -m|i-j|$，**固定线性** | $b$ 是**可学习查表**（桶表） |
| 参数 | 0 | num_buckets 个标量 |
| 外推 | 直线无限延伸 | 桶封闭（超 max_distance 落末桶） |

## 二、核心原理与直觉

### 2.1 近因偏置（Recency Bias）

语言模型的注意力分布强烈偏向**附近的 token**（局部语法、指代消解），且随距离大致**线性衰减**。ALiBi 把这个先验直接"焊死"进模型结构：距离越远，logits 被惩罚得越多，softmax 后权重越小。

### 2.2 为什么零参数还能外推（面试必考）

对比三类编码在"超训练长度"时的行为：

| 编码 | 训练期见过什么 | 超长时 |
|------|---------------|--------|
| 可学习表 | 位置索引 $0..L_{train}$ | 索引越界 → 崩溃 |
| RoPE | 相位组合 $(m\theta_0, \dots)$ | 全新相位向量 → OOD |
| **ALiBi** | **距离值 $0..L_{train}$ 全覆盖** | **同一根惩罚直线，延长即可** |

**三条理由**：

1. **距离值域训练期全覆盖**：训练中所有可能出现的距离 $|i-j| \in [0, L_{train})$ 都被模型见过，外推只是把同一条直线延到更大的距离——**没有引入任何新的输入分布**；
2. **零参数**：模型从未"记忆"绝对位置，不存在"位置长尾"或"位置 OOD"的根源；位置信息完全由固定的距离函数提供；
3. **符合语言先验**：线性惩罚天然聚焦近处，与真实注意力模式一致，模型无需再学"位置有什么用"。

### 2.3 实现上的本质

ALiBi 不改变输入 embedding、不旋转 q/k，只在 attention logits 上加一个**预计算的偏置矩阵** $M$：

$$M_{i,j} = -m_h \cdot (i-j) \quad (i \ge j, \text{因果})$$

前向一次查表 + 加法，比任何位置编码都便宜（$O(L^2)$ 的掩码矩阵加法，但只做一次且与批量无关）。

## 三、源码实现

### 3.1 纯 PyTorch 手写实现

```python
import torch
import torch.nn as nn
import math

def build_alibi_slopes(num_heads, interpolate=False):
    """生成每头的斜率 m_h（几何序列，零参数）"""
    if interpolate:                      # MPT 式：2^{-h}
        return 2 ** (-torch.arange(1, num_heads + 1).float())
    # 论文式：2^{-8h/H}，h = 1..H
    return 2 ** (-8 * torch.arange(1, num_heads + 1).float() / num_heads)

def build_alibi_mask(num_heads, max_len, slopes=None):
    """预计算 ALiBi 偏置矩阵: [H, L, L]（因果：只惩罚 i > j 的项）"""
    H = num_heads
    slopes = slopes if slopes is not None else build_alibi_slopes(H)
    m = torch.arange(max_len)
    # 相对距离矩阵: [L, L]，第 (i,j) 元素 = i - j（query i 到 key j 的距离）
    rel = m.unsqueeze(1) - m.unsqueeze(0)              # [L, L]
    rel = rel.clamp(min=0)                             # 因果：j > i 时取 0（不惩罚）
    alibi = -slopes.reshape(H, 1, 1) * rel.unsqueeze(0)  # [H, L, L]
    return alibi

class ALiBiAttention(nn.Module):
    """带 ALiBi 的多头注意力：qk^T/sqrt(d) + ALiBi 偏置"""

    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.register_buffer("alibi", build_alibi_mask(num_heads, max_len))

    def forward(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.alibi[:, :L, :L]        # 加上线性偏置
        attn = torch.softmax(scores, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)

# 测试
torch.manual_seed(0)
attn = ALiBiAttention(d_model=8, num_heads=4, max_len=10)
out = attn(torch.randn(2, 10, 8))
print(out.shape)                                       # torch.Size([2, 10, 8])
print(attn.alibi.shape)                                # torch.Size([4, 10, 10])
print(attn.alibi[0, 5, :4])  # 头 0（斜率 2^{-2}=0.25），query 位置 5 对前 4 个 key 的惩罚
# tensor([-1.25, -1.00, -0.75, -0.50])（距离 5/4/3/2，每远一格多罚 0.25）
```

### 3.2 与官方实现对比验证

参考实现：transformers `LlamaModel` 之外的 MPT/BLOOM 实现，以及 HF 社区 `build_alibi_tensor`。核心就一个公式：`alibi = -slopes[:, None, None] * (i - j)`。这里内嵌官方等价写法（MPT 风格）直接对比：

```python
def build_alibi_tensor_mp(attention_mask, num_heads, dtype):
    """MPT 官方实现思路（transformers/models/mpt/modeling_mpt.py）：
       相对距离用 (arange - arange.T) 一次生成，再乘斜率"""
    L = attention_mask.shape[-1]
    slopes = torch.pow(2, -torch.arange(1, num_heads + 1).float())   # 2^{-h}
    # 官方按行构造：distance(i,j) = i - j（因果），clamp 到 >= 0
    row = torch.arange(L, device=attention_mask.device, dtype=dtype)
    distances = row.view(1, 1, L, 1) - row.view(1, 1, 1, L)   # [1,1,L,L]
    distances = distances.clamp(min=0)
    alibi = -slopes.view(1, num_heads, 1, 1) * distances       # [1,H,L,L]
    return alibi

# 对比验证：手写版 vs MPT 版（同一斜率公式 2^{-h}）
torch.manual_seed(0)
H, L = 4, 6
mine = build_alibi_mask(H, L, slopes=build_alibi_slopes(H, interpolate=True))
mp = build_alibi_tensor_mp(torch.ones(1, 1, L), H, torch.float32)
print(torch.allclose(mine, mp.squeeze(0), atol=1e-6))   # True
```

### 3.3 可视化：偏置矩阵与注意力衰减

```python
# 无 matplotlib 依赖：直接打印 8 头的斜率与惩罚曲线
alibi = build_alibi_mask(8, 32)
print("各头斜率 m_h (2^{-8h/8}):", build_alibi_slopes(8).numpy().round(4))
# 各头斜率: [0.5   0.25  0.125 0.0625 0.0312 0.0156 0.0078 0.0039]
pen0 = alibi[0, 16].numpy()    # 头 0（最陡），query 位置 16 对所有 key 的惩罚
pen7 = alibi[7, 16].numpy()    # 头 7（最平缓）
print("head0 惩罚(前 6 个 key):", pen0[:6].round(3))
# head0 惩罚(前 6 个 key): [-8. -7.5 -7. -6.5 -6. -5.5]（每远一格多罚 0.5）
print("head7 惩罚(前 6 个 key):", pen7[:6].round(3))
# head7 惩罚(前 6 个 key): [-0.062 -0.059 -0.055 -0.051 -0.047 -0.043]（几乎平坦）
# 结论: 头 0 强近因偏置（远 key 几乎不可见），头 7 几乎无偏置（可看全局）
```

## 四、性质分析

### 4.1 外推性：好（机制性保证）

- 训练长度 $L_{train}$ 内，所有距离值 $|i-j|$ 均出现过 → 外推不引入新输入分布；
- 实测：BLOOM 从 2048 直接外推到 4096（甚至 8192）无崩溃，PPL 平滑；
- **代价**：注意力受距离惩罚，长文本上"远距信息读取"能力比 RoPE 弱——这正是后续模型更倾向 RoPE + 插值的原因之一。

### 4.2 相对位置：严格成立（线性形式）

偏置只依赖 $|i-j|$，是**最朴素、最直接**的相对位置注入——不需要任何推导，距离直接进入 logits。

### 4.3 参数

- **0 个参数**：斜率是固定几何序列，不可训练；
- 额外计算：一个 $[H, L, L]$ 掩码矩阵的加法（可预计算、可并入 causal mask，几乎免费）；
- 与注意力计算完全正交：不改 embedding、不改 q/k/v。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 零参数、实现最简单（一行偏置加法） | 线性惩罚是"硬衰减"，长程信息读取能力受限 |
| **外推性最好**（距离值域全覆盖，无新分布） | 无法表达更细的位置结构（如方向、绝对位置） |
| 训练更快收敛（结构先验强） | 无旋转变换，与线性注意力/部分 kernel 的融合不如 RoPE 成熟 |
| 头间几何序列斜率 = 多尺度距离感受野 | 斜率需人工设定（几何序列超参） |
| 与因果掩码天然合并 | |

## 六、与同类对比

### 6.1 ALiBi vs RoPE（面试最常问）

| 维度 | ALiBi | RoPE |
|------|-------|------|
| 机制 | 减性：$\text{score} - m\|i-j\|$ | 乘性：q/k 旋转，$\cos((m-n)\theta)$ 调制 |
| 相对位置 | 线性、直接 | 多频余弦、严格距离函数 |
| 外推 | **好**（无需任何技术） | 差（需 PI/NTK/YaRN） |
| 长程读取 | 惩罚线性增长 → 弱 | 低频段保留 → 强 |
| 参数/成本 | 0 / 掩码加法 | 0 / q,k 旋转 |
| 代表模型 | BLOOM、MPT、GPT-NeoX | LLaMA、Qwen、Mistral |

**面试观点**：ALiBi 赢在外推省事，RoPE 赢在长程能力与生态（FlashAttention 融合、M-RoPE 多模态扩展）。现代主流（Qwen/LLaMA 系）选 RoPE+插值；ALiBi 仍是"零成本外推"的最优解。

### 6.2 ALiBi vs 可学习/T5 偏置

- 可学习偏置（T5 桶表）：参数封闭、桶映射可外推，但需训练学习偏置函数；
- ALiBi：偏置函数是**固定的**，训练前就已确定，省掉学习成本与参数，但失去了数据驱动的灵活性（如学出"中等距离有偏好"的非单调结构）。

## 七、高频面试问答

**Q1：ALiBi 为什么零参数还能外推？**
① 惩罚只依赖距离，而距离值域 $[0, L_{train})$ 训练期全覆盖，外推只是延长同一条直线，无新分布；② 零参数 → 模型从不记忆绝对位置，不存在位置 OOD；③ 近因偏置符合语言先验，结构与任务天然匹配。

**Q2：ALiBi 的斜率怎么设定？为什么是几何序列？**
$m_h = 2^{-8h/H}$（或 $2^{-h}$），几何序列让各头覆盖从"强近因偏置"到"几乎无偏置"的谱系，不同头关注不同距离尺度——与 RoPE 的多频率维度思想一致。

**Q3：ALiBi 相比 RoPE 的优缺点？**
优：外推直接（无需插值技术）、实现最简单；缺：线性惩罚使长程信息读取变弱（远距 key 被罚死），且无法表达旋转类结构（无法直接扩展 M-RoPE 多模态）。

**Q4：ALiBi 与 T5 relative bias 的区别？**
都是 logits 加偏置；ALiBi 偏置是固定线性函数（零参数），T5 是可学习桶表（参数封闭，远距落末桶）。ALiBi 更强先验、更省参数，T5 更灵活。

**Q5：为什么 ALiBi 不旋转 q/k 也能表达相对位置？**
相对位置只需要"距离信息"——ALiBi 直接把距离以惩罚形式加进 logits，信息注入路径最短。旋转（RoPE）是另一种注入路径（乘性调制），两者表达同一先验的不同实现。

**Q6：BLOOM 怎么用 ALiBi 外推的？**
BLOOM 训练 2048，推理直接用 4096+：掩码矩阵按新长度重新生成（斜率不变），注意力天然接受更大距离值，PPL 平滑无尖峰。对比 RoPE 模型同场景会 PPL 暴涨。

**Q7：ALiBi 适合什么场景？**
中小模型快速上线、需要"零成本外推"的场景（如长文档理解、多轮对话）；以及实现简单性优先的部署（无需 RoPE kernel 支持）。长程密集检索场景（RAG）建议 RoPE+插值。

## 八、自我检验

- [ ] 能写出 $\text{score} = qk^\top - m\|i-j\|$ 与斜率公式 $m_h = 2^{-8h/H}$
- [ ] 能完整回答"零参数为什么能外推"（三点论）
- [ ] 能说明"距离值域训练期全覆盖"与 RoPE"相位 OOD"的本质区别
- [ ] 能手写 build_alibi_mask（因果 clamp + 每头斜率广播）
- [ ] 能对比 MPT 斜率（$2^{-h}$）与论文斜率（$2^{-8h/H}$）及实现等价性
- [ ] 能对比 ALiBi vs RoPE vs T5 bias 的机制/外推/长程能力
- [ ] 能回答 7 个面试追问
