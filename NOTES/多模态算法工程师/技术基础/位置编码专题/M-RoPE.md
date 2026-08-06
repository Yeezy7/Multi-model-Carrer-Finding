# M-RoPE：多模态三维旋转位置编码（Qwen2-VL）

> 本模块索引见 [位置编码专题详解](位置编码专题详解.md)

## 一、定义与公式

M-RoPE（Multimodal Rotary Position Embedding）是 Qwen2-VL 提出的 RoPE 三维推广：把 token 的位置从标量 $m$ 变成三元组 $(m_t, m_h, m_w)$，分别对应**时间（帧）/ 高度（行）/ 宽度（列）**，三个轴共享同一频率公式、按维度段独立旋转。

### 1.1 问题背景

视频/多图输入中，token 的位置天然是三维的：第几帧 $t$、帧内第几行 $h$、第几列 $w$。若只用 2D 空间位置（或 1D 序号）：

1. **时间混淆**：不同帧、同一空间位置的 patch 无法区分 → 注意力把跨帧内容当同一画面；
2. **运动无法建模**：帧间时序关系（物体的移动轨迹）信息丢失。

### 1.2 位置三元组

| 轴 | 符号 | 含义 | 示例（视频第 3 帧第 5 行第 7 列） |
|----|------|------|-----------------------------|
| 时间 | $m_t$ | 帧序号（或片段索引） | $m_t = 3$ |
| 高度 | $m_h$ | 帧内 patch 行号 | $m_h = 5$ |
| 宽度 | $m_w$ | 帧内 patch 列号 | $m_w = 7$ |

### 1.3 mrope_section：维度分组（核心公式）

head_dim 的一半（RoPE 作用的配对维度数）被**分成三段**，每段分配给一个轴：

$$\text{mrope\_section} = [16, 24, 24], \qquad 16 + 24 + 24 = 64 = \frac{d}{2} \quad (d = 128)$$

- 时间轴用频率下标 $i = 0..15$（16 个配对维度）；
- 高度轴用 $i = 16..39$（24 个）；
- 宽度轴用 $i = 40..63$（24 个）。

每个轴使用**同一个频率函数**（与 RoPE 相同），只是下标区间不同，再乘上自己的坐标：

$$\phi(\text{token}) = \big[\, m_t \cdot \theta_{0..15}, \;\; m_h \cdot \theta_{16..39}, \;\; m_w \cdot \theta_{40..63} \,\big], \qquad \theta_i = 10000^{-2i/d}$$

旋转方式与 RoPE 完全一致（每对相邻维度旋转 $\phi$ 弧度），只是不同维度的旋转角来自不同轴的坐标。

### 1.4 各模态的 position id 分配规则

| 模态 | 位置三元组 | 说明 |
|------|-----------|------|
| **文本** | $(i, i, i)$ 或 $(t, i, i)$ | 纯文本沿用一维 RoPE：时间/行/列取同一索引；与图像/视频并列时用当前时间 $t$ 对齐 |
| **图像** | $(\text{const}, h, w)$ | 时间轴取常数（如 0 或当前帧索引），高度/宽度用真实 patch 坐标 |
| **视频** | $(t, h, w)$ | 三轴都用真实坐标：帧号 $t$、帧内行 $h$、列 $w$ |

> **关键设计**：同一序列中文本、图像、视频 token 的 position id 是**混合坐标**——文本是 $(i,i,i)$ 标量型、图像是 $(t, h, w)$ 空间型，三者共存于一次前向。

## 二、核心原理与直觉

### 2.1 为什么分段旋转能区分"跨帧同位置"

同一空间位置 $(h,w)$、不同帧 $t_1 \ne t_2$ 的两个 patch：

- 时间段（$i=0..15$）旋转角不同：$t_1 \theta_i \ne t_2 \theta_i$；
- 空间段（$i=16..63$）旋转角相同：$h\theta_i, w\theta_i$ 不变。

因此两个 patch 的 q/k 向量在时间段维度上相位不同 → 内积中时间维度贡献被调制 → **注意力天然区分"跨帧同位置"与"同帧近邻"**。

### 2.2 为什么位置 id 小利于外推（面试必考）

RoPE 外推崩溃的根源是**旋转角超出训练区间**（相位 OOD，见 [RoPE](RoPE.md) 4.2）。M-RoPE 的坐标选择直接影响旋转角大小：

1. **文本 token 用 $(i,i,i)$**：序列上第 1000 个文本 token 的旋转角是 $1000\theta_i$——与 1D RoPE 相同，受训练长度限制；
2. **图像/视频 token 用真实坐标**：图像分辨率高时（如 56×56 网格），行坐标只到 55，旋转角最多 $55\theta_i \approx$ 很小——**图像 token 的旋转角天然远小于训练序列长度**；
3. **关键点**：长视频 = 帧数 × 帧内 patch 数。若用**扁平化一维索引**，视频第 100 帧的 patch 位置是 $100 \times 56 \times 56 \approx 313,600$，旋转角爆表 → 崩溃；用 $(t,h,w)$ 三维坐标后，**帧内坐标被"分解"回小值**，时间轴旋转角只依赖帧号 $t$（≤ 总帧数），空间轴只依赖 $h, w$（≤ 分辨率）——所有旋转角都控制在**与训练时相近的小量级**。

> **一句话**：M-RoPE 让"位置数值"不随 token 总量爆炸——视频多帧时每个轴的坐标仍然很小，从而让长视频外推成为可能。这也是 M-RoPE 论文宣称"大幅提升长视频理解能力"的机制基础。

### 2.3 与 2D 位置编码的对比直觉

| 方案 | 位置表示 | 帧间区分 | 外推 |
|------|---------|---------|------|
| 1D 序号（ViT 式） | 扁平索引 $hW+t$ | 无 | 差（索引爆炸） |
| 2D 分解（行列） | $(h, w)$ | 无（时间丢失） | 中 |
| **M-RoPE** | $(t, h, w)$ 分段旋转 | **有**（时间段） | **好**（坐标天然小） |

## 三、源码实现

### 3.1 手写实现：构造三维 position ids 并计算旋转

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MRopeConfig:
    """Qwen2-VL 默认配置"""
    def __init__(self):
        self.head_dim = 128
        self.mrope_section = torch.tensor([16, 24, 24], dtype=torch.int64)
        self.rotary_base = 10000.0

def precompute_mrope_freqs(cfg):
    """生成三段的旋转角度表：返回 e^{i*angle} 复数 [max_len, head_dim/2]"""
    d = cfg.head_dim
    theta = cfg.rotary_base ** (-torch.arange(0, d, 2).float() / d)   # [64]
    max_len = 8192
    m = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(m, theta)                    # 角度 [max_len, 64]
    return torch.polar(torch.ones_like(freqs), freqs)  # [max_len, 64] 复数

def build_3d_position_ids(text_lens, img_meta, video_meta):
    """构造混合模态的 3D position ids: [B, 3, L]
       text_lens: 各样本文本 token 数; img_meta: [(t, h, w), ...];
       video_meta: [(T, h, w), ...] 每帧的网格尺寸"""
    B = len(text_lens)
    t_ids, h_ids, w_ids = [], [], []
    for b in range(B):
        t, h, w = [], [], []
        offset = 0   # 文本时间轴计数
        # 1. 文本 token：位置 (i, i, i)，与前后视觉 token 时间对齐（offset 累加）
        for _ in range(text_lens[b]):
            t.append(offset); h.append(offset); w.append(offset); offset += 1
        # 2. 图像 token：位置 (当前帧 t, 行, 列)
        for (im_t, im_h, im_w) in img_meta.get(b, []):
            for r in range(im_h):
                for c in range(im_w):
                    t.append(im_t); h.append(r); w.append(c)
        # 3. 视频 token：位置 (帧号, 行, 列)
        for (vt, vh, vw) in video_meta.get(b, []):
            for tt in range(vt):
                for r in range(vh):
                    for c in range(vw):
                        t.append(tt); h.append(r); w.append(c)
        t_ids.append(t); h_ids.append(h); w_ids.append(w)
    # 组装 [B, 3, L]（逐 batch 逐轴填充，不足长度补 0）
    seqs = [[t_ids[b], h_ids[b], w_ids[b]] for b in range(B)]
    L = max(len(ax) for b in range(B) for ax in seqs[b])
    ids = torch.zeros(B, 3, L, dtype=torch.int64)
    for b in range(B):
        for a in range(3):
            ids[b, a, : len(seqs[b][a])] = torch.tensor(seqs[b][a])
    return ids

def apply_mrope(q, k, freqs_cis, position_ids, mrope_section):
    """按 mrope_section 分段旋转: q/k [B, L, H, D]；position_ids [B, 3, L]"""
    B, L, H, D = q.shape
    sec = [0] + list(torch.cumsum(mrope_section, dim=0))   # [0, 16, 40, 64]
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    for axis in range(3):                       # 逐轴处理
        s, e = sec[axis], sec[axis + 1]
        ids = position_ids[:, axis]             # [B, L] 该轴坐标
        # 本段频率列: 第 s//2..e//2-1 对维度（各轴用频率表的独立下标区间）
        fr = freqs_cis[ids, s // 2 : e // 2].unsqueeze(2)   # [B, L, 1, np]
        # 取出本段维度，按相邻配对旋转
        qc = torch.view_as_complex(
            q[..., s:e].reshape(B, L, H, -1, 2).to(torch.float32))
        kc = torch.view_as_complex(
            k[..., s:e].reshape(B, L, H, -1, 2).to(torch.float32))
        q_rot[..., s:e] = torch.view_as_real(qc * fr).flatten(-2)
        k_rot[..., s:e] = torch.view_as_real(kc * fr).flatten(-2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)

# 测试：视频（2 帧 2x2）+ 图像（1 张 2x2）+ 文本（3 token）
cfg = MRopeConfig()
freqs = precompute_mrope_freqs(cfg)
pos = build_3d_position_ids(text_lens=[3],
                            img_meta={0: [(0, 2, 2)]},
                            video_meta={0: [(2, 2, 2)]})
print(pos.shape)   # torch.Size([1, 3, 15])：3 文本 + 4 图像 + 8 视频 patch
print(pos[0, 0])   # 时间轴: [0,1,2, 0,0,0,0, 0,0,0,0, 1,1,1,1]（文本/图像/视频帧0/帧1）
print(pos[0, 2])   # 宽度轴: [0,1,2, 0,1,0,1, 0,1,0,1,0,1,0,1]（列坐标逐 patch 递增）

q = torch.randn(1, 15, 2, 128)
k = torch.randn(1, 15, 2, 128)
q2, k2 = apply_mrope(q, k, freqs, pos, cfg.mrope_section)
print(q2.shape)    # torch.Size([1, 15, 2, 128])
```

### 3.2 与官方实现对比验证

参考实现：transformers `Qwen2VLForConditionalGeneration` 的 `apply_multimodal_rotary_pos_emb`（建模文件 modeling_qwen2_vl.py）。官方实现要点：① 频率表按 max_pos 预计算并**重复拼接**（cat 一次等于 4 个位置）；② 用 `mrope_section` 的 cumsum 分段；③ 时间轴对文本按"相对时间"递增、对图像/视频取帧索引。内嵌官方核心逻辑验证分段一致性：

```python
def rotate_half(x):
    """官方 rotate_half（对偶布局）"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# 等价性验证：官方"cos/sin 分段逐元素 + rotate_half"与 3.1 "复数乘法"一致
# （分段区间、坐标索引完全一致，仅段内配对布局不同）
torch.manual_seed(0)
D, L, H = 128, 8, 2
q = torch.randn(1, L, H, D)
theta = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
m = torch.arange(L, dtype=torch.float32)
freqs_c = torch.polar(torch.ones_like(torch.outer(m, theta)),
                      torch.outer(m, theta))          # [L, 64] 复数
axes = torch.arange(L).unsqueeze(0).unsqueeze(0).expand(1, 3, L)  # 纯文本 (i,i,i)
sec = [0, 16, 40, 64]

outA = torch.empty_like(q)
outB = torch.empty_like(q)
for a in range(3):
    s, e = sec[a], sec[a + 1]
    ids = axes[:, a]                                  # [1, L]
    np_ = (e - s) // 2                                # 本段配对维度数
    fr = freqs_c[ids, s // 2 : e // 2].unsqueeze(2)   # [1,L,1,np] 本段频率
    qseg = q[..., s:e]                                # [1,L,H,2np] 交错布局
    # 写法 A：复数（相邻配对，3.1 实现），输入/输出都交错
    qc = torch.view_as_complex(qseg.reshape(1, L, H, -1, 2).to(torch.float32))
    outA[..., s:e] = torch.view_as_real(qc * fr).flatten(-2)
    # 写法 B：官方 rotate_half（cos/sin 复制成整段），输入/输出半区对偶
    qseg_b = qseg.reshape(1, L, H, -1, 2).transpose(-1, -2).reshape(1, L, H, e - s)
    cos = torch.cat((fr.real, fr.real), dim=-1)       # [1,L,1,2np]
    sin = torch.cat((fr.imag, fr.imag), dim=-1)
    b = qseg_b * cos + rotate_half(qseg_b) * sin
    # 把 B 的输出从"先所有第一分量、再第二分量"重排回交错后与 A 比较
    outB[..., s:e] = b.reshape(1, L, H, 2, -1).transpose(-1, -2).reshape(1, L, H, e - s)

print(torch.allclose(outA, outB, atol=1e-5))   # True（两种布局数学等价）
```

> 说明：两种写法的区别仅是**配对布局**——复数写法按"相邻两维配对"，官方写法按"前半/后半对偶"；分段区间、坐标索引完全一致，数学等价。生产环境以官方 `modeling_qwen2_vl.py` 为准。

## 四、性质分析

### 4.1 相对位置：三轴各自成立

对同一轴内两个 token：时间维度内积 $\langle R_{t_1}q, R_{t_2}k\rangle$ 只依赖 $t_1 - t_2$（沿用 RoPE 正交性推导），空间轴同理。跨模态 token（文本 vs 图像）之间，位置差由各轴坐标差共同决定。

### 4.2 外推性：坐标分解是关键

| 场景 | 1D 扁平索引 | M-RoPE 三元组 |
|------|------------|---------------|
| 100 帧视频（56×56） | 最大索引 ~313K（崩溃） | $t \le 100$，$h,w \le 56$（小量级） |
| 长文本 32K | $i \le 32K$（同 RoPE） | 时间轴 $i \le 32K$（同 RoPE，需插值） |

**结论**：M-RoPE 的坐标分解使"空间/时间坐标小"成为天然优势——视觉 token 的外推上限由**单帧分辨率**和**帧数**决定而非 token 总量；文本部分仍与 1D RoPE 相同，超长文本仍需插值技术配合。

### 4.3 参数

- 位置参数 **0**（与 RoPE 相同，频率函数固定）；
- mrope_section 是**模型超参**（Qwen2-VL 用 [16,24,24]），可随 head_dim 调整（保持三段和为 $d/2$）。

### 4.4 文本与视觉的对齐

文本 token 的时间轴取**当前时间索引**（累计递增），使其与图像/视频的帧索引在同一时间线上——注意力才能正确建立"这段文本描述第几帧"的关系。这是 M-RoPE 能跨模态对齐的关键工程细节。

## 五、优缺点

| 优点 | 缺点 |
|------|------|
| 天然支持视频/多图/文本混合序列（无分块） | 位置 id 构造复杂（需按模态分支生成） |
| 任意分辨率、任意帧数零插值 | mrope_section 需人工设计（分段比例影响性能） |
| 长视频 token 坐标小 → 外推友好 | 文本超长仍需 RoPE 插值技术 |
| 时间轴独立维度 → 显式建模时序运动 | 实现复杂度高于 2D 分解/ALiBi |
| 与 FlashAttention 融合（Qwen2-VL 官方 kernel） | 需模型训练时原生支持（无法后装） |
| 跨模态时间对齐（文本-视频同时间线） | |

## 六、与同类对比

| 维度 | M-RoPE | RoPE | 2D 分解 | ALiBi |
|------|--------|------|---------|-------|
| 位置维度 | 3D $(t,h,w)$ | 1D $m$ | 2D $(r,c)$ | 距离 $\|i-j\|$ |
| 视觉支持 | 视频+图像+文本 | 需人工合并成 1D | 图像（无时间） | 无专门设计 |
| 帧间区分 | **有**（时间段） | 无 | 无 | 无 |
| 外推 | 好（坐标分解） | 差（需插值） | 中 | 好 |
| 参数 | 0 | 0 | $(H+W)d$ | 0 |
| 代表模型 | Qwen2-VL / Qwen3-VL | LLaMA、Qwen | 2D-ViT | BLOOM、MPT |

**关键对比结论**：
1. M-RoPE vs RoPE：M-RoPE 是 RoPE 的多模态泛化（同频率、同旋转，只是坐标多维、维度分段）；纯文本时退化为 RoPE；
2. M-RoPE vs 2D 分解：2D 分解只解决图像分辨率，不解决帧间时间混淆；M-RoPE 三段分别编码时间/行/列；
3. M-RoPE vs ALiBi：ALiBi 零成本外推但无法表达结构坐标；M-RoPE 用坐标分解天然控制旋转角量级，长视频场景更优。

## 七、高频面试问答

**Q1：M-RoPE 解决什么问题？**
视频/多图中 token 位置是三维的（帧、行、列）。1D/2D 编码无法区分"跨帧同位置"（时间混淆）且无法建模时序运动。M-RoPE 用 $(t,h,w)$ 三元组 + 维度分段旋转解决。

**Q2：mrope_section 是什么？为什么是 [16,24,24]？**
head_dim 的一半（64 对）按 [16,24,24] 分成时间/高度/宽度三段，各段用同频率函数、各自坐标旋转。和为 64 = d/2；比例可调（时间 16 最低频承载长程运动，空间 48 承担分辨率）。

**Q3：为什么位置 id 小利于外推？**
RoPE 崩溃源于旋转角超训练区间。M-RoPE 用真实坐标：图像 token 的旋转角只依赖 $h,w$（≤分辨率）、视频只依赖 $t$（≤帧数），不会随 token 总量爆炸（对比扁平索引 313K）。视觉 token 旋转角天然小 → 长视频外推。

**Q4：文本、图像、视频的 position id 分别怎么给？**
文本 $(i,i,i)$（或对齐到当前时间）；图像 $(\text{const}, h, w)$；视频 $(t, h, w)$。文本时间轴累计递增实现跨模态时间对齐。

**Q5：M-RoPE 和 2D 分解位置编码的区别？**
2D 分解（$P_{row}[r]+P_{col}[c]$）是加性、可学习、只有空间；M-RoPE 是旋转、零参数、带时间轴。视频场景 2D 分解完全无法区分帧。

**Q6：纯文本任务 M-RoPE 退化成什么？**
退化为标准 RoPE：三个轴取相同索引 $(i,i,i)$，三段旋转合并后等价于对全部维度按 $i\theta$ 旋转（频率函数相同，只是下标区间不同）。

**Q7：M-RoPE 能处理超长文本吗？**
视觉部分天然外推好；文本部分仍受 RoPE 限制，需配合长度外推技术（NTK/YaRN）——Qwen2-VL 即用"long context 微调 + 插值"扩展文本长度。

**Q8：为什么时间轴分配 16 个维度而非更多？**
低维度=低频为主（$i$ 小，波长长），时间轴需要承载长程运动（跨帧关系）而非局部精度；空间轴 48 维承担高分辨率细节。这是 Qwen2-VL 验证过的平衡。

## 八、自我检验

- [ ] 能写出 M-RoPE 的位置三元组 $(m_t, m_h, m_w)$ 与 mrope_section 公式
- [ ] 能说明 [16,24,24] 的构成与 $\sum = d/2$ 约束
- [ ] 能给出文本/图像/视频三种 position id 分配规则
- [ ] 能解释"坐标分解 → 旋转角小 → 外推友好"的完整链条
- [ ] 能手写 build_3d_position_ids 与 apply_mrope（分段旋转）
- [ ] 能验证官方写法（cos/sin + rotate_half 分段）与复数写法的等价性
- [ ] 能对比 M-RoPE vs RoPE / 2D 分解 / ALiBi
- [ ] 能回答 8 个面试追问
