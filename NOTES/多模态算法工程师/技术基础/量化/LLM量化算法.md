---
learned: false
---

# LLM 量化算法：GPTQ、AWQ、NF4

> 本模块索引见 [量化详解](量化详解.md)

## 一、LLM 量化的特殊性

LLM 与 CNN/传统模型相比，量化有几个独特之处，决定了算法的设计方向：

### 1.1 激活分布含显著离群点（outlier channels）

LLM 的**激活**（尤其注意力后的 hidden states）不是正态分布：少数通道（channel）的激活值比其他通道大出**几十到上百倍**，且这些离群通道在不同 token 间**稳定出现**（叫 systematic outliers）。原因：注意力/FFN 里部分隐藏单元形成"专精"通路，输出恒大。

- 后果：只要存在一个离群通道，per-tensor 量化的 scale 就被它拉大，其余所有通道的相对误差急剧增大；
- 对策：**per-channel / per-group 量化**、**激活感知的权重保护**（AWQ 的思想）、**SmoothQuant**（把离群从激活转移到权重再量化）。

```python
# 离群点对 per-tensor 量化的破坏（示意）
import torch
torch.manual_seed(0)
x = torch.randn(4096) * 0.5
x[::100] *= 20                      # 1% 离群点
qmax = 127
scale = x.abs().max() / qmax        # 被离群点撑大
err = (x - torch.round(x / scale).clamp(-qmax, qmax) * scale).abs().mean()
print(f"含离群点 INT8 平均绝对误差: {err.item():.4f}")   # 0.0511
```

### 1.2 权重分布集中但分层差异大

- 权重近似**零中心正态分布**，但**不同层**的方差差异很大（embedding/head 的权重幅度与中间层不同）——所以好的算法按层、按通道分别定标；
- 权重**冗余度高**：LLM 参数量大、有大量近似线性相关的行/列，删掉/扰动一部分权重影响有限——这是"误差补偿量化"（GPTQ）可行的前提。

### 1.3 INT4 是主流目标

- INT8 权重损失无感（7B 掉 <0.1 perplexity），但只省一半显存；
- 要让 7B 单卡跑（~4GB）、70B 单机多卡跑，必须 **INT4**（≈4.15 bits/param，含 scale 开销）；
- 因此 2023 年以来的主流算法（GPTQ/AWQ/NF4）几乎都围绕 4-bit 权重展开。

## 二、GPTQ：基于误差补偿的逐层量化（重点）

**核心思想**：不孤立地量化每个权重，而是"量化一个权重后，用数学推导出的最优量去**补偿**其余权重，把误差吸收掉"。一层一层做，不需要训练，**one-shot**。

### 2.1 目标函数与闭式解

对某线性层，输入激活为 $X \in \mathbb{R}^{n\times d}$（$n$ 个输入通道、$d$ 个样本），权重为 $W \in \mathbb{R}^{k\times n}$。目标是量化 $W$ 成 $\hat{W}$ 时，**尽量保持层输出不变**：

$$\min_{\hat{W}} \; \left\| W X - \hat{W} X \right\|_F^2$$

展开损失 $L = \|WX - \hat{W}X\|^2$ 对 $W$ 的二阶近似，其 **Hessian 矩阵**为：

$$H = 2\, X X^{\top} + \lambda I$$

（$\lambda I$ 是阻尼项，保证可逆；$H \in \mathbb{R}^{n\times n}$，由校准数据 $X$ 决定。）

**为什么量化误差要用 Hessian 加权**：损失对第 $i$ 个权重列的敏感度不是均匀的——某个列若与校准激活的相关性高（$H_{ii}$ 大），扰动它损失升得多。因此**误差补偿方向要沿着 $H^{-1}$**（把误差往"对输出影响小"的方向推）。

### 2.2 逐列量化 + 误差补偿（OBQ/GPTQ 的核心循环）

对第 $i$ 列权重 $w_i \in \mathbb{R}^k$，量化得 $\hat{w}_i$，产生误差 $\delta_i = \hat{w}_i - w_i$（列向量）。OBQ 证明了：**在只量化第 $i$ 列的前提下，把其余列调整成**

$$W_{:,j} \;\leftarrow\; W_{:,j} - \frac{\delta_i}{\left[H^{-1}\right]_{ii}}\, \left[H^{-1}\right]_{i,j}, \qquad j \ne i$$

就能在保持 $H$ 约束下**最优地吸收**这次量化误差。直观理解：$H^{-1}$ 的每一列表示"该权重方向对损失的敏感度"，把误差 $\delta_i$ 按敏感度比例分摊给其余列，整体输出误差最小。

**算法循环**（简版）：

```text
H = 2·X·Xᵀ + λI ;  Hinv = H⁻¹
for i = 1..n:                    # 逐列处理
    w_i  = W[:, i]               # 当前要量化的列
    δ_i  = quant(w_i) - w_i      # 量化误差
    W[:, i] = quant(w_i)
    d = Hinv[i, :] / Hinv[i, i]  # 补偿权重（Hinv 的第 i 行按对角归一）
    W     -= outer(δ_i, d)       # 补偿其余列
    # 同时用秩 1 更新 Hinv（避免重算逆矩阵）→ GPTQ 的关键加速
```

> **GPTQ 相对 OBQ 的三个工程改进**：① **lazy batch**——批量处理若干列再一次性更新，避免逐列反复更新 Hinv；② **量化顺序**——按激活幅度降序（`--act-order`）先量化重要列，7B 上 PPL 从 7.15 降到 6.09；③ **group 量化（g128）**——每 128 列共享 scale，省 scale 存储，g128 时约 4.15 bits/param。

### 2.3 实测效果（LLaMA，WikiText-2 PPL）

| 模型 | FP16 | INT4 RTN（朴素） | INT4 GPTQ | GPTQ g128 |
| --- | --- | --- | --- | --- |
| LLaMA-7B | 5.68 | 6.29 | 6.09 | 5.85 |
| LLaMA-13B | 5.09 | 5.53 | 5.36 | — |
| LLaMA-30B | 4.10 | 4.54 | 4.45 | — |
| LLaMA-65B | 3.53 | 3.92 | 3.84 | — |

要点：**模型越大，量化损失越小**（65B 只掉 0.31，7B 掉 0.41）；**GPTQ 明显优于朴素 RTN**（误差补偿起效）。

## 三、AWQ：激活感知量化（重点）

**核心思想**：**权重不是平等的**。某些输入通道对应的激活很大，这些通道的权重对输出误差贡献最大，是"重要通道"。AWQ 用**激活分布**识别它们，再用 **per-channel 缩放**在量化前"保护"它们——完全不需要训练，也不需要 Hessian。

### 3.1 方法

1. 从校准数据统计每通道激活均值 $m_i = \mathrm{E}|x_i|$；
2. 计算缩放因子 $s_i = m_i^{\alpha}$，$\alpha \in [0,1]$（实验最优 $\alpha \approx 0.5$）；
3. 量化前把权重按通道缩放 $W' = W \cdot s$，量化后除以 $s$ 还原（数学等价）：

$$\hat{W} = \frac{\text{quant}(W \cdot s)}{s} \quad\Rightarrow\quad \hat{W}X = \text{quant}(W \cdot s) \cdot \frac{X}{s}$$

（推理时 $X/s$ 被**吸收进上一层权重**，不额外开销。）

### 3.2 为什么缩放有效

per-tensor 量化的步长由 $\max |W'|$ 决定。缩放 $s_i$ 让：
- **重要通道（激活大）** 的权重被放大 → 在共享步长下相对误差变小（"优先保护"）；
- **非重要、但权重大的干扰通道** 被缩小 → 压低全局步长，惠及所有通道。

代价分析：$\alpha$ 太小（不缩放）保护不足；$\alpha$ 太大（过度放大重要通道）把全局步长撑大，反而伤及其他通道——所以取中间值 $\alpha=0.5$。

### 3.3 AWQ vs GPTQ 对比

| 维度 | GPTQ | AWQ |
| --- | --- | --- |
| 数学依据 | Hessian 误差重建（二阶） | 激活感知通道保护（一阶） |
| 需要 Hessian/逆矩阵 | 是（计算更重） | 否（只统计激活均值） |
| 校准耗时 | 相对久 | **极快**（快 ~2 倍） |
| 是否需要训练 | 否（one-shot） | 否（one-shot） |
| 效果 | 好 | 相当或略好（尤其小模型） |
| 思路一句话 | "量化后把误差算清并补回来" | "量化前把重要的先护住" |

> **工程结论**：新做 INT4 量化**优先 AWQ**（更快更稳）；存量 GPTQ checkpoint 直接用（生态成熟、格式现成）。

## 四、NF4：4-bit 归一化浮点（重点）

**核心思想**：权重大致服从零中心正态分布。既然分布已知，就**按标准正态分布的分位数设计 16 个量化档位**，让**每个分箱覆盖等概率质量**——这是信息论意义下对正态分布的最优 4-bit 类型。用于 **QLoRA**（低比特微调）。

### 4.1 构造（等概率分箱）

把 $N(0,1)$ 的累积概率 $[0,1]$ 均匀切成段，取分位数作档位（bitsandbytes 的具体实现为非对称：8 正 + 7 负 + 0）：

$$q_i = Q_X\!\left(\frac{i}{2^{k-1}+1}\right), \quad \text{归一化到 } [-1,1], \quad \text{并强制一个精确的 } 0$$

16 个档位（与实际 bitsandbytes 完全一致）：

```text
[-1.000, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.000,
  0.080,  0.161,  0.246,  0.338,  0.441,  0.563,  0.723, 1.000]
```

- **为什么优于均匀 INT4/FP4**：正态分布中间密两边疏，均匀档位在中心区域浪费精度、在尾部精度不足；等概率分箱在"权重最多的地方"放最多档位；
- **注意它不是浮点编码**：4-bit 值只是查表索引，没有符号/指数/尾数分解。

### 4.2 与 QLoRA 的配合

- NF4 通常配 **blocksize=64**：每 64 个权重一块、各块独立 scale，更细的粒度进一步降误差；
- **Double Quantization（DQ）**：再把 32-bit 的 scale 二次量化成 8-bit，平均每个权重再省 ~0.37 bit（→ 约 4.127 bits/param）；
- 效果：QLoRA 用 NF4+DQ 微调，MMLU 等基准**追上 16-bit LoRA**；同条件下 FP4 会稳定掉约 1 个点。

## 五、部署格式与工具

### 5.1 GPTQ / AWQ 的 HF 权重格式与推理

两种算法产出的 checkpoint 都是 HF `safetensors` 格式，包含：**量化权重（packed INT4）+ per-group scale + zero/g_idx**。

> 以下两个加载片段为**用法示意**：需要联网下载模型并安装 `auto-gptq`/`autoawq`/`bitsandbytes` 等依赖，本地无 GPU/网络时不会执行。真正的可运行 demo 见第七节。

```python
# 加载已量化模型（transformers 直接支持，底层走 AutoGPTQ/autoawq）
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ", device_map="auto")
tok = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
out = model.generate(**tok("Hello", return_tensors="pt").to("cuda"), max_new_tokens=32)
print(tok.decode(out[0]))
```

- 量化需另装库：GPTQ 用 `auto-gptq`，AWQ 用 `autoawq`（推理时也依赖 `awq-inference` 的自定义 kernel）；
- 部署到生产环境建议用 **vLLM / TensorRT-LLM**，它们的 INT4 kernel 已集成（见 TensorRT 模块）。

### 5.2 GGUF / llama.cpp（CPU 与 Apple Silicon）

- **GGUF** 是 llama.cpp 的模型格式，内置多种 K-quant（`Q4_K_M`、`Q5_K_S`、`Q8_0` 等），按重要度给每层分配不同位宽（重要性越高位数越多），CPU/Apple Silicon 友好；
- **MMIQ**（Mixed-Matrix IQ，2025）是新一代逐层混合精度量化：先评估每层对量化损失的敏感度，再按敏感度分配 2~8 bit，比固定 Q4 质量更好。

### 5.3 bitsandbytes（NF4 动态加载）

```python
# 需安装 bitsandbytes；运行时把权重 NF4 量化进显存（不落盘、不改文件）
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_4bit=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
```

- 与 GPTQ/AWQ 的区别：**动态量化**（加载时实时算，每次加载都要重新量化），不产固定格式文件；专为训练（QLoRA）设计，权重冻结 + LoRA 微调。

## 六、精度-速度-显存三角对比

以 **7B 模型**为参照（权重显存、自回归解码吞吐、质量损失）：

| 方案 | 权重显存 | 解码吞吐 | 质量损失（参考） |
| --- | --- | --- | --- |
| FP16 | ~14 GB | 1×（基线） | 0（基线） |
| INT8（weight-only） | ~7 GB | ~1.2–1.5× | 几乎无感（PPL +<0.1） |
| INT4 GPTQ/AWQ | ~3.6 GB | ~1.5–2× | LLaMA-7B PPL +0.4~0.6；下游掉 1–2% |
| NF4（QLoRA 场景） | ~3.5 GB | ~1.5× | 微调后追平 16-bit LoRA |

70B 参考：FP16 ≈ 140 GB、INT8 ≈ 70 GB、INT4 ≈ 35 GB（不含 KV cache 与激活）。

**吞吐提升机制拆解**：LLM 生成是**带宽瓶颈**（decode 每步只算 1 个 token 却要读全部权重）。INT4 把每次读取的字节数降到 1/4，带宽压力骤减，吞吐显著提升；但**只省权重带宽**，不加速 matmul 本身太多——所以量化对"小 batch decode"收益最大，对长序列 prefill（算术密集）收益有限（详见 [KV Cache优化技术](../KVCache/KV%20Cache优化技术.md)）。

## 七、源码实现（可运行）

### 7.1 GPTQ 核心：逐列量化 + Hessian 误差补偿

```python
import torch

torch.manual_seed(0)

def quant_sym(x, bits=4):
    """对称 INT 量化→反量化（per-tensor）"""
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().max() / qmax
    q = torch.clamp(torch.round(x / scale.clamp_min(1e-12)), -qmax, qmax)
    return q * scale

def rel_err(a, b):
    return ((a - b).norm() / a.norm()).item()

def gptq_simple(W, X, bits=4):
    """W: (k, n) 权重；X: (n, d) 层输入（n 个输入通道, d 个样本）
    目标 min||WX - WhatX||²，H = X·Xᵀ + λI 近似二阶信息"""
    k, n = W.shape
    lam = 1e-5 * W.pow(2).mean()
    H = X @ X.t() + lam * torch.eye(n)
    Hinv = torch.linalg.inv(H)
    Wq = W.clone()
    for i in range(n):                       # 逐列量化
        w = Wq[:, i].clone()
        q = quant_sym(w, bits)
        delta = q - w                        # 本列量化误差 (k,)
        Wq[:, i] = q
        if i + 1 < n:
            d = Hinv[i, i + 1:] / Hinv[i, i] # 补偿系数（按敏感度分摊）
            Wq[:, i + 1:] -= torch.outer(delta, d)  # 误差补偿后续列
    return Wq

k, n, d = 64, 32, 512
X = torch.randn(n, d) * 0.5
X[0, :] *= 20                                # 制造一个离群通道
W = torch.randn(k, n) * 0.3

err_naive = rel_err(W @ X, quant_sym(W, 4) @ X)      # 朴素逐元素量化
err_gptq  = rel_err(W @ X, gptq_simple(W, X, 4) @ X)  # 补偿量化
print(f"朴素逐元素 INT4 激活误差: {err_naive:.4f}")
print(f"补偿量化   INT4 激活误差: {err_gptq:.4f}")
```

输出：

```text
朴素逐元素 INT4 激活误差: 0.1306
补偿量化   INT4 激活误差: 0.1010
```

**解读**：两种方法的**权重**误差几乎一样（0.134 vs 0.132），但**激活输出**误差从 0.131 降到 0.101——说明补偿量化把误差"推"到了对输出影响小的方向，这正是 GPTQ 的奥义。

### 7.2 AWQ 最小演示：重要通道缩放保护

```python
torch.manual_seed(2)
n_in, n_out, ns = 16, 64, 4096

X = torch.randn(n_in, ns)
X[3] *= 20                       # 通道 3 是"重要通道"（激活很大）
W = torch.randn(n_out, n_in) * 0.3
W[:, 5] = 3.0                    # 通道 5：大权重但小激活的干扰通道

def quant_error_scaled(W, X, alpha=0.5):
    s = X.abs().mean(dim=1).pow(alpha)     # 逐通道缩放因子
    What = quant_sym(W * s, 4) / s         # 缩放→量化→还原
    return rel_err(W @ X, What @ X)

err_no_scale = rel_err(W @ X, quant_sym(W, 4) @ X)
err_awq = quant_error_scaled(W, X, 0.5)
print(f"无缩放   INT4 激活误差: {err_no_scale:.4f}")
print(f"缩放保护 INT4 激活误差: {err_awq:.4f}")
s = X.abs().mean(dim=1).pow(0.5)
print(f"重要通道3 s={s[3].item():.2f}  干扰通道5 s={s[5].item():.2f}")
```

输出：

```text
无缩放   INT4 激活误差: 0.3437
缩放保护 INT4 激活误差: 0.1335
重要通道3 s=4.02  干扰通道5 s=0.89
```

**解读**：不缩放时，干扰通道 5 的 3.0 大权重把全局步长撑大，所有通道一起受损（误差 0.34）；AWQ 把重要通道 3 放大 4 倍、把干扰通道 5 缩小到 0.89，误差降到 0.13（**2.6 倍改善**）。

### 7.3 NF4 量化表：等概率分位数映射

```python
import torch.distributions as D

normal = D.Normal(0, 1)

def create_normal_map(offset=0.9677083):
    """复刻 bitsandbytes NF4 表：非对称 8 正 + 7 负 + 0，等概率分箱"""
    v1 = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])   # 8 个正值
    v3 = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])  # 7 个负值
    v = torch.cat([v1, torch.zeros(1), v3])
    return v.sort().values / v.max()        # 归一化到 [-1, 1]

nf4 = create_normal_map()
print(f"NF4 16 个档位: {[f'{x:.3f}' for x in nf4.tolist()]}")

def quant_nf4(x, table):
    scale = x.abs().max() / table.abs().max()
    x_norm = x / scale.clamp_min(1e-12)
    idx = torch.abs(x_norm.unsqueeze(0) - table.view(-1, 1)).argmin(dim=0)
    return table[idx] * scale

def quant_uni4(x):
    qmax = 7
    scale = x.abs().max() / qmax
    return torch.clamp(torch.round(x / scale.clamp_min(1e-12)), -qmax, qmax) * scale

x = torch.randn(4096) * 2.0
err_nf4  = rel_err(x, quant_nf4(x, nf4))
err_uni4 = rel_err(x, quant_uni4(x))
print(f"正态数据重构误差: NF4 {err_nf4:.4f} vs 均匀INT4 {err_uni4:.4f}")
```

输出：

```text
NF4 16 个档位: ['-1.000', '-0.696', '-0.525', '-0.395', '-0.284', '-0.185', '-0.091', '0.000', '0.080', '0.161', '0.246', '0.338', '0.441', '0.563', '0.723', '1.000']
正态数据重构误差: NF4 0.1198 vs 均匀INT4 0.1847
```

**解读**：同样 4-bit，NF4 靠"把档位放在概率密度高的地方"比均匀 INT4 误差低约 35%——这是 QLoRA 选它的理由。

## 八、高频面试问答

**Q1：GPTQ 的原理？**
对每层权重，用校准激活算 Hessian $H = 2XX^\top$，逐列量化，量化误差按 $H^{-1}$ 的敏感度分摊补偿给其余列，目标是 $\min\|\hat{W}X - WX\|^2$。one-shot、无需训练，模型越大效果越好。

**Q2：AWQ 和 GPTQ 的区别？**
AWQ 靠激活分布找"重要通道"，用 per-channel 缩放保护它们，无 Hessian、更快；GPTQ 靠 Hessian 做误差重建补偿，数学上更"二阶"。效果相当，新量化选 AWQ。

**Q3：NF4 为什么好？**
按标准正态分布等概率分箱设计 16 个档位，在权重分布密集处放更多档位，信息论上对正态权重最优；配 blocksize=64 + double quantization，QLoRA 微调可追平 16-bit LoRA，比 FP4 稳好约 1 个点。

**Q4：INT4 会不会掉很多点？**
不会崩。7B 上 WikiText-2 PPL 大约 +0.4~0.6，下游任务通常掉 1~2%；70B 级几乎可以忽略。但激活必须量化时 INT4 就危险了（离群点），所以主流 INT4 都是 weight-only。

**Q5：为什么 LLM 激活有离群点？**
注意力/FFN 部分隐藏单元形成稳定"专精"通路，少数通道输出恒大；它们在不同 token 间可预测地出现，导致激活带少数大值通道。

**Q6：weight-only 量化是什么意思？为什么够用？**
只量化权重、激活保持 FP16。因为推理瓶颈是带宽（读权重），量化权重就省了大头显存/带宽；激活量化困难（分布动态+离群），不必要就不做。

**Q7：per-tensor、per-channel、per-group 的区别？**
scale 的共享范围不同：整张量/每行每列/每 128 列一组。范围越小、误差越小、但存储 scale 开销越大；LLM INT4 常用 g128 平衡。

**Q8：GPTQ/AWQ 量化的模型怎么部署？**
HF safetensors 格式直接 `AutoModelForCausalLM.from_pretrained` 加载（GPTQ/AWQ 各自依赖 auto-gptq/autoawq），或转 GGUF 用 llama.cpp，生产用 vLLM/TensorRT-LLM 的 INT4 kernel。

## 九、常见误区

1. **"量化就是四舍五入再存整数"** —— 朴素 RTN 只是基线；主流算法都做误差补偿（GPTQ）或分布感知（NF4）/通道保护（AWQ），差别巨大（7B 上 6.29 vs 5.85 PPL）。
2. **"INT4 一定比 INT8 掉点多"** —— 掉点主要来自**激活离群点与定标方式**，不是位宽本身；g128 的 INT4 在 70B 上可能比 per-tensor 的 INT8 更好。
3. **"NF4 和 FP4 差不多"** —— 差 1 个点左右（QLoRA 实证），对正态权重 NF4 是信息论最优。
4. **"量化只省显存不省带宽"** —— 省权重带宽是 LLM 推理提速的主要来源（decode 是带宽瓶颈）；这也是量化+KV 量化叠加收益大的原因。
5. **"GPTQ/AWQ 需要重新训练"** —— 都是 one-shot PTQ，不需要训练；QAT 才是需要训练的路线。

## 十、自我检验

- [ ] 能说出 LLM 量化的三点特殊性（离群激活/分层权重差异/INT4 主流）
- [ ] 能写出 GPTQ 目标 $\min\|WX - \hat{W}X\|^2$、Hessian $H = 2XX^\top$ 与误差补偿更新公式
- [ ] 能解释 AWQ 的 per-channel 缩放为什么有效（重要通道保护）
- [ ] 能画出 NF4 等概率分箱构造并列出 16 个档位
- [ ] 知道 GPTQ/AWQ/NF4 的部署工具（auto-gptq/autoawq/GGUF/bitsandbytes）
- [ ] 能背出 7B 的 FP16/INT8/INT4 权重显存（14/7/3.6 GB）
- [ ] 能回答 8 个面试追问并跑通第七节三个 demo
