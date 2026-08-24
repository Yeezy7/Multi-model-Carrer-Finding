---
learned: false
---

# 预训练目标与 Scaling Law：模型学什么、长大多少

> 本模块索引见 [Tokenizer与大模型基础详解](Tokenizer与大模型基础详解.md)

## 一、预训练目标：三种主流范式

预训练目标定义了"模型在无标注语料上优化什么"。主流有三种，对应三代模型架构。

### 1.1 对比总表

| 目标 | 形式 | 损失 | 代表模型 | 特点 |
| --- | --- | --- | --- | --- |
| **Next-Token Prediction（自回归，causal LM）** | 从左到右，用前 $t$ 个 token 预测第 $t+1$ 个 | 交叉熵（只在每个位置预测右侧一位） | GPT 系、LLaMA、Qwen、所有主流 LLM | 天然支持生成与少样本；训练目标 = 推理目标 |
| **Masked LM（去噪/双向）** | 随机遮住 15% token，用**双向**上下文预测被遮住的 token | 交叉熵（只算被遮位置） | BERT、ELECTRA、RoBERTa | 双向表示强，但不能直接生成 |
| **Prefix LM（前缀语言模型）** | 前段 token 互相可见（双向），后段只预测下一个 | 交叉熵（后段自回归） | UniLM、GLM、早期 T5 变体 | 兼顾编码与生成，混合模式 |

三种目标的核心差异是**注意力的掩码范围**：

- 自回归：只能看左侧（causal mask）；
- Masked LM：能看两侧，但只对 15% 被遮位置算损失；
- Prefix LM：前缀内双向、后缀自回归。

> 主流大模型（GPT 系）只用 Next-Token Prediction。Masked LM 的"双向预训练 → 微调到生成"的错配，是它被大模型时代淘汰的主要原因之一。

### 1.2 教师强制（Teacher Forcing）：训练怎么用 ground truth

自回归训练如果"边生成边训练"，一步错步步错（训练慢、误差累积）。所以训练时**不管模型预测对错，下一位置的输入一律用真实 token（ground truth）**，即"教师强制"：

$$\mathcal{L} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)$$

$$x_{t+1} \text{ 的输入永远取真实值，而不是模型自己生成的 token}$$

- **训练阶段**：整条序列一次前向，所有位置同时算损失（并行）；
- **推理阶段**：模型用自己的输出当输入（自回归），因此训练/推理分布不同（exposure bias）——这是"贪心解码 vs 采样"等解码策略要处理的问题（见 [解码策略详解](../解码策略/解码策略详解.md)）。

## 二、Scaling Law（重点）：模型长大多少，效果涨多少

### 2.1 幂律（Power Law）关系

**Loss 与模型参数量、数据量之间不是线性关系，而是幂律关系：随着规模增大，loss 持续下降但收益递减（log-log 坐标下是直线）。**

GPT-3（Kaplan et al., 2020）的形式（固定计算量、只变模型大小）：

$$L(N) \approx C \cdot N^{-\alpha}, \quad \alpha \approx 0.076$$

Chinchilla（Hoffmann et al., 2022）的双因子形式：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

拟合常数：$E = 1.69,\ A = 406.4,\ \alpha = 0.34,\ B = 410.7,\ \beta = 0.28$。

- $E$ 是不可约损失（熵下限，语料本身的不确定性）；
- $\frac{A}{N^{\alpha}}$：模型容量不够导致的损失；
- $\frac{B}{D^{\beta}}$：数据不够导致的损失；
- 直觉：**参数翻 10 倍，容量项损失只降到原来的 $10^{-0.34} \approx 46\%$**——收益不是 10 倍，而是"按幂律打折"，这就是"规模红利边际递减"的数学表达。

### 2.2 训练计算量公式

训练一个参数量 $N$ 的模型、过 $D$ 个 token，总计算量约为：

$$C \approx 6ND \text{ FLOPs}$$

（前向 + 反向各约 $2ND$，参数更新约 $2ND$ 的常数因子；混合精度可再打折。）示例：7B 模型训 2T token → $6 \times 7\times10^9 \times 2\times10^{12} \approx 8.4\times10^{22}$ FLOPs（详见 [分布式训练详解](../分布式训练/分布式训练详解.md)）。

### 2.3 Chinchilla 结论：数据量 ≈ 参数量 × 20

Chinchilla 的关键发现：**在固定计算预算下，之前（Kaplan/GPT-3 时代）的"大模型、少数据"路线是次优的；最优配比是"更小的模型 + 更多的数据"，约 1 个参数配 20 个训练 token**：

- 70B 参数 → 约 $70 \times 20 = 1.4\text{T}$ tokens；
- 7B 参数 → 约 140B tokens（之前 GPT-3 只用 300B tokens 训 175B，即"欠数据"状态）；
- 同时代的 Gopher（280B，300B tokens）与 Chinchilla（70B，1.4T tokens）算力相同，后者每个任务平均强 7.8%——**同样的钱，花在数据上比花在参数上更值**。

为什么是"20 倍"？把幂律代入固定算力 $C = 6ND$：对给定的 $C$ 求使 $L(N,D)$ 最小的 $(N, D)$。由于 $L$ 对 $N$ 的幂指数 $\alpha$ 与对 $D$ 的指数 $\beta$ 接近，最优解大致落在 $D^* \propto N^*$ 的对角线上（即"参数与数据同比缩放"），比例约为 20:1。精确解（第三节代码演示）显示损失在最优附近非常平坦，20:1 是工程上最好的近似。

### 2.4 Compute-Optimal 的权衡：预算不变时选大模型还是大数据

| 策略 | 配比示例（同算力） | 结果 |
| --- | --- | --- |
| Kaplan 式（参数优先） | 280B 参数 + 300B tokens（D/N ≈ 1） | 欠数据：loss 偏高 |
| Chinchilla 式（数据优先） | 70B + 1.4T（D/N = 20） | loss 显著更低 |
| 极值（公式精确最优） | 33B + 3.0T（D/N ≈ 90） | 与 70B/1.4T 几乎持平（损失差 <0.5%） |

**工程结论**：预算一定时，先按 ~20:1 配比决定 (N, D)；大模型欠数据是常见错误；在最优附近损失面平缓，具体选 70B/1.4T 还是 30B/3T 差别不大，选型由部署（显存、时延）决定。

## 三、损失函数与训练细节

### 3.1 交叉熵（Cross-Entropy）

每个位置的目标是预测下一个 token 的分布，用交叉熵：

$$\mathcal{L}_t = -\log p_\theta(x_{t+1} \mid x_{1:t}), \quad \mathcal{L} = \frac{1}{T}\sum_t \mathcal{L}_t$$

- 词表 $V$ 大 → softmax 归一化分母计算昂贵，常用**分层 softmax / 词表并行切分**优化；
- 数值上等价于 `F.cross_entropy(logits, labels)`（内部做 log-softmax，数值稳定）。

### 3.2 序列长度、Batch 与学习率

| 超参 | 主流取值 | 影响 |
| --- | --- | --- |
| 序列长度 | 2K~128K（LLM 时代 4K~8K 起步） | 越长注意力越贵（$O(T^2)$），但长程建模更强；决定上下文窗口下限 |
| Global batch | 256K~4M tokens（如 1M = 32×8K 序列） | 梯度估计稳定度；MoE/DP 下受显存与通信约束 |
| 学习率 | 峰值 1e-4~3e-4（AdamW），Warmup + Cosine 衰减 | 预训练后期 loss 对 lr 很敏感，通常按数据量 10%~20% 做 warmup |
| 权重衰减 | 0.01~0.1 | 正则化；Embedding 常用更小值 |

常用批大小换算：$1024 \times 4096 \text{ tokens} = 4\text{M tokens/step}$。

## 四、上下文窗口与长文本

### 4.1 训练窗口 ≠ 推理能力

模型只能在"训练时见过的长度"附近稳定工作：

- 在 4K 内训练的模型，直接推理 32K 文本会退化——注意力分数尺度漂移、位置编码没见过超长位置；
- 所谓"上下文 128K"来自**训练窗口本身拉长**（长上下文预训练/续训），或**位置编码外推**（RoPE 的 NTK/缩放、ALiBi、YaRN 等，见 [长度外推技术](../位置编码专题/长度外推技术.md)）。

### 4.2 长文本训练的代价

$$\text{注意力计算} \propto O(T^2), \quad \text{KV Cache} \propto O(T)$$

8K→32K 窗口，单序列注意力算力涨 16 倍、KV Cache 涨 4 倍。这也是长上下文模型要用 GQA、FlashAttention、稀疏注意力（见 [注意力机制专题详解](../注意力机制专题/注意力机制专题详解.md)）的原因。

## 五、源码实现（可运行）

### 5.1 小 GPT：在 tiny 语料上演示 next-token 训练

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

CORPUS = (
    "人工智能是研究如何让计算机模拟人类智能的科学。"
    "大语言模型通过预测下一个词来学习语言的规律。"
    "深度学习依赖海量数据和强大的计算资源。"
    "变压器架构奠定了现代大模型的基础。"
    "分词器把文本变成编号，模型再把编号变回文本。"
)

chars = sorted(set(CORPUS))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in CORPUS])
V = len(chars)
print(f"字符表大小 V = {V}: {''.join(chars)}")

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d=64, h=2, n_layer=2, ctx=32):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln(h))

def get_batch(split, batch_size=16, ctx=32):
    n = len(data) - ctx - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x, y

model = MiniGPT(vocab_size=V)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for step in range(401):
    x, y = get_batch(16)
    logits = model(x)
    loss = lossf(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  loss = {loss.item():.4f}")

@torch.no_grad()
def generate(prompt, n_new=25, temperature=0.8):
    model.eval()
    ids = [stoi[c] for c in prompt]
    for _ in range(n_new):
        x = torch.tensor([ids[-model.ctx:]])
        logits = model(x)[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        ids.append(torch.multinomial(probs, 1).item())
    return "".join(itos[i] for i in ids)

print("\n生成示例（temperature=0.8）:")
print(generate("人工智能", 20))
print(generate("分词器把文本变成", 20))
```

运行输出：

```
字符表大小 V = 70: 。一下个习了人代何依再分压变号和器回型基大奠如学定工度强律成把拟据数文是智本机来构架模测海深源现的研础科究算类编能规言计让词语资赖过通量预，
step    0  loss = 4.4067
step  100  loss = 0.1871
step  200  loss = 0.0587
step  300  loss = 0.0224
step  400  loss = 0.0247

生成示例（temperature=0.8）:
人工智能是研究如何让计算机模拟人类智能的科学。大
分词器把文本变成编号，模型再把编号变回文本变回文本变压器
```

要点：loss 从 4.41（≈ $\ln 70$，纯随机猜 70 类）降到 ~0.02；tiny 语料上模型很快过拟合，生成内容开始复读语料——这正是"预测下一个 token"学习机制的最小可运行演示（教师强制：训练时永远喂 ground truth）。

### 5.2 用合成数据拟合幂律，演示 Scaling Law 的形状

```python
import numpy as np

rng = np.random.default_rng(42)

# 用 Chinchilla 论文的真实拟合常数生成合成数据（loss 在 1.7~2.1 之间）
TRUE = {"E": 1.69, "A": 406.4, "alpha": 0.34, "B": 410.7, "beta": 0.28}
N = np.logspace(8.5, 12.0, 14)        # 参数 300M ~ 1T
D = np.logspace(11.0, 13.5, 14)       # 数据 100B ~ 32T
NN, DD = np.meshgrid(N, D)
L_true = TRUE["E"] + TRUE["A"] * NN ** (-TRUE["alpha"]) + TRUE["B"] * DD ** (-TRUE["beta"])
L = L_true + rng.normal(0, 0.006, L_true.shape)   # 加噪声模拟"实测"

# 网格搜索 (alpha, beta)，每个候选用线性最小二乘解出 (E, A, B)
best = None
for alpha in np.arange(0.20, 0.42, 0.01):
    for beta in np.arange(0.16, 0.40, 0.01):
        X = np.stack([np.ones_like(NN.ravel()),
                      NN.ravel() ** (-alpha), DD.ravel() ** (-beta)], axis=1)
        coef, *_ = np.linalg.lstsq(X, L.ravel(), rcond=None)
        pred = X @ coef
        mse = np.mean((pred - L.ravel()) ** 2)
        if best is None or mse < best[0]:
            best = (mse, alpha, beta, coef)
mse, alpha, beta, (E, A, B) = best
print("=== 拟合 L(N,D) = E + A*N^-alpha + B*D^-beta ===")
print(f"真实: E={TRUE['E']}, A={TRUE['A']}, alpha={TRUE['alpha']}, B={TRUE['B']}, beta={TRUE['beta']}")
print(f"恢复: E={E:.2f}, A={A:.1f}, alpha={alpha:.2f}, B={B:.1f}, beta={beta:.2f}, MSE={mse:.2e}")

# 单因子幂律 L = c * N^(-a)：log-log 线性回归（Kaplan/GPT-3 的风格）
Ns = np.logspace(9, 12, 20)
L_N = 3.0 * Ns ** (-0.076)            # GPT-3 报告: L ∝ N^-0.076
slope, intercept = np.polyfit(np.log(Ns), np.log(L_N), 1)
print("\n=== 单因子拟合 log L = -a*log N + log c （GPT-3 风格）===")
print(f"恢复指数 a = {-slope:.4f}（真实 0.076），常数 c = {np.exp(intercept):.2f}（真实 3.0）")

# 计算最优：固定算力预算 C = 6*N*D 下最小化 L
C = 6 * 70e9 * 1.4e12                  # 70B 模型训练 1.4T token 的预算
Ng = np.logspace(9.0, 11.8, 2000)
Dg = C / (6 * Ng)
Lg = E + A * Ng ** (-alpha) + B * Dg ** (-beta)
i = np.argmin(Lg)
print("\n=== 计算最优（算力预算 C = 6*70B*1.4T）===")
print(f"最优 N* = {Ng[i]/1e9:.0f}B, D* = {Dg[i]/1e12:.1f}T, D*/N* = {Dg[i]/Ng[i]:.0f}:1")
for n, d in [(70e9, 300e9), (280e9, 300e9), (70e9, 1.4e12), (Ng[i], Dg[i])]:
    loss = E + A * n ** (-alpha) + B * d ** (-beta)
    print(f"  L({n/1e9:.0f}B, {d/1e12:.1f}T) = {loss:.4f}")
print("→ 同样算力下，(70B, 1.4T) 明显优于 (280B, 300B) 的'大模型少数据'路线")
```

运行输出：

```
=== 拟合 L(N,D) = E + A*N^-alpha + B*D^-beta ===
真实: E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28
恢复: E=1.70, A=405.5, alpha=0.34, B=522.7, beta=0.29, MSE=2.82e-05

=== 单因子拟合 log L = -a*log N + log c （GPT-3 风格）===
恢复指数 a = 0.0760（真实 0.076），常数 c = 3.00（真实 3.0）

=== 计算最优（算力预算 C = 6*70B*1.4T）===
最优 N* = 33B, D* = 3.0T, D*/N* = 90:1
  L(70B, 0.3T) = 2.0244
  L(280B, 0.3T) = 1.9931
  L(70B, 1.4T) = 1.9360
  L(33B, 3.0T) = 1.9295
→ 同样算力下，(70B, 1.4T) 明显优于 (280B, 300B) 的'大模型少数据'路线
```

三个演示要点：

1. **幂律形状可复现**：加噪声的合成数据能被原模型精确恢复（α=0.34、β=0.29 与真实值 0.34/0.28 一致；单因子 0.076 精确恢复）——这就是"scaling law 是稳定可预测"的数学基础；
2. **算力预算下的最优解**：固定 $C = 6ND$，网格搜索得 N*=33B、D*=3T（比值 90:1 是精确最优）；而 (70B, 1.4T) 的损失 1.936 与最优 1.930 几乎持平——**损失面在最优附近很平**，所以业界常引用的"20:1 经验法则"与"70B → 1.4T"是良好的工程近似；
3. **(280B, 300B) 是次优的**：同算力下损失 1.993 > 1.936，量化证明了"大模型少数据"路线的浪费。

## 六、与多模态的关系

### 6.1 VLM 预训练同样遵循数据规模定律

- 视觉塔 + LLM 的联合预训练中，**损失同样随数据量/参数幂律下降**（CLIP 论文：图文对的幂律 $L \propto (\text{pairs})^{-\gamma}$，$\gamma \approx 0.3$，与文本 LLM 同量级）；
- **视觉编码器冻结 + 只训投影层/LLM 主干**是常见策略（[参数高效微调PEFT](../参数高效微调PEFT/参数高效微调PEFT详解.md) 化），因为视觉塔的数据需求独立且巨大；
- 训 70B 级 VLM：文本部分按 20:1 配数据，视觉部分按"图像质量 × 图文配比"单独规划。

### 6.2 图像-文本配比（Data Mix）

| 配比（文本 tokens : 图像） | 效果倾向 | 代表做法 |
| --- | --- | --- |
| 图像极少（如 1% token 占比） | 文本能力优先，视觉易"遗忘" | 部分图文预训练阶段 |
| 约 10% 图像 token | 视觉/文本平衡 | 主流 VLM（如 Qwen-VL 类） |
| 图像主导（视觉指令数据+图片字幕） | 视觉任务强，对话能力弱 | 只做视觉 SFT 的场景 |

关键工程点：**一张图 = 256~1024 个视觉 token，对应配比要把 token 数换算成"图数"**。比如"1:1 token 配比"= 1 个文本 token 配 1 个视觉 token，大约是一句 50 词的话配 1~2 张图。视觉 token 压缩（Pixel Shuffle）与图像分块（Tiling）都在调这个杠杆（见 [MoE详解](../MoE/MoE详解.md) 第八节）。

## 七、高频面试问答

**Q1：Scaling Law 是什么？**
损失（困惑度）与模型参数量、数据量呈幂律关系 $L(N,D) = E + AN^{-\alpha} + BD^{-\beta}$：规模越大 loss 越低，但边际递减；且 loss 可以脱离训练精确预测，是"训练前算账"的理论基础。

**Q2：为什么 70B 需要 1.4T tokens 数据？**
Chinchilla 发现固定算力下最优配比约 1 参数配 20 个 token：$70B \times 20 = 1.4T$。数据少于这个量，模型"欠训练"（容量没喂饱）；多于这个量，边际收益低于把算力花在更大模型上。

**Q3：为什么同样算力下"小模型+多数据"优于"大模型+少数据"？**
看幂律指数：容量项与数据项接近对称（α≈0.34、β≈0.28），但"大模型+少数据"时 $B/D^\beta$ 项大得离谱（D/N≈1 时数据严重不足）。Chinchilla 实证：Gopher(280B,300B) vs Chinchilla(70B,1.4T)，同算力，后者更强。

**Q4：Next-Token Prediction 和 Masked LM 有什么区别？**
掩码范围与损失位置不同：NTP 因果单向、全位置算损失、可生成；MLM 双向、只算 15% 遮住位置、不能生成。大模型时代 NTP 成为唯一主流。

**Q5：什么是教师强制？有什么问题？**
训练时用真实 token 作为下一位置输入，而不是模型自己的输出。问题是训练/推理分布不一致（exposure bias），推理时生成误差会累积——这正是解码策略（贪心、Beam、采样）要缓解的问题。

**Q6：上下文窗口是训练决定的还是模型结构决定的？**
两者共同决定：结构（位置编码类型）决定"能否外推"，训练长度决定"训没训过"。4K 训练的模型用 RoPE/ALiBi 可外推到 32K（[长度外推技术](../位置编码专题/长度外推技术.md)），但质量打折；长上下文还是要拉长训练窗口或续训。

**Q7：Tokenizer 对训练有影响吗？**
有。词表/分词粒度决定序列长度（训练 FLOPs 与 KV 成本）、embedding 参数量（$V\times d$）、每 token 样本的语义粒度。多语言配比不当会让某些语言 token 稀疏学不好。词表是"模型的第一个超参"。

**Q8：VLM 的预训练目标还是 next-token 吗？**
是，但目标序列混合了文本与视觉 token：文本 token 正常预测下一个，视觉 token 对应的位置通常不参与损失（视觉特征已经由编码器给定）。所以 VLM 的 LM 头仍然只输出文本词表。

## 八、自我检验

- [ ] 能画出三种预训练目标（NTP / MLM / Prefix LM）的掩码与损失位置
- [ ] 能写出 $L(N,D) = E + AN^{-\alpha} + BD^{-\beta}$ 并解释三项含义
- [ ] 能背出 α≈0.34、β≈0.28、$C \approx 6ND$、D/N≈20 四个关键数字
- [ ] 能推导并说出"70B → 1.4T tokens"的来源
- [ ] 能讲清教师强制的机制与 exposure bias
- [ ] 能复现 5.1 的小 GPT 训练与 5.2 的幂律拟合（亲自跑一遍）
- [ ] 能说出 VLM 预训练的"两套 token 混合 + 图像文本配比"
- [ ] 能回答第七节全部 8 个追问
