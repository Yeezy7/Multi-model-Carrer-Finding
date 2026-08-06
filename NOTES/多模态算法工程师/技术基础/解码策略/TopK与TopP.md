# Top-K 与 Top-P 采样

> 本模块索引见 [解码策略详解](解码策略详解.md)

## 一、定义与公式

### 1.1 Top-K 采样

只从概率最高的 K 个 token 中采样（其余 token 概率置零并重归一化）：

$$\mathcal{V}_t = \{v \in V : \text{rank}(p(v \mid x_{<t})) \le K\}, \qquad p'(v) = \frac{p(v)}{\sum_{v' \in \mathcal{V}_t} p(v')} \quad (v \in \mathcal{V}_t)$$

### 1.2 Top-P 采样 / 核采样（Nucleus Sampling, Holtzman et al., 2019）

从概率最高的 token 开始累加，直到累积概率超过阈值 $p$，把这一组 token 作为候选集：

$$\mathcal{V}_p = \left\{ v_{(1)}, v_{(2)}, \dots, v_{(k)} : \sum_{i=1}^{k} p(v_{(i)}) \ge p \right\}$$

其中 $v_{(i)}$ 按概率降序排列，$k$ 是最小满足条件的数量。候选集大小**随分布形态自适应**。

## 二、核心原理

### 2.1 Top-K 的"K 固定"缺陷（必考）

Top-K 的 K 是固定值，但不同时间步的概率分布形态差异巨大：

| 分布形态 | 例子 | Top-K 的问题 |
|----------|------|-------------|
| **尖峰分布**（很尖） | "1+1=___"，正确答案概率 0.98 | K=50 时混入大量无意义 token，稀释正确概率 |
| **平坦分布**（很平） | "继续讲故事……" 后续 500 个 token 概率相近 | K=10 时把所有合理候选都剪掉了 |
| 极端情形 | 尖峰时 K=1 才合理；平坦时 K=500 也不够 | **单一 K 无法同时适配** |

**结论**：Top-K 假设"所有时间步的合理候选数相同"，这是错的。因此单独使用效果一般，常与 Top-P 叠加。

### 2.2 Top-P 为什么自适应（必考）

| 分布 | Top-K 表现 | Top-P 表现 |
|------|-----------|-----------|
| 尖峰分布（1+1=2） | K 固定 → 混入噪音或剪掉对的 | 累积概率很快到 0.95 → 候选集只有 1~2 个 token，精准 |
| 平坦分布（自由续写） | K=10 剪掉合理候选 | 累积到 0.95 需要几十上百个 token → 候选多，保留多样性 |
| 长尾分布 | 可能采到概率 0.0001 的乱 token | 长尾概率小、累加贡献低，被自动排除 |

**一句话**：Top-P 永远"只装下累积概率 p 的核（nucleus）"——分布尖时候选少、分布平时候选多。

### 2.3 工程默认值

- 默认 $p = 0.9 \sim 0.95$；K 常用 40~100（OpenAI 早期默认 top_k=40）；
- p 太小 → 多样性降低、接近 greedy；p 太大（≈1）→ 长尾噪音进入候选集；
- 两者叠加：`top_k=50, top_p=0.9`，候选集是两者的交集，先做 top_k 再做 top_p。

### 2.4 实现细节：为什么 `cumsum - probs > p` 而不是 `cumsum > p`

```text
mask = cumsum - sorted_probs > top_p     # 减掉当前项再判断
```

**保证第一个（概率最大的）token 总在候选集里**——即使它自己的概率已经超过 p（比如 $p(v_1)=0.98 > 0.9$，此时 `cumsum - v₁ = 0` 不满足 `> p`，v₁ 保留）。

## 三、源码实现

### 3.1 过滤函数 + 尖峰/平坦分布适配演示

```python
import torch

def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """只保留 logits 最大的 k 个，其余置 -inf（softmax 后概率为 0）"""
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    threshold = torch.topk(logits, k).values[..., -1, None]
    return torch.where(logits < threshold, float("-inf"), logits)

def filter_top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """核采样过滤：按概率降序累加，超过 p 之后的置 -inf"""
    sorted_z, sorted_idx = logits.sort(descending=True)
    probs = torch.softmax(sorted_z, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    mask = (cumsum - probs) > p           # 关键：减掉自身，保证最大项必留
    sorted_z[mask] = float("-inf")
    return sorted_z.scatter(-1, sorted_idx, sorted_z)

sharp = torch.tensor([4.0, -1.0, -1.1, -1.2, -1.3])     # 尖峰："1+1=2"（无并列，便于演示）
flat  = torch.tensor([0.5] * 5)                         # 平坦：自由续写

def fmt(z):
    return torch.softmax(z, -1).double().numpy().round(3).tolist()

print("原始分布  sharp:", fmt(sharp), " flat:", fmt(flat))
print("top_k=3   sharp:", fmt(filter_top_k(sharp, 3)), " flat:", fmt(filter_top_k(flat, 3)))
print("top_p=0.6 sharp:", fmt(filter_top_p(sharp, 0.6)), " flat:", fmt(filter_top_p(flat, 0.6)))
# 原始分布  sharp: [0.977, 0.007, 0.006, 0.005, 0.005]  flat: [0.2, 0.2, 0.2, 0.2, 0.2]
# top_k=3   sharp: [0.987, 0.007, 0.006, 0.0, 0.0]      ← 尖峰时硬塞 2 个噪音进候选
#           flat:  [0.2, 0.2, 0.2, 0.2, 0.2]           ← 平坦时 5 个全保留，过滤无效
# top_p=0.6 sharp: [1.0, 0.0, 0.0, 0.0, 0.0]            ← 尖峰时只留 1 个候选，精准！
#           flat:  [0.25, 0.25, 0.25, 0.25, 0.0]        ← 平坦时保留 4 个，多样性不丢
```

> **读表结论**：同一个 K=3，在尖峰分布里把概率 0.005 的噪音 token 硬塞进候选集（top_k 只看数量不看概率，K=50 时这种噪音会累积稀释正确答案）、在平坦分布里没起到任何过滤作用（K 大于合理候选数）；而 top_p=0.6 在尖峰时收敛到 1 个候选、在平坦时保留 4 个候选——候选集大小随分布自适应。

### 3.2 手写 Top-P 采样循环 + 文本条形图可视化（玩具分布）

```python
import torch

torch.manual_seed(0)

VOCAB = ["a", "b", "c", "<eos>"]
EOS = 3

def toy_probs(prefix: tuple) -> torch.Tensor:
    if not prefix:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    last = prefix[-1]
    if last == 0:
        return torch.tensor([0.0, 0.6, 0.4, 0.0])     # 相对平坦：b(0.6) c(0.4)
    if last == 1:
        return torch.tensor([0.0, 0.9, 0.0, 0.1])     # 尖峰：b(0.9) 主导
    return torch.tensor([0.0, 0.0, 0.0, 1.0])

def toy_logits(prefix: tuple) -> torch.Tensor:
    p = toy_probs(prefix)
    return torch.log(torch.where(p > 0, p, torch.full_like(p, 1e-7)))

def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    threshold = torch.topk(logits, k).values[..., -1, None]
    return torch.where(logits < threshold, float("-inf"), logits)

def filter_top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    sorted_z, sorted_idx = logits.sort(descending=True)
    probs = torch.softmax(sorted_z, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    mask = (cumsum - probs) > p
    sorted_z[mask] = float("-inf")
    return sorted_z.scatter(-1, sorted_idx, sorted_z)

def decode(p=0.9, max_len=10):
    prefix, tokens = [], []
    for _ in range(max_len):
        logits = filter_top_p(toy_logits(tuple(prefix)), p)
        v = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
        if v == EOS:
            break
        prefix.append(v)
        tokens.append(VOCAB[v])
    return "".join(tokens)

print("top_p=0.5 :", [decode(p=0.5) for _ in range(5)])
# top_p=0.5 : ['abbbbbbbbb', 'abbbbbbbbb', 'abbbbbbbbb', 'abbbbbbbbb', 'abbbbbbbbb']
#             ← p 小 → "a" 之后累积 0.6 只装下 b，全掉进 b 陷阱
print("top_p=0.99:", [decode(p=0.99) for _ in range(5)])
# top_p=0.99: ['ac', 'abbbbbbbb', 'abbb', 'ac', 'abbbbbbbbb']
#             ← p 大 → 平坦处 b/c 都被保留，绕开陷阱的比例明显上升

def bar(z, title):
    p = torch.softmax(z, -1)
    print(title)
    for i, (name, pi) in enumerate(zip(["v1", "v2", "v3", "v4", "v5"], p)):
        print(f"  {name}: {'#' * int(pi * 50):<50} {pi:.2f}")

sharp = torch.tensor([4.0, -1.0, -1.1, -1.2, -1.3])
bar(sharp, "原始（尖峰，v1 概率 0.98）")
bar(filter_top_k(sharp, 3), "top_k=3（硬塞 2 个噪音候选）")
bar(filter_top_p(sharp, 0.9), "top_p=0.9（只剩 v1，精准）")
# 原始（尖峰，v1 概率 0.98）
#   v1: ################################################   0.98
#   v2:                                                    0.01
#   v3:                                                    0.01
#   v4:                                                    0.01
#   v5:                                                    0.00
# top_k=3（硬塞 2 个噪音候选）
#   v1: ################################################# 0.99
#   v2:                                                    0.01
#   v3:                                                    0.01
#   v4:                                                    0.00
#   v5:                                                    0.00
# top_p=0.9（只剩 v1，精准）
#   v1: ################################################## 1.00
#   v2:                                                    0.00
#   v3:                                                    0.00
#   v4:                                                    0.00
#   v5:                                                    0.00
```

> **条形图读法**：同一个尖峰分布——原始 5 个 token 都在候选；top_k=3 只是"按数量截断"，概率 0.005 的噪音 token 依然留在候选集里参与重归一化（K 越大这类噪音越多，稀释正确答案，见 3.1 表格）；top_p=0.9 只剩 v1 一个候选，采样必然选对。

### 3.3 与 transformers generate 对比（top_k + top_p 叠加）

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
cfg = GPT2Config(vocab_size=100, n_layer=2, n_head=4, n_embd=128, n_positions=64)
model = GPT2LMHeadModel(cfg).eval()
model.config.pad_token_id, model.config.eos_token_id = 0, 0
input_ids = torch.tensor([[1, 2]])
attn = torch.ones_like(input_ids)

def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    threshold = torch.topk(logits, k).values[..., -1, None]
    return torch.where(logits < threshold, float("-inf"), logits)

def filter_top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    sorted_z, sorted_idx = logits.sort(descending=True)
    probs = torch.softmax(sorted_z, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    mask = (cumsum - probs) > p
    sorted_z[mask] = float("-inf")
    return sorted_z.scatter(-1, sorted_idx, sorted_z)

gen = model.generate(input_ids, attention_mask=attn, do_sample=True, temperature=0.7,
                     top_k=50, top_p=0.9, max_new_tokens=3,
                     return_dict_in_generate=True, output_scores=True)
logits = model(input_ids, attention_mask=attn).logits[:, -1]
my_probs = torch.softmax(filter_top_p(filter_top_k(logits / 0.7, 50), 0.9), -1)
hf_probs = gen.scores[0][0].softmax(-1)
print("与 HF 分布最大偏差:", (my_probs - hf_probs).abs().max().item())
# 与 HF 分布最大偏差: 1.1e-08   ← 顺序：温度 → top_k → top_p，与 HF 完全一致
```

## 四、深入分析

### 4.1 组合使用的正确姿势

- **Top-K 先裁掉极端长尾**（省计算、防意外），**Top-P 再动态收核**——候选集是两者交集；
- p 与 K 的关系：K 是"绝对上限"，p 是"动态下限"；K 很小 p 很大 ≈ 只有 top_k 生效，K 很大 p 很小 ≈ 只有 top_p 生效；
- 业界默认：`top_k=50, top_p=0.9~0.95, temperature=0.7~1.0`。

### 4.2 多样性 / 质量权衡

- p 太小 → 候选集小 → 接近 greedy，输出死板、重复风险高；
- p 太大 → 长尾噪音 token 进入候选集 → 输出"胡言乱语"；
- 多模态模型的部分位置（视觉 token 注入后的注意力分布）容易产生尖锐分布，尤其需要 top_p 兜底，否则每个 token 都可能从尾巴里采。

### 4.3 边界细节

1. **第一个 token 的保留**：`cumsum - probs > p` 的写法保证最大概率 token 必在候选集（2.4）；
2. **重归一化**：过滤后必须重新 softmax/归一化，否则候选集内概率之和小于 1；
3. **平局处理**：Top-K 的阈值处若存在并列 logits（严格小于才过滤），并列 token 会全部保留——HuggingFace 的 `TopKLogitsWarper` 同样是严格小于，行为一致；
4. **p 与量化**：量化会扰动 logits，top_p 边界处的 token 是否被包含可能抖动，评测时固定 seed 对比。

## 五、优缺点

| 策略 | 优点 | 缺点 |
|------|------|------|
| Top-K | 实现简单、硬性保证不采长尾 | K 固定不自适应，尖/平分布都别扭 |
| Top-P | 候选集随分布自适应、自动排除长尾、实现简单 | 需要排序+累积（小开销）；p 需按任务调 |
| 组合 | 既有绝对上限又有动态裁剪 | 多一个超参数，调参成本略升 |

## 六、与同类对比

| 策略 | 核心机制 | 自适应 | 一句话点评 |
|------|---------|--------|-----------|
| Top-K | 固定前 K 个 | 否 | 硬截断，尖/平场景都别扭 |
| Top-P | 累积概率达 p | 是 | 默认首选，动态收核 |
| Min-P | 相对最大概率的阈值 | 是 | 对分布平滑度更敏感，DeepSeek 推荐 min_p=0.02~0.05 与 top_p 联用 |
| Typical | 与负熵的距离 | 是 | 信息论视角，更贴近人类语言但参数敏感 |
| 温度 | 缩放 logits | 否 | 只调形状，不截长尾，必须配合上面三者 |

## 七、高频面试问答

**Q1：Top-P 为什么比 Top-K 好？**
Top-K 的候选数固定，无法适配分布形态；Top-P 动态截断到累积概率 p——尖时候选少（精准），平时候选多（保留多样性），永远只保留概率质量占比 p 的"核"，且长尾噪音被自动排除。默认 p=0.9~0.95。

**Q2：Top-K 的 K 固定有什么问题？请举例。**
尖峰分布（1+1=2，正确答案概率 0.98）：K=50 混入大量无意义 token，稀释正确概率；平坦分布（自由续写，500 个 token 概率相近）：K=10 把所有合理候选剪光。单一 K 无法同时适配两种形态。

**Q3：Top-P 实现时为什么用 `cumsum - probs > p` 而不是 `cumsum > p`？**
保证第一个（概率最大的）token 总在候选集里：当它自己的概率超过 p 时（如 0.98 > 0.9），`cumsum - probs` 为 0，不会触发置零；否则最大概率 token 可能被过滤掉，候选集为空或丢失最合理的候选。

**Q4：top_k 和 top_p 同时设置时执行顺序是什么？**
先 top_k 截断再 top_p 截断，最终候选集是两者交集；更完整的管线是 repetition_penalty → 温度 → top_k → top_p → softmax → 采样（HuggingFace 的 LogitsWarper 顺序）。

**Q5：p 太大或太小分别会怎样？**
p 太小 → 候选集小，退化为 greedy，多样性差、重复风险高；p 太大（≈1）→ 长尾噪音进入候选集，可能输出乱 token。0.9~0.95 是工程甜点区。

**Q6：Top-P 会改变 token 的排序吗？**
不会。它只把尾巴上的 token 置零并重归一化，候选集内部仍保持原有相对概率；排序改变只发生在 Top-K 的边界处（置零即除名）。

**Q7：Min-P 和 Top-P 的区别？**
Top-P 看"累积概率"，Min-P 看"相对最大概率"：只保留 $p(v) \ge \min\_p \cdot \max p$ 的 token。Min-P 在低熵（尖）分布下更稳定，两者可叠加使用（如 DeepSeek 官方配置）。

## 八、自我检验

- [ ] 能写出 Top-K 与 Top-P 的候选集公式
- [ ] 能用尖峰/平坦两个例子讲清"K 固定"的缺陷
- [ ] 能写出 top_k 过滤与 top_p 过滤的完整 torch 实现
- [ ] 能解释 `cumsum - probs > p` 的动机
- [ ] 能复现"尖峰时 top_p 只剩 1 个候选、平坦时保留多个"的数值演示
- [ ] 知道 top_k → top_p 的执行顺序及完整采样管线
- [ ] 能说出默认值 top_k=50、top_p=0.9~0.95
- [ ] 知道 Min-P、Typical 的核心思想（一句话各够）
- [ ] 能跑通手写 top-p 采样循环并解释 p 大小对结果的影响
- [ ] 能回答 7 个高频面试追问
