# Greedy 与 Beam Search 解码

> 本模块索引见 [解码策略详解](解码策略详解.md)

## 一、定义与公式

### 1.1 Greedy Decoding（贪心解码）

每一步直接取当前概率分布中概率最大的 token：

$$x_t = \arg\max_{v \in V} \; p(v \mid x_{<t})$$

直到生成结束符 `<eos>` 或达到最大长度 $L$。整个过程**前向贪婪**：只看局部最优，不回溯、不重试。

### 1.2 Beam Search（束搜索）

每一步维护 **B 个候选序列**（B 为束宽 beam size）：

$$S(y_{1:t}) = \sum_{i=1}^{t} \log p(y_i \mid y_{<i})$$

打分用**对数概率累加**（而不是概率连乘），原因见 3.4。最终返回得分最高的完整序列。

### 1.3 长度归一化（Length Penalty）

纯 log 概率累加有**长度偏差**：每多一个 token 就多乘一个 $\log p < 0$，得分必然更小，导致 Beam 系统性偏好短序列。标准做法（GNMT，Wu et al. 2017）：

$$\text{score}(y) = \frac{1}{|y|^{\alpha}} \sum_{i=1}^{|y|} \log p(y_i \mid y_{<i})$$

- $\alpha = 1$：均匀长度归一化（HuggingFace 默认 `length_penalty=1.0`）；
- $\alpha > 0$：鼓励长序列；$\alpha < 0$：鼓励短序列。

> 注意：有的框架写作 $\log P / T$（除以温度形式），本质相同——都是把"总得分"折算成"平均每步得分"。

## 二、核心原理

### 2.1 Greedy 的局部最优问题（必考）

Greedy 每一步选概率最大的 token，但**第 $t$ 步的最优 token 不代表整句最优**：一个概率略低的 token 可能把句子带进高概率的后续分支，而概率最高的 token 可能通向"死胡同"。

| 步骤 | Greedy 的选择 | 全局最优 |
|------|--------------|---------|
| step1 | 取 $p$ 最大的 token（局部最优） | 不等于全局最优路径的首 token |
| step2 | 继续取当前最大 | 此时可能已经走进低概率分支 |
| 结果 | 整句概率低 | 换一条路整句概率可能高一个数量级 |

### 2.2 Beam Search 的完整算法

1. **step1**：计算所有 token 的概率，保留概率最大的 B 个作为候选；
2. **step t**：对当前 B 个候选各自扩展词表 $V$，得到 $B \times |V|$ 个新候选；
3. 按**累计 log 概率得分**排序，保留得分最高的 B 个；
4. 重复直到所有候选结束（或达到 $L$），返回得分最高的完整序列。

```text
Beam Search 示例（B=2, 词表 {a, b, <eos>}）:
step1: 所有序列        a(0.4)  b(0.35)  →  保留 a, b
step2: 扩展            aa(0.16) ab(0.24) ba(0.14) bb(0.21)
      →  保留得分最高的 ab(0.24), bb(0.21)
step3: 扩展            aba(0.096) ... abb(0.084) bba(0.126) bbb(0.084)
      →  保留 bba(0.126), aba(0.096) ...
```

复杂度 $O(B \cdot |V| \cdot L)$：B 每翻一倍，计算量近似翻倍。

### 2.3 束宽 B 的影响

| 束宽 | 质量 | 速度/内存 | 说明 |
|------|------|-----------|------|
| B=1 | 次优 | 最快 | 退化为 greedy |
| B=4~8 | 明显提升（翻译/摘要） | 中等 | 工程常用默认 |
| B=16~32 | 边际收益递减 | 慢 | 竞赛刷分偶尔用 |
| B→∞ | 理论全局最优 | 指数爆炸 | 不现实 |

> **记忆点**：B 越大越接近全局最优，但收益边际递减；同时**重复问题越严重**（B 条相似路径互相踩同一条路）。

## 三、源码实现

### 3.1 玩具概率分布：一个会"骗"过 greedy 的转移表 + 手写 Greedy

```python
import torch

VOCAB = ["a", "b", "c", "<eos>"]
EOS = 3

def toy_probs(prefix: tuple) -> torch.Tensor:
    """玩具模型：直接给出条件概率表，故意设计成
    greedy 会掉进重复陷阱、beam 能绕开的结构。"""
    if not prefix:                     # 起始：必为 a
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    last = prefix[-1]
    if last == 0:                      # a 之后：b(0.6) 比 c(0.4) 更可能
        return torch.tensor([0.0, 0.6, 0.4, 0.0])
    if last == 1:                      # b 之后：0.9 概率继续 b（重复陷阱）
        return torch.tensor([0.0, 0.9, 0.0, 0.1])
    return torch.tensor([0.0, 0.0, 0.0, 1.0])   # c 之后：直接结束

def toy_logits(prefix: tuple) -> torch.Tensor:
    p = toy_probs(prefix)
    return torch.log(torch.where(p > 0, p, torch.full_like(p, 1e-7)))  # log(0) 用极小值代替

def probs(prefix: tuple) -> torch.Tensor:
    return torch.softmax(toy_logits(prefix), dim=-1)

def greedy_decode(max_len: int = 8):
    """手写 greedy：每一步 argmax，不回溯。"""
    prefix, total = [], 1.0
    for _ in range(max_len):
        p = probs(tuple(prefix))
        nxt = int(p.argmax())
        total *= p[nxt].item()         # 累计真实概率（用于对比）
        if nxt == EOS:
            break
        prefix.append(nxt)
    return prefix, total

print("分布检查: p(b|a) =", probs((0,)).numpy().round(3), " p(c|a) =", round(probs((0,))[2].item(), 3))
# 分布检查: p(b|a) = [0. 0.6 0.4 0.]  p(c|a) = 0.4

path, prob = greedy_decode()
print("greedy:", "".join(VOCAB[t] for t in path), f"总概率={prob:.4f}")
# greedy: abbbbbbb 总概率=0.3189
# 解释：a→b(0.6) 是局部最优，但 b 之后 0.9 概率继续 b（重复陷阱），
#       直到 max_len 耗尽也没等到 eos；总概率 0.6*0.9^6 = 0.3189
#       对比：全局最优路径 a→c→eos 的总概率是 0.4，greedy 选错了
```

### 3.2 手写 Beam Search（完整实现）

```python
import torch

VOCAB = ["a", "b", "c", "<eos>"]
EOS = 3

def toy_probs(prefix: tuple) -> torch.Tensor:
    if not prefix:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])
    last = prefix[-1]
    if last == 0:
        return torch.tensor([0.0, 0.6, 0.4, 0.0])
    if last == 1:
        return torch.tensor([0.0, 0.9, 0.0, 0.1])
    return torch.tensor([0.0, 0.0, 0.0, 1.0])

def probs(prefix: tuple) -> torch.Tensor:
    p = toy_probs(prefix)
    return torch.softmax(torch.log(torch.where(p > 0, p, torch.full_like(p, 1e-7))), dim=-1)

def beam_search(num_beams: int = 2, max_len: int = 8, alpha: float = 0.0) -> tuple:
    """标准 beam search：
    1) 对 B 个候选各扩展词表 V，得到 B*V 个新候选
    2) 按累计 log 概率排序，保留 top-B
    3) 完成的序列（以 eos 结尾）移入 finished 池
    4) 最终按长度归一化得分选出最优（alpha 为长度惩罚指数）"""
    beams, scores = [()], torch.tensor([0.0])
    finished = []                                   # (归一化得分, 序列)
    for _ in range(max_len):
        if not beams:
            break
        cand_ids, cand_scores = [], []
        for i, seq in enumerate(beams):
            logp = probs(seq).log()
            for v in range(len(VOCAB)):
                cand_ids.append(seq + (v,))
                cand_scores.append(scores[i] + logp[v])
        k = min(num_beams * 2, len(cand_scores))    # 候选不足时 topk 取全部
        top_scores, top_idx = torch.topk(torch.stack(cand_scores), k)
        new_beams, new_scores = [], []
        for s, idx in zip(top_scores, top_idx):
            seq = cand_ids[idx]
            if seq[-1] == EOS:                       # 完成的候选移入 finished
                finished.append((s.item() / (len(seq) ** alpha), seq))
            else:
                new_beams.append(seq)
                new_scores.append(s)
            if len(new_beams) == num_beams:
                break
        beams, scores = new_beams, torch.stack(new_scores)
    if finished:
        return max(finished, key=lambda x: x[0])[1]
    return beams[0]

for B in (1, 2, 3):
    path = beam_search(num_beams=B)
    total = 1.0
    for i in range(len(path)):
        total *= probs(path[:i])[path[i]].item()
    print(f"B={B}:", "".join(VOCAB[t] for t in path), f"总概率={total:.4f}")
# B=1: abbbbbbb 总概率=0.3189   ← 退化为 greedy，掉进重复陷阱
# B=2: aceos    总概率=0.4000   ← 保留了 a-c 这条"次优但更优"的路
# B=3: aceos    总概率=0.4000   ← 候选更多但最优不变
```

### 3.3 长度归一化演示（为什么需要 length penalty）

```python
import math

s1 = math.log(0.9) + math.log(0.9)                        # 2 步，每步 0.9
s2 = math.log(0.9) + math.log(0.9) + math.log(0.99)       # 3 步，多一步 0.99
print(f"不归一化: S1={s1:.4f} > S2={s2:.4f}   → 序列越长得分越低，偏好短句")
# 不归一化: S1=-0.2107 > S2=-0.2208   → 序列越长得分越低，偏好短句
print(f"归一化α=1: S1={s1/2:.4f} < S2={s2/3:.4f} → 平均每步质量相当，长句被公平对待")
# 归一化α=1: S1=-0.1054 < S2=-0.0736 → 平均每步质量相当，长句被公平对待
```

### 3.4 与 transformers generate 对比（greedy + beam 精确对齐）

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
cfg = GPT2Config(vocab_size=100, n_layer=2, n_head=4, n_embd=128, n_positions=64)
model = GPT2LMHeadModel(cfg).eval()
model.config.pad_token_id, model.config.eos_token_id = 0, 0
input_ids = torch.tensor([[1, 2]])
attn = torch.ones_like(input_ids)

def my_greedy(model, ids, max_new=8):
    for _ in range(max_new):
        nxt = model(ids, attention_mask=torch.ones_like(ids)).logits[:, -1].argmax(-1)
        if nxt.item() == model.config.eos_token_id:
            break
        ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
    return ids

def my_beam(model, ids, num_beams=2, max_new=8, alpha=0.0):
    device = ids.device
    beams, scores, finished = [ids], torch.zeros(1), []
    for _ in range(max_new):
        if not beams:
            break
        lprobs = torch.log_softmax(model(torch.cat(beams)).logits[:, -1], dim=-1)
        flat = (scores.unsqueeze(1) + lprobs).flatten()
        top_scores, top_idx = torch.topk(flat, min(num_beams * 2, len(flat)))
        new_beams, new_scores = [], []
        for s, idx in zip(top_scores, top_idx):
            b, tok = divmod(int(idx), lprobs.shape[-1])
            ids_i = torch.cat([beams[b], torch.tensor([[tok]], device=device)], dim=1)
            if tok == model.config.eos_token_id:
                finished.append((s.item() / len(ids_i[0]) ** alpha, ids_i))
            else:
                new_beams.append(ids_i)
                new_scores.append(s)
            if len(new_beams) == num_beams:
                break
        beams, scores = new_beams, torch.stack(new_scores)
    return max(finished, key=lambda x: x[0])[1] if finished else beams[0]

out_hf = model.generate(input_ids, attention_mask=attn, do_sample=False, max_new_tokens=8)
out_my = my_greedy(model, input_ids)
print("greedy 对齐:", torch.equal(out_hf, out_my))
# greedy 对齐: True   ← 逐 token 完全一致

out_hf = model.generate(input_ids, attention_mask=attn, do_sample=False,
                        num_beams=2, max_new_tokens=8, length_penalty=0.0)
out_my = my_beam(model, input_ids, num_beams=2)
print("beam 对齐:", torch.equal(out_hf, out_my))
# beam 对齐: True   ← 打分/剪枝逻辑一致，最优序列相同
```

> **对齐前提**：① 打分同为 log-softmax 累加；② 长度惩罚指数一致（`length_penalty=0.0` 对应 $\alpha=0$）；③ 完成的 beam 单独存放、最终比较。

## 四、深入分析

### 4.1 重复性（两者共同的病）

1. **Greedy 放大高频倾向**：一旦进入重复循环，循环内每个 token 的条件概率都高，模型"自己出不来"（3.1 的玩具分布就是演示）；
2. **Beam 的叠加效应**：B 条路径源自同一个父序列，彼此高度相似，重复模式会被多条路径同时选中并互相加强；
3. **注意力崩溃**：重复位置 attention 失焦，模型"忘了"前面说过什么；
4. **训练/推理不一致**：训练用 teacher forcing，推理用自生成前缀，分布偏移导致退化。

### 4.2 多样性问题

- Greedy 永远只走一条路，**零多样性**——同一 prompt 永远同一输出；
- Beam 的 B 条候选往往是同一句的微小改写（"集束内无差别"），浪费束宽；
- 缓解：`no_repeat_ngram_size`、repetition penalty（见[重复惩罚](重复惩罚.md)）、Diverse Beam Search（分组 + 组间互斥惩罚）。

### 4.3 与采样对比

| 维度 | Beam Search | 采样（Top-P + 温度） |
|------|------------|---------------------|
| 目标 | 逼近最优解（忠实、准确） | 合理且多样的答案 |
| 候选 | 固定 B 条路径 | 每步独立掷骰子 |
| 确定性 | 高（同输入同输出） | 低（需固定 seed） |
| 失败模式 | 重复、平庸、"翻译腔" | 发散、幻觉、答非所问 |

### 4.4 长度偏差

- 不归一化 → 偏好短序列，容易"草草结束"（见 3.3 数值演示）；
- 归一化过头 → 偏好长废话，重复滚雪球；
- 工程上翻译/摘要用 `length_penalty=1.0` 起步，按任务调 α。

## 五、优缺点

| 策略 | 优点 | 缺点 |
|------|------|------|
| Greedy | 实现简单（一个 argmax）、确定、快、省内存 | 局部最优 ≠ 全局最优、重复、零多样性 |
| Beam | 全局最优近似、质量高、适合得分可比较任务 | 慢（B 倍计算）、重复最严重、长度偏差、多样性差 |

## 六、与同类对比

| 策略 | 确定性 | 多样性 | 速度 | 重复风险 | 适用场景 |
|------|--------|--------|------|---------|---------|
| Greedy | 高 | 最低 | 最快 | 高 | 评测、精确答案 |
| Beam Search | 高 | 低 | 慢 | 最高 | 翻译、摘要、caption |
| 温度采样 | 低 | 随 T 升高 | 快 | 中 | 对话、创意写作 |
| Top-P 采样 | 低 | 自适应 | 快 | 低 | 通用生成默认 |

> **经验法则**：目标"逼近最优答案"→ Beam；目标"合理且多样"→ 采样。翻译/摘要的评测指标（BLEU/ROUGE）与 Beam 高度契合，所以 Beam 是它们的默认选择。

## 七、高频面试问答

**Q1：Greedy 和 Beam Search 的区别？**
Greedy 每步只取当前概率最大的 token，一条路走到底，局部最优；Beam Search 每步保留 top-B 候选、按累计 log 概率选最优，是全局最优的近似。Beam 质量更高但更慢、更容易重复；B=1 时 Beam 退化为 Greedy。

**Q2：能不能举一个"greedy 选错"的例子？**
可以：a 之后 $p(b)=0.6, p(c)=0.4$，greedy 选 b；但 b 之后 0.9 概率继续 b（重复陷阱），整条路径概率只有 0.287；而 c 之后立即结束，路径总概率 0.4。greedy 的局部最优把句子带进了低概率分支。

**Q3：为什么用 log 概率累加而不是概率连乘？**
两个原因：① 数值下溢——几十个 0.1 连乘趋近 0；② 概率连乘对长度敏感，log 后把乘法变加法，数值稳定且得分可比较。

**Q4：Beam Search 为什么要长度归一化？**
每多一个 token 就多乘一个 $\log p < 0$，总得分必然更小，Beam 系统性偏好短序列、过早结束。除以 $|y|^{\alpha}$ 抵消该偏差：不归一化偏好短句，归一化过头偏好长废话。

**Q5：束宽 B 增大一定更好吗？**
不是。收益边际递减（B=4→8 提升明显，B=32→64 几乎无差），计算量翻倍，且重复问题随 B 增大更严重。B 增大只能降低错过最优的概率，不能保证找到全局最优。

**Q6：为什么对话任务不用 Beam Search？**
对话的"正确答案"不唯一，Beam 会选出最"套路"的那条（平庸），且多条 beam 高度相似、多样性差；采样才能带来自然感。翻译/摘要的得分可比较（BLEU/ROUGE），Beam 的"逼近最优"属性与评测指标契合。

**Q7：Beam Search 是全局最优搜索吗？**
不是。它是贪心剪枝的启发式近似：每步只留 top-B，可能剪掉真正的最优路径。只有 B→∞ 才精确，那是指数复杂度。它只是"更接近全局最优"。

## 八、自我检验

- [ ] 能写出 greedy 公式与 beam 的累计 log 概率打分公式
- [ ] 能手推 B=2 下 beam 的一步扩展与剪枝过程
- [ ] 能解释 greedy 局部最优问题并自造一个反例分布
- [ ] 能写出长度归一化公式 $\text{score} = \frac{1}{|y|^{\alpha}} \sum \log p$ 及 α 的三种取值含义
- [ ] 能说出 log 概率累加的两个原因
- [ ] 能写出完整的手写 beam search 循环（候选扩展 → topk 剪枝 → eos 单独存放）
- [ ] 能解释 beam 为什么重复最严重、B 越大越重复
- [ ] 知道 beam 的复杂度 $O(B \cdot |V| \cdot L)$ 与收益边际递减
- [ ] 能说清翻译/摘要用 beam、对话用采样的原因
- [ ] 能回答 7 个高频面试追问
