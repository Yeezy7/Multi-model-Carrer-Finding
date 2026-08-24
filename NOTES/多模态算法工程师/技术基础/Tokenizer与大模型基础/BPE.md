---
learned: false
---

# BPE：字节对编码，最经典的分词算法

> 本模块索引见 [Tokenizer与大模型基础详解](Tokenizer与大模型基础详解.md)

## 一、定义与公式

**BPE（Byte Pair Encoding，字节对编码）是一种自底向上的子词分词算法：从字符（或字节）级词表出发，反复统计相邻符号对的频率，把"出现最多的相邻对"合并成新符号，迭代直到词表达到目标大小。**

- 提出：Sennrich et al., 2016（*Neural Machine Translation of Rare Words with Subword Units*），最初用于机器翻译；
- 地位：GPT-2 / GPT-3 / GPT-4、LLaMA 系、DeepSeek 系等绝大多数现代 LLM 的分词底座（GPT 系为 byte-level 变体，见第四节）；
- 核心思想一句话：**高频相邻对合并成一个"子词"**，让模型在"词表足够小"和"序列足够短"之间取平衡。

### 1.1 算法流程（5 步）

设语料 $C$、目标合并次数 $K$、词尾标记 `</w>`：

1. **预分词**：按空格/标点把语料切成词（word 级单位）；
2. **初始化词表**：每个词拆成字符序列，词尾加 `</w>` 标记（区分"low"与"lower"的边界）；
3. **统计相邻对频率**：对语料中所有词的相邻字符对计数：
   $$f(a, b) = \sum_{w \in C} \text{count}(w) \cdot \mathbb{1}[(a, b) \text{ 相邻出现在 } w \text{ 中}]$$
4. **合并**：取频率最高的对 $(a, b)$，把语料中所有 $ab$ 合并为 $ab$，新符号加入词表，$(a,b)$ 记入 merges 表；
5. **迭代**：重复 3~4 共 $K$ 次，或直到词表达到预设大小 $V$。

> **同频平局怎么办？** 真实实现里用确定性的字典序/编码序打破平局，保证同语料同词表可复现。

### 1.2 逐步示例：用 "low / lower / lowest" 完整跑一遍

语料只有三个词（各出现 1 次）。初始词表：`l, o, w, e, r, s, t, </w>`。

**第 0 步**统计相邻对频率：

| 相邻对 | 频率 | 出现在 |
| --- | --- | --- |
| (l, o) | 3 | low / lower / lowest |
| (o, w) | 3 | low / lower / lowest |
| (w, e) | 2 | lower / lowest |
| (w, </w>) | 1 | low |
| (e, r) | 1 | lower |
| (r, </w>) | 1 | lower |
| (e, s) | 1 | lowest |
| (s, t) | 1 | lowest |
| (t, </w>) | 1 | lowest |

平局由顺序打破，得到前 8 步合并序列（与第三节代码输出一致）：

```
step 0: (l,o) -> lo      step 4: (lowe,r) -> lower
step 1: (lo,w) -> low    step 5: (lower,</w>) -> lower</w>
step 2: (low,e) -> lowe  step 6: (lowe,s) -> lowes
step 3: (low,</w>) -> low</w>   step 7: (lowes,t) -> lowest
```

观察三个现象：

1. **先出现的是全局高频对**：`(l,o)`、`(o,w)` 出现在全部 3 个词里，排最前；`(e,r)` 只在 lower 里出现 1 次，排后；
2. **合并是可递归的**：`lo` 是第 0 步产物，第 1 步继续合并 `(lo,w)` 得 `low`，第 2 步合并 `(low,e)` 得 `lowe`——**"子词像搭积木"**；
3. **词尾标记防止过度合并**：若没有 `</w>`，`low` 和 `lower` 的 `l-o-w` 会被同一个 `low` 覆盖，词边界信息丢失。`(low,</w>)` 出现在第 3 步，形成独立 token `low</w>`。

合并 8 步后每个词的表示：

| 词 | 最终切分 |
| --- | --- |
| low | `[low</w>]` |
| lower | `[lower</w>]` |
| lowest | `[lowest, </w>]` |

## 二、数学/算法细节

### 2.1 merges 表：BPE 的"训练产物"

BPE 训练产出不是词表本身，而是一个**有序合并规则表**：

$$\text{merges} = [(a_1, b_1), (a_2, b_2), \dots, (a_K, b_K)]$$

每条规则 $(a, b) \to ab$ 的**先后顺序（rank）就是优先级**。合并规则 + 初始字符集，两者合起来定义了完整的 tokenizer：

$$\text{vocab} = \{\text{基础字符}\} \cup \{a_ib_i : (a_i, b_i) \in \text{merges}\}$$

注意：**即使某个合并产物后来被更大的 token 吞并（如 `lo` 被 `low` 覆盖），它仍然留在词表里**——因为编码时"low"这个词可能在 `l-o-w` 处恰好停在 `lo` 这一步。所以词表大小 = 基础符号数 + 合并次数 $K$（GPT-2：256 字节 + ~50K 次合并 ≈ 50257）。

### 2.2 编码过程（推理时如何切词）

给定文本和 merges 表，编码是从**单个字符**重新走一遍合并过程，但只在"规则允许"的路径上走：

1. 文本按空格预分词 → 字符序列 + `</w>`；
2. 从左到右扫描，找出所有**能被合并规则命中的相邻对**中 rank 最小的一对，合并；
3. 重复直到没有可合并的相邻对；
4. 每个片段查词表得 id。

**为什么是"rank 最小"？** 因为 rank 表示该规则在训练时被应用的顺序——训练时语料里每一处该对的合并都发生过，编码时按同样顺序合并，就能得到与训练一致的切分。

### 2.3 解码过程（为什么解码是唯一的）

解码极其简单：把 token 序列**直接拼接**，`</w>` 替换为空格：

$$\text{text} = \text{concat}(t_1, t_2, \dots, t_n), \quad </\text{w}> \to \text{空格}$$

**解码唯一性**的来源：每个 token 都是"一个字符串"，拼接本身没有歧义；`</w>` 是词边界标记、绝不会在正常文本中出现，替换后原文无损恢复。相比之下编码是"多对一"的（多种切分可能对应同一文本，但 merges 表保证只有一种会被选中）——**训练时怎么合并、推理时就怎么合并，这是一对一可逆的**。

### 2.4 罕见词（OOV）的 fallback

- **字符级 BPE**：词表外字符 → `<unk>` token（信息丢失，GPT 之前的 NMT 做法）；
- **Byte-level BPE**：任何字符都能 UTF-8 编码成字节，而字节天然在词表里 → **OOV 被彻底消灭**（见第四节）；
- 罕见词的典型命运：被拆成"常见前缀 + 若干单字符"，如 `lowly` → `low` + `l` + `y`（演示见第三节输出：`lowers` → `lower + s + </w>`）。

## 三、源码实现（可运行）

纯 Python 实现完整 BPE 训练 + 编码 + 解码，小语料逐步演示。**直接运行本文件对应代码即可复现全部输出**（Python 3.9+，无第三方依赖）：

```python
from collections import Counter

CORPUS = ["low", "lower", "lowest"]
EOW = "</w>"

def get_stats(vocab):
    """统计 vocab 中所有相邻符号对的频率。vocab: {符号元组: 词频}"""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """把 vocab 中所有 pair (a,b) 合并为 a+b"""
    a, b = pair
    new_sym = a + b
    out = {}
    for word, freq in vocab.items():
        i, merged = 0, []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(new_sym)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        out[tuple(merged)] = freq
    return out

def train_bpe(corpus, num_merges):
    """训练 BPE：返回 merges 表（合并顺序）+ 每步的词表示快照"""
    vocab = Counter()
    for w in corpus:
        vocab[tuple(list(w) + [EOW])] += 1
    merges, snapshots = [], [("初始", dict(vocab))]
    for step in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: pairs[p])
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        snapshots.append((f"step {step}", dict(vocab)))
    return merges, snapshots

def encode(text, merges):
    """编码：按 merges 表的先后顺序贪心合并（rank 越小越优先）"""
    rank = {p: i for i, p in enumerate(merges)}
    syms = tuple(list(text) + [EOW])
    while len(syms) > 1:
        cand = [(rank[(a, b)], (a, b)) for a, b in zip(syms, syms[1:])
                if (a, b) in rank]
        if not cand:
            break
        _, best = min(cand)
        new, i = [], 0
        while i < len(syms):
            if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                new.append(best[0] + best[1])
                i += 2
            else:
                new.append(syms[i])
                i += 1
        syms = new
    return syms

def decode(ids, id2sym, eow=EOW):
    """解码：唯一。逐 token 拼接，</w> 换成空格即可恢复原文"""
    text = "".join(id2sym[i] for i in ids)
    return text.replace(eow, " ").strip()

merges, snapshots = train_bpe(CORPUS, num_merges=8)
print("=== 第 0 步：字符级词表与相邻对频率 ===")
pairs0 = get_stats(snapshots[0][1])
for word, freq in snapshots[0][1].items():
    print(f"{freq}x {''.join(word[:-1]):7s} 符号: {list(word)}")
for pair, freq in pairs0.most_common():
    print(f"  pair {pair} : {freq}")

print("\n=== 逐步合并（每步选最高频相邻对）===")
for i, (name, vocab) in enumerate(snapshots):
    if i == 0:
        continue
    a, b = merges[i - 1]
    segs = ["[" + ", ".join(w) + "]" for w in vocab]
    print(f"{name}: 合并 ({a!r},{b!r}) -> {a + b!r}")
    print(f"       词表示: {' | '.join(segs)}")

final_vocab = snapshots[-1][1]
# 与真实实现一致：最终词表 = 基础字符 + </w> + 所有合并产物（即使后来被更大的词吞并）
symbols = sorted(set("lowestr") | {EOW} | {a + b for a, b in merges})
sym2id = {s: i for i, s in enumerate(symbols)}
id2sym = {i: s for s, i in sym2id.items()}
print(f"\n最终符号表（{len(symbols)} 个，含 {EOW}）: {symbols}")

print("\n=== 编码 / 解码 ===")
for text in ["low", "lower", "lowest", "lowers"]:
    ids = [sym2id[s] for s in encode(text, merges)]
    toks = [id2sym[i] for i in ids]
    print(f"encode({text!r}) -> {toks} -> ids {ids} -> decode {decode(ids, id2sym)!r}")

oov = "lowly"  # 'y' 不在训练词表 -> OOV，真实实现回落为 <unk> 或字节级编码
try:
    ids = [sym2id[s] for s in encode(oov, merges)]
    print(f"encode({oov!r}) -> ids {ids}")
except KeyError as e:
    print(f"encode({oov!r}) -> OOV 字符 {e} 不在词表 -> 回落 <unk>（或 byte-level 按字节编码）")
```

运行输出（节选关键几行）：

```
=== 第 0 步：字符级词表与相邻对频率 ===
1x low     符号: ['l', 'o', 'w', '</w>']
1x lower   符号: ['l', 'o', 'w', 'e', 'r', '</w>']
1x lowest  符号: ['l', 'o', 'w', 'e', 's', 't', '</w>']
  pair ('l', 'o') : 3
  pair ('o', 'w') : 3
  pair ('w', 'e') : 2
  ...
step 3: 合并 ('low','</w>') -> 'low</w>'
       词表示: [low</w>] | [lowe, r, </w>] | [lowe, s, t, </w>]
step 4: 合并 ('lowe','r') -> 'lower'
       词表示: [low</w>] | [lower, </w>] | [lowe, s, t, </w>]
  ...
最终符号表（16 个，含 </w>）: ['</w>', 'e', 'l', 'lo', 'low', 'low</w>', 'lowe',
'lower', 'lower</w>', 'lowes', 'lowest', 'o', 'r', 's', 't', 'w']

=== 编码 / 解码 ===
encode('low') -> ['low</w>'] -> ids [5] -> decode 'low'
encode('lower') -> ['lower</w>'] -> ids [8] -> decode 'lower'
encode('lowest') -> ['lowest', '</w>'] -> ids [10, 0] -> decode 'lowest'
encode('lowers') -> ['lower', 's', '</w>'] -> ids [7, 13, 0] -> decode 'lowers'
encode('lowly') -> OOV 字符 'y' 不在词表 -> 回落 <unk>（或 byte-level 按字节编码）
```

（完整输出含 8 步合并全过程与词表示快照，请直接运行验证。）

## 四、Byte-level BPE：GPT-2 的解决方案

### 4.1 字符级 BPE 的三个痛点

1. **多语言**：字符集随语言变化，拉丁字母 + 中文 + 阿拉伯文混合时，字符级词表必然膨胀；
2. **罕见 Unicode**：emoji、生僻字、数学符号——语料里出现但频率极低，合并学不好；
3. **OOV 无法根治**：任何字符级词表都覆盖不了"未来可能出现的新字符"。

### 4.2 思路：先编码成字节，再在字节上做 BPE

- 任何 Unicode 文本先用 **UTF-8** 编码成字节序列（1 个汉字 3 字节、1 个 emoji 4 字节）；
- 字节只有 **256 个可能取值**，作为初始词表完美覆盖一切文本（见第三节输出的字节示例）；
- 在字节序列上跑标准 BPE 合并，词表 = 256 字节 + 合并产物（GPT-2：50257）。

$$\text{文本} \xrightarrow{\text{UTF-8}} \text{字节序列} \xrightarrow{\text{BPE}} \text{子词 token}$$

```python
emoji = "你好 👋"
print(list(emoji.encode('utf-8')))
# [228, 189, 160, 229, 165, 189, 32, 240, 159, 145, 139]  (11 字节)
print(len(set(emoji.encode('utf-8'))))  # 10 个不同字节 -> 基础词表只需 256
```

### 4.3 收益与代价

| 维度 | 字符级 BPE | Byte-level BPE（GPT-2 起） |
| --- | --- | --- |
| OOV | 存在（→ `<unk>`） | **彻底消灭** |
| 多语言/emoji | 需预收集字符集 | 天然支持 |
| 词表 | 30K~100K | 256 + merges，通常 50K~150K |
| 代价 | — | **每个 token 不是完整字符**：单字节 token 的 embedding 语义弱；序列平均更长（中文 1 字 ≈ 1.5~3 字节）；"hello" 可能被拆成 "hell"+"o" 导致合并语义残损 |

GPT-2 论文（Radford et al., 2019）给出的选择理由：**"字节级 BPE 在不牺牲效率的前提下，覆盖了所有 Unicode 文本"**。代价则是 tokenizer 需要把 UTF-8 字节与 Unicode 字符做双向映射（`bytes_to_unicode` 表），保证解码正确。

## 五、优缺点

| 优点 | 缺点 |
| --- | --- |
| 语言无关：无需词典/分词器，任意文本可训练 | 合并规则是**贪心**的：每步只看局部最高频，全局最优不保证 |
| 词表大小完全可控（改 $K$ 即可） | 词表固定：训练完就定死，新增词汇需重训或扩展 |
| 子词天然表达词形变化（run/runs/running 共享前缀） | 同频平局靠任意顺序打破，不同语料可能得到不同词表 |
| 实现简单、可复现（纯计数 + 合并） | 频率 ≠ 语义：高频的"of the"也可能被合并成无意义片段 |
| 解码无损可逆 | 对空格等字符不做归一化，预处理责任交给调用方 |

## 六、与 WordPiece / SentencePiece 对比

| 对比项 | BPE | WordPiece（BERT） | Unigram（SentencePiece 默认） |
| --- | --- | --- | --- |
| 合并/选择标准 | **最高频相邻对** | **使 LM 似然提升最大**的对 | **剪枝后语料损失最小**的子词 |
| 训练方向 | 自底向上（字符→大） | 自底向上（字符→大） | 自顶向下（大词表→小） |
| 词内标记 | `</w>` 词尾 | `##` 词首 | `▁` 词首空格 |
| 是否可概率化 | 否（确定性切分） | 否 | **是**（每个子词有分数，可采样） |
| 代表模型 | GPT 系、LLaMA、DeepSeek | BERT、ELECTRA | T5、Gemma、LLaMA（SP 框架） |
| 多语言友好度 | 需 byte 级变体 | 弱（英文优先） | 强（原始文本训练 + ▁） |

三者的详细推导、打分公式与实现见 [WordPiece与SentencePiece.md](WordPiece与SentencePiece.md) 第一、二、五节。

## 七、高频面试问答

**Q1：BPE 是怎么工作的？**
三步：字符初始化 → 统计相邻对频率 → 合并最高频对并记入 merges 表，迭代 $K$ 次。编码时从字符出发按 rank 顺序贪心合并，解码时直接拼接。

**Q2：为什么不直接按字符分词？**
字符粒度信息密度低、序列过长，Transformer 注意力 $O(T^2)$ 成本暴涨，且单字符缺乏语义；整词级则词表爆炸（数十万）、OOV 严重。BPE 的子词粒度是两者的折中。

**Q3：Byte-level BPE 的好处是什么？**
UTF-8 字节只有 256 种，天然覆盖全部 Unicode（含 emoji、生僻字、多语言），OOV 从根上消灭；代价是 token 可能"切在字符中间"，序列变长。

**Q4：BPE 和 WordPiece 的区别？**
BPE 选最高频相邻对；WordPiece 选"合并后语言模型打分提升最大"的对，更语义化。实际效果接近，WordPiece 的 `##` 前缀约定让英文子词形态更规范。

**Q5：同频的相邻对怎么处理？**
用确定性规则（如字典序）打破平局，保证同语料可复现。这也是为什么不同实现的分词结果可能略有差异。

**Q6：为什么 merges 表比词表本身更重要？**
merges 表（有序合并规则）完整定义了编码逻辑——词表只是它的衍生物（基础符号 + 合并产物）。保存 tokenizer 时 merges 必须随 vocab 一起存。

**Q7：解码为什么是唯一的？**
token 是纯字符串，拼接无歧义；`</w>` 是保留词边界标记，替换为空格后原文无损恢复。BPE 是可逆编码。

**Q8：中文用 BPE 效果如何？**
中文按字/词切分差异小，BPE 收益不如英文；但中文+英文混排语料上，BPE 能自然学出"汉字对"与"英文词根"，是主流中文模型（结合 SentencePiece）的选择。

## 八、自我检验

- [ ] 能徒手写出 BPE 五步算法流程
- [ ] 能手工跑一遍 "low/lower/lowest" 的合并过程并解释每步选择
- [ ] 能说清 merges 表、vocab、encode、decode 四者的关系
- [ ] 能解释"解码为什么唯一"与"编码为什么贪心"
- [ ] 能说出 GPT-2 用 byte-level 的三个动机与两个代价
- [ ] 能讲清 BPE vs WordPiece vs Unigram 的打分标准差异
- [ ] 能回答第八节全部 8 个追问
