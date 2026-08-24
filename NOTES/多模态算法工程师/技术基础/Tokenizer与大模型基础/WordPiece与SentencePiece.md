---
learned: false
---

# WordPiece 与 SentencePiece：LM 打分与"不预分词"的路线

> 本模块索引见 [Tokenizer与大模型基础详解](Tokenizer与大模型基础详解.md)

## 一、WordPiece（BERT 的分词器）

**WordPiece 是 BERT 系模型的分词算法：与 BPE 同为"自底向上合并"，但合并的打分标准不是频率，而是"合并后让语言模型打分提升最大"。**

### 1.1 核心机制一：贪心最长匹配（编码侧）

推理（编码）时，WordPiece 对每个词做**从左到右的贪心最长匹配**：

1. 词首从完整单词开始试，词表里有就取走；没有就缩短一个字符再试；
2. 取走一个片段后，剩下的部分从**词首匹配变为"词中片段"**，候选 token 加 `##` 前缀；
3. 直到词被切完；若某处一个字符都匹配不上 → 整个词标 `<unk>`。

示例（BERT 词表）：`unhappiness → ["un", "##happi", "##ness"]`、`jumping → ["jump", "##ing"]`。

`##`（连续符号）标记语义：**"这个 token 不是词的开头"**。解码时去掉 `##` 再拼接即可还原单词，词边界信息由 `##` 保留。

### 1.2 核心机制二：合并标准是"LM 似然提升"（训练侧）

BPE 选最高频相邻对，WordPiece 选"**若把它们合并成一个新 token，语料在语言模型下的似然提升最大**"的那一对。

记语言模型对语料的对数似然为 $\log P(\text{corpus})$，合并候选对 $(a, b) \to ab$ 的分数：

$$\text{score}(a, b) = \log P_{ab}(\text{corpus}) - \log P(\text{corpus})$$

其中 $P_{ab}$ 是把 $(a,b)$ 替换成 $ab$ 后的分词结果下的似然。实践中用 unigram/bigram 计数近似（语言模型打分）：

$$\text{score}(a, b) \approx \log \frac{\text{count}(ab)}{\text{count}(a) \times \text{count}(b)} \times \text{语料大小}$$

（直觉：`count(ab)` 是合并后的收益，`count(a)·count(b)` 是"纯偶然相邻"的期望次数，比值越大越说明 $ab$ 是个真组合。）

### 1.3 与 BPE 的本质区别

| 维度 | BPE | WordPiece |
| --- | --- | --- |
| 训练打分 | 频率 $\max f(a,b)$ | 似然提升 $\max \Delta \log P$ |
| 词内标记 | `</w>`（词尾） | `##`（词首，即"非词首"） |
| 典型规模 | 32K~128K | 30K（BERT 系） |
| 编码 | 按 merges rank 重演合并 | 贪心最长匹配 |

> 重要澄清：**WordPiece 的"贪心最长匹配"是编码策略，训练仍靠迭代合并**（score 最高的对被合并，重复直到词表目标大小），很多教程把两者混为一谈。

## 二、Unigram LM：SentencePiece 的训练算法

Unigram（Kudo, 2018）走了一条**相反的路**：不从小到大合并，而是**先造一个大词表，再迭代剪枝**。

### 2.1 概率化分词：每个子词都有分数

假设词表里有 $V$ 个子词，每个子词 $x_i$ 有分数 $p(x_i)$（$\sum p(x_i) = 1$，即一个 unigram 语言模型）。一个词 $X$ 的分词不是唯一的，其概率是所有可行切分之和：

$$P(X) = \sum_{\text{seg} \in \text{SEG}(X)} \prod_{t \in \text{seg}} p(t)$$

实际分词（Viterbi）取概率最大的切分，损失函数是语料的**负对数似然**：

$$\mathcal{L} = -\sum_{X \in \text{corpus}} \log P(X)$$

### 2.2 训练流程：EM + 迭代剪枝

1. **造种子词表**：从所有可能子串（如长度 ≤ 16）里按概率筛出候选，词表远大于目标（如目标 32K，先造 100K）；
2. **E 步**：固定词表，用 EM 更新每个子词的概率 $p(t)$，使 $\mathcal{L}$ 最小（期望子词频次归一化）；
3. **M 步**：计算"若删掉子词 $t$，$\mathcal{L}$ 会增加多少"（loss 增量），把损失增量最小的子词删掉；
4. **迭代**：重复 2~3，直到词表达到目标大小。

$$\text{损失增量}(t) = \mathcal{L}_{V \setminus \{t\}} - \mathcal{L}_V, \quad \text{删掉增量最小的 } t$$

### 2.3 正则化：subword regularization（dropout）

因为分词是**概率化的**，训练时可以**随机采样**而不是取 Viterbi 最优切分（每次 forward 用不同的切分）：

$$P(\text{seg}) \propto \prod_{t \in \text{seg}} p(t) \quad (\text{以概率 } \alpha \text{ 替换最优切分})$$

效果：让模型"见过"同一文本的不同切分，对分词错误更鲁棒——这是 Unigram 相对 BPE/WordPiece 的独有优势（BPE 的切分是确定的，没有这个自由度）。

## 三、SentencePiece 框架（重点）

**SentencePiece（Kudo & Richardson, 2018）不是一种算法，而是一个"分词框架"：它把"训练数据预处理"（空格分词、归一化）也纳入流程，核心设计是两个。**

### 3.1 设计一：直接对原始文本训练，不预分词

- 传统分词器（BPE/WordPiece 的实现）第一步是**按空格预分词**——这在英文上没问题，但**日语、中文等"没有空格"的语言直接被切成整句**，预分词把语言假设硬编码进了算法；
- SentencePiece 直接吃原始文本，**把空格也当作普通字符对待**，词表统一由算法决定；
- 因此它对"无空格语言"（中日韩）天然友好，这也是多语言模型（LLaMA、T5、Gemma、LLM-jp 等）普遍选用它的原因。

### 3.2 设计二：`▁`（U+2581）标记空格

- 空格不直接出现在文本里，而是被转义成一个特殊字符 `▁`；
- 每个 token 若以 `▁` 开头，表示原文中该处**前面有一个空格**（词首标记）：
  - `"I love NLP"` → `["▁I", "▁love", "▁N", "LP"]`
  - `"我爱北京"` → `["▁我", "爱", "北京"]`
- `▁` 的作用类似 WordPiece 的 `##` 但方向相反：**`▁` 标词首，`##` 标"非词首"**；
- 解码时把 `▁` 换回空格即可无损还原，因此**不必预分词也能正确解码**。

### 3.3 为什么适合多语言 / 日语 / 中文

| 语言类型 | 空格预分词 | 问题 | SentencePiece 的做法 |
| --- | --- | --- | --- |
| 英文 | 可用 | — | 空格当普通字符，`▁` 标记 |
| 中文 | 不可用（无空格） | 整句变一个"词" | 整句内部自由切子词 |
| 日语 | 不可用（无空格） | 同上，还混三种文字 | 片假名/汉字/平假名各自成子词 |
| 混合语料 | 不可用 | 词级处理会崩 | 统一按字符/子词切分 |

补充说明：SentencePiece 内部仍可配置 `split_by_whitespace`、`normalization_rule_tsv`（NFKC 归一化，把全角→半角、复合字→分解形）、`byte_fallback`（未知字符回退为 UTF-8 字节）等参数——**框架可配置，但默认行为是"不预分词"**。

## 四、源码实现（可运行）

### 4.1 手写最小 WordPiece（纯标准库，可直接运行）

```python
from collections import Counter

class MiniWordPiece:
    """最小 WordPiece 实现（教学版）：
    真实 WordPiece 用"使 LM 似然提升最大"作为合并打分，这里简化为频率打分
    （这是文献里最常见的简化教学实现），## 前缀语义保持一致。"""

    def __init__(self, word_freq, num_merges=12):
        self.word_freq = word_freq
        chars = set()
        for w in word_freq:
            chars.add(w[0])
            for c in w[1:]:
                chars.add("##" + c)
        self.vocab = sorted(chars)
        self.merges = []
        print("初始词表（字符级 + ## 前缀）:")
        print(self.vocab)
        for step in range(num_merges):
            pairs = self._count_pairs()
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            new_tok = best[0] + best[1].replace("##", "")
            self.merges.append((best, new_tok, pairs[best]))
            self.vocab.append(new_tok)
        self.vocab.sort()

    def _segment(self, word):
        """当前词表下的贪心最长匹配切分（训练与推理共用）"""
        if word in self.vocab:
            return [word]
        toks, i = [], 0
        while i < len(word):
            found = False
            for j in range(len(word), i, -1):
                cand = word[i:j]
                pref = "##" if i > 0 else ""
                if pref + cand in self.vocab:
                    toks.append(pref + cand)
                    i = j
                    found = True
                    break
            if not found:
                toks.append("[UNK]")
                i += 1
        return toks

    def _count_pairs(self):
        """对每个词做当前切分，统计相邻 token 对频率"""
        pairs = Counter()
        for w, f in self.word_freq.items():
            toks = self._segment(w)
            for a, b in zip(toks, toks[1:]):
                pairs[(a, b)] += f
        return pairs

    def encode(self, text):
        toks = []
        for w in text.split():
            toks += self._segment(w)
        return toks

WORDS = {"happy": 10, "happiness": 8, "unhappy": 5, "unhappiness": 4,
         "the": 30, "this": 10, "that": 8, "is": 20}
wp = MiniWordPiece(WORDS, num_merges=10)

print("\n=== 合并过程（频率最高的相邻 token 对）===")
for (a, b), new_tok, freq in wp.merges:
    print(f"  {a!r} + {b!r} -> {new_tok!r}  (pair 频次 {freq})")

print(f"\n=== 最终词表（{len(wp.vocab)} 个）===")
print(wp.vocab)

print("\n=== 编码示例（贪心最长匹配）===")
for text in ["happy", "happiness", "unhappy", "the", "unhappiness", "happily"]:
    toks = wp.encode(text)
    print(f"  encode({text!r}) -> {toks}")
```

运行输出（节选）：

```
初始词表（字符级 + ## 前缀）:
['##a', '##e', '##h', '##i', '##n', '##p', '##s', '##t', '##y', 'h', 'i', 't', 'u']

=== 合并过程（频率最高的相邻 token 对）===
  't' + '##h' -> 'th'  (pair 频次 48)
  'th' + '##e' -> 'the'  (pair 频次 30)
  '##a' + '##p' -> '##ap'  (pair 频次 27)
  '##ap' + '##p' -> '##app'  (pair 频次 27)
  ...
  '##in' + '##e' -> '##ine'  (pair 频次 12)
  '##ine' + '##s' -> '##ines'  (pair 频次 12)
  '##ines' + '##s' -> '##iness'  (pair 频次 12)

=== 编码示例（贪心最长匹配）===
  encode('happy') -> ['happ', '##y']
  encode('happiness') -> ['happ', '##iness']
  encode('unhappy') -> ['u', '##n', '##h', '##app', '##y']
  encode('the') -> ['the']
  encode('unhappiness') -> ['u', '##n', '##h', '##app', '##iness']
  encode('happily') -> ['happ', '##i', '[UNK]', '##y']   # 'l' 未训练 -> [UNK]
```

观察：**词首 token 不带 `##`，词中/词尾片段一律带 `##`**——这就是 WordPiece 与 BPE（`</w>` 在词尾）最直观的差异。

### 4.2 SentencePiece 官方用法（需安装 `sentencepiece`）

> 本机未安装 sentencepiece，以下输出在临时虚拟环境（Python 3.9 + sentencepiece 0.2.2）中实际运行验证。运行前执行：`pip install sentencepiece`。

```python
import os
import sentencepiece as spm

os.makedirs("spm_demo_dir", exist_ok=True)
os.chdir("spm_demo_dir")

CORPUS = """今天天气真不错 今天下午我要去公园
我们去公园散步 天气好的时候去公园散步很舒服
Tokenization is the first step of language modeling
Large language models rely on subword tokenization"""

with open("spm_corpus.txt", "w", encoding="utf-8") as f:
    f.write(CORPUS)

def silence_fd(fd=2):
    """把指定 fd 重定向到 /dev/null（SentencePiece 的 C++ 日志直接写 stderr）"""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    def restore():
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)
    return restore

restore = silence_fd()
try:
    spm.SentencePieceTrainer.train(
        input="spm_corpus.txt",
        model_prefix="spm_demo",
        vocab_size=64,
        model_type="unigram",          # 训练算法：unigram（SentencePiece 默认）
        character_coverage=1.0,        # 全字符覆盖（小语料）
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
finally:
    restore()

sp = spm.SentencePieceProcessor(model_file="spm_demo.model")
print("词表大小:", sp.get_piece_size())
print("前 12 个 piece:", [sp.id_to_piece(i) for i in range(12)])
print("含 ▁(空格标记) 的 piece:", [sp.id_to_piece(i) for i in range(sp.get_piece_size())
                                if sp.id_to_piece(i).startswith("▁")])

for s in ["今天天气真不错", "今天下午我要去公园", "Tokenization is the first step"]:
    print(f"\nencode({s!r}) -> {sp.encode(s, out_type=str)}")
    print(f"    ids: {sp.encode(s, out_type=int)}")

toks = sp.encode("今天下午我要去公园")
print("\ndecode(ids) ->", sp.decode(toks))
print("decode 原生 token 串 ->", sp.decode(sp.encode("Tokenization is the first step",
                                                      out_type=int)))
```

运行输出：

```
词表大小: 64
前 12 个 piece: ['<pad>', '<unk>', '<s>', '</s>', '▁', 'r', 'i', 's', '天气', '▁今天', 'e', 'f']
含 ▁(空格标记) 的 piece: ['▁', '▁今天', '▁language', '▁model', '▁t']

encode('今天天气真不错') -> ['▁今天', '天气', '真', '不', '错']
    ids: [9, 8, 41, 35, 44]

encode('今天下午我要去公园') -> ['▁今天', '下', '午', '我', '要', '去公园']
    ids: [9, 34, 37, 16, 43, 38]

encode('Tokenization is the first step') -> ['▁', 'T', 'okenization', '▁', 'i',
's', '▁t', 'h', 'e', '▁', 'f', 'i', 'r', 'st', '▁', 'st', 'e', 'p']

decode(ids) -> 今天下午我要去公园
decode 原生 token 串 -> Tokenization is the first step
```

三处值得注意：

1. **中文整句直接切子词**：`"今天天气真不错"` 被切成 `▁今天 + 天气 + 真 + 不 + 错`，没有空格也能工作——这就是"不预分词"；
2. **`▁` 是词首标记**：`▁今天` 表示原文空格后的第一个字；英文里 `▁` 常单独成 token（因为训练语料空格极多）；
3. **decode 无损还原**：`▁` 换回空格，中文与英文都能精确复原。

## 五、对比总表：BPE vs WordPiece vs Unigram

| 对比项 | BPE | WordPiece | Unigram |
| --- | --- | --- | --- |
| 打分标准 | 相邻对**频率** | 合并后**LM 似然提升** | **语料负对数似然**（剪枝损失） |
| 训练方向 | 自底向上（字符→大） | 自底向上（字符→大） | 自顶向下（大词表→剪枝） |
| 词边界标记 | `</w>`（词尾） | `##`（非词首） | `▁`（词首，SentencePiece） |
| 概率化/可采样 | 否 | 否 | **是**（subword regularization） |
| 训练数据要求 | 需预分词（SP 例外） | 需预分词 | 原始文本（SP 框架） |
| 代表模型 | GPT 系、LLaMA、DeepSeek | BERT、ELECTRA | T5、Gemma、LLM-jp |
| 多语言能力 | 需 byte 级变体 | 弱 | 强 |
| 典型词表 | 32K~128K | ~30K | 32K~256K |

**一句话总结**：BPE 与 WordPiece 是同一条"自底向上"路线上的两个打分变体，Unigram 是"自顶向下剪枝"的另一条路线；SentencePiece 是一个"把空格当字符、不预分词"的框架，可承载 BPE 或 Unigram 两种算法（`model_type="bpe"` 或 `"unigram"`）。

## 六、词表大小选择的权衡

词表 $V$ 的经典选择：3 万（BERT 风格）、5 万（GPT-2/LLaMA-1 风格）、10 万+（多语言/大模型）。关键权衡：

### 6.1 显存与参数

$$\text{首尾两层参数} = 2 \times V \times d \quad (\text{Embedding + LM Head，通常权重共享})$$

| 词表 $V$ | 维度 $d$ | 首尾参数 | BF16 显存 |
| --- | --- | --- | --- |
| 30K | 768（BERT-base） | 0.46 亿 | 92 MB |
| 50K | 4096（7B 级） | 4.10 亿 | 820 MB |
| 100K | 4096 | 8.19 亿 | 1.64 GB |
| 250K | 8192（70B 级） | 40.96 亿 | 8.2 GB |

对 70B 模型（约 8TB 权重），8.2 GB 的 embedding 占比不大；但对 3B~7B 小模型，首尾两层可能占 10%~25%，词表从 32K 提到 128K 会显著涨参。

### 6.2 效果与工程

- **V 太小（<20K）**：罕见词拆得碎（"Bytedance" → 8+ 子词），序列变长、长程依赖压力大；中文场景字频高，可接受；
- **V 适中（30K~50K）**：覆盖绝大多数英文单词与中文常用字，序列长度与词表大小平衡最佳，**多数模型的选择**；
- **V 很大（100K+）**：多语言模型（每个语言分词表份额）与"大 token 词表"实验（如 50K→128K 提升代码/数学能力）。代价：embedding 显存↑、每 token 的样本更稀疏、softmax 计算量↑（LM Head 是 $T \times V$ 矩阵乘）；
- **经验法则**：英语+代码 32K~50K 够用；中英双语 50K~100K；多语言 100K~250K。**先算显存账，再拍规模**。

## 七、高频面试问答

**Q1：WordPiece 和 BPE 到底差在哪？**
打分标准：BPE 合并最高频相邻对，WordPiece 合并"使 LM 似然提升最大"的对；标记约定：BPE 用 `</w>` 标词尾，WordPiece 用 `##` 标非词首。实现上 BPE 更简单，效果相近。

**Q2：`##` 和 `▁` 有什么区别？**
`##` 表示"不是词的开头"（WordPiece，词首无标记）；`▁` 表示"词的开头"（SentencePiece，把空格转义成一个可见字符）。`▁` 还承担了"不预分词也能解码"的功能。

**Q3：Unigram 为什么可以剪枝？**
它把分词建模成概率模型：删掉一个子词只会增加语料负对数似然一点点（该子词可以被其他切分覆盖），把"损失增量最小"的子词删掉，就能在保持模型性能的同时缩小词表。

**Q4：为什么中文/日语模型偏爱 SentencePiece？**
中日文没有空格，BPE/WordPiece 依赖的"预分词"直接失效；SentencePiece 不预分词、把空格当普通字符，用 `▁` 保留边界信息，天然适配无空格语言。

**Q5：subword regularization（Unigram dropout）有什么用？**
Unigram 分词是概率化的，训练时随机采样多种切分，让模型对"同一句话的不同分词"都鲁棒；BPE/WordPiece 切分确定，没有这个机制，但实践中通过足够多的数据掩盖了差异。

**Q6：词表越大越好吗？**
不是。词表大 → embedding/softmax 显存与算力↑、每个 token 样本稀疏；词表小 → 序列变长、罕见词碎。最优值取决于语言构成、模型规模与部署显存预算，业界主流 32K~100K。

**Q7：WordPiece 编码遇到词表外的词怎么办？**
贪心最长匹配失败到单字符都不匹配时整词标 `<unk>`（BERT 是 100 个 unk 变体）。byte-level/byte_fallback 方案（GPT 系）则从根上消灭 OOV。

**Q8：SentencePiece 训练参数里哪些最影响结果？**
`vocab_size`（词表）、`model_type`（bpe/unigram）、`character_coverage`（稀有字符覆盖率）、`normalization_rule_tsv`（NFKC 等归一化）、`byte_fallback`（OOV 回退字节）、`split_digits`（数字拆分）。这些参数直接决定多语言与数学场景的表现。

## 八、自我检验

- [ ] 能说清 WordPiece 的"贪心最长匹配 + LM 打分 + ## 前缀"三件套
- [ ] 能画出 Unigram 训练流程（种子词表 → EM → 剪枝 → 迭代）
- [ ] 能解释 SentencePiece "不预分词 + ▁ 标记空格"为什么适合中日文
- [ ] 能手写 MiniWordPiece 并说出输出里每个 token 的含义
- [ ] 能说出 spm 训练的关键参数及其作用
- [ ] 能给出 BPE / WordPiece / Unigram 三者的打分标准一句话
- [ ] 能估算不同词表大小下的 embedding 显存（$V \times d \times 2$ 字节）
- [ ] 能回答第七节全部 8 个追问
