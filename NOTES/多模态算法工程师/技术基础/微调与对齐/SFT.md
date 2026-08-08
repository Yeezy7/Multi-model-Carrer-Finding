# SFT 监督微调

> 本模块索引见 [微调与对齐详解](微调与对齐详解.md)

## 一、定义与公式

### 1.1 什么是 SFT

SFT（Supervised Fine-Tuning，监督微调）：用**人工标注的"指令-回答"对**，在预训练模型上继续做**监督式的序列生成训练**，让模型学会"给定指令 → 输出对应回答"的行为。

预训练模型只会续写，SFT 是让模型学会"对话"的第一步，也是 RLHF/DPO 的公共前提。

### 1.2 目标函数：只对回答部分算交叉熵

模型按自回归方式生成，第 $t$ 个位置的 logits 预测第 $t+1$ 个 token。记模型参数为 $\theta$，输入序列 $x_{1:T} = [\text{system}, \text{user}, \text{assistant 回答}]$，SFT 的损失为：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \log P_\theta(x_{i+1} \mid x_{1:i})$$

其中 $\mathcal{A}$ 是**回答区间的 token 位置集合**（回答文本 + 结尾符），$|\mathcal{A}|$ 是回答区间的 token 数。非回答位置（system/user/格式 token）**完全不参与损失**。

### 1.3 mask 公式的推导

**第 1 步**：全序列的普通语言模型（LM）损失（预训练目标）：

$$\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{T-1}\sum_{i=1}^{T-1} \log P_\theta(x_{i+1} \mid x_{1:i})$$

**第 2 步**：引入二值 mask $m_i$（$m_i = 1$ 当且仅当 $x_{i+1}$ 是回答区间的 token），把分母从"全部位置"改成"被 mask 的位置数"：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_{i} m_i}\sum_{i=1}^{T-1} m_i \cdot \log P_\theta(x_{i+1} \mid x_{1:i})$$

**第 3 步（等价实现）**：把非回答位置的标签替换为 `ignore_index = -100`。交叉熵损失对 `-100` 的位置自动忽略（既不计入分母、梯度也为 0），因此：

```text
labels[i] = x[i]          当 i 属于回答区间
labels[i] = -100          当 i 属于 prompt/格式区间
```

**第 4 步（shift 一位）**：位置 $i$ 的 logits 预测的是 $x_{i+1}$，所以实现时 `logits = logits[:, :-1]`、`labels = labels[:, 1:]` 对齐后计算。mask 与 shift 的顺序无影响，但**必须同时偏移**。

### 1.4 损失加权恒等式（理解 mask 的关键）

设 $n = n_p + n_a$（全部有效位置 = prompt 区 + 回答区），$\mathcal{L}_p, \mathcal{L}_a$ 分别为两区各自的平均损失，则全序列损失是两区的加权平均；SFT 的"回答-only"模式等价于**把 $\mathcal{L}_p$ 项删掉**（3.1 节代码数值验证）：

$$\mathcal{L}_{\text{full}} = \frac{n_p}{n_p + n_a}\mathcal{L}_p + \frac{n_a}{n_p + n_a}\mathcal{L}_a$$


## 二、核心原理

### 2.1 为什么 SFT 有效：知识早已在预训练里

SFT 数据的量级（几千~几十万条）远不足以教会模型新知识。SFT 有效的前提是：

- **知识已经存在于预训练权重中**（世界知识、语言能力都在预训练阶段习得）；
- SFT 做的是**"重新布线"（re-routing）**：把预训练阶段"按前缀续写"的调用方式，改成"按指令回答"的调用方式；
- 直观类比：预训练是一个内容齐全但不会查资料的图书馆，SFT 教的是"用户提问 → 去对应书架上取书"的检索动作，而不是往图书馆里放新书。

实验佐证：只让模型在 SFT 数据上训练，回答的**格式**和**调用模式**学得很快（几个 epoch 内），但回答的**知识密度**取决于预训练底子。

### 2.2 指令遵循从哪来：格式 + 行为克隆

InstructGPT 论文（Ouyang et al., 2022）揭示：仅用 **1.3k 条人工演示数据**做 SFT，模型就具备了基本的指令遵循能力。这说明：

- 指令遵循本质是**行为克隆**（behavioral cloning）：模型模仿"人类在给定指令时应该输出什么"；
- 指令遵循的主干是**格式学习**：`<用户指令> → <回答>` 的映射模式在大量数据中反复出现，模型快速学会"看到问题就回答，而不是续写"；
- SFT 提升的是"调用分布"（哪些知识被调用），几乎不改变"知识本身"——所以 SFT 之后的知识评测（MMLU 等）通常只小幅波动。

### 2.3 为什么 SFT 通常只要 1~3 个 epoch

| 原因 | 解释 |
| --- | --- |
| 数据量小 | SFT 数据通常 1k~100k 条，一个 epoch 已经遍历完，不需要多轮 |
| 过拟合风险 | 多 epoch 会让模型**背诵**训练回答、丧失泛化；回答风格僵化，且窄分布下加速遗忘预训练能力 |
| 知识不在 SFT 里 | SFT 不教知识，反复看同样几条数据不会增加知识 |
| 经验数据 | LIMA/InstructGPT 均报告 1~3 epoch 为最佳区间，更多 epoch 收益递减甚至变差 |

> 与预训练对比：预训练要几万亿 token 数十个 epoch 等价（数据重复率低）；SFT 是"微调"，数据小且高度重复，训练信号饱和极快。

### 2.4 三个关键认知

1. **SFT 学的是"行为分布"而非"新知识"**：别指望用 SFT 教模型新事实；
2. **SFT 的输出是 RLHF/DPO 的起点**：不做 SFT 直接做偏好优化，模型连"问答格式"都没有，RLHF/DPO 无从谈起；
3. **SFT 质量的上限由数据决定**：数据里没有的指令模式，SFT 学不出来（泛化靠预训练）。

## 三、源码实现

### 3.1 完整可运行：迷你因果 LM 演示 loss mask 计算

下面的代码用一个小因果语言模型（2 层 Transformer）演示：① ChatML 风格的"多模态"问答数据如何构造；② `build_labels` 如何把 prompt 区置 `-100`；③ 全序列 loss 与回答-only loss 的差异（含损失恒等式验证）；④ 用两种 loss 各训一个模型，对比"prompt 区 loss"——直观展示全序列训练会把容量浪费在"预测用户问题"上。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------- 1. 迷你因果语言模型（2 层 Transformer，仅作演示） ----------

class TinyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, attn_mask):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        att = att.masked_fill(attn_mask == 0, float("-inf"))  # 因果 + padding mask
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)
class TinyBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos_emb, std=0.02)
    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :T]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
        if attn_mask is not None:
            causal = causal[None, None, :, :] & attn_mask.unsqueeze(1).unsqueeze(2)
        for blk in self.blocks:
            x = blk(x, causal)
        return self.lm_head(self.ln_f(x))
def shifted_ce(logits, labels, ignore_index=-100):
    """SFT 标准损失：logits 第 t 位预测 labels 第 t+1 位；-100 位置不参与"""
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)
# ---------- 2. 多模态 SFT 样本构造（ChatML 风格，<image> 为图像占位符） ----------

IM_START, IM_END, IMG, PAD = 0, 1, 2, 3
ROLE2ID = {"system": 4, "user": 5, "assistant": 6}
TEXTS = ["You are a helpful assistant.", "What color is the cat?", "The cat is orange.",
         "How many apples are there?", "There are three apples."]
CHAR2ID = {c: 10 + i for i, c in enumerate(sorted(set("".join(TEXTS))))}
VOCAB_SIZE = 10 + len(CHAR2ID)
def build_sft_sequence(sample):
    """sample: [(role, text), ...] → token id 序列（user 消息前插入图像占位符）"""
    ids = []
    for role, text in sample:
        ids += [IM_START, ROLE2ID[role]]
        if role == "user":
            ids.append(IMG)
        ids += [CHAR2ID[c] for c in text]
        ids.append(IM_END)
    return ids
def build_labels(input_ids, ignore_index=-100):
    """核心函数：只把 assistant 回答部分的 token 留作标签，其余置 -100"""
    labels = torch.full_like(input_ids, ignore_index)
    in_answer = False
    i = 0
    while i < input_ids.size(0):
        t = input_ids[i].item()
        if t == IM_START:
            in_answer = (i + 1 < input_ids.size(0)
                         and input_ids[i + 1].item() == ROLE2ID["assistant"])
            i += 2          # 跳过 <|im_start|> 与角色名
            continue
        if t == IM_END:
            in_answer = False
            i += 1
            continue
        if in_answer:
            labels[i] = t
        i += 1
    return labels
def build_prompt_labels(input_ids, ignore_index=-100):
    """与 build_labels 互补：只保留 prompt 区（system/user）为标签，回答区置 -100"""
    labels = input_ids.clone()
    in_answer = False
    i = 0
    while i < input_ids.size(0):
        t = input_ids[i].item()
        if t == IM_START:
            in_answer = (i + 1 < input_ids.size(0)
                         and input_ids[i + 1].item() == ROLE2ID["assistant"])
            i += 2
            continue
        if t == IM_END:
            in_answer = False
            i += 1
            continue
        if in_answer:
            labels[i] = ignore_index
        i += 1
    return labels
# 两条"多模态"问答样本
samples = [
    build_sft_sequence([("system", "You are a helpful assistant."),
                        ("user", "What color is the cat?"),
                        ("assistant", "The cat is orange.")]),
    build_sft_sequence([("system", "You are a helpful assistant."),
                        ("user", "How many apples are there?"),
                        ("assistant", "There are three apples.")]),
]
max_len = max(len(s) for s in samples)
input_ids = torch.full((len(samples), max_len), PAD, dtype=torch.long)
for i, s in enumerate(samples):
    input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
attn_mask = (input_ids != PAD).long()
labels = torch.stack([build_labels(row) for row in input_ids])
rev = {v: k for k, v in CHAR2ID.items()}
ans = [(i, t) for i, t in enumerate(labels[0].tolist()) if t != -100]
print("回答区间的 token 位置:", [p for p, _ in ans])
# 输出: 回答区间的 token 位置: [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
print("被 mask 保留的文本:", "".join(rev[t] for _, t in ans))
# 输出: 被 mask 保留的文本: The cat is orange.
# ---------- 3. 全序列 loss vs 回答-only loss ----------

torch.manual_seed(0)
model = TinyCausalLM(VOCAB_SIZE)
with torch.no_grad():
    logits = model(input_ids, attn_mask)
loss_full = shifted_ce(logits, input_ids)   # 全序列模式：labels = input_ids（预训练式）
loss_answer = shifted_ce(logits, labels)    # 回答-only 模式：真正的 SFT
n_full = input_ids[:, 1:].numel()
n_answer = (labels[:, 1:] != -100).sum().item()
print(f"全序列 loss: {loss_full:.4f}   参与 token 数: {n_full}")
# 输出: 全序列 loss: 3.7914   参与 token 数: 172
print(f"回答-only loss: {loss_answer:.4f}   参与 token 数: {n_answer}")
# 输出: 回答-only loss: 3.5870   参与 token 数: 41
print(f"差异: 全序列额外优化 {n_full - n_answer} 个 prompt/格式/填充 token 的预测")
# 输出: 差异: 全序列额外优化 131 个 prompt/格式/填充 token 的预测
# 损失恒等式验证：全序列 = 两区损失按 token 数加权平均（见 1.4 节公式）
prompt_labels = torch.stack([build_prompt_labels(r) for r in input_ids])
n_prompt = (prompt_labels[:, 1:] != -100).sum().item()
l_prompt = shifted_ce(logits, prompt_labels).item()
l_answer = shifted_ce(logits, labels).item()
print(f"恒等式: ({n_prompt}*{l_prompt:.4f} + {n_answer}*{l_answer:.4f}) / {n_full} = "
      f"{(n_prompt * l_prompt + n_answer * l_answer) / n_full:.4f} ≈ 全序列 {loss_full:.4f}")
# 输出: 恒等式: (131*3.8554 + 41*3.5870) / 172 = 3.7914 ≈ 全序列 3.7914
# ---------- 4. 两种 loss 各训一个模型，看"prompt 区 loss" ----------

def train(model, labels_, steps=60, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(input_ids, attn_mask)
        loss = shifted_ce(logits, labels_)
        loss.backward()
        opt.step()
torch.manual_seed(0)
m_full = TinyCausalLM(VOCAB_SIZE)   # 全序列 loss 训练
torch.manual_seed(0)
m_mask = TinyCausalLM(VOCAB_SIZE)   # 回答-only loss 训练（初始化完全相同）
train(m_full, input_ids)
train(m_mask, labels)
with torch.no_grad():
    lf = shifted_ce(m_full(input_ids, attn_mask), prompt_labels)
    lm = shifted_ce(m_mask(input_ids, attn_mask), prompt_labels)
print(f"训练后 prompt 区 loss | 全序列训练: {lf:.4f} | 回答-only 训练: {lm:.4f}")
# 输出: 训练后 prompt 区 loss | 全序列训练: 0.0235 | 回答-only 训练: 5.3593
```

**实验结论（代码输出直接说明）**：
- 全序列训练把 prompt 区 loss 压到 **0.024**——模型把容量浪费在"预测用户的提问"上（预训练式的无意义技能）；
- 回答-only 训练 prompt 区 loss 仍是 **5.36**——模型完全不管 prompt 区的预测，只优化回答；
- 两种模式在回答能力上等效（回答区 loss 都在下降），但全序列模式**额外消耗了模型容量与过拟合风险**。

> 注意：回答-only 模式下 prompt token 仍会通过注意力回流获得小梯度（回答要"看"问题），但"预测问题下一个词"这一直接损失项为 0——这正是 mask 的作用边界。

### 3.2 多模态 SFT 数据的 ChatML 格式示例

真实的 ChatML（如 LLaVA / Qwen 系）文本模板：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
What is in this picture?<|im_end|>
<|im_start|>assistant
There is a black cat sitting on the sofa.<|im_end|>
```

要点：
- `<image>` 是**特殊 token**（词表里注册的特殊符号），对应"整张图"；高分辨率模型会展开成多个占位符，其 label 必须置 `-100`（模型不可能"预测"图片）；
- system 可省（不需要时只保留 user/assistant 两轮）；
- 多轮对话 = 多组 `user → assistant` 交替，回答区全部保留。

### 3.3 真实框架里的等价实现

`build_labels` 在工程上无需手写——HuggingFace 生态已内置：

```python
from transformers import DataCollatorForSeq2Seq
collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
from trl import SFTTrainer   # 自动处理 label mask
trainer = SFTTrainer(model=model, train_dataset=dataset, dataset_text_field="text")
```

> 面试点：mask 实现 = label 置 `-100` + `cross_entropy(ignore_index=-100)`；框架里 `DataCollatorForSeq2Seq` / `SFTTrainer` 已内置。

## 四、数据与格式

### 4.1 三角色结构：system / user / assistant

| 角色 | 作用 | 注意事项 |
| --- | --- | --- |
| system | 全局设定（角色、风格、约束） | 多任务数据集里 system 会变，注意别让模型把 system 当回答 |
| user | 用户指令/问题 | 内容可含图像占位符（多模态） |
| assistant | 期望的回答 | **唯一参与损失的角色** |

格式统一是 SFT 的隐形工程质量：ChatML（`<|im_start|>`）是当前事实标准，训练与推理必须用**同一个模板**。

### 4.2 为什么只算回答 loss

1. **训练-推理对齐**：推理时 prompt 是"输入"而非"要生成的内容"，损失目标应与之一致；
2. **防止浪费容量**：全序列 loss 会把大量梯度花在"预测用户问题"上（见 3.1 实验：prompt 区 loss 被压到 0.024）；
3. **防格式过拟合与噪声**：格式 token 参与损失会"背模板"；不同样本的 system/user 内容千差万别，硬预测等于引入噪声目标。

### 4.3 数据质量 > 数量（LIMA 的 1000 条结论）

LIMA（Meta, 2023）论文：仅用 **1000 条精心挑选的高质量 SFT 数据**，微调后的模型在多数评测上与 GPT-4 相当或接近。核心结论：

- **指令遵循是"对齐的分布外泛化"**：预训练已提供全部能力，SFT 只需"最少量、最高质量"的演示指明方向（前提是底模足够强）；
- 质量维度：指令多样、回答正确完整、**风格像人类标注者**（不是模型生成的流水账）；数据量 1k~100k 区间内，**质量增益大于翻倍数量**。

### 4.4 数据配比（经验值）

| 数据类别 | 比例建议 | 作用 |
| --- | --- | --- |
| 通用指令问答 | 主体（60%~80%） | 学会通用指令遵循（领域任务数据按需混入） |
| 多轮对话 | 10%~20% | 学会上下文对话与指代 |
| 纯文本/通用语料（回放） | 10% 左右 | 防灾难性遗忘 |
| 拒答/安全样本 | 少量 | 学会拒绝 |

> 面试记忆点：**通用数据打底 + 少量领域数据 + 少量回放数据**，比例按遗忘监控动态调。

## 五、多模态 SFT 特有问题

### 5.1 视觉塔冻结

LLaVA 式的多模态模型 = 视觉塔（CLIP ViT）+ 投影层（Projector）+ LLM。SFT 阶段惯例：

| 方案 | 冻结什么 | 何时用 |
| --- | --- | --- |
| 冻结视觉塔 + 训练投影层与 LLM | ViT 全冻 | 默认方案（LLaVA 第一阶段预训练投影层后冻结） |
| 视觉塔加 LoRA | ViT 只训 LoRA 分支 | 视觉域差异大（医学/文档/遥感） |
| 解冻视觉塔全量 | 不冻 | 极少用（显存大、易遗忘 CLIP 能力） |

冻结原因：① CLIP 表征已在海量图文对上预训练好，SFT 数据量远不足以重训；② 省显存省计算；③ 防灾难性遗忘（视觉通用表征退化）。

### 5.2 图文数据配比

| 数据类型 | 示例 | 作用 | 建议比例 |
| --- | --- | --- | --- |
| 图像描述（captioning） | "一张图 → 描述文字" | 教"看图说话"，防止无视图像 | 1 份 |
| 指令问答（instruction QA） | "图里有几个苹果？" | 教"按指令用图" | 5~10 份 |
| 纯文本指令 | 无图问答 | 防语言能力退化 | 10%~20% 额外混入 |

> 经验：caption 太少 → 模型"不看图直接答"；caption 太多 → 回答啰嗦、指令跟随弱。LLaVA 系常用 ~1:5~1:10 的问答/描述比。

### 5.3 图像描述 vs 指令问答

| 维度 | 图像描述 | 指令问答 |
| --- | --- | --- |
| 输出 | 一段完整描述（自由形式） | 针对具体问题的答案（限定） |
| 学到的行为 | 看图 → 全面叙述 | 看图 + 读问题 → 提取所需信息 |
| 数据成本与风险 | 低（caption 数据集直接来）；多则回答冗长、指令跟随弱 | 高（需人工/强模型构造）；多则忽视整体描述 |

### 5.4 幻觉（hallucination）

多模态 SFT 的头号质量问题：模型描述**图中不存在的物体/属性**。SFT 层面能做的：

1. **数据清洗**：用 CLIP 相似度/检测器过滤"图-文不符"的配对数据（图里没有猫，回答却说有猫的样本直接删除）；
2. **拒答与诚实样本**：加入"图中信息不足时回答'无法确定'"的样本；
3. **根本缓解在偏好优化阶段**：SFT 只能数据侧缓解，DPO/RLHF 直接打压幻觉回答才是强手段（LLaVA-RLHF、RLHF-V 等）。

> 面试记忆点：SFT 对幻觉是"数据侧防御"，偏好优化是"目标侧攻击"，后者更有效。

## 六、优缺点与失败模式

### 6.1 优点

- 简单可靠：监督学习，无采样、无 RL，工程门槛低；1~3 epoch 即收敛；
- 效果好：高质量小数据（LIMA 1000 条）就能改变行为；且是 RLHF/DPO 的公共起点。

### 6.2 失败模式

| 失败模式 | 现象 | 根因 | 对策 |
| --- | --- | --- | --- |
| 过拟合/背诵 | 回答固定模板化、换说法就崩 | epoch 过多、数据重复、数据多样性不足 | 1~3 epoch、去重、增多样 |
| 灾难性遗忘/知识损坏 | 通用能力下降、MMLU 类评测下跌 | 数据分布过窄、LR 过大、epoch 过多、与预训练目标冲突 | 混入通用回放数据、LoRA 微调、低学习率 |
| 幻觉放大 | 多模态答非图所有 | 图文不匹配数据、模型被"顺着说" | 数据清洗、拒答样本、偏好优化 |
| 格式崩坏 | 重复输出、吐特殊 token | 模板不一致、mask 没做对 | 统一 ChatML 模板、label mask 检查 |
| 指令跟随退化 | 复杂指令不会，只会简单问答 | 数据里复杂指令占比低 | 指令复杂度分层采样 |

## 七、高频面试问答

**Q1：SFT 为什么只对回答部分算 loss？**
训练目标应与推理一致：推理时模型只需生成回答，prompt 是输入。全序列 loss 会让模型花容量"预测用户问题"（实验里 prompt 区 loss 被压到 0.024），浪费梯度、加速过拟合，还会让格式 token 参与损失导致模板过拟合。实现上用 `-100` 标签 + `ignore_index`。

**Q2：SFT 为什么通常只训 1~3 个 epoch？**
SFT 数据量小（1k~100k），一个 epoch 已遍历完；知识不在 SFT 数据里，多 epoch 不增加知识，反而导致背诵、回答僵化、加速遗忘预训练能力。LIMA/InstructGPT 均验证 1~3 epoch 最优。

**Q3：SFT 和预训练的区别？**
预训练：互联网海量文本、全序列 LM loss、目标是学语言统计与知识、训练量大（数万亿 token）；SFT：人工指令数据、回答-only mask loss、目标是学指令遵循行为、训练量小（1~3 epoch）。SFT 可以看作"带 mask 的、数据换成指令对的预训练式微调"。

**Q4：SFT 能教会模型新知识吗？**
基本不能。SFT 数据量级不足以承载新知识，它做的是"重新布线"——把预训练已有的知识按"指令→回答"的路径调用。想让模型知道新事实要靠预训练/继续预训练或检索增强（RAG）。

**Q5：LIMA 为什么 1000 条数据就够？**
指令遵循是"对齐的分布外泛化"：预训练已提供全部能力，SFT 只需少量高质量演示指明方向（前提是底模够强、质量够高；领域差异大或模型弱时仍需加量）。

**Q6：label mask 具体怎么实现？**
构造 labels：复制 input_ids，把回答区间以外的位置替换成 -100（含 system/user 文本、`<|im_start|>` 等格式 token、图像占位符）；损失用 `F.cross_entropy(..., ignore_index=-100)`；注意 logits/labels 都要 shift 一位。框架里 `DataCollatorForSeq2Seq` / `SFTTrainer` 已内置。

**Q7：多模态 SFT 中视觉塔为什么冻结？**
CLIP ViT 已在海量图文对上预训练，视觉表征通用且强；SFT 数据量远不足以重训，且解冻会遗忘 CLIP 能力、显存暴涨。视觉域差异大（医学/文档）时改为视觉塔 LoRA。投影层和 LLM 是默认训练面。

**Q8：SFT 训完还需要 RLHF/DPO 吗？**
看目标。只要"会对话"→ 高质量 SFT 足够（LIMA 路线）；要"偏好对齐"（更有用、更诚实、更少幻觉）→ 需要 DPO/RLHF，因为 SFT 只学分布、不学偏好——同一个问题的"好回答 vs 差回答"在 SFT 里没有排序信号。

## 八、自我检验

- [ ] 能写出 SFT 目标函数与回答-only mask 的完整推导（全序列 → mask → ignore_index），并解释 shift 一位与 mask 的配合
- [ ] 能说明为什么 SFT 只算回答 loss（推理一致性 + 防浪费 + 防格式过拟合）
- [ ] 能解释"SFT 不教知识、只重新布线"及 InstructGPT 1.3k / LIMA 1000 条的证据
- [ ] 能说出 SFT 1~3 epoch 的四个原因
- [ ] 能手写 build_labels（ChatML 解析 + -100 mask）与 shifted_ce
- [ ] 能复述 3.1 实验的关键数字（全序列 172 vs 回答-only 41 个 token、prompt 区 loss 0.024 vs 5.36）
- [ ] 能说出 ChatML 三角色结构及 label 中图像占位符必须 -100
- [ ] 能给出多模态数据配比经验（caption : 指令问答 ≈ 1:5~1:10，混入 10%~20% 纯文本回放）
- [ ] 能讲清图像描述 vs 指令问答的区别与训练风险
- [ ] 能说明 SFT 对幻觉是"数据侧防御"，偏好优化才是"目标侧攻击"
- [ ] 能列举 6 个失败模式及对策（过拟合、遗忘、幻觉、格式崩坏、指令退化、知识损坏）
- [ ] 能回答 8 个高频面试追问
