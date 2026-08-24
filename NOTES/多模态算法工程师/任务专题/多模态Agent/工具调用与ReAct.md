---
learned: false
---

# 工具调用与ReAct：让模型会"用工具"，还会"循环思考"

> 本模块索引见 [多模态Agent详解](多模态Agent详解.md)

## 一、工具调用（Function Calling）原理

### 1.1 核心机制：模型输出结构化 JSON

Function Calling 的本质：**把"要不要调用工具、调用哪个、传什么参数"变成一个标准化的文本生成任务**。模型在生成时不是自由写作文，而是输出一段约定格式的 JSON：

```json
{
  "name": "ocr",
  "arguments": {"image_path": "scene1.png"}
}
```

- `name`：要调用的工具名（必须是模型已知的工具列表之一）；
- `arguments`：该工具的参数，键名与工具 schema 的参数定义一致。

模型给出这段 JSON 后，**调用权在应用侧**：你的代码执行真实工具，把结果作为新消息放回对话，模型再基于结果继续（这是 Agent 循环的起点，详见第三节）。

### 1.2 工具 Schema 定义（name / description / parameters）

模型要"知道有哪些工具、什么时候用、参数怎么传"，靠的是**工具描述**——通常用 JSON Schema 描述参数结构：

```json
{
  "type": "function",
  "function": {
    "name": "ocr",
    "description": "对图片执行光学字符识别，返回识别出的文字。当需要读取图片中的文字、价格、票据信息时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "image_path": {
          "type": "string",
          "description": "图片文件路径"
        },
        "lang": {
          "type": "string",
          "enum": ["zh", "en"],
          "default": "zh",
          "description": "识别语言"
        }
      },
      "required": ["image_path"]
    }
  }
}
```

设计要点：

| 字段 | 作用 | 教训 |
| --- | --- | --- |
| `name` | 唯一标识，必须稳定 | 改名会导致旧数据失效；不要带空格/特殊字符 |
| `description` | **决定模型会不会用这个工具** | 要写"何时用"+"何时不用"，写得太泛模型会乱调 |
| `parameters` | 参数结构 + 合法性约束 | `required` 不写全 → 模型漏参数；`enum` 能显著降低幻觉参数 |

> **经验法则**：description 里明确"什么时候不该用这个工具"，比只写"什么时候用"更能降低幻觉调用（详见第六节）。

### 1.3 System Prompt 注入工具列表

模型不知道工具，工具列表要写进提示词。两种主流方式：

1. **系统提示注入**（OpenAI/Anthropic 风格）：把全部工具 schema 转成文本放进 system prompt，模型从上下文"学会"工具的存在；
2. **模型内部原生支持**（Qwen 等）：在预训练/后训练阶段就把工具调用格式教给模型，调用时仍需注入 schema（模型需要实时知道当前可用哪些工具）。

```python
system_prompt = (
    "你是多模态助手，可以调用以下工具：\n"
    "1. ocr(image_path: str, lang: str='zh')：OCR 识别图片文字\n"
    "2. retrieve(query: str)：检索知识库\n"
    "如需工具请严格输出 JSON：{\"name\": ..., \"arguments\": {...}}"
)
```

### 1.4 训练数据格式：对话中含 tool_calls / observation 轮次

训练阶段，模型的"工具调用能力"来自**对话格式的监督数据**。一条完整样本是这样的轮次序列：

```
用户: 这张货架图里酸奶多少钱？        ← user（可含图像）
助手: {"name": "ocr", "arguments": {...}}   ← assistant(tool_calls)，只输出调用不输出回答
工具: 货架标签：酸奶 12.8 元          ← observation（工具真实返回）
助手: 酸奶 12.8 元/盒。               ← assistant（最终回答）
```

训练时对每个 token 做交叉熵，**只有 assistant 轮次的 token 参与损失**：

$$\mathcal{L} = -\sum_{t \in \text{assistant 轮次}} \log P_\theta(w_t \mid w_{<t})$$

要点：

- **中间轮次（tool_calls）与最终回答分开成两个 assistant 消息**——让模型学会"先调用、后总结"的节奏，而不是边调用边编答案；
- 工具返回的 observation 要**真实的执行结果**（哪怕失败信息），不能造假，否则模型学到"编造工具结果"；
- 多轮样例要构造"连续调 2~3 个工具"的轨迹，让模型学会按需多次调用。

## 二、函数调用与 SFT 数据构造：多模态场景示例

以"图文一致性项目里调 OCR 工具"为例，构造一条完整的多模态 SFT 样本：

```json
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image_path": "product_1.png"},
      {"type": "text", "text": "图上的商品名是什么？我要用它去查商品库。"}
    ]
  },
  {
    "role": "assistant",
    "tool_calls": [{
      "name": "ocr",
      "arguments": {"image_path": "product_1.png", "lang": "zh"}
    }],
    "content": null
  },
  {
    "role": "tool",
    "name": "ocr",
    "content": "文案内容：『冰萃柠檬气泡水 500ml』"
  },
  {
    "role": "assistant",
    "content": "OCR 识别出商品名是「冰萃柠檬气泡水」。"
  }
]
```

构造注意事项：

1. **图像轮次用 content 数组**（OpenAI Chat 格式），图像以路径/URL/字节引用，训练时转成视觉 token；
2. **user 指令里写明"要用工具做什么"**（"用它去查商品库"），模型学到的是"识别→检索"的习惯；
3. **observation 放真实 OCR 输出**，包括可能带噪声的结果（如漏字），让模型学会"结果可能不完美，必要时复核"；
4. 负样本同样重要：构造"图片里没文字 → 模型回答不适用并说明"，避免见到图就调 OCR（工具调用幻觉的 SFT 层防线）。

## 三、ReAct 循环

### 3.1 三步循环：Thought → Action → Observation

```
                ┌──────────────────────────────────────────┐
                │              消息历史（循环状态）              │
                │  user: 帮我看货架图酸奶价格和库存               │
                └──────────────────────────────────────────┘
                                │
                                ▼
              ┌───────────────────────────────────────────┐
              │  Thought（模型思考）：                      │
              │  "需要先 OCR 提取图片文字"                    │
              │  Action（模型动作）：                       │
              │  {"name": "ocr", "arguments": {...}}       │
              └───────────────────────────────────────────┘
                                │  应用侧执行真实工具
                                ▼
              ┌───────────────────────────────────────────┐
              │  Observation（工具返回）：                  │
              │  "货架标签：酸奶 12.8 元，牛奶 9.9 元"        │
              └───────────────────────────────────────────┘
                                │  结果拼回消息历史
                                └──────────→ 回到 Thought（循环）
```

- **Thought**：模型对自己说"现在是什么情况、下一步做什么"——让推理可解释、可纠错；
- **Action**：调用工具（结构化 JSON）或直接给答案；
- **Observation**：工具的真实返回，是循环里唯一的"外部事实源"；
- 循环终止条件：模型输出最终答案，或达到最大步数。

### 3.2 多轮工具调用的状态维护（消息历史）

ReAct 的本质是**带工具轮次的序列决策**。状态就是消息历史 $h_t$，每轮追加：

$$h_{t+1} = h_t \oplus \{\text{Thought}_t, \text{Action}_t, \text{Observation}_t\}$$

工程上三条铁律：

1. **Observation 必须真实拼接**：工具返回什么就放什么，不能覆盖/截断到丢失信息；
2. **每条消息带角色**：`user` / `assistant(tool_calls)` / `tool` 严格交替，这是模型训练时学到的格式，顺序错了模型会"发懵"；
3. **上下文要控制**：长循环里 history 会膨胀，用截断/摘要压缩，但**不得删掉还没被"消费"的关键 observation**（模型可能下一轮还要用）。

### 3.3 ReAct 与单纯 Function Calling 的区别

| 维度 | Function Calling | ReAct Agent |
| --- | --- | --- |
| 本质 | 模型输出结构化调用（一次动作） | 动作 + 推理循环（多步决策） |
| 是否多轮 | 单次调用，调用完就结束 | 循环直到答案或上限 |
| 是否携带 Thought | 无 | 有（推理轨迹，可解释、可纠错） |
| 状态 | 无状态 | 消息历史即状态 |
| 失败处理 | 交给应用侧 | 模型看到失败 Observation 后可自行换策略重试 |

**一句话**：Function Calling 是"会按格式调工具"，ReAct 是"带着思考地反复调工具"——工程上通常用 Function Calling 作为 Action 的底层实现，外面套 ReAct 循环。

## 四、多模态工具链设计：OCR→检测→检索→VLM 复核

单工具调用解决不了"图文是否一致"这类复合任务，需要**编排链**。以图文一致性校验（如电商商品主图与商品描述核对）为例：

```
输入: 商品图 product.png + 商品标题/描述文本
 │
 ▼
[1] OCR ──────────→ 提取图片内文字（"冰萃柠檬气泡水 500ml 无糖"）
 │
 ▼
[2] 目标检测 ──────→ 定位图中商品本体与包装（bbox），确认图里确实有该商品
 │
 ▼
[3] 图像/文本检索 ──→ 拿 OCR 结果与标题查商品库，取标准资料
 │
 ▼
[4] VLM 复核 ──────→ 综合 图文文本比对 + 视觉内容，输出一致/不一致 + 差异点
 │
 ▼
输出: {"consistent": false, "diff": ["标题写『550ml』，图内标签是『500ml』"]}
```

每一步的决策点（模型判断，不硬编码）：

| 决策点 | 判断内容 | 失败分支 |
| --- | --- | --- |
| OCR 后 | 文字是否完整？ | 不完整 → 提高分辨率重试 / 转 VLM 直接读图 |
| 检测后 | 图中是否有商品本体？ | 无 → 判定"图物不符"提前终止 |
| 检索后 | 知识库有无匹配记录？ | 无 → 只做图文直读比对，并标注低置信 |
| 复核前 | 证据是否冲突？ | 冲突 → 追加一次检测/检索确认，再决定 |

**为什么这样分层而不是全交给 VLM**：OCR/检测小模型快且便宜，可以逐帧/多图并行；VLM 只负责语义综合（"图里这瓶水是不是标题这款"），全链路成本降一个量级。这也是 [多模态Agent详解](多模态Agent详解.md) 第七节 Q6 的答案。

## 五、源码实现：手写一个最小工具调用 Agent

下面实现一个完整的 ReAct 循环：两个模拟工具（OCR、知识库检索）+ 一个规则式"模型"（生产环境替换为真实 LLM），完整输出 Thought / Action / Observation。

```python
import json


def run_ocr(image_path: str) -> str:
    """模拟 OCR 工具：真实实现是视觉模型读图输出文字"""
    db = {
        "scene1.png": "货架标签：酸奶 12.8 元，牛奶 9.9 元",
        "invoice.jpg": "增值税发票 价税合计 3,600.00 元",
    }
    return db.get(image_path, "未识别到文字")


def run_retrieve(query: str) -> str:
    """模拟知识库检索工具"""
    db = {"酸奶": "库存 128 件，可正常下单", "牛奶": "库存 45 件，需补货"}
    for k, v in db.items():
        if k in query:
            return f"{k}：{v}"
    return "知识库无相关记录"


def execute_tool(name: str, arguments: dict) -> str:
    """工具调度器：按 name 分发到具体实现"""
    if name == "ocr":
        return run_ocr(arguments["image_path"])
    if name == "retrieve":
        return run_retrieve(arguments["query"])
    raise ValueError(f"未知工具: {name}")


def fake_llm(messages):
    """模拟支持 function calling 的 LLM：按规则输出 Thought + Action JSON"""
    last = messages[-1]
    if last["role"] == "user":
        return {
            "thought": "用户上传了货架图并询问酸奶价格与库存，先用 OCR 提取标签文字",
            "action": "tool", "name": "ocr", "arguments": {"image_path": "scene1.png"},
        }
    if last["role"] == "tool":
        if "库存" in last["content"]:
            return {
                "thought": "库存 128 件充足，信息齐全，可以直接回答用户",
                "action": "answer", "content": "酸奶 12.8 元/盒，库存 128 件，可正常下单。",
            }
        if "酸奶" in last["content"]:
            return {
                "thought": "OCR 已拿到价格 12.8 元，还需确认库存，调用检索工具",
                "action": "tool", "name": "retrieve", "arguments": {"query": "酸奶 库存"},
            }
    return {"thought": "无可用信息，如实告知", "action": "answer", "content": "无法处理该请求。"}


def react_loop(user_message: str, max_steps: int = 4) -> str:
    """ReAct 主循环：消息历史即状态，循环直到模型给答案或达上限"""
    messages = [{"role": "user", "content": user_message}]
    print(f"[USER] {user_message}")
    for step in range(1, max_steps + 1):
        decision = fake_llm(messages)   # 真实场景: response = llm.chat(messages)
        print(f"\n[Step {step}] Thought: {decision['thought']}")
        if decision["action"] == "answer":
            print(f"Action: 直接回答 → {decision['content']}")
            return decision["content"]
        print(f"Action: 调用工具 {decision['name']}({json.dumps(decision['arguments'], ensure_ascii=False)})")
        result = execute_tool(decision["name"], decision["arguments"])
        print(f"Observation: {result}")
        messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        messages.append({"role": "tool", "content": result})
    print(f"\n[上限] 已达 max_steps={max_steps}，停止循环")
    return messages[-1]["content"]


if __name__ == "__main__":
    answer = react_loop("请看一下这张货架图，告诉我酸奶多少钱、库存够不够")
    print(f"\n最终回答：{answer}")
```

运行输出（Python 3.10+）：

```
[USER] 请看一下这张货架图，告诉我酸奶多少钱、库存够不够

[Step 1] Thought: 用户上传了货架图并询问酸奶价格与库存，先用 OCR 提取标签文字
Action: 调用工具 ocr({"image_path": "scene1.png"})
Observation: 货架标签：酸奶 12.8 元，牛奶 9.9 元

[Step 2] Thought: OCR 已拿到价格 12.8 元，还需确认库存，调用检索工具
Action: 调用工具 retrieve({"query": "酸奶 库存"})
Observation: 酸奶：库存 128 件，可正常下单

[Step 3] Thought: 库存 128 件充足，信息齐全，可以直接回答用户
Action: 直接回答 → 酸奶 12.8 元/盒，库存 128 件，可正常下单。

最终回答：酸奶 12.8 元/盒，库存 128 件，可正常下单。
```

代码对应关系（面试可以指着讲）：

| 代码位置 | 对应真实系统 |
| --- | --- |
| `fake_llm` | LLM 推理：`llm.chat(messages)` 返回 `tool_calls` |
| `execute_tool` | 应用侧工具调度：按 `name` 分发、参数校验 |
| `messages.append(...)` | 消息历史维护：严格交替 `assistant(tool_calls)` → `tool` |
| `max_steps` | 循环上限：防止死循环与无限烧钱 |

## 六、常见问题

### 6.1 工具调用幻觉（编造工具、编造参数）

- **现象**：模型调用不存在的工具；或调用真实工具但参数是编的（如"库存=9999"）；
- **根因**：工具 schema 描述不清、训练数据里少见该工具、模型把"猜测"当成"事实"；
- **防线**（层层加固）：
  1. schema 层：`enum`/`required`/`description` 写死约束；
  2. 应用层：**参数校验 + 白名单**，非法调用直接返回失败 Observation，不执行；
  3. 数据层：SFT 加负样本（"无权限时不许调工具"）；RLHF 惩罚幻觉调用；
  4. 架构层：关键参数只允许来自工具返回（检索到的值），不允许模型"自由发挥"数值。

### 6.2 工具返回错误处理

- 工具失败要**把失败信息原样返回**给模型（Observation: "OCR 超时"），让模型决定重试/换工具/放弃，而不是应用侧静默吞掉；
- 重试要有退避：同一工具连续失败 2 次 → 换策略（转 VLM 直读、换分辨率、告知用户）。

### 6.3 循环上限

- 必须设置 `max_steps`（常见 5~15），否则死循环烧钱；
- 达上限时的兜底：输出"已尽力但仍未完成 + 当前已收集到的信息"，而不是编一个答案。

## 七、面试问答

**Q1：function calling 是怎么训练的？**
把它当成一种**特殊格式的文本生成任务**做 SFT：构造"user → assistant(tool_calls JSON) → tool(observation) → assistant(回答)"的对话样本，只在 assistant 轮次上算交叉熵；再配合大量真实工具返回、负样本（不该调时不调）和 RLHF 偏好对齐。

**Q2：ReAct 是什么？**
一个思考-行动-观察循环范式（ReAct = Reason + Act）：模型每轮先输出 Thought（推理轨迹），再输出 Action（工具调用），应用执行后把 Observation 拼回消息历史继续循环，直到给出答案或达上限。比 CoT 多"与外部世界交互"的能力，可纠错、可解释。

**Q3：工具调用幻觉怎么防？**
schema 写清约束（enum/required/何时不用）→ 应用层白名单与参数校验，非法调用不执行并返回失败 → 训练加负样本与 RLHF 惩罚 → 关键数值只允许来自工具返回。

**Q4：多轮工具调用时状态怎么维护？**
消息历史即状态：按 user / assistant(tool_calls) / tool 严格交替拼接，Observation 必须真实；长循环用截断/摘要控制上下文，但已产生的关键 observation 不能删。$h_{t+1} = h_t \oplus \{\text{Thought}_t, \text{Action}_t, \text{Observation}_t\}$

**Q5：多模态工具链为什么要分层（OCR→检测→检索→VLM）？**
小模型（OCR/检测）便宜快，适合逐帧并行粗处理；VLM 昂贵但擅长语义综合。分层编排在成本、延迟、可解释性（每步可单独 debug）上全面优于"一张图全交给 VLM"。

**Q6：ReAct 和 Function Calling 是一回事吗？**
不是。Function Calling 是模型输出结构化调用的**能力**；ReAct 是把"思考+调用+观察"循环起来直到任务完成的**范式**。实际工程中常以 Function Calling 为 Action 的实现，外面套 ReAct 循环。

**Q7：工具返回图片，Observation 怎么表示？**
Observation 可以是多模态消息（图像路径/URL + 文本描述），文本里写清"该图在循环中的用途"，避免 VLM 把工具返回图与用户原图混淆；必要时可附检测框/坐标等结构化信息。

## 八、自我检验

- [ ] 能说出 Function Calling 的完整链路：schema → 注入 prompt → 模型输出 JSON → 应用执行 → 结果回填
- [ ] 能写出一条含 tool_calls / observation 轮次的 SFT 样本 JSON
- [ ] 能画出 ReAct 的 Thought→Action→Observation 循环并说出终止条件
- [ ] 能说清 ReAct 与单纯 Function Calling 的三个区别
- [ ] 能画出 OCR→检测→检索→VLM 复核工具链并说出每步决策点
- [ ] 能独立运行第五节代码并复述输出（3 步循环：OCR → 检索 → 回答）
- [ ] 能说出防工具调用幻觉的四层防线
- [ ] 能回答 7 个面试追问
