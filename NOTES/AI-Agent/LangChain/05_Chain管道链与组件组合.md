# Chain 管道链与组件组合

## 一、Chain 的核心机制

LangChain 用 `|` 运算符将多个 Runnable 组件串联成处理流水线（Pipeline），这是 LangChain 最具标志性的设计。

```
输入 → PromptTemplate → Model → OutputParser → 输出
```

## 二、`|` 运算符的原理

### 2.1 `__or__` 魔术方法

Python 的 `|` 运算符对应 `__or__` 魔术方法。当执行 `a | b` 时，Python 调用 `a.__or__(b)`。

LangChain 对每个 Runnable 组件重写了 `__or__` 方法：
- `a | b` 不立即执行，而是返回一个新的 RunnableSequence 对象
- 这个新对象封装了 a 和 b 的执行顺序
- 新的 RunnableSequence 也实现了 `__or__`，可继续串联

### 2.2 链的惰性求值

`prompt | model` 只是定义了执行流程，不会立即执行。需要调用 `.invoke(input)` 或 `.stream(input)` 才会真正运行。

## 三、Runnable 接口

Runnable 是 LangChain 的统一接口标准，所有可组合组件都实现了该接口：

| 方法 | 作用 | 返回 |
|------|------|------|
| `invoke(input)` | 同步执行，等待完整结果 | 完整输出 |
| `stream(input)` | 流式执行，逐块返回 | Generator |
| `batch(inputs)` | 批量执行多个输入 | 结果列表 |

**实现 Runnable 的组件类型：**
- 所有 Model（LLM、ChatModel）
- 所有 PromptTemplate
- 所有 OutputParser
- RunnableSequence（由 `|` 串联生成的对象）

## 四、典型链模式

### 4.1 基本链

```
chain = prompt | model
```

输入模板变量 → 渲染提示词 → 调用模型 → 原始输出

### 4.2 带解析器链

```
chain = prompt | model | parser
```

增加输出解析步骤，将模型原始输出转为可用格式

### 4.3 多阶段链

```
chain = prompt | model | parser | model | parser
```

前一阶段的输出作为后一阶段的输入，适用于多步处理

## 五、RunnableSequence 的类型

由 `|` 生成的链是一个 RunnableSequence 对象。**链的类型取决于最后一个组件的类型**：
- 理论上 chain 本身没有固定的类型分类，它是动态组合的
- 每一个中间的 chain 仍然是一个 Runnable，可以继续与其他组件组合

## 六、组件化设计原则

LangChain 的链式设计遵循以下原则：

1. **单一职责** — 每个组件只做一件事（模板只构建提示词，模型只调用 API，解析器只处理输出）
2. **统一接口** — 所有组件实现 Runnable 接口，保证可组合性
3. **管道复用** — 不同组件可自由组合成不同 Chain，实现代码复用
4. **透明可控** — 每个环节的输入输出都可观测和调试
