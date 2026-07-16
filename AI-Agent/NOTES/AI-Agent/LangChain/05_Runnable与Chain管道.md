# Runnable 与 Chain 管道

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
- RunnableLambda（自定义函数包装器）
- RunnablePassthrough（直通组件）

## 四、RunnableLambda（自定义函数包装器）

### 4.1 作用

将任意 Python 函数包装为 Runnable，使其可以参与 `|` 链式调用。解决的核心问题：在链的中间步骤做自定义的数据转换。

### 4.2 使用场景

**场景一：类型转换**

模型输出是 AIMessage 对象，但下一个模板需要 dict。用 RunnableLambda 做中间转换：
```
prompt → model → RunnableLambda(lambda ai_msg: {"name": ai_msg.content}) → prompt → model
```

**场景二：调试日志**

在链中插入打印函数，观察中间结果：
```
prompt → print_prompt | model
```

**场景三：数据格式化**

将检索结果格式化为字符串，再传入模板：
```
retriever → RunnableLambda(format_func) → prompt
```

### 4.3 简写形式

RunnableLambda 可以直接用 lambda 表达式简写：
```
chain = prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | prompt | model
```

## 五、RunnablePassthrough（直通组件）

### 5.1 作用

将输入原样传递给下游组件，不做任何修改。在需要并行传递同一个输入时非常有用。

### 5.2 典型场景

在 RAG 链中，用户查询需要同时传给检索器和作为模板的 input 变量：
```
{
    "input": RunnablePassthrough(),
    "context": retriever | format_func
} | prompt | model
```

- `RunnablePassthrough()` 将用户查询原样传给 prompt 的 `input` 变量
- 检索器负责检索相关文档，传给 prompt 的 `context` 变量

## 六、典型链模式

### 6.1 基本链

```
chain = prompt | model
```

### 6.2 带解析器链

```
chain = prompt | model | parser
```

### 6.3 多阶段链

```
chain = prompt | model | parser | prompt2 | model2 | parser2
```

### 6.4 并行输入链（字典输入）

```
chain = {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | model
```

## 七、组件化设计原则

1. **单一职责** —— 每个组件只做一件事
2. **统一接口** —— 所有组件实现 Runnable 接口，保证可组合性
3. **管道复用** —— 不同组件可自由组合成不同 Chain
4. **透明可控** —— 每个环节的输入输出都可观测和调试
