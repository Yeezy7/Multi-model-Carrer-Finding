# LangChain 模型调用体系

## 一、LLM vs ChatModel

LangChain 提供了两种模型类型来适配不同场景：

| 对比维度 | LLM（Tongyi） | ChatModel（ChatTongyi） |
|---------|---------------|------------------------|
| 输入类型 | 纯文本字符串 | 消息列表（多角色） |
| 消息角色 | 无角色区分 | system / user / assistant / tool |
| 适用场景 | 简单问答、文本生成 | 多轮对话、复杂交互 |
| 返回类型 | 字符串 | `AIMessage` 对象 |
| 导入路径 | `langchain_community.llms.tongyi` | `langchain_community.chat_models.tongyi` |

## 二、两种调用模式

| 方法 | 行为 | 返回 |
|------|------|------|
| `invoke()` | 等待完整结果后返回 | 完整输出 |
| `stream()` | 逐块返回，实时迭代 | Generator / 迭代器 |

invoke 适合短文本或需要完整结果后再处理的场景；stream 适合打字机效果或长文本实时展示。

## 三、消息系统

### 3.1 消息对象类型

| 类名 | 角色 | 替代元组简写 |
|------|------|-------------|
| `SystemMessage` | system | `("system", "内容")` |
| `HumanMessage` | user | `("user", "内容")` |
| `AIMessage` | assistant | `("assistant", "内容")` |

### 3.2 消息简写形式

LangChain 允许用元组 `(role, content)` 替代消息对象，称为简写形式（Shorthand）。框架内部会自动将其转换为对应的消息对象。

```
("system", "你是一个助手")  →  SystemMessage(content="你是一个助手")
("user", "你好")           →  HumanMessage(content="你好")
("assistant", "你好")      →  AIMessage(content="你好")
```

这种机制大幅简化了代码编写，尤其在动态构建消息列表时非常方便。

## 四、嵌入模型（Embedding Model）

### 4.1 作用

嵌入模型将文本转换为高维向量，是语义搜索和 RAG 的基石。相似的文本在向量空间中距离更近。

### 4.2 主要方法

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `embed_query(text)` | 单条文本 | 单个向量 | 用户查询向量化 |
| `embed_documents(texts)` | 文本列表 | 向量列表 | 文档批量向量化 |

### 4.3 与 LLM 的关键区别

| 特性 | LLM / ChatModel | Embedding Model |
|------|----------------|-----------------|
| 输出 | 自然语言文本 | 高维浮点数向量 |
| 调用方式 | invoke / stream | embed_query / embed_documents |
| 数据用途 | 阅读 / 对话 | 相似度计算 / 检索 |
