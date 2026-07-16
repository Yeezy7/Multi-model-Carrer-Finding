# RAG 检索增强生成

## 一、RAG 核心概念

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识库与大语言模型结合的技术。核心思想是：**先检索相关文档，再基于检索结果让模型生成回答。**

**为什么需要 RAG：**
- LLM 的知识有截止日期，无法获取最新信息
- LLM 的训练数据不包含特定领域知识
- RAG 让模型可以引用外部资料，减少幻觉

## 二、RAG 完整流程

```
用户提问
    ↓
向量检索（Retriever）→ 从向量库中检索相关文档
    ↓
提示词构建（Prompt）→ 将检索结果 + 用户问题组合成提示词
    ↓
模型生成（Generate）→ LLM 基于提示词生成回答
```

## 三、检索阶段

### 3.1 向量检索器

通过 `vector_store.as_retriever()` 将向量存储转为检索器。

**检索器特性：**
- 输入：查询字符串（用户问题）
- 输出：`list[Document]`（相关文档列表）
- 可配置参数：`k`（返回文档数量）、`filter`（元数据过滤）

### 3.2 检索结果格式化

检索到的 Document 对象需要转为字符串，才能传入提示词模板：

```
docs → "".join([doc.page_content for doc in docs]) → context 字符串
```

## 四、提示词构建

### 4.1 RAG 提示词模板

```
system: 以我提供的已知参考资料为主，专业地回答用户问题。参考资料：{context}
user:   用户提问：{input}
```

**设计要点：**
- 明确告诉模型"参考资料"的位置
- 要求模型"以参考资料为主"，减少幻觉
- 将检索结果作为上下文注入

### 4.2 带历史消息的 RAG

```
system: 以参考资料回答问题
system: 对话历史记录：
MessagesPlaceholder("history")
user:   请回答：{input}
```

## 五、生成阶段

模型收到包含检索结果的提示词后，基于参考资料生成回答。关键在于：
- 提示词中明确引用来源
- 模型回答应基于检索结果，而非自身知识
- 输出解析器将 AIMessage 转为纯文本

## 六、RAG 链的典型结构

### 6.1 基本 RAG 链

```
retriever | format_func → context
                    ↓
prompt → model → parser → 回答
```

### 6.2 带 RunnablePassthrough 的 RAG 链

```
{
    "input": RunnablePassthrough(),
    "context": retriever | format_func
} | prompt | model | parser
```

**设计思想：**
- `RunnablePassthrough()` 将用户查询原样传给 prompt 的 input
- 检索器负责检索相关文档，传给 prompt 的 context
- 两个分支并行处理，最终合并到 prompt

### 6.3 带历史消息的完整 RAG 链

```
{
    "input": RunnablePassthrough(),
    "context": retriever | format_func,
    "history": RunnableWithMessageHistory 自动注入
} | prompt | model | parser
```

## 七、RAG 项目中的关键组件

| 组件 | 职责 |
|------|------|
| 文档加载器 | 读取原始文件（txt/pdf/csv/json） |
| 文本分割器 | 将长文档切分为小块 |
| 嵌入模型 | 将文本转为向量 |
| 向量存储 | 存储向量，支持相似度检索 |
| 检索器 | 接收查询，返回相关文档 |
| 提示词模板 | 将检索结果 + 用户问题组合 |
| LLM | 基于提示词生成回答 |
| 输出解析器 | 将模型输出转为可用格式 |
