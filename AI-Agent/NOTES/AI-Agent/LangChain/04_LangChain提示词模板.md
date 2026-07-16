# LangChain 提示词模板

## 一、PromptTemplate（通用文本模板）

### 1.1 核心思想

将静态提示词与动态变量分离，实现提示词的复用。模板中的变量用 `{variable_name}` 占位，调用时注入具体值。

### 1.2 使用方法

- 创建：`PromptTemplate.from_template("模板字符串")`
- 注入变量：`.format(variable=value)` 或 `.invoke({"variable": value})`
- 输出类型：`StringPromptValue`（纯文本）

### 1.3 适用场景

- LLM 模型（非 ChatModel）
- 不需要角色区分的简单文本生成
- 模板中变量较少的情况

## 二、FewShotPromptTemplate（少样本模板）

### 2.1 核心思想

在提示词中嵌入若干输入-输出示例（few-shot），引导模型理解任务。本质上是在前缀和后缀之间插入一组示例。

### 2.2 组成要素

| 要素 | 说明 |
|------|------|
| `example_prompt` | 每条示例的格式模板（PromptTemplate） |
| `examples` | 示例数据列表（list[dict]） |
| `prefix` | 提示词前缀，任务描述 |
| `suffix` | 提示词后缀，当前问题 |
| `input_variables` | 输入变量列表 |

### 2.3 执行流程

```
prefix + example1 + example2 + ... + suffix = 完整提示词
```

## 三、ChatPromptTemplate（聊天模板）

### 3.1 核心思想

专门为 ChatModel 设计，支持多角色消息和动态历史消息插入。

### 3.2 消息组成

- 支持 system / user / assistant 多种角色
- 每条消息可以是字符串模板，也可以是 MessagesPlaceholder

### 3.3 MessagesPlaceholder（消息占位符）

**解决的问题：** 在聊天模板中预留位置，运行时动态插入不定长度的历史消息。

**用法：**
- 模板中放置 `MessagesPlaceholder("variable_name")`
- 调用时传入对应的历史消息数据（`list[tuple(role, content)]`）
- 框架自动将历史消息嵌入到模板中

**适用场景：** 多轮对话、带历史记忆的聊天应用。

### 3.4 输出类型

ChatPromptTemplate 的输出是 `ChatPromptValue`，其中包含完整的消息列表，可直接传给 ChatModel。

## 四、三种模板对比

| 模板类型 | 输出类型 | 角色支持 | 适用模型 |
|---------|---------|---------|---------|
| PromptTemplate | StringPromptValue | 无 | LLM |
| FewShotPromptTemplate | StringPromptValue | 无（通过 prefix/suffix 间接） | LLM |
| ChatPromptTemplate | ChatPromptValue | 原生支持多角色 | ChatModel |

## 五、选择指南

- 简单文本生成 → `PromptTemplate`
- 需要示例引导 → `FewShotPromptTemplate`
- 多轮对话 / 带角色 → `ChatPromptTemplate`
- 需要历史消息 → `ChatPromptTemplate` + `MessagesPlaceholder`
