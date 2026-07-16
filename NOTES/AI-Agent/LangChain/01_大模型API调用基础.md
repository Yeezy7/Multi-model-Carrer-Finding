# 大模型 API 调用基础

## 一、调用流程总览

调用大语言模型（LLM）本质是通过 HTTP 请求向模型服务端发送对话数据，并接收模型返回的文本。标准流程分为三步：

1. **创建客户端（Client）** — 初始化 SDK 客户端，配置 API Key 和 Base URL
2. **调用模型** — 构建消息列表（messages），指定模型名称，发起请求
3. **处理结果** — 从响应中提取模型生成的文本内容

## 二、客户端配置

使用国内大模型（如阿里云通义千问）时，只需将 `base_url` 指向兼容 OpenAI 接口格式的网关地址，API Key 使用对应平台提供的密钥。这种**兼容模式**让开发者可以用同一套 OpenAI SDK 调用不同厂商的模型服务。

| 配置项 | 说明 | 示例值 |
|--------|------|--------|
| `api_key` | 身份认证密钥，应从环境变量读取 | `os.environ.get("OPENAI_API_KEY")` |
| `base_url` | API 网关地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `model` | 具体模型名称 | `qwen3-max`, `qwen-max` |

## 三、消息系统（Messages）

LLM 的输入是一个消息列表（messages），每条消息包含 `role`（角色）和 `content`（内容）。

### 3.1 三种核心角色

| 角色 | 含义 | 作用 |
|------|------|------|
| `system` | 系统指令 | 设定 AI 的身份、行为规则、输出格式，优先级最高 |
| `user` | 用户消息 | 用户的提问或指令 |
| `assistant` | AI 回复 | 模型的回答，也可用于提供示例（few-shot） |

### 3.2 消息与"记忆"的本质

**关键认知：模型本身不记忆之前的对话。** 每次请求都是独立的。要让模型"记住"历史，必须在每次请求时将完整的对话历史包含在 messages 列表中。

消息组织原则：
- system 消息放在最前面，定义 AI 的全局行为
- user 和 assistant 消息按时间交替排列
- 模型通过读取所有历史消息来"理解"上下文
- 实际应用需注意 token 长度限制，超出时需做截断或摘要

## 四、非流式输出 vs 流式输出

| 模式 | 请求参数 | 返回方式 | 适用场景 |
|------|----------|----------|----------|
| **非流式** | 默认 | 等待完整生成后一次性返回 | 短文本、即时问答 |
| **流式** | `stream=True` | 逐 token 返回，实时展示 | 长文本、打字机效果 |

流式输出的处理要点：
- 响应是一个迭代器，需用 for 循环逐块读取
- 每个 chunk 包含 `choices[0].delta.content`（可能为 None）
- 使用 `end=""` 控制不换行，`flush=True` 实时刷新

## 五、System Prompt 的作用

System Prompt 是引导模型行为最有效的手段，主要作用：

1. **角色定义** — 给 AI 一个专业身份（"你是一个 Python 编程专家"）
2. **行为约束** — 控制风格（"话不多"、"回答简洁"）
3. **格式控制** — 规定输出格式（"按 JSON 字符串输出"）
4. **边界设定** — 定义处理规则（"不清楚的分类为'不清楚类别'"）

## 六、调用模式对比

| 模式 | API | 流式支持 | 返回类型 |
|------|-----|----------|----------|
| OpenAI SDK 调用 | `client.chat.completions.create()` | 支持 | `response.choices[0].message.content` |
| LangChain LLM | `model.invoke()` / `model.stream()` | 支持 | 字符串 |
| LangChain ChatModel | `model.invoke()` / `model.stream()` | 支持 | `AIMessage` 对象 |
