# RAG 项目工程化架构

## 一、项目整体架构

一个完整的 RAG + Agent 项目通常包含以下模块：

```
project/
├── config/              # 配置文件
│   ├── rag.yml          # 模型配置
│   ├── chroma.yml       # 向量库配置
│   ├── prompts.yml      # 提示词路径配置
│   └── agent.yml        # Agent 配置
├── model/               # 模型工厂
│   └── factory.py       # 模型实例化工厂
├── rag/                 # RAG 服务
│   ├── vector_store.py  # 向量存储服务
│   └── rag_service.py   # RAG 总结服务
├── agent/               # Agent 服务
│   ├── react_agent.py   # ReAct Agent
│   └── tools/           # 工具和中间件
│       ├── agent_tools.py
│       └── middleware.py
├── prompts/             # 提示词模板
│   ├── main_prompt.txt
│   ├── rag_summarize.txt
│   └── report_prompt.txt
└── utils/               # 工具类
    ├── config_handler.py
    ├── prompt_loader.py
    ├── file_handler.py
    ├── logger_handler.py
    └── path_tool.py
```

## 二、配置管理（Config）

### 2.1 YAML 配置文件

使用 YAML 文件管理所有配置，便于维护和切换环境。

**配置分类：**
- `rag.yml` —— 模型名称（chat_model_name, embedding_model_name）
- `chroma.yml` —— 向量库参数（collection_name, chunk_size, k 等）
- `prompts.yml` —— 提示词文件路径
- `agent.yml` —— Agent 相关配置

### 2.2 配置加载

统一的配置加载函数，项目启动时一次性加载所有配置：

```python
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
```

**好处：** 配置集中管理，修改配置不需要改代码。

## 三、模型工厂（Model Factory）

### 3.1 工厂模式

使用抽象工厂模式创建模型实例，解耦模型创建与使用。

**抽象基类：** `BaseModelFactory` 定义 `generate()` 方法
**具体工厂：** `ChatModelFactory` 和 `EmbeddingsModelFactory`

### 3.2 单例模式

工厂生成的模型实例在整个项目中共享：

```python
chat_model = ChatModelFactory().generate()
embedding_model = EmbeddingsModelFactory().generate()
```

**好处：** 避免重复创建模型实例，节省资源。

## 四、提示词管理

### 4.1 外部文件存储

提示词存储在独立的 .txt 文件中，而非硬编码在代码里。

**优势：**
- 提示词修改不需要改代码
- 非技术人员也可以调整提示词
- 不同场景使用不同提示词文件

### 4.2 提示词加载器

统一的加载函数，从配置文件中读取路径，再加载内容：

```python
load_system_prompt()        # 加载主提示词
load_rag_summarize_prompt() # 加载 RAG 摘要提示词
load_report_prompt()        # 加载报告提示词
```

## 五、文件处理

### 5.1 MD5 去重

通过计算文件的 MD5 哈希值，避免重复处理同一个文件。

**流程：**
1. 计算文件 MD5
2. 检查 MD5 是否已存在于记录文件
3. 如果不存在，处理文件并保存 MD5
4. 如果已存在，跳过处理

### 5.2 多格式支持

支持 txt 和 pdf 两种格式的文件加载：
- `txt_loader()` —— 使用 TextLoader
- `pdf_loader()` —— 使用 PyPDFLoader

## 六、日志系统

### 6.1 日志配置

- 控制台输出 INFO 级别日志
- 文件记录 DEBUG 级别日志
- 日志格式：时间 - 名称 - 级别 - 文件:行号 - 信息

### 6.2 日志用途

- 记录 RAG 检索结果
- 记录模型调用参数
- 记录工具调用过程
- 记录错误和异常

## 七、向量存储服务

### 7.1 职责

- 初始化 Chroma 向量库
- 提供检索器接口
- 支持文档加载和分割
- MD5 去重避免重复入库

### 7.2 文档加载流程

```
遍历数据目录 → 检查文件类型 → 计算 MD5 去重
    → 加载文档 → 分割文档 → 存入向量库
```

## 八、RAG 总结服务

### 8.1 职责

接收用户问题，检索相关文档，基于文档生成回答。

### 8.2 流程

```
用户问题 → 检索相关文档 → 格式化文档为字符串
    → 构建提示词 → 调用模型 → 返回回答
```

## 九、设计模式总结

| 模式 | 应用 | 好处 |
|------|------|------|
| 工厂模式 | 模型创建 | 解耦创建与使用 |
| 单例模式 | 模型实例 | 节省资源 |
| 配置分离 | YAML 文件 | 易维护 |
| 提示词外部化 | txt 文件 | 易调整 |
| MD5 去重 | 文件处理 | 避免重复 |
| 中间件 | Agent 扩展 | 灵活可插拔 |
