# AI Agent 求职导向学习路线

你的方向可以定为：

> **AI 应用工程师 / AI Agent 开发工程师**
> 核心竞争力：**后端工程 + RAG + Agent 工作流 + MCP + 评测与部署**

不要把目标定成“掌握 LangChain、Dify、Coze”。目前字节的 Agent 开发实习岗位已经明确要求编程能力、数据结构、后端服务开发与部署；百度的智能体岗位则覆盖任务规划、工具调用、记忆、多智能体协同，以及效果、效率、成本一体化评测。([字节跳动招聘][1])

因此，学习路线应围绕一条完整能力链展开：

```text
Python后端
→ 大模型API与Tool Calling
→ RAG检索系统
→ Agent状态机与工作流
→ MCP与外部工具
→ 评测、可观测性和安全
→ 部署与高并发
→ 完整业务项目
```

建议采用 **12周求职版本**，每周投入约20—25小时。你的简历材料中已经出现过 Python、FastAPI、Docker、模型部署和视频流联调，可以压缩基础阶段，但仍需按实际掌握程度重新验证。

---

## 一、最终技能栈

### 主语言与后端

* Python
* FastAPI
* Pydantic
* asyncio
* PostgreSQL
* Redis
* Docker / Docker Compose
* Git
* Linux
* HTTP、SSE、WebSocket
* 基础消息队列

### LLM与Agent

* Prompt与上下文管理
* Structured Output
* Function Calling / Tool Calling
* Agent Loop
* RAG
* Memory
* Planning
* Human-in-the-loop
* Multi-Agent基础
* MCP
* Guardrails
* Agent Evaluation
* Tracing与Observability

### 框架选择

不要同时学习五六个框架，主线建议：

* **LangGraph：主要Agent编排框架**
* **OpenAI Agents SDK：理解另一种Agent Runtime设计**
* **MCP Python SDK：开发工具服务**
* FastAPI：业务后端
* PostgreSQL + pgvector，或者 Milvus：知识检索
* Redis：缓存、状态和任务控制

LangGraph重点解决持久化执行、流式输出、Human-in-the-loop和状态管理；OpenAI Agents SDK则提供工具、Handoff、Guardrail、Session和Tracing等Agent运行能力。([Docs by LangChain][2])

---

# 二、第1—2周：Python后端与软件工程基础

Agent岗位的基本盘仍然是软件工程。模型调用只是系统中的一个模块。

## 学习内容

### Python

重点掌握：

* 类型标注
* dataclass与Pydantic
* 装饰器
* 上下文管理器
* 迭代器与生成器
* 异常处理
* asyncio
* async/await
* 并发控制
* pytest
* 日志系统

### FastAPI

* 路由与请求模型
* Dependency Injection
* 异步接口
* SSE流式输出
* WebSocket基础
* 中间件
* 全局异常处理
* 身份认证基础
* OpenAPI文档
* 服务生命周期管理

### 数据库与缓存

* PostgreSQL基本CRUD
* 索引和事务
* SQLAlchemy
* Redis缓存
* Redis分布式锁基础
* 对话记录、任务状态和用户配置的数据建模

### 计算机基础

每周保持：

* 数据结构与算法题
* HTTP、TCP基本原理
* 进程、线程、协程
* 数据库索引与事务
* Linux常用命令

## 阶段项目

完成一个不使用Agent框架的后端服务：

> **流式大模型对话服务**

功能包括：

* 多轮对话
* SSE流式返回
* PostgreSQL保存会话
* Redis缓存
* 请求超时与重试
* Token和费用统计
* Docker Compose启动
* pytest接口测试

## 验收标准

能够独立解释：

* 协程和线程的区别
* SSE与WebSocket的区别
* Redis与PostgreSQL各自适合存什么
* 为什么接口要设置Timeout
* 如何处理模型API限流和失败重试
* 如何避免一个慢请求阻塞其他请求

---

# 三、第3周：不依赖框架实现Agent Loop

不要一开始直接调用框架的`create_agent()`。先自己实现Agent核心循环。

## 学习内容

* System Prompt
* User Message
* Assistant Message
* Tool Message
* JSON Schema
* Structured Output
* Function Calling
* 工具注册
* 工具参数校验
* 工具执行结果回传
* 最大迭代次数
* 停止条件
* 错误恢复
* 上下文截断

## 自己实现的核心流程

```text
接收用户任务
    ↓
模型判断是否调用工具
    ↓
解析并校验工具参数
    ↓
执行工具
    ↓
将结果返回模型
    ↓
继续推理或输出最终答案
```

## 阶段项目

> **个人任务助理Agent**

至少实现以下工具：

* 计算器
* 网页信息查询
* 本地文件读取
* 数据库查询
* 待办事项管理
* 当前时间查询

同时实现：

* 工具白名单
* 参数类型校验
* 工具执行超时
* 最大调用次数
* 运行日志
* 每次执行的Token与费用统计

此时先做单Agent，不要做多Agent。

---

# 四、第4—5周：RAG与知识库系统

大量Agent岗位会涉及企业知识库、内部文档和业务数据。不能只会调用向量数据库。

## 1. 文档处理

学习：

* PDF、Word、Markdown、HTML解析
* 文本清洗
* Chunk切分
* 固定长度与语义切分
* 表格和标题结构保留
* Metadata设计
* 文档版本管理

## 2. 检索

学习：

* Embedding
* 余弦相似度
* Dense Retrieval
* BM25
* Hybrid Search
* Metadata Filter
* Query Rewrite
* Multi-query Retrieval
* Reranker
* Top-K选择

## 3.生成

学习：

* 引用来源
* Context拼接
* 上下文去重
* 长文档压缩
* 无答案拒答
* 防止无依据生成
* 多轮检索

## 4.评测

自己建立至少100条测试集，记录：

* Recall@K
* MRR
* Hit Rate
* 答案正确率
* 引用准确率
* 无答案识别率
* 平均延迟
* Token消耗

## 阶段项目

> **带引用和评测体系的企业知识库Agent**

必须包含：

* 文档上传与增量更新
* 混合检索
* Reranker
* 来源引用
* 无答案拒答
* 管理后台或简单页面
* 自动评测脚本
* 错误案例分析

不要只展示“上传PDF后可以聊天”。

---

# 五、第6—7周：LangGraph与状态化工作流

完成基础Agent Loop后，再进入框架。

LangGraph的重点不是调用模型，而是将复杂任务拆成节点、边和显式状态，并支持暂停、恢复、持久化、流式执行和人工审批。([Docs by LangChain][2])

## 学习内容

### Graph基础

* State
* Node
* Edge
* Conditional Edge
* Router
* Subgraph
* Retry Policy
* Checkpointer
* Store

### 执行控制

* Durable Execution
* Interrupt
* Resume
* Human-in-the-loop
* Timeout
* Retry
* Fallback
* 状态恢复
* 幂等性

### Memory

区分：

* 当前请求状态
* 短期会话记忆
* 长期用户记忆
* 业务数据库
* 检索知识库

不要把所有历史消息都直接塞入Prompt。LangGraph的持久化层区分Checkpointer和Store，分别服务于执行状态与跨交互记忆。([Docs by LangChain][3])

## 阶段项目

把前面的知识库Agent改造成状态图：

```text
用户输入
→ 意图分类
→ 是否需要检索
→ Query改写
→ 文档检索
→ Rerank
→ 答案生成
→ 事实检查
→ 输出或重新检索
```

增加：

* 中断后恢复
* 人工审批
* 错误重试
* 运行状态查询
* 流式返回中间进度
* 会话持久化

---

# 六、第8周：MCP与工具生态

MCP不是Agent框架，而是AI应用连接外部数据和工具的协议。官方规范将Server能力分为Resources、Prompts和Tools；Tools可以查询数据库、调用API或执行计算。([Model Context Protocol][4])

## 学习内容

* MCP Host
* MCP Client
* MCP Server
* Tools
* Resources
* Prompts
* Transport
* JSON-RPC基础
* Capability Negotiation
* 本地Server与远程Server
* OAuth与权限控制基础
* MCP Inspector

## 必做实践

自己开发两个MCP Server。

### Server 1：数据库分析服务

提供：

* 查看表结构
* 执行只读SQL
* 生成统计结果
* 导出CSV
* SQL白名单
* 查询超时
* 返回行数限制

### Server 2：个人办公工具

提供：

* 文件搜索
* 文档读取
* 日程查询
* 待办管理
* 报告保存

需要解决：

* 参数校验
* 工具权限
* 敏感字段脱敏
* 超时和错误返回
* 审计日志

---

# 七、第9周：多Agent，但控制复杂度

多Agent不是Agent数量越多越好。只有角色隔离、权限隔离或上下文隔离确有必要时才使用。

## 需要掌握的两种模式

### Manager模式

```text
主Agent
├── 检索Agent
├── 数据分析Agent
└── 报告生成Agent
```

主Agent统一负责调用子Agent。

### Handoff模式

当前Agent将控制权移交给更适合的Agent。OpenAI Agents SDK将Handoff用于不同专业Agent之间的任务委派。([OpenAI][5])

## 实践要求

对比：

* 单Agent方案
* 多Agent方案

至少分析：

* 成功率
* 延迟
* Token消耗
* 调试复杂度
* 错误传播
* 上下文污染

结论不必是多Agent更优。很多任务中，确定性工作流加一个Agent更稳定。

---

# 八、第10周：Agent评测与可观测性

这是最容易被初学者忽略、但最能拉开项目差距的部分。

百度当前智能体岗位已经明确要求建立效果、效率和资源成本一体化评测体系。([百度校园招聘][6])

## 1.评测指标

### 任务效果

* Task Success Rate
* 答案正确率
* 完成步骤数
* 工具选择准确率
* 工具参数准确率
* 引用准确率
* 人工介入率

### 系统效率

* 端到端延迟
* P50/P95/P99
* 模型调用次数
* 工具调用次数
* 重试次数
* Token消耗
* 单任务费用

### 稳定性

* 工具失败恢复率
* 超时率
* 死循环率
* 非法工具调用率
* 状态恢复成功率

## 2.评测方式

建立三层评测：

1. 单工具单元测试
2. 工作流节点测试
3. 端到端任务测试

测试集至少包含：

* 正常任务
* 模糊指令
* 信息不足
* 工具异常
* 权限不足
* 超长输入
* Prompt Injection
* 要求执行危险操作

## 3.可观测性

记录完整Trace：

```text
用户请求
→ Agent决策
→ 模型输入输出
→ 工具调用
→ 工具结果
→ 状态变化
→ 最终输出
```

OpenAI Agents SDK的Tracing会记录模型生成、工具调用、Handoff和Guardrail事件，可作为你设计自建Trace系统时的参考。([OpenAI][7])

---

# 九、第11周：安全、权限与Human-in-the-loop

Agent能够执行工具后，风险明显高于普通聊天应用。

## 学习内容

* Prompt Injection
* Indirect Prompt Injection
* 工具越权
* 敏感信息泄露
* SQL注入
* 路径穿越
* SSRF基础
* 工具白名单
* 最小权限原则
* 输入输出Guardrail
* 沙箱执行
* 审计日志

## 必须人工审批的操作

例如：

* 发送邮件
* 删除文件
* 修改数据库
* 创建订单
* 执行付款
* 运行Shell命令
* 向外部系统提交数据

Human-in-the-loop的典型模式是：执行不可逆操作前暂停工作流，等待用户审批，之后从保存的状态继续运行。([Docs by LangChain][8])

---

# 十、第12周：生产化部署与压测

## 服务能力

实现：

* SSE流式输出
* 请求取消
* Timeout
* Retry
* Circuit Breaker
* Rate Limit
* 队列和背压
* 会话隔离
* API鉴权
* 健康检查
* Graceful Shutdown

## 部署

使用：

* Docker Compose
* Nginx
* PostgreSQL
* Redis
* Agent服务
* MCP服务
* 监控服务

增加：

* GitHub Actions
* 自动测试
* 镜像构建
* 配置文件管理
* 日志轮转
* 环境变量与Secret管理

## 压测指标

记录：

* 并发用户数
* P50/P95/P99延迟
* 请求成功率
* Token吞吐
* 数据库连接数
* Redis命中率
* 模型API失败率
* 单请求成本

---

# 十一、两个简历项目

## 项目一：AI原生浏览器Agent

这是主项目，重点体现Agent编排、工具调用和系统工程。

### 功能定位

用户使用自然语言完成跨网页任务，例如：

* 搜索和比较多个页面
* 提取结构化信息
* 汇总网页证据
* 填写表单
* 下载和整理文件
* 执行多步网页任务
* 对高风险操作进行人工确认

### 技术模块

* Playwright浏览器控制
* DOM与Accessibility Tree解析
* 页面元素定位
* 任务规划与状态图
* 工具调用
* 页面内容压缩
* 短期和长期记忆
* Human-in-the-loop
* MCP工具接入
* 失败恢复
* Trace与评测

### 必须做的指标

* 任务完成率
* 平均执行步骤数
* 页面元素定位准确率
* 人工接管率
* 平均延迟
* Token消耗
* 失败恢复率

不要把项目写成“调用模型控制浏览器”，而要体现任务状态、可靠性和评测。

---

## 项目二：企业文档与数据分析Agent

这个项目体现RAG、SQL、报告生成和业务集成。

### 功能定位

统一处理：

* PDF、Word和网页文档
* 数据库表
* CSV/Excel数据
* 企业知识库
* 自然语言数据查询
* 自动生成分析报告

### 核心技术

* 混合检索与Rerank
* Text-to-SQL
* SQL安全执行
* 多步骤分析工作流
* MCP数据工具
* 引用和证据追踪
* 报告生成
* 人工审批
* 自动评测
* 成本与延迟统计

### 项目指标

* 检索Recall@K
* SQL执行正确率
* 报告事实准确率
* 引用准确率
* 任务完成率
* P95延迟
* 单任务成本

---

# 十二、面试准备同步进行

每周至少安排4—5小时，不要等项目完成后再准备。

## 编程与基础

* 链表、树、哈希表、堆
* BFS、DFS
* 二分查找
* 动态规划基础
* 并发与异步
* TCP、HTTP
* 数据库索引和事务
* Redis
* Linux

## Agent高频问题

需要能回答：

* Agent和普通Workflow有什么区别
* 什么任务不适合使用Agent
* 如何防止Agent死循环
* 如何选择工具
* 如何设计工具Schema
* Memory和RAG有什么区别
* 如何控制上下文长度
* 如何评估Agent
* 多Agent一定优于单Agent吗
* 如何处理Prompt Injection
* 如何恢复失败的长任务
* 如何降低Token成本
* 如何保证高风险操作安全
* MCP和Function Calling有什么区别

---

# 十三、学习优先级

## 第一优先级

* Python后端
* Tool Calling
* RAG
* LangGraph
* Agent Evaluation
* Docker部署
* 数据结构与算法

## 第二优先级

* MCP
* Redis与PostgreSQL
* Human-in-the-loop
* Tracing
* 安全与权限
* 多Agent

## 暂时不要投入过多时间

* 同时学习多个Agent框架
* Dify/Coze界面操作
* 复杂前端
* 纯Prompt技巧
* 从头训练大模型
* 多智能体“社会模拟”
* 没有业务目标的AutoGPT式Demo
* 为项目强行加入区块链、知识图谱等技术

---

# 十四、12周学习与产出表

| 周期    | 学习重点                     | 项目产出            |
| ----- | ------------------------ | --------------- |
| 第1—2周 | Python后端、数据库、Redis、异步    | 流式LLM服务         |
| 第3周   | Tool Calling、Agent Loop  | 无框架工具Agent      |
| 第4—5周 | RAG、混合检索、Rerank、评测       | 企业知识库Agent      |
| 第6—7周 | LangGraph、状态、Memory、HITL | 状态化Agent工作流     |
| 第8周   | MCP Client/Server        | 两个MCP Server    |
| 第9周   | 多Agent和Handoff           | 单Agent/多Agent对比 |
| 第10周  | Evaluation、Tracing、成本    | 自动评测与监控         |
| 第11周  | 安全、权限、Guardrail          | 审批和审计机制         |
| 第12周  | 部署、压测、CI/CD              | 可在线演示的完整系统      |

---

## 最终求职标准

完成路线后，你需要形成以下证据链：

```text
能写后端服务
→ 能实现Agent Loop
→ 能构建RAG
→ 能编排长任务
→ 能连接外部工具
→ 能处理状态和故障
→ 能评测效果
→ 能控制成本和风险
→ 能部署和压测
```

简历定位建议写成：

> **AI应用工程师 / AI Agent开发工程师**

而不是：

> 熟悉LangChain、Dify、Coze等Agent框架。

前者强调系统交付能力，后者容易被识别为框架调用型候选人。

[1]: https://jobs.bytedance.com/campus/m/position/detail/7591478820613671221?recomId=3ba5353b-eb47-11f0-86b2-043f72c0777e&sourceJobId=7559894479974500616&utm_source=chatgpt.com "AI Agent开发实习生-抖音平台产品"
[2]: https://docs.langchain.com/oss/python/langgraph/overview?utm_source=chatgpt.com "LangGraph overview - Docs by LangChain"
[3]: https://docs.langchain.com/oss/python/langgraph/persistence?utm_source=chatgpt.com "Persistence - Docs by LangChain"
[4]: https://modelcontextprotocol.io/docs/getting-started/intro?utm_source=chatgpt.com "What is the Model Context Protocol (MCP)?"
[5]: https://openai.github.io/openai-agents-python/handoffs/?utm_source=chatgpt.com "Handoffs - OpenAI Agents SDK"
[6]: https://talent.baidu.com/jobs/list?recommendCode=IS3TJS "百度校园招聘"
[7]: https://openai.github.io/openai-agents-python/tracing/?utm_source=chatgpt.com "Tracing - OpenAI Agents SDK"
[8]: https://docs.langchain.com/oss/python/langgraph/interrupts?utm_source=chatgpt.com "Interrupts - Docs by LangChain"
