# AI Infra 学习路线（大模型基础设施方向）

> **目标岗位**：AI Infra / LLM Infra / 推理优化 / 训练框架工程师
> **核心方向**：大模型推理部署 + 训练框架 + GPU 编程 + 系统底层
> **时间规划**：16 周（可根据实际进度调整）
> **前置基础**：PyTorch、Python、Docker 基础

---

## 一、岗位能力画像

根据字节、腾讯、阿里、华为、小米、美团、百度等大厂 AI Infra 岗位 JD 提炼：

### 必备技能（Must Have）

| 技能 | 具体内容 | 重要度 |
|------|---------|-------|
| **Python** | 高质量工程代码、异步编程、类型注解 | ★★★★★ |
| **C++** | 推理框架源码阅读、算子开发、性能关键路径 | ★★★★★ |
| **PyTorch** | 深入理解 autograd、Module、分布式原语、torch.compile | ★★★★★ |
| **CUDA 编程** | kernel 编写/优化、thread/block/grid、shared memory、memory coalescing | ★★★★★ |
| **Linux 系统编程** | 进程/线程、内存管理、文件 I/O、网络编程、Shell | ★★★★☆ |
| **大模型训练流程** | 数据并行、模型并行、流水线并行、混合精度训练 | ★★★★☆ |

### 核心技能（Core Skills）

| 技能 | 具体内容 | 重要度 |
|------|---------|-------|
| **推理框架** | vLLM / SGLang / TensorRT-LLM / Triton Inference Server | ★★★★★ |
| **分布式训练框架** | DeepSpeed ZeRO / Megatron-LM / FSDP / ColossalAI | ★★★★☆ |
| **模型量化与压缩** | FP16/BF16/INT8/INT4、GPTQ、AWQ、GGUF、蒸馏、剪枝 | ★★★★☆ |
| **GPU 性能分析** | Nsight Systems / Nsight Compute / CUPTI / DCGM | ★★★★☆ |
| **集群调度** | Kubernetes、Slurm、GPU 调度策略、弹性伸缩 | ★★★☆☆ |

### 加分项（Nice to Have）

| 技能 | 具体内容 |
|------|---------|
| **RDMA/高速网络** | InfiniBand、NCCL 通信优化、Ring AllReduce |
| **自定义算子开发** | Triton Language、TVM、算子融合、FlashAttention 实现 |
| **模型编译优化** | torch.compile、TorchInductor、Triton backend |
| **AI 编译器** | TVM、MLIR、计算图优化、内存规划 |
| **向量数据库/RAG** | FAISS / Milvus / Qdrant、检索基础设施 |

---

## 二、16 周学习路线

### 第 1-2 周：C++ 基础 + Linux 系统编程

**目标**：补齐 C++ 能力，能读懂推理框架源码；掌握 Linux 系统编程基础

**C++ 必学内容**：
- [ ] 指针、引用、智能指针（unique_ptr/shared_ptr/weak_ptr）
- [ ] STL 容器（vector/map/unordered_map）与算法
- [ ] 多态、虚函数、模板基础
- [ ] 右值引用与移动语义
- [ ] Lambda 表达式
- [ ] 头文件/编译/链接基础（理解 .h 和 .cpp 关系）
- [ ] GDB 基础调试

**Linux 系统编程必学内容**：
- [ ] 进程与线程（fork、pthread、std::thread）
- [ ] 内存管理（malloc/free、mmap、虚拟内存概念）
- [ ] 文件 I/O（open/read/write、文件描述符）
- [ ] 网络编程基础（socket、TCP/IP 概念）
- [ ] Shell 脚本（grep/awk/sed/管道/重定向）
- [ ] 进程管理（ps/top/htop/kill/nice）
- [ ] 性能工具（strace/ltrace/perf top）

**动手实验**：
- [ ] 用 C++ 写一个简单的矩阵乘法，用 GDB 调试
- [ ] 用 C++ 写一个多线程 Producer-Consumer 队列
- [ ] 写一个 Shell 脚本自动监控 GPU 状态并记录日志
- [ ] 用 strace 跟踪一个 Python 进程的系统调用

**产出**：
```
C++ 基础练习代码 + Linux 系统编程笔记
```

---

### 第 3-4 周：CUDA 编程基础

**目标**：掌握 CUDA 编程模型，能编写和调试简单 CUDA kernel

**理论学习**：
- [ ] GPU 硬件架构（SM、Warp、Thread Block、Grid）
- [ ] CUDA 编程模型（kernel launch、threadIdx/blockIdx）
- [ ] 内存层次（Global Memory、Shared Memory、Registers、Constant Memory）
- [ ] Memory Coalescing 原则
- [ ] Warp Divergence 问题
- [ ] Occupancy 概念与优化
- [ ] CUDA Streams 与异步执行
- [ ] Unified Memory 基础

**动手实验**：
- [ ] 配置 CUDA 开发环境（nvcc、Nsight）
- [ ] 写 vector add kernel
- [ ] 写矩阵乘法 kernel（naive → shared memory 优化）
- [ ] 写 layernorm kernel
- [ ] 用 Nsight Systems 分析 kernel 执行 timeline
- [ ] 用 ncu (Nsight Compute) 分析 kernel 的 memory throughput

**必答面试题**：
1. CUDA 中 thread、block、grid 的关系是什么？
2. Shared Memory 的作用是什么？为什么能加速？
3. 什么是 Memory Coalescing？为什么重要？
4. Warp Divergence 是什么？如何避免？
5. 如何计算一个 CUDA kernel 的 occupancy？

**产出**：
```
CUDA kernel 练习代码 + 性能分析笔记
```

---

### 第 5-6 周：LLM 推理基础 + vLLM 部署

**目标**：理解 LLM 推理原理，跑通 vLLM 部署，建立性能评测基线

**理论学习**：
- [ ] Prefill / Decode 阶段区别与瓶颈分析
- [ ] KV Cache 原理与显存占用计算
- [ ] Attention 计算瓶颈（compute-bound vs memory-bound）
- [ ] Continuous Batching vs Static Batching
- [ ] PagedAttention 核心思想
- [ ] 吞吐(tokens/s) vs 延迟(latency) vs QPS 区别
- [ ] TTFT（Time To First Token）定义与影响因素

**动手实验**：
- [ ] vLLM 部署 Qwen2.5-7B-Instruct（单卡）
- [ ] vLLM 部署 Qwen2.5-14B-Instruct（双卡 tensor parallel）
- [ ] 编写并发压测脚本（locust 或 asyncio）
- [ ] 测量不同并发数(1/4/8/16/32)下的 P50/P95 latency
- [ ] 测量不同 max_model_len(2048/4096/8192)下的显存占用
- [ ] 测量不同输入长度(128/512/1024/2048 tokens)下的 TTFT
- [ ] 输出完整 benchmark 表格

**必答面试题**：
1. LLM 推理为什么分 prefill 和 decode？各阶段瓶颈是什么？
2. KV Cache 的显存占用如何估算？和哪些因素相关？
3. Continuous Batching 解决了什么问题？
4. PagedAttention 为什么能提高显存利用率？
5. 为什么 batch size 增大后吞吐上升但延迟也增加？

**产出**：
```
vLLM 推理性能评测报告（含 benchmark 表格）
```

---

### 第 7-8 周：SGLang + 多模态推理瓶颈分析

**目标**：掌握 SGLang 核心特性，深入分析 VLM 推理瓶颈

**理论学习**：
- [ ] SGLang 架构与 RadixAttention
- [ ] Prefix Caching 原理与适用场景
- [ ] Structured Output (JSON mode / function calling)
- [ ] Speculative Decoding 原理
- [ ] VLM 视觉 Token 与 Prefill 延迟关系
- [ ] 高分辨率图像对推理性能的影响

**动手实验**：
- [ ] SGLang 部署 Qwen2.5-7B，对比 vLLM 性能
- [ ] 测试 RadixAttention 在多轮对话场景的加速效果
- [ ] 部署 Qwen2.5-VL-7B / InternVL2.5-8B
- [ ] 分析不同图片分辨率下视觉 token 数量变化
- [ ] 对比单图/多图/OCR 截图/图表的 prefill latency
- [ ] 实现视觉 token 裁剪策略（限制最大 token 数）

**必答面试题**：
1. SGLang 的 RadixAttention 和 vLLM 的 Prefix Caching 有什么区别？
2. Prefix Caching 适合哪些场景？
3. VLM 为什么比纯 LLM 推理更慢？瓶颈在哪？
4. 视觉 token 裁剪/动态分辨率策略如何影响精度和速度？
5. Speculative Decoding 为什么能加速？

**产出**：
```
SGLang vs vLLM 对比评测 + VLM 视觉 Token 优化实验报告
```

---

### 第 9-10 周：TensorRT-LLM + 量化 + GPU 性能分析

**目标**：掌握 NVIDIA 推理优化工具链，深入量化和性能分析

**理论学习**：
- [ ] TensorRT-LLM 架构与优化 Pass
- [ ] FP16 / BF16 / INT8 / INT4 量化原理
- [ ] GPTQ / AWQ / GGUF 量化方法区别
- [ ] FlashAttention 算法原理
- [ ] 算子融合(Operator Fusion)基本思想
- [ ] CUDA Kernel 性能分析方法论

**动手实验**：
- [ ] TensorRT-LLM 部署 Qwen2.5-7B（FP16）
- [ ] TensorRT-LLM INT8/INT4 量化部署对比
- [ ] 编写 TensorRT-LLM vs vLLM benchmark 脚本
- [ ] 用 Nsight Systems 分析推理 timeline（看 prefill/decode 分布）
- [ ] 用 Nsight Compute 分析 kernel 的 compute/memory 瓶颈
- [ ] 用 Triton Language 写 layernorm kernel（体验编译优化）

**必答面试题**：
1. TensorRT-LLM 相比 vLLM 有什么优势？
2. FP16 / INT8 / INT4 量化对精度和速度各有什么影响？
3. GPTQ 和 AWQ 的核心区别是什么？
4. FlashAttention 为什么能加速 Attention 计算？
5. 如何用 Nsight 定位 GPU 推理性能瓶颈？

**产出**：
```
TensorRT-LLM 部署与量化实验报告 + 性能对比表格
```

---

### 第 11-12 周：分布式训练框架（DeepSpeed / Megatron-LM / FSDP）

**目标**：掌握大模型分布式训练的核心框架和并行策略

**理论学习**：
- [ ] 数据并行（DDP）原理与局限
- [ ] DeepSpeed ZeRO Stage 1/2/3 区别
- [ ] Megatron-LM 的 Tensor Parallel + Pipeline Parallel
- [ ] PyTorch FSDP 原理
- [ ] 混合精度训练（AMP）与 Loss Scaling
- [ ] 梯度累积(Gradient Accumulation)
- [ ] 梯度检查点(Gradient Checkpointing)与显存换时间

**动手实验**：
- [ ] 用 DeepSpeed ZeRO-2 微调一个 7B 模型
- [ ] 用 FSDP 微调一个 7B 模型，对比 DeepSpeed
- [ ] 分析不同 ZeRO Stage 的显存占用
- [ ] 实现梯度检查点，观察显存和速度 trade-off
- [ ] 用 Megatron-LM 跑一个小规模 TP 实验（2-4 卡）
- [ ] 训练时监控 GPU 利用率、显存、通信带宽

**必答面试题**：
1. DeepSpeed ZeRO Stage 1/2/3 分别优化了什么？
2. Tensor Parallel 和 Pipeline Parallel 各自适用什么场景？
3. FSDP 和 DeepSpeed ZeRO-3 有什么异同？
4. 梯度检查点为什么能省显存？代价是什么？
5. 混合精度训练中 Loss Scaling 的作用是什么？

**产出**：
```
DeepSpeed vs FSDP 分布式训练对比实验报告
```

---

### 第 13-14 周：Triton Inference Server + 集群调度

**目标**：掌握生产级模型服务框架，了解集群调度基础

**理论学习**：
- [ ] Triton Inference Server 架构
- [ ] Dynamic Batching 原理与配置
- [ ] Ensemble Models 与 Pipeline 设计
- [ ] Model Warmup 与缓存策略
- [ ] gRPC vs HTTP 推理接口
- [ ] Kubernetes 基础（Pod、Deployment、Service、GPU 资源调度）
- [ ] Slurm 基础（作业提交、GPU 分配）
- [ ] GPU 调度策略（MIG、Time-slicing、vGPU）

**动手实验**：
- [ ] Triton 部署 CLIP/SigLIP 图像编码模型
- [ ] 开启 Dynamic Batching，对比静态 batch 性能
- [ ] 构建 Ensemble Pipeline：OCR → VLM → 后处理
- [ ] Docker 打包 Triton + 模型 + 配置
- [ ] 用 K8s 部署一个推理服务（minikube 或 kind）
- [ ] 编写 K8s Deployment yaml，配置 GPU 资源请求
- [ ] 测试 K8s 环境下的模型服务扩缩容

**必答面试题**：
1. Triton Dynamic Batching 什么时候有用？怎么配置？
2. Triton Ensemble 和简单串联调用有什么区别？
3. K8s 中如何调度 GPU 资源？
4. 如何设计一个支持多模态输入的推理服务架构？
5. 模型更新时如何做到零停机切换？

**产出**：
```
Triton 多模型推理服务部署 + K8s 调度 Demo
```

---

### 第 15-16 周：RDMA/高速网络 + 完整项目 + 简历包装

**目标**：了解通信优化基础，整合所有技能构建完整项目

**RDMA / 高速网络学习**：
- [ ] NCCL 通信原理（Ring AllReduce、Tree AllReduce）
- [ ] InfiniBand / RoCE 基本概念
- [ ] GPU 间通信带宽瓶颈分析
- [ ] 通信与计算 overlap 策略

**完整项目设计**：

> **面向多模态大模型的 AI Infra 推理与训练平台**

```
┌──────────────────────────────────────────────────────────┐
│                    AI Infra 平台                           │
├──────────────────────────────────────────────────────────┤
│  [推理层]  vLLM / SGLang / TensorRT-LLM / Triton          │
│  [训练层]  DeepSpeed ZeRO / FSDP / 混合精度 / 梯度检查点      │
│  [优化层]  量化 / KV Cache / Prefix Caching / TP / PP      │
│  [调度层]  K8s GPU 调度 / 弹性伸缩                           │
│  [通信层]  NCCL / 多卡通信优化                                │
│  [服务层]  FastAPI / gRPC / 流式输出                         │
│  [监控层]  显存 / 延迟 / 吞吐 / P95 / GPU利用率              │
│  [评测层]  自动化 benchmark / 多维度对比                      │
└──────────────────────────────────────────────────────────┘
```

**实现任务**：
- [ ] 推理模块：vLLM/SGLang/TensorRT-LLM 部署 LLM/VLM
- [ ] 训练模块：DeepSpeed/FSDP 微调 7B 模型
- [ ] 优化模块：量化、KV Cache 优化、Continuous Batching
- [ ] 服务模块：FastAPI + Docker 一键启动
- [ ] 监控模块：GPU 显存/利用率/延迟实时监控
- [ ] 评测模块：自动化 benchmark 脚本
- [ ] 完整 README + 架构图 + 性能对比表格

**简历写法**：
```
AI Infra / 推理与训练基础设施：
- 熟悉大模型推理服务架构，掌握 vLLM、SGLang、TensorRT-LLM、Triton Inference Server 等框架
- 理解 DeepSpeed ZeRO、FSDP、Megatron-LM 等分布式训练框架，具备 7B+ 模型微调经验
- 掌握 CUDA 编程基础，能编写/调试 CUDA kernel，熟悉 Nsight Systems/Compute 性能分析
- 理解 Prefill/Decode、KV Cache、Continuous Batching、Prefix Caching、PagedAttention 等机制
- 掌握 Tensor Parallel、Pipeline Parallel 等多卡并行策略，了解 NCCL 通信优化
- 了解 FP16/INT8/INT4 量化（GPTQ/AWQ）、FlashAttention、算子融合等优化技术
- 具备 C++ 基础，能阅读推理框架部分源码
- 熟悉 Linux 系统编程、Docker 容器化、K8s 基础调度、FastAPI 服务开发
```

---

## 三、每周时间分配建议

```
每天 6 小时：
├── 1.5h：理论学习（论文/文档/源码阅读）
├── 3h：动手实验（编码/部署/压测/优化）
├── 1h：整理笔记 + 面试问题
└── 0.5h：复盘当天 + 更新进度

每天 4 小时：
├── 1h：理论学习
├── 2h：动手实验
└── 1h：整理 + 复盘
```

---

## 四、关键面试问题清单

### C++ / 系统编程
1. 智能指针 shared_ptr 和 unique_ptr 的区别？循环引用怎么解决？
2. 虚函数的实现原理（vtable）？
3. C++ 中 move 语义解决了什么问题？
4. Linux 中进程和线程的区别？协程呢？
5. TCP 三次握手、四次挥手的过程？
6. 什么是虚拟内存？页表的作用？

### CUDA / GPU 编程
1. CUDA 中 thread、block、grid 的关系？
2. Shared Memory 的作用？为什么能加速？
3. 什么是 Memory Coalescing？
4. Warp Divergence 是什么？如何避免？
5. 如何计算 CUDA kernel 的 occupancy？
6. CUDA Stream 的作用？如何实现异步执行？

### 推理系统
1. LLM 推理为什么分 prefill 和 decode？
2. KV Cache 的显存占用如何估算？
3. Continuous Batching 和普通 Batching 的区别？
4. PagedAttention 解决了什么问题？
5. Prefix Caching 适合哪些场景？
6. Speculative Decoding 为什么能加速？
7. Tensor Parallel 和 Pipeline Parallel 区别？
8. 为什么 P95 latency 比平均延迟更重要？

### 分布式训练
1. DeepSpeed ZeRO Stage 1/2/3 分别优化了什么？
2. FSDP 和 DeepSpeed ZeRO-3 的异同？
3. 梯度检查点为什么能省显存？代价是什么？
4. 混合精度训练中 Loss Scaling 的作用？
5. Tensor Parallel 和 Pipeline Parallel 各自适用什么场景？
6. AllReduce 在分布式训练中的作用？

### 工程部署
1. PyTorch、ONNXRuntime、TensorRT 的区别？
2. Triton Dynamic Batching 什么时候有用？
3. 如何设计模型推理服务的 benchmark？
4. 如何定位 GPU 利用率低的问题？
5. K8s 中如何调度 GPU 资源？
6. 如何优化高并发推理服务？

---

## 五、学习资源

| 资源 | 用途 |
|------|------|
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | GPU 编程核心 |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/) | 性能分析 |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/) | Kernel 级分析 |
| [vLLM 文档](https://docs.vllm.ai/) | 推理框架核心 |
| [SGLang 文档](https://sgl-project.github.io/) | 高性能推理运行时 |
| [TensorRT-LLM 文档](https://docs.nvidia.com/tensorrt-llm/) | NVIDIA 推理优化 |
| [Triton 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/) | 模型服务框架 |
| [DeepSpeed 文档](https://www.deepspeed.ai/tutorials/) | 分布式训练 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 大规模训练 |
| [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 原生分布式 |
| [FlashAttention 论文](https://arxiv.org/abs/2205.14135) | Attention 优化 |
| [PagedAttention 论文](https://arxiv.org/abs/2309.06180) | KV Cache 管理 |

---

## 六、学习顺序（一句话版）

```
C++ 基础 + Linux 系统编程
    ↓
CUDA 编程（kernel 编写 + Nsight 分析）
    ↓
vLLM 部署 Qwen-7B + 写压测脚本
    ↓
SGLang 对比 + VLM 视觉 token 瓶颈分析
    ↓
TensorRT-LLM 量化部署 + GPU 性能分析
    ↓
DeepSpeed/FSDP 分布式训练 7B 模型
    ↓
Triton 服务化 + K8s 调度基础
    ↓
RDMA/NCCL 通信优化（了解）
    ↓
整合完整 AI Infra 项目
    ↓
整理 benchmark + 简历包装
```

**最终目标**：成为既懂推理部署优化、又有训练框架能力、还能写 CUDA kernel 的 AI Infra 工程师。
