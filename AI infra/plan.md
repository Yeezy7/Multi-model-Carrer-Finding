你学 AI Infra，不能按“运维 / DevOps”去学。对你来说，最合适的是：

> **大模型 / 多模态模型推理基础设施方向：模型部署、推理加速、显存优化、服务化、并发调度、性能评测。**

也就是从 **AI Infra 里的 inference serving** 切入，而不是一上来学 Kubernetes、CI/CD、SRE 那套。

---

## 1. 先明确你要学的 AI Infra 子方向

AI Infra 大致分 4 类：

| 子方向             | 内容                                                        | 是否适合你      |
| --------------- | --------------------------------------------------------- | ---------- |
| 模型推理部署          | vLLM、SGLang、TensorRT-LLM、Triton、ONNX、TensorRT             | **最适合**    |
| GPU / CUDA 性能优化 | CUDA kernel、FlashAttention、Triton kernel、Nsight Profiling | 适合进阶       |
| 训练基础设施          | 分布式训练、ZeRO、FSDP、Megatron、DeepSpeed                        | 可以了解，不建议主攻 |
| MLOps / DevOps  | K8s、CI/CD、监控、灰度、弹性伸缩                                      | 辅助，不建议你主攻  |

你现在应该先学第一类，逐步补第二类。因为你已有 YOLO、Qwen-VL、SigLIP、TensorRT、Docker、FastAPI 这些基础，直接转“大模型推理部署优化”更自然。

---

## 2. 学习主线：从“会部署”到“懂推理系统”

### 第一阶段：补推理基础，不要急着啃源码

你先把这些概念搞清楚：

| 概念                       | 必须理解到什么程度                                 |
| ------------------------ | ----------------------------------------- |
| Prefill / Decode         | 知道首 token 为什么慢，decode 为什么受 KV Cache 影响    |
| KV Cache                 | 知道显存占用如何随 batch、sequence length、layer 数增长 |
| Attention                | 知道 self-attention 的计算和显存瓶颈                |
| Batch / Dynamic batching | 知道为什么并发请求不能简单堆 batch                      |
| Quantization             | FP16、BF16、INT8、INT4、AWQ、GPTQ 各自影响         |
| Tensor Parallel          | 多卡部署大模型时怎么切权重                             |
| P95 / P99 latency        | 面试和项目里必须会说的服务指标                           |
| Throughput               | tokens/s、req/s、QPS 的区别                    |

这一步不要只看理论。你可以直接用 Qwen2.5-VL / Qwen3-VL / InternVL 做实验。

你要能回答这些问题：

> 同一个模型，为什么 batch size 变大后吞吐上升但延迟变高？
> 为什么长上下文会显著吃显存？
> 为什么 prefill 和 decode 的优化策略不一样？
> 为什么多模态模型的视觉 token 会拖慢推理？

这些比“会 LangChain”有含金量得多。

---

## 3. 第二阶段：掌握 4 个核心推理框架

你不用全都精通，但要形成主次。

### 3.1 vLLM：必须学

vLLM 是当前大模型推理部署最应该先学的框架之一，官方定位就是高吞吐、显存高效的 LLM 推理与服务引擎，核心能力包括 PagedAttention、continuous batching、OpenAI-compatible API 等。([vLLM][1])

你需要会：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

然后做这些实验：

| 实验               | 目标                         |
| ---------------- | -------------------------- |
| 单卡 vs 双卡部署       | 看显存、吞吐、延迟变化                |
| 不同 max_model_len | 看 KV Cache 对显存的影响          |
| 不同并发数            | 测 QPS、tokens/s、P95 latency |
| 不同输入长度           | 分析 prefill latency         |
| 不同输出长度           | 分析 decode throughput       |

你最终要写出一张 benchmark 表：

| 模型 | GPU | 并发 | 输入长度 | 输出长度 | P50 | P95 | tokens/s | 显存 |
| -- | --- | -: | ---: | ---: | --: | --: | -------: | -: |

这就是 AI Infra 简历的核心素材。

---

### 3.2 SGLang：建议学

SGLang 更偏高性能推理运行时和复杂 LLM 程序执行。它的文档明确提到 RadixAttention、prefix caching、prefill-decode disaggregation、speculative decoding、continuous batching、paged attention、多种并行和量化能力。([sgl-project.github.io][2])

你学 SGLang 的重点不是“又会一个框架”，而是理解：

| 能力                            | 对应意义                          |
| ----------------------------- | ----------------------------- |
| RadixAttention                | 多轮对话 / 共享前缀请求下复用 KV Cache     |
| Prefix caching                | RAG、Agent、多轮对话常用优化            |
| Structured output             | JSON / function calling 场景更稳定 |
| Speculative decoding          | 用小模型辅助大模型加速生成                 |
| Prefill-decode disaggregation | 高并发服务中的架构优化方向                 |

如果你以后投推理系统、LLM serving、AI Infra 岗，SGLang 是加分项。

---

### 3.3 TensorRT-LLM：进阶重点

TensorRT-LLM 是 NVIDIA 面向 LLM 推理优化的库，官方说明其目标是加速并优化 NVIDIA GPU 上的大语言模型推理。([NVIDIA Docs][3]) GitHub 介绍中也提到它包含 attention、GEMM、MoE 等自定义 kernel，以及 prefill-decode disaggregation、speculative decoding 等运行时优化。([GitHub][4])

你不需要一开始就深入源码，但要会：

| 内容          | 学到什么程度                            |
| ----------- | --------------------------------- |
| 模型转换        | HuggingFace → TensorRT-LLM engine |
| FP16 / BF16 | 基础部署                              |
| INT8 / INT4 | 了解量化部署                            |
| 多卡推理        | tensor parallel                   |
| benchmark   | latency、throughput、显存             |

这块和你之前 SigLIP ONNX + TensorRT 的项目能衔接。你可以把简历从“会 TensorRT”升级到“理解大模型推理引擎优化”。

---

### 3.4 Triton Inference Server：部署层要学

Triton 不是只服务 LLM，它更像生产环境里的模型服务框架。官方文档里 dynamic batching 的定义是把一个或多个推理请求动态合并成 batch，以提升吞吐。([NVIDIA Docs][5]) Triton 也支持 ensemble models，可以把多个模型组合成推理流水线。([NVIDIA Docs][6])

你学 Triton 是为了做“工程闭环”：

```text
前处理 → 图像编码模型 → 文本编码模型 → 后处理 → 返回结果
```

或者：

```text
OCR → VLM → 检索 → rerank → 答案生成
```

这对多模态项目特别有用。

---

## 4. 第三阶段：补系统基础，但不要学偏

你要补的系统基础按优先级排：

### 必学

| 模块        | 内容                                   |
| --------- | ------------------------------------ |
| Linux     | 进程、显存查看、磁盘、端口、日志、shell               |
| Docker    | 镜像构建、GPU 容器、nvidia-container-runtime |
| Python 服务 | FastAPI、异步请求、流式输出                    |
| 性能测试      | wrk、ab、locust、requests 并发脚本          |
| GPU 监控    | nvidia-smi、DCGM、显存、SM 利用率            |
| Profiling | PyTorch Profiler、Nsight Systems 基础   |

### 中期学

| 模块              | 内容                                                     |
| --------------- | ------------------------------------------------------ |
| C++             | 能读推理框架部分源码即可                                           |
| CUDA            | 理解 kernel、thread/block、shared memory、memory coalescing |
| Triton Language | 能写简单矩阵乘、layernorm、算子 benchmark                         |
| Kubernetes      | 会部署服务即可，不要先深挖运维                                        |

### 暂时别重学

| 内容           | 原因             |
| ------------ | -------------- |
| Java 后端      | 和你的方向关系不大      |
| 完整 DevOps 体系 | 容易把你带偏成平台运维    |
| 大规模分布式训练     | 门槛高，短期不如推理部署见效 |
| 从零写推理框架      | 时间成本过高         |

---

## 5. 适合你的 8 周学习计划

### 第 1–2 周：LLM 推理基础 + vLLM 部署

目标：把 Qwen / InternVL 跑成 OpenAI API 服务。

任务：

1. 部署 7B / 14B 模型。
2. 测单卡、双卡显存。
3. 写并发压测脚本。
4. 记录 P50、P95、tokens/s。
5. 分析输入长度、输出长度、并发数对性能的影响。

产出：

```text
Qwen2.5-7B / 14B vLLM 推理性能评测报告
```

---

### 第 3–4 周：多模态模型推理瓶颈分析

目标：把你的优势放进去。

任务：

1. 部署 Qwen-VL / InternVL。
2. 测不同图片分辨率下视觉 token 数量。
3. 测 prefill latency、decode latency。
4. 对比单图、多图、OCR 截图、图表图像。
5. 做视觉 token 裁剪或分辨率自适应策略。

产出：

```text
高分辨率 VLM 视觉 Token 裁剪与推理加速实验
```

这个项目非常适合你，不像普通 Agent 项目。

---

### 第 5–6 周：TensorRT / TensorRT-LLM / Triton

目标：补模型部署工程能力。

任务：

1. 把 SigLIP / CLIP 图像编码器导出 ONNX。
2. TensorRT FP16 构建 engine。
3. Triton 部署图像编码模型。
4. 开启 dynamic batching。
5. 对比 PyTorch / ONNXRuntime / TensorRT / Triton 的延迟和吞吐。

产出：

```text
多模态图文表征模型 TensorRT + Triton 服务化部署
```

---

### 第 7–8 周：做成完整 AI Infra 项目

目标：形成可写简历、可面试讲解的项目。

建议项目名：

> **面向多模态大模型的推理服务与性能优化平台**

核心功能：

1. 支持 vLLM / SGLang 启动 LLM / VLM 服务。
2. 支持单卡 / 双卡模型部署。
3. 自动压测不同并发、输入长度、输出长度。
4. 统计 TTFT、P95 latency、tokens/s、显存峰值。
5. 支持多模态输入，统计视觉 token 与 prefill latency。
6. 支持 FastAPI + Docker 一键启动。
7. 支持简单 Web 页面展示 benchmark 结果。

简历可以写成：

> 构建面向多模态大模型的推理服务与性能评测平台，基于 vLLM / SGLang 部署 Qwen-VL、InternVL 等模型，设计并发压测脚本统计 TTFT、P95 latency、tokens/s 与 GPU 显存占用；针对高分辨率图像输入导致视觉 token 过多、prefill 阶段耗时高的问题，实现视觉 token 裁剪与动态分辨率策略，在精度基本稳定的前提下降低端到端推理延迟。

---

## 6. 你需要掌握的面试问题

学完以后，你至少要能回答这些：

### 推理系统问题

```text
1. LLM 推理为什么分 prefill 和 decode？
2. KV Cache 的显存占用怎么估算？
3. continuous batching 和普通 batching 有什么区别？
4. PagedAttention 解决了什么问题？
5. prefix caching 适合哪些场景？
6. speculative decoding 为什么能加速？
7. tensor parallel 和 pipeline parallel 区别是什么？
8. 为什么 P95 latency 比平均延迟更重要？
```

### 多模态推理问题

```text
1. VLM 为什么比 LLM 推理更慢？
2. 图片分辨率和视觉 token 数量是什么关系？
3. OCR 截图、图表、自然图像的推理瓶颈一样吗？
4. 如何在不明显损失精度的情况下降低视觉 token？
5. 多图输入为什么容易导致显存爆炸？
```

### 工程部署问题

```text
1. PyTorch、ONNXRuntime、TensorRT 的区别是什么？
2. Triton dynamic batching 什么时候有用？
3. 如何设计一个模型推理服务的 benchmark？
4. 如何定位 GPU 利用率低的问题？
5. 如何优化高并发推理服务？
```

---

## 7. 推荐你的技能栈写法

你简历里可以改成这种：

```text
AI Infra / 推理部署：
熟悉大模型推理服务流程，掌握 vLLM、SGLang、TensorRT、ONNXRuntime、Triton Inference Server 等工具；
理解 Prefill/Decode、KV Cache、Continuous Batching、Prefix Caching、Tensor Parallel、量化推理等机制；
具备 LLM/VLM 服务化部署、性能压测、显存分析、P95 延迟优化与 Docker 化部署经验。
```

项目关键词应该是：

```text
vLLM / SGLang / TensorRT-LLM / Triton / ONNX / TensorRT
Prefill latency / decode latency / TTFT / P95 / tokens/s
KV Cache / visual token pruning / batch inference / prefix caching
multi-GPU serving / Docker / FastAPI / benchmark
```

---

## 8. 最实际的学习顺序

你不要按课程从头学。按这个顺序做：

```text
vLLM 部署 Qwen-7B
        ↓
写压测脚本，统计延迟和吞吐
        ↓
换成 Qwen-VL / InternVL，分析视觉 token 瓶颈
        ↓
做 token pruning / 分辨率自适应优化
        ↓
补 TensorRT / Triton，把 CLIP/SigLIP 部署成高性能服务
        ↓
整理 benchmark 表和项目 README
        ↓
再补 CUDA / Triton kernel / Nsight
```

你的目标不是“成为运维工程师”，而是成为：

> **能把多模态大模型部署起来、测清楚瓶颈、做出推理优化的算法工程师。**

[1]: https://docs.vllm.ai/?utm_source=chatgpt.com "vLLM Documentation"
[2]: https://sgl-project.github.io/?utm_source=chatgpt.com "SGLang Documentation — SGLang"
[3]: https://docs.nvidia.com/tensorrt-llm/index.html?utm_source=chatgpt.com "NVIDIA TensorRT-LLM"
[4]: https://github.com/NVIDIA/TensorRT-LLM?utm_source=chatgpt.com "NVIDIA/TensorRT-LLM"
[5]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html?utm_source=chatgpt.com "Dynamic Batching & Concurrent Model Execution"
[6]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html?utm_source=chatgpt.com "Ensemble Models — NVIDIA Triton Inference Server"
