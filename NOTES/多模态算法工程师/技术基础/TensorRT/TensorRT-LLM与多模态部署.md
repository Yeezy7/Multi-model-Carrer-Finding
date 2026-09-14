---
learned: false
---

# TensorRT-LLM 与多模态部署

> 本模块索引见 [TensorRT详解](TensorRT详解.md)

上一章讲的是"一个静态模型图如何压榨精度与性能"。但 LLM/VLM 部署的本质不是"跑一张图"，而是"跑一个自回归解码循环"——每生成一个 token 都要管理 KV cache、做 attention、采样。TensorRT-LLM 就是 NVIDIA 为这个循环量身打造的框架：在原生 TensorRT 的图优化之上，把 KV cache 分页、连续批处理、采样、多卡并行全部内建。本章讲清它是什么、核心能力、与 vLLM 的取舍、VLM 多模态部署的完整管线，以及性能 benchmark 方法论。

## 一、TensorRT-LLM 是什么

TensorRT-LLM 是 NVIDIA 于 2023 年开源的 LLM/VLM 推理框架，底层仍是 TensorRT，但专门面向**自回归解码**这一动态过程：

- **构建层**：`trtllm-build` 把 HuggingFace 权重 + 模型配置编译成针对特定 GPU 型号的静态 engine（含精心手写的 CUDA 算子：FlashAttention、fused GQA、量化 GEMM 等）；
- **运行时层**：C++ 运行时负责 engine 之外的动态逻辑——KV cache 分配与分页、in-flight batching 调度、采样（temperature/top-p/top-k/beam search）、多 GPU 通信。

**与原生 TensorRT 的关系**（一句话）：

> 原生 TRT 优化"一张静态图"（CNN、视觉塔单次前向）；TRT-LLM 在 TRT 之上优化"一个循环"（自回归解码 + 动态调度）。

具体地：

| 维度 | 原生 TensorRT | TensorRT-LLM |
| --- | --- | --- |
| 输入形态 | 静态输入 shape（或固定 range） | 动态序列长度，运行时逐 token 扩展 |
| KV cache | 不感知（无状态） | 内建 PagedAttention 管理 |
| 批处理 | 固定 batch | in-flight batching（token 级调度） |
| 采样 | 需要用户自己实现 | 内建（温度/top-p/beam） |
| 多卡 | 需手工切分图 | `--tp_size` / `--pp_size` 一键 |
| 典型用途 | CNN、视觉塔、固定流程 | LLM / VLM 生成 |

## 二、核心能力（重点）

### 2.1 PagedAttention（KV cache 分页）

KV cache 是解码期显存大头：长度为 $L$、$N_{\text{layers}}$ 层、$H$ 个 KV 头、每头维度 $D$ 时：

$$\text{KV 显存} = N_{\text{layers}} \times 2 \times H \times D \times L \times \text{dtype\_bytes}$$

朴素实现为每个序列预分配 `max_seq_len` 的连续显存，短序列浪费、长序列碎片化。**PagedAttention** 把 KV cache 切成固定大小"页"（如每页 64 个 token 的块），用页表按需分配，机制类比操作系统虚拟内存：

- 消除碎片：只有真正用到的页才分配；
- 高利用率：空闲页可被任意序列复用；
- 吞吐直接受益：同样显存能跑更大的 batch。

（原理细节与显存计算见 [KV Cache原理与实现](../KVCache/KV Cache原理与实现.md)。）

### 2.2 In-flight batching（连续批处理）

传统 static batching 以"序列"为单位：一批里某个序列生成了 `max_new_tokens` 前，整批一起等。**In-flight batching** 以"token"为单位调度：任一序列每生成一个 token 就参与一次调度，生成完的序列立即让出槽位给新请求。效果：GPU 永远在处理"这一时刻还在生成的 token"，利用率大幅提升，代价是 runtime 需要维护每个序列的状态（KV cache 页、已生成 token、采样状态）——这正是"运行时在引擎外管理"的原因。

### 2.3 KV cache 量化

长上下文 + 大 batch 时 KV cache 常占显存 50%+，可量化到 INT8/FP8 减半（详见 [精度与量化](精度与量化.md) 6.2）：

```text
trtllm-build --kv_cache_quant_mode fp8 ...   # 或 int8，配合对应 GPU
```

风险是误差沿序列长度累积，长上下文必须实测。

### 2.4 FP8 / 低精度推理

Hopper+ 上支持 FP8（E4M3）权重/激活/KV，Blackwell 支持 FP4。TRT-LLM 把 FP8 量化也做进构建流程（`--use_fp8`），无需外部校准工具。低精度让相同显存塞下更大模型或更长上下文，是"大模型单卡跑得动"的关键手段之一（详见 [FP8与TF32](../参数精度/FP8与TF32.md)）。

### 2.5 Speculative decoding（投机解码）

自回归每步只出一个 token、受限于访存带宽。投机解码用小 draft 模型一次性生成 $k$ 个候选 token，大模型并行验证：对了就一次接受 $k$ 个，错了回退到首个错误位置。TRT-LLM 内建对 Medusa 等投机头的支持，短 prompt 场景可显著降低延迟。

### 2.6 多 GPU：TP / PP

- **张量并行（TP）**：把注意力头 / FFN 矩阵按列/行切到多卡，每步同步（NCCL all-reduce），单 batch 延迟低；
- **流水线并行（PP）**：按层切，卡间只传激活，适合跨节点，但有空泡（bubble）；
- 用法：`trtllm-build --tp_size 2 --pp_size 2 ...`，运行时自动做通信与 KV cache 分布。

### 2.7 运行时在引擎外管理

engine 是"静态计算图 + 权重"，不含任何序列状态。KV cache 的分配/复用、页表、采样参数、beam search 树、调度队列**全部在运行时层（C++ runtime / Python 封装）**：

- 同一个 engine 换不同 SamplingParams（温度、top-p）无需重编译；
- 支持流式输出（逐 token 吐给客户端）；
- engine 与运行时解耦 → engine 只负责"给定 KV 与输入，算下一个 logit"。

## 三、与 vLLM 对比

| 维度 | TensorRT-LLM | vLLM |
| --- | --- | --- |
| 底层方案 | TensorRT engine + 自研 CUDA 算子（编译型） | 自研 PagedAttention CUDA kernel（即时编译型，vLLM 源于 PagedAttention 论文） |
| 延迟/吞吐 | 峰值优化充分，官方 benchmark 常领先 | 与 TRT-LLM 接近，社区评测互有胜负 |
| 显存 | PagedAttention + KV 量化（INT8/FP8/FP4） | PagedAttention + KV 量化 + chunked prefill |
| 易用性 | **每 GPU 型号需 `trtllm-build` 重新构建 engine**，构建耗时（分钟~小时） | 开箱即用，首次运行自动编译 kernel 并缓存 |
| 模型支持 | 靠 NVIDIA 逐个适配，新模型有滞后 | 社区贡献，主流新模型通常当天可跑 |
| 社区生态 | 背靠 NVIDIA，企业级支持、与 TensorRT 工具链一体 | 社区极活跃、文档多、研究友好 |
| 硬件绑定 | 强（engine 绑定 GPU 算力/驱动/CUDA） | 弱（运行时按卡选 kernel，但同样仅 NVIDIA GPU） |
| 适用场景 | 生产环境、追求极致性能、多卡集群 | 快速验证、模型快速迭代、研究实验 |

**选型建议**：

- 已经上了 NVIDIA 全家桶（Triton/GPU 集群/监控），追求吞吐与低延迟 → TRT-LLM；
- 需要频繁换模型、验证新架构、快速上线 → vLLM（先跑起来，必要时再为生产单独优化 TRT-LLM）；
- 两者都基于 PagedAttention，性能差距在 10% 量级，工程权衡往往比峰值性能更重要。

## 四、多模态 VLM 部署（重点）

### 4.1 VLM 的结构回顾

```
             ┌─────────────┐      ┌──────────┐      ┌─────────────┐
   图像 ────►│  视觉塔 ViT/ │─────►│  投影层   │─────►│             │
   (pixels)  │  SigLIP     │      │(MLP/Linear)│     │    LLM      │
             └─────────────┘      └──────────┘      │(自回归生成)  │
   文本 ────► Tokenizer ──► 文本 token ────────────►│             │
                                                    └─────────────┘
```

视觉塔把图像编码为视觉特征，投影层映射到 LLM 的 embedding 维度，与文本 token 拼接后一起进入 LLM。关键数字：一张 336×336 图 → 576 个视觉 token（LLaVA-1.5 的 ViT-L 14×14 patch），512×512 图（SigLIP 16×16 patch）→ 1024 个 token。

### 4.2 TRT-LLM 的 VLM 支持

TRT-LLM 对 VLM 的处理是**"引擎外预处理 + 引擎内生成"**的分工：

- **图像预处理 + 视觉塔前向**：在引擎外完成——用 HuggingFace processor 做 resize/crop/归一化/patchify，视觉塔（ViT/SigLIP）用 PyTorch FP16 前向，或把视觉塔单独编译成原生 TensorRT engine 提速；
- **LLM 部分**：用 TRT-LLM 构建的引擎，视觉 token 作为 prompt 前缀（prefix）进入 prefill；
- 官方支持模型（随版本演进）：LLaVA / LLaVA-Next、VILA、NVILA、Phi-3-vision、Qwen-VL 等，构建时传入含视觉配置的 HF 模型目录即可。

为什么视觉塔放引擎外：① 视觉塔前向**无状态、无自回归**，原生 TRT 或 HF 就能跑，不需要 TRT-LLM 的解码调度；② 图像预处理与数据增强逻辑易变，放 Python 侧改起来快；③ 视觉 token 是一次性产生的，不需要参与逐 token 调度。

### 4.3 视觉 token 的 prefill 特点

VLM 推理与纯文本 LLM 的最大区别在 **prefill 阶段**：

- 一次请求就注入几百~上千个视觉 token（+ 系统提示 + 历史对话），prefill 极长；
- prefill 是计算密集（并行度高），decode 是访存密集（带宽受限）；
- 长 prefill 直接拉高 **TTFT**（首个 token 延迟），体验上"图加载慢"；
- 视觉 token 的 KV cache 同样占显存，多图/高分辨率场景更大。

应对手段：**chunked prefill**（把长 prefill 切成块，与 decode 请求交织执行，避免长 prefill 独占 GPU）、视觉 token 的 KV cache 量化、必要时对视觉特征做降采样/压缩（部分模型内置）。

### 4.4 多模态部署 pipeline 图

```
                     ┌──────────────────────────────────────────┐
   用户图像 ──────────►│  ① 图像预处理（Python，引擎外）              │
                     │     resize/crop/归一化/patchify           │
                     └────────────────────┬─────────────────────┘
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │  ② 视觉塔前向（原生 TRT engine 或 HF FP16） │
                     │     ViT/SigLIP → 视觉特征                  │
                     └────────────────────┬─────────────────────┘
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │  ③ 投影层：MLP/Linear → 视觉 token 序列     │
                     └────────────────────┬─────────────────────┘
                                          ▼
   用户文本 ──► Tokenizer ──► 文本 token ──┼──► ④ 拼接 [视觉 token + 文本 token]
                                          │        （视觉 token 是前缀）
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │  ⑤ TRT-LLM LLM 引擎（自回归生成）           │
                     │     PagedAttention + KV cache + 采样      │
                     └────────────────────┬─────────────────────┘
                                          ▼
                                    ⑥ 输出文本 token（流式）
```

**要点**：只有 ⑤ 走 TRT-LLM；①~④ 是引擎外 Python 逻辑（或用原生 TRT 编译的视觉塔引擎）。整条链路的延迟 = 预处理 + 视觉塔前向 + 投影 + LLM prefill + LLM decode，瓶颈往往在视觉塔前向与 LLM prefill 两端。

### 4.5 显存与吞吐要点

- 视觉塔参数量小（ViT-L ~300M vs LLM 7B），但 batch 大时视觉前向仍是计算热点，可并行跑多个 batch；
- 大规模部署可把视觉塔与 LLM 拆分到不同卡：一张卡只跑视觉塔 + 投影（吞吐高），其余卡跑 LLM 引擎；
- 长视觉前缀 + 长对话上下文时，KV cache 量化收益明显。

## 五、构建与运行示例

> 需 Linux + NVIDIA GPU + CUDA 环境运行（当前开发环境为 macOS，以下为标准可运行示例）

### 5.1 安装

```bash
# 要求 Linux + NVIDIA GPU（Ampere 及以上，H100 可开 FP8），CUDA ≥ 12.x
pip install tensorrt-llm
# 或从源码构建（获取最新模型支持）
```

### 5.2 构建 LLM engine（trtllm-build）

```bash
trtllm-build \
  --model_dir ./Llama-3.1-8B-Instruct-hf \   # HF 模型目录（config.json + safetensors）
  --dtype bfloat16 \                          # 引擎计算/存储精度
  --kv_cache_type paged \                     # KV cache 使用分页（默认推荐）
  --gemm_plugin auto \                        # 使用优化后的 GEMM 插件
  --max_batch_size 32 \                       # 运行时最大并发序列数（影响显存预算）
  --max_input_len 8192 \                      # 单序列最大输入 token 数
  --max_seq_len 16384 \                       # 最大总序列长度（含生成的 token）
  --output_dir ./llama31-8b-engine            # engine 输出目录
```

预期行为：构建过程打印各层 kernel 选择与权重转换信息；成功后 `./llama31-8b-engine` 下生成 `config.json`、`model.lark`（权重权重文件）与算子元数据。构建是**一次性成本**，跑几分钟到几十分钟，之后推理不再需要原始权重。

常用追加参数：

```bash
trtllm-build ... \
  --tp_size 2 \            # 张量并行卡数
  --pp_size 1 \            # 流水线并行卡数
  --use_fp8 \              # FP8 权重/激活（需 H100+）
  --kv_cache_quant_mode fp8 \   # KV cache 量化
  --max_num_tokens 8192 \       # in-flight batching 的单批最大 token 数（调吞吐用）
```

### 5.3 构建 VLM engine

```bash
# 以官方支持的 VLM（如 VILA/NVILA/LLaVA 系列）为例：目录内含视觉塔配置
trtllm-build \
  --model_dir ./VILA1.5-7b-hf \
  --dtype bfloat16 \
  --kv_cache_type paged \
  --max_batch_size 16 \
  --max_input_len 4096 \
  --max_seq_len 8192 \
  --output_dir ./vila7b-engine
```

预期行为：VLM 模型配置里声明了视觉塔，`trtllm-build` 会同时处理 LLM 部分（并生成对应的视觉塔组件配置）；具体是否含内置视觉塔 engine 视版本而定——不内置时视觉塔在 Python 端跑（见 4.2）。官方示例位于仓库 `examples/multimodal/` 下，接口随版本演进，以对应版本 README 为准。

### 5.4 Python 运行（LLM）

```python
# 需 Linux + NVIDIA GPU + CUDA 环境运行
from tensorrt_llm import LLM, SamplingParams

# 加载引擎：model 参数传 engine 目录即可（也可直接传 HF 模型目录自动构建）
llm = LLM(model="./llama31-8b-engine")

params = SamplingParams(max_new_tokens=256, temperature=0.7, top_p=0.9)
outputs = llm.generate(["Explain paged attention in one paragraph."], params)

for out in outputs[0].outputs:      # 支持 beam search → 多个输出
    print(out.text)
```

预期行为：加载 engine 后首次生成较慢（kernel 初始化 + CUDA 图），之后逐 token 流式返回；`generate` 返回结果对象，`.outputs[].text` 为生成文本。

### 5.5 Python 运行（VLM：视觉部分在引擎外）

```python
# 需 Linux + NVIDIA GPU + CUDA 环境运行
# 以 LLaVA/VILA 类模型为例：图像预处理 + 视觉塔前向在 Python 端完成
from transformers import AutoProcessor, AutoModel
from tensorrt_llm import LLM, SamplingParams

# ① 预处理 + 视觉塔 + 投影（引擎外，FP16）
processor = AutoProcessor.from_pretrained(vlm_hf_dir)
vision_tower = AutoModel.from_pretrained(vlm_hf_dir).vision_tower.half().cuda()
proj = get_projector(vlm_hf_dir)                       # 投影层参数

inputs = processor(images=image, text="Describe this image.",
                   return_tensors="pt").to("cuda")
with torch.no_grad():
    feats = vision_tower(inputs.pixel_values.half())   # 视觉特征
    vision_tokens = proj(feats)                        # → LLM 维度视觉 token

# ② 视觉 token 作为前缀送入 TRT-LLM 引擎（与文本一起编码）
llm = LLM(model="./vila7b-engine")
prompt = build_prompt_with_vision(vision_tokens, inputs)  # 具体拼接方式见官方示例
outputs = llm.generate([prompt], SamplingParams(max_new_tokens=128))
print(outputs[0].outputs[0].text)
```

预期行为：整条链路 = 图像前处理 → 视觉塔前向（GPU）→ 投影 → TRT-LLM 生成。视觉塔前向用 PyTorch 即可满足大部分场景；追求极致的把视觉塔单独编译成原生 TensorRT engine 再在 Python 里调用（见第七节）。

### 5.6 部署为服务

```bash
# 需 Linux + NVIDIA GPU + CUDA 环境运行
trtllm-serve --engine_dir ./llama31-8b-engine --port 8000
# OpenAI 兼容接口：POST /v1/chat/completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama31","messages":[{"role":"user","content":"Hi"}]}'
```

预期行为：服务启动后暴露 OpenAI 风格的 `/v1/completions`、`/v1/chat/completions` 接口，支持流式（`"stream": true`）。

## 六、性能指标与 benchmark 方法论

### 6.1 核心指标

| 指标 | 定义 | 公式/含义 |
| --- | --- | --- |
| TTFT | Time To First Token，首个输出 token 延迟 | 包含 prefill 全量计算，VLM 里含视觉塔前向 + 长视觉 prefill |
| TPOT | Time Per Output Token，解码阶段平均每 token 耗时 | $\text{TPOT} = \frac{\text{总耗时} - \text{TTFT}}{\text{生成 token 数}}$ |
| P95 延迟 | 95 分位端到端延迟（含排队） | 反映尾延迟与抖动 |
| 吞吐 | 每秒生成的 token 数（tokens/s）或请求数（QPS） | 受 batch 影响大 |
| 显存峰值 | 推理过程峰值显存 | 权重 + 激活 + KV cache，决定能开多大 batch |

端到端延迟估算：

$$\text{端到端} \approx \text{TTFT} + \text{TPOT} \times (\text{生成 token 数})$$

### 6.2 对比表设计（batch × 序列长度 × 精度）

| 配置（batch × 输入长度） | 精度 | TTFT (ms) | TPOT (ms) | P95 (ms) | 吞吐 (tok/s) | 显存峰值 (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 × 128 | FP16 | 45 | 22 | 60 | 45 | 17 |
| 8 × 128 | FP16 | 45 | 26 | 90 | 300 | 24 |
| 32 × 128 | FP16 | 46 | 34 | 150 | 900 | 60 |
| 8 × 8192 | FP16 | 1800 | 26 | 2000 | 300 | 78 |
| 8 × 8192 | FP8 | 1400 | 20 | 1600 | 380 | 42 |
| 8 × 8192 | FP16 + KV-INT8 | 1800 | 26 | 2000 | 300 | 52 |

（上表为格式示例，数值随模型/卡型变化。要点：**吞吐随 batch 增大而增大、延迟随 batch 增大而增大**，存在 sweet spot；长上下文主要吃显存与 TTFT；KV 量化对长上下文收益显著。）

### 6.3 方法论注意事项

1. **固定输入输出长度**：TTFT/TPOT 只在同长度下可比，用固定 prompt 长度与固定输出 token 数；
2. **用真实请求分布**：混合短对话与长文档、带流式读取，而不是单一 query 刷吞吐；
3. **预热后统计**：第一次请求含 kernel 初始化/CUDA 图编译，丢弃 warmup 数据；
4. **多轮重复取中位数/P95**：自回归有随机性，重复多次；
5. **构建时间 vs 推理时间分开衡量**：`trtllm-build` 是一次性成本（分钟~小时级），吞吐/延迟是每请求成本，做 ROI 时别把两者混在一起；
6. **官方工具**：`trtllm-bench latency --model ...` 与 `trtllm-bench throughput --model ...` 提供标准测量流程，直接用并上报结果。

### 6.4 调优清单

- batch 容量不够 → 开大 `--max_num_tokens`、降 KV cache（量化）、减 `--max_seq_len`；
- TTFT 高（长 prompt/VLM）→ chunked prefill、context parallel（长上下文）、视觉 token 压缩；
- TPOT 高 → 确认是带宽受限（量化/投机解码）、还是 kernel 选择不佳（`--gemm_plugin auto`、FlashAttention）；
- 多卡 → 用 `--tp_size` 扩大单请求并行度（降延迟）或开大 batch（提吞吐）。

## 七、与纯 TensorRT 部署 VLM 的差异

"不引入 TRT-LLM，用原生 TRT 也能部署 VLM"——但工程量差一个数量级：

| 维度 | 纯 TensorRT（自己拼） | TensorRT-LLM（一站式） |
| --- | --- | --- |
| 视觉塔 | ONNX → 独立 engine（或 HF） | Python/HF 前向或内置 |
| LLM 引擎 | ONNX → engine（需处理动态 seq、GQA 支持） | `trtllm-build` 自动处理 |
| 解码循环 | **手写**：for 循环逐 token 前向，手动维护 KV buffer、attention mask、位置 | 内建 |
| KV cache | 手写管理（预分配整块 or 自己分页） | 内建 PagedAttention |
| 采样 | 手写（temperature/top-p/beam） | 内建 SamplingParams |
| 长上下文 | 自己处理 chunk/内存 | 内建（paged + KV 量化） |
| 多卡 | 手动切分 + 自己写 NCCL 通信 | `--tp_size` / `--pp_size` |
| 流式/服务 | 自己写 HTTP + 状态机 | `trtllm-serve` |
| 开发量 | 周级（且易错：KV 越界、mask 错位） | 小时~天级 |

**结论**：纯 TRT 路线理论上可以做到极限性能（每一环都自己定制），但 LLM 的解码管线（KV cache 生命周期、批调度、采样状态）远比静态图复杂，自己拼的工程风险高、可维护性差。**默认选 TRT-LLM**；只有"视觉塔前向"这种无状态部分值得用原生 TRT 单独提速，再作为组件接进 TRT-LLM 管线。

## 八、高频面试问答

**Q1：TensorRT 和 TensorRT-LLM 的区别？**
原生 TRT 是把静态图编译成高效 engine（层融合、精度选择、kernel 替换），适合 CNN/视觉塔等单次前向；TRT-LLM 面向 LLM 的自回归循环，在 TRT 之上内建 KV cache 分页（PagedAttention）、连续批处理、采样、多卡并行——"TRT 优化一张图，TRT-LLM 优化一个循环"。

**Q2：TRT-LLM 和 vLLM 选哪个？**
看场景：追求峰值性能 + 已有 NVIDIA 生产栈 + 模型列表固定 → TRT-LLM（代价是每 GPU 型号重新构建 engine）；模型迭代快、需要快速验证/上线 + 社区生态 → vLLM。两者底层都是 PagedAttention，性能差距 10% 量级，工程与生态往往更重要。

**Q3：engine 为什么绑定 GPU 型号？**
engine 在构建时选择了具体架构的算子/kernel（SM 数量、Tensor Core 指令、寄存器压力）并按该卡的算力做了图优化，还依赖精确的 CUDA/驱动版本。换卡不重新构建会直接加载失败（算力不匹配）或行为未定义。这既是 TRT-LLM 性能的来源，也是它"不灵活"的来源。

**Q4：VLM 的视觉部分怎么优化？**
三块：① 视觉塔前向（ViT 计算 + 卷积 patchify）可编译成原生 TRT engine 提速，或多 batch 并行；② 视觉 token 数量大 → 长 prefill 拉高 TTFT，用 chunked prefill / 上下文并行 / 视觉 token 压缩；③ 视觉 token 的 KV cache 量化省显存。另外预处理（resize/patchify）在 CPU/GPU 端都要优化，不能只看 LLM。

**Q5：PagedAttention 为什么省显存、提吞吐？**
把 KV cache 切成固定页 + 页表按需分配，消除了"预分配 max_seq_len"的浪费与碎片；空闲页可被任意序列复用 → 同样显存跑更大 batch → 吞吐直接上升。类比虚拟内存的按需分页。

**Q6：为什么 VLM 部署 TTFT 普遍偏高？**
TTFT 含 prefill 全量计算，而 VLM 一次就注入几百上千个视觉 token（+ 长对话历史），prefill 极长且计算密集；加上视觉塔前向本身要几十 ms。优化方向是压缩视觉 token、chunked prefill、视觉塔提速。

**Q7：FP8 推理需要什么硬件？**
Hopper（H100/H200）起 Tensor Core 原生支持 FP8（E4M3/E5M2），Blackwell 更强；A100/T4 不支持，只能用 INT8/FP16。TRT-LLM 里 `--use_fp8` 构建、`--kv_cache_quant_mode fp8` 量化 KV。

**Q8：单卡放不下模型怎么办？**
先省显存：量化（FP8/INT8 + KV 量化）、减小 max_seq_len、PagedAttention 已内置。还放不下再上多卡：`--tp_size` 张量并行（权重切到多卡，延迟低）或 `--pp_size` 流水线并行（适合跨节点）。TP 通信开销大，跨节点优先 PP + 每节点内 TP。

## 九、常见误区

1. **"TensorRT engine 是跨 GPU 通用的"** —— 错。engine 绑定算力/驱动/CUDA 版本，A100 构建的 engine 不能跑在 T4 或 L40S 上，必须每 GPU 型号重新构建。

2. **"INT8 一定比 FP16 快很多"** —— 错。加速来自带宽与 Tensor Core，小模型或访存密集层（LayerNorm、非 matmul）加速有限，可能只快 20~30%；且量化流程有成本与掉点风险。先 benchmark 再决定。

3. **"VLM 部署只有 LLM 是瓶颈"** —— 错。视觉塔前向 + 几百上千个视觉 token 的长 prefill 常常是 TTFT 的主要贡献者；图像预处理与投影也会在 CPU/GPU 端排队。

4. **"TRT-LLM 支持所有模型"** —— 错。TRT-LLM 按官方模型逐个适配（config 转换、自定义算子），冷门模型/新架构需要自己写 config 甚至自定义 kernel，有适配成本。

5. **"换显卡把 engine 目录拷过去就行"** —— 错（同 1）。且 CUDA/驱动版本不一致时可能加载即失败。部署脚本里应包含"目标卡型号 → 构建 engine"的自动化。

## 十、自我检验

- [ ] 能用一句话说清 TensorRT 与 TensorRT-LLM 的分工（一张图 vs 一个循环）
- [ ] 能讲清 PagedAttention 为什么省显存、in-flight batching 为什么提吞吐
- [ ] 能说出 TRT-LLM 与 vLLM 在易用性/硬件绑定/生态上的核心差异
- [ ] 能画出 VLM 部署完整管线图并指出哪些在引擎外、哪些在引擎内
- [ ] 知道视觉 token 的 prefill 特点与 TTFT 的关系，及三种优化手段
- [ ] 能写出 trtllm-build 的常用参数（dtype/kv_cache_type/gemm_plugin/tp_size）与含义
- [ ] 能说清 TTFT/TPOT/P95/吞吐/显存峰值的定义与 benchmark 设计要点
- [ ] 能对比"纯 TRT 自己拼 VLM"与 TRT-LLM 的工程量差异
- [ ] 能指出 5 条常见误区并纠正
- [ ] 能回答 8 个面试追问
