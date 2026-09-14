---
learned: false
---

# TensorRT详解：把模型"编译"成 GPU 最擅长的执行形态

> 本模块索引见 [TensorRT详解](TensorRT详解.md)

## 子篇索引

| 文件名                                            | 一句话简介                                                                                                                             |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| [构建流程与图优化.md](构建流程与图优化.md)                     | 从 ONNX 到 engine 的完整构建链路：模型解析→精度设置→图优化（Conv+BN 折叠、QKV/Softmax 融合）→tactic 自动选 kernel→动态 shape→workspace，附 trtexec 与 Python API 标准示例 |
| [精度与量化.md](精度与量化.md)                           | FP32/TF32/FP16/INT8/FP8 的精度谱系：什么时候用什么精度、INT8 校准原理、量化掉点排查与精度/速度评测方法                                                                |
| [TensorRT-LLM与多模态部署.md](TensorRT-LLM与多模态部署.md) | LLM 为什么不能直接用 TRT：TensorRT-LLM 的 KV cache、PagedAttention、连续 batching，以及"ViT 视觉塔 + TensorRT-LLM 主干"的 VLM 部署方案                       |

> 阅读顺序：先读本文建立整体认知 → [构建流程与图优化.md](构建流程与图优化.md)（本模块核心：engine 是怎么"编译"出来的）→ [精度与量化.md](精度与量化.md)（精度怎么选）→ [TensorRT-LLM与多模态部署.md](TensorRT-LLM与多模态部署.md)（LLM/VLM 怎么落地）。

## 一、一句话解释

**TensorRT（TRT）是 NVIDIA 的深度学习推理优化引擎：读入训练好的模型（通常为 ONNX）→ 做图优化 → 自动挑选最快的 CUDA kernel → 生成一个针对"特定 GPU 型号 + 特定精度 + 特定输入 shape"的可执行文件 engine（.engine），推理速度通常比 PyTorch 快 2~10 倍。**

拆开来看：

- **读入模型**：支持 ONNX、UFF、TensorFlow 等格式；PyTorch 模型先导出成 ONNX（见第三节流程图）；
- **图优化**：把静态计算图"编译"成 GPU 最优的执行计划——层融合、kernel 选择、精度权衡（详见子篇一）；
- **生成 engine**：engine 是最终产物，一段序列化的可执行计算图，加载后直接推理；
- **比 PyTorch 快 2~10 倍**：CNN/检测类模型通常 2~5 倍，融合收益大的可达 10 倍；已经逼近硬件极限或瓶颈在显存带宽的模型收益小。

**一个类比**：PyTorch 是"解释执行"（每行代码现解释现跑，灵活但慢），TensorRT 是"**编译执行**"（针对目标机器提前优化成机器码，快但绑定机器）。engine 绑定 GPU 型号，正是"编译目标具体到某张卡"的代价——这是面试最容易考的一个点。

**几个常见误解**：

1. TensorRT 不是"更快的 PyTorch 运行时"，而是独立的推理引擎，产物是 engine 文件；
2. 不是"开箱即用"：需要构建流程（导出 ONNX、配置精度、动态 shape 等），开发成本比 ONNX Runtime 高；
3. 不是"模型越大收益越大"：推理收益最大的是算子类型丰富的 CNN/Transformer；瓶颈在显存带宽的小模型收益有限。

### 1.1 模块名词速记

| 名词               | 是什么                                 | 对应本文/子篇章节                  |
| ---------------- | ----------------------------------- | -------------------------- |
| ONNX             | 框架无关的静态计算图中间格式，TRT 的主要输入            | 第三节                        |
| Network          | ONNX 图解析到 TRT 后的内部 IR（张量 + 层）       | [构建流程与图优化.md](构建流程与图优化.md) |
| Builder / Config | 构建器与构建配置（精度 flag、profile、workspace） | [构建流程与图优化.md](构建流程与图优化.md) |
| Engine           | 构建产物，序列化后可加载推理，绑定 GPU 型号            | 第三节 / 第七节                  |
| Profile          | 动态 shape 的 min/opt/max 三元组          | [构建流程与图优化.md](构建流程与图优化.md) |
| Tactic           | 某层的候选 CUDA kernel 实现（构建时实测挑选）       | [构建流程与图优化.md](构建流程与图优化.md) |
| Plugin           | 自定义算子（NMS、RoIAlign 等），注册进计算图参与构建    | [构建流程与图优化.md](构建流程与图优化.md) |
| Calibration      | INT8 量化的校准流程（收集分布 → 算 scale）        | [精度与量化.md](精度与量化.md)       |
| trtexec          | TRT 自带的命令行构建/benchmark 工具           | [构建流程与图优化.md](构建流程与图优化.md) |

> 这张表建议背下来：面试聊到 TensorRT 时，能顺手说出这些名词的准确含义，是"真做过部署"和"只是听过"的分水岭。

## 二、为什么快：五个核心机制

| 机制                         | 一句话原理                                                   | 效果                      |
| -------------------------- | ------------------------------------------------------- | ----------------------- |
| ① 层融合（Layer Fusion）        | 把相邻算子合并成一个 kernel（如 Conv+BN+ReLU → 1 个），中间数据留在寄存器/共享内存  | 减少 kernel 启动次数与中间张量显存读写 |
| ② kernel 自动选择（Tactic）      | 每个层在**目标 GPU 上实测**多套 CUDA 实现（不同 tile/算法），选最快的写死进 engine | 每层都用当前卡上最快的实现           |
| ③ 显存池（Workspace）           | 构建时静态分析中间张量生命周期，推理时从预分配池复用显存                            | 推理零动态分配，显存峰值与延迟更低       |
| ④ 精度选择（FP16/INT8/TF32/FP8） | 用低精度计算换吞吐（INT8 约 4×FP32），配合校准控制精度损失                     | 计算吞吐成倍提升                |
| ⑤ 算子库深度优化                  | 底层基于 cuDNN/cuBLAS/CUTLASS 深度调优，专为 Tensor Core 设计 kernel | 单算子本身就跑得比通用框架快          |

### 2.1 层融合（最大头收益）

CNN 里大量算子是逐元素的（Bias、ReLU、BN、残差相加），它们"计算量 ≈ 0"，瓶颈全在显存读写。把它们并进前一个算子的 kernel 尾部（epilogue），中间数据不落显存，等于白赚。例子：Conv+Bias+ReLU 三个 kernel 合成一个；Transformer 里 QKV 三个线性层合成一个 GEMM、softmax 整段合成一个 kernel。**融合是"为什么快"的第一答案**。

### 2.2 kernel 自动选择（Tactic）

同一个"卷积"在 GPU 上有几十种实现：SIMT（标量核）还是 Tensor Core、tile 多大、NHWC 还是 NCHW、走 cuDNN 还是自研 CUTLASS kernel……TensorRT 构建时把每层所有候选 kernel 在目标 GPU 上**真实跑一遍计时**，选最快的固化进 engine。这就是"engine 必须针对具体 GPU 现测现选、无法跨卡复用"的根源。

### 2.3 显存池（Workspace）

eager 模式每次前向都要动态分配/释放中间张量（cudaMalloc 开销 + 碎片）。TRT 构建时按计算图算出每个中间张量的生命周期，推理时从预分配的池里复用同一块显存，推理全程零分配。显存峰值和延迟同时下降。

### 2.4 精度选择

FP16 计算吞吐是 FP32 的 2 倍（带宽减半），INT8 再翻倍；Ampere+ 的 TF32 在接近 FP32 精度下提供接近 FP16 的吞吐。TRT 不是简单全图降精度，而是逐层权衡：敏感层保持高精度，其余层降精度。降低精度还同时压缩了权重与激活的显存带宽需求——对大模型推理，带宽往往是真瓶颈。

### 2.5 算子库深度优化

TensorRT 的单算子 kernel 本身就是 NVIDIA 精心调过的：针对每种 Tensor Core 架构（Turing/Ampere/Hopper）分别优化、支持 FP8 等新格式、FlashAttention 风格的融合注意力 kernel 等。通用框架（如 ONNX Runtime 的 CPU/CUDA EP）做不到这个投入程度。

### 2.6 五机制的关系

- ① 和 ② 是**框架级**优化（编译器的活），决定"少做无用功"；③ ④ ⑤ 是**工程级**优化（运行时与库的活），决定"每步做得多快"；
- ② 是 ① 的前提：融合后的每个复合 kernel 也是按 tactic 机制挑选/生成的；
- ④ 会放大 ①~③ 的收益：低精度下带宽压力减半，融合省掉的显存往返价值更高；
- 面试被追问"哪个贡献最大"：不同模型不同——卷积多的模型融合收益最大，GEMM 密集的模型 tactic + Tensor Core 收益最大，带宽瓶颈的模型精度选择收益最大。

> 面试作答顺序建议：先讲"编译型优化"（融合 + 选 kernel）这类框架级思路，再补"工程型优化"（显存池 + 精度），最后落到"库级优化"。

## 三、在部署链路中的位置

```
训练侧                                  部署侧
┌───────────────┐  导出  ┌──────────────┐  构建  ┌────────────────────┐  序列化 ┌──────────────┐
│ PyTorch 模型   │ ─────► │  ONNX 模型    │ ─────► │  TensorRT 编译      │ ──────► │  .engine 文件  │
│ (训练产物)      │        │ (静态计算图)    │        │ (图优化+选kernel+量化)│        │ (GPU 型号绑定) │
└───────────────┘        └──────────────┘        └────────────────────┘        └──────┬───────┘
                                                                                     │ 加载
                                                                                     ▼
                                                                             ┌────────────────┐
                                                                                 │ 推理服务/在线脚本 │
                                                                                 │ (Triton/自研)   │
                                                                             └────────────────┘
```

- **为什么中间要转 ONNX**：ONNX 是框架无关的静态计算图。只有"完整拿到计算图"才能做融合与预编译；eager 模式"算一步跑一步"，没有全图，无从优化——这是"训练框架 vs 推理引擎"的本质分界；
- **PyTorch 能不能直接到 TRT**：能（torch2trt、torch_tensorrt 等桥接工具），但主流且可控的路径仍是"onnx.export 导出 → TRT 构建"；
- **量化在哪一步做**：可以在 ONNX 阶段做（PTQ 导出 INT8），也可以交给 TRT 的 INT8 校准流程做（推荐，见 [精度与量化.md](精度与量化.md)）；
- **LLM 不走这条链路**：自回归生成需要 KV cache 与批调度，普通 TRT 管不了，走 TensorRT-LLM（见子篇三）；
- **推理侧**：engine 可直接嵌入脚本/服务，或由 Triton Inference Server 托管（多模型、动态 batch 场景）；LLM 场景用 vLLM 或 TensorRT-LLM。

### 3.1 为什么这条链路里"编译"说得通

把上面的图换一种说法，TensorRT 做的事情与编译器一一对应：

| 编译器概念 | TensorRT 对应物 |
| --- | --- |
| 源代码 | ONNX 静态计算图 |
| 中间表示（IR） | TRT Network（张量 + 层） |
| 优化 pass | 层融合、常量折叠、死代码消除、精度分配 |
| 指令选择 | tactic 基准测试挑 kernel |
| 目标机器 | 具体 GPU 型号（sm 版本）+ CUDA/cuDNN 版本 |
| 可执行文件 | .engine 序列化文件 |

这个对应关系是面试讲"engine 是什么"的最佳框架：**engine = 针对指定机器编译好的可执行程序，换机器要重新编译**。

### 3.2 几个"为什么"的答案

- **为什么 PyTorch 推理慢**：eager 模式每个算子独立启动 kernel、中间张量全落显存、没有跨算子优化——这三条恰好是 TRT 的优化目标；
- **为什么 ONNX 导出要固定动态轴**：onnx.export 时默认把所有轴固定成训练时的 shape；要做动态 shape 必须显式声明 dynamic_axes（详见子篇一第五节）；
- **为什么构建要在目标卡上做**：tactic 挑选依赖实测，TRT 不允许"跨卡预测最优 kernel"，所以构建机与部署机的 GPU 型号必须一致。

## 四、与同类推理框架对比

| 对比项 | PyTorch eager | ONNX Runtime | TensorRT | vLLM |
| --- | --- | --- | --- | --- |
| 定位 | 训练框架（推理是附加能力） | 跨平台推理引擎 | NVIDIA GPU 专用推理引擎 | LLM 专用推理服务框架 |
| 延迟 | 高：逐算子启动 kernel | 中：有基础图优化 | 最低：融合 + 编译 | 低：CUDA Graph 等优化 |
| 吞吐 | 低 | 中 | 高 | 最高：连续 batching + PagedAttention |
| 显存占用 | 高（中间张量多、框架开销大） | 中 | 低（融合 + 显存池复用） | 低（KV cache 按页管理） |
| 开发成本 | 零 | 低：换 runtime 即可 | 高：构建 + 校准 + 动态 shape 工程 | 中：按模型格式接入 |
| 适用模型 | 任意 | CNN / Transformer / 任意 ONNX | CNN / ViT / 检测分割 / VLM 组件 | 仅 LLM / 多模态 LLM |
| 硬件要求 | CPU / GPU / Apple 芯片等 | CPU / GPU / NPU 等 | 仅 NVIDIA GPU | NVIDIA GPU（Ampere+ 最佳） |
| 典型加速比 | 1×（基准） | 1.5~3× | 2~10× | 相对 naive 自回归数倍~数十倍 |

几点解读：

1. **ONNX Runtime 是"性价比路线"**：改一行代码（runtime 换 ORT）就能拿到基础图优化，且能跑 CPU/国产卡，适合快速提效与跨平台；
2. **TensorRT 是"性能上限路线"**：NVIDIA 卡上单次前向延迟最低、单引擎吞吐高，但要承担构建/校准/动态 shape 的工程成本，且绑定卡型；
3. **vLLM 只回答 LLM 问题**：它的核心创新在调度层（PagedAttention、连续 batching），LLM 延迟/吞吐双优；如果你要的是"CNN/ViT 最快"，它不适用；
4. **同场景可以叠加**：多模态服务里常见"ViT 用 TensorRT + LLM 用 vLLM/TensorRT-LLM"的组合。

**一句话选型**：CNN/检测/ViT 上线 → TensorRT；要跨平台（CPU/国产卡）→ ONNX Runtime；LLM 在线服务 → vLLM 或 TensorRT-LLM；零改动先提速 → 先把 PyTorch 换 ONNX Runtime 试试。

### 4.1 为什么加速比是"2~10 倍"而不是固定值

加速比取决于模型结构与当前运行时有多"浪费"：

- **算子种类越多、逐元素算子占比越高**（CNN 典型）→ 融合收益越大，可达 5~10 倍；
- **GEMM 密集**（ViT/LLM 的注意力+MLP）→ 收益主要来自 tactic + Tensor Core + 精度，通常 2~3 倍；
- **已经很小/带宽受限**（几 MB 权重的小模型）→ eager 与编译后的差距缩小，可能只有 1.2~1.5 倍；
- **PyTorch 的对比基线也影响数字**：对比"PyTorch + FP32"还是"PyTorch + torch.compile + FP16"，后者已经把部分收益吃掉了。

> 面试话术：不要背死"10 倍"，正确说法是"**结构越依赖逐元素/卷积算子、基线越朴素，加速比越接近上限**"。

### 4.2 何时不该选 TensorRT

- 模型/输入 shape 天天变（实验阶段）→ 构建成本不划算，先用 ONNX Runtime；
- 需要跑 CPU/国产卡/端侧 → 硬件不允许；
- 纯 LLM 服务 → 直接 vLLM / TensorRT-LLM，别在普通 TRT 上折腾；
- 模型带大量动态控制流（while/if 依赖数据）→ ONNX 都难导出，TRT 更不支持，考虑专门引擎。

## 五、适用场景

### 5.1 视觉模型（收益最大）

- **CNN 分类/检测/分割**：卷积 + 激活 + BN + 残差的组合最容易被融合，YOLO/ResNet 系部署首选 TRT，典型 2~5×；
- **ViT 类**：主体是 GEMM（attention + MLP），走 cuBLAS/Tensor Core 高速 kernel，同样显著提速；
- **后处理进 engine**：NMS、RoIAlign、上采样等可用 plugin 或自定义 kernel 并进 engine，避免 CPU 端来回搬运。

### 5.2 LLM：必须用 TensorRT-LLM

- 自回归生成不是"一次前向"，而是"逐 token 循环 + KV cache 累积"，普通 TRT 的静态图无法表达这种动态循环；
- TensorRT-LLM 在 TRT 之上补全了 LLM 推理的全部机制：PagedAttention、KV cache 管理、连续 batching（inflight batching）、FP8/INT4 量化、投机解码、多卡并行（TP/PP）；
- 工程节奏建议：先用 vLLM 快速验证效果，再为了极致性能/与 Triton 生态整合迁移 TensorRT-LLM（详见 [TensorRT-LLM与多模态部署.md](TensorRT-LLM与多模态部署.md)）。

### 5.3 多模态 VLM：视觉塔 + LLM 组合

- **视觉塔（ViT）→ 标准 TRT engine**：图像编码是单次前向；输入分辨率可变 → 必须用动态 shape（见 [构建流程与图优化.md](构建流程与图优化.md) 第五节）；
- **投影层**：可并入视觉塔 engine，也可独立成一个小 engine；
- **LLM 主干 → TensorRT-LLM**：视觉 token 与文本 token 拼接后进 LLM，KV cache 由 TensorRT-LLM 托管；
- 例：LLaVA/Qwen-VL 类模型的部署 = 1 个 ViT engine + 1 个 TensorRT-LLM 服务，中间用采样/拼接插件或服务层粘合。视觉塔吞吐吃紧时，一张图产生的视觉 token 数量大，ViT engine 的优化收益直接放大。

### 5.4 部署工程视角

- engine 按"GPU 型号 × 精度 × shape 档位"是构建产物矩阵，建议在 CI 里构建并归档（带 TRT/CUDA 版本标签）；
- 动态 shape 的 engine 显存按 maxShape 预留，容量规划要留足；
- 推理服务用 Triton 时，TRT 是它的首选 backend，可自动处理并发与批处理（Triton 内部有它的动态 batch，与 TRT 动态 shape 是两层事）；
- 视觉/多模态服务的典型链路：请求 → 图像预处理（CPU/GPU 都行）→ ViT engine → token 组装 → LLM（TensorRT-LLM/vLLM）→ 后处理 → 返回；瓶颈定位要在**整条链路**上测，不要只看 ViT engine 单点。

### 5.5 性能验证方法论（简述）

- **单引擎**：用 trtexec 直接测（Latency mean/min/max、Throughput），它给出的是"纯 engine 能力"；
- **端到端**：用真实请求 + 真实前处理/后处理测 p99 延迟与吞吐，通常比 trtexec 数字差（有前后处理与排队）；
- **对比口径**：PyTorch / ONNX Runtime / TensorRT 三套跑同一输入、同一精度、同一 warmup 与迭代次数，比延迟与吞吐；
- **显存**：nvidia-smi 观测峰值，注意动态 shape 按 max 预留导致"实际占得比跑的时刻多"。

> 完整评测与量化掉点排查见 [精度与量化.md](精度与量化.md)。

## 六、硬件要求

| 项 | 要求 |
| --- | --- |
| 算力 | 必须 NVIDIA GPU（CUDA）；AMD/Intel 集成显卡不支持 |
| 实用架构 | Turing（T4/RTX20）/ Ampere（A100/A30/A10/L4）/ Hopper（H100）/ Blackwell（B200） |
| TF32 | Ampere+（sm_80+）原生支持，TRT 默认开启 |
| INT8 | Turing+ 有 INT8 Tensor Core，Ampere+ 效果最佳 |
| FP8 | 需要 Hopper（H100，sm_90）及以上 + TensorRT 9.0+ |
| 软件栈 | CUDA + cuDNN 版本需与 TRT 匹配（安装包内绑定，不能混用） |
| 显存 | 按"权重 + 激活缓冲 + workspace + 框架开销"预估；动态 shape 按 maxShape 预留 |
| 可移植性 | **.engine 与 GPU 型号强绑定**：sm 版本不同不可复用，需按卡型分别构建 |

补充说明：

- **构建机与部署机可以不同**：只要 GPU 型号相同（sm 版本一致），在 CI 机器上构建、拷到部署机加载即可；型号不同必须重构建；
- **低端/老卡能用吗**：能，TensorRT 支持到较老的 Maxwell 架构，但 Tensor Core 相关的精度/速度优势没有；实际生产建议 Turing 起步；
- **为什么强调 Ampere+**：TF32、更成熟的 INT8、FP8（Hopper+）都是 Ampere 之后才有；面试问"FP8 要什么卡"答案就是 Hopper（H100）及以上；
- **CUDA 版本注意**：TRT 安装包绑定 CUDA/cuDNN 版本，升级 CUDA 不代表 TRT 自动变好——engine 里固化的 kernel 与库版本强相关，升级后通常建议重新构建；
- **端侧/边缘（Jetson）**：Jetson 自带精简版 TRT，同样的构建 API，只是算子覆盖与性能档位不同——部署到 Jetson 也要单独构建 engine。

## 七、常见面试考点预告

| 考点               | 一句话答案                                         | 详阅                                             |
| ---------------- | --------------------------------------------- | ---------------------------------------------- |
| engine 是什么       | 针对特定 GPU/精度/shape 编译好的可执行计算图，加载即推理            | [构建流程与图优化.md](构建流程与图优化.md)                     |
| TensorRT 为什么快    | 层融合 + tactic 自动选 kernel + 显存池 + 低精度 + 深度算子库   | [构建流程与图优化.md](构建流程与图优化.md)                     |
| 层融合有哪些           | Conv+Bias+ReLU、Conv+BN 折叠、QKV/Softmax 融合、残差融合 | [构建流程与图优化.md](构建流程与图优化.md)                     |
| FP16/INT8 怎么选    | 精度够用即可；INT8 需校准，量化后必须评测                       | [精度与量化.md](精度与量化.md)                           |
| 动态 shape         | min/opt/max 三元组，运行时可变输入形状，多模态必备               | [构建流程与图优化.md](构建流程与图优化.md)                     |
| LLM 为什么不用普通 TRT  | 自回归 + KV cache + 批调度需要 TensorRT-LLM           | [TensorRT-LLM与多模态部署.md](TensorRT-LLM与多模态部署.md) |
| engine 为什么绑定 GPU | kernel 是按目标卡实测挑选/编译的，跨 sm 不兼容                 | [构建流程与图优化.md](构建流程与图优化.md)                     |

> 面试高频串讲：先答"engine 是什么"（编译产物）→ 再答"为什么快"（五个机制，重点是融合与 tactic）→ 被追问时展开"层融合公式"与"动态 shape 原理"→ 最后落到"LLM 场景用 TensorRT-LLM"。每个展开都在对应子篇。

### 7.1 考点速答（先给结论，推导见子篇）

**Q：engine 和 PyTorch 模型文件有什么区别？**
PyTorch 的 `.pth` 是"权重 + 网络定义"，可被任意平台动态解释执行；engine 是把图优化 + kernel 选择全部做完后的**可执行产物**，快但绑定 GPU 型号、精度与 shape 档位，不能当通用权重分发。

**Q：TensorRT 快 10 倍的说法靠谱吗？**
靠谱但有条件。加速来自融合（省 kernel 启动 + 省显存往返）+ tactic（每层选最快 kernel）+ 低精度（FP16/INT8）+ 显存池。逐元素算子/卷积占比高的模型收益最大；瓶颈在带宽的模型收益小。

**Q：FP16 和 INT8 能直接用吗？**
FP16 在多数视觉模型上可直接打开（注意溢出与精度验证）；INT8 必须做**校准**：拿代表性数据统计激活分布，算量化 scale，把精度损失压到可接受范围，再上评测集确认不掉点。

**Q：动态 shape 里的 min/opt/max 到底怎么填？**
min = 最小合法输入（如 batch=1、最小分辨率）；opt = 最常见输入（kernel 按它挑选）；max = 最大合法输入（显存按它预留）。opt 填错了性能会显著变差——这是动态 shape 最常见的失误。

**Q：普通 TensorRT 能部署 LLM 吗？**
不能直接。LLM 是自回归循环 + KV cache 动态增长，普通 TRT 静态图表达不了；需要 TensorRT-LLM（PagedAttention、连续 batching、KV cache 管理）或 vLLM。

**Q：engine 为什么不能跨 GPU 用？**
kernel 是按目标卡的 sm 架构（计算能力）实测挑选并编译的；不同架构（如 sm_80 的 A100 vs sm_86 的 RTX 4090）的指令与性能特性不同，强跑会出错或退化为慢路径。生产上按 GPU 型号构建产物矩阵。

## 八、自我检验

- [ ] 能用一句话说出 TensorRT 是什么（读入模型 → 图优化 → 生成 GPU 专属 engine，快 2~10 倍）
- [ ] 能说出五个加速机制，每个一句话，并说明哪个贡献最大取决于什么
- [ ] 能画出 PyTorch → ONNX → TRT engine → 推理服务的链路，并说明为什么中间要 ONNX
- [ ] 能用"编译器六要素"（IR/优化 pass/指令选择/目标机/可执行文件）类比 TRT 的构建
- [ ] 能说出 PyTorch eager / ONNX Runtime / TensorRT / vLLM 的适用边界与加速比量级
- [ ] 能说出 LLM 为什么不能用普通 TRT、VLM 怎么拆成"视觉塔 + LLM"
- [ ] 能说出硬件要求：NVIDIA GPU 专属；TF32/INT8 需 Ampere+；FP8 需 Hopper+
- [ ] 能说出 7 个面试考点中的任意 5 个并指到对应子篇
- [ ] 能背下 9 个名词速记（ONNX/Network/Builder/Engine/Profile/Tactic/Plugin/Calibration/trtexec）
- [ ] 已通读 [构建流程与图优化.md](构建流程与图优化.md) 并完成其自我检验

> 下一步：进入本模块核心 [构建流程与图优化.md](构建流程与图优化.md)，把"engine 怎么编译出来"这件事彻底讲透。
