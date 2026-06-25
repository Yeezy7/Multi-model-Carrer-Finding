下面这版按“**大厂多模态算法岗真实筛选逻辑**”来规划，不按课程目录堆知识点。

我看了一些岗位后，结论是：你不能只按“传统 CV 算法工程师”准备，而应该定位成：

> **多模态视觉算法工程师：VLM / 图文视频理解 / 跨模态检索 / 视觉推理 / 模型部署优化**

这比单纯做 YOLO、分割、异常检测更贴近现在大厂需求。

---

## 1. 大厂岗位要求提炼

字节 Seed 明确面向 2027 届大模型方向应届生和在校实习生招聘，并且热招岗位中包含“多模态世界模型算法研究员/专家”等方向。([字节跳动 Seed][1]) Top Seed 对候选人的要求更偏研究型，强调在 AI 具体方向有深度技术见解、高质量论文或代表性开源工作。([字节跳动 Seed][2])

字节电商多模态大模型岗位更偏业务落地，要求做图文、图视频多模态底座的预训练和对齐，并优化数据采集、模型训练、部署、推理流程；要求熟悉 InternVL、LLaVA-Next、DeepSeek-VL 等多模态模型，具备图像搜索、图像/视频分类、目标检测、图文/视频文本多模态项目经验。([牛客网][3])

华为云多模态算法岗强调多模态表示学习、融合、对齐、内容理解、生成、跨模态检索、多模态匹配，并要求掌握 NLP/CV/ML/DL 常用模型、Python/C/C++、TensorFlow/PyTorch 等框架。([华为云][4])

阿里达摩院基础大模型视觉实习岗要求理解多模态内容算法及下游任务，包括图像/视频生成、图像/视频表征、检测、分割等，同时要求 PyTorch、Python/C++、Linux、论文阅读和科研能力。([牛客网][5])

小红书 VLM Post-training 岗位更接近当前多模态大模型主流方向，要求做 VLM 的 SFT/RL/Post-training，提升图文、视频与文本的语义对齐、视觉 Reasoning、多模态 Agent、Tool-use、长视频理解和多帧推理能力，并熟悉 LLaVA、Qwen-VL、InternVL 等主流架构。([牛企直聘][6])

**提炼成学习重点就是：**

| 能力层级  | 大厂真正看什么                                                                 |
| ----- | ----------------------------------------------------------------------- |
| 基础算法  | ML/DL、Transformer、CV/NLP 基础、PyTorch 熟练度                                 |
| 多模态建模 | CLIP/SigLIP、BLIP/LLaVA/Qwen-VL/InternVL、图文对齐、VQA、Grounding、OCR、多图/视频理解  |
| 训练与微调 | 对比学习、SFT、LoRA、DPO/RLHF 基础、数据清洗、指令数据构造、评测体系                              |
| 业务落地  | 电商商品理解、内容审核、图文搜索、视频理解、文档图表理解、工业质检                                       |
| 工程部署  | FastAPI、Docker、vLLM/Transformers、ONNX、TensorRT、Triton、显存优化、P95/QPS/吞吐评估 |
| 加分项   | 顶会论文、开源项目、竞赛、可复现 benchmark、真实业务指标                                       |

---

## 2. 结合你的当前基础，主线应该这样定

你现在简历里已经有：YOLOv8 检测、RTSP 视频流接入、Qwen3-VL-32B 部署与视觉 Token 裁剪、SigLIP 图文对齐微调蒸馏、ONNX + TensorRT FP16 部署、零样本异常检测论文和工业巡检项目。

所以你不要再把大量时间花在“传统检测/分割基础项目”上。你的短板更像是：

1. **VLM 训练/后训练能力不够完整**：会部署和调用，但需要补 SFT、LoRA、多模态指令数据、评测。
2. **视频/多图/复杂视觉推理不够突出**：大厂内容场景非常看重视频理解、图文内容理解、长图文档、空间关系、多帧推理。
3. **业务闭环不够强**：需要把模型能力落到电商、内容搜索、审核、工业质检这种具体场景。
4. **工程指标需要继续强化**：P95、QPS、显存、吞吐、TensorRT/vLLM/Triton 对比，是简历区分度来源。

---

## 3. 推荐学习路线：10 周冲刺版

### 第 1 阶段：补齐多模态基础，别只会调模型

时间：第 1–2 周

你需要掌握：

* Transformer 基础：Attention、KV Cache、RoPE、MLP、LayerNorm、prefill/decode。
* LLM 基础：tokenizer、SFT、LoRA、DPO、指令微调数据格式。
* 视觉编码器：ViT、DINOv2、CLIP、SigLIP。
* 多模态对齐：image encoder + projector/resampler + LLM 的结构。
* 常见模型架构：CLIP/SigLIP、BLIP-2、LLaVA、Qwen-VL、InternVL。

最低产出：

* 手写一个简化版 CLIP/SigLIP 图文对齐训练脚本；
* 跑通 Qwen-VL / InternVL 的图片问答、OCR、图文理解、多图输入；
* 写一篇 README：解释 VLM 的输入流程、视觉 token 生成、projector、LLM 解码。

这一阶段不需要追求模型效果，目标是面试能讲清楚“VLM 为什么能看图”。

---

### 第 2 阶段：做 VLM 微调和评测

时间：第 3–4 周

你要从“会用 VLM”升级到“知道怎么改 VLM”。

学习内容：

* LoRA / QLoRA 微调；
* 多模态 SFT 数据格式；
* 图文问答、OCR 问答、图表问答、商品属性识别数据构造；
* hallucination、OCR 错误、细粒度属性错误的评估；
* 指标：Accuracy、F1、Recall@K、EM、BLEU/ROUGE 只做辅助，不要过度依赖。

建议做一个小任务：

> **商品图文一致性判断 / 文档截图问答 / 工业缺陷描述生成**

输入图片 + 文本，输出结构化 JSON，比如：

```json
{
  "is_match": true,
  "object": "黑色双肩包",
  "attributes": ["尼龙材质", "双拉链", "正面口袋"],
  "risk": "无明显图文不一致"
}
```

最低产出：

* 用 3k–10k 条自建或公开数据做 LoRA 微调；
* 对比 base 模型和微调后模型；
* 记录准确率、错误案例、显存占用、训练耗时。

---

### 第 3 阶段：做多模态检索与内容理解系统

时间：第 5–6 周

这是最适合你投大厂应用算法岗的项目主线。字节、电商、搜索、内容理解岗位都很看重这类能力。

系统应该包括：

* 图文 embedding：CLIP / SigLIP；
* 向量库：FAISS / Milvus；
* 检索任务：文本搜图、图搜图、图文匹配、视频片段检索；
* VLM 复核：用 Qwen-VL / InternVL 对 Top-K 结果做 rerank 或解释；
* OCR：处理商品图、海报、文档截图；
* 视频：抽帧 + caption + OCR + embedding；
* 评测：Recall@1、Recall@5、mAP、Temporal IoU、P95 latency。

你已有 SigLIP 微调蒸馏和跨模态召回项目，可以继续升级，而不是另起炉灶。重点是把它做成“业务系统”，不是单纯图文检索 demo。

最低产出：

* 10k–50k 图文/视频样本；
* 可视化检索页面；
* 支持文本查图、图搜图、视频片段定位；
* 有明确指标：Recall@K、P95、QPS、显存。

---

### 第 4 阶段：补视频理解、多图推理、视觉 Reasoning

时间：第 7–8 周

现在多模态岗位已经不只看单图 VQA。小红书这类 VLM 岗位明确提到视频时序理解、多图推理、空间关系推理、多模态 Agent、Tool-use。([牛企直聘][6])

你要补：

* 视频抽帧策略：均匀抽帧、镜头切分、关键帧抽取；
* 多帧输入：frame caption、时间戳、事件摘要；
* 多图推理：前后对比、变化检测、流程判断；
* 空间关系：目标位置、相对关系、区域理解；
* Tool-use：OCR、检测器、检索库、规则库、计算工具。

建议你做一个小模块：

> **多模态视觉 Agent：输入一段视频或多张图片，自动调用 OCR / 检索 / VLM / 规则库，输出结构化分析报告。**

注意，不要做成泛泛的 Agent。要围绕一个场景，例如：

* 商品短视频内容理解；
* 工业巡检视频异常分析；
* 文档截图问答；
* 电商图文一致性审核；
* 高空作业流程合规判断。

最低产出：

* 支持多图/视频输入；
* 有任务分解过程；
* 有工具调用日志；
* 有结构化输出；
* 有失败案例分析。

---

### 第 5 阶段：部署、推理优化和简历包装

时间：第 9–10 周

这一步决定你和普通“会调 API 的多模态项目”拉开多少差距。

你需要做：

* FastAPI 服务；
* Docker 一键部署；
* batch inference；
* embedding 缓存；
* FP16 / INT8；
* ONNX / TensorRT；
* vLLM 或 Transformers 推理对比；
* P50 / P95 / P99 延迟；
* QPS、吞吐、显存峰值；
* README + 架构图 + benchmark 表格。

你已经有 Qwen3-VL 视觉 Token 裁剪和 SigLIP TensorRT 部署基础，可以继续往“工程可复现 + 指标完整”方向打磨。

最低产出：

| 项目指标 | 必须有                                       |
| ---- | ----------------------------------------- |
| 模型效果 | Recall@K / Accuracy / F1 / VQA Accuracy   |
| 推理性能 | P95 latency / QPS / 显存                    |
| 工程交付 | FastAPI / Docker / README / Demo          |
| 对比实验 | base vs 微调，PyTorch vs TensorRT，缓存前 vs 缓存后 |
| 错误分析 | 至少 20 个典型失败案例                             |

---

## 4. 你应该重点学什么，不该学什么

### 必学

* PyTorch 深度使用：Dataset、Dataloader、AMP、DDP 基础、LoRA 微调；
* Transformer / LLM 基础：prefill、decode、KV Cache、attention mask；
* CLIP / SigLIP：对比学习、hard negative、图文检索；
* Qwen-VL / InternVL / LLaVA：结构、输入格式、SFT、推理；
* 多模态数据构建：caption 清洗、OCR、图文匹配、负样本构造；
* 检索与 RAG：FAISS / Milvus、rerank、结构化问答；
* 部署：FastAPI、Docker、ONNX、TensorRT、vLLM/Transformers；
* 评测：Recall@K、mAP、Accuracy、P95、QPS、显存。

### 暂时不作为主线

* Java：不是多模态算法岗核心；
* 复杂 CUDA kernel：除非你转 AI Infra / 推理优化岗；
* 纯后端微服务：不是你主线；
* 从零训练大模型：个人资源不现实；
* 扩散模型全栈：除非你明确转视觉生成；
* 传统 CV 细碎任务：不要继续堆 YOLO 项目。

---

## 5. 最适合你的最终能力画像

你秋招简历和面试应该塑造成：

> **具备 CV/异常检测研究背景，熟悉 CLIP/SigLIP/Qwen-VL/InternVL 等视觉语言模型，能够完成图文/视频多模态理解、跨模态检索、VLM 微调、结构化评测和推理部署优化；关注开放场景内容理解与工业视觉落地。**

这个定位比“计算机视觉算法工程师”更窄、更准，也比泛泛的“AI 应用工程师”更有算法含量。

你的路线可以概括成一句话：

> **用 2 周补 VLM 原理，2 周做 LoRA/SFT，2 周做图文视频检索系统，2 周补多图/视频推理和 Agent，最后 2 周做部署优化与简历指标。**

[1]: https://seed.bytedance.com/zh/career "加入我们 - 字节跳动Seed"
[2]: https://seed.bytedance.com/zh/topseed "Top Seed人才计划 - 字节跳动Seed"
[3]: https://www.nowcoder.com/jobs/detail/374385 "多模态大模型算法工程师_字节跳动社招_牛客网"
[4]: https://www.huaweicloud.com/careers/doctor/20220706190636232.html "多模态算法工程师_华为云"
[5]: https://mnowpick.nowcoder.com/m/intern/detail?jobId=239625 "阿里巴巴-达摩院基础大模型团队（视觉）-研究型实习生_阿里巴巴集团实习_牛客网"
[6]: https://jobs.niuqizp.com/job-vks55Zz55.html "小红书(xiaohongshu)招聘基础模型算法工程师 - VLM Post-training  招聘城市有:北京|上海"
