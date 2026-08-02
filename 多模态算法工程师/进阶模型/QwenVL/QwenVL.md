---
title: Qwen-VL 系列 — 从 Qwen-VL 到 Qwen3-VL 的架构演进
description: Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL 的架构演进、动态分辨率、M-RoPE、视觉 Token 处理与工程实践
category: multimodal
tags: [Qwen-VL, VLM, Dynamic Resolution, M-RoPE, Visual Token, OCR, Video Understanding]
status: draft
order: 1
---

# Qwen-VL 系列：从 Qwen-VL 到 Qwen3-VL

## 一句话解释

**Qwen-VL 系列**是通义千问的视觉语言模型家族，核心设计思路是 **ViT 视觉编码器 + 视觉 Token 压缩/合并 + LLM 生成**，并在四代演进中逐步解决了分辨率适配（动态分辨率）、位置编码（M-RoPE）、超长上下文（1M token）与高分辨率（6B ViT）等 VLM 关键难题。

---

## 1. 系列演进总览

| 版本 | 时间 | 视觉编码器 | 视觉-语言连接 | 核心突破 |
|------|------|-----------|--------------|---------|
| Qwen-VL | 2023.09 | OpenCLIP ViT-bigG（冻结） | 单层 Cross-Attention（re-attention） | 首次发布，图文对话 |
| Qwen-VL-Chat | 2023.09 | 同上 | 同上 | 增强指令跟随 |
| Qwen2-VL | 2024.09 | ViT 675M（patch14） | MLP + 2×2 卷积合并 | Naive Dynamic Resolution、M-RoPE |
| Qwen2.5-VL | 2025.01 | ViT 600M（patch14） | Window Attention + Visual Token Merger | 文档解析、视频时间戳定位、Agent |
| Qwen3-VL | 2025.09 | ViT 6B | 高阶 Token 合并 | MoE 架构、双思维模式、1M 上下文、长视频 |

> 面试记忆点：**四代演进主线 = 分辨率从固定到动态 + 位置编码从 2D 到 3D（时空） + Token 压缩从固定卷积到自适应 + 视觉编码器越来越大。**

---

## 2. Qwen-VL（2023）：第一代

### 2.1 架构

```
图像 → OpenCLIP ViT-bigG（冻结）→ 视觉特征(448 tokens, 28×16)
                                            ↓
                            单层 Cross-Attention（re-attention，可训练）
                                            ↓
                                   256 tokens（压缩 448→256）
                                            ↓
                                  Qwen-7B LLM 生成
```

- 视觉编码器：OpenCLIP 的 ViT-bigG（约 19 亿参数，**冻结**），分辨率固定 224×224
- 关键模块：**单层 Cross-Attention 适配器**，把 ViT 输出的 448 个视觉 token 压缩到 256 个，再送入 LLM —— 思路类似 BLIP-2 的 Q-Former，但更轻
- LLM：Qwen-7B，支持中英文、图像和文本交错输入

### 2.2 训练三阶段

1. **预训练**：1.4B 图文对，只训适配器，冻结 ViT 和 LLM
2. **多任务预训练**：VQA、图文匹配、OCR、视觉定位等任务数据混合
3. **指令微调**：350K 对话数据，得到 Qwen-VL-Chat

---

## 3. Qwen2-VL（2024）：动态分辨率 + M-RoPE

第二代是架构大改版，三个核心创新：

### 3.1 Naive Dynamic Resolution（原生动态分辨率）

- **旧问题**：固定 resize 到 224/448 会把小字、小物体压糊，长宽比变形
- **新方案**：图像按最小像素边（如 28×28 的 patch 网格）切分，**支持任意分辨率**，直接输入原始长宽比
- 图像被分成可变数量的 patch 网格（如 256×256 的图 → 9×9 grid），每格由 patch14 的 ViT 处理

### 3.2 M-RoPE（Multimodal Rotary Position Embedding）

- 把 RoPE 从文本的 1D 扩展到 **3D 时空位置编码（时间、高、宽）**
- 视频 token 带时间位置、图像 token 带空间位置，文本 token 用传统 1D RoPE
- 效果：**视频时序理解 + 空间定位**在一个统一的编码框架下完成，是后续视频定位能力的根基

### 3.3 视觉 Token 压缩

- ViT 输出特征先过 **2×2 卷积**，把相邻 2×2 的视觉特征合并成 1 个 token（token 数减为 1/4）
- 2×2 卷积权重可学习，属于"固定比例压缩"

### 3.4 其他能力

- ViT 从冻结变为 675M 可训练，patch 14，加 MLP 投影层
- 支持交错图文输入（一张序列中多张图）、20 分钟以上长视频
- 多语言 OCR 大幅增强

---

## 4. Qwen2.5-VL（2025）：文档解析 + 视频定位 + Agent

### 4.1 视觉编码器升级

- ViT 600M，patch14，引入 **Window Attention**：高分辨率图像分窗计算注意力，显存和计算随分辨率线性增长而非平方
- 新增 **Visual Token Merger**：用 Cross-Attention 让"可学习的压缩向量"与视觉特征交互，**自适应**决定保留多少视觉 token（替代 Qwen2-VL 的固定 2×2 卷积）

### 4.2 能力增强

- **文档解析**：版面分析、表格还原、公式 OCR 大幅提升，接近商用 OCR 服务
- **视频时间戳定位**：能回答"事件发生在第几秒"，靠 M-RoPE 的时间维度
- **绝对坐标定位**：输出物体在图像中的绝对坐标（无需固定坐标区间映射）
- **Agent 能力**：作为视觉 Agent 调用工具（OCR、检测器、搜索）
- 尺寸：3B / 7B / 32B / 72B

---

## 5. Qwen3-VL（2025）：MoE + 双思维模式 + 长视频

> 你简历里的"Qwen3-VL-32B 部署与视觉 Token 裁剪"就是这个版本，面试必背。

### 5.1 MoE 架构

- 代表模型 **Qwen3-VL-30B-A3B**：总参数约 30B，激活仅约 3B（MoE 稀疏激活）
- 视觉编码器升级到 **6B**，支持超高分辨率（>1M 视觉 token）
- 部署时显存友好：MoE 激活参数少，但全量权重仍需多卡

### 5.2 双思维模式（Thinking / Non-Thinking）

- **Thinking 模式**：先输出推理链（CoT）再给结论，适合复杂视觉推理
- **Non-Thinking 模式**：直接输出，低延迟，适合快速问答
- 训练时用 RL 强化了思考模式的推理能力

### 5.3 超长上下文

- 上下文窗口达 **1M token**，可处理数小时视频
- 视频 token 做更激进的压缩（M-RoPE 时间维度 + 帧合并），这是"视觉 Token 裁剪"优化空间大的原因

---

## 6. 关键技术细节（面试深挖）

### 6.1 视觉 Token 完整流程

```
原始图像 → 动态网格切分（patch14）→ ViT 编码 → [窗口注意力]
  → Token 合并（卷积 / Cross-Attention Merger）→ MLP 投影
  → 与文本 token 拼接 → LLM 自回归生成
```

面试必答：**一张 448×448 图在 Qwen2-VL 下约产生 32×32/4 ≈ 256 个视觉 token（2×2 卷积合并后）**；高分辨率图 token 数翻倍，显存与 KV cache 压力剧增 → 这是视觉 Token 裁剪优化的动机。

### 6.2 M-RoPE 为什么重要

- 传统 2D 位置编码无法表达视频的时间先后
- M-RoPE 把位置拆成 (时间, 高, 宽) 三个维度，视频理解、跨帧推理、时间戳定位都依赖它
- 记忆口诀：**文本 1D、图像 2D、视频 3D**

### 6.3 动态分辨率 vs 固定分辨率

| 项目 | 固定分辨率（Qwen-VL） | 动态分辨率（Qwen2+） |
|------|---------------------|---------------------|
| 小物体/小字 | 压缩模糊 | 保持清晰 |
| 长宽比 | 变形 | 不变形 |
| 计算量 | 固定 | 随分辨率线性增长 |
| 显存 | 可控 | 高分辨率爆显存风险 |

---

## 7. 工程部署实践

- **推理框架**：vLLM 原生支持 Qwen2-VL / Qwen2.5-VL / Qwen3-VL，注意 vision encoder 的显存占用
- **显存优化三板斧**：
  1. 视觉 Token 裁剪/采样（控制送入 LLM 的 token 数，KV cache 大头）
  2. FP16/INT8/INT4 量化（LLM 部分）
  3. 高分辨率上限限制（限制 grid 最大 patch 数）
- **性能指标**：prefill 时视觉 token 全量计算（算力密集），decode 时看 KV cache（显存密集）

---

## 8. 面试高频问题

1. Qwen-VL 到 Qwen3-VL 各代解决了什么问题？
2. 动态分辨率怎么做？为什么比固定分辨率好？
3. M-RoPE 是什么？为什么视频理解需要它？
4. 一张高分辨率图会生成多少视觉 token？怎么算？
5. 视觉 Token 裁剪有哪些方法？剪多了会损失什么？
6. Qwen3-VL 的 MoE 和 Thinking 模式各有什么意义？
7. Qwen2.5-VL 的 Visual Token Merger 为什么比 2×2 卷积好？
