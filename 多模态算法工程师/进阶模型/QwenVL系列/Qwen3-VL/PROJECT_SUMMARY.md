# Qwen3-VL 复杂文档多任务理解项目总结

## 1. 项目目标

本项目基于 `Qwen3-VL-8B-Instruct`，构建面向财务报告、表格和图表等复杂文档的多模态训练与评测流程，重点验证：

- 财报问答和数值推理（FinQA）；
- 财务表格视觉理解（FinTabNet）；
- 结构化字段抽取（KIE）；
- Qwen3-VL 官方 `image + conversations` 数据格式；
- Transformers + PEFT + NF4 QLoRA 微调流程。

项目目录：

```text
多模态算法工程师/进阶模型/QwenVL系列/Qwen3-VL/
```

## 2. 数据处理流程

### 2.1 公开文档数据

项目最初完成了 DocVQA、ChartQA、InfographicVQA 的下载、转换和原模型基线评测。训练数据统一整理成：

```json
{
  "id": "sample-id",
  "image": "images/example.png",
  "conversations": [
    {"from": "human", "value": "<image>\\nQuestion"},
    {"from": "gpt", "value": "Answer"}
  ],
  "task": "docvqa"
}
```

### 2.2 财务数据

当前正式财务训练清单由以下数据组成：

| 数据集 | 训练用途 | 当前划分 |
|---|---|---|
| FinQA | 财报问答和数值推理 | `train` 用于训练，`dev` 用于验证 |
| FinTabNet | 财务表格理解和结构抽取 | `train` 用于训练，`val` 用于验证 |

重新整理后的清单统计：

| 清单 | 总样本数 | FinQA | FinTabNet |
|---|---:|---:|---:|
| `train.jsonl` | 94,639 | 6,198 | 88,441 |
| `dev.jsonl` | 11,376 | 871 | 10,505 |

数据构建脚本：

```bash
.venv/bin/python build_financial_data.py --force
```

这次划分修正了此前把 FinQA train/dev 和 FinTabNet train/val 混在一起的问题。正式训练没有使用 FinQA dev 或 FinTabNet val，避免验证集泄漏。

## 3. 训练流程

训练入口是 `train_sft.py`，核心流程如下：

1. 读取 Qwen3-VL 官方格式 JSONL；
2. 使用 Processor 构造文本 token 和图像 token；
3. 加载 Qwen3-VL-8B-Instruct；
4. 使用 4-bit NF4 量化和 BF16 计算；
5. 冻结基座模型，只训练 LoRA 参数；
6. 只对 assistant response 计算 loss；
7. 保存 PEFT adapter 和训练状态。

本次正式财务训练使用：

- 训练样本：94,639；
- 训练轮数：1；
- optimizer steps：5,915；
- `max_length=8192`；
- gradient accumulation：16；
- LoRA 可训练参数：15,335,424；
- 总参数量：8,782,459,120；
- 最终训练损失：`0.2966`；
- 训练耗时：约 16 小时 55 分。

最终 adapter：

```text
outputs/checkpoints/qwen3vl_financial_official_v1
```

训练日志：

```text
outputs/logs/qwen3vl_financial_official_v1.log
```

## 4. 评测结果

### 4.1 原模型公开数据基线

原始 Qwen3-VL 模型在公开文档数据上的已验证结果：

| 数据集 | Split | 指标 | 原模型 |
|---|---|---|---:|
| DocVQA | validation / 5,349 | ANLS | 94.05 |
| ChartQA | test / 2,500 | Relaxed Accuracy | 84.12% |
| InfographicVQA | validation / 2,801 | ANLS | 77.09 |

这些结果是原模型基线，不是财务 adapter 的结果。

### 4.2 正式财务验证集

使用最终 adapter `qwen3vl_financial_official_v1` 对独立 `dev.jsonl` 评测：

| 任务 | 样本数 | 指标 | 微调结果 |
|---|---:|---|---:|
| FinQA dev | 871 | ANLS | **60.26%** |
| FinTabNet val / KIE | 10,505 | 字段级 micro F1 | **39.81%** |

FinQA 原模型在同一 871 条 dev 集上的结果为 `31.39%`，因此：

```text
60.26 - 31.39 = +28.87 个百分点
```

KIE 详细结果：

- Precision：43.28%；
- Recall：36.85%；
- F1：39.81%；
- JSON 有效率：85.14%；
- 运行时错误：0。

结果文件：

```text
outputs/eval/official_finqa_dev/summary.json
outputs/eval/official_kie_val/summary.json
```

## 5. 结果解读

### 已验证的结论

1. QLoRA 训练流程可以正常完成，并成功保存 adapter；
2. FinQA 在严格独立 dev 集上从 31.39% 提升到 60.26%；
3. 正式验证集没有被训练数据直接使用；
4. KIE 推理流程能够处理全部 10,505 条验证样本，且没有运行时错误；
5. KIE 当前具备一定字段抽取能力，但距离高质量结构化抽取还有明显差距。

### 目前不能宣称的结果

项目描述中曾出现以下目标数字：

- DocVQA ANLS `86.9 → 91.2`；
- KIE F1 `90.4%`；
- 困难样本提升 `2.1` 个百分点。

这些数字目前没有被本项目的正式实验复现，不能作为当前项目结果。特别是当前 KIE 验证结果为 `39.81%`，不能写成 `90.4%`。

## 6. 当前主要问题

### 6.1 KIE JSON 合法率不足

当前 JSON 有效率为 85.14%，约 14.86% 的输出无法被严格解析。无效 JSON 会直接降低 KIE 评测结果。

### 6.2 KIE 字段召回率偏低

KIE precision 为 43.28%，recall 为 36.85%。recall 更低，说明模型存在较多字段遗漏，而不是只存在格式问题。

### 6.3 评测速度较慢

当前 `evaluate_manifest.py` 对每条样本单独执行：

1. 打开图片；
2. Processor 编码；
3. Qwen3-VL 图像和文本推理；
4. 生成 JSON；
5. 写入预测文件。

KIE 验证集有 10,505 条样本，且使用 `max_new_tokens=2048`，因此完整评测耗时较长。当前脚本支持按已有 `predictions.jsonl` 断点续跑。

## 7. 常用命令

### 7.1 查看训练数据

```bash
wc -l datasets/Qwen3VL-Financial-SFT/train.jsonl
wc -l datasets/Qwen3VL-Financial-SFT/dev.jsonl
```

### 7.2 FinQA 验证

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python evaluate_manifest.py \
  --manifest datasets/Qwen3VL-Financial-SFT/dev.jsonl \
  --tasks finqa \
  --adapter outputs/checkpoints/qwen3vl_financial_official_v1 \
  --output-dir outputs/eval/official_finqa_dev \
  --load-in-4bit \
  --max-new-tokens 128
```

### 7.3 KIE 验证

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python evaluate_manifest.py \
  --manifest datasets/Qwen3VL-Financial-SFT/dev.jsonl \
  --tasks kie \
  --adapter outputs/checkpoints/qwen3vl_financial_official_v1 \
  --output-dir outputs/eval/official_kie_val \
  --load-in-4bit \
  --max-new-tokens 2048
```

## 8. 后续改进计划

1. 对 KIE 输出增加 JSON schema 约束或更明确的字段格式指令；
2. 分析无效 JSON 和字段遗漏样本，区分 OCR、表格结构和生成格式问题；
3. 对 KIE 使用更高质量、字段定义稳定的训练标注；
4. 对比 512、768 和动态分辨率；
5. 对比 LoRA rank 8、16、32；
6. 使用固定协议比较不同 task sampling 比例；
7. 将 KIE 评测改造成 batch inference 或多 GPU 分片，提高评测速度；
8. 在所有实验中同时保存 base、adapter、样本数、协议和错误数。

## 9. 最终结论

当前项目已经完成了从数据整理、官方格式转换、QLoRA 多任务训练到独立验证集评测的完整闭环。现阶段最可靠的结论是：

> Qwen3-VL-8B 的财务 QLoRA 微调显著提升了 FinQA 财报问答能力，FinQA dev ANLS 达到 60.26%；但 KIE 字段抽取 F1 仅为 39.81%，JSON 合法率和字段召回率仍是主要瓶颈。

因此，项目可以作为“完整训练和评测流程已跑通、FinQA 已取得明确提升、KIE 仍需专项优化”的真实实验总结。
