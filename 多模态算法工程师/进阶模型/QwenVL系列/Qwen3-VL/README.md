# Qwen3-VL-8B 复杂文档多任务理解

本目录是一套可运行、可断点续训、可审计的文档多模态 SFT 流程，覆盖 Document VQA、Chart QA、Infographic VQA 和自定义 JSON 字段抽取。模型固定为本地 `Qwen3-VL-8B-Instruct`，训练采用 Transformers + PEFT NF4 QLoRA。

## 当前已经验证的结果

原始模型使用 BF16、greedy decoding、Dynamic Resolution（262,144–1,048,576 pixels）完成了 10,650 条全量基线评测，进程退出码为 0，运行错误数为 0。

| 数据集 | Split / 样本数 | 指标 | 原模型结果 |
|---|---:|---|---:|
| DocVQA | validation / 5,349 | ANLS（0.5 threshold） | **94.05** |
| ChartQA | test / 2,500 | Relaxed Accuracy（数值 5% tolerance） | **84.12%** |
| InfographicVQA | validation / 2,801 | ANLS（0.5 threshold） | **77.09** |

ChartQA 进一步拆分为 augmented 94.40%、human 73.84%。完整配置和逐样本预测位于：

```text
outputs/base_eval/bf16_dynamic_1m_full/
├── summary.json
├── docvqa.jsonl
├── chartqa.jsonl
└── infographicvqa.jsonl
```

原项目描述中的“DocVQA 86.9 → 91.2、KIE F1 90.4%、困难集 +2.1”不是本仓库当前实验结果。尤其原模型 DocVQA 已达到 94.05，因此这些数字不能继续作为真实结论；SFT 后指标必须由同协议复测产生。

## 数据

原始 Parquet 位于仓库根目录 `datasets/`。训练数据构建命令：

```bash
.venv/bin/python build_multitask_data.py
```

生成 `datasets/Qwen3VL-SFT/train.jsonl`，训练清单采用 Qwen3-VL 官方的 `image + conversations` 格式，额外保留 `task` 方便多任务采样：

```json
{
  "id": "docvqa:123",
  "image": "images/docvqa/456-abcdef.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the invoice total?"},
    {"from": "gpt", "value": "$125.00"}
  ],
  "task": "docvqa",
  "source": "lmms-lab/DocVQA",
  "metadata": {}
}
```

训练入口读取官方格式后，在内存中提取 human/gpt 两轮对话，用于任务采样和 assistant-only loss；磁盘上的 JSONL 本身仍保持官方格式。

当前清单包含 77,836 条问答和 30,629 张去重图片：

| Task | 问答数 | 去重图片数 |
|---|---:|---:|
| DocVQA | 39,463 | 10,194 |
| ChartQA | 28,299 | 18,317 |
| InfographicVQA | 10,074 | 2,118 |

InfographicVQA 训练数据来自 `AI-ModelScope/the_cauldron:infographic_vqa`，评测仍使用独立的 validation split，不混入训练数据。

### 合同、财报和 KIE 自定义数据

自定义 JSONL 每行必须提供图片和 JSON response：

```json
{"id":"contract-001","image_path":"pages/001.png","instruction":"Extract contract_number, party_a and effective_date.","response":{"contract_number":"CT-001","party_a":"Example Ltd.","effective_date":"2026-01-01"},"source":"internal-contracts"}
```

导入时会验证 JSON、复制并去重图片：

```bash
.venv/bin/python build_multitask_data.py \
  --kie-jsonl /path/to/contracts.jsonl /path/to/financial_reports.jsonl \
  --force
```

导入 KIE 后，训练时需要显式给它非零采样权重，例如：

```bash
--task-weights docvqa=0.40,chartqa=0.25,infographicvqa=0.20,kie=0.15
```

仓库目前没有获得真实合同/财报标注，因此不会用虚构样本报告 KIE F1。

## 环境

当前已验证环境：Python 3.10、PyTorch 2.4.1+cu124、Transformers 4.57.3、PEFT 0.18.0、RTX 4090 24GB。

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Qwen3-VL 的 processor 和动态分辨率用法参考 [Transformers 官方文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)，LoRA 配置与数据处理边界参考 [Qwen3-VL 官方 finetune 代码](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)，NF4 QLoRA 配置参考 [PEFT 官方量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)。

## 原模型评测

固定使用 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python evaluate_base.py \
  --datasets all \
  --output-dir outputs/base_eval/bf16_dynamic_1m_full \
  --log-every 25
```

脚本逐条 flush JSONL，重复执行会按 `sample_id` 断点续跑。使用 adapter 复测时只增加：

```bash
--adapter outputs/checkpoints/qwen3vl_8b_multitask_qlora
```

## 多任务 QLoRA SFT

先验证数据和采样，不加载模型：

```bash
.venv/bin/python train_sft.py --dry-run
```

单卡 24GB 的正式起点配置：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python train_sft.py \
  --task-weights docvqa=0.45,chartqa=0.30,infographicvqa=0.25 \
  --resolution-mode dynamic \
  --min-pixels 262144 \
  --max-pixels 1048576 \
  --lora-r 16 \
  --lora-alpha 32 \
  --hard-augment-probability 0.20 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --epochs 1
```

训练行为：

- 4-bit NF4 + double quantization，BF16 compute；
- 冻结视觉编码器和基座权重，只训练语言 attention 的 `q/k/v/o_proj` LoRA；
- loss 仅覆盖 assistant response，用户提示和视觉 token 全部 mask；
- epoch 内按任务权重确定性采样，下一 epoch 自动换种子；
- 20% 训练图片执行低清、模糊、低对比度或灰度增强；
- 自动从输出目录最新 `checkpoint-*` 续训，并保存 `run_config.json`。

已额外完成 400 个采样样本、100 optimizer steps 的持续训练 pilot。相同 524,288-pixel 上限、每个数据集固定前 100 条的回归检查为：DocVQA 92.55→92.97（+0.41），ChartQA 68.00→67.00（-1.00），InfographicVQA 72.45→74.47（+2.02）。该 pilot 仅用于验证长于单步的训练和对照评测链路，不是正式性能结论；对比记录见 `outputs/pilot/comparison_100each.json`。

## 消融协议

采用 one-factor-at-a-time，其他条件保持不变：

1. Resolution：约 512/768 个视觉 token 的 fixed pixel budget、约 256–1024 个视觉 token 的 dynamic pixel budget；
2. LoRA rank：8、16、32，保持 `alpha = 2 × rank`；
3. Task sampling：DocVQA-only、natural mix、0.45/0.30/0.25 balanced mix；
4. Hard augmentation：0 与 0.20。

每个实验必须保存 adapter、`run_config.json`、全量预测 JSONL、`summary.json` 和退出码。不要把 smoke、部分 split 或失败后残留文件当成正式结果。

## 结构化指标与 Failure Case

KIE prediction JSONL 中包含 `prediction` 和 `response` 后，计算字段级 micro F1 与 JSON 合法率：

```bash
.venv/bin/python evaluate_kie_outputs.py \
  --predictions outputs/kie/predictions.jsonl \
  --output outputs/kie/metrics.json
```

也可以直接对统一 held-out manifest 推理并评分：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python evaluate_manifest.py \
  --manifest /path/to/kie_test.jsonl \
  --tasks kie \
  --adapter outputs/checkpoints/qwen3vl_8b_multitask_qlora \
  --output-dir outputs/kie/held_out
```

对基线或 SFT 预测生成候选错误清单：

```bash
.venv/bin/python analyze_failures.py \
  --prediction-dir outputs/base_eval/bf16_dynamic_1m_full \
  --output-dir outputs/analysis/base_bf16_dynamic_1m
```

脚本只给出 `OCR/layout`、`numeric reading`、`cross-region/numeric reasoning` 启发式候选标签；根因必须结合原图人工复核。
