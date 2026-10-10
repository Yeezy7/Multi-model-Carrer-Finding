# 长视频时序检索：LoRA-R 第一阶段

这一阶段只实现 **Query-Frame 相关性检索**，不接入 LoRA-U、候选片段问答和 Visual Token Pruning。

## 当前链路

```text
视频级标注
  → 0.5 FPS 全局抽帧
  → Qwen3-VL 图像 + Query
  → LoRA 微调输出 1/0
  → logit(1) - logit(0)
  → S_q(t)
```

这里的 LoRA-R 是 Qwen3-VL 的相关性判别适配器。训练目标是让模型对单帧只输出 `1`（相关）或 `0`（不相关），推理时使用两个标签 logit 的差值作为相关性分数，而不是使用自由生成文本。

## 输入标注格式

`build_retrieval_manifest.py` 接收 JSONL：

```json
{
  "video_id": "video_001",
  "video_path": "videos/video_001.mp4",
  "query": "When does the man open the refrigerator?",
  "gt_segments": [{"start": 315.2, "end": 328.7}]
}
```

可以额外提供 `duration`。如果一个 Query 对应多个事件区间，将多个区间放进 `gt_segments`。

## ActivityNet Captions 小子集

先用公开标注和视频元数据选择样本，不下载视频：

```bash
python prepare_activitynet_subset.py \
  --annotations /path/to/train.json \
  --metadata /path/to/metadata.jsonl \
  --output-dir data/activitynet_subset \
  --num-pairs 100 \
  --min-segment-seconds 8 \
  --max-video-mb 30 \
  --dry-run
```

去掉 `--dry-run` 后会写出 `activitynet_subset.jsonl` 和 `download_manifest.jsonl`。确认下载清单后，再加 `--download-videos` 获取选中的视频。原始 ActivityNet Captions 只提供时间标注，视频文件需要单独获取；因此视频下载失败的样本必须从最终清单中剔除，不能当作已完成。

## 生成帧级清单

先执行不写盘的检查：

```bash
python build_retrieval_manifest.py \
  --annotations data/annotations/video_queries.jsonl \
  --output-dir data/retrieval \
  --global-fps 0.5 \
  --negative-ratio 2 \
  --dry-run
```

确认视频路径和标注无误后，去掉 `--dry-run`。脚本会生成：

```text
data/retrieval/
├── frames/<video_id>/<timestamp>.jpg
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── summary.json
```

当前正样本定义为落在 GT 区间前后 `0.5` 秒容差内的全局帧；负样本默认按正样本数量的 2 倍下采样，并优先保留 GT 边界附近的 Hard Negative。`--hard-negative-window 4` 控制边界附近的秒数，`--hard-negative-ratio 1` 控制每个正帧最多保留的边界负帧比例。

如果要在当前 100 个视频上保留多个 ActivityNet Query，可以先运行：

```bash
python expand_activitynet_queries.py \
  --annotations data/annotations/train.json \
  --base-subset data/activitynet_subset/activitynet_subset.jsonl \
  --output data/activitynet_subset/activitynet_multiquery.jsonl \
  --queries-per-video 3
```

同一视频的多个 Query 必须通过视频 ID 放在同一个 train/val/test split 中，不能按 Query 随机拆分。

## 训练 LoRA-R

先只检查清单：

```bash
python train_retrieval_lora.py \
  --train-path data/retrieval/train.jsonl \
  --val-path data/retrieval/val.jsonl \
  --dry-run
```

批次检查需要加载 processor，但不会训练：

```bash
python train_retrieval_lora.py \
  --train-path data/retrieval/train.jsonl \
  --validate-batch-only
```

正式训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train_retrieval_lora.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --train-path data/retrieval/train.jsonl \
  --val-path data/retrieval/val.jsonl \
  --output-dir outputs/retrieval_lora \
  --qlora \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --epochs 1
```

当前训练脚本冻结基座参数，并对名称匹配的 `q/k/v/o_proj` 模块添加 LoRA；正式实验前应打印命中的模块，确认是否只覆盖预期的语言/跨模态模块。

## 输出相关性分数

```bash
CUDA_VISIBLE_DEVICES=0 python score_retrieval.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --adapter outputs/retrieval_lora \
  --manifest data/retrieval/test.jsonl \
  --output outputs/retrieval_scores.jsonl
```

输出中新增：

- `score`：`sigmoid(logit(1) - logit(0))`，作为 `S_q(t)`；
- `prediction`：以 0.5 为阈值的帧级预测。

当前已补充 `build_temporal_candidates.py`，从 `score_retrieval.py` 的帧分数生成候选时间窗口：先使用局部时间平滑得到 Temporal Importance，再通过时间 NMS 选择种子帧，最后扩展为固定长度候选窗口，并计算窗口命中率与 Temporal IoU Recall。对 test 视频的完整 0.5 FPS 帧进行打分后，32 秒窗口的 video macro Recall@5（tIoU≥0.3）为 0.811，命中率@5 为 0.985，Recall@10（tIoU≥0.3）为 0.900；64 秒窗口的 tIoU 更高，但片段过长，会增加后续 VLM 理解成本，因此暂以 32 秒作为候选片段配置。

```bash
python build_temporal_candidates.py \
  --scores outputs/retrieval_scores.jsonl \
  --output outputs/temporal_candidates.jsonl \
  --metrics-output outputs/temporal_metrics.json \
  --top-k 1 5 10 \
  --window-seconds 32 \
  --nms-seconds 4
```

可以使用 `evaluate_retrieval.py` 计算 Query 宏平均和 video 宏平均的 Recall@K、Precision@K、AP：

```bash
python evaluate_retrieval.py \
  --scores outputs/retrieval_scores.jsonl \
  --output outputs/retrieval_metrics.json
```
