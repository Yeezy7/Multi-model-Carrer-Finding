# 实施状态与实验门槛

## 已完成

- 三套公开数据下载与 Parquet 完整性检查；
- Qwen3-VL-8B-Instruct 官方权重下载和四个 shard SHA256 校验；
- GPU 1 上完成 10,650 条 BF16 Dynamic Resolution 原模型基线；
- 官方式 ANLS、ChartQA relaxed accuracy、结构化字段 micro F1；
- 77,836 条 Image + Instruction + Response 三任务训练清单；
- NF4 QLoRA、多任务 sampling、困难样本增强、adapter 复测和失败分析入口。
- 1-step 反向传播 smoke 与 100-step/400-sample 持续训练 pilot，均退出码 0。

## 仍需真实数据或计算才能完成

1. 导入获得授权的合同和财务报表 KIE 标注，冻结 train/val/test split；
2. 跑完多任务正式训练，不以单步 smoke 代替；
3. 用与 base 完全相同的三个公开评测 split 复测 adapter；
4. 固定困难评测子集，再比较 augmentation=0 与 0.20；
5. 完成 one-factor-at-a-time 分辨率、LoRA rank、task sampling 消融；
6. 人工复核 Failure Case 候选，确认 OCR、跨区域关联和数值推理根因。

## 结果接受条件

- 训练进程退出码 0，保存最终 adapter 和 `run_config.json`；
- 评测样本数分别为 5,349、2,500、2,801，且运行错误数为 0；
- KIE F1 只能来自独立 held-out、合法 JSON 可统计的真实领域标注；
- 每项提升均同时报告 base、SFT、绝对差值和协议；
- 原项目中的 91.2、90.4%、+2.1 在复现实验前只视为目标，不写成结果。
